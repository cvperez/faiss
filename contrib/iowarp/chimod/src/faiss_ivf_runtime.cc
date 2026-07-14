/*
 * faiss_ivf ChiMod — runtime implementation.
 *
 * Contains the server-side task processing logic: opens a FAISS IndexIVF
 * (metadata only), serves batched k-NN searches by fetching probed
 * inverted lists from CTE in the same address space, and reports stats.
 */

#include "../include/clio_runtime/faiss_ivf/faiss_ivf_runtime.h"

#include <clio_ctp/serialize/msgpack_wrapper.h>

#include <faiss/IndexIVF.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/index_io.h>
#include <faiss/invlists/InvertedLists.h>
#include <faiss/utils/Heap.h>

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace clio::run::faiss_ivf {

namespace {

/** Current steady-clock time in microseconds. */
inline chi::u64 NowUs() {
  return static_cast<chi::u64>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

/**
 * PtrInvertedLists - read-only InvertedLists view over CTE-fetched
 * buffers. get_codes/get_ids return the fetched pointers; list_size comes
 * from the "sizes" blob for fetched lists and 0 otherwise (lists that were
 * not probed are never scanned). All mutation entry points throw.
 * release_codes/release_ids inherit the base-class no-op defaults.
 */
struct PtrInvertedLists : faiss::InvertedLists {
  std::vector<const uint8_t*> codes_;
  std::vector<const faiss::idx_t*> ids_;
  std::vector<size_t> sizes_;

  PtrInvertedLists(size_t nlist_in, size_t code_size_in)
      : faiss::InvertedLists(nlist_in, code_size_in),
        codes_(nlist_in, nullptr),
        ids_(nlist_in, nullptr),
        sizes_(nlist_in, 0) {}

  void set_list(size_t list_no, size_t size, const uint8_t* codes,
                const faiss::idx_t* ids) {
    codes_[list_no] = codes;
    ids_[list_no] = ids;
    sizes_[list_no] = size;
  }

  size_t list_size(size_t list_no) const override {
    return sizes_[list_no];
  }

  const uint8_t* get_codes(size_t list_no) const override {
    return codes_[list_no];
  }

  const faiss::idx_t* get_ids(size_t list_no) const override {
    return ids_[list_no];
  }

  size_t add_entries(size_t, size_t, const faiss::idx_t*,
                     const uint8_t*) override {
    FAISS_THROW_MSG("PtrInvertedLists is read-only");
  }

  void update_entries(size_t, size_t, size_t, const faiss::idx_t*,
                      const uint8_t*) override {
    FAISS_THROW_MSG("PtrInvertedLists is read-only");
  }

  void resize(size_t, size_t) override {
    FAISS_THROW_MSG("PtrInvertedLists is read-only");
  }
};

// Scan one fetched list for all (query, coarse_dis) pairs touching it,
// in parallel across per-thread scanners. Plain function on purpose:
// gcc 11 ICEs when an OpenMP region sits inside a C++20 coroutine body.
// Safe because every pair updates a distinct per-query heap.
bool ScanListParallel(
    const std::vector<std::pair<chi::u32, float>>& plist,
    const std::vector<std::unique_ptr<faiss::InvertedListScanner>>& scanners,
    const float* q, size_t d, int64_t l, size_t sz, const uint8_t* codes,
    const faiss::idx_t* ids, float* D_out, faiss::idx_t* I_out, size_t k) {
  bool ok = true;
  if (plist.size() == 1) {
    // Scalar fast path: no OMP region for single-query lists.
    auto* sc = scanners[0].get();
    try {
      const size_t qi = plist[0].first;
      sc->set_query(q + qi * d);
      sc->set_list(l, plist[0].second);
      sc->scan_codes(sz, codes, ids, D_out + qi * k, I_out + qi * k, k);
    } catch (const std::exception&) {
      ok = false;
    }
    return ok;
  }
  const int nthreads = static_cast<int>(scanners.size());
#pragma omp parallel for num_threads(nthreads) schedule(dynamic, 1)
  for (size_t pi = 0; pi < plist.size(); ++pi) {
    auto* sc = scanners[omp_get_thread_num()].get();
    const size_t qi = plist[pi].first;
    try {
      sc->set_query(q + qi * d);
      sc->set_list(l, plist[pi].second);
      sc->scan_codes(sz, codes, ids, D_out + qi * k, I_out + qi * k, k);
    } catch (const std::exception&) {
      ok = false;  // benign flag race; read only after the region
    }
  }
  return ok;
}

}  // namespace

// Method implementations for Runtime class

// Virtual method implementations (Init, Run, Del, SaveTask, LoadTask,
// NewCopy, Aggregate) are in autogen/faiss_ivf_lib_exec.cc

//===========================================================================
// Method implementations
//===========================================================================

chi::TaskResume Runtime::Create(ctp::ipc::FullPtr<CreateTask> task,
                                chi::RunContext& rctx) {
  CLIO_TASK_BODY_BEGIN
  HLOG(kDebug, "faiss_ivf: Executing Create task for pool {}", task->pool_id_);

  // Container is already initialized via Init() before Create is called
  CreateParams params = task->GetParams();
  pipeline_mode_ = params.pipeline_mode_;

  HLOG(kDebug, "faiss_ivf: Container created for pool: {} (pipeline_mode: {})",
       pool_name_, pipeline_mode_);
  (void)rctx;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::OpenIndex(ctp::ipc::FullPtr<OpenIndexTask> task,
                                   chi::RunContext& rctx) {
  CLIO_TASK_BODY_BEGIN
  (void)rctx;
  chi::ScopedCoMutex lock(open_mu_);

  std::string index_path = task->index_path_.str();
  std::string tag_name = task->tag_name_.str();

  if (opened_ && index_path == opened_index_path_ &&
      tag_name == opened_tag_name_) {
    // Same volume already open: report the current index metadata.
    task->ntotal_ = static_cast<chi::u64>(ivf_->ntotal);
    task->d_ = static_cast<chi::u32>(ivf_->d);
    task->nlist_ = static_cast<chi::u32>(ivf_->nlist);
    task->SetReturnCode(0);
    CLIO_CO_RETURN;
  }
  // Different (or first) volume: (re)load and replace the container state.
  opened_ = false;
  ivf_ = nullptr;
  index_owner_.reset();
  sizes_.clear();

  // Load index metadata only; the IVF data lives in CTE. SKIP_IVF_DATA
  // works for OnDisk ("ilod") index files; plain ArrayInvertedLists
  // ("ilar") files reject it, so fall back to a full read — the loaded
  // lists are simply never scanned (search uses the CTE-fetched view).
  faiss::Index* raw = nullptr;
  try {
    raw = faiss::read_index(index_path.c_str(), faiss::IO_FLAG_SKIP_IVF_DATA);
  } catch (const std::exception&) {
    try {
      raw = faiss::read_index(index_path.c_str());
    } catch (const std::exception& e) {
      HLOG(kError, "faiss_ivf: read_index('{}') failed: {}", index_path,
           e.what());
      task->SetReturnCode(2);
      CLIO_CO_RETURN;
    }
  }
  std::unique_ptr<faiss::Index> owner(raw);

  auto* ivf = dynamic_cast<faiss::IndexIVF*>(raw);
  if (ivf == nullptr) {
    HLOG(kError, "faiss_ivf: '{}' is not an IndexIVF", index_path);
    task->SetReturnCode(1);
    CLIO_CO_RETURN;
  }

  // Single-threaded FAISS inside the cooperative runtime.
  omp_set_num_threads(1);

  // In-process CTE client: bind straight to the canonical composed CTE
  // pool (512.0). CLIO_CTE_CLIENT_INIT is client-process code — its
  // blocking Wait() would deadlock this cooperative worker.
  cte_.Init(clio::cte::core::kCtePoolId);
  auto* cte = &cte_;

  // Get-or-create the tag holding the inverted list blobs.
  auto tag_fut = cte->AsyncGetOrCreateTag(tag_name);
  CLIO_CO_AWAIT(tag_fut);
  if (tag_fut->GetReturnCode() != 0) {
    HLOG(kError, "faiss_ivf: GetOrCreateTag('{}') failed (rc={})", tag_name,
         tag_fut->GetReturnCode());
    task->SetReturnCode(3);
    CLIO_CO_RETURN;
  }
  clio::cte::core::TagId tag_id = tag_fut->tag_id_;

  // Fetch the "sizes" blob: int64[nlist] of list sizes.
  auto* ipc = CLIO_IPC;
  const size_t nlist = ivf->nlist;
  const chi::u64 sizes_bytes =
      static_cast<chi::u64>(nlist) * sizeof(int64_t);
  ctp::ipc::FullPtr<char> buf = ipc->AllocateBuffer(sizes_bytes);
  if (buf.IsNull()) {
    HLOG(kError, "faiss_ivf: AllocateBuffer({}) failed for sizes blob",
         sizes_bytes);
    task->SetReturnCode(4);
    CLIO_CO_RETURN;
  }
  auto sizes_fut = cte->AsyncGetBlob(tag_id, "sizes", 0, sizes_bytes, 0,
                                     buf.shm_.template Cast<void>());
  CLIO_CO_AWAIT(sizes_fut);
  if (sizes_fut->GetReturnCode() != 0) {
    HLOG(kError, "faiss_ivf: GetBlob('sizes') failed (rc={})",
         sizes_fut->GetReturnCode());
    ipc->FreeBuffer(buf);
    task->SetReturnCode(5);
    CLIO_CO_RETURN;
  }

  const int64_t* sizes_ptr = reinterpret_cast<const int64_t*>(buf.ptr_);
  sizes_.assign(sizes_ptr, sizes_ptr + nlist);
  ipc->FreeBuffer(buf);

  chi::u64 ntotal = 0;
  for (size_t i = 0; i < nlist; ++i) {
    ntotal += static_cast<chi::u64>(sizes_[i]);
  }
  ivf->ntotal = static_cast<faiss::idx_t>(ntotal);

  // Commit state.
  index_owner_ = std::move(owner);
  ivf_ = ivf;
  tag_id_ = tag_id;
  opened_ = true;
  opened_index_path_ = index_path;
  opened_tag_name_ = tag_name;

  task->ntotal_ = ntotal;
  task->d_ = static_cast<chi::u32>(ivf_->d);
  task->nlist_ = static_cast<chi::u32>(nlist);
  task->SetReturnCode(0);

  HLOG(kInfo, "faiss_ivf: opened '{}' (d={}, nlist={}, ntotal={}, tag='{}')",
       index_path, task->d_, task->nlist_, ntotal, tag_name);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::Search(ctp::ipc::FullPtr<SearchTask> task,
                                chi::RunContext& rctx) {
  CLIO_TASK_BODY_BEGIN
  (void)rctx;
  if (!opened_ || ivf_ == nullptr) {
    task->SetReturnCode(1);
    CLIO_CO_RETURN;
  }

  const chi::u32 nq = task->nq_;
  const chi::u32 k = task->k_;
  chi::u32 nprobe = task->nprobe_;
  if (nq == 0 || k == 0 || nprobe == 0 ||
      task->d_ != static_cast<chi::u32>(ivf_->d)) {
    task->SetReturnCode(2);
    CLIO_CO_RETURN;
  }
  if (nprobe > ivf_->nlist) {
    nprobe = static_cast<chi::u32>(ivf_->nlist);
  }
  const chi::u32 mode =
      (task->mode_ == kSearchModeDefault) ? pipeline_mode_ : task->mode_;

  auto* ipc = CLIO_IPC;
  // NOTE: ShmPtr -> raw pointer via CLIO_IPC->ToFullPtr, as done for
  // blob_data_ in clio-core's compressor_runtime.cc.
  const float* q = reinterpret_cast<const float*>(
      ipc->ToFullPtr<char>(task->queries_.template Cast<char>()).ptr_);
  float* D_out = reinterpret_cast<float*>(
      ipc->ToFullPtr<char>(task->distances_out_.template Cast<char>()).ptr_);
  int64_t* I_out = reinterpret_cast<int64_t*>(
      ipc->ToFullPtr<char>(task->labels_out_.template Cast<char>()).ptr_);
  if (q == nullptr || D_out == nullptr || I_out == nullptr) {
    task->SetReturnCode(2);
    CLIO_CO_RETURN;
  }

  // Coarse quantization: assign each query to nprobe lists.
  const size_t nassign = static_cast<size_t>(nq) * nprobe;
  std::vector<float> coarse_dis(nassign);
  std::vector<faiss::idx_t> assign(nassign);
  try {
    ivf_->quantizer->search(nq, q, nprobe, coarse_dis.data(), assign.data());
  } catch (const std::exception& e) {
    HLOG(kError, "faiss_ivf: coarse quantization failed: {}", e.what());
    task->SetReturnCode(5);
    CLIO_CO_RETURN;
  }

  // Unique sorted set of non-empty probed lists, plus per-list probe map
  // (query, coarse distance) for the pipelined path.
  std::unordered_map<int64_t, std::vector<std::pair<chi::u32, float>>> probes;
  std::vector<int64_t> lists;
  for (chi::u32 qi = 0; qi < nq; ++qi) {
    for (chi::u32 j = 0; j < nprobe; ++j) {
      const faiss::idx_t l = assign[static_cast<size_t>(qi) * nprobe + j];
      if (l < 0 || sizes_[l] <= 0) {
        continue;
      }
      auto& vec = probes[l];
      if (vec.empty()) {
        lists.push_back(l);
      }
      vec.emplace_back(qi, coarse_dis[static_cast<size_t>(qi) * nprobe + j]);
    }
  }
  std::sort(lists.begin(), lists.end());

  // Fetch the probed lists from CTE with a BOUNDED window of in-flight
  // AsyncGetBlob tasks. Never issue thousands of tasks at once from a
  // handler: the runtime queues (queue_depth per worker) are shared with
  // the CTE handlers consuming them, and unbounded issuance livelocks the
  // whole runtime (observed with 4096 lists on ondisk_nb10M).
  // Blob "list/<i>" layout: uint8_t codes[size*code_size] ++ int64 ids[size].
  // Sized so that even `runtime.num_threads` concurrent SearchTasks stay
  // well under the per-worker queue_depth (1024): 8 tasks x 64 = 512.
  constexpr size_t kMaxInflight = 64;
  auto* cte = &cte_;
  const size_t code_size = ivf_->code_size;
  const size_t ntofetch = lists.size();
  HLOG(kInfo,
       "faiss_ivf: Search nq={} k={} nprobe={} mode={} unique_lists={}", nq,
       k, nprobe, mode, ntofetch);
  std::vector<ctp::ipc::FullPtr<char>> bufs(ntofetch);
  std::vector<chi::u64> nbytes(ntofetch, 0);
  std::vector<chi::Future<clio::cte::core::GetBlobTask>> futs(ntofetch);

  auto issue_one = [&](size_t i) -> bool {
    const int64_t l = lists[i];
    const chi::u64 bytes = static_cast<chi::u64>(sizes_[l]) * code_size +
                           static_cast<chi::u64>(sizes_[l]) * sizeof(int64_t);
    bufs[i] = ipc->AllocateBuffer(bytes);
    if (bufs[i].IsNull()) {
      HLOG(kError, "faiss_ivf: AllocateBuffer({}) failed for list {}", bytes,
           l);
      return false;
    }
    nbytes[i] = bytes;
    futs[i] = cte->AsyncGetBlob(tag_id_,
                                std::string("list/") + std::to_string(l), 0,
                                bytes, 0, bufs[i].shm_.template Cast<void>());
    return true;
  };

  task->SetReturnCode(0);
  const bool is_l2 = (ivf_->metric_type == faiss::METRIC_L2);
  chi::u64 bytes_fetched = 0;
  size_t nfetched = 0;

  if (mode == 0) {
    //=======================================================================
    // v0: fetch ALL probed lists (windowed issuance, all buffers stay
    // resident), then run search_preassigned over a pointer view.
    // NOTE: whole-probed-set residency — only viable when the probed set
    // fits the runtime allocator; v1 is the scalable path.
    //=======================================================================
    const size_t nfetch = ntofetch;
    chi::u64 t_wait0 = NowUs();
    bool fetch_failed = false;
    size_t issued = 0;
    size_t completed = 0;
    while (completed < issued || (!fetch_failed && issued < nfetch)) {
      while (!fetch_failed && issued < nfetch &&
             issued - completed < kMaxInflight) {
        if (!issue_one(issued)) {
          fetch_failed = true;
          break;
        }
        ++issued;
      }
      if (completed < issued) {
        CLIO_CO_AWAIT(futs[completed]);
        if (futs[completed]->GetReturnCode() != 0) {
          HLOG(kError, "faiss_ivf: GetBlob('list/{}') failed (rc={})",
               lists[completed], futs[completed]->GetReturnCode());
          fetch_failed = true;
        }
        ++completed;
      }
    }
    stat_fetch_wait_us_ += NowUs() - t_wait0;
    nfetched = completed;

    if (fetch_failed) {
      for (size_t i = 0; i < nfetch; ++i) {
        if (!bufs[i].IsNull()) {
          ipc->FreeBuffer(bufs[i]);
          bufs[i] = ctp::ipc::FullPtr<char>();
        }
      }
      task->SetReturnCode(4);
      CLIO_CO_RETURN;
    }

    // Build the pointer view. Construct with ivf_'s nlist/code_size so
    // replace_invlists' compatibility asserts pass.
    PtrInvertedLists view(ivf_->nlist, code_size);
    for (size_t i = 0; i < ntofetch; ++i) {
      const int64_t l = lists[i];
      const size_t sz = static_cast<size_t>(sizes_[l]);
      const uint8_t* codes = reinterpret_cast<const uint8_t*>(bufs[i].ptr_);
      const faiss::idx_t* ids = reinterpret_cast<const faiss::idx_t*>(
          bufs[i].ptr_ + sz * code_size);
      view.set_list(l, sz, codes, ids);
    }

    // Temporarily swap the view in; restore the previous invlists after
    // the search. own_invlists is cleared first so neither swap deletes.
    // v0_mu_ serializes the swap: concurrent v0 tasks on other workers
    // must not see each other's views on the shared ivf_.
    chi::ScopedCoMutex v0_lock(v0_mu_);
    faiss::InvertedLists* saved = ivf_->invlists;
    const bool saved_own = ivf_->own_invlists;
    ivf_->own_invlists = false;
    try {
      ivf_->replace_invlists(&view, false);
      ivf_->nprobe = nprobe;
      chi::u64 t_scan0 = NowUs();
      ivf_->search_preassigned(nq, q, k, assign.data(), coarse_dis.data(),
                               D_out, I_out, false);
      stat_scan_us_ += NowUs() - t_scan0;
    } catch (const std::exception& e) {
      HLOG(kError, "faiss_ivf: search_preassigned failed: {}", e.what());
      task->SetReturnCode(5);
    }
    ivf_->replace_invlists(saved, false);
    ivf_->own_invlists = saved_own;
  } else {
    //=======================================================================
    // v1: pipelined — scan each list as soon as its fetch completes,
    // maintaining per-query result heaps directly in the output buffers.
    //=======================================================================
    const float init_dis = is_l2 ? std::numeric_limits<float>::max()
                                 : std::numeric_limits<float>::lowest();
    for (size_t i = 0; i < static_cast<size_t>(nq) * k; ++i) {
      D_out[i] = init_dis;
      I_out[i] = -1;
    }

    // One scanner per scan thread: set_query/set_list mutate scanner
    // state, so threads must not share one. Scanning one list's touching
    // queries in parallel is safe — each (query, probe) pair updates a
    // distinct per-query heap. kScanThreads matches the baseline's 8 OMP
    // scan threads (the comparability budget); the OMP region contains no
    // co_await, so it never suspends mid-parallelism.
    constexpr int kScanThreads = 8;
    std::vector<std::unique_ptr<faiss::InvertedListScanner>> scanners;
    bool scanners_ok = true;
    try {
      for (int t = 0; t < kScanThreads; ++t) {
        scanners.emplace_back(ivf_->get_InvertedListScanner(false));
      }
    } catch (const std::exception& e) {
      HLOG(kError, "faiss_ivf: get_InvertedListScanner failed: {}", e.what());
      task->SetReturnCode(5);
      scanners_ok = false;
    }

    // Windowed streaming: keep at most kMaxInflight fetches outstanding,
    // scan each list the moment it lands, and FREE its buffer right after
    // scanning — peak shm residency is O(window), so this path scales to
    // volumes far larger than the runtime allocator.
    std::vector<bool> done(ntofetch, false);
    size_t issued = 0;
    size_t completed = 0;
    bool stop_issuing = false;
    while (completed < issued || (!stop_issuing && issued < ntofetch)) {
      while (!stop_issuing && issued < ntofetch &&
             issued - completed < kMaxInflight) {
        if (!issue_one(issued)) {
          task->SetReturnCode(3);
          stop_issuing = true;
          break;
        }
        ++issued;
      }
      bool progressed = false;
      for (size_t i = 0; i < issued; ++i) {
        if (done[i] || !futs[i].IsComplete()) {
          continue;
        }
        // Completed: the await returns immediately and finalizes the
        // future.
        CLIO_CO_AWAIT(futs[i]);
        done[i] = true;
        ++completed;
        ++nfetched;
        progressed = true;

        if (futs[i]->GetReturnCode() != 0) {
          HLOG(kError, "faiss_ivf: GetBlob('list/{}') failed (rc={})",
               lists[i], futs[i]->GetReturnCode());
          task->SetReturnCode(4);
        } else if (scanners_ok) {
          const int64_t l = lists[i];
          const size_t sz = static_cast<size_t>(sizes_[l]);
          const uint8_t* codes =
              reinterpret_cast<const uint8_t*>(bufs[i].ptr_);
          const faiss::idx_t* ids = reinterpret_cast<const faiss::idx_t*>(
              bufs[i].ptr_ + sz * code_size);
          const auto& plist = probes[l];

          chi::u64 t_scan0 = NowUs();
          // OMP region lives in a plain function: gcc 11 ICEs on OpenMP
          // pragmas inside C++20 coroutine bodies.
          if (!ScanListParallel(plist, scanners, q, ivf_->d, l, sz, codes,
                                ids, D_out, I_out, k)) {
            HLOG(kError, "faiss_ivf: scan failed on list {}", l);
            task->SetReturnCode(5);
          }
          stat_scan_us_ += NowUs() - t_scan0;
        }
        // Scanned (or failed): release the buffer immediately.
        bytes_fetched += nbytes[i];
        ipc->FreeBuffer(bufs[i]);
        bufs[i] = ctp::ipc::FullPtr<char>();
      }
      if (!progressed && completed < issued) {
        // Nothing ready: yield cooperatively and account the wait.
        chi::u64 t_wait0 = NowUs();
        CLIO_CO_AWAIT(chi::yield());
        stat_fetch_wait_us_ += NowUs() - t_wait0;
      }
    }

    // Sort each query's heap into ascending (L2) / descending (IP) order.
    for (chi::u32 qi = 0; qi < nq; ++qi) {
      if (is_l2) {
        faiss::maxheap_reorder(k, D_out + static_cast<size_t>(qi) * k,
                               I_out + static_cast<size_t>(qi) * k);
      } else {
        faiss::minheap_reorder(k, D_out + static_cast<size_t>(qi) * k,
                               I_out + static_cast<size_t>(qi) * k);
      }
    }
  }

  // Release any still-held list buffers (v0: all of them; v1: none — freed
  // inline after scanning) and update statistics.
  for (size_t i = 0; i < bufs.size(); ++i) {
    if (!bufs[i].IsNull()) {
      bytes_fetched += nbytes[i];
      ipc->FreeBuffer(bufs[i]);
      bufs[i] = ctp::ipc::FullPtr<char>();
    }
  }
  stat_searches_ += nq;
  stat_lists_fetched_ += nfetched;
  stat_bytes_fetched_ += bytes_fetched;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::Stats(ctp::ipc::FullPtr<StatsTask> task,
                               chi::RunContext& rctx) {
  CLIO_TASK_BODY_BEGIN
  task->searches_ = stat_searches_;
  task->lists_fetched_ = stat_lists_fetched_;
  task->bytes_fetched_ = stat_bytes_fetched_;
  task->fetch_wait_us_ = stat_fetch_wait_us_;
  task->scan_us_ = stat_scan_us_;
  if (task->reset_ != 0) {
    stat_searches_ = 0;
    stat_lists_fetched_ = 0;
    stat_bytes_fetched_ = 0;
    stat_fetch_wait_us_ = 0;
    stat_scan_us_ = 0;
  }
  task->SetReturnCode(0);
  (void)rctx;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::Monitor(ctp::ipc::FullPtr<MonitorTask> task,
                                 chi::RunContext& rctx) {
  CLIO_TASK_BODY_BEGIN
  // Report container statistics as msgpack.
  msgpack::sbuffer sbuf;
  msgpack::packer<msgpack::sbuffer> pk(sbuf);

  pk.pack_map(7);
  pk.pack("opened");
  pk.pack(opened_);
  pk.pack("pipeline_mode");
  pk.pack(static_cast<uint32_t>(pipeline_mode_));
  pk.pack("searches");
  pk.pack(static_cast<uint64_t>(stat_searches_));
  pk.pack("lists_fetched");
  pk.pack(static_cast<uint64_t>(stat_lists_fetched_));
  pk.pack("bytes_fetched");
  pk.pack(static_cast<uint64_t>(stat_bytes_fetched_));
  pk.pack("fetch_wait_us");
  pk.pack(static_cast<uint64_t>(stat_fetch_wait_us_));
  pk.pack("scan_us");
  pk.pack(static_cast<uint64_t>(stat_scan_us_));

  task->results_[container_id_] = std::string(sbuf.data(), sbuf.size());
  task->SetReturnCode(0);
  (void)rctx;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::Destroy(ctp::ipc::FullPtr<DestroyTask> task,
                                 chi::RunContext& rctx) {
  CLIO_TASK_BODY_BEGIN
  HLOG(kDebug, "faiss_ivf: Executing Destroy task - Pool ID: {}",
       task->target_pool_id_);

  // Initialize output values
  task->return_code_ = 0;
  task->error_message_ = "";

  // Drop the FAISS index state.
  ivf_ = nullptr;
  index_owner_.reset();
  sizes_.clear();
  opened_ = false;

  HLOG(kDebug, "faiss_ivf: Container destroyed successfully");
  (void)rctx;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::u64 Runtime::GetWorkRemaining() const {
  // No work tracking
  return 0;
}

//===========================================================================
// Task Serialization Method Implementations in autogen/faiss_ivf_lib_exec.cc
//===========================================================================

}  // namespace clio::run::faiss_ivf

// Define ChiMod entry points using CLIO_TASK_CC macro
CLIO_TASK_CC(clio::run::faiss_ivf::Runtime)
