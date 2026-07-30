/*
 * faiss_ivf ChiMod — runtime implementation.
 *
 * Server-side task processing: OpenIndex loads a FAISS IndexIVF (metadata
 * only) and binds the CTE tag holding the inverted lists; Search coarse-
 * quantizes the queries, then for each probed list reads it from CTE,
 * scans it, and frees the buffer. The per-list read copies the bytes out
 * of the tier — the cost the zero-copy read API proposed to clio-core
 * would remove (see UPSTREAM_PROPOSAL_IOWARP.md), letting the module scan
 * the RAM tier's own bytes in place.
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
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace clio::run::faiss_ivf {

namespace {

/** Current steady-clock time in microseconds. */
inline clio::run::u64 NowUs() {
  return static_cast<clio::run::u64>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

// Scan one fetched list for all (query, coarse_dis) pairs touching it,
// in parallel across per-thread scanners. Plain function on purpose:
// gcc 11 ICEs when an OpenMP region sits inside a C++20 coroutine body.
// Safe because every pair updates a distinct per-query heap.
bool ScanListParallel(
    const std::vector<std::pair<clio::run::u32, float>>& plist,
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

// Copy a burst of lists straight out of the RAM tier via the client's
// zero-IPC read (TryReadBlobShm), nthreads-wide. Plain function for the
// same gcc-11 OMP-in-coroutine reason as ScanListParallel. Thread-safe
// per clio-core dev core_client.h: seqlocked record lookup, shared_mutex-
// guarded MapRamBdev (#817), placement_gen_ re-validated across the copy.
// fast[j] stays 0 on a miss (file-tier / truncated / moved blob) and the
// caller falls back to AsyncGetBlob for that list.
void FetchBurstShm(clio::cte::core::Client& cte,
                   const clio::cte::core::TagId& tag,
                   const std::vector<int64_t>& lists,
                   const std::vector<int64_t>& sizes, size_t code_size,
                   size_t b0, size_t b1, int nthreads,
                   const std::vector<clio::run::u32>& slot_of,
                   std::vector<ctp::ipc::FullPtr<char>>& slabs,
                   std::vector<uint8_t>& fast) {
#pragma omp parallel for num_threads(nthreads) schedule(dynamic, 1)
  for (size_t j = b0; j < b1; ++j) {
    const int64_t l = lists[j];
    const size_t sz = static_cast<size_t>(sizes[l]);
    const size_t bytes = sz * (code_size + sizeof(int64_t));
    fast[j] = cte.TryReadBlobShm(tag,
                                 std::string("list/") + std::to_string(l),
                                 slabs[slot_of[j]].ptr_, bytes)
                  ? 1
                  : 0;
  }
}

}  // namespace

// Method implementations for Runtime class

// Virtual method implementations (Init, Run, Del, SaveTask, LoadTask,
// NewCopy, Aggregate) are in autogen/faiss_ivf_lib_exec.cc

//===========================================================================
// Method implementations
//===========================================================================

clio::run::TaskResume Runtime::Create(clio::run::shared_ptr<CreateTask>& task) {
  CLIO_TASK_BODY_BEGIN
  HLOG(kDebug, "faiss_ivf: Executing Create task for pool {}", task->pool_id_);

  // Container is already initialized via Init() before Create is called.
  // CreateParams::pipeline_mode_ is accepted for client compatibility but
  // ignored: there is a single search path (read each probed list from CTE
  // on demand, scan, free).
  CreateParams params = task->GetParams();
  (void)params;

  HLOG(kDebug, "faiss_ivf: Container created for pool: {}", pool_name_);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::OpenIndex(clio::run::shared_ptr<OpenIndexTask>& task) {
  CLIO_TASK_BODY_BEGIN
  clio::run::ScopedCoMutex lock(open_mu_);

  std::string index_path = task->index_path_.str();
  std::string tag_name = task->tag_name_.str();

  if (opened_ && index_path == opened_index_path_ &&
      tag_name == opened_tag_name_) {
    // Same volume already open: report the current index metadata.
    task->ntotal_ = static_cast<clio::run::u64>(ivf_->ntotal);
    task->d_ = static_cast<clio::run::u32>(ivf_->d);
    task->nlist_ = static_cast<clio::run::u32>(ivf_->nlist);
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
  const clio::run::u64 sizes_bytes =
      static_cast<clio::run::u64>(nlist) * sizeof(int64_t);
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

  clio::run::u64 ntotal = 0;
  for (size_t i = 0; i < nlist; ++i) {
    ntotal += static_cast<clio::run::u64>(sizes_[i]);
  }
  ivf->ntotal = static_cast<faiss::idx_t>(ntotal);

  // The inverted lists stay in CTE; Search reads each probed list on
  // demand. OpenIndex only binds the tag and records the list sizes.

  // Commit state.
  index_owner_ = std::move(owner);
  ivf_ = ivf;
  tag_id_ = tag_id;
  opened_ = true;
  opened_index_path_ = index_path;
  opened_tag_name_ = tag_name;

  task->ntotal_ = ntotal;
  task->d_ = static_cast<clio::run::u32>(ivf_->d);
  task->nlist_ = static_cast<clio::run::u32>(nlist);
  task->SetReturnCode(0);

  HLOG(kInfo, "faiss_ivf: opened '{}' (d={}, nlist={}, ntotal={}, tag='{}')",
       index_path, task->d_, task->nlist_, ntotal, tag_name);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Search(clio::run::shared_ptr<SearchTask>& task) {
  CLIO_TASK_BODY_BEGIN
  if (!opened_ || ivf_ == nullptr) {
    task->SetReturnCode(1);
    CLIO_CO_RETURN;
  }

  const clio::run::u32 nq = task->nq_;
  const clio::run::u32 k = task->k_;
  clio::run::u32 nprobe = task->nprobe_;
  if (nq == 0 || k == 0 || nprobe == 0 ||
      task->d_ != static_cast<clio::run::u32>(ivf_->d)) {
    task->SetReturnCode(2);
    CLIO_CO_RETURN;
  }
  if (nprobe > ivf_->nlist) {
    nprobe = static_cast<clio::run::u32>(ivf_->nlist);
  }
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
  // (query, coarse distance).
  std::unordered_map<int64_t, std::vector<std::pair<clio::run::u32, float>>> probes;
  std::vector<int64_t> lists;
  for (clio::run::u32 qi = 0; qi < nq; ++qi) {
    for (clio::run::u32 j = 0; j < nprobe; ++j) {
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

  const size_t code_size = ivf_->code_size;
  const size_t ntoscan = lists.size();
  HLOG(kInfo, "faiss_ivf: Search nq={} k={} nprobe={} unique_lists={}", nq,
       k, nprobe, ntoscan);
  task->SetReturnCode(0);
  const bool is_l2 = (ivf_->metric_type == faiss::METRIC_L2);

  // Per-query result heaps live directly in the output buffers.
  const float init_dis = is_l2 ? std::numeric_limits<float>::max()
                               : std::numeric_limits<float>::lowest();
  for (size_t i = 0; i < static_cast<size_t>(nq) * k; ++i) {
    D_out[i] = init_dis;
    I_out[i] = -1;
  }

  // One scanner per scan thread: set_query/set_list mutate scanner state,
  // so threads must not share one. kScanThreads is the 8-thread scan
  // budget; the OMP region in ScanListParallel contains no co_await, so it
  // never suspends mid-parallelism.
  constexpr int kScanThreads = 8;
  std::vector<std::unique_ptr<faiss::InvertedListScanner>> scanners;
  try {
    for (int t = 0; t < kScanThreads; ++t) {
      scanners.emplace_back(ivf_->get_InvertedListScanner(false));
    }
  } catch (const std::exception& e) {
    HLOG(kError, "faiss_ivf: get_InvertedListScanner failed: {}", e.what());
    task->SetReturnCode(5);
    CLIO_CO_RETURN;
  }

  // Read each probed list from CTE, scan it, free it. The copy out of the
  // tier is the per-search cost the proposed zero-copy read API would
  // remove (scan the RAM tier's own bytes in place). Keep up to
  // kMaxInflight AsyncGetBlobs outstanding — never issue unbounded tasks
  // from a handler, the runtime queues are shared with the CTE handlers.
  // Poll-then-finalize: only CLIO_CO_AWAIT a future that IsComplete() (the
  // await returns immediately), scan on arrival, free the buffer, and
  // yield only while fetches are outstanding — awaiting a still-pending
  // future with more sub-tasks in flight can resume a destroyed coroutine
  // frame (SIGSEGV in ResumeCoroutine).
  constexpr size_t kMaxInflight = 64;
  // Zero-IPC burst width: on dev, AsyncGetBlob's shm fast path memcpys
  // INLINE on this coroutine's worker (core_client.h TryShmGet), so the
  // async pipeline no longer overlaps reads. Instead, copy RAM-resident
  // lists in kFetchBatch-sized bursts across the scan threads, and keep
  // the RPC pipeline only for misses.
  constexpr size_t kFetchBatch = 16;
  // Fixed pool of reusable list buffers ("slabs"), sized to the largest
  // probed list. Per-list AllocateBuffer/FreeBuffer must NOT be used here:
  // the dev allocator does not recycle multi-MB buffers — segments grow by
  // roughly the bytes pushed and stay mapped, so one cold pass at nb50M
  // grew the daemon by ~14 GB (102 x ~140 MB segments) and the OOM killer
  // took the node. The pool caps buffer memory at kMaxInflight x max_list.
  size_t max_bytes = 0;
  for (size_t j = 0; j < ntoscan; ++j) {
    const size_t szj = static_cast<size_t>(sizes_[lists[j]]);
    max_bytes = std::max(max_bytes, szj * (code_size + sizeof(int64_t)));
  }
  const size_t nslabs = std::min(kMaxInflight, ntoscan);
  std::vector<ctp::ipc::FullPtr<char>> slabs(nslabs);
  for (size_t s = 0; s < nslabs; ++s) {
    slabs[s] = ipc->AllocateBuffer(max_bytes);
    if (slabs[s].IsNull()) {
      HLOG(kError, "faiss_ivf: AllocateBuffer({}) failed during search",
           max_bytes);
      for (size_t t = 0; t < s; ++t) {
        ipc->FreeBuffer(slabs[t]);
      }
      task->SetReturnCode(5);
      CLIO_CO_RETURN;
    }
  }
  std::vector<clio::run::u32> free_slots(nslabs);
  for (size_t s = 0; s < nslabs; ++s) {
    free_slots[s] = static_cast<clio::run::u32>(nslabs - 1 - s);
  }
  std::vector<clio::run::u32> slot_of(ntoscan, 0);
  std::vector<clio::run::Future<clio::cte::core::GetBlobTask>> futs(ntoscan);
  // uint8_t (not vector<bool>): written concurrently from the OMP burst.
  std::vector<uint8_t> fast(ntoscan, 0);
  std::vector<bool> done(ntoscan, false);
  // Attach the shm metadata cache ONCE, serially, before any parallel
  // readers: AttachShmCache writes shm_root_ unsynchronized. The env var
  // is a kill-switch back to the pure RPC pipeline, no rebuild needed.
  const bool shm_direct =
      std::getenv("FAISS_IVF_NO_SHM_DIRECT") == nullptr &&
      !clio::cte::core::Client::ForceNetEnv() &&
      (cte_.HasShmCache() || cte_.AttachShmCache());
  const clio::run::u64 t_loop0 = NowUs();
  clio::run::u64 scan_us = 0;
  size_t issued = 0;
  size_t completed = 0;
  while (completed < issued || issued < ntoscan) {
    while (issued < ntoscan && issued - completed < kMaxInflight) {
      // Grab a batch of slots from the pool, zero-IPC-copy the whole
      // batch across the scan threads, then issue RPC gets for the
      // misses only. issued - completed < kMaxInflight bounds the slots
      // in use, so free_slots can never underflow here.
      const size_t b0 = issued;
      const size_t b1 =
          std::min({ntoscan, b0 + kFetchBatch, completed + kMaxInflight});
      for (size_t j = b0; j < b1; ++j) {
        slot_of[j] = free_slots.back();
        free_slots.pop_back();
      }
      if (shm_direct) {
        FetchBurstShm(cte_, tag_id_, lists, sizes_, code_size, b0, b1,
                      kScanThreads, slot_of, slabs, fast);
      }
      for (size_t j = b0; j < b1; ++j) {
        if (fast[j]) {
          continue;  // bytes already in the slab, no task needed
        }
        const size_t sz = static_cast<size_t>(sizes_[lists[j]]);
        const clio::run::u64 bytes =
            static_cast<clio::run::u64>(sz) * (code_size + sizeof(int64_t));
        futs[j] = cte_.AsyncGetBlob(
            tag_id_, std::string("list/") + std::to_string(lists[j]), 0,
            bytes, 0, slabs[slot_of[j]].shm_.template Cast<void>());
      }
      issued = b1;
    }
    bool progressed = false;
    for (size_t i = 0; i < issued; ++i) {
      if (done[i] || (!fast[i] && !futs[i].IsComplete())) {
        continue;
      }
      if (!fast[i]) {
        CLIO_CO_AWAIT(futs[i]);  // completed: returns immediately
      }
      done[i] = true;
      ++completed;
      progressed = true;
      const int64_t l = lists[i];
      const size_t sz = static_cast<size_t>(sizes_[l]);
      if (!fast[i] && futs[i]->GetReturnCode() != 0) {
        HLOG(kError, "faiss_ivf: GetBlob('list/{}') failed (rc={})", l,
             futs[i]->GetReturnCode());
        task->SetReturnCode(5);
      } else {
        stat_lists_fetched_ += 1;
        stat_bytes_fetched_ +=
            static_cast<clio::run::u64>(sz) * (code_size + sizeof(int64_t));
        const char* base = slabs[slot_of[i]].ptr_;
        const uint8_t* codes = reinterpret_cast<const uint8_t*>(base);
        const faiss::idx_t* ids =
            reinterpret_cast<const faiss::idx_t*>(base + sz * code_size);
        const clio::run::u64 s0 = NowUs();
        if (!ScanListParallel(probes[l], scanners, q, ivf_->d, l, sz, codes,
                              ids, D_out, I_out, k)) {
          HLOG(kError, "faiss_ivf: scan failed on list {}", l);
          task->SetReturnCode(5);
        }
        scan_us += NowUs() - s0;
      }
      // Return the slot to the pool (slabs are reused, never freed
      // per-list).
      free_slots.push_back(slot_of[i]);
      if (!fast[i]) {
        // Drop the future NOW: on dev a Future owns its GetBlobTask via
        // shared_ptr, and holding every awaited future until the end of
        // the search pins one shm task per probed list (~16k at nb100M)
        // until the client allocators are exhausted and the daemon
        // stalls. Resetting caps live tasks at kMaxInflight.
        futs[i] = clio::run::Future<clio::cte::core::GetBlobTask>();
      }
    }
    if (!progressed && completed < issued) {
      CLIO_CO_AWAIT(clio::run::yield());
    }
  }
  for (size_t s = 0; s < nslabs; ++s) {
    ipc->FreeBuffer(slabs[s]);
  }
  const clio::run::u64 loop_us = NowUs() - t_loop0;
  stat_scan_us_ += scan_us;
  stat_fetch_wait_us_ += (loop_us > scan_us ? loop_us - scan_us : 0);

  // Sort each query's heap into ascending (L2) / descending (IP) order.
  for (clio::run::u32 qi = 0; qi < nq; ++qi) {
    if (is_l2) {
      faiss::maxheap_reorder(k, D_out + static_cast<size_t>(qi) * k,
                             I_out + static_cast<size_t>(qi) * k);
    } else {
      faiss::minheap_reorder(k, D_out + static_cast<size_t>(qi) * k,
                             I_out + static_cast<size_t>(qi) * k);
    }
  }

  stat_searches_ += nq;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Stats(clio::run::shared_ptr<StatsTask>& task) {
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
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Monitor(clio::run::shared_ptr<MonitorTask>& task) {
  CLIO_TASK_BODY_BEGIN
  // Report container statistics as msgpack.
  msgpack::sbuffer sbuf;
  msgpack::packer<msgpack::sbuffer> pk(sbuf);

  pk.pack_map(6);
  pk.pack("opened");
  pk.pack(opened_);
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
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Destroy(clio::run::shared_ptr<DestroyTask>& task) {
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
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::u64 Runtime::GetWorkRemaining() const {
  // No work tracking
  return 0;
}

//===========================================================================
// Task Serialization Method Implementations in autogen/faiss_ivf_lib_exec.cc
//===========================================================================

}  // namespace clio::run::faiss_ivf

// Define ChiMod entry points using CLIO_TASK_CC macro
CLIO_TASK_CC(clio::run::faiss_ivf::Runtime)
