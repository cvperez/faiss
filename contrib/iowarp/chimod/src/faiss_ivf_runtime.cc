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
#include <clio_runtime/pool_manager.h>

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
#include <fstream>
#include <functional>
#include <limits>
#include <sstream>
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

// /proc/diskstats snapshot for one whole-disk device. Whitespace tokens,
// 0-indexed: [2]=name, [5]=sectors read, [9]=sectors written,
// [12]=io_ticks ms — the same fields telemetry_sampler.py reads. Returns
// false (outputs zeroed) if the device line is absent.
bool ReadDiskstats(const std::string& dev, clio::run::u64* rd_sectors,
                   clio::run::u64* wr_sectors, clio::run::u64* io_ticks_ms) {
  *rd_sectors = *wr_sectors = *io_ticks_ms = 0;
  std::ifstream f("/proc/diskstats");
  std::string line;
  while (std::getline(f, line)) {
    std::istringstream is(line);
    std::vector<std::string> tok;
    std::string t;
    while (is >> t) {
      tok.push_back(t);
    }
    if (tok.size() > 12 && tok[2] == dev) {
      *rd_sectors = strtoull(tok[5].c_str(), nullptr, 10);
      *wr_sectors = strtoull(tok[9].c_str(), nullptr, 10);
      *io_ticks_ms = strtoull(tok[12].c_str(), nullptr, 10);
      return true;
    }
  }
  return false;
}

// Tier device to sample in Stats. Env FAISS_IVF_DISK_DEV (set for the
// daemon by the launch scripts); default matches telemetry_sampler.py.
const std::string& DiskDev() {
  static const std::string dev = [] {
    const char* e = std::getenv("FAISS_IVF_DISK_DEV");
    return std::string(e ? e : "nvme0n1");
  }();
  return dev;
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
  list_local_.clear();

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

  // Owner-mode locality map. The faiss pool and the CTE pool are created
  // independently but with the same rule (one container per node,
  // ContainerId == NodeId), so hash % num_containers resolves to the same
  // node in both — an invariant to assert, not assume (a pool created
  // while the host list differed would silently break locality).
  auto* pm = CLIO_POOL_MANAGER;
  const clio::run::PoolInfo* faiss_pi = pm->GetPoolInfo(pool_id_);
  const clio::run::PoolInfo* cte_pi =
      pm->GetPoolInfo(clio::cte::core::kCtePoolId);
  if (faiss_pi == nullptr || cte_pi == nullptr ||
      faiss_pi->num_containers_ == 0 ||
      faiss_pi->num_containers_ != cte_pi->num_containers_) {
    HLOG(kError,
         "faiss_ivf: pool layout mismatch (faiss containers={} cte "
         "containers={})",
         faiss_pi ? faiss_pi->num_containers_ : 0,
         cte_pi ? cte_pi->num_containers_ : 0);
    task->SetReturnCode(7);
    CLIO_CO_RETURN;
  }
  // Same hash as the CTE core's HashBlobToContainer (owner routing) — it
  // is private to the core Runtime, so replicate it, as the cte cache
  // module does (cache_runtime.cc IsBlobOwnerLocal). If core ever changes
  // the hash only LOCALITY degrades (lists still partition exactly-once
  // across containers); correctness is unaffected. ListOwnerContainer is
  // the one shared definition — the Add client routes with the same
  // function, so ownership and routing can never drift apart.
  {
    const clio::run::u32 num_containers = faiss_pi->num_containers_;
    list_local_.assign(nlist, 0);
    for (size_t l = 0; l < nlist; ++l) {
      list_local_[l] =
          (ListOwnerContainer(tag_id.major_, tag_id.minor_,
                              static_cast<int64_t>(l),
                              num_containers) == container_id_)
              ? 1
              : 0;
    }
  }

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
  if (adding_.load(std::memory_order_acquire)) {
    // The client contract is drain-searches-before-add; a Search arriving
    // mid-add means that contract broke. Fail loudly rather than scan a
    // half-written blob / torn sizes_.
    HLOG(kError, "faiss_ivf: Search received while an Add is in flight");
    task->SetReturnCode(9);
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
  // Owner-filtered broadcast mode: scan only the lists this container
  // owns; export sorted partials for the origin-side AggregateOut merge.
  const bool owner_mode = SearchOwnerMode(task->mode_);
  const bool metric_is_l2 = (ivf_->metric_type == faiss::METRIC_L2);
  if (owner_mode &&
      (((task->mode_ & kSearchModeIP) != 0) == metric_is_l2)) {
    // The IP bit drives the merge direction in AggregateOut, which cannot
    // see the index — a mismatch would merge in the wrong order silently.
    HLOG(kError, "faiss_ivf: mode/metric mismatch (mode={:#x}, metric={})",
         task->mode_, static_cast<int>(ivf_->metric_type));
    task->SetReturnCode(7);
    CLIO_CO_RETURN;
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
      if (owner_mode && !list_local_[l]) {
        continue;  // peer-owned list: its owner's replica scans it
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
  HLOG(kInfo,
       "faiss_ivf: Search nq={} k={} nprobe={} unique_lists={} owner_mode={}",
       nq, k, nprobe, ntoscan, owner_mode ? 1 : 0);
  task->SetReturnCode(0);
  const bool is_l2 = metric_is_l2;

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
  // Poll-then-finalize: sweep for IsComplete() futures and scan those
  // first; when a sweep makes no progress, await the oldest pending RPC
  // future directly (see the #856 note at the bottom of the loop — the
  // historical yield-based park starves on dev >= d7b1f053).
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
  size_t got = 0;
  {
    // Reuse cached slabs (see slab_cache_ in the header — per-search
    // Allocate/Free OOM-kills the daemon at per-query search rates).
    std::lock_guard<std::mutex> lk(slab_mu_);
    if (slab_bytes_ < max_bytes) {
      for (auto& s : slab_cache_) {
        ipc->FreeBuffer(s);
      }
      slab_cache_.clear();
      // 2x headroom: under a write workload the max probed list grows a
      // few percent per write window, and every record-high would
      // otherwise re-key the cache — each re-key strands the freed slabs
      // in never-recycled segments (~0.9 GB a pop, the daemon-death creep
      // diagnosed in job 23298). With 2x, ~20 windows of uniform growth
      // fit without a single re-key.
      slab_bytes_ = static_cast<clio::run::u64>(max_bytes) * 2;
    }
    while (got < nslabs && !slab_cache_.empty()) {
      slabs[got++] = slab_cache_.back();
      slab_cache_.pop_back();
    }
  }
  for (size_t s = got; s < nslabs; ++s) {
    // Allocate at the CACHE capacity (>= max_bytes), so every slab is
    // interchangeable in the cache regardless of which search made it.
    slabs[s] = ipc->AllocateBuffer(slab_bytes_);
    if (slabs[s].IsNull()) {
      HLOG(kError, "faiss_ivf: AllocateBuffer({}) failed during search",
           (clio::run::u64)slab_bytes_);
      std::lock_guard<std::mutex> lk(slab_mu_);
      for (size_t t = 0; t < s; ++t) {
        slab_cache_.push_back(slabs[t]);
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
  constexpr uint8_t kGetRetries = 3;
  std::vector<uint8_t> retries(ntoscan, 0);
  // Attach the shm metadata cache ONCE, serially, before any parallel
  // readers: AttachShmCache writes shm_root_ unsynchronized. The env var
  // is a kill-switch back to the pure RPC pipeline, no rebuild needed.
  const bool shm_direct =
      std::getenv("FAISS_IVF_NO_SHM_DIRECT") == nullptr &&
      !clio::cte::core::Client::ForceNetEnv() &&
      (cte_.HasShmCache() || cte_.AttachShmCache());
  // FAISS_IVF_VERIFY_SHM=1: diagnostic mode — after every zero-IPC fast-path
  // hit, ALSO fetch the same list via the RPC GetBlob path and byte-compare.
  // Proves (or refutes) that the raw shm read returns the same bytes the
  // official API returns. Roughly halves throughput; diagnostics only.
  const bool verify_shm =
      shm_direct && std::getenv("FAISS_IVF_VERIFY_SHM") != nullptr;
  ctp::ipc::FullPtr<char> vslab;
  clio::run::u64 verify_diverge = 0;
  clio::run::u64 verify_rpcfail = 0;
  if (verify_shm) {
    vslab = ipc->AllocateBuffer(max_bytes);
    if (vslab.IsNull()) {
      HLOG(kError, "faiss_ivf: VERIFY_SHM scratch AllocateBuffer failed");
      task->SetReturnCode(5);
      std::lock_guard<std::mutex> lk(slab_mu_);
      for (size_t t = 0; t < nslabs; ++t) {
        slab_cache_.push_back(slabs[t]);
      }
      CLIO_CO_RETURN;
    }
  }
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
      // Bounded retry on RPC failure: sustained NVMe-heavy campaigns can
      // wedge a bdev route transiently (RouteLocal rc=4 storms observed at
      // nb130M after ~5 volume-scale passes) — re-issue the get up to
      // kGetRetries times before failing the search.
      if (!fast[i] && futs[i]->GetReturnCode() != 0 && retries[i] < kGetRetries) {
        ++retries[i];
        HLOG(kError,
             "faiss_ivf: GetBlob('list/{}') rc={} — retry {}/{}",
             lists[i], futs[i]->GetReturnCode(), retries[i],
             (clio::run::u32)kGetRetries);
        const size_t rsz = static_cast<size_t>(sizes_[lists[i]]);
        const clio::run::u64 rbytes =
            static_cast<clio::run::u64>(rsz) * (code_size + sizeof(int64_t));
        futs[i] = cte_.AsyncGetBlob(
            tag_id_, std::string("list/") + std::to_string(lists[i]), 0,
            rbytes, 0, slabs[slot_of[i]].shm_.template Cast<void>());
        progressed = true;
        continue;  // not done; the re-issued future completes later
      }
      done[i] = true;
      ++completed;
      progressed = true;
      const int64_t l = lists[i];
      const size_t sz = static_cast<size_t>(sizes_[l]);
      if (fast[i] && verify_shm) {
        // Cross-check the fast-path bytes against the RPC path. Direct
        // await (safe and required on dev >= d7b1f053, see #856 note at
        // the bottom of the loop).
        const clio::run::u64 vbytes =
            static_cast<clio::run::u64>(sz) * (code_size + sizeof(int64_t));
        auto vfut = cte_.AsyncGetBlob(
            tag_id_, std::string("list/") + std::to_string(l), 0, vbytes, 0,
            vslab.shm_.template Cast<void>());
        CLIO_CO_AWAIT(vfut);
        if (vfut->GetReturnCode() != 0) {
          HLOG(kError,
               "faiss_ivf: VERIFY_SHM rpc get failed list {} (rc={}, {} B)",
               l, vfut->GetReturnCode(), vbytes);
          ++verify_rpcfail;
        } else if (std::memcmp(slabs[slot_of[i]].ptr_, vslab.ptr_,
                               static_cast<size_t>(vbytes)) != 0) {
          HLOG(kError,
               "faiss_ivf: VERIFY_SHM DIVERGENCE list {} ({} B): shm fast "
               "path bytes != RPC GetBlob bytes",
               l, vbytes);
          ++verify_diverge;
        }
        vfut = clio::run::Future<clio::cte::core::GetBlobTask>();
      }
      if (!fast[i] && futs[i]->GetReturnCode() != 0) {
        HLOG(kError, "faiss_ivf: GetBlob('list/{}') failed (rc={}, {} B)", l,
             futs[i]->GetReturnCode(),
             static_cast<clio::run::u64>(sz) * (code_size + sizeof(int64_t)));
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
      // No list became ready this sweep — await the OLDEST still-pending
      // RPC future directly. On dev >= d7b1f053 (#856) the event-resume
      // guard is strict: a suspended fiber is resumed only by the exact
      // future it awaits, and the old poll-then-yield pattern can starve
      // (observed as an infinite cold-pass hang at nb100M: the yielded
      // Search fiber pinned worker 0 while its lane was rescue-adopted
      // away, GetBlob completions no longer resume a yield-parked
      // parent). Direct await of a pending subtask future is the pattern
      // the CTE core itself uses on this HEAD, and #856's exactly-once
      // completion makes it safe (no spurious double-resume of this
      // frame — the pre-#856 SIGSEGV hazard this loop was originally
      // designed around).
      for (size_t i = 0; i < issued; ++i) {
        if (!done[i] && !fast[i]) {
          CLIO_CO_AWAIT(futs[i]);
          break;
        }
      }
    }
  }
  {
    // Return the slabs to the cache instead of FreeBuffer (never
    // recycled). Bounded: at most kSlabCacheCap retained — that is the
    // peak concurrent demand (4 exp2 readers x 64), and anything beyond
    // would have been resident at peak anyway.
    constexpr size_t kSlabCacheCap = 512;
    std::lock_guard<std::mutex> lk(slab_mu_);
    for (size_t s = 0; s < nslabs; ++s) {
      if (slab_cache_.size() < kSlabCacheCap) {
        slab_cache_.push_back(slabs[s]);
      } else {
        ipc->FreeBuffer(slabs[s]);
      }
    }
  }
  if (verify_shm) {
    HLOG(kInfo,
         "faiss_ivf: VERIFY_SHM summary: {} divergence(s), {} rpc "
         "failure(s) across fast-path hits",
         verify_diverge, verify_rpcfail);
    if (verify_diverge > 0 || verify_rpcfail > 0) {
      task->SetReturnCode(6);  // distinct rc: fast path diverged from RPC
    }
    ipc->FreeBuffer(vslab);
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

  if (owner_mode && task->IsRemote()) {
    // Broadcast replica on a remote daemon: D_out/I_out here are
    // daemon-local EXPOSE scratch (LoadTaskArchive::bulk allocated them,
    // freed by ~SearchTask via TASK_DATA_OWNER) — the client never sees
    // them. Export the sorted per-query partials; they travel back via
    // SerializeOut and merge in AggregateOut on the origin. When executed
    // locally (single-node broadcast short-circuit, or legacy routing)
    // D_out/I_out ARE the client's buffers and part_* stays empty — the
    // bench falls back to the buffers when the vectors are empty.
    const size_t n = static_cast<size_t>(nq) * k;
    task->part_d_.assign(D_out, D_out + n);
    task->part_i_.assign(I_out, I_out + n);
  }

  stat_searches_ += nq;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Add(clio::run::shared_ptr<AddTask>& task) {
  CLIO_TASK_BODY_BEGIN
  if (!opened_ || ivf_ == nullptr) {
    task->SetReturnCode(1);
    CLIO_CO_RETURN;
  }
  const clio::run::u32 n = task->n_;
  const size_t code_size = ivf_->code_size;
  const size_t nseg = task->list_ids_.size();
  const size_t nlist = ivf_->nlist;
  if (n == 0 || nseg == 0 || task->list_offs_.size() != nseg + 1 ||
      task->d_ != static_cast<clio::run::u32>(ivf_->d) ||
      task->code_size_ != static_cast<clio::run::u32>(code_size) ||
      task->list_offs_.front() != 0 ||
      task->list_offs_.back() != static_cast<int64_t>(n)) {
    task->SetReturnCode(2);
    CLIO_CO_RETURN;
  }
  auto* ipc = CLIO_IPC;
  const uint8_t* new_codes = reinterpret_cast<const uint8_t*>(
      ipc->ToFullPtr<char>(task->codes_.template Cast<char>()).ptr_);
  const int64_t* new_ids = reinterpret_cast<const int64_t*>(
      ipc->ToFullPtr<char>(task->ids_.template Cast<char>()).ptr_);
  if (new_codes == nullptr || new_ids == nullptr) {
    task->SetReturnCode(2);
    CLIO_CO_RETURN;
  }
  // Validate every segment BEFORE touching any blob: monotone offsets,
  // in-range list ids, and — the routing contract — this container owns
  // every target list. A non-owned list means the client's
  // ListOwnerContainer routing diverged from list_local_: fail the whole
  // task loudly (rc=8), nothing partially applied.
  //
  // The RMW is TAIL-ONLY: in the codes||ids layout an append leaves
  // [0, old_sz*code_size) untouched — only the tail (new codes + the
  // shifted old ids + new ids) changes. Reading and putting just that
  // tail cuts bytes pushed per uniform 2.5M-vector window ~15x (full-blob
  // rewrites pushed the whole volume every window and grew the daemon to
  // OOM within 6 windows — allocator gotcha at PutBlob scale).
  size_t max_tail_bytes = 0;
  for (size_t j = 0; j < nseg; ++j) {
    const int64_t l = task->list_ids_[j];
    const int64_t cnt = task->list_offs_[j + 1] - task->list_offs_[j];
    if (l < 0 || l >= static_cast<int64_t>(nlist) || cnt <= 0) {
      task->SetReturnCode(2);
      CLIO_CO_RETURN;
    }
    if (!list_local_[l]) {
      HLOG(kError,
           "faiss_ivf: Add misroute — list {} not owned by container {}", l,
           container_id_);
      task->SetReturnCode(8);
      CLIO_CO_RETURN;
    }
    const size_t old_sz = static_cast<size_t>(sizes_[l]);
    const size_t tail = cnt * (code_size + sizeof(int64_t)) +
        old_sz * sizeof(int64_t);
    max_tail_bytes = std::max(max_tail_bytes, tail);
  }

  // One reusable slab sized for the largest grown list, drawn from the
  // shared slab cache (never per-task Allocate/Free — the dev allocator
  // does not recycle multi-MB buffers; see the Search slab comment). A
  // grown list can exceed the search slabs' capacity, in which case the
  // cache is dropped and re-keyed to the larger size.
  ctp::ipc::FullPtr<char> slab;
  {
    std::lock_guard<std::mutex> lk(slab_mu_);
    if (slab_bytes_ < max_tail_bytes) {
      for (auto& s : slab_cache_) {
        ipc->FreeBuffer(s);
      }
      slab_cache_.clear();
      // Same 2x headroom as Search (re-keys strand segments).
      slab_bytes_ = static_cast<clio::run::u64>(max_tail_bytes) * 2;
    }
    if (!slab_cache_.empty()) {
      slab = slab_cache_.back();
      slab_cache_.pop_back();
    }
  }
  if (slab.IsNull()) {
    slab = ipc->AllocateBuffer(slab_bytes_);
  }
  if (slab.IsNull()) {
    HLOG(kError, "faiss_ivf: Add AllocateBuffer({}) failed", max_tail_bytes);
    task->SetReturnCode(5);
    CLIO_CO_RETURN;
  }

  adding_.store(true, std::memory_order_release);
  const clio::run::u64 t0 = NowUs();
  clio::run::u64 add_bytes = 0;
  clio::run::u64 added = 0;
  bool failed = false;
  for (size_t j = 0; j < nseg && !failed; ++j) {
    const int64_t l = task->list_ids_[j];
    const size_t cnt =
        static_cast<size_t>(task->list_offs_[j + 1] - task->list_offs_[j]);
    const size_t off = static_cast<size_t>(task->list_offs_[j]);
    const size_t old_sz = static_cast<size_t>(sizes_[l]);
    const size_t new_sz = old_sz + cnt;
    const std::string name = "list/" + std::to_string(l);
    // Tail-only RMW. New blob = codes_old ‖ codes_new ‖ ids_old ‖ ids_new;
    // bytes before old_sz*code_size are untouched. The changed tail is
    // [codes_new][ids_old][ids_new] starting at offset old_sz*code_size.
    // Fetch ids_old straight into its FINAL slab position (no memmove),
    // splice the new codes/ids around it, put the tail at its offset —
    // CTE extends the blob in place (block-based ExtendBlob +
    // ModifyExistingData).
    const clio::run::u64 tail_off =
        static_cast<clio::run::u64>(old_sz) * code_size;
    const clio::run::u64 tail_bytes =
        static_cast<clio::run::u64>(cnt) * (code_size + sizeof(int64_t)) +
        static_cast<clio::run::u64>(old_sz) * sizeof(int64_t);
    if (old_sz > 0) {
      const clio::run::u64 ids_bytes =
          static_cast<clio::run::u64>(old_sz) * sizeof(int64_t);
      int grc = -1;
      for (int attempt = 0; attempt < 3 && grc != 0; ++attempt) {
        auto gfut = cte_.AsyncGetBlob(tag_id_, name, tail_off, ids_bytes, 0,
                                      slab.shm_.template Cast<void>());
        CLIO_CO_AWAIT(gfut);
        grc = static_cast<int>(gfut->GetReturnCode());
        if (grc != 0) {
          HLOG(kError,
               "faiss_ivf: Add GetBlob('{}') rc={} attempt {} ({} B)", name,
               grc, attempt + 1, ids_bytes);
        }
      }
      if (grc != 0) {
        task->SetReturnCode(5);
        failed = true;
        break;
      }
      // ids_old to its final slot (dst > src, memmove handles overlap),
      // then codes_new over the vacated prefix.
      std::memmove(slab.ptr_ + cnt * code_size, slab.ptr_,
                   static_cast<size_t>(ids_bytes));
    }
    std::memcpy(slab.ptr_, new_codes + off * code_size, cnt * code_size);
    std::memcpy(slab.ptr_ + cnt * code_size + old_sz * sizeof(int64_t),
                new_ids + off, cnt * sizeof(int64_t));
    int prc = -1;
    for (int attempt = 0; attempt < 3 && prc != 0; ++attempt) {
      auto pfut = cte_.AsyncPutBlob(tag_id_, name, tail_off, tail_bytes,
                                    slab.shm_.template Cast<void>());
      CLIO_CO_AWAIT(pfut);
      prc = static_cast<int>(pfut->GetReturnCode());
      if (prc != 0) {
        HLOG(kError, "faiss_ivf: Add PutBlob('{}') rc={} attempt {} ({} B)",
             name, prc, attempt + 1, tail_bytes);
      }
    }
    if (prc != 0) {
      task->SetReturnCode(5);
      failed = true;
      break;
    }
    sizes_[l] = static_cast<int64_t>(new_sz);
    ivf_->ntotal += static_cast<faiss::idx_t>(cnt);
    added += cnt;
    add_bytes += tail_bytes;
  }
  {
    std::lock_guard<std::mutex> lk(slab_mu_);
    slab_cache_.push_back(slab);
  }
  adding_.store(false, std::memory_order_release);

  stat_adds_ += 1;
  stat_add_vectors_ += added;
  stat_add_bytes_ += add_bytes;
  stat_add_us_ += NowUs() - t0;
  task->added_ = added;
  task->ntotal_after_ = static_cast<clio::run::u64>(ivf_->ntotal);
  if (!failed) {
    task->SetReturnCode(0);
    HLOG(kInfo, "faiss_ivf: Add appended {} vectors across {} lists", added,
         nseg);
  }
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
  task->adds_ = stat_adds_;
  task->add_vectors_ = stat_add_vectors_;
  task->add_bytes_ = stat_add_bytes_;
  task->add_us_ = stat_add_us_;
  // Node-local disk snapshot: cumulative kernel counters (never reset by
  // reset_ — the caller computes deltas), summed across containers by
  // AggregateOut. containers_ = 1 here makes the aggregate count nodes.
  task->containers_ = 1;
  ReadDiskstats(DiskDev(), &task->disk_rd_sectors_, &task->disk_wr_sectors_,
                &task->disk_io_ticks_ms_);
  if (task->reset_ != 0) {
    stat_searches_ = 0;
    stat_lists_fetched_ = 0;
    stat_bytes_fetched_ = 0;
    stat_fetch_wait_us_ = 0;
    stat_scan_us_ = 0;
    stat_adds_ = 0;
    stat_add_vectors_ = 0;
    stat_add_bytes_ = 0;
    stat_add_us_ = 0;
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

  pk.pack_map(10);
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
  pk.pack("adds");
  pk.pack(static_cast<uint64_t>(stat_adds_));
  pk.pack("add_vectors");
  pk.pack(static_cast<uint64_t>(stat_add_vectors_));
  pk.pack("add_bytes");
  pk.pack(static_cast<uint64_t>(stat_add_bytes_));
  pk.pack("add_us");
  pk.pack(static_cast<uint64_t>(stat_add_us_));

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
  list_local_.clear();
  opened_ = false;
  {
    auto* ipc = CLIO_IPC;
    std::lock_guard<std::mutex> lk(slab_mu_);
    for (auto& s : slab_cache_) {
      ipc->FreeBuffer(s);
    }
    slab_cache_.clear();
    slab_bytes_ = 0;
  }

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
