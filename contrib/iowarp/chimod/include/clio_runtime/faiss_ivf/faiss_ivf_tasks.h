/*
 * faiss_ivf ChiMod — task definitions.
 *
 * Out-of-tree CLIO ChiMod that hosts FAISS IVF search inside the CLIO
 * runtime. Inverted lists live in CTE (tag/blob storage): one blob
 * "list/<i>" per non-empty list holding uint8_t codes[size*code_size]
 * immediately followed by int64 ids[size], plus a blob "sizes" =
 * int64[nlist] of list sizes.
 */

#ifndef FAISS_IVF_TASKS_H_
#define FAISS_IVF_TASKS_H_

#include <clio_runtime/clio_runtime.h>
#include "autogen/faiss_ivf_methods.h"
// Include admin tasks for GetOrCreatePoolTask / DestroyTask / MonitorTask
#include <clio_runtime/admin/admin_tasks.h>

#include <cstdint>
#include <vector>

/**
 * Task struct definitions for faiss_ivf
 *
 * Defines the tasks for Create, OpenIndex, Search and Stats methods.
 */

namespace clio::run::faiss_ivf {

using MonitorTask = clio::run::admin::MonitorTask;

/** Sentinel for SearchTask::mode_: use the container default. */
GLOBAL_CROSS_CONST clio::run::u32 kSearchModeDefault = 0xFFFFFFFF;

/** SearchTask::mode_ bit: owner-filtered broadcast — each container scans
 *  only the inverted lists whose placement hash lands on it, and returns
 *  per-query partial top-k via part_d_/part_i_ (merged in AggregateOut).
 *  Pair this bit with PoolQuery::Broadcast(); with a single-target query
 *  it returns only that node's partial results. */
GLOBAL_CROSS_CONST clio::run::u32 kSearchModeOwner = 0x1;
/** SearchTask::mode_ bit: metric is inner product (merge keeps LARGEST
 *  distances). AggregateOut has no access to the faiss index, so the
 *  merge direction must ride on the task. */
GLOBAL_CROSS_CONST clio::run::u32 kSearchModeIP = 0x2;

/** True iff mode selects owner-filtered search. The legacy sentinel
 *  (all bits set) must stay legacy, so test it explicitly. */
CTP_CROSS_FUN inline bool SearchOwnerMode(clio::run::u32 mode) {
  return mode != kSearchModeDefault && (mode & kSearchModeOwner) != 0;
}

/**
 * CreateParams for faiss_ivf chimod
 * Contains configuration parameters for faiss_ivf container creation
 */
struct CreateParams {
  // Retained for wire compatibility; IGNORED by the runtime. There is a
  // single search path: read each probed list from CTE on demand, scan,
  // free.
  clio::run::u32 pipeline_mode_;

  // Required: chimod library name for module manager
  static constexpr const char* chimod_lib_name = "clio_faiss_ivf";

  // Constructor with parameters (also serves as default)
  CreateParams(clio::run::u32 pipeline_mode = 0) : pipeline_mode_(pipeline_mode) {}

  // Serialization support for cereal
  template <class Archive>
  void serialize(Archive& ar) {
    ar(pipeline_mode_);
  }

  /**
   * Load configuration from PoolConfig (for compose mode).
   * No-op: faiss_ivf has no YAML-configurable state.
   */
  void LoadConfig(const clio::run::PoolConfig& pool_config) { (void)pool_config; }
};

/**
 * CreateTask - Initialize the faiss_ivf container
 * Type alias for GetOrCreatePoolTask with CreateParams (kGetOrCreatePool)
 */
using CreateTask = clio::run::admin::GetOrCreatePoolTask<CreateParams>;

/**
 * OpenIndexTask - Open a FAISS IndexIVF (metadata only, IVF data skipped)
 * and bind the CTE tag holding the inverted lists.
 */
struct OpenIndexTask : public clio::run::Task {
  IN clio::run::priv::string index_path_;  // Path to the FAISS index file
  IN clio::run::priv::string tag_name_;    // CTE tag holding "sizes" + "list/<i>"
  OUT clio::run::u64 ntotal_;              // Total number of indexed vectors
  OUT clio::run::u32 d_;                   // Vector dimensionality
  OUT clio::run::u32 nlist_;               // Number of inverted lists

  /** SHM default constructor */
  OpenIndexTask()
      : clio::run::Task(),
        index_path_(CLIO_PRIV_ALLOC),
        tag_name_(CLIO_PRIV_ALLOC),
        ntotal_(0),
        d_(0),
        nlist_(0) {}

  /** Emplace constructor */
  explicit OpenIndexTask(
      const clio::run::TaskId& task_node,
      const clio::run::PoolId& pool_id,
      const clio::run::PoolQuery& pool_query,
      const std::string& index_path,
      const std::string& tag_name)
      : clio::run::Task(task_node, pool_id, pool_query, Method::kOpenIndex),
        index_path_(CLIO_PRIV_ALLOC, index_path),
        tag_name_(CLIO_PRIV_ALLOC, tag_name),
        ntotal_(0),
        d_(0),
        nlist_(0) {
    // Initialize task
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kOpenIndex;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /** Destructor */
  ~OpenIndexTask() {}

  /** Serialize IN and INOUT parameters for network transfer */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive& ar) {
    Task::SerializeIn(ar);
    ar(index_path_, tag_name_);
  }

  /** Serialize OUT and INOUT parameters */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive& ar) {
    Task::SerializeOut(ar);
    ar(ntotal_, d_, nlist_);
  }

  /** Fix up priv::string SSO pointer after a raw copy (e.g. cudaMemcpy).
   * dev renamed the old FixupSsoPointer(); re-pointing the self-referential
   * SSO data_ is done via SetSsoState(GetSsoState()). Heap strings need no
   * fixup. Vestigial on this CPU-only build (never called by the runtime). */
  CTP_CROSS_FUN void FixupAfterCopy() {
    if (index_path_.UsingSso()) index_path_.SetSsoState(index_path_.GetSsoState());
    if (tag_name_.UsingSso()) tag_name_.SetSsoState(tag_name_.GetSsoState());
  }

  /** Copy from another OpenIndexTask */
  void Copy(const ctp::ipc::FullPtr<OpenIndexTask>& other) {
    // Copy base Task fields
    Task::Copy(other.template Cast<Task>());
    index_path_ = other->index_path_;
    tag_name_ = other->tag_name_;
    ntotal_ = other->ntotal_;
    d_ = other->d_;
    nlist_ = other->nlist_;
  }

  /** Aggregate replica results into this task */
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task>& other_base) {
    Task::AggregateOut(other_base);
    Copy(other_base.template Cast<OpenIndexTask>());
  }
};

/**
 * SearchTask - Batched k-NN search.
 *
 * queries_ holds nq*d float32 (input, BULK_XFER). distances_out_ (nq*k
 * float32) and labels_out_ (nq*k int64) are client-preallocated output
 * buffers (BULK_EXPOSE in, BULK_XFER out) — modeled on cte GetBlobTask.
 */
struct SearchTask : public clio::run::Task {
  IN clio::run::u32 nq_;                       // Number of queries
  IN clio::run::u32 k_;                        // Neighbors per query
  IN clio::run::u32 nprobe_;                   // Lists probed per query
  IN clio::run::u32 d_;                        // Query dimensionality
  IN clio::run::u32 mode_;               // kSearchMode* bits (or the sentinel)
  IN ctp::ipc::ShmPtr<> queries_;        // nq*d float32 (shared memory)
  IN ctp::ipc::ShmPtr<> distances_out_;  // nq*k float32, client-preallocated
  IN ctp::ipc::ShmPtr<> labels_out_;     // nq*k int64, client-preallocated
  // Owner mode only: per-query partial top-k, nq*k each, per-query-major,
  // sorted per query (L2 ascending / IP descending). Plain serialized
  // vectors, NOT bulk regions: broadcast replica subtasks share the
  // origin's ShmPtrs (Copy copies them raw), so bulk-XFERing D/I back
  // would land every replica on the client's buffers — a last-writer-wins
  // race. Vector OUT fields are the clio-core-endorsed broadcast shape
  // (see cte SemanticSearchTask); they live on whichever process owns the
  // task instance and cross boundaries only via SerializeOut.
  OUT std::vector<float> part_d_;
  OUT std::vector<int64_t> part_i_;

  /** SHM default constructor */
  CTP_CROSS_FUN SearchTask()
      : clio::run::Task(),
        nq_(0),
        k_(0),
        nprobe_(0),
        d_(0),
        mode_(kSearchModeDefault),
        queries_(ctp::ipc::ShmPtr<>::GetNull()),
        distances_out_(ctp::ipc::ShmPtr<>::GetNull()),
        labels_out_(ctp::ipc::ShmPtr<>::GetNull()) {}

  /** Emplace constructor */
  CTP_CROSS_FUN explicit SearchTask(
      const clio::run::TaskId& task_node,
      const clio::run::PoolId& pool_id,
      const clio::run::PoolQuery& pool_query,
      clio::run::u32 nq, clio::run::u32 k, clio::run::u32 nprobe, clio::run::u32 d, clio::run::u32 mode,
      ctp::ipc::ShmPtr<> queries,
      ctp::ipc::ShmPtr<> distances_out,
      ctp::ipc::ShmPtr<> labels_out)
      : clio::run::Task(task_node, pool_id, pool_query, Method::kSearch),
        nq_(nq),
        k_(k),
        nprobe_(nprobe),
        d_(d),
        mode_(mode),
        queries_(queries),
        distances_out_(distances_out),
        labels_out_(labels_out) {
    // Initialize task
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kSearch;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /** Destructor — frees buffers when this task owns them (receiver-side
   * copies made by LoadTaskArchive::bulk; see cte PutBlobTask/GetBlobTask
   * destructors for the rationale). Client-created tasks have
   * task_flags_.Clear() so the client's buffers are left alone. */
  CTP_CROSS_FUN ~SearchTask() {
#if !CTP_IS_DEVICE_PASS
    if (task_flags_.Any(TASK_DATA_OWNER)) {
      auto* ipc_manager = CLIO_CPU_IPC;
      if (ipc_manager) {
        if (!queries_.IsNull()) {
          ipc_manager->FreeBuffer(queries_.template Cast<char>());
        }
        if (!distances_out_.IsNull()) {
          ipc_manager->FreeBuffer(distances_out_.template Cast<char>());
        }
        if (!labels_out_.IsNull()) {
          ipc_manager->FreeBuffer(labels_out_.template Cast<char>());
        }
      }
    }
#endif
  }

  /** Serialize IN and INOUT parameters. */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive& ar) {
    Task::SerializeIn(ar);
    ar(nq_, k_, nprobe_, d_, mode_, queries_, distances_out_, labels_out_);
    ar.bulk(queries_,
            static_cast<clio::run::u64>(nq_) * d_ * sizeof(float), BULK_XFER);
    ar.bulk(distances_out_,
            static_cast<clio::run::u64>(nq_) * k_ * sizeof(float), BULK_EXPOSE);
    ar.bulk(labels_out_,
            static_cast<clio::run::u64>(nq_) * k_ * sizeof(int64_t), BULK_EXPOSE);
  }

  /** Serialize OUT and INOUT parameters. Only the result buffers travel
   * back — the IN-only ShmPtr fields must not be echoed (see cte
   * PutBlobTask::SerializeOut comment). mode_ is serialized FIRST so the
   * save and load sides branch identically from the wire, whatever their
   * local state (bulk framing follows call order, so both sides must
   * execute the same ar()/ar.bulk() sequence). */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive& ar) {
    Task::SerializeOut(ar);
    ar(mode_);
    if (SearchOwnerMode(mode_)) {
      // Partial top-k travels as plain serialized vectors; no bulk on
      // D/I (see part_d_ comment — replica bulk would alias the client
      // buffers and race).
      ar(part_d_, part_i_);
    } else {
      ar.bulk(distances_out_,
              static_cast<clio::run::u64>(nq_) * k_ * sizeof(float),
              BULK_XFER);
      ar.bulk(labels_out_,
              static_cast<clio::run::u64>(nq_) * k_ * sizeof(int64_t),
              BULK_XFER);
    }
  }

  /** Copy from another SearchTask */
  void Copy(const ctp::ipc::FullPtr<SearchTask>& other) {
    // Copy base Task fields
    Task::Copy(other.template Cast<Task>());
    nq_ = other->nq_;
    k_ = other->k_;
    nprobe_ = other->nprobe_;
    d_ = other->d_;
    mode_ = other->mode_;
    queries_ = other->queries_;
    distances_out_ = other->distances_out_;
    labels_out_ = other->labels_out_;
    part_d_ = other->part_d_;
    part_i_ = other->part_i_;
  }

  /** Aggregate replica results into this task.
   *
   * Owner mode: MERGE the replica's per-query sorted partial top-k into
   * this task's part_d_/part_i_ (Copy would keep only the last replica —
   * the bug cte SemanticSearchTask documents). Runs serially per replica
   * on the origin node's net-recv thread, so no locking; the strict total
   * order (distance, then id) makes the merged result independent of
   * replica arrival order. Single-node broadcast short-circuits locally
   * and never calls this. */
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task>& other_base) {
    Task::AggregateOut(other_base);  // rc / completer propagation
    auto other = other_base.template Cast<SearchTask>();
    if (!SearchOwnerMode(mode_)) {
      Copy(other);  // legacy last-writer semantics (single-target routing)
      return;
    }
    const size_t n = static_cast<size_t>(nq_) * k_;
    if (other->part_d_.size() != n || other->part_i_.size() != n) {
      return;  // failed/empty replica; its rc was propagated above
    }
    if (part_d_.size() != n) {
      // First replica to arrive: adopt wholesale.
      part_d_ = other->part_d_;
      part_i_ = other->part_i_;
      return;
    }
    // Per query: two-pointer merge of two sorted k-lists, keep best k.
    // Sentinel slots (L2: FLT_MAX/-1, IP: lowest()/-1) sort last and fall
    // out of the merge naturally.
    const bool ip = (mode_ & kSearchModeIP) != 0;
    std::vector<float> md(k_);
    std::vector<int64_t> mi(k_);
    for (clio::run::u32 qi = 0; qi < nq_; ++qi) {
      float* ad = part_d_.data() + static_cast<size_t>(qi) * k_;
      int64_t* ai = part_i_.data() + static_cast<size_t>(qi) * k_;
      const float* bd = other->part_d_.data() + static_cast<size_t>(qi) * k_;
      const int64_t* bi = other->part_i_.data() + static_cast<size_t>(qi) * k_;
      clio::run::u32 x = 0, y = 0;
      for (clio::run::u32 j = 0; j < k_; ++j) {
        bool take_a;
        if (ad[x] != bd[y]) {
          take_a = ip ? (ad[x] > bd[y]) : (ad[x] < bd[y]);
        } else {
          take_a = ai[x] <= bi[y];  // deterministic tie-break: smaller id
        }
        if (take_a) {
          md[j] = ad[x];
          mi[j] = ai[x];
          ++x;
        } else {
          md[j] = bd[y];
          mi[j] = bi[y];
          ++y;
        }
      }
      for (clio::run::u32 j = 0; j < k_; ++j) {
        ad[j] = md[j];
        ai[j] = mi[j];
      }
    }
  }
};

/**
 * StatsTask - Report (and optionally reset) container statistics.
 */
struct StatsTask : public clio::run::Task {
  IN clio::run::u32 reset_;           // Non-zero: reset counters after reading
  OUT clio::run::u64 searches_;       // Number of query vectors searched
  OUT clio::run::u64 lists_fetched_;  // Number of inverted lists fetched from CTE
  OUT clio::run::u64 bytes_fetched_;  // Bytes fetched from CTE
  OUT clio::run::u64 fetch_wait_us_;  // Microseconds spent waiting on CTE fetches
  OUT clio::run::u64 scan_us_;        // Microseconds spent scanning codes

  /** SHM default constructor */
  StatsTask()
      : clio::run::Task(),
        reset_(0),
        searches_(0),
        lists_fetched_(0),
        bytes_fetched_(0),
        fetch_wait_us_(0),
        scan_us_(0) {}

  /** Emplace constructor */
  explicit StatsTask(
      const clio::run::TaskId& task_node,
      const clio::run::PoolId& pool_id,
      const clio::run::PoolQuery& pool_query,
      clio::run::u32 reset)
      : clio::run::Task(task_node, pool_id, pool_query, Method::kStats),
        reset_(reset),
        searches_(0),
        lists_fetched_(0),
        bytes_fetched_(0),
        fetch_wait_us_(0),
        scan_us_(0) {
    // Initialize task
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kStats;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  template <typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive& ar) {
    Task::SerializeIn(ar);
    ar(reset_);
  }

  template <typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive& ar) {
    Task::SerializeOut(ar);
    ar(searches_, lists_fetched_, bytes_fetched_, fetch_wait_us_, scan_us_);
  }

  /** Copy from another StatsTask */
  void Copy(const ctp::ipc::FullPtr<StatsTask>& other) {
    // Copy base Task fields
    Task::Copy(other.template Cast<Task>());
    reset_ = other->reset_;
    searches_ = other->searches_;
    lists_fetched_ = other->lists_fetched_;
    bytes_fetched_ = other->bytes_fetched_;
    fetch_wait_us_ = other->fetch_wait_us_;
    scan_us_ = other->scan_us_;
  }

  /** Aggregate replica results into this task. Counters are SUMMED across
   *  containers (a broadcast Stats on N nodes must report cluster totals;
   *  Copy would report only the last replica's). */
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task>& other_base) {
    Task::AggregateOut(other_base);
    auto other = other_base.template Cast<StatsTask>();
    searches_ += other->searches_;
    lists_fetched_ += other->lists_fetched_;
    bytes_fetched_ += other->bytes_fetched_;
    fetch_wait_us_ += other->fetch_wait_us_;
    scan_us_ += other->scan_us_;
  }
};

/**
 * Standard DestroyTask for faiss_ivf
 */
using DestroyTask = clio::run::admin::DestroyTask;

}  // namespace clio::run::faiss_ivf

#endif  // FAISS_IVF_TASKS_H_
