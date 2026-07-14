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

/**
 * Task struct definitions for faiss_ivf
 *
 * Defines the tasks for Create, OpenIndex, Search and Stats methods.
 */

namespace clio::run::faiss_ivf {

using MonitorTask = clio::run::admin::MonitorTask;

/** Sentinel for SearchTask::mode_: use the container default. */
GLOBAL_CROSS_CONST chi::u32 kSearchModeDefault = 0xFFFFFFFF;

/**
 * CreateParams for faiss_ivf chimod
 * Contains configuration parameters for faiss_ivf container creation
 */
struct CreateParams {
  // 0 = v0 (fetch-all then search_preassigned),
  // 1 = v1 (pipelined fetch + scan)
  chi::u32 pipeline_mode_;

  // Required: chimod library name for module manager
  static constexpr const char* chimod_lib_name = "clio_faiss_ivf";

  // Constructor with parameters (also serves as default)
  CreateParams(chi::u32 pipeline_mode = 0) : pipeline_mode_(pipeline_mode) {}

  // Serialization support for cereal
  template <class Archive>
  void serialize(Archive& ar) {
    ar(pipeline_mode_);
  }

  /**
   * Load configuration from PoolConfig (for compose mode).
   * No-op: faiss_ivf has no YAML-configurable state.
   */
  void LoadConfig(const chi::PoolConfig& pool_config) { (void)pool_config; }
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
struct OpenIndexTask : public chi::Task {
  IN chi::priv::string index_path_;  // Path to the FAISS index file
  IN chi::priv::string tag_name_;    // CTE tag holding "sizes" + "list/<i>"
  OUT chi::u64 ntotal_;              // Total number of indexed vectors
  OUT chi::u32 d_;                   // Vector dimensionality
  OUT chi::u32 nlist_;               // Number of inverted lists

  /** SHM default constructor */
  OpenIndexTask()
      : chi::Task(),
        index_path_(CLIO_PRIV_ALLOC),
        tag_name_(CLIO_PRIV_ALLOC),
        ntotal_(0),
        d_(0),
        nlist_(0) {}

  /** Emplace constructor */
  explicit OpenIndexTask(
      const chi::TaskId& task_node,
      const chi::PoolId& pool_id,
      const chi::PoolQuery& pool_query,
      const std::string& index_path,
      const std::string& tag_name)
      : chi::Task(task_node, pool_id, pool_query, Method::kOpenIndex),
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

  /** Fix up priv::string SSO pointer after cudaMemcpy */
  CTP_CROSS_FUN void FixupAfterCopy() {
    index_path_.FixupSsoPointer();
    tag_name_.FixupSsoPointer();
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
  void Aggregate(const ctp::ipc::FullPtr<chi::Task>& other_base) {
    Task::Aggregate(other_base);
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
struct SearchTask : public chi::Task {
  IN chi::u32 nq_;                       // Number of queries
  IN chi::u32 k_;                        // Neighbors per query
  IN chi::u32 nprobe_;                   // Lists probed per query
  IN chi::u32 d_;                        // Query dimensionality
  IN chi::u32 mode_;                     // 0=v0, 1=v1, kSearchModeDefault
  IN ctp::ipc::ShmPtr<> queries_;        // nq*d float32 (shared memory)
  IN ctp::ipc::ShmPtr<> distances_out_;  // nq*k float32, client-preallocated
  IN ctp::ipc::ShmPtr<> labels_out_;     // nq*k int64, client-preallocated

  /** SHM default constructor */
  CTP_CROSS_FUN SearchTask()
      : chi::Task(),
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
      const chi::TaskId& task_node,
      const chi::PoolId& pool_id,
      const chi::PoolQuery& pool_query,
      chi::u32 nq, chi::u32 k, chi::u32 nprobe, chi::u32 d, chi::u32 mode,
      ctp::ipc::ShmPtr<> queries,
      ctp::ipc::ShmPtr<> distances_out,
      ctp::ipc::ShmPtr<> labels_out)
      : chi::Task(task_node, pool_id, pool_query, Method::kSearch),
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
            static_cast<chi::u64>(nq_) * d_ * sizeof(float), BULK_XFER);
    ar.bulk(distances_out_,
            static_cast<chi::u64>(nq_) * k_ * sizeof(float), BULK_EXPOSE);
    ar.bulk(labels_out_,
            static_cast<chi::u64>(nq_) * k_ * sizeof(int64_t), BULK_EXPOSE);
  }

  /** Serialize OUT and INOUT parameters. Only the result buffers travel
   * back — the IN-only ShmPtr fields must not be echoed (see cte
   * PutBlobTask::SerializeOut comment). */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive& ar) {
    Task::SerializeOut(ar);
    ar.bulk(distances_out_,
            static_cast<chi::u64>(nq_) * k_ * sizeof(float), BULK_XFER);
    ar.bulk(labels_out_,
            static_cast<chi::u64>(nq_) * k_ * sizeof(int64_t), BULK_XFER);
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
  }

  /** Aggregate replica results into this task */
  void Aggregate(const ctp::ipc::FullPtr<chi::Task>& other_base) {
    Task::Aggregate(other_base);
    Copy(other_base.template Cast<SearchTask>());
  }
};

/**
 * StatsTask - Report (and optionally reset) container statistics.
 */
struct StatsTask : public chi::Task {
  IN chi::u32 reset_;           // Non-zero: reset counters after reading
  OUT chi::u64 searches_;       // Number of query vectors searched
  OUT chi::u64 lists_fetched_;  // Number of inverted lists fetched from CTE
  OUT chi::u64 bytes_fetched_;  // Bytes fetched from CTE
  OUT chi::u64 fetch_wait_us_;  // Microseconds spent waiting on CTE fetches
  OUT chi::u64 scan_us_;        // Microseconds spent scanning codes

  /** SHM default constructor */
  StatsTask()
      : chi::Task(),
        reset_(0),
        searches_(0),
        lists_fetched_(0),
        bytes_fetched_(0),
        fetch_wait_us_(0),
        scan_us_(0) {}

  /** Emplace constructor */
  explicit StatsTask(
      const chi::TaskId& task_node,
      const chi::PoolId& pool_id,
      const chi::PoolQuery& pool_query,
      chi::u32 reset)
      : chi::Task(task_node, pool_id, pool_query, Method::kStats),
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

  /** Aggregate replica results into this task */
  void Aggregate(const ctp::ipc::FullPtr<chi::Task>& other_base) {
    Task::Aggregate(other_base);
    Copy(other_base.template Cast<StatsTask>());
  }
};

/**
 * Standard DestroyTask for faiss_ivf
 */
using DestroyTask = clio::run::admin::DestroyTask;

}  // namespace clio::run::faiss_ivf

#endif  // FAISS_IVF_TASKS_H_
