/*
 * faiss_ivf ChiMod — runtime (server-side container) declaration.
 */

#ifndef FAISS_IVF_RUNTIME_H_
#define FAISS_IVF_RUNTIME_H_

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/container.h>
#include <clio_runtime/comutex.h>

#include <clio_cte/core/core_client.h>  // clio::cte::core::TagId, CTE client

#include <faiss/Index.h>
#include <faiss/IndexIVF.h>

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "faiss_ivf_tasks.h"
#include "autogen/faiss_ivf_methods.h"
#include "faiss_ivf_client.h"

namespace clio::run::faiss_ivf {

/**
 * Runtime implementation for faiss_ivf container
 */
class Runtime : public clio::run::Container {
 public:
  // CreateParams type used by CLIO_TASK_CC macro for lib_name access
  using CreateParams = clio::run::faiss_ivf::CreateParams;

 private:
  // FAISS index state (metadata only; IVF data lives in CTE)
  std::unique_ptr<faiss::Index> index_owner_;
  faiss::IndexIVF* ivf_ = nullptr;
  std::vector<int64_t> sizes_;  // per-list sizes, from the "sizes" blob
  // Owner-mode locality map, precomputed at OpenIndex: list_local_[l] != 0
  // iff blob "list/<l>"'s placement hash % num_containers == container_id_
  // (== this node, since ContainerId == NodeId). Partitions the lists
  // exactly-once across the pool's containers; on one node every list is
  // local.
  std::vector<uint8_t> list_local_;
  clio::cte::core::TagId tag_id_;
  bool opened_ = false;
  std::string opened_index_path_;  // volume identity: OpenIndex with a
  std::string opened_tag_name_;    // different path/tag replaces the state
  clio::run::CoMutex open_mu_;

  // In-process CTE client bound directly to the canonical CTE pool
  // (kCtePoolId = 512.0). NEVER call CLIO_CTE_CLIENT_INIT from a handler:
  // its blocking create_task.Wait() deadlocks the cooperative worker.
  // Search reads its probed lists through this client on demand.
  clio::cte::core::Client cte_;

  // Statistics. Atomics: concurrent SearchTasks may run on different
  // workers (the bench splits query batches into parallel tasks).
  std::atomic<clio::run::u64> stat_searches_{0};
  std::atomic<clio::run::u64> stat_lists_fetched_{0};   // lists read from CTE
  std::atomic<clio::run::u64> stat_bytes_fetched_{0};   // bytes read from CTE
  std::atomic<clio::run::u64> stat_fetch_wait_us_{0};   // time waiting on CTE reads
  std::atomic<clio::run::u64> stat_scan_us_{0};         // time scanning codes

  // Client for making calls to this ChiMod
  Client client_;

 public:
  /** Constructor */
  Runtime() = default;

  /** Destructor */
  virtual ~Runtime() = default;

  /**
   * Initialize container with pool information
   */
  void Init(const clio::run::PoolId& pool_id, const std::string& pool_name,
            clio::run::u32 container_id = 0) override;

  /**
   * Execute a method on a task (dev: no RunContext; task is a shared_ptr)
   */
  clio::run::TaskResume Run(clio::run::u32 method,
                            clio::run::shared_ptr<clio::run::Task> task_ptr) override;

  //===========================================================================
  // Method implementations
  //===========================================================================

  /** Handle Create task */
  clio::run::TaskResume Create(clio::run::shared_ptr<CreateTask>& task);

  /** Handle OpenIndex task */
  clio::run::TaskResume OpenIndex(clio::run::shared_ptr<OpenIndexTask>& task);

  /** Handle Search task */
  clio::run::TaskResume Search(clio::run::shared_ptr<SearchTask>& task);

  /** Handle Stats task */
  clio::run::TaskResume Stats(clio::run::shared_ptr<StatsTask>& task);

  /** Handle Monitor task */
  clio::run::TaskResume Monitor(clio::run::shared_ptr<MonitorTask>& task);

  /** Handle Destroy task */
  clio::run::TaskResume Destroy(clio::run::shared_ptr<DestroyTask>& task);

  /**
   * Get remaining work count for this container
   */
  clio::run::u64 GetWorkRemaining() const override;

  //===========================================================================
  // Task Serialization Methods (implemented in autogen/faiss_ivf_lib_exec.cc)
  //===========================================================================

  /** Serialize task parameters for network transfer (unified method) */
  void SaveTask(clio::run::u32 method, clio::run::SaveTaskArchive& archive,
                clio::run::shared_ptr<clio::run::Task>& task_ptr) override;

  /** Deserialize task parameters into an existing task */
  void LoadTask(clio::run::u32 method, clio::run::LoadTaskArchive& archive,
                clio::run::shared_ptr<clio::run::Task>& task_ptr) override;

  /** Allocate and deserialize task parameters from network transfer */
  clio::run::shared_ptr<clio::run::Task> AllocLoadTask(
      clio::run::u32 method, clio::run::LoadTaskArchive& archive) override;

  /** Deserialize task input parameters using LocalSerialize */
  void LocalLoadTask(clio::run::u32 method, clio::run::DefaultLoadArchive& archive,
                     clio::run::shared_ptr<clio::run::Task>& task_ptr) override;

  /** Allocate and deserialize task input parameters using LocalSerialize */
  clio::run::shared_ptr<clio::run::Task> LocalAllocLoadTask(
      clio::run::u32 method, clio::run::DefaultLoadArchive& archive) override;

  /** Serialize task output parameters using LocalSerialize */
  void LocalSaveTask(clio::run::u32 method, clio::run::DefaultSaveArchive& archive,
                     clio::run::shared_ptr<clio::run::Task>& task_ptr) override;

  /** Create a new copy of a task (deep copy for distributed execution) */
  clio::run::shared_ptr<clio::run::Task> NewCopyTask(
      clio::run::u32 method, clio::run::shared_ptr<clio::run::Task>& orig_task_ptr,
      bool deep) override;

  /** Create a new task of the specified method type */
  clio::run::shared_ptr<clio::run::Task> NewTask(clio::run::u32 method) override;

  /** Aggregate replica OUT fields (N->1 gather) */
  void AggregateOut(clio::run::u32 method, clio::run::shared_ptr<clio::run::Task>& orig_task,
                    const clio::run::shared_ptr<clio::run::Task>& replica_task) override;

  /** Aggregate member IN fields into the collective task */
  void AggregateIn(clio::run::u32 method, clio::run::shared_ptr<clio::run::Task>& agg_task,
                   const clio::run::shared_ptr<clio::run::Task>& member_task) override;
};

}  // namespace clio::run::faiss_ivf

#endif  // FAISS_IVF_RUNTIME_H_
