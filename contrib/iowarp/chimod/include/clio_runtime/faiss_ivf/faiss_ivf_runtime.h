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
class Runtime : public chi::Container {
 public:
  // CreateParams type used by CLIO_TASK_CC macro for lib_name access
  using CreateParams = clio::run::faiss_ivf::CreateParams;

 private:
  // FAISS index state (metadata only; IVF data lives in CTE)
  std::unique_ptr<faiss::Index> index_owner_;
  faiss::IndexIVF* ivf_ = nullptr;
  std::vector<int64_t> sizes_;  // per-list sizes, from the "sizes" blob
  clio::cte::core::TagId tag_id_;
  bool opened_ = false;
  std::string opened_index_path_;  // volume identity: OpenIndex with a
  std::string opened_tag_name_;    // different path/tag replaces the state
  chi::CoMutex open_mu_;

  // In-process CTE client bound directly to the canonical CTE pool
  // (kCtePoolId = 512.0). NEVER call CLIO_CTE_CLIENT_INIT from a handler:
  // its blocking create_task.Wait() deadlocks the cooperative worker.
  clio::cte::core::Client cte_;

  // Container default search mode (from CreateParams::pipeline_mode_)
  chi::u32 pipeline_mode_ = 0;

  // Serializes v0's temporary invlists swap on the shared ivf_ (v0 is the
  // ablation path; v1 has no shared mutable FAISS state and runs fully
  // concurrently).
  chi::CoMutex v0_mu_;

  // Statistics. Atomics: concurrent SearchTasks may run on different
  // workers (the bench splits query batches into parallel tasks).
  std::atomic<chi::u64> stat_searches_{0};
  std::atomic<chi::u64> stat_lists_fetched_{0};
  std::atomic<chi::u64> stat_bytes_fetched_{0};
  std::atomic<chi::u64> stat_fetch_wait_us_{0};
  std::atomic<chi::u64> stat_scan_us_{0};

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
  void Init(const chi::PoolId& pool_id, const std::string& pool_name,
            chi::u32 container_id = 0) override;

  /**
   * Execute a method on a task
   */
  chi::TaskResume Run(chi::u32 method, ctp::ipc::FullPtr<chi::Task> task_ptr,
                      chi::RunContext& rctx) override;

  //===========================================================================
  // Method implementations
  //===========================================================================

  /** Handle Create task */
  chi::TaskResume Create(ctp::ipc::FullPtr<CreateTask> task,
                         chi::RunContext& rctx);

  /** Handle OpenIndex task */
  chi::TaskResume OpenIndex(ctp::ipc::FullPtr<OpenIndexTask> task,
                            chi::RunContext& rctx);

  /** Handle Search task */
  chi::TaskResume Search(ctp::ipc::FullPtr<SearchTask> task,
                         chi::RunContext& rctx);

  /** Handle Stats task */
  chi::TaskResume Stats(ctp::ipc::FullPtr<StatsTask> task,
                        chi::RunContext& rctx);

  /** Handle Monitor task */
  chi::TaskResume Monitor(ctp::ipc::FullPtr<MonitorTask> task,
                          chi::RunContext& rctx);

  /** Handle Destroy task */
  chi::TaskResume Destroy(ctp::ipc::FullPtr<DestroyTask> task,
                          chi::RunContext& rctx);

  /**
   * Get remaining work count for this container
   */
  chi::u64 GetWorkRemaining() const override;

  //===========================================================================
  // Task Serialization Methods (implemented in autogen/faiss_ivf_lib_exec.cc)
  //===========================================================================

  /** Serialize task parameters for network transfer (unified method) */
  void SaveTask(chi::u32 method, chi::SaveTaskArchive& archive,
                ctp::ipc::FullPtr<chi::Task> task_ptr) override;

  /** Deserialize task parameters into an existing task */
  void LoadTask(chi::u32 method, chi::LoadTaskArchive& archive,
                ctp::ipc::FullPtr<chi::Task> task_ptr) override;

  /** Allocate and deserialize task parameters from network transfer */
  ctp::ipc::FullPtr<chi::Task> AllocLoadTask(
      chi::u32 method, chi::LoadTaskArchive& archive) override;

  /** Deserialize task input parameters using LocalSerialize */
  void LocalLoadTask(chi::u32 method, chi::DefaultLoadArchive& archive,
                     ctp::ipc::FullPtr<chi::Task> task_ptr) override;

  /** Allocate and deserialize task input parameters using LocalSerialize */
  ctp::ipc::FullPtr<chi::Task> LocalAllocLoadTask(
      chi::u32 method, chi::DefaultLoadArchive& archive) override;

  /** Serialize task output parameters using LocalSerialize */
  void LocalSaveTask(chi::u32 method, chi::DefaultSaveArchive& archive,
                     ctp::ipc::FullPtr<chi::Task> task_ptr) override;

  /** Create a new copy of a task (deep copy for distributed execution) */
  ctp::ipc::FullPtr<chi::Task> NewCopyTask(
      chi::u32 method, ctp::ipc::FullPtr<chi::Task> orig_task_ptr,
      bool deep) override;

  /** Create a new task of the specified method type */
  ctp::ipc::FullPtr<chi::Task> NewTask(chi::u32 method) override;
  void Aggregate(chi::u32 method, ctp::ipc::FullPtr<chi::Task> orig_task,
                 const ctp::ipc::FullPtr<chi::Task>& replica_task) override;
  void DelTask(chi::u32 method, ctp::ipc::FullPtr<chi::Task> task_ptr) override;
};

}  // namespace clio::run::faiss_ivf

#endif  // FAISS_IVF_RUNTIME_H_
