/*
 * faiss_ivf ChiMod — client API.
 */

#ifndef FAISS_IVF_CLIENT_H_
#define FAISS_IVF_CLIENT_H_

#include <clio_runtime/clio_runtime.h>

#include "faiss_ivf_tasks.h"

/**
 * Client API for faiss_ivf
 *
 * Provides async methods for external programs to submit tasks to the
 * runtime. All methods return Future objects - call Wait() to block for
 * completion. Task cleanup is automatic when the Future goes out of scope
 * after Wait().
 */

namespace clio::run::faiss_ivf {

class Client : public chi::ContainerClient {
 public:
  /** Default constructor */
  CTP_CROSS_FUN Client() = default;

  /** Constructor with pool ID */
  CTP_CROSS_FUN explicit Client(const chi::PoolId& pool_id) { Init(pool_id); }

  /**
   * Create the container (asynchronous)
   * @param pool_query Pool routing information
   * @param pool_name Unique name for the pool (user-provided)
   * @param custom_pool_id Explicit pool ID for the pool being created
   * @param params Create parameters (pipeline mode)
   * @return Future for the CreateTask
   */
  chi::Future<CreateTask> AsyncCreate(
      const chi::PoolQuery& pool_query,
      const std::string& pool_name,
      const chi::PoolId& custom_pool_id,
      const CreateParams& params = CreateParams()) {
    auto* ipc_manager = CLIO_CPU_IPC;

    // CreateTask is a GetOrCreatePoolTask, which must be handled by admin pool
    // Pass 'this' as client pointer for PostWait callback
    auto task = ipc_manager->NewTask<CreateTask>(
        chi::CreateTaskId(),
        chi::kAdminPoolId,  // Send to admin pool for GetOrCreatePool processing
        pool_query,
        CreateParams::chimod_lib_name,  // chimod name from CreateParams
        pool_name,                      // user-provided pool name
        custom_pool_id,                 // target pool ID to create
        this,                           // Client pointer for PostWait
        params);                        // CreateParams with configuration

    return ipc_manager->Send(task);
  }

  /**
   * Monitor container state - asynchronous
   */
  chi::Future<MonitorTask> AsyncMonitor(const chi::PoolQuery& pool_query,
                                        const std::string& query) {
    auto* ipc_manager = CLIO_CPU_IPC;
    auto task = ipc_manager->NewTask<MonitorTask>(
        chi::CreateTaskId(), pool_id_, pool_query, query);
    return ipc_manager->Send(task);
  }

  /**
   * Open a FAISS IndexIVF (metadata only) and bind the CTE tag holding
   * the inverted lists (asynchronous)
   * @param pool_query Pool routing information
   * @param index_path Path to the FAISS index file
   * @param tag_name CTE tag name holding "sizes" and "list/<i>" blobs
   * @return Future for the OpenIndexTask
   */
  chi::Future<OpenIndexTask> AsyncOpenIndex(const chi::PoolQuery& pool_query,
                                            const std::string& index_path,
                                            const std::string& tag_name) {
    auto* ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<OpenIndexTask>(
        chi::CreateTaskId(), pool_id_, pool_query, index_path, tag_name);

    return ipc_manager->Send(task);
  }

  /**
   * Batched k-NN search (asynchronous)
   *
   * The caller allocates the three shared-memory buffers via
   * CLIO_IPC->AllocateBuffer(n) and passes buf.shm_.template Cast<void>().
   * queries_shm holds nq*d float32; dists_shm (nq*k float32) and
   * labels_shm (nq*k int64) receive the results. Buffers must stay alive
   * until the future completes.
   *
   * @param pool_query Pool routing information
   * @param nq Number of queries
   * @param k Neighbors per query
   * @param nprobe Lists probed per query
   * @param d Query dimensionality
   * @param mode Retained for wire compatibility; ignored by the runtime
   * @param queries_shm Shared-memory pointer to queries
   * @param dists_shm Shared-memory pointer for output distances
   * @param labels_shm Shared-memory pointer for output labels
   * @return Future for the SearchTask
   */
  chi::Future<SearchTask> AsyncSearch(const chi::PoolQuery& pool_query,
                                      chi::u32 nq, chi::u32 k,
                                      chi::u32 nprobe, chi::u32 d,
                                      chi::u32 mode,
                                      ctp::ipc::ShmPtr<> queries_shm,
                                      ctp::ipc::ShmPtr<> dists_shm,
                                      ctp::ipc::ShmPtr<> labels_shm) {
    auto* ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<SearchTask>(
        chi::CreateTaskId(), pool_id_, pool_query, nq, k, nprobe, d, mode,
        queries_shm, dists_shm, labels_shm);

    return ipc_manager->Send(task);
  }

  /**
   * Fetch container statistics (asynchronous)
   * @param pool_query Pool routing information
   * @param reset Non-zero: reset counters after reading
   * @return Future for the StatsTask
   */
  chi::Future<StatsTask> AsyncStats(const chi::PoolQuery& pool_query,
                                    chi::u32 reset) {
    auto* ipc_manager = CLIO_CPU_IPC;

    auto task = ipc_manager->NewTask<StatsTask>(
        chi::CreateTaskId(), pool_id_, pool_query, reset);

    return ipc_manager->Send(task);
  }
};

}  // namespace clio::run::faiss_ivf

#endif  // FAISS_IVF_CLIENT_H_
