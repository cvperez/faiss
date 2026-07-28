/*
 * cte_client — connect a client process to the CLIO runtime + CTE.
 *
 * Shared bootstrap used by the ingest tool and the benchmark harness.
 * Including this header also pulls in the CTE client / shared-memory
 * macros (CLIO_CTE_CLIENT, CLIO_IPC) those tools use.
 */

#pragma once

#include <clio_cte/core/core_client.h>
#include <clio_runtime/clio_runtime.h>

namespace faiss_iowarp {

// Connect this process to the CLIO runtime + CTE exactly once.
// Safe to call repeatedly from any thread. Returns false on failure.
bool EnsureIOWarpClient();

} // namespace faiss_iowarp
