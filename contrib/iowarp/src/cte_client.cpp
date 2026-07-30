#include "cte_client.h"

#include <chrono>
#include <thread>

namespace faiss_iowarp {

bool EnsureIOWarpClient() {
    static bool ok = [] {
        if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false)) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        return clio::cte::core::CLIO_CTE_CLIENT_INIT();
    }();
    return ok;
}

} // namespace faiss_iowarp
