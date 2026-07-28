/*
 * cte_smoke.cpp — minimal standalone CTE client round-trip.
 *
 * Purpose: verify that code compiled against the pinned clio-core v2.1.0
 * headers links and runs against the iowarp-core pip wheel's shared
 * libraries, before any FAISS integration is written on top.
 *
 * Sequence (mirrors clio_cte_bench.cc): init runtime client -> get/create
 * tag -> AsyncPutBlob 4 KiB -> AsyncGetBlob into a second buffer ->
 * byte-compare -> exit 0 on success.
 *
 * Requires a running runtime:
 *   CLIO_SERVER_CONF=config/local_smoke.yaml clio_run start &
 */

#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>

#include <clio_cte/core/core_client.h>
#include <clio_runtime/clio_runtime.h>

int main() {
    if (!chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, false)) {
        std::fprintf(stderr, "smoke: runtime client init failed\n");
        return 1;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
        std::fprintf(stderr, "smoke: CTE client init failed\n");
        return 1;
    }

    auto* cte = CLIO_CTE_CLIENT;
    constexpr size_t kSize = 4096;

    auto tag_fut = cte->AsyncGetOrCreateTag("faiss_iowarp_smoke");
    tag_fut.Wait();
    auto tag_id = tag_fut->tag_id_;

    auto put_buf = CLIO_IPC->AllocateBuffer(kSize);
    auto get_buf = CLIO_IPC->AllocateBuffer(kSize);
    for (size_t i = 0; i < kSize; ++i) {
        put_buf.ptr_[i] = static_cast<char>(i * 31 + 7);
    }
    std::memset(get_buf.ptr_, 0, kSize);
    ctp::ipc::ShmPtr<> put_ptr = put_buf.shm_.template Cast<void>();
    ctp::ipc::ShmPtr<> get_ptr = get_buf.shm_.template Cast<void>();

    auto put_fut = cte->AsyncPutBlob(tag_id, "blob0", 0, kSize, put_ptr);
    put_fut.Wait();
    if (put_fut->return_code_.load() != 0) {
        std::fprintf(stderr, "smoke: PutBlob rc=%u\n",
                     put_fut->return_code_.load());
        return 1;
    }

    auto get_fut = cte->AsyncGetBlob(tag_id, "blob0", 0, kSize, 0, get_ptr);
    get_fut.Wait();
    if (get_fut->return_code_.load() != 0) {
        std::fprintf(stderr, "smoke: GetBlob rc=%u\n",
                     get_fut->return_code_.load());
        return 1;
    }

    int rc = std::memcmp(put_buf.ptr_, get_buf.ptr_, kSize);
    CLIO_IPC->FreeBuffer(put_buf);
    CLIO_IPC->FreeBuffer(get_buf);

    if (rc != 0) {
        std::fprintf(stderr, "smoke: FAIL — payload mismatch\n");
        return 1;
    }
    std::printf("smoke: PASS — 4 KiB put/get round-trip byte-identical\n");
    return 0;
}
