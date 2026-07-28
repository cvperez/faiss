#include "ivf_cte_ingest.h"

#include <cstring>
#include <deque>
#include <string>

#include <faiss/impl/FaissAssert.h>

#include "cte_client.h"

using faiss::idx_t;

namespace faiss_iowarp {

namespace {

struct InFlightPut {
    ctp::ipc::FullPtr<char> buf;
    chi::Future<clio::cte::core::PutBlobTask> fut;
};

void drain_one(std::deque<InFlightPut>& q) {
    auto& front = q.front();
    front.fut.Wait();
    FAISS_THROW_IF_NOT_MSG(
            front.fut->return_code_.load() == 0, "CTE PutBlob failed");
    CLIO_IPC->FreeBuffer(front.buf);
    q.pop_front();
}

} // namespace

size_t IngestIvfToCte(
        const faiss::InvertedLists* src,
        const std::string& tag_name,
        size_t batch) {
    FAISS_THROW_IF_NOT_MSG(
            EnsureIOWarpClient(), "IOWarp client init failed");
    auto* cte = CLIO_CTE_CLIENT;

    const size_t nlist = src->nlist;
    const size_t code_size = src->code_size;

    auto tag_fut = cte->AsyncGetOrCreateTag(tag_name);
    tag_fut.Wait();
    auto tag_id = tag_fut->tag_id_;

    std::deque<InFlightPut> inflight;
    size_t put_bytes = 0;
    for (size_t l = 0; l < nlist; ++l) {
        size_t sz = src->list_size(l);
        if (sz == 0) {
            continue;
        }
        size_t bytes = sz * (code_size + sizeof(idx_t));
        InFlightPut p;
        p.buf = CLIO_IPC->AllocateBuffer(bytes);
        {
            faiss::InvertedLists::ScopedCodes codes(src, l);
            faiss::InvertedLists::ScopedIds ids(src, l);
            std::memcpy(p.buf.ptr_, codes.get(), sz * code_size);
            std::memcpy(
                    p.buf.ptr_ + sz * code_size,
                    ids.get(),
                    sz * sizeof(idx_t));
        }
        ctp::ipc::ShmPtr<> sp = p.buf.shm_.template Cast<void>();
        std::string name = "list/" + std::to_string(l);
        p.fut = cte->AsyncPutBlob(tag_id, name, 0, bytes, sp);
        inflight.push_back(std::move(p));
        if (inflight.size() >= batch) {
            drain_one(inflight);
        }
        put_bytes += bytes;
    }
    while (!inflight.empty()) {
        drain_one(inflight);
    }

    // "sizes" blob last: its presence signals a completed ingest.
    {
        size_t bytes = nlist * sizeof(idx_t);
        auto buf = CLIO_IPC->AllocateBuffer(bytes);
        auto* dst = reinterpret_cast<idx_t*>(buf.ptr_);
        for (size_t l = 0; l < nlist; ++l) {
            dst[l] = static_cast<idx_t>(src->list_size(l));
        }
        ctp::ipc::ShmPtr<> sp = buf.shm_.template Cast<void>();
        auto f = cte->AsyncPutBlob(tag_id, "sizes", 0, bytes, sp);
        f.Wait();
        FAISS_THROW_IF_NOT_MSG(
                f->return_code_.load() == 0, "CTE PutBlob(sizes) failed");
        CLIO_IPC->FreeBuffer(buf);
    }
    return put_bytes;
}

} // namespace faiss_iowarp
