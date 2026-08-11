#include "ivf_cte_ingest.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include <faiss/impl/FaissAssert.h>

#include "cte_client.h"

using faiss::idx_t;

namespace faiss_iowarp {

namespace {

// One reusable in-flight PutBlob slot. Buffers are allocated ONCE at max
// list size and reused for every put: on clio-core dev the client segment
// allocator does not recycle multi-MB payload buffers across segments —
// per-list AllocateBuffer/FreeBuffer grew the client pool by ~40% of the
// bytes pushed (67 x ~140 MB segments at nb50M), and those segments stay
// mapped by the daemon after the ingest client exits, OOMing the node
// before the bench starts.
struct PutSlot {
    ctp::ipc::FullPtr<char> buf;
    clio::run::Future<clio::cte::core::PutBlobTask> fut;
    bool busy = false;
};

void drain_slot(PutSlot& s) {
    s.fut.Wait();
    FAISS_THROW_IF_NOT_MSG(
            s.fut->return_code_.load() == 0, "CTE PutBlob failed");
    // Drop the future now: it owns the PutBlobTask via shared_ptr; holding
    // it would pin one shm task per put until the end of the ingest.
    s.fut = clio::run::Future<clio::cte::core::PutBlobTask>();
    s.busy = false;
}

} // namespace

size_t IngestIvfShardsToCte(
        const std::vector<const faiss::InvertedLists*>& srcs,
        const std::string& tag_name,
        size_t batch,
        const std::function<bool(int64_t)>& list_filter,
        bool write_sizes) {
    FAISS_THROW_IF_NOT_MSG(!srcs.empty(), "CTE ingest: no source shards");
    FAISS_THROW_IF_NOT_MSG(
            EnsureIOWarpClient(), "IOWarp client init failed");
    auto* cte = CLIO_CTE_CLIENT;

    const size_t nlist = srcs[0]->nlist;
    const size_t code_size = srcs[0]->code_size;
    for (const auto* src : srcs) {
        FAISS_THROW_IF_NOT_MSG(
                src->nlist == nlist && src->code_size == code_size,
                "CTE ingest: shard nlist/code_size mismatch");
    }

    auto combined_size = [&](size_t l) {
        size_t sz = 0;
        for (const auto* src : srcs) {
            sz += src->list_size(l);
        }
        return sz;
    };

    auto tag_fut = cte->AsyncGetOrCreateTag(tag_name);
    tag_fut.Wait();
    auto tag_id = tag_fut->tag_id_;

    // Small ring of reusable max-size buffers (see PutSlot above). A slot
    // is safe to overwrite once its put completed: the payload has been
    // copied into the tier by then (same lifecycle the old per-list
    // FreeBuffer relied on).
    size_t max_bytes = 0;
    for (size_t l = 0; l < nlist; ++l) {
        if (list_filter && !list_filter(static_cast<int64_t>(l))) {
            continue;
        }
        const size_t sz = combined_size(l);
        max_bytes = std::max(max_bytes, sz * (code_size + sizeof(idx_t)));
    }
    if (max_bytes == 0) {
        max_bytes = 1;  // filter selected nothing on this node
    }
    const size_t ring_n = std::max<size_t>(1, std::min<size_t>(batch, 8));
    std::vector<PutSlot> ring(ring_n);
    for (auto& s : ring) {
        s.buf = CLIO_IPC->AllocateBuffer(max_bytes);
        FAISS_THROW_IF_NOT_MSG(
                !s.buf.IsNull(), "CTE ingest: AllocateBuffer failed");
    }

    size_t put_bytes = 0;
    size_t next = 0;
    for (size_t l = 0; l < nlist; ++l) {
        if (list_filter && !list_filter(static_cast<int64_t>(l))) {
            continue;
        }
        size_t sz = combined_size(l);
        if (sz == 0) {
            continue;
        }
        size_t bytes = sz * (code_size + sizeof(idx_t));
        PutSlot& s = ring[next];
        next = (next + 1) % ring_n;
        if (s.busy) {
            drain_slot(s);
        }
        // Codes of every shard back-to-back, then ids of every shard: the
        // blob keeps the single codes||ids split Search parses at
        // base + sz*code_size.
        size_t coff = 0;
        size_t ioff = sz * code_size;
        for (const auto* src : srcs) {
            size_t r = src->list_size(l);
            if (r == 0) {
                continue;
            }
            faiss::InvertedLists::ScopedCodes codes(src, l);
            faiss::InvertedLists::ScopedIds ids(src, l);
            std::memcpy(s.buf.ptr_ + coff, codes.get(), r * code_size);
            std::memcpy(s.buf.ptr_ + ioff, ids.get(), r * sizeof(idx_t));
            coff += r * code_size;
            ioff += r * sizeof(idx_t);
        }
        ctp::ipc::ShmPtr<> sp = s.buf.shm_.template Cast<void>();
        std::string name = "list/" + std::to_string(l);
        s.fut = cte->AsyncPutBlob(tag_id, name, 0, bytes, sp);
        s.busy = true;
        put_bytes += bytes;
    }
    for (auto& s : ring) {
        if (s.busy) {
            drain_slot(s);
        }
        CLIO_IPC->FreeBuffer(s.buf);
    }

    // "sizes" blob last: its presence signals a completed ingest. In an
    // owner-local multi-node ingest only rank 0 writes it (unfiltered
    // contents — the shards give every list's size locally); the caller
    // must sequence it AFTER all ranks' list puts.
    if (write_sizes) {
        size_t bytes = nlist * sizeof(idx_t);
        auto buf = CLIO_IPC->AllocateBuffer(bytes);
        auto* dst = reinterpret_cast<idx_t*>(buf.ptr_);
        for (size_t l = 0; l < nlist; ++l) {
            dst[l] = static_cast<idx_t>(combined_size(l));
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

size_t IngestIvfToCte(
        const faiss::InvertedLists* src,
        const std::string& tag_name,
        size_t batch) {
    return IngestIvfShardsToCte({src}, tag_name, batch);
}

} // namespace faiss_iowarp
