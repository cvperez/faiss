/*
 * ivf_to_iowarp — ingest an on-disk FAISS IVF index into IOWarp CTE.
 *
 * Usage:
 *   ivf_to_iowarp <populated.index> <tag_name> [--verify N] [--batch B]
 *
 * Reads the index with the stock OnDiskInvertedLists mmap hook (the
 * .ivfdata path is recorded inside the .index file), then puts one CTE
 * blob "list/<i>" per non-empty list (codes ++ ids, compact) plus the
 * "sizes" blob, pipelining up to B puts in flight (default 64).
 * Lists are copied through get_codes()/get_ids(), which is what makes
 * capacity-doubled (size < capacity) files come out compact.
 *
 * --verify N: after ingestion, read back N random non-empty lists from
 * CTE and byte-compare against the mmap pointers; exits non-zero on any
 * mismatch.
 *
 * Requires a running CLIO runtime (clio_run start).
 */

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <deque>
#include <random>
#include <string>
#include <vector>

#include <faiss/IndexIVF.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/index_io.h>
#include <faiss/invlists/InvertedLists.h>

#include "IOWarpInvertedLists.h"

using faiss::idx_t;

namespace {

struct InFlightPut {
    ctp::ipc::FullPtr<char> buf;
    chi::Future<clio::cte::core::PutBlobTask> fut;
};

void drain_one(std::deque<InFlightPut>& q) {
    auto& front = q.front();
    front.fut.Wait();
    FAISS_THROW_IF_NOT_MSG(
            front.fut->return_code_.load() == 0, "PutBlob failed");
    CLIO_IPC->FreeBuffer(front.buf);
    q.pop_front();
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(
                stderr,
                "usage: %s <populated.index> <tag_name> [--verify N] "
                "[--batch B]\n",
                argv[0]);
        return 2;
    }
    std::string index_path = argv[1];
    std::string tag_name = argv[2];
    int verify_n = 0;
    size_t batch = 64;
    for (int i = 3; i < argc; ++i) {
        if (!strcmp(argv[i], "--verify") && i + 1 < argc) {
            verify_n = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--batch") && i + 1 < argc) {
            batch = static_cast<size_t>(atoll(argv[++i]));
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", argv[i]);
            return 2;
        }
    }

    std::printf("[ingest] reading %s (mmap)\n", index_path.c_str());
    std::unique_ptr<faiss::Index> owner(
            faiss::read_index(index_path.c_str()));
    auto* ivf = dynamic_cast<faiss::IndexIVF*>(owner.get());
    FAISS_THROW_IF_NOT_MSG(ivf, "not an IVF index");
    const faiss::InvertedLists* src = ivf->invlists;
    size_t nlist = src->nlist;
    size_t code_size = src->code_size;
    std::printf(
            "[ingest] nlist=%zu code_size=%zu ntotal=%" PRId64 "\n",
            nlist,
            code_size,
            ivf->ntotal);

    FAISS_THROW_IF_NOT_MSG(
            faiss_iowarp::EnsureIOWarpClient(), "IOWarp client init failed");
    auto* cte = CLIO_CTE_CLIENT;
    auto tag_fut = cte->AsyncGetOrCreateTag(tag_name);
    tag_fut.Wait();
    auto tag_id = tag_fut->tag_id_;

    std::deque<InFlightPut> inflight;
    size_t put_lists = 0;
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
        put_lists++;
        put_bytes += bytes;
        if (put_lists % 1024 == 0) {
            std::printf(
                    "[ingest] %zu lists, %.2f GiB\n",
                    put_lists,
                    put_bytes / (1024.0 * 1024 * 1024));
        }
    }
    while (!inflight.empty()) {
        drain_one(inflight);
    }

    // sizes blob
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
                f->return_code_.load() == 0, "PutBlob(sizes) failed");
        CLIO_IPC->FreeBuffer(buf);
    }
    std::printf(
            "[ingest] done: %zu non-empty lists, %.2f GiB compact "
            "(tag %s)\n",
            put_lists,
            put_bytes / (1024.0 * 1024 * 1024),
            tag_name.c_str());

    if (verify_n > 0) {
        std::vector<size_t> nonempty;
        for (size_t l = 0; l < nlist; ++l) {
            if (src->list_size(l) > 0) {
                nonempty.push_back(l);
            }
        }
        std::mt19937_64 rng(12345);
        int fails = 0;
        for (int i = 0; i < verify_n && !nonempty.empty(); ++i) {
            size_t l = nonempty[rng() % nonempty.size()];
            size_t sz = src->list_size(l);
            size_t bytes = sz * (code_size + sizeof(idx_t));
            auto buf = CLIO_IPC->AllocateBuffer(bytes);
            ctp::ipc::ShmPtr<> sp = buf.shm_.template Cast<void>();
            std::string name = "list/" + std::to_string(l);
            auto f = cte->AsyncGetBlob(tag_id, name, 0, bytes, 0, sp);
            f.Wait();
            bool ok = f->return_code_.load() == 0;
            if (ok) {
                faiss::InvertedLists::ScopedCodes codes(src, l);
                faiss::InvertedLists::ScopedIds ids(src, l);
                ok = !std::memcmp(buf.ptr_, codes.get(), sz * code_size) &&
                        !std::memcmp(
                                buf.ptr_ + sz * code_size,
                                ids.get(),
                                sz * sizeof(idx_t));
            }
            CLIO_IPC->FreeBuffer(buf);
            if (!ok) {
                std::fprintf(stderr, "[verify] MISMATCH list %zu\n", l);
                fails++;
            }
        }
        if (fails) {
            std::fprintf(stderr, "[verify] FAILED (%d mismatches)\n", fails);
            return 1;
        }
        std::printf("[verify] PASS — %d lists byte-identical\n", verify_n);
    }
    return 0;
}
