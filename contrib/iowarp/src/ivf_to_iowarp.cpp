/*
 * ivf_to_iowarp — ingest an on-disk FAISS IVF index into IOWarp CTE.
 *
 * Usage:
 *   ivf_to_iowarp <populated.index> <tag_name> [--verify N|all] [--batch B]
 *
 * Reads the index with the stock OnDiskInvertedLists mmap hook (the
 * .ivfdata path is recorded inside the .index file), then puts one CTE
 * blob "list/<i>" per non-empty list (codes ++ ids, compact) plus the
 * "sizes" blob. Lists are copied through get_codes()/get_ids(), which is
 * what makes capacity-doubled (size < capacity) files come out compact.
 *
 * --verify N:   after ingestion, read back N random non-empty lists from
 *               CTE and byte-compare against the mmap pointers; exits
 *               non-zero on any mismatch.
 * --verify all: read back EVERY non-empty list through the RPC GetBlob
 *               path and byte-compare. This is the write-path proof: the
 *               puts were API-correct (put -> Wait -> rc==0), so any
 *               failure here happened inside the tiering engine between
 *               the put and this read. Reports failing list ids and
 *               distinguishes GetBlob rc!=0 from byte mismatches.
 *
 * Requires a running CLIO runtime (clio_run start).
 */

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <faiss/IndexIVF.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/index_io.h>
#include <faiss/invlists/InvertedLists.h>

#include "cte_client.h"
#include "ivf_cte_ingest.h"

using faiss::idx_t;

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
    int verify_n = 0;  // -1 = all non-empty lists
    size_t batch = 64;
    for (int i = 3; i < argc; ++i) {
        if (!strcmp(argv[i], "--verify") && i + 1 < argc) {
            ++i;
            verify_n = !strcmp(argv[i], "all") ? -1 : atoi(argv[i]);
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

    size_t put_bytes = faiss_iowarp::IngestIvfToCte(src, tag_name, batch);
    std::printf(
            "[ingest] done: %.2f GiB compact (tag %s)\n",
            put_bytes / (1024.0 * 1024 * 1024),
            tag_name.c_str());

    if (verify_n != 0) {
        auto* cte = CLIO_CTE_CLIENT;
        auto tag_fut = cte->AsyncGetOrCreateTag(tag_name);
        tag_fut.Wait();
        auto tag_id = tag_fut->tag_id_;

        std::vector<size_t> nonempty;
        size_t max_bytes = 0;
        for (size_t l = 0; l < nlist; ++l) {
            size_t sz = src->list_size(l);
            if (sz > 0) {
                nonempty.push_back(l);
                max_bytes = std::max(max_bytes, sz * (code_size + sizeof(idx_t)));
            }
        }

        // Which lists to verify: a random sample (--verify N) or every
        // non-empty list in order (--verify all).
        std::vector<size_t> to_check;
        if (verify_n < 0) {
            to_check = nonempty;
        } else {
            std::mt19937_64 rng(12345);
            for (int i = 0; i < verify_n && !nonempty.empty(); ++i) {
                to_check.push_back(nonempty[rng() % nonempty.size()]);
            }
        }
        std::printf(
                "[verify] reading back %zu list(s) via RPC GetBlob\n",
                to_check.size());

        // ONE reused buffer: per-list AllocateBuffer/FreeBuffer segments are
        // not recycled by the dev allocator, and a full-volume verify would
        // otherwise balloon this client by ~40%% of the volume size.
        auto buf = CLIO_IPC->AllocateBuffer(max_bytes);
        FAISS_THROW_IF_NOT_MSG(!buf.IsNull(), "verify: AllocateBuffer failed");

        constexpr int kMaxDetail = 20;  // per-failure log lines cap
        long get_fails = 0;
        long mismatches = 0;
        size_t done = 0;
        for (size_t l : to_check) {
            size_t sz = src->list_size(l);
            size_t bytes = sz * (code_size + sizeof(idx_t));
            ctp::ipc::ShmPtr<> sp = buf.shm_.template Cast<void>();
            std::string name = "list/" + std::to_string(l);
            auto f = cte->AsyncGetBlob(tag_id, name, 0, bytes, 0, sp);
            f.Wait();
            int rc = static_cast<int>(f->return_code_.load());
            if (rc != 0) {
                if (get_fails + mismatches < kMaxDetail) {
                    std::fprintf(
                            stderr,
                            "[verify] GETFAIL list %zu (rc=%d, %zu bytes)\n",
                            l,
                            rc,
                            bytes);
                }
                get_fails++;
            } else {
                faiss::InvertedLists::ScopedCodes codes(src, l);
                faiss::InvertedLists::ScopedIds ids(src, l);
                bool same =
                        !std::memcmp(buf.ptr_, codes.get(), sz * code_size) &&
                        !std::memcmp(
                                buf.ptr_ + sz * code_size,
                                ids.get(),
                                sz * sizeof(idx_t));
                if (!same) {
                    if (get_fails + mismatches < kMaxDetail) {
                        std::fprintf(
                                stderr,
                                "[verify] MISMATCH list %zu (%zu bytes)\n",
                                l,
                                bytes);
                    }
                    mismatches++;
                }
            }
            if (++done % 2048 == 0) {
                std::printf(
                        "[verify] ... %zu/%zu checked (getfail=%ld "
                        "mismatch=%ld)\n",
                        done,
                        to_check.size(),
                        get_fails,
                        mismatches);
            }
        }
        CLIO_IPC->FreeBuffer(buf);

        if (get_fails || mismatches) {
            std::fprintf(
                    stderr,
                    "[verify] FAILED — %ld GetBlob failure(s), %ld byte "
                    "mismatch(es) out of %zu lists\n",
                    get_fails,
                    mismatches,
                    to_check.size());
            return 1;
        }
        std::printf(
                "[verify] PASS — %zu lists byte-identical\n", to_check.size());
    }
    return 0;
}
