/*
 * ivf_to_iowarp — ingest on-disk FAISS IVF index shard(s) into IOWarp CTE.
 *
 * Usage:
 *   ivf_to_iowarp <populated.index> <tag_name> [flags]
 *   ivf_to_iowarp --shards s0.index s1.index ... --tag <tag_name> [flags]
 *   flags: [--verify N|all] [--batch B] [--expect-ntotal N]
 *
 * Reads each index with the stock OnDiskInvertedLists mmap hook (the
 * .ivfdata path is recorded inside the .index file), then puts one CTE
 * blob "list/<i>" per non-empty list plus the "sizes" blob. With multiple
 * shards (which must share nlist/code_size — same trained quantizer,
 * globally-unique ids), each blob holds every shard's codes back-to-back
 * followed by every shard's ids, preserving the single codes||ids split
 * the ChiMod's Search parses — ingesting R shards is search-equivalent to
 * ingesting their merged index. Lists are copied through
 * get_codes()/get_ids(), which is what makes capacity-doubled
 * (size < capacity) files come out compact.
 *
 * --expect-ntotal N: assert the shards' summed ntotal is exactly N before
 *               ingesting (exit 3 otherwise). Guards against historical
 *               shard files grown in place by old mixed-workload rounds.
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
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <faiss/IndexIVF.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/index_io.h>
#include <faiss/invlists/InvertedLists.h>

#include "cte_client.h"
#include "ivf_cte_ingest.h"

#ifdef HAVE_FAISS_IVF_CHIMOD
#include <clio_runtime/faiss_ivf/faiss_ivf_tasks.h>  // ListOwnerContainer
#endif

using faiss::idx_t;

static void usage(const char* argv0) {
    std::fprintf(
            stderr,
            "usage: %s <populated.index> <tag_name> [flags]\n"
            "       %s --shards s0.index s1.index ... --tag <tag_name> "
            "[flags]\n"
            "flags: [--verify N|all] [--batch B] [--expect-ntotal N]\n",
            argv0,
            argv0);
}

int main(int argc, char** argv) {
    std::vector<std::string> index_paths;
    std::string tag_name;
    int verify_n = 0;  // -1 = all non-empty lists
    size_t batch = 64;
    int64_t expect_ntotal = -1;
    // --owner-only R:N — ingest (and verify) only the lists container R of
    // an N-container cluster owns, so every put is NODE-LOCAL when run on
    // that node. A single head-node ingest pushes (N-1)/N of the bytes
    // cross-node and OOMs peer daemons (~24 GB/node of recv staging).
    // Rank 0 additionally writes the "sizes" blob (full contents).
    int owner_rank = -1, owner_n = 0;

    int i = 1;
    if (argc >= 2 && !strcmp(argv[1], "--shards")) {
        for (i = 2; i < argc && argv[i][0] != '-'; ++i) {
            index_paths.push_back(argv[i]);
        }
    } else if (argc >= 3) {
        index_paths.push_back(argv[1]);
        tag_name = argv[2];
        i = 3;
    }
    for (; i < argc; ++i) {
        if (!strcmp(argv[i], "--tag") && i + 1 < argc) {
            tag_name = argv[++i];
        } else if (!strcmp(argv[i], "--verify") && i + 1 < argc) {
            ++i;
            verify_n = !strcmp(argv[i], "all") ? -1 : atoi(argv[i]);
        } else if (!strcmp(argv[i], "--batch") && i + 1 < argc) {
            batch = static_cast<size_t>(atoll(argv[++i]));
        } else if (!strcmp(argv[i], "--expect-ntotal") && i + 1 < argc) {
            expect_ntotal = atoll(argv[++i]);
        } else if (!strcmp(argv[i], "--owner-only") && i + 1 < argc) {
            ++i;
            if (sscanf(argv[i], "%d:%d", &owner_rank, &owner_n) != 2 ||
                owner_rank < 0 || owner_n <= owner_rank) {
                std::fprintf(stderr, "bad --owner-only %s (want R:N)\n",
                             argv[i]);
                return 2;
            }
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", argv[i]);
            usage(argv[0]);
            return 2;
        }
    }
    if (index_paths.empty() || tag_name.empty()) {
        usage(argv[0]);
        return 2;
    }

    std::vector<std::unique_ptr<faiss::Index>> owners;
    std::vector<const faiss::InvertedLists*> srcs;
    int64_t ntotal = 0;
    for (const auto& path : index_paths) {
        std::printf("[ingest] reading %s (mmap)\n", path.c_str());
        owners.emplace_back(faiss::read_index(path.c_str()));
        auto* ivf = dynamic_cast<faiss::IndexIVF*>(owners.back().get());
        FAISS_THROW_IF_NOT_MSG(ivf, "not an IVF index");
        srcs.push_back(ivf->invlists);
        ntotal += ivf->ntotal;
        FAISS_THROW_IF_NOT_MSG(
                srcs.back()->nlist == srcs[0]->nlist &&
                        srcs.back()->code_size == srcs[0]->code_size,
                "shard nlist/code_size mismatch");
    }
    const size_t nlist = srcs[0]->nlist;
    const size_t code_size = srcs[0]->code_size;
    std::printf(
            "[ingest] shards=%zu nlist=%zu code_size=%zu ntotal=%" PRId64
            "\n",
            srcs.size(),
            nlist,
            code_size,
            ntotal);
    if (expect_ntotal >= 0 && ntotal != expect_ntotal) {
        std::fprintf(
                stderr,
                "[ingest] ntotal %" PRId64 " != expected %" PRId64
                " — dirty shard set?\n",
                ntotal,
                expect_ntotal);
        return 3;
    }

    // Combined size of list l across all shards.
    auto list_sz = [&](size_t l) {
        size_t sz = 0;
        for (const auto* src : srcs) {
            sz += src->list_size(l);
        }
        return sz;
    };

    std::function<bool(int64_t)> owned;  // empty = ingest everything
    if (owner_rank >= 0) {
#ifdef HAVE_FAISS_IVF_CHIMOD
        FAISS_THROW_IF_NOT_MSG(
                faiss_iowarp::EnsureIOWarpClient(), "client init failed");
        auto tf = CLIO_CTE_CLIENT->AsyncGetOrCreateTag(tag_name);
        tf.Wait();
        const uint32_t tmaj = static_cast<uint32_t>(tf->tag_id_.major_);
        const uint32_t tmin = static_cast<uint32_t>(tf->tag_id_.minor_);
        const uint32_t r = static_cast<uint32_t>(owner_rank);
        const uint32_t n = static_cast<uint32_t>(owner_n);
        owned = [tmaj, tmin, r, n](int64_t l) {
            return clio::run::faiss_ivf::ListOwnerContainer(tmaj, tmin, l, n)
                    == r;
        };
        std::printf("[ingest] owner-only mode: container %d of %d\n",
                    owner_rank, owner_n);
#else
        std::fprintf(stderr, "--owner-only needs a chimod-enabled build\n");
        return 2;
#endif
    }

    size_t put_bytes = faiss_iowarp::IngestIvfShardsToCte(
            srcs, tag_name, batch, owned, /*write_sizes=*/owner_rank <= 0);
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
            // Owner-only mode: verify only OUR lists — peer ranks' puts
            // may still be in flight when this rank verifies.
            if (owned && !owned(static_cast<int64_t>(l))) {
                continue;
            }
            size_t sz = list_sz(l);
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
            for (int j = 0; j < verify_n && !nonempty.empty(); ++j) {
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

        // Blob layout: codes of every shard back-to-back, then ids of every
        // shard — compare region by region against the shards' mmaps.
        auto list_matches = [&](size_t l, size_t sz) {
            size_t coff = 0;
            size_t ioff = sz * code_size;
            for (const auto* src : srcs) {
                size_t r = src->list_size(l);
                if (r == 0) {
                    continue;
                }
                faiss::InvertedLists::ScopedCodes codes(src, l);
                faiss::InvertedLists::ScopedIds ids(src, l);
                if (std::memcmp(buf.ptr_ + coff, codes.get(), r * code_size) ||
                    std::memcmp(
                            buf.ptr_ + ioff, ids.get(), r * sizeof(idx_t))) {
                    return false;
                }
                coff += r * code_size;
                ioff += r * sizeof(idx_t);
            }
            return true;
        };

        constexpr int kMaxDetail = 20;  // per-failure log lines cap
        long get_fails = 0;
        long mismatches = 0;
        size_t done = 0;
        for (size_t l : to_check) {
            size_t sz = list_sz(l);
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
            } else if (!list_matches(l, sz)) {
                if (get_fails + mismatches < kMaxDetail) {
                    std::fprintf(
                            stderr,
                            "[verify] MISMATCH list %zu (%zu bytes)\n",
                            l,
                            bytes);
                }
                mismatches++;
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
