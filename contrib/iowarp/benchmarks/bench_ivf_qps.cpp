/*
 * bench_ivf_qps — correctness selftest + QPS harness for the faiss_ivf
 * ChiMod (FAISS IVF search inside the CLIO runtime, inverted lists in CTE).
 *
 * Selftest (CLIO runtime must be running):
 *   bench_ivf_qps --selftest-chimod
 *     Builds a small IVF,Flat index, ingests its lists into CTE, opens it
 *     in the ChiMod, and requires the ChiMod's (D, I) to be bitwise-
 *     identical to stock FAISS (single-batch and split-batch searches).
 *
 * Timed mode:
 *   bench_ivf_qps --protocol step3 --index populated.index --tag faiss_ivf::vol
 *                 --queries bigann_query.bvecs [--csv out.csv] [--label vol]
 *                 [--nq 500] [--threads 8] [--k 10] [--passes 3]
 *                 [--nprobe N] [--inflight N] [--route owner|split]
 *                 [--dump-di prefix]
 *     Drives the ChiMod with one batched search per pass; passes are
 *     labeled cold, warm0, warm1... Per pass: QPS, majflt delta,
 *     /proc/self/io read_bytes delta, an FNV-1a hash of (D, I), and a
 *     canonical order-independent hash (per-query-sorted pairs).
 *     --route owner (default): broadcast + owner-filtered scan, results
 *     merged in AggregateOut (data-local multi-node). --route split:
 *     historical query-split DirectHash fan-out.
 */

#include <sys/resource.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <omp.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/index_io.h>

#include "cte_client.h"
#include "ivf_cte_ingest.h"

#ifdef HAVE_FAISS_IVF_CHIMOD
#include <clio_runtime/faiss_ivf/faiss_ivf_client.h>
#endif

using faiss::idx_t;

namespace {

/*** ------------------------- utilities -------------------------------- ***/

long majflt_now() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_majflt;
}

long long io_read_bytes_now() {
    std::ifstream f("/proc/self/io");
    std::string key;
    long long val;
    while (f >> key >> val) {
        if (key == "read_bytes:") {
            return val;
        }
    }
    return -1;
}

double now_s() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// Load the first nq vectors of a .bvecs file (int32 dim + uint8[dim] per
// record) as float32.
std::vector<float> load_bvecs_queries(const std::string& path, size_t nq,
                                      int& d_out) {
    FILE* f = fopen(path.c_str(), "rb");
    FAISS_THROW_IF_NOT_FMT(f, "cannot open %s", path.c_str());
    std::vector<float> out;
    int d = 0;
    for (size_t i = 0; i < nq; ++i) {
        int32_t dim;
        if (fread(&dim, sizeof(dim), 1, f) != 1) {
            break;
        }
        if (i == 0) {
            d = dim;
            out.reserve(nq * d);
        }
        FAISS_THROW_IF_NOT_MSG(dim == d, "inconsistent bvecs dims");
        std::vector<uint8_t> rec(d);
        FAISS_THROW_IF_NOT(fread(rec.data(), 1, d, f) == (size_t)d);
        for (int j = 0; j < d; ++j) {
            out.push_back(static_cast<float>(rec[j]));
        }
    }
    fclose(f);
    d_out = d;
    FAISS_THROW_IF_NOT_MSG(
            out.size() == nq * (size_t)d, "bvecs file shorter than nq");
    return out;
}

#ifdef HAVE_FAISS_IVF_CHIMOD

/*** ------------------------- chimod drivers --------------------------- ***/

// One batched search through the chimod; D/I sized nq*k by the caller.
bool chimod_search(
        clio::run::faiss_ivf::Client& client,
        size_t nq,
        int k,
        int nprobe,
        int d,
        const float* xq,
        float* D,
        idx_t* I) {
    auto qbuf = CLIO_IPC->AllocateBuffer(nq * d * sizeof(float));
    auto dbuf = CLIO_IPC->AllocateBuffer(nq * k * sizeof(float));
    auto ibuf = CLIO_IPC->AllocateBuffer(nq * k * sizeof(idx_t));
    std::memcpy(qbuf.ptr_, xq, nq * d * sizeof(float));
    auto fut = client.AsyncSearch(
            clio::run::PoolQuery::Local(),
            static_cast<clio::run::u32>(nq),
            static_cast<clio::run::u32>(k),
            static_cast<clio::run::u32>(nprobe),
            static_cast<clio::run::u32>(d),
            0,
            qbuf.shm_.template Cast<void>(),
            dbuf.shm_.template Cast<void>(),
            ibuf.shm_.template Cast<void>());
    fut.Wait();
    bool ok = fut->GetReturnCode() == 0;
    if (ok) {
        std::memcpy(D, dbuf.ptr_, nq * k * sizeof(float));
        std::memcpy(I, ibuf.ptr_, nq * k * sizeof(idx_t));
    } else {
        std::fprintf(
                stderr, "chimod search rc=%u\n", fut->GetReturnCode());
    }
    CLIO_IPC->FreeBuffer(qbuf);
    CLIO_IPC->FreeBuffer(dbuf);
    CLIO_IPC->FreeBuffer(ibuf);
    return ok;
}

// Split a query batch into `nsplit` concurrently in-flight SearchTasks —
// the CPU-budget knob (nsplit tasks across the runtime's workers). Per-query
// results are independent, so the split cannot change any output.
//
// Routing: owner_route=false preserves the historical query-split fan-out
// (sub-batch i -> DirectHash(i) -> container i % num_containers).
// owner_route=true broadcasts each sub-batch to EVERY container with the
// owner-filter mode bits: each node scans only the lists it owns and the
// partial top-k merge happens in SearchTask::AggregateOut — --inflight then
// means per-node concurrency, independent of node count.
bool chimod_search_parallel(
        clio::run::faiss_ivf::Client& client,
        size_t nq,
        int k,
        int nprobe,
        int d,
        int nsplit,
        uint32_t mode,
        bool owner_route,
        const float* xq,
        float* D,
        idx_t* I) {
    if (nsplit < 1) {
        nsplit = 1;
    }
    if (static_cast<size_t>(nsplit) > nq) {
        nsplit = static_cast<int>(nq);
    }
    struct Sub {
        ctp::ipc::FullPtr<char> q, dd, ii;
        clio::run::Future<clio::run::faiss_ivf::SearchTask> fut;
        size_t off = 0, cnt = 0;
    };
    std::vector<Sub> subs(nsplit);
    for (int i = 0; i < nsplit; ++i) {
        Sub& s = subs[i];
        s.off = nq * i / nsplit;
        s.cnt = nq * (i + 1) / nsplit - s.off;
        s.q = CLIO_IPC->AllocateBuffer(s.cnt * d * sizeof(float));
        s.dd = CLIO_IPC->AllocateBuffer(s.cnt * k * sizeof(float));
        s.ii = CLIO_IPC->AllocateBuffer(s.cnt * k * sizeof(idx_t));
        std::memcpy(s.q.ptr_, xq + s.off * d, s.cnt * d * sizeof(float));
        // owner_route: Broadcast — every container gets the sub-batch,
        // scans only its owned lists, partials merge in AggregateOut.
        // Otherwise DirectHash(i): sub-batch i lands on container
        // i % num_containers (single node: container 0 == old Local).
        s.fut = client.AsyncSearch(
                owner_route
                        ? clio::run::PoolQuery::Broadcast()
                        : clio::run::PoolQuery::DirectHash(
                                  static_cast<clio::run::u32>(i)),
                static_cast<clio::run::u32>(s.cnt),
                static_cast<clio::run::u32>(k),
                static_cast<clio::run::u32>(nprobe),
                static_cast<clio::run::u32>(d),
                mode,
                s.q.shm_.template Cast<void>(),
                s.dd.shm_.template Cast<void>(),
                s.ii.shm_.template Cast<void>());
    }
    bool ok = true;
    for (auto& s : subs) {
        s.fut.Wait();
        if (s.fut->GetReturnCode() != 0) {
            std::fprintf(
                    stderr,
                    "chimod subtask rc=%u\n",
                    s.fut->GetReturnCode());
            ok = false;
        } else {
            const size_t n = s.cnt * static_cast<size_t>(k);
            if (s.fut->part_d_.size() == n && s.fut->part_i_.size() == n) {
                // Multi-node owner path: the merged global top-k arrived
                // as serialized vectors (replica D/I never travels back).
                std::memcpy(D + s.off * k, s.fut->part_d_.data(),
                            n * sizeof(float));
                std::memcpy(I + s.off * k, s.fut->part_i_.data(),
                            n * sizeof(idx_t));
            } else {
                // Legacy routing or single-node-degenerate broadcast: the
                // handler wrote the shm buffers directly (part_* empty).
                std::memcpy(D + s.off * k, s.dd.ptr_, n * sizeof(float));
                std::memcpy(I + s.off * k, s.ii.ptr_, n * sizeof(idx_t));
            }
        }
        CLIO_IPC->FreeBuffer(s.q);
        CLIO_IPC->FreeBuffer(s.dd);
        CLIO_IPC->FreeBuffer(s.ii);
    }
    return ok;
}

/*** ------------------------- selftest --------------------------------- ***/

// Equivalence test: tiny L2 index, reference (D,I) from stock FAISS, then
// the ChiMod's search (single-batch and split-batch) must match bitwise.
int run_selftest_chimod() {
    FAISS_THROW_IF_NOT_MSG(
            faiss_iowarp::EnsureIOWarpClient(), "IOWarp client init failed");
    const int d = 32, nlist = 64, k = 10, nprobe = 8;
    const size_t nb = 100000, nt = 20000, nq = 200;
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> xb(nb * d), xt(nt * d), xq(nq * d);
    for (auto& v : xb) v = nd(rng);
    for (auto& v : xt) v = nd(rng);
    for (auto& v : xq) v = nd(rng);

    faiss::IndexFlatL2 quantizer(d);
    faiss::IndexIVFFlat index(&quantizer, d, nlist, faiss::METRIC_L2);
    index.own_fields = false;
    index.train(nt, xt.data());
    index.add(nb, xb.data());
    index.nprobe = nprobe;

    std::vector<float> D_ref(nq * k), D_new(nq * k);
    std::vector<idx_t> I_ref(nq * k), I_new(nq * k);
    omp_set_num_threads(8);
    index.search(nq, xq.data(), k, D_ref.data(), I_ref.data());

    std::string tag = "faiss_ivf::selftest_chimod_" + std::to_string(getpid());
    faiss_iowarp::IngestIvfToCte(index.invlists, tag);
    std::string index_file =
            "/tmp/faiss_ivf_selftest_" + std::to_string(getpid()) + ".index";
    faiss::write_index(&index, index_file.c_str());

    clio::run::faiss_ivf::Client client;
    auto create_fut = client.AsyncCreate(
            clio::run::PoolQuery::Local(),
            "faiss_ivf_bench",
            clio::run::PoolId(600, 0));
    create_fut.Wait();
    auto open_fut = client.AsyncOpenIndex(
            clio::run::PoolQuery::Local(), index_file, tag);
    open_fut.Wait();
    if (open_fut->GetReturnCode() != 0) {
        std::fprintf(
                stderr,
                "[selftest chimod] OpenIndex rc=%u\n",
                open_fut->GetReturnCode());
        return 1;
    }
    std::printf(
            "[selftest chimod] opened: ntotal=%" PRIu64 " d=%u nlist=%u\n",
            (uint64_t)open_fut->ntotal_,
            open_fut->d_,
            open_fut->nlist_);

    bool all_ok = true;
    // Two consecutive searches: each reads its lists from CTE fresh, so
    // both must be bitwise-identical to stock FAISS.
    for (int rep = 0; rep < 2; ++rep) {
        bool ok = chimod_search(
                client, nq, k, nprobe, d, xq.data(), D_new.data(),
                I_new.data());
        bool d_ok = ok &&
                !std::memcmp(
                        D_ref.data(),
                        D_new.data(),
                        D_ref.size() * sizeof(float));
        bool i_ok = ok &&
                !std::memcmp(
                        I_ref.data(),
                        I_new.data(),
                        I_ref.size() * sizeof(idx_t));
        std::printf(
                "[selftest chimod rep%d] D %s, I %s\n",
                rep,
                d_ok ? "identical" : "MISMATCH",
                i_ok ? "identical" : "MISMATCH");
        all_ok = all_ok && d_ok && i_ok;
    }
    // Split-batch path (4 concurrent subtasks) must also be identical.
    {
        bool ok = chimod_search_parallel(
                client, nq, k, nprobe, d, 4, 0, false, xq.data(),
                D_new.data(), I_new.data());
        bool eq = ok &&
                !std::memcmp(
                        D_ref.data(),
                        D_new.data(),
                        D_ref.size() * sizeof(float)) &&
                !std::memcmp(
                        I_ref.data(),
                        I_new.data(),
                        I_ref.size() * sizeof(idx_t));
        std::printf(
                "[selftest chimod split4] %s\n",
                eq ? "D and I identical" : "MISMATCH");
        all_ok = all_ok && eq;
    }
    // Owner-filtered broadcast path. On one node it must degenerate to the
    // exact legacy behavior (all lists local, handler writes the client
    // buffers, part_* empty) — bitwise identical to stock FAISS.
    {
        bool ok = chimod_search_parallel(
                client, nq, k, nprobe, d, 2,
                clio::run::faiss_ivf::kSearchModeOwner, true, xq.data(),
                D_new.data(), I_new.data());
        bool eq = ok &&
                !std::memcmp(
                        D_ref.data(),
                        D_new.data(),
                        D_ref.size() * sizeof(float)) &&
                !std::memcmp(
                        I_ref.data(),
                        I_new.data(),
                        I_ref.size() * sizeof(idx_t));
        std::printf(
                "[selftest chimod owner2] %s\n",
                eq ? "D and I identical" : "MISMATCH");
        all_ok = all_ok && eq;
    }
    unlink(index_file.c_str());
    std::printf("[selftest chimod] %s\n", all_ok ? "PASS" : "FAIL");
    return all_ok ? 0 : 1;
}

/*** ------------------------- timed harness ---------------------------- ***/

struct Args {
    std::string index_path;
    std::string queries_path;
    std::string tag;
    std::string csv;
    std::string label = "unknown";
    size_t nq = 500;
    int threads = 8;
    int k = 10;
    int passes = 3;
    int nprobe_override = 0;
    // Concurrent SearchTasks per pass (0 = default: 1 for owner route,
    // --threads for split route).
    int inflight = 0;
    // "owner": broadcast + owner-filtered scan + AggregateOut merge
    // (data-local; the default). "split": historical query-split
    // DirectHash fan-out, kept for A/B comparison.
    std::string route = "owner";
    // Non-empty: write raw (D, I) per pass to <prefix>.<pass>.bin.
    std::string dump_di;
};

// Canonical (order-independent) hash of the results: per query, sort the k
// (D, I) pairs by (D, then I) and FNV-1a the sorted stream. Owner-route
// merging can legitimately reorder equal-distance ties vs the single-node
// heap order, so di_hash may differ while the result SET is identical —
// canon compares the sets.
uint64_t canon_hash(const float* D, const idx_t* I, size_t nq, int k) {
    uint64_t h = 1469598103934665603ULL;
    auto fnv = [&h](const void* p, size_t n) {
        const uint8_t* b = static_cast<const uint8_t*>(p);
        for (size_t i = 0; i < n; ++i) {
            h = (h ^ b[i]) * 1099511628211ULL;
        }
    };
    std::vector<std::pair<float, idx_t>> row(k);
    for (size_t qi = 0; qi < nq; ++qi) {
        for (int j = 0; j < k; ++j) {
            row[j] = {D[qi * k + j], I[qi * k + j]};
        }
        std::sort(row.begin(), row.end());
        for (int j = 0; j < k; ++j) {
            fnv(&row[j].first, sizeof(float));
            fnv(&row[j].second, sizeof(idx_t));
        }
    }
    return h;
}

// Dump raw results: magic "DIQ1", u64 nq, u64 k, float D[nq*k], i64 I[nq*k].
void dump_di_file(const std::string& path, const float* D, const idx_t* I,
                  size_t nq, int k) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "WARN: cannot write %s\n", path.c_str());
        return;
    }
    uint64_t nq64 = nq, k64 = static_cast<uint64_t>(k);
    fwrite("DIQ1", 1, 4, f);
    fwrite(&nq64, sizeof(nq64), 1, f);
    fwrite(&k64, sizeof(k64), 1, f);
    fwrite(D, sizeof(float), nq * k, f);
    fwrite(I, sizeof(idx_t), nq * k, f);
    fclose(f);
}

int run_timed(const Args& a) {
    FAISS_THROW_IF_NOT_MSG(!a.index_path.empty(), "--index required");
    FAISS_THROW_IF_NOT_MSG(!a.queries_path.empty(), "--queries required");
    FAISS_THROW_IF_NOT_MSG(!a.tag.empty(), "--tag required");

    // Load index metadata only (the lists live in CTE, read by the ChiMod);
    // the client needs d / nlist / nprobe here.
    std::unique_ptr<faiss::Index> owner(faiss::read_index(
            a.index_path.c_str(), faiss::IO_FLAG_SKIP_IVF_DATA));
    auto* ivf = dynamic_cast<faiss::IndexIVF*>(owner.get());
    FAISS_THROW_IF_NOT_MSG(ivf, "not an IVF index");

    FAISS_THROW_IF_NOT_MSG(
            faiss_iowarp::EnsureIOWarpClient(), "client init failed");
    clio::run::faiss_ivf::Client chimod_client;
    chimod_client
            .AsyncCreate(
                    clio::run::PoolQuery::Local(),
                    "faiss_ivf_bench",
                    clio::run::PoolId(600, 0))
            .Wait();
    // Broadcast: EVERY node's container must open the index (per-container
    // state) — with Local, a second node's container would fail every
    // search routed to it with rc=1. Single-node this degenerates to the
    // one local container.
    auto open_fut = chimod_client.AsyncOpenIndex(
            clio::run::PoolQuery::Broadcast(), a.index_path, a.tag);
    open_fut.Wait();
    FAISS_THROW_IF_NOT_FMT(
            open_fut->GetReturnCode() == 0,
            "chimod OpenIndex rc=%u",
            open_fut->GetReturnCode());

    int d = 0;
    std::vector<float> xq = load_bvecs_queries(a.queries_path, a.nq, d);
    FAISS_THROW_IF_NOT_FMT(
            d == ivf->d, "query dim %d != index dim %d", d, (int)ivf->d);

    int nprobe = a.nprobe_override > 0
            ? a.nprobe_override
            : std::max<size_t>(1, ivf->nlist / 64);
    omp_set_num_threads(a.threads);

    FAISS_THROW_IF_NOT_MSG(
            a.route == "owner" || a.route == "split",
            "--route must be owner or split");
    const bool owner_route = (a.route == "owner");
    const uint32_t mode = owner_route
            ? (clio::run::faiss_ivf::kSearchModeOwner |
               (ivf->metric_type == faiss::METRIC_L2
                        ? 0u
                        : clio::run::faiss_ivf::kSearchModeIP))
            : 0u;
    // Owner route broadcasts every sub-batch to all nodes, so inflight is
    // per-node concurrency: default 1 (one SearchTask x 8 scan threads per
    // node). Split route keeps the historical default of --threads.
    const int nsplit = a.inflight > 0
            ? a.inflight
            : (owner_route ? 1 : a.threads);

    std::printf(
            "[bench] volume=%s nlist=%zu nprobe=%d nq=%zu k=%d threads=%d "
            "route=%s inflight=%d ntotal=%" PRId64 "\n",
            a.label.c_str(),
            ivf->nlist,
            nprobe,
            a.nq,
            a.k,
            a.threads,
            a.route.c_str(),
            nsplit,
            (int64_t)open_fut->ntotal_);

    FILE* csv = nullptr;
    if (!a.csv.empty()) {
        bool fresh = !std::ifstream(a.csv).good();
        csv = fopen(a.csv.c_str(), "a");
        FAISS_THROW_IF_NOT_FMT(csv, "cannot open %s", a.csv.c_str());
        if (fresh) {
            fprintf(csv,
                    "timestamp,volume,pass,nq,k,nprobe,threads,"
                    "elapsed_s,qps,majflt,majflt_per_q,read_bytes,notes\n");
        }
    }

    std::vector<float> D(a.nq * a.k);
    std::vector<idx_t> I(a.nq * a.k);
    for (int p = 0; p < a.passes; ++p) {
        std::string pass =
                p == 0 ? "cold" : ("warm" + std::to_string(p - 1));
        long mf0 = majflt_now();
        long long rb0 = io_read_bytes_now();
        double t0 = now_s();
        FAISS_THROW_IF_NOT_MSG(
                chimod_search_parallel(
                        chimod_client,
                        a.nq,
                        a.k,
                        nprobe,
                        d,
                        nsplit,
                        mode,
                        owner_route,
                        xq.data(),
                        D.data(),
                        I.data()),
                "chimod search failed");
        double el = now_s() - t0;
        long mf = majflt_now() - mf0;
        long long rb = io_read_bytes_now() - rb0;
        double qps = a.nq / el;
        // FNV-1a over the (D, I) buffers: a cheap bitwise-integrity gate at
        // real-volume scale (identical searches produce identical hashes).
        uint64_t rhash = 1469598103934665603ULL;
        auto fnv = [&rhash](const void* p, size_t n) {
            const uint8_t* b = static_cast<const uint8_t*>(p);
            for (size_t i = 0; i < n; ++i) {
                rhash = (rhash ^ b[i]) * 1099511628211ULL;
            }
        };
        fnv(D.data(), D.size() * sizeof(float));
        fnv(I.data(), I.size() * sizeof(idx_t));
        uint64_t chash = canon_hash(D.data(), I.data(), a.nq, a.k);
        std::printf(
                "  %-6s qps=%9.1f  elapsed=%8.3fs  majflt/q=%8.1f  "
                "read_MB=%8.1f  di_hash=%016llx  canon=%016llx\n",
                pass.c_str(),
                qps,
                el,
                mf / (double)a.nq,
                rb / (1024.0 * 1024),
                (unsigned long long)rhash,
                (unsigned long long)chash);
        if (!a.dump_di.empty()) {
            dump_di_file(
                    a.dump_di + "." + pass + ".bin",
                    D.data(),
                    I.data(),
                    a.nq,
                    a.k);
        }
        if (csv) {
            char notes[128];
            snprintf(notes, sizeof(notes),
                     "dihash=%016llx;canon=%016llx;inflight=%d;route=%s",
                     (unsigned long long)rhash,
                     (unsigned long long)chash,
                     nsplit,
                     a.route.c_str());
            fprintf(csv,
                    "%ld,%s,%s,%zu,%d,%d,%d,%.4f,%.1f,%ld,%.3f,%lld,%s\n",
                    (long)time(nullptr),
                    a.label.c_str(),
                    pass.c_str(),
                    a.nq,
                    a.k,
                    nprobe,
                    a.threads,
                    el,
                    qps,
                    mf,
                    mf / (double)a.nq,
                    rb,
                    notes);
        }
    }
    if (csv) {
        fclose(csv);
    }
    // Broadcast + summing AggregateOut => cluster-total counters.
    auto sf = chimod_client.AsyncStats(clio::run::PoolQuery::Broadcast(), 1);
    sf.Wait();
    if (sf->GetReturnCode() == 0) {
        std::printf(
                "[stats] searches=%llu lists_read=%llu GB_read=%.2f "
                "read_wait_s=%.2f scan_s=%.2f\n",
                (unsigned long long)sf->searches_,
                (unsigned long long)sf->lists_fetched_,
                sf->bytes_fetched_ / (1024.0 * 1024 * 1024),
                sf->fetch_wait_us_ / 1e6,
                sf->scan_us_ / 1e6);
    }
    return 0;
}

#endif // HAVE_FAISS_IVF_CHIMOD

} // namespace

int main(int argc, char** argv) {
    // Line-buffer stdout so progress survives `| tee` in batch jobs.
    setvbuf(stdout, nullptr, _IOLBF, 0);
#ifndef HAVE_FAISS_IVF_CHIMOD
    (void)argc;
    (void)argv;
    std::fprintf(stderr, "built without chimod support\n");
    return 3;
#else
    Args a;
    bool timed = false;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&](const char* opt) -> std::string {
            FAISS_THROW_IF_NOT_FMT(i + 1 < argc, "%s needs a value", opt);
            return argv[++i];
        };
        if (s == "--selftest-chimod") return run_selftest_chimod();
        else if (s == "--protocol") timed = (next("--protocol") == "step3");
        else if (s == "--index") a.index_path = next("--index");
        else if (s == "--queries") a.queries_path = next("--queries");
        else if (s == "--tag") a.tag = next("--tag");
        else if (s == "--csv") a.csv = next("--csv");
        else if (s == "--label") a.label = next("--label");
        else if (s == "--nq") a.nq = atoll(next("--nq").c_str());
        else if (s == "--threads") a.threads = atoi(next("--threads").c_str());
        else if (s == "--k") a.k = atoi(next("--k").c_str());
        else if (s == "--passes") a.passes = atoi(next("--passes").c_str());
        else if (s == "--nprobe") a.nprobe_override = atoi(next("--nprobe").c_str());
        else if (s == "--inflight") a.inflight = atoi(next("--inflight").c_str());
        else if (s == "--route") a.route = next("--route");
        else if (s == "--dump-di") a.dump_di = next("--dump-di");
        else {
            std::fprintf(stderr, "unknown arg: %s\n", s.c_str());
            return 2;
        }
    }
    if (timed) {
        return run_timed(a);
    }
    std::fprintf(stderr,
                 "nothing to do: pass --selftest-chimod or --protocol step3\n");
    return 2;
#endif
}
