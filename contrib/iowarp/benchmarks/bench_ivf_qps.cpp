/*
 * bench_ivf_qps — correctness selftest + QPS harness for the faiss_ivf
 * ChiMod (FAISS IVF search inside the CLIO runtime, inverted lists in CTE).
 *
 * Selftest (CLIO runtime must be running):
 *   bench_ivf_qps --selftest-chimod
 *     Builds a small IVF,Flat index, ingests its lists into CTE, opens it
 *     in the ChiMod, and requires the ChiMod's (D, I) to be bitwise-
 *     identical to stock FAISS (single-batch, split-batch, owner-broadcast
 *     and post-Add searches).
 *
 * Timed mode:
 *   bench_ivf_qps --protocol step3 --index populated.index --tag faiss_ivf::vol
 *                 --queries bigann_query.bvecs [--csv out.csv] [--label vol]
 *                 [--nq 500] [--threads 8] [--k 10] [--passes 3]
 *                 [--nprobe N] [--inflight N] [--route owner|split]
 *                 [--dump-di prefix] [--json-out out.json]
 *                 [--nb-m N] [--nodes K]
 *     Drives the ChiMod with one batched search per pass; passes are
 *     labeled cold, warm0, warm1... Per pass: QPS, majflt delta,
 *     /proc/self/io read_bytes delta, an FNV-1a hash of (D, I), a
 *     canonical order-independent hash (per-query-sorted pairs), and a
 *     cluster-Stats bracket yielding /proc/diskstats deltas with the mmap
 *     study's semantics (per-node mean active %, summed MB/s).
 *     --json-out additionally writes a performance-study record
 *     (perf_study_exp1) with plot-compatible field names; --nb-m/--nodes
 *     stamp the record's nb_M / n_nodes.
 *     --route owner (default): broadcast + owner-filtered scan, results
 *     merged in AggregateOut (data-local multi-node). --route split:
 *     historical query-split DirectHash fan-out.
 */

#include <unistd.h>

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <omp.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/index_io.h>

#include "cte_client.h"
#include "ivf_cte_ingest.h"

#include "bench_common.h"

#ifdef HAVE_FAISS_IVF_CHIMOD
#include <clio_runtime/faiss_ivf/faiss_ivf_client.h>
#endif

using faiss::idx_t;

namespace {

using bench_common::canon_hash;
using bench_common::load_bvecs_queries;
using bench_common::now_s;

#ifdef HAVE_FAISS_IVF_CHIMOD

using bench_common::chimod_search;
using bench_common::chimod_search_parallel;
using bench_common::dump_di_file;
using bench_common::fetch_cluster_stats;
using bench_common::fprint_pass_json;
using bench_common::AddSender;
using bench_common::group_add_batch;
using bench_common::PassResult;
using bench_common::run_one_pass;

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
    // SHARED filesystem, not /tmp: the broadcast OpenIndex makes every
    // node's container read this file — a node-local path leaves remote
    // containers unopened (rc=1 on every routed search/add).
    const char* home = getenv("HOME");
    std::string index_file = std::string(home ? home : "/tmp") +
            "/.faiss_ivf_selftest_" + std::to_string(getpid()) + ".index";
    faiss::write_index(&index, index_file.c_str());

    clio::run::faiss_ivf::Client client;
    auto create_fut = client.AsyncCreate(
            clio::run::PoolQuery::Local(),
            "faiss_ivf_bench",
            clio::run::PoolId(600, 0));
    create_fut.Wait();
    // Broadcast: on a multi-node cluster EVERY container must open the
    // index — the split4 leg routes sub-batches to every container, and
    // an unopened container fails its search with rc=1 (exactly what the
    // first n4 run showed with a Local open).
    auto open_fut = client.AsyncOpenIndex(
            clio::run::PoolQuery::Broadcast(), index_file, tag);
    open_fut.Wait();
    if (open_fut->GetReturnCode() != 0) {
        std::fprintf(
                stderr,
                "[selftest chimod] OpenIndex rc=%u\n",
                open_fut->GetReturnCode());
        return 1;
    }
    uint32_t ncont = 1;
    {
        auto cs = fetch_cluster_stats(client);
        if (cs.ok && cs.containers > 0) {
            ncont = static_cast<uint32_t>(cs.containers);
        }
    }
    std::printf(
            "[selftest chimod] opened: ntotal=%" PRIu64
            " d=%u nlist=%u containers=%u\n",
            (uint64_t)open_fut->ntotal_,
            open_fut->d_,
            open_fut->nlist_,
            ncont);

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
    // Add leg: append 10k fresh vectors through the AddTask client path
    // (grouped by owner container with the cluster's real container
    // count) AND through stock faiss; the post-add searches must again
    // match bitwise. Per-list append order equals input order on both
    // sides. The post-add search MUST run in owner-broadcast mode: after
    // an add, each container's in-memory sizes_ is current only for its
    // OWNED lists (a legacy full scan from one container would size
    // peer-owned reads with stale pre-add sizes). Gaussian float
    // distances make exact ties vanishingly unlikely, so the owner merge
    // is bitwise-deterministic here even multi-node.
    {
        const size_t nb2 = 10000;
        std::vector<float> xb2(nb2 * d);
        for (auto& v : xb2) v = nd(rng);
        std::vector<int64_t> ids2(nb2);
        for (size_t i = 0; i < nb2; ++i) {
            ids2[i] = static_cast<int64_t>(nb + i);
        }
        // Owner grouping needs the CTE tag identity (same hash inputs the
        // runtime uses for list_local_).
        auto tag_fut = CLIO_CTE_CLIENT->AsyncGetOrCreateTag(tag);
        tag_fut.Wait();
        auto tag_id = tag_fut->tag_id_;
        const size_t code_size = index.code_size;
        auto shares = group_add_batch(
                xb2.data(), ids2.data(), nb2, d, &quantizer,
                static_cast<uint32_t>(tag_id.major_),
                static_cast<uint32_t>(tag_id.minor_), ncont);
        AddSender sender(client, d, code_size, 256ull << 20);
        int64_t added = sender.send(shares);
        bool ok = added == static_cast<int64_t>(nb2);
        if (!ok) {
            std::fprintf(
                    stderr, "[selftest chimod add] appended %" PRId64
                            " of %zu\n",
                    added, nb2);
        }
        index.add(nb2, xb2.data());
        index.search(nq, xq.data(), k, D_ref.data(), I_ref.data());
        ok = ok &&
                chimod_search_parallel(
                        client, nq, k, nprobe, d, 2,
                        clio::run::faiss_ivf::kSearchModeOwner, true,
                        xq.data(), D_new.data(), I_new.data());
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
                "[selftest chimod add] %s\n",
                eq ? "D and I identical after add" : "MISMATCH");
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
    // Performance-study JSON record (perf_study_exp1).
    std::string json_out;
    long nb_m = 0;   // total DB size in millions of vectors (JSON metadata)
    int nodes = 0;   // topology (JSON metadata; cross-checked vs Stats)
};

// Load index metadata only (the lists live in CTE, read by the ChiMod).
// SKIP_IVF_DATA works for OnDisk ("ilod") index files; plain
// ArrayInvertedLists ("ilar") files — e.g. the study's trained-only
// quantizer skeletons — reject it, so fall back to a full read (the
// loaded lists are never scanned here). Mirrors Runtime::OpenIndex.
std::unique_ptr<faiss::Index> read_index_skeleton(const std::string& path) {
    try {
        return std::unique_ptr<faiss::Index>(faiss::read_index(
                path.c_str(), faiss::IO_FLAG_SKIP_IVF_DATA));
    } catch (const std::exception&) {
        return std::unique_ptr<faiss::Index>(
                faiss::read_index(path.c_str()));
    }
}

// Per-node RAM threshold used across the study for the ratio metadata
// (the mmap study's page-cache boundary on 46.6 GiB nodes).
constexpr double kRamThresholdM = 92.3;

void write_exp1_json(
        const Args& a,
        const faiss::IndexIVF* ivf,
        uint64_t ntotal,
        int nprobe,
        int nsplit,
        const std::vector<PassResult>& passes) {
    FILE* f = fopen(a.json_out.c_str(), "w");
    FAISS_THROW_IF_NOT_FMT(f, "cannot write %s", a.json_out.c_str());
    const double per_node =
            a.nodes > 0 ? static_cast<double>(a.nb_m) / a.nodes : 0.0;
    fprintf(f, "{\n");
    fprintf(f, "  \"experiment\": \"perf_study_exp1\",\n");
    fprintf(f, "  \"nb_M\": %ld,\n", a.nb_m);
    fprintf(f, "  \"n_nodes\": %d,\n", a.nodes);
    fprintf(f, "  \"nb_M_per_node\": %.3f,\n", per_node);
    fprintf(f, "  \"per_node_ram_ratio\": %.4f,\n",
            per_node / kRamThresholdM);
    fprintf(f, "  \"backend\": \"cte_chimod\",\n");
    fprintf(f, "  \"volume\": \"%s\",\n", a.label.c_str());
    fprintf(f, "  \"factory\": \"IVF%zu,Flat\",\n", ivf->nlist);
    fprintf(f, "  \"nlist\": %zu,\n", ivf->nlist);
    fprintf(f, "  \"nprobe\": %d,\n", nprobe);
    fprintf(f, "  \"ntotal\": %" PRIu64 ",\n", ntotal);
    fprintf(f, "  \"nq\": %zu,\n", a.nq);
    fprintf(f, "  \"k\": %d,\n", a.k);
    fprintf(f, "  \"threads\": %d,\n", a.threads);
    fprintf(f, "  \"route\": \"%s\",\n", a.route.c_str());
    fprintf(f, "  \"inflight\": %d,\n", nsplit);
    fprintf(f, "  \"warm_runs\": %d,\n",
            a.passes > 1 ? a.passes - 1 : 0);
    fprintf(f, "  \"passes\": [\n");
    for (size_t i = 0; i < passes.size(); ++i) {
        fprint_pass_json(f, passes[i], i + 1 == passes.size());
    }
    fprintf(f, "  ]\n}\n");
    fclose(f);
    std::printf("[bench] wrote %s\n", a.json_out.c_str());
}

int run_timed(const Args& a) {
    FAISS_THROW_IF_NOT_MSG(!a.index_path.empty(), "--index required");
    FAISS_THROW_IF_NOT_MSG(!a.queries_path.empty(), "--queries required");
    FAISS_THROW_IF_NOT_MSG(!a.tag.empty(), "--tag required");

    // The client needs d / nlist / nprobe here.
    std::unique_ptr<faiss::Index> owner = read_index_skeleton(a.index_path);
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

    // Cross-check --nodes against the cluster's container count.
    {
        auto s = fetch_cluster_stats(chimod_client);
        if (s.ok && a.nodes > 0 &&
            s.containers != static_cast<uint64_t>(a.nodes)) {
            std::fprintf(
                    stderr,
                    "[bench] FATAL: --nodes %d but cluster reports %" PRIu64
                    " containers\n",
                    a.nodes,
                    s.containers);
            return 4;
        }
    }

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
    std::vector<PassResult> results;
    for (int p = 0; p < a.passes; ++p) {
        std::string pass =
                p == 0 ? "cold" : ("warm" + std::to_string(p - 1));
        PassResult r = run_one_pass(
                chimod_client,
                pass,
                a.nq,
                a.k,
                nprobe,
                d,
                nsplit,
                mode,
                owner_route,
                xq.data(),
                D.data(),
                I.data());
        std::printf(
                "  %-6s qps=%9.1f  elapsed=%8.3fs  majflt/q=%8.1f  "
                "read_MB=%8.1f  disk_act=%5.1f%%  disk_rd_MBps=%7.1f  "
                "di_hash=%016llx  canon=%016llx\n",
                pass.c_str(),
                r.qps,
                r.elapsed_s,
                r.majflt / (double)a.nq,
                r.read_bytes / (1024.0 * 1024),
                r.disk.ok ? r.disk.active_pct : -1.0,
                r.disk.ok ? r.disk.read_mbps : -1.0,
                (unsigned long long)r.dihash,
                (unsigned long long)r.canon);
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
                     (unsigned long long)r.dihash,
                     (unsigned long long)r.canon,
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
                    r.elapsed_s,
                    r.qps,
                    r.majflt,
                    r.majflt / (double)a.nq,
                    r.read_bytes,
                    notes);
        }
        results.push_back(std::move(r));
    }
    if (csv) {
        fclose(csv);
    }
    if (!a.json_out.empty()) {
        write_exp1_json(
                a, ivf, open_fut->ntotal_, nprobe, nsplit, results);
    }
    // Broadcast + summing AggregateOut => cluster-total counters.
    auto sf = chimod_client.AsyncStats(clio::run::PoolQuery::Broadcast(), 1);
    sf.Wait();
    if (sf->GetReturnCode() == 0) {
        std::printf(
                "[stats] searches=%llu lists_read=%llu GB_read=%.2f "
                "read_wait_s=%.2f scan_s=%.2f nodes=%llu\n",
                (unsigned long long)sf->searches_,
                (unsigned long long)sf->lists_fetched_,
                sf->bytes_fetched_ / (1024.0 * 1024 * 1024),
                sf->fetch_wait_us_ / 1e6,
                sf->scan_us_ / 1e6,
                (unsigned long long)sf->containers_);
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
        else if (s == "--json-out") a.json_out = next("--json-out");
        else if (s == "--nb-m") a.nb_m = atol(next("--nb-m").c_str());
        else if (s == "--nodes") a.nodes = atoi(next("--nodes").c_str());
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
