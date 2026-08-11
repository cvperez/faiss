/*
 * bench_ivf_mixed — mixed 80/20 read/write benchmark for the faiss_ivf
 * ChiMod (performance study, Exp 2). Mirrors the mmap study's
 * ondisk_step4_exp2.py semantics with the CTE ChiMod as the backend:
 *
 *   Stage 1  OpenIndex (broadcast).
 *   Stage 2  baseline read-only passes (cold + warm0 + warm1, one batched
 *            owner-broadcast search per pass — same as bench_ivf_qps).
 *   Stage 3  mixed stage, FAISS_MIXED_DURATION_S wall seconds:
 *            - readers: T_READ single-query owner-broadcast searches kept
 *              in flight from ONE thread (closed-loop concurrency T_READ,
 *              per-query completion timestamps — the mmap study's 4
 *              reader threads without new client thread-safety
 *              assumptions);
 *            - writer: a std::thread streams BigANN vectors from offset
 *              NB_M, quantizer-assigns and owner-groups them (pure FAISS,
 *              no clio calls — assignment of a 2.5M batch takes minutes
 *              and must not starve the readers); the main loop then
 *              drains the readers, brackets the write window, sends the
 *              AddTasks (concurrent across containers, sequential per
 *              container), resumes the readers and paces to the 80/20
 *              write fraction: sleep = t_add * (1-f)/f.
 *            - per-query completions bucket into 10 s QPS windows tagged
 *              in_write when they overlap a write window; the whole stage
 *              is bracketed by broadcast Stats for mixed_disk_active.
 *   Stage 4  the client rewrites the "sizes" blob (tracked per-list
 *            deltas), then post-write passes (post_cold, post_warm0/1).
 *
 * The JSON record (perf_study_exp2) is rewritten atomically every ~60 s
 * during Stage 3 so a crashed run still salvages its windows.
 *
 * No-headroom cells (NB_M == available corpus): the writer never starts,
 * the readers run the full duration, achieved_write_fraction = 0 — same
 * as the mmap study's 500M cells.
 */

#include <unistd.h>

#include <atomic>
#include <cinttypes>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <vector>

#include <omp.h>

#include <faiss/IndexIVF.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/index_io.h>

#include "cte_client.h"

#include "bench_common.h"

#ifdef HAVE_FAISS_IVF_CHIMOD
#include <clio_runtime/faiss_ivf/faiss_ivf_client.h>
#endif

using faiss::idx_t;

#ifndef HAVE_FAISS_IVF_CHIMOD
int main() {
    std::fprintf(stderr, "built without chimod support\n");
    return 3;
}
#else

namespace {

using bench_common::AddSender;
using bench_common::AddShare;
using bench_common::ClusterStats;
using bench_common::DiskWindow;
using bench_common::disk_window;
using bench_common::fetch_cluster_stats;
using bench_common::fprint_pass_json;
using bench_common::group_add_batch;
using bench_common::load_bvecs_queries;
using bench_common::now_s;
using bench_common::PassResult;
using bench_common::run_one_pass;

constexpr double kRamThresholdM = 92.3;
constexpr int kBvecsRec = 132;  // 4-byte dim prefix + 128 uint8

struct Args {
    std::string index_path;    // trained skeleton (quantizer)
    std::string queries_path;  // bigann_query.bvecs
    std::string bigann_path;   // bigann_base.bvecs (write stream source)
    std::string tag;
    std::string label = "unknown";
    std::string json_out;
    long nb_m = 0;  // total DB size in millions (write stream starts here)
    int nodes = 1;
    size_t nq = 500;
    int threads = 8;
    int k = 10;
    int nprobe_override = 0;
    int t_read = 4;                    // outstanding single-query searches
    size_t batch_size = 2500000;       // vectors per write batch
    double write_fraction = 0.20;
    double mixed_duration_s = 3600;
    double qps_window_s = 10.0;
    size_t add_max_bytes = 256ull << 20;  // AddTask payload cap
};

/*** ---------------- writer thread: prepare add batches ---------------- ***/

// One prepared write batch: owner-grouped shares + bookkeeping.
struct PreparedBatch {
    std::vector<AddShare> shares;
    size_t n = 0;
    int64_t id_begin = 0;
};

// Streams BigANN vectors from offset start_vec, assigns them with the
// (local) quantizer, groups them per owner container, and hands each
// PreparedBatch to the main loop through a single-slot queue. Pure FAISS
// + file I/O — every clio client call stays on the main thread.
class BatchPreparer {
  public:
    BatchPreparer(
            const Args& a,
            const faiss::Index* quantizer,
            size_t d,
            uint32_t tag_major,
            uint32_t tag_minor,
            int64_t start_vec,
            int64_t end_vec)
            : a_(a),
              quantizer_(quantizer),
              d_(d),
              tag_major_(tag_major),
              tag_minor_(tag_minor),
              cursor_(start_vec),
              end_(end_vec) {
        if (cursor_ < end_) {
            thread_ = std::thread([this] { Run(); });
        }
    }

    ~BatchPreparer() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    bool exhausted() {
        std::lock_guard<std::mutex> lk(mu_);
        return done_ && !ready_;
    }

    // Non-blocking: take the prepared batch if one is waiting.
    bool try_take(PreparedBatch& out) {
        std::lock_guard<std::mutex> lk(mu_);
        if (!ready_) {
            return false;
        }
        out = std::move(slot_);
        ready_ = false;
        cv_.notify_all();
        return true;
    }

  private:
    void Run() {
        FILE* f = fopen(a_.bigann_path.c_str(), "rb");
        if (!f) {
            std::fprintf(stderr, "[writer] cannot open %s\n",
                         a_.bigann_path.c_str());
            std::lock_guard<std::mutex> lk(mu_);
            done_ = true;
            return;
        }
        std::vector<uint8_t> raw;
        std::vector<float> vecs;
        std::vector<int64_t> ids;
        while (true) {
            int64_t n;
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (stop_) {
                    break;
                }
                n = std::min<int64_t>(
                        static_cast<int64_t>(a_.batch_size), end_ - cursor_);
            }
            if (n <= 0) {
                break;
            }
            raw.resize(static_cast<size_t>(n) * kBvecsRec);
            if (fseeko(f, cursor_ * kBvecsRec, SEEK_SET) != 0 ||
                fread(raw.data(), kBvecsRec, n, f) !=
                        static_cast<size_t>(n)) {
                std::fprintf(stderr, "[writer] short read at vec %" PRId64
                                     "\n", cursor_);
                break;
            }
            vecs.resize(static_cast<size_t>(n) * d_);
            for (int64_t i = 0; i < n; ++i) {
                const uint8_t* rec = raw.data() + i * kBvecsRec + 4;
                float* dst = vecs.data() + i * d_;
                for (size_t j = 0; j < d_; ++j) {
                    dst[j] = static_cast<float>(rec[j]);
                }
            }
            ids.resize(n);
            for (int64_t i = 0; i < n; ++i) {
                ids[i] = cursor_ + i;
            }
            PreparedBatch b;
            b.n = static_cast<size_t>(n);
            b.id_begin = cursor_;
            b.shares = group_add_batch(
                    vecs.data(), ids.data(), b.n, d_, quantizer_,
                    tag_major_, tag_minor_,
                    static_cast<uint32_t>(a_.nodes));
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [this] { return !ready_ || stop_; });
            if (stop_) {
                break;
            }
            slot_ = std::move(b);
            ready_ = true;
            cursor_ += n;
        }
        fclose(f);
        std::lock_guard<std::mutex> lk(mu_);
        done_ = true;
    }

    const Args& a_;
    const faiss::Index* quantizer_;
    const size_t d_;
    const uint32_t tag_major_, tag_minor_;
    int64_t cursor_;
    const int64_t end_;
    std::thread thread_;
    std::mutex mu_;
    std::condition_variable cv_;
    PreparedBatch slot_;
    bool ready_ = false;
    bool done_ = false;
    bool stop_ = false;
};

/*** ---------------------- reader slots (T_READ) ----------------------- ***/

// One outstanding single-query owner-broadcast search. Buffers are
// allocated once and reused for the whole run (small, but hours of
// per-query Allocate/Free churn is exactly what the dev allocator
// dislikes).
struct ReaderSlot {
    ctp::ipc::FullPtr<char> q, dd, ii;
    clio::run::Future<clio::run::faiss_ivf::SearchTask> fut;
    double t_issue = 0;
    bool busy = false;
};

struct Completion {
    double t;       // seconds since mixed-stage start
    double lat_ms;  // request latency
};

struct WriteEvent {
    double t_start = 0, t_end = 0;  // seconds since mixed-stage start
    double add_s = 0;               // == t_end - t_start
    int64_t vectors = 0;
    int64_t id_begin = 0;
};

/*** ---------------------------- JSON out ------------------------------ ***/

struct MixedState {
    std::vector<PassResult> baseline;
    std::vector<PassResult> postwrite;
    std::vector<Completion> completions;
    std::vector<WriteEvent> writes;
    DiskWindow mixed_disk;
    double mixed_elapsed_s = 0;
    double achieved_wf = 0;
    uint64_t ntotal_base = 0;
    int64_t vectors_added = 0;
    long search_errors = 0;
};

void write_json(
        const Args& a,
        const faiss::IndexIVF* ivf,
        int nprobe,
        const MixedState& st) {
    const std::string tmp = a.json_out + ".tmp";
    FILE* f = fopen(tmp.c_str(), "w");
    if (!f) {
        std::fprintf(stderr, "WARN: cannot write %s\n", tmp.c_str());
        return;
    }
    const double per_node = static_cast<double>(a.nb_m) / a.nodes;
    fprintf(f, "{\n");
    fprintf(f, "  \"experiment\": \"perf_study_exp2\",\n");
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
    fprintf(f, "  \"nq\": %zu,\n", a.nq);
    fprintf(f, "  \"k\": %d,\n", a.k);
    fprintf(f, "  \"threads\": %d,\n", a.threads);
    fprintf(f, "  \"t_read\": %d,\n", a.t_read);
    fprintf(f, "  \"batch_size\": %zu,\n", a.batch_size);
    fprintf(f, "  \"write_fraction\": %.3f,\n", a.write_fraction);
    fprintf(f, "  \"mixed_duration_s\": %.1f,\n", a.mixed_duration_s);
    fprintf(f, "  \"qps_window_s\": %.1f,\n", a.qps_window_s);
    fprintf(f, "  \"ntotal_base\": %" PRIu64 ",\n", st.ntotal_base);
    fprintf(f, "  \"vectors_added\": %" PRId64 ",\n", st.vectors_added);
    fprintf(f, "  \"ntotal_final\": %" PRIu64 ",\n",
            st.ntotal_base + static_cast<uint64_t>(st.vectors_added));
    fprintf(f, "  \"achieved_write_fraction\": %.4f,\n", st.achieved_wf);
    fprintf(f, "  \"search_errors\": %ld,\n", st.search_errors);

    fprintf(f, "  \"baseline_passes\": [\n");
    for (size_t i = 0; i < st.baseline.size(); ++i) {
        fprint_pass_json(f, st.baseline[i], i + 1 == st.baseline.size());
    }
    fprintf(f, "  ],\n");

    // 10 s QPS windows from the completion timestamps; a window is
    // in_write when it overlaps any write window. Only fully elapsed
    // windows are emitted (a partial tail window is not a throughput
    // measurement).
    const double W = a.qps_window_s;
    const size_t nwin = st.mixed_elapsed_s >= W
            ? static_cast<size_t>(st.mixed_elapsed_s / W)
            : 0;
    std::vector<std::vector<double>> lat(nwin);
    for (const auto& c : st.completions) {
        const size_t w = static_cast<size_t>(c.t / W);
        if (w < nwin) {
            lat[w].push_back(c.lat_ms);
        }
    }
    fprintf(f, "  \"qps_series\": [\n");
    for (size_t w = 0; w < nwin; ++w) {
        const double w0 = w * W, w1 = (w + 1) * W;
        bool in_write = false;
        for (const auto& e : st.writes) {
            if (e.t_start < w1 && e.t_end > w0) {
                in_write = true;
                break;
            }
        }
        auto& v = lat[w];
        std::sort(v.begin(), v.end());
        auto pct = [&](double p) {
            return v.empty()
                    ? 0.0
                    : v[std::min(v.size() - 1,
                                 static_cast<size_t>(p * v.size()))];
        };
        fprintf(f,
                "    {\"t_s\": %.1f, \"qps\": %.4f, \"n_queries\": %zu, "
                "\"in_write\": %s, \"p50_ms\": %.2f, \"p95_ms\": %.2f, "
                "\"p99_ms\": %.2f}%s\n",
                w0, v.size() / W, v.size(), in_write ? "true" : "false",
                pct(0.50), pct(0.95), pct(0.99),
                w + 1 == nwin ? "" : ",");
    }
    fprintf(f, "  ],\n");

    fprintf(f, "  \"write_events\": [\n");
    for (size_t i = 0; i < st.writes.size(); ++i) {
        const auto& e = st.writes[i];
        fprintf(f,
                "    {\"t_start_s\": %.3f, \"t_end_s\": %.3f, "
                "\"add_duration_s\": %.3f, \"vectors_added\": %" PRId64
                ", \"id_begin\": %" PRId64 "}%s\n",
                e.t_start, e.t_end, e.add_s, e.vectors, e.id_begin,
                i + 1 == st.writes.size() ? "" : ",");
    }
    fprintf(f, "  ],\n");

    if (st.mixed_disk.ok) {
        fprintf(f,
                "  \"mixed_disk_active\": {\"active_pct\": %.3f, "
                "\"read_mbps\": %.3f, \"write_mbps\": %.3f},\n",
                st.mixed_disk.active_pct, st.mixed_disk.read_mbps,
                st.mixed_disk.write_mbps);
    } else {
        fprintf(f, "  \"mixed_disk_active\": null,\n");
    }

    fprintf(f, "  \"postwrite_passes\": [\n");
    for (size_t i = 0; i < st.postwrite.size(); ++i) {
        fprint_pass_json(f, st.postwrite[i], i + 1 == st.postwrite.size());
    }
    fprintf(f, "  ]\n}\n");
    fclose(f);
    if (rename(tmp.c_str(), a.json_out.c_str()) != 0) {
        std::fprintf(stderr, "WARN: rename to %s failed\n",
                     a.json_out.c_str());
    }
}

/*** ---------------------------- sizes blob ----------------------------- ***/

// Read-modify-write the global "sizes" blob with the client's tracked
// per-list deltas. Runtime Adds do NOT touch "sizes" (cross-container RMW
// on one blob would race); each container's in-memory sizes_ is current
// for its owned lists, and this final rewrite makes the volume
// re-OPENable with the grown sizes.
bool rewrite_sizes_blob(
        const clio::cte::core::TagId& tag_id,
        size_t nlist,
        const std::vector<int64_t>& deltas) {
    auto* cte = CLIO_CTE_CLIENT;
    const size_t bytes = nlist * sizeof(int64_t);
    auto buf = CLIO_IPC->AllocateBuffer(bytes);
    FAISS_THROW_IF_NOT_MSG(!buf.IsNull(), "sizes: AllocateBuffer failed");
    auto g = cte->AsyncGetBlob(tag_id, "sizes", 0, bytes, 0,
                               buf.shm_.template Cast<void>());
    g.Wait();
    if (g->GetReturnCode() != 0) {
        std::fprintf(stderr, "[sizes] GetBlob rc=%u\n", g->GetReturnCode());
        CLIO_IPC->FreeBuffer(buf);
        return false;
    }
    auto* p = reinterpret_cast<int64_t*>(buf.ptr_);
    for (size_t l = 0; l < nlist; ++l) {
        p[l] += deltas[l];
    }
    auto put = cte->AsyncPutBlob(tag_id, "sizes", 0, bytes,
                                 buf.shm_.template Cast<void>());
    put.Wait();
    const bool ok = put->GetReturnCode() == 0;
    if (!ok) {
        std::fprintf(stderr, "[sizes] PutBlob rc=%u\n",
                     put->GetReturnCode());
    }
    CLIO_IPC->FreeBuffer(buf);
    return ok;
}

/*** ------------------------------ main --------------------------------- ***/

int run(const Args& a) {
    FAISS_THROW_IF_NOT_MSG(!a.index_path.empty(), "--index required");
    FAISS_THROW_IF_NOT_MSG(!a.queries_path.empty(), "--queries required");
    FAISS_THROW_IF_NOT_MSG(!a.tag.empty(), "--tag required");
    FAISS_THROW_IF_NOT_MSG(!a.json_out.empty(), "--json-out required");
    FAISS_THROW_IF_NOT_MSG(a.nb_m > 0, "--nb-m required");

    // Trained skeleton: quantizer + metadata. "ilar" files reject
    // SKIP_IVF_DATA, hence the fallback (same as bench_ivf_qps).
    std::unique_ptr<faiss::Index> owner;
    try {
        owner.reset(faiss::read_index(
                a.index_path.c_str(), faiss::IO_FLAG_SKIP_IVF_DATA));
    } catch (const std::exception&) {
        owner.reset(faiss::read_index(a.index_path.c_str()));
    }
    auto* ivf = dynamic_cast<faiss::IndexIVF*>(owner.get());
    FAISS_THROW_IF_NOT_MSG(ivf, "not an IVF index");
    FAISS_THROW_IF_NOT_MSG(
            ivf->metric_type == faiss::METRIC_L2, "exp2 assumes L2");

    FAISS_THROW_IF_NOT_MSG(
            faiss_iowarp::EnsureIOWarpClient(), "client init failed");
    clio::run::faiss_ivf::Client client;
    client.AsyncCreate(
                  clio::run::PoolQuery::Local(),
                  "faiss_ivf_bench",
                  clio::run::PoolId(600, 0))
            .Wait();
    auto open_fut = client.AsyncOpenIndex(
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
    const int nprobe = a.nprobe_override > 0
            ? a.nprobe_override
            : std::max<size_t>(1, ivf->nlist / 64);
    omp_set_num_threads(a.threads);
    const uint32_t mode = clio::run::faiss_ivf::kSearchModeOwner;
    const size_t code_size = ivf->code_size;
    const size_t nlist = ivf->nlist;

    // Cross-check --nodes against the cluster (the owner grouping and the
    // runtime's list_local_ both hash mod this count — a mismatch would
    // rc=8 every AddTask).
    {
        auto s = fetch_cluster_stats(client);
        FAISS_THROW_IF_NOT_FMT(
                !s.ok || s.containers == static_cast<uint64_t>(a.nodes),
                "--nodes %d but cluster reports %" PRIu64 " containers",
                a.nodes, s.containers);
    }

    // CTE tag identity for the owner hash + the final sizes rewrite.
    auto tag_fut = CLIO_CTE_CLIENT->AsyncGetOrCreateTag(a.tag);
    tag_fut.Wait();
    const auto tag_id = tag_fut->tag_id_;

    MixedState st;
    st.ntotal_base = open_fut->ntotal_;

    std::printf(
            "[mixed] volume=%s nlist=%zu nprobe=%d nq=%zu k=%d t_read=%d "
            "batch=%zu wf=%.2f duration=%.0fs ntotal=%" PRIu64 "\n",
            a.label.c_str(), nlist, nprobe, a.nq, a.k, a.t_read,
            a.batch_size, a.write_fraction, a.mixed_duration_s,
            st.ntotal_base);

    // ---- Stage 2: baseline read-only passes -----------------------------
    std::vector<float> D(a.nq * a.k);
    std::vector<idx_t> I(a.nq * a.k);
    for (int p = 0; p < 3; ++p) {
        std::string label = p == 0 ? "cold" : ("warm" + std::to_string(p - 1));
        PassResult r = run_one_pass(
                client, label, a.nq, a.k, nprobe, d, 1, mode, true,
                xq.data(), D.data(), I.data());
        std::printf("  [baseline] %-6s qps=%9.1f canon=%016llx\n",
                    label.c_str(), r.qps, (unsigned long long)r.canon);
        st.baseline.push_back(std::move(r));
    }
    write_json(a, ivf, nprobe, st);

    // ---- Stage 3: mixed --------------------------------------------------
    // Write-stream bounds: BigANN vectors [NB, avail) — none at the top
    // corpus size (pure-read cell).
    int64_t avail = 0;
    {
        struct stat sb;
        if (!a.bigann_path.empty() && stat(a.bigann_path.c_str(), &sb) == 0) {
            avail = sb.st_size / kBvecsRec;
        }
    }
    const int64_t write_begin = a.nb_m * 1000000LL;
    const int64_t write_end = std::max(avail, write_begin);
    if (write_end == write_begin) {
        std::printf("[mixed] no write headroom (avail=%" PRId64
                    ") — pure-read mixed stage\n", avail);
    }
    BatchPreparer preparer(
            a, ivf->quantizer, d,
            static_cast<uint32_t>(tag_id.major_),
            static_cast<uint32_t>(tag_id.minor_),
            write_begin, write_end);
    // Persistent per-container payload buffers for the whole mixed stage
    // (per-window Allocate/Free churns never-recycled client segments).
    AddSender add_sender(client, d, code_size, a.add_max_bytes);

    std::vector<ReaderSlot> slots(a.t_read);
    for (auto& s : slots) {
        s.q = CLIO_IPC->AllocateBuffer(d * sizeof(float));
        s.dd = CLIO_IPC->AllocateBuffer(a.k * sizeof(float));
        s.ii = CLIO_IPC->AllocateBuffer(a.k * sizeof(idx_t));
        FAISS_THROW_IF_NOT_MSG(
                !s.q.IsNull() && !s.dd.IsNull() && !s.ii.IsNull(),
                "reader: AllocateBuffer failed");
    }
    size_t qcursor = 0;
    std::vector<int64_t> list_deltas(nlist, 0);
    const double t_start = now_s();
    double next_write_t = t_start;  // eligible immediately, like exp2.py
    double next_json_t = t_start + 60;
    double add_time_total = 0;
    ClusterStats mix_s0 = fetch_cluster_stats(client);

    auto issue = [&](ReaderSlot& s) {
        std::memcpy(s.q.ptr_, xq.data() + qcursor * d, d * sizeof(float));
        qcursor = (qcursor + 1) % a.nq;
        s.t_issue = now_s();
        s.fut = client.AsyncSearch(
                clio::run::PoolQuery::Broadcast(), 1,
                static_cast<clio::run::u32>(a.k),
                static_cast<clio::run::u32>(nprobe),
                static_cast<clio::run::u32>(d), mode,
                s.q.shm_.template Cast<void>(),
                s.dd.shm_.template Cast<void>(),
                s.ii.shm_.template Cast<void>());
        s.busy = true;
    };
    auto retire = [&](ReaderSlot& s) {
        s.fut.Wait();
        const double t = now_s();
        if (s.fut->GetReturnCode() != 0) {
            ++st.search_errors;
        } else {
            st.completions.push_back(
                    {t - t_start, (t - s.t_issue) * 1000.0});
        }
        s.fut = clio::run::Future<clio::run::faiss_ivf::SearchTask>();
        s.busy = false;
    };
    auto drain_all = [&]() {
        for (auto& s : slots) {
            if (s.busy) {
                retire(s);
            }
        }
    };

    for (auto& s : slots) {
        issue(s);
    }
    while (now_s() - t_start < a.mixed_duration_s) {
        // Readers: refill completed slots.
        bool progressed = false;
        for (auto& s : slots) {
            if (s.busy && s.fut.IsComplete()) {
                retire(s);
                issue(s);
                progressed = true;
            }
        }
        // Writer: when a prepared batch is waiting and the 80/20 pacing
        // allows, run one write window.
        PreparedBatch batch;
        if (now_s() >= next_write_t && preparer.try_take(batch)) {
            drain_all();
            WriteEvent e;
            const double w0 = now_s();
            e.t_start = w0 - t_start;
            e.id_begin = batch.id_begin;
            int64_t added = add_sender.send(batch.shares);
            const double w1 = now_s();
            e.t_end = w1 - t_start;
            e.add_s = w1 - w0;
            if (added < 0) {
                std::fprintf(stderr, "[mixed] write batch FAILED at id %"
                             PRId64 " — stopping writer\n", batch.id_begin);
                st.writes.push_back(e);
                // Keep reading; no further writes (preparer drains).
                next_write_t = t_start + a.mixed_duration_s;
            } else {
                e.vectors = added;
                st.vectors_added += added;
                add_time_total += e.add_s;
                for (const auto& share : batch.shares) {
                    for (size_t j = 0; j + 1 < share.list_offs.size(); ++j) {
                        list_deltas[share.list_ids[j]] +=
                                share.list_offs[j + 1] - share.list_offs[j];
                    }
                }
                st.writes.push_back(e);
                // 80/20 pacing: writer busy-fraction ≈ write_fraction.
                next_write_t = w1 +
                        e.add_s * (1.0 - a.write_fraction) /
                                a.write_fraction;
                std::printf("[mixed] wrote %" PRId64 " vecs in %.1fs "
                            "(next window in %.0fs)\n",
                            added, e.add_s, next_write_t - w1);
            }
            for (auto& s : slots) {
                issue(s);
            }
            progressed = true;
        }
        if (now_s() >= next_json_t) {
            st.mixed_elapsed_s = now_s() - t_start;
            st.achieved_wf = add_time_total / st.mixed_elapsed_s;
            write_json(a, ivf, nprobe, st);
            next_json_t = now_s() + 60;
        }
        if (!progressed) {
            // Nothing completed: block briefly on the oldest reader
            // rather than spinning.
            for (auto& s : slots) {
                if (s.busy) {
                    retire(s);
                    issue(s);
                    break;
                }
            }
        }
    }
    drain_all();
    st.mixed_elapsed_s = now_s() - t_start;
    st.achieved_wf = add_time_total / st.mixed_elapsed_s;
    ClusterStats mix_s1 = fetch_cluster_stats(client);
    st.mixed_disk = disk_window(mix_s0, mix_s1, st.mixed_elapsed_s);
    for (auto& s : slots) {
        CLIO_IPC->FreeBuffer(s.q);
        CLIO_IPC->FreeBuffer(s.dd);
        CLIO_IPC->FreeBuffer(s.ii);
    }
    std::printf("[mixed] stage done: %.0fs, %zu completions, %zu writes, "
                "%" PRId64 " vectors added, wf=%.3f\n",
                st.mixed_elapsed_s, st.completions.size(),
                st.writes.size(), st.vectors_added, st.achieved_wf);
    write_json(a, ivf, nprobe, st);

    // ---- Stage 4: sizes rewrite + post-write passes ---------------------
    if (st.vectors_added > 0) {
        if (!rewrite_sizes_blob(tag_id, nlist, list_deltas)) {
            std::fprintf(stderr, "[mixed] sizes rewrite FAILED\n");
        }
    }
    for (int p = 0; p < 3; ++p) {
        std::string label =
                p == 0 ? "post_cold" : ("post_warm" + std::to_string(p - 1));
        PassResult r = run_one_pass(
                client, label, a.nq, a.k, nprobe, d, 1, mode, true,
                xq.data(), D.data(), I.data());
        std::printf("  [postwrite] %-10s qps=%9.1f canon=%016llx\n",
                    label.c_str(), r.qps, (unsigned long long)r.canon);
        st.postwrite.push_back(std::move(r));
    }
    write_json(a, ivf, nprobe, st);
    return st.search_errors == 0 ? 0 : 5;
}

} // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    Args a;
    if (const char* e = std::getenv("CHIMOD_ADD_MAX_BYTES")) {
        a.add_max_bytes = strtoull(e, nullptr, 10);
    }
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&](const char* opt) -> std::string {
            FAISS_THROW_IF_NOT_FMT(i + 1 < argc, "%s needs a value", opt);
            return argv[++i];
        };
        if (s == "--index") a.index_path = next("--index");
        else if (s == "--queries") a.queries_path = next("--queries");
        else if (s == "--bigann") a.bigann_path = next("--bigann");
        else if (s == "--tag") a.tag = next("--tag");
        else if (s == "--label") a.label = next("--label");
        else if (s == "--json-out") a.json_out = next("--json-out");
        else if (s == "--nb-m") a.nb_m = atol(next("--nb-m").c_str());
        else if (s == "--nodes") a.nodes = atoi(next("--nodes").c_str());
        else if (s == "--nq") a.nq = atoll(next("--nq").c_str());
        else if (s == "--threads") a.threads = atoi(next("--threads").c_str());
        else if (s == "--k") a.k = atoi(next("--k").c_str());
        else if (s == "--nprobe") a.nprobe_override = atoi(next("--nprobe").c_str());
        else if (s == "--t-read") a.t_read = atoi(next("--t-read").c_str());
        else if (s == "--batch-size") a.batch_size = atoll(next("--batch-size").c_str());
        else if (s == "--write-fraction") a.write_fraction = atof(next("--write-fraction").c_str());
        else if (s == "--mixed-duration") a.mixed_duration_s = atof(next("--mixed-duration").c_str());
        else if (s == "--qps-window") a.qps_window_s = atof(next("--qps-window").c_str());
        else if (s == "--add-max-bytes") a.add_max_bytes = strtoull(next("--add-max-bytes").c_str(), nullptr, 10);
        else {
            std::fprintf(stderr, "unknown arg: %s\n", s.c_str());
            return 2;
        }
    }
    return run(a);
}

#endif // HAVE_FAISS_IVF_CHIMOD
