/*
 * bench_common — shared harness code for the faiss_ivf ChiMod benchmarks
 * (bench_ivf_qps: selftest + read-only passes; bench_ivf_mixed: mixed
 * 80/20 read/write). Factored out of bench_ivf_qps.cpp unchanged, plus
 * the performance-study additions: per-pass cluster Stats bracketing
 * (disk metrics with the mmap study's /proc/diskstats semantics) and the
 * Add-path client (owner grouping + sequential-per-container sends).
 */

#pragma once

#include <sys/resource.h>
#include <sys/time.h>

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <faiss/Index.h>
#include <faiss/impl/FaissAssert.h>

#ifdef HAVE_FAISS_IVF_CHIMOD
#include <clio_runtime/faiss_ivf/faiss_ivf_client.h>
#endif

namespace bench_common {

using faiss::idx_t;

/*** ------------------------- utilities -------------------------------- ***/

inline long majflt_now() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_majflt;
}

inline long long io_read_bytes_now() {
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

inline double now_s() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// Load the first nq vectors of a .bvecs file (int32 dim + uint8[dim] per
// record) as float32.
inline std::vector<float> load_bvecs_queries(const std::string& path,
                                             size_t nq, int& d_out) {
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

// FNV-1a over the raw (D, I) buffers: a cheap bitwise-integrity gate at
// real-volume scale (identical searches produce identical hashes).
inline uint64_t di_hash(const float* D, const idx_t* I, size_t nq, int k) {
    uint64_t h = 1469598103934665603ULL;
    auto fnv = [&h](const void* p, size_t n) {
        const uint8_t* b = static_cast<const uint8_t*>(p);
        for (size_t i = 0; i < n; ++i) {
            h = (h ^ b[i]) * 1099511628211ULL;
        }
    };
    fnv(D, nq * k * sizeof(float));
    fnv(I, nq * k * sizeof(idx_t));
    return h;
}

// Canonical (order-independent) hash of the results: per query, sort the k
// (D, I) pairs by (D, then I) and FNV-1a the sorted stream. Owner-route
// merging can legitimately reorder equal-distance ties vs the single-node
// heap order, so di_hash may differ while the result SET is identical —
// canon compares the sets.
inline uint64_t canon_hash(const float* D, const idx_t* I, size_t nq, int k) {
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
inline void dump_di_file(const std::string& path, const float* D,
                         const idx_t* I, size_t nq, int k) {
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

#ifdef HAVE_FAISS_IVF_CHIMOD

/*** ------------------------- chimod drivers --------------------------- ***/

// One batched search through the chimod; D/I sized nq*k by the caller.
inline bool chimod_search(
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
inline bool chimod_search_parallel(
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

/*** ------------------- cluster stats + disk metrics ------------------- ***/

// One broadcast Stats snapshot: cluster-summed ChiMod counters plus the
// summed node-local /proc/diskstats counters (cumulative; containers = N).
struct ClusterStats {
    bool ok = false;
    uint64_t searches = 0, lists_fetched = 0, bytes_fetched = 0;
    uint64_t fetch_wait_us = 0, scan_us = 0;
    uint64_t adds = 0, add_vectors = 0, add_bytes = 0, add_us = 0;
    uint64_t containers = 0;
    uint64_t disk_io_ticks_ms = 0, disk_rd_sectors = 0, disk_wr_sectors = 0;
};

inline ClusterStats fetch_cluster_stats(
        clio::run::faiss_ivf::Client& client, bool reset = false) {
    ClusterStats s;
    auto fut = client.AsyncStats(
            clio::run::PoolQuery::Broadcast(), reset ? 1u : 0u);
    fut.Wait();
    if (fut->GetReturnCode() != 0) {
        std::fprintf(
                stderr, "cluster stats rc=%u\n", fut->GetReturnCode());
        return s;
    }
    s.ok = true;
    s.searches = fut->searches_;
    s.lists_fetched = fut->lists_fetched_;
    s.bytes_fetched = fut->bytes_fetched_;
    s.fetch_wait_us = fut->fetch_wait_us_;
    s.scan_us = fut->scan_us_;
    s.adds = fut->adds_;
    s.add_vectors = fut->add_vectors_;
    s.add_bytes = fut->add_bytes_;
    s.add_us = fut->add_us_;
    s.containers = fut->containers_;
    s.disk_io_ticks_ms = fut->disk_io_ticks_ms_;
    s.disk_rd_sectors = fut->disk_rd_sectors_;
    s.disk_wr_sectors = fut->disk_wr_sectors_;
    return s;
}

// Disk-activity window between two Stats snapshots, with the mmap study's
// exact formulas (ondisk_step4_exp1.py::_disk_active_metrics):
//   active_pct = Δio_ticks_ms / (elapsed_s*1000 * N) * 100  (per-node mean)
//   read/write MB/s = Δsectors * 512 / elapsed_s / 1e6      (cluster sum)
struct DiskWindow {
    bool ok = false;
    double active_pct = 0, read_mbps = 0, write_mbps = 0;
};

inline DiskWindow disk_window(
        const ClusterStats& s0, const ClusterStats& s1, double elapsed_s) {
    DiskWindow w;
    if (!s0.ok || !s1.ok || s1.containers == 0 || elapsed_s <= 0) {
        return w;
    }
    const double n = static_cast<double>(s1.containers);
    w.ok = true;
    w.active_pct = (s1.disk_io_ticks_ms - s0.disk_io_ticks_ms) /
            (elapsed_s * 1000.0 * n) * 100.0;
    w.read_mbps =
            (s1.disk_rd_sectors - s0.disk_rd_sectors) * 512.0 / elapsed_s /
            1e6;
    w.write_mbps =
            (s1.disk_wr_sectors - s0.disk_wr_sectors) * 512.0 / elapsed_s /
            1e6;
    return w;
}

/*** ------------------------- timed pass ------------------------------- ***/

// One batched search pass bracketed by broadcast Stats — everything the
// exp1 JSON needs for a "passes" entry.
struct PassResult {
    std::string label;
    size_t nq = 0;
    double elapsed_s = 0, qps = 0;
    long majflt = 0;
    long long read_bytes = 0;
    uint64_t dihash = 0, canon = 0;
    DiskWindow disk;
    uint64_t lists_fetched = 0, bytes_fetched = 0;  // deltas over the pass
    uint64_t nodes = 0;
};

inline PassResult run_one_pass(
        clio::run::faiss_ivf::Client& client,
        const std::string& label,
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
    PassResult r;
    r.label = label;
    r.nq = nq;
    ClusterStats s0 = fetch_cluster_stats(client);
    long mf0 = majflt_now();
    long long rb0 = io_read_bytes_now();
    double t0 = now_s();
    FAISS_THROW_IF_NOT_MSG(
            chimod_search_parallel(
                    client, nq, k, nprobe, d, nsplit, mode, owner_route, xq,
                    D, I),
            "chimod search failed");
    r.elapsed_s = now_s() - t0;
    r.majflt = majflt_now() - mf0;
    r.read_bytes = io_read_bytes_now() - rb0;
    r.qps = nq / r.elapsed_s;
    r.dihash = di_hash(D, I, nq, k);
    r.canon = canon_hash(D, I, nq, k);
    ClusterStats s1 = fetch_cluster_stats(client);
    r.disk = disk_window(s0, s1, r.elapsed_s);
    if (s0.ok && s1.ok) {
        r.lists_fetched = s1.lists_fetched - s0.lists_fetched;
        r.bytes_fetched = s1.bytes_fetched - s0.bytes_fetched;
        r.nodes = s1.containers;
    }
    return r;
}

// Emit one "passes" JSON array entry (plot-compatible field names:
// label/qps/disk_active_pct/disk_read_mbps/disk_write_mbps).
inline void fprint_pass_json(FILE* f, const PassResult& r, bool last) {
    fprintf(f,
            "    {\"label\": \"%s\", \"nq\": %zu, \"elapsed_s\": %.6f, "
            "\"qps\": %.4f, ",
            r.label.c_str(), r.nq, r.elapsed_s, r.qps);
    if (r.disk.ok) {
        fprintf(f,
                "\"disk_active_pct\": %.3f, \"disk_read_mbps\": %.3f, "
                "\"disk_write_mbps\": %.3f, ",
                r.disk.active_pct, r.disk.read_mbps, r.disk.write_mbps);
    } else {
        fprintf(f,
                "\"disk_active_pct\": null, \"disk_read_mbps\": null, "
                "\"disk_write_mbps\": null, ");
    }
    fprintf(f,
            "\"major_faults\": %ld, \"read_bytes\": %lld, "
            "\"cte_lists_fetched\": %" PRIu64 ", "
            "\"cte_bytes_fetched\": %" PRIu64 ", "
            "\"di_hash\": \"%016" PRIx64 "\", "
            "\"canon_hash\": \"%016" PRIx64 "\"}%s\n",
            r.majflt, r.read_bytes, r.lists_fetched, r.bytes_fetched,
            r.dihash, r.canon, last ? "" : ",");
}

/*** ------------------------- add-path client -------------------------- ***/

// One container's share of an add batch: vectors grouped by target list
// (lists ascending, input order preserved within a list — matching stock
// faiss add_core semantics so single-node results stay bitwise-identical).
struct AddShare {
    uint32_t container = 0;
    std::vector<int64_t> list_ids;   // ascending
    std::vector<int64_t> list_offs;  // prefix offsets in vectors, nlists+1
    std::vector<float> vecs;         // n*d float32 (codes for IVF-Flat)
    std::vector<int64_t> ids;        // n
    size_t n = 0;
};

// Assign a batch with the (local) coarse quantizer and split it into
// per-owner-container shares routed by ListOwnerContainer — the SAME hash
// the runtime's list_local_ map uses, so a share can never land on a
// container that does not own its lists.
inline std::vector<AddShare> group_add_batch(
        const float* xb,
        const int64_t* xids,
        size_t n,
        size_t d,
        const faiss::Index* quantizer,
        uint32_t tag_major,
        uint32_t tag_minor,
        uint32_t num_containers) {
    std::vector<float> dis(n);
    std::vector<idx_t> assign(n);
    quantizer->search(n, xb, 1, dis.data(), assign.data());
    // (owner, list) -> input rows, in input order.
    std::map<uint32_t, std::map<int64_t, std::vector<size_t>>> buckets;
    for (size_t i = 0; i < n; ++i) {
        const int64_t l = assign[i];
        FAISS_THROW_IF_NOT_MSG(l >= 0, "quantizer produced a negative list");
        const uint32_t owner = clio::run::faiss_ivf::ListOwnerContainer(
                tag_major, tag_minor, l, num_containers);
        buckets[owner][l].push_back(i);
    }
    std::vector<AddShare> shares;
    for (auto& [owner, lists] : buckets) {
        AddShare s;
        s.container = owner;
        s.list_offs.push_back(0);
        for (auto& [l, rows] : lists) {
            s.list_ids.push_back(l);
            for (size_t i : rows) {
                s.vecs.insert(
                        s.vecs.end(), xb + i * d, xb + (i + 1) * d);
                s.ids.push_back(xids[i]);
            }
            s.list_offs.push_back(static_cast<int64_t>(s.ids.size()));
        }
        s.n = s.ids.size();
        shares.push_back(std::move(s));
    }
    return shares;
}

// Plan of one AddTask: which slice of a share's payload it carries.
struct AddTaskPlan {
    std::vector<int64_t> lists;
    std::vector<int64_t> offs;       // prefix offsets in vectors
    std::vector<size_t> src_begin;   // payload source row per segment
    size_t n = 0;
};

// Chop a share into AddTasks capped at max_bytes of payload each (on list
// boundaries where possible, inside a list when a single segment exceeds
// the cap — appends stay ordered because tasks are strictly sequential
// per container).
inline std::vector<AddTaskPlan> plan_add_tasks(
        const AddShare& share, size_t code_size, size_t max_bytes) {
    const size_t per_vec = code_size + sizeof(int64_t);
    const size_t max_vecs = std::max<size_t>(1, max_bytes / per_vec);
    std::vector<AddTaskPlan> plans;
    size_t seg = 0;
    size_t seg_off = 0;
    while (seg < share.list_ids.size()) {
        AddTaskPlan p;
        p.offs.push_back(0);
        while (seg < share.list_ids.size() && p.n < max_vecs) {
            const size_t seg_total = static_cast<size_t>(
                    share.list_offs[seg + 1] - share.list_offs[seg]);
            const size_t avail = seg_total - seg_off;
            const size_t take = std::min(avail, max_vecs - p.n);
            p.lists.push_back(share.list_ids[seg]);
            p.src_begin.push_back(
                    static_cast<size_t>(share.list_offs[seg]) + seg_off);
            p.n += take;
            p.offs.push_back(static_cast<int64_t>(p.n));
            seg_off += take;
            if (seg_off == seg_total) {
                ++seg;
                seg_off = 0;
            }
        }
        plans.push_back(std::move(p));
    }
    return plans;
}

// Sends add shares: CONCURRENT across containers (disjoint lists),
// strictly sequential within a container (the read-modify-write on a
// list's blob must see the previous append). One in-flight AddTask per
// container, refilled on completion.
//
// STATEFUL on purpose: each container's payload buffers are allocated
// ONCE at the task-size cap and reused for every task of every write
// window — per-task Allocate/Free churns ~1.3 GB of never-recycled
// client segments per 2.5M-vector window (dev allocator gotcha) and
// contributed to a daemon OOM in the first hour-long mixed run.
class AddSender {
  public:
    AddSender(clio::run::faiss_ivf::Client& client,
              size_t d,
              size_t code_size,
              size_t max_bytes)
            : client_(client),
              d_(d),
              code_size_(code_size),
              max_vecs_(std::max<size_t>(
                      1, max_bytes / (code_size + sizeof(int64_t)))) {}

    ~AddSender() {
        for (auto& [c, lane] : lanes_) {
            if (!lane.cbuf.IsNull()) {
                CLIO_IPC->FreeBuffer(lane.cbuf);
            }
            if (!lane.ibuf.IsNull()) {
                CLIO_IPC->FreeBuffer(lane.ibuf);
            }
        }
    }

    // Returns total vectors appended, or -1 if any task failed.
    int64_t send(const std::vector<AddShare>& shares) {
        struct Active {
            Lane* lane;
            const AddShare* share;
            std::vector<AddTaskPlan> plans;
            size_t next = 0;
        };
        std::vector<Active> act;
        for (const auto& s : shares) {
            if (s.n == 0) {
                continue;
            }
            Active a;
            a.lane = &LaneFor(s.container);
            a.share = &s;
            a.plans = plan_add_tasks(
                    s, code_size_, max_vecs_ * (code_size_ + sizeof(int64_t)));
            act.push_back(std::move(a));
        }
        int64_t appended = 0;
        bool failed = false;
        auto issue = [&](Active& a) {
            const AddTaskPlan& p = a.plans[a.next];
            for (size_t j = 0; j < p.lists.size(); ++j) {
                const size_t cnt =
                        static_cast<size_t>(p.offs[j + 1] - p.offs[j]);
                std::memcpy(
                        a.lane->cbuf.ptr_ +
                                static_cast<size_t>(p.offs[j]) * code_size_,
                        a.share->vecs.data() + p.src_begin[j] * d_,
                        cnt * code_size_);
                std::memcpy(
                        a.lane->ibuf.ptr_ +
                                static_cast<size_t>(p.offs[j]) *
                                        sizeof(int64_t),
                        a.share->ids.data() + p.src_begin[j],
                        cnt * sizeof(int64_t));
            }
            a.lane->fut = client_.AsyncAdd(
                    clio::run::PoolQuery::DirectHash(a.share->container),
                    static_cast<clio::run::u32>(p.n),
                    static_cast<clio::run::u32>(d_),
                    static_cast<clio::run::u32>(code_size_),
                    p.lists, p.offs,
                    a.lane->cbuf.shm_.template Cast<void>(),
                    a.lane->ibuf.shm_.template Cast<void>());
            a.lane->busy = true;
            ++a.next;
        };
        auto retire = [&](Active& a) {
            a.lane->fut.Wait();  // complete (or the lone pending one)
            if (a.lane->fut->GetReturnCode() != 0) {
                std::fprintf(
                        stderr, "chimod add rc=%u (container %u)\n",
                        a.lane->fut->GetReturnCode(), a.share->container);
                failed = true;
            } else {
                appended += static_cast<int64_t>(a.lane->fut->added_);
            }
            a.lane->fut =
                    clio::run::Future<clio::run::faiss_ivf::AddTask>();
            a.lane->busy = false;
        };
        for (auto& a : act) {
            issue(a);  // one in-flight task per container
        }
        while (!failed) {
            bool any_busy = false;
            bool progressed = false;
            for (auto& a : act) {
                if (!a.lane->busy) {
                    continue;
                }
                if (a.lane->fut.IsComplete()) {
                    retire(a);
                    if (!failed && a.next < a.plans.size()) {
                        issue(a);
                        any_busy = true;
                    }
                    progressed = true;
                } else {
                    any_busy = true;
                }
            }
            if (!any_busy) {
                break;
            }
            if (!progressed) {
                // Nothing completed this sweep: block on the first busy
                // lane instead of spinning.
                for (auto& a : act) {
                    if (a.lane->busy) {
                        retire(a);
                        if (!failed && a.next < a.plans.size()) {
                            issue(a);
                        }
                        break;
                    }
                }
            }
        }
        // On failure, drain whatever is still in flight.
        for (auto& a : act) {
            if (a.lane->busy) {
                retire(a);
            }
        }
        return failed ? -1 : appended;
    }

  private:
    struct Lane {
        ctp::ipc::FullPtr<char> cbuf, ibuf;
        clio::run::Future<clio::run::faiss_ivf::AddTask> fut;
        bool busy = false;
    };

    Lane& LaneFor(uint32_t container) {
        auto [it, fresh] = lanes_.try_emplace(container);
        if (fresh) {
            it->second.cbuf = CLIO_IPC->AllocateBuffer(max_vecs_ * code_size_);
            it->second.ibuf =
                    CLIO_IPC->AllocateBuffer(max_vecs_ * sizeof(int64_t));
            FAISS_THROW_IF_NOT_MSG(
                    !it->second.cbuf.IsNull() && !it->second.ibuf.IsNull(),
                    "add: AllocateBuffer failed");
        }
        return it->second;
    }

    clio::run::faiss_ivf::Client& client_;
    const size_t d_, code_size_, max_vecs_;
    std::map<uint32_t, Lane> lanes_;
};

#endif  // HAVE_FAISS_IVF_CHIMOD

}  // namespace bench_common
