# FAISS × IOWarp: IVF-on-disk over CTE — Implementation Plan

**Status:** ready to execute (written on the dev machine; execution happens on Ares)
**Repos involved:** this faiss fork (`cvperez/faiss`, upstream v1.14.3) and
`iowarp/clio-core`, consumed as the **`iowarp-core` pip wheel** (native
shared libraries + `clio_run`; user requirement — no source build), with
matching-version headers fetched from the clio-core source tree (headers
only, never compiled).
**Full design rationale:** `IVF_ONDISK_IOWARP_PROPOSAL.md` at the TFM workspace
root (`TFM-RAG-INFRASTRUCTURE/`). This file is self-contained: everything
needed to implement and run is here.

---

## Progress log (2026-07-11, Ares login node)

- [x] **Phase 0** — faiss built (`FAISS_OPT_LEVEL=generic`, shared, no python) →
  `~/faiss-install`; iowarp-core wheel 2.1.0 installed; headers from the
  `~/clio-core-v2.1.0` worktree. Deviations: (a) explicit
  `-DBLAS_LIBRARIES=/lib/x86_64-linux-gnu/libblas.so.3` (no dev symlinks on
  the node), (b) `zmq.h` v4.3.5 vendored under `third_party/zmq/` (wheel
  statically links+exports libzmq but ships no header), (c) the in-tree CTP
  compile definitions are replicated in `CMakeLists.txt`
  (`IOWARP_COMPILE_DEFS`, StdThread model). `cte_smoke` put/get round-trip
  **PASS** against the wheel runtime.
- [x] **Phase 1 (Level 0)** — `IOWarpInvertedLists` + `ivf_to_iowarp` +
  `bench_ivf_qps` build clean; `--selftest` **PASS**: bitwise-identical
  `(D, I)` vs `ArrayInvertedLists` for METRIC_L2 and METRIC_INNER_PRODUCT.
- [x] Gate 3: `ivf_to_iowarp --verify 64` on the real `ondisk_nb10M` pair —
  **PASS** (4.84 GiB compact, 64 random lists byte-identical).
- [x] **Phase 3 (chimod)** — `libclio_faiss_ivf_{client,runtime}.so` build
  against wheel libs + pinned headers; the wheel runtime dlopens the
  out-of-tree module and composes pool 600.0. `--selftest-chimod` **PASS**:
  v0 and v1 both bitwise-identical `(D, I)` to stock FAISS. Key fix: never
  call `CLIO_CTE_CLIENT_INIT()` inside a handler (its blocking
  `create_task.Wait()` deadlocks the cooperative worker and wedges the
  whole runtime); the runtime binds a CTE client directly to
  `kCtePoolId` (512.0). Additional vendored headers: msgpack-c 6.1.0
  (+ generated `sysdep.h`/`pack_template.h`), cereal 1.3.2.
- [x] **Phase 4 (2026-07-12/13)** — campaign iterations:
  (a) First chain failed: compute-partition images have no BLAS (bundled
  real copies into `~/faiss-install/lib`; env script no longer relinks on
  compute nodes); `sudo drop_caches` unavailable in batch — switched to the
  baseline's own `posix_fadvise(DONTNEED)` mechanism. mmap re-runs removed
  per user (baseline numbers already exist).
  (b) Pilot delivered first data: **L0 iowarp on nb10M = 14.5–16.3 QPS flat
  (cold==warm, 0 majflt, 0 disk)** — fetch-RTT-bound at ~32k per-probe
  round-trips/pass; quantifies the client-boundary cost.
  (c) chimod pass hung: Search issued all 4096 GetBlobs at once →
  runtime-queue livelock (queue_depth 1024). Fixed with windowed issuance
  (128 in flight) in v0/v1; v1 now frees each list right after scanning
  (peak shm O(window) — scales past the allocator size); OpenIndex now
  replaces state when path/tag change. Local validation on real nb10M:
  selftest v0/v1 PASS, **chimod v1 = 73–75 QPS** (~5× L0) on the login
  node.
  (d) Pilot 21894 completed the full matrix on nb10M (exclusive node):
  **L0 = 14.5–15.1 QPS** (flat, 0 majflt); **chimod v1 single-task =
  62–66 QPS**; task-split sweep monotonically WORSE (inflight 2/4/8 =
  53/42/34 QPS — splitting sacrifices cross-task fetch dedup; workload is
  fetch-bound at this scale). User decision: L0 not run at larger volumes
  (its client-boundary cost is quantified; would cost hours at 178M).
  (e) **Intra-task parallel scan** (2026-07-13): each arrived list's
  (query, probe) pairs now scan across 8 per-thread scanners inside the
  one task — full fetch dedup AND the baseline's 8-thread CPU budget.
  (OMP region hoisted to a plain function: gcc 11 ICEs on OMP inside
  C++20 coroutines.) Correctness: v0/v1/split all bitwise-identical.
  Login-node validation on real nb10M: **125–162 QPS** (vs 65 before;
  vs L0 15; step2 mmap warm reference ~81–93 at 50M). kStats: fetch-wait
  still ≥ scan time → remaining headroom is the GetBlob copy (zero-copy
  peek API = future upstream CTE proposal). Campaign paused for analysis
  per user; comparability config is now inflight=1 (8 scan threads inside
  the task == baseline's 8 OMP threads).
- Experiment note: pre-built volumes from the user's step2/step3 campaigns
  are reused (no §5.1 index builds); the comparability protocol against
  those baselines is pinned in `README.md`/plan file.

---

## 0. Goal and context

Measured on 1 Ares node: FAISS `OnDiskInvertedLists` (inverted lists in a
mmapped `.ivfdata` file, placement delegated to the OS page cache) delivers
good QPS while the index fits in RAM, but once the index exceeds RAM, page-ins
dominate and QPS collapses (prior Experiments 1/2: Panel A QPS-vs-size,
Figures 3a/3b I/O + page-cache residency).

**Hypothesis:** storing the inverted lists as IOWarp/CTE blobs — placement and
tiering managed explicitly by CTE's Data Placement Engine instead of the page
cache — improves QPS in the index ≫ RAM regime. Two levels:

- **Level 0** — `IOWarpInvertedLists`, a client-side `faiss::InvertedLists`
  backend: FAISS search runs unchanged in the client process; lists are
  fetched from CTE via shared memory.
- **Level 1** — a `faiss_ivf` ChiMod: FAISS IVF search runs *inside* the CLIO
  runtime, next to the data; only queries in / top-k out cross the client
  boundary; list fetch and code scanning overlap via coroutines.

**Experiment (both levels):** 1 Ares compute node (48 GB RAM), two synthetic
`IVF,Flat` indexes — **24 GB (0.5× RAM)** and **96 GB (2× RAM)** — measure
QPS, plot systems side by side.

This is written as a contribution to FAISS: everything lives in this
`contrib/iowarp/` folder, self-contained, styled after `demos/rocksdb_ivf/`
(the existing precedent for an external-dependency `InvertedLists` backend).

### Assumptions — confirm with the user before running

1. **Dataset:** synthetic random float32, d=128, `IVF,Flat`. Recall is NOT
   measured (outputs are identical by construction once the backend is
   correct), so random data is fine and gives exact size control. If the user
   prefers the dataset from their earlier baseline runs, ask where it lives.
2. **Ares environment:** unknown what is already installed under the user's
   account. Phase 0 is discovery-first; only build what is missing.
3. **Baseline:** the mmap baseline is **NOT re-run** — the user already has
   those results (their Experiments 1/2, Panel A). Before the first plot, ask
   the user for the baseline QPS values (or CSV) at the 24 GB and 96 GB
   points **and the exact run conditions they were measured under** (dataset
   and index construction — d, nlist, index type — plus nprobe, k, thread
   count, query batch size, warmup). The new IOWarp runs must replicate those
   conditions — including building the same kind of index, which overrides
   the synthetic defaults in §5.1 if they differ — otherwise overlaying the
   series is invalid. Record the baseline conditions in `results/RESULTS.md`.
4. **Tests are the user's responsibility.** Do not write test suites. The only
   checks implemented are the inline sanity checks in §7.

---

## 1. Target layout of `contrib/iowarp/`

Standalone CMake project (`find_package(faiss)` + pip-wheel discovery for
iowarp-core — the wheel exports no CMake package, see §3), mirroring
`demos/rocksdb_ivf/`:

```
contrib/iowarp/
├── README.md                     # contribution-facing: design, build, usage
├── IMPLEMENTATION_PLAN.md        # this file
├── CMakeLists.txt                # level0 lib + tools; add_subdirectory(chimod) if clio runtime found
├── IOWarpInvertedLists.h         # Level 0 backend (namespace faiss_iowarp)
├── IOWarpInvertedLists.cpp
├── ivf_to_iowarp.cpp             # ingestion tool: populated.index + .ivfdata -> CTE blobs
├── bench_ivf_qps.cpp             # QPS harness, --backend {mmap|iowarp|chimod}
├── chimod/                       # Level 1
│   ├── clio_mod.yaml
│   ├── CMakeLists.txt            # add_clio_module_client / add_clio_module_runtime
│   ├── include/clio_runtime/faiss_ivf/
│   │   ├── faiss_ivf_tasks.h
│   │   ├── faiss_ivf_client.h
│   │   └── faiss_ivf_runtime.h
│   └── src/
│       ├── faiss_ivf_client.cc
│       ├── faiss_ivf_runtime.cc
│       └── autogen/faiss_ivf_lib_exec.cc
├── scripts/
│   ├── 00_env_ares.sh            # discovery + builds (clio-core, faiss, contrib/iowarp/)
│   ├── 10_gen_and_build_index.py # synthetic data, shards, merge_ondisk -> .ivfdata
│   ├── 20_ingest_cte.sh          # start clio_run + run ivf_to_iowarp
│   ├── 30_run_bench.sh           # per-backend runs, drop_caches, CSV out
│   └── 40_plot_qps.py            # QPS figure (2 sizes × N systems)
├── config/
│   └── ares_cte.yaml             # CLIO runtime config: RAM tier + NVMe file tier (+ chimod compose)
└── results/                      # CSVs + figures (gitignore contents, keep .gitkeep)
```

---

## 2. Ares practicalities (from the Ares user guide)

- Login: `ssh ares` (ares.cs.iit.edu, SSH key auth; IIT network/VPN required).
- **Never run heavy work on the login node.** Allocate a compute node:
  `salloc -N 1 --exclusive` (or `sbatch`); walltime limit 48 h; FCFS queue.
- Compute node specs: 2× Xeon Silver 4114 (20 cores / 40 threads, 2.2–3.0 GHz,
  skylake-avx512), **48 GiB RAM**, ~256 GB local NVMe at `/mnt/nvme/$USER`,
  512 GB SATA SSD at `/mnt/ssd/$USER`, 1 TB HDD at `/mnt/hdd/$USER`.
- Home (`/home/$USER` = `/mnt/common/$USER`) is shared NFS — code lives there,
  **data and indexes live on the node-local NVMe** (`/mnt/nvme/$USER`).
- Page-cache drop between runs: `module load user-scripts` then
  `sudo drop_caches` (or `sudo /mnt/repo/software/user-scripts/drop_caches`).
- Toolchains: `module avail` (Lmod) or Spack for cmake/gcc/python if the
  system ones are too old. clio-core needs a C++ toolchain with coroutines
  (gcc ≥ 11 works; Ubuntu 22.04 default gcc is 11).
- Clean up NVMe/SSD after experiments (shared courtesy).

---

## 3. Phase 0 — environment discovery and builds

Run on a compute node (`salloc -N 1 --exclusive`, then ssh to it).
`scripts/00_env_ares.sh` implements this phase; make it idempotent
(check-then-build).

**Discovery checklist:**

```bash
which clio_run                       # iowarp-core wheel on PATH?
python3 -c "import iowarp_core; print(iowarp_core.__version__)"   # wheel + version
ls ~/faiss ~/clio-core-headers 2>/dev/null   # repos cloned?
module avail 2>&1 | grep -i -E "cmake|gcc|python|user-scripts"
python3 -c "import faiss"            # python bindings available?
df -h /mnt/nvme/$USER                # NVMe free space (need ~200 GB for the 96 GB run)
ldd --version | head -1              # wheel is manylinux_2_34: needs glibc >= 2.34
```

**clio-core via the pip wheel (required path — the native libraries come
from pip, no source build):**

```bash
pip install iowarp-core              # v2.1.0 at time of writing
PKG=$(python3 -c "import iowarp_core, os; print(os.path.dirname(iowarp_core.__file__))")
export PATH="$PKG/bin:$PATH"                       # clio_run, clio_cte_bench
export LD_LIBRARY_PATH="$PKG/lib:$LD_LIBRARY_PATH" # libclio_*.so
clio_run --help    # verify; ~/.clio/clio.yaml is seeded on first import
```

Wheel contents (verified against iowarp_core-2.1.0, manylinux_2_34):
`bin/` has `clio_run` + both benches; `lib/` has every native library we
link — `libclio_cte_core_client.so` (exports `clio::cte::core::
CLIO_CTE_CLIENT_INIT`, `AsyncPutBlob`, ...), `libclio_run_cxx.so`,
admin/bdev clients, and even the compiled `MOD_NAME` template module;
`ext/` has the `clio_cte_core_ext` / `clio_cee` Python modules; `data/`
has the default config. Built with GCC 14 (Red Hat, manylinux_2_34).
What the wheel does NOT ship: the C++ header tree (only one umbrella
`clio_runtime.h` whose includes are absent) and any CMake package.

**Headers therefore come from a matching-version source checkout — cloned
for its `include/` trees only, never built:**

```bash
git clone --depth 1 --branch v2.1.0 https://github.com/iowarp/clio-core.git ~/clio-core-headers
# The tag MUST match the wheel version above; if the tag name differs,
# find the commit whose version metadata says 2.1.0. ABI compatibility
# between our objects (compiled from these headers) and the wheel's .so
# files depends on this pin.
```

`contrib/iowarp/CMakeLists.txt` discovers both halves itself (no
`find_package(clio-core)`): the wheel's `lib/` dir via the Python import
above, and the checkout's include dirs (`context-runtime/include`,
`context-transfer-engine/core/include`, `context-transport-primitives/
include`, ...). Fallback ONLY if an ABI mismatch surfaces on Ares: full
source build per clio-core's INSTALL.md, which restores the exported
CMake package path.

**If the faiss fork is missing or not built:**

```bash
git clone <the cvperez/faiss fork> ~/faiss    # or rsync the working tree
cd ~/faiss
cmake -B build . -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=ON \
  -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DFAISS_OPT_LEVEL=avx512 \
  -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=$HOME/faiss-install
make -C build -j 20 faiss swigfaiss
(cd build/faiss/python && python3 setup.py build)
cmake --install build
export PYTHONPATH="$(ls -d ~/faiss/build/faiss/python/build/lib*/):$PYTHONPATH"
```

Python bindings are needed by `10_gen_and_build_index.py` (it uses
`faiss.contrib.ondisk.merge_ondisk`). BLAS: Ubuntu's libopenblas is fine
(`module load` or apt package should already be present; check
`ldconfig -p | grep openblas`).

**Build this folder:**

```bash
cd ~/faiss/contrib/iowarp
cmake -B build . -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$HOME/faiss-install" \
  -DIOWARP_HEADERS_DIR="$HOME/clio-core-headers"
# The iowarp-core wheel (libs/bins) is discovered automatically through
# `python3 -c "import iowarp_core"`; override with -DIOWARP_WHEEL_DIR=...
cmake --build build -j 20
```

---

## 4. Phase 1 — Level 0: `IOWarpInvertedLists`

### 4.1 Blob data model

```
CTE tag:  faiss_ivf::<index_name>
  blob "sizes"      idx_t[nlist]  — entries per list (0 for empty lists; no blob for empty lists)
  blob "list/<i>"   uint8_t codes[size_i * code_size] ++ idx_t ids[size_i]
```

Same per-list layout as `OnDiskInvertedLists` (codes then ids, contiguous),
so ingestion from `.ivfdata` is a straight per-list copy. One blob per list =
CTE places/migrates each list independently — the whole point of the design.

### 4.2 Class (verified FAISS interface, `faiss/invlists/InvertedLists.h:77-108`)

```cpp
// IOWarpInvertedLists.h — namespace faiss_iowarp (mirrors faiss_rocksdb)
struct IOWarpInvertedLists : faiss::InvertedLists {
    IOWarpInvertedLists(size_t nlist, size_t code_size,
                        const std::string& tag_name);   // connects, loads "sizes"

    size_t list_size(size_t list_no) const override;    // from cached sizes_
    const uint8_t* get_codes(size_t list_no) const override;
    const faiss::idx_t* get_ids(size_t list_no) const override;
    void release_codes(size_t list_no, const uint8_t* codes) const override;
    void release_ids(size_t list_no, const faiss::idx_t* ids) const override;
    void prefetch_lists(const faiss::idx_t* list_nos, int n) const override;

    // write path (used by ivf_to_iowarp; direct add works but is not the fast path)
    size_t add_entries(size_t list_no, size_t n_entry,
                       const faiss::idx_t* ids, const uint8_t* code) override;
    void update_entries(...) override;   // FAISS_THROW_MSG("not supported")
    void resize(size_t list_no, size_t new_size) override;  // metadata-only update of sizes_
};
```

Internal state: CTE client pointer, `TagId`, `std::vector<size_t> sizes_`, and
a `mutable std::mutex` + `std::unordered_map<size_t, PendingFetch>` where
`PendingFetch` holds the allocated shm buffer and the CTE `Future` (and a
refcount, since `get_codes` and `get_ids` both pin the same buffer).

### 4.3 Fetch pipeline (this is where the win comes from)

- `prefetch_lists(list_nos, n)`: for each list not already pending/cached:
  `auto buf = CLIO_IPC->AllocateBuffer(bytes(list));`
  `ctp::ipc::ShmPtr<> p(buf.shm_);`
  `fut = cte->AsyncGetBlob(tag_id, "list/<i>", 0, bytes, /*flags=*/0, p);`
  Store `{buf, fut}` in the map. Do NOT wait.
- `get_codes(l)`: look up the entry (if absent — cold call — issue the fetch
  now), `fut.Wait()`, return `(const uint8_t*)buf.ptr_`.
- `get_ids(l)`: same buffer, pointer offset `size_l * code_size`.
- `release_codes/release_ids`: decrement refcount; at zero,
  `CLIO_IPC->FreeBuffer(buf)` and erase the map entry.

FAISS calls `prefetch_lists` with **all** `n*nprobe` selected lists right
after coarse quantization and before `search_preassigned` (5 call sites in
`faiss/IndexIVF.cpp`), so every fetch is in flight before the first
`get_codes` blocks — the async pipelining is inherited from stock FAISS with
zero changes to search code.

CTE client API (canonical signatures in
`clio-core/context-transfer-engine/core/include/clio_cte/core/core_client.h`):
`CLIO_CTE_CLIENT_INIT()`, `CLIO_CTE_CLIENT`, `AsyncGetOrCreateTag(name)`,
`AsyncPutBlob(tag, name, offset, size, shm_ptr)`,
`AsyncGetBlob(tag, name, offset, size, flags, shm_ptr)`,
`AsyncGetBlobSize`, futures with `.Wait()` / `operator->`.

Thread-safety note: the bench runs multiple search threads; FAISS may call
`prefetch_lists`/`get_codes` concurrently from OMP threads. Keep the map
mutex-protected and hold it only for map operations, never across `Wait()`.

### 4.4 `ivf_to_iowarp` (ingestion tool)

CLI: `ivf_to_iowarp <populated.index> <ivfdata_file> <tag_name> [--verify N]`

1. `read_index(populated.index, IO_FLAG_SKIP_IVF_DATA)` → nlist, code_size,
   list sizes (with this flag FAISS loads sizes but not data;
   `index_read_warn_on_null_invlists` may warn — fine). The per-list
   `{offset, capacity}` table lives in the OnDisk metadata; simplest robust
   alternative: `read_index(populated.index)` *without* skip on the machine
   doing ingestion is wrong (loads via mmap hook, POSIX-only but Ares is
   Linux — actually fine and simplest: the OnDiskInvertedLists object then
   exposes `get_codes/get_ids` pointers directly). **Recommended:** plain
   `read_index` (mmap attach via the "ilod" hook), then for each list
   `il->get_codes(i)/get_ids(i)` and `AsyncPutBlob` the concatenation.
2. Put blobs in pipelined batches (e.g. 64 futures in flight), then `"sizes"`.
3. `--verify N`: read back N random lists via `AsyncGetBlob`, byte-compare
   against the mmap pointers. Abort non-zero on mismatch.

### 4.5 `bench_ivf_qps` (one harness, all backends)

```
bench_ivf_qps --index populated.index [--ivfdata merged.ivfdata]
              --backend {mmap|iowarp|chimod} [--tag faiss_ivf::exp]
              --nq 10000 --k 10 --nprobe 16 --threads 20
              --warmup 500 --seed 42 --csv out.csv
```

- Queries: random float32 d=128 with fixed seed (self-contained, no query file).
- `mmap`: `read_index(populated.index)` as-is (stock OnDiskInvertedLists).
- `iowarp`: `read_index(populated.index, IO_FLAG_SKIP_IVF_DATA)`, then
  `index_ivf->replace_invlists(new IOWarpInvertedLists(nlist, code_size, tag), true)`
  (`faiss/IndexIVF.h:475`). Set `index->ntotal` from the sizes sum.
- `chimod` (Phase 3): drives the client `AsyncSearch` with in-flight batches.
- Measures wall-clock over the timed query set; reports QPS, p50/p95/p99
  per-query latency. Batch size 1 per search call across `--threads` OMP/std
  threads (per-query latency regime, matching the prior baseline experiments —
  adjust if the user's Panel A used batched queries).
- CSV schema (documented in README):
  `timestamp,backend,index_gb,nlist,nprobe,k,threads,nq,qps,p50_ms,p95_ms,p99_ms,notes`

### 4.6 Sanity check (inline, not a test suite)

`bench_ivf_qps --selftest`: builds a tiny index (100k vectors, nlist=64) in
memory, ingests it to CTE via `add_entries`, runs the same 1000 queries
through `ArrayInvertedLists` and `IOWarpInvertedLists`, asserts identical
`(D, I)`. Run once after every build. Formal tests are the user's job.

---

## 5. Phase 2 — Experiment A (mmap vs Level 0)

### 5.1 Index builds (`scripts/10_gen_and_build_index.py`)

| | 0.5× RAM | 2× RAM |
|---|---|---|
| Target size | 24 GB | 96 GB |
| Vectors (d=128 Flat, 520 B/vec) | 46,000,000 | 184,000,000 |
| nlist | 16384 | 16384 |
| avg list size | ~1.5 MB | ~5.8 MB |
| shards | 4 | 8 |

- Train `IVF16384,Flat` on 2M random vectors (same seed family) → `trained.index`.
- Per shard: generate vectors with `np.random.default_rng(seed+shard)` in
  chunks, `add_with_ids`, write `block_<j>.index`, free RAM. Largest shard
  ~12 GB (96 GB / 8) — fits the 48 GB node during build.
- `faiss.contrib.ondisk.merge_ondisk(trained, blocks, merged_<size>.ivfdata)`
  → write `populated_<size>.index`. Delete block files after merge (NVMe
  budget: 96 GB ivfdata + transient shards; ~200 GB free needed at peak —
  checked in Phase 0).
- Everything under `/mnt/nvme/$USER/faiss_iowarp_exp/`.
- Build time note: data generation + add for 184M vectors is hours, not
  minutes; run under `sbatch` with generous walltime, keep artifacts for reuse.

### 5.2 CTE config (`config/ares_cte.yaml`)

Based on the seeded `~/.clio/clio.yaml`, with:

```yaml
compose:
  - mod_name: clio_bdev            # required DRAM allocator
    pool_name: "ram::chi_default_bdev"
    pool_query: local
    pool_id: "301.0"
    bdev_type: ram
    capacity: "36g"                # leave headroom out of 48g
  - mod_name: clio_cte_core
    pool_name: cte_main
    pool_query: local
    pool_id: "512.0"
    storage:
      - path: "ram::cte_ram_tier"
        bdev_type: ram
        capacity_limit: "30g"      # the explicit "RAM budget" of the tiered system
        score: 1.0
      - path: "/mnt/nvme/$USER/cte_tier"
        bdev_type: file
        capacity_limit: "120g"
        score: 0.3
    dpe: { dpe_type: "max_bw" }
# Phase 3 adds:
#  - mod_name: clio_faiss_ivf
#    pool_name: faiss_ivf_main
#    pool_query: local
#    pool_id: "600.0"
```

Start with `CLIO_SERVER_CONF=config/ares_cte.yaml clio_run start &`
(`scripts/20_ingest_cte.sh` does start → ingest → verify).

**Fairness note (write into RESULTS.md):** the pre-existing mmap baseline
had the node's free RAM as page cache (~44 GB); the CTE RAM tier is capped
at 30 GB plus the runtime's own memory. State both budgets explicitly next
to the plot — the comparison is against previously measured numbers, so the
conditions of those runs (see assumption 3 in §0) must be documented, not
assumed.

### 5.3 Run protocol (`scripts/30_run_bench.sh`)

The mmap baseline is **not run** — its results already exist (§0 assumption
3); match its conditions. For each `size ∈ {24, 96}` with `backend=iowarp`
(and later `backend=chimod`):

1. `sudo drop_caches` (module `user-scripts`).
2. Kill + restart `clio_run` fresh with `config/ares_cte.yaml`, re-ingest
   (fresh restart per size is the clean default).
3. Warmup: 500 queries (not timed) — lets CTE promote hot lists. Match the
   baseline's warmup policy if it differed.
4. Timed run with **the same nq/k/nprobe/threads as the existing baseline**
   (placeholder default: 10,000 queries, k=10, nprobe=16, 20 threads —
   replace with the baseline's actual values) → CSV row.
5. Optional secondary sweep at the 96 GB size: nprobe ∈ {4, 16, 64}.

Also capture per run (into `results/`): `/proc/meminfo` snapshots, and the
CTE tier occupancy if exposed (else note ingest-time placement split). The
`--backend mmap` mode stays implemented in the harness for debugging and for
verification step 4 in §7, but it is not part of the experiment matrix.

### 5.4 Plot (`scripts/40_plot_qps.py`)

Grouped bar chart (or two-point lines): x = index size {24 GB, 96 GB},
y = QPS, series = {mmap (page cache) — **from the user's existing results**,
IOWarp L0} — Level 1 series added in Phase 4. The script takes the baseline
numbers as an input CSV (`--baseline baseline.csv`, same schema as §4.5) so
they are versioned alongside the new measurements, never hardcoded.
Matplotlib, output `results/qps_comparison.png` + the CSVs it was drawn from.
Expected story: parity (or mild L0 overhead) at 24 GB; divergence at 96 GB.

---

## 6. Phase 3 — Level 1: the `faiss_ivf` ChiMod

With the pip wheel there is no exported CMake package, so
`chimod/CMakeLists.txt` writes the two shared-library targets by hand
(client + runtime `.so`), compiled against the fetched v2.1.0 headers and
linked against the wheel's `lib/` — replicating what
`add_clio_module_client` / `add_clio_module_runtime` do in-tree (crib the
exact compile flags and link sets from `cmake/ClioCoreCommon.cmake` in the
header checkout). Loading is plausible by construction: the wheel itself
ships the compiled template module (`libchimaera_MOD_NAME_{client,
runtime}.so`), confirming `clio_run` dlopens module libraries by name — our
runtime `.so` just has to be on `LD_LIBRARY_PATH` when `clio_run` starts.
Verify early in Phase 3 that the wheel runtime accepts an out-of-tree
module; if it hard-fails, the fallback is the source-build path (§3).
Start from the template `clio-core/context-runtime/modules/MOD_NAME/` — copy
its structure including `autogen/MOD_NAME_lib_exec.cc` and adapt (rename
MOD_NAME→faiss_ivf, keep the switch-case dispatch pattern; the template's
exec file is small and mechanical).

### 6.1 `clio_mod.yaml`

```yaml
module_name: faiss_ivf
namespace: clio::run
version: 1.0.0
kCreate: 0
kDestroy: 1
kMonitor: 9
kOpenIndex: 10
kSearch: 11
kStats: 12
```

### 6.2 Tasks (`faiss_ivf_tasks.h`)

- `OpenIndexTask`: IN `index_path` (priv::string — path to `populated.index`
  readable from the node, NVMe or NFS), IN `tag_name`; OUT `ok`, `ntotal`.
- `SearchTask`: IN `nq (u32)`, `k (u32)`, `nprobe (u32)`,
  `queries (priv::vector<float>, nq*d)`; OUT `distances (priv::vector<float>,
  nq*k)`, `labels (priv::vector<int64>, nq*k)`.
  `SerializeIn`: params + queries; `SerializeOut`: distances + labels.
- `StatsTask`: OUT counters — searches served, lists fetched, bytes fetched,
  fetch-wait µs, scan µs (enough to quantify overlap and data movement).
- Lifecycle reuse from admin module as in the template:
  `using CreateTask = clio::run::admin::GetOrCreatePoolTask<CreateParams>;`
  with `CreateParams::chimod_lib_name = "clio_faiss_ivf"`.

### 6.3 Runtime container (`faiss_ivf_runtime.cc`)

State: `std::unique_ptr<faiss::IndexIVF>` (loaded with
`IO_FLAG_SKIP_IVF_DATA` — quantizer + metadata only), `sizes` from the CTE
`"sizes"` blob, CTE client, `TagId`, stats counters.

`kOpenIndex`: load index metadata, `AsyncGetOrCreateTag`, fetch `"sizes"`.
Call `omp_set_num_threads(1)` — CLIO workers are the parallelism; FAISS's
internal OMP must not oversubscribe them.

`kSearch` — implement v0 first, v1 second (keep both behind a task flag or
container option so the ablation "v0 vs v1" from the proposal is runnable):

- **v0 (correctness):** quantizer->search(nq, q, nprobe) → dedupe the probed
  list set across the batch → issue all CTE `AsyncGetBlob`s → `co_await` /
  `Wait` all → build a trivial pointer-table `InvertedLists` view (subclass
  whose `get_codes/get_ids` return the fetched shm pointers, `prefetch_lists`
  a no-op) → `index->search_preassigned(nq, q, k, assign, coarse_dis, D, I,
  false)` (`faiss/IndexIVF.h:132`) → copy D/I into OUT fields, free buffers.
- **v1 (performance):** same quantize + dedupe, then a manual scanner loop —
  `scanner = index->get_InvertedListScanner()` (`faiss/IndexIVF.h:358`);
  maintain per-query heaps; as each list's fetch completes (`co_await` the
  earliest / poll futures with coroutine yields), for every (query, probe)
  touching that list: `scanner->set_query(q_i)`, `scanner->set_list(l,
  coarse_dis)`, `scanner->scan_codes(size, codes, ids, heap_d, heap_i, k)`
  (`faiss/IndexIVF.h:516-542`). I/O of pending lists overlaps scanning of
  arrived ones; chunk very large lists (e.g. 256k codes per slice) so the
  coroutine yields periodically and never hogs a worker.

Handlers use the template's `CLIO_TASK_BODY_BEGIN` / `CLIO_CO_RETURN` /
`CLIO_TASK_BODY_END` coroutine macros; runtime coding rules are in
`clio-core/AGENTS.md` (no blocking calls on workers, coroutine-aware locks).

### 6.4 Client + build + compose

- `faiss_ivf_client.h`: `AsyncOpenIndex(...)`, `AsyncSearch(pool_query,
  queries, nq, k, nprobe)`, `AsyncStats()` — standard
  `ContainerClient` pattern (`NewTask` → `Send` → `Future`).
- `chimod/CMakeLists.txt`: `add_clio_module_client(LIB_NAME clio_faiss_ivf
  SOURCES src/faiss_ivf_client.cc)` + `add_clio_module_runtime(LIB_NAME
  clio_faiss_ivf SOURCES src/faiss_ivf_runtime.cc
  src/autogen/faiss_ivf_lib_exec.cc)`; link `faiss` into the runtime lib.
- Compose entry (pool 600.0) in `config/ares_cte.yaml` (§5.2); the module
  `.so` directory must be on `LD_LIBRARY_PATH` when `clio_run` starts.
- `bench_ivf_qps --backend chimod`: init client, `AsyncOpenIndex` once, then
  timed loop sending `SearchTask`s with a configurable number in flight
  (`--inflight`, default = threads) to keep workers busy.

---

## 7. Phase 4 — Experiment B + verification

Repeat §5.3 exactly with `--backend chimod` (both sizes, both v0 and v1 if
time allows; v1 is the headline). Regenerate the plot: 2 sizes × 3 systems
(mmap from the user's existing results, IOWarp L0, IOWarp chimod). Write `results/RESULTS.md`: the figure,
the CSV, run conditions (node, budgets, drop_caches, warmup), and 2–3
paragraphs of interpretation (including `kStats` bytes-fetched vs
bytes-returned, and fetch-wait vs scan time as evidence of overlap).

**Verification checklist (run in this order, each gates the next):**

1. `contrib/iowarp/` builds clean against installed faiss + clio-core.
2. `ivf_to_iowarp --verify 64` passes (byte-identical lists in CTE).
3. `bench_ivf_qps --selftest` passes (identical `(D,I)` Array vs IOWarp L0).
4. Optional harness cross-check: one short `--backend mmap` run at 24 GB
   should land in the ballpark of the user's existing Panel-A number at that
   size (same-order QPS). This is a methodology sanity check only — it does
   NOT replace the user's baseline results; if wildly off, the harness
   conditions differ from their original methodology — stop and reconcile
   with the user before the timed experiments.
5. Chimod: `(D,I)` of v0 == `(D,I)` of Level 0 on the same CTE tag (small
   index); v1 == v0.
6. Only then run the full-size timed experiments.

---

## 8. Documentation requirements (user: "document everything")

- `contrib/iowarp/README.md` — the contribution-facing doc: motivation (2 paragraphs),
  architecture diagram (levels), blob layout, build instructions (with and
  without the chimod), CLI reference for the three binaries, script-by-script
  experiment reproduction guide, CSV schema, and the Ares-specific notes.
- Every script starts with a header comment: purpose, inputs, outputs, example
  invocation.
- `results/RESULTS.md` per experiment campaign (conditions + figure + reading).
- Keep `IMPLEMENTATION_PLAN.md` updated: check off phases as they complete,
  record deviations (e.g. actual vector counts, chosen nprobe) inline.

## 9. Suggested execution order on Ares (summary)

```
Phase 0  scripts/00_env_ares.sh          (discovery; build only what's missing)
Phase 1  IOWarpInvertedLists + ivf_to_iowarp + bench_ivf_qps (+ selftest)
Phase 2  10_gen_and_build_index.py (24g, 96g) -> 20_ingest -> 30_run_bench (iowarp) -> 40_plot (overlay user's existing mmap baseline)
Phase 3  chimod (v0 -> verify -> v1)
Phase 4  30_run_bench (chimod) -> 40_plot (final 3-system figure) -> RESULTS.md
```
