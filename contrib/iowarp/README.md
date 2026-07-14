# FAISS × IOWarp: IVF inverted lists on the CLIO Context Transfer Engine

This folder contains an external-dependency `InvertedLists` backend (in the
spirit of `demos/rocksdb_ivf/`) that stores FAISS IVF inverted lists as
[IOWarp](https://github.com/iowarp) CTE blobs, a ChiMod that runs IVF search
inside the CLIO runtime, and the scripts to benchmark both against stock
`OnDiskInvertedLists` on a single node.

## Motivation

FAISS's `OnDiskInvertedLists` keeps the inverted lists in a mmapped
`.ivfdata` file and delegates all placement decisions to the OS page cache.
While the index fits in RAM this works beautifully — warm searches run at
memory speed. Once the index is a small multiple of RAM, however, every
query's `nprobe` list fetches compete for cache residency, page-ins dominate
the search path, and QPS collapses by orders of magnitude (our step2/step3
baselines on one 48 GiB Ares node: ~93 QPS warm for a 25 GB index vs ~0.2–2
QPS for 49–173 GB indexes). The page cache is reactive, per-4KiB-page, and
blind to the access unit that actually matters — the inverted list.

IOWarp's Context Transfer Engine (CTE) manages placement explicitly: data
lives in named blobs, a Data Placement Engine tiers them across a capped RAM
tier and node-local NVMe by score, and clients fetch them asynchronously over
shared memory. Storing *one blob per inverted list* makes the placement unit
equal to the access unit, and FAISS's own `prefetch_lists` hook (called with
all `n × nprobe` lists before scanning) gives CTE the full fetch set up
front — every fetch is in flight before the first scan blocks. The study
measures whether that explicit, list-granular management beats the page
cache in the index ≫ RAM regime.

## Architecture

Two integration levels, benchmarked with the same harness:

```
Level 0 — client-side backend               Level 1 — search inside the runtime
=============================               ====================================

  client process                              client process
 ┌───────────────────────────┐               ┌──────────────────────────┐
 │ bench_ivf_qps             │               │ bench_ivf_qps            │
 │  faiss::IndexIVF (search) │               │  faiss_ivf client        │
 │   └ IOWarpInvertedLists   │               │   AsyncSearch(queries)   │
 │      prefetch/get_codes   │               └────────────┬─────────────┘
 └────────────┬──────────────┘                 queries in │ top-k out
              │ AsyncGetBlob (shm)                        │ (task, shm)
 ┌────────────▼──────────────┐               ┌────────────▼─────────────┐
 │ clio_run (CLIO runtime)   │               │ clio_run (CLIO runtime)  │
 │  clio_cte_core (512.0)    │               │  clio_faiss_ivf (600.0)  │
 │   ├ RAM tier   (30 GiB)   │               │   quantize → fetch lists │
 │   └ NVMe tier (120 GiB)   │               │   → scan (coroutines,    │
 │  DPE: max_bw              │               │     I/O–compute overlap) │
 └───────────────────────────┘               │  clio_cte_core (512.0)   │
                                             │   ├ RAM tier / NVMe tier │
   baseline: same IndexIVF over              └──────────────────────────┘
   OnDiskInvertedLists (mmap +
   OS page cache) — no runtime
```

- **Level 0 — `IOWarpInvertedLists`** (`IOWarpInvertedLists.{h,cpp}`,
  namespace `faiss_iowarp`): FAISS search runs unchanged in the client;
  `prefetch_lists` issues async CTE fetches, `get_codes`/`get_ids` wait on
  the corresponding future. Zero changes to FAISS search code.
- **Level 1 — `clio_faiss_ivf` ChiMod** (`chimod/`): the index (metadata
  only) is opened inside the runtime; a `SearchTask` carries queries in and
  top-k out. List fetch and code scanning overlap via runtime coroutines.

## Blob data model

```
CTE tag:  faiss_ivf::<volume>
  blob "sizes"      idx_t[nlist] — entries per list (empty lists: no list blob)
  blob "list/<i>"   uint8_t codes[size_i * code_size] ++ idx_t ids[size_i]
```

Per-list layout (codes then ids, contiguous) matches `OnDiskInvertedLists`,
so ingestion from `.ivfdata` is a straight per-list copy. One blob per list
means CTE places and migrates each list independently.

## Building

Three ingredients (see `IMPLEMENTATION_PLAN.md` §3 for the rationale):

1. **iowarp-core pip wheel** — the native CLIO libraries and `clio_run`
   come from pip, never from source:

   ```shell
   pip install iowarp-core            # v2.1.0 at time of writing
   PKG=$(python3 -c "import iowarp_core, os; print(os.path.dirname(iowarp_core.__file__))")
   export PATH="$PKG/bin:$PATH"
   export LD_LIBRARY_PATH="$PKG/lib:$LD_LIBRARY_PATH"
   ```

2. **clio-core headers, pinned to the wheel version** — the wheel ships no
   header tree, so headers come from a source checkout used for its
   `include/` dirs only (the tag MUST match the wheel version; ABI
   compatibility depends on it):

   ```shell
   git clone --depth 1 --branch v2.1.0 https://github.com/iowarp/clio-core.git \
       ~/clio-core-v2.1.0
   ```

3. **faiss, installed as shared libraries:**

   ```shell
   cmake -B build . -DCMAKE_BUILD_TYPE=Release -DFAISS_OPT_LEVEL=avx512 \
     -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DFAISS_ENABLE_EXTRAS=OFF \
     -DBUILD_TESTING=OFF -DFAISS_ENABLE_MKL=OFF -DBUILD_SHARED_LIBS=ON \
     -DCMAKE_INSTALL_PREFIX=$HOME/faiss-install
   cmake --build build -j && cmake --install build
   ```

Then this folder (standalone CMake project):

```shell
cd contrib/iowarp
cmake -B build . -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$HOME/faiss-install" \
  -DIOWARP_HEADERS_DIR="$HOME/clio-core-v2.1.0"
cmake --build build -j
```

The wheel's `lib/` dir is discovered through `python3 -c "import
iowarp_core"` (override with `-DIOWARP_WHEEL_DIR=...`).

**With / without the chimod:** Level 0 (`ivf_to_iowarp`, `bench_ivf_qps`
with `--backend mmap|iowarp`) needs only the above. Level 1 additionally
builds `chimod/` into `libclio_faiss_ivf_{client,runtime}.so`; the build
skips it if the runtime headers are unusable, and `scripts/20_ingest_cte.sh`
strips the `clio_faiss_ivf` compose entry from the config when the runtime
lib is absent — so everything up to `--backend chimod` works without it.

`scripts/00_env_ares.sh` automates all of the above idempotently.

## CLI reference

### `ivf_to_iowarp` — ingest an on-disk index into CTE

```
ivf_to_iowarp <populated.index> <ivfdata_file> <tag_name> [--verify N]
```

Attaches the `.ivfdata` via mmap, copies every non-empty list into blob
`list/<i>` under `<tag_name>` (pipelined puts), writes the `"sizes"` blob
last. `--verify N` reads back N random lists and byte-compares them against
the mmap pointers; exits non-zero on mismatch. Requires a running `clio_run`.

### `bench_ivf_qps` — one QPS harness, all backends

```
bench_ivf_qps --backend {mmap|iowarp|chimod} --protocol step3
              --index <populated.index> [--ivfdata <merged.ivfdata>]
              --queries <bigann_query.bvecs> --csv <out.csv>
bench_ivf_qps --selftest
```

- `--backend mmap` — stock `OnDiskInvertedLists` (page cache); kept for
  harness cross-checks against the pre-existing baseline numbers.
- `--backend iowarp` — Level 0: loads the index with
  `IO_FLAG_SKIP_IVF_DATA` and `replace_invlists` with `IOWarpInvertedLists`.
- `--backend chimod` — Level 1: drives `clio_faiss_ivf::AsyncSearch`.
- `--protocol step3` — the comparability protocol (below): decodes the
  first 500 bvecs queries to float32, runs 1 cold + 2 warm passes, one
  batched search call per pass, appends one CSV row per pass.
- `--selftest` — builds a tiny index in memory, ingests it, and asserts
  identical `(D, I)` between `ArrayInvertedLists` and `IOWarpInvertedLists`.
  Run once after every build.

### `clio_run` — the CLIO runtime (from the iowarp-core wheel)

```
CLIO_SERVER_CONF=<rendered ares_cte.yaml> clio_run start &
```

Composes the pools in `config/ares_cte.yaml`: `clio_bdev` (DRAM allocator,
36 GiB), `clio_cte_core` (pool 512.0; 30 GiB RAM tier score 1.0 + 120 GiB
NVMe file tier score 0.3, `max_bw` DPE), and `clio_faiss_ivf` (pool 600.0)
when built. The scripts render `${USER}` in the config before starting it.

## The comparability protocol (do not change silently)

The mmap baseline numbers were measured under exactly these conditions; the
IOWarp runs must reproduce them or the overlay is invalid:

- queries: **first 500** of the 10k BigANN queries
  (`bigann_query.bvecs`, d=128, uint8), decoded to float32
- `k = 10`, `nprobe = max(1, nlist // 64)`, **8 OMP threads**
- **one batched search call per pass**; passes = 1 **cold** (immediately
  after `sudo drop_caches`) + 2 **warm**
- QPS = 500 / elapsed, per pass
- side metrics per pass: `majflt` delta, `/proc/self/io` `read_bytes`
  delta, and (iowarp/chimod) CTE bytes fetched

## Pre-built experiment volumes

No index building is needed — populated `IVF<nlist>,Flat` pairs (BigANN
base vectors, d=128) already exist on NFS under
`/mnt/common/$USER/faiss/slurm_bench/work/`:

| volume               | .ivfdata | nlist  | note                                        |
|----------------------|----------|--------|---------------------------------------------|
| `ondisk_nb10M`       | 4.9 GB   | 4096   | smoke / verify volume                        |
| `ondisk_nb50M`       | 25 GB    | 8192   | fits-in-RAM regime; baseline ~93 QPS warm    |
| `ondisk_nb100M`      | 49 GB    | 16384  | ≈2× RAM; baseline ~1–2 QPS                   |
| `ondisk_step3_nb44M` | 43 GB    | 8192   | capacity-doubled post-write; baseline 92.6 QPS |
| `ondisk_step3_nb178M`| 173 GB   | 16384  | capacity-doubled; baseline 0.2 QPS           |

Queries: `/mnt/common/$USER/faiss/slurm_bench/data/bigann/bigann_query.bvecs`.

## Reproducing the experiments (script by script)

All scripts live in `scripts/`, start with a purpose/inputs/outputs header,
and log into `results/`. Everything runs on an **Ares compute node** — never
the login node.

1. **`00_env_ares.sh`** — discovery + idempotent builds: iowarp-core wheel,
   pinned header worktree, faiss install, this folder. Re-run any time.

   ```shell
   bash scripts/00_env_ares.sh
   ./build/bench_ivf_qps --selftest        # gate: must pass after every build
   ```

2. **`20_ingest_cte.sh <volume> [tag]`** — kill any running `clio_run`,
   start a fresh one with the rendered `config/ares_cte.yaml`, ingest the
   volume with `--verify 16`, leave the runtime up.

   ```shell
   bash scripts/20_ingest_cte.sh ondisk_nb10M
   ```

3. **`30_run_bench.sh <volume> [backend...]`** — per backend: prepare
   (fresh runtime + re-ingest for iowarp/chimod; runtime stopped for mmap),
   `sudo drop_caches`, then `bench_ivf_qps --protocol step3` appending to
   `results/qps_<volume>.csv`.

   ```shell
   bash scripts/30_run_bench.sh ondisk_nb50M iowarp
   bash scripts/30_run_bench.sh ondisk_step3_nb178M mmap iowarp chimod
   ```

4. **`sbatch_bench.sh`** — SLURM wrapper around steps 1+3 (exclusive node):

   ```shell
   mkdir -p results/logs
   sbatch --export=ALL,FAISS_VOLUME=ondisk_nb50M,FAISS_BACKENDS="iowarp chimod" \
       scripts/sbatch_bench.sh
   ```

5. **`40_plot_qps.py`** — grouped bars (x = volume, series = system,
   warm-pass QPS). The mmap baseline is **not re-run**: it comes in as a
   CSV so the numbers stay versioned, never hardcoded.

   ```shell
   python3 scripts/40_plot_qps.py --baseline results/baseline_panelA.csv
   # -> results/qps_comparison.{png,pdf}
   ```

## CSV schemas

**Bench output** (`results/qps_<volume>.csv`, one row per pass, written by
`bench_ivf_qps --protocol step3`):

```
timestamp,volume,backend,pass,nlist,nprobe,k,threads,nq,elapsed_s,qps,majflt,read_bytes,cte_bytes_fetched,notes
```

`backend ∈ {mmap, iowarp, chimod}`, `pass ∈ {cold, warm1, warm2}`;
`cte_bytes_fetched` is empty for mmap. The plot script requires only
`backend,pass,qps` and derives `volume` from the filename if absent.

**Baseline input** (`--baseline`, the user's pre-existing step2/step3
numbers, same pass convention):

```
volume,system,pass,qps
ondisk_nb50M,mmap-baseline,warm1,93.1
```

## Ares notes

- **Never run on the login node.** `salloc -N 1 --exclusive` or `sbatch`
  (walltime limit 48 h). `--exclusive` matters: the baselines were measured
  with the whole ~46 GiB page cache owned by the job.
- Compute node: 2× Xeon Silver 4114 (skylake-avx512), 48 GiB RAM, ~256 GB
  NVMe at `/mnt/nvme/$USER` (CTE file tier lives at
  `/mnt/nvme/$USER/cte_tier`), NFS home at `/mnt/common/$USER`.
- Page-cache drop: `module load user-scripts && sudo drop_caches` (fallback:
  `sudo /mnt/repo/software/user-scripts/drop_caches`). `30_run_bench.sh`
  does this before every cold pass.
- Indexes and queries are read from NFS; the CTE NVMe tier is node-local.
  Clean up `/mnt/nvme/$USER` after experiments (`sbatch_bench.sh` removes
  the tier directory on exit).

## Fairness note (state this next to every plot)

The mmap baseline and the IOWarp runs do **not** get the same fast-memory
budget: the baseline had the node's free RAM as page cache (~44 GiB,
implicit), while CTE runs under an explicit 30 GiB RAM-tier cap plus the
runtime's own allocator (36 GiB `clio_bdev` arena) and spills to a 120 GiB
NVMe file tier (score 0.3). Any figure comparing the systems must state
both budgets alongside it — `40_plot_qps.py` prints them in the figure
footnote, and `results/RESULTS.md` for each campaign must record the exact
run conditions (node, budgets, drop_caches, passes) next to the numbers.
