# FAISS × IOWarp — results on clio-core `dev` (zero-IPC read)

The `clio_faiss_ivf` ChiMod, ported to clio-core `dev` and measured against the
v2.1.0-wheel baseline. Two campaigns are reported:

* **dev, before fixes** — dev HEAD `609c526a` (2026-07-29/30 jobs). The port
  compiled and returned correct results but regressed at nb10M and the daemon
  died at scale.
* **dev, after fixes** — dev HEAD `4cc0d780` (latest `origin/dev`, 2026-07-30
  jobs 22484/22485/22487), with three ChiMod-side adaptations to dev's changed
  runtime semantics (described below). The ChiMod algorithm is unchanged.

## Protocol (identical to the v2.1.0 baseline campaign)

First 500 BigANN queries, `k = 10`, `nprobe = nlist/64`, **8 scan threads inside
one SearchTask** (`inflight = 1`, matching the baseline's 8 OMP threads),
`FAISS_OPT_LEVEL = avx512`, one batched search per pass, 1 cold pass
(`posix_fadvise(DONTNEED)`) + 2 warm passes. CTE config: RAM tier 30 GB
(score 1.0) + NVMe file tier 120 GB (score 0.3), `max_bw` DPE,
`runtime.num_threads = 8`. Single exclusive Ares compute node (48 GB RAM).

## Results

QPS per pass (cold / warm0 / warm1). read_s and scan_s are totals over all
three passes.

| Volume | Config | QPS cold | QPS warm | read_s | scan_s | Placement |
|---|---|---|---|---|---|---|
| **nb10M** (4.8 GB, nprobe 64) | v2.1.0 avx512 (job 21917) | 130.0 | 124.7 / 134.9 | 5.74 | 4.10 | RAM |
| | dev before (job 22445) | 104.2 | 119.8 / 117.0 | 8.46 | 4.59 | RAM |
| | **dev after (job 22487)** | **251.2** | **256.6 / 257.4** | **1.37** | 4.13 | RAM |
| **nb50M** (24 GB, nprobe 128) | v2.1.0 avx512 (job 21919) | 20.8 | 21.9 / 26.0 | 39.60 | 20.94 | RAM+NVMe |
| | dev before (job 22447) | 24.4 | 28.9 / 29.5 | 32.72 | 21.56 | ~21 GB on NVMe¹ |
| | **dev after (job 22484)** | **52.1** | **54.5 / 55.0** | **6.67** | 20.38 | RAM (0 on NVMe) |
| **nb100M** (49 GB, nprobe 256) | v2.1.0 scalar² (2026-07-13) | 8.9 | 7.8 / 7.9 | 122.67 | 47.76 | RAM+NVMe |
| | dev before (job 22476) | 11.4 | daemon died³ | — | — | RAM+NVMe |
| | **dev after (job 22485)** | **17.7** | **20.4 / 21.7** | **33.41** | 41.74 | 30 GB RAM + 19 GB NVMe |
| **nb178M** (86 GB, nprobe 256) | v2.1.0 scalar² (2026-07-13) | 4.0 | 3.9 / 4.0 | 258.47 | 103.69 | RAM+NVMe |
| | dev before | not attempted (nb100M-class crash) | | | | |
| | **dev after (job 22497)** | **8.7** | **6.6 / 6.6** | **127.34** | 80.30 | 87 GB NVMe, file-tier only⁴ |

**Correctness.** Every dev-after pass returns `(D, I)` bitwise-identical to
stock FAISS (FNV-1a di_hash): nb10M `0e04f8171f7d1898`, nb50M
`32085d09bb65391b` — the same hashes as the v2.1.0 and mmap baselines. nb100M
prints `4d0bb1d73138720a` on all three passes (no stock-FAISS reference hash
exists for nb100M; the hash is also identical to the pre-fix cold pass, i.e.
consistent across two different ChiMod binaries); nb178M prints
`b1bda05cb0b86813` on all three passes (likewise self-consistent, no
reference). Zero page-ins on warm passes throughout.

**Bottom line.** On latest dev with the port fixed, the ChiMod is ~2× the
v2.1.0 baseline at every size (130→251, 21→52, 8.9→17.7, 4.0→8.7 cold QPS),
the previously-fatal nb100M/nb178M passes complete, and read time stops
dominating wherever data is RAM-resident: read_s dropped 5.74→1.37 (nb10M),
39.60→6.67 (nb50M), 122.67→33.41 (nb100M). At the RAM-resident sizes scan_s
is now the bottleneck — exactly the profile the zero-copy view/pin API
proposal targets next. nb178M is NVMe-bandwidth-bound (read 127 s vs scan
80 s over 3 passes, ~2.0 GB/s sustained io_uring reads), the regime where
CTE's explicit placement beats mmap by 16× (6.6 vs 0.4 warm).

### CTE (dev, fixed) vs the mmap baseline

Same format as `docs/report.tex` Tables 1–2, with the CTE column replaced by
this campaign's numbers (mmap column reproduced from report.tex Table 2 —
mmap does not involve clio-core, so those baselines are unchanged):

| Index size | nprobe | CTE dev QPS (cold/warm) | mmap QPS (cold/warm) | mmap page-ins/pass |
|---|---|---|---|---|
| 4.8 GB | 64 | **251 / 257** | 28 / 238 | 46k / <700 |
| 24 GB | 128 | **52 / 55** | 11 / 93 | 213k / <200 |
| 49 GB | 256 | **17.7 / 20.4** | 2.0 / 5.5 | 757k / ~535k |
| 87 GB | 256 | **8.7 / 6.6**⁴ | 0.4 / 0.4 | 2.8M / ~2.7M |

(500 queries, K = 10, nprobe = nlist/64, 8 threads, 48 GB node. CTE page-ins
≈ 0 on every pass, cold included.)

The v2.1.0 campaign's story was a crossover: mmap won warm while the index
fit in RAM, CTE won once it didn't. That crossover still exists but has
narrowed sharply: **cold, CTE now wins at every size** (251 vs 28, 52 vs 11,
17.7 vs 2.0, 8.7 vs 0.4); **warm in-RAM**, mmap's page cache is comparable
at 4.8 GB (238–360 vs 257) and still ahead at 24 GB, though the gap shrank
from 3.4× (93 vs ~27) to 1.7× (93 vs 55); **out-of-core** (49 GB+), CTE is
3–16× ahead warm (20.4 vs 5.5; 6.6 vs 0.4) with zero page-ins versus
~535 k–2.7 M per pass.

⁴ nb178M ran **file-tier only** (`CTE_RAM_TIER_GB=0`, all 86 GB on NVMe) —
see the footnote in Provenance for why the mixed-tier config is not usable
at this size on dev.

## What was wrong, and what changed

The ChiMod port to dev was line-for-line equivalent to the v2.1.0 version —
the regressions came from clio-core `dev` changing runtime semantics
underneath it. Three adaptations were needed
(`chimod/src/faiss_ivf_runtime.cc`, `src/ivf_cte_ingest.cpp`):

1. **Reads lost their concurrency.** Dev's `AsyncGetBlob` has a default-on
   zero-IPC fast path that performs the whole copy *synchronously on the
   calling thread* and returns an already-complete future. The ChiMod's
   64-deep async pipeline therefore degenerated into serial copies on one
   cooperative worker (nb10M read_s 4.93→8.46 despite 100 % RAM residency).
   *Fix:* RAM-resident lists are now copied with direct `TryReadBlobShm`
   calls, 8-wide across the scan threads in bursts of 16; only misses
   (NVMe-tier lists) go through the async RPC pipeline, which still overlaps
   with scanning as before. Kill-switch: `FAISS_IVF_NO_SHM_DIRECT=1`.

2. **Futures pinned one shm task per probed list.** On dev a `Future` owns its
   task via `shared_ptr`; holding all `ntoscan` futures for the whole search
   pinned ~16 k tasks at nb100M until the client shm allocators exhausted the
   segment and the daemon stalled and died (`chimod subtask rc=-1` in jobs
   22446/22472/22476). *Fix:* each future is dropped as soon as its list is
   scanned; at most 64 tasks are ever live.

3. **Per-list buffers were never recycled.** Dev's segment allocator does not
   reuse freed multi-MB buffers — client segments grow by roughly the bytes
   pushed and stay mapped by the daemon. Ingest grew 67 × ~140 MB segments at
   nb50M (daemon RSS 35 GB before the bench even started, job 22481) and one
   search pass grew the daemon by ~14 GB more (job 22482); both ended in the
   OOM killer taking the daemon. *Fix:* ingest reuses a ring of ≤ 8
   preallocated max-list-size buffers; the search loop uses a fixed pool of
   64 reusable slabs. Steady-state memory is flat: at nb50M the whole 24 GB
   volume is now RAM-resident with ~12 GB still free through all passes.

Also fixed for diagnosability: `telemetry_sampler.py` now samples the dev
daemon's RSS (pgrep pattern) and true NVMe tier occupancy (sparse-aware
`st_blocks` on the `cte_tier_node*` bdev files) — this is what made the OOM
staircase visible.

## Provenance and caveats

¹ The dev-before nb50M run predates clearing stale `~/.clio/bdev_perf` stats:
the `max_bw` DPE mis-scored RAM vs NVMe and ~21 GB of the volume landed on
NVMe, so its numbers are mostly the RPC/NVMe path, not zero-IPC. The dev-after
nb50M run is fully RAM-resident — placement, not just code, differs between
those rows.

² No avx512 v2.1.0 baseline exists for nb100M: the 8.9/7.8/7.9 row is the
2026-07-13 scalar-kernel campaign quoted in `docs/report.tex` (the nb10M/nb50M
report.tex rows — 148 and 21 QPS — are likewise scalar; the avx512 re-runs
used here are jobs 21917–21919 from 2026-07-14, see `RESULTS.md` §3).

³ Job 22476 log: cold pass completed, then client shm allocators climbed past
~120 × ~140 MB, a worker stalled with NaN load, the daemon vanished, and
`bench_ivf_qps` aborted after the 1200 s reconnect timeout.

⁴ **nb178M could not run the standard mixed-tier config on dev — a
placement-record bug at large spill volumes.** Two rejected attempts:
job 22495 (RAM 30 GB + 56 GB spill, fast path on) "completed" a cold pass at
41.8 QPS in 12 s with **zero NVMe reads** — the shm metadata cache still
claimed direct-readability for spilled blobs, so `TryReadBlobShm` copied
stale RAM-tier bytes (garbage, hash `b781267ff858a9b8` invalid) — then the
daemon stalled and died; job 22496 (same config, `FAISS_IVF_NO_SHM_DIRECT=1`)
did honest io_uring reads but the daemon's own GetBlob returned **rc=1 for
42 of ~10.4 k spilled blobs** — same stale records, correctly failing
instead of silently lying. At nb100M's 19 GB spill all records are correct
(sustained 1–1.6 GB/s NVMe reads, consistent hashes), so the corruption
onsets somewhere between 19 GB and 56 GB of ingest-time spill. Workaround
(v2.1.0 §9.1 precedent): `CTE_RAM_TIER_GB=0` — all 86 GB placed directly on
the file tier at Put time, no reorganization, zero failures (job 22497).
Upstream-worthy dev bug.

* `qps_dev_*.csv` files contain both campaigns' rows — select by timestamp/job
  (before: ts ≤ 1785351239; after: jobs 22484/22485/22487; an intermediate
  nb10M row set from job 22480, 199/216/215 QPS, ran fixes 1+2 without the
  slab pool of fix 3).
* Rejected result sets are quarantined out of this table:
  `nvme_misplaced_609/` (NVMe-misplaced runs), `stale_3c22ee6d/` (older dev
  HEAD), and jobs 22469/22470/22472 (LD_LIBRARY_PATH failures / ingest death
  at dev `609c526a`).
* clio-core dev provenance: before = `609c526a`, after = `4cc0d780`
  (`Merge pull request #875`, fetched 2026-07-30). Between the two, dev added
  nothing touching `core_client.h`/`future.h`; the improvements come from the
  ChiMod-side fixes, not the 20 new dev commits.
* Issue-by-issue analysis of what dev resolves vs the upstream proposal:
  `ISSUES_RESOLVED_DEV_ANALYSIS.md`. The copy tax itself (one memcpy per
  list) remains — dev is zero-IPC, not zero-copy; the view/pin API proposed
  in `docs/UPSTREAM_PROPOSAL_IOWARP.md` is still the path to removing the
  remaining read_s entirely.
