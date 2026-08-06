# FAISS × IOWarp — results on clio-core `dev` (zero-IPC read)

The `clio_faiss_ivf` ChiMod on **clio-core dev `8715591a`** (2026-08-05
campaign, the current results). Earlier dev campaigns (`609c526a`,
`4cc0d780`) and the five ChiMod-side adaptations to dev's changed runtime
semantics are documented in `ISSUES_RESOLVED_DEV_ANALYSIS.md` and in the
git history of this file.

## Protocol (identical to the v2.1.0 baseline campaign)

First 500 BigANN queries, `k = 10`, `nprobe = nlist/64`, **8 scan threads
inside one SearchTask** (`inflight = 1`, matching the baseline's 8 OMP
threads), `FAISS_OPT_LEVEL = avx512`, one batched search per pass, 1 cold
pass (`posix_fadvise(DONTNEED)`) + 2 warm passes. CTE config: RAM tier
30 GB (score 1.0) + NVMe file tier 120 GB (score 0.3), `max_bw` DPE,
`runtime.num_threads = 8`. One exclusive Ares compute node (48 GB RAM).

**Every run in this campaign additionally passed a FULL ingest
verification** (`ivf_to_iowarp --verify all`): every non-empty list read
back through the RPC `GetBlob` API and byte-compared against the source —
0 failures at all sizes. The large-spill corruption seen on earlier dev
HEADs is fixed upstream: nb178M now runs the standard mixed-tier config
(RAM 30 GB + 57 GB NVMe spill) clean, so no upstream issue was filed.

## 1-node results (dev `8715591a`)

QPS per pass (cold / warm0 / warm1); verify-all clean on every row.

| Volume | Config | QPS cold | QPS warm | Placement | Job |
|---|---|---|---|---|---|
| **nb10M** (4.8 GB, nprobe 64) | v2.1.0 avx512 baseline | 130.0 | 124.7 / 134.9 | RAM | 21917 |
| | **dev 8715591a** | **237.2** | **268.9 / 276.0** | RAM | 22753 |
| **nb50M** (24 GB, nprobe 128) | v2.1.0 avx512 baseline | 20.8 | 21.9 / 26.0 | RAM+NVMe | 21919 |
| | **dev 8715591a** | **51.2** | **53.6 / 54.0** | RAM (0 on NVMe) | 22757 |
| **nb100M** (49 GB, nprobe 256) | v2.1.0 scalar¹ baseline | 8.9 | 7.8 / 7.9 | RAM+NVMe | 2026-07-13 |
| | **dev 8715591a** | **18.1** | **21.1 / 21.4** | 30 GB RAM + 19 GB NVMe | 22784 |
| **nb178M** (86 GB, nprobe 256) | v2.1.0 scalar¹ baseline | 4.0 | 3.9 / 4.0 | RAM+NVMe | 2026-07-13 |
| | **dev 8715591a** | **8.8** | **9.4 / 9.4** | 30 GB RAM + 57 GB NVMe | 22786 |

**Correctness.** `(D, I)` bitwise-identical to stock FAISS where a
reference exists (FNV-1a di_hash): nb10M `0e04f8171f7d1898`, nb50M
`32085d09bb65391b`. nb100M `4d0bb1d73138720a` and nb178M
`b1bda05cb0b86813` are identical on every pass and identical across
different ChiMod binaries, configs, and dev HEADs. Zero page-ins on warm
passes throughout; verify-all = 0 failures everywhere.

**Bottom line.** The ChiMod is ~2× the v2.1.0 baseline at every size
(130→237, 21→51, 8.9→18, 4.0→8.8 cold QPS). At RAM-resident sizes scan
time is the bottleneck (the profile the zero-copy view/pin proposal
targets); at nb100M/nb178M the NVMe-resident share is read at ~2 GB/s via
io_uring, and the mixed-tier RAM share lifts nb178M warm 42 % over the old
file-tier-only workaround (9.4 vs 6.6).

## CTE (dev) vs the mmap baseline

Same format as `docs/report.tex` Tables 1–2, CTE column = this campaign
(mmap column reproduced from report.tex Table 2 — mmap does not involve
clio-core, so those baselines are unchanged):

| Index size | nprobe | CTE dev QPS (cold/warm) | mmap QPS (cold/warm) | mmap page-ins/pass |
|---|---|---|---|---|
| 4.8 GB | 64 | **237 / 269** | 28 / 238 | 46k / <700 |
| 24 GB | 128 | **51 / 54** | 11 / 93 | 213k / <200 |
| 49 GB | 256 | **18.1 / 21.1** | 2.0 / 5.5 | 757k / ~535k |
| 87 GB | 256 | **8.8 / 9.4** | 0.4 / 0.4 | 2.8M / ~2.7M |

(500 queries, K = 10, nprobe = nlist/64, 8 threads, 48 GB node. CTE
page-ins ≈ 0 on every pass, cold included.)

The v2.1.0 story was a crossover: mmap won warm while the index fit in
RAM, CTE won once it didn't. The crossover has narrowed sharply: **cold,
CTE wins at every size** (237 vs 28, 51 vs 11, 18 vs 2.0, 8.8 vs 0.4);
**warm in-RAM**, CTE now edges mmap's low end at 4.8 GB (269 vs 238–360)
and trails only at 24 GB (54 vs 93, was 27 vs 93 on v2.1.0);
**out-of-core** (49 GB+), CTE is 4–24× ahead warm (21.1 vs 5.5; 9.4 vs
0.4) with zero page-ins versus ~535 k–2.7 M per pass.

## 2-node evaluation

Implementation walkthrough with code: `docs/MULTINODE.md`. Summary:
`networking.hostfile` + `swim: enabled: false` rendered into the config,
one `clio_run` per node via srun with per-host TCP readiness polls,
per-node tier prep/teardown, `AsyncOpenIndex` broadcast to all containers,
`AsyncSearch` sub-batches fanned by `DirectHash(i)` (one SearchTask with 8
scan threads per node at `--inflight 2`), Stats broadcast with summed
aggregation. Submit: `sbatch --nodes=2 --ntasks=2 --ntasks-per-node=1 …`.
Single-node behavior unchanged (verified, job 22789).

| Volume | 1-node QPS (warm) | 2-node QPS (warm) | Verdict |
|---|---|---|---|
| nb10M (job 22791) | 269–276 | 22.5 | correct, network-bound |
| nb50M (job 22793) | 54 | 4.6 | correct, network-bound |
| nb100M (jobs 22795, 22818) | 21 | — | ingest+verify-all pass; search OOMs node 1 |

* **Correctness is perfect**: cluster forms, blobs hash-spread ~50/50,
  verify-all passes over cross-node reads, and both nb10M and nb50M
  produce the exact single-node di_hash — bitwise-identical distributed
  search.
* **Performance is network-bound by design of the fan-out.** The zero-IPC
  fast path is node-local, and each query sub-batch probes nearly the
  whole index, so every node reads ~half its lists from the peer over the
  RPC fabric (~215 MB/s effective) — read_wait 124 s (nb10M) / 615 s
  (nb50M) cluster-total vs 3.8 / 7.2 s single-node. Splitting by QUERY
  cannot exploit combined RAM; a data-local design (scan each list on its
  owner node, merge partial top-k heaps) is the actual multi-node
  architecture — future ChiMod work.
* **nb100M is additionally blocked by an upstream limit**: daemon-side shm
  segments grow with bytes moved cross-node and never recycle (the receive
  staging / serving path — 131 segments ≈ 18 GB after ingest, 188 ≈ 26 GB
  during the first search pass) until the OOM killer takes the peer daemon
  (`srun: task 1: Killed`, jobs 22795 and 22818, the latter with tier caps
  lowered to 22 GB). At nb50M's ~12 GB/node the growth fits; at nb100M's
  ~24 GB/node it cannot. Worth raising with the maintainers alongside the
  #856 strict-event-resume note (see `ISSUES_RESOLVED_DEV_ANALYSIS.md`).

## Provenance and caveats

¹ No avx512 v2.1.0 baseline exists for nb100M/nb178M: those rows are the
2026-07-13 scalar-kernel campaign quoted in `docs/report.tex`; the avx512
baselines used for nb10M/nb50M are jobs 21917–21919 (2026-07-14, see
`RESULTS.md` §3).

* Every dev number above: clio-core dev `8715591a`, contrib `build-dev`
  (`IOWARP_USE_DEV=ON`, avx512 faiss), `CHIMOD_INFLIGHTS=1` (1-node) / `=2`
  (2-node), `VERIFY_N=all`. Raw artifacts per job id:
  `logs/<job>_qps_dev.out`, `qps_dev_<volume>.csv` (rows from earlier
  campaigns remain in the CSVs — select by job/timestamp),
  `telemetry_dev_<volume>_<job>.csv`, `ares_cte_rendered_*.yaml`,
  per-node daemon logs `clio_run_*_node<i>.log`.
* The fifth dev-semantics adaptation made during this campaign: #856
  (`d7b1f053`) strict event-resume starves yield-polling in handler
  coroutines — the Search loop now awaits the oldest pending RPC future
  directly (8-hour hang otherwise, job 22758). Details in
  `ISSUES_RESOLVED_DEV_ANALYSIS.md`.
* History (earlier campaigns, the four prior adaptations, and the
  since-fixed large-spill corruption analysis): git history of this file
  and `ISSUES_RESOLVED_DEV_ANALYSIS.md`.
