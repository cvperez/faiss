# FAISS × IOWarp — results on clio-core `dev` (zero-IPC read)

This document gathers the measurements obtained when the `clio_faiss_ivf`
ChiMod is run against a **clio-core `dev`** source install (HEAD `609c526a`)
instead of the pinned v2.1.0 wheel. The ChiMod itself is unchanged; the only
difference is the clio-core version underneath, which adds the zero-IPC
shared-memory read to `AsyncGetBlob` (issue #783).

## Setup

Same protocol as the v2.1.0 baselines: the first 500 BigANN queries, `k = 10`,
`nprobe = nlist/64`, 8 threads, `FAISS_OPT_LEVEL = avx512`, one batched search,
`inflight = 1`. The reported QPS is the steady-state value (the passes agreed to
within measurement noise, so a single figure is given per configuration).

For each run three quantities are recorded, as in `RESULTS.md`:

- **read_s** — time spent reading inverted lists from CTE that is not overlapped
  by scanning (`fetch_wait_s` in the v2.1.0 logs, `read_wait_s` on dev).
- **scan_s** — time spent computing distances.
- **QPS** — queries per second.

## Correctness

At every size the ChiMod returns `(D, I)` **bitwise-identical** to stock FAISS
(FNV-1a `di_hash`): nb10M `0e04f8171f7d1898`, nb50M `32085d09bb65391b` — the same
hashes as the v2.1.0 and mmap baselines.

## Throughput

| Index size | Backend | read_s | scan_s | QPS | GB read |
|---|---|---|---|---|---|
| nb10M — 4.8 GB   | v2.1.0 | 4.93   | 3.96   | 148 | 14.49 |
| nb10M — 4.8 GB   | dev    | 8.46   | 4.59   | 118 | 14.49 |
| nb50M — 24 GB    | v2.1.0 | 33.32  | 21.36  | 29  | 72.38 |
| nb50M — 24 GB    | dev    | 32.72  | 21.56  | 29  | 72.38 |
| nb100M — 49 GB   | v2.1.0 | 122.67 | 47.76  | 7.9 | 144.83 |
| nb100M — 49 GB   | dev    | —      | —      | —   | —     |
| nb178M — 87 GB   | v2.1.0 | 258.47 | 103.69 | 3.9 | 257.81 |
| nb178M — 87 GB   | dev    | —      | —      | —   | —     |

`GB read` and the number of lists fetched are identical between v2.1.0 and dev at
each size (nb10M: 12 252 lists; nb50M: 24 462 lists), so the two backends move
exactly the same volume of data.

Notes on the dev rows:

- **nb10M** was run with the inverted-list blobs confirmed 100 % RAM-resident
  (NVMe tier occupancy 0), so the zero-IPC shared-memory read served every list.
- **nb50M** landed on the NVMe file tier (21 GB spilled), so the zero-IPC read
  did not apply; the numbers are the NVMe path and match v2.1.0.
- **nb100M and nb178M**: no dev QPS was obtained — see below.

## Placement of the inverted lists

The zero-IPC read only serves a blob that is RAM-resident. Which tier a blob
lands on is decided by the `max_bw` DPE from the measured per-tier bandwidth. A
stale `~/.clio/bdev_perf` file reported the RAM tier at 476 MB/s against 1637 MB/s
for NVMe, so the lists were placed on NVMe and the zero-IPC read never fired.
Clearing the stale perf stats (now done automatically in `20_ingest_cte.sh`)
lets the RAM tier win placement; this is what made the nb10M dev run RAM-resident.
Placement was not fully repeatable (nb50M still spilled to NVMe on re-measure).

## Mixed-tier ingest (nb100M)

nb100M was ingested with a 30 GB RAM tier plus NVMe spill — the configuration
that corrupted the heap on v2.1.0 (`RESULTS.md` §9.1, which forced
`CTE_RAM_TIER_GB = 0`). On dev the ingest completed cleanly:
`[ingest] done: 48.43 GiB compact`, `[verify] PASS — 16 lists byte-identical`,
with no corruption. Big volumes can therefore use a real RAM tier plus NVMe spill
on dev.

## Search at large sizes (nb100M / nb178M)

The nb100M **search** did not complete: after ingest, `bench_ivf_qps` aborted
(`chimod_search_parallel` returned failure) while the runtime logged a growing
`HANGWATCH` scheduler backlog (47–61 outstanding tasks over 60–74 k processed,
recurring stall/lane-rescue warnings). Memory was not exhausted (37.9 GiB free).
The same failure occurred once on the RAM-resident nb50M run. nb178M was not run.

## Provenance

- clio-core `dev` `609c526a`; deps in `~/iowarp-dev-deps`; contrib built with
  `-DIOWARP_USE_DEV=ON`.
- Runner `sbatch_bench_dev.sh`; dev CSVs `qps_dev_<volume>.csv`; per-run logs
  `clio_run_*`, `bench_*`, `ingest_*`, `telemetry_dev_*`, `logs/<job>_qps_dev.out`.
- Jobs: nb10M 22445; nb50M 22447 (NVMe) / 22446 (RAM, crashed); nb100M 22448.
- v2.1.0 baseline figures from the existing `bench_ondisk_*_chimod_*` logs
  (`fetch_wait_s` / `scan_s`) and `RESULTS.md`.
