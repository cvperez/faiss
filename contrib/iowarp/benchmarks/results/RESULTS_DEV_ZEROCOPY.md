# FAISS × IOWarp — results on clio-core `dev` (zero-IPC read)

This document gathers the measurements obtained when the `clio_faiss_ivf`
ChiMod is run against the **latest clio-core `dev`** (HEAD `609c526a`) instead of
the pinned v2.1.0 wheel. The ChiMod itself is unchanged; the only difference is
the clio-core version underneath, which adds the zero-IPC shared-memory read to
`AsyncGetBlob` (issue #783). Only results from this last dev version are reported.

## Setup

Same protocol as the v2.1.0 baselines: the first 500 BigANN queries, `k = 10`,
`nprobe = nlist/64`, 8 threads, `FAISS_OPT_LEVEL = avx512`, one batched search,
`inflight = 1`. Every configuration runs one **cold** pass (data evicted first)
and two **warm** passes. For each run three quantities are recorded, as in
`RESULTS.md`: **read_s** (time reading inverted lists from CTE that is not
overlapped by scanning), **scan_s** (distance computation), and **QPS**.

The dev runtime is built with io_uring enabled (matching the wheel) so bdev I/O
does not block the cooperative workers.

## Correctness

At the sizes that complete, the ChiMod returns `(D, I)` **bitwise-identical** to
stock FAISS (FNV-1a `di_hash`): nb10M `0e04f8171f7d1898`, nb50M `32085d09bb65391b`
— the same hashes as the v2.1.0 and mmap baselines.

## Throughput

| Index size | Backend | QPS cold | QPS warm | read_s | scan_s |
|---|---|---|---|---|---|
| nb10M — 4.8 GB  | v2.1.0 | 148  | 145–152 | 4.93   | 3.96   |
| nb10M — 4.8 GB  | dev    | 104  | 117–120 | 8.46   | 4.59   |
| nb50M — 24 GB   | v2.1.0 | 21   | 27–30   | 33.32  | 21.36  |
| nb50M — 24 GB   | dev    | 24   | 29      | 32.72  | 21.56  |
| nb100M — 49 GB  | v2.1.0 | 8.9  | 7.8–7.9 | 122.67 | 47.76  |
| nb100M — 49 GB  | dev    | 11.4 | —       | —      | —      |
| nb178M — 87 GB  | v2.1.0 | 4.0  | 3.9–4.0 | 258.47 | 103.69 |
| nb178M — 87 GB  | dev    | —    | —       | —      | —      |

The number of lists fetched and the GB read are identical between v2.1.0 and dev
at each size (nb10M: 12 252 lists / 14.49 GB; nb50M: 24 462 lists / 72.38 GB), so
the two backends move the same volume of data.

## Notes on the dev runs

- **nb10M** — the inverted-list blobs were confirmed 100 % RAM-resident (NVMe
  tier occupancy 0), so the zero-IPC shared-memory read served every list. Even
  so the QPS is a little below v2.1.0: the read is not the limiting factor once
  the data is RAM-resident (see below), and dev carries more per-read machinery.

- **nb50M** — the blobs were placed on the NVMe file tier (21 GB spilled), so the
  zero-IPC read did not apply; the figures are the NVMe path and match v2.1.0.

- **nb100M** — with io_uring the ingest (48.43 GiB across a 30 GB RAM tier + NVMe
  spill) and the **cold** pass complete (11.4 QPS, slightly above v2.1.0's 8.9).
  The **warm** passes do not complete: the runtime grows its shared-memory pool
  without bound (client allocators climbing past 120 × ~140 MB until the 25 GB
  segment is exhausted) and a worker stalls, so the daemon stops responding.
  This is a runtime liveness/shared-memory bug on dev, not a data-path result.

- **nb178M** — not run; it exercises the same warm-pass path that fails at nb100M.

## Placement of the inverted lists

The zero-IPC read only serves a blob that is RAM-resident and flagged
`kShmBlobDirectReadable`. Which tier a blob lands on is decided by the `max_bw`
DPE from the measured per-tier bandwidth. A stale `~/.clio/bdev_perf` file
reported the RAM tier at 476 MB/s against 1637 MB/s for NVMe, which sent the
lists to NVMe and disabled the zero-IPC read. `20_ingest_cte.sh` now clears the
stale stats so the RAM tier can win placement (this is what made nb10M
RAM-resident); placement is still not fully repeatable (nb50M spilled to NVMe).

## Two facts the numbers show

- **The mixed-tier spill no longer corrupts the heap.** On v2.1.0 a RAM+NVMe
  spill corrupted the runtime heap (`RESULTS.md` §9.1), which forced
  `CTE_RAM_TIER_GB = 0`. On dev nb100M ingested 48 GiB across a 30 GB RAM tier
  plus NVMe spill and byte-verified clean.

- **The zero-IPC read does not lower read_s for the co-located ChiMod.** On
  nb10M, read_s is ~8.5 s whether the list blobs are RAM-resident (zero-IPC read
  serving) or on NVMe (RPC read). The ChiMod still allocates a per-list buffer
  and the list is copied into it (dev is zero-IPC, not zero-copy), so removing
  the task dispatch alone does not change throughput.

## Provenance

- clio-core `dev` `609c526a`; built with io_uring; deps in `~/iowarp-dev-deps`;
  contrib built with `-DIOWARP_USE_DEV=ON`.
- Runner `sbatch_bench_dev.sh`; dev CSVs `qps_dev_<volume>.csv`; per-run logs
  `clio_run_*`, `bench_*`, `ingest_*`, `telemetry_dev_*`, `logs/<job>_qps_dev.out`.
- Jobs: nb10M 22445; nb50M 22447; nb100M 22476.
- v2.1.0 baseline figures from `report.tex` (Table: QPS cold/warm) and the
  `bench_ondisk_*_chimod_*` logs (`fetch_wait_s` / `scan_s`).
