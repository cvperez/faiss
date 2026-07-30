# FAISS × IOWarp — dev-branch (zero-IPC AsyncGetBlob) experiment results

**Status: COMPLETE** (2026-07-29). nb10M (clean, RAM-placed, fast-path confirmed);
nb50M (NVMe-placed ok + one RAM-placed crash); nb100M (spill ingest fixed, search
crashed). nb178M not run — the nb100M search crash answers the remaining question
and the same instability would recur.

## TL;DR

- All three report blockers are **resolved** on dev; the ChiMod runs with
  **bitwise-identical** results; the #783 shm RAM bdev and the zero-IPC read are
  **confirmed live** (instrumented HIT, NVMe occupancy 0).
- The zero-IPC read **does not speed up the co-located ChiMod** — `read_wait_s` is
  identical whether the fast path hits or misses, because dev is *zero-IPC, not
  zero-copy* and the ChiMod still allocates+copies a per-list buffer.
- The v2.1.0 **spill-heap-corruption (§9.1) is fixed** — nb100M ingested 48 GiB
  across a 30 GB RAM tier + NVMe spill and byte-verified clean.
- **New dev instability:** search on large volumes crashes via `HANGWATCH`
  scheduler stalls / daemon hang (nb50M RAM path once; nb100M search reliably).

## What this measures

The Level-1 `clio_faiss_ivf` ChiMod, **unchanged in behaviour**, rebuilt and run
against a **clio-core `dev` source install** (HEAD `609c526a`, ~277 commits ahead
of the pinned v2.1.0) instead of the v2.1.0 pip wheel. `dev` makes the zero-IPC
shared-memory read **native to `AsyncGetBlob`, default-on** (commit `48203a53`,
issue #783), backed by a shared-memory RAM bdev + an SHM metadata cache; a later
commit (`425acdcf`, #753) adds a runtime-side extent **pin/drain** around reads.

Protocol held fixed vs the v2.1.0 baselines: first 500 BigANN queries, k=10,
`nprobe=nlist/64`, 8 threads, cold + 2 warm passes, `FAISS_OPT_LEVEL=avx512`,
same windowed-fetch search (≤64 in-flight `AsyncGetBlob` → per-list buffer →
scan → free). Comparisons are at **matched `inflight`**.

## ⚠️ Critical setup finding — the fast path only fires with RAM placement

The zero-IPC direct read only serves a blob that is **RAM-resident and flagged
`kShmBlobDirectReadable`**. On the first dev runs it **never fired**: a stale
`~/.clio/bdev_perf/*.perf` reported the RAM tier at **476 MB/s** (vs NVMe
**1637 MB/s**), so the `max_bw` DPE placed every inverted-list blob on **NVMe**,
where it can never be direct-readable. Instrumenting `TryReadBlobShm` showed
`not-direct … flags=0` on **every** list → silent fallback to the RPC read path
(so dev looked like v2.1.0 + fallback overhead).

**Fix (now baked into `20_ingest_cte.sh`):** clear the stale perf stats so the
tiers tie and the RAM tier's higher score wins placement. With that, instrumentation
shows `[DIAG-SHM] HIT (zero-IPC read served from shm)` on the list blobs, and the
NVMe file-tier occupancy after the run is **0** — i.e. the volume is 100%
RAM-resident and the zero-IPC path is genuinely serving every read.

## Correctness (gate — all passed)
Bitwise-identical `(D, I)` to stock FAISS at every size (FNV-1a `di_hash`):
selftest (single + split-4) identical; nb10M `0e04f8171f7d1898`,
nb50M `32085d09bb65391b` — matching the v2.1.0/mmap baselines exactly.

## Headline result: the zero-IPC path does NOT speed up the co-located ChiMod

nb10M, matched `inflight=1`, **RAM placement confirmed (NVMe occupancy = 0,
fast path HITing)**:

| metric | v2.1.0 (wheel) | dev (fast path FIRING) | dev (fast path MISSING, blobs on NVMe) |
|---|---|---|---|
| read/fetch_wait_s | 4.93 | **8.46** | 8.46–8.71 |
| scan_s | 3.96 | 4.59 | 4.29 |
| warm QPS | ~148 | **117–120** | 116–120 |
| GB read / lists | 14.49 / 12252 | 14.49 / 12252 | identical |

**The decisive observation:** `read_wait_s` is **≈8.5 s whether the fast path
hits or misses** (RAM vs NVMe placement). If the read were the bottleneck, serving
it zero-IPC-from-shm would collapse that number — it does not move at all. So for
this ChiMod the read/copy is **not** the dominant cost once data is RAM-resident;
the per-list **buffer management + fetch/scan serialization** is (heavier on dev,
which also logs `HANGWATCH` scheduler stalls and now carries the #753 pin/drain).

**Why zero-IPC can't help here.** The ChiMod's windowed design still does
`AllocateBuffer` → `AsyncGetBlob(dst=buffer)` → scan → `FreeBuffer` for every
list. The dev fast path only removes the *task dispatch*; it still **memcpys the
list into that buffer** (zero-IPC ≠ zero-copy). The buffer alloc/free +
coroutine scheduling around 12 252 reads dominates, so eliminating the IPC leaves
throughput unchanged (slightly lower here, from dev's added per-read machinery).

The optimization the report actually asked for — a **zero-copy view**: resolve →
map → scan the tier's bytes **in place**, with *no per-list buffer and no memcpy*
— is what would help the co-located ChiMod, and `dev` does **not** provide it (it
is zero-IPC only). This is consistent with the report's §"Scope of the Improvement".

## nb50M — dev ≈ v2.1.0 (and two dev-maturity findings)

nb50M (24 GB), inflight=1:

| run | placement | read_wait_s | scan_s | warm QPS | outcome |
|---|---|---|---|---|---|
| v2.1.0 | (RAM/page-cache) | 33.32 | 21.36 | ~27–30 | ok |
| dev 22447 | **NVMe (21 GB spilled)** | 32.72 | 21.56 | 28.9–29.5 | ok, di_hash ✓ |
| dev 22446 | RAM (NVMe=0) | — | — | cold 20.2 only | **daemon crashed on warm pass** |

Two findings beyond the headline:
- **Placement is non-deterministic.** Clearing the stale perf stats does not
  reliably move blobs to RAM: the RAM bdev's self-benchmark keeps returning a low
  bandwidth, so `max_bw` often still spills to NVMe (21 GB here). Reliable RAM
  placement needs a RAM-only tier config (no NVMe) — but see next.
- **The RAM/fast-path route is unstable at 24 GB.** The one run that did land on
  RAM (22446) **crashed on the warm pass**: the `clio_run` daemon hung (client
  hit a 60 s server timeout, `chimod subtask rc=-1`) amid runaway client-shm
  growth and scheduler stalls (`stalls_detected`, `HANGWATCH`) — not OOM
  (37.9 GiB free). Transient (the NVMe-placed retry completed), but a real
  dev-maturity signal for this workload. When it *does* complete, **NVMe-placed
  dev equals v2.1.0** — expected, since the fast path can't serve NVMe blobs.

## nb100M (49 GB) — spill corruption FIXED; search crashes

Run at `CTE_RAM_TIER_GB=30` (30 GB RAM tier + NVMe spill — the exact config that
corrupted the v2.1.0 heap, RESULTS.md §9.1):

- **Ingest + spill: clean.** `[ingest] done: 48.43 GiB compact`, `[verify] PASS —
  16 lists byte-identical`. No glibc `corrupted double-linked list`, no garbage
  pool-ids. **The v2.1.0 spill-heap-corruption bug is fixed on dev** — big volumes
  can now use a real RAM tier + NVMe spill instead of `CTE_RAM_TIER_GB=0`.
- **Search: crashed.** After ingest, `chimod_search_parallel` failed and
  `bench_ivf_qps` aborted; the daemon showed a persistent `HANGWATCH` backlog
  (47–61 outstanding tasks across 60–74k processed, recurring `stalls_detected` /
  lane-rescues) — the same scheduler-stall instability seen intermittently at
  nb50M, here reliably fatal at 49 GB. Not heap corruption; a scheduler/IPC
  liveness failure under sustained search load.

v2.1.0 baselines for reference (all-NVMe, `CTE_RAM_TIER_GB=0`, inflight=1):
nb100M fetch_wait=122.67 s, warm 7.8–7.9 QPS; nb178M fetch_wait=258.47 s,
warm 3.9–4.0 QPS. No clean dev search QPS could be captured at these sizes due to
the crash. **nb178M not run** — it would exhibit the same search instability.

## Provenance
- clio-core `dev` `609c526a`; trimmed build (RUNTIME+CTE) → `~/clio-core-dev-install`;
  deps in `~/iowarp-dev-deps` (apt-get download + dpkg -x, no sudo).
- contrib branch `iowarp-dev-zerocopy`, `-DIOWARP_USE_DEV=ON` → `build-dev/`.
- Runner `sbatch_bench_dev.sh` (dev toggle; clears stale bdev perf stats; reports
  NVMe-tier occupancy). Dev CSVs `qps_dev_<volume>.csv`.
- Jobs: nb10M 22445 (clean); diagnosis 22442–22444; nb50M 22446; nb100M pending.
- Fast-path firing verified by temporary `TryShmGet`/`TryReadBlobShm`
  instrumentation (since reverted); placement verified by NVMe occupancy = 0.
