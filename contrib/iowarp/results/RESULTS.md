# FAISS × IOWarp — Findings log

## 9. No-buffer Level 1 (2026-07-19) — the GetBlob copy tax, measured directly

**What changed.** Per user direction, the chimod's per-search buffer
machinery (v0 fetch-all and v1 windowed fetch+free) was deleted outright:
`OpenIndex` reads every inverted list out of CTE exactly once, directly
into a module-resident arena (raw-pointer GetBlob destinations — no
staging buffer, no memcpy), and `Search` scans that arena in place with
zero `AllocateBuffer`/`GetBlob`/`FreeBuffer`. This makes search behave
exactly as it would under the proposed CTE zero-copy view API — except it
costs a second resident copy of the volume, which is the argument for
upstreaming the API instead ([UPSTREAM_PROPOSAL_IOWARP.md]).

**Correctness.** New per-pass FNV-1a `di_hash` over the full (D, I)
output: chimod == mmap == L0 bitwise at nb10M and nb50M on the real
volumes (plus the existing selftests, incl. repeat-search and split4).

**QPS (exclusive Ares node, step3 protocol: 500 BigANN queries, k=10,
nprobe=nlist/64, 8 scan threads, cold + 2 warm):**

| volume | chimod WITH buffer (07-14) | chimod NO buffer | mmap same node | speedup |
|---|---|---|---|---|
| nb10M (4.84 GB) | 125–152 QPS | **330 / 341 / 342** | 28 cold / 238–360 warm | ~2.4× |
| nb50M (24.2 GB) | 21–30 QPS | **64 / 75 / 75** | 5.7 cold / 70.5–71.9 warm | ~2.7× |

- nb50M scan time is now 6.7 s/pass == mmap's warm scan (7.0 s): with the
  per-search copies deleted, Level 1 search is memory-bandwidth-bound like
  mmap, and cold == warm with zero major faults (placement explicit).
- One-time open load: 4.84 GB in ~5 s (nb10M), 24.2 GB in ~22 s from the
  NVMe file tier (nb50M) — the only data movement left; per-pass CTE
  bytes fetched is 0.
- Volumes ≥ node RAM (nb100M 49 GB, nb178M 87 GB) cannot run this design:
  the resident arena must fit RAM. That is precisely the limitation the
  zero-copy view API removes (scan the tier's own bytes; tiering keeps
  working out-of-core, where chimod already won 20×).

**Run configuration notes (fairness):** nb50M ran CTE file-tier-only
(`CTE_RAM_TIER_GB=0`) because wheel v2.1.0 corrupts its heap when blobs
overflow a RAM tier smaller than the volume (§9.1); the RAM budget is
consumed by the resident arena instead (24 GB arena vs the baseline's
~44 GB page cache — state both next to plots).

### 9.1 Wheel findings hit while getting here (all reproduced, documented for upstream)

1. **Mixed-tier overflow corrupts the runtime heap** (glibc "corrupted
   double-linked list" / garbage pool-ids in routing): any config where
   the ingested volume exceeds the RAM tier and spills to the file tier.
   RAM-only-that-fits and file-tier-only are both stable. Workaround:
   `CTE_RAM_TIER_GB=0` for big volumes; stale `<path>_node<i>` tier files
   now removed by `20_ingest_cte.sh`.
2. **Awaiting a still-pending future with more sub-tasks in flight** can
   let a straggler completion resume the destroyed coroutine frame →
   SIGSEGV in `Worker::ResumeCoroutine` (gdb backtrace captured, hit at
   8192-fetch scale). Safe discipline: poll `IsComplete()`, await only
   completed futures, `chi::yield()` while fetches are outstanding — and
   never yield with nothing in flight (no wake-up ever arrives: hang).
3. **Raw-pointer GetBlob destinations work in-process** (null-allocator
   FullPtr): fetches land directly in module memory — used to delete the
   staging copy from the open-time load as well.
4. **NUMA:** a single-threaded first touch of the arena put 24 GB on one
   socket and quartered scan bandwidth (29 s vs 7 s per pass);
   `FirstTouchParallel` (8 threads, static chunks) restores mmap-class
   page distribution. mmap gets this for free from parallel page faults.

**Provenance:** jobs 22065/22069 (nb10M), 22068 (nb50M mmap reference +
crash), 22071 (NUMA-slow), 22072 (final nb50M); with-buffer comparison
rows in `qps_ondisk_{nb10M,nb50M}.csv` from the 07-14 AVX-512 campaign;
`results/logs/<job>_qps.out`, `clio_run_*.log`, `telemetry_*_2206*.csv`.

---

**Campaign status:** in progress (2026-07-13). Volumes nb100M / nb44M /
nb178M running (jobs 21908–21910, chimod inflight=1, telemetry on).
This file consolidates everything established so far; final tables and
figures land when the chain completes.

## 1. Run conditions (comparability contract)

Protocol identical to the user's step2/step3 baselines
(`ondisk_step2_search.py:run_pass` / `ondisk_step3_mixed.py:run_baseline_pass`):
first 500 of the 10k BigANN queries (bvecs→f32), k=10,
nprobe = max(1, nlist/64), 8-thread CPU budget, ONE batched search per
pass, passes = cold + 2 warm, cold = `posix_fadvise(DONTNEED)` on the
volume (the baseline's own mechanism; NOT `drop_caches`).
- mmap baseline numbers are the user's existing measurements (step2 for
  nb50M/nb100M; step3 **Stage-4** for nb44M/nb178M — those match the
  surviving capacity-doubled files). Never re-run.
- IOWarp RAM budget: CTE RAM tier capped at 30 GB + NVMe tier 120 GB
  (`config/ares_cte.yaml`); baseline had ~44 GB of page cache. State both
  next to every plot.
- chimod CPU budget: 8 scan threads INSIDE one task (`inflight=1`) ==
  baseline's 8 OMP threads.

## 2. Correctness (all gates passed)

Bitwise-identical `(D, I)` vs stock FAISS in every configuration: L0
(L2 + IP metrics), chimod v0 (fetch-all) and v1 (pipelined), split-batch
(4 concurrent subtasks). Ingestion byte-verified against mmap pointers on
the real volumes (login + compute nodes). Recall is equal by
construction; it is not measured.

## 3. Headline QPS so far (ondisk_nb10M, 10M vectors, 4.9 GB, exclusive node)

> **PROVISIONAL — scalar-kernel numbers.** Everything in this section and
> the per-volume results measured before 2026-07-14 used a
> `FAISS_OPT_LEVEL=generic` build, while the mmap baseline's python wheel
> scans with AVX-512 — a ~3-4× kernel handicap AGAINST IOWarp on
> scan-bound volumes. Ratios measured under identical binaries (L0 vs L1,
> the inflight sweep) are valid as architecture comparisons; any absolute
> QPS and ANY comparison against the mmap baseline (regimes, crossover
> point) awaits the AVX-512 campaign (jobs 21917-21921) and will replace
> these tables.

| System | cold / warm QPS | majflt/q | disk reads |
|---|---|---|---|
| L0 — IOWarp tiering only, compute in client | 15.0 / 14.5–15.1 | 0 | ~0 |
| L1 chimod, serial scan (1 worker) | 61.6 / 65–66 | 0 | 0 |
| **L1 chimod, parallel scan (8 threads in-task)** | **148 / 146–152** | 0 | 0 |
| mmap warm reference (step2, at 50M) | 81–93 | — | — |

Both IOWarp systems: **cold ≈ warm, zero major faults** — no warmup
transient; placement is explicit and deterministic. (nb10M mmap itself was
never measured by the baseline campaigns.)

## 4. The client-boundary cost (why L0 is slow — its role is this measurement)

Per probed list L0 pays: shm alloc + task IPC round-trip + tag/blob/tier
lookup + memcpy of the list into the client + free; and with no
cross-query reuse (buffers die at refcount 0), hot lists are re-fetched
per touching query: ~32,000 round-trips per 500-query pass vs ~4,096
unique lists (~8× amplification). ≈1 ms overhead per ≈0.2 ms of scan.
The chimod removes the amplification and the boundary (fetch once,
scan in place, return top-k = k·12 B/query) → the 15 → 150 QPS gap.
kStats at nb10M: 14.5 GB fetched, fetch-wait 4.9 s vs scan 4.0 s over
3 passes — fetch/scan genuinely overlap; the residual fetch cost is the
GetBlob memcpy (see §7 upstream proposal).

## 5. Parallelism-vs-dedup ablation (task-split sweep, nb10M, exclusive node)

Splitting the batch across concurrent SearchTasks monotonically HURTS at
this scale — each subtask re-fetches lists its siblings already have:

| concurrent tasks | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| warm QPS (serial-scan build) | 65 | 53 | 42 | 34 |

Resolution: parallelize INSIDE the task (per-arrived-list scan across 8
per-thread scanners; safe — each (query,probe) pair updates a distinct
per-query heap). Keeps full fetch dedup AND the 8-thread budget →
§3's 150 QPS. Design lesson for near-data batch processing: mmap's
threads get dedup for free via shared address space; task parallelism
does not — dedup must be engineered.

## 6. Engineering findings (integration + runtime)

1. **No changes to clio-core or FAISS core.** Stock pip wheel v2.1.0 +
   pinned v2.1.0 headers; the unmodified runtime dlopens the out-of-tree
   chimod. FAISS integrated via public `InvertedLists`/scanner seams.
   Wheel gaps solved in-tree: vendored zmq 4.3.5 / msgpack-c 6.1.0 /
   cereal 1.3.2 headers; in-tree CTP compile definitions replicated.
2. **Never block a cooperative worker**: `CLIO_CTE_CLIENT_INIT()` inside
   a handler deadlocks the runtime (blocking `Wait()`); bind an
   in-process client to `kCtePoolId` instead.
3. **Never issue unbounded tasks from a handler**: 4096 simultaneous
   GetBlobs livelocked the runtime queues (depth 1024). Windowed
   issuance (64 in flight); v1 frees each list after scanning → peak shm
   O(window), scales past allocator size.
4. gcc 11 ICEs on OpenMP regions inside C++20 coroutines — hoist the
   parallel region into a plain function.
5. Ares compute-partition images lack BLAS/LAPACK (debug nodes have
   them); real copies bundled into `~/faiss-install/lib`.
   `sudo drop_caches` unavailable in batch jobs.
6. CTE stores lists compactly: the capacity-doubled step3 files shrink
   on ingest (43→~22 GB, 173→~87 GB expected; confirms at 21909/21910) —
   L0/L1 structurally avoid the mmap slot-doubling pathology that
   permanently degraded the baseline's post-write files.
7. Ingestion is NFS-read-bound (~4.84 GB in ~2 min).

## 7. Follow-ups / upstream proposals

- CTE zero-copy read ("pin blob, return tier pointer") would remove the
  remaining GetBlob memcpy — RAM-tier reads would approach mmap cost with
  tier control retained. Natural IOWarp upstream proposal, mirroring the
  §8 (FAISS async-InvertedLists) proposal in IMPLEMENTATION_PLAN.md.
  **Feasibility verified in v2.1.0 source** (see UPSTREAM_PROPOSAL_IOWARP.md
  for the full analysis + file:line cites): no zero-copy path exists today
  — RAM bdev pages are private process-heap (not shm), GetBlobInfo's
  placement fields are disabled in the source, and no pin prevents
  reorganize/eviction from reclaiming bytes mid-read; but block addresses
  are already lifetime-stable, so upstream needs only a local view
  accessor, the (already drafted) blob→blocks resolve, and a pin
  refcount. Decision: NOT worked around locally (a module-side cache
  would duplicate the tier's RAM role); the numbers stand as the
  motivation for the upstream API.
- L0 per-batch pinned cache (fix the 8× re-fetch) if L0 is ever revisited.
- Telemetry (`telemetry_sampler.py`, 5 s ticks, phase-tagged): NVMe
  rd/wr MB/s + active %, CTE NVMe-tier occupancy, /dev/shm + runtime RSS
  (RAM tier), NFS read rate — the Figure-3a analogue; plots pending data.

## 8. Data inventory

- `results/qps_<volume>.csv` — per-pass QPS + majflt + read_bytes (+
  mode/inflight in notes). nb10M rows: L0 (jobs 21884/21894), chimod
  serial + sweep (21894), chimod parallel-scan (21903).
- `results/telemetry_<volume>_<job>.csv` — from job 21908 onward.
- `results/logs/<job>_qps.out`, `results/ingest_*.log`,
  `results/clio_run_*.log` — full provenance.
