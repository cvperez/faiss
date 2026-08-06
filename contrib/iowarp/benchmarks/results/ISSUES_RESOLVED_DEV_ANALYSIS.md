# Are the contribution's issues resolved on clio-core `dev`, and how do the fixes affect performance?

**Status: COMPLETE** (2026-07-29) — resolution verified by code + build +
correctness + runtime instrumentation; performance measured on nb10M/nb50M/nb100M.
Measured numbers: [RESULTS_DEV_ZEROCOPY.md](RESULTS_DEV_ZEROCOPY.md).

**UPDATED 2026-08-05 — the large-spill corruption is RESOLVED on dev
`8715591a`.** The nb178M mixed-tier gate (job 22786, RAM 30 GB + 57 GB
spill) now passes a FULL read-back verification — `ivf_to_iowarp --verify
all`, every one of 16 384 lists byte-identical via RPC GetBlob right after
ingest — and completes all search passes at 8.8/9.4/9.4 QPS with the same
di_hash as the file-tier-only control from the previous HEAD. The fix
landed upstream between `4cc0d780` and `8715591a` (the #909/#910
"safe-bdev concurrent alloc race" family is the plausible vehicle; our
allocator-race hypothesis below, §4 discussion, matched that shape). **The
planned upstream corruption issue is withdrawn as unnecessary.** One NEW
dev-semantics adaptation was required this round (the fifth): #856's strict
event-resume guard starves yield-polling in handlers — see
RESULTS_DEV_ZEROCOPY.md §"2026-08-05 campaign". Two findings remain worth
mentioning to the maintainers informally: `ReadData` still collapses all
bdev failure kinds into `rc=1`, and on 2-node runs the daemon-side shm
segments grow with cross-node bytes moved and never recycle (OOM at
nb100M scale — jobs 22795/22818, `clio_run_*_node1.log`).

**UPDATED 2026-07-30** — the first campaign's two negative conclusions
("no speedup for the co-located ChiMod" and "search is unstable at scale on
dev") are **superseded**. Both were artifacts of running a v2.1.0-shaped
ChiMod against dev's changed runtime semantics. After three ChiMod-side
adaptations (parallel `TryReadBlobShm` bursts, prompt future release, pooled
buffers — commit `4b026bfc`) and updating to dev HEAD `4cc0d780`, the ChiMod
is **~2× the v2.1.0 baseline at every size** (251 / 52 / 17.7 cold QPS at
nb10M / nb50M / nb100M) with bitwise-identical results, and nb100M completes
all passes. Sections below are annotated where superseded.

## Summary verdict

The blockers this contribution documented (report.tex §"Why This Cannot Be Done
Today"; UPSTREAM_PROPOSAL_IOWARP.md; RESULTS.md §7) are **resolved on clio-core
`dev`** (verified at HEAD `609c526a`; final campaign on `4cc0d780` — the 20
commits between them touch neither `core_client.h` nor `future.h`). The
ChiMod builds and runs against dev with
**bitwise-identical** results, the shared-memory storage path is live, and the
zero-IPC read was **confirmed firing** (instrumented `[DIAG-SHM] HIT`, NVMe
occupancy 0) once a placement pitfall was fixed.

~~**But** — measured — the fixes do **not** speed up the Level-1 (co-located)
ChiMod~~ **[superseded 2026-07-30]**: that measurement ran the unadapted
port. Dev's zero-IPC `AsyncGetBlob` copies *synchronously on the calling
thread*, so the ChiMod's 64-deep async pipeline had silently degenerated
into serial copies on one cooperative worker — the fast path was being
measured with its parallelism turned off. Once the ChiMod fetches
RAM-resident lists with direct `TryReadBlobShm` calls 8-wide across the scan
threads (and stops leaking futures/buffers, see below), the zero-IPC read is
worth **~2× end-to-end** (nb10M 130→251, nb50M 21→52, nb100M 8.9→17.7 cold
QPS vs v2.1.0; read_s 5.74→1.37, 39.60→6.67, 122.67→33.41).

What remains true: `dev` is **zero-IPC, not zero-copy** — one memcpy per
list survives, and with the read fixed **scan_s is now the bottleneck at
every size**, which is precisely the cost the resolve+view+pin zero-copy
proposal targets. And it is on `dev`, **not yet on `main`**.

## Issue-by-issue resolution (code evidence on `origin/dev`)

| # | Issue (v2.1.0 state) | dev resolution | Verdict |
|---|---|---|---|
| 1 | RAM-tier bytes **private to the runtime**, not client-addressable | **#783** `MemBdevTransport::InitShmBacking` shm-backs the RAM bdev (`mem_bdev_transport.cc`) | **Resolved** — observed live: `[#783] RAM bdev '...' is shared-memory backed` |
| 2 | Blob→location **resolve fields disabled** | SHM metadata cache `ShmBlobRecord`/`ShmBlockDesc` + `IsDirectReadable()` + `TryGetBlobRecordShm` (`shm_metadata_cache.h`); published via `BuildShmBlobRecord`/`MirrorBlobToShm` (`core_runtime.cc`) | **Resolved** |
| 3 | No **pin** — blob can be relocated/reclaimed mid-read | **Two mechanisms on dev:** runtime read path now takes a real extent **pin/drain** (`425acdcf`/#753: `TryPinRead`/`UnpinRead`, drain-before-free); the client SHM fast path still uses the `placement_gen_` seqlock (a client-visible pin is explicitly "left as a follow-up", `core_client.h`) | **Resolved** (runtime pin) / follow-up (client pin) |
| 4 | Residual **copy tax** (`fetch_wait_s ≫ scan_s`) | `48203a53` native zero-IPC `AsyncGetBlob`, default-on; `TryReadBlobShm` public host API | **Largely resolved** — used correctly (parallel direct reads, not the sync-fast-path future API), read_s drops 4–8× and `fetch_wait ≫ scan` inverts to `scan > fetch_wait` at every size. The last memcpy per list remains (zero-IPC ≠ zero-copy) |
| 5 | Mixed-tier spill **corrupts the v2.1.0 heap** (RESULTS §9.1) | dev bdev/CTE tiering reworked (#858 lazy growth, #753 ClearBlob reorg, #857) | **Resolved (ingest)** — nb100M ingested 48 GiB across a 30 GB RAM tier + NVMe spill, byte-verified clean, no corruption (v2.1.0 corrupted here) |

Report capabilities: **Resolve** = `IsDirectReadable`+block descriptors ✅;
**View** = shm-mapped RAM bdev + direct read ✅; **Pin** = runtime extent pin
(#753) ✅ on the RPC path / seqlock on the client fast path.

## How the fixes affect performance (measured)

The report's premise: search is **data-movement-bound** (v2.1.0 nb100M
`fetch_wait=122.7 s ≫ scan=47.8 s`; scalar→AVX-512 barely changed QPS), so
removing the per-list copy should help. Testing that on dev:

1. **First, the fast path wasn't firing — a placement pitfall.** A stale
   `~/.clio/bdev_perf` RAM-tier reading (476 MB/s vs NVMe 1637 MB/s) made the
   `max_bw` DPE place list blobs on **NVMe**, where they are never
   `kShmBlobDirectReadable`. Instrumentation showed `not-direct flags=0` on every
   list → silent RPC fallback. **Fixed** by clearing the stale stats (now in
   `20_ingest_cte.sh`); the volume then sits 100% in RAM (NVMe occupancy 0) and
   `[DIAG-SHM] HIT` confirms the zero-IPC read serves every list.

2. ~~**Even firing, it doesn't speed up the ChiMod.**~~ **[superseded
   2026-07-30 — the measurement was confounded.]** The "decisive control"
   (`read_wait_s ≈ 8.5 s` on RAM hit and NVMe miss alike) was not showing
   that the read doesn't matter — it was showing that **on a hit, the
   inline synchronous memcpy in `TryShmGet` serialized all reads onto the
   one worker running the Search coroutine**, costing about what the RPC
   path costs. The zero-IPC fast path was firing with its concurrency
   removed. Same numbers, opposite conclusion.

3. **What actually dominated (root-caused via fixed telemetry).** Three
   v2.1.0 assumptions in the ChiMod turned toxic on dev: (a) the async
   pipeline serialized (above); (b) every fast-path hit synthesized a shm
   `GetBlobTask` that the pinned futures kept alive — ~16 k tasks/search at
   nb100M until the allocators exhausted and the daemon stalled; (c) dev's
   segment allocator never recycles freed multi-MB buffers, so per-list
   `AllocateBuffer`/`FreeBuffer` grew mapped segments by ~the bytes pushed
   (+10 GB in ingest, +14 GB per search pass) until the OOM killer took the
   daemon. Fixes (commit `4b026bfc`): 8-wide direct `TryReadBlobShm` bursts
   with the RPC pipeline kept for misses; futures dropped per-list; fixed
   pools of reusable buffers in both ingest and search. Result: 251 / 52 /
   17.7 cold QPS, memory flat through all passes.

4. **What would still help — and isn't in dev.** A true **zero-copy view**
   (resolve → map → scan the tier bytes **in place**, no per-list buffer, no
   memcpy) is the report's actual proposal; dev provides zero-IPC only. With
   the read fixed, scan_s now exceeds read_s at every size, so the remaining
   copy+scan is exactly where the view/pin API would bite. NVMe-tier lists
   still pay real disk bandwidth regardless.

## Two dev-maturity caveats surfaced by the campaign

- **Blob placement is non-deterministic** — *mitigated.* The stale
  `~/.clio/bdev_perf` reading was the trigger; with `20_ingest_cte.sh`
  clearing it, the 2026-07-30 campaign placed all three volumes correctly on
  the first try (nb10M and nb50M fully RAM-resident, nb100M the expected
  30 GB RAM + 19 GB NVMe split). Still worth checking per run (the job log
  prints NVMe tier occupancy) since the DPE remains bandwidth-driven.
- ~~**Search is unstable at scale on dev.**~~ **[root-caused 2026-07-30 —
  not a dev scheduler liveness bug, and partly it *was* memory
  exhaustion.]** The `chimod subtask rc=-1` / daemon-death signature had two
  concrete causes, both in how the v2.1.0-era ChiMod used dev: pinned
  futures holding ~16 k shm tasks per search (allocator exhaustion → worker
  stall → daemon gone), and non-recycled per-list buffers growing mapped
  segments until the **OOM killer** took the daemon (jobs 22481/22482 —
  daemon RSS staircased 27→56 GB within one pass). With the ChiMod fixed,
  nb50M and nb100M complete all passes with flat memory. What *is* worth
  reporting upstream as dev behavior: the client segment allocator's
  non-recycling of freed multi-MB buffers, and segments staying mapped in
  the daemon after the owning client exits.
- **Placement records corrupt at large ingest-time spill volumes**
  (found 2026-07-31, nb178M campaign). With RAM 30 GB + 56 GB NVMe spill,
  the shm metadata cache keeps claiming direct-readability for spilled
  blobs: the zero-IPC fast path then "reads" 86 GB in 12 s with **zero disk
  I/O** — silently returning stale RAM-tier bytes (job 22495; the
  placement_gen seqlock does not catch it because the stale record is
  *stable*, not mid-move) — and the daemon's own RPC GetBlob fails with
  rc=1 on 42 of ~10.4 k spilled blobs (job 22496). At nb100M's 19 GB spill
  every record is correct, so the corruption onsets between 19 and 56 GB of
  spill. Workaround: `CTE_RAM_TIER_GB=0` (direct-to-file-tier placement, no
  reorganization) runs clean — nb178M completed 3/3 passes at 8.7/6.6/6.6
  QPS with zero failures (job 22497). This is the most serious of the three
  upstream findings: the fast-path variant returns **wrong data without any
  error**.

## Bottom line

The maintainers' `dev` work **genuinely resolves the engineering blockers** —
shared-memory RAM tier (#783), blob→location resolve, a real read-time extent pin
(#753), **and the mixed-tier spill-heap-corruption (§9.1) is fixed** (clean 48 GiB
ingest + byte-verify). And — after adapting the ChiMod to dev's semantics
(sync fast path → parallel direct reads; shared_ptr futures → prompt release;
non-recycling allocator → pooled buffers) — the payoff is real: **~2× the
v2.1.0 baseline at every size** (251 / 52 / 17.7 cold QPS; nb100M completes),
bitwise-identical results throughout. The earlier "neutral" verdict measured
the unadapted port, i.e. dev's fast path with its concurrency accidentally
removed.

The report's actual ask still stands, sharpened: dev is **zero-IPC, not
zero-copy**, and with the read fixed **scan_s now exceeds read_s
everywhere** — the per-list memcpy+scan is the last data-movement cost, and
the **zero-copy scan-in-place view** (no per-list buffer, no memcpy) is what
would remove it. Upstream-worthy dev findings from this campaign: (1) the
client segment allocator does not recycle freed multi-MB buffers; (2) client
segments stay mapped in the daemon after the owning client exits; (3) the
zero-IPC `AsyncGetBlob` fast path silently serializes async pipelines (a
documented "completes-inline" note or an async-preserving variant would
spare the next porter); and (4) — most serious — at large ingest-time spill
volumes (between 19 and 56 GB) blob placement records go stale: the shm fast
path then returns **wrong bytes with no error and no I/O**, and the RPC path
fails rc=1 on the affected blobs (see the caveat above; workaround
`CTE_RAM_TIER_GB=0`).
