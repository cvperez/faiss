# Are the contribution's issues resolved on clio-core `dev`, and how do the fixes affect performance?

**Status: COMPLETE** (2026-07-29) — resolution verified by code + build +
correctness + runtime instrumentation; performance measured on nb10M/nb50M/nb100M.
Measured numbers: [RESULTS_DEV_ZEROCOPY.md](RESULTS_DEV_ZEROCOPY.md).

## Summary verdict

The blockers this contribution documented (report.tex §"Why This Cannot Be Done
Today"; UPSTREAM_PROPOSAL_IOWARP.md; RESULTS.md §7) are **resolved on clio-core
`dev`** (HEAD `609c526a`). The ChiMod builds and runs against dev with
**bitwise-identical** results, the shared-memory storage path is live, and the
zero-IPC read was **confirmed firing** (instrumented `[DIAG-SHM] HIT`, NVMe
occupancy 0) once a placement pitfall was fixed.

**But** — measured — the fixes do **not** speed up the Level-1 (co-located)
ChiMod, because the zero-IPC read removes an IPC round-trip that was cheap in
this configuration while leaving the actual cost — the per-list buffer + memcpy —
in place. `dev` is **zero-IPC, not zero-copy**.

Two caveats throughout: dev is a strong intermediate, **not** the full
resolve+view+pin zero-copy proposal; and it is on `dev`, **not yet on `main`**.

## Issue-by-issue resolution (code evidence on `origin/dev`)

| # | Issue (v2.1.0 state) | dev resolution | Verdict |
|---|---|---|---|
| 1 | RAM-tier bytes **private to the runtime**, not client-addressable | **#783** `MemBdevTransport::InitShmBacking` shm-backs the RAM bdev (`mem_bdev_transport.cc`) | **Resolved** — observed live: `[#783] RAM bdev '...' is shared-memory backed` |
| 2 | Blob→location **resolve fields disabled** | SHM metadata cache `ShmBlobRecord`/`ShmBlockDesc` + `IsDirectReadable()` + `TryGetBlobRecordShm` (`shm_metadata_cache.h`); published via `BuildShmBlobRecord`/`MirrorBlobToShm` (`core_runtime.cc`) | **Resolved** |
| 3 | No **pin** — blob can be relocated/reclaimed mid-read | **Two mechanisms on dev:** runtime read path now takes a real extent **pin/drain** (`425acdcf`/#753: `TryPinRead`/`UnpinRead`, drain-before-free); the client SHM fast path still uses the `placement_gen_` seqlock (a client-visible pin is explicitly "left as a follow-up", `core_client.h`) | **Resolved** (runtime pin) / follow-up (client pin) |
| 4 | Residual **copy tax** (`fetch_wait_s ≫ scan_s`) | `48203a53` native zero-IPC `AsyncGetBlob`, default-on | **Partially** — IPC removed, **memcpy remains** (zero-IPC ≠ zero-copy); no benefit to the co-located ChiMod (see below) |
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

2. **Even firing, it doesn't speed up the ChiMod.** nb10M inflight=1, RAM-placed:
   warm ~118 QPS, `read_wait_s=8.46` — versus v2.1.0 ~148 QPS / 4.93 s. And the
   decisive control: `read_wait_s ≈ 8.5 s` **whether the fast path hits (RAM) or
   misses (NVMe)**. The read is simply not the bottleneck once RAM-resident.

3. **Why.** The ChiMod's windowed design still `AllocateBuffer` → `AsyncGetBlob`
   (dst = that buffer) → scan → `FreeBuffer` per list. Zero-IPC removes the task
   dispatch but still **copies the list into the buffer**; the buffer alloc/free +
   coroutine scheduling around 12 252 reads (plus dev's #753 pin and `HANGWATCH`
   scheduler stalls) is what dominates. Removing the IPC leaves it unchanged.

4. **What would help — and isn't in dev.** A true **zero-copy view** (resolve →
   map → scan the tier bytes **in place**, no per-list buffer, no memcpy) is the
   report's actual proposal; dev provides zero-IPC only. NVMe-tier lists still
   pay real disk bandwidth regardless.

## Two dev-maturity caveats surfaced by the campaign

- **Blob placement is non-deterministic.** The RAM bdev's self-benchmark returns a
  low bandwidth, so `max_bw` frequently spills RAM-eligible volumes to NVMe (24 GB
  of nb50M; nb10M only stayed in RAM after clearing perf stats). Where blobs land
  on NVMe the fast path cannot apply at all, and dev then equals v2.1.0 exactly.
- **Search is unstable at scale on dev.** nb50M's one RAM-resident run and the
  nb100M search both hit a `HANGWATCH` scheduler-stall / daemon-hang failure
  (`chimod subtask rc=-1`, 60 s server timeout, `bench` abort) — not OOM, not heap
  corruption. Intermittent at 24 GB, reliably fatal at 49 GB. dev is an
  active-development branch (the #753 reader-pin landed mid-campaign); this is a
  liveness bug to report upstream, not a property of the contribution.

## Bottom line

The maintainers' `dev` work **genuinely resolves the engineering blockers** —
shared-memory RAM tier (#783), blob→location resolve, a real read-time extent pin
(#753), **and the mixed-tier spill-heap-corruption (§9.1) is fixed** (clean 48 GiB
ingest + byte-verify). The ChiMod runs correctly on it. But for this
contribution's own Level-1 ChiMod the measured performance effect is **neutral
(slightly negative)**: the zero-IPC read is confirmed firing yet the read was
never the bottleneck once RAM-resident, and the copy it leaves in place is —
because dev is **zero-IPC, not zero-copy**. The result reinforces the report's
actual ask: near-data search needs the **zero-copy scan-in-place view** (no
per-list buffer, no memcpy), which dev does not yet provide. Two dev-maturity
issues (non-deterministic RAM placement; search-at-scale liveness crashes) should
go upstream.
