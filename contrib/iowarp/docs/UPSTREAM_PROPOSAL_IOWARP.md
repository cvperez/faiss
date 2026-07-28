# Proposal to iowarp/clio-core: zero-copy blob reads for co-located compute

*From the FAISS × IOWarp integration (`contrib/iowarp`). Built against the
iowarp-core pip wheel v2.1.0, unmodified. Source references below are to the
matching clio-core v2.1.0 tree.*

## Summary

We host FAISS IVF vector search inside the CLIO runtime as a ChiMod, with
inverted lists stored one blob per list in CTE (RAM + NVMe tiers). The module
lives in the runtime's own process, next to the RAM tier that holds the bytes.
Yet its search is **copy-bound**: `AsyncGetBlob` is the only read primitive, and
it materializes a copy of every blob it reads — even when those bytes already
sit in the RAM tier of the caller's process. We propose a small
resolve / view / pin API that makes RAM-tier reads pointer-cheap for a
co-located module.

See [DESIGN_AND_SOLUTION.md](DESIGN_AND_SOLUTION.md) for how this cost arises in
the search path.

## Why reads are copy-bound

To score an inverted list, the module needs its codes and ids as ordinary
memory. CTE's only read path (`AsyncGetBlob`) copies the blob's bytes out of the
tier into a destination buffer before anything can touch them. A query batch
probes essentially every list, so a search copies about the whole index out of
CTE, one list at a time — the search waits on copies more than it computes. The
bucket-per-blob layout is not the issue (placement is explicit and per-list);
the issue is that a *read is a copy*, even for bytes already present in RAM in
the module's address space.

## Why we could not do zero-copy ourselves (v2.1.0 source analysis)

1. **RAM tier bytes are private process heap.** The RAM bdev stores data in
   process-local heap pages (`new char[…]`,
   `modules/bdev/src/bdev_runtime.cc:1075`) and reads memcpy out of them
   (`ReadFromRam`, `bdev_runtime.cc:1142`). Those pages are not in the shared-
   memory segments, so a `ShmPtr` cannot address them, and the page accessors
   are private.
2. **The blob→placement query is disabled.** `GetBlobInfo`'s block fields are
   drafted (`BlobBlockInfo{target_pool_id_, block_size_, block_offset_}`,
   `context-transfer-engine/core/include/…/core_tasks.h:1886`) but not populated
   or serialized — no caller can resolve a blob to (target, offset) today, even
   in-process.
3. **Nothing pins data.** `ReorganizeBlob`
   (`context-transfer-engine/core/src/core_runtime.cc:1240`), overwrites, and
   eviction free blocks for immediate reuse, and reads hold no lock during I/O —
   scanning tier memory directly would race with reclamation (use-after-free).
4. **The file tier never mmaps.** File-tier bytes are moved with async
   pread-style I/O, so they have no in-process address at all.

The encouraging part: a RAM-tier block's address is already **stable for the
block's lifetime**, and blocks are contiguous within a page. The storage already
behaves the way a borrowed view needs; only the access discipline is missing.

## Proposed API (minimal, conceptual)

1. **Resolve** — a runtime-local way to map `(tag, blob)` to its blocks
   `(target_pool, offset, size)`: either finish the drafted `GetBlobInfo` block
   fields or add a `ResolveBlob`.
2. **View** — on the RAM bdev, a `GetLocalView(offset, size) -> char*` that
   returns a pointer only when the range lies within one RAM page in the
   caller's process, and null otherwise (the caller then falls back to the copy
   path).
3. **Pin** — a per-blob (or per-block) read refcount honored by the block
   allocator, `ReorganizeBlob`, and eviction, so a borrowed view cannot be
   reclaimed mid-scan. `PinBlob -> {views, lease}` + `UnpinBlob(lease)` composes
   resolve + view + pin.

*(Optional, larger)* allocate RAM pages from the shm allocator so views also
work cross-process, and an mmap mode for the file bdev so the file tier can
participate.

With this, a co-located module scans the RAM tier's own bytes in place; the
per-search copy — an amount equal to the index size — disappears, while explicit
tiers, no eviction cliff, and per-list placement control remain. It generalizes
to any near-data module, not just FAISS.

## Secondary items from the same integration

- **Backpressure on handler-issued tasks.** Issuing very many `AsyncGetBlob`s
  from one handler can livelock the runtime (the queue is consumed by the same
  workers). We window client-side; `Send` yielding under queue pressure (or a
  documented hard limit) would prevent the failure mode.
- **Handler-safe CTE client init.** `CLIO_CTE_CLIENT_INIT()` called from a
  handler deadlocks the worker (a blocking `Wait()` inside). The working pattern
  — bind an in-process `Client` to `kCtePoolId` — deserves documentation in the
  module template, or an awaitable init.
- **Batched blob ops.** One task carrying N gets/puts; IVF batches touch many
  thousands of blobs.
- **C++ SDK packaging.** The pip wheel exports the needed symbols but ships no
  headers or CMake package; we build against a source checkout pinned to the
  wheel tag and vendor zmq/msgpack/cereal headers. A companion dev package would
  make out-of-tree ChiMods turnkey.
- **Client-visible tier telemetry.** Per-target occupancy / bytes-moved counters
  (we reconstruct them from `/proc/diskstats` and directory sizes).

## Offer

The `faiss_ivf` ChiMod plus its benchmark harness is available as a reproducible
near-data test case: coroutine handlers, in-process CTE calls, bulk shm data
paths, and an out-of-tree build against the wheel. Happy to contribute it as an
example module and to validate the resolve/view/pin API against it.
