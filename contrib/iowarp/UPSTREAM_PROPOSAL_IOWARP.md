# Proposal to iowarp/clio-core: zero-copy blob reads for co-located compute

*Prepared from the FAISS×IOWarp integration study (contrib/iowarp in the
cvperez/faiss fork). All measurements: Ares, 1 exclusive node (48 GiB RAM,
local NVMe), BigANN, iowarp-core pip wheel v2.1.0 unmodified.*

## Summary

We host FAISS IVF vector search inside the CLIO runtime as an out-of-tree
ChiMod, with inverted lists stored one-blob-per-list in CTE (RAM + NVMe
tiers). It works well — out-of-core it beats an mmap/page-cache baseline
by 20× (4.0 vs 0.2 QPS on a 90 GB index) with zero page faults and no
warmup cliff. But in the in-RAM regime the module is **copy-bound**:
`AsyncGetBlob` is the only read primitive and it materializes a copy of
every blob, even for a module living in the same address space as the
tier that holds the bytes. We propose a small pin/resolve/view API that
makes RAM-tier reads pointer-cheap.

## Evidence that reads are copy-bound

50M-vector IVF,Flat index (24 GB, fully RAM-tier resident), 500-query
batches, 8 threads, per-pass internal stats from the module:

- the entire 24 GB is copied out of CTE every pass (fetch dedup is
  perfect — each unique blob fetched exactly once);
- fetch-wait 13.2 s/pass vs scan 7.0 s/pass — waiting on copies exceeds
  useful compute;
- switching FAISS from scalar to AVX-512 kernels changed QPS by ~0% —
  the bottleneck is data movement, not distance computation;
- against warm mmap (where the cached page IS the scan target) the copy
  tax is the whole remaining gap: ~25 vs ~93 QPS.

## Why we could not do zero-copy ourselves (v2.1.0 source analysis)

1. RAM bdev stores data in process-local heap pages (`new char[1 GiB]`,
   `modules/bdev/src/bdev_runtime.cc:1075`); reads memcpy out
   (`ReadFromRam`, `bdev_runtime.cc:1181`). The pages are NOT in the shm
   segments, so `ShmPtr` cannot address them; `ram_pages_`/`GetRamPage`
   are private (`bdev_runtime.h:398-401,504-506`).
2. The blob→placement query is disabled: `GetBlobInfo`'s block fields are
   commented out of population (`core/src/core_runtime.cc:3283-3292`) and
   serialization (`core_tasks.h:1960`) — no caller can resolve a blob to
   (target, offset) today, even in-process.
3. Nothing pins data: `ReorganizeBlob` (`core_runtime.cc:1240-1346`),
   overwrites, and eviction free blocks for immediate reuse, and `GetBlob`
   holds no lock during I/O (`core_runtime.cc:1211`) — scanning tier
   memory directly would race with reclamation (use-after-free).
4. The file tier uses async pread-style I/O, never mmap
   (`bdev_runtime.cc:898-900`) — file-tier bytes have no in-process
   address at all.

The encouraging part: RAM-tier block addresses are already **stable for
the block's lifetime**, and blocks are contiguous within a 1 GiB page.
The hard infrastructure exists; only the access discipline is missing.

## Proposed API (minimal)

1. **bdev**: `GetLocalView(offset, size) -> char*` — non-null only when
   the range lies within one RAM page in the caller's process; null
   otherwise (caller falls back to the copy path).
2. **CTE core**: finish `GetBlobInfo` (re-enable the drafted
   `BlobBlockInfo{target_pool_id_, block_offset_, block_size_}`,
   `core_tasks.h:1886-1915`) or add a runtime-local
   `ResolveBlob(tag, name) -> blocks`.
3. **Pin/unpin** (the one genuinely new mechanism): a per-blob or
   per-block read refcount honored by `FreeBlocks` / `ReorganizeBlob` /
   eviction, so a borrowed view cannot be reclaimed mid-scan.
   `PinBlob -> {views, lease}` + `UnpinBlob(lease)` composes 1+2+3.
4. *(Optional, larger)* allocate RAM pages from the shm allocator so
   views work cross-process too; an mmap mode for the file bdev to let
   the file tier participate.

Expected impact for our workload: removes a per-pass copy equal to the
index size; projected ~3× in-RAM throughput (to mmap-class) while keeping
what the page cache cannot do — explicit tiers, no eviction cliff,
per-blob placement control. Generalizes to any near-data module.

## Secondary items from the same study

- **Backpressure on handler-issued tasks**: issuing ~4096 `AsyncGetBlob`s
  from one handler livelocked the runtime (queue_depth 1024, consumed by
  the same workers). We now window client-side; `Send` yielding under
  queue pressure (or a documented hard rule) would prevent the failure
  mode.
- **Handler-safe CTE client init**: `CLIO_CTE_CLIENT_INIT()` called from
  a handler deadlocks the worker (blocking `Wait()` inside). The working
  pattern — `Client c; c.Init(kCtePoolId);` — deserves documentation in
  the module template / AGENTS.md, or an awaitable init.
- **Batched blob ops**: one task carrying N gets/puts; IVF batches touch
  4k–16k blobs.
- **C++ SDK packaging**: the pip wheel exports all needed symbols but no
  headers/CMake package; we built against a source checkout pinned to the
  wheel tag and vendored zmq/msgpack/cereal headers. A companion dev
  package would make out-of-tree ChiMods turnkey.
- **Client-visible tier telemetry**: per-target occupancy / bytes-moved
  counters (we reconstructed them from /proc/diskstats and directory
  sizes).

## Offer

The `faiss_ivf` ChiMod + benchmark harness is available as a reproducible
test case: coroutine handlers, in-process CTE calls, bulk shm data paths,
out-of-tree build against the wheel, and a benchmark with a 20× win over
mmap out-of-core. Happy to contribute it as an example module and to
validate the pin API against it.
