# FAISS IVF search inside the CLIO runtime: design, the search-time copy problem, and the clio-core solution

This document explains what the contribution is, what happens when a search
runs, where the cost is, and the change to clio-core that removes it.

## 1. Overview

FAISS answers nearest-neighbour queries over large sets of vectors. An **IVF**
index partitions the vectors into `nlist` clusters ("inverted lists", or
buckets); a query is compared only against the vectors in its few nearest
buckets (`nprobe` of them), not the whole set.

We host this search **inside the CLIO runtime** as a module (a "ChiMod"), and
store the inverted lists in CLIO's **Context Transfer Engine (CTE)** — one blob
per bucket, tiered across a capped RAM tier and node-local NVMe. CTE manages
placement explicitly: each bucket is placed and migrated on its own, so the
unit CTE moves is exactly the unit a query needs.

## 2. Architecture

```
   client process                      CLIO runtime (clio_run)
 ┌───────────────────────┐           ┌───────────────────────────────┐
 │ queries (nq × d)      │  Search   │  faiss_ivf ChiMod             │
 │  ───────────────────► │  task     │   OpenIndex: metadata + tag   │
 │  ◄─────────────────── │  (shm)    │   Search:  quantize → probe   │
 │ top-k (distances,ids) │           │            → read lists → scan│
 └───────────────────────┘           │                │  reads       │
                                     │                ▼              │
                                     │  CTE  ┌──────────────────────┐ │
                                     │       │ RAM tier │ NVMe tier │ │
                                     │       │  one blob per bucket │ │
                                     │       └──────────────────────┘ │
                                     └───────────────────────────────┘
```

- **Client** submits a `SearchTask` carrying the queries over shared memory and
  receives the top-k distances and ids back.
- **ChiMod** runs the IVF search inside the runtime. It opens the index
  (metadata only) once, and on every search reads the buckets it needs from CTE.
- **CTE** stores the buckets as blobs and tiers them between RAM and NVMe.

## 3. How the inverted lists are stored in CTE

One index is one CTE **tag** (a namespace). Under it:

```
tag  "faiss_ivf::<volume>"
├── blob "sizes"      int64[nlist]        ← how many vectors are in each bucket
├── blob "list/0"     [ codes | ids ]     ← bucket 0
├── blob "list/1"     [ codes | ids ]     ← bucket 1
│   ...                                     (empty buckets have no blob)
└── blob "list/N-1"   [ codes | ids ]     ← bucket N-1
```

- **`sizes`** is the directory: it records each bucket's length, so the module
  knows which buckets exist and how big each blob is without reading any vectors.
- **`list/<i>`** holds bucket `i`'s encoded vectors (`codes`) immediately
  followed by their ids (`ids`), packed contiguously.

One blob per bucket means CTE's placement unit equals the search's access unit.

## 4. What happens during a search

1. The client submits a `SearchTask` (the queries, `k`, `nprobe`) over shared
   memory.
2. The ChiMod **coarse-quantizes**: for each query it finds the `nprobe` nearest
   buckets.
3. It collects the **deduplicated set of probed buckets** across the whole query
   batch, remembering which queries need each bucket.
4. For each probed bucket, the module **reads that bucket's blob from CTE** — it
   needs the codes and ids as ordinary memory to compute distances.
5. It **scans** each bucket (across 8 threads), updating every relevant query's
   running top-k.
6. It returns the top-k distances and ids to the client.

Step 4 is the one that matters.

## 5. The problem: reading a bucket means copying it

CTE's only way to read a blob **copies its bytes** out of the tier into a
separate buffer, and only then can the CPU look at them. The ChiMod runs in the
**same process** as the CTE RAM tier that holds the bytes — but it still cannot
get a pointer to those bytes; it must copy them to scan them.

The consequences follow directly:

- A batch of queries, with `nprobe` set to a useful fraction of `nlist`, probes
  **essentially every bucket** — the union of buckets the batch needs is close
  to the whole index.
- So a single search **copies about the whole index out of CTE**, one bucket at
  a time.
- The search therefore spends **more time waiting on byte copies than computing
  distances** — it is limited by data movement, not by the distance math. (Using
  wider SIMD distance kernels barely changes throughput, which confirms the copy
  is the bottleneck.)

The bucket-per-blob layout is doing its job — placement is explicit, each bucket
moves on its own, and there is no page-cache thrashing. The remaining cost is
purely that **a read is a copy**, even for bytes already sitting in RAM in the
module's own address space.

## 6. The solution: let a co-located module read the RAM tier in place

The fix belongs in **clio-core**: give a module that lives in the runtime's
process a way to read a RAM-tier blob **without copying it**. Conceptually three
capabilities compose to make this safe:

- **Resolve** — ask CTE where a blob's bytes physically are (which tier/target,
  and at what offset). CTE already tracks this internally; it is simply not
  exposed to callers today.
- **View** — obtain a CPU pointer to those bytes when the blob is in the RAM
  tier of the caller's own process (and nothing when it is not, so the caller
  can fall back to a normal copy).
- **Pin** — while the module holds such a view, guarantee the bytes are not
  moved or reclaimed by tiering, eviction, or reorganization, so the scan cannot
  read freed memory.

With these, the ChiMod would scan the RAM tier's **own bytes in place** and the
per-search copy disappears. The scan becomes limited only by memory bandwidth —
the same floor a warm in-memory index has — while keeping everything the page
cache cannot give: explicit tiers, no eviction cliff, and per-bucket placement
control that still lets the index spill to NVMe when it exceeds RAM.

This is feasible rather than speculative: a RAM-tier block's address is already
stable for the block's lifetime, and blocks are contiguous within a page. The
storage already behaves the way the view needs; only the access discipline
(resolve + view + pin) is missing. The concrete, file-level analysis of why this
cannot be done from outside clio-core today, and the minimal surface to add, is
in [UPSTREAM_PROPOSAL_IOWARP.md](UPSTREAM_PROPOSAL_IOWARP.md).
