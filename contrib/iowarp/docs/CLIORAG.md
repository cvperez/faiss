# ClioRAG — FAISS IVF Vector Search on IOWarp's Content Transfer Engine

**The definitive implementation report.** ClioRAG is a server-side vector
search engine: a ChiMod (`clio_faiss_ivf`) that hosts FAISS IVF search and
incremental writes *inside* the IOWarp clio-core runtime, with the inverted
lists stored as Content Transfer Engine (CTE) blobs tiered across RAM and
NVMe on one or many nodes. It replaces the classical mmap-backed on-disk
IVF deployment: instead of the OS page cache deciding what stays in memory,
CTE's tiering engine does — and the search executes where the data lives.

This document supersedes the narrative in `report.tex` (written against the
v2.1.0 read-only, single-node prototype). Everything below describes the
current system: clio-core dev `8715591a`, zero-IPC reads, owner-filtered
N-node broadcast search, the `kAdd` write path, multi-shard ingest, and the
size × topology performance study that mirrors the mmap
`ondisk_step4` experiment cell-for-cell.

---

## 1. Motivation

FAISS's `OnDiskInvertedLists` keeps IVF inverted lists in a memory-mapped
file, so an index can exceed RAM. The mmap size × topology study
(`ondisk_step4`, BigANN 44–500 M vectors on 1/2/4/8 nodes) quantified the
failure mode: throughput collapses at a **sharp knee** exactly where the
per-node working set crosses the per-node RAM boundary (~92.3 M vectors on
Ares's 46.6 GiB nodes) — 123 → 8.9 → 0.8 QPS at 0.5×/1×/2× RAM — because
the page cache thrashes and the SSD saturates at a fixed read ceiling while
per-query I/O keeps growing.

ClioRAG's thesis: an explicit tiering engine with an in-runtime search
module beats a page cache at every point of that curve, because

1. **placement is deliberate** (hot lists pinned in a RAM tier, spill to
   NVMe, never evicted by unrelated memory pressure),
2. **reads are zero-copy-adjacent** (RAM-tier blobs are read through shared
   memory with no per-list RPC), and
3. **search is data-local** (on N nodes, each node scans exactly the lists
   it owns; only queries and per-query top-k partials cross the network).

## 2. Architecture

```
 client (bench / application)
   │  AsyncOpenIndex ─ broadcast: every node binds index metadata + tag
   │  AsyncSearch ──── broadcast (owner mode): all nodes, disjoint lists
   │  AsyncAdd ─────── DirectHash(owner): exactly the owning node
   │  AsyncStats ───── broadcast: cluster-summed counters + diskstats
   ▼
 clio-core runtime, one container per node (ContainerId == NodeId)
   ┌──────────────────────────────────────────────────────────────┐
   │ clio_faiss_ivf (ClioRAG ChiMod)                              │
   │   quantizer (replicated, from the index skeleton on shared FS)│
   │   sizes_[nlist]   per-list sizes (from the "sizes" blob)      │
   │   list_local_[l]  ownership map (placement hash % N)          │
   │   slab cache      reusable list buffers (allocator gotcha #3) │
   └──────────────┬───────────────────────────────────────────────┘
                  │ TryReadBlobShm (zero-IPC, RAM tier) / AsyncGetBlob
   ┌──────────────▼───────────────────────────────────────────────┐
   │ CTE core: tag faiss_ivf::<volume>                            │
   │   blob "list/<l>" = codes ‖ ids   (one per non-empty list)   │
   │   blob "sizes"    = int64[nlist]  (written last = commit)    │
   │   tiers: ram (30 GB, score 1.0) + NVMe file (120 GB, 0.3)    │
   └──────────────────────────────────────────────────────────────┘
```

### 2.1 Storage layout

One CTE tag per index volume. Each non-empty inverted list `l` is one blob
`list/<l>` holding the list's codes (`size*code_size` bytes, raw float32
vectors for IVF-Flat) immediately followed by its ids (`size` × int64),
always compact. A final `"sizes"` blob (`int64[nlist]`) is written last —
its presence signals a completed ingest, and `OpenIndex` derives `ntotal`
and per-list sizes from it rather than trusting the index file.

The coarse quantizer is **not** ingested. Every node reads the small index
skeleton (quantizer + metadata) from the shared filesystem at `OpenIndex`
and runs coarse quantization locally; only inverted-list payloads live in
CTE.

### 2.2 Blob ownership and the shared placement hash

CTE routes a blob to container `HashBlobToContainer(tag, name) % N`. That
hash is private to the CTE core, so ClioRAG replicates it in ONE shared
function, `ListOwnerContainer(tag_major, tag_minor, l, N)`
(`faiss_ivf_tasks.h`), used by

- `Runtime::OpenIndex` to precompute `list_local_[l]` (the owner-filter
  map), and
- the write client to route `AddTask`s to exactly the owning container.

Because both sides call the same function, search filtering and write
routing can never drift apart; a misrouted add is rejected loudly (rc=8)
rather than silently misplaced. If clio-core ever changes its hash, only
*locality* degrades — lists still partition exactly-once across nodes.

## 3. The ChiMod interface

| method | id | routing | purpose |
|---|---|---|---|
| `Create` | 0 | Local | container bootstrap (admin `GetOrCreatePool`) |
| `Monitor` | 9 | any | msgpack counters per container |
| `OpenIndex` | 10 | **Broadcast** | bind index skeleton + CTE tag; build `sizes_`, `list_local_`; overwrite `ntotal` from the `"sizes"` blob |
| `Search` | 11 | Broadcast (owner mode) / any (legacy) | batched k-NN |
| `Stats` | 12 | Broadcast | cluster-summed counters + node `/proc/diskstats` |
| `Add` | 13 | **DirectHash(owner)** | append vectors to owned lists (RMW) |

### 3.1 Search

`SearchTask` carries `nq, k, nprobe, d`, a `mode_` bit-field, the query
buffer, and client-preallocated result buffers. The handler:

1. coarse-quantizes the batch locally (`quantizer->search`),
2. builds the set of unique probed non-empty lists — **in owner mode
   (`kSearchModeOwner`) it drops every list whose `list_local_[l]` is 0**
   (the peer's replica scans it instead: the probe set partitions
   exactly-once across nodes),
3. fetches each list: a zero-IPC shared-memory burst copy
   (`TryReadBlobShm`, 16-wide across the 8 scan threads) for RAM-resident
   blobs, an `AsyncGetBlob` RPC pipeline (≤64 in flight) for the misses
   (NVMe tier / remote), and
4. scans each fetched list across per-thread `InvertedListScanner`s into
   per-query heaps living directly in the output buffers, then reorders.

**Multi-node result path.** Broadcast replicas cannot write the client's
buffers (the replicated task aliases the origin's ShmPtrs — last-writer-
wins). In owner mode a remote replica instead serializes its per-query
sorted partial top-k (`part_d_/part_i_`) back on the wire, and
`SearchTask::AggregateOut` two-pointer-merges replicas per query under a
strict (distance, id) total order — the result is independent of arrival
order, and equal-distance ties break deterministically toward smaller id.
A single-node broadcast short-circuits: the handler writes the client
buffers directly, bit-for-bit the legacy behavior.

The `kSearchModeIP` bit carries the metric direction on the wire because
the merge runs where no index is available; a mode/metric mismatch is
rejected (rc=7).

### 3.2 Add — the write path

`AddTask` carries a batch grouped by target list: `list_ids_` (ascending),
`list_offs_` (prefix offsets), and two bulk payloads (`codes_`, `ids_`).
The **client** does the global work — coarse-assign the batch with its
local quantizer, group by `ListOwnerContainer`, chop each container's
share into ≤256 MB tasks — and sends each share to exactly its owner
(`DirectHash(owner)`), concurrently across containers but strictly
sequentially within one (the read-modify-write on a list must observe the
previous append).

The **runtime** validates every segment against `list_local_` (rc=8 on
misroute — the tripwire that caught a mono-routed client during
validation), then per list composes
`codes_old ‖ codes_new ‖ ids_old ‖ ids_new` in a reusable slab (fetch old
blob, `memmove` the old ids right, splice the new codes/ids) and
`PutBlob`s the grown list — preserving the single `codes‖ids` split the
scanner parses. `sizes_[l]` and `ntotal` update in memory.

Two deliberate design points:

- **The global `"sizes"` blob is not rewritten by `Add`** — concurrent
  per-container RMW of one blob would race. Each container's in-memory
  `sizes_` is current for its owned lists (all that owner-mode search
  reads), and the writing client — which computed every delta — rewrites
  `"sizes"` once after its write phase, making the volume re-openable.
- **Add/Search exclusion is a client contract, enforced loudly.** The
  mixed-workload bench drains in-flight searches before adding (exactly
  the mmap study's write-window semantics). The runtime keeps an
  `adding_` tripwire: a Search arriving mid-add gets rc=9 instead of a
  torn scan.

Correctness gate: the selftest appends 10 k vectors through the AddTask
path and through stock `faiss::IndexIVFFlat::add`, then requires the
post-add search results **bitwise identical** — which holds at 1, 4, and
8 nodes because per-list append order equals input order on both sides.

### 3.3 Stats and per-pass disk metrics

`StatsTask` reports the ChiMod counters (searches, lists/bytes fetched,
fetch-wait and scan time, add counters) **plus each node's cumulative
`/proc/diskstats` counters for the NVMe tier device** (sectors read/
written, io_ticks) and a `containers_=1` field. Broadcast aggregation sums
everything, so one Stats round-trip yields cluster totals and `N`. The
bench brackets every pass with two Stats calls and computes the mmap
study's exact metrics:

```
disk_active_pct = Δio_ticks_ms / (elapsed_s · 1000 · N) · 100   (per-node mean)
disk_read_mbps  = Δrd_sectors · 512 / elapsed_s / 10⁶           (cluster sum)
```

## 4. Ingest

`ivf_to_iowarp` reads populated FAISS index file(s) via the stock
`OnDiskInvertedLists` mmap hook and puts one blob per non-empty list plus
`"sizes"`, through a ring of ≤8 reusable max-size buffers.

**Multi-shard ingest** (`--shards s0.index s1.index … --tag T`) ingests N
per-rank shard files — which share a trained quantizer and globally-unique
ids — as ONE volume: per list, every shard's codes back-to-back, then every
shard's ids, preserving the single `codes‖ids` split. Because rank slices
are contiguous ascending id ranges, the concatenated per-list order equals
a merged index's order **exactly**: the validation gate demands (and gets)
bitwise-identical search results (`di_hash 9f02bf5dd69e36c5` on every pass,
single-shard vs two-shard, nb44M). This is what lets the performance study
reuse the mmap study's 3.9 TB of existing shards without building ~1.1 TB
of merged copies.

`--verify N|all` reads blobs back through the RPC `GetBlob` path and
byte-compares against the source mmaps — `all` is the write-path
correctness gate (on N nodes most reads are cross-node, making it a
cluster-correctness gate too). `--expect-ntotal` guards against
historical shard files grown in place by earlier experiments.

## 5. Multi-node operation

Cluster formation is config-only: a `networking.hostfile` (line order =
node id), SWIM probing off for benchmarks, one `clio_run` per node started
by the launcher (clio-core does not self-spawn) with a TCP readiness poll
(no built-in barrier). One container per node in both the CTE pool and the
ClioRAG pool; `OpenIndex` asserts the two pools agree on `N` (rc=7), the
invariant that makes `hash % N` name the same node in both.

Nothing in the code is specific to N=2: the same binaries and scripts ran
N ∈ {1, 2, 4, 8} unchanged. The known ceiling is clio-core's
`neighborhood_size` (32), beyond which broadcast stops resolving to one
query per container.

Cross-node traffic per search in owner mode: the query batch to each node
(≈256 KB at nq=500) and one sorted partial top-k back (≈60 KB) — versus
the query-split alternative shipping ~half the *index* per pass over RPC
(measured 10–12× slower at 2 nodes; abandoned).

## 6. Runtime lessons (clio-core dev)

Hard-won semantics any code against clio-core dev must respect
(chronology and analysis in `RESULTS_DEV_ZEROCOPY.md` and
`ISSUES_RESOLVED_DEV_ANALYSIS.md`):

1. **Zero-IPC reads serialize async pipelines.** Dev's `AsyncGetBlob`
   fast path memcpys inline on the calling coroutine. ClioRAG burst-copies
   RAM-resident lists across the scan threads (`TryReadBlobShm`) and keeps
   the RPC pipeline only for misses.
2. **Futures pin tasks.** A dev `Future` owns its task via `shared_ptr`;
   holding thousands pins shm. Drop futures as soon as consumed.
3. **The allocator never recycles multi-MB buffers.** Freed large buffers
   grow the segment pool forever. Every hot path uses preallocated
   reusable pools: the ingest PutSlot ring, the search slab pool, and — 
   after the mixed-workload bench OOM-killed a daemon in 60 s by
   allocating 64 slabs *per single-query search* — a **runtime-level slab
   cache**: retired slabs go on a mutex-guarded free stack shared by
   Search and Add, re-keyed upward when a bigger volume/grown list needs
   larger slabs, freed on Destroy.
4. **Never yield-park a handler coroutine.** The dev event-resume guard
   resumes a fiber only from the exact future it awaits; directly
   `co_await` the oldest pending future instead (the 8-hour-hang lesson).
5. **BLAS matters off the runtime too.** The system `libblas.so.3` on Ares
   is Netlib reference BLAS; a 2.5 M-vector coarse assignment against
   8192 centroids (~5.2 Tflop) simply never finishes inside a write cycle.
   The harness interposes the OpenBLAS shipped in the `faiss-cpu` wheel
   (`LD_PRELOAD`), taking assignment to ~136 s.

## 7. The performance study (size × topology)

`benchmarks/results/performance_study/` reproduces the mmap
`ondisk_step4` study cell-for-cell with ClioRAG as the backend — same
dataset (BigANN, `IVF{nlist},Flat`, 520 B/vec), same sizes
(44–500 M vectors), same topologies (n1: →185 M, n2: →370 M, n4/n8:
→500 M), same protocol (500 queries — 200 when disk-bound — k=10,
nprobe=nlist/64, cold + 2 warm passes, 8 scan threads), same metrics and
JSON schema, same eight figure families (`perf_study_plot.py` is the mmap
plot script, prefix-parameterized).

- **Exp 1 (read-only):** one batched owner-broadcast search per pass;
  per-pass QPS + Stats-bracketed disk metrics →
  `perf_study_exp1_nb{N}M_n{K}.json`.
- **Exp 2 (mixed 80/20):** `bench_ivf_mixed` keeps 4 single-query
  searches in flight from one event loop (closed-loop concurrency 4, the
  mmap study's reader semantics without multi-threaded-client
  assumptions); a writer thread streams BigANN from offset `NB_M`,
  assigns and owner-groups batches off-thread; each write window drains
  the readers, sends the AddTasks, and paces to the 20 % write fraction
  (`sleep = t_add·(1−f)/f`). 10-s QPS windows tagged `in_write`; headline
  = mean QPS outside write windows. Top-of-corpus cells run pure-read
  (no headroom), like the mmap 500 M cells.
- **Per node**: RAM tier 30 GB + NVMe file tier 120 GB, `max_bw` DPE. The
  CTE RAM boundary is therefore ~57.7 M vec/node, below the mmap study's
  92.3 M page-cache boundary (annotated on shared figures).
- **Orchestration**: one Slurm allocation per cell (`run_perf_study_cell.sh`
  ingests the mmap study's shards multi-shard, runs exp1 then exp2), a
  strictly serial dependency chain (`submit_perf_study.sh`) — validation
  gates linked `afterok` (a red gate cancels the whole campaign), grid
  cells `afterany` between themselves *and* `afterok` on the gate check —
  ending in the plot job. Idempotent: complete JSONs are skipped, so
  resubmission resumes a partial campaign.

### 7.1 Validation gates (all green before the grid runs)

| gate | what it proves | result |
|---|---|---|
| selftest, n1 | search + split + owner + **add** bitwise vs stock FAISS | PASS |
| verify-all, n1 | every blob byte-identical via RPC read-back | PASS (8192/8192) |
| multi-shard equivalence | 2-shard ingest ≡ merged index | di_hash bitwise identical, every pass |
| n4 cluster | first 4-node run: verify-all + full selftest | PASS |
| n8 cluster | first 8-node run: verify-all + full selftest | PASS |
| cross-topology | n4, n8 exp1 results ≡ n1 | canon hash identical |
| mixed short-run | write fraction ≈ 0.2, windows tagged, no rc=8/9 | (in progress) |

### 7.2 Results

Populated by the campaign as cells complete; figures land in
`benchmarks/results/performance_study/` as `perf_study_exp{1,2}_panel_{a,b}`,
`perf_study_active_time_*`, `perf_study_disk_throughput_*`. Raw per-cell
records: `perf_study_exp{1,2}_nb{N}M_n{K}.json`. First data points
(validation cells, nb44M): n1 owner-route 61.6 QPS cold / ~62 warm at
nprobe 128 with **zero disk activity** (fully RAM-tier-resident) and ~24
QPS single-query closed-loop at concurrency 4; per-query p50 ≈ 150 ms.

## 8. Relationship to the earlier campaigns

The 2026-08-05/06 campaigns (`RESULTS_DEV_ZEROCOPY.md`) measured the same
engine on pre-built single volumes (nb10M–nb178M) against the v2.1.0
baseline and mmap: ~2× the v2.1.0 ChiMod at every size, cold-QPS wins over
mmap at every size, and 4–24× warm wins out-of-core, with the 2-node
owner route scaling 1.9–2.6×. This study extends that evidence to a
systematic size × topology grid with a write workload — the regime where
the mmap deployment's knee and global write stalls were documented.

## 9. Limitations and future work

- **Broadcast beyond `neighborhood_size` (32)** needs clio-core's range
  splitting fixed upstream before >32-node clusters.
- **`Add` durability** is `"sizes"`-rewrite-on-commit by the writing
  client; a crash mid-write-phase leaves grown list blobs with a stale
  `"sizes"` (searches stay consistent per container; a re-open sees the
  old ntotal). An upstream `PutBlobVectored`-based sizes journal would
  close this.
- **Replicated placement hash**: a public `HashBlobToContainer` in
  clio-core would remove ClioRAG's copy (locality-only risk today).
- **OpenIndex rc masking**: a failed remote replica's return code can be
  masked by a later successful one under broadcast aggregation; failures
  currently surface at first search (rc=1) instead of at open.
- **Zero-copy scan**: the RAM-tier burst copy still copies each list once
  per search. The proposed view/pin read API
  (`UPSTREAM_PROPOSAL_IOWARP.md`) would let the scanner run over the
  tier's own bytes — the measured ~2× scan-time headroom at RAM-resident
  sizes.
