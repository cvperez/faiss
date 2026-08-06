# Running the FAISS IVF ChiMod on two (or more) nodes

How the multi-node scenario is implemented, layer by layer, with the actual
code. §§1–6 cover the infrastructure and the original query-split fan-out
(2026-08-05 campaign); §7 covers the owner-filtered broadcast that replaced
it as the default (2026-08-06 campaign — data-local scan, ~2× warm QPS per
added node). Measured results are in
[`../benchmarks/results/RESULTS_DEV_ZEROCOPY.md`](../benchmarks/results/RESULTS_DEV_ZEROCOPY.md);
this document explains *how it works*.

## 1. What clio-core gives us

clio-core forms a cluster from a single config key: `networking.hostfile`
(one hostname per line; **the line number is the node id**). Everything else
derives from it automatically:

* every pool is created with **one container per node**
  (`pool_manager.cc`: `num_containers = all_hosts.size()`), and
  `ContainerId == NodeId`;
* each node's CTE container registers **its own local tiers** — with
  `targets.neighborhood: 1` node *i* gets `ram::cte_ram_tier_node<i>` and
  `/mnt/nvme/$USER/cte_tier_node<i>`. Capacity is therefore per-node and
  additive: 2 nodes ⇒ 2 × 30 GB RAM tier + 2 × 120 GB NVMe;
* a blob's **owner node is chosen by hashing the blob name**
  (`HashBlobToContainer(tag_id, blob_name)` → `PoolQuery::DirectHash`), so
  an ingest from any one node spreads the `list/<i>` blobs ~50/50 across
  the cluster with no code changes on the ingest side;
* a `GetBlob`/`PutBlob` whose hash lands on a remote container routes over
  the network transparently (ZMQ; queries/results move as bulk transfers).

Two things clio-core does **not** do: `clio_run` does not self-spawn across
the hostfile (each node's daemon must be started separately), and there is
no startup barrier (nothing may talk to the cluster before every daemon has
bound its port).

One important asymmetry: the **zero-IPC shared-memory read is node-local by
construction** (`ShmBlobRecord.node_id_` — "only local is cacheable"). A
container can fast-path only the blobs its own node owns; peer-owned blobs
always go through the RPC path. This is what makes the query-split fan-out
network-bound (§6) — and what the owner-filtered broadcast exploits by
scanning every list on the node that owns it (§7).

## 2. Config rendering — `benchmarks/scripts/20_ingest_cte.sh`

The committed `config/ares_cte.yaml` stays single-node. At run time, when
the SLURM allocation has more than one node, the rendered copy gets the
hostfile injected into the `networking:` block and SWIM failure-probing
disabled (probes share the 8 task workers; a saturated benchmark pass can
get a node falsely declared dead):

```bash
# --- multi-node (2+ nodes): hostfile + SWIM off ------------------------------
MULTI_NODE=0
if [ -n "${IOWARP_HOSTFILE:-}" ] && [ "${SLURM_JOB_NUM_NODES:-1}" -gt 1 ]; then
    MULTI_NODE=1
    sed -i "/^  port:/a\\  hostfile: \"$IOWARP_HOSTFILE\"" "$RENDERED"
    printf '\nswim:\n  enabled: false\n' >> "$RENDERED"
fi
```

The rendered `networking:` block then reads:

```yaml
networking:
  port: 9413
  hostfile: "/mnt/common/<user>/.../results/hostfile_<jobid>"
  neighborhood_size: 32
swim:
  enabled: false
```

Per-node preparation (fresh tier files, stale-perf cleanup, killing old
daemons) must run **on every node**, so the script wraps those commands in
a helper that srun-broadcasts when multi-node and degrades to plain bash on
one node:

```bash
# Run a command on every node of the allocation (or just locally, 1-node).
on_all_nodes() {
    if [ "$MULTI_NODE" = 1 ]; then
        srun --ntasks-per-node=1 --nodes="$SLURM_JOB_NUM_NODES" --export=ALL bash -c "$*"
    else
        bash -c "$*"
    fi
}

on_all_nodes "mkdir -p /mnt/nvme/$USER/cte_tier; rm -f /mnt/nvme/$USER/cte_tier_node* 2>/dev/null || true"
...
on_all_nodes "pkill -u $USER -f clio_[r]un 2>/dev/null || true"
```

(The `clio_[r]un` bracket pattern is deliberate: the srun wrapper's own
command line contains the pattern text, and a plain `pkill -f clio_run`
would match — and SIGTERM — its own wrapper.)

## 3. Launching one daemon per node

`clio_run` stays in the **foreground of its srun task** (a backgrounded
child would die when the task exits); the srun itself is backgrounded and
the script then polls every host's RPC port — clio-core has no peer
barrier, so ingest must not start before all daemons have bound:

```bash
srun --ntasks-per-node=1 --nodes="$SLURM_JOB_NUM_NODES" --export=ALL \
     --output="$RESULTS/clio_run_${VOLUME}_${TS}_node%n.log" \
     bash -c "export CLIO_RESTART_LOG=/tmp/${USER}_restart_\$(hostname).bin; exec clio_run start" &
CLIO_PID=$!

# Readiness: poll the run2run ROUTER port on every host.
PORT="$(awk '/^  port:/{print $2; exit}' "$RENDERED")"
while read -r h; do
    for i in $(seq 1 60); do
        if timeout 1 bash -c "</dev/tcp/$h/$PORT" 2>/dev/null; then break; fi
        sleep 1
    done
done < "$IOWARP_HOSTFILE"
```

`CLIO_RESTART_LOG` is pointed at node-local `/tmp` because the default
lives in the shared home directory and two daemons would race on one file.
Per-node daemon logs land as `clio_run_<volume>_<ts>_node0.log` /
`_node1.log`.

The hostfile itself is generated by `sbatch_bench_dev.sh` from the SLURM
allocation (line order = node ids):

```bash
if [ "${SLURM_JOB_NUM_NODES:-1}" -gt 1 ]; then
    export IOWARP_HOSTFILE="$ROOT/benchmarks/results/hostfile_${SLURM_JOB_ID}"
    scontrol show hostnames "$SLURM_JOB_NODELIST" > "$IOWARP_HOSTFILE"
fi
```

## 4. Client-side changes — `benchmarks/bench_ivf_qps.cpp`

The first 2-node implementation (2026-08-05) changed only two client
routing decisions; the ChiMod runtime needed **no** multi-node changes for
this path — its per-list CTE fetch already falls back from the node-local
shm fast path to the RPC path, which routes cross-node transparently.
Of the three pieces below, **(a) and (c) apply to both routes and are
still current**; **(b) is the query-split fan-out, now the non-default
`--route split`** — §7's owner-filtered broadcast replaced it after the
measurements in §6 showed it network-bound.

**(a) `OpenIndex` must reach every container.** The index handle
(`ivf_`, `sizes_`, `tag_id_`) is per-container state, and `Search` fails
with rc=1 on a container that never opened. With the old
`PoolQuery::Local()` only node 0's container was opened, so every search
routed to node 1 would fail:

```cpp
// Broadcast: EVERY node's container must open the index (per-container
// state) — with Local, a second node's container would fail every
// search routed to it with rc=1. Single-node this degenerates to the
// one local container.
auto open_fut = chimod_client.AsyncOpenIndex(
        clio::run::PoolQuery::Broadcast(), a.index_path, a.tag);
```

(`OpenIndex` reads only the index *metadata* from the shared filesystem —
`/mnt/common` on Ares — the inverted lists come from CTE.)

**(b) Search sub-batches fan out by container id (`--route split`).** The
bench splits the query batch into `--inflight N` independent SearchTasks;
sub-batch `i` is routed with `DirectHash(i)`, which resolves to container
`i % num_containers` — so `--inflight 2` on two nodes puts exactly one
SearchTask (with its 8 scan threads) on each node. On one node,
`hash % 1 = 0` reproduces the old `Local()` behavior exactly:

```cpp
// DirectHash(i): sub-batch i lands on container i % num_containers.
// Single node this is container 0 (== the old Local behavior);
// multi-node it fans one SearchTask (with its 8 scan threads) out
// to each node's container — use --inflight <num nodes>.
s.fut = client.AsyncSearch(
        clio::run::PoolQuery::DirectHash(static_cast<clio::run::u32>(i)),
        static_cast<clio::run::u32>(s.cnt),
        ...);
```

Per-query results are independent, so the split cannot change any output —
the 2-node runs return the byte-identical `di_hash` of the 1-node runs.

**(c) Stats become cluster totals.** `AsyncStats` is broadcast, and the
task's reply-merge hook **sums** each container's counters instead of
copying the last reply
(`chimod/include/clio_runtime/faiss_ivf/faiss_ivf_tasks.h`):

```cpp
/** Aggregate replica results into this task. Counters are SUMMED across
 *  containers (a broadcast Stats on N nodes must report cluster totals;
 *  Copy would report only the last replica's). */
void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task>& other_base) {
  Task::AggregateOut(other_base);
  auto other = other_base.template Cast<StatsTask>();
  searches_ += other->searches_;
  lists_fetched_ += other->lists_fetched_;
  bytes_fetched_ += other->bytes_fetched_;
  fetch_wait_us_ += other->fetch_wait_us_;
  scan_us_ += other->scan_us_;
}
```

(`AsyncCreate` stays `Local()`: the admin module turns pool creation into a
broadcast itself.)

## 5. How to run

Owner-filtered broadcast (the default route, §7):

```bash
cd contrib/iowarp/benchmarks/scripts
sbatch --wait --nodes=2 --ntasks=2 --ntasks-per-node=1 \
    --export=ALL,FAISS_VOLUME=ondisk_nb50M,CHIMOD_INFLIGHTS=1,VERIFY_N=16 \
    sbatch_bench_dev.sh
```

The historical query-split fan-out for A/B comparison: add
`CHIMOD_ROUTE=split` and set `CHIMOD_INFLIGHTS=<node count>`.

Notes:

* **All three of `--nodes=2 --ntasks=2 --ntasks-per-node=1` are required**
  on the command line: the script's `#SBATCH --ntasks=1` directive
  otherwise constrains SLURM into shrinking the allocation back to one
  node (observed: job 22790 silently ran single-node).
* `CHIMOD_INFLIGHTS` is per-node concurrency on the owner route (every
  broadcast reaches every node) — keep it 1 at any node count to preserve
  the 1-node baseline's one-SearchTask × 8-scan-threads contract. Only on
  the split route must it equal the node count.
* Optional: `RUN_SELFTEST=1` runs `bench_ivf_qps --selftest-chimod` after
  ingest (four legs, including an owner-route broadcast leg, all bitwise
  vs stock FAISS); `--dump-di` artifacts land in `results/dumps/` and two
  runs are compared with `scripts/compare_di.py A.bin B.bin`.
* Everything degrades to the unchanged 1-node behavior when the
  allocation has a single node (verified: jobs 22789 query-split,
  22821/22822/22824 owner route — historical di_hash reproduced exactly).

What a healthy 2-node owner-route run prints (job 22823):

```
=== multi-node allocation: ares-comp-10 ares-comp-11
=== multi-node: 2 nodes (ares-comp-10 ares-comp-11 ), swim off
clio_run up on ares-comp-10:9413
clio_run up on ares-comp-11:9413
[verify] PASS — 16 lists byte-identical          <- includes cross-node reads
--- inflight=1 route=owner
  cold   qps=     99.5  ...  di_hash=32085d09bb65391b  canon=b3e14dd79e32cd77
  warm0  qps=    104.5  ...  di_hash=32085d09bb65391b  canon=b3e14dd79e32cd77
                              ^ same di_hash AND canon as the 1-node run
[stats] searches=3000 lists_read=24462 ...       <- lists_read == 1-node total:
ares-comp-10: 0  total                              each list scanned once
ares-comp-11: 0  total                           <- per-node tier occupancy
```

## 6. What the query-split measurements showed (2026-08-05 campaign)

| Volume | 1-node warm QPS | 2-node warm QPS | Correctness |
|---|---|---|---|
| nb10M (job 22791) | 269–276 | 22.5 | exact 1-node di_hash |
| nb50M (job 22793) | 54 | 4.6 | exact 1-node di_hash |
| nb100M (jobs 22795/22818) | 21 | — | ingest + verify-all pass; search OOMs the peer daemon |

* **Correctness is complete**: cluster formation, hash-spread ingest,
  cross-node verification, and bitwise-identical search results.
* **The query-split fan-out is network-bound by construction.** Each
  sub-batch probes nearly the whole index, so every node reads ~half its
  lists from the peer over RPC (~215 MB/s effective) — and the zero-IPC
  fast path cannot help, being node-local. Combined RAM capacity is real
  (each node holds half the blobs) but combined *bandwidth* is not
  exploitable this way.
* **The scalable design is data-local**: route the *scan* to the node that
  owns each list (split by list ownership, not by query), return partial
  top-k heaps, and merge k results per query. That turns the cross-node
  traffic from ~half the index per pass into `nq × k × 12 bytes`, and lets
  both nodes scan at local-RAM speed. **Implemented — see §7.**
* **Upstream limit at scale**: daemon-side shm segments grow with the
  bytes moved cross-node (remote-put staging and remote-read serving) and
  are never recycled — at nb100M (~24 GB/node) the peer daemon is
  OOM-killed (131→188 × ~140 MB segments; per-node logs
  `clio_run_ondisk_nb100M_*_node1.log`). nb10M/nb50M scales fit. Worth
  raising with the clio-core maintainers together with the #856
  strict-event-resume note.

## 7. Owner-filtered broadcast — the data-local implementation

The design of §6's last bullet, implemented (2026-08-06 campaign; measured
results in `../benchmarks/results/RESULTS_DEV_ZEROCOPY.md` §"2-node,
owner-filtered broadcast"). The query-split path of §4 is preserved as
`bench_ivf_qps --route split`; the new path is `--route owner` (the
default). Warm QPS: nb50M 105.6 on 2 nodes vs 56.1 on 1 (1.88×; the
query-split fan-out managed 4.6); nb100M 52.8 vs 20.7 (2.55×,
super-linear — see "capacity" below; query-split OOM'd the peer daemon).
Every pass reproduced the single-node `di_hash` bit-for-bit.

### Mechanics

**(a) Ownership is computed locally from the placement hash.** clio-core
routes a blob to its owner container with `HashBlobToContainer(tag_id,
blob_name)` → `DirectHash(h)` → `h % num_containers`. That function is
private to the CTE core runtime, but the CTE *cache* module sets the
precedent of replicating its four hash lines in module code
(`cache_runtime.cc IsBlobOwnerLocal`). `OpenIndex` does the same once per
volume, precomputing `list_local_[l]` for every list
(`faiss_ivf_runtime.cc`); it also asserts that the faiss pool (600.0) and
the CTE pool (512.0) have identical container counts — both are created
with one container per node (`ContainerId == NodeId`), which is what makes
`h % N` name the same node in both pools. If core ever changes the hash,
only locality degrades (visible as `bytes_fetched_` blowing up); the
lists still partition exactly-once, so correctness holds. On one node
`h % 1 == 0` — every list is local and the path degenerates to §4's
behavior automatically.

**(b) The search is broadcast; each node scans only what it owns.**
`SearchTask::mode_` — a previously ignored wire slot — carries two bits:
`kSearchModeOwner` (filter to owned lists) and `kSearchModeIP` (merge
direction; the merge runs where the faiss index is not visible). The bench
sends each sub-batch with `PoolQuery::Broadcast()`; the probe-map build in
`Runtime::Search` drops non-owned lists:

```cpp
if (l < 0 || sizes_[l] <= 0) continue;
if (owner_mode && !list_local_[l]) continue;  // peer's replica scans it
```

Every list a node scans is therefore one it owns — every fetch is eligible
for the zero-IPC shm fast path, and the RPC pipeline degenerates to
local-NVMe misses. `--inflight` becomes per-node concurrency (each
broadcast reaches all nodes), so `CHIMOD_INFLIGHTS=1` at any node count.

**(c) Partials return as serialized vectors, NOT via the D/I buffers.**
This is the critical wire-level change. A broadcast task is replicated
with `NewCopyTask` → `Copy()`, which copies the raw `ShmPtr`s — so every
replica subtask on the origin node aliases the client's `distances_out_`/
`labels_out_`. Had `SerializeOut` kept bulk-XFERing D/I, each returning
replica would memcpy over the same client buffer: a last-writer-wins race
(no clio-core broadcast task uses bulk OUT regions; the endorsed shape is
`SemanticSearchTask`'s serialized-vector merge). So in owner mode the
handler exports its sorted per-query partials into new OUT fields and
`SerializeOut` ships those instead (`faiss_ivf_tasks.h`):

```cpp
ar(mode_);                      // wire-carried branch selector
if (SearchOwnerMode(mode_)) ar(part_d_, part_i_);   // no bulk on D/I
else { ar.bulk(distances_out_, ...); ar.bulk(labels_out_, ...); }
```

The handler fills `part_*` only when `task->IsRemote()` — true iff the
task arrived through the network receive path, which on a multi-node
broadcast is every replica (loopback included: `SendIn` has no local
shortcut). On a single node the broadcast short-circuits (`IsTaskLocal`),
the origin runs the handler directly with `IsRemote() == false`, and the
handler writes the client's shm buffers exactly as before — the
single-node path is byte-identical to §4's.

**(d) The merge lives in `SearchTask::AggregateOut`** — the same hook
`StatsTask` uses to sum counters, called once per replica, serially, on
the origin node. First replica's partials are adopted wholesale;
each subsequent replica is merged per query (two-pointer merge of two
sorted k-lists, keep best k) under a strict total order — distance, then
id — so the result is independent of replica arrival order. Sentinel
slots (`FLT_MAX`/-1) sort last and fall out naturally. A failed replica's
non-zero rc propagates through `Task::AggregateOut` and fails the search.
The bench harvests `part_*` from the completed future when present
(multi-node) and falls back to the shm buffers when empty (single node /
legacy route).

### What the campaign showed

| Volume | 1-node warm QPS | 2-node warm QPS | Cross-check |
|---|---|---|---|
| nb50M (22822/22823) | 55.5–56.1 | 104.5–105.6 | bitwise di_hash; compare_di 500/500 exact |
| nb100M (22824/22825) | 20.4–20.7 | 52.5–52.8 | bitwise di_hash; compare_di 500/500 exact |

* `lists_read` equals the 1-node total at both sizes (24 462 / 48 867):
  each list is scanned exactly once across the cluster — the hash filter
  is a true partition. Cluster read_wait: 7.2 s vs 615 s for query-split
  (nb50M). (`searches` doubles vs 1-node — every node's replica processes
  the full query batch, and broadcast Stats sums per-container counters.)
* **nb100M is super-linear (2.55×) because 2 nodes add capacity, not just
  bandwidth**: 1-node spills 19 GB to NVMe; on 2 nodes the 49 GB volume is
  fully RAM-resident (NVMe occupancy 0 on both nodes). The peer-daemon
  OOM of jobs 22795/22818 is gone with the cross-node byte flow that fed
  it (per pass: ~256 KB of queries out + ~60 KB of partials back per node,
  instead of ~half the index).
* Correctness tooling: `bench_ivf_qps` now prints a canonical
  order-independent hash (`canon=`) next to `di_hash`, dumps raw (D, I)
  per pass with `--dump-di`, and `scripts/compare_di.py` classifies any
  1-node-vs-2-node differences (exact / order-only / k-boundary-tie /
  mismatch). In this campaign no ties were hit: all passes bitwise exact.
  The selftest gained an owner-route broadcast leg
  (`[selftest chimod owner2]`), bitwise-identical to stock FAISS.

### Limits

* Broadcast resolves to one query per container only for
  `N ≤ neighborhood_size` (default 32) nodes; beyond that clio-core
  delivers multi-container ranges to only the first container of each
  range — a latent cliff far above this campaign's scale.
* A non-colocated (TCP) client on a single-node cluster would get empty
  `part_*` and unwritten local buffers in owner mode; the bench is always
  a colocated shm client. Multi-node owner mode works for any client
  (results travel as serialized vectors).
* The replicated placement hash depends on libstdc++'s
  `std::hash<std::string>` — stable here (daemon and ChiMod are built with
  the same toolchain), fragile as a cross-toolchain contract; a public
  `HashBlobToContainer` in clio-core would remove the duplication (worth
  adding to the upstream list).
