#!/bin/bash
#SBATCH --job-name=perf_study
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --partition=compute
# --mem=0 = whole-node memory. Ares enforces ConstrainRAMSpace=yes: with no
# memory request the job cgroup caps at ~34.5 GB and the cgroup OOM killer
# silently takes clio_run mid-mixed-stage (daemon tier 25-30 GB + slab
# caches + the bench's ~4 GB writer buffers cross the cap) while the node
# still has free RAM — observed twice at nb44M exp2 before this line.
#SBATCH --output=/mnt/common/cvazquezperezdelacru/IOWARP/faiss/contrib/iowarp/benchmarks/results/logs/%j_perf.out

# run_perf_study_cell.sh — one cell of the CTE performance study
# (size x topology; the mmap ondisk_step4 study re-run with the
# clio_faiss_ivf ChiMod as the backend).
#
# One allocation = one cell: start an N-node clio-core dev cluster, ingest
# the size's step-4 per-rank mmap shards as ONE CTE volume (multi-shard
# ivf_to_iowarp — lists hash-distribute across the nodes), then run
# Exp 1 (read-only passes -> perf_study_exp1_nb{N}M_n{K}.json) and/or
# Exp 2 (mixed 80/20     -> perf_study_exp2_nb{N}M_n{K}.json).
#
# Submit with the topology on the command line (node count = cell column):
#   sbatch -N4 --export=ALL,NB_M=185 run_perf_study_cell.sh
#
# Inputs (via --export=ALL,...):
#   NB_M              total DB size in millions of vectors (required)
#   RUN_EXP           "1", "2" or "12" (default 12)
#   MIXED_DURATION_S  exp2 mixed-stage seconds (default 3600)
#   VERIFY_N          ingest verify: N lists or "all" (default 16)
#   RUN_SELFTEST      1 = bench_ivf_qps --selftest-chimod after ingest
#   BUILD_FIRST       1 = rebuild $BUILD_DIR before anything else
#   SHARD_SET         force which topology's shard files feed the ingest
#                     (e.g. SHARD_SET=2 ingests ..._n2_rank{0,1}); default:
#                     prefer n$SLURM_JOB_NUM_NODES, fall back to any clean set
#   OUT_DIR           JSON output dir (default results/performance_study)
#   JSON_SUFFIX       appended to the JSON stem (validation runs)
#   CTE_RAM_TIER_GB   RAM-tier cap per node (default 30)

set -euo pipefail

ROOT="${IOWARP_CONTRIB_ROOT:-/mnt/common/cvazquezperezdelacru/IOWARP/faiss/contrib/iowarp}"
NB_M="${NB_M:?export NB_M=<millions of vectors>}"
N_NODES="${SLURM_JOB_NUM_NODES:-1}"
RUN_EXP="${RUN_EXP:-12}"
MIXED_DURATION_S="${MIXED_DURATION_S:-3600}"
export VERIFY_N="${VERIFY_N:-16}"
OUT_DIR="${OUT_DIR:-$ROOT/benchmarks/results/performance_study}"
JSON_SUFFIX="${JSON_SUFFIX:-}"

SLURM_BENCH="/mnt/common/$USER/faiss/slurm_bench"
WORK="${IOWARP_WORK_DIR:-$SLURM_BENCH/work}"
QUERIES="$SLURM_BENCH/data/bigann/bigann_query.bvecs"
BIGANN="$SLURM_BENCH/data/bigann/bigann_base.bvecs"

VOL="perf_nb${NB_M}M_n${N_NODES}"
TAG="faiss_ivf::$VOL"
EXP1_JSON="$OUT_DIR/perf_study_exp1_nb${NB_M}M_n${N_NODES}${JSON_SUFFIX}.json"
EXP2_JSON="$OUT_DIR/perf_study_exp2_nb${NB_M}M_n${N_NODES}${JSON_SUFFIX}.json"
mkdir -p "$OUT_DIR" "$ROOT/benchmarks/results/logs"

echo "=== perf-study cell: nb${NB_M}M x n${N_NODES}  exp=$RUN_EXP  job=${SLURM_JOB_ID:-?}"
echo "=== host=$(hostname)  started=$(date)"
# Confirm the cgroup RAM cap this job actually got (see --mem=0 above).
awk '{print "=== my cgroup:", $0}' /proc/self/cgroup | head -3
find /sys/fs/cgroup -maxdepth 5 \( -name memory.max -o -name memory.limit_in_bytes \) \
        -path "*job_${SLURM_JOB_ID:-0}*" 2>/dev/null | while read -r cg; do
    echo "=== cgroup mem limit: $(cat "$cg" 2>/dev/null) ($cg)"
done

# --- capacity guard: bytes/node must fit RAM tier + NVMe file tier ------------
python3 - "$NB_M" "$N_NODES" <<'EOF'
import sys
nb_m, n = int(sys.argv[1]), int(sys.argv[2])
per_node_gb = nb_m * 1e6 * 520 / n / 1e9
cap_gb = 145  # 30 RAM tier + 120 NVMe, minus headroom
if per_node_gb > cap_gb:
    sys.exit(f"cell nb{nb_m}M/n{n}: {per_node_gb:.0f} GB/node > {cap_gb} GB cap")
print(f"capacity OK: {per_node_gb:.1f} GB/node")
EOF

# --- idempotency: skip experiments whose JSON already carries metrics ---------
exp_done() {  # exp_done <1|2> <json> -> exit 0 if complete
    python3 - "$1" "$2" <<'EOF'
import json, sys
exp, path = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(path))
except Exception:
    sys.exit(1)
if exp == "1":
    warm = [p for p in d.get("passes") or []
            if p.get("label", "").startswith("warm")
            and p.get("qps") is not None
            and p.get("disk_active_pct") is not None]
    sys.exit(0 if len(warm) >= 2 else 1)
series = d.get("qps_series") or []
sys.exit(0 if len(series) >= 2 and d.get("mixed_disk_active") else 1)
EOF
}
DO_EXP1=0; DO_EXP2=0
case "$RUN_EXP" in *1*) DO_EXP1=1;; esac
case "$RUN_EXP" in *2*) DO_EXP2=1;; esac
if [ "$DO_EXP1" = 1 ] && exp_done 1 "$EXP1_JSON"; then
    echo "=== exp1 already complete ($EXP1_JSON) — skipping"
    DO_EXP1=0
fi
if [ "$DO_EXP2" = 1 ] && exp_done 2 "$EXP2_JSON"; then
    echo "=== exp2 already complete ($EXP2_JSON) — skipping"
    DO_EXP2=0
fi
if [ "$DO_EXP1" = 0 ] && [ "$DO_EXP2" = 0 ] && [ "${RUN_SELFTEST:-0}" != 1 ]; then
    echo "=== nothing to do"; exit 0
fi

# --- shard-set selection + protocol parameters --------------------------------
# The study's shards live in $WORK as ondisk_step4_nb{N}M_n{K}_rank{r}.index
# (+ .ivfdata), all sharing the per-size trained quantizer (keyed by nlist)
# and globally-unique ids — so ANY complete, clean (sum ntotal == NB) set
# feeds the multi-shard ingest equally well.
SHARD_ENV="$(python3 - "$NB_M" "$WORK" "${SHARD_SET:-${N_NODES}}" <<'EOF'
import math, os, struct, sys
nb_m, work, prefer = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])

def nlist_for(nb_m):
    # scaling_common.nlist_for: next_pow2(ceil(sqrt(nb))) clipped [1024,32768]
    x = int(math.ceil(math.sqrt(nb_m * 1_000_000)))
    val = 1 if x <= 1 else 1 << (x - 1).bit_length()
    return max(1024, min(32768, val))

def ntotal_of(path):
    # write_index_header: fourcc(4) + d(int32) + ntotal(int64)
    with open(path, "rb") as f:
        hdr = f.read(16)
    return struct.unpack("<q", hdr[8:16])[0]

nlist = nlist_for(nb_m)
trained = os.path.join(work, f"ondisk_step4_nb{nb_m}M_nlist{nlist}_trained.index")
if not os.path.exists(trained):
    sys.exit(f"missing trained skeleton {trained}")

expect = nb_m * 1_000_000
order = [prefer] + [k for k in (1, 2, 4, 8) if k != prefer]
for k in order:
    shards = [os.path.join(work, f"ondisk_step4_nb{nb_m}M_n{k}_rank{r}.index")
              for r in range(k)]
    if not all(os.path.exists(s) for s in shards):
        continue
    total = sum(ntotal_of(s) for s in shards)
    if total != expect:
        print(f"# shard set n{k}: ntotal {total} != {expect} — dirty, skipping",
              file=sys.stderr)
        continue
    print(f"INGEST_SHARDS='{ ' '.join(shards) }'")
    print(f"INGEST_EXPECT_NTOTAL={expect}")
    print(f"TRAINED_INDEX='{trained}'")
    print(f"NLIST={nlist}")
    break
else:
    sys.exit(f"no clean shard set found for nb{nb_m}M in {work}")
EOF
)"
eval "$SHARD_ENV"
export INGEST_SHARDS INGEST_EXPECT_NTOTAL
echo "=== shards: $INGEST_SHARDS"
echo "=== trained skeleton: $TRAINED_INDEX (nlist=$NLIST)"

# nq: 200 for disk-bound cells (per-node ratio >= 1.25), 500 otherwise —
# the same rule as the mmap study's round-5 cell script.
NQ="$(awk -v nb="$NB_M" -v n="$N_NODES" 'BEGIN{print (nb/n/92.3 >= 1.25) ? 200 : 500}')"
echo "=== nq=$NQ (per-node ratio $(awk -v nb="$NB_M" -v n="$N_NODES" 'BEGIN{printf "%.2f", nb/n/92.3}'))"

# --- dev provider wiring (same as sbatch_bench_dev.sh) ------------------------
CLIO_DEV_PREFIX="${CLIO_DEV_PREFIX:-$HOME/clio-core-dev-install}"
DEP_PREFIX="${DEP_PREFIX:-$HOME/iowarp-dev-deps}"
export CLIO_BIN_DIR="$CLIO_DEV_PREFIX/bin"
export CLIO_LIB_DIR="$CLIO_DEV_PREFIX/lib"
export IOWARP_EXTRA_LIB="$DEP_PREFIX/usr/lib/x86_64-linux-gnu"
export BUILD_DIR="${BUILD_DIR:-$ROOT/build-dev}"
export FAISS_INSTALL="${FAISS_INSTALL:-$HOME/faiss-install}"
export CLIO_CLIENT_RETRY_TIMEOUT="${CLIO_CLIENT_RETRY_TIMEOUT:-1200}"
export CLIO_WAIT_SERVER="${CLIO_WAIT_SERVER:-120}"
# Tier device the ChiMod's Stats samples from /proc/diskstats (per node).
export FAISS_IVF_DISK_DEV="${FAISS_IVF_DISK_DEV:-nvme0n1}"
export CTE_RAM_TIER_GB="${CTE_RAM_TIER_GB:-30}"
# The bench binaries are invoked directly from THIS shell (selftest, exp1,
# exp2), so the runtime lib path must be set here too — 20_ingest_cte.sh
# sets it only for its own children (same list as its line 69).
export LD_LIBRARY_PATH="$CLIO_LIB_DIR:$FAISS_INSTALL/lib:$FAISS_INSTALL/lib64:$BUILD_DIR:$BUILD_DIR/chimod:$IOWARP_EXTRA_LIB:${LD_LIBRARY_PATH:-}"
# Interpose a REAL BLAS: the system libblas.so.3 is Netlib reference
# (single-threaded ~Gflops), which makes exp2's 2.5M-vector coarse
# assignment (~5.2 Tflop at nlist 8192) take >40 min — the writer never
# delivered a batch. The pip faiss-cpu wheel bundles OpenBLAS; preloading
# it binds libfaiss's sgemm_ there (measured: 136 s/batch, 8 threads).
# Applied uniformly (benches + daemons) so every cell runs the same BLAS.
FAISS_CPU_LIBS="$HOME/.local/lib/python3.10/site-packages/faiss_cpu.libs"
if [ -e "$FAISS_CPU_LIBS/libopenblas-r0-11edc3fa.3.15.so" ]; then
    export LD_LIBRARY_PATH="$FAISS_CPU_LIBS:$LD_LIBRARY_PATH"
    export LD_PRELOAD="$FAISS_CPU_LIBS/libopenblas-r0-11edc3fa.3.15.so${LD_PRELOAD:+:$LD_PRELOAD}"
    export OPENBLAS_NUM_THREADS=8
fi

if [ "${BUILD_FIRST:-0}" = 1 ]; then
    echo "=== rebuilding $BUILD_DIR"
    cmake --build "$BUILD_DIR" -j8
fi
for b in bench_ivf_qps bench_ivf_mixed ivf_to_iowarp chimod/libclio_faiss_ivf_runtime.so; do
    [ -e "$BUILD_DIR/$b" ] || { echo "ERROR: missing $BUILD_DIR/$b — build the contrib first" >&2; exit 1; }
done
git -C "$HOME/clio-core" rev-parse --short HEAD 2>/dev/null | sed 's/^/clio-core dev HEAD: /' || true

# --- multi-node wiring --------------------------------------------------------
if [ "$N_NODES" -gt 1 ]; then
    export IOWARP_HOSTFILE="$ROOT/benchmarks/results/hostfile_${SLURM_JOB_ID}"
    scontrol show hostnames "$SLURM_JOB_NODELIST" > "$IOWARP_HOSTFILE"
    echo "=== nodes: $(tr '\n' ' ' < "$IOWARP_HOSTFILE")"
fi
on_all_nodes() {
    if [ "$N_NODES" -gt 1 ]; then
        srun --ntasks-per-node=1 --nodes="$N_NODES" --export=ALL bash -c "$*"
    else
        bash -c "$*"
    fi
}

# --- telemetry on EVERY node (per-hostname CSVs) ------------------------------
# A long-lived srun step per node (a nohup'd child would be reaped when the
# helper step exits); concurrent steps fit because each uses 1 task/1 CPU
# of the exclusive allocation — same pattern as 20_ingest_cte.sh's clio_run.
export TELEMETRY_PHASE_FILE="$ROOT/benchmarks/results/.phase_${SLURM_JOB_ID:-local}"
echo "setup" > "$TELEMETRY_PHASE_FILE" || true
TELEMETRY_CMD="exec python3 '$ROOT/benchmarks/scripts/telemetry_sampler.py' \
    --out '$OUT_DIR/telemetry_${VOL}_${SLURM_JOB_ID:-local}_'\"\$(hostname)\"'.csv' \
    --phase-file '$TELEMETRY_PHASE_FILE' \
    --interval 5 --nvme-dev '$FAISS_IVF_DISK_DEV' \
    --tier-dir '/mnt/nvme/$USER/cte_tier'"
if [ "$N_NODES" -gt 1 ]; then
    srun --ntasks-per-node=1 --nodes="$N_NODES" --export=ALL \
        bash -c "$TELEMETRY_CMD" &
else
    bash -c "$TELEMETRY_CMD" &
fi
TELEMETRY_PID=$!
trap 'echo done > "$TELEMETRY_PHASE_FILE" 2>/dev/null || true; \
     kill "$TELEMETRY_PID" 2>/dev/null || true; \
     on_all_nodes "pkill -u $USER -f telemetry_[s]ampler 2>/dev/null || true; \
                   pkill -u $USER -f clio_[r]un 2>/dev/null || true; \
                   rm -rf /mnt/nvme/$USER/cte_tier*"' EXIT

# --- NVMe free-space fail-fast ------------------------------------------------
# The file tier's writes return 0 bytes when the node-local NVMe fills
# (other users' data varies per node: 66 GB free on some, 120+ on others) —
# observed as a mid-ingest ModifyExistingData WRITE FAILED + bdev route
# storm at nb185M/n1. Fail early with a clear message instead.
SPILL_GB=$(awk -v nb="$NB_M" -v n="$N_NODES" -v ram="${CTE_RAM_TIER_GB:-30}" \
    'BEGIN{s=nb*520/n/1000-ram; print (s>0)?int(s+6):0}')
if [ "$SPILL_GB" -gt 0 ]; then
    on_all_nodes "free_gb=\$(df -BG /mnt/nvme | awk 'NR==2{gsub(\"G\",\"\",\$4); print \$4}'); \
        echo \"\$(hostname): NVMe free \${free_gb}G, need ${SPILL_GB}G\"; \
        [ \"\$free_gb\" -ge $SPILL_GB ]" || {
        echo "ERROR: insufficient node-local NVMe for ${SPILL_GB}G spill — resubmit excluding low-space nodes" >&2
        exit 1
    }
fi

# --- runtime + ingest (leaves clio_run running) -------------------------------
# DAEMON_NO_SHM=1: make the DAEMON take the RPC read path (the env must be
# in clio_run's environment — Search runs server-side; setting it on the
# bench binary does nothing).
if [ "${DAEMON_NO_SHM:-0}" = 1 ]; then
    export FAISS_IVF_NO_SHM_DIRECT=1
fi
bash "$ROOT/benchmarks/scripts/20_ingest_cte.sh" "$VOL" "$TAG"

# DIAG=1: fine-grained memory forensics — per-process RssAnon/RssShmem/
# VmSwap for the daemon and benches + the job cgroup's real usage/limit,
# every 2 s. Used to pin down the silent daemon deaths during exp2.
if [ "${DIAG:-0}" = 1 ]; then
    (
        # Resolve this job's memory cgroup (v1 "memory:" line or v2 "0::").
        CG1=$(awk -F: '$2=="memory"{print "/sys/fs/cgroup/memory" $3; exit}' /proc/self/cgroup)
        CG2=$(awk -F: '$1=="0"{print "/sys/fs/cgroup" $3; exit}' /proc/self/cgroup)
        for f in "$CG1/memory.limit_in_bytes" "$CG2/memory.max"; do
            [ -r "$f" ] && echo "limit $(cat "$f") ($f)"
        done
        while true; do
            ts=$(date +%s)
            for p in $(pgrep -f "clio_[r]un") $(pgrep -f "bench_ivf_[mq]"); do
                awk -v ts="$ts" -v p="$p" \
                    '/^(VmRSS|RssAnon|RssShmem|VmSwap):/{printf "%s pid=%s %s %s kB\n", ts, p, $1, $2}' \
                    "/proc/$p/status" 2>/dev/null
            done
            # Full process table (pgid included: a group-kill takes the
            # wrapper subshell with the daemon; a plain OOM kill does not).
            ps -u "$USER" -o pid,pgid,rss,stat,comm --no-headers 2>/dev/null \
                | awk -v ts="$ts" '{print ts, "ps", $0}'
            [ -r "$CG1/memory.usage_in_bytes" ] && echo "$ts cgroup_usage $(cat "$CG1/memory.usage_in_bytes")"
            [ -r "$CG2/memory.current" ] && echo "$ts cgroup_usage $(cat "$CG2/memory.current")"
            # oom_kill counters: increments prove a cgroup OOM event.
            for ev in "$CG1/memory.oom_control" "$CG2/memory.events"; do
                [ -r "$ev" ] && awk -v ts="$ts" -v f="$ev" '{print ts, "memev", f":"$0}' "$ev"
            done
            sleep 2
        done
    ) >> "$OUT_DIR/diag_${VOL}_${SLURM_JOB_ID:-local}.log" 2>&1 &
    DIAG_PID=$!
fi

if [ "${RUN_SELFTEST:-0}" = 1 ]; then
    echo "=== selftest (incl. add leg)"
    "$BUILD_DIR/bench_ivf_qps" --selftest-chimod
fi

# --- evict the SOURCE files from the page cache (cold = CTE tiers only) -------
drop_page_cache() {
    python3 - "$@" <<'EOF'
import os, sys
for p in sys.argv[1:]:
    try:
        fd = os.open(p, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
    except OSError as e:
        print(f"fadvise {p}: {e}", file=sys.stderr)
EOF
}
SHARD_IVFDATA=""
for s in $INGEST_SHARDS; do
    SHARD_IVFDATA="$SHARD_IVFDATA ${s%.index}.ivfdata"
done
# shellcheck disable=SC2086
drop_page_cache $INGEST_SHARDS $SHARD_IVFDATA "$TRAINED_INDEX"

export OMP_NUM_THREADS=8

# --- Exp 1: read-only passes --------------------------------------------------
if [ "$DO_EXP1" = 1 ]; then
    echo "bench_chimod" > "$TELEMETRY_PHASE_FILE" || true
    "$BUILD_DIR/bench_ivf_qps" --protocol step3 \
        --index "$TRAINED_INDEX" --tag "$TAG" --queries "$QUERIES" \
        --label "$VOL" --nq "$NQ" --threads 8 --k 10 --passes 3 \
        --route owner --inflight 1 \
        --json-out "$EXP1_JSON" --nb-m "$NB_M" --nodes "$N_NODES" \
        --dump-di "$OUT_DIR/dumps_${VOL}${JSON_SUFFIX}" \
        --csv "$ROOT/benchmarks/results/qps_dev_${VOL}.csv" \
        2>&1 | tee "$OUT_DIR/exp1_${VOL}_${SLURM_JOB_ID:-local}.log"
    echo "idle" > "$TELEMETRY_PHASE_FILE" || true
fi

# --- Exp 2: mixed 80/20 -------------------------------------------------------
if [ "$DO_EXP2" = 1 ]; then
    if [ "$DO_EXP1" = 1 ]; then
        # Fresh runtime + re-ingest for exp2: a daemon that has already
        # served exp1's 3 volume-scale passes wedges its bdev routes during
        # exp2 on spilled volumes (GetBlob rc=1 / RouteLocal rc=4 storms at
        # nb130M, pass ~5-6 of one daemon lifetime — beyond anything the
        # 3-pass campaigns exercised). A fresh runtime also resets
        # accumulated segment growth before the hour-long mixed stage.
        echo "=== restarting runtime for exp2 (fresh daemon + re-ingest)"
        on_all_nodes "pkill -u $USER -f clio_[r]un 2>/dev/null || true"
        sleep 3
        bash "$ROOT/benchmarks/scripts/20_ingest_cte.sh" "$VOL" "$TAG"
    fi
    echo "bench_mixed" > "$TELEMETRY_PHASE_FILE" || true
    # NO_SHM_DIRECT: adds grow blobs (ExtendBlob adds blocks); the zero-IPC
    # read path's cached shm records can describe the pre-growth layout and
    # an in-place read of the grown size runs off the mapped extent —
    # observed as a silent daemon death during post-write searches. The RPC
    # GetBlob path consults live CTE metadata and is safe under growth.
    # Exp1 (read-only) keeps the fast path.
    FAISS_IVF_NO_SHM_DIRECT=1 \
    "$BUILD_DIR/bench_ivf_mixed" \
        --index "$TRAINED_INDEX" --tag "$TAG" --queries "$QUERIES" \
        --bigann "$BIGANN" --label "$VOL" \
        --nq "$NQ" --threads 8 --k 10 --t-read 4 \
        --batch-size 2500000 --write-fraction 0.20 \
        --mixed-duration "$MIXED_DURATION_S" \
        --json-out "$EXP2_JSON" --nb-m "$NB_M" --nodes "$N_NODES" \
        2>&1 | tee "$OUT_DIR/exp2_${VOL}_${SLURM_JOB_ID:-local}.log"
    echo "idle" > "$TELEMETRY_PHASE_FILE" || true
fi

# --- placement + cleanup (trap kills daemons/telemetry, clears tiers) ---------
echo "=== NVMe file-tier occupancy after run ==="
on_all_nodes "echo \"\$(hostname): \$(du -shc /mnt/nvme/$USER/cte_tier* 2>/dev/null | tail -1 || echo 'no NVMe tier files')\""
echo "Finished: $(date)"
