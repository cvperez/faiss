#!/usr/bin/env bash
# 30_run_bench.sh — timed QPS run of the faiss_ivf ChiMod for one volume.
#
# Per pass:
#   * first --nq of the BigANN queries, decoded bvecs -> float32
#   * k = 10, nprobe = max(1, nlist // 64), 8 scan threads
#   * one batched search per pass; passes = 1 cold + 2 warm
#   * QPS = nq / elapsed; side metrics: majflt, /proc/self/io read_bytes
#
# Prepares a fresh clio_run and ingests the volume into CTE (via
# 20_ingest_cte.sh), evicts the volume's file pages, then drives the
# ChiMod with `bench_ivf_qps --protocol step3`. The ChiMod reads each
# probed list from CTE on demand.
#
# Inputs:
#   $1  volume name (e.g. ondisk_nb50M)
#   Env overrides: IOWARP_WORK_DIR, FAISS_INSTALL, QUERIES,
#                  CHIMOD_INFLIGHTS (concurrent SearchTasks per pass; default 8)
#
# Outputs:
#   results/qps_<volume>.csv   one row per pass, appended
#   results/bench_<volume>_<ts>.log
#
# Example:
#   bash scripts/30_run_bench.sh ondisk_nb50M

set -euo pipefail

VOLUME="${1:?usage: 30_run_bench.sh <volume>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS="$ROOT/results"
IOWARP_WORK_DIR="${IOWARP_WORK_DIR:-/mnt/common/$USER/faiss/slurm_bench/work}"
QUERIES="${QUERIES:-/mnt/common/$USER/faiss/slurm_bench/data/bigann/bigann_query.bvecs}"
FAISS_INSTALL="${FAISS_INSTALL:-$HOME/faiss-install}"

INDEX="$IOWARP_WORK_DIR/$VOLUME.index"
IVFDATA="$IOWARP_WORK_DIR/$VOLUME.ivfdata"
CSV="$RESULTS/qps_$VOLUME.csv"
[ -f "$INDEX" ]   || { echo "ERROR: missing $INDEX" >&2; exit 1; }
[ -f "$QUERIES" ] || { echo "ERROR: missing $QUERIES" >&2; exit 1; }
mkdir -p "$RESULTS"

export OMP_NUM_THREADS=8

IOWARP_PKG="$(python3 -c 'import iowarp_core, os; print(os.path.dirname(iowarp_core.__file__))')"
export PATH="$IOWARP_PKG/bin:$PATH"
export LD_LIBRARY_PATH="$IOWARP_PKG/lib:$FAISS_INSTALL/lib:$FAISS_INSTALL/lib64:$ROOT/build:$ROOT/build/chimod:${LD_LIBRARY_PATH:-}"

drop_page_cache() {
    # Unprivileged posix_fadvise(DONTNEED) evicts the volume's file pages
    # (no sudo — `sudo drop_caches` is not available inside batch jobs).
    python3 - "$INDEX" "$IVFDATA" <<'PY'
import os, sys
for path in sys.argv[1:]:
    if not os.path.exists(path):
        continue
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
PY
    echo "volume pages evicted via fadvise ($(awk '/MemAvailable/{printf "%.1f GiB avail", $2/1024/1024}' /proc/meminfo))"
}

TS="$(date +%Y%m%d_%H%M%S)"
LOG="$RESULTS/bench_${VOLUME}_$TS.log"
echo
echo "########## $VOLUME ##########"

bash "$SCRIPT_DIR/20_ingest_cte.sh" "$VOLUME" "faiss_ivf::$VOLUME"
drop_page_cache

# CHIMOD_INFLIGHTS: space-separated list of concurrent SearchTask counts
# (the CPU-budget knob). Default: 8 = the 8-thread scan budget.
[ -n "${TELEMETRY_PHASE_FILE:-}" ] && echo "bench_chimod" > "$TELEMETRY_PHASE_FILE" || true
for INFLIGHT in ${CHIMOD_INFLIGHTS:-8}; do
    echo "--- inflight=$INFLIGHT"
    "$ROOT/build/bench_ivf_qps" \
        --protocol step3 \
        --index "$INDEX" \
        --queries "$QUERIES" \
        --tag "faiss_ivf::$VOLUME" \
        --label "$VOLUME" \
        --inflight "$INFLIGHT" \
        --csv "$CSV" 2>&1 | tee -a "$LOG"
done

echo "rows appended to $CSV"
