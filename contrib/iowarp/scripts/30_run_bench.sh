#!/usr/bin/env bash
# 30_run_bench.sh — timed QPS runs for one index volume across backends,
# reproducing the step2/step3 baseline protocol.
#
# Protocol (comparability contract — matches how the mmap baseline numbers
# were measured; do not change without re-stating it next to the plots):
#   * first 500 of the 10k BigANN queries, decoded bvecs -> float32
#   * k = 10, nprobe = max(1, nlist // 64), 8 OMP threads
#   * ONE batched search call per pass; passes = 1 cold + 2 warm
#   * cold pass runs right after `sudo drop_caches`
#   * QPS = 500 / elapsed per pass; side metrics: majflt,
#     /proc/self/io read_bytes, CTE bytes fetched (iowarp/chimod)
#   All of the above is implemented by `bench_ivf_qps --protocol step3`;
#   this script provides the drop_caches / fresh-runtime choreography.
#
# Per backend:
#   mmap    — stop any clio_run (page cache gets the whole node), drop
#             caches, run bench (cold + 2 warm in one process).
#   iowarp  — fresh clio_run + re-ingest (calls 20_ingest_cte.sh), drop
#             caches (CTE RAM tier is shm — unaffected; NVMe tier file
#             pages ARE dropped), run bench.
#   chimod  — same as iowarp; bench drives the clio_faiss_ivf module.
#
# Inputs:
#   $1      volume name (e.g. ondisk_nb50M, ondisk_step3_nb178M)
#   $2..$N  backends, any of: mmap iowarp chimod   (default: iowarp)
#   Env overrides: IOWARP_WORK_DIR, FAISS_INSTALL, QUERIES (see 20_ingest_cte.sh)
#
# Outputs:
#   results/qps_<volume>.csv            one row per (backend, pass), appended
#   results/bench_<volume>_<backend>_<ts>.log
#   results/meminfo_<volume>_<backend>_<ts>.txt
#
# Example:
#   bash scripts/30_run_bench.sh ondisk_nb50M iowarp
#   bash scripts/30_run_bench.sh ondisk_step3_nb178M mmap iowarp chimod

set -euo pipefail

VOLUME="${1:?usage: 30_run_bench.sh <volume> [backend...]}"
shift
BACKENDS=("${@:-iowarp}")

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

# Comparability: baseline ran with 8 read threads.
export OMP_NUM_THREADS=8

IOWARP_PKG="$(python3 -c 'import iowarp_core, os; print(os.path.dirname(iowarp_core.__file__))')"
export PATH="$IOWARP_PKG/bin:$PATH"
export LD_LIBRARY_PATH="$IOWARP_PKG/lib:$FAISS_INSTALL/lib:$FAISS_INSTALL/lib64:$ROOT/build:$ROOT/build/chimod:${LD_LIBRARY_PATH:-}"

drop_page_cache() {
    # Same mechanism as the step2/step3 baseline (ondisk_step2_search.py:153):
    # unprivileged posix_fadvise(DONTNEED) evicts the volume's file pages.
    # No sudo — `sudo drop_caches` is not available inside batch jobs.
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

for BACKEND in "${BACKENDS[@]}"; do
    case "$BACKEND" in mmap|iowarp|chimod) ;; *)
        echo "ERROR: unknown backend '$BACKEND' (want mmap|iowarp|chimod)" >&2; exit 1;;
    esac
done

for BACKEND in "${BACKENDS[@]}"; do
    TS="$(date +%Y%m%d_%H%M%S)"
    LOG="$RESULTS/bench_${VOLUME}_${BACKEND}_$TS.log"
    echo
    echo "########## $VOLUME / $BACKEND  ($(date)) ##########"

    if [ "$BACKEND" = "mmap" ]; then
        # Fairness: give the page cache the whole node, as the baseline had.
        pkill -u "$USER" -f clio_run 2>/dev/null || true
        sleep 2
    else
        bash "$SCRIPT_DIR/20_ingest_cte.sh" "$VOLUME" "faiss_ivf::$VOLUME"
    fi

    drop_page_cache
    head -5 /proc/meminfo > "$RESULTS/meminfo_${VOLUME}_${BACKEND}_$TS.txt"

    # CHIMOD_INFLIGHTS (chimod only): space-separated list of concurrent
    # SearchTask counts to sweep (comparability/ablation). Default: 8 =
    # the baseline's 8 scan threads.
    [ -n "${TELEMETRY_PHASE_FILE:-}" ] && echo "bench_$BACKEND" > "$TELEMETRY_PHASE_FILE" || true
    if [ "$BACKEND" = "chimod" ]; then
        for INFLIGHT in ${CHIMOD_INFLIGHTS:-8}; do
            echo "--- chimod inflight=$INFLIGHT"
            "$ROOT/build/bench_ivf_qps" \
                --backend chimod \
                --protocol step3 \
                --index "$INDEX" \
                --queries "$QUERIES" \
                --tag "faiss_ivf::$VOLUME" \
                --label "$VOLUME" \
                --inflight "$INFLIGHT" \
                --csv "$CSV" 2>&1 | tee -a "$LOG"
        done
    else
        "$ROOT/build/bench_ivf_qps" \
            --backend "$BACKEND" \
            --protocol step3 \
            --index "$INDEX" \
            --queries "$QUERIES" \
            --tag "faiss_ivf::$VOLUME" \
            --label "$VOLUME" \
            --csv "$CSV" 2>&1 | tee "$LOG"
    fi

    echo "rows appended to $CSV"
done

echo
echo "Done. Plot with: python3 scripts/40_plot_qps.py --baseline <baseline.csv>"
