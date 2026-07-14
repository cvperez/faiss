#!/bin/bash
#SBATCH --job-name=iowarp_qps
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --time=05:00:00
#SBATCH --partition=compute
#SBATCH --output=/mnt/common/cvazquezperezdelacru/IOWARP/faiss/contrib/iowarp/results/logs/%j_qps.out

# sbatch_bench.sh — SLURM wrapper: run the QPS experiment on one Ares
# compute node.
#
# Purpose:
#   Allocate an exclusive node (the whole ~46 GiB page cache must belong to
#   this job — same as the mmap baseline runs), make sure the environment is
#   built (00_env_ares.sh, idempotent), then run 30_run_bench.sh for one
#   volume across the requested backends.
#
# Inputs (via --export, like the step3 scripts):
#   FAISS_VOLUME    index volume name       (default: ondisk_nb10M — smoke)
#   FAISS_BACKENDS  space-separated list    (default: "iowarp")
#   Plus the env overrides of 00/20/30 (IOWARP_WORK_DIR, FAISS_INSTALL, ...).
#
# Outputs:
#   results/logs/<jobid>_qps.out          this job's log
#   results/qps_<volume>.csv              appended by bench_ivf_qps
#
# Examples:
#   mkdir -p results/logs     # sbatch needs the --output dir to exist
#   sbatch scripts/sbatch_bench.sh
#   sbatch --export=ALL,FAISS_VOLUME=ondisk_nb50M,FAISS_BACKENDS="mmap iowarp" \
#       scripts/sbatch_bench.sh
#   sbatch --time=12:00:00 \
#       --export=ALL,FAISS_VOLUME=ondisk_step3_nb178M,FAISS_BACKENDS="iowarp chimod" \
#       scripts/sbatch_bench.sh
#
# NOTE: sbatch spools a copy of this file, so paths below are absolute
# (adjust ROOT and the #SBATCH --output line if the checkout moves).

set -euo pipefail

ROOT="${IOWARP_CONTRIB_ROOT:-/mnt/common/cvazquezperezdelacru/IOWARP/faiss/contrib/iowarp}"
export FAISS_VOLUME="${FAISS_VOLUME:-ondisk_nb10M}"
FAISS_BACKENDS="${FAISS_BACKENDS:-iowarp}"

echo "=== IOWarp QPS  volume=${FAISS_VOLUME}  backends=${FAISS_BACKENDS}"
echo "=== job=${SLURM_JOB_ID:-?}  host=$(hostname)  started=$(date)"
echo "MemTotal:     $(awk '/MemTotal/{printf "%.2f GiB", $2/1024/1024}' /proc/meminfo)"
echo "MemAvailable: $(awk '/MemAvailable/{printf "%.2f GiB", $2/1024/1024}' /proc/meminfo)"
echo

bash "$ROOT/scripts/00_env_ares.sh"

# Telemetry: NVMe throughput + active time, per-tier occupancy, NFS read
# rate, phase-tagged (the baseline's Figure-3a analogue). Never fails the job.
export TELEMETRY_PHASE_FILE="$ROOT/results/.phase_${SLURM_JOB_ID:-local}"
echo "setup" > "$TELEMETRY_PHASE_FILE" || true
TELEMETRY_CSV="$ROOT/results/telemetry_${FAISS_VOLUME}_${SLURM_JOB_ID:-local}.csv"
python3 "$ROOT/scripts/telemetry_sampler.py" \
    --out "$TELEMETRY_CSV" --phase-file "$TELEMETRY_PHASE_FILE" \
    --interval 5 --tier-dir "/mnt/nvme/$USER/cte_tier" &
TELEMETRY_PID=$!
trap 'echo done > "$TELEMETRY_PHASE_FILE" 2>/dev/null; kill "$TELEMETRY_PID" 2>/dev/null' EXIT

# shellcheck disable=SC2086  # backends are intentionally word-split
bash "$ROOT/scripts/30_run_bench.sh" "$FAISS_VOLUME" $FAISS_BACKENDS

# Courtesy: stop the runtime and free node-local NVMe tier space.
pkill -u "$USER" -f clio_run 2>/dev/null || true
rm -rf "/mnt/nvme/$USER/cte_tier"

echo
echo "Finished: $(date)"
