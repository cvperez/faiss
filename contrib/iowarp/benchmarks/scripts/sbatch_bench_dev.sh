#!/bin/bash
#SBATCH --job-name=iowarp_qps_dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --partition=compute
#SBATCH --output=/mnt/common/cvazquezperezdelacru/IOWARP/faiss/contrib/iowarp/benchmarks/results/logs/%j_qps_dev.out

# sbatch_bench_dev.sh — QPS experiment against a clio-core *dev* source install
# (the zero-IPC AsyncGetBlob path), on one exclusive Ares node.
#
# Unlike sbatch_bench.sh this does NOT run 00_env_ares.sh (which pip-installs
# the v2.1.0 wheel and rebuilds the contrib against it). It assumes:
#   * clio-core dev built+installed at $CLIO_DEV_PREFIX  (Phase 1)
#   * dep prefix assembled at        $DEP_PREFIX         (Phase 0)
#   * faiss installed at             $FAISS_INSTALL
#   * contrib built with IOWARP_USE_DEV=ON into           $BUILD_DIR (build-dev)
#
# Inputs (via --export=ALL,...):
#   FAISS_VOLUME     index volume (default ondisk_nb10M)
#   CTE_RAM_TIER_GB  RAM-tier cap (default 30; 0 = file-tier only)
#   CHIMOD_INFLIGHTS concurrent SearchTasks (default 8; with
#                    CHIMOD_ROUTE=owner this is per-node concurrency — use 1)
#   CHIMOD_ROUTE     owner|split (default owner). owner = broadcast +
#                    owner-filtered data-local scan (partial top-k merged in
#                    AggregateOut); split = historical query-split fan-out.
#   RUN_SELFTEST     1 = run bench_ivf_qps --selftest-chimod after ingest
#   VERIFY_N         ingest verify: N sampled lists or "all" (default 16)
#
# Multi-node: submit with `sbatch --nodes=2 --ntasks=2 --ntasks-per-node=1`
# (all three flags on the command line). The script detects
# SLURM_JOB_NUM_NODES>1, renders a hostfile for clio-core
# (networking.hostfile) and 20_ingest_cte.sh starts one clio_run per node.
# With CHIMOD_ROUTE=owner every SearchTask reaches every node (data-local
# scan), so CHIMOD_INFLIGHTS stays 1; with CHIMOD_ROUTE=split use
# CHIMOD_INFLIGHTS=<num nodes> (query-split fan-out).

set -euo pipefail

ROOT="${IOWARP_CONTRIB_ROOT:-/mnt/common/cvazquezperezdelacru/IOWARP/faiss/contrib/iowarp}"
export FAISS_VOLUME="${FAISS_VOLUME:-ondisk_nb10M}"

# --- multi-node wiring --------------------------------------------------------
if [ "${SLURM_JOB_NUM_NODES:-1}" -gt 1 ]; then
    export IOWARP_HOSTFILE="$ROOT/benchmarks/results/hostfile_${SLURM_JOB_ID}"
    scontrol show hostnames "$SLURM_JOB_NODELIST" > "$IOWARP_HOSTFILE"
    echo "=== multi-node allocation: $(tr '\n' ' ' < "$IOWARP_HOSTFILE")"
fi
# Run a command on every node of the allocation (or just locally, 1-node).
on_all_nodes() {
    if [ "${SLURM_JOB_NUM_NODES:-1}" -gt 1 ]; then
        srun --ntasks-per-node=1 --nodes="$SLURM_JOB_NUM_NODES" --export=ALL bash -c "$*"
    else
        bash -c "$*"
    fi
}

# --- dev provider wiring (consumed by 20/30) --------------------------------
CLIO_DEV_PREFIX="${CLIO_DEV_PREFIX:-$HOME/clio-core-dev-install}"
DEP_PREFIX="${DEP_PREFIX:-$HOME/iowarp-dev-deps}"
export CLIO_BIN_DIR="$CLIO_DEV_PREFIX/bin"
export CLIO_LIB_DIR="$CLIO_DEV_PREFIX/lib"
export IOWARP_EXTRA_LIB="$DEP_PREFIX/usr/lib/x86_64-linux-gnu"
export BUILD_DIR="${BUILD_DIR:-$ROOT/build-dev}"
export FAISS_INSTALL="${FAISS_INSTALL:-$HOME/faiss-install}"
# Keep dev rows in their own CSV so the v2.1.0 baseline CSVs are preserved.
export QPS_CSV="$ROOT/benchmarks/results/qps_dev_${FAISS_VOLUME}.csv"
# Search routing (see header). Exported so 30_run_bench.sh sees it.
export CHIMOD_ROUTE="${CHIMOD_ROUTE:-owner}"

# A whole search pass at the large sizes takes longer than the client's default
# 60 s request timeout (v2.1.0 nb100M/nb178M passes ran 56-130 s), so the client
# would abandon a still-running search and abort. Give it a generous budget.
export CLIO_CLIENT_RETRY_TIMEOUT="${CLIO_CLIENT_RETRY_TIMEOUT:-1200}"
export CLIO_WAIT_SERVER="${CLIO_WAIT_SERVER:-120}"

echo "=== IOWarp QPS (DEV)  volume=${FAISS_VOLUME}  RAM_tier=${CTE_RAM_TIER_GB:-30}g  route=${CHIMOD_ROUTE}"
echo "=== job=${SLURM_JOB_ID:-?}  host=$(hostname)  started=$(date)"
echo "=== clio dev: $CLIO_DEV_PREFIX   build: $BUILD_DIR"
# Sanity-check clio_run with the runtime lib path the real run uses. The dep
# prefix carries REAL libzmq/libsodium/libaio/libmsgpackc .so files (not symlinks
# into /lib), so this works on compute nodes that lack those system libs.
LD_LIBRARY_PATH="$CLIO_LIB_DIR:$IOWARP_EXTRA_LIB:${LD_LIBRARY_PATH:-}" \
    "$CLIO_BIN_DIR/clio_run" --help >/dev/null 2>&1 && echo "clio_run(dev) OK" || echo "WARN: clio_run --help failed"
echo "MemAvailable: $(awk '/MemAvailable/{printf "%.2f GiB", $2/1024/1024}' /proc/meminfo)"

# Sanity: contrib dev binaries must exist (built with IOWARP_USE_DEV=ON).
for b in bench_ivf_qps ivf_to_iowarp chimod/libclio_faiss_ivf_runtime.so; do
    [ -e "$BUILD_DIR/$b" ] || { echo "ERROR: missing $BUILD_DIR/$b — build the contrib first" >&2; exit 1; }
done

# Record which clio-core commit this run measured, for provenance.
git -C "$HOME/clio-core" rev-parse --short HEAD 2>/dev/null | sed 's/^/clio-core dev HEAD: /' || true

# Telemetry (phase-tagged), same as the baseline harness.
export TELEMETRY_PHASE_FILE="$ROOT/benchmarks/results/.phase_${SLURM_JOB_ID:-local}"
echo "setup" > "$TELEMETRY_PHASE_FILE" || true
TELEMETRY_CSV="$ROOT/benchmarks/results/telemetry_dev_${FAISS_VOLUME}_${SLURM_JOB_ID:-local}.csv"
python3 "$ROOT/benchmarks/scripts/telemetry_sampler.py" \
    --out "$TELEMETRY_CSV" --phase-file "$TELEMETRY_PHASE_FILE" \
    --interval 5 --tier-dir "/mnt/nvme/$USER/cte_tier" &
TELEMETRY_PID=$!
trap 'echo done > "$TELEMETRY_PHASE_FILE" 2>/dev/null; kill "$TELEMETRY_PID" 2>/dev/null; on_all_nodes "pkill -u $USER -f clio_[r]un 2>/dev/null || true"' EXIT

bash "$ROOT/benchmarks/scripts/30_run_bench.sh" "$FAISS_VOLUME"

# Placement check: how much of the volume spilled to the NVMe file tier.
# ~0 => blobs are RAM-resident => the zero-IPC RAM direct-read path can serve
# them; a large number => they landed on NVMe (fast path cannot apply).
# Multi-node: per-node occupancy (each node has its own cte_tier_node<i>).
echo "=== NVMe file-tier occupancy after run (RAM placement if ~0) ==="
on_all_nodes "echo \"\$(hostname): \$(du -shc /mnt/nvme/$USER/cte_tier* 2>/dev/null | tail -1 || echo 'no NVMe tier files')\""

on_all_nodes "pkill -u $USER -f clio_[r]un 2>/dev/null || true"
on_all_nodes "rm -rf /mnt/nvme/$USER/cte_tier*"
echo "Finished: $(date)"
