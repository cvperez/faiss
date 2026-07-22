#!/usr/bin/env bash
# 20_ingest_cte.sh — start a fresh CLIO runtime and ingest one index volume
# into CTE blobs.
#
# Purpose:
#   Kill any running clio_run, start a fresh one with config/ares_cte.yaml,
#   wait for it to come up, then run ivf_to_iowarp (with --verify 16) to copy
#   the volume's inverted lists from the .ivfdata file into CTE blobs
#   (tag faiss_ivf::<volume>, one "list/<i>" blob per list + a "sizes" blob).
#   The runtime is LEFT RUNNING on exit — bench runs against it next.
#
# Inputs:
#   $1  volume name, e.g. ondisk_nb50M — resolves to
#       $IOWARP_WORK_DIR/<volume>.{index,ivfdata}
#   $2  (optional) CTE tag name          (default: faiss_ivf::<volume>)
#   Env overrides:
#     IOWARP_WORK_DIR   (default: /mnt/common/$USER/faiss/slurm_bench/work)
#     FAISS_INSTALL     (default: $HOME/faiss-install)
#     VERIFY_N          lists to read back and byte-compare (default: 16)
#     CLIO_START_WAIT   seconds to wait for clio_run startup (default: 15)
#
# Outputs:
#   results/clio_run_<volume>_<ts>.log   runtime stdout/stderr
#   results/ingest_<volume>_<ts>.log     ivf_to_iowarp output (incl. verify)
#   A live clio_run process + populated CTE tag.
#
# Example:
#   bash scripts/20_ingest_cte.sh ondisk_nb50M
#   bash scripts/20_ingest_cte.sh ondisk_step3_nb178M faiss_ivf::exp178

set -euo pipefail

VOLUME="${1:?usage: 20_ingest_cte.sh <volume> [tag]}"
TAG="${2:-faiss_ivf::$VOLUME}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS="$ROOT/results"
CONF_SRC="$ROOT/config/ares_cte.yaml"
IOWARP_WORK_DIR="${IOWARP_WORK_DIR:-/mnt/common/$USER/faiss/slurm_bench/work}"
FAISS_INSTALL="${FAISS_INSTALL:-$HOME/faiss-install}"
VERIFY_N="${VERIFY_N:-16}"
CLIO_START_WAIT="${CLIO_START_WAIT:-15}"
TS="$(date +%Y%m%d_%H%M%S)"

INDEX="$IOWARP_WORK_DIR/$VOLUME.index"
IVFDATA="$IOWARP_WORK_DIR/$VOLUME.ivfdata"
[ -f "$INDEX" ]   || { echo "ERROR: missing $INDEX" >&2; exit 1; }
[ -f "$IVFDATA" ] || { echo "ERROR: missing $IVFDATA" >&2; exit 1; }
mkdir -p "$RESULTS"

# --- runtime environment (wheel bins/libs + faiss + our build) ---------------
IOWARP_PKG="$(python3 -c 'import iowarp_core, os; print(os.path.dirname(iowarp_core.__file__))')"
export PATH="$IOWARP_PKG/bin:$PATH"
export LD_LIBRARY_PATH="$IOWARP_PKG/lib:$FAISS_INSTALL/lib:$FAISS_INSTALL/lib64:$ROOT/build:$ROOT/build/chimod:${LD_LIBRARY_PATH:-}"

# --- render the config (expand ${USER} and the RAM-tier cap; drop chimod ----
# entry if not built). CTE_RAM_TIER_GB default 30 sizes the CTE RAM tier.
RENDERED="$RESULTS/ares_cte_rendered_$TS.yaml"
sed -e "s|\${USER}|$USER|g" \
    -e "s|\${CTE_RAM_TIER_GB}|${CTE_RAM_TIER_GB:-30}|g" \
    "$CONF_SRC" > "$RENDERED"
if [ "${CTE_RAM_TIER_GB:-30}" = "0" ]; then
    echo "NOTE: CTE_RAM_TIER_GB=0 — stripping the RAM tier (file tier only)"
    sed -i '/# BEGIN cte_ram_tier/,/# END cte_ram_tier/d' "$RENDERED"
fi
if ! ls "$ROOT"/build/chimod/libclio_faiss_ivf_runtime.so >/dev/null 2>&1; then
    echo "NOTE: chimod runtime lib not built — stripping clio_faiss_ivf from compose"
    sed -i '/# BEGIN clio_faiss_ivf/,/# END clio_faiss_ivf/d' "$RENDERED"
fi
mkdir -p "/mnt/nvme/$USER/cte_tier"
# The file bdev stores the tier in a file named <path>_node<i>, which
# survives job exits on the node-local NVMe — remove stale ones so every
# run starts from a fresh tier.
rm -f "/mnt/nvme/$USER/cte_tier_node"* 2>/dev/null || true

# --- kill-then-restart clio_run ----------------------------------------------
echo "=== stopping any existing clio_run"
pkill -u "$USER" -f clio_run 2>/dev/null || true
sleep 2
pkill -9 -u "$USER" -f clio_run 2>/dev/null || true

CLIO_LOG="$RESULTS/clio_run_${VOLUME}_$TS.log"
echo "=== starting clio_run (conf: $RENDERED, log: $CLIO_LOG)"
CLIO_SERVER_CONF="$RENDERED" clio_run start >"$CLIO_LOG" 2>&1 &
CLIO_PID=$!
disown "$CLIO_PID"

for i in $(seq 1 "$CLIO_START_WAIT"); do
    kill -0 "$CLIO_PID" 2>/dev/null || { echo "ERROR: clio_run died at startup — see $CLIO_LOG" >&2; exit 1; }
    sleep 1
done
echo "clio_run up (pid $CLIO_PID) after ${CLIO_START_WAIT}s grace"

# --- ingest -------------------------------------------------------------------
INGEST_LOG="$RESULTS/ingest_${VOLUME}_$TS.log"
echo "=== ingesting $VOLUME -> tag '$TAG' (verify $VERIFY_N lists; log: $INGEST_LOG)"
# NB: the .ivfdata path is recorded inside the .index file; the binary
# takes only <index> <tag>. $IVFDATA above is validated for existence only.
[ -n "${TELEMETRY_PHASE_FILE:-}" ] && echo "ingest" > "$TELEMETRY_PHASE_FILE" || true
"$ROOT/build/ivf_to_iowarp" "$INDEX" "$TAG" --verify "$VERIFY_N" 2>&1 | tee "$INGEST_LOG"
[ -n "${TELEMETRY_PHASE_FILE:-}" ] && echo "idle" > "$TELEMETRY_PHASE_FILE" || true

echo "=== ingest OK — runtime left running (pid $CLIO_PID)"
