#!/usr/bin/env bash
# 00_env_ares.sh — Phase 0: environment discovery + idempotent builds on an
# Ares compute node.
#
# Purpose:
#   Check (and build only what is missing) everything the experiments need:
#     1. the iowarp-core pip wheel (clio_run on PATH, native libs)
#     2. the pinned clio-core header worktree (headers only, never compiled)
#     3. the faiss fork installed to $FAISS_INSTALL (shared libs, avx512)
#     4. this contrib/iowarp/ CMake project (ivf_to_iowarp, bench_ivf_qps,
#        chimod libs if the CLIO runtime headers allow)
#   Then print a discovery summary (binaries, data volumes, NVMe space).
#
# Inputs (env overrides, all optional):
#   FAISS_SRC           faiss checkout        (default: this repo, two dirs up)
#   FAISS_INSTALL       faiss install prefix  (default: $HOME/faiss-install)
#   IOWARP_HEADERS_DIR  clio-core header worktree pinned to the wheel version
#                       (default: $HOME/clio-core-v2.1.0)
#   IOWARP_WORK_DIR     pre-built index pairs (default:
#                       /mnt/common/$USER/faiss/slurm_bench/work)
#   JOBS                parallel build jobs   (default: nproc)
#
# Outputs:
#   $FAISS_INSTALL/{lib,include}/...          (faiss install, if it was missing)
#   contrib/iowarp/build/{ivf_to_iowarp,bench_ivf_qps,...}
#   Discovery summary on stdout.
#
# Example:
#   salloc -N 1 --exclusive          # NEVER run builds on the login node
#   ssh $SLURM_NODELIST
#   bash contrib/iowarp/scripts/00_env_ares.sh
#
# Idempotent: safe to re-run; each step is check-then-build.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"                      # contrib/iowarp
FAISS_SRC="${FAISS_SRC:-$(cd "$ROOT/../.." && pwd)}"      # the faiss checkout
FAISS_INSTALL="${FAISS_INSTALL:-$HOME/faiss-install}"
IOWARP_HEADERS_DIR="${IOWARP_HEADERS_DIR:-$HOME/clio-core-v2.1.0}"
IOWARP_WORK_DIR="${IOWARP_WORK_DIR:-/mnt/common/$USER/faiss/slurm_bench/work}"
QUERIES="${QUERIES:-/mnt/common/$USER/faiss/slurm_bench/data/bigann/bigann_query.bvecs}"
JOBS="${JOBS:-$(nproc)}"

log() { printf '\n== %s ==\n' "$*"; }

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "WARNING: no SLURM allocation detected — do NOT build or run on the" >&2
    echo "         Ares login node. Use: salloc -N 1 --exclusive" >&2
fi

# --- 1. iowarp-core pip wheel -----------------------------------------------
log "iowarp-core wheel"
if ! python3 -c "import iowarp_core" 2>/dev/null; then
    echo "iowarp_core not importable — installing the pip wheel"
    python3 -m pip install --user iowarp-core
fi
IOWARP_PKG="$(python3 -c 'import iowarp_core, os; print(os.path.dirname(iowarp_core.__file__))')"
IOWARP_VERSION="$(python3 -c 'import iowarp_core; print(iowarp_core.__version__)')"
export PATH="$IOWARP_PKG/bin:$PATH"
export LD_LIBRARY_PATH="$IOWARP_PKG/lib:${LD_LIBRARY_PATH:-}"
command -v clio_run >/dev/null || { echo "ERROR: clio_run not on PATH after wheel install" >&2; exit 1; }
echo "iowarp_core $IOWARP_VERSION at $IOWARP_PKG"
echo "clio_run: $(command -v clio_run)"

# --- 2. clio-core header worktree (pinned to the wheel version) -------------
log "clio-core headers ($IOWARP_HEADERS_DIR)"
if [ ! -d "$IOWARP_HEADERS_DIR" ]; then
    echo "missing — cloning tag v$IOWARP_VERSION (headers only, never built)"
    git clone --depth 1 --branch "v$IOWARP_VERSION" \
        https://github.com/iowarp/clio-core.git "$IOWARP_HEADERS_DIR"
fi
# ABI contract: the checkout tag MUST match the wheel version.
if [ -d "$IOWARP_HEADERS_DIR/.git" ]; then
    HDR_TAG="$(git -C "$IOWARP_HEADERS_DIR" describe --tags --always 2>/dev/null || echo unknown)"
    echo "header worktree at: $HDR_TAG (wheel is $IOWARP_VERSION — these must match)"
fi

# --- 3. faiss (shared, avx512, no python — the bench links C++ only) --------
log "faiss install ($FAISS_INSTALL)"
if [ -f "$FAISS_INSTALL/include/faiss/Index.h" ] && \
   ls "$FAISS_INSTALL"/lib*/libfaiss*.so >/dev/null 2>&1; then
    echo "already installed — skipping build"
else
    echo "building faiss from $FAISS_SRC"
    cmake -S "$FAISS_SRC" -B "$FAISS_SRC/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$FAISS_INSTALL" \
        -DFAISS_OPT_LEVEL=avx512 \
        -DFAISS_ENABLE_GPU=OFF \
        -DFAISS_ENABLE_PYTHON=OFF \
        -DFAISS_ENABLE_EXTRAS=OFF \
        -DBUILD_TESTING=OFF \
        -DFAISS_ENABLE_MKL=OFF \
        -DBUILD_SHARED_LIBS=ON
    cmake --build "$FAISS_SRC/build" -j "$JOBS"
    cmake --install "$FAISS_SRC/build"
fi

# --- 4. contrib/iowarp (this folder) -----------------------------------------
# Skip when binaries already exist: several compute-node images have NO
# BLAS/LAPACK (e.g. ares-comp-27), so linking there fails even though the
# login-built binaries run fine (libblas/liblapack are bundled into
# $FAISS_INSTALL/lib, which the run scripts put on LD_LIBRARY_PATH).
# FORCE_REBUILD=1 overrides.
log "contrib/iowarp build ($ROOT/build)"
if [ "${FORCE_REBUILD:-0}" != "1" ] && \
   [ -x "$ROOT/build/bench_ivf_qps" ] && [ -x "$ROOT/build/ivf_to_iowarp" ]; then
    echo "binaries already built — skipping (FORCE_REBUILD=1 to override)"
else
    cmake -S "$ROOT" -B "$ROOT/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="$FAISS_INSTALL" \
        -DIOWARP_HEADERS_DIR="$IOWARP_HEADERS_DIR"
    cmake --build "$ROOT/build" -j "$JOBS"
fi

# --- 5. discovery summary -----------------------------------------------------
log "summary"
for bin in ivf_to_iowarp bench_ivf_qps; do
    if [ -x "$ROOT/build/$bin" ]; then
        echo "OK    $ROOT/build/$bin"
    else
        echo "MISS  $ROOT/build/$bin" >&2
    fi
done
if ls "$ROOT"/build/chimod/libclio_faiss_ivf_runtime.so >/dev/null 2>&1; then
    echo "OK    chimod runtime lib (Level 1 available)"
else
    echo "NOTE  chimod runtime lib not built — Level 1 (backend=chimod) unavailable"
fi
echo
echo "Data volumes under $IOWARP_WORK_DIR:"
ls -lh "$IOWARP_WORK_DIR"/ondisk_*.index 2>/dev/null || echo "  (none found)"
[ -f "$QUERIES" ] && echo "Queries: $QUERIES" || echo "MISS  queries: $QUERIES" >&2
echo
echo "NVMe scratch:"
mkdir -p "/mnt/nvme/$USER" 2>/dev/null || true
df -h "/mnt/nvme/$USER" 2>/dev/null || echo "  /mnt/nvme/$USER not available on this node" >&2
echo
echo "Environment ready. Next: scripts/20_ingest_cte.sh <volume>  (or scripts/30_run_bench.sh)"
