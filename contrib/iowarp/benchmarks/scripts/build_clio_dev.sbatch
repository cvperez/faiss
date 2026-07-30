#!/bin/bash
#SBATCH --job-name=clio_dev_build
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --partition=compute
#SBATCH --output=/mnt/common/cvazquezperezdelacru/IOWARP/faiss/contrib/iowarp/benchmarks/results/logs/%j_clio_dev_build.out

# Build + install clio-core dev to a local prefix (no sudo).
set -e
source ~/iowarp-dev-deps/env.sh
cd ~/clio-core

echo "=== toolchain on $(hostname) ==="
gcc --version | head -1; cmake --version | head -1; echo "nproc=$(nproc)"

echo "=== build (all configured targets) ==="
cmake --build build-dev -j"$(nproc)"

echo "=== install to $HOME/clio-core-dev-install ==="
cmake --install build-dev

echo "=== verify install ==="
ls -la "$HOME/clio-core-dev-install/bin/clio_run" 2>&1 || echo "MISSING clio_run"
echo "--- installed client/runtime libs ---"
ls "$HOME/clio-core-dev-install/lib" | grep -E 'clio_(cte_core|run_cxx|ctp_host|admin|bdev)' || true
echo "--- find_package export ---"
ls "$HOME/clio-core-dev-install/lib/cmake/iowarp-core/" 2>&1 || echo "no cmake export dir"
echo "=== DONE ==="
