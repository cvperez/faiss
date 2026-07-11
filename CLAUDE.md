# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

Faiss is a C++20 library for efficient similarity search and clustering of dense vectors, with complete Python/numpy bindings (SWIG) and optional GPU support. This repo is a fork (`cvperez/faiss`) of `facebookresearch/faiss` (v1.14.3); notable additions over upstream include a Metal GPU backend for Apple Silicon (`faiss/gpu_metal/`) and a runtime SIMD dynamic-dispatch build mode (`FAISS_OPT_LEVEL=dd`).

## Build commands

```shell
# Configure (CPU-only, no Python, with tests — typical dev configuration)
cmake -B build . -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF \
  -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release

# Build the C++ library
make -C build -j faiss          # or: cmake --build build -j --target faiss
```

Key CMake options (see INSTALL.md for the full list):
- `FAISS_ENABLE_GPU` (default ON) — CUDA indexes; `FAISS_ENABLE_ROCM=ON` for AMD; `FAISS_ENABLE_CUVS=ON` for NVIDIA cuVS backends
- `FAISS_ENABLE_METAL` — Metal backend; defaults to ON on Apple Silicon only
- `FAISS_ENABLE_PYTHON` (default ON) — SWIG bindings (requires python3, numpy, swig)
- `FAISS_OPT_LEVEL` — SIMD level: `generic`, `avx2`, `avx512`, `avx512_spr` (x86); `sve` (aarch64); or `dd` for runtime dynamic dispatch compiled into the single main library
- `FAISS_ENABLE_SVS` — Intel Scalable Vector Search integration (fetched/built by CMake)
- `FAISS_ENABLE_C_API`, `BUILD_SHARED_LIBS`, `FAISS_ENABLE_MKL`

A BLAS implementation is the only hard dependency (MKL strongly recommended on Intel).

With non-generic `FAISS_OPT_LEVEL`, each SIMD level is a separate target/library (`faiss_avx2`, `faiss_avx512`, `faiss_avx512_spr`, `faiss_sve`) — build that target before `swigfaiss`. In `dd` mode the SIMD variants are compiled per-file into the main `faiss` target instead, selected at runtime.

### Python bindings

```shell
make -C build -j swigfaiss
(cd build/faiss/python && python setup.py install)   # or `build` instead of install
```

## Testing

```shell
# C++ tests (gtest, single binary build/tests/faiss_test)
make -C build -j faiss_test
ctest --test-dir build                                   # all tests
./build/tests/faiss_test --gtest_filter='TestHNSW*'      # single test / suite

# Python tests (after building swigfaiss with `python setup.py build`)
PYTHONPATH="$(ls -d ./build/faiss/python/build/lib*/)" pytest tests/test_*.py
PYTHONPATH=... pytest tests/test_index.py -k test_name   # single test

# GPU tests live in faiss/gpu/test/ (C++ and Python), Metal tests in faiss/gpu_metal/test/
```

Tests live in `tests/` (mixed C++ `test_*.cpp` and Python `test_*.py`). C++ test files must be registered in `tests/CMakeLists.txt`. gtest is fetched automatically by CMake.

## Coding style

- C++20; 4-space indentation, no tabs; 80-character lines (C++ and Python)
- `clang-format` with the repo's `.clang-format` is the source of truth for formatting

## Architecture

### The Index abstraction

Everything is built around `faiss/Index.h`: an index stores a set of `d`-dimensional float32 vectors identified by `int64` ids, and exposes `train()` / `add()` / `search()` (k-NN under L2 or inner product, see `MetricType.h`). `IndexBinary*` classes are the parallel hierarchy for binary vectors with Hamming distance. All API methods take raw C-style pointers/arrays (`float*`, `idx_t*`) — no fancy containers on API boundaries.

Indexes compose: many indexes wrap other indexes.
- `IndexIVF*` uses another Index as the coarse quantizer for cluster assignment
- `IndexPreTransform` applies a `VectorTransform` (PCA, OPQ, ...) before an inner index
- `IndexRefine` re-ranks results from a fast index with a more accurate one
- `IndexIDMap` adds arbitrary-id support to indexes that require sequential ids
- `IndexShards` / `IndexReplicas` distribute over multiple sub-indexes

`index_factory()` (`faiss/index_factory.cpp`) builds these compositions from strings like `"PCA80,IVF4096,PQ8+16"` — the preferred way to construct indexes and the vocabulary used in benchmarks/autotuning (`AutoTune.cpp`).

### Layering: faiss/ vs faiss/impl/

Top-level `faiss/Index*.{h,cpp}` files are the user-facing Index classes. `faiss/impl/` holds the underlying algorithm implementations they wrap: quantizers (`ProductQuantizer`, `ScalarQuantizer`, `ResidualQuantizer`/additive quantizers, `RaBitQuantizer`), graph structures (`HNSW`, `NSG`, `NNDescent`), result handling (`ResultHandler.h`), and I/O. For example `IndexHNSW` is a thin combination of `impl/HNSW` (graph) plus a storage index for the actual vectors.

Other key directories inside `faiss/`:
- `invlists/` — inverted list storage for IVF indexes (`InvertedLists` interface, in-memory `ArrayInvertedLists`, on-disk/mmap variants)
- `utils/` — distance computations, heaps, partitioning; the performance-critical SIMD kernels live here with per-architecture variants
- `impl/simdlib*`, `impl/simd_dispatch.h` — SIMD abstraction layer (AVX2/AVX512/NEON/SVE/RVV/emulated)
- `impl/index_read.cpp`, `impl/index_write.cpp` — serialization (`write_index`/`read_index`); `clone_index.cpp` for deep copies. New index types must be wired into both.

### FastScan family

`Index*FastScan` variants (PQ, additive quantizers, RaBitQ, plus IVF versions) are SIMD-register-resident reimplementations that process vectors in blocks of 32 with 4-bit lookup tables — separate classes from their non-FastScan counterparts, with code packing handled by `impl/CodePacker*`.

### GPU

`faiss/gpu/` contains CUDA implementations (`GpuIndexFlat`, `GpuIndexIVF*`, `GpuIndexCagra`). CPU↔GPU conversion goes through `GpuCloner`. ROCm support is generated from the CUDA sources via `faiss/gpu/hipify.sh` at configure time. cuVS provides alternative backends for IVF-Flat/IVF-PQ/CAGRA when enabled. `faiss/gpu_metal/` (fork-specific) implements Flat and IVFFlat on Apple Metal in Objective-C++.

### Python layer

`faiss/python/swigfaiss.swig` generates the raw bindings; `class_wrappers.py` replaces the raw methods with numpy-friendly wrappers (this is where the Pythonic `index.search(x, k)` signatures come from). `loader.py` selects the appropriate SIMD build at import time. `contrib/` is pure-Python utility code (datasets, evaluation, on-disk helpers, torch interop in `contrib/torch/`) tested by `tests/test_contrib*.py`.

`c_api/` is a plain-C wrapper around the C++ library, built only with `FAISS_ENABLE_C_API=ON`.
