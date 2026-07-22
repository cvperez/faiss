# FAISS × IOWarp

What is implemented in `contrib/iowarp/`: FAISS IVF search hosted inside the
CLIO runtime as a ChiMod, with inverted lists stored in CTE. For the design
narrative and the proposed clio-core change see
[DESIGN_AND_SOLUTION.md](DESIGN_AND_SOLUTION.md) and
[UPSTREAM_PROPOSAL_IOWARP.md](UPSTREAM_PROPOSAL_IOWARP.md); for build and usage
see [README.md](README.md).

## Components

```
contrib/iowarp/
├── cte_client.{h,cpp}           # connect a process to the CLIO runtime + CTE
├── ivf_cte_ingest.{h,cpp}       # copy a FAISS IVF's lists into CTE blobs
├── ivf_to_iowarp.cpp            # CLI: ingest a populated .index into CTE (+ --verify)
├── bench_ivf_qps.cpp            # selftest + QPS harness driving the ChiMod
├── chimod/                      # the faiss_ivf ChiMod
│   ├── clio_mod.yaml            # method ids (kOpenIndex/kSearch/kStats/...)
│   ├── include/clio_runtime/faiss_ivf/{faiss_ivf_tasks,client,runtime}.h
│   └── src/{faiss_ivf_client.cc, faiss_ivf_runtime.cc, autogen/faiss_ivf_lib_exec.cc}
├── config/                      # clio_run compose configs (RAM + NVMe tiers + the ChiMod)
└── scripts/                     # environment setup, ingest, run
```

The ChiMod runs at pool 600.0; CTE core is pool 512.0. The build is a standalone
CMake project: `find_package(faiss)` plus pip-wheel discovery for the iowarp-core
native libraries (the wheel exports no CMake package), compiled against a
clio-core header checkout pinned to the wheel version.

## Blob data model

```
CTE tag:  faiss_ivf::<volume>
  blob "sizes"      int64[nlist] — entries per list (empty lists: no list blob)
  blob "list/<i>"   uint8 codes[size_i * code_size] ++ int64 ids[size_i]
```

Per-list layout (codes then ids, contiguous) matches `OnDiskInvertedLists`, so
ingestion from a `.ivfdata` file is a straight per-list copy. One blob per list
means CTE places and migrates each list independently.

`ivf_cte_ingest.cpp::IngestIvfToCte` writes these blobs (pipelined puts, `sizes`
last); `ivf_to_iowarp.cpp` is the CLI wrapper with an optional `--verify N`
byte-check against the mmap pointers.

## The ChiMod

Container state (`faiss_ivf_runtime.h`): a `faiss::IndexIVF*` (metadata only,
loaded with `IO_FLAG_SKIP_IVF_DATA`), the per-list `sizes_`, the CTE `TagId`, an
in-process CTE `Client`, and fetch/scan statistics counters.

### OpenIndex

Binds the CTE tag, reads the `sizes` blob, loads the index metadata, and sets
`ntotal` from the size sum. It does not read any inverted list — the lists stay
in CTE. Re-opening a different `(index_path, tag)` replaces the state.

### Search

1. Coarse-quantize: `quantizer->search(nq, q, nprobe, ...)` assigns each query to
   its `nprobe` nearest lists.
2. Build the deduplicated set of probed non-empty lists, plus a per-list map of
   the `(query, coarse_distance)` pairs that touch it.
3. Read each probed list from CTE on demand: keep up to `kMaxInflight` (64)
   `AsyncGetBlob`s outstanding into shared-memory buffers; on each completion,
   scan that list and free its buffer.
4. Scan a list across `kScanThreads` (8) `InvertedListScanner`s — one scanner per
   thread, each `(query, probe)` pair updating that query's own top-k heap
   (`ScanListParallel`).
5. Reorder each query's heap and return the top-k distances and ids.

The per-list read copies the bytes out of the tier; that copy is the cost the
clio-core zero-copy proposal removes.

### Stats / Monitor

Report (and optionally reset) searches served, lists read, bytes read, read-wait
µs, and scan µs.

## Runtime discipline (constraints the implementation obeys)

- **Handler-safe CTE client init.** `CLIO_CTE_CLIENT_INIT()` blocks (`Wait()`) and
  deadlocks a cooperative worker if called from a handler; the runtime binds an
  in-process `Client` to `kCtePoolId` instead.
- **Bounded in-flight tasks.** Issuing unbounded `AsyncGetBlob`s from a handler
  livelocks the runtime queues; Search windows the reads (≤64 outstanding).
- **Poll-then-finalize.** Only `CLIO_CO_AWAIT` a future that `IsComplete()` (the
  await returns immediately), and `chi::yield()` only while reads are
  outstanding — awaiting a still-pending future with more sub-tasks in flight can
  resume a destroyed coroutine frame, and yielding with nothing outstanding never
  wakes.
- **No OpenMP region inside a coroutine body.** gcc 11 ICEs on that; the parallel
  scan lives in a plain function (`ScanListParallel`).

## Build dependencies

- **iowarp-core pip wheel** (native libraries + `clio_run`); no source build.
- **clio-core header checkout** pinned to the wheel version (headers only). The
  wheel ships no header tree or CMake package, so `CMakeLists.txt` discovers the
  wheel `lib/` via the Python import and the checkout's include dirs directly,
  and vendors zmq / msgpack / cereal headers the wheel links but does not ship.
- **faiss**, installed as shared libraries (`find_package(faiss)`); the AVX-512
  SIMD variant is linked when present.

## Verification

- `bench_ivf_qps --selftest-chimod` — build a tiny index, ingest it into CTE,
  open it in the ChiMod, and assert the ChiMod's `(D, I)` is bitwise-identical to
  stock FAISS for both single-batch and split-batch searches. Run after every
  build.
- `ivf_to_iowarp --verify N` — read back N random lists from CTE and byte-compare
  against the mmap pointers.
