# FAISS × IOWarp: IVF search inside the CLIO runtime

This folder hosts FAISS IVF vector search inside the [IOWarp](https://github.com/iowarp)
CLIO runtime as a module (a "ChiMod"), with the inverted lists stored in the
Context Transfer Engine (CTE). One blob per list, tiered across a capped RAM
tier and node-local NVMe.


## Motivation

FAISS's `OnDiskInvertedLists` stores the inverted lists in a memory-mapped (`.ivfdata`) file, relying entirely on the operating system's page cache to decide which data remains in memory. This approach works well when the index fits in RAM. However, as the index grows to several times the available memory, performance degrades significantly. Because the page cache manages memory at the granularity of individual 4 KiB pages rather than complete inverted lists, frequently accessed lists are repeatedly evicted and reloaded. As a result, the system experiences a large increase in page-ins, meaning data has to be fetched from disk much more often. This additional disk I/O leads to page cache thrashing, higher query latency, and a substantial drop in throughput.

CTE manages placement explicitly instead: each inverted list is a named blob,
placed and migrated on its own across RAM and NVMe tiers. The unit CTE moves is
exactly the unit a search reads. The ChiMod runs the IVF search next to that
storage, inside the runtime.

## Architecture

```
   client process                      CLIO runtime (clio_run)
 ┌───────────────────────┐           ┌───────────────────────────────┐
 │ bench_ivf_qps         │  Search   │  clio_faiss_ivf ChiMod (600.0)│
 │  queries ───────────► │  task     │   OpenIndex: metadata + tag   │
 │  ◄─────── top-k        │  (shm)    │   Search:  quantize → probe   │
 └───────────────────────┘           │            → read lists → scan│
                                     │  clio_cte_core (512.0)         │
                                     │   ├ RAM tier   └ NVMe tier     │
                                     └───────────────────────────────┘
```

- **`clio_faiss_ivf` ChiMod** (`chimod/`): the index (metadata only) is opened
  inside the runtime; a `SearchTask` carries queries in and top-k out.
  `OpenIndex` binds the CTE tag and records the list sizes; `Search`
  coarse-quantizes, then for each probed list reads it from CTE, scans it, and
  frees the buffer. The per-list read copies the bytes out of the tier. 

## Blob data model

```
CTE tag:  faiss_ivf::<volume>
  blob "sizes"      int64[nlist] — entries per list (empty lists: no list blob)
  blob "list/<i>"   uint8 codes[size_i * code_size] ++ int64 ids[size_i]
```

Per-list layout (codes then ids, contiguous) matches `OnDiskInvertedLists`, so
ingestion from a `.ivfdata` file is a straight per-list copy. One blob per list
means CTE places and migrates each list independently.

## Building

Three ingredients:

1. **iowarp-core pip wheel** — the native CLIO libraries and `clio_run` come
   from pip, never from source:

   ```shell
   pip install iowarp-core            # v2.1.0 at time of writing
   PKG=$(python3 -c "import iowarp_core, os; print(os.path.dirname(iowarp_core.__file__))")
   export PATH="$PKG/bin:$PATH"
   export LD_LIBRARY_PATH="$PKG/lib:$LD_LIBRARY_PATH"
   ```

2. **clio-core headers, pinned to the wheel version** — the wheel ships no
   header tree, so headers come from a source checkout used for its `include/`
   dirs only (the tag MUST match the wheel version; ABI compatibility depends
   on it):

   ```shell
   git clone --depth 1 --branch v2.1.0 https://github.com/iowarp/clio-core.git \
       ~/clio-core-v2.1.0
   ```

3. **faiss, installed as shared libraries:**

   ```shell
   cmake -B build . -DCMAKE_BUILD_TYPE=Release -DFAISS_OPT_LEVEL=avx512 \
     -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DFAISS_ENABLE_EXTRAS=OFF \
     -DBUILD_TESTING=OFF -DFAISS_ENABLE_MKL=OFF -DBUILD_SHARED_LIBS=ON \
     -DCMAKE_INSTALL_PREFIX=$HOME/faiss-install
   cmake --build build -j && cmake --install build
   ```

Then this folder (standalone CMake project):

```shell
cd contrib/iowarp
cmake -B build . -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$HOME/faiss-install" \
  -DIOWARP_HEADERS_DIR="$HOME/clio-core-v2.1.0"
cmake --build build -j
```

The wheel's `lib/` dir is discovered through `python3 -c "import iowarp_core"`
(override with `-DIOWARP_WHEEL_DIR=...`). The build produces the ingest tool
`ivf_to_iowarp`, the benchmark `bench_ivf_qps`, and the ChiMod libraries
`libclio_faiss_ivf_{client,runtime}.so`.

## Running

The ChiMod runs inside `clio_run`, whose pools are composed from
`config/ares_cte.yaml`: `clio_bdev` (DRAM allocator), `clio_cte_core`
(pool 512.0; RAM tier + NVMe file tier), and `clio_faiss_ivf` (pool 600.0). Start
the runtime with the rendered config:

```shell
CLIO_SERVER_CONF=<rendered ares_cte.yaml> clio_run start &
```

### `ivf_to_iowarp` — ingest an on-disk index into CTE

```
ivf_to_iowarp <populated.index> <tag_name> [--verify N] [--batch B]
```

Reads the index with the stock `OnDiskInvertedLists` mmap hook, copies every
non-empty list into blob `list/<i>` under `<tag_name>` (pipelined puts), and
writes the `"sizes"` blob last. `--verify N` reads back N random lists and
byte-compares them against the mmap pointers. Requires a running `clio_run`.





