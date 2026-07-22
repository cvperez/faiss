/*
 * ivf_cte_ingest — copy a FAISS IVF's inverted lists into IOWarp CTE.
 *
 * Writes one blob "list/<i>" per non-empty list (uint8 codes[size*code_size]
 * immediately followed by int64 ids[size], compact) plus a "sizes" blob
 * (int64[nlist]) under the given tag. This is the storage layout the
 * faiss_ivf ChiMod reads. Requires a running CLIO runtime.
 */

#pragma once

#include <cstddef>
#include <string>

#include <faiss/invlists/InvertedLists.h>

namespace faiss_iowarp {

// Ingest every non-empty inverted list of `src` into CTE under `tag_name`,
// pipelining up to `batch` blob puts in flight. Returns the total number of
// bytes written across the list blobs. Throws on any CTE failure.
size_t IngestIvfToCte(
        const faiss::InvertedLists* src,
        const std::string& tag_name,
        size_t batch = 64);

} // namespace faiss_iowarp
