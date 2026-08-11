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
#include <functional>
#include <string>
#include <vector>

#include <faiss/invlists/InvertedLists.h>

namespace faiss_iowarp {

// Ingest every non-empty inverted list of `src` into CTE under `tag_name`,
// pipelining up to `batch` blob puts in flight. Returns the total number of
// bytes written across the list blobs. Throws on any CTE failure.
size_t IngestIvfToCte(
        const faiss::InvertedLists* src,
        const std::string& tag_name,
        size_t batch = 64);

// Multi-shard variant: the shards must share nlist/code_size (same trained
// quantizer, globally-unique ids). Blob "list/<i>" holds the codes of every
// shard back-to-back followed by the ids of every shard back-to-back — the
// same single codes||ids split the ChiMod's Search parses, so ingesting R
// shards is search-equivalent to ingesting their merged index.
//
// list_filter (optional): ingest only lists for which it returns true —
// the owner-local multi-node ingest runs one ingest per node with a
// "this node owns list l" filter, so every put is node-local (a head-node
// ingest pushes (N-1)/N of the bytes cross-node and OOMs peer daemons
// after ~24 GB/node of network-moved data — upstream recv-staging growth).
// write_sizes: whether to write the final "sizes" blob (exactly ONE
// ingest — rank 0 — must, with the filter ignored for its contents).
size_t IngestIvfShardsToCte(
        const std::vector<const faiss::InvertedLists*>& srcs,
        const std::string& tag_name,
        size_t batch = 64,
        const std::function<bool(int64_t)>& list_filter = {},
        bool write_sizes = true);

} // namespace faiss_iowarp
