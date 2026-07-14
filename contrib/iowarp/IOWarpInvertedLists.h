/*
 * IOWarpInvertedLists — faiss::InvertedLists backend over IOWarp CTE.
 *
 * Each non-empty inverted list i is one CTE blob "list/<i>" laid out as
 *   uint8_t codes[size * code_size] ++ idx_t ids[size]        (compact)
 * plus one blob "sizes" = idx_t[nlist] with the list sizes, all under a
 * caller-chosen tag. One blob per list makes CTE's Data Placement Engine
 * place and migrate each list independently — the unit of placement equals
 * the unit of access.
 *
 * Read path: IndexIVF::search calls prefetch_lists() with every probed
 * list before scanning; we turn each into a pipelined AsyncGetBlob. The
 * later get_codes()/get_ids() wait on the in-flight future and pin the
 * fetched buffer; release_codes()/release_ids() unpin and free it. All of
 * list_size/get_codes/get_ids/release_* are called concurrently from OMP
 * threads (FAISS ScopedCodes/ScopedIds), so the fetch table is
 * mutex-guarded and entries are refcounted.
 *
 * Write path: add_entries() (read-modify-write of one blob) exists for
 * selftests and small ingests; bulk ingestion should use ivf_to_iowarp,
 * which puts whole lists directly. update_entries/resize are unsupported.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <faiss/invlists/InvertedLists.h>

#include <clio_cte/core/core_client.h>
#include <clio_runtime/clio_runtime.h>

namespace faiss_iowarp {

// Connect this process to the CLIO runtime + CTE exactly once.
// Safe to call repeatedly from any thread. Returns false on failure.
bool EnsureIOWarpClient();

struct IOWarpInvertedLists : faiss::InvertedLists {
    // Attaches to (or creates) the tag and loads the "sizes" blob if it
    // exists; otherwise starts with all-empty lists.
    IOWarpInvertedLists(
            size_t nlist,
            size_t code_size,
            const std::string& tag_name);
    ~IOWarpInvertedLists() override;

    /*** read interface (thread-safe) ***/
    size_t list_size(size_t list_no) const override;
    const uint8_t* get_codes(size_t list_no) const override;
    const faiss::idx_t* get_ids(size_t list_no) const override;
    void release_codes(size_t list_no, const uint8_t* codes) const override;
    void release_ids(size_t list_no, const faiss::idx_t* ids) const override;
    void prefetch_lists(const faiss::idx_t* list_nos, int n) const override;

    /*** write interface ***/
    size_t add_entries(
            size_t list_no,
            size_t n_entry,
            const faiss::idx_t* ids,
            const uint8_t* code) override;
    void update_entries(
            size_t list_no,
            size_t offset,
            size_t n_entry,
            const faiss::idx_t* ids,
            const uint8_t* code) override;
    void resize(size_t list_no, size_t new_size) override;

    // Writes the "sizes" blob. Call once after a sequence of add_entries.
    void persist_sizes();

    // Total entries across lists (for setting index->ntotal).
    size_t compute_total() const;

    const std::string& tag_name() const {
        return tag_name_;
    }

  private:
    struct PendingFetch {
        ctp::ipc::FullPtr<char> buf;
        chi::Future<clio::cte::core::GetBlobTask> fut;
        std::once_flag arrived; // first accessor Waits, others block on it
        int refs = 0;
    };

    size_t list_bytes(size_t list_no) const {
        return sizes_[list_no] * (code_size + sizeof(faiss::idx_t));
    }
    static std::string blob_name(size_t list_no);

    // Returns a pinned (refs incremented), completed fetch entry.
    PendingFetch* acquire(size_t list_no) const;
    void unpin(size_t list_no) const;

    std::string tag_name_;
    clio::cte::core::TagId tag_id_;
    std::vector<size_t> sizes_;

    mutable std::mutex mu_;
    mutable std::unordered_map<size_t, std::unique_ptr<PendingFetch>>
            fetches_;
};

} // namespace faiss_iowarp
