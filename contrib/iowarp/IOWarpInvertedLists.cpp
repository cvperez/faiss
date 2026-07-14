#include "IOWarpInvertedLists.h"

#include <chrono>
#include <cstring>
#include <thread>

#include <faiss/impl/FaissAssert.h>

namespace faiss_iowarp {

using clio::cte::core::TagId;

bool EnsureIOWarpClient() {
    static bool ok = [] {
        if (!chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, false)) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        return clio::cte::core::CLIO_CTE_CLIENT_INIT();
    }();
    return ok;
}

IOWarpInvertedLists::IOWarpInvertedLists(
        size_t nlist,
        size_t code_size,
        const std::string& tag_name)
        : faiss::InvertedLists(nlist, code_size),
          tag_name_(tag_name),
          sizes_(nlist, 0) {
    FAISS_THROW_IF_NOT_MSG(
            EnsureIOWarpClient(), "IOWarp client initialization failed");
    auto* cte = CLIO_CTE_CLIENT;

    auto tag_fut = cte->AsyncGetOrCreateTag(tag_name_);
    tag_fut.Wait();
    tag_id_ = tag_fut->tag_id_;

    // Load "sizes" if this tag was already populated.
    auto sz_fut = cte->AsyncGetBlobSize(tag_id_, "sizes");
    sz_fut.Wait();
    if (sz_fut->return_code_.load() == 0 && sz_fut->size_ > 0) {
        size_t bytes = nlist * sizeof(faiss::idx_t);
        FAISS_THROW_IF_NOT_FMT(
                sz_fut->size_ == bytes,
                "sizes blob is %zu bytes, expected %zu (nlist mismatch?)",
                (size_t)sz_fut->size_,
                bytes);
        auto buf = CLIO_IPC->AllocateBuffer(bytes);
        ctp::ipc::ShmPtr<> p = buf.shm_.template Cast<void>();
        auto get_fut = cte->AsyncGetBlob(tag_id_, "sizes", 0, bytes, 0, p);
        get_fut.Wait();
        FAISS_THROW_IF_NOT_MSG(
                get_fut->return_code_.load() == 0, "failed to read sizes");
        auto* stored = reinterpret_cast<const faiss::idx_t*>(buf.ptr_);
        for (size_t i = 0; i < nlist; ++i) {
            sizes_[i] = static_cast<size_t>(stored[i]);
        }
        CLIO_IPC->FreeBuffer(buf);
    }
}

IOWarpInvertedLists::~IOWarpInvertedLists() {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& [list_no, pf] : fetches_) {
        pf->fut.Wait();
        CLIO_IPC->FreeBuffer(pf->buf);
    }
    fetches_.clear();
}

std::string IOWarpInvertedLists::blob_name(size_t list_no) {
    return "list/" + std::to_string(list_no);
}

size_t IOWarpInvertedLists::list_size(size_t list_no) const {
    return sizes_[list_no];
}

void IOWarpInvertedLists::prefetch_lists(const faiss::idx_t* list_nos, int n)
        const {
    auto* cte = CLIO_CTE_CLIENT;
    std::lock_guard<std::mutex> lk(mu_);
    for (int i = 0; i < n; ++i) {
        faiss::idx_t key = list_nos[i];
        if (key < 0 || sizes_[key] == 0) {
            continue;
        }
        size_t l = static_cast<size_t>(key);
        if (fetches_.count(l)) {
            continue;
        }
        auto pf = std::make_unique<PendingFetch>();
        size_t bytes = list_bytes(l);
        pf->buf = CLIO_IPC->AllocateBuffer(bytes);
        ctp::ipc::ShmPtr<> p = pf->buf.shm_.template Cast<void>();
        pf->fut = cte->AsyncGetBlob(tag_id_, blob_name(l), 0, bytes, 0, p);
        fetches_.emplace(l, std::move(pf));
    }
}

IOWarpInvertedLists::PendingFetch* IOWarpInvertedLists::acquire(size_t list_no)
        const {
    PendingFetch* pf = nullptr;
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = fetches_.find(list_no);
        if (it == fetches_.end()) {
            // Cold access without prefetch: issue the fetch now.
            auto fresh = std::make_unique<PendingFetch>();
            size_t bytes = list_bytes(list_no);
            fresh->buf = CLIO_IPC->AllocateBuffer(bytes);
            ctp::ipc::ShmPtr<> p = fresh->buf.shm_.template Cast<void>();
            fresh->fut = CLIO_CTE_CLIENT->AsyncGetBlob(
                    tag_id_, blob_name(list_no), 0, bytes, 0, p);
            it = fetches_.emplace(list_no, std::move(fresh)).first;
        }
        it->second->refs++;
        pf = it->second.get(); // stable: owned by unique_ptr, erased only
                               // when refs drops to 0
    }
    // Complete the fetch outside the map lock; concurrent acquirers of the
    // same list block on the once_flag, not on the map.
    std::call_once(pf->arrived, [pf] {
        pf->fut.Wait();
        FAISS_THROW_IF_NOT_MSG(
                pf->fut->return_code_.load() == 0, "CTE GetBlob failed");
    });
    return pf;
}

void IOWarpInvertedLists::unpin(size_t list_no) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = fetches_.find(list_no);
    FAISS_THROW_IF_NOT(it != fetches_.end());
    if (--it->second->refs == 0) {
        CLIO_IPC->FreeBuffer(it->second->buf);
        fetches_.erase(it);
    }
}

const uint8_t* IOWarpInvertedLists::get_codes(size_t list_no) const {
    if (sizes_[list_no] == 0) {
        return nullptr;
    }
    PendingFetch* pf = acquire(list_no);
    return reinterpret_cast<const uint8_t*>(pf->buf.ptr_);
}

const faiss::idx_t* IOWarpInvertedLists::get_ids(size_t list_no) const {
    if (sizes_[list_no] == 0) {
        return nullptr;
    }
    PendingFetch* pf = acquire(list_no);
    return reinterpret_cast<const faiss::idx_t*>(
            pf->buf.ptr_ + sizes_[list_no] * code_size);
}

void IOWarpInvertedLists::release_codes(size_t list_no, const uint8_t* codes)
        const {
    if (codes == nullptr) {
        return;
    }
    unpin(list_no);
}

void IOWarpInvertedLists::release_ids(
        size_t list_no,
        const faiss::idx_t* ids) const {
    if (ids == nullptr) {
        return;
    }
    unpin(list_no);
}

size_t IOWarpInvertedLists::add_entries(
        size_t list_no,
        size_t n_entry,
        const faiss::idx_t* ids,
        const uint8_t* code) {
    if (n_entry == 0) {
        return 0;
    }
    FAISS_THROW_IF_NOT(list_no < nlist);
    auto* cte = CLIO_CTE_CLIENT;

    size_t old_size = sizes_[list_no];
    size_t new_size = old_size + n_entry;
    size_t new_bytes = new_size * (code_size + sizeof(faiss::idx_t));

    auto buf = CLIO_IPC->AllocateBuffer(new_bytes);
    uint8_t* codes_dst = reinterpret_cast<uint8_t*>(buf.ptr_);
    faiss::idx_t* ids_dst =
            reinterpret_cast<faiss::idx_t*>(buf.ptr_ + new_size * code_size);

    if (old_size > 0) {
        // Read-modify-write: fetch the current blob, splice codes/ids.
        size_t old_bytes = old_size * (code_size + sizeof(faiss::idx_t));
        auto old_buf = CLIO_IPC->AllocateBuffer(old_bytes);
        ctp::ipc::ShmPtr<> op = old_buf.shm_.template Cast<void>();
        auto gf = cte->AsyncGetBlob(
                tag_id_, blob_name(list_no), 0, old_bytes, 0, op);
        gf.Wait();
        FAISS_THROW_IF_NOT_MSG(
                gf->return_code_.load() == 0, "CTE GetBlob failed in add");
        std::memcpy(codes_dst, old_buf.ptr_, old_size * code_size);
        std::memcpy(
                ids_dst,
                old_buf.ptr_ + old_size * code_size,
                old_size * sizeof(faiss::idx_t));
        CLIO_IPC->FreeBuffer(old_buf);
    }
    std::memcpy(codes_dst + old_size * code_size, code, n_entry * code_size);
    std::memcpy(ids_dst + old_size, ids, n_entry * sizeof(faiss::idx_t));

    ctp::ipc::ShmPtr<> p = buf.shm_.template Cast<void>();
    auto pf = cte->AsyncPutBlob(tag_id_, blob_name(list_no), 0, new_bytes, p);
    pf.Wait();
    FAISS_THROW_IF_NOT_MSG(
            pf->return_code_.load() == 0, "CTE PutBlob failed in add");
    CLIO_IPC->FreeBuffer(buf);

    sizes_[list_no] = new_size;
    return old_size;
}

void IOWarpInvertedLists::update_entries(
        size_t,
        size_t,
        size_t,
        const faiss::idx_t*,
        const uint8_t*) {
    FAISS_THROW_MSG("update_entries not supported by IOWarpInvertedLists");
}

void IOWarpInvertedLists::resize(size_t, size_t) {
    FAISS_THROW_MSG("resize not supported by IOWarpInvertedLists");
}

void IOWarpInvertedLists::persist_sizes() {
    auto* cte = CLIO_CTE_CLIENT;
    size_t bytes = nlist * sizeof(faiss::idx_t);
    auto buf = CLIO_IPC->AllocateBuffer(bytes);
    auto* dst = reinterpret_cast<faiss::idx_t*>(buf.ptr_);
    for (size_t i = 0; i < nlist; ++i) {
        dst[i] = static_cast<faiss::idx_t>(sizes_[i]);
    }
    ctp::ipc::ShmPtr<> p = buf.shm_.template Cast<void>();
    auto pf = cte->AsyncPutBlob(tag_id_, "sizes", 0, bytes, p);
    pf.Wait();
    FAISS_THROW_IF_NOT_MSG(
            pf->return_code_.load() == 0, "CTE PutBlob(sizes) failed");
    CLIO_IPC->FreeBuffer(buf);
}

size_t IOWarpInvertedLists::compute_total() const {
    size_t total = 0;
    for (size_t s : sizes_) {
        total += s;
    }
    return total;
}

} // namespace faiss_iowarp
