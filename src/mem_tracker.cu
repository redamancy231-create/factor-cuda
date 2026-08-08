// factor-cuda -- mem_tracker runtime HWM allocator (see mem_tracker.h).
// ASCII-only comments (nvcc/GBK pitfall).
#include <utility>

#include "mem_tracker.h"

namespace factor_cuda {

bool MemTracker::checked_add_ok(size_t a, size_t b, size_t* out) {
  if (a > SIZE_MAX - b) return false;  // F5
  *out = a + b;
  return true;
}

cudaError_t MemTracker::Alloc(void** ptr, size_t bytes, const char* name,
                              const char* stage) {
  if (ptr == nullptr) return cudaErrorInvalidValue;
  if (bytes > SIZE_MAX - (kTrackAlignBytes - 1)) return cudaErrorInvalidValue;  // F5
  cudaError_t err = cudaMalloc(ptr, bytes);
  if (err != cudaSuccess) return err;
  const size_t aligned =
      ((bytes + kTrackAlignBytes - 1) / kTrackAlignBytes) * kTrackAlignBytes;
  // F4: exception-safe commit. Everything that can throw (string/hash/vector
  // allocations) happens inside the try; on throw, roll the device allocation
  // back so no leak and no half-updated bookkeeping escapes.
  try {
    size_t new_live = 0;
    if (!checked_add_ok(live_, aligned, &new_live)) {  // F5
      cudaFree(*ptr);
      *ptr = nullptr;
      return cudaErrorInvalidValue;
    }
    ActiveInfo info{name == nullptr ? std::string("?") : std::string(name), bytes,
                    aligned};
    active_.emplace(*ptr, std::move(info));
    events_.push_back(MemEvent{events_.size(), stage == nullptr ? "" : std::string(stage),
                               "alloc", active_[*ptr].name, bytes, aligned, new_live});
    live_ = new_live;
    if (live_ > peak_) peak_ = live_;
    ++alloc_count_;
  } catch (...) {
    cudaFree(*ptr);
    *ptr = nullptr;
    return cudaErrorMemoryAllocation;
  }
  return cudaSuccess;
}

cudaError_t MemTracker::Free(void* ptr) {
  if (ptr == nullptr) return cudaSuccess;  // cudaFree(nullptr) is a no-op
  auto it = active_.find(ptr);
  if (it == active_.end()) {
    // F1/F2 strict: an unknown non-null pointer is never silently freed. This
    // surfaces a missed AllocOrTrack (or double free) instead of corrupting the
    // HWM. Callers that intentionally mix ownership must track everything.
    ++unknown_free_count_;
    return cudaErrorInvalidValue;
  }
  const ActiveInfo info = it->second;
  cudaError_t err = cudaFree(ptr);
  if (err != cudaSuccess) return err;
  live_ -= info.aligned_bytes;
  active_.erase(it);
  // F3: free event reports logical and aligned bytes separately (was aligned twice).
  events_.push_back(MemEvent{events_.size(), "free", "free", info.name,
                             info.logical_bytes, info.aligned_bytes, live_});
  return cudaSuccess;
}

}  // namespace factor_cuda
