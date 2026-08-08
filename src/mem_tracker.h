// factor-cuda -- mem_tracker: runtime high-water-mark allocator.
//
// PoC 3 memory model infrastructure. Wraps cudaMalloc/cudaFree and tracks
// live bytes / peak (HWM) / per-buffer event log, aligned to 256 B to match
// the theoretical Timeline in benchmarks/compute_workspace_v1.py (align256).
// Calibration uses three points of view (protocol Sec 3.3):
//   theoretical formula  vs  tracker HWM  (expect ~0)
//   tracker HWM          vs  cudaMemGetInfo sampling (expect ~driver overhead)
//
// ASCII-only comments (nvcc/GBK pitfall).
#ifndef FACTOR_CUDA_MEM_TRACKER_H_
#define FACTOR_CUDA_MEM_TRACKER_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include <cuda_runtime.h>

namespace factor_cuda {

constexpr size_t kTrackAlignBytes = 256;  // matches Timeline.align256

struct MemEvent {
  size_t index;
  std::string stage;     // allocator call site stage label (e.g. "alloc", "cub_temp")
  std::string action;    // "alloc" | "free"
  std::string name;      // buffer name
  size_t logical_bytes;
  size_t aligned_bytes;
  size_t live_after;     // live bytes after this event (HWM accounting)
};

class MemTracker {
 public:
  MemTracker() = default;
  // F6: copying/moving would duplicate bookkeeping over the same device
  // pointers, so either copy could free an object the other still accounts.
  MemTracker(const MemTracker&) = delete;
  MemTracker& operator=(const MemTracker&) = delete;
  MemTracker(MemTracker&&) = delete;
  MemTracker& operator=(MemTracker&&) = delete;

  // Allocate bytes on the current device; records the event. ptr/bytes mirrors
  // cudaMalloc semantics. Returns cudaSuccess or the cudaMalloc error.
  cudaError_t Alloc(void** ptr, size_t bytes, const char* name, const char* stage);

  // Free a pointer previously Alloc'd by this tracker. nullptr is a safe no-op.
  // F1/F2 strict: an unknown NON-null pointer is an error (cudaErrorInvalidValue)
  // and is counted in unknown_free_count() -- it is never silently freed, so a
  // missed AllocOrTrack cannot corrupt the HWM silently. Returns cudaSuccess or
  // the cudaFree/ownership error.
  cudaError_t Free(void* ptr);

  size_t live_bytes() const { return live_; }
  size_t peak_live_bytes() const { return peak_; }
  size_t alloc_count() const { return alloc_count_; }
  size_t unknown_free_count() const { return unknown_free_count_; }  // F1/F2 diag
  const std::vector<MemEvent>& events() const { return events_; }

 private:
  struct ActiveInfo {
    std::string name;
    size_t logical_bytes;  // F3: keep both so the free event reports the truth
    size_t aligned_bytes;
  };

  static bool checked_add_ok(size_t a, size_t b, size_t* out);  // F5

  size_t live_ = 0;
  size_t peak_ = 0;
  size_t alloc_count_ = 0;
  size_t unknown_free_count_ = 0;
  std::unordered_map<void*, ActiveInfo> active_;
  std::vector<MemEvent> events_;
};

}  // namespace factor_cuda

#endif  // FACTOR_CUDA_MEM_TRACKER_H_
