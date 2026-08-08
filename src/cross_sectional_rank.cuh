// factor-cuda -- cross_sectional_rank v0 GPU kernel (host interface).
//
// Stable ordinal cross-sectional rank (CLAUDE.md Sec 1 cross_sectional_rank).
// Key transform is the single source of truth tests/fixtures/corr_math_v1.py
// (canonical_ordinal_key_f32). Ascending CUB segmented radix sort on key;
// non-finite -> 0xFFFFFFFF sentinel sorts last (excluded from ranking).
// Descending == ascending of negated values (contract direction).
//
// PoC 3 v0 kernel -- first real CUDA kernel of the memory model milestone.
// ASCII-only comments (nvcc/GBK pitfall).
#ifndef FACTOR_CUDA_CROSS_SECTIONAL_RANK_CUH_
#define FACTOR_CUDA_CROSS_SECTIONAL_RANK_CUH_

#include <cstdint>
#include <cstddef>

#include "mem_tracker.h"

// Cached device-buffer workspace for cs_rank_gpu (PoC 3 perf, 2026-08-05).
// Reusing one workspace across calls with the same panel shape (T,N) removes
// the per-call cudaMalloc/cudaFree overhead (measured ~7.7 ms at corpus scale
// 1218x5000 of a ~16.4 ms end-to-end). The workspace owns its buffers; call
// cs_rank_workspace_clear to release them (e.g. so a leak check that samples
// cudaMemGetInfo after the run sees them freed).
//
// Ownership and lifetime (review F01/F02/F06/F07, 2026-08-05):
//   - The workspace is bound to the CUDA device ordinal and MemTracker (or
//     nullptr) active when its buffers were (re)allocated. A later call with a
//     different device or tracker triggers a clear + re-allocation.
//   - cs_rank_workspace_clear always frees via the owner tracker recorded at
//     allocation time (idempotent; cudaFree/tracker Free are nullptr no-ops).
//   - The mask buffer is optional growing capacity: lazily allocated on the
//     first masked call and kept after, so alternating mask on/off does not
//     discard the whole workspace.
//   - Non-copyable / non-movable (owning raw-pointer aggregate; copying would
//     create two owners -> double free). Not safe for concurrent use from
//     multiple threads.
struct cs_rank_workspace {
  cs_rank_workspace() = default;
  cs_rank_workspace(const cs_rank_workspace&) = delete;
  cs_rank_workspace& operator=(const cs_rank_workspace&) = delete;

  uint8_t* d_valid = nullptr;
  uint32_t* d_keys_cur = nullptr;
  uint32_t* d_keys_alt = nullptr;
  uint32_t* d_vals_cur = nullptr;
  uint32_t* d_vals_alt = nullptr;
  int32_t* d_offsets = nullptr;
  float* d_X = nullptr;
  uint8_t* d_mask = nullptr;   // optional capacity: lazily allocated on first masked call, kept
  float* d_out = nullptr;
  void* d_temp = nullptr;      // CUB temp buffer (size queried once at acquire)
  size_t temp_bytes = 0;
  int64_t n_items = -1;        // shape key: T*N
  int N = -1;                  // shape key: N
  int device = -1;             // shape key: CUDA device ordinal buffers were allocated on
  factor_cuda::MemTracker* owner_tracker = nullptr;  // tracker buffers were allocated through
};

// Release all device buffers cached in the workspace (idempotent; safe on an
// empty / never-used workspace). Buffers are freed via the owner tracker
// recorded at allocation time. After this the workspace is reusable (next call
// re-allocates).
void cs_rank_workspace_clear(cs_rank_workspace* ws);

// Compute stable ordinal rank of a (T,N) float32 panel.
//   h_X    : (T,N) float32, C-contiguous (row-major), host.
//   h_mask : (T,N) uint8 bool (1=participate) or nullptr (=all finite).
//   T, N   : panel dims. Preconditions: T>=1, N>=1, N<=2^24, T*N<=INT32_MAX.
//   descending : if true, rank is ascending of negated values (rank 1 = max).
//   h_out  : (T,N) float32 output. Valid cell -> exact integer rank 1..K;
//            invalid cell -> quiet NaN payload 0x7fc00000.
//   tracker: optional MemTracker; if non-null, all device allocations route
//            through it (records live/peak/events). Default nullptr = plain
//            cudaMalloc/cudaFree. When ws != nullptr the workspace owns its
//            buffers persistently, so a tracker counts them as live until
//            cs_rank_workspace_clear -- do not combine with a tracker if you
//            assert "final live == 0" at the end of a measurement. A change of
//            tracker between calls on the same workspace re-allocates it.
//   ws     : optional workspace. When provided (and the panel shape T,N and the
//            device / tracker match the cached key), device buffers are reused
//            across calls -- no per-call malloc/free. On shape/device/tracker
//            mismatch the workspace is cleared and re-allocated. nullptr =
//            allocate and free every call (previous behavior).
// Returns 0 on success, otherwise nonzero (cudaError_t cast to int).
//
// 7-parameter form: original source AND binary ABI (review F09). An out-of-line
// overload forwarding to the 8-parameter form (ws = nullptr). Kept so object
// files compiled against the pre-workspace header still link.
int cs_rank_gpu(const float* h_X, const uint8_t* h_mask, int T, int N,
                bool descending, float* h_out,
                factor_cuda::MemTracker* tracker = nullptr);

// 8-parameter form: workspace-enabled (see ws above). ws has no default so its
// presence selects the cached path; the 7-parameter overload covers old callers.
int cs_rank_gpu(const float* h_X, const uint8_t* h_mask, int T, int N,
                bool descending, float* h_out,
                factor_cuda::MemTracker* tracker,
                cs_rank_workspace* ws);

#endif  // FACTOR_CUDA_CROSS_SECTIONAL_RANK_CUH_
