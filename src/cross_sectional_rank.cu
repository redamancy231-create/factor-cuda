// factor-cuda -- cross_sectional_rank v0 GPU kernel.
//
// 3-kernel pipeline:
//   1. preprocess_kernel : valid flag + canonical ordinal key (corr_math_v1.py
//      single source of truth) + payload = column index. Descending negates
//      the value first (contract: desc = ascending of negated). Non-finite ->
//      0xFFFFFFFF sentinel (sorts last, never participates).
//   2. cub::DeviceSegmentedRadixSort::SortPairs DoubleBuffer -- stable
//      ascending sort over regular segments [t*N, (t+1)*N). Valid cells
//      occupy the first K positions of each segment because invalid keys
//      (0xFFFFFFFF) sort last.
//   3. scatter_rank_kernel : rank = segment-local position + 1, scattered by
//      payload column index; invalid cell -> quiet NaN 0x7fc00000.
//
// PoC 3 v0 -- first real CUDA kernel of the memory model milestone.
// ASCII-only comments (nvcc/GBK pitfall).
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cub/device/device_segmented_radix_sort.cuh>

#include "cross_sectional_rank.cuh"
#include "parameter_scan.cuh"

namespace {

constexpr int kMaxN = (1 << 24);        // N > 2^24 -> rank not exactly representable
constexpr int kMaxTotal = INT32_MAX;    // T*N <= INT32_MAX (checked_mul)
constexpr uint32_t kInvalidKey = 0xFFFFFFFFu;
constexpr uint32_t kQuietNanPayload = 0x7fc00000u;

// canonical_ordinal_key_f32 -- mirrors tests/fixtures/corr_math_v1.py:189.
// Ascending key order == ascending numeric order; +0.0/-0.0 fold to one tie key.
__device__ __forceinline__ uint32_t canonical_ordinal_key_f32(float v) {
  if (!isfinite(v)) return kInvalidKey;
  uint32_t bits = (v == 0.0f) ? 0u : __float_as_uint(v);
  return (bits & 0x80000000u) ? (~bits) : (bits | 0x80000000u);
}

// Build sort key + column-index payload + valid flag for every cell.
__global__ void preprocess_kernel(const float* __restrict__ X,
                                  const uint8_t* __restrict__ mask, int N,
                                  bool descending, int total,
                                  uint32_t* __restrict__ d_keys,
                                  uint32_t* __restrict__ d_values,
                                  uint8_t* __restrict__ d_valid) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  float v = X[i];
  if (descending) v = -v;
  uint8_t valid = static_cast<uint8_t>(isfinite(v) && (mask == nullptr || mask[i]));
  d_valid[i] = valid;
  d_keys[i] = valid ? canonical_ordinal_key_f32(v) : kInvalidKey;
  d_values[i] = static_cast<uint32_t>(i % N);
}

// Scatter rank from sorted payload (column index) back to output. Valid cells
// occupy the first K positions of each segment, so rank == pos - t*N + 1.
// Each output cell is written exactly once (payload is a per-segment permutation).
__global__ void scatter_rank_kernel(const uint32_t* __restrict__ sorted_values,
                                    const uint8_t* __restrict__ d_valid, int N,
                                    int total, float* __restrict__ out) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= total) return;
  int t = p / N;
  int rank = p - t * N + 1;
  int col = static_cast<int>(sorted_values[p]);
  int out_idx = t * N + col;
  out[out_idx] = d_valid[out_idx] ? static_cast<float>(rank)
                                  : __uint_as_float(kQuietNanPayload);
}

cudaError_t LaunchPreprocess(const float* d_X, const uint8_t* d_mask, int N,
                             bool descending, int total, uint32_t* d_keys,
                             uint32_t* d_values, uint8_t* d_valid) {
  const int block = 256;
  // int64 arithmetic: total may be INT32_MAX (T*N guard), total+block-1 would
  // overflow signed int. total >= 1 guaranteed by host preconditions.
  const int grid = static_cast<int>(1 + (static_cast<int64_t>(total) - 1) / block);
  preprocess_kernel<<<grid, block>>>(d_X, d_mask, N, descending, total, d_keys,
                                     d_values, d_valid);
  return cudaGetLastError();
}

cudaError_t LaunchScatter(const uint32_t* d_sorted_values, const uint8_t* d_valid,
                          int N, int total, float* d_out) {
  const int block = 256;
  const int grid = static_cast<int>(1 + (static_cast<int64_t>(total) - 1) / block);
  scatter_rank_kernel<<<grid, block>>>(d_sorted_values, d_valid, N, total, d_out);
  return cudaGetLastError();
}

// Fill segment offsets on device: d_offsets[t] = t*N for t in [0, T].
// Avoids an 8GB host std::vector for the legal-but-extreme T=INT32_MAX input
// (review finding 2): an oversized device allocation is a checked cudaMalloc
// error, not an uncaught host bad_alloc. int64 loop avoids t==INT32_MAX wrap.
__global__ void make_offsets_kernel(int32_t* __restrict__ d_offsets, int T, int N) {
  int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t > static_cast<int64_t>(T)) return;
  d_offsets[t] = static_cast<int32_t>(t * N);  // t*N <= T*N <= INT32_MAX
}

cudaError_t LaunchMakeOffsets(int32_t* d_offsets, int T, int N) {
  const int block = 256;
  // cover T+1 elements; T*N<=INT32_MAX so T may reach INT32_MAX, use int64 grid
  const int64_t need = static_cast<int64_t>(T) + 1;
  const int64_t grid64 = 1 + (need - 1) / block;
  const int grid = grid64 > INT32_MAX ? INT32_MAX : static_cast<int>(grid64);
  make_offsets_kernel<<<grid, block>>>(d_offsets, T, N);
  return cudaGetLastError();
}

// Route alloc/free through the tracker when provided, else plain CUDA calls.
// Template so typed T** callers work (cudaMalloc is a C API and accepts void**).
template <typename T>
cudaError_t AllocOrTrack(T** ptr, size_t bytes, const char* name, const char* stage,
                         factor_cuda::MemTracker* t) {
  return t != nullptr ? t->Alloc(reinterpret_cast<void**>(ptr), bytes, name, stage)
                      : cudaMalloc(reinterpret_cast<void**>(ptr), bytes);
}

cudaError_t FreeOrTrack(void* ptr, factor_cuda::MemTracker* t) {
  return t != nullptr ? t->Free(ptr) : cudaFree(ptr);
}

// Free all device buffers, returning the first non-success error encountered.
cudaError_t FreeAllBuffers(uint8_t* d_valid, uint32_t* d_keys_cur, uint32_t* d_keys_alt,
                           uint32_t* d_vals_cur, uint32_t* d_vals_alt, int32_t* d_offsets,
                           float* d_X, uint8_t* d_mask, float* d_out, void* d_temp,
                           factor_cuda::MemTracker* tracker) {
  cudaError_t e = cudaSuccess;
  auto keep_first = [&e](cudaError_t r) {
    if (e == cudaSuccess && r != cudaSuccess) e = r;
  };
  keep_first(FreeOrTrack(d_valid, tracker));
  keep_first(FreeOrTrack(d_keys_cur, tracker));
  keep_first(FreeOrTrack(d_keys_alt, tracker));
  keep_first(FreeOrTrack(d_vals_cur, tracker));
  keep_first(FreeOrTrack(d_vals_alt, tracker));
  keep_first(FreeOrTrack(d_offsets, tracker));
  keep_first(FreeOrTrack(d_X, tracker));
  keep_first(FreeOrTrack(d_mask, tracker));  // nullptr is a no-op, safe
  keep_first(FreeOrTrack(d_out, tracker));
  keep_first(FreeOrTrack(d_temp, tracker));
  return e;
}

// Query (d_temp == nullptr) or execute (d_temp != nullptr) the CUB segmented
// radix sort on DoubleBuffer pairs. On execute, reports the sorted values
// buffer (d_vals.Current()) via out_values_current.
cudaError_t RunCubSort(void* d_temp, size_t& temp_bytes, uint32_t* keys_cur,
                       uint32_t* keys_alt, uint32_t* vals_cur, uint32_t* vals_alt,
                       const int32_t* d_offsets, int64_t total, int64_t T,
                       const uint32_t** out_values_current) {
  cub::DoubleBuffer<uint32_t> d_keys(keys_cur, keys_alt);
  cub::DoubleBuffer<uint32_t> d_vals(vals_cur, vals_alt);
  cudaError_t err = cub::DeviceSegmentedRadixSort::SortPairs(
      d_temp, temp_bytes, d_keys, d_vals, total, T, d_offsets, d_offsets + 1);
  if (err == cudaSuccess && out_values_current != nullptr) {
    *out_values_current = d_vals.Current();
  }
  return err;
}

// Allocate every core device buffer into the workspace (after a shape/device/
// tracker mismatch) and query + allocate the CUB temp buffer (size queried
// once). The mask buffer is NOT allocated here -- it is optional growing
// capacity, allocated lazily on the first masked call (review F02). On failure
// the workspace is cleared and its shape key reset so a retry starts clean.
cudaError_t AllocWorkspace(cs_rank_workspace* ws, factor_cuda::MemTracker* tracker,
                           int total, int T, int N) {
  const size_t n_items = static_cast<size_t>(total);
  cudaError_t err = AllocOrTrack(&ws->d_valid, n_items, "d_valid", "alloc", tracker);
  if (err == cudaSuccess) err = AllocOrTrack(&ws->d_keys_cur, n_items * sizeof(uint32_t), "d_keys_cur", "alloc", tracker);
  if (err == cudaSuccess) err = AllocOrTrack(&ws->d_keys_alt, n_items * sizeof(uint32_t), "d_keys_alt", "alloc", tracker);
  if (err == cudaSuccess) err = AllocOrTrack(&ws->d_vals_cur, n_items * sizeof(uint32_t), "d_vals_cur", "alloc", tracker);
  if (err == cudaSuccess) err = AllocOrTrack(&ws->d_vals_alt, n_items * sizeof(uint32_t), "d_vals_alt", "alloc", tracker);
  if (err == cudaSuccess) err = AllocOrTrack(&ws->d_offsets, (static_cast<size_t>(T) + 1) * sizeof(int32_t), "d_offsets", "alloc", tracker);
  if (err == cudaSuccess) err = AllocOrTrack(&ws->d_X, n_items * sizeof(float), "d_X", "alloc", tracker);
  if (err == cudaSuccess) err = AllocOrTrack(&ws->d_out, n_items * sizeof(float), "d_out", "alloc", tracker);
  if (err == cudaSuccess) {
    size_t q = 0;
    err = RunCubSort(nullptr, q, ws->d_keys_cur, ws->d_keys_alt, ws->d_vals_cur,
                     ws->d_vals_alt, ws->d_offsets, static_cast<int64_t>(total),
                     static_cast<int64_t>(T), nullptr);
    if (err == cudaSuccess) {
      // review F03: a q==0 query would leave d_temp null and a later execute
      // would be re-interpreted as a query. Ensure a non-null temp (>=1 byte)
      // so the execute call is a real execute for every legal non-empty sort.
      if (q == 0) q = 1;
      err = AllocOrTrack(&ws->d_temp, q, "cub_temp", "cub_temp", tracker);
      if (err == cudaSuccess) ws->temp_bytes = q;
    }
  }
  if (err != cudaSuccess) {
    cs_rank_workspace_clear(ws);  // release any partially-allocated buffers
    return err;
  }
  return cudaSuccess;
}

}  // namespace

void cs_rank_workspace_clear(cs_rank_workspace* ws) {
  if (ws == nullptr) return;
  // Always free via the owner tracker recorded at allocation time (review F07):
  // release routing must not depend on what the caller passes today.
  FreeAllBuffers(ws->d_valid, ws->d_keys_cur, ws->d_keys_alt, ws->d_vals_cur,
                 ws->d_vals_alt, ws->d_offsets, ws->d_X, ws->d_mask, ws->d_out,
                 ws->d_temp, ws->owner_tracker);
  ws->d_valid = nullptr; ws->d_keys_cur = nullptr; ws->d_keys_alt = nullptr;
  ws->d_vals_cur = nullptr; ws->d_vals_alt = nullptr; ws->d_offsets = nullptr;
  ws->d_X = nullptr; ws->d_mask = nullptr; ws->d_out = nullptr;
  ws->d_temp = nullptr; ws->temp_bytes = 0;
  ws->n_items = -1; ws->N = -1; ws->device = -1; ws->owner_tracker = nullptr;
}

// 7-parameter form: original source + binary ABI (review F09). Forwards to the
// 8-parameter form with ws = nullptr (allocate/free every call).
int cs_rank_gpu(const float* h_X, const uint8_t* h_mask, int T, int N,
                bool descending, float* h_out,
                factor_cuda::MemTracker* tracker) {
  return cs_rank_gpu(h_X, h_mask, T, N, descending, h_out, tracker, nullptr);
}

// 8-parameter form: workspace-enabled.
int cs_rank_gpu(const float* h_X, const uint8_t* h_mask, int T, int N,
                bool descending, float* h_out,
                factor_cuda::MemTracker* tracker,
                cs_rank_workspace* ws) {
  // ---- host preconditions (contract errors) --------------------------------
  if (h_X == nullptr || h_out == nullptr || T < 1 || N < 1) return -1;
  if (N > kMaxN) return -2;                     // N > 2^24
  if (static_cast<int64_t>(T) * N > kMaxTotal) return -3;  // checked_mul T*N <= INT32_MAX

  const int total = T * N;
  const size_t n_items = static_cast<size_t>(total);
  const bool need_mask = (h_mask != nullptr);

  // Locals that must survive the goto cleanup path (declared before any goto).
  size_t temp_bytes = 0;
  const uint32_t* values_current = nullptr;
  cudaError_t cleanup = cudaSuccess;

  uint8_t* d_valid = nullptr;
  uint32_t* d_keys_cur = nullptr;
  uint32_t* d_keys_alt = nullptr;
  uint32_t* d_vals_cur = nullptr;
  uint32_t* d_vals_alt = nullptr;
  int32_t* d_offsets = nullptr;
  float* d_X = nullptr;
  uint8_t* d_mask = nullptr;
  float* d_out = nullptr;
  void* d_temp = nullptr;

  cudaError_t err = cudaSuccess;

  if (ws != nullptr) {
    // Workspace path: reuse cached buffers when the shape key matches, else
    // clear + re-allocate (avoids the ~7.7 ms per-call malloc/free overhead).
    // Key = shape (T*N,N) + device ordinal + owner tracker (review F01/F07): a
    // device or tracker change re-allocates so stale / foreign pointers are
    // never reused. Mask presence is NOT part of the key -- d_mask is optional
    // capacity grown lazily below (review F02).
    int cur_dev = -1;
    cudaGetDevice(&cur_dev);
    if (ws->n_items != static_cast<int64_t>(n_items) || ws->N != N ||
        ws->device != cur_dev || ws->owner_tracker != tracker) {
      cs_rank_workspace_clear(ws);  // frees via the previous owner tracker
      ws->n_items = static_cast<int64_t>(n_items);
      ws->N = N;
      ws->device = cur_dev;
      ws->owner_tracker = tracker;
      err = AllocWorkspace(ws, tracker, total, T, N);
      if (err != cudaSuccess) return static_cast<int>(err);  // ws cleared + key reset
    }
    // Lazy mask capacity: allocate once on the first masked call and keep it,
    // so mask on/off alternation does not discard the workspace (review F02).
    // On alloc failure the key stays valid; a masked retry re-allocates the mask.
    if (need_mask && ws->d_mask == nullptr) {
      err = AllocOrTrack(&ws->d_mask, n_items, "d_mask", "alloc", tracker);
      if (err != cudaSuccess) return static_cast<int>(err);
    }
    d_valid = ws->d_valid;
    d_keys_cur = ws->d_keys_cur;
    d_keys_alt = ws->d_keys_alt;
    d_vals_cur = ws->d_vals_cur;
    d_vals_alt = ws->d_vals_alt;
    d_offsets = ws->d_offsets;
    d_X = ws->d_X;
    // Mask capacity may exist from a previous masked call; when THIS call has no
    // mask, pass nullptr so preprocess treats every finite cell as valid (a
    // stale non-null d_mask would wrongly apply old mask bits).
    d_mask = need_mask ? ws->d_mask : nullptr;
    d_out = ws->d_out;
    d_temp = ws->d_temp;       // already queried + allocated
    temp_bytes = ws->temp_bytes;
  } else {
    // Non-workspace path: allocate and free every call (previous behavior).
    err = AllocOrTrack(&d_valid, n_items, "d_valid", "alloc", tracker);
    if (err == cudaSuccess) err = AllocOrTrack(&d_keys_cur, n_items * sizeof(uint32_t), "d_keys_cur", "alloc", tracker);
    if (err == cudaSuccess) err = AllocOrTrack(&d_keys_alt, n_items * sizeof(uint32_t), "d_keys_alt", "alloc", tracker);
    if (err == cudaSuccess) err = AllocOrTrack(&d_vals_cur, n_items * sizeof(uint32_t), "d_vals_cur", "alloc", tracker);
    if (err == cudaSuccess) err = AllocOrTrack(&d_vals_alt, n_items * sizeof(uint32_t), "d_vals_alt", "alloc", tracker);
    if (err == cudaSuccess) err = AllocOrTrack(&d_offsets, (static_cast<size_t>(T) + 1) * sizeof(int32_t), "d_offsets", "alloc", tracker);
    if (err == cudaSuccess) err = AllocOrTrack(&d_X, n_items * sizeof(float), "d_X", "alloc", tracker);
    if (err == cudaSuccess && need_mask) err = AllocOrTrack(&d_mask, n_items, "d_mask", "alloc", tracker);
    if (err == cudaSuccess) err = AllocOrTrack(&d_out, n_items * sizeof(float), "d_out", "alloc", tracker);
    if (err != cudaSuccess) goto fail;
    // CUB temp: query once, then allocate (per call).
    err = RunCubSort(nullptr, temp_bytes, d_keys_cur, d_keys_alt, d_vals_cur,
                     d_vals_alt, d_offsets, static_cast<int64_t>(total),
                     static_cast<int64_t>(T), nullptr);
    if (err != cudaSuccess) goto fail;
    if (temp_bytes == 0) temp_bytes = 1;  // review F03: non-null temp => execute mode
    err = AllocOrTrack(&d_temp, temp_bytes, "cub_temp", "cub_temp", tracker);
    if (err != cudaSuccess) goto fail;
  }

  // ---- upload inputs -------------------------------------------------------
  err = cudaMemcpy(d_X, h_X, n_items * sizeof(float), cudaMemcpyHostToDevice);
  if (err == cudaSuccess && h_mask != nullptr) {
    err = cudaMemcpy(d_mask, h_mask, n_items, cudaMemcpyHostToDevice);
  }
  if (err != cudaSuccess) goto fail;

  // regular segments: begin[t] = t*N, end = begin + 1 (contiguous aliasing).
  // Offsets generated on device -- avoids an 8GB host vector at extreme T and
  // int overflow in a host loop (review finding 2).
  err = LaunchMakeOffsets(d_offsets, T, N);
  if (err != cudaSuccess) goto fail;

  // ---- stage 1: preprocess -------------------------------------------------
  err = LaunchPreprocess(d_X, d_mask, N, descending, total, d_keys_cur, d_vals_cur,
                         d_valid);
  if (err != cudaSuccess) goto fail;

  // ---- stage 2: CUB stable segmented radix sort (DoubleBuffer) ------------
  // Temp buffer is already queried + allocated: in the ws path at acquire, in
  // the non-ws path in the allocate branch above.
  err = RunCubSort(d_temp, temp_bytes, d_keys_cur, d_keys_alt, d_vals_cur,
                   d_vals_alt, d_offsets, static_cast<int64_t>(total),
                   static_cast<int64_t>(T), &values_current);
  if (err != cudaSuccess) goto fail;

  // ---- stage 3: scatter ranks ----------------------------------------------
  err = LaunchScatter(values_current, d_valid, N, total, d_out);
  if (err != cudaSuccess) goto fail;

  // ---- download output -----------------------------------------------------
  err = cudaMemcpy(h_out, d_out, n_items * sizeof(float), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) goto fail;

  if (ws == nullptr) {
    // Non-workspace path: free every call.
    cleanup = FreeAllBuffers(d_valid, d_keys_cur, d_keys_alt, d_vals_cur,
                             d_vals_alt, d_offsets, d_X, d_mask, d_out, d_temp,
                             tracker);
    if (err == cudaSuccess && cleanup != cudaSuccess) err = cleanup;
  }
  // Workspace path: buffers stay cached for the next call; caller clears via
  // cs_rank_workspace_clear when done with the workspace.
  return static_cast<int>(err);

fail:
  if (ws == nullptr) {
    FreeAllBuffers(d_valid, d_keys_cur, d_keys_alt, d_vals_cur, d_vals_alt,
                   d_offsets, d_X, d_mask, d_out, d_temp, tracker);
  }
  return static_cast<int>(err);  // failure path keeps the original primary error
}

// ---- parameter_scan v0 -------------------------------------------------------
//
// parameter_scan over cs_rank axes. G=4 dict order: (asc,masked), (asc,unmasked),
// (desc,masked), (desc,unmasked). Single H2D of X/mask; each group reuses the
// same device buffers (keys/values DoubleBuffer reset each call because
// RunCubSort constructs cub::DoubleBuffer with cur as current). CUB temp is
// queried once and shared across groups (same total/T for every group).
//
// Two-level failure semantics (contract CLAUDE.md Sec 4 failure): the ONLY
// downgrade to a per-group failure is a whitelist cudaError_t
// (InvalidConfiguration / LaunchOutOfResources) caught at the per-group launch
// checkpoint (launches + sync, before D2H). Those groups get group_status = the
// error and result None; remaining groups still run. Every other error
// (setup / upload / offsets / temp, non-whitelist launch or sync error, D2H) is
// a scan-level fatal: return the error, mark unreached groups -100, and the
// caller raises RuntimeError with no partial results.
namespace {

// Contract group-level downgrade whitelist (see header comment).
bool IsGroupWhitelistError(cudaError_t e) {
  return e == cudaErrorInvalidConfiguration || e == cudaErrorLaunchOutOfResources;
}

constexpr int kGroupNotExecuted = -100;  // group_status sentinel (not attempted)

}  // namespace

int parameter_scan_gpu(const float* h_X, const uint8_t* h_mask, int T, int N,
                       float* (&h_out)[4], int (&group_status)[4],
                       factor_cuda::MemTracker* tracker, double* h_time_ms,
                       float* h_time_gpu_ms, const int* h_active,
                       cs_rank_workspace* ws) {
  // Until a group is attempted it is "not executed". Set before the
  // precondition checks so every return path (incl. contract errors -1/-2/-3)
  // leaves all four group_status entries at kGroupNotExecuted.
  for (int g = 0; g < 4; ++g) group_status[g] = kGroupNotExecuted;

  // ---- host preconditions (contract errors) --------------------------------
  if (h_X == nullptr || T < 1 || N < 1) return -1;
  // Active groups require an output buffer; inactive groups (h_active=0) may
  // pass nullptr -- the adapter allocates only the active hosts (review F3).
  for (int g = 0; g < 4; ++g)
    if ((h_active == nullptr || h_active[g]) && h_out[g] == nullptr) return -1;
  if (N > kMaxN) return -2;
  if (static_cast<int64_t>(T) * N > kMaxTotal) return -3;

  const int total = T * N;
  const size_t n_items = static_cast<size_t>(total);

  size_t temp_bytes = 0;
  const uint32_t* values_current = nullptr;
  cudaError_t cleanup = cudaSuccess;
  // Declared here (before any goto fail) so no goto bypasses their
  // initialization (C++ transfer-of-control rule) and the fail path can
  // destroy the events.
  const bool use_timing = (h_time_ms != nullptr) || (h_time_gpu_ms != nullptr);
  cudaEvent_t ev_start = nullptr, ev_stop = nullptr;
  bool ev_created = false;  // event API returns checked (review F2); partial-create cleanup
  cudaError_t cleanup_err = cudaSuccess;  // success-path cleanup error (review F6)
  cudaError_t free_err = cudaSuccess;     // declared at top so goto fail does not bypass init

  uint8_t* d_valid = nullptr;
  uint32_t* d_keys_cur = nullptr;
  uint32_t* d_keys_alt = nullptr;
  uint32_t* d_vals_cur = nullptr;
  uint32_t* d_vals_alt = nullptr;
  int32_t* d_offsets = nullptr;
  float* d_X = nullptr;
  uint8_t* d_mask = nullptr;
  float* d_out = nullptr;
  void* d_temp = nullptr;

  cudaError_t err = cudaSuccess;
  if (ws != nullptr) {
    // Workspace path (P2 PoC4 perf, 2026-08-08): reuse the cached buffers when
    // the shape/device/tracker key matches (same cs_rank_workspace buffer set),
    // else clear + re-allocate -- removes the per-call cudaMalloc/cudaFree
    // overhead on the 4 groups' shared buffers. Mask is optional lazy capacity.
    int cur_dev = -1;
    cudaGetDevice(&cur_dev);
    if (ws->n_items != static_cast<int64_t>(n_items) || ws->N != N ||
        ws->device != cur_dev || ws->owner_tracker != tracker) {
      cs_rank_workspace_clear(ws);
      ws->n_items = static_cast<int64_t>(n_items);
      ws->N = N;
      ws->device = cur_dev;
      ws->owner_tracker = tracker;
      err = AllocWorkspace(ws, tracker, total, T, N);
      if (err != cudaSuccess) return static_cast<int>(err);  // ws cleared + key reset
    }
    if (h_mask != nullptr && ws->d_mask == nullptr) {
      err = AllocOrTrack(&ws->d_mask, n_items, "d_mask", "alloc", tracker);
      if (err != cudaSuccess) return static_cast<int>(err);
    }
    d_valid = ws->d_valid;
    d_keys_cur = ws->d_keys_cur;
    d_keys_alt = ws->d_keys_alt;
    d_vals_cur = ws->d_vals_cur;
    d_vals_alt = ws->d_vals_alt;
    d_offsets = ws->d_offsets;
    d_X = ws->d_X;
    d_mask = (h_mask != nullptr) ? ws->d_mask : nullptr;  // stale-mask guard
    d_out = ws->d_out;
    d_temp = ws->d_temp;       // queried + allocated at acquire
    temp_bytes = ws->temp_bytes;
  } else {
    // Non-workspace path: allocate and free every call (previous behavior).
    err = AllocOrTrack(&d_valid, n_items, "d_valid", "alloc", tracker);
    if (err == cudaSuccess) err = AllocOrTrack(&d_keys_cur, n_items * sizeof(uint32_t), "d_keys_cur", "alloc", tracker);
    if (err == cudaSuccess) err = AllocOrTrack(&d_keys_alt, n_items * sizeof(uint32_t), "d_keys_alt", "alloc", tracker);
    if (err == cudaSuccess) err = AllocOrTrack(&d_vals_cur, n_items * sizeof(uint32_t), "d_vals_cur", "alloc", tracker);
    if (err == cudaSuccess) err = AllocOrTrack(&d_vals_alt, n_items * sizeof(uint32_t), "d_vals_alt", "alloc", tracker);
    if (err == cudaSuccess) err = AllocOrTrack(&d_offsets, (static_cast<size_t>(T) + 1) * sizeof(int32_t), "d_offsets", "alloc", tracker);
    if (err == cudaSuccess) err = AllocOrTrack(&d_X, n_items * sizeof(float), "d_X", "alloc", tracker);
    if (err == cudaSuccess && h_mask != nullptr) err = AllocOrTrack(&d_mask, n_items, "d_mask", "alloc", tracker);
    if (err == cudaSuccess) err = AllocOrTrack(&d_out, n_items * sizeof(float), "d_out", "alloc", tracker);
    if (err != cudaSuccess) goto fail;
  }

  // ---- upload inputs once (single H2D across all groups) --------------------
  err = cudaMemcpy(d_X, h_X, n_items * sizeof(float), cudaMemcpyHostToDevice);
  if (err == cudaSuccess && h_mask != nullptr) {
    err = cudaMemcpy(d_mask, h_mask, n_items, cudaMemcpyHostToDevice);
  }
  if (err != cudaSuccess) goto fail;

  // regular segments + CUB temp size are identical for every group.
  err = LaunchMakeOffsets(d_offsets, T, N);
  if (err != cudaSuccess) goto fail;
  // BLOCKER F1 (ws review): on the workspace path d_temp was queried + allocated
  // at acquire (AllocWorkspace), so skip the query/alloc here -- a second
  // AllocOrTrack would orphan ws->d_temp and leak a fresh temp per call (the ws
  // success/fail paths never free it). Mirrors rolling_ic stage2a.
  if (ws == nullptr) {
    err = RunCubSort(nullptr, temp_bytes, d_keys_cur, d_keys_alt, d_vals_cur,
                     d_vals_alt, d_offsets, static_cast<int64_t>(total),
                     static_cast<int64_t>(T), nullptr);
    if (err != cudaSuccess) goto fail;
    if (temp_bytes > 0) {
      err = AllocOrTrack(&d_temp, temp_bytes, "cub_temp", "cub_temp", tracker);
      if (err != cudaSuccess) goto fail;
    }
  }

  // Optional per-group timing. Events are created once and reused. Every event
  // API return is checked (review F2): create/record/elapsed failures are
  // scan-level; a partially-created pair is cleaned up.
  if (use_timing) {
    cudaError_t e1 = cudaEventCreate(&ev_start);
    cudaError_t e2 = (e1 == cudaSuccess) ? cudaEventCreate(&ev_stop) : e1;
    if (e1 != cudaSuccess || e2 != cudaSuccess) {
      if (ev_start != nullptr) cudaEventDestroy(ev_start);
      if (ev_stop != nullptr) cudaEventDestroy(ev_stop);
      err = (e1 != cudaSuccess) ? e1 : e2;
      goto fail;
    }
    ev_created = true;
  }

  for (int g = 0; g < 4 && err == cudaSuccess; ++g) {
    // Active-group selector (Phase 1 adapter subset scans): skipped groups are
    // neither launched, D2H'd, nor timed; group_status stays kGroupNotExecuted.
    if (h_active != nullptr && !h_active[g]) {
      group_status[g] = kGroupNotExecuted;
      if (h_time_ms != nullptr) h_time_ms[g] = 0.0;
      if (h_time_gpu_ms != nullptr) h_time_gpu_ms[g] = 0.0f;
      continue;
    }

    const bool descending = (g >= 2);                    // group g<2 ascending
    const uint8_t* group_mask = (g % 2 == 0) ? d_mask : nullptr;  // odd=unmasked

    // Declared at the top of the loop body (value-initialized) so the record
    // block's goto-fail below does not bypass its initialization (C++
    // transfer-of-control rule). The actual start timestamp is taken AFTER the
    // event record so the record host-call overhead is not counted into
    // h_time_ms (review F5); both are gated by use_timing (review F4).
    std::chrono::steady_clock::time_point t0{};
    if (use_timing) {
      cudaError_t et = cudaEventRecord(ev_start);
      if (et != cudaSuccess) {
        err = et;
        break;
      }
      t0 = std::chrono::steady_clock::now();
    }

    // Per-group launch checkpoint: only the two whitelist errors downgrade this
    // group; everything else (incl. any async error surfaced by the sync) is a
    // scan-level fatal. The sync also drains errors so group attribution is
    // clean before the D2H (contract: only a launch-checkpoint error may
    // downgrade; a D2H-reported error must NOT).
    cudaError_t e0 = LaunchPreprocess(d_X, group_mask, N, descending, total,
                                      d_keys_cur, d_vals_cur, d_valid);
    if (e0 == cudaSuccess)
      e0 = RunCubSort(d_temp, temp_bytes, d_keys_cur, d_keys_alt, d_vals_cur,
                      d_vals_alt, d_offsets, static_cast<int64_t>(total),
                      static_cast<int64_t>(T), &values_current);
    if (e0 == cudaSuccess)
      e0 = LaunchScatter(values_current, d_valid, N, total, d_out);

    if (e0 == cudaSuccess) {
      // All launches succeeded: enqueue the device stop (stream order makes it
      // complete when the kernel completes, so time_gpu_ms = kernel device time;
      // NOT after the host sync, which would measure ~0), then sync. A sync
      // error here is a scan-level fatal (async errors are never whitelist).
      if (use_timing) {
        cudaError_t et = cudaEventRecord(ev_stop);
        if (et != cudaSuccess) {
          err = et;
          break;
        }
      }
      e0 = cudaDeviceSynchronize();
      if (e0 != cudaSuccess) {
        group_status[g] = static_cast<int>(e0);
        err = e0;
        break;
      }
      group_status[g] = 0;
      err = cudaMemcpy(h_out[g], d_out, n_items * sizeof(float),
                       cudaMemcpyDeviceToHost);
      if (err != cudaSuccess) {  // D2H is a scan-level error per contract
        group_status[g] = static_cast<int>(err);
        break;
      }
      if (use_timing) {
        const auto t1 = std::chrono::steady_clock::now();
        float gpu_ms = 0.0f;
        cudaError_t et = cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop);
        if (et != cudaSuccess) {
          err = et;
          break;
        }
        if (h_time_ms != nullptr)
          h_time_ms[g] =
              std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (h_time_gpu_ms != nullptr) h_time_gpu_ms[g] = gpu_ms;
      }
    } else if (IsGroupWhitelistError(e0)) {
      // Whitelist launch failure (review F1): the earlier successful launches
      // in this group were enqueued but not drained -- run a group-end sync
      // checkpoint so their async errors cannot bleed into the next group. A
      // sync error upgrades to scan-level (async errors are never whitelist).
      cudaError_t sync_err = cudaDeviceSynchronize();
      if (sync_err != cudaSuccess) {
        group_status[g] = static_cast<int>(sync_err);
        err = sync_err;
        break;
      }
      group_status[g] = static_cast<int>(e0);  // group failed; result = None
      if (h_time_ms != nullptr) h_time_ms[g] = 0.0;    // failed timing = 0.0
      if (h_time_gpu_ms != nullptr) h_time_gpu_ms[g] = 0.0f;
    } else {  // non-whitelist -> scan-level fatal, remaining groups not executed
      group_status[g] = static_cast<int>(e0);
      err = e0;
      break;
    }
  }
  if (err != cudaSuccess) goto fail;

  // Cleanup error propagation (review F6): a cleanup failure must not be
  // silently swallowed on the success path; the first cleanup error (event
  // destroy or FreeAllBuffers) is returned as a scan-level failure.
  if (ev_created) {
    cudaError_t d1 = cudaEventDestroy(ev_start);
    cudaError_t d2 = cudaEventDestroy(ev_stop);
    if (d1 != cudaSuccess) cleanup_err = d1;
    else if (d2 != cudaSuccess) cleanup_err = d2;
  }
  if (ws == nullptr) {
    free_err = FreeAllBuffers(d_valid, d_keys_cur, d_keys_alt,
                              d_vals_cur, d_vals_alt, d_offsets,
                              d_X, d_mask, d_out, d_temp, tracker);
  }
  if (free_err != cudaSuccess && cleanup_err == cudaSuccess) cleanup_err = free_err;
  // Scan completed (all active groups attempted); group_status is authoritative.
  if (cleanup_err != cudaSuccess) return static_cast<int>(cleanup_err);
  return 0;

fail:
  if (ev_created) {
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
  }
  if (ws == nullptr) {
    FreeAllBuffers(d_valid, d_keys_cur, d_keys_alt, d_vals_cur, d_vals_alt, d_offsets,
                   d_X, d_mask, d_out, d_temp, tracker);
  }
  return static_cast<int>(err);
}
