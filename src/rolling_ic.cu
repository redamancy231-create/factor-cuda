// factor-cuda -- rolling_ic v0 GPU kernel.
//
// Pipeline (per row t, independent cross-sections):
//   1. preprocess_kernel : per-cell valid flag + f64 canonical ordinal keys
//      (corr_math_v1.py single source of truth) + per-row valid count and
//      factor/return key min/max (constant-section detection via key min==max,
//      equivalent to np.ptp(valid subset)==0 in np_rolling_ic).
//   2. cub::DeviceSegmentedRadixSort::SortPairs DoubleBuffer over uint64 keys
//      for factor and returns (two passes); invalid sentinel 0xFF.. sorts last.
//   3. scatter_rank_double_kernel : rank = segment-local position + 1 written
//      to rank_f/rank_r (payload column index); invalid cell -> 0.
//   4. pearson_kernel : two-pass centered Pearson (float64) over the valid
//      cells of each row; fixed tree reduction (no atomicAdd), min_valid and
//      constant-section guards output NaN.
//
// Unified float64 path matches np_rolling_ic (upcast to f64; f32 survives
// exactly). ASCII-only comments (nvcc/GBK pitfall).
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <cub/device/device_segmented_radix_sort.cuh>

#include "rolling_ic.cuh"
#include "rolling_ic_impl.cuh"

namespace {

constexpr int kMaxTotal = INT32_MAX;      // T*N <= INT32_MAX (checked_mul)
constexpr uint64_t kInvalidKey64 = 0xFFFFFFFFFFFFFFFFull;
constexpr uint32_t kInvalidKey32 = 0xFFFFFFFFu;

// canonical_ordinal_key_f64 -- mirrors tests/fixtures/corr_math_v1.py:197.
// Ascending key order == ascending numeric order; +0.0/-0.0 fold to one key.
__device__ __forceinline__ uint64_t canonical_ordinal_key_f64(double v) {
  if (!isfinite(v)) return kInvalidKey64;
  uint64_t bits = (v == 0.0) ? 0ull : __double_as_longlong(v);
  return (bits & 0x8000000000000000ull) ? (~bits) : (bits | 0x8000000000000000ull);
}

// Fill segment offsets on device: d_offsets[t] = t*N for t in [0, T].
__global__ void make_offsets_kernel(int32_t* __restrict__ d_offsets, int T, int N) {
  int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t > static_cast<int64_t>(T)) return;
  d_offsets[t] = static_cast<int32_t>(t * N);  // t*N <= T*N <= INT32_MAX
}

cudaError_t LaunchMakeOffsets(int32_t* d_offsets, int T, int N) {
  const int block = 256;
  const int64_t need = static_cast<int64_t>(T) + 1;
  const int64_t grid64 = 1 + (need - 1) / block;
  const int grid = grid64 > INT32_MAX ? INT32_MAX : static_cast<int>(grid64);
  make_offsets_kernel<<<grid, block>>>(d_offsets, T, N);
  return cudaGetLastError();
}

// Query/execute the CUB segmented radix sort over a DoubleBuffer key pair.
template <typename KeyT>
cudaError_t RunSegSort(void* d_temp, size_t& temp_bytes, KeyT* keys_cur,
                       KeyT* keys_alt, uint32_t* vals_cur, uint32_t* vals_alt,
                       const int32_t* d_offsets, int64_t total, int64_t T,
                       const uint32_t** out_values_current) {
  cub::DoubleBuffer<KeyT> d_keys(keys_cur, keys_alt);
  cub::DoubleBuffer<uint32_t> d_vals(vals_cur, vals_alt);
  cudaError_t err = cub::DeviceSegmentedRadixSort::SortPairs(
      d_temp, temp_bytes, d_keys, d_vals, total, T, d_offsets, d_offsets + 1);
  if (err == cudaSuccess && out_values_current != nullptr) {
    *out_values_current = d_vals.Current();
  }
  return err;
}

// Route alloc/free through the tracker when provided, else plain CUDA calls.
template <typename T>
cudaError_t AllocOrTrack(T** ptr, size_t bytes, const char* name, const char* stage,
                         factor_cuda::MemTracker* t) {
  return t != nullptr ? t->Alloc(reinterpret_cast<void**>(ptr), bytes, name, stage)
                      : cudaMalloc(reinterpret_cast<void**>(ptr), bytes);
}

cudaError_t FreeOrTrack(void* ptr, factor_cuda::MemTracker* t) {
  return t != nullptr ? t->Free(ptr) : cudaFree(ptr);
}

cudaError_t FreeAllBuffers(double* d_F, double* d_R, uint8_t* d_fmask,
                           uint8_t* d_rmask, uint64_t* d_fkey_cur, uint64_t* d_fkey_alt,
                           uint64_t* d_rkey_cur, uint64_t* d_rkey_alt,
                           uint32_t* d_fvals_cur, uint32_t* d_fvals_alt,
                           uint32_t* d_rvals_cur, uint32_t* d_rvals_alt,
                           uint8_t* d_valid, double* d_rank_f, double* d_rank_r,
                           int32_t* d_offsets, uint32_t* d_valid_count,
                           uint64_t* d_fkey_min, uint64_t* d_fkey_max,
                           uint64_t* d_rkey_min, uint64_t* d_rkey_max, double* d_ic,
                           void* d_temp, factor_cuda::MemTracker* tracker) {
  cudaError_t e = cudaSuccess;
  auto keep_first = [&e](cudaError_t r) {
    if (e == cudaSuccess && r != cudaSuccess) e = r;
  };
  keep_first(FreeOrTrack(d_F, tracker));
  keep_first(FreeOrTrack(d_R, tracker));
  keep_first(FreeOrTrack(d_fmask, tracker));
  keep_first(FreeOrTrack(d_rmask, tracker));
  keep_first(FreeOrTrack(d_fkey_cur, tracker));
  keep_first(FreeOrTrack(d_fkey_alt, tracker));
  keep_first(FreeOrTrack(d_rkey_cur, tracker));
  keep_first(FreeOrTrack(d_rkey_alt, tracker));
  keep_first(FreeOrTrack(d_fvals_cur, tracker));
  keep_first(FreeOrTrack(d_fvals_alt, tracker));
  keep_first(FreeOrTrack(d_rvals_cur, tracker));
  keep_first(FreeOrTrack(d_rvals_alt, tracker));
  keep_first(FreeOrTrack(d_valid, tracker));
  keep_first(FreeOrTrack(d_rank_f, tracker));
  keep_first(FreeOrTrack(d_rank_r, tracker));
  keep_first(FreeOrTrack(d_offsets, tracker));
  keep_first(FreeOrTrack(d_valid_count, tracker));
  keep_first(FreeOrTrack(d_fkey_min, tracker));
  keep_first(FreeOrTrack(d_fkey_max, tracker));
  keep_first(FreeOrTrack(d_rkey_min, tracker));
  keep_first(FreeOrTrack(d_rkey_max, tracker));
  keep_first(FreeOrTrack(d_ic, tracker));
  keep_first(FreeOrTrack(d_temp, tracker));
  return e;
}

// Allocate every core device buffer into the workspace (after a shape/device/
// tracker mismatch) + query + allocate the CUB temp buffer. d_fmask / d_rmask
// are NOT allocated here -- optional lazy capacity, grown on the first masked
// call (synced to cs_rank_workspace review F02). On failure the workspace is
// cleared so a retry starts clean.
cudaError_t AllocRollingIcWorkspace(rolling_ic_workspace* ws,
                                    factor_cuda::MemTracker* tracker,
                                    int total, int T, int N) {
  const size_t n_items = static_cast<size_t>(total);
  const size_t n_rows = static_cast<size_t>(T);
  cudaError_t err = AllocOrTrack(&ws->d_F, n_items * sizeof(double), "d_F", "alloc", tracker);
  if (err == cudaSuccess) err = AllocOrTrack(&ws->d_R, n_items * sizeof(double), "d_R", "alloc", tracker);
  if (err == cudaSuccess) err = AllocOrTrack(&ws->d_fkey_cur, n_items * sizeof(uint64_t), "d_fkey_cur", "alloc", tracker);
  if (err == cudaSuccess) err = AllocOrTrack(&ws->d_fkey_alt, n_items * sizeof(uint64_t), "d_fkey_alt", "alloc", tracker);
  if (err == cudaSuccess) err = AllocOrTrack(&ws->d_rkey_cur, n_items * sizeof(uint64_t), "d_rkey_cur", "alloc", tracker);
  if (err == cudaSuccess) err = AllocOrTrack(&ws->d_rkey_alt, n_items * sizeof(uint64_t), "d_rkey_alt", "alloc", tracker);
  if (err == cudaSuccess) err = AllocOrTrack(&ws->d_fvals_cur, n_items * sizeof(uint32_t), "d_fvals_cur", "alloc", tracker);
  if (err == cudaSuccess) err = AllocOrTrack(&ws->d_fvals_alt, n_items * sizeof(uint32_t), "d_fvals_alt", "alloc", tracker);
  if (err == cudaSuccess) err = AllocOrTrack(&ws->d_rvals_cur, n_items * sizeof(uint32_t), "d_rvals_cur", "alloc", tracker);
  if (err == cudaSuccess) err = AllocOrTrack(&ws->d_rvals_alt, n_items * sizeof(uint32_t), "d_rvals_alt", "alloc", tracker);
  if (err == cudaSuccess) err = AllocOrTrack(&ws->d_valid, n_items, "d_valid", "alloc", tracker);
  if (err == cudaSuccess) err = AllocOrTrack(&ws->d_rank_f, n_items * sizeof(double), "d_rank_f", "alloc", tracker);
  if (err == cudaSuccess) err = AllocOrTrack(&ws->d_rank_r, n_items * sizeof(double), "d_rank_r", "alloc", tracker);
  if (err == cudaSuccess) err = AllocOrTrack(&ws->d_offsets, (n_rows + 1) * sizeof(int32_t), "d_offsets", "alloc", tracker);
  if (err == cudaSuccess) err = AllocOrTrack(&ws->d_valid_count, n_rows * sizeof(uint32_t), "d_valid_count", "alloc", tracker);
  if (err == cudaSuccess) err = AllocOrTrack(&ws->d_fkey_min, n_rows * sizeof(uint64_t), "d_fkey_min", "alloc", tracker);
  if (err == cudaSuccess) err = AllocOrTrack(&ws->d_fkey_max, n_rows * sizeof(uint64_t), "d_fkey_max", "alloc", tracker);
  if (err == cudaSuccess) err = AllocOrTrack(&ws->d_rkey_min, n_rows * sizeof(uint64_t), "d_rkey_min", "alloc", tracker);
  if (err == cudaSuccess) err = AllocOrTrack(&ws->d_rkey_max, n_rows * sizeof(uint64_t), "d_rkey_max", "alloc", tracker);
  if (err == cudaSuccess) err = AllocOrTrack(&ws->d_ic, n_rows * sizeof(double), "d_ic", "alloc", tracker);
  if (err == cudaSuccess) {
    // Query + allocate the CUB temp once (the two DoubleBuffer sorts share the
    // single temp buffer). Ensure >= 1 byte so an execute call is never
    // re-interpreted as a query (review F03 pattern).
    size_t q = 0;
    err = RunSegSort(nullptr, q, ws->d_fkey_cur, ws->d_fkey_alt, ws->d_fvals_cur,
                     ws->d_fvals_alt, ws->d_offsets, static_cast<int64_t>(total),
                     static_cast<int64_t>(T), nullptr);
    if (err == cudaSuccess) {
      if (q == 0) q = 1;
      err = AllocOrTrack(&ws->d_temp, q, "cub_temp", "cub_temp", tracker);
      if (err == cudaSuccess) ws->temp_bytes = q;
    }
  }
  if (err != cudaSuccess) {
    rolling_ic_workspace_clear(ws);  // release any partially-allocated buffers
    return err;
  }
  return cudaSuccess;
}

}  // namespace

// The three kernels below are declared in rolling_ic_impl.cuh (shared with the
// F/T chunking proof harness). Definitions live here at file scope so the
// chunked path can launch the exact production kernels with chunk-local
// buffers and re-cut segment offsets. Kernel bodies are unchanged by the
// chunked layout: preprocess/scatter/pearson take a slice base pointer and a
// (possibly chunk-local) total, and pearson uses blockIdx.x as the local row.
__global__ void preprocess_kernel(const double* __restrict__ F,
                                  const double* __restrict__ R,
                                  const uint8_t* __restrict__ fmask,
                                  const uint8_t* __restrict__ rmask, int N,
                                  int total, uint64_t* __restrict__ fkey,
                                  uint64_t* __restrict__ rkey,
                                  uint32_t* __restrict__ fvals,
                                  uint32_t* __restrict__ rvals,
                                  uint8_t* __restrict__ valid,
                                  uint32_t* __restrict__ valid_count,
                                  uint64_t* __restrict__ fkey_min,
                                  uint64_t* __restrict__ fkey_max,
                                  uint64_t* __restrict__ rkey_min,
                                  uint64_t* __restrict__ rkey_max) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  int t = i / N;
  double v = F[i], w = R[i];
  uint8_t ok = static_cast<uint8_t>(
      isfinite(v) && isfinite(w) && (fmask == nullptr || fmask[i]) &&
      (rmask == nullptr || rmask[i]));
  uint64_t fk = ok ? canonical_ordinal_key_f64(v) : kInvalidKey64;
  uint64_t rk = ok ? canonical_ordinal_key_f64(w) : kInvalidKey64;
  fkey[i] = fk;
  rkey[i] = rk;
  fvals[i] = static_cast<uint32_t>(i % N);  // payload = column index
  rvals[i] = static_cast<uint32_t>(i % N);
  valid[i] = ok;
  if (ok) {
    atomicAdd(&valid_count[t], 1u);
    atomicMin(&fkey_min[t], fk);
    atomicMax(&fkey_max[t], fk);
    atomicMin(&rkey_min[t], rk);
    atomicMax(&rkey_max[t], rk);
  }
}

__global__ void scatter_rank_double_kernel(const uint32_t* __restrict__ sorted_values,
                                           const uint8_t* __restrict__ valid, int N,
                                           int total, double* __restrict__ rank_out) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= total) return;
  int t = p / N;
  int rank = p - t * N + 1;
  int col = static_cast<int>(sorted_values[p]);
  int out_idx = t * N + col;
  rank_out[out_idx] = valid[out_idx] ? static_cast<double>(rank) : 0.0;
}

__global__ void pearson_kernel(const double* __restrict__ rank_f,
                               const double* __restrict__ rank_r,
                               const uint8_t* __restrict__ valid,
                               const uint32_t* __restrict__ valid_count,
                               const uint64_t* __restrict__ fkey_min,
                               const uint64_t* __restrict__ fkey_max,
                               const uint64_t* __restrict__ rkey_min,
                               const uint64_t* __restrict__ rkey_max, int N,
                               int min_valid, double* __restrict__ ic) {
  const int t = blockIdx.x;
  const int tid = threadIdx.x;
  if (static_cast<int>(valid_count[t]) < min_valid) {
    if (tid == 0) ic[t] = NAN;
    return;
  }
  if (fkey_min[t] == fkey_max[t] || rkey_min[t] == rkey_max[t]) {
    if (tid == 0) ic[t] = NAN;
    return;
  }
  __shared__ double sh1[256], sh2[256], sh3[256];

  // pass 1: sums (for means)
  double sx = 0.0, sy = 0.0;
  for (int j = tid; j < N; j += blockDim.x) {
    const int idx = t * N + j;
    if (valid[idx]) {
      sx += rank_f[idx];
      sy += rank_r[idx];
    }
  }
  sh1[tid] = sx;
  sh2[tid] = sy;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sh1[tid] += sh1[tid + s];
      sh2[tid] += sh2[tid + s];
    }
    __syncthreads();
  }
  const double mean_x = sh1[0] / static_cast<double>(valid_count[t]);
  const double mean_y = sh2[0] / static_cast<double>(valid_count[t]);

  // pass 2: centered sums of squares / cross
  double lxx = 0.0, lyy = 0.0, lxy = 0.0;
  for (int j = tid; j < N; j += blockDim.x) {
    const int idx = t * N + j;
    if (valid[idx]) {
      const double dx = rank_f[idx] - mean_x;
      const double dy = rank_r[idx] - mean_y;
      lxx += dx * dx;
      lyy += dy * dy;
      lxy += dx * dy;
    }
  }
  sh1[tid] = lxx;
  sh2[tid] = lyy;
  sh3[tid] = lxy;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sh1[tid] += sh1[tid + s];
      sh2[tid] += sh2[tid + s];
      sh3[tid] += sh3[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    const double sxx = sh1[0], syy = sh2[0], sxy = sh3[0];
    // safe_pearson: (sxy/sqrt(sxx))/sqrt(syy) -- no intermediate sxx*syy
    ic[t] = (sxx > 0.0 && syy > 0.0) ? (sxy / std::sqrt(sxx)) / std::sqrt(syy) : NAN;
  }
}

void rolling_ic_workspace_clear(rolling_ic_workspace* ws) {
  if (ws == nullptr) return;
  // Always free via the owner tracker recorded at allocation time (review F07
  // pattern): release routing must not depend on what the caller passes today.
  FreeAllBuffers(ws->d_F, ws->d_R, ws->d_fmask, ws->d_rmask, ws->d_fkey_cur,
                 ws->d_fkey_alt, ws->d_rkey_cur, ws->d_rkey_alt, ws->d_fvals_cur,
                 ws->d_fvals_alt, ws->d_rvals_cur, ws->d_rvals_alt, ws->d_valid,
                 ws->d_rank_f, ws->d_rank_r, ws->d_offsets, ws->d_valid_count,
                 ws->d_fkey_min, ws->d_fkey_max, ws->d_rkey_min, ws->d_rkey_max,
                 ws->d_ic, ws->d_temp, ws->owner_tracker);
  ws->d_F = nullptr; ws->d_R = nullptr; ws->d_fmask = nullptr; ws->d_rmask = nullptr;
  ws->d_fkey_cur = nullptr; ws->d_fkey_alt = nullptr; ws->d_rkey_cur = nullptr; ws->d_rkey_alt = nullptr;
  ws->d_fvals_cur = nullptr; ws->d_fvals_alt = nullptr; ws->d_rvals_cur = nullptr; ws->d_rvals_alt = nullptr;
  ws->d_valid = nullptr; ws->d_rank_f = nullptr; ws->d_rank_r = nullptr; ws->d_offsets = nullptr;
  ws->d_valid_count = nullptr; ws->d_fkey_min = nullptr; ws->d_fkey_max = nullptr;
  ws->d_rkey_min = nullptr; ws->d_rkey_max = nullptr; ws->d_ic = nullptr; ws->d_temp = nullptr;
  ws->temp_bytes = 0; ws->n_items = -1; ws->T = -1; ws->N = -1;
  ws->device = -1; ws->owner_tracker = nullptr;
}

int rolling_ic_gpu(const double* h_F, const double* h_R, const uint8_t* h_fmask,
                   const uint8_t* h_rmask, int T, int N, int min_valid,
                   double* h_out, factor_cuda::MemTracker* tracker,
                   double* h_rank_f_out, double* h_rank_r_out,
                   rolling_ic_workspace* ws) {
  // ---- host preconditions --------------------------------------------------
  if (h_F == nullptr || h_R == nullptr || h_out == nullptr || T < 1 || N < 1)
    return -1;
  if (min_valid < 2) return -4;  // contract: min_valid must be >= 2
  if (static_cast<int64_t>(T) * N > kMaxTotal) return -3;  // checked_mul T*N <= INT32_MAX

  const int total = T * N;
  const size_t n_items = static_cast<size_t>(total);
  const size_t n_rows = static_cast<size_t>(T);

  size_t temp_bytes = 0;
  cudaError_t cleanup = cudaSuccess;

  double* d_F = nullptr;
  double* d_R = nullptr;
  uint8_t* d_fmask = nullptr;
  uint8_t* d_rmask = nullptr;
  uint64_t* d_fkey_cur = nullptr;
  uint64_t* d_fkey_alt = nullptr;
  uint64_t* d_rkey_cur = nullptr;
  uint64_t* d_rkey_alt = nullptr;
  uint32_t* d_fvals_cur = nullptr;
  uint32_t* d_fvals_alt = nullptr;
  uint32_t* d_rvals_cur = nullptr;
  uint32_t* d_rvals_alt = nullptr;
  uint8_t* d_valid = nullptr;
  double* d_rank_f = nullptr;
  double* d_rank_r = nullptr;
  int32_t* d_offsets = nullptr;
  uint32_t* d_valid_count = nullptr;
  uint64_t* d_fkey_min = nullptr;
  uint64_t* d_fkey_max = nullptr;
  uint64_t* d_rkey_min = nullptr;
  uint64_t* d_rkey_max = nullptr;
  double* d_ic = nullptr;
  void* d_temp = nullptr;

  cudaError_t err = cudaSuccess;
  if (ws != nullptr) {
    // Workspace path (P2 PoC4 perf, 2026-08-08): reuse cached buffers when the
    // shape (T*N, T, N) / device / tracker key matches, else clear + re-allocate
    // -- removes the per-call cudaMalloc/cudaFree overhead on ~20 buffers.
    int cur_dev = -1;
    cudaGetDevice(&cur_dev);
    if (ws->n_items != static_cast<int64_t>(n_items) || ws->T != T || ws->N != N ||
        ws->device != cur_dev || ws->owner_tracker != tracker) {
      rolling_ic_workspace_clear(ws);
      ws->n_items = static_cast<int64_t>(n_items);
      ws->T = T; ws->N = N; ws->device = cur_dev; ws->owner_tracker = tracker;
      err = AllocRollingIcWorkspace(ws, tracker, total, T, N);
      if (err != cudaSuccess) return static_cast<int>(err);  // ws cleared + key reset
    }
    // Optional mask capacity: lazily grown on the first masked call (F02).
    if (h_fmask != nullptr && ws->d_fmask == nullptr) {
      err = AllocOrTrack(&ws->d_fmask, n_items, "d_fmask", "alloc", tracker);
      if (err != cudaSuccess) return static_cast<int>(err);
    }
    if (h_rmask != nullptr && ws->d_rmask == nullptr) {
      err = AllocOrTrack(&ws->d_rmask, n_items, "d_rmask", "alloc", tracker);
      if (err != cudaSuccess) return static_cast<int>(err);
    }
    d_F = ws->d_F; d_R = ws->d_R;
    d_fmask = (h_fmask != nullptr) ? ws->d_fmask : nullptr;  // stale-mask guard
    d_rmask = (h_rmask != nullptr) ? ws->d_rmask : nullptr;
    d_fkey_cur = ws->d_fkey_cur; d_fkey_alt = ws->d_fkey_alt;
    d_rkey_cur = ws->d_rkey_cur; d_rkey_alt = ws->d_rkey_alt;
    d_fvals_cur = ws->d_fvals_cur; d_fvals_alt = ws->d_fvals_alt;
    d_rvals_cur = ws->d_rvals_cur; d_rvals_alt = ws->d_rvals_alt;
    d_valid = ws->d_valid; d_rank_f = ws->d_rank_f; d_rank_r = ws->d_rank_r;
    d_offsets = ws->d_offsets; d_valid_count = ws->d_valid_count;
    d_fkey_min = ws->d_fkey_min; d_fkey_max = ws->d_fkey_max;
    d_rkey_min = ws->d_rkey_min; d_rkey_max = ws->d_rkey_max;
    d_ic = ws->d_ic; d_temp = ws->d_temp; temp_bytes = ws->temp_bytes;
  } else {
    // Non-workspace path: allocate and free every call (previous behavior).
    err = AllocOrTrack(&d_F, n_items * sizeof(double), "d_F", "alloc", tracker);
    if (err == cudaSuccess) err = AllocOrTrack(&d_R, n_items * sizeof(double), "d_R", "alloc", tracker);
    if (err == cudaSuccess && h_fmask != nullptr) err = AllocOrTrack(&d_fmask, n_items, "d_fmask", "alloc", tracker);
    if (err == cudaSuccess && h_rmask != nullptr) err = AllocOrTrack(&d_rmask, n_items, "d_rmask", "alloc", tracker);
    if (err == cudaSuccess) err = AllocOrTrack(&d_fkey_cur, n_items * sizeof(uint64_t), "d_fkey_cur", "alloc", tracker);
    if (err == cudaSuccess) err = AllocOrTrack(&d_fkey_alt, n_items * sizeof(uint64_t), "d_fkey_alt", "alloc", tracker);
    if (err == cudaSuccess) err = AllocOrTrack(&d_rkey_cur, n_items * sizeof(uint64_t), "d_rkey_cur", "alloc", tracker);
    if (err == cudaSuccess) err = AllocOrTrack(&d_rkey_alt, n_items * sizeof(uint64_t), "d_rkey_alt", "alloc", tracker);
    if (err == cudaSuccess) err = AllocOrTrack(&d_fvals_cur, n_items * sizeof(uint32_t), "d_fvals_cur", "alloc", tracker);
    if (err == cudaSuccess) err = AllocOrTrack(&d_fvals_alt, n_items * sizeof(uint32_t), "d_fvals_alt", "alloc", tracker);
    if (err == cudaSuccess) err = AllocOrTrack(&d_rvals_cur, n_items * sizeof(uint32_t), "d_rvals_cur", "alloc", tracker);
    if (err == cudaSuccess) err = AllocOrTrack(&d_rvals_alt, n_items * sizeof(uint32_t), "d_rvals_alt", "alloc", tracker);
    if (err == cudaSuccess) err = AllocOrTrack(&d_valid, n_items, "d_valid", "alloc", tracker);
    if (err == cudaSuccess) err = AllocOrTrack(&d_rank_f, n_items * sizeof(double), "d_rank_f", "alloc", tracker);
    if (err == cudaSuccess) err = AllocOrTrack(&d_rank_r, n_items * sizeof(double), "d_rank_r", "alloc", tracker);
    if (err == cudaSuccess) err = AllocOrTrack(&d_offsets, (n_rows + 1) * sizeof(int32_t), "d_offsets", "alloc", tracker);
    if (err == cudaSuccess) err = AllocOrTrack(&d_valid_count, n_rows * sizeof(uint32_t), "d_valid_count", "alloc", tracker);
    if (err == cudaSuccess) err = AllocOrTrack(&d_fkey_min, n_rows * sizeof(uint64_t), "d_fkey_min", "alloc", tracker);
    if (err == cudaSuccess) err = AllocOrTrack(&d_fkey_max, n_rows * sizeof(uint64_t), "d_fkey_max", "alloc", tracker);
    if (err == cudaSuccess) err = AllocOrTrack(&d_rkey_min, n_rows * sizeof(uint64_t), "d_rkey_min", "alloc", tracker);
    if (err == cudaSuccess) err = AllocOrTrack(&d_rkey_max, n_rows * sizeof(uint64_t), "d_rkey_max", "alloc", tracker);
    if (err == cudaSuccess) err = AllocOrTrack(&d_ic, n_rows * sizeof(double), "d_ic", "alloc", tracker);
    if (err != cudaSuccess) goto fail;
  }

  // ---- upload inputs -------------------------------------------------------
  err = cudaMemcpy(d_F, h_F, n_items * sizeof(double), cudaMemcpyHostToDevice);
  if (err == cudaSuccess) err = cudaMemcpy(d_R, h_R, n_items * sizeof(double), cudaMemcpyHostToDevice);
  if (err == cudaSuccess && h_fmask != nullptr) err = cudaMemcpy(d_fmask, h_fmask, n_items, cudaMemcpyHostToDevice);
  if (err == cudaSuccess && h_rmask != nullptr) err = cudaMemcpy(d_rmask, h_rmask, n_items, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) goto fail;

  // ---- per-row stats buffers initialization --------------------------------
  // valid_count -> 0; key min -> all-ones; key max -> 0 (memset byte-wise).
  if (err == cudaSuccess) err = cudaMemset(d_valid_count, 0, n_rows * sizeof(uint32_t));
  if (err == cudaSuccess) err = cudaMemset(d_fkey_min, 0xFF, n_rows * sizeof(uint64_t));
  if (err == cudaSuccess) err = cudaMemset(d_fkey_max, 0, n_rows * sizeof(uint64_t));
  if (err == cudaSuccess) err = cudaMemset(d_rkey_min, 0xFF, n_rows * sizeof(uint64_t));
  if (err == cudaSuccess) err = cudaMemset(d_rkey_max, 0, n_rows * sizeof(uint64_t));
  if (err != cudaSuccess) goto fail;

  // ---- regular segments -----------------------------------------------------
  err = LaunchMakeOffsets(d_offsets, T, N);
  if (err != cudaSuccess) goto fail;

  // ---- stage 1: preprocess --------------------------------------------------
  {
    const int block = 256;
    const int grid = static_cast<int>(1 + (static_cast<int64_t>(total) - 1) / block);
    preprocess_kernel<<<grid, block>>>(d_F, d_R, d_fmask, d_rmask, N, total, d_fkey_cur,
                                       d_rkey_cur, d_fvals_cur, d_rvals_cur, d_valid,
                                       d_valid_count, d_fkey_min, d_fkey_max, d_rkey_min,
                                       d_rkey_max);
    err = cudaGetLastError();
    if (err != cudaSuccess) goto fail;
  }

  // ---- stage 2a: sort factor keys, scatter factor ranks ---------------------
  // Workspace path: d_temp was queried + allocated at acquire, so skip the
  // query/alloc here (a second AllocOrTrack would leak the cached temp).
  if (ws == nullptr) {
    err = RunSegSort(nullptr, temp_bytes, d_fkey_cur, d_fkey_alt, d_fvals_cur, d_fvals_alt,
                     d_offsets, static_cast<int64_t>(total), static_cast<int64_t>(T), nullptr);
    if (err != cudaSuccess) goto fail;
    if (temp_bytes > 0) {
      err = AllocOrTrack(&d_temp, temp_bytes, "cub_temp", "cub_temp", tracker);
      if (err != cudaSuccess) goto fail;
    }
  }
  {
    const uint32_t* f_vals = nullptr;
    err = RunSegSort(d_temp, temp_bytes, d_fkey_cur, d_fkey_alt, d_fvals_cur, d_fvals_alt,
                     d_offsets, static_cast<int64_t>(total), static_cast<int64_t>(T), &f_vals);
    if (err != cudaSuccess) goto fail;
    const int block = 256;
    const int grid = static_cast<int>(1 + (static_cast<int64_t>(total) - 1) / block);
    scatter_rank_double_kernel<<<grid, block>>>(f_vals, d_valid, N, total, d_rank_f);
    err = cudaGetLastError();
    if (err != cudaSuccess) goto fail;
  }

  // ---- stage 2b: sort return keys, scatter return ranks ---------------------
  {
    const uint32_t* r_vals = nullptr;
    err = RunSegSort(d_temp, temp_bytes, d_rkey_cur, d_rkey_alt, d_rvals_cur, d_rvals_alt,
                     d_offsets, static_cast<int64_t>(total), static_cast<int64_t>(T), &r_vals);
    if (err != cudaSuccess) goto fail;
    const int block = 256;
    const int grid = static_cast<int>(1 + (static_cast<int64_t>(total) - 1) / block);
    scatter_rank_double_kernel<<<grid, block>>>(r_vals, d_valid, N, total, d_rank_r);
    err = cudaGetLastError();
    if (err != cudaSuccess) goto fail;
  }

  // ---- optional rank dump (F/T chunking proof) -------------------------------
  // The chunked path must produce bitwise-identical intermediate rank arrays;
  // the final IC alone is non-injective over rank pairs, so compare the ranks
  // directly when the caller requests the outputs.
  if (h_rank_f_out != nullptr && h_rank_r_out != nullptr) {
    err = cudaMemcpy(h_rank_f_out, d_rank_f, n_items * sizeof(double), cudaMemcpyDeviceToHost);
    if (err == cudaSuccess) err = cudaMemcpy(h_rank_r_out, d_rank_r, n_items * sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto fail;
  }

  // ---- stage 3: per-row Pearson ----------------------------------------------
  {
    const int block = 256;
    pearson_kernel<<<T, block>>>(d_rank_f, d_rank_r, d_valid, d_valid_count, d_fkey_min,
                                 d_fkey_max, d_rkey_min, d_rkey_max, N, min_valid, d_ic);
    err = cudaGetLastError();
    if (err != cudaSuccess) goto fail;
  }

  // ---- download output --------------------------------------------------------
  err = cudaMemcpy(h_out, d_ic, n_rows * sizeof(double), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) goto fail;

  if (ws == nullptr) {
    cleanup = FreeAllBuffers(d_F, d_R, d_fmask, d_rmask, d_fkey_cur, d_fkey_alt, d_rkey_cur,
                             d_rkey_alt, d_fvals_cur, d_fvals_alt, d_rvals_cur, d_rvals_alt,
                             d_valid, d_rank_f, d_rank_r, d_offsets, d_valid_count, d_fkey_min,
                             d_fkey_max, d_rkey_min, d_rkey_max, d_ic, d_temp, tracker);
  }
  if (err == cudaSuccess && cleanup != cudaSuccess) err = cleanup;
  return static_cast<int>(err);

fail:
  if (ws == nullptr) {
    FreeAllBuffers(d_F, d_R, d_fmask, d_rmask, d_fkey_cur, d_fkey_alt, d_rkey_cur, d_rkey_alt,
                   d_fvals_cur, d_fvals_alt, d_rvals_cur, d_rvals_alt, d_valid, d_rank_f,
                   d_rank_r, d_offsets, d_valid_count, d_fkey_min, d_fkey_max, d_rkey_min,
                   d_rkey_max, d_ic, d_temp, tracker);
  }
  return static_cast<int>(err);
}
