// factor-cuda -- factor_corr v0 GPU kernel.
//
// Pipeline (per factor pair (i,j), pooled over all T*N rows):
//   1. transpose_preprocess : (T,N,F) row-major -> (F,T*N) column-major so each
//      factor column is contiguous (coalesced pair reads) + per-cell valid flag
//      = mask(row) AND finite(value). Mask shared across factor columns.
//   2. reduce_p1_kernel : per pair, per-thread strided accumulation of Partial1
//      (count / sum_x / sum_y / min_x / max_x / min_y / max_y) over jointly
//      valid rows, fixed shared-memory binary-tree reduce (no atomicAdd).
//   3. finalize_p1_kernel : means = sum/count per pair.
//   4. reduce_p2_kernel : per pair, centered accumulation of Partial2
//      (sxx / syy / sxy) using the pass-1 means.
//   5. finalize_p2_kernel : safe_pearson r; bias_metric + |r|>1 + finite trigger
//      (corpus state machine). count<2 -> NaN.
//   6. Kahan re-run for triggered pairs only (CompensatedSum pass 1 + pass 2,
//      compensated means, safe_pearson finalize), overwriting d_corr.
//   7. writeback_kernel : strict lower triangle mirrored to upper triangle so
//      r[i,j] == r[j,i] bitwise; diagonal derived from the computed value --
//      1.0 iff the computed self-correlation is finite (non-degenerate column
//      with positive centered variance), else NaN (count<2 / zero or
//      underflowed variance / non-finite) (review F1, 2026-08-05).
//
// Numerics follow tests/fixtures/corr_math_v1.py: safe_pearson expression
// (sxy/sqrt(sxx))/sqrt(syy); CompensatedSum add/merge conventions; Partial1/2/
// K1/K2 ABI. Reduction order is fixed (deterministic on one device), matching
// the CLAUDE.md determinism clause (no atomicAdd, fixed tree).
//
// The per-thread accumulation operators and the fixed binary-tree reduction are
// shared device-inline helpers declared in factor_corr_impl.cuh, so the F/T
// chunking minimal proof 2 continuation kernels (poc3_factor_corr_selfcheck.cu)
// run byte-identical accumulation/tree logic (see factor_corr_impl.cuh header).
//
// PoC 3 v0. ASCII-only comments (nvcc/GBK pitfall).
#include <cmath>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

#include "factor_corr.cuh"
#include "factor_corr_impl.cuh"

// Build pair table: p = i*(i+1)/2 + j (lower triangle incl diagonal, i>=j).
__global__ void make_pairs_kernel(int* __restrict__ d_pairs, int F) {
  int i = blockIdx.x;
  if (i >= F) return;
  int base = i * (i + 1) / 2;
  for (int j = 0; j <= i; ++j) {
    d_pairs[2 * (base + j)] = i;
    d_pairs[2 * (base + j) + 1] = j;
  }
}

// Transpose (T*N, F) row-major -> (F, T*N) column-major; valid = mask & finite.
// d_Xt layout: column f starts at f*R (R = T*N), contiguous within column.
__global__ void transpose_preprocess(const double* __restrict__ d_src,
                                     const uint8_t* __restrict__ d_mask, int R,
                                     int F, double* __restrict__ d_Xt,
                                     uint8_t* __restrict__ d_valid) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= R) return;
  const double* row = d_src + (size_t)r * F;
  uint8_t m = (d_mask != nullptr) ? d_mask[r] : 1u;
  for (int f = 0; f < F; ++f) {
    double v = row[f];
    uint8_t ok = static_cast<uint8_t>(m != 0u && isfinite(v));
    d_Xt[(size_t)f * R + r] = v;
    d_valid[(size_t)f * R + r] = ok;
  }
}

// Pass 1 (normal): per pair Partial1 over jointly valid rows. One block per
// pair; threads stride rows in increasing order (fixed), then shared binary
// tree reduce (fixed order) -- the shared accum_p1_cell / tree_reduce_p1_store
// helpers make this byte-identical to the chunked continuation path.
__global__ void reduce_p1_kernel(const double* __restrict__ d_Xt,
                                 const uint8_t* __restrict__ d_valid,
                                 const int* __restrict__ d_pairs, int R,
                                 Partial1* __restrict__ d_gp1) {
  const int p = blockIdx.x;
  const int i = d_pairs[2 * p];
  const int j = d_pairs[2 * p + 1];
  const double* xi = d_Xt + (size_t)i * R;
  const double* xj = d_Xt + (size_t)j * R;
  const uint8_t* vi = d_valid + (size_t)i * R;
  const uint8_t* vj = d_valid + (size_t)j * R;
  const int tid = threadIdx.x;

  double cnt = 0.0, sx = 0.0, sy = 0.0;
  double mnx = kBigPos, mxx = kBigNeg, mny = kBigPos, mxy = kBigNeg;
  for (int r = tid; r < R; r += blockDim.x) {
    if (vi[r] != 0u && vj[r] != 0u) {
      accum_p1_cell(cnt, sx, sy, mnx, mxx, mny, mxy, xi[r], xj[r]);
    }
  }
  __shared__ double scnt[256], ssx[256], ssy[256], smnx[256], smxx[256], smny[256], smxy[256];
  scnt[tid] = cnt; ssx[tid] = sx; ssy[tid] = sy;
  smnx[tid] = mnx; smxx[tid] = mxx; smny[tid] = mny; smxy[tid] = mxy;
  tree_reduce_p1_store(scnt, ssx, ssy, smnx, smxx, smny, smxy, tid, blockDim.x, d_gp1, p);
}

// Finalize pass 1: means per pair (n>=1 for pairs reaching pass 2).
__global__ void finalize_p1_kernel(const Partial1* __restrict__ d_gp1, int P,
                                   double* __restrict__ d_means) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= P) return;
  const Partial1& g = d_gp1[p];
  double n = g.count;
  d_means[2 * p] = (n > 0.0) ? g.sum_x / n : 0.0;
  d_means[2 * p + 1] = (n > 0.0) ? g.sum_y / n : 0.0;
}

// Pass 2 (normal): per pair centered Partial2 using pass-1 means.
__global__ void reduce_p2_kernel(const double* __restrict__ d_Xt,
                                 const uint8_t* __restrict__ d_valid,
                                 const int* __restrict__ d_pairs,
                                 const double* __restrict__ d_means, int R,
                                 Partial2* __restrict__ d_gp2) {
  const int p = blockIdx.x;
  const int i = d_pairs[2 * p];
  const int j = d_pairs[2 * p + 1];
  const double* xi = d_Xt + (size_t)i * R;
  const double* xj = d_Xt + (size_t)j * R;
  const uint8_t* vi = d_valid + (size_t)i * R;
  const uint8_t* vj = d_valid + (size_t)j * R;
  const double mx = d_means[2 * p];
  const double my = d_means[2 * p + 1];
  const int tid = threadIdx.x;

  double sxx = 0.0, syy = 0.0, sxy = 0.0;
  for (int r = tid; r < R; r += blockDim.x) {
    if (vi[r] != 0u && vj[r] != 0u) {
      accum_p2_cell(sxx, syy, sxy, xi[r], mx, xj[r], my);
    }
  }
  __shared__ double sh1[256], sh2[256], sh3[256];
  sh1[tid] = sxx;
  sh2[tid] = syy;
  sh3[tid] = sxy;
  tree_reduce_p2_store(sh1, sh2, sh3, tid, blockDim.x, d_gp2, p);
}

// Finalize pass 2: safe_pearson r + trigger. count<2 -> NaN. Exact-constant
// operand on the joint valid set (min==max) -> NaN with NO Kahan trigger (review
// F3: a float mean may not reconstruct a repeated constant like 0.1 exactly, so
// centered pass 2 could manufacture a tiny syy>0 and emit a finite correlation;
// the explicit min==max check is the contract-mandated exact-zero-variance gate).
// bias_metric = max( max|x|/sqrt(sxx/n), max|y|/sqrt(syy/n) ) from Partial1
// min/max and Partial2 sxx/syy. Trigger = bias_metric > 1e8 OR |r| > 1 OR
// non-finite.
__global__ void finalize_p2_kernel(const Partial1* __restrict__ d_gp1,
                                   const Partial2* __restrict__ d_gp2, int P,
                                   double* __restrict__ d_corr,
                                   uint8_t* __restrict__ d_trigger) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= P) return;
  const Partial1& g1 = d_gp1[p];
  const Partial2& g2 = d_gp2[p];
  double n = g1.count;
  double r = NAN;
  uint8_t trig = 0u;
  if (n >= 2.0) {
    const bool const_x = (g1.min_x == g1.max_x);
    const bool const_y = (g1.min_y == g1.max_y);
    if (const_x || const_y) {
      r = NAN;  // exact-constant operand -> NaN; skip Kahan (wouldn't help)
    } else {
      r = safe_pearson(g2.sxy, g2.sxx, g2.syy);
      double mx_abs = fmax(fabs(g1.min_x), fabs(g1.max_x));
      double my_abs = fmax(fabs(g1.min_y), fabs(g1.max_y));
      double bx = mx_abs / std::sqrt(g2.sxx / n);
      double by = my_abs / std::sqrt(g2.syy / n);
      double bias = fmax(bx, by);
      if (bias > kTriggerThreshold || fabs(r) > 1.0 || !isfinite(r)) trig = 1u;
    }
  }
  d_corr[p] = r;
  d_trigger[p] = trig;
}

// Kahan pass 1: compensated count/sum over jointly valid rows. One block per
// triggered pair. Threads stride in increasing order and each accumulates its
// own Kahan state, then a fixed shared-memory binary-tree merge (add right.sum,
// add -right.c per corr_math_v1). blockDim is host-selected: 256 for long
// panels (each thread walks many elements, compensation effective), 1 for
// short panels (single thread walks all elements; a multi-thread split on
// R < 256 gives <=1 element/thread and Kahan degenerates to a plain sum which
// loses ~1.5e-4 on a 1e12 bias, moving corr by ~6e-9).
__global__ void kahan_reduce_p1_kernel(const double* __restrict__ d_Xt,
                                       const uint8_t* __restrict__ d_valid,
                                       const int* __restrict__ d_pairs,
                                       const int* __restrict__ d_trig_pairs,
                                       int R, PartialK1* __restrict__ d_gk1) {
  const int k = blockIdx.x;
  const int p = d_trig_pairs[k];
  const int i = d_pairs[2 * p];
  const int j = d_pairs[2 * p + 1];
  const double* xi = d_Xt + (size_t)i * R;
  const double* xj = d_Xt + (size_t)j * R;
  const uint8_t* vi = d_valid + (size_t)i * R;
  const uint8_t* vj = d_valid + (size_t)j * R;
  const int tid = threadIdx.x;

  double cnt = 0.0;
  KahanAcc sx, sy;
  sx.sum = 0.0; sx.c = 0.0;
  sy.sum = 0.0; sy.c = 0.0;
  for (int r = tid; r < R; r += blockDim.x) {
    if (vi[r] != 0u && vj[r] != 0u) {
      cnt += 1.0;
      sx.add(xi[r]);
      sy.add(xj[r]);
    }
  }
  __shared__ double scnt[256], ssumx[256], scx[256], ssumy[256], scy[256];
  scnt[tid] = cnt;
  ssumx[tid] = sx.sum; scx[tid] = sx.c;
  ssumy[tid] = sy.sum; scy[tid] = sy.c;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      scnt[tid] += scnt[tid + s];
      KahanAcc rx; rx.sum = ssumx[tid + s]; rx.c = scx[tid + s];
      KahanAcc lx; lx.sum = ssumx[tid]; lx.c = scx[tid]; lx.merge(rx);
      ssumx[tid] = lx.sum; scx[tid] = lx.c;
      KahanAcc ry; ry.sum = ssumy[tid + s]; ry.c = scy[tid + s];
      KahanAcc ly; ly.sum = ssumy[tid]; ly.c = scy[tid]; ly.merge(ry);
      ssumy[tid] = ly.sum; scy[tid] = ly.c;
    }
    __syncthreads();
  }
  if (tid == 0) {
    PartialK1 out;
    out.count = scnt[0];
    out.sum_x = ssumx[0]; out.c_x = scx[0];
    out.sum_y = ssumy[0]; out.c_y = scy[0];
    d_gk1[k] = out;
  }
}

// Kahan finalize pass 1: compensated means per triggered pair.
__global__ void kahan_finalize_p1_kernel(const PartialK1* __restrict__ d_gk1,
                                         int K, double* __restrict__ d_kmeans) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= K) return;
  const PartialK1& g = d_gk1[k];
  double n = g.count;
  d_kmeans[2 * k] = (n > 0.0) ? g.represented_x() / n : 0.0;
  d_kmeans[2 * k + 1] = (n > 0.0) ? g.represented_y() / n : 0.0;
}

// Kahan pass 2: compensated centered sums of squares / cross using compensated
// means. Same blockDim strategy as pass 1 (256 for long panels, 1 for short).
__global__ void kahan_reduce_p2_kernel(const double* __restrict__ d_Xt,
                                       const uint8_t* __restrict__ d_valid,
                                       const int* __restrict__ d_pairs,
                                       const int* __restrict__ d_trig_pairs,
                                       const double* __restrict__ d_kmeans,
                                       int R, PartialK2* __restrict__ d_gk2) {
  const int k = blockIdx.x;
  const int p = d_trig_pairs[k];
  const int i = d_pairs[2 * p];
  const int j = d_pairs[2 * p + 1];
  const double* xi = d_Xt + (size_t)i * R;
  const double* xj = d_Xt + (size_t)j * R;
  const uint8_t* vi = d_valid + (size_t)i * R;
  const uint8_t* vj = d_valid + (size_t)j * R;
  const double mx = d_kmeans[2 * k];
  const double my = d_kmeans[2 * k + 1];
  const int tid = threadIdx.x;

  KahanAcc a_xx, a_yy, a_xy;
  a_xx.sum = 0.0; a_xx.c = 0.0;
  a_yy.sum = 0.0; a_yy.c = 0.0;
  a_xy.sum = 0.0; a_xy.c = 0.0;
  for (int r = tid; r < R; r += blockDim.x) {
    if (vi[r] != 0u && vj[r] != 0u) {
      double dx = xi[r] - mx, dy = xj[r] - my;
      a_xx.add(dx * dx);
      a_yy.add(dy * dy);
      a_xy.add(dx * dy);
    }
  }
  __shared__ double sxx[256], cxx[256], syy[256], cyy[256], sxy[256], cxy[256];
  sxx[tid] = a_xx.sum; cxx[tid] = a_xx.c;
  syy[tid] = a_yy.sum; cyy[tid] = a_yy.c;
  sxy[tid] = a_xy.sum; cxy[tid] = a_xy.c;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      KahanAcc rx; rx.sum = sxx[tid + s]; rx.c = cxx[tid + s];
      KahanAcc lx; lx.sum = sxx[tid]; lx.c = cxx[tid]; lx.merge(rx);
      sxx[tid] = lx.sum; cxx[tid] = lx.c;
      KahanAcc ry; ry.sum = syy[tid + s]; ry.c = cyy[tid + s];
      KahanAcc ly; ly.sum = syy[tid]; ly.c = cyy[tid]; ly.merge(ry);
      syy[tid] = ly.sum; cyy[tid] = ly.c;
      KahanAcc rz; rz.sum = sxy[tid + s]; rz.c = cxy[tid + s];
      KahanAcc lz; lz.sum = sxy[tid]; lz.c = cxy[tid]; lz.merge(rz);
      sxy[tid] = lz.sum; cxy[tid] = lz.c;
    }
    __syncthreads();
  }
  if (tid == 0) {
    PartialK2 out;
    out.sxx = sxx[0]; out.c_xx = cxx[0];
    out.syy = syy[0]; out.c_yy = cyy[0];
    out.sxy = sxy[0]; out.c_xy = cxy[0];
    d_gk2[k] = out;
  }
}

// Kahan finalize pass 2: safe_pearson of compensated sums, overwrite d_corr.
__global__ void kahan_finalize_p2_kernel(const PartialK2* __restrict__ d_gk2,
                                         const int* __restrict__ d_trig_pairs,
                                         int K, double* __restrict__ d_corr) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= K) return;
  const PartialK2& g = d_gk2[k];
  double sxx = g.represented_xx();
  double syy = g.represented_yy();
  double sxy = g.represented_xy();
  d_corr[d_trig_pairs[k]] = safe_pearson(sxy, sxx, syy);
}

// Writeback: strict lower triangle mirrored to upper (bitwise equal); diagonal
// derived from the COMPUTED path's value (review F1, 2026-08-05): finalize_p2
// (and the Kahan re-run) already emit NaN when the column's centered variance
// underflows to zero or is zero, so the diagonal is 1.0 iff the computed
// self-correlation is finite, else NaN. This matches the frozen oracle
// (np.corrcoef var underflow -> NaN) instead of blindly forcing 1.0 from a
// count>=2 && min!=max test.
__global__ void writeback_kernel(const double* __restrict__ d_corr,
                                 const int* __restrict__ d_pairs, int F, int P,
                                 double* __restrict__ d_out) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= P) return;
  const int i = d_pairs[2 * p];
  const int j = d_pairs[2 * p + 1];
  if (i == j) {
    double v = d_corr[p];  // computed diagonal (normal + Kahan paths)
    d_out[i * F + j] = isfinite(v) ? 1.0 : NAN;  // explicit 1.0 when non-degenerate
  } else {
    double v = d_corr[p];
    d_out[i * F + j] = v;
    d_out[j * F + i] = v;  // mirror: bitwise equal by construction
  }
}

namespace {

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

cudaError_t FreeAllBuffers(void* d_Xt, void* d_valid, void* d_pairs, void* d_gp1,
                           void* d_means, void* d_gp2, void* d_corr,
                           void* d_trigger, void* d_trig_pairs, void* d_gk1,
                           void* d_kmeans, void* d_gk2, void* d_out,
                           void* d_F, void* d_mask,
                           factor_cuda::MemTracker* tracker) {
  cudaError_t e = cudaSuccess;
  auto keep_first = [&e](cudaError_t r) {
    if (e == cudaSuccess && r != cudaSuccess) e = r;
  };
  keep_first(FreeOrTrack(d_Xt, tracker));
  keep_first(FreeOrTrack(d_valid, tracker));
  keep_first(FreeOrTrack(d_pairs, tracker));
  keep_first(FreeOrTrack(d_gp1, tracker));
  keep_first(FreeOrTrack(d_means, tracker));
  keep_first(FreeOrTrack(d_gp2, tracker));
  keep_first(FreeOrTrack(d_corr, tracker));
  keep_first(FreeOrTrack(d_trigger, tracker));
  keep_first(FreeOrTrack(d_trig_pairs, tracker));
  keep_first(FreeOrTrack(d_gk1, tracker));
  keep_first(FreeOrTrack(d_kmeans, tracker));
  keep_first(FreeOrTrack(d_gk2, tracker));
  keep_first(FreeOrTrack(d_out, tracker));
  keep_first(FreeOrTrack(d_F, tracker));
  keep_first(FreeOrTrack(d_mask, tracker));  // nullptr is a safe no-op
  return e;
}

}  // namespace

int factor_corr_gpu(const double* h_F, const uint8_t* h_mask, int T, int N,
                    int F, double* h_out, factor_cuda::MemTracker* tracker,
                    uint8_t* h_trigger_out) {
  // ---- host preconditions (contract errors) --------------------------------
  if (h_F == nullptr || h_out == nullptr || T < 1 || N < 1 || F < 1) return -1;
  if (static_cast<int64_t>(T) * N > INT32_MAX) return -2;  // checked_mul T*N
  if (F > kMaxF) return -3;                                // F > 128 grid cap

  const int R = T * N;                   // pooled row count
  const int P = F * (F + 1) / 2;         // pairs (lower triangle incl diag)
  const size_t cols_bytes = static_cast<size_t>(F) * R * sizeof(double);
  const size_t valid_bytes = static_cast<size_t>(F) * R;

  cudaError_t cleanup = cudaSuccess;

  double* d_Xt = nullptr;
  uint8_t* d_valid = nullptr;
  int* d_pairs = nullptr;
  Partial1* d_gp1 = nullptr;
  double* d_means = nullptr;
  Partial2* d_gp2 = nullptr;
  double* d_corr = nullptr;
  uint8_t* d_trigger = nullptr;
  int* d_trig_pairs = nullptr;
  PartialK1* d_gk1 = nullptr;
  double* d_kmeans = nullptr;
  PartialK2* d_gk2 = nullptr;
  double* d_out = nullptr;
  double* d_F = nullptr;
  uint8_t* d_mask = nullptr;

  cudaError_t err = AllocOrTrack(&d_Xt, cols_bytes, "d_Xt", "alloc", tracker);
  if (err == cudaSuccess) err = AllocOrTrack(&d_valid, valid_bytes, "d_valid", "alloc", tracker);
  if (err == cudaSuccess) err = AllocOrTrack(&d_pairs, static_cast<size_t>(P) * 2 * sizeof(int), "d_pairs", "alloc", tracker);
  if (err == cudaSuccess) err = AllocOrTrack(&d_gp1, static_cast<size_t>(P) * sizeof(Partial1), "d_gp1", "alloc", tracker);
  if (err == cudaSuccess) err = AllocOrTrack(&d_means, static_cast<size_t>(P) * 2 * sizeof(double), "d_means", "alloc", tracker);
  if (err == cudaSuccess) err = AllocOrTrack(&d_gp2, static_cast<size_t>(P) * sizeof(Partial2), "d_gp2", "alloc", tracker);
  if (err == cudaSuccess) err = AllocOrTrack(&d_corr, static_cast<size_t>(P) * sizeof(double), "d_corr", "alloc", tracker);
  if (err == cudaSuccess) err = AllocOrTrack(&d_trigger, static_cast<size_t>(P), "d_trigger", "alloc", tracker);
  if (err == cudaSuccess) err = AllocOrTrack(&d_trig_pairs, static_cast<size_t>(P) * sizeof(int), "d_trig_pairs", "alloc", tracker);
  if (err == cudaSuccess) err = AllocOrTrack(&d_gk1, static_cast<size_t>(P) * sizeof(PartialK1), "d_gk1", "alloc", tracker);
  if (err == cudaSuccess) err = AllocOrTrack(&d_kmeans, static_cast<size_t>(P) * 2 * sizeof(double), "d_kmeans", "alloc", tracker);
  if (err == cudaSuccess) err = AllocOrTrack(&d_gk2, static_cast<size_t>(P) * sizeof(PartialK2), "d_gk2", "alloc", tracker);
  if (err == cudaSuccess) err = AllocOrTrack(&d_out, static_cast<size_t>(F) * F * sizeof(double), "d_out", "alloc", tracker);
  if (err == cudaSuccess) err = AllocOrTrack(&d_F, static_cast<size_t>(R) * F * sizeof(double), "d_F", "alloc", tracker);
  if (err == cudaSuccess && h_mask != nullptr) err = AllocOrTrack(&d_mask, static_cast<size_t>(R), "d_mask", "alloc", tracker);
  if (err != cudaSuccess) goto fail;

  // ---- pair table ----------------------------------------------------------
  make_pairs_kernel<<<F, 1>>>(d_pairs, F);
  err = cudaGetLastError();
  if (err != cudaSuccess) goto fail;

  // ---- upload inputs -------------------------------------------------------
  err = cudaMemcpy(d_F, h_F, static_cast<size_t>(R) * F * sizeof(double), cudaMemcpyHostToDevice);
  if (err == cudaSuccess && h_mask != nullptr) {
    err = cudaMemcpy(d_mask, h_mask, static_cast<size_t>(R), cudaMemcpyHostToDevice);
  }
  if (err != cudaSuccess) goto fail;

  // ---- transpose + valid ---------------------------------------------------
  {
    const int block = 256;
    const int grid = static_cast<int>(1 + (static_cast<int64_t>(R) - 1) / block);
    transpose_preprocess<<<grid, block>>>(d_F, d_mask, R, F, d_Xt, d_valid);
    err = cudaGetLastError();
    if (err != cudaSuccess) goto fail;
  }
  // d_F is no longer needed after transpose; free now to lower peak. Only
  // clear the pointer on success -- if the free errored (review F4), the fail
  // path's FreeAllBuffers must still see d_F non-null to retry the release.
  cleanup = FreeOrTrack(d_F, tracker);
  if (err == cudaSuccess && cleanup != cudaSuccess) err = cleanup;
  if (cleanup == cudaSuccess) d_F = nullptr;
  if (err != cudaSuccess) goto fail;

  // ---- pass 1 (normal) ------------------------------------------------------
  reduce_p1_kernel<<<P, 256>>>(d_Xt, d_valid, d_pairs, R, d_gp1);
  err = cudaGetLastError();
  if (err != cudaSuccess) goto fail;
  {
    const int block = 256;
    const int grid = (P + block - 1) / block;
    finalize_p1_kernel<<<grid, block>>>(d_gp1, P, d_means);
    err = cudaGetLastError();
    if (err != cudaSuccess) goto fail;
  }

  // ---- pass 2 (normal) ------------------------------------------------------
  reduce_p2_kernel<<<P, 256>>>(d_Xt, d_valid, d_pairs, d_means, R, d_gp2);
  err = cudaGetLastError();
  if (err != cudaSuccess) goto fail;
  {
    const int block = 256;
    const int grid = (P + block - 1) / block;
    finalize_p2_kernel<<<grid, block>>>(d_gp1, d_gp2, P, d_corr, d_trigger);
    err = cudaGetLastError();
    if (err != cudaSuccess) goto fail;
  }

  // ---- Kahan re-run for triggered pairs --------------------------------------
  {
    std::vector<uint8_t> host_trig(static_cast<size_t>(P));
    err = cudaMemcpy(host_trig.data(), d_trigger, static_cast<size_t>(P), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto fail;
    std::vector<int> trig_idx;
    trig_idx.reserve(static_cast<size_t>(P));
    for (int p = 0; p < P; ++p) {
      if (host_trig[static_cast<size_t>(p)] != 0u) trig_idx.push_back(p);
    }
    const int K = static_cast<int>(trig_idx.size());
    if (K > 0) {
      err = cudaMemcpy(d_trig_pairs, trig_idx.data(), static_cast<size_t>(K) * sizeof(int),
                       cudaMemcpyHostToDevice);
      if (err != cudaSuccess) goto fail;

      // Kahan scans: blockDim = 256 when the panel is long enough that each
      // thread sees several elements (compensation effective); blockDim = 1
      // on short panels so the single thread still walks all elements (a
      // strided multi-thread split on R < 256 gives <=1 element/thread and
      // Kahan degenerates to a plain sum). Triggered pairs are rare, so the
      // serialized case is acceptable.
      const int kb = (R >= 256) ? 256 : 1;
      kahan_reduce_p1_kernel<<<K, kb>>>(d_Xt, d_valid, d_pairs, d_trig_pairs, R, d_gk1);
      err = cudaGetLastError();
      if (err != cudaSuccess) goto fail;
      {
        const int block = 256;
        const int grid = (K + block - 1) / block;
        kahan_finalize_p1_kernel<<<grid, block>>>(d_gk1, K, d_kmeans);
        err = cudaGetLastError();
        if (err != cudaSuccess) goto fail;
      }
      kahan_reduce_p2_kernel<<<K, kb>>>(d_Xt, d_valid, d_pairs, d_trig_pairs, d_kmeans, R, d_gk2);
      err = cudaGetLastError();
      if (err != cudaSuccess) goto fail;
      {
        const int block = 256;
        const int grid = (K + block - 1) / block;
        kahan_finalize_p2_kernel<<<grid, block>>>(d_gk2, d_trig_pairs, K, d_corr);
        err = cudaGetLastError();
        if (err != cudaSuccess) goto fail;
      }
    }
  }

  // ---- writeback + download ------------------------------------------------
  {
    const int block = 256;
    const int grid = (P + block - 1) / block;
    writeback_kernel<<<grid, block>>>(d_corr, d_pairs, F, P, d_out);
    err = cudaGetLastError();
    if (err != cudaSuccess) goto fail;
  }
  err = cudaMemcpy(h_out, d_out, static_cast<size_t>(F) * F * sizeof(double), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) goto fail;
  if (h_trigger_out != nullptr) {
    err = cudaMemcpy(h_trigger_out, d_trigger, static_cast<size_t>(P), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto fail;
  }

  cleanup = FreeAllBuffers(d_Xt, d_valid, d_pairs, d_gp1, d_means, d_gp2, d_corr,
                           d_trigger, d_trig_pairs, d_gk1, d_kmeans, d_gk2, d_out,
                           d_F, d_mask, tracker);
  if (err == cudaSuccess && cleanup != cudaSuccess) err = cleanup;
  return static_cast<int>(err);

fail:
  FreeAllBuffers(d_Xt, d_valid, d_pairs, d_gp1, d_means, d_gp2, d_corr, d_trigger,
                 d_trig_pairs, d_gk1, d_kmeans, d_gk2, d_out, d_F, d_mask, tracker);
  return static_cast<int>(err);  // failure path keeps the original primary error
}
