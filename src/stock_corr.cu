// factor-cuda -- stock_corr v2 GPU kernel (masked-GEMM, upper triangle).
//
// v1 (v0's per-pair two-pass O(N^2*T)) -> one-pass masked-GEMM computing every
// pair (i,j) i<=j in a single sweep over T, sharing column reads via shared-
// memory tiles. v2 adds a fully-valid FAST path (de-mean Gram, 1 accumulator).
//
// Pipeline:
//   1. transpose_preprocess : (T,N) row-major -> two (N,T) column-major
//      matrices: d_Xm (valid values, invalid -> 0.0) and d_M (valid -> 1.0,
//      invalid -> 0.0). Column i contiguous (coalesced reads).
//   2. col_stats_kernel : per column count/sum/min/max over its valid rows
//      (diagonal degeneracy decision + dispatch).
//   3. host dispatch : every column count==T (fully valid) -> FAST path;
//      any partial validity -> GENERAL path.
//   4a. FAST path (demean_kernel + fast_gemm_corr_kernel): de-mean each column
//      by its serial-Kahan mean (bit-matching the reference), precompute
//      S2=sum(xd^2), then 1 accumulator Sxy=sum(xd_i*xd_j) per cell; corr =
//      Sxy/sqrt(S2_i)/sqrt(S2_j). Algebraically identical to the two-pass
//      reference (same xd), so no cancellation detection / fall-back; compute
//      is 1/6 of the general path. Hierarchical accumulation keeps the error
//      T-independent.
//   4b. GENERAL path (gemm_corr_kernel + fallback_kernel): 6 accumulators over
//      t: n, sumx, sumy, sumxy, sumx2, sumy2 -> uncentered-Gram corr.
//      Cancellation detection (|uncentered term| > 3x the residual, |r|>1) and
//      near-zero-var (exact-constant on the JOINT set) write NaN; the separated
//      fallback_kernel then recomputes upper-triangle NaN cells with a per-pair
//      two-pass centered Pearson (Kahan mean + min/max gate), mirroring
//      benchmarks/backends.py _gemm_cancel_mask semantics.
//   5. writeback_kernel : strict lower triangle mirrored (r[i,j]==r[j,i]
//      bitwise); diagonal = 1.0 for count>=2 && min!=max else NaN.
//
// Numerics follow tests/fixtures/corr_math_v1.py (safe_pearson, Kahan) and
// benchmarks/backends.py _masked_gemm_stats.
//
// PoC 3 v2. ASCII-only comments (nvcc/GBK pitfall).
#include <cmath>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

#include "stock_corr.cuh"
#include "stock_corr_impl.cuh"

// Kernel definitions live in namespace stock_corr_impl (declared in
// stock_corr_impl.cuh); the shared constants / ColStats / KahanAcc /
// safe_pearson / two_pass_centered are defined in the header so the N-blocking
// proof harness launches the SAME production kernels. Host-side helpers stay in
// an anonymous namespace below. using-namespace keeps the driver code below
// terse; the kernel symbols remain unambiguous because factor_corr's kernels
// are global-scope (stock's are scoped).
using namespace stock_corr_impl;

namespace stock_corr_impl {

// Transpose (T,N) row-major -> (N,T) column-major; Xm = valid?value:0,
// M = valid?1:0. valid = mask & finite.
__global__ void transpose_preprocess(const double* __restrict__ d_src,
                                     const uint8_t* __restrict__ d_mask, int T,
                                     int N, double* __restrict__ d_Xm,
                                     double* __restrict__ d_M) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= T * N) return;
  int t = e / N, i = e % N;
  double v = d_src[static_cast<size_t>(e)];
  uint8_t m = (d_mask != nullptr) ? d_mask[static_cast<size_t>(e)] : 1u;
  uint8_t ok = static_cast<uint8_t>(m != 0u && isfinite(v));
  d_Xm[(size_t)i * T + t] = ok ? v : 0.0;
  d_M[(size_t)i * T + t] = ok ? 1.0 : 0.0;
}

// Per-column stats: count/sum/min/max over valid rows (valid = M==1). One block
// per column; strided + shared binary-tree reduce.
__global__ void col_stats_kernel(const double* __restrict__ d_Xm,
                                 const double* __restrict__ d_M, int T,
                                 ColStats* __restrict__ d_stats) {
  const int i = blockIdx.x;
  const double* xi = d_Xm + (size_t)i * T;
  const double* mi = d_M + (size_t)i * T;
  const int tid = threadIdx.x;

  double cnt = 0.0, s = 0.0, mn = kBigPos, mx = kBigNeg;
  double mn_abs = kBigPos, mx_abs = 0.0;
  for (int t = tid; t < T; t += blockDim.x) {
    if (mi[t] != 0.0) {
      double x = xi[t];
      cnt += 1.0;
      s += x;
      mn = fmin(mn, x);
      mx = fmax(mx, x);
      double ax = fabs(x);
      mx_abs = fmax(mx_abs, ax);
      if (ax != 0.0) mn_abs = fmin(mn_abs, ax);  // min NONZERO |x|
    }
  }
  __shared__ double scnt[256], ss[256], smn[256], smx[256], smna[256], smxa[256];
  scnt[tid] = cnt; ss[tid] = s; smn[tid] = mn; smx[tid] = mx;
  smna[tid] = mn_abs; smxa[tid] = mx_abs;
  __syncthreads();
  for (int s_ = blockDim.x / 2; s_ > 0; s_ >>= 1) {
    if (tid < s_) {
      scnt[tid] += scnt[tid + s_];
      ss[tid] += ss[tid + s_];
      smn[tid] = fmin(smn[tid], smn[tid + s_]);
      smx[tid] = fmax(smx[tid], smx[tid + s_]);
      smna[tid] = fmin(smna[tid], smna[tid + s_]);
      smxa[tid] = fmax(smxa[tid], smxa[tid + s_]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    ColStats c;
    c.count = scnt[0]; c.sum = ss[0]; c.min = smn[0]; c.max = smx[0];
    c.min_abs = smna[0]; c.max_abs = smxa[0];
    d_stats[i] = c;
  }
}

// Upper-triangle masked-GEMM correlation kernel (GENERAL path), 1 output cell
// per thread. Block (i0,j0) covers rows [i0,i0+16) x cols [j0,j0+16); thread
// (tx,ty) computes (i0+tx, j0+ty). 1D grid enumerates ONLY upper-triangle
// tiles so no block is wasted (2D full grid idles ~half on the strict lower
// triangle); a 16x16 tile keeps the block count high enough to fill the GPU at
// small N (e.g. N=500 -> 528 blocks). Only cells with i<=j are written (the
// strict lower triangle is mirrored in writeback). Cancellation / |r|>1 /
// near-zero-var cells write NaN; the separated fallback_kernel recomputes them
// with two_pass_centered. Deterministic.
__global__ void gemm_corr_kernel(const double* __restrict__ d_Xm,
                                 const double* __restrict__ d_M, int T, int N,
                                 int nt, double* __restrict__ d_corr) {
  // 1D grid -> upper-triangle tile (i_block <= j_block), row-major: row ib
  // holds (ib,ib)..(ib,nt-1); prefix P(ib)=ib*nt-ib*(ib-1)/2. Binary search the
  // largest ib with P(ib) <= b.
  const int b = blockIdx.x;
  int lo = 0, hi = nt - 1;
  while (lo < hi) {
    int mid = (lo + hi + 1) / 2;
    long long pm = (long long)mid * nt - (long long)mid * (mid - 1) / 2;
    if (pm <= b) lo = mid; else hi = mid - 1;
  }
  const int ib = lo;
  const long long p = (long long)ib * nt - (long long)ib * (ib - 1) / 2;
  const int jb = ib + static_cast<int>(b - p);
  const int i0 = ib * kTile;
  const int j0 = jb * kBCols;

  const int tx = threadIdx.x & (kTile - 1);  // row within tile (0..15)
  const int ty = threadIdx.x >> 4;           // col within tile (0..15)

  extern __shared__ double smem[];  // 8-byte aligned by the compiler (review F15)
  double* sh_Ax = smem;                                // [16][kKtile+1]
  double* sh_Am = sh_Ax + kTile * kRowPad;
  double* sh_Bx = sh_Am + kTile * kRowPad;             // [16][kKtile+1]
  double* sh_Bm = sh_Bx + kBCols * kRowPad;

  // Hierarchical (two-level) accumulation so the fast-path error is independent
  // of T (review F8): each kKtile window is summed serially into a local partial
  // (klen terms, error ~klen*eps), then Kahan-merged into the running total
  // (error ~2*eps). Total ~(klen+2)*eps ~ 7.5e-15 regardless of T, vs ~T*eps for
  // a single serial chain (which breaks 1e-12 beyond T ~ 4500). Joint-value
  // reuse (review F17) keeps the inner loop at 12 FP64 flops per cell-t.
  KahanAcc rn, rsx, rsy, rsxy, rsx2, rsy2;
  rn.sum = 0.0;  rn.c = 0.0;
  rsx.sum = 0.0; rsx.c = 0.0;
  rsy.sum = 0.0; rsy.c = 0.0;
  rsxy.sum = 0.0; rsxy.c = 0.0;
  rsx2.sum = 0.0; rsx2.c = 0.0;
  rsy2.sum = 0.0; rsy2.c = 0.0;

  for (int k0 = 0; k0 < T; k0 += kKtile) {
    const int klen = (k0 + kKtile <= T) ? kKtile : (T - k0);
    // stage A tile: rows [i0,i0+16) x K-window [k0,k0+klen)
    for (int e = threadIdx.x; e < kTile * klen; e += kBlock) {
      int ii = e / klen, kk = e % klen;
      int gi = i0 + ii;
      int t = k0 + kk;
      double xv = (gi < N) ? d_Xm[(size_t)gi * T + t] : 0.0;
      double mv = (gi < N) ? d_M[(size_t)gi * T + t] : 0.0;
      sh_Ax[ii * kRowPad + kk] = xv;
      sh_Am[ii * kRowPad + kk] = mv;
    }
    // stage B tile: rows [j0,j0+16) x K-window
    for (int e = threadIdx.x; e < kBCols * klen; e += kBlock) {
      int jj = e / klen, kk = e % klen;
      int gj = j0 + jj;
      int t = k0 + kk;
      double xv = (gj < N) ? d_Xm[(size_t)gj * T + t] : 0.0;
      double mv = (gj < N) ? d_M[(size_t)gj * T + t] : 0.0;
      sh_Bx[jj * kRowPad + kk] = xv;
      sh_Bm[jj * kRowPad + kk] = mv;
    }
    __syncthreads();

    const int i = i0 + tx, j = j0 + ty;
    if (i < N && j < N && i <= j) {
      double nl = 0.0, sxl = 0.0, syl = 0.0, sxyl = 0.0, sx2l = 0.0, sy2l = 0.0;
      for (int kk = 0; kk < klen; ++kk) {
        const double ax = sh_Ax[tx * kRowPad + kk];
        const double am = sh_Am[tx * kRowPad + kk];
        const double bx = sh_Bx[ty * kRowPad + kk];
        const double bm = sh_Bm[ty * kRowPad + kk];
        const double jx = ax * bm;  // sx / sx2 shared product
        const double jy = am * bx;  // sy / sy2 shared product
        nl   += am * bm;
        sxl  += jx;
        syl  += jy;
        sxyl += ax * bx;
        sx2l += ax * jx;            // ax*ax*bm
        sy2l += bx * jy;            // am*bx*bx
      }
      // Kahan-merge the window partial into the running total.
      rn.add(nl);
      rsx.add(sxl);
      rsy.add(syl);
      rsxy.add(sxyl);
      rsx2.add(sx2l);
      rsy2.add(sy2l);
    }
    __syncthreads();
  }

  const int i = i0 + tx, j = j0 + ty;
  if (i < N && j < N && i <= j) {
    CellRes r = finalize_cell(rn.sum - rn.c, rsx.sum - rsx.c, rsy.sum - rsy.c,
                              rsxy.sum - rsxy.c, rsx2.sum - rsx2.c,
                              rsy2.sum - rsy2.c);
    // v2: fall-back cells are written as NaN here; the separated fallback_kernel
    // (launched after this kernel) recomputes upper-triangle NaN cells with the
    // two-pass centered Pearson. Keeps the two-pass out of the hot loop (less
    // register pressure / divergence).
    d_corr[(size_t)i * N + j] = r.fallback ? NAN : r.r;
  }
}

// Per-column serial-Kahan de-mean for the fully-valid fast path (v2). One
// thread per column (blockDim 256, grid ceil(N/256)). Pass 1 computes the
// column mean with a serial Kahan sum over the valid rows IN ROW ORDER --
// bit-matching the selfcheck cpu_pair_corr / corr_math_v1 reference so that on
// large-bias inputs the de-meaned values are identical and |dr| stays <= 1e-12
// (any other reduction order shifts the mean by ~ulp(mu) ~ 2.4e-3 on bias_1e12
// data, which the Gram amplifies to dr ~ 1e-5). Pass 2 writes xd = x - mean in
// place and accumulates S2 = sum(xd^2) with serial Kahan (== reference sxx).
// The host only dispatches here when every column is fully valid (count == T),
// so the joint set of every pair is the full column and column stats are exact.
__global__ void demean_kernel(double* __restrict__ d_Xm, int T, int N,
                              double* __restrict__ d_s2) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= N) return;
  double* xcol = d_Xm + (size_t)col * T;
  KahanAcc sk;
  sk.sum = 0.0; sk.c = 0.0;
  for (int t = 0; t < T; ++t) sk.add(xcol[t]);  // serial, row order
  const double mu = sk.sum / T;                 // == reference mean (uncompensated)
  KahanAcc s2k;
  s2k.sum = 0.0; s2k.c = 0.0;
  for (int t = 0; t < T; ++t) {
    double xd = xcol[t] - mu;
    xcol[t] = xd;
    s2k.add(xd * xd);
  }
  d_s2[col] = s2k.sum - s2k.c;
}

// Fully-valid fast-path correlation kernel (v2): 1 accumulator per cell.
// The de-mean Gram r = Sxy/sqrt(S2_i)/sqrt(S2_j) with a serial-Kahan column
// mean is algebraically identical to the two-pass centered reference (same
// xd), so NO cancellation detection / fall-back is needed and the compute is
// 1/6 of the general 6-accumulator GEMM. Hierarchical accumulation (per-kKtile
// serial partial + Kahan merge) keeps the error T-independent (review F8).
// safe_pearson's sequential division avoids the S2i*S2j underflow for
// tiny-scale (~1e-150) data.
//
// Exact-constant gate (review F1, 2026-08-05): S2==0 does NOT reliably mark a
// constant column -- the serial-Kahan mean may not round back to the column
// value (e.g. 0.1 at T=3 -> mu=0.10000000000000002), leaving a nonzero residue
// xd and S2>0. The reference's min==max gate (cpu_pair_corr / corr_math_v1)
// must be mirrored: a constant operand -> NaN. For a fully-valid panel the
// column min/max (from col_stats) equal the pair's joint min/max, so the gate
// is per-column. Only i<=j cells are written (writeback mirrors).
__global__ void fast_gemm_corr_kernel(const double* __restrict__ d_Xd,
                                      const double* __restrict__ d_s2,
                                      const ColStats* __restrict__ d_stats,
                                      int T, int N, int nt,
                                      double* __restrict__ d_corr) {
  const int b = blockIdx.x;
  int lo = 0, hi = nt - 1;
  while (lo < hi) {
    int mid = (lo + hi + 1) / 2;
    long long pm = (long long)mid * nt - (long long)mid * (mid - 1) / 2;
    if (pm <= b) lo = mid; else hi = mid - 1;
  }
  const int ib = lo;
  const long long p = (long long)ib * nt - (long long)ib * (ib - 1) / 2;
  const int jb = ib + static_cast<int>(b - p);
  const int i0 = ib * kTile;
  const int j0 = jb * kBCols;
  const int tx = threadIdx.x & (kTile - 1);
  const int ty = threadIdx.x >> 4;

  extern __shared__ double smem[];  // 8-byte aligned
  double* sh_Ax = smem;             // [16][kKtile+1]
  double* sh_Bx = sh_Ax + kTile * kRowPad;  // [16][kKtile+1]
  // no M tiles: fully valid panel, every cell participates

  KahanAcc rsxy;
  rsxy.sum = 0.0; rsxy.c = 0.0;
  for (int k0 = 0; k0 < T; k0 += kKtile) {
    const int klen = (k0 + kKtile <= T) ? kKtile : (T - k0);
    for (int e = threadIdx.x; e < kTile * klen; e += kBlock) {
      int ii = e / klen, kk = e % klen;
      int gi = i0 + ii;
      int t = k0 + kk;
      sh_Ax[ii * kRowPad + kk] = (gi < N) ? d_Xd[(size_t)gi * T + t] : 0.0;
    }
    for (int e = threadIdx.x; e < kBCols * klen; e += kBlock) {
      int jj = e / klen, kk = e % klen;
      int gj = j0 + jj;
      int t = k0 + kk;
      sh_Bx[jj * kRowPad + kk] = (gj < N) ? d_Xd[(size_t)gj * T + t] : 0.0;
    }
    __syncthreads();
    const int i = i0 + tx, j = j0 + ty;
    if (i < N && j < N && i <= j) {
      double sxyl = 0.0;
      for (int kk = 0; kk < klen; ++kk)
        sxyl += sh_Ax[tx * kRowPad + kk] * sh_Bx[ty * kRowPad + kk];
      rsxy.add(sxyl);
    }
    __syncthreads();
  }
  const int i = i0 + tx, j = j0 + ty;
  if (i < N && j < N && i <= j) {
    // exact-constant operand -> NaN (min==max gate mirrors the reference and
    // the general path's joint-constant detection; S2 alone is unreliable).
    const ColStats& si = d_stats[i];
    const ColStats& sj = d_stats[j];
    if (si.min == si.max || sj.min == sj.max) {
      d_corr[(size_t)i * N + j] = NAN;
    } else {
      double Sxy = rsxy.sum - rsxy.c;
      d_corr[(size_t)i * N + j] = safe_pearson(Sxy, d_s2[i], d_s2[j]);
    }
  }
}

// Separated two-pass fall-back for the general path (v2). Scans the
// upper-triangle cells INCLUDING the diagonal (i<=j) of d_corr and recomputes
// any NaN cell with the two-pass centered Pearson (the reference algorithm;
// degenerate cells -- n<2, exact-constant joint, var underflow -- keep NaN
// from two_pass). The diagonal is included so writeback can derive its value
// from the computed path (review F1: a diagonal must be NaN when the column's
// centered variance underflows to zero, not blindly 1.0 from min!=max).
__global__ void fallback_kernel(const double* __restrict__ d_Xm,
                                const double* __restrict__ d_M, int T, int N,
                                double* __restrict__ d_corr,
                                int* __restrict__ d_fb_count) {
  const int64_t k = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total = (int64_t)N * (N + 1) / 2;  // i<=j (incl diagonal)
  if (k >= total) return;
  int lo = 0, hi = N - 1;
  while (lo < hi) {
    int mid = (lo + hi + 1) / 2;
    long long pm = (long long)mid * N - (long long)mid * (mid - 1) / 2;
    if (pm <= k) lo = mid; else hi = mid - 1;
  }
  const int i = lo;
  const long long p = (long long)i * N - (long long)i * (i - 1) / 2;
  const int j = i + static_cast<int>(k - p);
  double& v = d_corr[(size_t)i * N + j];
  if (isnan(v)) {
    v = two_pass_centered(d_Xm, d_M, T, i, j);
    // Diagnostic counter (evidence only, not part of the numeric output; the
    // count is deterministic -- the set of NaN cells is fixed for fixed input).
    if (d_fb_count != nullptr) atomicAdd(d_fb_count, 1);
  }
}

// Writeback: strict lower triangle mirrored to upper (bitwise equal); diagonal
// derived from the COMPUTED path's value (review F1): the fast/general kernels
// already emit NaN for a column whose centered variance underflows or is zero,
// so the diagonal is 1.0 iff the computed value is finite, else NaN. This
// matches the high-precision reference (var underflow -> NaN) instead of
// blindly forcing 1.0 from count>=2 && min!=max.
__global__ void writeback_kernel(const ColStats* __restrict__ d_stats,
                                 const double* __restrict__ d_corr, int N,
                                 double* __restrict__ d_out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N * N) return;
  int i = idx / N, j = idx % N;
  if (i == j) {
    double v = d_corr[idx];  // computed diagonal (fast/general + fallback)
    d_out[idx] = isfinite(v) ? 1.0 : NAN;  // explicit 1.0 when non-degenerate
  } else if (i > j) {
    d_out[idx] = d_corr[(size_t)j * N + i];  // mirror: bitwise equal
  } else {
    d_out[idx] = d_corr[idx];
  }
}

}  // namespace stock_corr_impl

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

cudaError_t FreeAllBuffers(void* d_Xm, void* d_M, void* d_stats, void* d_s2,
                           void* d_corr, void* d_out, void* d_X, void* d_mask,
                           void* d_fb_count, factor_cuda::MemTracker* tracker) {
  cudaError_t e = cudaSuccess;
  auto keep_first = [&e](cudaError_t r) {
    if (e == cudaSuccess && r != cudaSuccess) e = r;
  };
  keep_first(FreeOrTrack(d_Xm, tracker));
  keep_first(FreeOrTrack(d_M, tracker));
  keep_first(FreeOrTrack(d_stats, tracker));
  keep_first(FreeOrTrack(d_s2, tracker));  // nullptr is a safe no-op
  keep_first(FreeOrTrack(d_corr, tracker));
  keep_first(FreeOrTrack(d_out, tracker));
  keep_first(FreeOrTrack(d_X, tracker));
  keep_first(FreeOrTrack(d_mask, tracker));  // nullptr is a safe no-op
  keep_first(FreeOrTrack(d_fb_count, tracker));  // nullptr is a safe no-op
  return e;
}

}  // namespace

int stock_corr_gpu(const double* h_X, const uint8_t* h_mask, int T, int N,
                   double* h_out, factor_cuda::MemTracker* tracker,
                   StockCorrRunStats* stats) {
  // ---- host preconditions (contract errors) --------------------------------
  if (h_X == nullptr || h_out == nullptr || T < 1 || N < 1) return -1;
  if (static_cast<int64_t>(T) * N > INT32_MAX) return -2;  // checked_mul T*N
  if (static_cast<int64_t>(N) * N > INT32_MAX) return -3;  // output grid cap

  const size_t cols_bytes = static_cast<size_t>(N) * T * sizeof(double);
  const size_t nn = static_cast<size_t>(N) * N;

  cudaError_t cleanup = cudaSuccess;
  bool full_valid = false;  // set in the stats/dispatch block; declared here so
                            // earlier `goto fail` does not bypass initialization

  double* d_Xm = nullptr;
  double* d_M = nullptr;
  ColStats* d_stats = nullptr;
  double* d_s2 = nullptr;
  double* d_corr = nullptr;
  double* d_out = nullptr;
  double* d_X = nullptr;
  uint8_t* d_mask = nullptr;
  int* d_fb_count = nullptr;  // fallback diagnostic counter (stats only)

  cudaError_t err = AllocOrTrack(&d_Xm, cols_bytes, "d_Xm", "alloc", tracker);
  if (err == cudaSuccess) err = AllocOrTrack(&d_M, cols_bytes, "d_M", "alloc", tracker);
  if (err == cudaSuccess) err = AllocOrTrack(&d_stats, static_cast<size_t>(N) * sizeof(ColStats), "d_stats", "alloc", tracker);
  if (err == cudaSuccess) err = AllocOrTrack(&d_s2, static_cast<size_t>(N) * sizeof(double), "d_s2", "alloc", tracker);
  if (err == cudaSuccess) err = AllocOrTrack(&d_corr, nn * sizeof(double), "d_corr", "alloc", tracker);
  if (err == cudaSuccess) err = AllocOrTrack(&d_out, nn * sizeof(double), "d_out", "alloc", tracker);
  if (err == cudaSuccess) err = AllocOrTrack(&d_X, static_cast<size_t>(T) * N * sizeof(double), "d_X", "alloc", tracker);
  if (err == cudaSuccess && h_mask != nullptr) err = AllocOrTrack(&d_mask, static_cast<size_t>(T) * N, "d_mask", "alloc", tracker);
  // Fall-back diagnostic counter is allocated ONLY when stats is requested, so
  // the default (stats == nullptr) allocation set is unchanged (calibration
  // theory formula stays valid).
  if (err == cudaSuccess && stats != nullptr) {
    err = AllocOrTrack(reinterpret_cast<void**>(&d_fb_count), sizeof(int),
                       "d_fb_count", "alloc", tracker);
  }
  if (err != cudaSuccess) goto fail;

  // ---- upload inputs -------------------------------------------------------
  err = cudaMemcpy(d_X, h_X, static_cast<size_t>(T) * N * sizeof(double), cudaMemcpyHostToDevice);
  if (err == cudaSuccess && h_mask != nullptr) {
    err = cudaMemcpy(d_mask, h_mask, static_cast<size_t>(T) * N, cudaMemcpyHostToDevice);
  }
  if (err != cudaSuccess) goto fail;

  // ---- transpose + mask/Xm preprocess -------------------------------------
  {
    const int block = 256;
    const int64_t total = static_cast<int64_t>(T) * N;
    const int grid = static_cast<int>(1 + (total - 1) / block);
    transpose_preprocess<<<grid, block>>>(d_X, d_mask, T, N, d_Xm, d_M);
    err = cudaGetLastError();
    if (err != cudaSuccess) goto fail;
  }
  // d_X no longer needed after transpose; free now (only on success).
  cleanup = FreeOrTrack(d_X, tracker);
  if (err == cudaSuccess && cleanup != cudaSuccess) err = cleanup;
  if (cleanup == cudaSuccess) d_X = nullptr;
  if (err != cudaSuccess) goto fail;

  // ---- column stats ---------------------------------------------------------
  col_stats_kernel<<<N, kBlock>>>(d_Xm, d_M, T, d_stats);
  err = cudaGetLastError();
  if (err != cudaSuccess) goto fail;

  // ---- correlation domain precondition (review F14) -------------------------
  // max|x| <= 1e150 and min nonzero |x| >= 1e-150 over the VALID (participating)
  // cells; violation is a contract precondition error (-4 -> Python ValueError,
  // no correlation output promised). Checked before the GEMM so it fails fast.
  // The same D2H stats also drive the fast/general dispatch.
  {
    std::vector<ColStats> h_stats(static_cast<size_t>(N));
    err = cudaMemcpy(h_stats.data(), d_stats, static_cast<size_t>(N) * sizeof(ColStats),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto fail;
    double max_abs = 0.0, min_abs = kBigPos;
    full_valid = true;
    for (const ColStats& c : h_stats) {
      max_abs = fmax(max_abs, c.max_abs);
      min_abs = fmin(min_abs, c.min_abs);
      if (c.count < static_cast<double>(T)) full_valid = false;  // partial validity
    }
    if (max_abs > kDomainMaxAbs || min_abs < kDomainMinAbs) {
      FreeAllBuffers(d_Xm, d_M, d_stats, d_s2, d_corr, d_out, d_X, d_mask,
                     d_fb_count, tracker);
      return -4;
    }
  }

  // Record the actual dispatch path (evidence; see corpus_parity_v1.py).
  if (stats != nullptr) stats->selected_path = full_valid ? 0 : 1;

  // ---- dispatch: fully-valid (every column count == T) -> fast path ---------
  // The de-mean Gram fast path is algebraically identical to the two-pass
  // reference ONLY when every pair's joint valid set equals its full column
  // (fully valid panel); with partial validity the per-column S2 does not match
  // the pair joint set, so those panels go the general 6-accumulator GEMM.
  // 1D grid of only the upper-triangle tiles: nt*(nt+1)/2 blocks (no idle).
  {
    const int nt = (N + kTile - 1) / kTile;
    const int64_t tiles = static_cast<int64_t>(nt) * (nt + 1) / 2;
    const size_t smem_fast =
        static_cast<size_t>(2 * kTile) * kRowPad * sizeof(double);        // 8448 B
    const size_t smem_gen =
        static_cast<size_t>(2 * kTile + 2 * kBCols) * kRowPad * sizeof(double);  // 16896 B
    if (full_valid) {
      demean_kernel<<<(N + kBlock - 1) / kBlock, kBlock>>>(d_Xm, T, N, d_s2);
      err = cudaGetLastError();
      if (err != cudaSuccess) goto fail;
      fast_gemm_corr_kernel<<<static_cast<unsigned>(tiles), kBlock, smem_fast>>>(
          d_Xm, d_s2, d_stats, T, N, nt, d_corr);
      err = cudaGetLastError();
      if (err != cudaSuccess) goto fail;
    } else {
      gemm_corr_kernel<<<static_cast<unsigned>(tiles), kBlock, smem_gen>>>(
          d_Xm, d_M, T, N, nt, d_corr);
      err = cudaGetLastError();
      if (err != cudaSuccess) goto fail;
      const int64_t ut = (int64_t)N * (N + 1) / 2;  // upper incl diagonal (F1)
      if (d_fb_count != nullptr) {
        err = cudaMemset(d_fb_count, 0, sizeof(int));
        if (err != cudaSuccess) goto fail;
      }
      fallback_kernel<<<static_cast<int>(1 + (ut - 1) / kBlock), kBlock>>>(
          d_Xm, d_M, T, N, d_corr, d_fb_count);
      err = cudaGetLastError();
      if (err != cudaSuccess) goto fail;
      if (stats != nullptr) {
        int fb = 0;
        err = cudaMemcpy(&fb, d_fb_count, sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) goto fail;
        stats->fallback_count = fb;
      }
    }
  }

  // ---- writeback + download --------------------------------------------------
  {
    const int block = 256;
    const int grid = static_cast<int>(1 + (static_cast<int64_t>(nn) - 1) / block);
    writeback_kernel<<<grid, block>>>(d_stats, d_corr, N, d_out);
    err = cudaGetLastError();
    if (err != cudaSuccess) goto fail;
  }
  err = cudaMemcpy(h_out, d_out, nn * sizeof(double), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) goto fail;

  cleanup = FreeAllBuffers(d_Xm, d_M, d_stats, d_s2, d_corr, d_out, d_X, d_mask,
                           d_fb_count, tracker);
  if (err == cudaSuccess && cleanup != cudaSuccess) err = cleanup;
  return static_cast<int>(err);

fail:
  FreeAllBuffers(d_Xm, d_M, d_stats, d_s2, d_corr, d_out, d_X, d_mask,
                 d_fb_count, tracker);
  return static_cast<int>(err);  // failure path keeps the original primary error
}
