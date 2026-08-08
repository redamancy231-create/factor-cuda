// factor-cuda -- stock_corr v2 GPU kernels (shared declarations).
//
// Shared declarations of the production stock_corr kernels + the per-thread
// helpers, so the stock N-blocking minimal proof (poc/poc3_stock_corr_selfcheck.cu)
// can launch the EXACT production kernels on tile-local buffers and show the
// result is bitwise identical to the non-blocked path. No algorithm is
// duplicated into the proof harness (avoids the F4 self-reference trap where a
// copy of the algorithm is mistaken for the implementation).
//
// Definitions live in src/stock_corr.cu inside namespace stock_corr_impl.
// N-blocking changes ONLY which columns each tile's kernel reads (the N-axis is
// split into block_width-wide blocks; each tile holds only the tile's columns)
// and which output cells are consumed -- the per-cell accumulation order is
// byte-identical to production because every pair is computed by the same
// kernel on the same column data (pair-axis blocking, "pair independent").
//
// The stock_corr kernels are kept in a named namespace (NOT the global scope)
// because factor_corr.cu also defines a global `transpose_preprocess` (via
// factor_corr_impl.cuh); the calibration / corpus-parity executables link both
// factor_corr.cu and stock_corr.cu, so global symbols must not collide.
//
// ABI: ColStats mirrors the production struct (align 8, 48 bytes).
// ASCII-only comments (nvcc/GBK pitfall).
#ifndef FACTOR_CUDA_STOCK_CORR_IMPL_CUH_
#define FACTOR_CUDA_STOCK_CORR_IMPL_CUH_

#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

namespace stock_corr_impl {

constexpr double kBigPos = 1e300;   // init sentinel for min
constexpr double kBigNeg = -1e300;  // init sentinel for max
constexpr int kBlock = 256;         // threads per block
constexpr int kTile = 16;           // A/B tile edge (16x16 output cells/block)
constexpr int kBCols = 16;          // B-tile rows = output columns per block
constexpr int kKtile = 32;          // K (T) tile width
constexpr int kRowPad = kKtile + 1; // kKtile data + 1 pad (bank-conflict-free)
constexpr double kCancelRatio = 3.0;   // _gemm_cancel_mask threshold (H-01, v2 100->3:
                                       // with hierarchical reduction the GEMM error is
                                       // ~7.5e-15*ratio, so ratio 100 leaves R in (3,100]
                                       // pairs at up to ~7.5e-13 -- tighten to 3 for a
                                       // 30x margin; corpus panel fallback 0.01%->0.10%)
constexpr double kTiny = 1e-300;       // near-zero-var joint-constant gate only
constexpr double kConstVarRel = 1e-10; // near-zero-var joint-constant gate
constexpr double kDomainMaxAbs = 1e150;   // correlation numeric domain (CLAUDE.md
constexpr double kDomainMinAbs = 1e-150;  // Sec 2: max|x|<=1e150, min-nonzero>=1e-150)
// Below this T a plain serial two-pass second pass is provably < 1e-12 worst
// case (T*eps ~ 4.5e-13 at 2048); at/above it the fall-back uses Kahan (F10).
constexpr int kTwoPassCompensateT = 2048;
// The tile-staging loops and the (tx,ty) = (x&15, x>>4) cell map require exactly
// 256 threads per block (16 rows x 16 cols). Every launch uses kBlock; this
// documents the invariant the kernel's shared-memory layout depends on (F15).
static_assert(kBlock == 256, "gemm_corr_kernel staging/thread map needs 256");

// Column stats (single-pass over T rows) -- diagonal degeneracy decision plus
// the correlation domain precondition (review F14): min/max of |value| over
// valid cells feed the host's max|x|<=1e150 / min-nonzero-|x|>=1e-150 check.
struct alignas(8) ColStats {
  double count;    // valid count
  double sum;      // sum of valid values
  double min;      // min of valid values
  double max;      // max of valid values
  double min_abs;  // min NONZERO |value| over valid cells (kBigPos if none)
  double max_abs;  // max |value| over valid cells (0 if none)
};
static_assert(sizeof(ColStats) == 48, "ColStats 48 bytes");

// safe_pearson -- corr_math_v1.py:17 with complete finite guards (the stock_corr
// variant adds isfinite(sxy) and isfinite(r) vs the factor_corr shared header;
// keep this variant -- the production kernel's bit semantics depend on it).
__device__ __forceinline__ double safe_pearson(double sxy, double sxx, double syy) {
  if (!(sxx > 0.0 && syy > 0.0) || !isfinite(sxx) || !isfinite(syy) ||
      !isfinite(sxy)) return NAN;
  double r = (sxy / std::sqrt(sxx)) / std::sqrt(syy);
  return isfinite(r) ? r : NAN;
}

// Kahan (CompensatedSum) state -- corr_math_v1.py:25. Real value is sum - c.
struct KahanAcc {
  double sum;
  double c;
  __device__ __forceinline__ void add(double x) {
    double y = x - c;
    double t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }
};

// Finalize one output cell from its 6 GEMM accumulators. Returns the Gram-form
// corr (== safe_pearson(cov_u, varx_u, vary_u), ddof cancels) or flags a
// fallback: cancellation (|uncentered term| > kCancelRatio*residual), |r|>1
// sanity, or near-zero-var joint exact-constant gate. The caller writes NaN for
// flagged cells; the separated fallback_kernel recomputes them with
// two_pass_centered. Deterministic (fixed t-order).
struct CellRes {
  double r;
  bool fallback;
};
__device__ __forceinline__ CellRes finalize_cell(double n, double sx, double sy,
                                                 double sxy, double sx2,
                                                 double sy2) {
  CellRes res = {NAN, false};
  if (n < 2.0) return res;
  const double cov_u = sxy - sx * sy / n;
  const double varx_u = sx2 - sx * sx / n;
  const double vary_u = sy2 - sy * sy / n;
  // Cancellation detection (_gemm_cancel_mask semantics; review F7: NO 1e-300
  // additive -- the comparison is already cross-multiplied so no divide-by-zero
  // protection is needed, and kTiny swallows real cancellation near the
  // contract's low-scale bound (term~1e-300, residual~1e-304 -> ratio 1e4 but
  // suppressed). Synced with benchmarks/backends.py _gemm_cancel_mask).
  bool fix = (fabs(sx * sy / n) > kCancelRatio * fabs(cov_u)) ||
             (fabs(sx * sx / n) > kCancelRatio * fabs(varx_u)) ||
             (fabs(sy * sy / n) > kCancelRatio * fabs(vary_u));
  // joint exact-constant gate: near-zero centered variance on the joint set
  // (0.1-type constants leave fp residue ~1e-16 vs true 0; relative threshold)
  const double xmag = fmax(fabs(sx2), fabs(sx * sx / n));
  const double ymag = fmax(fabs(sy2), fabs(sy * sy / n));
  const bool const_x = (fabs(varx_u) <= kConstVarRel * (xmag + kTiny));
  const bool const_y = (fabs(vary_u) <= kConstVarRel * (ymag + kTiny));
  const double r_geom = safe_pearson(cov_u, varx_u, vary_u);
  if (!isfinite(r_geom) || fabs(r_geom) > 1.0) fix = true;  // |r|>1 sanity
  if (fix || const_x || const_y) {
    res.fallback = true;
    return res;
  }
  res.r = r_geom;
  return res;
}

// Per-pair two-pass centered Pearson over the joint valid set, used as the
// fall-back for cancellation / near-zero-var pairs. Reads global Xm/M directly
// (fall-back pairs are rare on canonical data, so no tile staging needed).
// Kahan mean for order-independence on large bias; min/max exact-constant gate
// on the JOINT set (0.1-type constants).
//
// Second pass (review F10): compensated (Kahan) only at long T, where a plain
// serial sxx/syy/sxy worst case ~T*eps exceeds the 1e-12 budget. At
// T <= kTwoPassCompensateT the plain worst case is < 1e-12 and plain keeps the
// common fall-back path fast (the fall-back rate on the corpus panel is
// ~0.1% at kCancelRatio 3, so Kahan-everywhere measurably regresses).
// __forceinline__ (not plain __device__): the header is included by both
// src/stock_corr.cu and the proof harness TU, so a non-inlined __device__
// function would emit a duplicate symbol at link time (same discipline as the
// factor_corr_impl.cuh helpers). The fall-back is a cold path, so the forced
// inline is free.
__device__ __forceinline__ double two_pass_centered(
    const double* __restrict__ d_Xm, const double* __restrict__ d_M, int T,
    int i, int j) {
  const double* xi = d_Xm + (size_t)i * T;
  const double* mi = d_M + (size_t)i * T;
  const double* xj = d_Xm + (size_t)j * T;
  const double* mj = d_M + (size_t)j * T;
  double n = 0.0, mnx = kBigPos, mxx = kBigNeg, mny = kBigPos, mxy = kBigNeg;
  KahanAcc sx, sy;
  sx.sum = 0.0; sx.c = 0.0; sy.sum = 0.0; sy.c = 0.0;
  for (int t = 0; t < T; ++t) {
    if (mi[t] != 0.0 && mj[t] != 0.0) {
      double x = xi[t], y = xj[t];
      n += 1.0;
      sx.add(x);
      sy.add(y);
      mnx = fmin(mnx, x); mxx = fmax(mxx, x);
      mny = fmin(mny, y); mxy = fmax(mxy, y);
    }
  }
  if (n < 2.0) return NAN;
  if (mnx == mxx || mny == mxy) return NAN;  // exact-constant joint operand
  double mx = sx.sum / n;
  double my = sy.sum / n;
  double sxx, syy, sxy;
  if (T > kTwoPassCompensateT) {
    KahanAcc ksxx, ksyy, ksxy;
    ksxx.sum = 0.0; ksxx.c = 0.0;
    ksyy.sum = 0.0; ksyy.c = 0.0;
    ksxy.sum = 0.0; ksxy.c = 0.0;
    for (int t = 0; t < T; ++t) {
      if (mi[t] != 0.0 && mj[t] != 0.0) {
        double dx = xi[t] - mx, dy = xj[t] - my;
        ksxx.add(dx * dx);
        ksyy.add(dy * dy);
        ksxy.add(dx * dy);
      }
    }
    sxx = ksxx.sum - ksxx.c; syy = ksyy.sum - ksyy.c; sxy = ksxy.sum - ksxy.c;
  } else {
    sxx = 0.0; syy = 0.0; sxy = 0.0;
    for (int t = 0; t < T; ++t) {
      if (mi[t] != 0.0 && mj[t] != 0.0) {
        double dx = xi[t] - mx, dy = xj[t] - my;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
      }
    }
  }
  return safe_pearson(sxy, sxx, syy);
}

// ---- production kernel declarations (definitions in src/stock_corr.cu) -------
// Transpose (T,N) row-major -> (N,T) column-major; Xm = valid?value:0,
// M = valid?1:0. valid = mask & finite.
__global__ void transpose_preprocess(const double* __restrict__ d_src,
                                     const uint8_t* __restrict__ d_mask, int T,
                                     int N, double* __restrict__ d_Xm,
                                     double* __restrict__ d_M);

// Per-column stats: count/sum/min/max over valid rows (valid = M==1). One block
// per column; strided + shared binary-tree reduce.
__global__ void col_stats_kernel(const double* __restrict__ d_Xm,
                                 const double* __restrict__ d_M, int T,
                                 ColStats* __restrict__ d_stats);

// Upper-triangle masked-GEMM correlation kernel (GENERAL path), 1 output cell
// per thread. Block (i0,j0) covers rows [i0,i0+16) x cols [j0,j0+16); thread
// (tx,ty) computes (i0+tx, j0+ty). 1D grid enumerates ONLY upper-triangle
// tiles. Only cells with i<=j are written (the strict lower triangle is
// mirrored in writeback). Cancellation / |r|>1 / near-zero-var cells write NaN;
// the separated fallback_kernel recomputes them with two_pass_centered.
__global__ void gemm_corr_kernel(const double* __restrict__ d_Xm,
                                 const double* __restrict__ d_M, int T, int N,
                                 int nt, double* __restrict__ d_corr);

// Per-column serial-Kahan de-mean for the fully-valid fast path (v2). One
// thread per column. Writes xd = x - mean in place and accumulates S2.
__global__ void demean_kernel(double* __restrict__ d_Xm, int T, int N,
                              double* __restrict__ d_s2);

// Fully-valid fast-path correlation kernel (v2): 1 accumulator per cell.
// r = Sxy/sqrt(S2_i)/sqrt(S2_j); exact-constant operand -> NaN via the
// per-column min==max gate from d_stats.
__global__ void fast_gemm_corr_kernel(const double* __restrict__ d_Xd,
                                      const double* __restrict__ d_s2,
                                      const ColStats* __restrict__ d_stats,
                                      int T, int N, int nt,
                                      double* __restrict__ d_corr);

// Separated two-pass fall-back for the general path (v2). Scans the
// upper-triangle cells INCLUDING the diagonal (i<=j) of d_corr and recomputes
// any NaN cell with the two-pass centered Pearson.
__global__ void fallback_kernel(const double* __restrict__ d_Xm,
                                const double* __restrict__ d_M, int T, int N,
                                double* __restrict__ d_corr,
                                int* __restrict__ d_fb_count);

// Writeback: strict lower triangle mirrored to upper (bitwise equal); diagonal
// derived from the COMPUTED path's value (review F1).
__global__ void writeback_kernel(const ColStats* __restrict__ d_stats,
                                 const double* __restrict__ d_corr, int N,
                                 double* __restrict__ d_out);

}  // namespace stock_corr_impl

#endif  // FACTOR_CUDA_STOCK_CORR_IMPL_CUH_
