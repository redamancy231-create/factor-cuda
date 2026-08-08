// factor-cuda -- stock_corr v2 GPU kernel (host interface).
//
// Cross-sectional stock correlation matrix (CLAUDE.md Sec 2 stock_corr).
// Input X (T,N) float64 (host) and optional mask (T,N) uint8. Output (N,N)
// float64 Pearson correlation over each pair's pooled valid set:
// valid_pair(t) = mask(t,i) AND mask(t,j) AND finite(X[t,i]) AND finite(X[t,j]).
// Diagonal is 1.0 iff the computed column self-correlation is finite (i.e. the
// column is non-degenerate and its centered variance does not underflow), else
// NaN -- mirroring the high-precision reference (review F1, var underflow ->
// NaN rather than a blind count>=2 && min!=max -> 1.0).
//
// Numeric paths (single source of truth tests/fixtures/corr_math_v1.py +
// benchmarks/backends.py _masked_gemm_stats):
//  - FAST path (every column count == T, i.e. fully valid panel): de-mean each
//    column by its serial-Kahan mean, precompute S2=sum(xd^2), then per pair a
//    single accumulator Sxy=sum(xd_i*xd_j); corr = Sxy/sqrt(S2_i)/sqrt(S2_j).
//    Algebraically identical to the two-pass reference (same xd), so it stays
//    within 1e-12 of the oracle for ALL fully-valid panels including
//    large-bias (bias_1e12) inputs, with no cancellation detection needed.
//  - GENERAL path (any partial validity): one-pass masked-GEMM accumulating 6
//    products over the pair's joint valid set (n, sumx, sumy, sumxy, sumx2,
//    sumy2), then uncentered-Gram corr. The 6 accumulators use hierarchical
//    reduction (per-kKtile serial partial + Kahan merge), so the accumulation
//    error is ~(klen+2)*eps, independent of T. Cancellation detection
//    (|uncentered term| > 3x residual, |r|>1) and near-zero-var exact-constant
//    joint detection write NaN; a separated fallback_kernel recomputes
//    upper-triangle NaN cells with a per-pair two-pass centered Pearson (Kahan
//    mean + Kahan sxx/syy/sxy + min==max gate) so large-bias masked inputs stay
//    within 1e-12.
// Domain precondition (CLAUDE.md Sec 2 correlation): max|x|<=1e150 and min
// nonzero |x|>=1e-150 over the VALID (participating) cells; violation returns
// -4 (Python maps to ValueError).
//
// PoC 3 v2. ASCII-only comments (nvcc/GBK pitfall).
#ifndef FACTOR_CUDA_STOCK_CORR_CUH_
#define FACTOR_CUDA_STOCK_CORR_CUH_

#include <cstdint>

#include "mem_tracker.h"

// Optional run-time diagnostics for stock_corr_gpu (2026-08-05, corpus-parity
// evidence): which dispatch path actually executed and how many upper-triangle
// cells the fall-back recomputed. Not part of the numeric output; nullptr
// (default) keeps the exact previous behavior and allocation set (the fall-back
// counter device buffer is only allocated when stats is non-null).
struct StockCorrRunStats {
  int selected_path = -1;   // 0 = fast (fully valid panel), 1 = general
  int fallback_count = 0;   // cells recomputed by fallback_kernel (general path)
};

// Compute the (N,N) stock correlation matrix.
//   h_X     : (T,N) float64, C-contiguous (row-major: index t*N + i).
//   h_mask  : (T,N) uint8 bool (1=participate) or nullptr (=all finite).
//   T, N    : panel dims. Preconditions: T>=1, N>=1, T*N<=INT32_MAX,
//             N*N <= INT32_MAX (output grid cap; N <= 46340).
//   h_out   : (N,N) float64 correlation matrix; upper triangle (i<=j) computed,
//             strict lower triangle mirrored from it so r[i,j] == r[j,i]
//             bitwise. Diagonal 1.0 or NaN. (Review F13: doc corrected -- the
//             kernel computes the UPPER triangle, not the strict lower.)
//   tracker : optional MemTracker; all device allocations routed through it.
// Returns 0 on success, otherwise nonzero error code:
//   -1 null pointer or dim < 1     -2 T*N > INT32_MAX     -3 N*N > INT32_MAX
//   -4 correlation domain violation over valid cells (max|x|>1e150 or a nonzero
//      |x|<1e-150); the output matrix is unspecified (Python maps to ValueError).
int stock_corr_gpu(const double* h_X, const uint8_t* h_mask, int T, int N,
                   double* h_out, factor_cuda::MemTracker* tracker = nullptr,
                   StockCorrRunStats* stats = nullptr);

#endif  // FACTOR_CUDA_STOCK_CORR_CUH_
