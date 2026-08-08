// factor-cuda -- factor_corr v0 GPU kernel (host interface).
//
// Cross-sectional factor correlation matrix (CLAUDE.md Sec 2 factor_corr).
// Input F3 (T,N,F) float64 (host, already upcast from f32 by the adapter) and
// optional mask (T,N) uint8. Output (F,F) float64 Pearson correlation over the
// pooled valid set of each factor pair: valid(t,i) = mask(t,i) AND finite in
// both factor columns. Diagonal is 1.0 iff the computed self-correlation is
// finite (non-degenerate column with positive centered variance), NaN for
// count<2 / zero or underflowed variance / non-finite (review F1, 2026-08-05:
// a tiny-adjacent column whose centered squares underflow -> NaN, not a blind
// 1.0 from min!=max).
//
// Numeric path (single source of truth tests/fixtures/corr_math_v1.py):
//   normal two-pass centered Pearson (fixed reduction order), with a Kahan
//   (CompensatedSum) re-run when bias_metric > 1e8 OR |r| > 1 OR non-finite,
//   matching the corpus trigger state machine. safe_pearson finalize:
//   corr = (sxy/sqrt(sxx))/sqrt(syy).
//
// PoC 3 v0 -- factor_corr is the most complex correlation operator. ASCII-only
// comments (nvcc/GBK pitfall).
#ifndef FACTOR_CUDA_FACTOR_CORR_CUH_
#define FACTOR_CUDA_FACTOR_CORR_CUH_

#include <cstdint>

#include "mem_tracker.h"

// Compute the (F,F) factor correlation matrix.
//   h_F     : (T,N,F) float64, C-contiguous (row-major: index t*N*F + i*F + f).
//   h_mask  : (T,N) uint8 bool (1=participate) or nullptr (=all finite).
//   T, N, F : panel dims. Preconditions: T>=1, N>=1, F>=1,
//             T*N <= INT32_MAX (implementation dimension cap; element offset
//             and pair payload are 32-bit), F <= 128 (pair count grid cap).
//   h_out   : (F,F) float64 correlation matrix; lower triangle computed and
//             mirrored so r[i,j] == r[j,i] bitwise. Diagonal 1.0 or NaN.
//   tracker : optional MemTracker; all device allocations routed through it.
//   h_trigger_out : optional (F(F+1)/2) uint8 output receiving the per-pair
//             Kahan trigger bitset (1 = pair was re-run in compensated
//             arithmetic). nullptr = do not download. Added for the F/T
//             minimal proof 2 so the harness can assert the chunked and
//             non-chunked pipelines select identical trigger sets (analogous
//             to the rolling_ic optional rank outputs).
// Returns 0 on success, otherwise nonzero error code:
//   -1 null pointer or dim < 1     -2 T*N > INT32_MAX
//   -3 F > 128                      -4 F(F+1)/2 grid cap exceeded (unused v0)
int factor_corr_gpu(const double* h_F, const uint8_t* h_mask, int T, int N,
                    int F, double* h_out,
                    factor_cuda::MemTracker* tracker = nullptr,
                    uint8_t* h_trigger_out = nullptr);

#endif  // FACTOR_CUDA_FACTOR_CORR_CUH_
