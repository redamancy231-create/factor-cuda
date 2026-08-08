// factor-cuda -- factor_corr v0 GPU kernels (shared declarations).
//
// Declarations of the production factor_corr kernels + the shared per-thread
// accumulation / fixed-tree-reduction helpers, so the F/T chunking minimal
// proof 2 (poc/poc3_factor_corr_selfcheck.cu) can launch continuation kernels
// that share the EXACT accumulation operator and tree-reduction order with the
// production kernels. No algorithm is duplicated into the proof harness
// (avoids the F4 self-reference trap where a copy of the algorithm is mistaken
// for the implementation).
//
// Definitions live in src/factor_corr.cu. Chunking changes ONLY how the
// per-thread strided accumulation is sliced across chunks (256-aligned chunk
// boundaries; non-final chunk lengths = multiples of the pinned blockDim 256)
// and defers the fixed shared-memory binary-tree reduce until all chunks have
// run; the accumulation operator and tree order are byte-identical to
// production. See reviews/ft_chunking_design_spec_workflow_2026-08-05.md
// (factor_corr row) and reviews/ft_kahan_residency_decision_2026-08-05.md
// (minimal proof 2 implementation points).
//
// ABI: Partial1/Partial2/PartialK1/PartialK2 mirror corr_math_v1 STRUCT_ABI
// (align 8). static_asserts pin size/offsetof. ASCII-only comments (nvcc/GBK
// pitfall).
//
// Refactor-equivalence note (GPT-5.6-Sol review F-02, 2026-08-05): the claim
// that the __forceinline__ extraction into the shared accumulators / tree
// helpers (accum_p1_cell / accum_p2_cell / tree_reduce_p1_store /
// tree_reduce_p2_store) is bit-semantics-preserving vs the pre-refactor
// anonymous-namespace kernels is backed by THREE independent bodies of
// evidence: (1) an actual re-run of the pre-refactor factor_corr_gpu (git
// 5f9fe1e) vs the refactored one on identical deterministic panels -- 3 shapes
// (normal path / Kahan-trigger + mask / degenerate columns) dump bit-identical
// (F,F) output (scratch/reconstruct_check/, fc /b S1/S2/S3_IDENTICAL); (2) the
// existing factor_corr selfcheck suite ALL PASS after the refactor; (3) frozen
// corpus parity gate_closed=True (factor max|dr|=2.653e-14, unchanged) and the
// 11-case memory calibration (max HWM 2381 MiB, unchanged).
#ifndef FACTOR_CUDA_FACTOR_CORR_IMPL_CUH_
#define FACTOR_CUDA_FACTOR_CORR_IMPL_CUH_

#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

constexpr int kMaxF = 128;          // F > 128 -> pair grid cap (guard -3)
constexpr double kBigPos = 1e300;   // init sentinel for min
constexpr double kBigNeg = -1e300;  // init sentinel for max
constexpr double kTriggerThreshold = 1e8;  // corpus BIAS_THRESHOLD

// Partial structs -- ABI mirrors corr_math_v1 STRUCT_ABI (align 8).
struct alignas(8) Partial1 {
  double count;   // offset 0
  double sum_x;   // 8
  double sum_y;   // 16
  double min_x;   // 24
  double max_x;   // 32
  double min_y;   // 40
  double max_y;   // 48
};
static_assert(sizeof(Partial1) == 56, "Partial1 ABI 56");
static_assert(offsetof(Partial1, count) == 0 && offsetof(Partial1, max_y) == 48,
              "Partial1 field offsets");

struct alignas(8) Partial2 {
  double sxx;  // 0
  double syy;  // 8
  double sxy;  // 16
};
static_assert(sizeof(Partial2) == 24, "Partial2 ABI 24");
static_assert(offsetof(Partial2, sxx) == 0 && offsetof(Partial2, sxy) == 16,
              "Partial2 field offsets");

struct alignas(8) PartialK1 {
  double count;  // 0
  double sum_x;  // 8
  double c_x;    // 16
  double sum_y;  // 24
  double c_y;    // 32
  __device__ __forceinline__ double represented_x() const { return sum_x - c_x; }
  __device__ __forceinline__ double represented_y() const { return sum_y - c_y; }
};
static_assert(sizeof(PartialK1) == 40, "PartialK1 ABI 40");
static_assert(offsetof(PartialK1, count) == 0 && offsetof(PartialK1, c_y) == 32,
              "PartialK1 field offsets");

struct alignas(8) PartialK2 {
  double sxx;   // 0
  double c_xx;  // 8
  double syy;   // 16
  double c_yy;  // 24
  double sxy;   // 32
  double c_xy;  // 40
  __device__ __forceinline__ double represented_xx() const { return sxx - c_xx; }
  __device__ __forceinline__ double represented_yy() const { return syy - c_yy; }
  __device__ __forceinline__ double represented_xy() const { return sxy - c_xy; }
};
static_assert(sizeof(PartialK2) == 48, "PartialK2 ABI 48");
static_assert(offsetof(PartialK2, sxx) == 0 && offsetof(PartialK2, c_xy) == 40,
              "PartialK2 field offsets");

// safe_pearson -- corr_math_v1.py:17. (sxy/sqrt(sxx))/sqrt(syy); NaN if not
// both sxx/syy > 0 and finite (zero-variance / empty / second-order overflow).
__device__ __forceinline__ double safe_pearson(double sxy, double sxx, double syy) {
  if (!(sxx > 0.0 && syy > 0.0) || !isfinite(sxx) || !isfinite(syy)) return NAN;
  return (sxy / std::sqrt(sxx)) / std::sqrt(syy);
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
  // merge right into this: this.add(right.sum); this.add(-right.c) (F5-02).
  __device__ __forceinline__ void merge(const KahanAcc& right) {
    add(right.sum);
    add(-right.c);
  }
};

// ---- shared per-thread accumulation operators --------------------------------
// Both the production reduce kernels and the chunked continuation kernels call
// these, so the accumulation is byte-identical across both paths by
// construction (single source of truth for the stride loop body).
__device__ __forceinline__ void accum_p1_cell(double& cnt, double& sx, double& sy,
                                              double& mnx, double& mxx, double& mny,
                                              double& mxy, double x, double y) {
  cnt += 1.0;
  sx += x;
  sy += y;
  mnx = fmin(mnx, x);
  mxx = fmax(mxx, x);
  mny = fmin(mny, y);
  mxy = fmax(mxy, y);
}

__device__ __forceinline__ void accum_p2_cell(double& sxx, double& syy, double& sxy,
                                              double x, double mx, double y, double my) {
  double dx = x - mx, dy = y - my;
  sxx += dx * dx;
  syy += dy * dy;
  sxy += dx * dy;
}

// ---- fixed shared-memory binary-tree reduce (production order) ---------------
// __syncthreads()-guarded; stores the winner into d_gp1[p] / d_gp2[p]. Used by
// the production reduce kernels and by the continuation finalize-from-pp
// kernels, so the tree reduction order is identical across both paths.
__device__ __forceinline__ void tree_reduce_p1_store(double* scnt, double* ssx,
                                                     double* ssy, double* smnx,
                                                     double* smxx, double* smny,
                                                     double* smxy, int tid, int dim,
                                                     Partial1* d_gp1, int p) {
  __syncthreads();
  for (int s = dim / 2; s > 0; s >>= 1) {
    if (tid < s) {
      scnt[tid] += scnt[tid + s];
      ssx[tid] += ssx[tid + s];
      ssy[tid] += ssy[tid + s];
      smnx[tid] = fmin(smnx[tid], smnx[tid + s]);
      smxx[tid] = fmax(smxx[tid], smxx[tid + s]);
      smny[tid] = fmin(smny[tid], smny[tid + s]);
      smxy[tid] = fmax(smxy[tid], smxy[tid + s]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    Partial1 out;
    out.count = scnt[0];
    out.sum_x = ssx[0];
    out.sum_y = ssy[0];
    out.min_x = smnx[0];
    out.max_x = smxx[0];
    out.min_y = smny[0];
    out.max_y = smxy[0];
    d_gp1[p] = out;
  }
}

__device__ __forceinline__ void tree_reduce_p2_store(double* sh1, double* sh2,
                                                     double* sh3, int tid, int dim,
                                                     Partial2* d_gp2, int p) {
  __syncthreads();
  for (int s = dim / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sh1[tid] += sh1[tid + s];
      sh2[tid] += sh2[tid + s];
      sh3[tid] += sh3[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    Partial2 out;
    out.sxx = sh1[0];
    out.syy = sh2[0];
    out.sxy = sh3[0];
    d_gp2[p] = out;
  }
}

// ---- production kernel declarations (definitions in src/factor_corr.cu) ------
__global__ void make_pairs_kernel(int* __restrict__ d_pairs, int F);

__global__ void transpose_preprocess(const double* __restrict__ d_src,
                                     const uint8_t* __restrict__ d_mask, int R,
                                     int F, double* __restrict__ d_Xt,
                                     uint8_t* __restrict__ d_valid);

__global__ void reduce_p1_kernel(const double* __restrict__ d_Xt,
                                 const uint8_t* __restrict__ d_valid,
                                 const int* __restrict__ d_pairs, int R,
                                 Partial1* __restrict__ d_gp1);

__global__ void finalize_p1_kernel(const Partial1* __restrict__ d_gp1, int P,
                                   double* __restrict__ d_means);

__global__ void reduce_p2_kernel(const double* __restrict__ d_Xt,
                                 const uint8_t* __restrict__ d_valid,
                                 const int* __restrict__ d_pairs,
                                 const double* __restrict__ d_means, int R,
                                 Partial2* __restrict__ d_gp2);

__global__ void finalize_p2_kernel(const Partial1* __restrict__ d_gp1,
                                   const Partial2* __restrict__ d_gp2, int P,
                                   double* __restrict__ d_corr,
                                   uint8_t* __restrict__ d_trigger);

__global__ void kahan_reduce_p1_kernel(const double* __restrict__ d_Xt,
                                       const uint8_t* __restrict__ d_valid,
                                       const int* __restrict__ d_pairs,
                                       const int* __restrict__ d_trig_pairs,
                                       int R, PartialK1* __restrict__ d_gk1);

__global__ void kahan_finalize_p1_kernel(const PartialK1* __restrict__ d_gk1,
                                         int K, double* __restrict__ d_kmeans);

__global__ void kahan_reduce_p2_kernel(const double* __restrict__ d_Xt,
                                       const uint8_t* __restrict__ d_valid,
                                       const int* __restrict__ d_pairs,
                                       const int* __restrict__ d_trig_pairs,
                                       const double* __restrict__ d_kmeans,
                                       int R, PartialK2* __restrict__ d_gk2);

__global__ void kahan_finalize_p2_kernel(const PartialK2* __restrict__ d_gk2,
                                         const int* __restrict__ d_trig_pairs,
                                         int K, double* __restrict__ d_corr);

__global__ void writeback_kernel(const double* __restrict__ d_corr,
                                 const int* __restrict__ d_pairs, int F, int P,
                                 double* __restrict__ d_out);

// ---- continuation kernels (minimal proof 2) ---------------------------------
// Definitions live in poc/poc3_factor_corr_selfcheck.cu. blockDim MUST be the
// pinned 256 (production reduce kernels launch <<<P, 256>>>). Each block
// handles one pair p; lane `tid` accumulates the strided global indices
// r = r0 + tid + k*256 over [r0, r1). The per-thread accumulator lives in
// d_pp[p*256 + tid] and is carried across chunks (first=1 initializes it on
// the first chunk). After all chunks, finalize_pX_from_pp_kernel runs the
// SAME fixed binary tree as production.
__global__ void reduce_p1_cont_kernel(const double* __restrict__ d_Xt,
                                      const uint8_t* __restrict__ d_valid,
                                      const int* __restrict__ d_pairs, int R,
                                      int r0, int r1, int first,
                                      Partial1* __restrict__ d_pp);

__global__ void reduce_p2_cont_kernel(const double* __restrict__ d_Xt,
                                      const uint8_t* __restrict__ d_valid,
                                      const int* __restrict__ d_pairs,
                                      const double* __restrict__ d_means, int R,
                                      int r0, int r1, int first,
                                      Partial2* __restrict__ d_pp);

__global__ void finalize_p1_from_pp_kernel(const Partial1* __restrict__ d_pp,
                                           int P, Partial1* __restrict__ d_gp1);

__global__ void finalize_p2_from_pp_kernel(const Partial2* __restrict__ d_pp,
                                           int P, Partial2* __restrict__ d_gp2);

#endif  // FACTOR_CUDA_FACTOR_CORR_IMPL_CUH_
