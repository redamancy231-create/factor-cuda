// factor-cuda -- rolling_ic v0 GPU kernels (shared declarations).
//
// Declarations of the production rolling_ic kernels so the F/T chunking minimal
// proof (poc/poc3_rolling_ic_selfcheck.cu) can launch the EXACT production
// kernels with chunk-local buffers and re-cut CUB segment offsets -- no kernel
// logic is duplicated into the proof harness (avoids the F4 self-reference trap
// where a copy of the algorithm is mistaken for the implementation).
//
// Definitions live in src/rolling_ic.cu. Chunking changes ONLY launch ranges /
// base pointers / segment offsets; kernel bodies are byte-identical. The
// pearson_kernel blockDim MUST stay pinned at 256 (see CLAUDE.md "blockDim
// 钉死 256" in reviews/ft_chunking_design_spec_workflow_2026-08-05.md) so the
// per-row Pearson tree is a pure function of (row content, N, blockDim) --
// chunked and non-chunked launches then reduce identically.
#ifndef FACTOR_CUDA_ROLLING_IC_IMPL_CUH_
#define FACTOR_CUDA_ROLLING_IC_IMPL_CUH_

#include <cstdint>

// Per-cell valid flag + canonical f64 ordinal keys + sort payload (column
// index) + per-row stats (valid count / factor&return key min&max).
// When called from the chunked path, F/R/fmask/rmask point at a chunk's first
// row and `total` is the chunk's item count; stats/keys/valid are chunk-local
// buffers. `t = i / N` is then the chunk-local row.
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
                                  uint64_t* __restrict__ rkey_max);

// Scatter rank from the CUB-sorted payload (column index) into a double rank
// array. Valid cells occupy the first K positions of each segment, so
// rank == p - t*N + 1; invalid cell -> 0. `total` is the item count of the
// slice (chunk-local in the chunked path).
__global__ void scatter_rank_double_kernel(const uint32_t* __restrict__ sorted_values,
                                           const uint8_t* __restrict__ valid, int N,
                                           int total, double* __restrict__ rank_out);

// Two-pass centered Pearson (float64) over the valid cells of one row.
// Fixed reduction order: per-thread strided loop then binary tree over shared
// memory (deterministic; no atomicAdd). min_valid / constant guards -> NaN.
// grid = number of rows; blockDim MUST be 256 (chunked path keeps it pinned).
__global__ void pearson_kernel(const double* __restrict__ rank_f,
                               const double* __restrict__ rank_r,
                               const uint8_t* __restrict__ valid,
                               const uint32_t* __restrict__ valid_count,
                               const uint64_t* __restrict__ fkey_min,
                               const uint64_t* __restrict__ fkey_max,
                               const uint64_t* __restrict__ rkey_min,
                               const uint64_t* __restrict__ rkey_max, int N,
                               int min_valid, double* __restrict__ ic);

#endif  // FACTOR_CUDA_ROLLING_IC_IMPL_CUH_
