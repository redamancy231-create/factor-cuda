// factor-cuda -- rolling_ic v0 GPU kernel (host interface).
//
// Daily cross-sectional Spearman IC (CLAUDE.md Sec 3 rolling_ic). For each row
// t: valid cells = isfinite(f) & isfinite(r) & factor_mask & fwd_mask. If
// valid count < min_valid -> NaN. If the factor or return values are constant
// on the valid set (ptp == 0) -> NaN. Otherwise stable ordinal ranks 1..m of
// the two sides, and Pearson (float64, two-pass centered) of the rank pair.
//
// v0 uses a unified float64 path (inputs upcast to f64; f32 values survive
// exactly, so ties match the reference np_rolling_ic). ASCII-only comments.
#ifndef FACTOR_CUDA_ROLLING_IC_CUH_
#define FACTOR_CUDA_ROLLING_IC_CUH_

#include <cstdint>

#include "mem_tracker.h"

// Cached device-buffer workspace for rolling_ic_gpu (P2 PoC4 perf, 2026-08-08).
// Mirrors cs_rank_workspace: reusing one workspace across calls with the same
// panel shape (T,N) removes the per-call cudaMalloc/cudaFree overhead (~20
// buffers at corpus scale). The workspace owns its buffers; call
// rolling_ic_workspace_clear to release them (e.g. before a leak check).
//
// Ownership and lifetime (synced to the cs_rank_workspace review F01/F02/F07):
//   - Bound to the CUDA device ordinal + MemTracker active at (re)allocation; a
//     later call with a different device/tracker clears + re-allocates.
//   - rolling_ic_workspace_clear frees via the owner tracker recorded at
//     allocation time (idempotent; nullptr no-ops).
//   - d_fmask / d_rmask are optional growing capacity (lazily allocated on the
//     first masked call, kept after -- mask on/off alternation does not discard
//     the workspace).
//   - Non-copyable / non-movable (owning raw-pointer aggregate).
struct rolling_ic_workspace {
  rolling_ic_workspace() = default;
  rolling_ic_workspace(const rolling_ic_workspace&) = delete;
  rolling_ic_workspace& operator=(const rolling_ic_workspace&) = delete;

  double* d_F = nullptr;
  double* d_R = nullptr;
  uint8_t* d_fmask = nullptr;   // optional lazy capacity
  uint8_t* d_rmask = nullptr;   // optional lazy capacity
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
  void* d_temp = nullptr;       // CUB temp (size queried once at acquire)
  size_t temp_bytes = 0;
  int64_t n_items = -1;         // shape key: T*N
  int T = -1;                   // shape key: T (offsets / per-row stats sizes)
  int N = -1;                   // shape key: N
  int device = -1;              // CUDA device ordinal buffers were allocated on
  factor_cuda::MemTracker* owner_tracker = nullptr;
};

// Release all device buffers cached in the workspace (idempotent; safe on an
// empty / never-used workspace). After this the workspace is reusable.
void rolling_ic_workspace_clear(rolling_ic_workspace* ws);

// Compute rolling Spearman IC of a (T,N) panel.
//   h_F      : (T,N) float64 factor values (inputs already upcast to f64).
//   h_R      : (T,N) float64 forward returns.
//   h_fmask  : (T,N) uint8 factor mask (1=participate) or nullptr (=all).
//   h_rmask  : (T,N) uint8 returns mask or nullptr.
//   T, N     : panel dims. Preconditions: T>=1, N>=1, T*N<=INT32_MAX
//              (implementation dimension cap; offsets/payload are 32-bit).
//   min_valid: minimum valid count per row; below -> NaN. Contract (CLAUDE.md
//              Sec 3) requires min_valid >= 2 -- values < 2 return error -4.
//   h_out    : (T,) float64 IC; NaN rows for invalid/constant/insufficient.
//   tracker  : optional MemTracker (device allocations routed through it).
//   h_rank_f_out / h_rank_r_out: optional (T*N) float64 outputs receiving the
//              per-cell stable-ordinal factor/return ranks (0 for invalid
//              cells), dumped after the scatter stages. Used by the F/T
//              chunking proof to assert the intermediate rank arrays are
//              bitwise identical between the chunked and non-chunked paths
//              (the final IC alone is non-injective over rank pairs).
// Numerics: IC tolerance 1e-12 vs the np_rolling_ic reference holds at
// practical panel sizes (validated to N <= 5000); the two-pass centered
// Pearson uses fixed reduction order (deterministic on one device). The
// exact-1e-12 guarantee at extreme N (approaching INT32_MAX) is not proven.
// Returns 0 on success, otherwise nonzero error code.
//   ws     : optional rolling_ic_workspace for the cached path (P2 PoC4,
//            2026-08-08). Shape/device/tracker key match reuses the buffers;
//            mismatch clears + re-allocates. nullptr = allocate/free every call.
//            NOTE: ws was appended with a default, so this is SOURCE compatible
//            only -- it does NOT preserve the pre-workspace binary ABI (unlike
//            cs_rank_gpu's F09 7-param overload). No external binary consumer
//            exists today; keep as-is for PoC scope.
int rolling_ic_gpu(const double* h_F, const double* h_R,
                   const uint8_t* h_fmask, const uint8_t* h_rmask, int T, int N,
                   int min_valid, double* h_out,
                   factor_cuda::MemTracker* tracker = nullptr,
                   double* h_rank_f_out = nullptr,
                   double* h_rank_r_out = nullptr,
                   rolling_ic_workspace* ws = nullptr);

#endif  // FACTOR_CUDA_ROLLING_IC_CUH_
