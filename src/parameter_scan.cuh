// factor-cuda -- parameter_scan v0 GPU kernel (host interface).
//
// Parameter scan over the cross_sectional_rank axes. PoC 1 frozen scan set:
// direction x mask_mode, 4 groups in canonical dict order:
//   g0 (ascending, masked)  g1 (ascending, unmasked)
//   g2 (descending, masked) g3 (descending, unmasked)
// (CLAUDE.md Sec 4 parameter_scan; benchmarks/backends.py np_parameter_scan).
// Each group result is a (T,N) float32 ordinal-rank matrix, bitwise-identical
// to a single cs_rank_gpu call with the matching (descending, use-mask) args.
// X (and mask) are uploaded ONCE and reused across all 4 groups (single H2D);
// each group's rank result is D2H-materialized to its own host buffer
// (contract: parameter_scan groups are always CPU-resident results).
//
// PoC 3 v0. ASCII-only comments (nvcc/GBK pitfall).
#ifndef FACTOR_CUDA_PARAMETER_SCAN_CUH_
#define FACTOR_CUDA_PARAMETER_SCAN_CUH_

#include <cstdint>

#include "mem_tracker.h"
// The parameter_scan device buffer set (d_valid / keys / vals / offsets / d_X /
// d_mask / d_out / d_temp) is identical to cs_rank's, so it REUSES
// cs_rank_workspace for the cached-buffer path (P2 PoC4 perf, 2026-08-08) --
// no separate workspace type is needed.
#include "cross_sectional_rank.cuh"

// Compute the G=4 cross-sectional-rank parameter scan.
//   h_X     : (T,N) float32, C-contiguous (row-major), host.
//   h_mask  : (T,N) uint8 bool (1=participate) or nullptr (=all finite).
//   T, N    : panel dims. Preconditions: T>=1, N>=1, N<=2^24, T*N<=INT32_MAX.
//   h_out   : reference to a length-4 array of (T,N) float32 host buffers, one
//             per group in dict order above. Each pointer must be non-null and
//             have capacity T*N. Binding by reference-to-array forces exactly 4
//             pointers at compile time (a shorter array is a compile error).
//   group_status : reference to a length-4 int array receiving the per-group
//             outcome (only for groups the scan actually reached):
//               0            = success (h_out[g] is valid)
//               positive int = group-level (whitelist) cudaError_t:
//                   cudaErrorInvalidConfiguration / cudaErrorLaunchOutOfResources
//                   caught at the per-group launch checkpoint -> h_out[g] is
//                   unspecified (contract: that group's result is None); the
//                   scan continues with the remaining groups.
//               -100         = group not executed (scan aborted before it)
//   tracker : optional MemTracker; if non-null, all device allocations route
//             through it. Default nullptr = plain cudaMalloc/cudaFree.
//   h_time_ms / h_time_gpu_ms : optional per-group timing outputs ([4] each;
//             nullptr = off). h_time_ms = host wall-clock ms from group launch
//             to D2H + cudaDeviceSynchronize (excludes the shared H2D, matches
//             the contract time_ms); h_time_gpu_ms = cudaEvent device ms from
//             launch to kernel completion (excludes D2H, matches time_gpu_ms).
//             Whitelist-failed and skipped groups store 0.0. Added for the
//             Phase 1 fc.parameter_scan adapter (2026-08-06).
//   h_active : optional [4] int selector (1 = execute group, 0 = skip);
//             nullptr = all 4 executed (previous behavior). Skipped groups are
//             NOT launched/D2H'd/timed and their group_status stays -100
//             (kGroupNotExecuted) with zero timing -- used by the adapter for
//             effective-spec subset scans (Phase 1, 2026-08-06). The fixed
//             BINDING_INDEX order (asc-masked, asc-unmasked, desc-masked,
//             desc-unmasked) is unchanged for active groups.
// Returns:
//   0        scan completed: all ACTIVE groups were attempted. group_status is
//            the authoritative per-group outcome (0 = ok, whitelist error =
//            failed group, -100 = skipped/aborted). Callers MUST consult
//            group_status; a 0 return does not imply every group succeeded.
//   negative contract error (nothing executed; group_status all -100):
//            -1 null pointer or dim < 1    -2 N > 2^24    -3 T*N > INT32_MAX
//   positive scan-level fatal cudaError_t (setup/upload/offsets/temp, a
//            non-whitelist launch or sync error, or D2H failure). Per the
//            parameter_scan contract (CLAUDE.md Sec 4 failure) these raise
//            RuntimeError with no partial results; group_status records which
//            groups completed (0) and which were not reached (-100).
// Two-level failure semantics match the contract's group-level whitelist:
// only a kernel launch check at the per-group sync checkpoint downgrades a
// group (result None, others continue); every other CUDA failure is scan-level.
//   ws     : optional cs_rank_workspace (shared buffer set) for the cached path
//            (P2 PoC4, 2026-08-08). Shape/device/tracker key match reuses the
//            buffers; mismatch clears + re-allocates. nullptr = allocate/free
//            every call. NOTE: ws was appended with a default, so this is
//            SOURCE compatible only -- it does NOT preserve the pre-workspace
//            binary ABI (unlike cs_rank_gpu's F09 7-param overload). No
//            external binary consumer exists today; keep as-is for PoC scope.
int parameter_scan_gpu(const float* h_X, const uint8_t* h_mask, int T, int N,
                       float* (&h_out)[4], int (&group_status)[4],
                       factor_cuda::MemTracker* tracker = nullptr,
                       double* h_time_ms = nullptr,
                       float* h_time_gpu_ms = nullptr,
                       const int* h_active = nullptr,
                       cs_rank_workspace* ws = nullptr);

#endif  // FACTOR_CUDA_PARAMETER_SCAN_CUH_
