// factor-cuda -- PoC 3 stock_corr v1 selfcheck.
//
// Verifies src/stock_corr.cu (masked-GEMM upper triangle) against an in-process
// CPU reference equivalent to benchmarks/backends.py np_stock_corr /
// tests/fixtures/corr_oracle_v1.py: per output entry (i,j) the Pearson
// correlation over the pair's pooled valid set
// { t : mask(t,i) AND mask(t,j) AND finite in both columns }. |dr|<=1e-12;
// NaN position match; diagonal 1.0/NaN decision; triangle mirror (r[i,j]==r[j,i]
// bitwise).
//
// Covers:
//   - hand-built anchors: correlated columns, anticorrelated, constant column,
//     all-invalid rows, single valid row, scale mismatch, exact-constant.
//   - randomized panels with NaN / +-inf / +-0 / ties / masks / constant cols /
//     all-invalid rows / masked-out cells.
//   - long-T adversarial (T up to 262144: internal-accumulation cancellation +
//     long-T bias, review F8) and tiny-scale values ~1e-150 (review F7).
//   - correlation domain precondition (max|x|<=1e150 / min-nonzero>=1e-150 ->
//     -4, review F14).
//   - error-path smoke (null / dim / grid cap).
//
// The CPU reference uses Kahan for BOTH passes so it stays within ~1e-13 of the
// oracle at any T (a serial second pass would itself exceed 1e-12 beyond
// T ~ 4500).
//
// ASCII-only comments (nvcc/GBK pitfall).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <utility>
#include <vector>
#include <cuda_runtime.h>
#include "stock_corr.cuh"
#include "stock_corr_impl.cuh"

// Shared stock_corr kernel declarations (definitions in src/stock_corr.cu, scope
// namespace stock_corr_impl) so the N-blocking driver below launches the EXACT
// production kernels on tile-local buffers. using-namespace keeps the launches
// terse; no name clashes with the CPU reference helpers in this TU.
using namespace stock_corr_impl;

namespace {

struct Lcg {
  uint64_t s;
  explicit Lcg(uint64_t seed) : s(seed) {}
  uint32_t next() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<uint32_t>(s >> 33);
  }
  double uniform() { return static_cast<double>(next() >> 8) / 16777216.0; }
};

double nan_payload() {
  uint64_t b = 0x7ff8000000000000ull;
  double d;
  std::memcpy(&d, &b, sizeof(double));
  return d;
}

double mk_f64(uint64_t bits) {
  double d;
  std::memcpy(&d, &bits, sizeof(double));
  return d;
}

// CPU reference for a single pair of columns (i,j): oracle semantics (two-pass
// centered Pearson over the common valid subset).
double cpu_pair_corr(const double* X, int T, int N, int i, int j,
                     const uint8_t* mask) {
  std::vector<double> xv, yv;
  for (int t = 0; t < T; ++t) {
    bool okx = std::isfinite(X[static_cast<size_t>(t) * N + i]) &&
               (mask == nullptr || mask[static_cast<size_t>(t) * N + i] != 0u);
    bool oky = std::isfinite(X[static_cast<size_t>(t) * N + j]) &&
               (mask == nullptr || mask[static_cast<size_t>(t) * N + j] != 0u);
    if (okx && oky) {
      xv.push_back(X[static_cast<size_t>(t) * N + i]);
      yv.push_back(X[static_cast<size_t>(t) * N + j]);
    }
  }
  const size_t n = xv.size();
  if (n < 2) return nan_payload();
  // review F3 analog: exact-constant operand on the JOINT valid set -> NaN
  double mnx = xv[0], mxx = xv[0], mny = yv[0], mxy = yv[0];
  // Kahan (CompensatedSum) mean so the reference matches the GPU's
  // order-independent mean on large-bias inputs (review F4).
  struct CK { double sum = 0.0, c = 0.0; void add(double x){ double y=x-c, t=sum+y; c=(t-sum)-y; sum=t; } };
  CK kax, kay;
  for (size_t k = 0; k < n; ++k) {
    kax.add(xv[k]); kay.add(yv[k]);
    mnx = std::min(mnx, xv[k]); mxx = std::max(mxx, xv[k]);
    mny = std::min(mny, yv[k]); mxy = std::max(mxy, yv[k]);
  }
  if (mnx == mxx || mny == mxy) return nan_payload();
  double am = kax.sum / static_cast<double>(n);
  double bm = kay.sum / static_cast<double>(n);
  // Second pass compensated too (long-T tests T up to 262144): a serial sxx at
  // T > ~4500 has the same T*eps error the stock_corr v1 review flagged for the
  // GPU, so an uncompensated reference could not validate the kernel at long T.
  CK ksxx, ksyy, ksxy;
  for (size_t k = 0; k < n; ++k) {
    double dx = xv[k] - am, dy = yv[k] - bm;
    ksxx.add(dx * dx); ksyy.add(dy * dy); ksxy.add(dx * dy);
  }
  double sxx = ksxx.sum - ksxx.c, syy = ksyy.sum - ksyy.c;
  double sxy = ksxy.sum - ksxy.c;
  if (!(sxx > 0.0 && syy > 0.0)) return nan_payload();
  return (sxy / std::sqrt(sxx)) / std::sqrt(syy);
}

// Full (N,N) CPU reference: per-entry pair correlation; diagonal 1.0/NaN by the
// degenerate decision (count>=2 and min!=max on the column's own valid set).
void cpu_stock_corr(const double* X, const uint8_t* mask, int T, int N,
                    double* out) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      if (i == j) {
        // F1 (review): the diagonal must follow the computed (high-precision)
        // self-correlation -- NaN when the column's centered variance
        // underflows to zero (e.g. tiny-adjacent 1e-150 values), 1.0 otherwise.
        // A blind count>=2 && min!=max -> 1.0 would wrongly force finite.
        double rc = cpu_pair_corr(X, T, N, i, i, mask);
        out[static_cast<size_t>(i) * N + j] = std::isnan(rc) ? nan_payload() : 1.0;
      } else {
        out[static_cast<size_t>(i) * N + j] =
            cpu_pair_corr(X, T, N, i, j, mask);
      }
    }
  }
}

int g_fail = 0;

bool corr_match(double gpu, double cpu) {
  if (std::isnan(gpu) && std::isnan(cpu)) return true;
  if (std::isnan(gpu) || std::isnan(cpu)) return false;
  return std::abs(gpu - cpu) <= 1e-12;
}

void check_matrix(const char* name, const double* X, const uint8_t* mask,
                  int T, int N) {
  std::vector<double> gpu(static_cast<size_t>(N) * N);
  std::vector<double> cpu(static_cast<size_t>(N) * N);
  int rc = stock_corr_gpu(X, mask, T, N, gpu.data());
  cpu_stock_corr(X, mask, T, N, cpu.data());
  if (rc != 0) {
    printf("  [FAIL] %s (rc=%d)\n", name, rc);
    ++g_fail;
    return;
  }
  bool ok = true;
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      if (!corr_match(gpu[static_cast<size_t>(i) * N + j],
                      cpu[static_cast<size_t>(i) * N + j]))
        ok = false;
  bool mirrored = true;
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) {
      double a = gpu[static_cast<size_t>(i) * N + j];
      double b = gpu[static_cast<size_t>(j) * N + i];
      if (std::memcmp(&a, &b, sizeof(double)) != 0) mirrored = false;
    }
  printf("  [%s] %s (T=%d N=%d, mirror=%s)\n", ok ? "PASS" : "FAIL", name,
         T, N, mirrored ? "OK" : "BROKEN");
  if (!ok || !mirrored) {
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N; ++j)
        if (!corr_match(gpu[static_cast<size_t>(i) * N + j],
                        cpu[static_cast<size_t>(i) * N + j]))
          printf("      mismatch (%d,%d): gpu=%.12g cpu=%.12g\n", i, j,
                 gpu[static_cast<size_t>(i) * N + j],
                 cpu[static_cast<size_t>(i) * N + j]);
    ++g_fail;
  }
}

// F6 (review 2026-08-05): anchors must lock the EXPECTED semantics with an
// explicit golden assertion, independent of the in-process CPU reference. Runs
// the kernel, cross-checks the full matrix vs the CPU reference, AND asserts
// the off-diagonal (0,1) value (finite golden range or NaN for degenerate).
void check_anchor(const char* name, const double* X, const uint8_t* mask,
                  int T, int N, bool expect_nan, double lo, double hi) {
  std::vector<double> gpu(static_cast<size_t>(N) * N);
  int rc = stock_corr_gpu(X, mask, T, N, gpu.data());
  if (rc != 0) {
    printf("  [FAIL] %s (rc=%d)\n", name, rc);
    ++g_fail;
    return;
  }
  std::vector<double> cpu(static_cast<size_t>(N) * N);
  cpu_stock_corr(X, mask, T, N, cpu.data());
  bool ok = true;
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      if (!corr_match(gpu[static_cast<size_t>(i) * N + j],
                      cpu[static_cast<size_t>(i) * N + j]))
        ok = false;
  const double v = gpu[static_cast<size_t>(1)];  // off-diagonal (0,1)
  bool golden = expect_nan ? std::isnan(v) : (v >= lo && v <= hi);
  printf("  [%s] %s (offdiag[0,1]=%.6g expect %s)\n",
         (ok && golden) ? "PASS" : "FAIL", name, v,
         expect_nan ? "NaN" : "finite golden");
  if (!ok || !golden) ++g_fail;
}

void random_panel(Lcg* rng, std::vector<double>& X, std::vector<uint8_t>& M,
                  int T, int N, bool with_masks, int special) {
  for (int t = 0; t < T; ++t) {
    for (int i = 0; i < N; ++i) {
      int idx = t * N + i;
      uint32_t rr = rng->next() % 100;
      double v;
      if (rr < 60) { v = rng->uniform() * 20.0 - 10.0; }
      else if (rr < 70) { v = 0.0; }
      else if (rr < 78) { v = nan_payload(); }
      else if (rr < 86) { v = (i % 2 == 0) ? mk_f64(0x7ff0000000000000ull) : -mk_f64(0x7ff0000000000000ull); }
      else { v = static_cast<double>(rng->next() % 5); }
      X[static_cast<size_t>(idx)] = v;
      M[static_cast<size_t>(idx)] = (rng->next() % 100) < 80 ? 1 : 0;
    }
  }
  if (with_masks) {
    for (int i = 0; i < N; ++i) M[static_cast<size_t>(i)] = 0;  // row 0 all masked
    if (T >= 2)
      for (int i = 0; i < N; ++i) M[static_cast<size_t>(N + i)] = 1;
  }
  if (special == 1 && N >= 1)  // column 0 constant
    for (int t = 0; t < T; ++t) X[static_cast<size_t>(t) * N] = 5.0;
  if (special == 2 && N >= 2)  // col 1 constant = 0.1 (not exactly representable)
    for (int t = 0; t < T; ++t) X[static_cast<size_t>(t) * N + 1] = 0.1;
  if (special == 3 && T >= 2)  // row 1 all invalid
    for (int i = 0; i < N; ++i) X[static_cast<size_t>(N + i)] = nan_payload();
}

bool vec_bitwise_eq(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    double x = a[i], y = b[i];
    if (std::memcmp(&x, &y, sizeof(double)) != 0) return false;
  }
  return true;
}

// ===========================================================================
// corr width (pair-axis) blocking minimal proof -- stock_corr N-blocking.
// Splits the N axis into blocks of `block_width`; for each lower-triangle tile
// (block_a, block_b) with a>=b, runs the PRODUCTION kernels on tile-local
// buffers holding only the tile's columns (d_F_tile = N_tile*T, the
// N-blocking residency model of memory_budget_v1.py). Every pair is computed
// independently (pair-axis blocking), so the result must be bitwise identical
// to the non-blocked path. No continuation state is carried -- the same trivial
// "pair independent" blocking as factor_corr F-blocking.
//
// KEY DIFFERENCE vs factor_corr F-blocking: the (N,N) output is O(N^2) and the
// two full-size device buffers d_corr/d_out (~N^2*8 each) exceed the budget at
// N=22600 (memory_budget review M4). This driver holds NO full-size device corr
// buffer: each tile's result is copied to host and written into the host output
// matrix, so device residency is bounded by the tile (output is STREAMED
// tile-by-tile). Mirror of the strict half is done on host.
//
// Dispatch is global: a host pass-1 computes every column's valid count and the
// domain bounds (exact integer count / precise min-max on the same mask&finite
// test as the device col_stats) so every tile takes the SAME path as the
// production non-blocked call (fast iff every column count==T) and the -4
// domain check fails fast before any output is written.
//
// Error codes (review SC-ERR-2: this is a PoC driver, not the production
// interface -- negative = contract, -7 collapses the CUDA runtime classes that
// production reports as positive cudaError_t):
//   -1 null / dim<1   -2 T*N>INT32_MAX   -3 N*N>INT32_MAX   -4 domain
//   -6 block_width<1 or >N   -7 any cudaMalloc / memcpy / launch failure
// NOTE (review SC-ERR-5): block_width=1 on large N yields ~N^2/2 tiles, each
// doing a host gather + 2 memcpy + kernels + D2H -- an effectively unbounded
// runtime with no guard. Prefer block_width >= 8; the boundary is safe (no
// crash/UB), just slow.
//
// Returns 0 on success; prints per-tile execution evidence. ASCII-only.
int stock_corr_gpu_nblock(const double* h_X, const uint8_t* h_mask, int T, int N,
                          int block_width, double* h_out) {
  if (h_X == nullptr || h_out == nullptr || T < 1 || N < 1) return -1;
  if (static_cast<int64_t>(T) * N > INT32_MAX) return -2;
  if (static_cast<int64_t>(N) * N > INT32_MAX) return -3;
  if (block_width < 1 || block_width > N) return -6;

  // ---- pass 1 (host): global per-column valid count + domain bounds ---------
  // Exact integer count / precise min-max; identical decision to the device
  // col_stats (col_stats.sum is not load-bearing for dispatch or any kernel
  // gate, so it is not consulted here).
  std::vector<int> col_count(static_cast<size_t>(N), 0);
  double max_abs = 0.0, min_abs = kBigPos;
  for (int t = 0; t < T; ++t) {
    for (int i = 0; i < N; ++i) {
      const size_t e = static_cast<size_t>(t) * N + i;
      const double v = h_X[e];
      const uint8_t m = h_mask != nullptr ? h_mask[e] : 1u;
      if (m == 0u || !std::isfinite(v)) continue;
      ++col_count[i];
      const double ax = std::fabs(v);
      max_abs = std::fmax(max_abs, ax);
      if (ax != 0.0) min_abs = std::fmin(min_abs, ax);
    }
  }
  if (max_abs > kDomainMaxAbs || min_abs < kDomainMinAbs) return -4;
  bool full_valid = true;
  for (int i = 0; i < N; ++i)
    if (col_count[static_cast<size_t>(i)] < T) { full_valid = false; break; }

  // ---- tile buffers (worst tile = off-diagonal: Ba + Bb <= 2*block_width) ----
  const int n_blocks = (N + block_width - 1) / block_width;
  // Worst off-diagonal tile needs Ba+Bb <= 2*block_width columns, and also
  // <= N (the two blocks are disjoint). Clamp (external review MINOR-2): at
  // block_width=N the single tile is N columns wide, so 2N linear + (2N)^2
  // d_corr_tile would be a 4x over-allocation.
  const int max_cols = std::min(N, 2 * block_width);
  const size_t tile_T = static_cast<size_t>(T);
  const size_t buf_x = static_cast<size_t>(max_cols) * tile_T;

  double* d_F_tile = nullptr;
  uint8_t* d_mask_tile = nullptr;
  double* d_Xm_tile = nullptr;
  double* d_M_tile = nullptr;
  ColStats* d_stats_tile = nullptr;
  double* d_s2_tile = nullptr;
  // PoC memory note (external review MINOR-6): the host std::vector
  // allocations below (out, per-tile F_tile/M_tile/tile_corr) may throw
  // std::bad_alloc once the raw device pointers are live; with no RAII a
  // thrown exception leaks the device buffers. Accepted for this proof
  // harness (host OOM is out of scope; the production interface keeps device
  // pointers until explicit FreeAllBuffers).
  double* d_corr_tile = nullptr;
  cudaError_t err = cudaSuccess;
  err = cudaMalloc(&d_F_tile, buf_x * sizeof(double));
  if (err == cudaSuccess) err = cudaMalloc(&d_mask_tile, buf_x);
  if (err == cudaSuccess) err = cudaMalloc(&d_Xm_tile, buf_x * sizeof(double));
  if (err == cudaSuccess) err = cudaMalloc(&d_M_tile, buf_x * sizeof(double));
  if (err == cudaSuccess)
    err = cudaMalloc(&d_stats_tile, static_cast<size_t>(max_cols) * sizeof(ColStats));
  if (err == cudaSuccess)
    err = cudaMalloc(&d_s2_tile, static_cast<size_t>(max_cols) * sizeof(double));
  if (err == cudaSuccess)
    err = cudaMalloc(&d_corr_tile, static_cast<size_t>(max_cols) * max_cols * sizeof(double));
  if (err != cudaSuccess) {
    printf("  nblock cudaMalloc FAIL (err=%d)\n", static_cast<int>(err));
    cudaFree(d_F_tile); cudaFree(d_mask_tile); cudaFree(d_Xm_tile);
    cudaFree(d_M_tile); cudaFree(d_stats_tile); cudaFree(d_s2_tile);
    cudaFree(d_corr_tile);
    return -7;
  }

  std::vector<double> out(static_cast<size_t>(N) * N, std::nan(""));

  for (int a = 0; a < n_blocks && err == cudaSuccess; ++a) {
    for (int b = 0; b <= a && err == cudaSuccess; ++b) {
      const int fa0 = a * block_width, fa1 = std::min((a + 1) * block_width, N);
      const int fb0 = b * block_width, fb1 = std::min((b + 1) * block_width, N);
      const int Ba = fa1 - fa0, Bb = fb1 - fb0;
      const bool diag = (a == b);
      const int tile_cols = diag ? Ba : (Ba + Bb);

      // 1. gather tile columns (host). Diagonal tile = [A]; off-diagonal tile =
      // [B | A] so every needed cross cell (lj in B at local row, Bb+li in A at
      // local col) has row < col and lands in the production kernels'
      // upper-triangle enumeration, AND the S2/safe_pearson argument order is
      // the production (S2[j], S2[i]) -- the bitwise contract depends on it
      // (the sequential division in safe_pearson is not commutative).
      std::vector<double> F_tile(static_cast<size_t>(tile_cols) * tile_T);
      std::vector<uint8_t> M_tile(static_cast<size_t>(tile_cols) * tile_T);
      for (int t = 0; t < T; ++t) {
        for (int c = 0; c < tile_cols; ++c) {
          const int gcol = diag ? (fa0 + c)
                                : (c < Bb ? (fb0 + c) : (fa0 + (c - Bb)));
          const size_t e = static_cast<size_t>(t) * N + gcol;
          F_tile[static_cast<size_t>(t) * tile_cols + c] = h_X[e];
          M_tile[static_cast<size_t>(t) * tile_cols + c] =
              h_mask != nullptr ? h_mask[e] : 1u;
        }
      }
      err = cudaMemcpy(d_F_tile, F_tile.data(),
                       static_cast<size_t>(tile_cols) * tile_T * sizeof(double),
                       cudaMemcpyHostToDevice);
      if (err != cudaSuccess) break;
      err = cudaMemcpy(d_mask_tile, M_tile.data(),
                       static_cast<size_t>(tile_cols) * tile_T,
                       cudaMemcpyHostToDevice);
      if (err != cudaSuccess) break;

      // 2. transpose (production kernel)
      {
        const int block = 256;
        const int64_t total = static_cast<int64_t>(T) * tile_cols;
        const int grid = static_cast<int>(1 + (total - 1) / block);
        transpose_preprocess<<<grid, block>>>(d_F_tile, d_mask_tile, T,
                                              tile_cols, d_Xm_tile, d_M_tile);
        err = cudaGetLastError();
        if (err != cudaSuccess) break;
      }

      // 3. column stats (production kernel; fast path reads min/max gate)
      col_stats_kernel<<<tile_cols, kBlock>>>(d_Xm_tile, d_M_tile, T,
                                              d_stats_tile);
      err = cudaGetLastError();
      if (err != cudaSuccess) break;

      // 4. dispatch by the GLOBAL full_valid decision (same path as the
      // production non-blocked call), on the tile-local mini-matrix (N=tile_cols).
      const int nt = (tile_cols + kTile - 1) / kTile;
      const int64_t tiles = static_cast<int64_t>(nt) * (nt + 1) / 2;
      const size_t smem_fast =
          static_cast<size_t>(2 * kTile) * kRowPad * sizeof(double);
      const size_t smem_gen =
          static_cast<size_t>(2 * kTile + 2 * kBCols) * kRowPad * sizeof(double);
      if (full_valid) {
        demean_kernel<<<(tile_cols + kBlock - 1) / kBlock, kBlock>>>(
            d_Xm_tile, T, tile_cols, d_s2_tile);
        err = cudaGetLastError();
        if (err != cudaSuccess) break;
        fast_gemm_corr_kernel<<<static_cast<unsigned>(tiles), kBlock, smem_fast>>>(
            d_Xm_tile, d_s2_tile, d_stats_tile, T, tile_cols, nt, d_corr_tile);
        err = cudaGetLastError();
        if (err != cudaSuccess) break;
      } else {
        gemm_corr_kernel<<<static_cast<unsigned>(tiles), kBlock, smem_gen>>>(
            d_Xm_tile, d_M_tile, T, tile_cols, nt, d_corr_tile);
        err = cudaGetLastError();
        if (err != cudaSuccess) break;
        const int64_t ut = static_cast<int64_t>(tile_cols) * (tile_cols + 1) / 2;
        // d_fb_count deliberately nullptr: this proof harness exposes no
        // fallback stats (review F1-dfb-count-dead -- the counter was dead
        // plumbing, never D2H'd; production fallback_kernel accepts nullptr).
        fallback_kernel<<<static_cast<int>(1 + (ut - 1) / kBlock), kBlock>>>(
            d_Xm_tile, d_M_tile, T, tile_cols, d_corr_tile, nullptr);
        err = cudaGetLastError();
        if (err != cudaSuccess) break;
      }

      // 5. D2H tile result -> host writeback + mirror
      std::vector<double> tile_corr(static_cast<size_t>(tile_cols) * tile_cols);
      err = cudaMemcpy(tile_corr.data(), d_corr_tile,
                       static_cast<size_t>(tile_cols) * tile_cols * sizeof(double),
                       cudaMemcpyDeviceToHost);
      if (err != cudaSuccess) break;
      if (diag) {
        // block-local UPPER triangle incl diagonal (production computes i<=j;
        // the strict lower half is mirrored below). Diagonal: 1.0 when the
        // computed value is finite, else NaN (same bit pattern as production
        // writeback, reviews F1/F3).
        for (int li = 0; li < Ba; ++li)
          for (int lj = li; lj < Ba; ++lj) {
            double v = tile_corr[static_cast<size_t>(li) * tile_cols + lj];
            if (li == lj)
              // Same NaN source as production writeback_kernel (review F2: a
              // hard-coded canonical payload would silently diverge from the
              // NAN macro on a toolchain whose NAN is non-canonical).
              v = std::isfinite(v) ? 1.0 : NAN;
            // External review INFO-8: a diagonal cell (li==lj) writes the SAME
            // host address twice with the identical value (out[i*N+j] and
            // out[j*N+i] coincide when i==j); every non-diagonal cell is
            // written exactly once. Values coincide, no effect.
            const int i = fa0 + li, j = fa0 + lj;
            out[static_cast<size_t>(i) * N + j] = v;
            out[static_cast<size_t>(j) * N + i] = v;
          }
      } else {
        // cross B x A (global i in block_a > j in block_b), local layout [B | A]
        // puts the needed cell (lj in B at local row, Bb+li in A at local col)
        // at tile_corr[lj * tile_cols + (Bb+li)]; safe_pearson then reads
        // (S2[j], S2[i]) -- the production argument order.
        for (int lj = 0; lj < Bb; ++lj)
          for (int li = 0; li < Ba; ++li) {
            const double v =
                tile_corr[static_cast<size_t>(lj) * tile_cols + (Bb + li)];
            const int i = fa0 + li, j = fb0 + lj;
            out[static_cast<size_t>(i) * N + j] = v;  // lower (i > j)
            out[static_cast<size_t>(j) * N + i] = v;  // upper mirror (j < i)
          }
      }
      printf("  nblock tile (a=%d b=%d) cols=%d path=%s\n",
             a, b, tile_cols, full_valid ? "fast" : "general");
    }
  }

  if (err == cudaSuccess)
    std::memcpy(h_out, out.data(), static_cast<size_t>(N) * N * sizeof(double));
  else
    printf("  nblock FAIL err=%d (%s)\n", static_cast<int>(err),
           cudaGetErrorString(err));

  cudaFree(d_F_tile); cudaFree(d_mask_tile); cudaFree(d_Xm_tile);
  cudaFree(d_M_tile); cudaFree(d_stats_tile); cudaFree(d_s2_tile);
  cudaFree(d_corr_tile);
  return (err == cudaSuccess) ? 0 : -7;
}

// Run one N-blocking case: production (non-blocked) vs nblock output -- (N,N)
// bitwise equal + mirror check. check_cpu anchors against the independent CPU
// reference (1e-12, informational).
void run_stock_nblock_case(const char* name, const std::vector<double>& X,
                           const std::vector<uint8_t>& mask, int T, int N,
                           int block_width, bool check_cpu) {
  std::vector<double> full(static_cast<size_t>(N) * N);
  std::vector<double> blocked(static_cast<size_t>(N) * N);
  int rc1 = stock_corr_gpu(X.data(), mask.empty() ? nullptr : mask.data(),
                           T, N, full.data());
  int rc2 = stock_corr_gpu_nblock(X.data(), mask.empty() ? nullptr : mask.data(),
                                  T, N, block_width, blocked.data());
  if (rc1 != 0 || rc2 != 0) {
    printf("  [FAIL] %s (rc1=%d rc2=%d)\n", name, rc1, rc2);
    ++g_fail;
    return;
  }
  const bool bit_eq = vec_bitwise_eq(full, blocked);
  bool mirrored = true;
  for (int i = 0; i < N && mirrored; ++i)
    for (int j = 0; j < N; ++j) {
      double a = blocked[static_cast<size_t>(i) * N + j];
      double b = blocked[static_cast<size_t>(j) * N + i];
      if (std::memcmp(&a, &b, sizeof(double)) != 0) { mirrored = false; break; }
    }
  printf("  [%s] %s nblock (T=%d N=%d block=%d) bitwise=%s mirror=%s\n",
         (bit_eq && mirrored) ? "PASS" : "FAIL", name, T, N, block_width,
         bit_eq ? "YES" : "no", mirrored ? "OK" : "BROKEN");
  if (!bit_eq) {
    for (size_t i = 0; i < full.size(); ++i) {
      uint64_t bf, bc;
      std::memcpy(&bf, &full[i], 8);
      std::memcpy(&bc, &blocked[i], 8);
      if (bf != bc) {
        printf("      first bitwise mismatch idx=%zu full=%.17g blocked=%.17g\n",
               i, full[i], blocked[i]);
        break;
      }
    }
    ++g_fail;
  }
  if (!mirrored) ++g_fail;
  if (check_cpu && bit_eq) {
    std::vector<double> cpu(static_cast<size_t>(N) * N);
    cpu_stock_corr(X.data(), mask.empty() ? nullptr : mask.data(), T, N,
                   cpu.data());
    bool cpu_ok = true;
    for (size_t i = 0; i < cpu.size(); ++i)
      if (!corr_match(blocked[i], cpu[i])) { cpu_ok = false; break; }
    printf("  [%s] %s nblock vs CPU reference (1e-12, informational)\n",
           cpu_ok ? "PASS" : "FAIL", name);
    if (!cpu_ok) ++g_fail;
  }
}

// ===========================================================================
// --hwm mode: measure the device HWM of stock_corr_gpu_nblock, closing the
// memory_budget_v1 nblock model prediction (N22600 block=256 -> 16.9 MiB) to a
// measured value. Runs ONLY when the exe is invoked as
//   poc3_stock_corr_selfcheck.exe --hwm
// The default selfcheck body is untouched (argc/argv branch returns early).
//
// KEY MEASUREMENT FACT: the nblock device peak is N-INDEPENDENT for N >=
// 2*block_width (max_cols = min(N, 2*block_width) = 2*block_width, and every
// tile buffer is sized by max_cols, not N). So measuring N=5000 anchors the
// N=22600 M4 closure: device residency is tile-local; only the host loop
// length (tile count) and the host N*N output grow with N. The N=22600 host
// output (~4 GiB) and full O(N^2*T) compute are NOT run here (would take hours);
// the device-peak claim is N-independent and closed by the N=5000 measurement,
// the host-side O(N^2) output is disclosed separately.
//
// Measurement is the calibration "driver sample" (third) leg only:
//   driver_peak = free_before - min_free   (background cudaMemGetInfo thread)
// Acceptance adjudicated in benchmarks/stock_nblock_hwm_v1.py:
//   fit       : driver_peak <= available budget (no separate stock-nblock
//               allocation-chain probe exists -- review F3/contract: the
//               factor-fblock scratch/alloc_probe.cu targets the fblock path;
//               the stock delta vs model is reported, not probe-verified)
//   exhausted : margin (fb_case - driver_peak == min_free) ~ 0 with rc==0
//               (WDDM shared-memory fallback; NOT a usable fit)
//   OOM       : rc != 0 (genuine cudaMalloc failure)
// Every case gets its OWN sampler + free_before snapshot so a neighbour's
// allocation peak cannot pollute the min_free attribution.
//
// ASCII-only comments (nvcc/GBK pitfall).
struct NblkHwmCase {
  const char* kind;  // "nblock" or "production"
  int T, N, block;
  int reps;
  size_t fb = 0;            // per-case free_before (before the case's allocs)
  size_t driver_peak = 0;   // fb - min_free during the case
  int samples = 0;          // successful sampler polls (health handshake)
  int rc = 0;
};

void NblkRunSampler(std::atomic<size_t>* min_free, std::atomic<int>* sample_count,
                    std::atomic<bool>* stop, int dev) {
  cudaError_t set_err = cudaSetDevice(dev);
  if (set_err != cudaSuccess) return;  // cannot sample on this device
  while (!stop->load()) {
    size_t f = 0, tot = 0;
    if (cudaMemGetInfo(&f, &tot) == cudaSuccess) {
      sample_count->fetch_add(1, std::memory_order_relaxed);
      size_t cur = min_free->load();
      while (f < cur) {
        if (min_free->compare_exchange_weak(cur, f)) break;
      }
    }
  }
}

void PrintNblkCase(const NblkHwmCase& c) {
  // fb=<per-case free_before> lets the Python report compute the exhausted gate
  // per case (margin = fb_case - driver_peak == min_free) instead of against a
  // single header free_before (baseline drift could mislabel).
  printf("FBLC|kind=%s|T=%d|N=%d|block=%d|reps=%d|fb=%zu|driver_peak=%zu|samples=%d|rc=%d\n",
         c.kind, c.T, c.N, c.block, c.reps, c.fb, c.driver_peak, c.samples, c.rc);
  printf("  [%s] %s T=%d N=%d block=%d: driver_peak %.2f MiB rc=%d\n",
         c.rc == 0 ? "RUN" : "OOM", c.kind, c.T, c.N, c.block,
         c.driver_peak / 1048576.0, c.rc);
}

int run_hwm_stock_nblock(int dev) {
  int guard_fail = 0;
  const int T = 1218;
  const int N = 5000;

  size_t free0 = 0, total = 0;
  cudaMemGetInfo(&free0, &total);
  printf("HWM mode: GPU free_before %.0f MiB / total %.0f MiB\n",
         free0 / 1048576.0, total / 1048576.0);

  // ---- panel: values do not affect HWM; low-bias uniform (fast path). -------
  printf("== hwm: building N=%d panel (%lld doubles, %.0f MiB host) ==\n",
         N, static_cast<int64_t>(T) * N,
         static_cast<double>(T) * N * 8 / 1048576.0);
  std::vector<double> X(static_cast<size_t>(T) * N);
  {
    Lcg rng(0x5EED6C00u);
    for (int64_t i = 0; i < static_cast<int64_t>(T) * N; ++i)
      X[static_cast<size_t>(i)] = rng.uniform() * 4.0 - 2.0;
  }

  // ---- per-case measurement: own sampler + own free_before -------------------
  auto measure = [&](const char* kind, int Nn, int block,
                     const std::vector<double>& Xp, int reps) -> NblkHwmCase {
    NblkHwmCase c;
    c.kind = kind; c.T = T; c.N = Nn; c.block = block; c.reps = reps;
    size_t fb = 0, total2 = 0;
    if (cudaMemGetInfo(&fb, &total2) != cudaSuccess) fb = 0;
    c.fb = fb;
    std::atomic<size_t> min_free{fb};
    std::atomic<int> sample_count{0};
    std::atomic<bool> stop{false};
    std::thread sampler(NblkRunSampler, &min_free, &sample_count, &stop, dev);
    // Handshake: wait for at least one successful sample before running the
    // case, so a dead sampler cannot silently yield driver_peak==0.
    for (int i = 0; i < 2000 && sample_count.load() == 0; ++i)
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    int rc = 0;
    for (int i = 0; i < reps; ++i) {
      std::vector<double> out(static_cast<size_t>(Nn) * Nn);
      if (std::strcmp(kind, "nblock") == 0)
        rc = stock_corr_gpu_nblock(Xp.data(), nullptr, T, Nn, block, out.data());
      else
        rc = stock_corr_gpu(Xp.data(), nullptr, T, Nn, out.data());
      if (rc != 0) break;  // OOM case: stop after first failure
    }
    stop.store(true);
    sampler.join();
    cudaDeviceSynchronize();
    c.rc = rc;
    c.samples = sample_count.load();
    c.driver_peak = (fb == 0) ? 0 : fb - min_free.load();
    return c;
  };

  // ---- GUARD: nblock vs production bitwise (load-bearing, exit 2). ----------
  // Proves the measured implementation did not drift from production. Covers
  // BOTH the fully-valid FAST path and the masked GENERAL path (external review
  // MINOR-10: the previous guard only exercised the fast path). A single GUARD|
  // record (parse_guard requires exactly pass=1, r1=r2=0).
  {
    const int Gn = 300, Gbw = 60;
    bool all_eq = true;
    int gr1 = 0, gr2 = 0;
    // fast path (fully valid, mask=nullptr)
    {
      std::vector<double> GX(static_cast<size_t>(Gn * 1218));
      Lcg rng(0x5EED6D00u);
      for (double& v : GX) v = rng.uniform() * 4.0 - 2.0;
      std::vector<double> full(static_cast<size_t>(Gn) * Gn);
      std::vector<double> blk(static_cast<size_t>(Gn) * Gn);
      int a = stock_corr_gpu(GX.data(), nullptr, 1218, Gn, full.data());
      int b = stock_corr_gpu_nblock(GX.data(), nullptr, 1218, Gn, Gbw, blk.data());
      bool eq = (a == 0 && b == 0) && vec_bitwise_eq(full, blk);
      printf("  [%s] fast N=%d block=%d nblock vs production bitwise (r1=%d r2=%d)\n",
             eq ? "PASS" : "FAIL", Gn, Gbw, a, b);
      if (!eq) { all_eq = false; gr1 = a; gr2 = b; }
    }
    // general path (masked)
    {
      std::vector<double> GX(static_cast<size_t>(Gn * 1218));
      std::vector<uint8_t> GM(static_cast<size_t>(Gn * 1218), 1u);
      Lcg rng(0x5EED6E00u);
      for (int64_t i = 0; i < static_cast<int64_t>(Gn) * 1218; ++i) {
        GX[static_cast<size_t>(i)] = rng.uniform() * 4.0 - 2.0;
        if (rng.next() % 100 < 7) GM[static_cast<size_t>(i)] = 0u;
      }
      std::vector<double> full(static_cast<size_t>(Gn) * Gn);
      std::vector<double> blk(static_cast<size_t>(Gn) * Gn);
      int a = stock_corr_gpu(GX.data(), GM.data(), 1218, Gn, full.data());
      int b = stock_corr_gpu_nblock(GX.data(), GM.data(), 1218, Gn, Gbw, blk.data());
      bool eq = (a == 0 && b == 0) && vec_bitwise_eq(full, blk);
      printf("  [%s] masked N=%d block=%d nblock vs production bitwise (r1=%d r2=%d)\n",
             eq ? "PASS" : "FAIL", Gn, Gbw, a, b);
      if (!eq) { all_eq = false; gr1 = a; gr2 = b; }
    }
    printf("GUARD|pass=%d|r1=%d|r2=%d\n", all_eq ? 1 : 0, gr1, gr2);
    if (!all_eq) guard_fail = 1;
  }

  // ---- nblock block ladder: N-independent peak (N=5000 anchors N=22600). ----
  printf("== hwm: N=%d nblock block ladder ==\n", N);
  for (int block : {256, 128, 64, 32}) {
    NblkHwmCase c = measure("nblock", N, block, X, /*reps=*/1);
    PrintNblkCase(c);
  }

  // ---- production (non-blocked) N=5000: model 526.93 MiB, expected fit. -----
  printf("== hwm: production (non-blocked) N=%d ==\n", N);
  {
    NblkHwmCase c = measure("production", N, /*block=*/0, X, /*reps=*/1);
    PrintNblkCase(c);
  }

  // Context health after the run (driver must not be poisoned).
  cudaError_t err = cudaGetLastError();
  printf("  post-run cudaGetLastError: %s\n", cudaGetErrorString(err));
  printf("== summary: adjudication (delta vs model) in benchmarks/stock_nblock_hwm_v1.py ==\n");
  if (guard_fail) {
    printf("GUARD FAIL: nblock drift detected; run failing\n");
    return 2;
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  int dev = 0;
  cudaError_t err = cudaGetDevice(&dev);
  if (err != cudaSuccess) { printf("cudaGetDevice FAIL\n"); return 1; }
  if (argc > 1 && std::strcmp(argv[1], "--hwm") == 0)
    return run_hwm_stock_nblock(dev);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, dev);
  printf("GPU: %s (cc %d.%d), SM %d\n", prop.name, prop.major, prop.minor,
         prop.multiProcessorCount);
  printf("== stock_corr v2 selfcheck ==\n");

  // ---- error-path smoke ------------------------------------------------------
  {
    double dummy_in[4] = {1.0, 2.0, 3.0, 4.0};
    double dummy_out[4];
    int rc_null = stock_corr_gpu(nullptr, nullptr, 1, 2, dummy_out);
    int rc_t0 = stock_corr_gpu(dummy_in, nullptr, 0, 2, dummy_out);
    int rc_n0 = stock_corr_gpu(dummy_in, nullptr, 1, 0, dummy_out);
    std::vector<double> big_in(1 * 46341, 1.0);
    // F7 (review): N*N > INT32_MAX is checked (-3) BEFORE h_out is touched, so a
    // single-element dummy output suffices -- no 46341^2 (~17 GB) allocation.
    double dummy_out_cap[1];
    int rc_cap = stock_corr_gpu(big_in.data(), nullptr, 1, 46341, dummy_out_cap);
    printf("error-path smoke: null=%d T0=%d N0=%d Ncap=%d (all expect nonzero)\n",
           rc_null, rc_t0, rc_n0, rc_cap);
    if (rc_null == 0 || rc_t0 == 0 || rc_n0 == 0 || rc_cap == 0) {
      printf("  [FAIL] error-path smoke\n");
      ++g_fail;
    } else {
      printf("  [PASS] error-path smoke\n");
    }
  }

  // ---- hand-built anchors -----------------------------------------------------
  // F6 (review): all inputs written in TRUE row-major order (each row is
  // [col0_t, col1_t]) so names/comments match the real columns, and each anchor
  // asserts an explicit golden value (off-diagonal +1/-1/NaN) independent of
  // the in-process CPU reference.
  printf("== anchors ==\n");
  {
    // correlated: x == y exactly -> r = +1
    const double x1[10] = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5};
    check_anchor("correlated", x1, nullptr, 5, 2, false, 1.0 - 1e-12, 1.0);
    // anticorrelated: y = 6 - x -> r = -1
    const double x2[10] = {1, 5, 2, 4, 3, 3, 4, 2, 5, 1};
    check_anchor("anticorrelated", x2, nullptr, 5, 2, false, -1.0, -1.0 + 1e-12);
    // constant column (col1 = 5 always) -> NaN off-diag, NaN diag
    const double x3[10] = {1, 5, 2, 5, 3, 5, 4, 5, 5, 5};
    check_anchor("const_col", x3, nullptr, 5, 2, true, 0, 0);
    // all_invalid col0 (every row's col0 = NaN) -> NaN off-diag and diag;
    // col1 = 1..5 -> diag 1.0
    const double x4[10] = {nan_payload(), 1,
                           nan_payload(), 2,
                           nan_payload(), 3,
                           nan_payload(), 4,
                           nan_payload(), 5};
    check_anchor("all_invalid_col", x4, nullptr, 5, 2, true, 0, 0);
    // exact-constant 0.1 column -> NaN off-diag
    const double x5[10] = {1, 0.1, 2, 0.1, 3, 0.1, 4, 0.1, 5, 0.1};
    check_anchor("const_0.1", x5, nullptr, 5, 2, true, 0, 0);
    // scale mismatch: x~O(1), y~O(1e-9) but perfectly correlated -> r = +1
    const double x6[10] = {1, 1e-9, 2, 2e-9, 3, 3e-9, 4, 4e-9, 5, 5e-9};
    check_anchor("scale_mismatch", x6, nullptr, 5, 2, false, 1.0 - 1e-12, 1.0);
    // F1 (review): tiny-adjacent values -- centered variance underflows to
    // zero -> off-diag AND diagonal must be NaN (not 1.0 from min!=max).
    // Values are in-domain (min-nonzero >= 1e-150), fully valid -> fast path.
    {
      const double xt[4] = {1e-150, 1.0,
                            std::nextafter(1e-150, 1e150), 2.0};
      check_anchor("tiny_adjacent", xt, nullptr, 2, 2, true, 0, 0);
    }
    // large-bias pair (review F4): x=1e12+a, y=1e12*(1+1e-6)+b (corpus bias_1e12
    // structure) as a (T=100,N=2) panel. Two-pass centered must stay within
    // 1e-12 of the CPU reference despite the 1e12 offset.
    {
      const int T = 100, N = 2;
      Lcg rng(0xCAFEu);
      std::vector<double> X(static_cast<size_t>(T) * N);
      for (int t = 0; t < T; ++t) {
        double a = rng.uniform() * 2.0 - 1.0;
        double b = rng.uniform() * 2.0 - 1.0;
        X[static_cast<size_t>(t) * N + 0] = 1e12 + a;
        X[static_cast<size_t>(t) * N + 1] = 1e12 * (1.0 + 1e-6) + b;
      }
      check_matrix("bias_1e12_pair", X.data(), nullptr, T, N);
    }
  }

  // ---- v2 fast-path dispatch --------------------------------------------------
  // Fully-valid panels (all columns count==T) go the de-mean-Gram FAST path;
  // any partial validity goes the GENERAL path. Both must match the CPU
  // reference. A partially-valid panel that were wrongly routed to FAST would
  // treat its masked-out cell (stored as 0.0 after transpose) as data and
  // break S2, so these cases pin the dispatch boundary.
  printf("== v2 fast/general dispatch ==\n");
  {
    const int T = 50, N = 8;
    Lcg rng(0x2E52u);
    std::vector<double> X(static_cast<size_t>(T) * N);
    for (int t = 0; t < T; ++t)
      for (int i = 0; i < N; ++i)
        X[static_cast<size_t>(t) * N + i] = rng.uniform() * 0.1 - 0.05;
    check_matrix("fully_valid_returns", X.data(), nullptr, T, N);      // fast
    std::vector<uint8_t> M(T * N, 1u);
    check_matrix("fully_valid_ones_mask", X.data(), M.data(), T, N);   // fast
    // one masked-out cell -> general (dispatch boundary; would fail if fast)
    std::vector<uint8_t> M1(T * N, 1u);
    M1[0] = 0u;
    check_matrix("single_masked_cell", X.data(), M1.data(), T, N);     // general
    // several masked-out cells -> general
    std::vector<uint8_t> M5(T * N, 1u);
    for (int k = 0; k < 5; ++k) M5[static_cast<size_t>(k) * N + (k % N)] = 0u;
    check_matrix("several_masked_cells", X.data(), M5.data(), T, N);   // general
  }
  {
    // Fully-valid N>tile boundary (fast path tile edge: N not a multiple of 16).
    const int T = 32, N = 257;
    Lcg rng(0x2E57u);
    std::vector<double> X(static_cast<size_t>(T) * N);
    for (int t = 0; t < T; ++t)
      for (int i = 0; i < N; ++i)
        X[static_cast<size_t>(t) * N + i] = rng.uniform() * 2.0 - 1.0;
    check_matrix("fully_valid_N257", X.data(), nullptr, T, N);         // fast
  }
  {
    // Fully-valid LARGE-OFFSET panel: every column shifted by +1e6 (fast path
    // must de-mean and match the reference; the general path would also
    // cancellation-detect and fall back, but the dispatch sends this to fast).
    const int T = 40, N = 4;
    Lcg rng(0x0FF5u);
    std::vector<double> X(static_cast<size_t>(T) * N);
    for (int t = 0; t < T; ++t)
      for (int i = 0; i < N; ++i)
        X[static_cast<size_t>(t) * N + i] = 1e6 + rng.uniform() * 0.1 - 0.05;
    check_matrix("bias_offset_fully_valid", X.data(), nullptr, T, N);  // fast
  }
  {
    // Fully-valid near-constant column (0.1 + 1e-9 noise) -- not exactly
    // constant, so the reference computes a (tiny) correlation and the fast
    // path must match it.
    const int T = 30, N = 3;
    Lcg rng(0x0C0Fu);
    std::vector<double> X(static_cast<size_t>(T) * N);
    for (int t = 0; t < T; ++t) {
      for (int i = 0; i < N; ++i)
        X[static_cast<size_t>(t) * N + i] = rng.uniform() * 2.0 - 1.0;
      X[static_cast<size_t>(t) * N + 2] = 0.1 + 1e-9 * (rng.uniform() - 0.5);
    }
    check_matrix("near_const_col", X.data(), nullptr, T, N);           // fast
  }
  {
    // Exact-constant column at a "dirty" T (review F1): T=3, col1=0.1 constant.
    // The serial-Kahan mean rounds to 0.10000000000000002 != 0.1 -> xd != 0 ->
    // S2 > 0; WITHOUT the fast-path min==max gate this would emit a finite r
    // instead of NaN. The shipped const_0.1 anchor (T=5) rounds cleanly and
    // masks this. Fully valid -> FAST path.
    const double xc[6] = {1, 0.1, 2, 0.1, 3, 0.1};
    check_matrix("fast_const_0.1_T3", xc, nullptr, 3, 2);              // fast
    const double xd[6] = {1, 0.7, 2, 0.7, 3, 0.7};  // 0.7 always dirty at T=3
    check_matrix("fast_const_0.7_T3", xd, nullptr, 3, 2);              // fast
  }
  {
    // Masked bias_1e12 (partial validity) -> general path (6 accumulators +
    // cancellation detection + two-pass fall-back). The fast path must NOT
    // handle this (column-demean vs joint-demean would differ ~4e-8).
    const int T = 100, N = 4;
    Lcg rng(0xB1A5u);
    std::vector<double> X(static_cast<size_t>(T) * N);
    std::vector<uint8_t> M(static_cast<size_t>(T) * N, 1u);
    for (int t = 0; t < T; ++t) {
      for (int i = 0; i < N; ++i) {
        double a = rng.uniform() * 2.0 - 1.0;
        X[static_cast<size_t>(t) * N + i] = 1e12 + a;
        if ((t + i) % 17 == 0) M[static_cast<size_t>(t) * N + i] = 0u;
      }
    }
    check_matrix("masked_bias_1e12", X.data(), M.data(), T, N);        // general
  }

  // ---- N-blocking (pair-axis width blocking) -----------------------------------
  // Every case: production stock_corr_gpu vs stock_corr_gpu_nblock output
  // bitwise-equal (the "pair independent -> bitwise identical" contract) +
  // mirror OK. block widths include non-divisor (ragged last block) and the
  // B=1 / B=N degenerate tiles.
  printf("== stock_corr N-blocking ==\n");
  {
    // fully-valid fast path, several block widths
    for (int bw : {8, 4, 3, 1, 24}) {
      const int T = 40, N = 24;
      Lcg rng(0x0B10u + static_cast<uint32_t>(bw));
      std::vector<double> X(static_cast<size_t>(T) * N);
      for (int t = 0; t < T; ++t)
        for (int i = 0; i < N; ++i)
          X[static_cast<size_t>(t) * N + i] = rng.uniform() * 2.0 - 1.0;
      run_stock_nblock_case("nblock-fullyvalid", X, std::vector<uint8_t>(),
                            T, N, bw, false);
    }
    // explicit all-ones mask (nullptr vs mask both fully valid -> fast)
    {
      const int T = 40, N = 24, bw = 8;
      Lcg rng(0x0B20u);
      std::vector<double> X(static_cast<size_t>(T) * N);
      std::vector<uint8_t> M(static_cast<size_t>(T) * N, 1u);
      for (int t = 0; t < T; ++t)
        for (int i = 0; i < N; ++i)
          X[static_cast<size_t>(t) * N + i] = rng.uniform() * 2.0 - 1.0;
      run_stock_nblock_case("nblock-nullmask", X, M, T, N, bw, false);
    }
    // partial validity -> general path (6 accumulators + fallback)
    {
      const int T = 40, N = 24, bw = 8;
      Lcg rng(0x0B30u);
      std::vector<double> X(static_cast<size_t>(T) * N);
      std::vector<uint8_t> M(static_cast<size_t>(T) * N, 1u);
      for (int t = 0; t < T; ++t)
        for (int i = 0; i < N; ++i)
          X[static_cast<size_t>(t) * N + i] = rng.uniform() * 2.0 - 1.0;
      for (int k = 0; k < 7; ++k) M[static_cast<size_t>(k) * N + (k % N)] = 0u;
      run_stock_nblock_case("nblock-masked", X, M, T, N, bw, false);
    }
    // N>tile boundary + ragged block crossing the 16-cell tile edge
    {
      const int T = 32, N = 257, bw = 16;
      Lcg rng(0x0B40u);
      std::vector<double> X(static_cast<size_t>(T) * N);
      for (int t = 0; t < T; ++t)
        for (int i = 0; i < N; ++i)
          X[static_cast<size_t>(t) * N + i] = rng.uniform() * 2.0 - 1.0;
      run_stock_nblock_case("nblock-N257", X, std::vector<uint8_t>(), T, N, bw,
                            false);
    }
    // masked large-bias -> general path + two-pass fall-back exercised
    {
      const int T = 100, N = 12, bw = 4;
      Lcg rng(0x0B50u);
      std::vector<double> X(static_cast<size_t>(T) * N);
      std::vector<uint8_t> M(static_cast<size_t>(T) * N, 1u);
      for (int t = 0; t < T; ++t)
        for (int i = 0; i < N; ++i) {
          double a = rng.uniform() * 2.0 - 1.0;
          X[static_cast<size_t>(t) * N + i] = 1e12 + a;
          if ((t + i) % 17 == 0) M[static_cast<size_t>(t) * N + i] = 0u;
        }
      run_stock_nblock_case("nblock-bias-masked", X, M, T, N, bw, false);
    }
    // constant columns + diagonal: NaN diagonal handling (reviews F1/F3)
    {
      const int T = 8, N = 10, bw = 3;
      Lcg rng(0x0B60u);
      std::vector<double> X(static_cast<size_t>(T) * N);
      for (int t = 0; t < T; ++t)
        for (int i = 0; i < N; ++i)
          X[static_cast<size_t>(t) * N + i] = rng.uniform() * 2.0 - 1.0;
      for (int t = 0; t < T; ++t) X[static_cast<size_t>(t) * N + 0] = 5.0;
      for (int t = 0; t < T; ++t) X[static_cast<size_t>(t) * N + 1] = 0.1;
      run_stock_nblock_case("nblock-const-diag", X, std::vector<uint8_t>(),
                            T, N, bw, false);
    }
    // short T (every tile < kKtile window; dirty 0.1 constant col)
    {
      const int T = 3, N = 5, bw = 2;
      const double xs[15] = {1, 0.1, 3, 2, 5, 1, 0.1, 4, 3, 6, 1, 0.1, 5, 4, 7};
      std::vector<double> X(xs, xs + 15);
      run_stock_nblock_case("nblock-shortT", X, std::vector<uint8_t>(), T, N, bw,
                            false);
    }
    // independent CPU anchor (bias + mask + check_cpu) -- informational 1e-12
    {
      const int T = 40, N = 30, bw = 6;
      Lcg rng(0x0B70u);
      std::vector<double> X(static_cast<size_t>(T) * N);
      std::vector<uint8_t> M(static_cast<size_t>(T) * N, 1u);
      for (int t = 0; t < T; ++t)
        for (int i = 0; i < N; ++i) {
          double a = rng.uniform() * 2.0 - 1.0;
          X[static_cast<size_t>(t) * N + i] = 1e9 + a;
          if ((t + i) % 19 == 0) M[static_cast<size_t>(t) * N + i] = 0u;
        }
      run_stock_nblock_case("nblock-cpu-anchor", X, M, T, N, bw, true);
    }
    // error-path smoke: block_width contract + null input
    {
      std::vector<double> X(static_cast<size_t>(4 * 3), 1.0);
      std::vector<double> out(static_cast<size_t>(3) * 3);
      int r_bw0 = stock_corr_gpu_nblock(X.data(), nullptr, 4, 3, 0, out.data());
      int r_bwN = stock_corr_gpu_nblock(X.data(), nullptr, 4, 3, 99, out.data());
      int r_null = stock_corr_gpu_nblock(nullptr, nullptr, 4, 3, 1, out.data());
      printf("  [%s] nblock error-path smoke (bw0=%d bw>N=%d null=%d)\n",
             (r_bw0 != 0 && r_bwN != 0 && r_null != 0) ? "PASS" : "FAIL",
             r_bw0, r_bwN, r_null);
      if (r_bw0 == 0 || r_bwN == 0 || r_null == 0) ++g_fail;
    }
    // -4 domain fail-fast (review SC-ERR-1: the nblock -4 host pass-1 branch
    // was dead in the selfcheck). max|x|=2e150 > 1e150 -> -4.
    {
      std::vector<double> X(static_cast<size_t>(2 * 2));
      X[0] = 1e150; X[1] = 2e150; X[2] = 1.0; X[3] = 1.0;
      std::vector<double> out(static_cast<size_t>(2) * 2, 0.0);
      int rc = stock_corr_gpu_nblock(X.data(), nullptr, 2, 2, 1, out.data());
      printf("  [%s] nblock -4 domain rc=%d (expect -4)\n",
             rc == -4 ? "PASS" : "FAIL", rc);
      if (rc != -4) ++g_fail;
    }
    // -3 output grid cap (review SC-ERR-4): T*N fine but N*N > INT32_MAX.
    {
      std::vector<double> big_in(static_cast<size_t>(1) * 46341, 1.0);
      double dummy_out_cap[1];
      int rc = stock_corr_gpu_nblock(big_in.data(), nullptr, 1, 46341, 8,
                                     dummy_out_cap);
      printf("  [%s] nblock -3 N*N cap rc=%d (expect -3)\n",
             rc == -3 ? "PASS" : "FAIL", rc);
      if (rc != -3) ++g_fail;
    }
    // determinism: two nblock runs bitwise identical (review SC-ERR-4).
    {
      const int T = 40, N = 24;
      Lcg rng(0x0B80u);
      std::vector<double> X(static_cast<size_t>(T) * N);
      std::vector<uint8_t> M(static_cast<size_t>(T) * N, 1u);
      for (int t = 0; t < T; ++t)
        for (int i = 0; i < N; ++i) {
          X[static_cast<size_t>(t) * N + i] = rng.uniform() * 2.0 - 1.0;
          if ((t + i) % 13 == 0) M[static_cast<size_t>(t) * N + i] = 0u;
        }
      std::vector<double> o1(static_cast<size_t>(N) * N),
          o2(static_cast<size_t>(N) * N);
      int r1 = stock_corr_gpu_nblock(X.data(), M.data(), T, N, 8, o1.data());
      int r2 = stock_corr_gpu_nblock(X.data(), M.data(), T, N, 8, o2.data());
      bool same = (r1 == 0 && r2 == 0) && vec_bitwise_eq(o1, o2);
      printf("  [%s] nblock determinism (rc1=%d rc2=%d)\n",
             same ? "PASS" : "FAIL", r1, r2);
      if (!same) ++g_fail;
    }
  }

  // ---- randomized panels -------------------------------------------------------
  printf("== randomized checks ==\n");
  {
    const int kSizes[][2] = {{5, 2}, {8, 4}, {16, 8}, {20, 16}, {32, 24},
                             {16, 257}, {8, 300}};  // N>256 boundary (review F1)
    for (int run = 0; run < 3; ++run) {
      for (const auto& sz : kSizes) {
        int T = sz[0], N = sz[1];
        Lcg rng(0x5EEDu + static_cast<uint32_t>(run) * 7919u +
                static_cast<uint32_t>(T) * 131u + static_cast<uint32_t>(N) * 17u);
        std::vector<double> X(static_cast<size_t>(T) * N);
        std::vector<uint8_t> M(static_cast<size_t>(T) * N);
        for (int mask_mode = 0; mask_mode < 2; ++mask_mode) {
          for (int special = 0; special < 4; ++special) {
            random_panel(&rng, X, M, T, N, mask_mode != 0, special);
            char nm[128];
            std::snprintf(nm, sizeof(nm), "rand T=%d N=%d mask=%d sp=%d run=%d",
                          T, N, mask_mode, special, run);
            check_matrix(nm, X.data(), mask_mode ? M.data() : nullptr, T, N);
          }
        }
      }
    }
  }

  // ---- long-T adversarial + tiny-scale (reviews F7/F8) ---------------------------
  // The old single-serial-chain accumulation has a ~T*eps error that exceeds the
  // 1e-12 budget beyond T ~ 4500; the hierarchical reduction in the GEMM kernel
  // must keep |dr| <= 1e-12 for ANY T. Long-T panels here are N=2 (CPU reference
  // stays O(T)) so T can go to 262144.
  printf("== long-T + tiny-scale ==\n");
  {
    // F8 internal-accumulation stress: zero-mean random magnitudes ~1e6 -- heavy
    // sign cancellation inside sumx/sumy/sumxy that the fixed Gram-term ratio
    // cannot see (ratio stays ~1/sqrt(T) << 100 so the fast path runs).
    for (int T : {4096, 16384, 65536, 262144}) {
      const int N = 2;
      Lcg rng(0xF81u + static_cast<uint32_t>(T));
      std::vector<double> X(static_cast<size_t>(T) * N);
      for (int t = 0; t < T; ++t) {
        double u = rng.uniform() * 2.0 - 1.0;
        double v = rng.uniform() * 2.0 - 1.0;
        X[static_cast<size_t>(t) * N + 0] = 1e6 * u;
        X[static_cast<size_t>(t) * N + 1] = 1e6 * v;
      }
      check_matrix("f8_long_cancel", X.data(), nullptr, T, N);
    }
    // F8 long-T bias (corpus bias_1e12 structure at long T): falls back to the
    // compensated two-pass; both passes Kahan on both GPU and CPU reference.
    for (int T : {4096, 16384}) {
      const int N = 2;
      Lcg rng(0xB1A5u + static_cast<uint32_t>(T));
      std::vector<double> X(static_cast<size_t>(T) * N);
      for (int t = 0; t < T; ++t) {
        double u = rng.uniform() * 2.0 - 1.0;
        double v = rng.uniform() * 2.0 - 1.0;
        X[static_cast<size_t>(t) * N + 0] = 1e12 + u;
        X[static_cast<size_t>(t) * N + 1] = 1e12 * (1.0 + 1e-6) + v;
      }
      check_matrix("f8_long_bias", X.data(), nullptr, T, N);
    }
    // F7 tiny-scale boundary: values ~1e-150 so products ~1e-300 (the regime
    // where the old 1e-300 additive suppressed the cancellation fallback).
    // Kept >= 1e-150 so the correlation domain precondition (>= 1e-150) holds.
    for (int T : {64, 4096}) {
      const int N = 2;
      Lcg rng(0xF7A7u + static_cast<uint32_t>(T));
      std::vector<double> X(static_cast<size_t>(T) * N);
      for (int t = 0; t < T; ++t) {
        double u = rng.uniform() * 2.0 - 1.0;
        double v = rng.uniform() * 2.0 - 1.0;
        X[static_cast<size_t>(t) * N + 0] = 1e-150 * (1.5 + 0.5 * u);
        X[static_cast<size_t>(t) * N + 1] = 1e-150 * (1.5 + 0.5 * v);
      }
      check_matrix("f7_tiny_scale", X.data(), nullptr, T, N);
    }
    // F5 (review): the fully-valid long-T panels above all take the FAST path.
    // Add mask-forced GENERAL-path long-T coverage (bias/cancel/tiny) so the
    // 6-accumulator + fallback path is exercised at long T too. Every 17/23/29th
    // row of col0 is masked out -> count<T -> general. These are
    // reduction-order-sensitive inputs (large bias / tiny magnitude), judged
    // against the high-precision Kahan reference (contract revision).
    for (int T : {4096, 16384}) {
      const int N = 2;
      Lcg rng(0xF5A5u + static_cast<uint32_t>(T));
      std::vector<double> X(static_cast<size_t>(T) * N);
      std::vector<uint8_t> M(static_cast<size_t>(T) * N, 1u);
      for (int t = 0; t < T; ++t) {
        double u = rng.uniform() * 2.0 - 1.0;
        double v = rng.uniform() * 2.0 - 1.0;
        X[static_cast<size_t>(t) * N + 0] = 1e12 + u;
        X[static_cast<size_t>(t) * N + 1] = 1e12 * (1.0 + 1e-6) + v;
        if (t % 17 == 0) M[static_cast<size_t>(t) * N + 0] = 0u;
      }
      check_matrix("masked_long_bias", X.data(), M.data(), T, N);
    }
    for (int T : {4096}) {
      const int N = 2;
      Lcg rng(0xF5A7u + static_cast<uint32_t>(T));
      std::vector<double> X(static_cast<size_t>(T) * N);
      std::vector<uint8_t> M(static_cast<size_t>(T) * N, 1u);
      for (int t = 0; t < T; ++t) {
        double u = rng.uniform() * 2.0 - 1.0;
        double v = rng.uniform() * 2.0 - 1.0;
        X[static_cast<size_t>(t) * N + 0] = 1e6 * u;
        X[static_cast<size_t>(t) * N + 1] = 1e6 * v;
        if (t % 23 == 0) M[static_cast<size_t>(t) * N + 0] = 0u;
      }
      check_matrix("masked_long_cancel", X.data(), M.data(), T, N);
    }
    for (int T : {4096}) {
      const int N = 2;
      Lcg rng(0xF5B1u + static_cast<uint32_t>(T));
      std::vector<double> X(static_cast<size_t>(T) * N);
      std::vector<uint8_t> M(static_cast<size_t>(T) * N, 1u);
      for (int t = 0; t < T; ++t) {
        double u = rng.uniform() * 2.0 - 1.0;
        double v = rng.uniform() * 2.0 - 1.0;
        X[static_cast<size_t>(t) * N + 0] = 1e-150 * (1.5 + 0.5 * u);
        X[static_cast<size_t>(t) * N + 1] = 1e-150 * (1.5 + 0.5 * v);
        if (t % 29 == 0) M[static_cast<size_t>(t) * N + 0] = 0u;
      }
      check_matrix("masked_long_tiny", X.data(), M.data(), T, N);
    }
  }

  // ---- correlation domain precondition (review F14) ------------------------------
  printf("== domain -4 ==\n");
  {
    std::vector<double> out(4);
    // max|x| = 2e150 > 1e150 -> -4 (both cells valid and finite).
    const double big[4] = {1e150, 2e150, 1.0, 1.0};
    int rc = stock_corr_gpu(big, nullptr, 2, 2, out.data());
    printf("  [%s] max|x|>1e150 -> rc=%d (expect -4)\n", rc == -4 ? "PASS" : "FAIL", rc);
    if (rc != -4) ++g_fail;
    // min nonzero |x| = 5e-324 (subnormal) < 1e-150 -> -4.
    const double denormal[4] = {5e-324, 1.0, 1.0, 1.0};
    rc = stock_corr_gpu(denormal, nullptr, 2, 2, out.data());
    printf("  [%s] min-nonzero|x|<1e-150 -> rc=%d (expect -4)\n", rc == -4 ? "PASS" : "FAIL", rc);
    if (rc != -4) ++g_fail;
    // Boundary |x| = 1e150 exactly is allowed (max|x| <= 1e150).
    const double edge[4] = {1e150, -1e150, 1.0, 1.0};
    rc = stock_corr_gpu(edge, nullptr, 2, 2, out.data());
    printf("  [%s] boundary |x|=1e150 -> rc=%d (expect 0)\n", rc == 0 ? "PASS" : "FAIL", rc);
    if (rc != 0) ++g_fail;
  }

  // ---- determinism ---------------------------------------------------------------
  printf("== determinism ==\n");
  {
    Lcg rng(0xDEEF0u);
    const int T = 8, N = 6;
    std::vector<double> X(static_cast<size_t>(T) * N);
    for (double& v : X) v = rng.uniform() * 20.0 - 10.0;
    std::vector<double> o1(static_cast<size_t>(N) * N), o2(static_cast<size_t>(N) * N);
    int rc1 = stock_corr_gpu(X.data(), nullptr, T, N, o1.data());
    int rc2 = stock_corr_gpu(X.data(), nullptr, T, N, o2.data());
    bool same = (rc1 == 0 && rc2 == 0) &&
                (std::memcmp(o1.data(), o2.data(), o1.size() * sizeof(double)) == 0);
    printf("  [%s] two runs bitwise identical (rc1=%d rc2=%d)\n",
           same ? "PASS" : "FAIL", rc1, rc2);
    if (!same) ++g_fail;
  }

  printf("== summary ==\n");
  if (g_fail == 0) {
    printf("ALL PASS (stock_corr v2 1e-12 vs CPU reference)\n");
    return 0;
  }
  printf("FAILURES: %d\n", g_fail);
  return 1;
}
