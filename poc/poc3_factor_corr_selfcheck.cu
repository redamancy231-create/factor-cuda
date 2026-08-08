// factor-cuda -- PoC 3 factor_corr v0 selfcheck.
//
// Verifies src/factor_corr.cu against an in-process CPU reference equivalent to
// benchmarks/backends.py np_factor_corr / tests/fixtures/corr_oracle_v1.py:
// per output entry (i,j) the Pearson correlation over the pooled valid set
// { (t,idx) : mask(t,idx) AND finite in both factor columns }. |dr|<=1e-12;
// NaN position match; diagonal 1.0/NaN decision; triangle mirror (r[i,j]==r[j,i]
// bitwise).
//
// Covers:
//   - 16 corpus anchors (corr_anchors.h, from corr_corpus_v1 manifest) as
//     F=2 panels (col0=a, col1=b) with optional masks.
//   - randomized panels with NaN / +-inf / +-0 / ties / masks / constant
//     columns / all-invalid rows / masked-out rows.
//   - error-path smoke (null / dim / F cap).
//
// Reduction-order note: for the extreme-bias corpus cases (bias_1e15,
// f64_ulp_bias) numpy's own mean is reduction-order sensitive (oracle wrapper
// clause); the GPU Kahan path is more accurate. The selfcheck compares against
// a longdouble-verified two-pass reference; see mismatch tolerance below.
//
// ASCII-only comments (nvcc/GBK pitfall).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <cuda_runtime.h>
#include "factor_corr.cuh"
#include "factor_corr_impl.cuh"
#include "corr_anchors.h"

namespace {

// Deterministic LCG for reproducible panels.
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

// Kahan (CompensatedSum) state -- mirrors corr_math_v1.py / the GPU kernel.
struct CKahan {
  double sum = 0.0;
  double c = 0.0;
  void add(double x) {
    double y = x - c;
    double t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }
  double represented() const { return sum - c; }
};

// CPU reference for a single pair: mirrors the GPU algorithm exactly.
//   normal two-pass centered Pearson (sequential accumulation over the pooled
//   valid subset), then the corpus trigger state machine: if
//   bias_metric > 1e8 OR |r| > 1 OR non-finite -> compensated re-run.
double cpu_pair_corr(const double* x, const double* y, const uint8_t* mask,
                     int R) {
  std::vector<double> xv, yv;
  xv.reserve(static_cast<size_t>(R));
  yv.reserve(static_cast<size_t>(R));
  for (int r = 0; r < R; ++r) {
    bool ok = std::isfinite(x[r]) && std::isfinite(y[r]) &&
              (mask == nullptr || mask[r] != 0u);
    if (ok) { xv.push_back(x[r]); yv.push_back(y[r]); }
  }
  const size_t n = xv.size();
  if (n < 2) return nan_payload();

  double mnx = xv[0], mxx = xv[0], mny = yv[0], mxy = yv[0];
  double am = 0.0, bm = 0.0;
  for (size_t i = 0; i < n; ++i) {
    am += xv[i]; bm += yv[i];
    mnx = std::min(mnx, xv[i]); mxx = std::max(mxx, xv[i]);
    mny = std::min(mny, yv[i]); mxy = std::max(mxy, yv[i]);
  }
  am /= static_cast<double>(n);
  bm /= static_cast<double>(n);
  double sxx = 0.0, syy = 0.0, sxy = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double dx = xv[i] - am, dy = yv[i] - bm;
    sxx += dx * dx; syy += dy * dy; sxy += dx * dy;
  }
  double r = nan_payload();
  // review F3: exact-constant operand on the joint valid set -> NaN, no Kahan
  const bool const_x = (mnx == mxx);
  const bool const_y = (mny == mxy);
  bool trig = false;
  if (!(const_x || const_y)) {
    if (sxx > 0.0 && syy > 0.0)
      r = (sxy / std::sqrt(sxx)) / std::sqrt(syy);

    // trigger: corpus _trigger state machine
    double mx_abs = std::max(std::fabs(mnx), std::fabs(mxx));
    double my_abs = std::max(std::fabs(mny), std::fabs(mxy));
    double bx = mx_abs / std::sqrt(sxx / static_cast<double>(n));
    double by = my_abs / std::sqrt(syy / static_cast<double>(n));
    double bias = std::max(bx, by);
    trig = (bias > 1e8) || std::fabs(r) > 1.0 || !std::isfinite(r);
  }
  if (trig) {
    // compensated re-run: Kahan sums for means, then Kahan centered sums.
    CKahan ksx, ksy;
    for (double v : xv) ksx.add(v);
    for (double v : yv) ksy.add(v);
    double kmx = ksx.represented() / static_cast<double>(n);
    double kmy = ksy.represented() / static_cast<double>(n);
    CKahan a_xx, a_yy, a_xy;
    for (size_t i = 0; i < n; ++i) {
      double dx = xv[i] - kmx, dy = yv[i] - kmy;
      a_xx.add(dx * dx); a_yy.add(dy * dy); a_xy.add(dx * dy);
    }
    double ksxx = a_xx.represented(), ksyy = a_yy.represented(), ksxy = a_xy.represented();
    if (ksxx > 0.0 && ksyy > 0.0)
      r = (ksxy / std::sqrt(ksxx)) / std::sqrt(ksyy);
    else
      r = nan_payload();
  }
  return r;
}

// Full (F,F) CPU reference: per-entry two-pass centered Pearson over the pair's
// own pooled valid subset; diagonal follows the computed self-correlation
// (review F1, 2026-08-05): 1.0 iff finite, else NaN -- mirrors the kernel's
// writeback. F3 is (T*N, F) row-major (as the GPU kernel consumes it).
void cpu_factor_corr(const double* F3, const uint8_t* mask, int T, int N,
                     int F, double* out) {
  const int R = T * N;
  std::vector<double> xi(static_cast<size_t>(R)), xj(static_cast<size_t>(R));
  for (int i = 0; i < F; ++i) {
    for (int j = 0; j < F; ++j) {
      if (i == j) {
        // diagonal: correlation of the column with itself -- NaN when the
        // column's centered variance underflows to zero (e.g. tiny-adjacent
        // 1e-150 values) or the column is degenerate, 1.0 otherwise. A blind
        // count>=2 && min!=max -> 1.0 would wrongly force finite.
        for (int r = 0; r < R; ++r)
          xi[static_cast<size_t>(r)] = F3[r * F + i];
        double rc = cpu_pair_corr(xi.data(), xi.data(), mask, R);
        out[i * F + j] = std::isfinite(rc) ? 1.0 : nan_payload();
      } else {
        for (int r = 0; r < R; ++r) {
          xi[static_cast<size_t>(r)] = F3[r * F + i];
          xj[static_cast<size_t>(r)] = F3[r * F + j];
        }
        out[i * F + j] = cpu_pair_corr(xi.data(), xj.data(), mask, R);
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

void report(const char* name, bool ok, double gpu, double cpu) {
  printf("  [%s] %s (gpu=%.12g cpu=%.12g)\n", ok ? "PASS" : "FAIL", name, gpu, cpu);
  if (!ok) ++g_fail;
}

// Compare a full GPU (F,F) matrix against the CPU reference; mirror check.
void check_matrix(const char* name, const double* F3, const uint8_t* mask,
                  int T, int N, int F) {
  std::vector<double> gpu(static_cast<size_t>(F) * F);
  std::vector<double> cpu(static_cast<size_t>(F) * F);
  int rc = factor_corr_gpu(F3, mask, T, N, F, gpu.data());
  cpu_factor_corr(F3, mask, T, N, F, cpu.data());
  if (rc != 0) {
    printf("  [FAIL] %s (rc=%d)\n", name, rc);
    ++g_fail;
    return;
  }
  bool ok = true;
  for (int i = 0; i < F; ++i) {
    for (int j = 0; j < F; ++j) {
      if (!corr_match(gpu[static_cast<size_t>(i) * F + j],
                      cpu[static_cast<size_t>(i) * F + j])) {
        ok = false;
      }
    }
  }
  // triangle mirror: r[i,j] == r[j,i] bitwise for off-diagonal
  bool mirrored = true;
  for (int i = 0; i < F; ++i) {
    for (int j = 0; j < F; ++j) {
      double a = gpu[static_cast<size_t>(i) * F + j];
      double b = gpu[static_cast<size_t>(j) * F + i];
      if (std::memcmp(&a, &b, sizeof(double)) != 0) { mirrored = false; }
    }
  }
  printf("  [%s] %s (T=%d N=%d F=%d, mirror=%s)\n", ok ? "PASS" : "FAIL",
         name, T, N, F, mirrored ? "OK" : "BROKEN");
  if (!ok || !mirrored) {
    for (int i = 0; i < F; ++i)
      for (int j = 0; j < F; ++j)
        if (!corr_match(gpu[static_cast<size_t>(i) * F + j],
                        cpu[static_cast<size_t>(i) * F + j]))
          printf("      mismatch (%d,%d): gpu=%.12g cpu=%.12g\n", i, j,
                 gpu[static_cast<size_t>(i) * F + j],
                 cpu[static_cast<size_t>(i) * F + j]);
    ++g_fail;
  }
}

// random panel; may force constant columns / all-invalid / masked rows.
void random_panel(Lcg* rng, std::vector<double>& F3, std::vector<uint8_t>& M,
                  int T, int N, int F, bool with_masks, int special) {
  for (int r = 0; r < T * N; ++r) {
    for (int f = 0; f < F; ++f) {
      int idx = r * F + f;
      uint32_t rr = rng->next() % 100;
      double v;
      if (rr < 60) { v = rng->uniform() * 20.0 - 10.0; }
      else if (rr < 70) { v = 0.0; }
      else if (rr < 78) { v = nan_payload(); }
      else if (rr < 86) { v = (f % 2 == 0) ? mk_f64(0x7ff0000000000000ull) : -mk_f64(0x7ff0000000000000ull); }
      else { v = static_cast<double>(rng->next() % 5); }
      F3[static_cast<size_t>(idx)] = v;
      M[static_cast<size_t>(r)] = (rng->next() % 100) < 80 ? 1 : 0;
    }
  }
  if (with_masks) {
    for (int j = 0; j < N; ++j) M[static_cast<size_t>(j)] = 0;  // row 0 all masked
    if (T >= 2) for (int j = 0; j < N; ++j) M[static_cast<size_t>(N + j)] = 1;
  }
  if (special == 1 && F >= 1) {  // column 0 constant (exactly representable)
    for (int r = 0; r < T * N; ++r) F3[static_cast<size_t>(r * F)] = 5.0;
  }
  if (special == 2 && T >= 2) {  // row 1 all invalid (NaN across all F)
    for (int j = 0; j < N; ++j)
      for (int f = 0; f < F; ++f)
        F3[static_cast<size_t>((N + j) * F + f)] = nan_payload();
  }
  if (special == 3 && F >= 2) {  // col 1 constant = 0.1 (NOT exactly representable):
    // a float mean may not reconstruct 0.1 bit-exactly, so a centered pass 2
    // could manufacture a tiny syy>0. Review F3 regression: any pair touching
    // col 1 must still be NaN (exact-constant operand).
    for (int r = 0; r < T * N; ++r) F3[static_cast<size_t>(r * F + 1)] = 0.1;
  }
}

}  // namespace

// ===========================================================================
// F/T chunking minimal proof 2 -- factor_corr continuation kernels.
// Definitions (declarations live in src/factor_corr_impl.cuh). blockDim MUST
// be the pinned 256 (production reduce kernels launch <<<P, 256>>>). Each block
// handles one pair p; lane `tid` accumulates the strided global indices
// r = r0 + tid + k*256 over [r0, r1). The per-thread accumulator lives in
// d_pp[p*256 + tid] and is carried across chunks (first=1 initializes it on the
// first chunk; every other chunk continues from the stored state). Non-final
// chunk lengths must be multiples of 256 so the per-thread index sequence
// equals the non-chunked strided sequence (tid, tid+256, ...). After all
// chunks the finalize_pX_from_pp kernels run the SAME fixed binary tree as
// production (shared tree_reduce_pX_store), so pass 1 / pass 2 results are
// bit-identical to the non-chunked path by construction.
// ===========================================================================

__global__ void reduce_p1_cont_kernel(const double* __restrict__ d_Xt,
                                      const uint8_t* __restrict__ d_valid,
                                      const int* __restrict__ d_pairs, int R,
                                      int r0, int r1, int first,
                                      Partial1* __restrict__ d_pp) {
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
  if (!first) {
    const Partial1 a = d_pp[(size_t)p * 256 + tid];
    cnt = a.count; sx = a.sum_x; sy = a.sum_y;
    mnx = a.min_x; mxx = a.max_x; mny = a.min_y; mxy = a.max_y;
  }
  for (int r = r0 + tid; r < r1; r += blockDim.x) {
    if (vi[r] != 0u && vj[r] != 0u) {
      accum_p1_cell(cnt, sx, sy, mnx, mxx, mny, mxy, xi[r], xj[r]);
    }
  }
  Partial1 out;
  out.count = cnt; out.sum_x = sx; out.sum_y = sy;
  out.min_x = mnx; out.max_x = mxx; out.min_y = mny; out.max_y = mxy;
  d_pp[(size_t)p * 256 + tid] = out;
}

__global__ void reduce_p2_cont_kernel(const double* __restrict__ d_Xt,
                                      const uint8_t* __restrict__ d_valid,
                                      const int* __restrict__ d_pairs,
                                      const double* __restrict__ d_means, int R,
                                      int r0, int r1, int first,
                                      Partial2* __restrict__ d_pp) {
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
  if (!first) {
    const Partial2 a = d_pp[(size_t)p * 256 + tid];
    sxx = a.sxx; syy = a.syy; sxy = a.sxy;
  }
  for (int r = r0 + tid; r < r1; r += blockDim.x) {
    if (vi[r] != 0u && vj[r] != 0u) {
      accum_p2_cell(sxx, syy, sxy, xi[r], mx, xj[r], my);
    }
  }
  Partial2 out;
  out.sxx = sxx; out.syy = syy; out.sxy = sxy;
  d_pp[(size_t)p * 256 + tid] = out;
}

__global__ void finalize_p1_from_pp_kernel(const Partial1* __restrict__ d_pp,
                                           int P, Partial1* __restrict__ d_gp1) {
  const int p = blockIdx.x;
  const int tid = threadIdx.x;
  const Partial1 a = d_pp[(size_t)p * 256 + tid];
  __shared__ double scnt[256], ssx[256], ssy[256], smnx[256], smxx[256], smny[256], smxy[256];
  scnt[tid] = a.count; ssx[tid] = a.sum_x; ssy[tid] = a.sum_y;
  smnx[tid] = a.min_x; smxx[tid] = a.max_x; smny[tid] = a.min_y; smxy[tid] = a.max_y;
  tree_reduce_p1_store(scnt, ssx, ssy, smnx, smxx, smny, smxy, tid, blockDim.x, d_gp1, p);
}

__global__ void finalize_p2_from_pp_kernel(const Partial2* __restrict__ d_pp,
                                           int P, Partial2* __restrict__ d_gp2) {
  const int p = blockIdx.x;
  const int tid = threadIdx.x;
  const Partial2 a = d_pp[(size_t)p * 256 + tid];
  __shared__ double sh1[256], sh2[256], sh3[256];
  sh1[tid] = a.sxx; sh2[tid] = a.syy; sh3[tid] = a.sxy;
  tree_reduce_p2_store(sh1, sh2, sh3, tid, blockDim.x, d_gp2, p);
}

namespace {

// ===========================================================================
// F/T chunking minimal proof 2 -- factor_corr chunked driver + harness.
// ===========================================================================

bool vec_bitwise_eq(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size()) return false;
  return std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

bool vec_bitwise_eq_u8(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
  if (a.size() != b.size()) return false;
  return std::memcmp(a.data(), b.data(), a.size()) == 0;
}

// Chunked factor_corr driver for the minimal proof 2. Runs the production
// pipeline but slices the two normal-pass reductions across T-row chunks that
// carry per-thread continuation state in d_pp (256-pinned blockDim; non-final
// chunk lengths = multiples of 256 so the per-thread strided index sequence
// equals the non-chunked one). Inputs are uploaded and transposed once in full
// (decision v2 path A: keep the full d_Xt + d_valid as the Kahan re-run data
// source). The Kahan re-run keeps a single full-R launch, byte-identical to
// production -- it is NOT part of the continuation proof (decision v2).
//   chunks        : per-chunk T-row counts, must sum to T; every non-final
//                   chunk's c*N must be a multiple of 256.
//   h_trigger_out : optional (P) uint8 output receiving the per-pair Kahan
//                   trigger bitset for cross-checking against production.
//   reset_pp      : negative-control flag -- when 1, EVERY chunk re-initializes
//                   d_pp (fresh-start, NO continuation). With >1 chunk this
//                   MUST produce a bitwise mismatch vs the non-chunked result,
//                   proving continuation is load-bearing (not vacuous).
// Returns 0 on success; per-chunk execution evidence is printed.
int factor_corr_gpu_chunked(const double* h_F, const uint8_t* h_mask, int T, int N,
                            int F, const std::vector<int>& chunks, double* h_out,
                            uint8_t* h_trigger_out = nullptr, int reset_pp = 0) {
  if (h_F == nullptr || h_out == nullptr || T < 1 || N < 1 || F < 1) return -1;
  if (static_cast<int64_t>(T) * N > INT32_MAX) return -2;
  if (F > kMaxF) return -3;
  if (chunks.empty()) return -5;

  // Pre-validate every chunk BEFORE any allocation (int64): each c >= 1, prefix
  // sum never exceeds T, total == T, and every NON-final chunk's c*N is a
  // multiple of 256 (block discipline -- the continuation index sequence equals
  // the non-chunked strided sequence only then).
  int64_t prefix = 0;
  for (size_t ci = 0; ci < chunks.size(); ++ci) {
    const int c = chunks[ci];
    if (c < 1) return -7;
    prefix += c;
    if (prefix > T) return -8;
    if (ci + 1 < chunks.size()) {
      if ((static_cast<int64_t>(c) * N) % 256 != 0) return -9;
    }
  }
  if (prefix != T) return -6;

  const int R = T * N;
  const int P = F * (F + 1) / 2;

  cudaError_t err = cudaSuccess;
  const char* stage = "setup";

  double* d_F = nullptr;
  uint8_t* d_mask = nullptr;
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
  Partial1* d_pp1 = nullptr;
  Partial2* d_pp2 = nullptr;

  stage = "alloc";
  err = cudaMalloc(&d_F, static_cast<size_t>(R) * F * sizeof(double));
  if (err == cudaSuccess && h_mask != nullptr) err = cudaMalloc(&d_mask, static_cast<size_t>(R));
  if (err == cudaSuccess) err = cudaMalloc(&d_Xt, static_cast<size_t>(F) * R * sizeof(double));
  if (err == cudaSuccess) err = cudaMalloc(&d_valid, static_cast<size_t>(F) * R);
  if (err == cudaSuccess) err = cudaMalloc(&d_pairs, static_cast<size_t>(P) * 2 * sizeof(int));
  if (err == cudaSuccess) err = cudaMalloc(&d_gp1, static_cast<size_t>(P) * sizeof(Partial1));
  if (err == cudaSuccess) err = cudaMalloc(&d_means, static_cast<size_t>(P) * 2 * sizeof(double));
  if (err == cudaSuccess) err = cudaMalloc(&d_gp2, static_cast<size_t>(P) * sizeof(Partial2));
  if (err == cudaSuccess) err = cudaMalloc(&d_corr, static_cast<size_t>(P) * sizeof(double));
  if (err == cudaSuccess) err = cudaMalloc(&d_trigger, static_cast<size_t>(P));
  if (err == cudaSuccess) err = cudaMalloc(&d_trig_pairs, static_cast<size_t>(P) * sizeof(int));
  if (err == cudaSuccess) err = cudaMalloc(&d_gk1, static_cast<size_t>(P) * sizeof(PartialK1));
  if (err == cudaSuccess) err = cudaMalloc(&d_kmeans, static_cast<size_t>(P) * 2 * sizeof(double));
  if (err == cudaSuccess) err = cudaMalloc(&d_gk2, static_cast<size_t>(P) * sizeof(PartialK2));
  if (err == cudaSuccess) err = cudaMalloc(&d_out, static_cast<size_t>(F) * F * sizeof(double));
  if (err == cudaSuccess) err = cudaMalloc(&d_pp1, static_cast<size_t>(P) * 256 * sizeof(Partial1));
  if (err == cudaSuccess) err = cudaMalloc(&d_pp2, static_cast<size_t>(P) * 256 * sizeof(Partial2));
  if (err != cudaSuccess) goto fail;

  stage = "pairs";
  make_pairs_kernel<<<F, 1>>>(d_pairs, F);
  err = cudaGetLastError();
  if (err != cudaSuccess) goto fail;

  stage = "upload";
  err = cudaMemcpy(d_F, h_F, static_cast<size_t>(R) * F * sizeof(double), cudaMemcpyHostToDevice);
  if (err == cudaSuccess && h_mask != nullptr) {
    err = cudaMemcpy(d_mask, h_mask, static_cast<size_t>(R), cudaMemcpyHostToDevice);
  }
  if (err != cudaSuccess) goto fail;

  stage = "transpose";
  {
    const int block = 256;
    const int grid = static_cast<int>(1 + (static_cast<int64_t>(R) - 1) / block);
    transpose_preprocess<<<grid, block>>>(d_F, d_mask, R, F, d_Xt, d_valid);
    err = cudaGetLastError();
    if (err != cudaSuccess) goto fail;
  }
  cudaFree(d_F);
  d_F = nullptr;

  // ---- normal pass 1 (chunked continuation) --------------------------------
  stage = "p1-cont";
  {
    int t0 = 0, r0 = 0;
    for (size_t ci = 0; ci < chunks.size(); ++ci) {
      const int c = chunks[ci];
      const int r1 = r0 + c * N;
      reduce_p1_cont_kernel<<<P, 256>>>(d_Xt, d_valid, d_pairs, R, r0, r1,
                                        (reset_pp != 0 || ci == 0) ? 1 : 0, d_pp1);
      err = cudaGetLastError();
      if (err != cudaSuccess) goto fail;
      printf("    chunk[%zu] rows[%d,%d) R[%d,%d) p1-cont P=%d block=256%s\n",
             ci, t0, t0 + c, r0, r1, P, (reset_pp != 0) ? " (reset_pp)" : "");
      t0 += c;
      r0 = r1;
    }
    finalize_p1_from_pp_kernel<<<P, 256>>>(d_pp1, P, d_gp1);
    err = cudaGetLastError();
    if (err != cudaSuccess) goto fail;
    {
      const int grid = (P + 255) / 256;
      finalize_p1_kernel<<<grid, 256>>>(d_gp1, P, d_means);
      err = cudaGetLastError();
      if (err != cudaSuccess) goto fail;
    }
  }

  // ---- normal pass 2 (chunked continuation) --------------------------------
  stage = "p2-cont";
  {
    int t0 = 0, r0 = 0;
    for (size_t ci = 0; ci < chunks.size(); ++ci) {
      const int c = chunks[ci];
      const int r1 = r0 + c * N;
      reduce_p2_cont_kernel<<<P, 256>>>(d_Xt, d_valid, d_pairs, d_means, R, r0, r1,
                                        (reset_pp != 0 || ci == 0) ? 1 : 0, d_pp2);
      err = cudaGetLastError();
      if (err != cudaSuccess) goto fail;
      t0 += c;
      r0 = r1;
    }
    finalize_p2_from_pp_kernel<<<P, 256>>>(d_pp2, P, d_gp2);
    err = cudaGetLastError();
    if (err != cudaSuccess) goto fail;
    {
      const int grid = (P + 255) / 256;
      finalize_p2_kernel<<<grid, 256>>>(d_gp1, d_gp2, P, d_corr, d_trigger);
      err = cudaGetLastError();
      if (err != cudaSuccess) goto fail;
    }
  }

  // ---- Kahan re-run: single full-R launch, identical to production ---------
  stage = "kahan";
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
      const int kb = (R >= 256) ? 256 : 1;
      kahan_reduce_p1_kernel<<<K, kb>>>(d_Xt, d_valid, d_pairs, d_trig_pairs, R, d_gk1);
      err = cudaGetLastError();
      if (err != cudaSuccess) goto fail;
      {
        const int grid = (K + 255) / 256;
        kahan_finalize_p1_kernel<<<grid, 256>>>(d_gk1, K, d_kmeans);
        err = cudaGetLastError();
        if (err != cudaSuccess) goto fail;
      }
      kahan_reduce_p2_kernel<<<K, kb>>>(d_Xt, d_valid, d_pairs, d_trig_pairs, d_kmeans, R, d_gk2);
      err = cudaGetLastError();
      if (err != cudaSuccess) goto fail;
      {
        const int grid = (K + 255) / 256;
        kahan_finalize_p2_kernel<<<grid, 256>>>(d_gk2, d_trig_pairs, K, d_corr);
        err = cudaGetLastError();
        if (err != cudaSuccess) goto fail;
      }
    }
    printf("    Kahan: K=%d triggered pairs, single full-R launch (R=%d kb=%d)\n",
           K, R, (R >= 256) ? 256 : 1);
  }

  // ---- writeback + download ------------------------------------------------
  stage = "writeback";
  {
    const int grid = (P + 255) / 256;
    writeback_kernel<<<grid, 256>>>(d_corr, d_pairs, F, P, d_out);
    err = cudaGetLastError();
    if (err != cudaSuccess) goto fail;
  }
  stage = "d2h";
  err = cudaMemcpy(h_out, d_out, static_cast<size_t>(F) * F * sizeof(double), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) goto fail;
  if (h_trigger_out != nullptr) {
    err = cudaMemcpy(h_trigger_out, d_trigger, static_cast<size_t>(P), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto fail;
  }

  cudaFree(d_mask); cudaFree(d_Xt); cudaFree(d_valid); cudaFree(d_pairs);
  cudaFree(d_gp1); cudaFree(d_means); cudaFree(d_gp2); cudaFree(d_corr);
  cudaFree(d_trigger); cudaFree(d_trig_pairs); cudaFree(d_gk1); cudaFree(d_kmeans);
  cudaFree(d_gk2); cudaFree(d_out); cudaFree(d_pp1); cudaFree(d_pp2);
  return 0;

fail:
  fprintf(stderr, "  chunked driver FAIL at stage=%s err=%d (%s)\n", stage,
          static_cast<int>(err), cudaGetErrorString(err));
  cudaFree(d_mask); cudaFree(d_Xt); cudaFree(d_valid); cudaFree(d_pairs);
  cudaFree(d_gp1); cudaFree(d_means); cudaFree(d_gp2); cudaFree(d_corr);
  cudaFree(d_trigger); cudaFree(d_trig_pairs); cudaFree(d_gk1); cudaFree(d_kmeans);
  cudaFree(d_gk2); cudaFree(d_out); cudaFree(d_pp1); cudaFree(d_pp2);
  cudaFree(d_F);
  return static_cast<int>(err);
}

// Run one chunk-proof case. Production (non-chunked) vs chunked continuation:
// full (F,F) corr bitwise equal + trigger bitset byte equal. check_cpu anchors
// the chunked output against the independent CPU reference (1e-12) to rule out
// "equal but wrong". expect_equal=false is a negative control that MUST FAIL
// (the chunked output must differ from production) -- proves the bitwise
// assertion is load-bearing, not vacuous.
// expected_K (>=0): hard-assert the Kahan branch count derived from the chunked
// trigger bitset, so a future drift that eliminates Kahan branch coverage in
// BOTH paths (while corr/trigger bitwise equality still holds) is caught
// (GPT-5.6-Sol F-01). -1 = do not check (negative controls).
void run_factor_chunk_case(const char* name, const std::vector<double>& F3,
                           const std::vector<uint8_t>& mask, int T, int N, int F,
                           const std::vector<int>& chunks, bool expect_equal,
                           bool check_cpu, int reset_pp, int expected_K) {
  const int P = F * (F + 1) / 2;
  std::vector<double> full(static_cast<size_t>(F) * F), chunked(static_cast<size_t>(F) * F);
  std::vector<uint8_t> ft(static_cast<size_t>(P)), ct(static_cast<size_t>(P));
  int rc1 = factor_corr_gpu(F3.data(), mask.empty() ? nullptr : mask.data(), T, N, F,
                            full.data(), nullptr, ft.data());
  int rc2 = factor_corr_gpu_chunked(F3.data(), mask.empty() ? nullptr : mask.data(),
                                    T, N, F, chunks, chunked.data(), ct.data(), reset_pp);
  if (rc1 != 0 || rc2 != 0) {
    printf("  [FAIL] %s (rc1=%d rc2=%d)\n", name, rc1, rc2);
    ++g_fail;
    return;
  }
  const bool corr_eq = vec_bitwise_eq(full, chunked);
  const bool trig_eq = vec_bitwise_eq_u8(ft, ct);
  const bool ok = (corr_eq && trig_eq) == expect_equal;
  printf("  [%s] %s (T=%d N=%d F=%d chunks=%zu%s) corr_eq=%s trigger_eq=%s\n",
         ok ? "PASS" : "FAIL", name, T, N, F, chunks.size(),
         reset_pp != 0 ? " reset_pp" : "", corr_eq ? "YES" : "no",
         trig_eq ? "YES" : "no");
  // F-01 (GPT-5.6-Sol): hard-assert the Kahan branch count from the chunked
  // trigger bitset. corr/trigger bitwise equality alone could stay ALL-PASS if
  // both paths drifted in lockstep and the Kahan branch silently disappeared;
  // pinning expected_K keeps the branch coverage honest.
  if (expected_K >= 0) {
    int k_chunk = 0;
    for (uint8_t v : ct) if (v != 0u) ++k_chunk;
    if (k_chunk != expected_K) {
      printf("  [FAIL] %s Kahan branch coverage drift: expected_K=%d got=%d\n",
             name, expected_K, k_chunk);
      ++g_fail;
    }
  }
  if (!ok) {
    if (expect_equal) {
      for (size_t i = 0; i < full.size(); ++i) {
        uint64_t bf, bc;
        std::memcpy(&bf, &full[i], 8);
        std::memcpy(&bc, &chunked[i], 8);
        if (bf != bc) {
          printf("      first corr bitwise mismatch idx=%zu full=%.17g chunked=%.17g\n",
                 i, full[i], chunked[i]);
          break;
        }
      }
    }
    ++g_fail;
  }
  if (check_cpu && expect_equal && ok) {
    std::vector<double> cpu(static_cast<size_t>(F) * F);
    cpu_factor_corr(F3.data(), mask.empty() ? nullptr : mask.data(), T, N, F, cpu.data());
    bool cpu_ok = true;
    for (size_t i = 0; i < cpu.size(); ++i) {
      if (!corr_match(chunked[i], cpu[i])) { cpu_ok = false; break; }
    }
    printf("  [%s] %s chunked vs CPU reference (1e-12, informational)\n",
           cpu_ok ? "PASS" : "FAIL", name);
    if (!cpu_ok) ++g_fail;
  }
}

// ===========================================================================
// streaming (item 2) -- factor_corr input/transpose streaming driver.
// Kahan residency decision v2 (A) keeps d_Xt/d_valid resident in FULL; item 2
// removes the pre-transpose d_F overlap by uploading + transposing the input in
// per-slice chunks, so the full (T*N, F) d_F never exists on device. This
// closes memory_budget_v1's streaming (item 2) column: F=128 current peak
// (~12.6 GiB, d_F+d_Xt+d_valid overlap) -> streaming (d_Xt+d_valid+d_pp+d_F_chunk).
//   - transpose sub-chunks of max_transpose_rows upload a fixed d_F_chunk buffer
//     and run a RANGE transpose into d_Xt/d_valid at global row offset r0. The
//     transpose is element-independent (one thread per row), so the range kernel
//     is bitwise identical to production transpose_preprocess on the same rows.
//   - the two normal-pass reductions then slice across the `chunks` boundaries
//     exactly as minimal proof 2 (reduce_p1_cont / reduce_p2_cont carry
//     per-thread continuation state in d_pp; non-final chunk c*N multiple of 256).
//   - Kahan re-run stays a single full-R launch (decision A, reads d_Xt/d_valid).
// Returns 0 on success; negative on contract-error codes (-1..-11, validated
// BEFORE any allocation; -11 = R too close to INT32_MAX for the int-r loops,
// external F-05); positive cudaError_t value on a runtime CUDA failure
// (stream-4, internal review). chunk rows = chunks[i] * N.
// ===========================================================================

__global__ void transpose_preprocess_range(const double* __restrict__ d_src,
                                           const uint8_t* __restrict__ d_mask,
                                           int R, int r0, int r1, int F,
                                           double* __restrict__ d_Xt,
                                           uint8_t* __restrict__ d_valid) {
  // One thread per row in [r0, r1). d_src is the chunk-local (chunk_R, F)
  // buffer; d_Xt/d_valid are the global (F, R) arrays with stride R. Same
  // per-row logic as production transpose_preprocess (factor_corr_impl.cuh:206).
  int r = r0 + blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= r1) return;
  const double* row = d_src + (size_t)(r - r0) * F;
  uint8_t m = (d_mask != nullptr) ? d_mask[r] : 1u;
  for (int f = 0; f < F; ++f) {
    double v = row[f];
    uint8_t ok = static_cast<uint8_t>(m != 0u && isfinite(v));
    d_Xt[(size_t)f * R + r] = v;
    d_valid[(size_t)f * R + r] = ok;
  }
}

int factor_corr_gpu_stream(const double* h_F, const uint8_t* h_mask, int T, int N,
                           int F, const std::vector<int>& chunks,
                           int max_transpose_rows, double* h_out,
                           uint8_t* h_trigger_out = nullptr) {
  if (h_F == nullptr || h_out == nullptr || T < 1 || N < 1 || F < 1) return -1;
  if (static_cast<int64_t>(T) * N > INT32_MAX) return -2;
  // F-05 (external MAJOR): the continuation / range-transpose loops use int r
  // and advance r += blockDim.x (<=256); at R near INT32_MAX the final stride
  // could overflow signed int (negative wrap / out-of-bounds). Tighten R to a
  // safe ceiling -- far above any physically realizable panel (R=INT32_MAX
  // alone needs >8 GiB device residency) but guarantees r0+tid+256 never
  // overflows. (The production factor_corr_gpu shares the same int-r loops; the
  // streaming driver is tightened here as the reviewed surface.)
  if (static_cast<int64_t>(T) * N > static_cast<int64_t>(INT32_MAX) - 65536) return -11;
  if (F > kMaxF) return -3;
  if (chunks.empty()) return -5;
  if (max_transpose_rows < 1) return -10;

  // Pre-validate every chunk BEFORE any allocation (int64): each c >= 1, prefix
  // sum never exceeds T, total == T, and every NON-final chunk's c*N is a
  // multiple of 256 (block discipline -- the continuation index sequence equals
  // the non-chunked strided sequence only then). Same contract as the chunked
  // driver (minimal proof 2).
  int64_t prefix = 0;
  for (size_t ci = 0; ci < chunks.size(); ++ci) {
    const int c = chunks[ci];
    if (c < 1) return -7;
    prefix += c;
    if (prefix > T) return -8;
    if (ci + 1 < chunks.size()) {
      if ((static_cast<int64_t>(c) * N) % 256 != 0) return -9;
    }
  }
  if (prefix != T) return -6;

  const int R = T * N;
  const int P = F * (F + 1) / 2;

  cudaError_t err = cudaSuccess;
  const char* stage = "setup";
  // F-08 (external MINOR): keep the FIRST cudaFree error on the success path.
  // Declared before any goto so the fail label never bypasses initialization.
  cudaError_t free_err = cudaSuccess;
  auto keep_free = [&free_err](cudaError_t r) {
    if (free_err == cudaSuccess && r != cudaSuccess) free_err = r;
  };

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
  uint8_t* d_mask = nullptr;
  double* d_F_chunk = nullptr;
  Partial1* d_pp1 = nullptr;
  Partial2* d_pp2 = nullptr;

  stage = "alloc";
  err = cudaMalloc(&d_Xt, static_cast<size_t>(F) * R * sizeof(double));
  if (err == cudaSuccess) err = cudaMalloc(&d_valid, static_cast<size_t>(F) * R);
  if (err == cudaSuccess) err = cudaMalloc(&d_pairs, static_cast<size_t>(P) * 2 * sizeof(int));
  if (err == cudaSuccess) err = cudaMalloc(&d_gp1, static_cast<size_t>(P) * sizeof(Partial1));
  if (err == cudaSuccess) err = cudaMalloc(&d_means, static_cast<size_t>(P) * 2 * sizeof(double));
  if (err == cudaSuccess) err = cudaMalloc(&d_gp2, static_cast<size_t>(P) * sizeof(Partial2));
  if (err == cudaSuccess) err = cudaMalloc(&d_corr, static_cast<size_t>(P) * sizeof(double));
  if (err == cudaSuccess) err = cudaMalloc(&d_trigger, static_cast<size_t>(P));
  if (err == cudaSuccess) err = cudaMalloc(&d_trig_pairs, static_cast<size_t>(P) * sizeof(int));
  if (err == cudaSuccess) err = cudaMalloc(&d_gk1, static_cast<size_t>(P) * sizeof(PartialK1));
  if (err == cudaSuccess) err = cudaMalloc(&d_kmeans, static_cast<size_t>(P) * 2 * sizeof(double));
  if (err == cudaSuccess) err = cudaMalloc(&d_gk2, static_cast<size_t>(P) * sizeof(PartialK2));
  if (err == cudaSuccess) err = cudaMalloc(&d_out, static_cast<size_t>(F) * F * sizeof(double));
  if (err == cudaSuccess && h_mask != nullptr) err = cudaMalloc(&d_mask, static_cast<size_t>(R));
  // F-07 (external MINOR): allocate by the CLAMPED per-sub-chunk row count --
  // the transpose loop uses min(max_transpose_rows, R-r0), so an unclamped
  // max_transpose_rows (e.g. INT_MAX for a T=N=F=1 panel) would request ~16 GiB
  // for 1 row. Clamp to R (>= the actual max sub-chunk).
  if (err == cudaSuccess) err = cudaMalloc(&d_F_chunk,
      static_cast<size_t>(std::min(max_transpose_rows, R)) * F * sizeof(double));
  if (err == cudaSuccess) err = cudaMalloc(&d_pp1,
      static_cast<size_t>(P) * 256 * sizeof(Partial1));
  if (err == cudaSuccess) err = cudaMalloc(&d_pp2,
      static_cast<size_t>(P) * 256 * sizeof(Partial2));
  if (err != cudaSuccess) goto fail;

  stage = "pairs";
  make_pairs_kernel<<<F, 1>>>(d_pairs, F);
  err = cudaGetLastError();
  if (err != cudaSuccess) goto fail;

  // ---- upload mask once (R bytes; tiny vs the panel) -----------------------
  stage = "mask-upload";
  if (h_mask != nullptr) {
    err = cudaMemcpy(d_mask, h_mask, static_cast<size_t>(R), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail;
  }

  // ---- stage A: streamed upload + range transpose (d_F never resident) -----
  stage = "transpose";
  {
    const int block = 256;
    int r0 = 0;
    while (r0 < R) {
      const int chunk_rows = std::min(max_transpose_rows, R - r0);
      err = cudaMemcpy(d_F_chunk, h_F + static_cast<size_t>(r0) * F,
                       static_cast<size_t>(chunk_rows) * F * sizeof(double),
                       cudaMemcpyHostToDevice);
      if (err != cudaSuccess) goto fail;
      const int grid = static_cast<int>(1 + (chunk_rows - 1) / block);
      transpose_preprocess_range<<<grid, block>>>(d_F_chunk,
                                                  h_mask ? d_mask : nullptr,
                                                  R, r0, r0 + chunk_rows, F,
                                                  d_Xt, d_valid);
      err = cudaGetLastError();
      if (err != cudaSuccess) goto fail;
      r0 += chunk_rows;
    }
  }

  // ---- stage B: normal pass 1 (chunked continuation) -----------------------
  stage = "p1-cont";
  {
    int r0 = 0;
    for (size_t ci = 0; ci < chunks.size(); ++ci) {
      const int c = chunks[ci];
      const int r1 = r0 + c * N;
      reduce_p1_cont_kernel<<<P, 256>>>(d_Xt, d_valid, d_pairs, R, r0, r1,
                                        (ci == 0) ? 1 : 0, d_pp1);
      err = cudaGetLastError();
      if (err != cudaSuccess) goto fail;
      r0 = r1;
    }
    finalize_p1_from_pp_kernel<<<P, 256>>>(d_pp1, P, d_gp1);
    err = cudaGetLastError();
    if (err != cudaSuccess) goto fail;
    {
      const int grid = (P + 255) / 256;
      finalize_p1_kernel<<<grid, 256>>>(d_gp1, P, d_means);
      err = cudaGetLastError();
      if (err != cudaSuccess) goto fail;
    }
  }

  // ---- stage C: normal pass 2 (chunked continuation) -----------------------
  stage = "p2-cont";
  {
    int r0 = 0;
    for (size_t ci = 0; ci < chunks.size(); ++ci) {
      const int c = chunks[ci];
      const int r1 = r0 + c * N;
      reduce_p2_cont_kernel<<<P, 256>>>(d_Xt, d_valid, d_pairs, d_means, R, r0, r1,
                                        (ci == 0) ? 1 : 0, d_pp2);
      err = cudaGetLastError();
      if (err != cudaSuccess) goto fail;
      r0 = r1;
    }
    finalize_p2_from_pp_kernel<<<P, 256>>>(d_pp2, P, d_gp2);
    err = cudaGetLastError();
    if (err != cudaSuccess) goto fail;
    {
      const int grid = (P + 255) / 256;
      finalize_p2_kernel<<<grid, 256>>>(d_gp1, d_gp2, P, d_corr, d_trigger);
      err = cudaGetLastError();
      if (err != cudaSuccess) goto fail;
    }
  }

  // ---- stage D: Kahan re-run: single full-R launch (decision A) ------------
  stage = "kahan";
  {
    std::vector<uint8_t> host_trig(static_cast<size_t>(P));
    err = cudaMemcpy(host_trig.data(), d_trigger, static_cast<size_t>(P),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto fail;
    std::vector<int> trig_idx;
    trig_idx.reserve(static_cast<size_t>(P));
    for (int p = 0; p < P; ++p) {
      if (host_trig[static_cast<size_t>(p)] != 0u) trig_idx.push_back(p);
    }
    const int K = static_cast<int>(trig_idx.size());
    if (K > 0) {
      err = cudaMemcpy(d_trig_pairs, trig_idx.data(),
                       static_cast<size_t>(K) * sizeof(int), cudaMemcpyHostToDevice);
      if (err != cudaSuccess) goto fail;
      const int kb = (R >= 256) ? 256 : 1;
      kahan_reduce_p1_kernel<<<K, kb>>>(d_Xt, d_valid, d_pairs, d_trig_pairs, R, d_gk1);
      err = cudaGetLastError();
      if (err != cudaSuccess) goto fail;
      {
        const int grid = (K + 255) / 256;
        kahan_finalize_p1_kernel<<<grid, 256>>>(d_gk1, K, d_kmeans);
        err = cudaGetLastError();
        if (err != cudaSuccess) goto fail;
      }
      kahan_reduce_p2_kernel<<<K, kb>>>(d_Xt, d_valid, d_pairs, d_trig_pairs, d_kmeans, R, d_gk2);
      err = cudaGetLastError();
      if (err != cudaSuccess) goto fail;
      {
        const int grid = (K + 255) / 256;
        kahan_finalize_p2_kernel<<<grid, 256>>>(d_gk2, d_trig_pairs, K, d_corr);
        err = cudaGetLastError();
        if (err != cudaSuccess) goto fail;
      }
    }
  }

  // ---- stage E: writeback + download ---------------------------------------
  stage = "writeback";
  {
    const int grid = (P + 255) / 256;
    writeback_kernel<<<grid, 256>>>(d_corr, d_pairs, F, P, d_out);
    err = cudaGetLastError();
    if (err != cudaSuccess) goto fail;
  }
  stage = "d2h";
  err = cudaMemcpy(h_out, d_out, static_cast<size_t>(F) * F * sizeof(double),
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) goto fail;
  if (h_trigger_out != nullptr) {
    err = cudaMemcpy(h_trigger_out, d_trigger, static_cast<size_t>(P),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto fail;
  }

  // F-08: keep the FIRST cudaFree error on the success path (declared at the
  // top of the function) -- a failed release after a successful compute must not
  // silently return 0 (a leaked allocation would pollute the next HWM case's
  // free-before).
  keep_free(cudaFree(d_mask)); keep_free(cudaFree(d_Xt)); keep_free(cudaFree(d_valid));
  keep_free(cudaFree(d_pairs)); keep_free(cudaFree(d_gp1)); keep_free(cudaFree(d_means));
  keep_free(cudaFree(d_gp2)); keep_free(cudaFree(d_corr)); keep_free(cudaFree(d_trigger));
  keep_free(cudaFree(d_trig_pairs)); keep_free(cudaFree(d_gk1)); keep_free(cudaFree(d_kmeans));
  keep_free(cudaFree(d_gk2)); keep_free(cudaFree(d_out)); keep_free(cudaFree(d_F_chunk));
  keep_free(cudaFree(d_pp1)); keep_free(cudaFree(d_pp2));
  return static_cast<int>(free_err);

fail:
  fprintf(stderr, "  stream driver FAIL at stage=%s err=%d (%s)\n", stage,
          static_cast<int>(err), cudaGetErrorString(err));
  cudaFree(d_mask); cudaFree(d_Xt); cudaFree(d_valid); cudaFree(d_pairs);
  cudaFree(d_gp1); cudaFree(d_means); cudaFree(d_gp2); cudaFree(d_corr);
  cudaFree(d_trigger); cudaFree(d_trig_pairs); cudaFree(d_gk1); cudaFree(d_kmeans);
  cudaFree(d_gk2); cudaFree(d_out); cudaFree(d_F_chunk); cudaFree(d_pp1); cudaFree(d_pp2);
  return static_cast<int>(err);
}

// Run one streaming case: production (non-blocked) vs streamed output -- corr
// (F,F) bitwise equal + trigger bitset byte equal. check_cpu anchors the
// streamed output against the independent CPU reference (1e-12) to rule out
// "equal but wrong". expected_K (>=0) hard-asserts the Kahan branch count from
// the streamed trigger bitset, so a drift that eliminates Kahan coverage in
// BOTH paths is caught (same discipline as the chunked proof, F-01).
void run_factor_stream_case(const char* name, const std::vector<double>& F3,
                            const std::vector<uint8_t>& mask, int T, int N, int F,
                            const std::vector<int>& chunks, int max_transpose_rows,
                            bool check_cpu, int expected_K) {
  const int P = F * (F + 1) / 2;
  std::vector<double> full(static_cast<size_t>(F) * F), streamed(static_cast<size_t>(F) * F);
  std::vector<uint8_t> ft(static_cast<size_t>(P)), st(static_cast<size_t>(P));
  int rc1 = factor_corr_gpu(F3.data(), mask.empty() ? nullptr : mask.data(), T, N, F,
                            full.data(), nullptr, ft.data());
  int rc2 = factor_corr_gpu_stream(F3.data(), mask.empty() ? nullptr : mask.data(),
                                   T, N, F, chunks, max_transpose_rows,
                                   streamed.data(), st.data());
  if (rc1 != 0 || rc2 != 0) {
    printf("  [FAIL] %s (rc1=%d rc2=%d)\n", name, rc1, rc2);
    ++g_fail;
    return;
  }
  const bool corr_eq = vec_bitwise_eq(full, streamed);
  const bool trig_eq = vec_bitwise_eq_u8(ft, st);
  const bool ok = corr_eq && trig_eq;
  printf("  [%s] %s stream (T=%d N=%d F=%d chunks=%zu tt=%d) corr_eq=%s trigger_eq=%s\n",
         ok ? "PASS" : "FAIL", name, T, N, F, chunks.size(), max_transpose_rows,
         corr_eq ? "YES" : "no", trig_eq ? "YES" : "no");
  if (!ok) {
    for (size_t i = 0; i < full.size(); ++i) {
      uint64_t bf, bs;
      std::memcpy(&bf, &full[i], 8);
      std::memcpy(&bs, &streamed[i], 8);
      if (bf != bs) {
        printf("      first corr bitwise mismatch idx=%zu full=%.17g streamed=%.17g\n",
               i, full[i], streamed[i]);
        break;
      }
    }
    ++g_fail;
  }
  if (expected_K >= 0) {
    int k_stream = 0;
    for (uint8_t v : st) if (v != 0u) ++k_stream;
    if (k_stream != expected_K) {
      printf("  [FAIL] %s Kahan branch coverage drift: expected_K=%d got=%d\n",
             name, expected_K, k_stream);
      ++g_fail;
    }
  }
  if (check_cpu && ok) {
    std::vector<double> cpu(static_cast<size_t>(F) * F);
    cpu_factor_corr(F3.data(), mask.empty() ? nullptr : mask.data(), T, N, F, cpu.data());
    bool cpu_ok = true;
    for (size_t i = 0; i < cpu.size(); ++i) {
      if (!corr_match(streamed[i], cpu[i])) { cpu_ok = false; break; }
    }
    printf("  [%s] %s stream vs CPU reference (1e-12, informational)\n",
           cpu_ok ? "PASS" : "FAIL", name);
    if (!cpu_ok) ++g_fail;
  }
}

// ===========================================================================
// corr width (pair-axis) blocking minimal proof -- factor_corr F-blocking.
// Splits the F axis into blocks of `block_width`; for each lower-triangle tile
// (block_a, block_b) with a>=b, runs the PRODUCTION kernels on tile-local
// buffers holding only the tile's factor columns (d_Xt_tile = (Ba+Bb)*R, the
// F-blocking residency model of memory_budget_v1.py). Every pair is computed
// independently (pair-axis blocking), so the result must be bitwise identical
// to the non-blocked path. No continuation state is carried -- this is the
// trivial "pair-independent" blocking (ft_chunking_design_spec width 分块).
//   block_width   : F-axis block width (1..F).
//   h_trigger_out : optional (P) uint8 trigger bitset for cross-check.
// Returns 0 on success; prints per-tile execution evidence.
int factor_corr_gpu_fblock(const double* h_F, const uint8_t* h_mask, int T, int N,
                           int F, int block_width, double* h_out,
                           uint8_t* h_trigger_out = nullptr) {
  if (h_F == nullptr || h_out == nullptr || T < 1 || N < 1 || F < 1) return -1;
  if (static_cast<int64_t>(T) * N > INT32_MAX) return -2;
  if (F > kMaxF) return -3;
  if (block_width < 1 || block_width > F) return -6;
  const int R = T * N;
  const int P = F * (F + 1) / 2;
  const int n_blocks = (F + block_width - 1) / block_width;
  const int max_cols = 2 * block_width;
  const int max_pairs = block_width * block_width + block_width * (block_width + 1) / 2;

  std::vector<double> corr(static_cast<size_t>(F) * F, std::nan(""));
  std::vector<uint8_t> trigger(static_cast<size_t>(P), 0);

  cudaError_t err = cudaSuccess;
  double *d_Xt_tile = nullptr;
  int *d_pairs_tile = nullptr;
  uint8_t *d_valid_tile = nullptr;
  Partial1 *d_gp1_tile = nullptr;
  double *d_means_tile = nullptr;
  Partial2 *d_gp2_tile = nullptr;
  double *d_corr_tile = nullptr, *d_F_tile = nullptr;
  uint8_t *d_mask_tile = nullptr;
  PartialK1 *d_gk1_tile = nullptr;
  double *d_kmeans_tile = nullptr;
  PartialK2 *d_gk2_tile = nullptr;
  uint8_t *d_trigger_tile = nullptr;
  int *d_trig_pairs_tile = nullptr;

  const size_t cols_bytes = static_cast<size_t>(max_cols) * R * sizeof(double);
  const size_t pairs_cap = static_cast<size_t>(max_pairs);
  err = cudaMalloc(&d_Xt_tile, cols_bytes);
  if (err == cudaSuccess) err = cudaMalloc(&d_valid_tile, static_cast<size_t>(max_cols) * R);
  if (err == cudaSuccess) err = cudaMalloc(&d_F_tile, cols_bytes);
  // transpose reads d_mask[r] (one mask byte per row, production d_mask = R).
  if (err == cudaSuccess) err = cudaMalloc(&d_mask_tile, static_cast<size_t>(R));
  if (err == cudaSuccess) err = cudaMalloc(&d_pairs_tile, pairs_cap * 2 * sizeof(int));
  if (err == cudaSuccess) err = cudaMalloc(&d_gp1_tile, pairs_cap * sizeof(Partial1));
  if (err == cudaSuccess) err = cudaMalloc(&d_means_tile, pairs_cap * 2 * sizeof(double));
  if (err == cudaSuccess) err = cudaMalloc(&d_gp2_tile, pairs_cap * sizeof(Partial2));
  if (err == cudaSuccess) err = cudaMalloc(&d_corr_tile, pairs_cap * sizeof(double));
  if (err == cudaSuccess) err = cudaMalloc(&d_trigger_tile, pairs_cap);
  if (err == cudaSuccess) err = cudaMalloc(&d_trig_pairs_tile, pairs_cap * sizeof(int));
  if (err == cudaSuccess) err = cudaMalloc(&d_gk1_tile, pairs_cap * sizeof(PartialK1));
  if (err == cudaSuccess) err = cudaMalloc(&d_kmeans_tile, pairs_cap * 2 * sizeof(double));
  if (err == cudaSuccess) err = cudaMalloc(&d_gk2_tile, pairs_cap * sizeof(PartialK2));
  if (err != cudaSuccess) {
    printf("  fblock cudaMalloc FAIL at alloc (err=%d)\n", static_cast<int>(err));
    cudaFree(d_Xt_tile); cudaFree(d_valid_tile); cudaFree(d_F_tile); cudaFree(d_mask_tile);
    cudaFree(d_pairs_tile); cudaFree(d_gp1_tile); cudaFree(d_means_tile); cudaFree(d_gp2_tile);
    cudaFree(d_corr_tile); cudaFree(d_trigger_tile); cudaFree(d_trig_pairs_tile);
    cudaFree(d_gk1_tile); cudaFree(d_kmeans_tile); cudaFree(d_gk2_tile);
    return -7;
  }

  for (int a = 0; a < n_blocks && err == cudaSuccess; ++a) {
    for (int b = 0; b <= a && err == cudaSuccess; ++b) {
      const int fa0 = a * block_width, fa1 = std::min((a + 1) * block_width, F);
      const int fb0 = b * block_width, fb1 = std::min((b + 1) * block_width, F);
      const int Ba = fa1 - fa0, Bb = fb1 - fb0;
      // Diagonal tile (a==b) covers the block's own lower-triangle pairs and
      // holds only the block's Ba columns; off-diagonal tile (a>b) covers the
      // cross pairs block_a x block_b (i in A, j in B, always i>j) and holds
      // [block_b cols | block_a cols]. No block is duplicated.
      const bool diag_tile = (a == b);
      const int tile_cols = diag_tile ? Ba : (Ba + Bb);
      const int tile_pairs = diag_tile ? (Ba * (Ba + 1) / 2) : (Ba * Bb);
      if (tile_pairs == 0) continue;

      // 1. upload tile columns; local layout for off-diag [block_b | block_a]
      // h_F is row-major (R, F): h_F[r*F + col]; h_mask is per-row (R) u8,
      // shared by every factor column (production d_mask = R bytes).
      std::vector<double> F_tile(static_cast<size_t>(tile_cols) * R);
      for (int r = 0; r < R; ++r) {
        for (int t = 0; t < tile_cols; ++t) {
          const int gcol = diag_tile ? (fa0 + t)
                                     : ((t < Bb) ? (fb0 + t) : (fa0 + (t - Bb)));
          F_tile[static_cast<size_t>(r) * tile_cols + t] =
              h_F[static_cast<size_t>(r) * F + gcol];
        }
      }
      err = cudaMemcpy(d_F_tile, F_tile.data(),
                       static_cast<size_t>(tile_cols) * R * sizeof(double),
                       cudaMemcpyHostToDevice);
      if (err != cudaSuccess) { printf("  fblock err after F memcpy (%d)\n", (int)err); break; }
      if (h_mask) {
        err = cudaMemcpy(d_mask_tile, h_mask, static_cast<size_t>(R),
                         cudaMemcpyHostToDevice);
        if (err != cudaSuccess) { printf("  fblock err after mask memcpy (%d)\n", (int)err); break; }
      }

      // 2. transpose (production kernel)
      {
        const int block = 256;
        const int grid = static_cast<int>(1 + (static_cast<int64_t>(R) - 1) / block);
        transpose_preprocess<<<grid, block>>>(d_F_tile, h_mask ? d_mask_tile : nullptr,
                                              R, tile_cols, d_Xt_tile, d_valid_tile);
        err = cudaGetLastError();
        if (err != cudaSuccess) { printf("  fblock err after transpose (%d)\n", (int)err); break; }
      }

      // 3. tile pair table (host): diagonal tile -> block diagonal pairs;
      // off-diagonal tile -> all cross pairs (block_a x block_b)
      std::vector<int> pairs;
      pairs.reserve(static_cast<size_t>(tile_pairs) * 2);
      if (diag_tile) {
        for (int li = 0; li < Ba; ++li)
          for (int lj = 0; lj <= li; ++lj) { pairs.push_back(li); pairs.push_back(lj); }
      } else {
        for (int li = Bb; li < Bb + Ba; ++li)
          for (int lj = 0; lj < Bb; ++lj) { pairs.push_back(li); pairs.push_back(lj); }
      }
      err = cudaMemcpy(d_pairs_tile, pairs.data(),
                       static_cast<size_t>(tile_pairs) * 2 * sizeof(int),
                       cudaMemcpyHostToDevice);
      if (err != cudaSuccess) break;

      // 4. production kernels (tile-local)
      reduce_p1_kernel<<<tile_pairs, 256>>>(d_Xt_tile, d_valid_tile, d_pairs_tile, R, d_gp1_tile);
      err = cudaGetLastError();
      if (err != cudaSuccess) break;
      {
        const int block = 256;
        finalize_p1_kernel<<<(tile_pairs + block - 1) / block, block>>>(d_gp1_tile, tile_pairs, d_means_tile);
        err = cudaGetLastError();
        if (err != cudaSuccess) break;
      }
      reduce_p2_kernel<<<tile_pairs, 256>>>(d_Xt_tile, d_valid_tile, d_pairs_tile, d_means_tile, R, d_gp2_tile);
      err = cudaGetLastError();
      if (err != cudaSuccess) break;
      {
        const int block = 256;
        finalize_p2_kernel<<<(tile_pairs + block - 1) / block, block>>>(d_gp1_tile, d_gp2_tile, tile_pairs, d_corr_tile, d_trigger_tile);
        err = cudaGetLastError();
        if (err != cudaSuccess) break;
      }

      // 5. Kahan (tile-local); keep tile trigger for global writeback
      int K = 0;
      std::vector<uint8_t> tile_trig(static_cast<size_t>(tile_pairs), 0u);
      err = cudaMemcpy(tile_trig.data(), d_trigger_tile, static_cast<size_t>(tile_pairs),
                       cudaMemcpyDeviceToHost);
      if (err != cudaSuccess) break;
      for (uint8_t v : tile_trig) if (v != 0u) ++K;
      if (K > 0) {
        std::vector<int> trig_idx;
        for (int p = 0; p < tile_pairs; ++p)
          if (tile_trig[static_cast<size_t>(p)] != 0u) trig_idx.push_back(p);
        err = cudaMemcpy(d_trig_pairs_tile, trig_idx.data(),
                         static_cast<size_t>(K) * sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) break;
        const int kb = (R >= 256) ? 256 : 1;
        kahan_reduce_p1_kernel<<<K, kb>>>(d_Xt_tile, d_valid_tile, d_pairs_tile, d_trig_pairs_tile, R, d_gk1_tile);
        err = cudaGetLastError();
        if (err != cudaSuccess) break;
        {
          const int block = 256;
          kahan_finalize_p1_kernel<<<(K + block - 1) / block, block>>>(d_gk1_tile, K, d_kmeans_tile);
          err = cudaGetLastError();
          if (err != cudaSuccess) break;
        }
        kahan_reduce_p2_kernel<<<K, kb>>>(d_Xt_tile, d_valid_tile, d_pairs_tile, d_trig_pairs_tile, d_kmeans_tile, R, d_gk2_tile);
        err = cudaGetLastError();
        if (err != cudaSuccess) break;
        {
          const int block = 256;
          kahan_finalize_p2_kernel<<<(K + block - 1) / block, block>>>(d_gk2_tile, d_trig_pairs_tile, K, d_corr_tile);
          err = cudaGetLastError();
          if (err != cudaSuccess) break;
        }
      }

      // 6. writeback to global corr + trigger (host; mirror upper triangle after loop)
      std::vector<double> tile_corr(static_cast<size_t>(tile_pairs));
      err = cudaMemcpy(tile_corr.data(), d_corr_tile,
                       static_cast<size_t>(tile_pairs) * sizeof(double),
                       cudaMemcpyDeviceToHost);
      if (err != cudaSuccess) break;
      {
        int pi = 0;
        if (diag_tile) {
          for (int li = 0; li < Ba; ++li)
            for (int lj = 0; lj <= li; ++lj, ++pi) {
              const int i = fa0 + li, j = fa0 + lj;  // global i>=j within block
              double v = tile_corr[static_cast<size_t>(pi)];
              // F3: diagonal NaN uses the production writeback pattern
              // (CUDA NAN = 0x7ff8000000000000) to avoid payload drift.
              if (i == j) v = std::isfinite(v) ? 1.0 : mk_f64(0x7ff8000000000000ull);
              corr[static_cast<size_t>(i) * F + j] = v;
              trigger[static_cast<size_t>(i) * (i + 1) / 2 + j] = tile_trig[static_cast<size_t>(pi)];
            }
        } else {
          for (int li = Bb; li < Bb + Ba; ++li)
            for (int lj = 0; lj < Bb; ++lj, ++pi) {
              const int i = fa0 + (li - Bb), j = fb0 + lj;  // global i>j
              corr[static_cast<size_t>(i) * F + j] = tile_corr[static_cast<size_t>(pi)];
              trigger[static_cast<size_t>(i) * (i + 1) / 2 + j] = tile_trig[static_cast<size_t>(pi)];
            }
        }
      }
      printf("  fblock tile (a=%d b=%d) cols=%d pairs=%d K=%d\n",
             a, b, tile_cols, tile_pairs, K);
    }
  }

  if (err == cudaSuccess) {
    // mirror upper triangle (production writeback mirrors; keep bitwise)
    for (int i = 0; i < F; ++i)
      for (int j = i + 1; j < F; ++j)
        corr[static_cast<size_t>(i) * F + j] = corr[static_cast<size_t>(j) * F + i];
    std::memcpy(h_out, corr.data(), static_cast<size_t>(F) * F * sizeof(double));
    if (h_trigger_out)
      std::memcpy(h_trigger_out, trigger.data(), static_cast<size_t>(P));
  }

  if (err != cudaSuccess)
    printf("  fblock FAIL err=%d (%s)\n", static_cast<int>(err),
           cudaGetErrorString(err));
  cudaFree(d_Xt_tile); cudaFree(d_valid_tile); cudaFree(d_F_tile); cudaFree(d_mask_tile);
  cudaFree(d_pairs_tile); cudaFree(d_gp1_tile); cudaFree(d_means_tile); cudaFree(d_gp2_tile);
  cudaFree(d_corr_tile); cudaFree(d_trigger_tile); cudaFree(d_trig_pairs_tile);
  cudaFree(d_gk1_tile); cudaFree(d_kmeans_tile); cudaFree(d_gk2_tile);
  return (err == cudaSuccess) ? 0 : -7;
}

// Run one F-blocking case: production (non-blocked) vs fblock output -- corr
// (F,F) bitwise equal + trigger bitset byte equal (expected_K pins Kahan branch
// coverage). check_cpu anchors against the independent CPU reference (1e-12).
void run_factor_fblock_case(const char* name, const std::vector<double>& F3,
                            const std::vector<uint8_t>& mask, int T, int N, int F,
                            int block_width, bool check_cpu, int expected_K) {
  const int P = F * (F + 1) / 2;
  std::vector<double> full(static_cast<size_t>(F) * F), blocked(static_cast<size_t>(F) * F);
  std::vector<uint8_t> ft(static_cast<size_t>(P)), bt(static_cast<size_t>(P));
  int rc1 = factor_corr_gpu(F3.data(), mask.empty() ? nullptr : mask.data(), T, N, F,
                            full.data(), nullptr, ft.data());
  int rc2 = factor_corr_gpu_fblock(F3.data(), mask.empty() ? nullptr : mask.data(),
                                   T, N, F, block_width, blocked.data(), bt.data());
  if (rc1 != 0 || rc2 != 0) {
    printf("  [FAIL] %s (rc1=%d rc2=%d)\n", name, rc1, rc2);
    ++g_fail;
    return;
  }
  const bool corr_eq = vec_bitwise_eq(full, blocked);
  const bool trig_eq = vec_bitwise_eq_u8(ft, bt);
  printf("  [%s] %s fblock (T=%d N=%d F=%d block=%d) corr_eq=%s trigger_eq=%s\n",
         (corr_eq && trig_eq) ? "PASS" : "FAIL", name, T, N, F, block_width,
         corr_eq ? "YES" : "no", trig_eq ? "YES" : "no");
  if (!(corr_eq && trig_eq)) {
    for (size_t i = 0; i < full.size(); ++i) {
      uint64_t bf, bc;
      std::memcpy(&bf, &full[i], 8);
      std::memcpy(&bc, &blocked[i], 8);
      if (bf != bc) {
        printf("      first corr bitwise mismatch idx=%zu full=%.17g blocked=%.17g\n",
               i, full[i], blocked[i]);
        break;
      }
    }
    ++g_fail;
  }
  if (expected_K >= 0) {
    int k = 0;
    for (uint8_t v : bt) if (v != 0u) ++k;
    if (k != expected_K) {
      printf("  [FAIL] %s Kahan branch coverage drift: expected_K=%d got=%d\n",
             name, expected_K, k);
      ++g_fail;
    }
  }
  if (check_cpu && corr_eq && trig_eq) {
    std::vector<double> cpu(static_cast<size_t>(F) * F);
    cpu_factor_corr(F3.data(), mask.empty() ? nullptr : mask.data(), T, N, F, cpu.data());
    bool cpu_ok = true;
    for (size_t i = 0; i < cpu.size(); ++i) {
      if (!corr_match(blocked[i], cpu[i])) { cpu_ok = false; break; }
    }
    printf("  [%s] %s fblock vs CPU reference (1e-12, informational)\n",
           cpu_ok ? "PASS" : "FAIL", name);
    if (!cpu_ok) ++g_fail;
  }
}

}  // namespace

// ===========================================================================
// --hwm-f128 mode: measure the device HWM of factor_corr_gpu_fblock on the
// F=128 (N=5000, T=1218) scenario plus F=12 cross-check anchors, closing the
// memory_budget_v1 fblock(项③) model prediction (F128 block=32 -> 6325.11 MiB)
// to a measured value. Runs ONLY when the exe is invoked as
//   poc3_factor_corr_selfcheck.exe --hwm-f128
// The default selfcheck body is untouched (argc/argv branch returns early).
//
// Measurement is the calibration "driver sample" (third) leg only:
//   driver_peak = free_before - min_free   (background cudaMemGetInfo thread)
// factor_corr_gpu_fblock takes no MemTracker, so the tracker-HWM and
// delta_formula legs of the 5-op calibration do not apply. Acceptance is
// adjudicated in benchmarks/factor_fblock_hwm_v1.py:
//   fit cases : 0 <= driver_peak - model_peak <= 64 MiB
//   OOM cases : rc != 0 (cudaMalloc failure), driver_peak < model_peak
// Every (T,N,F,block) case gets its OWN sampler + free_before snapshot so a
// neighbour's allocation peak cannot pollute the min_free attribution.
//
// ASCII-only comments (nvcc/GBK pitfall).
struct FblkHwmCase {
  const char* kind;  // "fblock" or "production"
  int T, N, F, block;
  int reps;
  size_t fb = 0;            // per-case free_before (before the case's allocs)
  size_t driver_peak = 0;   // fb - min_free during the case
  int samples = 0;          // successful sampler polls during the case (health, review F2)
  int rc = 0;
};

void FblkRunSampler(std::atomic<size_t>* min_free, std::atomic<int>* sample_count,
                    std::atomic<bool>* stop, int dev) {
  // Review F2 (external): a sampler that fails silently (bad device, all
  // cudaMemGetInfo failing) must NOT yield driver_peak==0 that the report would
  // read as a "fit". We count successful samples and let the caller handshake.
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

void PrintFblkCase(const FblkHwmCase& c) {
  // fb=<per-case free_before> lets the Python report compute the exhausted gate
  // per case (margin = fb_case - driver_peak == min_free) instead of against a
  // single header free_before (review F2: baseline drift could mislabel).
  printf("FBLC|kind=%s|T=%d|N=%d|F=%d|block=%d|reps=%d|fb=%zu|driver_peak=%zu|samples=%d|rc=%d\n",
         c.kind, c.T, c.N, c.F, c.block, c.reps, c.fb, c.driver_peak, c.samples, c.rc);
  printf("  [%s] %s T=%d N=%d F=%d block=%d: driver_peak %.2f MiB rc=%d\n",
         c.rc == 0 ? "RUN" : "OOM", c.kind, c.T, c.N, c.F, c.block,
         c.driver_peak / 1048576.0, c.rc);
}

int run_hwm_f128(int dev) {
  int guard_fail = 0;
  const int T = 1218;
  const int N = 5000;
  const int F = 128;
  const int64_t R = static_cast<int64_t>(T) * N;

  size_t free0 = 0, total = 0;
  cudaMemGetInfo(&free0, &total);
  printf("HWM mode: GPU free_before %.0f MiB / total %.0f MiB\n",
         free0 / 1048576.0, total / 1048576.0);

  // ---- F=128 panel: single construction, reused across all blocks. ----------
  // Values do not affect HWM (peak depends only on allocation sizes); low-bias
  // uniform [-2,2] keeps the Kahan branch un-triggered (faster, same HWM).
  printf("== hwm-f128: building F=%d N=%d panel (%lld doubles, %.0f MiB host) ==\n",
         F, N, R * F, static_cast<double>(R) * F * 8 / 1048576.0);
  std::vector<double> F3(static_cast<size_t>(R) * F);
  std::vector<uint8_t> mask(static_cast<size_t>(R));
  {
    Lcg rng(0x5EEDC0DEu);
    for (int64_t i = 0; i < R * F; ++i)
      F3[static_cast<size_t>(i)] = rng.uniform() * 4.0 - 2.0;
    for (int64_t i = 0; i < R; ++i)
      mask[static_cast<size_t>(i)] = (rng.next() % 100) < 98 ? 1 : 0;
  }

  // ---- per-case measurement: own sampler + own free_before -------------------
  auto measure = [&](const char* kind, int Tn, int Nn, int Fn, int block,
                     const std::vector<double>& Fp,
                     const std::vector<uint8_t>& msk, int reps) -> FblkHwmCase {
    FblkHwmCase c;
    c.kind = kind; c.T = Tn; c.N = Nn; c.F = Fn; c.block = block; c.reps = reps;
    size_t fb = 0, total2 = 0;
    // Health: a failed cudaMemGetInfo leaves fb=0, which the report must reject
    // (driver_peak = fb - min_free would otherwise be meaningless).
    if (cudaMemGetInfo(&fb, &total2) != cudaSuccess) fb = 0;
    c.fb = fb;
    std::atomic<size_t> min_free{fb};
    std::atomic<int> sample_count{0};
    std::atomic<bool> stop{false};
    std::thread sampler(FblkRunSampler, &min_free, &sample_count, &stop, dev);
    // Handshake (review F2): wait for at least one successful sample before
    // running the case, so a dead sampler cannot silently yield driver_peak==0.
    for (int i = 0; i < 2000 && sample_count.load() == 0; ++i)
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    int rc = 0;
    for (int i = 0; i < reps; ++i) {
      std::vector<double> out(static_cast<size_t>(Fn) * Fn);
      std::vector<uint8_t> trig(static_cast<size_t>(Fn) * (Fn + 1) / 2);
      if (std::strcmp(kind, "fblock") == 0)
        rc = factor_corr_gpu_fblock(Fp.data(), msk.empty() ? nullptr : msk.data(),
                                    Tn, Nn, Fn, block, out.data(), trig.data());
      else
        rc = factor_corr_gpu(Fp.data(), msk.empty() ? nullptr : msk.data(),
                             Tn, Nn, Fn, out.data());
      if (rc != 0) break;  // OOM case: stop after first failure
    }
    stop.store(true);
    sampler.join();
    cudaDeviceSynchronize();
    c.rc = rc;
    size_t fa = 0;
    cudaMemGetInfo(&fa, &total);
    c.samples = sample_count.load();
    c.driver_peak = (fb == 0) ? 0 : fb - min_free.load();
    return c;
  };

  // ---- cross-check anchors: F=12 block=6 (model==impl at block<=F/2). -------
  // Model predicts 1190.63 / 2381.24 MiB == calibration current HWM 1191/2381.
  printf("== hwm-f128: F=12 cross-check anchors (block=6) ==\n");
  for (int Nn : {5000, 10000}) {
    const int Fn = 12;
    const int64_t Rn = static_cast<int64_t>(T) * Nn;
    std::vector<double> F3x(static_cast<size_t>(Rn) * Fn);
    std::vector<uint8_t> maskx(static_cast<size_t>(Rn));
    {
      Lcg rng(0xF12C0DEu);
      for (int64_t i = 0; i < Rn * Fn; ++i)
        F3x[static_cast<size_t>(i)] = rng.uniform() * 4.0 - 2.0;
      for (int64_t i = 0; i < Rn; ++i)
        maskx[static_cast<size_t>(i)] = (rng.next() % 100) < 98 ? 1 : 0;
    }
    FblkHwmCase c = measure("fblock", T, Nn, Fn, /*block=*/6, F3x, maskx, /*reps=*/2);
    PrintFblkCase(c);
    // bitwise guard: fblock vs production on the same small panel (cheap,
    // 1.2 KB output) -- proves the measured implementation did not drift.
    // Review F3: make it LOAD-BEARING -- a drift must fail the run (GUARD|
    // record is parsed by the report generator; run_hwm_f128 returns nonzero)
    // and the trigger bitsets are compared too (not just corr).
    if (Nn == 5000) {
      std::vector<double> full(static_cast<size_t>(Fn) * Fn);
      std::vector<double> blk(static_cast<size_t>(Fn) * Fn);
      std::vector<uint8_t> ft(static_cast<size_t>(Fn) * (Fn + 1) / 2);
      std::vector<uint8_t> bt(static_cast<size_t>(Fn) * (Fn + 1) / 2);
      int r1 = factor_corr_gpu(F3x.data(), maskx.data(), T, Nn, Fn, full.data(),
                               nullptr, ft.data());
      int r2 = factor_corr_gpu_fblock(F3x.data(), maskx.data(), T, Nn, Fn, 6,
                                      blk.data(), bt.data());
      bool eq = (r1 == 0 && r2 == 0) && vec_bitwise_eq(full, blk) &&
                vec_bitwise_eq_u8(ft, bt);
      printf("GUARD|pass=%d|r1=%d|r2=%d\n", eq ? 1 : 0, r1, r2);
      printf("  [%s] F=12 block=6 fblock vs production bitwise+trigger (r1=%d r2=%d)\n",
             eq ? "PASS" : "FAIL", r1, r2);
      if (!eq) guard_fail = 1;
    }
  }

  // ---- F=128 blocks {8,16,32} (fit ladder) then 64 (expected OOM) last. ----
  // block=1 omitted: its 8256-tile host loop (~48 MB host staging per tile,
  // ~396 GB total) is prohibitive (~30+ min on this panel) and adds no decision
  // value -- block=8/16/32 already show the peak-vs-blockwidth decreasing
  // ladder and block=1's 203 MiB model peak is far below budget. reps=1 is
  // sufficient (the tile buffers are held for the whole kernel execution, so
  // the background sampler has a wide window; the allocation instant is
  // synchronous on the host side).
  printf("== hwm-f128: F=%d N=%d blocks ==\n", F, N);
  for (int block : {8, 16, 32, 64}) {
    FblkHwmCase c = measure("fblock", T, N, F, block, F3, mask, /*reps=*/1);
    PrintFblkCase(c);
  }

  // ---- production (non-blocked) F=128: expected cudaMalloc OOM. -------------
  // Confirms the model's current-peak claim (12645 MiB > device) is real,
  // though the 12645 MiB peak itself is unreachable on this device.
  printf("== hwm-f128: production (non-blocked) F=%d ==\n", F);
  {
    FblkHwmCase c = measure("production", T, N, F, /*block=*/0, F3, mask, /*reps=*/1);
    PrintFblkCase(c);
  }

  // Context health after the OOM cases (driver must not be poisoned).
  cudaError_t err = cudaGetLastError();
  printf("  post-OOM cudaGetLastError: %s\n", cudaGetErrorString(err));
  printf("== summary: adjudication (delta vs model) in benchmarks/factor_fblock_hwm_v1.py ==\n");
  if (guard_fail) {
    printf("GUARD FAIL: fblock drift detected; run failing\n");
    return 2;
  }
  return 0;
}

// --hwm-stream mode: measure the device HWM of the streaming (item 2)
// factor_corr input/transpose path on the F=128 (N=5000, T=1218) scenario plus
// F=12 cross-check anchors, closing the memory_budget_v1 streaming(项②) model
// prediction (F128 ~6698 MiB = d_Xt+d_valid+d_pp+d_F_chunk) to a measured value.
// Runs ONLY when invoked as `poc3_factor_corr_selfcheck.exe --hwm-stream`.
// Measurement is the driver-sample leg (fb - min_free via background sampler);
// acceptance adjudicated in benchmarks/factor_stream_hwm_v1.py (fail-closed).
// FBLC|kind=stream uses `block` = max_transpose_rows (the fixed d_F_chunk
// buffer rows); kind=production uses block=0. Continuation chunks = 37x32 + 34
// rows (every non-final chunk 32*5000 = 160000 = 625*256, block discipline).
int run_hwm_stream(int dev) {
  int guard_fail = 0;
  const int T = 1218;
  const int N = 5000;
  const int F = 128;
  const int max_tt = 4096;
  const int64_t R = static_cast<int64_t>(T) * N;

  auto mk_chunks32 = []() {
    std::vector<int> c;
    for (int i = 0; i < 37; ++i) c.push_back(32);
    c.push_back(34);
    return c;
  };

  size_t free0 = 0, total = 0;
  cudaMemGetInfo(&free0, &total);
  printf("HWM mode: GPU free_before %.0f MiB / total %.0f MiB\n",
         free0 / 1048576.0, total / 1048576.0);

  // ---- F=128 panel: single construction, reused across all cases. ----------
  // Values do not affect HWM (peak depends only on allocation sizes); low-bias
  // uniform [-2,2] keeps the Kahan branch un-triggered (faster, same HWM).
  printf("== hwm-stream: building F=%d N=%d panel (%lld doubles, %.0f MiB host) ==\n",
         F, N, R * F, static_cast<double>(R) * F * 8 / 1048576.0);
  std::vector<double> F3(static_cast<size_t>(R) * F);
  std::vector<uint8_t> mask(static_cast<size_t>(R));
  {
    Lcg rng(0x5EEDC0DEu);
    for (int64_t i = 0; i < R * F; ++i)
      F3[static_cast<size_t>(i)] = rng.uniform() * 4.0 - 2.0;
    for (int64_t i = 0; i < R; ++i)
      mask[static_cast<size_t>(i)] = (rng.next() % 100) < 98 ? 1 : 0;
  }

  // ---- per-case measurement: own sampler + own free_before -------------------
  auto measure = [&](const char* kind, int Tn, int Nn, int Fn,
                     const std::vector<int>& chunks, int tt,
                     const std::vector<double>& Fp,
                     const std::vector<uint8_t>& msk, int reps) -> FblkHwmCase {
    FblkHwmCase c;
    c.kind = kind; c.T = Tn; c.N = Nn; c.F = Fn; c.block = tt; c.reps = reps;
    size_t fb = 0, total2 = 0;
    // Health: a failed cudaMemGetInfo leaves fb=0, which the report must reject.
    if (cudaMemGetInfo(&fb, &total2) != cudaSuccess) fb = 0;
    c.fb = fb;
    std::atomic<size_t> min_free{fb};
    std::atomic<int> sample_count{0};
    std::atomic<bool> stop{false};
    std::thread sampler(FblkRunSampler, &min_free, &sample_count, &stop, dev);
    // Handshake: wait for at least one successful sample before running.
    for (int i = 0; i < 2000 && sample_count.load() == 0; ++i)
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    int rc = 0;
    for (int i = 0; i < reps; ++i) {
      std::vector<double> out(static_cast<size_t>(Fn) * Fn);
      std::vector<uint8_t> trig(static_cast<size_t>(Fn) * (Fn + 1) / 2);
      if (std::strcmp(kind, "stream") == 0)
        rc = factor_corr_gpu_stream(Fp.data(), msk.empty() ? nullptr : msk.data(),
                                    Tn, Nn, Fn, chunks, tt, out.data(), trig.data());
      else
        rc = factor_corr_gpu(Fp.data(), msk.empty() ? nullptr : msk.data(),
                             Tn, Nn, Fn, out.data());
      if (rc != 0) break;
    }
    stop.store(true);
    sampler.join();
    cudaDeviceSynchronize();
    c.rc = rc;
    size_t fa = 0;
    cudaMemGetInfo(&fa, &total);
    c.samples = sample_count.load();
    c.driver_peak = (fb == 0) ? 0 : fb - min_free.load();
    return c;
  };

  // ---- F=12 cross-check anchors + GUARD (streaming vs production bitwise). --
  printf("== hwm-stream: F=12 cross-check anchors (tt=%d) ==\n", max_tt);
  for (int Nn : {5000, 10000}) {
    const int Fn = 12;
    const int64_t Rn = static_cast<int64_t>(T) * Nn;
    std::vector<double> F3x(static_cast<size_t>(Rn) * Fn);
    std::vector<uint8_t> maskx(static_cast<size_t>(Rn));
    {
      Lcg rng(0xF12C0DEu);
      for (int64_t i = 0; i < Rn * Fn; ++i)
        F3x[static_cast<size_t>(i)] = rng.uniform() * 4.0 - 2.0;
      // F-10 (external MINOR): col0 big-bias (1e12) so the Kahan re-run branch
      // is EXERCISED on this panel -- a guard panel that never triggers Kahan
      // could not catch a drift that silently kills the decision-A full-R Kahan
      // path. (HWM measures allocation peaks, so the bias does not affect the
      // measurement.)
      for (int64_t r = 0; r < Rn; ++r)
        F3x[static_cast<size_t>(r) * Fn] = 1e12 + rng.uniform() * 20.0 - 10.0;
      for (int64_t i = 0; i < Rn; ++i)
        maskx[static_cast<size_t>(i)] = (rng.next() % 100) < 98 ? 1 : 0;
    }
    FblkHwmCase c = measure("stream", T, Nn, Fn, mk_chunks32(), max_tt, F3x, maskx, /*reps=*/2);
    PrintFblkCase(c);
    // bitwise guard on the cheap F=12 N=5000 panel (1.2 KB output) -- proves the
    // measured implementation did not drift. Load-bearing: a drift must fail the
    // run (GUARD| parsed by the report; run_hwm_stream returns nonzero).
    if (Nn == 5000) {
      std::vector<double> full(static_cast<size_t>(Fn) * Fn);
      std::vector<double> strm(static_cast<size_t>(Fn) * Fn);
      std::vector<uint8_t> ft(static_cast<size_t>(Fn) * (Fn + 1) / 2);
      std::vector<uint8_t> bt(static_cast<size_t>(Fn) * (Fn + 1) / 2);
      int r1 = factor_corr_gpu(F3x.data(), maskx.data(), T, Nn, Fn, full.data(),
                               nullptr, ft.data());
      int r2 = factor_corr_gpu_stream(F3x.data(), maskx.data(), T, Nn, Fn,
                                      mk_chunks32(), max_tt, strm.data(), bt.data());
      bool eq = (r1 == 0 && r2 == 0) && vec_bitwise_eq(full, strm) &&
                vec_bitwise_eq_u8(ft, bt);
      // F-10: assert the Kahan branch actually fired (K>0) on the bias panel --
      // a zero-K guard would not exercise the decision-A full-R Kahan re-run.
      int k_guard = 0;
      for (uint8_t v : bt) if (v != 0u) ++k_guard;
      if (k_guard == 0) eq = false;
      printf("GUARD|pass=%d|r1=%d|r2=%d\n", eq ? 1 : 0, r1, r2);
      printf("  [%s] F=12 stream vs production bitwise+trigger (r1=%d r2=%d K=%d)\n",
             eq ? "PASS" : "FAIL", r1, r2, k_guard);
      if (!eq) guard_fail = 1;
    }
  }

  // ---- F=128 streaming: the key case. Current 12.6 GiB (d_F+d_Xt+d_valid
  // overlap) -> streaming ~6.9 GiB (d_Xt+d_valid+d_pp+d_F_chunk). reps=1 is
  // sufficient (the buffer set is held for the whole execution; the background
  // sampler has a wide window).
  printf("== hwm-stream: F=%d N=%d streaming (tt=%d, chunks 37x32+34) ==\n", F, N, max_tt);
  {
    FblkHwmCase c = measure("stream", T, N, F, mk_chunks32(), max_tt, F3, mask, /*reps=*/1);
    PrintFblkCase(c);
  }

  // ---- production (non-streamed) F=128 control: expected vram exhaustion / OOM.
  printf("== hwm-stream: production (non-streamed) F=%d control ==\n", F);
  {
    FblkHwmCase c = measure("production", T, N, F, mk_chunks32(), /*tt=*/0, F3, mask, /*reps=*/1);
    PrintFblkCase(c);
  }

  // Context health (driver must not be poisoned by the exhausted case).
  cudaError_t err = cudaGetLastError();
  printf("  post-run cudaGetLastError: %s\n", cudaGetErrorString(err));
  printf("== summary: adjudication in benchmarks/factor_stream_hwm_v1.py ==\n");
  if (guard_fail) {
    printf("GUARD FAIL: stream drift detected; run failing\n");
    return 2;
  }
  return 0;
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  int dev = 0;
  cudaError_t err = cudaGetDevice(&dev);
  if (err != cudaSuccess) { printf("cudaGetDevice FAIL\n"); return 1; }
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, dev);
  printf("GPU: %s (cc %d.%d), SM %d\n", prop.name, prop.major, prop.minor,
         prop.multiProcessorCount);
  if (argc > 1 && std::strcmp(argv[1], "--hwm-f128") == 0) {
    return run_hwm_f128(dev);
  }
  if (argc > 1 && std::strcmp(argv[1], "--hwm-stream") == 0) {
    return run_hwm_stream(dev);
  }
  printf("== factor_corr v0 selfcheck ==\n");

  // ---- error-path smoke: guards must reject invalid inputs ------------------
  {
    double dummy_in[4] = {1.0, 2.0, 3.0, 4.0};
    double dummy_out[2];
    int rc_null = factor_corr_gpu(nullptr, nullptr, 1, 2, 2, dummy_out);
    int rc_t0 = factor_corr_gpu(dummy_in, nullptr, 0, 2, 2, dummy_out);
    int rc_f0 = factor_corr_gpu(dummy_in, nullptr, 1, 2, 0, dummy_out);
    std::vector<double> big_in(4 * 129, 1.0);
    std::vector<double> big_out(static_cast<size_t>(129) * 129);
    int rc_fcap = factor_corr_gpu(big_in.data(), nullptr, 2, 2, 129, big_out.data());
    printf("error-path smoke: null=%d T0=%d F0=%d Fcap=%d (all expect nonzero)\n",
           rc_null, rc_t0, rc_f0, rc_fcap);
    if (rc_null == 0 || rc_t0 == 0 || rc_f0 == 0 || rc_fcap == 0) {
      printf("  [FAIL] error-path smoke: guards did not reject invalid input\n");
      ++g_fail;
    } else {
      printf("  [PASS] error-path smoke (null/dim/cap guards)\n");
    }
  }

  // ---- corpus anchors (F=2 panels: col0=a, col1=b) ---------------------------
  printf("== corpus anchors ==\n");
  {
    for (int ai = 0; ai < kCorrAnchorCount; ++ai) {
      const CorrAnchor& an = kCorrAnchors[ai];
      const int n = an.n;
      const int T = 1, N = n, F = 2;
      std::vector<double> F3(static_cast<size_t>(T) * N * F);
      std::vector<uint8_t> mask(static_cast<size_t>(T) * N, 1);
      for (int r = 0; r < n; ++r) {
        F3[static_cast<size_t>(r) * F + 0] = an.a[static_cast<size_t>(r)];
        F3[static_cast<size_t>(r) * F + 1] = an.b[static_cast<size_t>(r)];
        if (!an.ma.empty()) mask[static_cast<size_t>(r)] = an.ma[static_cast<size_t>(r)] ? 1 : 0;
        // mb is the same mask in corpus (pairwise both sides), fold into mask
        if (!an.mb.empty() && !an.ma.empty())
          mask[static_cast<size_t>(r)] =
              (an.ma[static_cast<size_t>(r)] && an.mb[static_cast<size_t>(r)]) ? 1 : 0;
      }
      std::vector<double> gpu(static_cast<size_t>(F) * F);
      std::vector<double> cpu(static_cast<size_t>(F) * F);
      int rc = factor_corr_gpu(F3.data(), mask.data(), T, N, F, gpu.data());
      cpu_factor_corr(F3.data(), mask.data(), T, N, F, cpu.data());
      if (rc != 0) {
        printf("  [FAIL] %s (rc=%d)\n", an.id, rc);
        ++g_fail;
        continue;
      }
      // Hard criterion: GPU must match the CPU reference implementing the SAME
      // algorithm (normal two-pass + trigger + Kahan re-run) to 1e-12. This
      // proves the kernel faithfully implements its intended numerics.
      bool ok = corr_match(gpu[1], cpu[1]);  // entry (0,1)
      // Informational: match against the manifest expected (numpy oracle). For
      // 1e15-bias inputs (bias_1e15, f64_ulp_bias) numpy's own mean is
      // reduction-order sensitive (oracle wrapper clause, CLAUDE.md Sec 2
      // uncentered_gram edge cases) -- GPU == CPU-reference is the guarantee,
      // numpy exp may differ; that difference is a known, allowed divergence.
      double exp = an.expected;
      bool exp_ok = an.expected_is_nan ? std::isnan(gpu[1]) : (std::isnan(gpu[1]) ? false : std::abs(gpu[1] - exp) <= 1e-12);
      bool sensitive = (std::string(an.branch) == "kahan" && !exp_ok);
      printf("  [%s] %s branch=%s (gpu=%.12g cpu=%.12g exp=%.12g%s)\n",
             ok ? "PASS" : "FAIL", an.id, an.branch,
             gpu[1], cpu[1], an.expected_is_nan ? nan_payload() : exp,
             sensitive ? " [exp-red-order-sensitive]" : "");
      if (!ok) {
        printf("      detail: gpu[0,1]=%.17g cpu[0,1]=%.17g (algorithm mismatch)\n",
               gpu[1], cpu[1]);
        ++g_fail;
      } else if (!exp_ok && !sensitive) {
        printf("      detail: gpu[0,1]=%.17g exp=%.17g (numpy mismatch, not flagged)\n",
               gpu[1], exp);
        ++g_fail;
      } else if (!exp_ok) {
        printf("      note: gpu=%.12g exp=%.12g (numpy reduction-order-sensitive, allowed)\n",
               gpu[1], exp);
      }
    }
  }

  // ---- review F1 diagonal regressions: var-underflow diagonal -> NaN ---------
  // The frozen oracle (np.corrcoef) returns NaN when a column's centered
  // variance underflows to zero (e.g. tiny-adjacent 1e-150 values). A blind
  // count>=2 && min!=max -> 1.0 would wrongly force finite; the diagonal must
  // follow the computed self-correlation. Assert golden NaN/1.0 on BOTH the
  // GPU kernel and the CPU reference.
  printf("== F1 diagonal regressions ==\n");
  {
    auto check = [](const char* name, const std::vector<double>& F3,
                    const std::vector<uint8_t>& mask, int T, int N, int F,
                    const std::vector<bool>& expect_nan) {
      std::vector<double> gpu(static_cast<size_t>(F) * F);
      std::vector<double> cpu(static_cast<size_t>(F) * F);
      int rc = factor_corr_gpu(F3.data(), mask.empty() ? nullptr : mask.data(),
                               T, N, F, gpu.data());
      cpu_factor_corr(F3.data(), mask.empty() ? nullptr : mask.data(),
                      T, N, F, cpu.data());
      bool ok = (rc == 0);
      for (int i = 0; i < F && ok; ++i) {
        double g = gpu[static_cast<size_t>(i) * F + i];
        double c = cpu[static_cast<size_t>(i) * F + i];
        if (std::isnan(g) != expect_nan[static_cast<size_t>(i)] ||
            std::isnan(c) != expect_nan[static_cast<size_t>(i)]) ok = false;
      }
      printf("  [%s] %s (T=%d N=%d F=%d)\n", ok ? "PASS" : "FAIL", name, T, N, F);
      if (!ok) ++g_fail;
    };
    // tiny-adjacent column: [1e-150, nextafter(1e-150, +inf)] -- centered
    // squares underflow to zero -> diag NaN (NOT 1.0 from min!=max).
    {
      std::vector<double> F3(2);  // T=1 N=2 F=1
      F3[0] = 1e-150;
      F3[1] = std::nextafter(1e-150, 1e150);
      check("tiny_adjacent diag", F3, {}, 1, 2, 1, {true});
    }
    // non-degenerate column -> diag 1.0 (control).
    {
      std::vector<double> F3(3);  // T=1 N=3 F=1
      F3[0] = 1.0; F3[1] = 2.0; F3[2] = 3.0;
      check("non-degenerate diag", F3, {}, 1, 3, 1, {false});
    }
    // exact-constant column -> diag NaN.
    {
      std::vector<double> F3(3);  // T=1 N=3 F=1
      F3[0] = 5.0; F3[1] = 5.0; F3[2] = 5.0;
      check("constant diag", F3, {}, 1, 3, 1, {true});
    }
    // count<2 (N=1 single row) -> diag NaN; exercises the N=1 path.
    {
      std::vector<double> F3(1);  // T=1 N=1 F=1
      F3[0] = 1.0;
      check("count<2 (N=1) diag", F3, {}, 1, 1, 1, {true});
    }
    // N=2 two factors: tiny-adjacent col0 (diag NaN), healthy col1 (diag 1.0);
    // one bad column must not contaminate the other.
    {
      std::vector<double> F3(4);  // T=1 N=2 F=2, row-major (row, factor)
      F3[0] = 1e-150; F3[1] = 1.0;  // row0: col0 tiny, col1 = 1
      F3[2] = std::nextafter(1e-150, 1e150); F3[3] = 2.0;  // row1
      check("two-factor mixed diag", F3, {}, 1, 2, 2, {true, false});
    }
  }

  // ---- randomized panels -----------------------------------------------------
  printf("== randomized checks ==\n");
  {
    const int kSizes[][3] = {{1, 5, 2}, {1, 32, 3}, {3, 7, 2}, {5, 100, 4},
                             {8, 128, 2}, {20, 300, 3}};
    for (int run = 0; run < 3; ++run) {
      for (const auto& sz : kSizes) {
        int T = sz[0], N = sz[1], F = sz[2];
        Lcg rng(0xCAFEu + static_cast<uint32_t>(run) * 7919u +
                static_cast<uint32_t>(T) * 131u + static_cast<uint32_t>(N) +
                static_cast<uint32_t>(F) * 17u);
        std::vector<double> F3(static_cast<size_t>(T) * N * F);
        std::vector<uint8_t> M(static_cast<size_t>(T) * N);
        for (int mask_mode = 0; mask_mode < 2; ++mask_mode) {
          for (int special = 0; special < 4; ++special) {
            random_panel(&rng, F3, M, T, N, F, mask_mode != 0, special);
            char nm[128];
            std::snprintf(nm, sizeof(nm), "rand T=%d N=%d F=%d mask=%d sp=%d run=%d",
                          T, N, F, mask_mode, special, run);
            check_matrix(nm, F3.data(), mask_mode ? M.data() : nullptr, T, N, F);
          }
        }
      }
    }
  }

  // ---- determinism: same input twice -> bitwise identical --------------------
  printf("== determinism ==\n");
  {
    Lcg rng(0xDEEF0u);
    const int T = 4, N = 64, F = 3;
    std::vector<double> F3(static_cast<size_t>(T) * N * F);
    for (double& v : F3) v = rng.uniform() * 20.0 - 10.0;
    std::vector<double> o1(static_cast<size_t>(F) * F), o2(static_cast<size_t>(F) * F);
    int rc1 = factor_corr_gpu(F3.data(), nullptr, T, N, F, o1.data());
    int rc2 = factor_corr_gpu(F3.data(), nullptr, T, N, F, o2.data());
    bool same = (rc1 == 0 && rc2 == 0) &&
                (std::memcmp(o1.data(), o2.data(), o1.size() * sizeof(double)) == 0);
    printf("  [%s] two runs bitwise identical (rc1=%d rc2=%d)\n",
           same ? "PASS" : "FAIL", rc1, rc2);
    if (!same) ++g_fail;
  }

  // ---- F/T chunking minimal proof 2: factor_corr continuation demo ----------
  // Chunked continuation (production kernels + d_pp state-carry) must equal the
  // non-chunked production pipeline bitwise on the full (F,F) corr matrix AND
  // the per-pair Kahan trigger bitset. Closes the corr "可分块 (continuation
  // correctness)" criterion (design spec Sec 4 #5; decision v2). Negative
  // controls prove the bitwise assertion is load-bearing.
  printf("== F/T chunking minimal proof 2 (factor_corr chunked vs non-chunked) ==\n");
  {
    auto mk_main_chunks = []() {
      std::vector<int> c;
      for (int i = 0; i < 37; ++i) c.push_back(32);  // 37x32 rows = 1184
      c.push_back(34);                                // + 34 = 1218
      return c;
    };

    // main-shape: T=1218 N=5000 F=12, col0 big-bias 1e12 (Kahan-triggers every
    // pair touching it), col1 normal, cols 2+ mixed random with NaN/inf/ties.
    // chunks [32x37, 34]: every non-final chunk is 32 rows * 5000 = 160000 =
    // 625 * 256 (block discipline); the tail 34 rows (170000) is unconstrained.
    {
      Lcg rng(0xFAB0C0DEu);
      const int T = 1218, N = 5000, F = 12;
      std::vector<double> F3(static_cast<size_t>(T) * N * F);
      std::vector<uint8_t> M(static_cast<size_t>(T) * N, 1);
      for (int r = 0; r < T * N; ++r) {
        for (int f = 0; f < F; ++f) {
          double v;
          if (f == 0) v = 1e12 + rng.uniform() * 20.0 - 10.0;
          else if (f == 1) v = rng.uniform() * 4.0 - 2.0;
          else {
            uint32_t rr = rng.next() % 100;
            if (rr < 55) v = rng.uniform() * 20.0 - 10.0;
            else if (rr < 65) v = 0.0;
            else if (rr < 73) v = nan_payload();
            else if (rr < 81) v = (f % 2 == 0) ? mk_f64(0x7ff0000000000000ull)
                                               : -mk_f64(0x7ff0000000000000ull);
            else v = static_cast<double>(rng.next() % 5);
          }
          F3[static_cast<size_t>(r) * F + f] = v;
        }
        M[static_cast<size_t>(r)] = (rng.next() % 100) < 90 ? 1 : 0;
      }
      for (int j = 0; j < N; ++j) M[static_cast<size_t>(j)] = 0;  // row 0 masked
      std::vector<int> chunks = mk_main_chunks();
      run_factor_chunk_case("main-shape", F3, M, T, N, F, chunks, true, false, 0, /*expected_K=*/12);
      // null-mask: exercises the chunked driver's nullptr-mask base-pointer path
      run_factor_chunk_case("main-shape-nullmask", F3, std::vector<uint8_t>(), T, N, F,
                            chunks, true, false, 0, /*expected_K=*/17);

      // F-blocking (pair-axis) minimal proof: F=12 split into block=4 (3 tiles)
      // and block=5 (non-uniform tiles [5,5,2]). Pair-independent blocking must
      // be bitwise identical to the non-blocked path (corr + trigger bitset).
      run_factor_fblock_case("fblock-main", F3, M, T, N, F, /*block_width=*/4,
                             /*check_cpu=*/false, /*expected_K=*/12);
      run_factor_fblock_case("fblock-main-b5", F3, M, T, N, F, /*block_width=*/5,
                             /*check_cpu=*/false, /*expected_K=*/12);
      run_factor_fblock_case("fblock-main-nullmask", F3, std::vector<uint8_t>(),
                             T, N, F, /*block_width=*/4, /*check_cpu=*/false,
                             /*expected_K=*/17);

      // F2: degenerate block widths -- block=1 (single-pair tiles) and
      // block=F (one diagonal tile, max_cols=2F over-allocate path).
      run_factor_fblock_case("fblock-b1", F3, M, T, N, F, /*block_width=*/1,
                             /*check_cpu=*/false, /*expected_K=*/12);
      run_factor_fblock_case("fblock-bF", F3, M, T, N, F, /*block_width=*/F,
                             /*check_cpu=*/false, /*expected_K=*/12);
    }

    // F1/F4: CPU-anchored small panel (independent correctness anchor) and
    // short-R (kb=1) branch. ASCII-only comments (nvcc/GBK discipline).
    {
      Lcg rng(0xFB0C0D1u);
      // F1: T=40 N=60 F=6, col0 bias 1e9 triggers Kahan (6 touching pairs),
      // other cols normal, mask ~8% off; block=2; check_cpu anchors full CPU.
      {
        const int T2 = 40, N2 = 60, F2 = 6;
        std::vector<double> S(static_cast<size_t>(T2) * N2 * F2);
        std::vector<uint8_t> M2(static_cast<size_t>(T2) * N2, 1);
        for (int r = 0; r < T2 * N2; ++r) {
          for (int f = 0; f < F2; ++f) {
            double v = (f == 0) ? (1e9 + rng.uniform() * 2.0 - 1.0)
                                : rng.uniform() * 4.0 - 2.0;
            S[static_cast<size_t>(r) * F2 + f] = v;
          }
          M2[static_cast<size_t>(r)] = (rng.next() % 100) < 92 ? 1 : 0;
        }
        run_factor_fblock_case("fblock-cpu-anchor", S, M2, T2, N2, F2,
                               /*block_width=*/2, /*check_cpu=*/true, /*expected_K=*/7);
      }
      // F4: T*N=240 < 256 -> kb=1 branch (col0 bias triggers 12 pairs)
      {
        const int T3 = 40, N3 = 6, F3s = 12;
        std::vector<double> S3(static_cast<size_t>(T3) * N3 * F3s);
        for (int r = 0; r < T3 * N3; ++r)
          for (int f = 0; f < F3s; ++f)
            S3[static_cast<size_t>(r) * F3s + f] =
                (f == 0) ? (1e6 + rng.uniform() - 0.5) : rng.uniform() * 2.0 - 1.0;
        run_factor_fblock_case("fblock-shortR", S3, std::vector<uint8_t>(), T3, N3, F3s,
                               /*block_width=*/4, /*check_cpu=*/true, /*expected_K=*/3);
      }
    }

    // K=0: low-bias finite-only panel -> no Kahan trigger; trigger set all-zero
    // in BOTH paths.
    {
      Lcg rng(0x0F0F0F01u);
      const int T = 300, N = 100, F = 4;
      std::vector<double> F3(static_cast<size_t>(T) * N * F);
      for (double& v : F3) v = rng.uniform() * 4.0 - 2.0;
      std::vector<int> chunks = {64, 64, 172};  // 64*100=6400=25*256 (non-final)
      run_factor_chunk_case("K=0-low-bias", F3, std::vector<uint8_t>(), T, N, F,
                            chunks, true, true, 0, /*expected_K=*/0);
    }

    // short-R boundaries: R=255 (single chunk; Kahan kb=1), R=256 (single
    // full-256 chunk), R=257 (256-multiple chunk + 1 tail). Continuation must
    // hold across the blockDim boundary and the tail.
    {
      Lcg rng(0x510A5EEDu);
      const int F = 3;
      for (int RR = 255; RR <= 257; ++RR) {
        std::vector<double> F3(static_cast<size_t>(RR) * F);
        for (size_t i = 0; i < F3.size(); ++i) F3[i] = rng.uniform() * 4.0 - 2.0;
        // col0 big-bias so a pair triggers Kahan at short R too
        for (int r = 0; r < RR; ++r) F3[static_cast<size_t>(r) * F] = 1e6 + rng.uniform() * 2.0 - 1.0;
        std::vector<int> chunks = (RR == 257) ? std::vector<int>{256, 1}
                                              : std::vector<int>{RR};
        char nm[64];
        std::snprintf(nm, sizeof(nm), "short-R=%d", RR);
        run_factor_chunk_case(nm, F3, std::vector<uint8_t>(), RR, 1, F, chunks, true, true, 0,
                              /*expected_K=*/(RR == 257) ? 0 : 1);
      }
    }

    // trigger 邻界: col0 bias_metric ~1.7e8 (>1e8, triggers), col1 ~8.7e7
    // (<1e8, no trigger), col2 normal, col3 all-NaN (non-finite trigger), col4
    // exact constant (NaN, no trigger). The trigger bitset must be byte-equal.
    {
      Lcg rng(0x7E6E4E4Du);
      const int T = 1218, N = 5000, F = 5;
      std::vector<double> F3(static_cast<size_t>(T) * N * F);
      for (int r = 0; r < T * N; ++r) {
        double a = rng.uniform() * 2.0 - 1.0;                 // [-1,1]
        F3[static_cast<size_t>(r) * F + 0] = 1e8 + a;          // bias ~ 1.73e8 > 1e8
        F3[static_cast<size_t>(r) * F + 1] = 5e7 + a;          // bias ~ 8.66e7 < 1e8
        F3[static_cast<size_t>(r) * F + 2] = rng.uniform() * 4.0 - 2.0;
        F3[static_cast<size_t>(r) * F + 3] = nan_payload();    // all NaN
        F3[static_cast<size_t>(r) * F + 4] = 5.0;              // exact constant
      }
      std::vector<int> chunks = mk_main_chunks();
      run_factor_chunk_case("trigger-boundaries", F3, std::vector<uint8_t>(), T, N, F,
                            chunks, true, false, 0, /*expected_K=*/5);
    }

    // adversarial input: mixed-magnitude columns (Kahan-chain / mean counter-
    // examples from the design spec: [.. 1e16 ..] cancellations + tiny values)
    // stress the continuation path; must stay bitwise equal to non-chunked.
    {
      Lcg rng(0xAD9EAD9Eu);
      const int T = 600, N = 500, F = 4;
      std::vector<double> F3(static_cast<size_t>(T) * N * F);
      const double chain[] = {-1.0, -1e8, 2.5, -1.0, 1.0, 1.0, -1e-5, 2.5, 1e16};
      for (int i = 0; i < T * N; ++i) {
        F3[static_cast<size_t>(i) * F + 0] = chain[i % 9];           // Kahan-chain values
        F3[static_cast<size_t>(i) * F + 1] = 1e12 + rng.uniform() * 2.0 - 1.0;  // big bias
        F3[static_cast<size_t>(i) * F + 2] = rng.uniform() * 4.0 - 2.0;
        F3[static_cast<size_t>(i) * F + 3] = (i % 7 == 0) ? -0.0 : rng.uniform() * 4.0 - 2.0;
      }
      std::vector<int> chunks = {128, 128, 128, 128, 88};  // 128*500=64000=250*256
      run_factor_chunk_case("adversarial-mixed-magnitude", F3, std::vector<uint8_t>(), T, N,
                            F, chunks, true, true, 0, /*expected_K=*/5);
    }

    // Negative control 1: fresh-start per chunk (reset_pp, NO continuation).
    // With >1 chunk this MUST produce a bitwise mismatch vs production --
    // proves the state-carry continuation is load-bearing, not vacuous
    // (design spec "independent-chunk + merge is a false-pass trap").
    {
      Lcg rng(0xDEADBEEFu);
      const int T = 300, N = 100, F = 4;
      std::vector<double> F3(static_cast<size_t>(T) * N * F);
      for (int i = 0; i < T * N; ++i) {
        F3[static_cast<size_t>(i) * F + 0] = 1e8 + rng.uniform() * 2.0 - 1.0;  // triggers
        for (int f = 1; f < F; ++f) F3[static_cast<size_t>(i) * F + f] = rng.uniform() * 4.0 - 2.0;
      }
      std::vector<int> chunks = {64, 64, 172};
      run_factor_chunk_case("neg-fresh-start", F3, std::vector<uint8_t>(), T, N, F,
                            chunks, /*expect_equal=*/false, false, /*reset_pp=*/1,
                            /*expected_K=*/-1);
    }

    // Negative control 2: a non-final chunk whose c*N is not a multiple of 256
    // (block discipline violation) must be REJECTED by the driver pre-check
    // (return -9) -- the discipline is enforced, not silently run.
    {
      Lcg rng(0xBADF00Du);
      const int T = 1218, N = 5000, F = 4;
      std::vector<double> F3(static_cast<size_t>(T) * N * F);
      for (double& v : F3) v = rng.uniform() * 4.0 - 2.0;
      std::vector<uint8_t> M(static_cast<size_t>(T) * N, 1);
      std::vector<double> out(static_cast<size_t>(F) * F);
      std::vector<int> bad = {32, 33, 1153};  // 33*5000 = 165000 not a multiple of 256
      int rc = factor_corr_gpu_chunked(F3.data(), M.data(), T, N, F, bad, out.data());
      bool ok = (rc == -9);
      printf("  [%s] neg-bad-chunk-reject non-final c*N not 256-multiple (rc=%d expect -9)\n",
             ok ? "PASS" : "FAIL", rc);
      if (!ok) ++g_fail;
    }
  }

  // ---- streaming (item 2): input/transpose streaming vs non-blocked ---------
  // Streamed upload + range transpose (d_F never fully resident) must equal the
  // production pipeline bitwise (corr + trigger) for ANY max_transpose_rows:
  // the range transpose is element-independent, so sub-chunk boundaries cannot
  // perturb the result. Continuation chunks are the same block-disciplined
  // boundaries as minimal proof 2 (non-final c*N multiple of 256). Negative
  // controls: a block-discipline violation (bad chunk, -9) and max_transpose_rows
  // < 1 (-10) must be rejected by the driver pre-check.
  printf("== streaming (item 2): input/transpose streaming vs non-blocked ==\n");
  {
    auto mk_chunks32 = []() {
      std::vector<int> c;
      for (int i = 0; i < 37; ++i) c.push_back(32);  // 37x32 = 1184
      c.push_back(34);                                // + 34 = 1218
      return c;
    };

    // main-shape: T=1218 N=5000 F=12, col0 big-bias 1e12 (Kahan-triggers every
    // pair touching it -> expected_K=12; NaN/inf cells are marked invalid at
    // transpose, so they cannot add triggers). max_transpose_rows spans a small
    // sub-chunk (4096), one sub-chunk per continuation chunk (160000 = 32*5000)
    // and a non-multiple of the 256 grid (5000).
    {
      Lcg rng(0xFAB0C0DEu);
      const int T = 1218, N = 5000, F = 12;
      std::vector<double> F3(static_cast<size_t>(T) * N * F);
      std::vector<uint8_t> M(static_cast<size_t>(T) * N, 1);
      for (int r = 0; r < T * N; ++r) {
        for (int f = 0; f < F; ++f) {
          double v;
          if (f == 0) v = 1e12 + rng.uniform() * 20.0 - 10.0;
          else if (f == 1) v = rng.uniform() * 4.0 - 2.0;
          else {
            uint32_t rr = rng.next() % 100;
            if (rr < 55) v = rng.uniform() * 20.0 - 10.0;
            else if (rr < 65) v = 0.0;
            else if (rr < 73) v = nan_payload();
            else if (rr < 81) v = (f % 2 == 0) ? mk_f64(0x7ff0000000000000ull)
                                               : -mk_f64(0x7ff0000000000000ull);
            else v = static_cast<double>(rng.next() % 5);
          }
          F3[static_cast<size_t>(r) * F + f] = v;
        }
        M[static_cast<size_t>(r)] = (rng.next() % 100) < 90 ? 1 : 0;
      }
      for (int j = 0; j < N; ++j) M[static_cast<size_t>(j)] = 0;  // row 0 masked
      std::vector<int> chunks = mk_chunks32();
      run_factor_stream_case("stream-main-tt4096", F3, M, T, N, F, chunks,
                             /*max_transpose_rows=*/4096, /*check_cpu=*/true,
                             /*expected_K=*/12);
      run_factor_stream_case("stream-main-tt160000", F3, M, T, N, F, chunks,
                             /*max_transpose_rows=*/160000, /*check_cpu=*/false,
                             /*expected_K=*/12);
      run_factor_stream_case("stream-main-tt5000", F3, M, T, N, F, chunks,
                             /*max_transpose_rows=*/5000, /*check_cpu=*/false,
                             /*expected_K=*/12);
      // null-mask: exercises the nullptr-mask base-pointer path. K=17 here (vs
      // 12 with the mask): with every row valid, the col0-bias pairs plus extra
      // floating-point diagonal self-corr triggers fire; same K as the chunked
      // null-mask case (expected_K must match the CHUNKED null-mask value, 17).
      run_factor_stream_case("stream-main-nullmask", F3, std::vector<uint8_t>(), T, N, F,
                             chunks, /*max_transpose_rows=*/4096, /*check_cpu=*/false,
                             /*expected_K=*/17);
    }

    // F=128 small panel: bitwise correctness of the HWM scenario (F cap 128,
    // R=512, chunks [32,32] with 32*8=256 non-final). col0 bias 1e12 fires the
    // Kahan branch on the 128 col0-touching pairs PLUS a data-dependent set of
    // floating-point diagonal self-corr triggers (observed K=166). expected_K=-1
    // (the corr/trigger bitwise equality and the CPU anchor are the load-bearing
    // assertions here; pinning a data-specific K is brittle).
    {
      Lcg rng(0x5F128001u);
      const int T = 64, N = 8, F = 128;
      std::vector<double> F3(static_cast<size_t>(T) * N * F);
      for (int r = 0; r < T * N; ++r) {
        F3[static_cast<size_t>(r) * F + 0] = 1e12 + rng.uniform() * 20.0 - 10.0;
        for (int f = 1; f < F; ++f)
          F3[static_cast<size_t>(r) * F + f] = rng.uniform() * 4.0 - 2.0;
      }
      std::vector<int> chunks = {32, 32};
      run_factor_stream_case("stream-F128", F3, std::vector<uint8_t>(), T, N, F,
                             chunks, /*max_transpose_rows=*/4096, /*check_cpu=*/true,
                             /*expected_K=*/-1);
    }

    // short-R: R=255 -> Kahan kb=1 branch; single chunk (no block discipline).
    // col0 1e6 -> diagonal self-corr is 1+1ulp > 1, triggering |r|>1 on the
    // diagonal pair only (expected_K=1; verified against production).
    {
      Lcg rng(0x510A5EEDu);
      const int T = 255, N = 1, F = 3;
      std::vector<double> F3(static_cast<size_t>(T) * N * F);
      for (int i = 0; i < T * N * F; ++i)
        F3[static_cast<size_t>(i)] = rng.uniform() * 4.0 - 2.0;
      for (int r = 0; r < T; ++r)
        F3[static_cast<size_t>(r) * F] = 1e6 + rng.uniform() * 2.0 - 1.0;
      std::vector<int> chunks = {255};
      run_factor_stream_case("stream-shortR", F3, std::vector<uint8_t>(), T, N, F,
                             chunks, /*max_transpose_rows=*/100, /*check_cpu=*/true,
                             /*expected_K=*/1);
    }

    // negative controls (stream-3, internal review): every contract-error code
    // must be exercised, not just -9/-10. -9 (block discipline) and -10 (tt<1)
    // were the original pair; add -5 (empty chunks), -6 (prefix != T), -7 (c<1),
    // -8 (prefix > T), -2 (T*N > INT32_MAX) and -3 (F > kMaxF) -- all validated
    // BEFORE any allocation, so dummy inputs suffice.
    {
      Lcg rng(0xBADF00Du);
      const int T = 1218, N = 5000, F = 4;
      std::vector<double> F3(static_cast<size_t>(T) * N * F);
      for (double& v : F3) v = rng.uniform() * 4.0 - 2.0;
      std::vector<double> out(static_cast<size_t>(F) * F);
      auto expect = [&](const char* name, int rc, int want) {
        bool ok = (rc == want);
        printf("  [%s] stream-neg-%s (rc=%d expect %d)\n",
               ok ? "PASS" : "FAIL", name, rc, want);
        if (!ok) ++g_fail;
      };
      std::vector<int> bad = {32, 33, 1153};  // 33*5000 = 165000 not 256-multiple
      expect("bad-chunk", factor_corr_gpu_stream(F3.data(), nullptr, T, N, F, bad,
                                                 4096, out.data()), -9);
      expect("tt0", factor_corr_gpu_stream(F3.data(), nullptr, T, N, F,
                                           std::vector<int>{1218}, 0, out.data()), -10);
      expect("empty-chunks", factor_corr_gpu_stream(F3.data(), nullptr, T, N, F,
                                                    std::vector<int>{}, 4096, out.data()), -5);
      expect("prefix-lt-T", factor_corr_gpu_stream(F3.data(), nullptr, T, N, F,
                                                   std::vector<int>{100}, 4096, out.data()), -6);
      expect("c0", factor_corr_gpu_stream(F3.data(), nullptr, T, N, F,
                                          std::vector<int>{0, 1218}, 4096, out.data()), -7);
      expect("prefix-gt-T", factor_corr_gpu_stream(F3.data(), nullptr, T, N, F,
                                                   std::vector<int>{2000}, 4096, out.data()), -8);
      // -2: T*N = 2.5e9 > INT32_MAX; rejected before allocation (dummy input).
      double dummy = 1.0;
      expect("t-ovf", factor_corr_gpu_stream(&dummy, nullptr, 50000, 50000, 4,
                                             std::vector<int>{1}, 1, out.data()), -2);
      // -11 (F-05): R within (INT32_MAX-65536, INT32_MAX] -- the int-r loops
      // could overflow; rejected before allocation. 46341*46340 = 2147439540
      // (> INT32_MAX-65536, <= INT32_MAX).
      expect("r-int32-ceiling", factor_corr_gpu_stream(&dummy, nullptr, 46341, 46340, 4,
                                                       std::vector<int>{1}, 1, out.data()), -11);
      // -3: F = 129 > kMaxF (128).
      expect("f-cap", factor_corr_gpu_stream(&dummy, nullptr, 1, 1, 129,
                                             std::vector<int>{1}, 1, out.data()), -3);
    }
  }

  printf("== summary ==\n");
  if (g_fail == 0) {
    printf("ALL PASS (factor_corr v0 1e-12 vs CPU reference + corpus anchors)\n");
    return 0;
  }
  printf("FAILURES: %d\n", g_fail);
  return 1;
}
