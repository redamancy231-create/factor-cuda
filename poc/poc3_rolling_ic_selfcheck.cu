// factor-cuda -- PoC 3 rolling_ic v0 selfcheck.
//
// Verifies src/rolling_ic.cu against an in-process CPU reference equivalent to
// benchmarks/backends.py np_rolling_ic: two-sided isfinite&mask intersection,
// min_valid guard, ptp==0 constant guard, stable ordinal ranks, two-pass
// centered Pearson (float64). IC tolerance 1e-12; NaN position match.
// Covers parity anchors (ic_tie/ic_nan/ic_inf/ic_valid_29_30_31/ic_all_invalid)
// plus randomized panels with NaN / +-inf / +-0 / ties / masks / constant rows.
//
// ASCII-only comments (nvcc/GBK pitfall).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <utility>
#include <vector>
#include <cuda_runtime.h>
#include <cub/device/device_segmented_radix_sort.cuh>
#include "rolling_ic.cuh"
#include "rolling_ic_impl.cuh"

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

// two-pass centered Pearson; sequential-division form (sxy/sqrt(sxx))/sqrt(syy)
// matching backends._two_pass_corr / safe_pearson (the 2026-08-05
// product-denominator fix, commit ba5df9f). No intermediate sxx*syy product so
// positive-subnormal variance is not misjudged as zero.
double two_pass_corr(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() < 2) return nan_payload();
  double am = 0.0, bm = 0.0;
  for (double x : a) am += x;
  for (double x : b) bm += x;
  am /= static_cast<double>(a.size());
  bm /= static_cast<double>(b.size());
  double sxx = 0.0, syy = 0.0, sxy = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    double dx = a[i] - am, dy = b[i] - bm;
    sxx += dx * dx;
    syy += dy * dy;
    sxy += dx * dy;
  }
  if (sxx > 0.0 && syy > 0.0) return (sxy / std::sqrt(sxx)) / std::sqrt(syy);
  return nan_payload();
}

// stable ordinal ranks 1..m (ties by original index), np argsort(kind=stable).
std::vector<double> ordinal_rank_1d(const std::vector<double>& vals) {
  std::vector<std::pair<double, int>> items(vals.size());
  for (size_t i = 0; i < vals.size(); ++i) items[i] = {vals[i], static_cast<int>(i)};
  std::stable_sort(items.begin(), items.end(),
                   [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                     return a.first < b.first;
                   });
  std::vector<double> r(vals.size());
  for (size_t k = 0; k < vals.size(); ++k) r[items[k].second] = static_cast<double>(k + 1);
  return r;
}

// CPU reference equivalent to np_rolling_ic.
void cpu_rolling_ic(const double* F, const double* R, const uint8_t* fmask,
                    const uint8_t* rmask, int T, int N, int min_valid, double* out) {
  for (int t = 0; t < T; ++t) {
    const double* f = F + static_cast<size_t>(t) * N;
    const double* r = R + static_cast<size_t>(t) * N;
    std::vector<double> fv, rv;
    for (int j = 0; j < N; ++j) {
      int g = t * N + j;
      bool ok = std::isfinite(f[j]) && std::isfinite(r[j]) &&
                (fmask == nullptr || fmask[g]) && (rmask == nullptr || rmask[g]);
      if (ok) { fv.push_back(f[j]); rv.push_back(r[j]); }
    }
    if (static_cast<int>(fv.size()) < min_valid) { out[t] = nan_payload(); continue; }
    // constant guard: ptp == 0 on the valid subset
    double fmin = fv[0], fmax = fv[0], rmin = rv[0], rmax = rv[0];
    for (double x : fv) { fmin = std::min(fmin, x); fmax = std::max(fmax, x); }
    for (double x : rv) { rmin = std::min(rmin, x); rmax = std::max(rmax, x); }
    if (fmin == fmax || rmin == rmax) { out[t] = nan_payload(); continue; }
    std::vector<double> ra = ordinal_rank_1d(fv);
    std::vector<double> rb = ordinal_rank_1d(rv);
    out[t] = two_pass_corr(ra, rb);
  }
}

int g_fail = 0;

bool ic_match(double gpu, double cpu) {
  if (std::isnan(gpu) && std::isnan(cpu)) return true;
  if (std::isnan(gpu) || std::isnan(cpu)) return false;
  return std::abs(gpu - cpu) <= 1e-12;
}

void report(const char* name, bool ok, double gpu, double cpu) {
  printf("  [%s] %s (gpu=%.12g cpu=%.12g)\n", ok ? "PASS" : "FAIL", name, gpu, cpu);
  if (!ok) ++g_fail;
}

// Run a single-row (or multi-row) check: GPU vs CPU reference.
void check_panel(const char* name, const double* F, const double* R, const uint8_t* fmask,
                 const uint8_t* rmask, int T, int N, int min_valid) {
  std::vector<double> gpu(static_cast<size_t>(T));
  std::vector<double> cpu(static_cast<size_t>(T));
  int rc = rolling_ic_gpu(F, R, fmask, rmask, T, N, min_valid, gpu.data());
  cpu_rolling_ic(F, R, fmask, rmask, T, N, min_valid, cpu.data());
  char detail[160];
  if (rc != 0) {
    std::snprintf(detail, sizeof(detail), "rc=%d", rc);
    printf("  [FAIL] %s (%s)\n", name, detail);
    ++g_fail;
    return;
  }
  bool ok = true;
  for (int t = 0; t < T; ++t) {
    if (!ic_match(gpu[static_cast<size_t>(t)], cpu[static_cast<size_t>(t)])) { ok = false; break; }
  }
  printf("  [%s] %s (T=%d N=%d min_valid=%d)\n", ok ? "PASS" : "FAIL", name, T, N, min_valid);
  if (!ok) {
    for (int t = 0; t < T; ++t)
      if (!ic_match(gpu[static_cast<size_t>(t)], cpu[static_cast<size_t>(t)]))
        printf("      mismatch row %d: gpu=%.12g cpu=%.12g\n", t, gpu[static_cast<size_t>(t)],
               cpu[static_cast<size_t>(t)]);
    ++g_fail;
  }
}

// Randomize a panel; may force constant rows / all-invalid rows / dense ties.
void random_panel(Lcg* rng, std::vector<double>& F, std::vector<double>& R,
                  std::vector<uint8_t>& fM, std::vector<uint8_t>& rM, int T, int N,
                  bool with_masks, bool force_constant_row, bool force_invalid_row) {
  for (int i = 0; i < T * N; ++i) {
    uint32_t rr = rng->next() % 100;
    double v, w;
    if (rr < 60) { v = rng->uniform() * 20.0 - 10.0; w = rng->uniform() * 20.0 - 10.0; }
    else if (rr < 70) { v = 0.0; w = 0.0; }  // +-0 folds
    else if (rr < 78) { v = nan_payload(); w = nan_payload(); }
    else if (rr < 86) { v = mk_f64(0x7ff0000000000000ull); w = -mk_f64(0x7ff0000000000000ull); }
    else { v = static_cast<double>(rng->next() % 5); w = static_cast<double>(rng->next() % 5); }  // ties
    F[i] = v; R[i] = w;
    fM[i] = ((rng->next() % 100) < 80) ? 1 : 0;
    rM[i] = ((rng->next() % 100) < 85) ? 1 : 0;
  }
  if (with_masks) {
    for (int j = 0; j < N; ++j) { fM[j] = 0; rM[j] = 0; }              // row 0 all masked
    if (T >= 2) for (int j = 0; j < N; ++j) { fM[N + j] = 1; rM[N + j] = 1; }
  }
  if (force_constant_row && T >= 2) {  // row 1: factor constant, returns varied
    for (int j = 0; j < N; ++j) { F[N + j] = 5.0; R[N + j] = static_cast<double>(j % 3); fM[N + j] = 1; rM[N + j] = 1; }
  }
  if (force_invalid_row && T >= 2) {  // row 1: all invalid (NaN both sides)
    for (int j = 0; j < N; ++j) { F[N + j] = nan_payload(); R[N + j] = nan_payload(); fM[N + j] = 1; rM[N + j] = 1; }
  }
}

// ---- F/T chunking minimal proof 1: chunked rolling_ic pipeline ---------------
//
// The chunked path re-launches the EXACT production kernels (declared in
// rolling_ic_impl.cuh) over T-row chunks, changing ONLY launch ranges, base
// pointers and CUB segment offsets:
//   * chunk-local compute buffers (capacity = max_chunk*N, NOT T*N),
//   * CUB segment offsets re-cut per chunk to 0-based [0,N,2N,...],
//   * input slices via base pointers into a single full-size upload.
// Kernel bodies are byte-identical to the non-chunked path. If every IC row
// equals the non-chunked result bitwise, the "可分块" criterion holds and the
// CUB segment re-cut black box is closed at device level (see
// reviews/ft_chunking_design_spec_workflow_2026-08-05.md Sec 4 #8).
constexpr int kFTBlock = 256;  // pearson blockDim pinned (matches production)

// Full bitwise comparison (uint64 bits incl. NaN payloads).
bool ic_bitwise(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    uint64_t ba, bb;
    std::memcpy(&ba, &a[i], sizeof(ba));
    std::memcpy(&bb, &b[i], sizeof(bb));
    if (ba != bb) return false;
  }
  return true;
}

// CUB segmented radix sort over a DoubleBuffer pair (chunk-local offsets).
template <typename KeyT>
cudaError_t ChunkSegSort(void* d_temp, size_t& temp_bytes, KeyT* keys_cur,
                         KeyT* keys_alt, uint32_t* vals_cur, uint32_t* vals_alt,
                         const int32_t* d_offsets, int64_t total, int64_t segs,
                         const uint32_t** out_values) {
  cub::DoubleBuffer<KeyT> dk(keys_cur, keys_alt);
  cub::DoubleBuffer<uint32_t> dv(vals_cur, vals_alt);
  cudaError_t e = cub::DeviceSegmentedRadixSort::SortPairs(
      d_temp, temp_bytes, dk, dv, total, segs, d_offsets, d_offsets + 1);
  if (e == cudaSuccess && out_values != nullptr) *out_values = dv.Current();
  return e;
}

// Chunked rolling_ic driver. `chunks` must be non-empty and sum to T (each c>=1
// and no prefix sum may exceed T). Returns 0 on success; prints per-chunk
// execution evidence (row range / segs / items / CUB temp bytes) so a reviewer
// can confirm the pipeline genuinely ran as a sequence of chunk-local re-cut
// launches, not as one full panel.
//   h_rank_f_out/h_rank_r_out: optional (T*N) outputs receiving the chunk-local
//              ranks, so the proof can assert the intermediate rank arrays are
//              bitwise identical to the non-chunked path (the final IC alone is
//              non-injective over rank pairs).
//   break_offsets_side: 0 = normal; 1/2 = use IN-BOUNDS but misaligned offsets
//              (segment boundaries shifted by one cell) for the factor/returns
//              CUB sort only -- the negative control that proves the re-cut is
//              load-bearing (a broken re-cut must produce a bitwise mismatch).
int rolling_ic_gpu_chunked(const double* h_F, const double* h_R,
                           const uint8_t* h_fmask, const uint8_t* h_rmask, int T,
                           int N, int min_valid, const std::vector<int>& chunks,
                           double* h_out, double* h_rank_f_out = nullptr,
                           double* h_rank_r_out = nullptr,
                           int break_offsets_side = 0) {
  if (h_F == nullptr || h_R == nullptr || h_out == nullptr || T < 1 || N < 1)
    return -1;
  if (min_valid < 2) return -4;
  if (static_cast<int64_t>(T) * N > INT32_MAX) return -3;
  if (chunks.empty()) return -5;
  // Pre-validate every chunk BEFORE any allocation (int64): each c >= 1, prefix
  // sum never exceeds T, and the total equals T. An int32 sum allows {T+1,-1}
  // to pass sum==T and read OOB in the first chunk before the c<1 branch.
  int64_t prefix = 0;
  for (int c : chunks) {
    if (c < 1) return -7;
    prefix += c;
    if (prefix > T) return -8;
  }
  if (prefix != T) return -6;

  const int max_chunk = *std::max_element(chunks.begin(), chunks.end());
  const int total = T * N;
  const size_t n_items = static_cast<size_t>(total);
  const size_t cap = static_cast<size_t>(max_chunk) * static_cast<size_t>(N);

  // Declared before the first `goto ft_fail` so the jumps don't bypass their
  // initialization (C++ rule). h_offsets holds the per-chunk re-cut 0-based
  // CUB segment offsets [0,N,2N,...] (chunk-local, not full-panel).
  std::vector<int32_t> h_offsets(static_cast<size_t>(max_chunk) + 1);
  for (int k = 0; k <= max_chunk; ++k)
    h_offsets[static_cast<size_t>(k)] = static_cast<int32_t>(static_cast<int64_t>(k) * N);
  int t0 = 0;

  cudaError_t err = cudaSuccess;
  const char* ft_stage = "setup";  // failure stage (diagnostics)

  double* d_F = nullptr;
  double* d_R = nullptr;
  uint8_t* d_fmask = nullptr;
  uint8_t* d_rmask = nullptr;
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
  uint32_t* d_vcount = nullptr;
  uint64_t* d_fkmin = nullptr;
  uint64_t* d_fkmax = nullptr;
  uint64_t* d_rkmin = nullptr;
  uint64_t* d_rkmax = nullptr;
  double* d_ic = nullptr;
  int32_t* d_offsets = nullptr;
  void* d_temp = nullptr;
  size_t temp_alloc = 0;

  ft_stage = "alloc";
  err = cudaMalloc(&d_F, n_items * sizeof(double));
  if (err == cudaSuccess) err = cudaMalloc(&d_R, n_items * sizeof(double));
  if (err == cudaSuccess && h_fmask != nullptr) err = cudaMalloc(&d_fmask, n_items);
  if (err == cudaSuccess && h_rmask != nullptr) err = cudaMalloc(&d_rmask, n_items);
  if (err == cudaSuccess) err = cudaMalloc(&d_fkey_cur, cap * sizeof(uint64_t));
  if (err == cudaSuccess) err = cudaMalloc(&d_fkey_alt, cap * sizeof(uint64_t));
  if (err == cudaSuccess) err = cudaMalloc(&d_rkey_cur, cap * sizeof(uint64_t));
  if (err == cudaSuccess) err = cudaMalloc(&d_rkey_alt, cap * sizeof(uint64_t));
  if (err == cudaSuccess) err = cudaMalloc(&d_fvals_cur, cap * sizeof(uint32_t));
  if (err == cudaSuccess) err = cudaMalloc(&d_fvals_alt, cap * sizeof(uint32_t));
  if (err == cudaSuccess) err = cudaMalloc(&d_rvals_cur, cap * sizeof(uint32_t));
  if (err == cudaSuccess) err = cudaMalloc(&d_rvals_alt, cap * sizeof(uint32_t));
  if (err == cudaSuccess) err = cudaMalloc(&d_valid, cap);
  if (err == cudaSuccess) err = cudaMalloc(&d_rank_f, cap * sizeof(double));
  if (err == cudaSuccess) err = cudaMalloc(&d_rank_r, cap * sizeof(double));
  if (err == cudaSuccess) err = cudaMalloc(&d_vcount, static_cast<size_t>(max_chunk) * sizeof(uint32_t));
  if (err == cudaSuccess) err = cudaMalloc(&d_fkmin, static_cast<size_t>(max_chunk) * sizeof(uint64_t));
  if (err == cudaSuccess) err = cudaMalloc(&d_fkmax, static_cast<size_t>(max_chunk) * sizeof(uint64_t));
  if (err == cudaSuccess) err = cudaMalloc(&d_rkmin, static_cast<size_t>(max_chunk) * sizeof(uint64_t));
  if (err == cudaSuccess) err = cudaMalloc(&d_rkmax, static_cast<size_t>(max_chunk) * sizeof(uint64_t));
  if (err == cudaSuccess) err = cudaMalloc(&d_ic, static_cast<size_t>(max_chunk) * sizeof(double));
  if (err == cudaSuccess) err = cudaMalloc(&d_offsets, static_cast<size_t>(max_chunk + 1) * sizeof(int32_t));
  if (err != cudaSuccess) goto ft_fail;

  // single full-size upload; chunk rows are sliced via base pointers
  ft_stage = "upload";
  err = cudaMemcpy(d_F, h_F, n_items * sizeof(double), cudaMemcpyHostToDevice);
  if (err == cudaSuccess) err = cudaMemcpy(d_R, h_R, n_items * sizeof(double), cudaMemcpyHostToDevice);
  if (err == cudaSuccess && h_fmask != nullptr) err = cudaMemcpy(d_fmask, h_fmask, n_items, cudaMemcpyHostToDevice);
  if (err == cudaSuccess && h_rmask != nullptr) err = cudaMemcpy(d_rmask, h_rmask, n_items, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) goto ft_fail;

  for (size_t ci = 0; ci < chunks.size(); ++ci) {
    const int c = chunks[ci];
    const int64_t c_total64 = static_cast<int64_t>(c) * N;
    const int c_total = static_cast<int>(c_total64);

    // Broken offsets for the negative control (break_offsets_side): an
    // IN-BOUNDS but misaligned segmentation (each segment boundary shifted by
    // one cell) so the sort ranks within wrong segments -> wrong ranks, which
    // the test must catch as a bitwise mismatch. Keeps compute-sanitizer clean
    // (no OOB), unlike the t0-based "forgot to re-cut" global offsets which
    // would reference positions >= c*N in the chunk-local buffer.
    std::vector<int32_t> h_offs_broken(static_cast<size_t>(c) + 1);
    for (int k = 0; k < c; ++k)
      h_offs_broken[static_cast<size_t>(k)] =
          static_cast<int32_t>(static_cast<int64_t>(k) * N + 1);
    h_offs_broken[static_cast<size_t>(c)] =
        static_cast<int32_t>(static_cast<int64_t>(c) * N);
    auto upload_offsets = [&](const std::vector<int32_t>& offs) -> cudaError_t {
      return cudaMemcpy(d_offsets, offs.data(),
                        (static_cast<size_t>(c) + 1) * sizeof(int32_t),
                        cudaMemcpyHostToDevice);
    };

    // per-chunk per-row stats init (sentinel memset, mirror of non-chunked)
    ft_stage = "stats-init";
    err = cudaMemset(d_vcount, 0, static_cast<size_t>(c) * sizeof(uint32_t));
    if (err == cudaSuccess) err = cudaMemset(d_fkmin, 0xFF, static_cast<size_t>(c) * sizeof(uint64_t));
    if (err == cudaSuccess) err = cudaMemset(d_fkmax, 0, static_cast<size_t>(c) * sizeof(uint64_t));
    if (err == cudaSuccess) err = cudaMemset(d_rkmin, 0xFF, static_cast<size_t>(c) * sizeof(uint64_t));
    if (err == cudaSuccess) err = cudaMemset(d_rkmax, 0, static_cast<size_t>(c) * sizeof(uint64_t));
    if (err != cudaSuccess) goto ft_fail;

    // factor-side offsets (used by the CUB temp query and the factor sort)
    ft_stage = "offsets";
    err = upload_offsets(break_offsets_side == 1 ? h_offs_broken : h_offsets);
    if (err != cudaSuccess) goto ft_fail;

    // stage 1: preprocess on the chunk rows (input slice via base pointer)
    ft_stage = "preprocess";
    {
      const int block = 256;
      const int grid = static_cast<int>(1 + (c_total - 1) / block);
      preprocess_kernel<<<grid, block>>>(
          d_F + static_cast<int64_t>(t0) * N, d_R + static_cast<int64_t>(t0) * N,
          (h_fmask != nullptr) ? d_fmask + static_cast<int64_t>(t0) * N : nullptr,
          (h_rmask != nullptr) ? d_rmask + static_cast<int64_t>(t0) * N : nullptr,
          N, c_total, d_fkey_cur, d_rkey_cur, d_fvals_cur, d_rvals_cur, d_valid,
          d_vcount, d_fkmin, d_fkmax, d_rkmin, d_rkmax);
      err = cudaGetLastError();
      if (err != cudaSuccess) goto ft_fail;
    }

    // CUB temp: query for this chunk's config; grow if needed
    ft_stage = "temp-query";
    {
      size_t q = 0;
      err = ChunkSegSort<uint64_t>(nullptr, q, d_fkey_cur, d_fkey_alt, d_fvals_cur,
                                   d_fvals_alt, d_offsets, c_total64, c, nullptr);
      if (err != cudaSuccess) goto ft_fail;
      if (q > temp_alloc) {
        ft_stage = "temp-alloc";
        if (d_temp != nullptr) cudaFree(d_temp);
        d_temp = nullptr;
        if (q > 0) {
          err = cudaMalloc(&d_temp, q);
          if (err != cudaSuccess) goto ft_fail;
        }
        temp_alloc = q;
      }
    }

    // stage 2a: sort factor keys, scatter factor ranks (chunk-local)
    ft_stage = "fsort";
    {
      const uint32_t* f_vals = nullptr;
      err = ChunkSegSort<uint64_t>(d_temp, temp_alloc, d_fkey_cur, d_fkey_alt,
                                   d_fvals_cur, d_fvals_alt, d_offsets, c_total64, c,
                                   &f_vals);
      if (err != cudaSuccess) goto ft_fail;
      const int block = 256;
      const int grid = static_cast<int>(1 + (c_total - 1) / block);
      ft_stage = "fscatter";
      scatter_rank_double_kernel<<<grid, block>>>(f_vals, d_valid, N, c_total, d_rank_f);
      err = cudaGetLastError();
      if (err != cudaSuccess) goto ft_fail;
    }

    // returns-side offsets (re-upload: the negative control may break this side)
    ft_stage = "offsets-r";
    err = upload_offsets(break_offsets_side == 2 ? h_offs_broken : h_offsets);
    if (err != cudaSuccess) goto ft_fail;

    // stage 2b: sort return keys, scatter return ranks
    ft_stage = "rsort";
    {
      const uint32_t* r_vals = nullptr;
      err = ChunkSegSort<uint64_t>(d_temp, temp_alloc, d_rkey_cur, d_rkey_alt,
                                   d_rvals_cur, d_rvals_alt, d_offsets, c_total64, c,
                                   &r_vals);
      if (err != cudaSuccess) goto ft_fail;
      const int block = 256;
      const int grid = static_cast<int>(1 + (c_total - 1) / block);
      ft_stage = "rscatter";
      scatter_rank_double_kernel<<<grid, block>>>(r_vals, d_valid, N, c_total, d_rank_r);
      err = cudaGetLastError();
      if (err != cudaSuccess) goto ft_fail;
    }

    // optional per-chunk rank dump (bitwise comparison vs the non-chunked path)
    if (h_rank_f_out != nullptr && h_rank_r_out != nullptr) {
      ft_stage = "rank-d2h";
      err = cudaMemcpy(h_rank_f_out + static_cast<size_t>(t0) * N, d_rank_f,
                       static_cast<size_t>(c_total) * sizeof(double),
                       cudaMemcpyDeviceToHost);
      if (err == cudaSuccess)
        err = cudaMemcpy(h_rank_r_out + static_cast<size_t>(t0) * N, d_rank_r,
                         static_cast<size_t>(c_total) * sizeof(double),
                         cudaMemcpyDeviceToHost);
      if (err != cudaSuccess) goto ft_fail;
    }

    // stage 3: pearson per chunk row (blockDim pinned to 256)
    ft_stage = "pearson";
    {
      pearson_kernel<<<c, kFTBlock>>>(d_rank_f, d_rank_r, d_valid, d_vcount, d_fkmin,
                                      d_fkmax, d_rkmin, d_rkmax, N, min_valid, d_ic);
      err = cudaGetLastError();
      if (err != cudaSuccess) goto ft_fail;
    }

    // D2H this chunk's IC into h_out + t0 (c doubles, one per row -- NOT c*N).
    // This synchronous copy is the chunk's sync point: any async kernel error
    // from THIS chunk surfaces here, so a failure reported at "d2h" may
    // originate in an earlier kernel of the chunk (diagnostic only; the error
    // is always caught, never a silent wrong answer).
    ft_stage = "d2h";
    err = cudaMemcpy(h_out + t0, d_ic, static_cast<size_t>(c) * sizeof(double),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto ft_fail;

    printf("    chunk[%zu] rows[%d,%d) segs=%d items=%d cub_temp=%zuB\n", ci, t0,
           t0 + c, c, c_total, temp_alloc);
    t0 += c;
  }

  cudaFree(d_F); cudaFree(d_R); cudaFree(d_fmask); cudaFree(d_rmask);
  cudaFree(d_fkey_cur); cudaFree(d_fkey_alt); cudaFree(d_rkey_cur); cudaFree(d_rkey_alt);
  cudaFree(d_fvals_cur); cudaFree(d_fvals_alt); cudaFree(d_rvals_cur); cudaFree(d_rvals_alt);
  cudaFree(d_valid); cudaFree(d_rank_f); cudaFree(d_rank_r);
  cudaFree(d_vcount); cudaFree(d_fkmin); cudaFree(d_fkmax); cudaFree(d_rkmin); cudaFree(d_rkmax);
  cudaFree(d_ic); cudaFree(d_offsets); cudaFree(d_temp);
  return 0;

ft_fail:
  fprintf(stderr, "  chunked driver FAIL at stage=%s err=%d (%s)\n", ft_stage,
          static_cast<int>(err), cudaGetErrorString(err));
  cudaFree(d_F); cudaFree(d_R); cudaFree(d_fmask); cudaFree(d_rmask);
  cudaFree(d_fkey_cur); cudaFree(d_fkey_alt); cudaFree(d_rkey_cur); cudaFree(d_rkey_alt);
  cudaFree(d_fvals_cur); cudaFree(d_fvals_alt); cudaFree(d_rvals_cur); cudaFree(d_rvals_alt);
  cudaFree(d_valid); cudaFree(d_rank_f); cudaFree(d_rank_r);
  cudaFree(d_vcount); cudaFree(d_fkmin); cudaFree(d_fkmax); cudaFree(d_rkmin); cudaFree(d_rkmax);
  cudaFree(d_ic); cudaFree(d_offsets); cudaFree(d_temp);
  return static_cast<int>(err);
}

// Deterministic panel with ties / NaN / +-inf / +-0 / masks / forced fully-masked
// (row 0) / forced constant-factor (row 1) / forced all-NaN (row 2) sections.
void proof_panel(Lcg* rng, std::vector<double>& F, std::vector<double>& R,
                 std::vector<uint8_t>& fM, std::vector<uint8_t>& rM, int T, int N) {
  for (int i = 0; i < T * N; ++i) {
    uint32_t rr = rng->next() % 100;
    double v, w;
    if (rr < 55) { v = rng->uniform() * 20.0 - 10.0; w = rng->uniform() * 20.0 - 10.0; }
    else if (rr < 65) {  // +-0: some cells get -0.0 (canonical key folds both)
      v = (rng->next() % 3 == 0) ? -0.0 : 0.0;
      w = (rng->next() % 3 == 0) ? -0.0 : 0.0;
    }
    else if (rr < 72) { v = nan_payload(); w = nan_payload(); }
    else if (rr < 79) { v = mk_f64(0x7ff0000000000000ull); w = -mk_f64(0x7ff0000000000000ull); }
    else { v = static_cast<double>(rng->next() % 5); w = static_cast<double>(rng->next() % 5); }
    F[i] = v; R[i] = w;
    fM[i] = ((rng->next() % 100) < 82) ? 1 : 0;
    rM[i] = ((rng->next() % 100) < 86) ? 1 : 0;
  }
  // row 0: fully masked (IC -> NaN via min_valid)
  if (T >= 1) for (int j = 0; j < N; ++j) { fM[j] = 0; rM[j] = 0; F[j] = 1.0; R[j] = 1.0; }
  // row 1: factor constant on the valid set (IC -> NaN via constant guard)
  if (T >= 2) for (int j = 0; j < N; ++j) {
    F[N + j] = 5.0; R[N + j] = static_cast<double>(j % 7); fM[N + j] = 1; rM[N + j] = 1;
  }
  // row 2: all NaN both sides (IC -> NaN)
  if (T >= 3) for (int j = 0; j < N; ++j) {
    F[2 * N + j] = nan_payload(); R[2 * N + j] = nan_payload(); fM[2 * N + j] = 1; rM[2 * N + j] = 1;
  }
}

// Run one chunk-proof case: non-chunked full pipeline vs chunked pipeline,
// assert all T IC rows bitwise equal (uint64 bits, incl. NaN payloads).
// Optionally also check the chunked output against the CPU reference (1e-12).
void run_chunk_case(const char* name, int T, int N, int min_valid,
                    const std::vector<int>& chunks, uint64_t seed, bool check_cpu) {
  Lcg rng(seed);
  std::vector<double> F(static_cast<size_t>(T) * N), R(static_cast<size_t>(T) * N);
  std::vector<uint8_t> fM(static_cast<size_t>(T) * N), rM(static_cast<size_t>(T) * N);
  proof_panel(&rng, F, R, fM, rM, T, N);
  std::vector<double> full(static_cast<size_t>(T)), chunked(static_cast<size_t>(T));
  std::vector<double> ref_rf(static_cast<size_t>(T) * N), ref_rr(static_cast<size_t>(T) * N);
  std::vector<double> ch_rf(static_cast<size_t>(T) * N), ch_rr(static_cast<size_t>(T) * N);
  int rc1 = rolling_ic_gpu(F.data(), R.data(), fM.data(), rM.data(), T, N, min_valid,
                           full.data(), nullptr, ref_rf.data(), ref_rr.data());
  int rc2 = rolling_ic_gpu_chunked(F.data(), R.data(), fM.data(), rM.data(), T, N,
                                   min_valid, chunks, chunked.data(), ch_rf.data(),
                                   ch_rr.data(), 0);
  if (rc1 != 0 || rc2 != 0) {
    printf("  [FAIL] %s (rc1=%d rc2=%d)\n", name, rc1, rc2);
    ++g_fail;
    return;
  }
  bool ok = ic_bitwise(full, chunked);
  // Intermediate rank arrays must also be bitwise identical: the final IC is
  // non-injective over rank pairs, so IC equality alone cannot prove the CUB
  // segment re-cut / rank intermediate state is preserved (GPT-5.6-Sol F1).
  bool rank_ok = ic_bitwise(ref_rf, ch_rf) && ic_bitwise(ref_rr, ch_rr);
  int first_bad = -1;
  if (!ok) {
    for (size_t t = 0; t < static_cast<size_t>(T); ++t) {
      uint64_t bf, bc;
      std::memcpy(&bf, &full[t], 8);
      std::memcpy(&bc, &chunked[t], 8);
      if (bf != bc) { first_bad = static_cast<int>(t); break; }
    }
  }
  printf("  [%s] %s (T=%d N=%d min_valid=%d chunks=%zu) bitwise chunked==nonchunked (IC + ranks)\n",
         ok && rank_ok ? "PASS" : "FAIL", name, T, N, min_valid, chunks.size());
  if (!ok) {
    printf("      first bitwise mismatch row %d: full=%.17g chunked=%.17g\n", first_bad,
           full[static_cast<size_t>(first_bad)], chunked[static_cast<size_t>(first_bad)]);
    ++g_fail;
  }
  if (!rank_ok) {
    printf("      rank arrays bitwise mismatch (factor/returns ranks differ)\n");
    ++g_fail;
  }
  if (check_cpu && ok) {
    std::vector<double> cpu(static_cast<size_t>(T));
    cpu_rolling_ic(F.data(), R.data(), fM.data(), rM.data(), T, N, min_valid, cpu.data());
    bool cpu_ok = true;
    for (size_t t = 0; t < static_cast<size_t>(T); ++t)
      if (!ic_match(chunked[t], cpu[t])) { cpu_ok = false; break; }
    printf("  [%s] %s chunked vs CPU reference (1e-12, informational)\n",
           cpu_ok ? "PASS" : "FAIL", name);
    if (!cpu_ok) ++g_fail;
  }
}

}  // namespace

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  int dev = 0;
  cudaError_t err = cudaGetDevice(&dev);
  if (err != cudaSuccess) { printf("cudaGetDevice FAIL\n"); return 1; }
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, dev);
  printf("GPU: %s (cc %d.%d), SM %d\n", prop.name, prop.major, prop.minor,
         prop.multiProcessorCount);
  printf("== rolling_ic v0 selfcheck ==\n");

  // ---- F3 error-path smoke: guards must reject invalid inputs --------------
  {
    double dummy_in[4] = {1.0, 2.0, 3.0, 4.0};
    double dummy_out[2];
    int rc_mv = rolling_ic_gpu(dummy_in, dummy_in, nullptr, nullptr, 1, 2, 1, dummy_out);  // min_valid<2
    int rc_t0 = rolling_ic_gpu(dummy_in, dummy_in, nullptr, nullptr, 0, 2, 2, dummy_out);  // T=0
    printf("error-path smoke: min_valid=1 rc=%d, T=0 rc=%d (both expect nonzero)\n", rc_mv, rc_t0);
    if (rc_mv == 0 || rc_t0 == 0) {
      printf("  [FAIL] error-path smoke: guards did not reject invalid input\n");
      ++g_fail;
    } else {
      printf("  [PASS] error-path smoke (min_valid/T guards)\n");
    }
  }

  // ---- parity anchors (from parity_anchors_v1 manifest) ---------------------
  {
    // ic_tie: F=[1,1,2], R1=[1,2,3] -> 1.0 ; R2=[2,1,3] -> 0.5 (min_valid=2)
    const double f_tie[] = {1.0, 1.0, 2.0};
    const double r_tie1[] = {1.0, 2.0, 3.0};
    const double r_tie2[] = {2.0, 1.0, 3.0};
    check_panel("ic_tie1", f_tie, r_tie1, nullptr, nullptr, 1, 3, 2);
    check_panel("ic_tie2", f_tie, r_tie2, nullptr, nullptr, 1, 3, 2);
  }
  {
    // ic_nan: F=[1,nan,2,3,4], R=[1,2,nan,3,4] -> valid {0,3,4} monotone -> 1.0
    const double f_nan[] = {1.0, nan_payload(), 2.0, 3.0, 4.0};
    const double r_nan[] = {1.0, 2.0, nan_payload(), 3.0, 4.0};
    check_panel("ic_nan", f_nan, r_nan, nullptr, nullptr, 1, 5, 2);
  }
  {
    // ic_inf: F=[1,inf,2,3,4], R=[1,2,-inf,3,4] -> valid {0,3,4} -> 1.0
    const double f_inf[] = {1.0, mk_f64(0x7ff0000000000000ull), 2.0, 3.0, 4.0};
    const double r_ninf[] = {1.0, 2.0, -mk_f64(0x7ff0000000000000ull), 3.0, 4.0};
    check_panel("ic_inf_neginf", f_inf, r_ninf, nullptr, nullptr, 1, 5, 2);
  }
  {
    // ic_factor_tradable_nan: F=[nan,1,2,3], R=[1,2,3,4] -> valid {1,2,3} -> 1.0
    const double f_tn[] = {nan_payload(), 1.0, 2.0, 3.0};
    const double r_tn[] = {1.0, 2.0, 3.0, 4.0};
    check_panel("ic_tradable_nan", f_tn, r_tn, nullptr, nullptr, 1, 4, 2);
  }
  {
    // ic_valid_29/30/31: monotone 0..n-1, min_valid=30 -> NaN / 1.0 / 1.0
    std::vector<double> f29(35), r29(35);
    for (int i = 0; i < 35; ++i) { f29[i] = static_cast<double>(i); r29[i] = static_cast<double>(i); }
    check_panel("ic_valid_29", f29.data(), r29.data(), nullptr, nullptr, 1, 29, 30);
    check_panel("ic_valid_30", f29.data(), r29.data(), nullptr, nullptr, 1, 30, 30);
    check_panel("ic_valid_31", f29.data(), r29.data(), nullptr, nullptr, 1, 31, 30);
  }
  {
    // ic_all_invalid: all NaN both sides -> NaN
    const double all_nan[] = {nan_payload(), nan_payload(), nan_payload()};
    check_panel("ic_all_invalid", all_nan, all_nan, nullptr, nullptr, 1, 3, 2);
  }
  {
    // constant factor row -> NaN (valid >= min_valid but factor constant)
    const double f_const[] = {3.0, 3.0, 3.0, 3.0};
    const double r_var[] = {1.0, 2.0, 3.0, 4.0};
    check_panel("ic_constant_factor", f_const, r_var, nullptr, nullptr, 1, 4, 2);
  }
  {
    // constant return row -> NaN
    const double f_var[] = {1.0, 2.0, 3.0, 4.0};
    const double r_const[] = {5.0, 5.0, 5.0, 5.0};
    check_panel("ic_constant_return", f_var, r_const, nullptr, nullptr, 1, 4, 2);
  }

  // ---- randomized panels: 1e-12 tolerance vs CPU reference -------------------
  printf("== randomized checks ==\n");
  {
    const int kSizes[][2] = {{1, 5}, {1, 32}, {3, 7}, {5, 100}, {8, 128}, {20, 300}};
    for (int run = 0; run < 3; ++run) {
      for (const auto& sz : kSizes) {
        int T = sz[0], N = sz[1];
        Lcg rng(0xC0FFEEu + static_cast<uint32_t>(run) * 7919u + static_cast<uint32_t>(T) * 131u +
                static_cast<uint32_t>(N));
        std::vector<double> F(static_cast<size_t>(T) * N), R(static_cast<size_t>(T) * N);
        std::vector<uint8_t> fM(static_cast<size_t>(T) * N), rM(static_cast<size_t>(T) * N);
        for (int mask_mode = 0; mask_mode < 2; ++mask_mode) {
          for (int special = 0; special < 2; ++special) {
            random_panel(&rng, F, R, fM, rM, T, N, mask_mode != 0, special != 0 && T >= 2,
                         false);
            char nm[128];
            std::snprintf(nm, sizeof(nm), "rand T=%d N=%d mask=%d const=%d run=%d", T, N,
                          mask_mode, special, run);
            check_panel(nm, F.data(), R.data(), mask_mode ? fM.data() : nullptr,
                        mask_mode ? rM.data() : nullptr, T, N, 2 + (run % 3));
          }
        }
      }
    }
  }

  // ---- F/T chunking minimal proof 1: device-level bitwise assertions ---------
  // chunked pipeline (production kernels, chunk-local buffers, re-cut CUB
  // segment offsets) must equal the non-chunked pipeline bitwise on every IC
  // row (uint64 bits, incl. NaN payloads). Closes the rank-class "可分块"
  // criterion + the CUB segment re-cut black box (design spec Sec 4 #8).
  printf("== F/T chunking minimal proof 1 (rolling_ic chunked vs non-chunked) ==\n");
  {
    const std::vector<int> main_chunks = {512, 512, 194};  // full chunks + short tail
    run_chunk_case("main-shape", 1218, 5000, 30, main_chunks, 0xFEED1234u, true);

    const std::vector<int> stress_chunks = {4, 3};         // wide-row stress + tail
    // check_cpu=true anchors the wide-N shape against an independent CPU oracle
    // (differential-only would risk "equal but wrong" at N=250000).
    run_chunk_case("stress-wide-N", 7, 250000, 30, stress_chunks, 0xCAFE1010u, true);

    const std::vector<int> extreme_chunks(32, 1);          // extreme re-cut: 1 row/chunk
    run_chunk_case("extreme-recut", 32, 1000, 30, extreme_chunks, 0xBEEF2020u, true);

    // null-mask: exercises the chunked driver's nullptr-mask base-pointer branch
    // (all masked cases above pass masks; GPT-5.6-Sol F4).
    {
      Lcg rng(0x0D0F0001u);
      const int T = 1218, N = 5000;
      std::vector<double> F(static_cast<size_t>(T) * N), R(static_cast<size_t>(T) * N);
      std::vector<uint8_t> fM(static_cast<size_t>(T) * N), rM(static_cast<size_t>(T) * N);
      proof_panel(&rng, F, R, fM, rM, T, N);
      std::vector<double> full(static_cast<size_t>(T)), chunked(static_cast<size_t>(T));
      int rc1 = rolling_ic_gpu(F.data(), R.data(), nullptr, nullptr, T, N, 30, full.data());
      int rc2 = rolling_ic_gpu_chunked(F.data(), R.data(), nullptr, nullptr, T, N, 30,
                                       main_chunks, chunked.data());
      bool nm_ok = (rc1 == 0 && rc2 == 0) && ic_bitwise(full, chunked);
      printf("  [%s] null-mask (T=%d N=%d chunks=%zu) bitwise (rc1=%d rc2=%d)\n",
             nm_ok ? "PASS" : "FAIL", T, N, main_chunks.size(), rc1, rc2);
      if (!nm_ok) ++g_fail;
    }

    // Negative control: an IN-BOUNDS but misaligned factor-side segmentation
    // (segment boundaries shifted by one cell, so the factor sort ranks within
    // wrong segments) MUST be caught as a bitwise mismatch. This proves the
    // chunk-local re-cut is load-bearing, not vacuous (GPT-5.6-Sol F2); the
    // in-bounds break keeps compute-sanitizer clean (no OOB).
    {
      Lcg rng(0x0BADDEC0u);
      const int T = 1218, N = 5000;
      std::vector<double> F(static_cast<size_t>(T) * N), R(static_cast<size_t>(T) * N);
      std::vector<uint8_t> fM(static_cast<size_t>(T) * N), rM(static_cast<size_t>(T) * N);
      proof_panel(&rng, F, R, fM, rM, T, N);
      std::vector<double> full(static_cast<size_t>(T)), broken(static_cast<size_t>(T));
      int rc1 = rolling_ic_gpu(F.data(), R.data(), fM.data(), rM.data(), T, N, 30,
                               full.data());
      int rc2 = rolling_ic_gpu_chunked(F.data(), R.data(), fM.data(), rM.data(), T, N,
                                       30, main_chunks, broken.data(), nullptr, nullptr,
                                       /*break_offsets_side=*/1);
      bool neg_ok = !(rc2 == 0 && ic_bitwise(full, broken));
      printf("  [%s] negative-control break-recut (rc1=%d rc2=%d; broken==full=%s)\n",
             neg_ok ? "PASS" : "FAIL", rc1, rc2,
             (rc2 == 0 && ic_bitwise(full, broken)) ? "YES(bad)" : "no");
      if (!neg_ok) ++g_fail;
    }

    // min_valid boundary: rows with exactly 29/30/31/32 valid cells (min_valid
    // = 30) must give NaN / IC / IC / IC in BOTH paths (GPT-5.6-Sol F4).
    {
      const int T = 4, N = 40;
      std::vector<double> F(static_cast<size_t>(T) * N, 0.0);
      std::vector<double> R(static_cast<size_t>(T) * N, 0.0);
      std::vector<uint8_t> fM(static_cast<size_t>(T) * N, 0);
      std::vector<uint8_t> rM(static_cast<size_t>(T) * N, 0);
      for (int t = 0; t < T; ++t) {
        const int nvalid = 29 + t;  // 29, 30, 31, 32
        for (int j = 0; j < N; ++j) {
          const int g = t * N + j;
          if (j < nvalid) {
            fM[g] = 1; rM[g] = 1;
            F[g] = static_cast<double>(j);  // monotone on the valid subset
            R[g] = static_cast<double>(j);  // monotone -> IC = 1.0
          }
        }
      }
      const std::vector<int> chunks = {2, 2};
      std::vector<double> full(static_cast<size_t>(T)), chunked(static_cast<size_t>(T));
      int rc1 = rolling_ic_gpu(F.data(), R.data(), fM.data(), rM.data(), T, N, 30,
                               full.data());
      int rc2 = rolling_ic_gpu_chunked(F.data(), R.data(), fM.data(), rM.data(), T, N,
                                       30, chunks, chunked.data());
      bool mv_ok = (rc1 == 0 && rc2 == 0) && ic_bitwise(full, chunked);
      printf("  [%s] min-valid-boundary (T=%d N=%d chunks={2,2}) bitwise (rows 29/30/31/32 valid)\n",
             mv_ok ? "PASS" : "FAIL", T, N);
      if (!mv_ok) {
        printf("      full IC = %.12g %.12g %.12g %.12g\n", full[0], full[1], full[2], full[3]);
        ++g_fail;
      }
    }
  }

  // ---- workspace path (P2 PoC4 perf, 2026-08-08): cached-buffer reuse must be
  // bitwise transparent -- ws vs non-ws identical, reuse, mask on/off, shape
  // switch (realloc), clear-then-reuse, and usable after a contract error.
  printf("== workspace path ==\n");
  {
    auto check_ws = [&](const char* name, const std::vector<double>& F,
                        const std::vector<double>& R,
                        const std::vector<uint8_t>& fM,
                        const std::vector<uint8_t>& rM,
                        int T, int N, int min_valid, rolling_ic_workspace* ws) {
      std::vector<double> full(static_cast<size_t>(T)), wsv(static_cast<size_t>(T));
      int rc1 = rolling_ic_gpu(F.data(), R.data(), fM.empty() ? nullptr : fM.data(),
                               rM.empty() ? nullptr : rM.data(), T, N, min_valid, full.data());
      int rc2 = rolling_ic_gpu(F.data(), R.data(), fM.empty() ? nullptr : fM.data(),
                               rM.empty() ? nullptr : rM.data(), T, N, min_valid, wsv.data(),
                               nullptr, nullptr, nullptr, ws);
      const bool ok = (rc1 == 0 && rc2 == 0) && ic_bitwise(full, wsv);
      printf("  [%s] %s (T=%d N=%d rc1=%d rc2=%d)\n", ok ? "PASS" : "FAIL", name, T, N, rc1, rc2);
      if (!ok) ++g_fail;
    };
    rolling_ic_workspace ws;
    Lcg rng(0x575331u);
    {
      const int T = 30, N = 20;
      std::vector<double> F(static_cast<size_t>(T) * N), R(static_cast<size_t>(T) * N);
      std::vector<uint8_t> fM(static_cast<size_t>(T) * N), rM(static_cast<size_t>(T) * N);
      for (int i = 0; i < T * N; ++i) {
        F[static_cast<size_t>(i)] = rng.uniform() * 4.0 - 2.0;
        R[static_cast<size_t>(i)] = rng.uniform() * 4.0 - 2.0;
        fM[static_cast<size_t>(i)] = (rng.next() % 100) < 90 ? 1 : 0;
        rM[static_cast<size_t>(i)] = (rng.next() % 100) < 90 ? 1 : 0;
      }
      check_ws("ws basic masked", F, R, fM, rM, T, N, 5, &ws);
      check_ws("ws reuse same shape", F, R, fM, rM, T, N, 5, &ws);  // cached reuse
      check_ws("ws mask-off (stale-mask guard)", F, R, std::vector<uint8_t>(),
               std::vector<uint8_t>(), T, N, 5, &ws);
      // shape switch -> realloc (ws key mismatch)
      const int T2 = 12, N2 = 40;
      std::vector<double> F2(static_cast<size_t>(T2) * N2), R2(static_cast<size_t>(T2) * N2);
      for (int i = 0; i < T2 * N2; ++i) {
        F2[static_cast<size_t>(i)] = rng.uniform() * 4.0 - 2.0;
        R2[static_cast<size_t>(i)] = rng.uniform() * 4.0 - 2.0;
      }
      check_ws("ws shape-switch", F2, R2, std::vector<uint8_t>(), std::vector<uint8_t>(),
               T2, N2, 5, &ws);
      // shape-switch then MASKED (P2-PoC4-03): realloc then lazy mask re-alloc
      std::vector<uint8_t> fM2(static_cast<size_t>(T2) * N2), rM2(static_cast<size_t>(T2) * N2);
      for (int i = 0; i < T2 * N2; ++i) {
        fM2[static_cast<size_t>(i)] = (rng.next() % 100) < 90 ? 1 : 0;
        rM2[static_cast<size_t>(i)] = (rng.next() % 100) < 90 ? 1 : 0;
      }
      check_ws("ws shape-switch then masked", F2, R2, fM2, rM2, T2, N2, 5, &ws);
    }
    rolling_ic_workspace_clear(&ws);
    {
      const int T = 4, N = 6;
      std::vector<double> F(static_cast<size_t>(T) * N), R(static_cast<size_t>(T) * N);
      for (int i = 0; i < T * N; ++i) {
        F[static_cast<size_t>(i)] = rng.uniform() * 4.0 - 2.0;
        R[static_cast<size_t>(i)] = rng.uniform() * 4.0 - 2.0;
      }
      check_ws("ws after-clear reuse", F, R, std::vector<uint8_t>(), std::vector<uint8_t>(),
               T, N, 2, &ws);
    }
    // contract error with ws: invalid input rejected, workspace still usable.
    {
      double dummy_out[8];
      int rc = rolling_ic_gpu(nullptr, nullptr, nullptr, nullptr, 1, 2, 2, dummy_out,
                              nullptr, nullptr, nullptr, &ws);
      const bool ok = (rc != 0);
      printf("  [%s] ws null input rejected (rc=%d)\n", ok ? "PASS" : "FAIL", rc);
      if (!ok) ++g_fail;
      const int T = 4, N = 6;
      std::vector<double> F(static_cast<size_t>(T) * N), R(static_cast<size_t>(T) * N);
      for (int i = 0; i < T * N; ++i) {
        F[static_cast<size_t>(i)] = rng.uniform() * 4.0 - 2.0;
        R[static_cast<size_t>(i)] = rng.uniform() * 4.0 - 2.0;
      }
      check_ws("ws usable after error", F, R, std::vector<uint8_t>(), std::vector<uint8_t>(),
               T, N, 2, &ws);
    }
    // ws + MemTracker (F2): reuse rounds must NOT grow alloc_count/live_bytes;
    // clear returns live to 0 (workspace owns its buffers until clear).
    {
      factor_cuda::MemTracker mt;
      rolling_ic_workspace ws2;
      const int T = 30, N = 20;
      std::vector<double> F(static_cast<size_t>(T) * N), R(static_cast<size_t>(T) * N);
      for (int i = 0; i < T * N; ++i) {
        F[static_cast<size_t>(i)] = rng.uniform() * 4.0 - 2.0;
        R[static_cast<size_t>(i)] = rng.uniform() * 4.0 - 2.0;
      }
      std::vector<double> out(static_cast<size_t>(T));
      bool ok = true;
      size_t alloc0 = 0, live0 = 0;
      for (int rep = 0; rep < 5; ++rep) {
        int rc = rolling_ic_gpu(F.data(), R.data(), nullptr, nullptr, T, N, 5,
                                out.data(), &mt, nullptr, nullptr, &ws2);
        if (rc != 0) { ok = false; break; }
        if (rep == 0) { alloc0 = mt.alloc_count(); live0 = mt.live_bytes(); }
        else if (mt.alloc_count() != alloc0 || mt.live_bytes() != live0) {
          ok = false;
          break;
        }
      }
      rolling_ic_workspace_clear(&ws2);
      if (mt.live_bytes() != 0) ok = false;
      printf("  [%s] ws tracker reuse no-leak (5 rounds, clear live==0)\n",
             ok ? "PASS" : "FAIL");
      if (!ok) ++g_fail;
    }
    rolling_ic_workspace_clear(&ws);
  }

  printf("== summary ==\n");
  if (g_fail == 0) {
    printf("ALL PASS (rolling_ic v0 parity + F/T chunking proof 1 bitwise)\n");
    return 0;
  }
  printf("FAILURES: %d\n", g_fail);
  return 1;
}
