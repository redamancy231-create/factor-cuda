// factor-cuda -- PoC 3 frozen-corpus cross-end parity runner (GPU side).
//
// Loads frozen panels from raw .bin files, runs the correlation GPU kernels,
// and dumps the raw GPU output matrices back to .bin for the Python side
// (benchmarks/corpus_parity_v1.py) to compare against the frozen oracle
// corr_oracle_v1.py (per-pair np.corrcoef).
//
// Five cases (closes stock_corr v2 review F4 "release gate" -- GPU kernel vs
// frozen wrapper on frozen inputs, WITH actual dispatch/fallback evidence):
//   1. factor_corr : full corpus factors (T,N,F) f64, masked            -> (F,F)
//   2. stock_corr  : corpus returns prefix (T,N_sub) f64, masked
//                    (expected general path)                             -> (N_sub,N_sub)
//   3. stock_corr  : all-valid fast panel prefix (T,N_sub) f64, no mask
//                    (expected fast path)                                -> (N_sub,N_sub)
//   4. stock_corr  : degenerate frozen panel (T_D,N_D) f64, no mask
//                    (all-valid fast path; constant col -> NaN diagonal) -> (N_D,N_D)
//   5. stock_corr  : low-bias fallback frozen panel (T_F,N_F) f64, masked
//                    (general path; independent N(0.5,1) columns trigger
//                     the cancellation detection -> fallback recompute;
//                     constant col -> NaN pairs)                         -> (N_F,N_F)
// For every stock_corr case the actual selected_path and fallback_count are
// reported on stdout (STATS lines) so the Python side can ASSERT the expected
// path, not just infer it from data.
//
// Usage:
//   poc3_corpus_parity.exe <factors.bin> <mask.bin> <returns.bin>
//     <returns_mask.bin> <fastpanel.bin> <degener.bin> <fallback.bin>
//     <fallback_mask.bin> <out_factor.bin> <out_stock.bin> <out_fast.bin>
//     <out_degen.bin> <out_fallback.bin> <T> <N> <F> <N_sub>
// Binary formats: f64 arrays as raw little-endian doubles; masks as raw uint8.
//
// ASCII-only comments (nvcc/GBK pitfall).
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#include "factor_corr.cuh"
#include "stock_corr.cuh"

namespace {

constexpr int kDegenerateT = 50, kDegenerateN = 4;   // frozen degenerate panel
constexpr int kFallbackT = 100, kFallbackN = 4;      // frozen fallback panel

bool ReadFile(const std::string& path, void* dst, size_t bytes) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }
  f.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(bytes));
  if (!f) { fprintf(stderr, "short read %s (%zu bytes wanted)\n", path.c_str(), bytes); return false; }
  return true;
}

bool WriteFile(const std::string& path, const void* src, size_t bytes) {
  std::ofstream f(path, std::ios::binary);
  if (!f) { fprintf(stderr, "cannot write %s\n", path.c_str()); return false; }
  f.write(reinterpret_cast<const char*>(src), static_cast<std::streamsize>(bytes));
  return f.good();
}

bool LoadF64(const std::string& path, std::vector<double>& v, size_t expected_bytes) {
  v.resize(expected_bytes / sizeof(double));
  return ReadFile(path, v.data(), expected_bytes);
}

bool LoadU8(const std::string& path, std::vector<uint8_t>& v, size_t bytes) {
  v.resize(bytes);
  return ReadFile(path, v.data(), bytes);
}

int RunStock(const char* name, const double* X, const uint8_t* mask, int T, int N,
             const char* out_path) {
  std::vector<double> out(static_cast<size_t>(N) * N);
  StockCorrRunStats stats;
  int rc = stock_corr_gpu(X, mask, T, N, out.data(), nullptr, &stats);
  printf("STATS|case=%s|T=%d|N=%d|rc=%d|selected_path=%d|fallback_count=%d\n",
         name, T, N, rc, stats.selected_path, stats.fallback_count);
  printf("stock_corr(%s) rc=%d out=(%d,%d)\n", name, rc, N, N);
  if (rc != 0) return rc;
  if (!WriteFile(out_path, out.data(), out.size() * sizeof(double))) return 1;
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 18) {
    fprintf(stderr,
            "usage: %s <factors.bin> <mask.bin> <returns.bin> <returns_mask.bin> "
            "<fastpanel.bin> <degener.bin> <fallback.bin> <fallback_mask.bin> "
            "<out_factor.bin> <out_stock.bin> <out_fast.bin> <out_degen.bin> "
            "<out_fallback.bin> <T> <N> <F> <N_sub>\n",
            argv[0]);
    return 2;
  }
  const std::string factors_path = argv[1], mask_path = argv[2],
                    returns_path = argv[3], returns_mask_path = argv[4],
                    fastpanel_path = argv[5], degener_path = argv[6],
                    fallback_path = argv[7], fallback_mask_path = argv[8],
                    out_factor_path = argv[9], out_stock_path = argv[10],
                    out_fast_path = argv[11], out_degen_path = argv[12],
                    out_fallback_path = argv[13];
  const int T = std::atoi(argv[14]), N = std::atoi(argv[15]),
            F = std::atoi(argv[16]), N_sub = std::atoi(argv[17]);

  int dev = 0;
  cudaError_t err = cudaGetDevice(&dev);
  if (err != cudaSuccess) { fprintf(stderr, "cudaGetDevice FAIL\n"); return 1; }

  const size_t R = static_cast<size_t>(T) * N;
  const size_t Rsub = static_cast<size_t>(T) * N_sub;
  const size_t n_fac = static_cast<size_t>(T) * N * F;
  const size_t Rdegen = static_cast<size_t>(kDegenerateT) * kDegenerateN;
  const size_t Rfb = static_cast<size_t>(kFallbackT) * kFallbackN;

  std::vector<double> factors, returns, fastpanel, degener, fallback;
  std::vector<uint8_t> mask, returns_mask, fallback_mask;
  if (!LoadF64(factors_path, factors, n_fac * sizeof(double))) return 1;
  if (!LoadU8(mask_path, mask, R)) return 1;
  if (!LoadF64(returns_path, returns, Rsub * sizeof(double))) return 1;
  if (!LoadU8(returns_mask_path, returns_mask, Rsub)) return 1;
  if (!LoadF64(fastpanel_path, fastpanel, Rsub * sizeof(double))) return 1;
  if (!LoadF64(degener_path, degener, Rdegen * sizeof(double))) return 1;
  if (!LoadF64(fallback_path, fallback, Rfb * sizeof(double))) return 1;
  if (!LoadU8(fallback_mask_path, fallback_mask, Rfb)) return 1;

  // ---- case 1: factor_corr full corpus --------------------------------------
  std::vector<double> outF(static_cast<size_t>(F) * F);
  int rc1 = factor_corr_gpu(factors.data(), mask.data(), T, N, F, outF.data());
  printf("factor_corr rc=%d out=(%d,%d)\n", rc1, F, F);
  if (rc1 != 0) return 1;
  if (!WriteFile(out_factor_path, outF.data(), outF.size() * sizeof(double))) return 1;

  // ---- case 2: stock_corr corpus returns prefix (masked, general) -----------
  if (RunStock("stock_corpus", returns.data(), returns_mask.data(), T, N_sub,
               out_stock_path.c_str()) != 0) return 1;

  // ---- case 3: stock_corr all-valid panel prefix (no mask, fast) ------------
  if (RunStock("stock_fast", fastpanel.data(), nullptr, T, N_sub,
               out_fast_path.c_str()) != 0) return 1;

  // ---- case 4: stock_corr degenerate frozen panel (no mask, fast, NaN diag) --
  if (RunStock("stock_degen", degener.data(), nullptr, kDegenerateT, kDegenerateN,
               out_degen_path.c_str()) != 0) return 1;

  // ---- case 5: stock_corr low-bias fallback frozen panel (masked, general) --
  if (RunStock("stock_fallback", fallback.data(), fallback_mask.data(),
               kFallbackT, kFallbackN, out_fallback_path.c_str()) != 0) return 1;

  printf("ALL GPU RUNS OK\n");
  return 0;
}
