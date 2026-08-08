// factor-cuda -- PoC 3 stock_corr v0 perf + memory probe.
//
// Times stock_corr_gpu end-to-end (H2D + kernels + D2H, host wall clock) at
// the formal gate scale (T=1218, N=500, f64 returns) and the extension scales
// (N=2000/5000) with a device free-memory peak estimate. Gate reference:
// stock_corr(N=500) exact_half 21.805625 ms (gate_config_v1.json, CuPy
// 43.611ms/2); extension N=2000 268.74ms / N=5000 1279.44ms.
//
// ASCII-only comments (nvcc/GBK pitfall).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>
#include <cuda_runtime.h>
#include "stock_corr.cuh"

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

double mk_f64(uint64_t bits) {
  double d;
  std::memcpy(&d, &bits, sizeof(double));
  return d;
}

void RunSampler(std::atomic<size_t>* min_free, std::atomic<bool>* stop, int dev) {
  cudaSetDevice(dev);
  while (!stop->load()) {
    size_t f = 0, tot = 0;
    if (cudaMemGetInfo(&f, &tot) == cudaSuccess) {
      size_t cur = min_free->load();
      while (f < cur) {
        if (min_free->compare_exchange_weak(cur, f)) break;
      }
    }
  }
}

// deterministic returns-like panel (mean ~0, ~8% non-finite, dense ties)
void make_panel(const char* what, int T, int N, std::vector<double>& X) {
  Lcg rng(0x5EEDC0DEu + static_cast<uint32_t>(N));
  X.assign(static_cast<size_t>(T) * N, 0.0);
  for (int t = 0; t < T; ++t) {
    for (int i = 0; i < N; ++i) {
      uint32_t r = rng.next() % 100;
      double v = r < 82 ? rng.uniform() * 0.1 - 0.05     // returns-like small
               : r < 88 ? 0.0
               : r < 94 ? mk_f64(0x7ff8000000000000ull | (static_cast<uint64_t>(rng.next()) & 0xFFFFull))
               : static_cast<double>(rng.next() % 5);
      X[static_cast<size_t>(t) * N + i] = v;
    }
  }
  printf("  panel %s: T=%d N=%d (T*N=%lld)\n", what, T, N,
         static_cast<int64_t>(T) * N);
}

// Load the fully-valid v2 panel (.bin: raw float64, T*N_COLS) so the GPU fast
// path and the CuPy gate reference run on the SAME panel (user decision:
// same-panel re-baseline, 2026-08-05).
int load_panel(int T, int N_COLS, std::vector<double>& X, const char* path) {
  FILE* f = fopen(path, "rb");
  if (!f) { printf("  cannot open panel %s\n", path); return 1; }
  X.assign(static_cast<size_t>(T) * N_COLS, 0.0);
  size_t got = fread(X.data(), sizeof(double), static_cast<size_t>(T) * N_COLS, f);
  fclose(f);
  if (got != static_cast<size_t>(T) * N_COLS) {
    printf("  short read %zu/%lld\n", got, static_cast<long long>(T) * N_COLS);
    return 1;
  }
  return 0;
}

int run_one(int T, int N, double gate_ms, const char* gate_name, int kReps,
            const std::vector<double>& X, const char* what) {
  std::vector<double> out(static_cast<size_t>(N) * N);

  int dev = 0;
  cudaGetDevice(&dev);
  size_t mem_before = 0, mem_total = 0;
  cudaMemGetInfo(&mem_before, &mem_total);

  int rc = stock_corr_gpu(X.data(), nullptr, T, N, out.data());
  if (rc != 0) { printf("  warmup FAIL rc=%d\n", rc); return 1; }
  printf("  warmup ok, corr[0,1]=%.6f\n", out[1]);

  std::vector<double> ms;
  for (int rep = 0; rep < kReps; ++rep) {
    auto t0 = std::chrono::steady_clock::now();
    rc = stock_corr_gpu(X.data(), nullptr, T, N, out.data());
    auto t1 = std::chrono::steady_clock::now();
    if (rc != 0) { printf("  FAIL rc=%d\n", rc); return 1; }
    ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  std::sort(ms.begin(), ms.end());
  double median = ms[kReps / 2];
  printf("  stock_corr[%s] N=%d end-to-end: median %.4f ms (gate %s %.4f ms)\n",
         what, N, median, gate_name, gate_ms);
  printf("    per-rep: ");
  for (double v : ms) printf("%.2f ", v);
  printf("\n    %s gate\n", median < gate_ms ? "BEATS" : "does NOT beat");

  // memory peak (only for the largest run to bound test time)
  std::atomic<size_t> min_free{mem_before};
  std::atomic<bool> stop_sampler{false};
  std::thread sampler(RunSampler, &min_free, &stop_sampler, dev);
  for (int rep = 0; rep < 3; ++rep)
    stock_corr_gpu(X.data(), nullptr, T, N, out.data());
  stop_sampler.store(true);
  sampler.join();
  cudaDeviceSynchronize();
  size_t mem_after = 0;
  cudaMemGetInfo(&mem_after, &mem_total);
  printf("    memory: peak_used ~%.1f MiB; net free delta %.1f MiB\n",
         (mem_before - min_free.load()) / 1048576.0,
         (mem_before - mem_after) / 1048576.0);
  return 0;
}

}  // namespace

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  int dev = 0;
  cudaError_t err = cudaGetDevice(&dev);
  if (err != cudaSuccess) { printf("cudaGetDevice FAIL\n"); return 1; }
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, dev);
  size_t mem_before = 0, mem_total = 0;
  cudaMemGetInfo(&mem_before, &mem_total);
  printf("GPU: %s (cc %d.%d), SM %d, total %.0f MiB, free %.1f MiB\n",
         prop.name, prop.major, prop.minor, prop.multiProcessorCount,
         mem_total / 1048576.0, mem_before / 1048576.0);
  printf("== stock_corr v2 perf ==\n");

  const int T = 1218;
  const int N_COLS = 5000;
  // gate references: GENERAL path -> corpus gate (informative); FAST path ->
  // new same-panel gate (CuPy on the fully-valid panel, filled after baseline).
  const double gate_gen[3][2] = {{21.80562500126255, 11},   // N=500, reps
                                 {268.7398750003922, 5},    // N=2000
                                 {1279.440849999446, 3}};   // N=5000
  // Same-panel v2 gates (CuPy exact_half on the fully-valid panel; evidence
  // benchmarks/results/runs/stock_corr_v2_rebaseline_20260805/gate.json):
  const double gate_fast[3] = {26.348400, 359.351800, 2382.366900};  // N=500/2000/5000

  // --- general path: returns-like panel with ~6% NaN -> partial validity ---
  {
    std::vector<double> X;
    make_panel("returns", T, 5000, X);
    std::vector<double> sub(static_cast<size_t>(T) * 500);
    for (int t = 0; t < T; ++t)
      for (int i = 0; i < 500; ++i) sub[(size_t)t * 500 + i] = X[(size_t)t * 5000 + i];
    int rc = run_one(T, 500, gate_gen[0][0], "21.8056 (corpus)", 11, sub, "general");
    if (rc != 0) return rc;
  }

  // --- fast path: fully-valid panel (same .bin as the CuPy gate reference) ---
  {
    std::vector<double> panel;
    if (load_panel(T, N_COLS, panel,
                   "benchmark_corpus/stock_corr_panel_v1_5000.bin"))
      return 1;
    const int sizes[3] = {500, 2000, 5000};
    for (int s = 0; s < 3; ++s) {
      int N = sizes[s];
      std::vector<double> sub(static_cast<size_t>(T) * N);
      for (int t = 0; t < T; ++t)
        for (int i = 0; i < N; ++i) sub[(size_t)t * N + i] = panel[(size_t)t * N_COLS + i];
      char gname[64];
      snprintf(gname, sizeof(gname), "%.4f (v2 gate)", gate_fast[s]);
      int rc = run_one(T, N, gate_fast[s], gname, (int)gate_gen[s][1], sub, "fast");
      if (rc != 0) return rc;
    }
  }

  printf("stock_corr perf+memo done.\n");
  return 0;
}
