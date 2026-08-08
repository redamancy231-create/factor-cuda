// factor-cuda -- PoC 3 factor_corr v0 perf + memory probe.
//
// Times factor_corr_gpu end-to-end (H2D + kernels + D2H, host wall clock) at
// corpus scale (1218 x 5000 x 12 factor columns, f64 inputs) and samples device
// free memory from a background thread for a peak estimate. Gate reference:
// factor_corr exact_half 1543.520875 ms (gate_config_v1.json, CuPy 3087ms/2).
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
#include "factor_corr.cuh"

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

}  // namespace

int main() {
  const int T = 1218, N = 5000, F = 12;
  const int64_t R = static_cast<int64_t>(T) * N;

  int dev = 0;
  cudaError_t err = cudaGetDevice(&dev);
  if (err != cudaSuccess) { printf("cudaGetDevice FAIL\n"); return 1; }
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, dev);
  size_t mem_before = 0, mem_total = 0;
  cudaMemGetInfo(&mem_before, &mem_total);
  printf("GPU: %s, total %.0f MiB, free before %.1f MiB\n", prop.name,
         mem_total / 1048576.0, mem_before / 1048576.0);
  printf("panel T=%d N=%d F=%d (T*N=%lld)\n", T, N, F, R);

  // deterministic f64 factor panel (~10% non-finite, dense ties), similar to
  // the rolling_ic perf probe so cross-op timing is comparable.
  std::vector<double> F3(static_cast<size_t>(R) * F);
  {
    Lcg rng(0x5EEDFAC7u);
    for (int64_t i = 0; i < R; ++i) {
      for (int f = 0; f < F; ++f) {
        uint32_t r = rng.next() % 100;
        double v = r < 78 ? rng.uniform() * 20.0 - 10.0
                 : r < 84 ? 0.0
                 : r < 92 ? mk_f64(0x7ff8000000000000ull | (static_cast<uint64_t>(rng.next()) & 0xFFFFull))
                 : static_cast<double>(rng.next() % 5);
        F3[static_cast<size_t>(i) * F + f] = v;
      }
    }
  }
  std::vector<double> out(static_cast<size_t>(F) * F);

  // ---- timing (end-to-end, median of 11) -----------------------------------
  int rc = factor_corr_gpu(F3.data(), nullptr, T, N, F, out.data());
  if (rc != 0) { printf("warmup FAIL rc=%d\n", rc); return 1; }
  printf("warmup ok, corr[0,1]=%.6f\n", out[1]);
  std::vector<double> ms;
  const int kReps = 11;
  for (int rep = 0; rep < kReps; ++rep) {
    auto t0 = std::chrono::steady_clock::now();
    rc = factor_corr_gpu(F3.data(), nullptr, T, N, F, out.data());
    auto t1 = std::chrono::steady_clock::now();
    if (rc != 0) { printf("FAIL rc=%d\n", rc); return 1; }
    ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  std::sort(ms.begin(), ms.end());
  double median = ms[kReps / 2];
  printf("factor_corr end-to-end: median %.4f ms (gate exact_half 1543.5209 ms)\n", median);
  printf("  per-rep: ");
  for (double v : ms) printf("%.2f ", v);
  printf("\n  %s gate\n", median < 1543.5208750004676 ? "BEATS" : "does NOT beat");

  // ---- memory peak ----------------------------------------------------------
  std::atomic<size_t> min_free{mem_before};
  std::atomic<bool> stop_sampler{false};
  std::thread sampler(RunSampler, &min_free, &stop_sampler, dev);
  for (int rep = 0; rep < 5; ++rep)
    factor_corr_gpu(F3.data(), nullptr, T, N, F, out.data());
  stop_sampler.store(true);
  sampler.join();
  cudaDeviceSynchronize();
  size_t mem_after = 0;
  cudaMemGetInfo(&mem_after, &mem_total);
  double peak_used = (mem_before - min_free.load()) / 1048576.0;
  printf("memory: peak sampling min_free=%.1f MiB -> peak_used ~%.1f MiB\n",
         min_free.load() / 1048576.0, peak_used);
  printf("memory: net free delta after run %.1f MiB\n",
         (mem_before - mem_after) / 1048576.0);
  printf("factor_corr perf+memo done.\n");
  return 0;
}
