// factor-cuda -- PoC 3 cs_rank v0 perf + memory probe.
//
// Times cs_rank_gpu end-to-end (H2D + kernels + D2H, host wall clock, aligned
// with PoC 2 baseline timing) at corpus scale (1218 x 5000), and samples device
// free memory from a background thread for a peak-bytes estimate (PoC 3 memory
// model). Gate reference: cs_rank exact_half 13.926 ms (gate_config_v1.json).
//
// ASCII-only comments (nvcc/GBK pitfall).
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <atomic>
#include <cuda_runtime.h>
#include "cross_sectional_rank.cuh"

namespace {

struct Lcg {
  uint64_t s;
  explicit Lcg(uint64_t seed) : s(seed) {}
  uint32_t next() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<uint32_t>(s >> 33);
  }
  float uniform() { return static_cast<float>(next() >> 8) / 16777216.0f; }
};

float mk_f32(uint32_t bits) {
  float f;
  std::memcpy(&f, &bits, sizeof(float));
  return f;
}

}  // namespace

int main() {
  const int T = 1218, N = 5000;  // corpus_synth_v1 panel shape
  const int64_t total = static_cast<int64_t>(T) * N;

  int dev = 0;
  cudaError_t err = cudaGetDevice(&dev);
  if (err != cudaSuccess) {
    printf("cudaGetDevice FAIL: %s\n", cudaGetErrorString(err));
    return 1;
  }
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, dev);
  size_t mem_before = 0, mem_total = 0;
  cudaMemGetInfo(&mem_before, &mem_total);
  printf("GPU: %s (cc %d.%d), total %.0f MiB, free before %.0f MiB\n", prop.name,
         prop.major, prop.minor, mem_total / 1048576.0, mem_before / 1048576.0);
  printf("panel T=%d N=%d (T*N=%lld)\n", T, N, total);

  // deterministic random float32 panel (uniform in [-10, 10), ~8% non-finite)
  std::vector<float> X(static_cast<size_t>(total));
  {
    Lcg rng(0x5EEDC0DEu);
    for (int64_t i = 0; i < total; ++i) {
      uint32_t r = rng.next() % 100;
      float v;
      if (r < 92) {
        v = rng.uniform() * 20.0f - 10.0f;
      } else if (r < 96) {
        uint32_t nb = 0x7fc00000u | (rng.next() & 0x1FFFFFu);
        std::memcpy(&v, &nb, sizeof(float));
      } else {
        v = (rng.next() & 1u) ? mk_f32(0x7f800000u) : mk_f32(0xff800000u);
      }
      X[static_cast<size_t>(i)] = v;
    }
  }
  std::vector<float> out(static_cast<size_t>(total));

  // ---- timing (workspace steady-state, median of 11) ------------------------
  // A persistent workspace caches all device buffers across calls (no per-call
  // malloc/free, measured ~7.7 ms of the ~16.4 ms). This is the realistic
  // steady-state for repeated calls (parameter scan / rolling window).
  cs_rank_workspace ws;
  auto tc0 = std::chrono::steady_clock::now();
  int rc = cs_rank_gpu(X.data(), nullptr, T, N, false, out.data(), nullptr, &ws);  // cold: allocates ws
  auto tc1 = std::chrono::steady_clock::now();
  if (rc != 0) {
    printf("cs_rank_gpu warmup FAIL rc=%d\n", rc);
    return 1;
  }
  double cold_ms = std::chrono::duration<double, std::milli>(tc1 - tc0).count();
  std::vector<double> ms;
  const int kReps = 11;
  for (int rep = 0; rep < kReps; ++rep) {
    auto t0 = std::chrono::steady_clock::now();
    rc = cs_rank_gpu(X.data(), nullptr, T, N, false, out.data(), nullptr, &ws);
    auto t1 = std::chrono::steady_clock::now();
    if (rc != 0) {
      printf("cs_rank_gpu FAIL rc=%d\n", rc);
      return 1;
    }
    ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  std::sort(ms.begin(), ms.end());
  double median = ms[kReps / 2];
  printf("cs_rank steady-state (workspace cached): median %.4f ms  (gate exact_half 13.926 ms)\n", median);
  printf("  cold first call (incl. workspace alloc): %.4f ms\n", cold_ms);
  printf("  per-rep: %s\n", [&]() {
    std::string s;
    for (double v : ms) { char b[16]; std::snprintf(b, sizeof(b), "%.2f ", v); s += b; }
    return s;
  }().c_str());
  printf("  steady-state %s gate (2x speedup baseline)\n",
         median < 13.926450001235935 ? "BEATS" : "does NOT beat");

  // ---- secondary regression: non-workspace (default) path + cold spread -----
  // The gate above proves the opt-in workspace path. The default ws=nullptr path
  // (allocate/free every call) and the cold first-call remain as regression
  // baselines (review F08).
  {
    std::vector<double> noms, colds;
    for (int rep = 0; rep < kReps; ++rep) {
      auto t0 = std::chrono::steady_clock::now();
      rc = cs_rank_gpu(X.data(), nullptr, T, N, false, out.data());  // no ws
      auto t1 = std::chrono::steady_clock::now();
      if (rc != 0) { printf("cs_rank_gpu FAIL rc=%d\n", rc); return 1; }
      noms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(noms.begin(), noms.end());
    double nomedian = noms[kReps / 2];
    // cold spread: clear + single call, repeated (each pays the alloc)
    for (int rep = 0; rep < 5; ++rep) {
      auto t0 = std::chrono::steady_clock::now();
      rc = cs_rank_gpu(X.data(), nullptr, T, N, false, out.data(), nullptr, &ws);
      auto t1 = std::chrono::steady_clock::now();
      cs_rank_workspace_clear(&ws);  // next call re-allocates -> cold
      if (rc != 0) { printf("cs_rank_gpu FAIL rc=%d\n", rc); return 1; }
      colds.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(colds.begin(), colds.end());
    double cold_p50 = colds[2], cold_p95 = colds[4];
    printf("  legacy non-workspace median: %.4f ms (alloc/free per call)\n", nomedian);
    printf("  cold (clear + realloc) p50=%.4f p95=%.4f ms\n", cold_p50, cold_p95);
  }

  // ---- memory peak (background free-memory sampler) ------------------------
  std::atomic<size_t> free_peak_sample{SIZE_MAX};
  std::atomic<bool> stop_sampler{false};
  std::thread sampler([&]() {
    while (!stop_sampler.load()) {
      size_t f = 0, tot = 0;
      if (cudaMemGetInfo(&f, &tot) == cudaSuccess) {
        size_t cur = free_peak_sample;
        while (f < cur) {
          if (free_peak_sample.compare_exchange_weak(cur, f)) break;
        }
      }
    }
  });
  for (int rep = 0; rep < 5; ++rep) {
    rc = cs_rank_gpu(X.data(), nullptr, T, N, false, out.data(), nullptr, &ws);
    if (rc != 0) { printf("FAIL\n"); return 1; }
  }
  stop_sampler.store(true);
  sampler.join();
  cudaDeviceSynchronize();
  cs_rank_workspace_clear(&ws);  // release cached buffers so the leak check is clean
  size_t mem_after = 0;
  cudaMemGetInfo(&mem_after, &mem_total);
  double peak_used = (mem_before - free_peak_sample.load()) / 1048576.0;
  printf("memory: peak sampling min_free=%.1f MiB -> peak_used ~%.1f MiB\n",
         free_peak_sample.load() / 1048576.0, peak_used);
  printf("memory: net free delta after run %.1f MiB (leak check)\n",
         (mem_before - mem_after) / 1048576.0);

  printf("cs_rank perf+memo done.\n");
  return 0;
}
