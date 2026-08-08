// factor-cuda -- PoC 3 parameter_scan v0 perf + memory probe.
//
// Times parameter_scan_gpu end-to-end (single H2D + G=4 group pipeline + D2H,
// host wall clock) at corpus scale (1218 x 5000) and samples device free
// memory from a background thread for a peak-bytes estimate (PoC 3 memory
// model). Gate reference: parameter_scan(G=4) exact_half 55.342775 ms
// (gate_config_v1.json).
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
#include "parameter_scan.cuh"

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
  std::vector<float> out[4];
  float* h_out[4];
  int gs[4];
  for (int g = 0; g < 4; ++g) {
    out[g].resize(static_cast<size_t>(total));
    h_out[g] = out[g].data();
  }
  auto gs_ok = [](const int s[4]) {
    for (int g = 0; g < 4; ++g) if (s[g] != 0) return false;
    return true;
  };

  // ---- timing (end-to-end, median of 11) -----------------------------------
  int rc = parameter_scan_gpu(X.data(), nullptr, T, N, h_out, gs);  // warmup
  if (rc != 0) {
    printf("parameter_scan_gpu warmup FAIL rc=%d\n", rc);
    return 1;
  }
  if (!gs_ok(gs)) {
    printf("parameter_scan_gpu warmup group_status not all ok\n");
    return 1;
  }
  printf("  warmup ok, group0 rank[0]=%.1f group3 rank[0]=%.1f\n",
         out[0][0], out[3][0]);
  std::vector<double> ms;
  const int kReps = 11;
  for (int rep = 0; rep < kReps; ++rep) {
    auto t0 = std::chrono::steady_clock::now();
    rc = parameter_scan_gpu(X.data(), nullptr, T, N, h_out, gs);
    auto t1 = std::chrono::steady_clock::now();
    if (rc != 0 || !gs_ok(gs)) {
      printf("parameter_scan_gpu FAIL rc=%d gs=(%d,%d,%d,%d)\n", rc, gs[0],
             gs[1], gs[2], gs[3]);
      return 1;
    }
    ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  std::sort(ms.begin(), ms.end());
  double median = ms[kReps / 2];
  printf("parameter_scan end-to-end: median %.4f ms  (gate exact_half 55.342775 ms)\n",
         median);
  printf("  per-rep: %s\n", [&]() {
    std::string s;
    for (double v : ms) { char b[16]; std::snprintf(b, sizeof(b), "%.2f ", v); s += b; }
    return s;
  }().c_str());
  printf("  end-to-end %s gate (2x speedup baseline)\n",
         median < 55.34277500009921 ? "BEATS" : "does NOT beat");

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
    rc = parameter_scan_gpu(X.data(), nullptr, T, N, h_out, gs);
    if (rc != 0 || !gs_ok(gs)) { printf("FAIL\n"); return 1; }
  }
  stop_sampler.store(true);
  sampler.join();
  cudaDeviceSynchronize();
  size_t mem_after = 0;
  cudaMemGetInfo(&mem_after, &mem_total);
  double peak_used = (mem_before - free_peak_sample.load()) / 1048576.0;
  printf("memory: peak sampling min_free=%.1f MiB -> peak_used ~%.1f MiB\n",
         free_peak_sample.load() / 1048576.0, peak_used);
  printf("memory: net free delta after run %.1f MiB (leak check)\n",
         (mem_before - mem_after) / 1048576.0);

  // ---- workspace steady-state (P2 PoC4 perf, 2026-08-08) --------------------
  // The parameter_scan buffer set IS the cs_rank_workspace set, so a single
  // shared workspace caches all buffers across the 4 groups and across calls.
  // Cold first call allocates; steady-state median vs the non-ws median above.
  {
    cs_rank_workspace ws;
    int rcc = parameter_scan_gpu(X.data(), nullptr, T, N, h_out, gs, nullptr,
                                 nullptr, nullptr, nullptr, &ws);
    if (rcc != 0 || !gs_ok(gs)) { printf("ws cold FAIL rc=%d\n", rcc); return 1; }
    for (int rep = 0; rep < 3; ++rep)
      parameter_scan_gpu(X.data(), nullptr, T, N, h_out, gs, nullptr, nullptr,
                         nullptr, nullptr, &ws);
    std::vector<double> wms;
    for (int rep = 0; rep < kReps; ++rep) {
      auto t0 = std::chrono::steady_clock::now();
      rc = parameter_scan_gpu(X.data(), nullptr, T, N, h_out, gs, nullptr, nullptr,
                              nullptr, nullptr, &ws);
      auto t1 = std::chrono::steady_clock::now();
      if (rc != 0 || !gs_ok(gs)) { printf("ws FAIL rc=%d\n", rc); return 1; }
      wms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(wms.begin(), wms.end());
    double ws_median = wms[kReps / 2];
    printf("parameter_scan workspace steady-state: median %.4f ms  (non-ws %.4f ms, speedup %.2fx)\n",
           ws_median, median, median / ws_median);
    cs_rank_workspace_clear(&ws);
    // P2-PoC4-04: ws-path memory recheck after clear -- free should return to
    // near the baseline (no per-call ws leak, incl. the F1 temp leak fix).
    size_t after_ws = 0;
    cudaMemGetInfo(&after_ws, &mem_total);
    printf("workspace memory: after-clear free %.1f MiB vs baseline %.1f MiB (delta %.1f MiB)\n",
           after_ws / 1048576.0, mem_before / 1048576.0,
           (mem_before - after_ws) / 1048576.0);
  }

  printf("parameter_scan perf+memo done.\n");
  return 0;
}
