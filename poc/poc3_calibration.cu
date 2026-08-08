// factor-cuda -- PoC 3 five-op three-way memory calibration.
//
// Extends the cs_rank-only mem_tracker selfcheck to all 5 PoC 3 kernels.
// For each (op, scale) case we measure memory from three points of view
// (poc34_workload_estimate.md Sec 3.3 calibration discipline):
//   (1) theoretical formula -- align256 sum of every device buffer, computed
//       from the documented allocation layout per kernel (no CUB temp);
//   (2) tracker HWM -- MemTracker live-byte peak while the op runs;
//   (3) driver sample -- background cudaMemGetInfo min-free thread.
// Deviations (two measures, per protocol):
//   formula(+temp) vs HWM  -- expect EXACT 0 (allocation is deterministic;
//       CUB temp is the only theory blind spot and is taken from the tracker);
//   HWM vs driver sample  -- expect 0 <= overhead <= kDriverTolerance.
// Also asserts final live == 0 (leak) and unknown-free == 0 (strict tracker).
//
// Scales: canonical 1218x5000 (all ops, F=12) and large N=10000 (validates
// model extrapolation; stock_corr N=10000 also validates the O(N^2) output
// irreducible claim, ~800 MB result matrix).
//
// Theory formulas derived from the actual AllocOrTrack chains:
//   cs_rank / parameter_scan (masked, n=T*N):
//     6*align(4n) + 2*align(n) + align(4(T+1))        [valid, keys2, vals2,
//       offsets, X, mask, out]
//   rolling_ic (f64, both masks, n=T*N):
//     8*align(8n) + 4*align(4n) + 3*align(n)
//       + align(4(T+1)) + align(4T) + 5*align(8T)     [F,R,4keys,4vals,valid,
//       rank2, offsets, counts, 4 min/max, ic]
//   factor_corr (R=T*N, P=F(F+1)/2, masked):
//     2*align(8FR) + align(FR) + align(8F^2) + align(R)
//       + align(8P)+align(56P)+align(16P)+align(24P)+align(8P)+align(P)
//       + align(4P)+align(40P)+align(16P)+align(48P)  [Xt, valid, F, mask,
//       pairs, gp1, means, gp2, corr, trigger, trig_pairs, gk1, kmeans, gk2, out]
//   stock_corr (masked, n=T*N):
//     3*align(8n) + align(n) + 2*align(8N^2) + align(48N) + align(8N)
//       [Xm, M, X, mask, corr, out, stats, s2]  (d_X is freed after transpose
//       but the peak is reached at the end of the alloc chain, so it counts)
// CUB temp (cs_rank/parameter_scan/rolling_ic) is the documented theory blind
// spot: taken from the tracker's "cub_temp" alloc event.
//
// ASCII-only comments (nvcc/GBK pitfall).
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <cuda_runtime.h>

#include "cross_sectional_rank.cuh"
#include "parameter_scan.cuh"
#include "rolling_ic.cuh"
#include "factor_corr.cuh"
#include "stock_corr.cuh"
#include "mem_tracker.h"

namespace {

struct Lcg {
  uint64_t s;
  explicit Lcg(uint64_t seed) : s(seed) {}
  uint32_t next() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<uint32_t>(s >> 33);
  }
  float uniform() { return static_cast<float>(next() >> 8) / 16777216.0f; }
  double uniform_d() { return static_cast<double>(next() >> 8) / 16777216.0; }
};

size_t align256(size_t b) { return ((b + 255u) / 256u) * 256u; }

// Background sampler: track min free bytes while the kernel runs. cudaMalloc is
// synchronous on the host side, so cudaMemGetInfo reflects live allocations
// immediately (no per-sample cudaDeviceSynchronize, which could miss the HWM).
// The new thread must pin the same CUDA device (mem_tracker review F9).
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

// ---- theory formulas (align256 sums, no CUB temp) ---------------------------

// Uniform theory signature: (T, N, F, masked). Unused params ignored.
size_t theory_cs_rank(int T, int N, int /*F*/, bool masked) {
  const size_t n = static_cast<size_t>(T) * N;
  size_t s = 6 * align256(4 * n) + align256(4 * (static_cast<size_t>(T) + 1));
  s += masked ? 2 * align256(n) : align256(n);
  return s;
}

size_t theory_parameter_scan(int T, int N, int F, bool masked) {
  // masked scan: same allocation set as cs_rank with a mask (mask allocated).
  return theory_cs_rank(T, N, F, /*masked=*/true);
}

size_t theory_rolling_ic(int T, int N, int /*F*/, bool /*masked*/) {  // f64 inputs, both masks
  const size_t n = static_cast<size_t>(T) * N;
  const size_t Tsz = static_cast<size_t>(T);
  return 8 * align256(8 * n) + 4 * align256(4 * n) + 3 * align256(n) +
         align256(4 * (Tsz + 1)) + align256(4 * Tsz) + 5 * align256(8 * Tsz);
}

size_t theory_factor_corr(int T, int N, int F, bool masked) {
  const size_t R = static_cast<size_t>(T) * N;
  const size_t P = static_cast<size_t>(F) * (F + 1) / 2;
  size_t s = 2 * align256(8 * F * R) + align256(F * R) + align256(8 * F * F) +
             align256(8 * P) + align256(56 * P) + align256(16 * P) +
             align256(24 * P) + align256(8 * P) + align256(P) +
             align256(4 * P) + align256(40 * P) + align256(16 * P) +
             align256(48 * P);
  s += masked ? align256(R) : 0;
  return s;
}

size_t theory_stock_corr(int T, int N, int /*F*/, bool masked) {
  const size_t n = static_cast<size_t>(T) * N;
  const size_t Nsz = static_cast<size_t>(N);
  size_t s = 3 * align256(8 * n) + 2 * align256(8 * Nsz * Nsz) +
             align256(48 * Nsz) + align256(8 * Nsz);
  s += masked ? align256(n) : 0;
  return s;
}

// ---- per-case measurement ---------------------------------------------------

struct CalibCase {
  std::string op;
  int T, N, F;
  bool masked;
  int reps;
  size_t theory_no_temp = 0;
  size_t temp_aligned = 0;
  size_t theory_with_temp = 0;
  size_t alloc_sum_all_reps = 0;  // sum of aligned bytes of ALL tracker alloc events across all reps
  size_t hwm = 0;
  size_t driver_peak = 0;
  long long delta_formula = 0;   // hwm - theory_with_temp (expect 0)
  long long overhead = 0;        // driver_peak - hwm (expect [0, tol])
  size_t final_live = 0;
  size_t unknown_free = 0;
  int rc = 0;
  bool pass = false;
  std::string note;  // e.g. formula text
};

// Generic measurement harness. run(tracker, masked) must return 0 on success
// and is called `reps` times so the sampler reliably hits the HWM window.
template <typename TheoryFn, typename RunFn>
CalibCase Measure(const std::string& op, int T, int N, int F, bool masked,
                  int reps, TheoryFn theory, RunFn run, int dev) {
  CalibCase c;
  c.op = op; c.T = T; c.N = N; c.F = F; c.masked = masked; c.reps = reps;
  c.theory_no_temp = theory(T, N, F, masked);

  factor_cuda::MemTracker tracker;
  size_t free_before = 0, total = 0;
  cudaMemGetInfo(&free_before, &total);
  std::atomic<size_t> min_free{free_before};
  std::atomic<bool> stop{false};
  std::thread sampler(RunSampler, &min_free, &stop, dev);

  int rc = 0;
  for (int i = 0; i < reps; ++i) {
    rc = run(&tracker, masked);
    if (rc != 0) break;
  }
  stop.store(true);
  sampler.join();
  cudaDeviceSynchronize();
  c.rc = rc;

  c.hwm = tracker.peak_live_bytes();
  for (const auto& e : tracker.events()) {
    if (e.action == "alloc") {
      c.alloc_sum_all_reps += e.aligned_bytes;
      // CUB temp is a per-call allocation; HWM is a single-call peak, so the
      // theory blind-spot temp must be the MAX single-call size, NOT the sum
      // across reps (sum inflates theory_with_temp by (reps-1) * temp).
      if (e.name == "cub_temp" && e.aligned_bytes > c.temp_aligned) {
        c.temp_aligned = e.aligned_bytes;
      }
    }
  }
  if (getenv("CALIB_DUMP_EVENTS") != nullptr) {
    printf("EVENTS[%s T=%d N=%d]:\n", c.op.c_str(), c.T, c.N);
    for (const auto& e : tracker.events()) {
      printf("  %-12s %-5s %-12s logical=%9zu aligned=%9zu live=%9zu\n",
             e.name.c_str(), e.action.c_str(), e.stage.c_str(), e.logical_bytes,
             e.aligned_bytes, e.live_after);
    }
  }
  c.theory_with_temp = c.theory_no_temp + c.temp_aligned;
  c.delta_formula = static_cast<long long>(c.hwm) -
                    static_cast<long long>(c.theory_with_temp);
  size_t free_after = 0;
  cudaMemGetInfo(&free_after, &total);
  c.driver_peak = free_before - min_free.load();
  c.overhead = static_cast<long long>(c.driver_peak) -
               static_cast<long long>(c.hwm);
  c.final_live = tracker.live_bytes();
  c.unknown_free = tracker.unknown_free_count();

  const size_t kDriverTolerance = 64ull * 1024 * 1024;
  c.pass = (rc == 0) && (c.delta_formula == 0) &&
           (c.overhead >= 0) &&
           (c.overhead <= static_cast<long long>(kDriverTolerance)) &&
           (c.final_live == 0) && (c.unknown_free == 0);
  return c;
}

void PrintCase(const CalibCase& c) {
  printf("CASE|op=%s|T=%d|N=%d|F=%d|masked=%d|reps=%d|theory_no_temp=%zu|"
         "alloc_sum_all_reps=%zu|temp_aligned=%zu|theory_with_temp=%zu|hwm=%zu|"
         "driver_peak=%zu|delta_formula=%lld|overhead=%lld|final_live=%zu|"
         "unknown_free=%zu|rc=%d|PASS=%d\n",
         c.op.c_str(), c.T, c.N, c.F, c.masked ? 1 : 0, c.reps,
         c.theory_no_temp, c.alloc_sum_all_reps, c.temp_aligned,
         c.theory_with_temp, c.hwm, c.driver_peak, c.delta_formula, c.overhead,
         c.final_live, c.unknown_free, c.rc, c.pass ? 1 : 0);
  printf("  %-4s T=%-5d N=%-6d %s theory %8.2f MiB + temp %7.2f = %8.2f "
         "| HWM %8.2f | driver %8.2f | d_formula %+lld B | overhead %+.2f MiB"
         " | live %zu | %s\n",
         c.op.c_str(), c.T, c.N, c.pass ? "PASS" : "FAIL",
         c.theory_no_temp / 1048576.0, c.temp_aligned / 1048576.0,
         c.theory_with_temp / 1048576.0, c.hwm / 1048576.0,
         c.driver_peak / 1048576.0, c.delta_formula,
         static_cast<double>(c.overhead) / 1048576.0, c.final_live,
         c.pass ? "" : "(see CASE line)");
}

}  // namespace

int main() {
  int dev = 0;
  cudaError_t err = cudaGetDevice(&dev);
  if (err != cudaSuccess) { printf("cudaGetDevice FAIL\n"); return 1; }
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, dev);
  size_t free0 = 0, total = 0;
  cudaMemGetInfo(&free0, &total);
  printf("GPU: %s, total %.0f MiB, free before %.0f MiB\n", prop.name,
         total / 1048576.0, free0 / 1048576.0);

  int all_pass = 1;

  // ---- cs_rank (masked, canonical + large) ----------------------------------
  {
    for (const auto& [T, N] : std::vector<std::pair<int, int>>{{1218, 5000},
                                                               {1218, 10000}}) {
      const size_t n = static_cast<size_t>(T) * N;
      // rebuild per scale via a scale-capturing lambda
      auto run = [&, T, N](factor_cuda::MemTracker* tr, bool msk) -> int {
        std::vector<float> X(n), out(n);
        std::vector<uint8_t> mask(n);
        Lcg rng(0x5EEDC0DEu);
        for (size_t i = 0; i < n; ++i) {
          X[i] = rng.uniform() * 20.0f - 10.0f;
          mask[i] = (rng.next() % 100) < 98 ? 1 : 0;
        }
        return cs_rank_gpu(X.data(), msk ? mask.data() : nullptr, T, N, false,
                           out.data(), tr);
      };
      CalibCase c = Measure("cs_rank", T, N, 0, /*masked=*/true, 5,
                            theory_cs_rank, run, dev);
      PrintCase(c);
      if (!c.pass) all_pass = 0;
    }
  }

  // ---- parameter_scan (masked, canonical + large) ---------------------------
  {
    for (const auto& [T, N] : std::vector<std::pair<int, int>>{{1218, 5000},
                                                               {1218, 10000}}) {
      auto run = [&, T, N](factor_cuda::MemTracker* tr, bool) -> int {
        const size_t n = static_cast<size_t>(T) * N;
        std::vector<float> X(n);
        std::vector<uint8_t> mask(n);
        std::vector<float> outs[4];
        for (auto& o : outs) o.assign(n, 0.0f);
        int status[4] = {0, 0, 0, 0};
        Lcg rng(0x5EEDC0DEu);
        for (size_t i = 0; i < n; ++i) {
          X[i] = rng.uniform() * 20.0f - 10.0f;
          mask[i] = (rng.next() % 100) < 98 ? 1 : 0;
        }
        float* arr[4] = {outs[0].data(), outs[1].data(), outs[2].data(),
                         outs[3].data()};
        return parameter_scan_gpu(X.data(), mask.data(), T, N, arr, status, tr);
      };
      CalibCase c = Measure("parameter_scan", T, N, 0, /*masked=*/true, 3,
                            theory_parameter_scan, run, dev);
      PrintCase(c);
      if (!c.pass) all_pass = 0;
    }
  }

  // ---- rolling_ic (f64, both masks, canonical + large) ----------------------
  {
    for (const auto& [T, N] : std::vector<std::pair<int, int>>{{1218, 5000},
                                                               {1218, 10000}}) {
      auto run = [&, T, N](factor_cuda::MemTracker* tr, bool) -> int {
        const size_t n = static_cast<size_t>(T) * N;
        std::vector<double> F(n), R(n), ic(static_cast<size_t>(T));
        std::vector<uint8_t> fmask(n), rmask(n);
        Lcg rng(0x5EEDC0DEu);
        for (size_t i = 0; i < n; ++i) {
          F[i] = rng.uniform_d() * 4.0 - 2.0;
          R[i] = rng.uniform_d() * 0.04 - 0.02;
          uint8_t f = (rng.next() % 100) < 98 ? 1 : 0;
          uint8_t r = (rng.next() % 100) < 99 ? 1 : 0;
          fmask[i] = f; rmask[i] = r;
        }
        return rolling_ic_gpu(F.data(), R.data(), fmask.data(), rmask.data(),
                              T, N, 30, ic.data(), tr);
      };
      CalibCase c = Measure("rolling_ic", T, N, 0, /*masked=*/true, 3,
                            theory_rolling_ic, run, dev);
      PrintCase(c);
      if (!c.pass) all_pass = 0;
    }
  }

  // ---- factor_corr (masked, canonical + large) ------------------------------
  {
    for (const auto& [T, N] :
         std::vector<std::pair<int, int>>{{1218, 5000}, {1218, 10000}}) {
      const int F = 12;
      auto run = [&, T, N, F](factor_cuda::MemTracker* tr, bool msk) -> int {
        const size_t R = static_cast<size_t>(T) * N;
        std::vector<double> X(R * F), out(static_cast<size_t>(F) * F);
        std::vector<uint8_t> mask(R);
        Lcg rng(0x5EEDC0DEu);
        for (size_t i = 0; i < R * F; ++i) X[i] = rng.uniform_d() * 4.0 - 2.0;
        for (size_t i = 0; i < R; ++i) mask[i] = (rng.next() % 100) < 98 ? 1 : 0;
        return factor_corr_gpu(X.data(), msk ? mask.data() : nullptr, T, N, F,
                               out.data(), tr);
      };
      CalibCase c = Measure("factor_corr", T, N, F, /*masked=*/true, 3,
                            theory_factor_corr, run, dev);
      PrintCase(c);
      if (!c.pass) all_pass = 0;
    }
  }

  // ---- stock_corr (masked, N=2000/5000/10000) ------------------------------
  {
    for (const int N : {2000, 5000, 10000}) {
      const int T = 1218;
      auto run = [&, T, N](factor_cuda::MemTracker* tr, bool msk) -> int {
        const size_t n = static_cast<size_t>(T) * N;
        std::vector<double> X(n), out(static_cast<size_t>(N) * N);
        std::vector<uint8_t> mask(n);
        Lcg rng(0x5EEDC0DEu);
        for (size_t i = 0; i < n; ++i) {
          X[i] = rng.uniform_d() * 0.04 - 0.02;
          mask[i] = (rng.next() % 100) < 98 ? 1 : 0;
        }
        return stock_corr_gpu(X.data(), msk ? mask.data() : nullptr, T, N,
                              out.data(), tr);
      };
      CalibCase c = Measure("stock_corr", T, N, 0, /*masked=*/true, 3,
                            theory_stock_corr, run, dev);
      PrintCase(c);
      if (!c.pass) all_pass = 0;
    }
  }

  printf("== summary ==\n");
  printf("%s (five-op three-way memory calibration)\n",
         all_pass ? "ALL PASS" : "FAILURES PRESENT");
  return all_pass ? 0 : 1;
}
