// factor-cuda -- PoC 3 mem_tracker selfcheck.
//
// Three-way memory-model calibration for cs_rank v0 at corpus scale:
//   (1) theoretical formula  -- align256 sum of all buffers, no CUB temp
//   (2) tracker HWM          -- MemTracker live-byte peak while cs_rank runs
//   (3) driver sample        -- background cudaMemGetInfo min-free thread
// Calibration protocol: theoretical vs tracker ~0 (allocation is deterministic);
// tracker vs driver ~driver overhead (< ~100 MB). The canonical theoretical
// no-temp HWM for cs_rank is 152,254,876 B (IMPLEMENTATION F-06).
//
// ASCII-only comments (nvcc/GBK pitfall).
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>
#include <cuda_runtime.h>
#include "cross_sectional_rank.cuh"
#include "mem_tracker.h"

namespace {

constexpr int kT = 1218, kN = 5000;
constexpr size_t kTheoreticalNoTemp = 152254876ull;  // align256 all buffers, no CUB temp

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

size_t align256(size_t b) { return ((b + 255u) / 256u) * 256u; }

// Background sampler: track min free bytes while the kernel runs. cudaMalloc is
// synchronous on the host side, so cudaMemGetInfo reflects live allocations
// immediately -- per-sample cudaDeviceSynchronize would only block the sampler
// and risk missing the HWM window (review F7). The main thread runs several
// cs_rank passes so the high-frequency poll reliably hits the HWM.
// F9: the new thread must pin the same CUDA device.
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
  int dev = 0;
  cudaError_t err = cudaGetDevice(&dev);
  if (err != cudaSuccess) { printf("cudaGetDevice FAIL\n"); return 1; }
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, dev);
  size_t free_before = 0, total = 0;
  cudaMemGetInfo(&free_before, &total);
  printf("GPU: %s, total %.0f MiB, free before %.1f MiB\n", prop.name,
         total / 1048576.0, free_before / 1048576.0);

  const size_t n = static_cast<size_t>(kT) * kN;
  std::vector<float> X(n);
  {
    Lcg rng(0x5EEDC0DEu);
    for (size_t i = 0; i < n; ++i) {
      uint32_t r = rng.next() % 100;
      if (r < 92) X[i] = rng.uniform() * 20.0f - 10.0f;
      else if (r < 96) X[i] = mk_f32(0x7fc00000u | (rng.next() & 0x1FFFFFu));
      else X[i] = (rng.next() & 1u) ? mk_f32(0x7f800000u) : mk_f32(0xff800000u);
    }
  }
  std::vector<float> out(n);

  // ---- F10 error-path smoke: guards must reject invalid input with a nonzero
  //      return code, not crash / corrupt. N>2^24 and T<1 are contract errors.
  {
    float dummy_out[4];
    int rc_big_n = cs_rank_gpu(X.data(), nullptr, 1, (1 << 24) + 1, false, dummy_out);
    int rc_zero_t = cs_rank_gpu(X.data(), nullptr, 0, 10, false, dummy_out);
    printf("error-path smoke: N>2^24 rc=%d, T=0 rc=%d (both expect nonzero)\n",
           rc_big_n, rc_zero_t);
    if (rc_big_n == 0 || rc_zero_t == 0) {
      printf("FAIL: host guards did not reject invalid input\n");
      return 1;
    }
  }

  // ---- theoretical formula (no CUB temp) -----------------------------------
  // canonical anchor 152,254,876 is the LOGICAL byte sum (IMPLEMENTATION F-06);
  // the aligned sum is what the tracker (align256) and cudaMalloc actually hold.
  size_t theory_logical = (6ull * n * 4u) + n + (static_cast<size_t>(kT) + 1) * 4u;
  size_t theory = align256(n * 4u) * 6u + align256(n) +
                  align256((static_cast<size_t>(kT) + 1) * 4u);
  printf("theoretical logical no-temp: %zu B (canonical anchor %zu) %s\n",
         theory_logical, kTheoreticalNoTemp,
         theory_logical == kTheoreticalNoTemp ? "MATCH" : "MISMATCH");
  printf("theoretical align256 no-temp: %zu B (%.2f MiB)\n", theory,
         theory / 1048576.0);
  if (theory_logical != kTheoreticalNoTemp) {
    printf("FAIL: logical theory mismatch with canonical anchor\n");
    return 1;
  }

  // ---- run with tracker + background sampler ------------------------------
  factor_cuda::MemTracker tracker;
  std::atomic<size_t> min_free{free_before};
  std::atomic<bool> stop_sampler{false};
  std::thread sampler(RunSampler, &min_free, &stop_sampler, dev);

  int rc = 0;
  for (int rep = 0; rep < 5; ++rep) {  // several passes so the sampler hits HWM
    rc = cs_rank_gpu(X.data(), nullptr, kT, kN, false, out.data(), &tracker);
    if (rc != 0) { printf("cs_rank_gpu FAIL rc=%d\n", rc); return 1; }
  }

  stop_sampler.store(true);
  sampler.join();
  cudaDeviceSynchronize();
  size_t free_after = 0;
  cudaMemGetInfo(&free_after, &total);

  // ---- tracker HWM vs theory -----------------------------------------------
  size_t hwm = tracker.peak_live_bytes();
  // temp bytes recorded by the tracker (aligned) is the only theory blind spot
  size_t temp_aligned = 0;
  for (const auto& e : tracker.events()) {
    if (e.name == "cub_temp" && e.action == "alloc") temp_aligned = e.aligned_bytes;
  }
  size_t theory_with_temp = theory + temp_aligned;
  printf("tracker HWM: %zu B (%.2f MiB), alloc events %zu\n", hwm, hwm / 1048576.0,
         tracker.alloc_count());
  printf("theory incl CUB temp (%zu B): %zu B (%.2f MiB)\n", temp_aligned,
         theory_with_temp, theory_with_temp / 1048576.0);
  printf("  tracker vs theory delta: %lld B (%+.2f MiB)\n",
         static_cast<long long>(hwm) - static_cast<long long>(theory_with_temp),
         (static_cast<double>(hwm) - static_cast<double>(theory_with_temp)) / 1048576.0);

  // ---- tracker HWM vs driver sample ----------------------------------------
  size_t driver_peak = free_before - min_free.load();  // free_before >= min_free
  printf("driver sample peak: %zu B (%.2f MiB)\n", driver_peak, driver_peak / 1048576.0);
  printf("  tracker vs driver delta: %lld B (%+.2f MiB)\n",
         static_cast<long long>(hwm) - static_cast<long long>(driver_peak),
         (static_cast<double>(hwm) - static_cast<double>(driver_peak)) / 1048576.0);
  printf("  leak check: free delta after run %.1f MiB\n",
         (static_cast<double>(free_before) - static_cast<double>(free_after)) / 1048576.0);

  // ---- verdict -------------------------------------------------------------
  const size_t kDriverTolerance = 64ull * 1024 * 1024;  // F8: bounded driver overhead
  bool ok = true;
  long long d1 = static_cast<long long>(hwm) - static_cast<long long>(theory_with_temp);
  if (d1 != 0) { ok = false; printf("FAIL: tracker != theory (should be exact, delta=%lld)\n", d1); }
  long long d2 = static_cast<long long>(hwm) - static_cast<long long>(driver_peak);
  // F8 two-sided assertion: driver sampled peak = tracker HWM + bounded driver
  // overhead (cudaMalloc real alignment + CUDA context/allocator). Must be
  // 0 <= overhead <= tolerance; a negative overhead means the sampler missed
  // the HWM, a too-large overhead means unexpected allocator/context cost.
  long long overhead = static_cast<long long>(driver_peak) - static_cast<long long>(hwm);
  printf("  driver overhead above tracker: %+.2f MiB (tolerance %u MiB)\n",
         static_cast<double>(overhead) / 1048576.0,
         static_cast<unsigned>(kDriverTolerance / 1048576));
  if (overhead < 0 || overhead > static_cast<long long>(kDriverTolerance)) {
    ok = false;
    printf("FAIL: driver overhead %lld B out of [0, %zu] (sampler missed HWM or excess cost)\n",
           overhead, kDriverTolerance);
  }
  if (tracker.live_bytes() != 0) {
    ok = false;
    printf("FAIL: tracker final live %zu != 0 (leak)\n", tracker.live_bytes());
  }
  if (tracker.unknown_free_count() != 0) {
    ok = false;  // F1/F2: strict tracker must never see an untracked pointer
    printf("FAIL: tracker saw %zu unknown non-null free (missed AllocOrTrack)\n",
           tracker.unknown_free_count());
  }

  printf("== summary ==\n");
  printf("%s\n", ok ? "ALL PASS (mem_tracker three-way calibration)" : "FAILURES");
  return ok ? 0 : 1;
}
