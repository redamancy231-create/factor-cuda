// Phase 0 self-check: verify nvcc x MSVC host compat + GPU availability + basic kernel
// Reference: PLAN.md Sec 7 Phase 0 (GPU vector add -> numpy check)
// Build: after vcvars64, `nvcc -arch=sm_89 phase0_selfcheck.cu -o phase0_selfcheck.exe`
#include <cstdio>
#include <cuda_runtime.h>

__global__ void vec_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int N = 1 << 20;  // 1,048,576

    // 1. GPU availability
    int dev = 0;
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) {
        printf("cudaGetDevice FAIL: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("GPU: %s (compute capability %d.%d), SM count %d\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    // 2. vector add 1.0 + 2.0 = 3.0
    float *h_a = new float[N], *h_b = new float[N], *h_c = new float[N];
    for (int i = 0; i < N; i++) { h_a[i] = 1.0f; h_b[i] = 2.0f; }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    vec_add<<<(N + 255) / 256, 256>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("kernel FAIL: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (h_c[i] != 3.0f) { ok = false; break; }
    }
    printf("vec_add: %s (1.0+2.0=3.0, N=%d)\n", ok ? "PASS" : "FAIL", N);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    delete[] h_a; delete[] h_b; delete[] h_c;
    return ok ? 0 : 1;
}
