#include <cstdio>
#include <cuda_runtime.h>

__global__ void stride_kernel(const int* __restrict__ A,
                              int* __restrict__ out,
                              int stride,
                              int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = tid * stride;
    int acc = 0;
    for (int i = 0; i < iters; ++i) {
        acc += A[(idx + i * stride) & ((1 << 20) - 1)]; // wrap
    }
    out[tid] = acc;
}

int main() {
    const int N = 1 << 20;
    const int iters = 1024;
    const int threads = 256;
    const int blocks = 80;

    int *d_A, *d_out;
    cudaMalloc(&d_A, N * sizeof(int));
    cudaMalloc(&d_out, blocks * threads * sizeof(int));
    cudaMemset(d_A, 1, N * sizeof(int));

    int strides[] = {1, 2, 4, 8, 16, 32, 64};
    for (int s : strides) {
        cudaDeviceSynchronize();
        float ms = 0;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        stride_kernel<<<blocks, threads>>>(d_A, d_out, s, iters);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);

        int total_threads = blocks * threads;
        double total_loads = (double)total_threads * iters;
        double loads_per_ms = total_loads / ms;
        double loads_per_s = loads_per_ms * 1e3;

        printf("Stride %d: time = %.3f ms, loads/s = %.3e\n",
               s, ms, loads_per_s);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    cudaFree(d_A);
    cudaFree(d_out);
    return 0;
}
