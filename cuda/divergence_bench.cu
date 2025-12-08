#include <cstdio>
#include <cuda_runtime.h>

__global__ void divergence_kernel(int* out, int iters, bool divergent) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int acc = 0;

    for (int i = 0; i < iters; ++i) {
        if (divergent) {
            if (threadIdx.x & 1) {
                acc += i;
            } else {
                acc -= i;
            }
        } else {
            // No divergence: all threads take same path
            acc += i;
        }
    }
    out[tid] = acc;
}

int main() {
    const int iters = 1<<20;
    const int threads = 256;
    const int blocks = 80;
    int total_threads = threads * blocks;

    int *d_out;
    cudaMalloc(&d_out, total_threads * sizeof(int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Non-divergent
    cudaEventRecord(start);
    divergence_kernel<<<blocks, threads>>>(d_out, iters, false);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_nd = 0;
    cudaEventElapsedTime(&ms_nd, start, stop);

    // Divergent
    cudaEventRecord(start);
    divergence_kernel<<<blocks, threads>>>(d_out, iters, true);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_d = 0;
    cudaEventElapsedTime(&ms_d, start, stop);

    double instrs = (double)iters * total_threads;
    printf("Non-divergent: %.3f ms, approx instrs/s = %.3e\n",
           ms_nd, instrs / (ms_nd / 1e3));
    printf("Divergent:     %.3f ms, approx instrs/s = %.3e\n",
           ms_d, instrs / (ms_d / 1e3));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_out);
    return 0;
}
