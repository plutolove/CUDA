#include <iostream>

#include "cuda_runtime.h"

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

const int BM = 32, BN = 32;

__global__ void naiveSgemm(float* __restrict__ a, float* __restrict__ b,
                           float* __restrict__ c, const int M, const int N,
                           const int K) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m < M && n < N) {
        float psum = 0.0;
        // #pragma unroll
        for (int k = 0; k < K; k++) {
            psum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
        }
        c[OFFSET(m, n, N)] = psum;
    }
}

void naiveSgemm_cpp(float* a, float* b, float* c, int M, int N, int K) {
    dim3 blockDim(BN, BM);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
    naiveSgemm<<<gridDim, blockDim>>>(a, b, c, M, N, K);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err)
                  << std::endl;
    }
    cudaDeviceSynchronize();
}
