#include <cuda_runtime.h>

#include "common/macro.h"

const int BlockSize = 32;

__global__ void sharedMemGemm(float* a, float* b, float* c, const int M,
                              const int N, const int K) {
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ float As[BlockSize][BlockSize];
  __shared__ float Bs[BlockSize][BlockSize];
  float sum = 0;
  for (int bkidx = 0; bkidx < CEIL_DIV(K, BlockSize); bkidx++) {
    As[threadIdx.y][threadIdx.x] = a[y * K + bkidx * BlockSize + threadIdx.x];
    Bs[threadIdx.y][threadIdx.x] = b[(bkidx * BlockSize + threadIdx.y) * N + x];
    __syncthreads();
    for (size_t k = 0; k < BlockSize; k++) {
      sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }
    __syncthreads();
  }
  c[y * N + x] = sum;
}
