#include <cuda_runtime.h>

#include "common/macro.h"

const int BlockSize = 32;

__global__ void sharedMemGemm(float* a, float* b, float* c, const int M,
                              const int N, const int K) {
  auto thread_idx = blockDim.x * threadIdx.y + threadIdx.x;
  int block_x = blockIdx.x;
  int block_y = blockIdx.y;
  __shared__ float As[BlockSize][BlockSize];
  __shared__ float Bs[BlockSize][BlockSize];
  int thread_x = thread_idx / BlockSize;
  int thread_y = thread_idx % BlockSize;
  for (int bkidx = 0; bkidx < CEIL_DIV(K, BlockSize); bkidx++) {
    {
      // a -> M * K
      if (block_x * BlockSize + thread_x < M &&
          bkidx * BlockSize + thread_x < K) {
        As[thread_x][thread_y] = a[(block_x * BlockSize + thread_x) * K +
                                   bkidx * BlockSize + thread_x];
      } else {
        As[thread_x][thread_y] = 0;
      }
    }
  }
}

