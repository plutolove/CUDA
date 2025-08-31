#include <cuda_runtime.h>

__global__ void naiveSgemm(float* a, float* b, float* c, const int M,
                           const int N, const int K) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  // a -> M * K
  // b -> K * N
  // c -> M * N
  if (x < M && y < N) {
    float sum = 0.0;
    for (int k = 0; k < K; k++) {
      sum += a[x * K + k] * b[k * N + y];
    }
    c[x * N + y] = sum;
  }
}

__global__ void SgemmCoalescing(float* a, float* b, float* c, const int M,
                                const int N, const int K) {
  auto thread_idx = blockDim.x * threadIdx.y + threadIdx.x;
  int x = blockIdx.x * blockDim.x + thread_idx / blockDim.x;
  int y = blockIdx.y * blockDim.y + thread_idx % blockDim.x;
  // a -> M * K
  // b -> K * N
  // c -> M * N
  if (x < M && y < N) {
    float sum = 0.0;
    for (int k = 0; k < K; k++) {
      sum += a[x * K + k] * b[k * N + y];
    }
    c[x * N + y] = sum;
  }
}
