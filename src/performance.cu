#include "matrix_perfmance.h"

float testPerformance(dim3 grid, dim3 block, GemmType gemm, float* a, float* b,
                      float* c, int M, int N, int K, int repeat) {
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start);

  for (int i = 0; i < repeat; i++) {
    gemm<<<grid, block>>>(a, b, c, M, N, K);
  }

  cudaEventRecord(end);
  cudaEventSynchronize(end);

  float msec, sec;
  cudaEventElapsedTime(&msec, start, end);
  sec = msec / 1000.0 / repeat;

  return sec;
}
