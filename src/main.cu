#include <cuda_runtime.h>

#include "matrix_perfmance.cuh"
#include "tensor.cuh"

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

void cpuSgemm(float* a, float* b, float* c, int M, int K, int N) {
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      float sum = 0;
      for (int k = 0; k < K; k++) {
        sum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
      }
      c[OFFSET(m, n, N)] = sum;
    }
  }
}

void naiveSgemm(float* a, float* b, float* c, int M, int N, int K);
void SgemmCoalescing(float* a, float* b, float* c, int M, int N, int K);

int main() {
  MatrixPerfmance<float> test_perf(&naiveSgemm, "naive");
  MatrixPerfmance<float> test_perf1(&SgemmCoalescing, "coalescing");
  test_perf();
  test_perf1();
  return 0;
}
