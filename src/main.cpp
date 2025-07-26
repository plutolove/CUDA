#include <cuda_runtime.h>

#include "matrix_perfmance.h"
#include "tensor.h"

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
void sgemm_V1(float* a, float* b, float* c, int M, int N, int K);

const int BM = 64;
const int BN = 64;
const int TN = 8;
const int TM = 8;

int main() {
  MatrixPerfmance test_perf(&sgemm_V1, 3, 3);
  test_perf(
      [](int N, int M) {
        // dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
        dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
        return gridDim;
      },
      []() {
        // dim3 blockDim(BN, BM);
        dim3 blockDim(BN / TN, BM / TM);
        return blockDim;
      });
  return 0;
}
