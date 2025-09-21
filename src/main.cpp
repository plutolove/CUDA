#include "common/macro.h"
#include "matrix_perfmance.h"

void naiveSgemm(float* a, float* b, float* c, const int M, const int N,
                const int K);
void SgemmCoalescing(float* a, float* b, float* c, const int M, const int N,
                     const int K);

void SgemmCoalescing_v1(float* a, float* b, float* c, const int M, const int N,
                        const int K);

void sharedMemGemm(float* a, float* b, float* c, const int M, const int N,
                   const int K);

void sharedMemGemm_v1(float* a, float* b, float* c, const int M, const int N,
                      const int K);

int main() {
  MatrixPerfmance<float> test_perf(&naiveSgemm, "naive");
  MatrixPerfmance<float> test_perf1(&SgemmCoalescing, "coalescing");
  MatrixPerfmance<float> test_perf2(
      [](int M, int N, int /*unused*/, int BM, int BN, int TM, int TN) {
        return dim3(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
      },
      &SgemmCoalescing_v1, "coalescing_v1");
  MatrixPerfmance<float> test_perf3(
      [](int M, int N, int /*unused*/, int BM, int BN, int TM, int TN) {
        return dim3(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
      },
      &sharedMemGemm, "shared_mem_gemm");

  MatrixPerfmance<float, 128, 128> test_perf4(
      [](int M, int N, int /*unused*/, int BM, int BN, int TM, int TN) {
        return dim3(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
      },
      [](int M, int N, int /*unused*/, int BM, int BN, int TM, int TN) {
        return dim3(CEIL_DIV(BN, TN), CEIL_DIV(BM, TM));
      },
      &sharedMemGemm_v1, "shared_mem_gemm_v1");

  // test_perf();
  // test_perf1();
  // test_perf2();
  // test_perf3();
  test_perf4();
  return 0;
}
