#include "common/macro.h"
#include "matrix_perfmance.h"

void naiveSgemm(float* a, float* b, float* c, const int M, const int N,
                const int K);
void SgemmCoalescing(float* a, float* b, float* c, const int M, const int N,
                     const int K);

void SgemmCoalescing_v1(float* a, float* b, float* c, const int M, const int N,
                        const int K);

int main() {
  MatrixPerfmance<float> test_perf(&naiveSgemm, "naive");
  MatrixPerfmance<float> test_perf1(&SgemmCoalescing, "coalescing");
  MatrixPerfmance<float> test_perf2(
      [](int M, int N, int /*unused*/, int BM, int BN) {
        return dim3(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
      },
      &SgemmCoalescing_v1, "coalescing_v1");
  test_perf();
  test_perf1();
  test_perf2();
  return 0;
}
