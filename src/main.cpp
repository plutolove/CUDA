#include "matrix_perfmance.h"

void naiveSgemm(float* a, float* b, float* c, const int M, const int N,
                const int K);
void SgemmCoalescing(float* a, float* b, float* c, const int M, const int N,
                     const int K);

int main() {
  MatrixPerfmance<float> test_perf(&naiveSgemm, "naive");
  MatrixPerfmance<float> test_perf1(&SgemmCoalescing, "coalescing");
  test_perf();
  test_perf1();
  return 0;
}
