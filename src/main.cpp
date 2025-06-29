#include <cstring>
#include <iostream>

#include "cuda_runtime.h"
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

void naiveSgemm_cpp(float* a, float* b, float* c, int M, int N, int K);

int main() {
    GPUTensor<float> lhs({16, 64});
    GPUTensor<float> rhs({64, 32});

    GPUTensor<float> result({16, 32});
    lhs.random_uniform();
    rhs.random_uniform();

    naiveSgemm_cpp(lhs.mutable_data(), rhs.mutable_data(),
                   result.mutable_data(), 16, 32, 64);

    // std::cerr << lhs.to_string() << std::endl;
    // std::cerr << rhs.to_string() << std::endl;
    std::cerr << result.to_string() << std::endl;
    return 0;
}
