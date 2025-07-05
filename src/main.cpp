#include <cstring>
#include <iostream>
#include <memory>
#include <random>

#include "folly/executors/CPUThreadPoolExecutor.h"
#include "folly/futures/Future.h"
#include "pin_mem_allocator.h"
#include "tensor.h"

// #define OFFSET(row, col, ld) ((row) * (ld) + (col))

// float testPerformance(void (*gemm)(float*, float*, float*, int, int, int),
//                       float* d_a, float* d_b, float* d_c, const int M,
//                       const int N, const int K, const int repeat);

// void cpuSgemm(float* a, float* b, float* c, int M, int K, int N) {
//     for (int m = 0; m < M; m++) {
//         for (int n = 0; n < N; n++) {
//             float sum = 0;
//             for (int k = 0; k < K; k++) {
//                 sum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
//             }
//             c[OFFSET(m, n, N)] = sum;
//         }
//     }
// }

// void naiveSgemm_cpp(float* a, float* b, float* c, int M, int N, int K);

// int main() {
//     GPUTensor<float> lhs({512, 512});
//     GPUTensor<float> rhs({512, 512});

//     GPUTensor<float> result({512, 512});
//     lhs.random_uniform();
//     rhs.random_uniform();

//     auto tc =
//         testPerformance(&naiveSgemm_cpp, lhs.mutable_data(),
//         rhs.mutable_data(),
//                         result.mutable_data(), 512, 512, 512, 10);

//     std::cerr << tc << std::endl;
//     // std::cerr << lhs.to_string() << std::endl;
//     // std::cerr << rhs.to_string() << std::endl;
//     std::cerr << result.to_string() << std::endl;
//     return 0;
// }

auto getThreadPool() {
    auto pool = std::make_shared<folly::CPUThreadPoolExecutor>(
        64, std::make_shared<folly::NamedThreadFactory>("test"));
    return pool;
}

size_t random_size() {
    static std::random_device rd;
    static std::mt19937 generator(rd());
    static std::uniform_int_distribution<int> distribution(1, 128);
    return distribution(generator);
}

int main() {
    auto pool = getThreadPool();

    for (size_t rep = 0; rep < 50; rep++) {
        std::vector<folly::Future<void*>> tasks;
        tasks.reserve(64);
        for (size_t i = 0; i < 64; i++) {
            tasks.push_back(folly::via(pool.get()).thenValue([](auto&&) {
                return PinMemAllocator::alloc(random_size());
            }));
        }
        std::cerr << "------- wait" << std::endl;
        auto result = folly::collectAll(tasks).get();
        std::cerr << "------- wait finished: " << result.size() << std::endl;
        for (auto&& val : result) {
            std::cerr << val.value() << std::endl;
        }

        std::cerr << "before free: " << result.size() << " "
                  << PinMemAllocator::getAllocBlockNum() << std::endl;
        for (auto&& val : result) {
            PinMemAllocator::free(val.value());
        }

        std::cerr << "after free: " << result.size() << " "
                  << PinMemAllocator::getAllocBlockNum() << std::endl;
    }
    return 0;
}
