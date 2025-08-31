#pragma once
#include <iostream>

#include "common/macro.h"
#include "fmt/format.h"
#include "tensor.cuh"

template <typename T>
using GemmType = void (*)(T*, T*, T*, int, int, int);

template <typename T>
float testPerformance(dim3 grid, dim3 block, GemmType<T> gemm, float* a,
                      float* b, float* c, int M, int N, int K, int repeat) {
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

template <typename T, size_t BM = 32, size_t BN = 32>
struct MatrixPerfmance {
  static const int SIZE = 6;
  static const int Loop = 6;
  const int M_list[SIZE] = {8, 1024, 1536, 2048, 3072, 4096};

  MatrixPerfmance(GemmType<T> gemm_, std::string name, int out_rep = 1,
                  int inner_rep = 1)
      : gemm(gemm_),
        name(std::move(name)),
        out_rep_num(out_rep),
        inner_rep_num(inner_rep) {
    lhs.reserve(SIZE);
    for (size_t i = 0; i < SIZE; i++) {
      lhs.emplace_back(std::initializer_list<int>{M_list[i], M_list[i]});
      rhs.emplace_back(std::initializer_list<int>{M_list[i], M_list[i]});
      lhs.back().random_uniform();
      rhs.back().random_uniform();
      result.emplace_back(std::initializer_list<int>{M_list[i], M_list[i]});
    }
  }

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

  void operator()() {
    std::cerr << fmt::format("{}:", name) << std::endl;
    for (int i = 0; i < Loop; i++) {
      const int M = M_list[i], N = M_list[i], K = M_list[i];

      dim3 blockDim(BN, BM);
      dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));

      double max_sec = 0.0;
      double min_sec = 1000;
      double total_sec = 0.0;

      auto* a = lhs[i].mutable_data();
      // auto a_cpu = lhs[i].to_host();
      auto* b = rhs[i].mutable_data();
      // auto b_cpu = lhs[i].to_host();
      auto* c = result[i].mutable_data();
      // auto c_cpu = result[i].to_host();

      // cpuSgemm(a_cpu.data(), b_cpu.data(), c_cpu.data(), M, N, K);
      // std::cerr << fmt::format("[{}]", fmt::join(c_cpu, ",")) << std::endl;

      for (int j = 0; j < out_rep_num; j++) {
        double this_sec = testPerformance(gridDim, blockDim, gemm, a, b, c, M,
                                          N, K, inner_rep_num);
        max_sec = std::max(max_sec, this_sec);
        min_sec = std::min(min_sec, this_sec);
        total_sec += this_sec;
      }
      // std::cerr << result[i].to_string() << std::endl;

      double avg_sec = total_sec / out_rep_num;
      double avg_Gflops =
          ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;

      std::cerr << fmt::format(
                       "    M={}, N={}, K={}, time={} {} {}, AVG "
                       "Performance={} Gflops",
                       M, N, K, min_sec, avg_sec, max_sec, avg_Gflops)
                << std::endl;
    }
  }

  GemmType<T> gemm;
  std::string name;
  int out_rep_num{1};
  int inner_rep_num{1};
  std::vector<GPUTensor<T>> lhs;
  std::vector<GPUTensor<T>> rhs;
  std::vector<GPUTensor<T>> result;
};

