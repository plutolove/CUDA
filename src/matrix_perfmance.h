#pragma once
#include <iostream>

#include "tensor.h"

using GemmType = void (*)(float*, float*, float*, int, int, int);
float testPerformance(dim3 grid, dim3 block, GemmType gemm, float* a, float* b,
                      float* c, int M, int N, int K, int repeat);

struct MatrixPerfmance {
  static const int SIZE = 15;
  const int M_list[SIZE] = {128,  192,  256,  384,  512,  768,   1024, 1536,
                            2048, 3072, 4096, 6144, 8192, 12288, 16384};
  const int N_list[SIZE] = {128,  192,  256,  384,  512,  768,   1024, 1536,
                            2048, 3072, 4096, 6144, 8192, 12288, 16384};
  const int K_list[SIZE] = {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024,
                            1024, 1024, 1024, 1024, 1024, 1024, 1024};

  MatrixPerfmance(GemmType gemm_, int out_rep, int inner_rep)
      : gemm(gemm_), out_rep_num(out_rep), inner_rep_num(inner_rep) {
    lhs.reserve(SIZE);
    for (size_t i = 0; i < SIZE; i++) {
      lhs.emplace_back(std::initializer_list<int>{M_list[i], K_list[i]});
      rhs.emplace_back(std::initializer_list<int>{K_list[i], N_list[i]});
      lhs.back().random_uniform();
      rhs.back().random_uniform();
      result.emplace_back(std::initializer_list<int>{M_list[i], N_list[i]});
    }
  }

  template <typename F1, typename F2>
  void operator()(F1&& get_grid, F2&& get_block) {
    for (int i = 0; i < SIZE; i++) {
      const int M = M_list[i], N = N_list[i], K = K_list[i];

      // dim3 blockDim(BN, BM);
      // dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
      auto blockDim = get_block();
      auto gridDim = get_grid(N, M);

      double max_sec = 0.0;
      double min_sec = 1000;
      double total_sec = 0.0;

      auto* a = lhs[i].mutable_data();
      auto* b = rhs[i].mutable_data();
      auto* c = result[i].mutable_data();

      for (int j = 0; j < out_rep_num; j++) {
        double this_sec = testPerformance(gridDim, blockDim, gemm, a, b, c, M,
                                          N, K, inner_rep_num);
        max_sec = std::max(max_sec, this_sec);
        min_sec = std::min(min_sec, this_sec);
        total_sec += this_sec;
      }

      double avg_sec = total_sec / out_rep_num;
      double avg_Gflops =
          ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;

      std::cerr
          << fmt::format(
                 "M={}, N={}, K={}, time={} {} {}, AVG Performance={} Gflops",
                 M, N, K, min_sec, avg_sec, max_sec, avg_Gflops)
          << std::endl;
      // printf(
      //     "M N K = %6d %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG "
      //     "Performance = %10.4lf Gflops\n",
      //     M, N, K, min_sec, avg_sec, max_sec, avg_Gflops);
    }
  }

  GemmType gemm;
  int out_rep_num{1};
  int inner_rep_num{1};
  std::vector<GPUTensor<float>> lhs;
  std::vector<GPUTensor<float>> rhs;
  std::vector<GPUTensor<float>> result;
};

