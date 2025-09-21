#pragma once
#include <functional>
#include <iostream>

#include "common/macro.h"
#include "tensor.h"

template <typename T>
float testPerformance(dim3 grid, dim3 block, GemmType<T> gemm, float* a,
                      float* b, float* c, int M, int N, int K, int repeat);

template <typename T, size_t BM = 32, size_t BN = 32, size_t TM = 8,
          size_t TN = 8>
struct MatrixPerfmance {
  static const int SIZE = 6;
  static const int Loop = 6;
  const int M_list[SIZE] = {256, 1024, 1536, 2048, 3072, 4096};

  MatrixPerfmance(
      std::function<dim3(int, int, int, int, int, int, int)> grid_func,
      std::function<dim3(int, int, int, int, int, int, int)> block_func,
      GemmType<T> gemm_, const std::string& name, int out_rep = 1,
      int inner_rep = 1)
      : grid_dim_func(std::move(grid_func)),
        block_dim_func(std::move(block_func)),
        gemm(gemm_),
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

  MatrixPerfmance(std::function<dim3(int, int, int, int, int, int, int)> func,
                  GemmType<T> gemm_, const std::string& name, int out_rep = 1,
                  int inner_rep = 1)
      : grid_dim_func(std::move(func)),
        gemm(gemm_),
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

  MatrixPerfmance(GemmType<T> gemm_, const std::string& name, int out_rep = 1,
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

  void cpuSgemm(float* a, float* b, float* c, int M, int K, int N) const {
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

  void debug_cpu(const GPUTensor<T>& a, const GPUTensor<T>& b,
                 const GPUTensor<T>& c, int M, int K, int N) const {
    auto a_cpu = a.to_host();
    auto b_cpu = b.to_host();
    auto c_cpu = c.to_host();
    cpuSgemm(a_cpu.data(), b_cpu.data(), c_cpu.data(), M, N, K);
    std::cerr << detail::to_string(c_cpu) << std::endl;
  }

  void operator()() {
    std::cerr << fmt::format("{}:", name) << std::endl;
    for (int i = 0; i < Loop; i++) {
      const int M = M_list[i], N = M_list[i], K = M_list[i];

      dim3 blockDim(BN, BM);
      dim3 gridDim(CEIL_DIV(M, BM), CEIL_DIV(N, BN));
      if (grid_dim_func) {
        gridDim = grid_dim_func(M, N, K, BM, BN, TM, TN);
      }

      if (block_dim_func) {
        blockDim = block_dim_func(M, N, K, BM, BN, TM, TN);
      }

      double max_sec = 0.0;
      double min_sec = 1000;
      double total_sec = 0.0;

      auto* a = lhs[i].mutable_data();
      auto* b = rhs[i].mutable_data();
      auto* c = result[i].mutable_data();

      if (enable_debug) {
        debug_cpu(lhs[i], rhs[i], result[i], M, N, K);
      }

      for (int j = 0; j < out_rep_num; j++) {
        double this_sec = testPerformance(gridDim, blockDim, gemm, a, b, c, M,
                                          N, K, inner_rep_num);
        max_sec = std::max(max_sec, this_sec);
        min_sec = std::min(min_sec, this_sec);
        total_sec += this_sec;
      }
      if (enable_debug) {
        std::cerr << result[i].to_string() << std::endl;
      }

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

  std::function<dim3(int, int, int, int, int, int, int)> grid_dim_func;
  std::function<dim3(int, int, int, int, int, int, int)> block_dim_func;
  GemmType<T> gemm;
  std::string name;
  int out_rep_num{1};
  int inner_rep_num{1};
  std::vector<GPUTensor<T>> lhs;
  std::vector<GPUTensor<T>> rhs;
  std::vector<GPUTensor<T>> result;
  bool enable_debug{false};
};

