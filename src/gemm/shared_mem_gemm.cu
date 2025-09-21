#include <cuda_runtime.h>

#include "common/macro.h"

const int BlockSize = 32;

// thread block(32, 32)
__global__ void sharedMemGemm(float* a, float* b, float* c, const int M,
                              const int N, const int K) {
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ float As[BlockSize][BlockSize];
  __shared__ float Bs[BlockSize][BlockSize];
  float sum = 0;
  for (int bkidx = 0; bkidx < CEIL_DIV(K, BlockSize); bkidx++) {
    As[threadIdx.y][threadIdx.x] = a[y * K + bkidx * BlockSize + threadIdx.x];
    Bs[threadIdx.y][threadIdx.x] = b[(bkidx * BlockSize + threadIdx.y) * N + x];
    __syncthreads();
    for (size_t k = 0; k < BlockSize; k++) {
      sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }
    __syncthreads();
  }
  c[y * N + x] = sum;
}

// BK:TILE_K=8 BM=BN=128
// TM=TN=8 增加计算密度 BM/TM=16 BN/TN=16
// dim3 blockDim(BN/TN, BM/TM) -> (16, 16)
// dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM)
__global__ void sharedMemGemm_v1(float* a, float* b, float* c, const int M,
                                 const int N, const int K) {
  static const int TM = 8;
  static const int TN = 8;

  static const int BK = 8;
  static const int BM = 128;
  static const int BN = 128;

  auto by = blockIdx.y;
  auto bx = blockIdx.x;
  auto ty = threadIdx.y;
  auto tx = threadIdx.x;

  auto tidx = ty * blockDim.x + tx;

  // blockTile 128 * 8 总共 1024条数据
  // blockDim(16, 16) 256个thread
  // 一个thread读4条数据
  // 128行，每行8条数据, 一行两个thread
  auto load_smem_a_m = tidx / 2;
  auto load_smem_a_k = tidx & 1 ? 0 : 4;

  // blockTile 8 * 128 总共1024条数据
  // blockDim(16, 16) 256个thread
  // 一个thread读4条数据
  // 8行，每行128条数据, 一行32个thread
  auto load_smem_b_k = tidx / 32;
  auto load_smem_b_n = tidx % 32 * 4;

  auto load_gmem_a_m = by * BM + load_smem_a_m;
  auto load_gmem_b_n = bx * BN + load_smem_b_n;

  if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;

  __shared__ float As[BM][BK];
  __shared__ float Bs[BK][BN];
  float Tc[TM][TN] = {0.0};
  float sum = 0;
  for (int bkidx = 0; bkidx < CEIL_DIV(K, BK); bkidx++) {
    auto load_gmem_a_k = bkidx * BK + load_smem_a_k;
    auto load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    FLOAT4(As[load_smem_a_m][load_smem_a_k]) = FLOAT4(a[load_gmem_a_addr]);
    auto load_gmem_b_k = bkidx * BK + load_smem_b_k;
    auto load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
    FLOAT4(Bs[load_smem_b_k][load_smem_b_n]) = FLOAT4(b[load_gmem_b_addr]);
    __syncthreads();
    for (int k = 0; k < BK; ++k) {
      for (int m = 0; m < TM; ++m) {
        for (int n = 0; n < TN; ++n) {
          Tc[m][n] += As[ty * TM + m][k] * Bs[k][tx * TN + n];
        }
      }
    }
    __syncthreads();
  }
  for (int m = 0; m < TM; m++) {
    auto store_gmem_m = by * BM + ty * TM + m;
    for (int n = 0; n < TN; n++) {
      auto store_gmem_n = bx * BN + tx * TN + n;
      c[store_gmem_m * N + store_gmem_n] = Tc[m][n];
    }
  }
}
