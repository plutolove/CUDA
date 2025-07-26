#include <cstdio>
#include <iostream>
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

const int BM = 32, BN = 32;

__global__ void naiveSgemm(float* __restrict__ a, float* __restrict__ b,
                           float* __restrict__ c, const int M, const int N,
                           const int K) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;
  if (m < M && n < N) {
    float psum = 0.0;
    // #pragma unroll
    for (int k = 0; k < K; k++) {
      psum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
    }
    c[OFFSET(m, n, N)] = psum;
  }
}

#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
__global__ void sgemm_V1(float* __restrict__ a, float* __restrict__ b,
                         float* __restrict__ c, const int M, const int N,
                         const int K) {
  const int BM = 64;
  const int BN = 64;
  const int BK = 8;
  const int TM = 8;
  const int TN = 8;

  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;

  __shared__ float s_a[BM][BK];
  __shared__ float s_b[BK][BN];

  float r_c[TM][TN] = {0.0};

  int load_a_smem_m = tid >> 1;         // tid/2, row of s_a
  int load_a_smem_k = (tid & 1) << 2;   // (tid % 2 == 0) ? 0 : 4, col of s_a
  int load_b_smem_k = tid >> 5;         // tid/32, row of s_b
  int load_b_smem_n = (tid & 31) << 2;  // (tid % 32) * 4, col of s_b

  int load_a_gmem_m = by * BM + load_a_smem_m;  // global row of a
  int load_b_gmem_n = bx * BN + load_b_smem_n;  // global col of b

  for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
    int load_a_gmem_k = bk * BK + load_a_smem_k;  // global col of a
    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
    FLOAT4(s_a[load_a_smem_m][load_a_smem_k]) = FLOAT4(a[load_a_gmem_addr]);
    int load_b_gmem_k = bk * BK + load_b_smem_k;  // global row of b
    int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
    FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(b[load_b_gmem_addr]);

    __syncthreads();

#pragma unroll
    for (int k = 0; k < BK; k++) {
#pragma unroll
      for (int m = 0; m < TM; m++) {
#pragma unroll
        for (int n = 0; n < TN; n++) {
          int comp_a_smem_m = ty * TM + m;
          int comp_b_smem_n = tx * TN + n;
          r_c[m][n] += s_a[comp_a_smem_m][k] * s_b[k][comp_b_smem_n];
        }
      }
    }

    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; i++) {
    int store_c_gmem_m = by * BM + ty * TM + i;
#pragma unroll
    for (int j = 0; j < TN; j += 4) {
      int store_c_gmem_n = bx * BN + tx * TN + j;
      int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
      FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][j]);
    }
  }
}
#undef OFFSET
#undef FLOAT4
