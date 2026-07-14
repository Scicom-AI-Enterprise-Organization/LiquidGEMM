// Fused W4A8 GEMM (v1, dp4a). Computes  Y = X @ Qhat_W^T  with the LiquidQuant weights
// read from GMEM as true 4-bit (nibble-packed), dequantized in registers (IMAD+XOR), and
// accumulated with __dp4a int8 MACs. This captures the W4A8 memory-traffic win — weights
// never round-trip through GMEM as int8 — which is the durable decode/memory-bound
// advantage on H20. Tensor-core (WGMMA) path for large-M/prefill is added separately.
//
//   X        : [M, K]     int8   (per-token quantized activations)
//   qweight  : [N, K/2]   uint8  (nibble-packed UINT4: byte = w_lo | w_hi<<4)
//   s_u8     : [N, G]     uint8  (per-group integer step, G = K/group_size)
//   offset_a : [N, G]     uint8  (per-group a = 128 + min)
//   s1       : [N]        float  (per-output-channel level-1 scale)
//   ascale   : [M]        float  (per-token activation scale)
//   Y        : [M, N]     float  (Y[m,n] = ascale[m]*s1[n]*sum_k X[m,k]*Qhat_W[n,k])

#include <torch/all.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace liquidgemm {

// Expand two packed bytes (4 nibbles) into a uint32 with one UINT4 value per byte lane,
// in weight order (matches pack_nibbles: byte = w_even | w_odd<<4).
__device__ __forceinline__ uint32_t expand2(uint32_t h) {
  uint32_t b0 = h & 0xFF, b1 = (h >> 8) & 0xFF;
  return (b0 & 0xF) | ((b0 >> 4) << 8) | ((b1 & 0xF) << 16) | ((b1 >> 4) << 24);
}

// ---- Small-M (decode) kernel: one warp streams one output column n, coalesced. ----
// Bandwidth-bound: each lane reads a 128-bit chunk of 4-bit weights, dequants in registers
// (IMAD+XOR), and dp4a-accumulates against the (L2-resident) activation rows.
constexpr int GEMV_MAXM = 16;
constexpr int GEMV_THREADS = 256;
constexpr int GEMV_WARPS = GEMV_THREADS / 32;

template <bool CACHE_X>
__global__ void w4a8_gemv_kernel(
    const int8_t* __restrict__ X, const uint8_t* __restrict__ qweight,
    const uint8_t* __restrict__ s_u8, const uint8_t* __restrict__ offset_a,
    const float* __restrict__ s1, const float* __restrict__ ascale,
    float* __restrict__ Y, int M, int N, int K, int G) {
  const int Kh = K >> 1;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps_per_block = blockDim.x >> 5;
  const int total_warps = gridDim.x * warps_per_block;

  // Optionally stage the (small) activation tile in shared memory so the block's warps
  // read X from smem instead of hammering L2 once per output column.
  extern __shared__ int8_t xsh[];
  if (CACHE_X) {
    for (int i = threadIdx.x; i < M * K; i += blockDim.x) xsh[i] = X[i];
    __syncthreads();
  }
  const int8_t* Xp = CACHE_X ? xsh : X;

  for (int n = blockIdx.x * warps_per_block + warp; n < N; n += total_warps) {
    int acc[GEMV_MAXM];
#pragma unroll
    for (int m = 0; m < GEMV_MAXM; ++m) acc[m] = 0;

    // Each lane owns 32 weights per step; the warp covers 32*32=1024 weights/step.
    for (int kc = lane * 32; kc < K; kc += 32 * 32) {
      const int g = kc >> 6;  // group (group_size == 64); 32 weights stay within one group
      const uint32_t s = s_u8[n * G + g];
      const uint32_t a_word = (uint32_t)offset_a[n * G + g] * 0x01010101u;
      const uint4 pkv = *reinterpret_cast<const uint4*>(&qweight[n * Kh + (kc >> 1)]);
      const uint32_t pk[4] = {pkv.x, pkv.y, pkv.z, pkv.w};
      int wq[8];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        wq[2 * i]     = (int)((expand2(pk[i] & 0xFFFF) * s + a_word) ^ 0x80808080u);
        wq[2 * i + 1] = (int)((expand2(pk[i] >> 16) * s + a_word) ^ 0x80808080u);
      }
      for (int m = 0; m < M; ++m) {
        const int* xr = reinterpret_cast<const int*>(&Xp[m * K + kc]);
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[m] = __dp4a(xr[j], wq[j], acc[m]);
      }
    }
#pragma unroll
    for (int m = 0; m < GEMV_MAXM; ++m) {
      if (m < M) {
        int v = acc[m];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
        if (lane == 0) Y[m * N + n] = (float)v * ascale[m] * s1[n];
      }
    }
  }
}

constexpr int BM = 16;   // output rows per block tile
constexpr int BN = 64;   // output cols per block tile
constexpr int BK = 64;   // reduction chunk == group_size (one group per chunk)
constexpr int THREADS = 256;
constexpr int OUT_PER_THREAD = (BM * BN) / THREADS;  // = 4

__global__ void w4a8_gemm_dp4a_kernel(
    const int8_t* __restrict__ X,        // [M, K]
    const uint8_t* __restrict__ qweight, // [N, K/2]
    const uint8_t* __restrict__ s_u8,    // [N, G]
    const uint8_t* __restrict__ offset_a,// [N, G]
    const float* __restrict__ s1,        // [N]
    const float* __restrict__ ascale,    // [M]
    float* __restrict__ Y,               // [M, N]
    int M, int N, int K, int G) {
  __shared__ int8_t xs[BM * BK];
  __shared__ int8_t ws[BN * BK];

  const int m0 = blockIdx.y * BM;
  const int n0 = blockIdx.x * BN;
  const int tid = threadIdx.x;

  // Each thread owns OUT_PER_THREAD outputs; precompute their (m,n) within the tile.
  int mloc[OUT_PER_THREAD], nloc[OUT_PER_THREAD];
  int acc[OUT_PER_THREAD];
#pragma unroll
  for (int o = 0; o < OUT_PER_THREAD; ++o) {
    int t = tid + o * THREADS;   // in [0, BM*BN)
    mloc[o] = t / BN;
    nloc[o] = t % BN;
    acc[o] = 0;
  }

  const int Kh = K >> 1;         // packed bytes per weight row
  for (int k0 = 0; k0 < K; k0 += BK) {
    const int g = k0 / BK;       // group index (BK == group_size)

    // Stage X tile [BM, BK] -> smem.
    for (int i = tid; i < BM * BK; i += THREADS) {
      int m = i / BK, k = i % BK;
      xs[i] = (m0 + m < M) ? X[(m0 + m) * K + (k0 + k)] : (int8_t)0;
    }
    // Stage + dequant W tile [BN, BK] -> smem (read 4-bit, IMAD+XOR per nibble pair).
    for (int i = tid; i < BN * (BK / 2); i += THREADS) {
      int j = i / (BK / 2);         // row within tile
      int kb = i % (BK / 2);        // packed-byte within chunk
      int n = n0 + j;
      uint32_t s = 0, a = 0, pb = 0;
      if (n < N) {
        s = s_u8[n * G + g];
        a = offset_a[n * G + g];
        pb = qweight[n * Kh + (k0 / 2) + kb];
      }
      uint32_t w_lo = pb & 0xF;
      uint32_t w_hi = (pb >> 4) & 0xF;
      // Qhat = (u4*s + a) XOR 0x80  (== u4*s + a - 128), guaranteed in [-128,127].
      int8_t q_lo = (int8_t)(((w_lo * s + a) ^ 0x80u) & 0xFF);
      int8_t q_hi = (int8_t)(((w_hi * s + a) ^ 0x80u) & 0xFF);
      ws[j * BK + 2 * kb] = q_lo;
      ws[j * BK + 2 * kb + 1] = q_hi;
    }
    __syncthreads();

    // Accumulate this chunk: 16 dp4a per output.
#pragma unroll
    for (int o = 0; o < OUT_PER_THREAD; ++o) {
      const int8_t* xr = &xs[mloc[o] * BK];
      const int8_t* wr = &ws[nloc[o] * BK];
#pragma unroll
      for (int kk = 0; kk < BK; kk += 4) {
        int xv = *reinterpret_cast<const int*>(xr + kk);
        int wv = *reinterpret_cast<const int*>(wr + kk);
        acc[o] = __dp4a(xv, wv, acc[o]);
      }
    }
    __syncthreads();
  }

  // Epilogue.
#pragma unroll
  for (int o = 0; o < OUT_PER_THREAD; ++o) {
    int m = m0 + mloc[o], n = n0 + nloc[o];
    if (m < M && n < N) {
      Y[m * N + n] = (float)acc[o] * ascale[m] * s1[n];
    }
  }
}

torch::Tensor w4a8_gemm(torch::Tensor X, torch::Tensor qweight, torch::Tensor s_u8,
                        torch::Tensor offset_a, torch::Tensor s1, torch::Tensor ascale,
                        int64_t N, int64_t K, int64_t group_size) {
  TORCH_CHECK(X.is_cuda() && qweight.is_cuda(), "inputs must be CUDA");
  TORCH_CHECK(X.dtype() == torch::kInt8, "X must be int8");
  TORCH_CHECK(qweight.dtype() == torch::kUInt8, "qweight must be uint8 (nibble-packed)");
  TORCH_CHECK(s1.dtype() == torch::kFloat32 && ascale.dtype() == torch::kFloat32,
              "s1/ascale must be float32");
  TORCH_CHECK(group_size == BK, "v1 requires group_size == 64");
  TORCH_CHECK(K % group_size == 0 && K % 2 == 0, "bad K");

  const int M = X.size(0);
  const int G = K / group_size;
  X = X.contiguous();
  qweight = qweight.contiguous();
  auto Y = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(X.device()));
  auto stream = at::cuda::getCurrentCUDAStream();

  if (M <= GEMV_MAXM) {
    // Decode / small-M: warp-per-column, bandwidth-bound.
    const int blocks = (N + GEMV_WARPS - 1) / GEMV_WARPS;  // one warp per column
    const size_t xbytes = (size_t)M * K;                   // stage X in smem if it fits
    // Empirically the smem-X staging slows M=1 (extra sync + per-block cooperative load
    // dominates when there are thousands of tiny blocks); keep it off pending profiling.
    const bool cache_x = false && xbytes <= 40 * 1024;
    if (cache_x) {
      w4a8_gemv_kernel<true><<<blocks, GEMV_THREADS, xbytes, stream>>>(
          X.data_ptr<int8_t>(), qweight.data_ptr<uint8_t>(), s_u8.data_ptr<uint8_t>(),
          offset_a.data_ptr<uint8_t>(), s1.data_ptr<float>(), ascale.data_ptr<float>(),
          Y.data_ptr<float>(), M, (int)N, (int)K, G);
    } else {
      w4a8_gemv_kernel<false><<<blocks, GEMV_THREADS, 0, stream>>>(
          X.data_ptr<int8_t>(), qweight.data_ptr<uint8_t>(), s_u8.data_ptr<uint8_t>(),
          offset_a.data_ptr<uint8_t>(), s1.data_ptr<float>(), ascale.data_ptr<float>(),
          Y.data_ptr<float>(), M, (int)N, (int)K, G);
    }
  } else {
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    w4a8_gemm_dp4a_kernel<<<grid, THREADS, 0, stream>>>(
        X.data_ptr<int8_t>(), qweight.data_ptr<uint8_t>(), s_u8.data_ptr<uint8_t>(),
        offset_a.data_ptr<uint8_t>(), s1.data_ptr<float>(), ascale.data_ptr<float>(),
        Y.data_ptr<float>(), M, (int)N, (int)K, G);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return Y;
}

}  // namespace liquidgemm
