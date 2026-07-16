// W4A8 GEMV v2 — the decode (M<=16) engine. Warp-per-output-row weight streaming at full
// coalescing (16B/lane), LiquidQuant dequant via mask-shift interleaved nibbles
// (AND / SHR+AND + IMAD + XOR per 4 elements — the paper's 2-instr datapath), dp4a
// accumulation, fused warp-reduce + ascale*s1 scale + bf16/fp16 cast epilogue.
//
// Uses a GEMV-specific row-major pack (pack_gemv_interleaved in Python): within each
// 8-nibble octet along K, byte b = w[8h+b] | w[8h+4+b] << 4, so
//   lo = word & 0x0F0F0F0F -> k = 8h..8h+3 (consecutive, dp4a-aligned with X)
//   hi = (word >> 4) & 0x0F0F0F0F -> k = 8h+4..8h+7

#include <torch/all.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace liquidgemm {

template <int MT, typename T>
__global__ __launch_bounds__(256) void w4a8_gemv2_kernel(
    const int8_t* __restrict__ X,       // [M, K]
    const uint8_t* __restrict__ Wp,     // [N, K/2] interleaved pack
    const uint8_t* __restrict__ s_u8,   // [N, G]
    const uint8_t* __restrict__ off_a,  // [N, G]
    const float* __restrict__ ascale,   // [M]
    const float* __restrict__ s1,       // [N]
    T* __restrict__ Y,                  // [M, N]
    int M, int N, int K, int G) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps_per_block = blockDim.x >> 5;
  const int total_warps = gridDim.x * warps_per_block;
  const int Kh = K >> 1;

  for (int n = blockIdx.x * warps_per_block + warp; n < N; n += total_warps) {
    int acc[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[m] = 0;

    // Each lane streams 32 weights (16 packed bytes) per step; warp covers 1024/step.
    // (An ILP-2 variant — 64 weights/lane/step — was measured 30-40% SLOWER on H20:
    // the doubled register footprint costs more in occupancy than the load ILP buys.)
    for (int kc = lane * 32; kc < K; kc += 32 * 32) {
      const int g = kc >> 6;  // 32 consecutive k stay within one group (g=64)
      const uint32_t s = __ldg(&s_u8[n * G + g]);
      const uint32_t aw = __ldg(&off_a[n * G + g]) * 0x01010101u;
      const uint4 pkv = *reinterpret_cast<const uint4*>(&Wp[n * Kh + (kc >> 1)]);
      const uint32_t pk[4] = {pkv.x, pkv.y, pkv.z, pkv.w};
      int wq[8];
#pragma unroll
      for (int h = 0; h < 4; ++h) {  // word h -> k octet [kc+8h, kc+8h+8)
        wq[2 * h]     = (int)(((pk[h] & 0x0F0F0F0Fu) * s + aw) ^ 0x80808080u);
        wq[2 * h + 1] = (int)((((pk[h] >> 4) & 0x0F0F0F0Fu) * s + aw) ^ 0x80808080u);
      }
#pragma unroll
      for (int m = 0; m < MT; ++m) {
        if (m < M) {
          const int* xr = reinterpret_cast<const int*>(&X[(long)m * K + kc]);
#pragma unroll
          for (int h = 0; h < 4; ++h) {
            acc[m] = __dp4a(xr[2 * h], wq[2 * h], acc[m]);
            acc[m] = __dp4a(xr[2 * h + 1], wq[2 * h + 1], acc[m]);
          }
        }
      }
    }
    // Fused epilogue: warp reduce + scale + cast, no intermediate buffers.
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      if (m < M) {
        int v = acc[m];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if (lane == 0) Y[(long)m * N + n] = (T)((float)v * ascale[m] * s1[n]);
      }
    }
  }
}

torch::Tensor w4a8_gemv2(torch::Tensor x_i8, torch::Tensor gpack, torch::Tensor s_u8,
                         torch::Tensor off_a, torch::Tensor ascale, torch::Tensor s1,
                         int64_t N, int64_t K, int64_t group_size, int64_t out_dtype) {
  TORCH_CHECK(x_i8.is_cuda() && x_i8.dtype() == torch::kInt8);
  TORCH_CHECK(gpack.dtype() == torch::kUInt8 && group_size == 64);
  TORCH_CHECK(K % 128 == 0, "gemv2 requires K % 128 == 0");
  const int M = x_i8.size(0);
  TORCH_CHECK(M <= 16, "gemv2 is the M<=16 decode path");
  x_i8 = x_i8.contiguous();
  gpack = gpack.contiguous();
  ascale = ascale.contiguous();
  s1 = s1.contiguous();
  const int G = K / group_size;
  auto dt = out_dtype == 1 ? torch::kFloat16
                           : (out_dtype == 2 ? torch::kFloat32 : torch::kBFloat16);
  auto y = torch::empty({M, (long)N}, torch::dtype(dt).device(x_i8.device()));
  auto stream = at::cuda::getCurrentCUDAStream();
  const int warps_per_block = 8;
  const int blocks = (int)((N + warps_per_block - 1) / warps_per_block);

  auto launch = [&](auto mt_tag, auto t_tag) {
    constexpr int MT = decltype(mt_tag)::value;
    using T = decltype(t_tag);
    w4a8_gemv2_kernel<MT, T><<<blocks, warps_per_block * 32, 0, stream>>>(
        x_i8.data_ptr<int8_t>(), gpack.data_ptr<uint8_t>(), s_u8.data_ptr<uint8_t>(),
        off_a.data_ptr<uint8_t>(), ascale.data_ptr<float>(), s1.data_ptr<float>(),
        reinterpret_cast<T*>(y.data_ptr()), M, (int)N, (int)K, G);
  };
  auto dispatch_t = [&](auto mt_tag) {
    if (dt == torch::kBFloat16) launch(mt_tag, __nv_bfloat16{});
    else if (dt == torch::kFloat16) launch(mt_tag, __half{});
    else launch(mt_tag, float{});
  };
  if (M <= 1) dispatch_t(std::integral_constant<int, 1>{});
  else if (M <= 4) dispatch_t(std::integral_constant<int, 4>{});
  else dispatch_t(std::integral_constant<int, 16>{});
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

}  // namespace liquidgemm
