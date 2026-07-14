// LiquidQuant dequant kernel: reconstruct INT8 weights from packed UINT4 via IMAD + XOR.
//
// This is the standalone validation of the paper's Eq.12 core arithmetic (2 ops / 4 elems):
//   u = Q_u4 * s_u8 + a        (one IMAD over 4 byte-lanes; product <= 240 so no carry)
//   Q_hat_i8 = u XOR 0x80      (biased byte -> two's-complement int8, == u - 128)
//
// Weights arrive as uint8 [N, K] with one UINT4 value (0-15) per byte, so four consecutive
// bytes are exactly the 4-lane packed word the kernel multiplies. Group scale/offset are
// uint8 [N, G] with G = K / group_size.

#include <torch/all.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>

namespace liquidgemm {

__global__ void dequant_weight_kernel(
    const uint32_t* __restrict__ qweight,  // [N, K/4] (reinterpreted uint8[N,K])
    const uint8_t* __restrict__ s_u8,      // [N, G]
    const uint8_t* __restrict__ offset_a,  // [N, G]
    uint32_t* __restrict__ out,            // [N, K/4] (reinterpreted int8[N,K])
    int N, int K, int group_size) {
  const int Kw = K >> 2;                       // 32-bit words per row (K/4)
  const int words_per_group = group_size >> 2; // K/4 words span group_size/4 lanes
  const int G = K / group_size;
  const long total = (long)N * Kw;

  for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += (long)gridDim.x * blockDim.x) {
    const int n = idx / Kw;
    const int w = idx - (long)n * Kw;
    const int g = w / words_per_group;
    const uint32_t s = s_u8[n * G + g];
    const uint32_t a = offset_a[n * G + g];
    const uint32_t a_word = a * 0x01010101u;   // broadcast offset into 4 lanes

    const uint32_t word = qweight[idx];
    const uint32_t u = word * s + a_word;       // IMAD: 4 lanes at once (product <= 240)
    out[idx] = u ^ 0x80808080u;                 // XOR: biased -> two's complement
  }
}

// Launcher. qweight/out are uint8/int8 [N, K]; K must be a multiple of 4.
torch::Tensor dequant_weight(torch::Tensor qweight, torch::Tensor s_u8,
                             torch::Tensor offset_a, int64_t N, int64_t K,
                             int64_t group_size) {
  TORCH_CHECK(qweight.is_cuda(), "qweight must be CUDA");
  TORCH_CHECK(qweight.dtype() == torch::kUInt8, "qweight must be uint8");
  TORCH_CHECK(s_u8.dtype() == torch::kUInt8 && offset_a.dtype() == torch::kUInt8,
              "s_u8/offset_a must be uint8");
  TORCH_CHECK(K % 4 == 0, "K must be a multiple of 4");
  TORCH_CHECK(K % group_size == 0, "K must be a multiple of group_size");

  qweight = qweight.contiguous();
  s_u8 = s_u8.contiguous();
  offset_a = offset_a.contiguous();
  auto out = torch::empty({N, K}, torch::dtype(torch::kInt8).device(qweight.device()));

  const long total = (long)N * (K / 4);
  const int threads = 256;
  const int blocks = (int)std::min<long>(65535, (total + threads - 1) / threads);
  auto stream = at::cuda::getCurrentCUDAStream();

  dequant_weight_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const uint32_t*>(qweight.data_ptr<uint8_t>()),
      s_u8.data_ptr<uint8_t>(), offset_a.data_ptr<uint8_t>(),
      reinterpret_cast<uint32_t*>(out.data_ptr<int8_t>()), (int)N, (int)K,
      (int)group_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace liquidgemm
