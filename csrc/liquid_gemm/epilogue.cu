// Fused helpers to remove per-call overhead from the vLLM W4A8 path:
//   quant_per_token : bf16/fp16 [M,K] -> (int8 [M,K], ascale [M])   (one kernel)
//   scale_epilogue  : int32 acc [M,N] * ascale[m] * s1[n] -> bf16/fp16 [M,N]  (one kernel)
// With cuBLASLt INT8 GEMM (torch._int_mm) between them, the linear is 3 kernels total
// instead of ~7 Python-orchestrated ops.

#include <torch/all.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace liquidgemm {

template <typename T>
__global__ void quant_per_token_kernel(const T* __restrict__ x, int8_t* __restrict__ xq,
                                        float* __restrict__ ascale, int M, int K) {
  const int m = blockIdx.x;
  if (m >= M) return;
  const T* xr = x + (long)m * K;
  float amax = 0.f;
  for (int k = threadIdx.x; k < K; k += blockDim.x)
    amax = fmaxf(amax, fabsf((float)xr[k]));
  __shared__ float sm[32];
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, o));
  if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = amax;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = (threadIdx.x < (blockDim.x + 31) / 32) ? sm[threadIdx.x] : 0.f;
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, o));
    if (threadIdx.x == 0) sm[0] = v;
  }
  __syncthreads();
  float sc = sm[0] / 127.0f;
  sc = sc > 0.f ? sc : 1e-8f;
  if (threadIdx.x == 0) ascale[m] = sc;
  const float inv = 1.0f / sc;
  int8_t* qr = xq + (long)m * K;
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    int q = __float2int_rn((float)xr[k] * inv);
    q = max(-127, min(127, q));
    qr[k] = (int8_t)q;
  }
}

template <typename T>
__global__ void scale_epilogue_kernel(const int* __restrict__ acc,
                                       const float* __restrict__ ascale,
                                       const float* __restrict__ s1, T* __restrict__ y,
                                       int M, int N) {
  const long tot = (long)M * N;
  for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < tot;
       idx += (long)gridDim.x * blockDim.x) {
    const int m = idx / N, n = idx - (long)m * N;
    y[idx] = (T)((float)acc[idx] * ascale[m] * s1[n]);
  }
}

std::vector<torch::Tensor> quant_per_token(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2, "x must be 2D CUDA");
  x = x.contiguous();
  const int M = x.size(0), K = x.size(1);
  auto xq = torch::empty({M, K}, torch::dtype(torch::kInt8).device(x.device()));
  auto ascale = torch::empty({M}, torch::dtype(torch::kFloat32).device(x.device()));
  auto stream = at::cuda::getCurrentCUDAStream();
  const int threads = 256;
  if (x.dtype() == torch::kBFloat16)
    quant_per_token_kernel<__nv_bfloat16><<<M, threads, 0, stream>>>(
        (const __nv_bfloat16*)x.data_ptr(), xq.data_ptr<int8_t>(), ascale.data_ptr<float>(), M, K);
  else if (x.dtype() == torch::kFloat16)
    quant_per_token_kernel<__half><<<M, threads, 0, stream>>>(
        (const __half*)x.data_ptr(), xq.data_ptr<int8_t>(), ascale.data_ptr<float>(), M, K);
  else
    quant_per_token_kernel<float><<<M, threads, 0, stream>>>(
        x.data_ptr<float>(), xq.data_ptr<int8_t>(), ascale.data_ptr<float>(), M, K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {xq, ascale};
}

torch::Tensor scale_epilogue(torch::Tensor acc, torch::Tensor ascale, torch::Tensor s1,
                             int64_t out_dtype) {
  TORCH_CHECK(acc.is_cuda() && acc.dim() == 2 && acc.dtype() == torch::kInt32);
  acc = acc.contiguous();
  const int M = acc.size(0), N = acc.size(1);
  auto dt = out_dtype == 1 ? torch::kFloat16 : (out_dtype == 2 ? torch::kFloat32 : torch::kBFloat16);
  auto y = torch::empty({M, N}, torch::dtype(dt).device(acc.device()));
  auto stream = at::cuda::getCurrentCUDAStream();
  const int threads = 256;
  const int blocks = (int)std::min<long>(65535, ((long)M * N + threads - 1) / threads);
  if (dt == torch::kBFloat16)
    scale_epilogue_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
        acc.data_ptr<int>(), ascale.data_ptr<float>(), s1.data_ptr<float>(),
        (__nv_bfloat16*)y.data_ptr(), M, N);
  else if (dt == torch::kFloat16)
    scale_epilogue_kernel<__half><<<blocks, threads, 0, stream>>>(
        acc.data_ptr<int>(), ascale.data_ptr<float>(), s1.data_ptr<float>(),
        (__half*)y.data_ptr(), M, N);
  else
    scale_epilogue_kernel<float><<<blocks, threads, 0, stream>>>(
        acc.data_ptr<int>(), ascale.data_ptr<float>(), s1.data_ptr<float>(),
        y.data_ptr<float>(), M, N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

}  // namespace liquidgemm
