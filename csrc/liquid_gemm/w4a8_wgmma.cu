// Real Hopper WGMMA path for LiquidGEMM (the paper's core kernel), built in stages.
// Stage 1 (this file): a correct INT8 WGMMA GEMM via CuTe atoms — proves the WGMMA
// tensor-core machinery (SM90_64x64x32_S32S8S8_SS_TN + swizzled smem + cp.async pipeline).
// Stage 2 adds in-register LiquidQuant 4-bit dequant; Stage 3 adds the ImFP warp
// specialization. The generic mainloop below is adapted from CUTLASS's cute tutorial
// examples/cute/tutorial/hopper/wgmma_sm90.cu (BSD-3, NVIDIA).

#include <torch/all.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>

using namespace cute;

namespace liquidgemm {

template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA,
          class TB, class BStride, class BSmemLayout, class TiledCopyB,
          class TC, class CStride, class TiledMma>
__global__ static __launch_bounds__(decltype(size(TiledMma{}))::value) void
wgmma_i8_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
                TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a,
                TB const* B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b,
                TC* C, CStride dC, TiledMma mma) {
  Tensor mA = make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), dA);  // (M,K)
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1, 2>(shape_MNK), dB);  // (N,K)
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), dC);  // (M,N)

  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});  // (BM,BK,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});  // (BN,BK,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});  // (BM,BN)

  extern __shared__ char smem_raw[];
  TA* smemA = reinterpret_cast<TA*>(smem_raw);
  TB* smemB = reinterpret_cast<TB*>(smem_raw + cosize_v<ASmemLayout> * sizeof(TA));
  Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);  // (BM,BK,PIPE)
  Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);  // (BN,BK,PIPE)

  ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tAsA = thr_copy_a.partition_D(as_position_independent_swizzle_tensor(sA));
  ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
  Tensor tBgB = thr_copy_b.partition_S(gB);
  Tensor tBsB = thr_copy_b.partition_D(as_position_independent_swizzle_tensor(sB));

  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);
  Tensor tCsB = thr_mma.partition_B(sB);
  Tensor tCgC = thr_mma.partition_C(gC);
  Tensor tCrA = thr_mma.make_fragment_A(tCsA);
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);
  clear(tCrC);

  auto K_TILE_MAX = size<3>(tAgA);
  auto K_PIPE_MAX = size<3>(tAsA);

  CUTE_UNROLL
  for (int k = 0; k < K_PIPE_MAX - 1; ++k) {
    copy(copy_a, tAgA(_, _, _, k), tAsA(_, _, _, k));
    copy(copy_b, tBgB(_, _, _, k), tBsB(_, _, _, k));
    cp_async_fence();
  }
  __syncthreads();

  int k_pipe_read = 0, k_pipe_write = K_PIPE_MAX - 1;
  CUTE_NO_UNROLL
  for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile) {
    int k_tile_next = k_tile + (K_PIPE_MAX - 1);
    k_tile_next = (k_tile_next >= K_TILE_MAX) ? K_TILE_MAX - 1 : k_tile_next;
    copy(copy_a, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, k_pipe_write));
    copy(copy_b, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, k_pipe_write));
    cp_async_fence();
    ++k_pipe_write;
    k_pipe_write = (k_pipe_write == K_PIPE_MAX) ? 0 : k_pipe_write;

    cp_async_wait<0>();
    warpgroup_fence_operand(tCrC);
    warpgroup_arrive();
    cute::gemm(mma, tCrA(_, _, _, k_pipe_read), tCrB(_, _, _, k_pipe_read), tCrC);
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(tCrC);
    ++k_pipe_read;
    k_pipe_read = (k_pipe_read == K_PIPE_MAX) ? 0 : k_pipe_read;
  }

  // Epilogue: write INT32 accumulators.
  copy(tCrC, tCgC);
}

// Stage 1: C[M,N] int32 = A[M,K] int8 @ B[N,K] int8^T  (TN, row-major inputs).
torch::Tensor wgmma_i8_gemm(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && a.dtype() == torch::kInt8 && b.dtype() == torch::kInt8);
  a = a.contiguous();
  b = b.contiguous();
  int M = a.size(0), K = a.size(1), N = b.size(0);
  TORCH_CHECK(b.size(1) == K, "K mismatch");
  TORCH_CHECK(M % 128 == 0 && N % 128 == 0 && K % 128 == 0,
              "Stage-1 WGMMA requires M,N,K multiples of 128");
  auto c = torch::empty({M, N}, torch::dtype(torch::kInt32).device(a.device()));

  auto prob = make_shape(M, N, K);
  auto dA = make_stride(K, Int<1>{});   // A [M,K] row-major (TN)
  auto dB = make_stride(K, Int<1>{});   // B [N,K] row-major
  auto dC = make_stride(N, Int<1>{});   // C [M,N] row-major
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<128>{};
  auto bP = Int<3>{};
  auto cta = make_shape(bM, bN, bK);

  auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<int8_t>{}, make_shape(bM, bK, bP));
  auto sB = tile_to_shape(GMMA::Layout_K_SW128_Atom<int8_t>{}, make_shape(bN, bK, bP));

  TiledCopy copyA = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, int8_t>{},
                                    Layout<Shape<_16, _8>, Stride<_8, _1>>{},
                                    Layout<Shape<_1, _16>>{});
  TiledCopy copyB = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, int8_t>{},
                                    Layout<Shape<_16, _8>, Stride<_8, _1>>{},
                                    Layout<Shape<_1, _16>>{});
  TiledMMA mma = make_tiled_mma(SM90_64x64x32_S32S8S8_SS_TN{});

  int smem_bytes = (cosize_v<decltype(sA)> + cosize_v<decltype(sB)>) * (int)sizeof(int8_t);
  dim3 grid(ceil_div(M, (int)bM), ceil_div(N, (int)bN));
  dim3 block(size(mma));
  auto stream = at::cuda::getCurrentCUDAStream();

  auto kernel = &wgmma_i8_device<decltype(prob), decltype(cta),
                                 int8_t, decltype(dA), decltype(sA), decltype(copyA),
                                 int8_t, decltype(dB), decltype(sB), decltype(copyB),
                                 int32_t, decltype(dC), decltype(mma)>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
  kernel<<<grid, block, smem_bytes, stream>>>(
      prob, cta, a.data_ptr<int8_t>(), dA, sA, copyA,
      b.data_ptr<int8_t>(), dB, sB, copyB, c.data_ptr<int32_t>(), dC, mma);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return c;
}

}  // namespace liquidgemm
