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

// two packed bytes (4 nibbles) -> uint32 with one UINT4 per byte lane, weight order.
__device__ __forceinline__ uint32_t wg_expand2(uint32_t h) {
  uint32_t b0 = h & 0xFF, b1 = (h >> 8) & 0xFF;
  return (b0 & 0xF) | ((b0 >> 4) << 8) | ((b1 & 0xF) << 16) | ((b1 >> 4) << 24);
}

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

// ---- Stage 2: fused W4A8 WGMMA. 4-bit weights streamed from GMEM, dequantized
// (LiquidQuant IMAD+XOR) into the swizzled WGMMA smem operand, INT8 WGMMA, INT32 out.
// Correctness-first (single-stage: load -> dequant -> wgmma per K-tile). ----
template <class ProblemShape, class CtaTiler,
          class AStride, class ASmemLayout, class TiledCopyA,
          class BSmemLayout, class TiledMma>
__global__ static __launch_bounds__(decltype(size(TiledMma{}))::value) void
w4a8_wgmma_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
                  int8_t const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a,
                  uint8_t const* Bpacked, BSmemLayout sB_layout,
                  uint8_t const* s_u8, uint8_t const* off_a, int32_t* C, int G) {
  using namespace cute;
  int M = size<0>(shape_MNK), N = size<1>(shape_MNK), K = size<2>(shape_MNK);
  auto BM = size<0>(cta_tiler), BN = size<1>(cta_tiler), BK = size<2>(cta_tiler);
  int Kh = K >> 1;

  Tensor mA = make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), dA);
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});  // (BM,BK,k)

  extern __shared__ char smem_raw[];
  int8_t* smemA = reinterpret_cast<int8_t*>(smem_raw);
  int8_t* smemB = smemA + cosize_v<ASmemLayout>;
  Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);           // (BM,BK)
  Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);           // (BN,BK)

  ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tAsA = thr_copy_a.partition_D(as_position_independent_swizzle_tensor(sA));

  TiledMma mma;
  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);
  Tensor tCsB = thr_mma.partition_B(sB);
  Tensor gC = local_tile(make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK),
                                     make_stride(N, Int<1>{})),
                         cta_tiler, cta_coord, Step<_1, _1, X>{});    // (BM,BN)
  Tensor tCgC = thr_mma.partition_C(gC);
  Tensor tCrA = thr_mma.make_fragment_A(tCsA);   // (MMA,MMA_M,MMA_K,PIPE)
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);   // (MMA,MMA_N,MMA_K,PIPE)
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);
  clear(tCrC);

  const int n0 = blockIdx.y * BN;
  const int bk = BK;  // 128
  const int K_TILES = K / bk;
  const int chunks_per_row = bk / 32;              // 32 weights (16 packed bytes) / chunk
  const int n_chunks = BN * chunks_per_row;

  // Dequant one K-tile of 4-bit weights into sB[buf] (coalesced 128-bit loads + 4-lane
  // SIMD LiquidQuant IMAD+XOR; one group scale/offset per 32 weights).
  auto dequant_tile = [&](int kt, int buf) {
    const int k0 = kt * bk;
    for (int c = threadIdx.x; c < n_chunks; c += blockDim.x) {
      int n = c / chunks_per_row;
      int kc = (c - n * chunks_per_row) * 32;
      int gn = n0 + n, gk = k0 + kc, g = gk >> 6;
      uint32_t s = s_u8[gn * G + g];
      uint32_t a_word = (uint32_t)off_a[gn * G + g] * 0x01010101u;
      const uint4 pkv = *reinterpret_cast<const uint4*>(&Bpacked[gn * Kh + (gk >> 1)]);
      const uint32_t pk[4] = {pkv.x, pkv.y, pkv.z, pkv.w};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        uint32_t lo = (wg_expand2(pk[i] & 0xFFFF) * s + a_word) ^ 0x80808080u;
        uint32_t hi = (wg_expand2(pk[i] >> 16) * s + a_word) ^ 0x80808080u;
        int base = kc + 8 * i;
        sB(n, base + 0, buf) = (int8_t)(lo & 0xFF);
        sB(n, base + 1, buf) = (int8_t)((lo >> 8) & 0xFF);
        sB(n, base + 2, buf) = (int8_t)((lo >> 16) & 0xFF);
        sB(n, base + 3, buf) = (int8_t)((lo >> 24) & 0xFF);
        sB(n, base + 4, buf) = (int8_t)(hi & 0xFF);
        sB(n, base + 5, buf) = (int8_t)((hi >> 8) & 0xFF);
        sB(n, base + 6, buf) = (int8_t)((hi >> 16) & 0xFF);
        sB(n, base + 7, buf) = (int8_t)((hi >> 24) & 0xFF);
      }
    }
  };

  // Generic-proxy smem writes (the dequant stores) must be made visible to the async
  // proxy (WGMMA smem descriptor reads) — otherwise a timing-dependent race.
  auto fence_proxy = [] { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); };

  // Prologue: fill buffer 0.
  copy(copy_a, tAgA(_, _, _, 0), tAsA(_, _, _, 0));
  cp_async_fence();
  dequant_tile(0, 0);
  cp_async_wait<0>();
  fence_proxy();
  __syncthreads();

  // Software-pipelined mainloop: issue WGMMA (async, tensor cores) on the current buffer,
  // then load+dequant the NEXT tile (CUDA cores) into the other buffer so the two overlap
  // (the ImFP idea, via double-buffering rather than warp specialization).
  for (int kt = 0; kt < K_TILES; ++kt) {
    int cur = kt & 1, nxt = (kt + 1) & 1;
    warpgroup_fence_operand(tCrC);
    warpgroup_arrive();
    cute::gemm(mma, tCrA(_, _, _, cur), tCrB(_, _, _, cur), tCrC);
    warpgroup_commit_batch();

    if (kt + 1 < K_TILES) {
      copy(copy_a, tAgA(_, _, _, kt + 1), tAsA(_, _, _, nxt));
      cp_async_fence();
      cp_async_wait<0>();
      dequant_tile(kt + 1, nxt);   // overlaps the in-flight WGMMA above
      fence_proxy();               // make dequant stores visible to WGMMA's async proxy
    }

    warpgroup_wait<0>();
    warpgroup_fence_operand(tCrC);
    __syncthreads();
  }
  copy(tCrC, tCgC);  // write INT32 accumulators; scaling done by scale_epilogue
}

torch::Tensor w4a8_wgmma(torch::Tensor x_i8, torch::Tensor packed, torch::Tensor s_u8,
                         torch::Tensor off_a, int64_t N, int64_t K, int64_t group_size) {
  TORCH_CHECK(x_i8.is_cuda() && x_i8.dtype() == torch::kInt8);
  TORCH_CHECK(packed.dtype() == torch::kUInt8 && group_size == 64);
  x_i8 = x_i8.contiguous();
  packed = packed.contiguous();
  int M = x_i8.size(0);
  TORCH_CHECK(M % 128 == 0 && N % 128 == 0 && K % 128 == 0,
              "Stage-2 WGMMA requires M,N,K multiples of 128 (pad callers)");
  int G = K / group_size;
  auto c = torch::empty({M, (int)N}, torch::dtype(torch::kInt32).device(x_i8.device()));

  auto prob = make_shape(M, (int)N, (int)K);
  auto dA = make_stride((int)K, Int<1>{});
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<128>{};
  auto bP = Int<2>{};   // double-buffer for dequant/WGMMA overlap
  auto cta = make_shape(bM, bN, bK);
  auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<int8_t>{}, make_shape(bM, bK, bP));
  auto sB = tile_to_shape(GMMA::Layout_K_SW128_Atom<int8_t>{}, make_shape(bN, bK, bP));
  TiledCopy copyA = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, int8_t>{},
                                    Layout<Shape<_16, _8>, Stride<_8, _1>>{},
                                    Layout<Shape<_1, _16>>{});
  TiledMMA mma = make_tiled_mma(SM90_64x64x32_S32S8S8_SS_TN{});

  int smem_bytes = (cosize_v<decltype(sA)> + cosize_v<decltype(sB)>) * (int)sizeof(int8_t);
  dim3 grid(ceil_div(M, (int)bM), ceil_div((int)N, (int)bN));
  dim3 block(size(mma));
  auto stream = at::cuda::getCurrentCUDAStream();
  auto kernel = &w4a8_wgmma_device<decltype(prob), decltype(cta),
                                   decltype(dA), decltype(sA), decltype(copyA),
                                   decltype(sB), decltype(mma)>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
  kernel<<<grid, block, smem_bytes, stream>>>(
      prob, cta, x_i8.data_ptr<int8_t>(), dA, sA, copyA,
      packed.data_ptr<uint8_t>(), sB, s_u8.data_ptr<uint8_t>(),
      off_a.data_ptr<uint8_t>(), c.data_ptr<int32_t>(), G);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return c;
}

// ---- Stage 4 (RS): the paper's actual datapath. Weights are the WGMMA *A* operand,
// sourced from REGISTERS (Y = (W·Xᵀ)ᵀ layout): each thread dequantizes its own A-fragment
// elements straight from 4-bit GMEM with IMAD+XOR — the weights never touch shared memory.
// Activations (int8) stream through swizzled smem as the B operand descriptor.
// Correct-by-construction: fragment(m,k) coordinates come from a CuTe identity-tensor
// partition, never a hand-derived PTX layout.
//
//   packed variant (fast path): weights are prepacked on the host in *fragment order*
//   (via the coords dumped by wgmma_rs_a_coords), so each thread loads its 64 nibbles
//   per (64x128) K-tile as two coalesced 128-bit words.

using RsMmaAtom = SM90_64x64x32_S32S8S8_RS_TN;
constexpr int RS_BM = 64;    // weight rows per CTA (wgmma M)
constexpr int RS_BN = 64;    // tokens per CTA (wgmma N)
constexpr int RS_BK = 128;   // K-tile (2 LiquidQuant groups)

// Dump the (m,k) coordinate of every A-fragment element for one (RS_BM, RS_BK) tile.
// out: int32 [128 threads][elems][2].
__global__ void wgmma_rs_a_coords_kernel(int* out, int elems) {
  TiledMMA mma = make_tiled_mma(RsMmaAtom{});
  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor cA = make_identity_tensor(Shape<Int<RS_BM>, Int<RS_BK>>{});
  Tensor tCcA = thr_mma.partition_A(cA);
  for (int i = 0; i < size(tCcA); ++i) {
    auto c = tCcA(i);
    out[(threadIdx.x * elems + i) * 2 + 0] = get<0>(c);
    out[(threadIdx.x * elems + i) * 2 + 1] = get<1>(c);
  }
}

torch::Tensor wgmma_rs_a_coords() {
  // elems per thread for (64,128) tile: 64*128 / 128 threads = 64
  const int elems = RS_BM * RS_BK / 128;
  auto out = torch::empty({128, elems, 2},
                          torch::dtype(torch::kInt32).device(torch::kCUDA));
  auto stream = at::cuda::getCurrentCUDAStream();
  wgmma_rs_a_coords_kernel<<<1, 128, 0, stream>>>(out.data_ptr<int>(), elems);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

template <bool PACKED, class BSmemLayout, class TiledCopyB, class TiledMma>
__global__ static __launch_bounds__(128) void
w4a8_wgmma_rs_device(int M, int N, int K,
                     int8_t const* __restrict__ X,     // [M,K] tokens (B operand)
                     uint8_t const* __restrict__ Wp,   // PACKED? fragment-order pack
                                                       //        : [N,K/2] nibble-packed
                     uint8_t const* __restrict__ s_u8, // [N,G]
                     uint8_t const* __restrict__ off_a,// [N,G]
                     int32_t* __restrict__ C,          // [M,N] int32 accum out
                     int G,
                     BSmemLayout sB_layout, TiledCopyB copy_b, TiledMma mma) {
  const int n0 = blockIdx.x * RS_BM;   // weight-row offset (output channel)
  const int t0 = blockIdx.y * RS_BN;   // token offset
  const int Kh = K >> 1;
  const int KT = K / RS_BK;

  extern __shared__ char smem_raw[];
  Tensor sB = make_tensor(make_smem_ptr(reinterpret_cast<int8_t*>(smem_raw)),
                          sB_layout);                                   // (BN,BK,PIPE=2)

  Tensor mX = make_tensor(make_gmem_ptr(X), make_shape(M, K), make_stride(K, Int<1>{}));
  Tensor gX = local_tile(mX, Shape<Int<RS_BN>, Int<RS_BK>>{},
                         make_coord(blockIdx.y, _));                    // (BN,BK,k)

  ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
  Tensor tBgB = thr_copy_b.partition_S(gX);
  Tensor tBsB = thr_copy_b.partition_D(as_position_independent_swizzle_tensor(sB));

  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  // A register fragments (double-buffered for dequant/WGMMA overlap).
  Tensor dumA = make_tensor(make_smem_ptr((int8_t*)nullptr),
                            make_layout(Shape<Int<RS_BM>, Int<RS_BK>>{},
                                        make_stride(Int<RS_BK>{}, Int<1>{})));
  Tensor tCrA0 = thr_mma.partition_fragment_A(dumA);   // (MMA,MMA_M,MMA_K) int8 rmem
  Tensor tCrA1 = thr_mma.partition_fragment_A(dumA);
  Tensor cA = make_identity_tensor(Shape<Int<RS_BM>, Int<RS_BK>>{});
  Tensor tCcA = thr_mma.partition_A(cA);               // static (m,k) coords per element

  Tensor tCsB = thr_mma.partition_B(sB);               // (MMA,MMA_N,MMA_K,PIPE)
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);

  Tensor cC = make_identity_tensor(Shape<Int<RS_BM>, Int<RS_BN>>{});
  Tensor tCcC = thr_mma.partition_C(cC);
  Tensor tCrC = thr_mma.partition_fragment_C(cC);      // int32 accum
  clear(tCrC);

  constexpr int FRAG = decltype(size(tCcA))::value;    // 64 elems/thread per K-tile
  static_assert(FRAG % 4 == 0);

  // Dequant one K-tile of weights into an A register fragment.
  auto fill_frag = [&](auto& frag, int kt) {
    const int k0 = kt * RS_BK;
    if constexpr (PACKED) {
      // Fragment-order pack: this thread's 64 nibbles at 32-byte offset, coalesced.
      const uint4* p = reinterpret_cast<const uint4*>(
          Wp + ((size_t)(blockIdx.x * KT + kt) * 128 + threadIdx.x) * (FRAG / 2));
      uint32_t* fr = reinterpret_cast<uint32_t*>(raw_pointer_cast(frag.data()));
      CUTE_UNROLL
      for (int v = 0; v < FRAG / 32; ++v) {            // one uint4 = 32 nibbles = 8 regs
        const uint4 pk = p[v];
        const uint32_t w[4] = {pk.x, pk.y, pk.z, pk.w};
        CUTE_UNROLL
        for (int h = 0; h < 4; ++h) {                  // each uint32 = 8 nibbles = 2 regs
          const int j = v * 8 + h * 2;                 // register pair index
          // group scale/offset: reg pair spans elems 8j0..; row/col from static coords
          const int e = j * 4;
          const int m0 = get<0>(tCcA(e)), k0a = get<1>(tCcA(e));
          const int m1 = get<0>(tCcA(e + 4)), k1a = get<1>(tCcA(e + 4));
          const uint32_t s0 = __ldg(&s_u8[(n0 + m0) * G + ((k0 + k0a) >> 6)]);
          const uint32_t a0 = __ldg(&off_a[(n0 + m0) * G + ((k0 + k0a) >> 6)]) * 0x01010101u;
          const uint32_t s1v = __ldg(&s_u8[(n0 + m1) * G + ((k0 + k1a) >> 6)]);
          const uint32_t a1 = __ldg(&off_a[(n0 + m1) * G + ((k0 + k1a) >> 6)]) * 0x01010101u;
          fr[j]     = (wg_expand2(w[h] & 0xFFFF) * s0 + a0) ^ 0x80808080u;
          fr[j + 1] = (wg_expand2(w[h] >> 16) * s1v + a1) ^ 0x80808080u;
        }
      }
    } else {
      // v1 reference path: per-element gather from the plain [N,K/2] nibble layout.
      CUTE_UNROLL
      for (int i = 0; i < FRAG; ++i) {
        const int m = get<0>(tCcA(i)), k = get<1>(tCcA(i));
        const int gn = n0 + m, gk = k0 + k;
        const uint32_t s = s_u8[gn * G + (gk >> 6)];
        const uint32_t a = off_a[gn * G + (gk >> 6)];
        const uint8_t pb = Wp[gn * Kh + (gk >> 1)];
        const uint32_t nib = (gk & 1) ? (pb >> 4) : (pb & 0xF);
        frag(i) = (int8_t)(((nib * s + a) ^ 0x80u) & 0xFF);
      }
    }
  };

  // Prologue: stage B tile 0 and dequant A fragment 0.
  copy(copy_b, tBgB(_, _, _, 0), tBsB(_, _, _, 0));
  cp_async_fence();
  fill_frag(tCrA0, 0);
  cp_async_wait<0>();
  __syncthreads();

  // Mainloop: WGMMA on (frag cur, sB cur) while CUDA cores stage tile kt+1 —
  // register-level dequant/MMA overlap (the ImFP idea within one warpgroup).
  for (int kt = 0; kt < KT; ++kt) {
    const int cur = kt & 1;
    auto& fragC = (cur == 0) ? tCrA0 : tCrA1;
    auto& fragN = (cur == 0) ? tCrA1 : tCrA0;

    warpgroup_fence_operand(fragC);
    warpgroup_fence_operand(tCrC);
    warpgroup_arrive();
    cute::gemm(mma, fragC, tCrB(_, _, _, cur), tCrC);
    warpgroup_commit_batch();

    if (kt + 1 < KT) {   // overlap: load + dequant next tile while WGMMA runs
      copy(copy_b, tBgB(_, _, _, kt + 1), tBsB(_, _, _, 1 - cur));
      cp_async_fence();
      fill_frag(fragN, kt + 1);
      cp_async_wait<0>();
    }

    warpgroup_wait<0>();
    warpgroup_fence_operand(tCrC);
    __syncthreads();
  }

  // Epilogue: transposed write C[token, channel] with token bounds check.
  CUTE_UNROLL
  for (int i = 0; i < decltype(size(tCrC))::value; ++i) {
    const int m = get<0>(tCcC(i)), n = get<1>(tCcC(i));
    const int tok = t0 + n;
    if (tok < M) C[(long)tok * N + (n0 + m)] = tCrC(i);
  }
}

torch::Tensor w4a8_wgmma_rs(torch::Tensor x_i8, torch::Tensor w, torch::Tensor s_u8,
                            torch::Tensor off_a, int64_t N, int64_t K,
                            int64_t group_size, bool packed) {
  TORCH_CHECK(x_i8.is_cuda() && x_i8.dtype() == torch::kInt8);
  TORCH_CHECK(w.dtype() == torch::kUInt8 && group_size == 64);
  TORCH_CHECK(N % RS_BM == 0 && K % RS_BK == 0, "N%64==0 and K%128==0 required");
  x_i8 = x_i8.contiguous();
  w = w.contiguous();
  const int M = x_i8.size(0);
  TORCH_CHECK(M % RS_BN == 0, "M must be padded to a multiple of 64 (wrapper pads)");
  const int G = K / group_size;
  auto c = torch::empty({M, (int)N}, torch::dtype(torch::kInt32).device(x_i8.device()));

  auto sB = tile_to_shape(GMMA::Layout_K_SW128_Atom<int8_t>{},
                          make_shape(Int<RS_BN>{}, Int<RS_BK>{}, Int<2>{}));
  TiledCopy copyB = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, int8_t>{},
                                    Layout<Shape<_16, _8>, Stride<_8, _1>>{},
                                    Layout<Shape<_1, _16>>{});
  TiledMMA mma = make_tiled_mma(RsMmaAtom{});

  const int smem_bytes = cosize_v<decltype(sB)> * (int)sizeof(int8_t);
  dim3 grid((int)N / RS_BM, M / RS_BN);
  dim3 block(128);
  auto stream = at::cuda::getCurrentCUDAStream();

  if (packed) {
    auto kernel = &w4a8_wgmma_rs_device<true, decltype(sB), decltype(copyB), decltype(mma)>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    kernel<<<grid, block, smem_bytes, stream>>>(
        M, (int)N, (int)K, x_i8.data_ptr<int8_t>(), w.data_ptr<uint8_t>(),
        s_u8.data_ptr<uint8_t>(), off_a.data_ptr<uint8_t>(), c.data_ptr<int32_t>(), G,
        sB, copyB, mma);
  } else {
    auto kernel = &w4a8_wgmma_rs_device<false, decltype(sB), decltype(copyB), decltype(mma)>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    kernel<<<grid, block, smem_bytes, stream>>>(
        M, (int)N, (int)K, x_i8.data_ptr<int8_t>(), w.data_ptr<uint8_t>(),
        s_u8.data_ptr<uint8_t>(), off_a.data_ptr<uint8_t>(), c.data_ptr<int32_t>(), G,
        sB, copyB, mma);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return c;
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
