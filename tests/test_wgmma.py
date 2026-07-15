"""Stage 1: real Hopper WGMMA int8 tensor-core GEMM must match torch._int_mm bit-exactly."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

try:
    import liquidgemm.ops  # noqa: F401  registers torch.ops.liquidgemm.*
    _HAVE = True
except Exception:
    _HAVE = False

SHAPES = [(128, 512, 4096), (256, 256, 512), (128, 6144, 4096), (512, 4096, 14336)]


@pytest.mark.skipif(not _HAVE, reason="extension not built")
@pytest.mark.parametrize("M,N,K", SHAPES)
def test_wgmma_i8_matches_int_mm(M, N, K):
    torch.manual_seed(0)
    a = torch.randint(-127, 127, (M, K), device="cuda", dtype=torch.int8)
    b = torch.randint(-127, 127, (N, K), device="cuda", dtype=torch.int8)
    got = torch.ops.liquidgemm.wgmma_i8_gemm(a, b)
    ref = torch._int_mm(a, b.t().contiguous())
    assert torch.equal(got, ref), f"WGMMA int8 mismatch at {M}x{N}x{K}"


@pytest.mark.skipif(not _HAVE, reason="extension not built")
@pytest.mark.parametrize("M,N,K", SHAPES)
def test_fused_w4a8_wgmma_matches_reference(M, N, K):
    """Fused 4-bit dequant + WGMMA int32 accumulation == reference."""
    from liquidgemm import quant
    from liquidgemm.pack import pack_nibbles
    torch.manual_seed(0)
    W = torch.randn(N, K) * 0.05
    qw = quant.quantize_weight(W, 64)
    packed = pack_nibbles(qw.qweight_u4).cuda().contiguous()
    x_i8 = torch.randint(-127, 127, (M, K), device="cuda", dtype=torch.int8)
    got = torch.ops.liquidgemm.w4a8_wgmma(x_i8, packed, qw.s_u8.cuda(), qw.offset_a.cuda(), N, K, 64)
    ref = torch._int_mm(x_i8, quant.dequantize_i8(qw).cuda().t().contiguous())
    assert torch.equal(got, ref), f"fused W4A8 WGMMA mismatch at {M}x{N}x{K}"


@pytest.mark.skipif(not _HAVE, reason="extension not built")
@pytest.mark.parametrize("M,N,K", SHAPES)
@pytest.mark.parametrize("packed", [False, True])
def test_rs_w4a8_wgmma_matches_reference(M, N, K, packed):
    """RS WGMMA (in-register dequant, the paper's datapath) == reference, v1 + prepacked."""
    from liquidgemm import quant, ops
    from liquidgemm.pack import pack_nibbles
    torch.manual_seed(0)
    W = torch.randn(N, K) * 0.05
    qw = quant.quantize_weight(W, 64)
    x_i8 = torch.randint(-127, 127, (M, K), device="cuda", dtype=torch.int8)
    if packed:
        w = ops.repack_rs_weight(qw).cuda()
        spack = ops.build_rs_scale_pack(qw).cuda()
    else:
        w = pack_nibbles(qw.qweight_u4).cuda().contiguous()
        spack = qw.s_u8.cuda()
    got = torch.ops.liquidgemm.w4a8_wgmma_rs(
        x_i8, w, spack, qw.s_u8.cuda(), qw.offset_a.cuda(), N, K, 64, packed)
    ref = torch._int_mm(x_i8, quant.dequantize_i8(qw).cuda().t().contiguous())
    assert torch.equal(got, ref), f"RS W4A8 WGMMA (packed={packed}) mismatch at {M}x{N}x{K}"


@pytest.mark.skipif(not _HAVE, reason="extension not built")
@pytest.mark.parametrize("M", [1, 30, 100])
def test_rs_w4a8_wgmma_pads_odd_M(M):
    """The op must accept any M (pads to 64 internally) — decode/concurrency shapes."""
    from liquidgemm import quant, ops
    torch.manual_seed(0)
    N, K = 512, 4096
    qw = quant.quantize_weight(torch.randn(N, K) * 0.05, 64)
    w = ops.repack_rs_weight(qw).cuda()
    spack = ops.build_rs_scale_pack(qw).cuda()
    x_i8 = torch.randint(-127, 127, (M, K), device="cuda", dtype=torch.int8)
    got = torch.ops.liquidgemm.w4a8_wgmma_rs(
        x_i8, w, spack, qw.s_u8.cuda(), qw.offset_a.cuda(), N, K, 64, True)
    assert got.shape == (M, N)
    xp = torch.cat([x_i8, x_i8.new_zeros(64 - M % 64 if M % 64 else 0, K)], 0)
    ref = torch._int_mm(xp, quant.dequantize_i8(qw).cuda().t().contiguous())[:M]
    assert torch.equal(got.contiguous(), ref)
