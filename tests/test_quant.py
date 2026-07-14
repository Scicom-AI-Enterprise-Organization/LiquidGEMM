"""Phase 1 correctness: LiquidQuant reference, the XOR dequant trick, SIMD lanes, packing."""

import torch
import pytest

from liquidgemm import quant, pack

torch.manual_seed(0)

# (name, N, K) for Llama-3.1-8B linear layers. N reduced where huge, K kept realistic.
LLAMA_SHAPES = [
    ("qkv", 1536, 4096),      # 6144 -> 1536 for test speed
    ("o", 1024, 4096),
    ("gate_up", 2048, 4096),  # 28672 -> 2048
    ("down", 1024, 14336),
]
GROUP_SIZES = [64, 128]


def _rand_weight(N, K):
    # Mildly heavy-tailed to exercise the protective range + per-group spread.
    W = torch.randn(N, K)
    W += 0.1 * torch.randn(N, 1) * torch.randn(N, K).abs()
    return W


@pytest.mark.parametrize("name,N,K", LLAMA_SHAPES)
@pytest.mark.parametrize("g", GROUP_SIZES)
def test_xor_equals_subtraction(name, N, K, g):
    """(Q_u4*s_u8 + a) XOR 0x80  ==  (Q_u4*s_u8 + a) - 128  (biased -> two's complement)."""
    qw = quant.quantize_weight(_rand_weight(N, K), group_size=g)
    via_xor = quant._dequant_i8_via_xor(qw)
    via_sub = quant._dequant_i8_via_sub(qw)
    assert torch.equal(via_xor, via_sub), "XOR 0x80 trick disagrees with u-128"


@pytest.mark.parametrize("name,N,K", LLAMA_SHAPES)
@pytest.mark.parametrize("g", GROUP_SIZES)
def test_overflow_invariants(name, N, K, g):
    """The guarantees that make IMAD+XOR safe: s_u8<=16, product<=240, u in [0,255]."""
    W = _rand_weight(N, K)
    qw = quant.quantize_weight(W, group_size=g)
    G = K // g

    assert qw.s_u8.max().item() <= 16 and qw.s_u8.min().item() >= 1
    assert qw.qweight_u4.max().item() <= 15

    q_u4 = qw.qweight_u4.to(torch.int64).view(N, G, g)
    s = qw.s_u8.to(torch.int64)[:, :, None]
    a = qw.offset_a.to(torch.int64)[:, :, None]
    prod = q_u4 * s
    u = prod + a
    assert prod.max().item() <= 240, "product exceeds 240 -> cross-byte carry possible"
    assert u.min().item() >= 0 and u.max().item() <= 255, "u escaped [0,255]"

    q_hat = quant.dequantize_i8(qw).to(torch.int32)
    assert q_hat.min().item() >= -128 and q_hat.max().item() <= 127


@pytest.mark.parametrize("name,N,K", LLAMA_SHAPES)
def test_simd_lane_dequant_matches(name, N, K):
    """4-lane packed-uint32 IMAD+XOR == per-element dequant (proves no cross-byte carry)."""
    qw = quant.quantize_weight(_rand_weight(N, K), group_size=64)
    simd = pack.simd_dequant_u32(qw)
    ref = quant.dequantize_i8(qw)
    assert torch.equal(simd, ref), "packed 32-bit lane dequant diverged from per-element"


@pytest.mark.parametrize("name,N,K", LLAMA_SHAPES)
def test_nibble_roundtrip(name, N, K):
    qw = quant.quantize_weight(_rand_weight(N, K), group_size=64)
    packed = pack.pack_nibbles(qw.qweight_u4)
    assert packed.shape[-1] == K // 2
    restored = pack.unpack_nibbles(packed)
    assert torch.equal(restored, qw.qweight_u4)


@pytest.mark.parametrize("name,N,K", LLAMA_SHAPES)
def test_weight_reconstruction_quality(name, N, K):
    W = _rand_weight(N, K)
    qw = quant.quantize_weight(W, group_size=64)
    W_hat = quant.dequantize(qw)
    rel = (W_hat - W).norm() / W.norm()
    cos = torch.nn.functional.cosine_similarity(W_hat.flatten(), W.flatten(), dim=0)
    print(f"[{name}] weight rel-err={rel:.4f} cos={cos:.5f}")
    assert rel < 0.15, f"weight rel-err too high: {rel:.4f}"
    assert cos > 0.985


@pytest.mark.parametrize("name,N,K", LLAMA_SHAPES)
def test_w4a8_matmul_vs_fp(name, N, K):
    """Int8-domain W4A8 GEMM should track full-precision X @ W^T closely (K averages error)."""
    M = 32
    W = _rand_weight(N, K)
    X = torch.randn(M, K)
    qw = quant.quantize_weight(W, group_size=64)
    x_i8, ascale = quant.quantize_activation(X)

    y_q = quant.w4a8_matmul(x_i8, ascale, qw)      # [M, N]
    y_fp = X @ W.t()                                # [M, N]
    rel = (y_q - y_fp).norm() / y_fp.norm()
    cos = torch.nn.functional.cosine_similarity(y_q.flatten(), y_fp.flatten(), dim=0)
    print(f"[{name}] matmul rel-err={rel:.4f} cos={cos:.5f}")
    assert cos > 0.99, f"matmul cosine too low: {cos:.4f}"


def test_full_qkv_shape_smoke():
    """One real full-size shape end-to-end (no N reduction)."""
    qw = quant.quantize_weight(_rand_weight(6144, 4096), group_size=64)
    assert qw.qweight_u4.shape == (6144, 4096)
    assert qw.s_u8.shape == (6144, 64) and qw.offset_a.shape == (6144, 64)
    assert torch.equal(pack.simd_dequant_u32(qw), quant.dequantize_i8(qw))
