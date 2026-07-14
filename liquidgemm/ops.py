"""Python wrappers over the compiled CUDA ops (``torch.ops.liquidgemm.*``).

Importing this module loads ``liquidgemm._C``, which registers the ops. Kept out of the
package ``__init__`` so the pure-Python reference (quant/pack) works without the build.
"""

from __future__ import annotations

import torch

from . import _C  # noqa: F401  (side effect: registers torch.ops.liquidgemm.*)
from .quant import LiquidQuantWeight
from .pack import pack_nibbles


# Fake/meta implementations so the ops are torch.compile / CUDA-graph capturable
# (lets vLLM run WITHOUT enforce_eager, amortizing per-layer launch overhead).
@torch.library.register_fake("liquidgemm::dequant_weight")
def _dequant_weight_fake(qweight, s_u8, offset_a, N, K, group_size):
    return qweight.new_empty((N, K), dtype=torch.int8)


@torch.library.register_fake("liquidgemm::w4a8_gemm")
def _w4a8_gemm_fake(X, qweight, s_u8, offset_a, s1, ascale, N, K, group_size):
    return X.new_empty((X.shape[0], N), dtype=torch.float32)


@torch.library.register_fake("liquidgemm::quant_per_token")
def _quant_per_token_fake(x):
    M, K = x.shape
    return (x.new_empty((M, K), dtype=torch.int8),
            x.new_empty((M,), dtype=torch.float32))


@torch.library.register_fake("liquidgemm::scale_epilogue")
def _scale_epilogue_fake(acc, ascale, s1, out_dtype):
    dt = {0: torch.bfloat16, 1: torch.float16, 2: torch.float32}[int(out_dtype)]
    return acc.new_empty(acc.shape, dtype=dt)


@torch.library.register_fake("liquidgemm::wgmma_i8_gemm")
def _wgmma_i8_gemm_fake(a, b):
    return a.new_empty((a.shape[0], b.shape[0]), dtype=torch.int32)


@torch.library.register_fake("liquidgemm::w4a8_wgmma")
def _w4a8_wgmma_fake(x_i8, packed, s_u8, off_a, N, K, group_size):
    return x_i8.new_empty((x_i8.shape[0], N), dtype=torch.int32)


@torch.library.register_fake("liquidgemm::w4a8_wgmma_rs")
def _w4a8_wgmma_rs_fake(x_i8, w, s_u8, off_a, N, K, group_size, packed):
    return x_i8.new_empty((x_i8.shape[0], N), dtype=torch.int32)


def dequant_weight_i8(qw: LiquidQuantWeight) -> torch.Tensor:
    """Reconstruct INT8 weights [N, K] on-device via the CUDA IMAD+XOR kernel."""
    return torch.ops.liquidgemm.dequant_weight(
        qw.qweight_u4.cuda(),
        qw.s_u8.cuda(),
        qw.offset_a.cuda(),
        qw.N,
        qw.K,
        qw.group_size,
    )


def w4a8_linear_unfused(
    x_i8: torch.Tensor,
    ascale: torch.Tensor,
    qw: LiquidQuantWeight,
    out_dtype: torch.dtype = torch.float16,
):
    """Unfused W4A8 GEMM correctness oracle: CUDA dequant -> int8 tensor-core matmul -> scale.

    Y[m,n] = ascale[m] * s1[n] * sum_k X_i8[m,k] * Qhat_i8[n,k]. Matches quant.w4a8_matmul
    exactly on the integer accumulation. This is NOT the fused kernel — it writes/reads the
    full int8 weights (extra GMEM traffic) and exists only to validate numerics and unblock
    downstream (vLLM) work while the fused kernel is built.
    """
    w_i8 = dequant_weight_i8(qw)                      # [N, K] int8 (cuda)
    x_i8 = x_i8.cuda().contiguous()
    M = x_i8.shape[0]
    # torch._int_mm (cuBLASLt) requires M > 16 — the decode regime our fused kernel targets
    # natively. Pad with zero rows for the oracle; zeros add nothing to the accumulation.
    pad = 32 - M if M < 32 else 0
    if pad:
        x_i8 = torch.cat([x_i8, x_i8.new_zeros(pad, x_i8.shape[1])], dim=0)
    acc = torch._int_mm(x_i8, w_i8.t().contiguous())  # [M(+pad), N] int32
    if pad:
        acc = acc[:M]
    y = acc.to(torch.float32) * ascale.cuda()[:, None] * qw.s1.cuda()[None, :]
    return y.to(out_dtype), acc


def repack_rs_weight(qw: LiquidQuantWeight) -> torch.Tensor:
    """Prepack UINT4 weights into WGMMA-RS *fragment order* for coalesced register fills.

    Uses the (m,k) coordinates dumped by the wgmma_rs_a_coords op (correct by
    construction — no hand-derived PTX layout). Output: uint8
    [N/64, K/128, 128 threads, 32 bytes]; thread t's 64 nibbles for tile (rb, kt) in
    fragment-element order, two nibbles per byte (low nibble = even element).
    """
    coords = torch.ops.liquidgemm.wgmma_rs_a_coords().cpu()  # [128, 64, 2]
    T, E, _ = coords.shape
    m = coords[..., 0].long()
    k = coords[..., 1].long()
    # Structural invariants the packed kernel relies on: each 32-bit register holds
    # 4 consecutive-k elements of ONE row (=> one LiquidQuant group per register).
    m4 = m.view(T, E // 4, 4)
    k4 = k.view(T, E // 4, 4)
    assert (m4 == m4[..., :1]).all(), "RS fragment register spans multiple rows"
    assert (k4 - k4[..., :1] == torch.arange(4)).all(), "RS fragment k not consecutive"
    assert (k4[..., 0] % 4 == 0).all(), "RS fragment k not 4-aligned"

    N, K = qw.N, qw.K
    RB, KT = N // 64, K // 128
    q = qw.qweight_u4.view(RB, 64, KT, 128).permute(0, 2, 1, 3)  # [RB, KT, 64rows, 128k]
    m, k = m.to(q.device), k.to(q.device)
    out = q[:, :, m, k]                                          # [RB, KT, 128thr, 64elem]
    packed = (out[..., 0::2] | (out[..., 1::2] << 4)).to(torch.uint8)
    return packed.contiguous()                                   # [RB, KT, 128, 32]


def w4a8_gemm_rs(x_i8: torch.Tensor, ascale: torch.Tensor, qw: LiquidQuantWeight,
                 rs_packed: torch.Tensor = None,
                 out_dtype: torch.dtype = torch.float16):
    """RS-WGMMA W4A8 linear: in-register dequant (the paper's datapath). Any M (op pads)."""
    K, N = qw.K, qw.N
    if rs_packed is None:
        w, packed = pack_nibbles(qw.qweight_u4).cuda().contiguous(), False
    else:
        w, packed = rs_packed.cuda(), True
    acc = torch.ops.liquidgemm.w4a8_wgmma_rs(
        x_i8.cuda().contiguous(), w, qw.s_u8.cuda(), qw.offset_a.cuda(),
        N, K, qw.group_size, packed)
    y = torch.ops.liquidgemm.scale_epilogue(
        acc, ascale.cuda().contiguous(), qw.s1.cuda(),
        {torch.bfloat16: 0, torch.float16: 1, torch.float32: 2}[out_dtype])
    return y


def w4a8_gemm(x_i8: torch.Tensor, ascale: torch.Tensor, qw: LiquidQuantWeight,
              out_dtype: torch.dtype = torch.float16):
    """Fused W4A8 GEMM (dp4a): 4-bit weights read from GMEM, dequant in registers.

    Returns Y [M, N] in ``out_dtype``. The kernel accumulates in fp32; the cast is applied
    on return.
    """
    packed = pack_nibbles(qw.qweight_u4).cuda().contiguous()  # [N, K//2] uint8
    y = torch.ops.liquidgemm.w4a8_gemm(
        x_i8.cuda().contiguous(), packed, qw.s_u8.cuda(), qw.offset_a.cuda(),
        qw.s1.cuda(), ascale.cuda(), qw.N, qw.K, qw.group_size)
    return y.to(out_dtype)
