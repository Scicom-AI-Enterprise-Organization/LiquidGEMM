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
