"""Python wrappers over the compiled CUDA ops (``torch.ops.liquidgemm.*``).

Importing this module loads ``liquidgemm._C``, which registers the ops. Kept out of the
package ``__init__`` so the pure-Python reference (quant/pack) works without the build.
"""

from __future__ import annotations

import torch

from . import _C  # noqa: F401  (side effect: registers torch.ops.liquidgemm.*)
from .quant import LiquidQuantWeight


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
