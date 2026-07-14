"""LiquidQuant (LQQ) reference quantizer — the numerical ground truth.

Two-level W4A8 quantization from LiquidGEMM (arXiv:2509.01229), matching QServe/QoQ's
progressive group quantization but with the overflow-safe IMAD+XOR dequant (Eq. 12).

Weight layout convention: ``W`` has shape ``[N, K]`` = ``[out_features, in_features]``.
- **Level 1** — per-output-channel (per-row) symmetric INT8 with a protective range
  ``[-119, 119]``. The 8-level margin below 127 equals ``max(s_u8)/2`` so that level-2
  rounding can never push the reconstructed value out of INT8.
- **Level 2** — per-group (``group_size`` along K) asymmetric UINT4 of the INT8 values,
  with an **integer** step ``s_u8 ∈ [1, 16]`` and offset ``a = 128 + min``.

Dequant (per element), all integer until the final scale:

    u        = Q_u4 * s_u8 + a          # IMAD; u ∈ [0, 255] (proved overflow-safe)
    Q_hat_i8 = u XOR 0x80               # biased byte -> two's-complement int8 == u - 128
    W_hat    = Q_hat_i8 * s1            # level-1 per-channel scale (float)

Because ``Q_u4 ≤ 15`` and ``s_u8 ≤ 16``, the product ``≤ 240`` never carries across byte
lanes — four elements packed in a uint32 dequant with a single IMAD + a single XOR.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Level-1 protective magnitude: 127 - max(s_u8)/2 = 127 - 8.
INT8_PROTECTIVE_MAX = 119
UINT4_MAX = 15
S_U8_MAX = 16


@dataclass
class LiquidQuantWeight:
    """Packed LiquidQuant weight (reference / host side).

    Shapes (with G groups per row, ``G = K // group_size``):
      qweight_u4 : [N, K]  uint8, values in [0, 15]
      s_u8       : [N, G]  uint8, integer group step in [1, 16]
      offset_a   : [N, G]  uint8, a = 128 + min(Q_i8_group), in [0, 255]
      s1         : [N]     float32, per-channel level-1 scale
    """

    qweight_u4: torch.Tensor
    s_u8: torch.Tensor
    offset_a: torch.Tensor
    s1: torch.Tensor
    group_size: int

    @property
    def N(self) -> int:
        return self.qweight_u4.shape[0]

    @property
    def K(self) -> int:
        return self.qweight_u4.shape[1]


def quantize_weight(W: torch.Tensor, group_size: int = 64) -> LiquidQuantWeight:
    """Quantize an ``[N, K]`` weight matrix with two-level LiquidQuant."""
    assert W.dim() == 2, "expected [N, K]"
    N, K = W.shape
    assert K % group_size == 0, f"K={K} not divisible by group_size={group_size}"
    G = K // group_size
    Wf = W.detach().to(torch.float32)

    # --- Level 1: per-channel symmetric INT8 in [-119, 119] ---
    s1 = Wf.abs().amax(dim=1).clamp_min(1e-12) / INT8_PROTECTIVE_MAX  # [N]
    q_i8 = torch.round(Wf / s1[:, None]).clamp_(-INT8_PROTECTIVE_MAX, INT8_PROTECTIVE_MAX)
    q_i8 = q_i8.to(torch.int32)  # keep as int32 for arithmetic; values in [-119, 119]

    # --- Level 2: per-group asymmetric UINT4 with integer step ---
    q_g = q_i8.view(N, G, group_size)
    gmin = q_g.amin(dim=2)  # [N, G] int32
    gmax = q_g.amax(dim=2)  # [N, G] int32
    # integer step; ceil guarantees the 15-step grid covers [min, max] (no clamp loss).
    s_u8 = torch.ceil((gmax - gmin).to(torch.float32) / UINT4_MAX).clamp_(1, S_U8_MAX)
    s_u8 = s_u8.to(torch.int32)  # [N, G] in [1, 16]

    q_u4 = torch.round((q_g - gmin[:, :, None]).to(torch.float32) / s_u8[:, :, None].to(torch.float32))
    q_u4 = q_u4.clamp_(0, UINT4_MAX).to(torch.int32).view(N, K)

    offset_a = (128 + gmin).to(torch.int32)  # [N, G] a = 128 + min

    # Overflow-safety invariants (the whole point of the scheme):
    assert int(s_u8.max()) <= S_U8_MAX and int(s_u8.min()) >= 1
    prod = q_u4.view(N, G, group_size) * s_u8[:, :, None]
    assert int(prod.max()) <= UINT4_MAX * S_U8_MAX  # <= 240, no cross-byte carry
    u = prod + offset_a[:, :, None]
    assert int(u.min()) >= 0 and int(u.max()) <= 255, f"u out of [0,255]: {int(u.min())}..{int(u.max())}"

    return LiquidQuantWeight(
        qweight_u4=q_u4.to(torch.uint8),
        s_u8=s_u8.to(torch.uint8),
        offset_a=offset_a.to(torch.uint8),
        s1=s1.to(torch.float32),
        group_size=group_size,
    )


def _dequant_i8_via_xor(qw: LiquidQuantWeight) -> torch.Tensor:
    """Reconstruct INT8 weights the way the kernel does: (Q_u4*s_u8 + a) XOR 0x80."""
    N, K, G = qw.N, qw.K, qw.K // qw.group_size
    q_u4 = qw.qweight_u4.to(torch.int32).view(N, G, qw.group_size)
    s_u8 = qw.s_u8.to(torch.int32)[:, :, None]
    a = qw.offset_a.to(torch.int32)[:, :, None]
    u = (q_u4 * s_u8 + a).view(N, K)  # in [0, 255]
    u8 = u.to(torch.uint8)
    q_hat = torch.bitwise_xor(u8, torch.tensor(0x80, dtype=torch.uint8)).view(torch.int8)
    return q_hat


def _dequant_i8_via_sub(qw: LiquidQuantWeight) -> torch.Tensor:
    """Mathematically-equivalent reconstruction: u - 128 (validates the XOR trick)."""
    N, K, G = qw.N, qw.K, qw.K // qw.group_size
    q_u4 = qw.qweight_u4.to(torch.int32).view(N, G, qw.group_size)
    s_u8 = qw.s_u8.to(torch.int32)[:, :, None]
    a = qw.offset_a.to(torch.int32)[:, :, None]
    q_hat = (q_u4 * s_u8 + a - 128).view(N, K)
    return q_hat.to(torch.int8)


def dequantize_i8(qw: LiquidQuantWeight) -> torch.Tensor:
    """Reconstructed INT8 weights ``Q_hat_i8`` (bit-exact XOR path)."""
    return _dequant_i8_via_xor(qw)


def dequantize(qw: LiquidQuantWeight) -> torch.Tensor:
    """Reconstructed float weights ``W_hat = Q_hat_i8 * s1`` (shape [N, K])."""
    return dequantize_i8(qw).to(torch.float32) * qw.s1[:, None]


def quantize_activation(X: torch.Tensor):
    """Per-token (per-row) symmetric INT8 activation quant. Returns (X_i8, ascale)."""
    assert X.dim() == 2, "expected [M, K]"
    Xf = X.detach().to(torch.float32)
    ascale = Xf.abs().amax(dim=1).clamp_min(1e-12) / 127.0  # [M]
    x_i8 = torch.round(Xf / ascale[:, None]).clamp_(-127, 127).to(torch.int8)
    return x_i8, ascale.to(torch.float32)


def w4a8_matmul(x_i8: torch.Tensor, ascale: torch.Tensor, qw: LiquidQuantWeight) -> torch.Tensor:
    """Int8-domain reference GEMM the kernel must match bit-for-bit.

    Computes ``Y[m, n] = ascale[m] * s1[n] * sum_k Q_hat_i8[n, k] * X_i8[m, k]``.
    Returns float32 ``Y`` of shape ``[M, N]``.
    """
    q_hat = dequantize_i8(qw).to(torch.int32)              # [N, K]
    acc = x_i8.to(torch.int32) @ q_hat.t()                 # [M, N] exact int32 accumulate
    y = acc.to(torch.float32) * ascale[:, None] * qw.s1[None, :]
    return y
