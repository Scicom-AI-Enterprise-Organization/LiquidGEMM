"""Weight packing + a faithful emulation of the kernel's SIMD dequant.

Two things live here:

1. ``pack_nibbles`` / ``unpack_nibbles`` — compact 2×UINT4-per-byte storage (round-trips).
2. ``simd_dequant_u32`` — models the kernel's core trick exactly: four UINT4 packed one
   per byte-lane of a uint32, dequantized with a single 32-bit multiply, a single 32-bit
   add, and a single 32-bit XOR (IMAD + XOR over 4 elements). The arithmetic is carried out
   on the packed 32-bit word (masked to 32 bits) and the byte lanes are then extracted, so
   any cross-byte carry WOULD corrupt a neighbouring lane — the test asserts it does not,
   which is exactly the ``product ≤ 240`` / ``u ≤ 255`` overflow-safety guarantee.

The kernel's true "dual-MMA packed" weight interleave (so one ``LDS.128`` feeds two WGMMAs)
is layout-specific and is added in Phase 2 alongside the CUDA kernel; Phase 1 only needs the
lane arithmetic to be proven correct.
"""

from __future__ import annotations

import torch

from .quant import LiquidQuantWeight

MASK32 = 0xFFFFFFFF


def pack_nibbles(q_u4: torch.Tensor) -> torch.Tensor:
    """Pack ``[N, K]`` uint8 nibbles (0-15) into ``[N, K//2]`` uint8 (low nibble first)."""
    assert q_u4.dtype == torch.uint8 and q_u4.shape[-1] % 2 == 0
    lo = q_u4[..., 0::2]
    hi = q_u4[..., 1::2]
    return (lo | (hi << 4)).to(torch.uint8)


def unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`pack_nibbles` -> ``[N, K]`` uint8 in [0, 15]."""
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    out = torch.stack([lo, hi], dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    return out.to(torch.uint8)


def pack_lanes_u32(q_u4: torch.Tensor) -> torch.Tensor:
    """Pack consecutive groups of 4 UINT4 into one uint32 (lane i in byte i).

    ``[N, K] -> [N, K//4]`` int64 (holding uint32 bit patterns).
    """
    assert q_u4.shape[-1] % 4 == 0
    N = q_u4.shape[0]
    v = q_u4.to(torch.int64).view(N, -1, 4)
    word = v[..., 0] | (v[..., 1] << 8) | (v[..., 2] << 16) | (v[..., 3] << 24)
    return word  # [N, K//4]


def simd_dequant_u32(qw: LiquidQuantWeight) -> torch.Tensor:
    """Emulate the kernel: 4-lane IMAD + XOR on packed uint32 -> reconstructed int8 [N, K].

    Uses 32-bit-masked word arithmetic so cross-byte carries are NOT hidden.
    """
    N, K, g = qw.N, qw.K, qw.group_size
    G = K // g
    assert g % 4 == 0, "group_size must be a multiple of 4 for 4-lane packing"
    words = pack_lanes_u32(qw.qweight_u4)  # [N, K//4] int64

    # Broadcast the per-group integer scale and offset to each packed word.
    lanes_per_group = g // 4
    s_u8 = qw.s_u8.to(torch.int64).repeat_interleave(lanes_per_group, dim=1)   # [N, K//4]
    a = qw.offset_a.to(torch.int64).repeat_interleave(lanes_per_group, dim=1)  # [N, K//4]
    a_word = a * 0x01010101  # broadcast the offset byte into all 4 lanes

    prod = (words * s_u8) & MASK32          # single IMAD (multiply part), 32-bit
    u_word = (prod + a_word) & MASK32       # + offset (add part of IMAD)
    xor_word = u_word ^ 0x80808080          # single XOR -> two's-complement bytes

    # Extract 4 signed int8 lanes.
    b = torch.stack([(xor_word >> (8 * i)) & 0xFF for i in range(4)], dim=-1)  # [N, K//4, 4]
    b = b.reshape(N, K).to(torch.uint8).view(torch.int8)
    return b
