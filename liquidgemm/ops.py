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
def _w4a8_wgmma_rs_fake(x_i8, w, spack, s_u8, off_a, N, K, group_size, packed):
    return x_i8.new_empty((x_i8.shape[0], N), dtype=torch.int32)


@torch.library.register_fake("liquidgemm::w4a8_wgmma_rs_fused")
def _w4a8_wgmma_rs_fused_fake(x_i8, w, s_u8, off_a, ascale, s1, N, K, group_size, out_dtype):
    dt = {0: torch.bfloat16, 1: torch.float16, 2: torch.float32}[int(out_dtype)]
    return x_i8.new_empty((x_i8.shape[0], N), dtype=dt)


@torch.library.register_fake("liquidgemm::w4a8_gemv2")
def _w4a8_gemv2_fake(x_i8, gpack, s_u8, off_a, ascale, s1, N, K, group_size, out_dtype):
    dt = {0: torch.bfloat16, 1: torch.float16, 2: torch.float32}[int(out_dtype)]
    return x_i8.new_empty((x_i8.shape[0], N), dtype=dt)


def pack_gemv_interleaved(qw: LiquidQuantWeight) -> torch.Tensor:
    """Row-major nibble pack for the GEMV, interleaved per 8-nibble octet along K so the
    kernel unpacks with mask/shift only: byte b of octet h = w[8h+b] | w[8h+4+b] << 4."""
    q = qw.qweight_u4  # [N, K] uint8 in 0..15
    N, K = q.shape
    o = q.view(N, K // 8, 2, 4)              # [N, octet, half, lane]
    packed = (o[:, :, 0, :] | (o[:, :, 1, :] << 4)).to(torch.uint8)
    return packed.reshape(N, K // 2).contiguous()


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
    # Interleave for mask-shift unpack (the paper's reorder trick): byte b of 32-bit word h
    # holds elem[8h+b] in the LOW nibble and elem[8h+4+b] in the HIGH nibble, so the kernel
    # unpacks a whole register with one AND (and its pair with SHR+AND) — no lane shuffling.
    ev = out.view(*out.shape[:-1], E // 8, 2, 4)                 # [..., word, regpair, lane]
    lo = ev[..., 0, :]                                           # reg 2h lanes
    hi = ev[..., 1, :]                                           # reg 2h+1 lanes
    packed = (lo | (hi << 4)).to(torch.uint8)                    # [..., word, 4 bytes]
    return packed.reshape(RB, KT, 128, E // 2).contiguous()      # [RB, KT, 128, 32]


def build_rs_scale_pack(qw: LiquidQuantWeight) -> torch.Tensor:
    """Prepack per-group (s_u8, offset_a) in RS fragment order: uint8 [N/64, K/128, 128, 8].

    Each thread's 8 bytes per K-tile hold the 4 (scale, offset) pairs it can need —
    (row m0/m1) x (group g0/g1) — laid out [g][r] pairs (s, a). One coalesced 8-byte
    load replaces 8 scattered byte loads per tile.
    """
    coords = torch.ops.liquidgemm.wgmma_rs_a_coords().cpu()  # [128, 64, 2]
    T, E, _ = coords.shape
    m = coords[..., 0].long()
    m_reg = m.view(T, E // 4, 4)[:, :, 0]        # row per register [128, 16]
    m0 = m[:, 0]                                  # elem-0 row per thread [128]
    # each thread touches exactly 2 distinct rows (asserted in repack_rs_weight)
    is_m1 = m_reg != m0[:, None]                  # [128, 16]
    m1 = torch.where(is_m1.any(1), m_reg[torch.arange(T), is_m1.float().argmax(1)], m0)

    N, K = qw.N, qw.K
    RB, KT = N // 64, K // 128
    G = K // qw.group_size
    assert G == KT * 2, "group_size 64 with BK=128 -> 2 groups per K-tile"
    dev = qw.s_u8.device
    rows2 = torch.stack([m0, m1], 1).to(dev)      # [128, 2(r)]
    sv = qw.s_u8.view(RB, 64, KT, 2)              # [RB, row, KT, g]
    av = qw.offset_a.view(RB, 64, KT, 2)
    s_sel = sv[:, rows2]                          # [RB, 128, 2(r), KT, 2(g)]
    a_sel = av[:, rows2]
    # -> [RB, KT, 128, g, r, (s,a)] -> flatten to 8 bytes in [g][r] pair order
    pk = torch.stack([s_sel, a_sel], -1)          # [RB,128,r,KT,g,2]
    pk = pk.permute(0, 3, 1, 4, 2, 5).contiguous()  # [RB,KT,128,g,r,2]
    return pk.view(RB, KT, 128, 8).to(torch.uint8)


def w4a8_gemm_rs(x_i8: torch.Tensor, ascale: torch.Tensor, qw: LiquidQuantWeight,
                 rs_packed: torch.Tensor = None,
                 out_dtype: torch.dtype = torch.float16):
    """RS-WGMMA W4A8 linear: in-register dequant (the paper's datapath). Any M (op pads)."""
    K, N = qw.K, qw.N
    if rs_packed is None:
        w, packed = pack_nibbles(qw.qweight_u4).cuda().contiguous(), False
    else:
        w, packed = rs_packed.cuda(), True
    spack = qw.s_u8.cuda()  # scale-pack rejected (perf-neutral, +25% memory); dummy arg
    acc = torch.ops.liquidgemm.w4a8_wgmma_rs(
        x_i8.cuda().contiguous(), w, spack, qw.s_u8.cuda(), qw.offset_a.cuda(),
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
