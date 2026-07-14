"""Kernel-level benchmark of the RS W4A8 WGMMA (in-register dequant) vs cuBLAS bf16
and the Stage-1 int8 WGMMA, across Llama-3.1-8B GEMM shapes and batch sizes."""

import torch

import liquidgemm.ops as ops
from liquidgemm import quant


def t(fn, it=50, w=10):
    for _ in range(w):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(it):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / it


LLAMA = [("qkv", 6144, 4096), ("o", 4096, 4096), ("gate_up", 28672, 4096), ("down", 4096, 14336)]

print(torch.cuda.get_device_name(0))
for name, N, K in LLAMA:
    W = torch.randn(N, K) * 0.02
    qw = quant.quantize_weight(W, 64)
    rs = ops.repack_rs_weight(qw).cuda()
    su8, off = qw.s_u8.cuda(), qw.offset_a.cuda()
    Wg = W.cuda().to(torch.bfloat16)
    wi8 = quant.dequantize_i8(qw).cuda().contiguous()

    # correctness spot check
    xc = torch.randint(-127, 127, (128, K), device="cuda", dtype=torch.int8)
    ok = torch.equal(
        torch.ops.liquidgemm.w4a8_wgmma_rs(xc, rs, su8, off, N, K, 64, True),
        torch._int_mm(xc, quant.dequantize_i8(qw).cuda().t().contiguous()))
    print(f"\n== {name} N={N} K={K}  correct={ok} ==")
    for M in [64, 128, 256, 512, 1024]:
        Xg = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        xi = torch.randint(-127, 127, (M, K), device="cuda", dtype=torch.int8)
        tb = t(lambda: Xg @ Wg.t())
        tr = t(lambda: torch.ops.liquidgemm.w4a8_wgmma_rs(xi, rs, su8, off, N, K, 64, True))
        line = f"  M={M:>4}: bf16 {tb*1e3:7.1f}us | RS-w4a8 {tr*1e3:7.1f}us ({tb/tr:.2f}x)"
        if M % 128 == 0:  # Stage-1 i8 kernel requires M%128
            ti = t(lambda: torch.ops.liquidgemm.wgmma_i8_gemm(xi, wi8))
            line += f" | i8-wgmma {ti*1e3:7.1f}us ({tb/ti:.2f}x)"
        print(line)
