"""Kernel microbenchmark: LiquidGEMM fused W4A8 vs torch bf16 (cuBLAS) vs unfused int8-mm.

Weights are packed/staged on-GPU once (as in real serving); only the GEMM op is timed.
Pin to an idle GPU:  CUDA_VISIBLE_DEVICES=6 python bench/microbench.py
"""

import argparse
import torch

from liquidgemm import quant, ops
from liquidgemm.pack import pack_nibbles

# (name, N, K) — real Llama-3.1-8B linear layers (down projects from K=14336).
LLAMA = [("qkv", 6144, 4096), ("o", 4096, 4096), ("gate_up", 28672, 4096), ("down", 4096, 14336)]
M_SWEEP = [1, 4, 16, 64, 256, 1024, 4096]


def _time(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def bench_shape(name, N, K, dtype=torch.bfloat16):
    dev = "cuda"
    W = (torch.randn(N, K) * 0.02)
    qw = quant.quantize_weight(W, group_size=64)

    # Pre-stage everything on-GPU (weights packed once, as at model load).
    Wg = W.to(dev, dtype)
    packed = pack_nibbles(qw.qweight_u4).to(dev).contiguous()
    s_u8 = qw.s_u8.to(dev)
    off = qw.offset_a.to(dev)
    s1 = qw.s1.to(dev)
    w_i8_full = ops.dequant_weight_i8(qw).contiguous()  # for unfused int8-mm

    rows = []
    for M in M_SWEEP:
        Xg = torch.randn(M, K, device=dev, dtype=dtype)
        x_i8 = torch.randint(-127, 127, (M, K), device=dev, dtype=torch.int8)
        ascale = torch.rand(M, device=dev) * 0.02 + 0.01

        t_bf16 = _time(lambda: Xg @ Wg.t())
        t_fused = _time(lambda: torch.ops.liquidgemm.w4a8_gemm(
            x_i8, packed, s_u8, off, s1, ascale, N, K, 64))

        def _unfused():
            m = x_i8.shape[0]
            xp = x_i8 if m >= 32 else torch.cat(
                [x_i8, x_i8.new_zeros(32 - m, K)], 0)
            acc = torch._int_mm(xp, w_i8_full.t().contiguous())
            return acc[:m]
        t_unf = _time(_unfused)

        flops = 2 * M * N * K
        w_bytes_4 = N * K // 2           # fused reads 4-bit weights
        rows.append((M, t_bf16, t_fused, t_unf, flops, w_bytes_4))

    print(f"\n== {name}  (N={N}, K={K}) ==")
    print(f"{'M':>5} | {'bf16 ms':>9} {'fused ms':>9} {'unfus ms':>9} | "
          f"{'fused GB/s':>10} {'fusedTFLOP':>10} | {'fused/bf16':>10}")
    for M, tb, tf, tu, flops, wb in rows:
        gbps = wb / (tf * 1e-3) / 1e9
        tflop = flops / (tf * 1e-3) / 1e12
        print(f"{M:>5} | {tb:9.4f} {tf:9.4f} {tu:9.4f} | "
              f"{gbps:10.0f} {tflop:10.1f} | {tb/tf:9.2f}x")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    args = ap.parse_args()
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    print(f"GPU: {torch.cuda.get_device_name(0)}  dtype={args.dtype}")
    for name, N, K in LLAMA:
        bench_shape(name, N, K, dt)
