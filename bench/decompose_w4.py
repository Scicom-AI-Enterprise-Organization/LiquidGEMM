"""Per-shape decode decomposition: real gemma-4-31B linear shapes, bf16 vs w4 path.

Reconciles kernel-level wins with the end-to-end gap: times each linear at M=1 in both
modes, sums over layers, and compares against measured e2e tokens/s.
"""

import glob
import json

import torch

import liquidgemm.ops as ops
from liquidgemm import quant

cfg_path = glob.glob(
    "/root/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/*/config.json")[0]
cfg = json.load(open(cfg_path))
tc = cfg.get("text_config", cfg)
H = tc["hidden_size"]
I = tc["intermediate_size"]
L = tc["num_hidden_layers"]
nh = tc["num_attention_heads"]
nkv = tc["num_key_value_heads"]
hd = tc.get("head_dim", H // nh)
qkv_out = (nh + 2 * nkv) * hd
shapes = [("qkv", qkv_out, H), ("o", H, nh * hd), ("gate_up", 2 * I, H), ("down", H, I)]
print(f"H={H} I={I} L={L} nh={nh} nkv={nkv} hd={hd}", flush=True)


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


tot_bf = tot_w4 = 0.0
for name, N, K in shapes:
    aligned = (N % 64 == 0 and K % 128 == 0)
    W = torch.randn(N, K) * 0.02
    Wg = W.cuda().to(torch.bfloat16)
    x = torch.randn(1, K, device="cuda", dtype=torch.bfloat16)
    tb = t(lambda: x @ Wg.t())
    if aligned:
        qw = quant.quantize_weight(W, 64)
        rs = ops.repack_rs_weight(qw).cuda()
        su8, off, s1 = qw.s_u8.cuda(), qw.offset_a.cuda(), qw.s1.cuda()

        def w4():
            xi, asc = torch.ops.liquidgemm.quant_per_token(x)
            return torch.ops.liquidgemm.w4a8_wgmma_rs_fused(
                xi, rs, su8, off, asc, s1, N, K, 64, 0)
        tw = t(w4)
    else:
        tw = tb  # plugin falls back to bf16 for unaligned layers
    print(f"{name:8s} N={N:6d} K={K:6d} aligned={aligned} "
          f"bf16 {tb*1e3:7.1f}us  w4 {tw*1e3:7.1f}us  ({tb/tw:.2f}x)", flush=True)
    tot_bf += tb
    tot_w4 += tw

print(f"\nper-layer linears: bf16 {tot_bf*1e3:.0f}us  w4 {tot_w4*1e3:.0f}us", flush=True)
print(f"x {L} layers: bf16 {tot_bf*L:.2f}ms  w4 {tot_w4*L:.2f}ms", flush=True)
print("measured e2e/token: bf16 ~19.1ms  w4 ~30.6ms", flush=True)
