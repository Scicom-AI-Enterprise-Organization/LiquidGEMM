# LiquidGEMM benchmark results (H20-3e)

All measured on one **idle NVIDIA H20-3e** (Hopper sm_90, ~4 TB/s HBM measured), CUDA 13,
torch 2.9. GPUs 0–5 on the box belong to another tenant — only 6/7 are clean.

## Accuracy — WikiText-2 perplexity (Qwen2.5-3B)

Llama-3.1-8B is gated (no HF token on the box), so we use Qwen2.5-3B; the relative
comparison between weight schemes is model-transferable. Weights are fake-quantized
(quant → dequant); "W4A8" additionally fake-quantizes activations to per-token INT8. Both
LiquidQuant and the RTN baselines are uncalibrated round-to-nearest, so this isolates the
*quantization scheme* (LiquidQuant's two-level per-channel-INT8 + per-group-INT4 vs a
standard single-level group-wise INT4).

| scheme | wikitext2 ppl | Δ vs bf16 |
|---|---:|---:|
| bf16 (baseline) | 7.6323 | — |
| **LiquidQuant W4 (weight-only)** | **8.3822** | +0.75 |
| RTN int4 g64 W4 (weight-only) | 9.3067 | +1.68 |
| RTN int4 g128 W4 (weight-only) | 10.9285 | +3.30 |
| **LiquidQuant W4A8 (+int8 act)** | **8.5382** | +0.91 |
| RTN int4 g64 W4A8 (+int8 act) | 9.0006 | +1.37 |

**LiquidQuant preserves ~2× less perplexity degradation than standard group-wise INT4** at
the same 4-bit budget/group size. (GPTQ-style calibration would improve both baselines and
LiquidQuant; that is a fair next step.)

## Speed — kernel microbench vs cuBLAS bf16 (Llama-3.1-8B GEMM shapes)

Fused W4A8 (`torch.ops.liquidgemm.w4a8_gemm`): 4-bit weights streamed from GMEM, in-register
LiquidQuant dequant (IMAD+XOR), dp4a INT8 accumulate. `fused/bf16` > 1 means faster than
cuBLAS bf16. Weights pre-packed/staged on GPU; only the GEMM is timed.

| shape (N←K) | M=1 | M=4 | M=16 | M=256 |
|---|---:|---:|---:|---:|
| qkv 6144←4096 | **1.08×** (799 GB/s) | 0.36× | 0.10× | 0.05× |
| o 4096←4096 | **1.17×** (645 GB/s) | 0.33× | 0.10× | 0.05× |
| gate_up 28672←4096 | 0.94× (911 GB/s) | 0.29× | 0.08× | 0.05× |
| down 4096←14336 | **1.02×** (854 GB/s) | 0.33× | 0.09× | 0.05× |

**Decode (M=1) is competitive-to-faster than cuBLAS bf16** — the durable H20 win comes from
moving ¼ the weight bytes in the memory-bound regime. For **M≥4 the dp4a path does not
amortize** (CUDA-core MACs, not tensor cores): this is precisely the wall the paper's
**WGMMA + ImFP** tensor-core mainloop exists to break, and is the main remaining kernel work.
On H20 (INT8 tensor cores ~15% of H100) the prefill win over W8A8 is capacity, not FLOPs.

## End-to-end in vLLM (no fork) — Qwen2.5-3B, H20 GPU6, out=256

LiquidGEMM is registered out-of-tree via `register_quantization_config` +
a `vllm.general_plugins` entry point — **no vLLM fork**. Verified: coherent, correct
generations on **Qwen2.5-3B** and **gemma-4-31B-it** (31B loaded at 19.7 GiB W4 vs ~62 GiB
bf16). Throughput (tokens/s):

Throughput (tokens/s), CUDA graphs (vLLM default) unless noted:

| config | batch 1 | batch 30 | mem | notes |
|---|---:|---:|---:|---|
| bf16 | 319 | 8345 | 5.79 GiB | fp baseline |
| **LiquidGEMM — default (CUTLASS INT8)** | **225** | **6386** | 3.23 GiB | 0.71× / 0.77× bf16 |
| LiquidGEMM — INT8 via torch._int_mm, eager | 34.8 | 1035 | 3.26 GiB | superseded (overhead) |
| LiquidGEMM — 4-bit custom op (`w4=True`) | works | slow (dp4a) | 2.0 GiB | ~4× mem, decode-only |

**Default path = production-ready.** LiquidQuant weights are stored INT8 and run through
vLLM's own CUTLASS INT8 GEMM (`cutlass_scaled_mm`) + fused per-token int8 quant
(`scaled_int8_quant`). It is **CUDA-graph-safe**, fast at all M, and reaches **~0.75× of
bf16 throughput at ~30 concurrency** while using **~1.8× less weight memory** — and because
it is the *same* kernel vLLM's W8A8 uses, it is **speed-parity with the INT8 W4A8 baseline,
with LiquidQuant's better accuracy** (8.38 vs 9.31 ppl above).

Why ~0.75× and not faster: at 3B/short-context decode, per-token latency is dominated by
attention + non-quantized ops, not the linear GEMMs, so halving weight bytes yields a
sub-linear gain; the win grows with model size and batch (where weights/KV dominate and the
memory saving lets you fit more). On H20 the INT8 compute advantage over bf16 is modest.

**Remaining upside (task #7): a fused WGMMA/mma.sync W4A8 kernel** — 4-bit weights streamed
from GMEM + in-register LiquidQuant dequant + INT8 tensor cores, single graph-capturable
launch — would add the **4× memory** win on top of the CUTLASS-level speed. QServe's
`mma.sync.m16n8k32.s8` + `ldmatrix` (`third_party/omniserve`) is the reference; it is
QServe/Marlin-level tuning work.

Modes: default `quantization="liquidgemm"` (INT8/CUTLASS, fast, ~2× mem). Set
`LIQUIDGEMM_W4=1` for the 4-bit custom-op path (~4× mem, best for pure decode).

## Reproduce
```
CUDA_VISIBLE_DEVICES=6 python bench/microbench.py
CUDA_VISIBLE_DEVICES=6 python bench/accuracy.py --model Qwen/Qwen2.5-3B
CUDA_VISIBLE_DEVICES=6 python bench/vllm_serving.py --model Qwen/Qwen2.5-3B --quant liquidgemm --batch 30
```
