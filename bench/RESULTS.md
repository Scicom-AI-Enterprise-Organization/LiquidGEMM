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

| config | batch 1 | batch 30 | notes |
|---|---:|---:|---|
| bf16, **CUDA graphs** (vLLM default) | 319 | 8345 | production baseline |
| bf16, eager | 88 | 2645 | |
| **LiquidGEMM (INT8 tensor-core), eager** | 34.8 | 1035 | fused quant→int_mm→scale |
| LiquidGEMM, CUDA graphs | ✗ | ✗ | crashes: `_int_mm` needs M>16, breaks capture |

**Realized wins:** (1) **memory** — INT8 mode ~2× (3.26 vs 5.79 GiB on Qwen2.5-3B), 4-bit
mode ~3–4× (2.0 GiB; gemma-31B 19.7 vs ~62 GiB) → fit bigger models / more KV / higher
concurrency; (2) **accuracy** — LiquidQuant W4 beats standard int4 (above).

**Not yet realized — throughput at concurrency.** LiquidGEMM (eager) is ~2.5× behind
bf16-eager and ~8× behind bf16-with-CUDA-graphs. Root causes, established empirically:
1. It isn't a mature fused tensor-core GEMM — `torch._int_mm` (cuBLASLt) carries per-call
   overhead and its **M>16 restriction breaks CUDA-graph capture**, forcing `enforce_eager`.
2. Eager mode forgoes the 3.6× CUDA-graph speedup that vLLM's bf16 enjoys.
3. H20's INT8 tensor cores (~15% of H100) give too little compute headroom to overcome (1).

**Path to a real throughput win** (the paper's core): a hand-written **fused WGMMA (or
mma.sync) INT8 W4A8 kernel** — 4-bit weights streamed from GMEM, in-register LiquidQuant
dequant, INT8 tensor cores, single launch, **CUDA-graph-capturable for all M** (replacing
`_int_mm`). QServe's `mma.sync.m16n8k32.s8` + `ldmatrix` (in `third_party/omniserve`) is the
reference. This is QServe/Marlin-level kernel engineering and is the main remaining work.

Current vLLM usage requires `enforce_eager=True`.

## Reproduce
```
CUDA_VISIBLE_DEVICES=6 python bench/microbench.py
CUDA_VISIBLE_DEVICES=6 python bench/accuracy.py --model Qwen/Qwen2.5-3B
CUDA_VISIBLE_DEVICES=6 python bench/vllm_serving.py --model Qwen/Qwen2.5-3B --quant liquidgemm --batch 30
```
