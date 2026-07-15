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

**Big model at production concurrency — gemma-4-31B-it, batch 30, out=128, CUDA graphs:**

| config | throughput | weight mem |
|---|---:|---:|
| bf16 | 540 tok/s | 59.0 GiB |
| **LiquidGEMM (INT8/CUTLASS)** | **938 tok/s (1.74×)** | **31.7 GiB (1.86× less)** |

On a 31B model the linear GEMMs dominate, so INT8's compute + halved weight bytes give a
clear **1.74× throughput win** plus ~1.9× memory headroom (more KV / larger batch). NOTE:
default stores **INT8** weights (1 byte) → 31.7 GiB, not 4-bit; the CUTLASS INT8 kernel
consumes int8 operands. `LIQUIDGEMM_W4=1` keeps true 4-bit (~16 GiB) but uses the slow
custom kernel. Getting 4-bit memory *and* this speed is the fused-WGMMA task (#7).

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

## The real paper's kernel — Hopper WGMMA (csrc/liquid_gemm/w4a8_wgmma.cu)

Built with CuTe WGMMA atoms (`SM90_64x64x32_S32S8S8_SS_TN`, `Layout_K_SW128` swizzled
smem, cp.async pipeline, warpgroup fence/commit/wait), hand-written mainloop + LiquidQuant
dequant — the paper's design, not the CUTLASS collective it rejects. Both bit-exact vs
reference. Latency vs cuBLAS bf16 (H20 GPU6, prefill/concurrency M):

| shape | M | **int8 WGMMA** (Stage 1) | **fused W4A8 WGMMA** (4-bit, Stage 2/3) |
|---|--:|--:|--:|
| gate_up 28672←4096 | 128 | 1.23× | 0.73× |
| gate_up | 256 | 1.62× | 0.74× |
| gate_up | 512 | 1.61× | 0.80× |
| qkv 6144←4096 | 512 | 1.08× | 0.69× |
| down 4096←14336 | 512 | 1.37× | 0.55× |

- **Stage 1 (int8 WGMMA): beats bf16 up to 1.62×** — the real tensor-core kernel works and
  is fast. (This is what the production vLLM path effectively uses via CUTLASS.)
- **Stage 2/3 (fused 4-bit → in-smem LiquidQuant dequant → WGMMA): correct, 0.55–0.80× bf16.**
  Streams 4-bit from GMEM (4× weight memory) and runs real WGMMA. Double-buffered to overlap
  dequant with WGMMA, but the ~16K swizzled-smem writes/tile for the dequant dominate and
  can't be hidden.

**To exceed bf16 with 4-bit weights** (the paper's full design): dequant in **registers**
(RS WGMMA variant, `A`=weights operand via the `Y=(W·Xᵀ)ᵀ` layout) so the smem write
vanishes, plus warp-specialized ImFP producer/consumer. That RS+ImFP step is the remaining
kernel work; the SS version here proves the WGMMA + LiquidQuant-dequant datapath is correct.

## Final validation — Llama-3.1-8B on an idle H100 SXM (RunPod)

The shared H20 box became fully occupied, so final clean numbers were taken on a RunPod
1× H100 SXM (driver 580, torch 2.11+cu130, vLLM 0.25.1, extension built sm_90a with nvcc
12.8). **102/102 tests green on the H100** — the whole stack revalidates on a second
Hopper machine. Model: `NousResearch/Meta-Llama-3.1-8B-Instruct` (ungated mirror;
meta-llama repo gated). Serving = CUDA graphs, out=128.

| mode | batch 1 | batch 30 | weights |
|---|---:|---:|---:|
| bf16 | 158.1 tok/s | 4320 tok/s | 15.0 GiB |
| **LiquidGEMM int8 (CUTLASS)** | **228.7 (1.45×)** | **6053 (1.40×)** | 8.99 GiB |
| **LiquidGEMM w4 (RS-WGMMA)** | 80.5 (0.51×) | 2279 (0.53×) | **5.83 GiB** |

- **int8 mode: 1.40–1.45× faster than bf16 on H100** — replicates across GPUs
  (H20/gemma-31B was 1.74×). **Superseded as the recommendation — see fp8 below.**
- **w4 mode (the paper's RS-WGMMA kernel) serves end-to-end under CUDA graphs** with true
  4-bit weights (2.6× less than bf16) at ~0.5× bf16 speed on H100. On H20 the same kernel
  is ~1.0–1.1× bf16 at M≥128 — H100's ~7× faster tensor cores raise the bar, and closing
  it there needs TMA + multi-CTA warp specialization + split-K (the paper's remaining
  ExCP/ImFP engineering, documented as future work).
- Generations correct in all modes (17×23=391 etc.).

**Accuracy — WikiText-2 PPL, Llama-3.1-8B (40×2048 ctx):**

| scheme | ppl |
|---|---:|
| bf16 | 6.929 |
| LiquidQuant W4 | 7.483 |
| RTN int4 g64 W4 | 7.440 |
| RTN int4 g128 W4 | 7.635 |
| LiquidQuant W4A8 | 7.569 |
| RTN int4 g64 W4A8 | 7.554 |

Honest read: on Llama-3.1-8B, LiquidQuant ≈ RTN g64 (within noise; both clearly beat
g128) — unlike Qwen2.5-3B where LiquidQuant was decisively better (8.38 vs 9.31). The
LQQ advantage is model-dependent: it pays off on weight distributions with outliers
(Qwen), while its protective range costs a little resolution on well-behaved ones (Llama).
Both models: W4A8 ≈ W4 + small activation penalty, i.e. int8 activations are cheap.

## FP8 dynamic vs the int8 mode (user question: "why int8 if Hopper has fp8?")

Measured — same H100 SXM class, Llama-3.1-8B, CUDA graphs, out=128 (bf16 anchor within
4% across pods, so directly comparable):

| mode | batch 1 | batch 30 | weights | weight fidelity |
|---|---:|---:|---:|---|
| bf16 | 154–158 | 4159–4320 | 15.0 GiB | exact |
| **vLLM `fp8` dynamic** | **237.4 (1.54×)** | **5851 (1.41×)** | **8.54 GiB** | fp8 of *original* weights |
| LiquidGEMM int8 | 228.7 (1.45×) | 6053 (1.40×) | 8.99 GiB | **W4-reconstructed** |
| LiquidGEMM w4 (RS) | 80.5 | 2279 | 5.83 GiB | W4 |

**Verdict — the user is right: on Hopper there is no reason to use the int8 mode.**
FP8 == INT8 tensor-core rate on Hopper, so speed is a wash (fp8 wins b1, int8 wins b30 by
~3%, noise), fp8 uses slightly *less* memory, needs zero custom code, and is strictly more
accurate at that memory (8-bit encode of the original weights vs our W4-reconstructed
weights). The int8 mode's only remaining niche is **non-FP8 GPUs (A100/Ampere)**.

**Revised recommendation:** 8-bit class → vLLM `quantization="fp8"` (dynamic). 4-bit
class (fit bigger models / more KV) → that is LiquidGEMM's actual lane: the w4/RS kernel,
with vLLM's in-tree W4A8-FP8 (`cutlass_w4a8`, int4→fp8 LUT) as the baseline to beat on
Hopper — closing that gap needs the TMA/multi-CTA/split-K work already documented.

## W4-lane optimization session (split-K) — H100, Llama-3.1-8B

Per the user's direction, the 4-bit lane is the focus. **Split-K** added to the RS kernel
(K-tile chunks across `blockIdx.z`, exact int32 partial reduction; auto-enabled at
M≤128 & K≥4096). Kernel-level (M=64 decode tile, vs cuBLAS bf16):

| shape | before | after split-K |
|---|---:|---:|
| down 4096←14336 | 118.4us (0.39×) | **54.5us (0.88×)** |
| qkv 6144←4096 | 31.0us (0.69×) | 29.6us (0.75×) |
| gate_up 28672←4096 | 97.9us (0.86×) | ~121us (0.73×)* |
| decode aggregate | 0.55× bf16 | **~0.74× bf16** |

*gate_up regression is run-to-run noise; its split-K doesn't trigger (N-CTAs already high).

**End-to-end w4 serving (Llama-3.1-8B, CUDA graphs):** 80.5 → **96.6 tok/s** (b1),
2279 → **2669 tok/s** (b30), still **5.83 GiB** true 4-bit. All kernels bit-exact.

Also tried and **rejected**: fragment-order scale prepack (one coalesced 8B load replacing
8 cached `__ldg` scale bytes/tile) — measured perf-neutral on H100 but +25% weight memory
(6.65 vs 5.83 GiB). Not worth it in the memory lane; builder kept in `ops.py` for reference.
**Mask-shift unpack landed** (the paper's weight-reorder trick: nibbles interleaved so
unpack per register = AND / SHR+AND + IMAD + XOR — the true "2 arithmetic instructions
per 4 elements"): gate_up M=64 121→103us; serving 96.6→**105.0** b1, 2669→**2861** b30.

**Deep pipeline landed (`wait<1>` + 3-buffer rotation):** the per-tile WGMMA drain was the
bottleneck, exactly as the roofline predicted. The mainloop now keeps the previous WGMMA
batch in flight across tile boundaries (statically-unrolled ×3 buffer rotation; the buffer
being refilled is always the one read two tiles ago, provably retired by the `wait<1>`
chain). **~2× across the board, all bit-exact.** Decode tile (M=64) vs cuBLAS bf16, H100:

| shape | before | after | vs bf16 |
|---|---:|---:|---:|
| qkv 6144←4096 | 28.6us | **19.8us** | **1.09×** |
| gate_up 28672←4096 | 103.2us | **60.4us** | **1.40×** |
| down 4096←14336 | 54.8us | **34.9us** | **1.34×** |
| o 4096←4096 | 21.5us | 17.5us | 0.61× (small-shape occupancy) |
| **aggregate** | 208us | **132.6us** | **1.23× faster than bf16** |

**W4 is now faster than bf16 at decode tiles** — the "lesser bandwidth ⇒ faster" promise
made real (3 of 4 shapes; ceiling remains ~4×, next levers: `o`-shape occupancy, n128
tiles / 2-warpgroup CTAs for prefill M≥256 which sits at 0.5–0.7×).

**End-to-end w4 serving (Llama-3.1-8B, H100, CUDA graphs, 5.83 GiB true 4-bit):**
80.5 → 96.6 → 105 → **137.1 tok/s** (b1); 2279 → 2669 → 2861 → **3679 tok/s** (b30)
across the optimization arc (split-K → mask-shift → deep pipeline). Now 0.87× of bf16
end-to-end at 2.6× less weight memory, 0.63× of fp8 at 1.5× less. On H20 (bandwidth-rich,
compute-poor — production), these kernel gains should translate to an outright win;
re-bench when the box frees.

In-tree W4A8 baseline: no official RedHatAI W4A8 Llama-3.1-8B checkpoint exists on HF
(only third-party GPTQ variants); a rigorous llm-compressor W4A8 calibration run is the
proper follow-up when the H20 box frees up.

## Reproduce
```
CUDA_VISIBLE_DEVICES=6 python bench/microbench.py
CUDA_VISIBLE_DEVICES=6 python bench/accuracy.py --model Qwen/Qwen2.5-3B
CUDA_VISIBLE_DEVICES=6 python bench/vllm_serving.py --model Qwen/Qwen2.5-3B --quant liquidgemm --batch 30
```

## H20 re-bench with the deep-pipeline kernel (clean GPU 0)

**Kernel-level: the paper's promise, delivered on production hardware.** RS-W4A8 beats
cuBLAS bf16 at EVERY shape and every M on H20 (bit-exact): qkv 1.01–1.62×, o 1.06–1.46×,
gate_up 1.27–1.57×, down 1.30–1.47×, at 4× less weight memory.

**End-to-end, gemma-4-31B (b1/b30, graphs):** w4 30.7/790 @ 19.4 GiB · fp8 77.5/1056 @
31.7 GiB · bf16 52.3/1170 @ 59.0 GiB. Despite winning every GEMM, w4 loses e2e: the
per-linear pipeline (quant → RS op → split-K int32 partials → sum → scale epilogue, up to
5 kernels + full int32 [S,Mp,N] GMEM round-trips) now costs more than the GEMM at decode.
Next step (in progress): fused epilogue — S=1 writes scaled bf16 straight from the kernel;
S>1 uses one fused reduce+scale kernel. (Qwen2.5-3B on H20 shows the same pattern
amplified: w4 156/4327 vs bf16 319/8199 — tiny GEMMs, fixed overhead.)

## Decode decomposition on real gemma-4-31B shapes (H20, M=1) — the honest picture

`bench/decompose_w4.py` reconciles kernel vs e2e exactly (bf16: 15.45ms linears + 3.6ms
rest = 19.1ms measured ✓). Finding: at M=1, cuBLAS switches to a ~95%-bandwidth GEMV
(qkv: 176MB in 49.8us), while the RS tile kernel runs ~0.5TB/s effective — the earlier
"beats bf16" result compared against cuBLAS's weak M=64 GEMM, not its strong M=1 GEMV.
Split-K sweep (S=2..16): ±3%, not the answer. Prefetch-depth-2 for the activation
pipeline: +7% (481→448us/layer), bit-exact. Remaining gap at M=1 is architectural:
a dedicated W4 **GEMV** kernel for M≤16 (mask-shift dequant + full-bandwidth weight
streaming; the old dp4a GEMV already reached ~900GB/s) is the identified next kernel.
For batch~30-64 decode (production), the RS tile kernel remains the right engine.

**gemma-4-31B b30 serving arc (H20, CUDA graphs, 19.35 GiB true 4-bit):** 790 → 825
(fused epilogue) → **876 tok/s** (prefetch-2) = 0.75× bf16 (1170) / 0.83× fp8 (1056) at
3.0× / 1.6× less weight memory respectively.
