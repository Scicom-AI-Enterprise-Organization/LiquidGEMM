# LiquidGEMM

An open implementation of **[LiquidGEMM: Hardware-Efficient W4A8 GEMM Kernel for
High-Performance LLM Serving](https://arxiv.org/abs/2509.01229)** (SC '25) — a from-scratch
Hopper (sm_90a) CUDA/CuTe kernel plus an out-of-tree **vLLM plugin** (no fork), with an
honest benchmark suite against bf16, FP8-dynamic, and INT8 baselines.

The paper has no official code release; the closest public reference is QServe/QoQ. This
repo implements both of the paper's contributions:

- **LiquidQuant (LQQ)** — two-level weight quantization: FP16 → per-channel INT8 in the
  protective range `[-119, 119]` → group-wise UINT4 (group 64). Dequant back to INT8 is
  overflow-safe **IMAD + XOR** — 2 arithmetic instructions per 4 elements — enabled by a
  nibble-interleaved weight packing (mask-shift unpack, no lane shuffling).
- **The RS-WGMMA kernel** (`csrc/liquid_gemm/w4a8_wgmma.cu`) — weights are the WGMMA **A
  operand sourced from registers** (`Y=(W·Xᵀ)ᵀ`): each thread dequantizes its fragment
  straight from 4-bit GMEM (weights never touch shared memory), activations stream through
  swizzled smem as the B descriptor, INT8 WGMMA accumulates in INT32. The mainloop keeps a
  WGMMA batch in flight across K-tiles (`wait<1>` + 3-buffer rotation — the paper's ImFP
  overlap), with **split-K** for long-K decode shapes and a fused reduce+scale+cast
  epilogue. Fragment layouts are correct-by-construction (CuTe identity-tensor coordinate
  dump → fragment-order prepack), and every kernel is validated **bit-exactly** against a
  pure-PyTorch reference.

## Measured results (see `bench/RESULTS.md` for everything)

**Kernel vs cuBLAS bf16, decode tile M=64, all bit-exact:**

| GEMM (Llama-3.1-8B shapes) | H20-3e | H100 SXM |
|---|---:|---:|
| qkv 6144←4096 | 1.01–1.62× | 1.09× |
| gate_up 28672←4096 | 1.27–1.57× | 1.40× |
| down 4096←14336 | 1.30–1.47× | 1.34× |

On H20 the kernel beats bf16 at **every shape and every M** with **4× less weight memory**.

**Accuracy (WikiText-2 PPL):** LiquidQuant W4 8.38 vs standard RTN int4-g64 9.31 on
Qwen2.5-3B (bf16 7.63); parity with RTN-g64 on Llama-3.1-8B — the LQQ edge is
model-dependent (outlier-heavy weights benefit most).

**End-to-end serving (vLLM, CUDA graphs):** Llama-3.1-8B on H100 — w4 mode 137 tok/s b1 /
3679 b30 at **5.83 GiB** (bf16: 158 / 4320 at 15.0 GiB). On Hopper, for the 8-bit class we
recommend vLLM's native `quantization="fp8"`; **LiquidGEMM's lane is the 4-bit class**
(fit bigger models / more KV). Per-linear framework overhead at small batch is the current
end-to-end gap on large models and is under active profiling.

## Use it in vLLM (no fork)

```bash
pip install -e .            # builds the sm_90a extension (needs CUDA >= 12.8, Hopper)
```

```python
from vllm import LLM
llm = LLM("Qwen/Qwen2.5-3B", quantization="liquidgemm")   # registered via entry point
```

Modes:
- default — LiquidQuant weights as INT8 through vLLM's CUTLASS `cutlass_scaled_mm`
  (~2× less memory, fastest; kept mainly for non-FP8 GPUs like A100).
- `LIQUIDGEMM_W4=1` — true 4-bit in VRAM via the RS-WGMMA kernel (~4× less memory).
- `LIQUIDGEMM_SPLITK=n` — override the split-K factor for tuning.

## Repo layout

```
csrc/liquid_gemm/     CUDA/CuTe kernels: w4a8_wgmma.cu (RS kernel), dequant, epilogues
liquidgemm/           quant.py (LQQ reference) · pack.py · ops.py · vllm_plugin.py
tests/                bit-exactness suite (quantizer, packing, every kernel)
bench/                rs_bench.py · microbench.py · accuracy.py · vllm_serving.py · RESULTS.md
third_party/          CUTLASS (headers, reference only — cloned at build time)
```

## Build & test (on a Hopper box)

```bash
export TORCH_CUDA_ARCH_LIST=9.0a
python setup.py build_ext --inplace
pytest tests/ -q                      # 100+ tests, all bit-exact comparisons
python bench/rs_bench.py              # kernel vs cuBLAS bf16
python bench/vllm_serving.py --model Qwen/Qwen2.5-3B --quant liquidgemm --batch 30
```

`LIQUIDGEMM_SKIP_CUDA_CHECK=1` permits building with nvcc one CUDA major older than the
torch runtime (e.g. nvcc 12.8 with torch+cu130) — sm_90a cubins are forward-compatible.

## Status & roadmap

Implemented: LiquidQuant · RS-WGMMA with in-register dequant · ImFP-style dequant/MMA
overlap · split-K · fused epilogue · vLLM plugin (CUDA-graph-safe) · full benchmark suite.
Open: per-linear framework overhead at small batch (profiling), prefill tiles (n128 /
2-warpgroup CTAs), TMA loads, llm-compressor-calibrated checkpoints.

MIT-style; see the paper for the original algorithm design.
