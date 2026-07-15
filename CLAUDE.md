# LiquidGEMM

Implementation of **LiquidGEMM** ([arXiv:2509.01229](https://arxiv.org/abs/2509.01229) —
"Hardware-Efficient W4A8 GEMM Kernel for High-Performance LLM Serving"): a from-scratch
CUDA/PTX **W4A8-INT8** GEMM kernel for NVIDIA Hopper, plus a vLLM integration and a
benchmark suite (speed + accuracy) against vLLM's existing INT8 W4A8.

There is **no official code release** for the paper; the closest public reference is
QServe/QoQ (`mit-han-lab/omniserve`), which we bootstrap the quantization algorithm from.

## What we're building

- **LiquidQuant (LQQ):** two-level quantization — FP16 → per-channel INT8 in `[-119,119]`
  → group-wise UINT4 (group size **64**). Dequant back to INT8 is just **IMAD + XOR**
  (2 arith instrs / 4 elems): `Q̂_i8 = (Q_u4·s_u8 + a) ⊕ 0x80`, overflow-safe because
  `Q_u4∈[0,15]`, `s_u8≤16` ⇒ product ≤ 240 < 256, and the `0x80` XOR maps UINT8→INT8.
- **W4A8 GEMM kernel:** INT8×INT8 **WGMMA** (`wgmma.mma_async.sync ...s8`, INT32 accum),
  computes `Y = (W·Xᵀ)ᵀ`. Epilogue applies group scale + per-channel `(s, z)` + activation
  scale → bf16/fp16.
- **Implicit Fine-Grained Pipeline (ImFP):** 1 TMA "Load" warpgroup + 2 "Compute"
  warpgroups, hardware-scheduled overlap of dequant (CUDA cores) and MMA (tensor cores),
  no software sync. (Deliberately NOT the CUTLASS/Machete explicit-sync collective, which
  the paper rejects as "ExCP".)

## Hardware / target

- **Remote box only** — the local machine (macOS ARM) has no GPU and is for authoring.
- Box: **8× NVIDIA H20-3e**, Hopper **sm_90** (compile for **`sm_90a`** — WGMMA/TMA need
  the `a` variant), ~140 GB HBM3e each, driver 570, **CUDA 13.0**, torch **2.9.0a0** (NGC
  nightly), 164 CPUs, 1.6 TB RAM.
- **H20 is bandwidth-rich / tensor-core-reduced** (INT8 ~296 TOPS ≈ 15% of H100; HBM
  ~4 TB/s). It goes compute-bound at ~8× lower arithmetic intensity than H100, so the
  paper's "unblock the tensor cores" win is smaller here. **Frame gains as: half the weight
  bytes in memory-bound decode + memory capacity, not the paper's 2.9× H800 numbers.**
- **Measured HBM bandwidth (idle GPU 6): ~3.9 TB/s copy/read, ~4.35 TB/s triad** — confirms
  the ~4 TB/s spec. 150.1 GB/GPU, 78 SMs, mem clock 3201 MHz, SM clock 1980 MHz, ECC on.

### ⚠️ Shared box — check GPU occupancy before every run
Another tenant runs jobs on this box and **which GPUs are free changes over time** (at
different points: 0–5 busy, then all 8, then only 0–1 free). **Always run `nvidia-smi`
first** and pin `CUDA_VISIBLE_DEVICES` to a fully idle GPU (0% util AND 0 MiB used) — a
contended GPU silently halves bandwidth (measured 1.77 TB/s under contention vs 3.9 TB/s
idle) and poisons every benchmark. If no GPU is free, use the RunPod H100 flow below.

## Remote workflow — use `claude-ping`, not raw ssh

The box is reached via [`claude-ping`](../claude-ping) over one persistent SSH tunnel.
Config lives in `./claude-ping.json` (host `8.222.165.68`, port `1023`, `root`,
`remote_dir=/share/LiquidGEMM`).

```bash
CP=/Users/husein.z/Documents/claude-ping/claude-ping
export CLAUDE_PING_CONFIG=/Users/husein.z/Documents/LiquidGEMM/claude-ping.json
$CP up                       # open the persistent master (once)
$CP sync                     # rsync repo -> /share/LiquidGEMM
$CP exec "<cmd>"             # run on the box (reuses tunnel, retries)
$CP gpu                      # nvidia-smi summary
$CP logs 200                 # tail a logfile (one-shot, returns)
$CP down                     # close the master
```

**Rules:** never hold `follow`/`shell` (streaming) from an agent — poll one-shot verbs.
Long builds/benchmarks run **detached** (`nohup ... > log 2>&1 &`) and we poll the log.

## Python env (remote)

Use **`uv`**; venv lives under **`/share`** (survives container churn better than `/root`):

```bash
$CP exec "export PATH=/usr/local/cuda/bin:\$PATH && uv venv /share/venvs/liquidgemm --python 3.12"
# reuse the container's torch 2.9 (do NOT reinstall torch); add only build/test deps.
$CP exec "source /share/venvs/liquidgemm/bin/activate && uv pip install ninja pybind11 pytest numpy"
```

`nvcc` is at `/usr/local/cuda/bin/nvcc` (CUDA 13.0) — **not on PATH by default**, always
`export PATH=/usr/local/cuda/bin:$PATH` first.

## Build & test

- Kernel is a torch CUDA extension (`csrc/`), built with `sm_90a`. CUTLASS headers are
  **read-only references** cloned to `third_party/cutlass` on the box (not vendored/synced).
- `pytest tests/` — LiquidQuant round-trip, packing identity, kernel-vs-bf16 tolerance.
- `bench/verify.py` — correctness vs bf16 across the Llama-3.1-8B GEMM shapes + M sweep.
- `bench/microbench.py` — latency/TFLOPs vs torch bf16 and QServe's W4A8-INT8 kernel.

Llama-3.1-8B GEMM shapes (N←K): QKV `6144←4096`, O `4096←4096`, gate/up `28672←4096`,
down `4096←14336`. M swept over {1,4,16,64,256,1024,4096}.

## Repo layout

```
csrc/liquid_gemm/   # CUDA/PTX: gemm kernel, ImFP mainloop, LiquidQuant dequant, torch op
liquidgemm/         # python pkg: quantizer (LQQ), packing, torch.ops wrapper, nn layers
tests/              # pytest correctness
bench/              # verify.py + microbench.py + e2e
third_party/        # cutlass (cloned on remote; reference only)
claude-ping.json    # remote-box connection config
```

## Baselines & references

- **QServe/QoQ** `mit-han-lab/omniserve` — `kernels/csrc/qgemm/w4a8_per_group/gemm_cuda.cu`
  is the algorithm ancestor + the microbench baseline. Ampere-style (`mma.sync`+`cp.async`);
  we keep its quant/packing, swap its `vadd` dequant for IMAD+XOR, and replace its mainloop
  with a Hopper WGMMA+TMA one.
- **CUTLASS** `sm90_mma_tma_gmma_rs_warpspecialized_mixed_input.hpp` + Machete
  `machete_mainloop.cuh` — read-only refs for Hopper TMA/SMEM/WGMMA-s8 primitives.
- **vLLM baseline:** `int8_w4a8` via llm-compressor (compressed-tensors W4A8-INT8), served
  in stock vLLM, on Llama-3.1-8B-Instruct.

## Status (see bench/RESULTS.md for all numbers; git log tells the story)

The paper's kernel is implemented and optimized: RS-WGMMA with in-register IMAD+XOR
dequant (mask-shift interleaved pack — true 2-instr/4-elem), **`wait<1>` + 3-buffer
deep pipeline** (no per-tile WGMMA drain), **split-K** (env `LIQUIDGEMM_SPLITK` to tune),
and a **fused reduce+scale+cast epilogue** (`w4a8_wgmma_rs_fused`). All bit-exact.

- **Kernel beats cuBLAS bf16 at every shape/M on H20** (1.0–1.6×) and at decode tiles on
  H100 (qkv 1.09×, gate_up 1.40×, down 1.34×) — at 4× less weight memory.
- **Production guidance (user decision, 2026-07-15):** 8-bit class → vLLM native
  `quantization="fp8"` (measured ≥ our int8 mode, better fidelity, zero custom code);
  int8 mode = non-FP8-GPU (A100) fallback only. **LiquidGEMM's lane = the 4-bit class**
  (`LIQUIDGEMM_W4=1`).
- **Open problem — e2e vs kernel gap on big models at decode:** gemma-31B/H20 w4 serves
  at 32.7 tok/s b1 vs bf16 52.3 despite every GEMM being faster; fused epilogue only
  bought +6%, so the per-linear overhead hypothesis is largely disproven. A torch-profiler
  trace of one decode step (bench/profile_w4.py → bench/profile_w4.out on the box) is the
  active investigation. Suspects: eager-mode profiling parity, vision-tower/bf16-fallback
  layers in gemma-4, pad-to-64 waste at M=1, attention kernels differing between modes.
- Also open: prefill tiles (n128 / 2-warpgroup CTAs), TMA loads, calibrated checkpoints.

Full plan: `/Users/husein.z/.claude/plans/functional-orbiting-kite.md`.

- **D0 done** — scaffold, remote env (`/share/venvs/liquidgemm`), rsync side-loaded, refs cloned.
- **Phase 1 done** — `liquidgemm/quant.py` + `pack.py`, 33 tests green (XOR trick, overflow
  invariants, 4-lane SIMD dequant, quant quality). Verified against QServe's actual scheme.
- **Phase 2 done (dp4a) / Phase 3 done:**
  - ✅ CUDA extension builds on CUDA 13 / torch 2.9 / sm_90a (`setup.py`, `csrc/liquid_gemm/`).
  - ✅ `torch.ops.liquidgemm.dequant_weight` (IMAD+XOR) — bit-exact vs reference.
  - ✅ `torch.ops.liquidgemm.w4a8_gemm` — **fused**: 4-bit weights from GMEM, in-register
    dequant, dp4a. Warp-per-column GEMV (M≤16) + tiled (M>16). Verified vs reference.
  - ✅ `ops.w4a8_linear_unfused` oracle (dequant → `torch._int_mm` → scale). 63 tests green.
  - ✅ Microbench (`bench/microbench.py`) + **accuracy** (`bench/accuracy.py`). See
    `bench/RESULTS.md`: decode M=1 ≈ **1.0–1.17× cuBLAS bf16** (~800–910 GB/s); LiquidQuant
    W4 ppl **8.38 vs standard RTN int4 9.31** (bf16 7.63) on Qwen2.5-3B.
- **vLLM integration done (no fork):** out-of-tree plugin `liquidgemm/vllm_plugin.py`
  registers `quantization="liquidgemm"` (+ `vllm.general_plugins` entry point). Dedicated
  venv `/share/venvs/vllm` (vLLM 0.25.1, torch 2.11+cu130). Two modes:
  - **default (INT8/CUTLASS):** LiquidQuant weights as int8 through vLLM's
    `cutlass_scaled_mm` — fast at all M, ~2× memory, CUDA-graph-safe. gemma-4-31B @ b30:
    **1.74× bf16**, 31.7 vs 59 GiB.
  - **`LIQUIDGEMM_W4=1` (RS-WGMMA, the paper's kernel):** true 4-bit in VRAM; in-register
    dequant → INT8 WGMMA (`csrc/liquid_gemm/w4a8_wgmma.cu`, `w4a8_wgmma_rs`). Kernel
    ~1.0–1.1× bf16 at M≥128 on H20 (qkv/gate_up), 4× less weight memory. Needs K%128, N%64.
- **The paper's kernel is implemented** (RS WGMMA + in-register IMAD+XOR + fragment-order
  4-bit prepack `repack_rs_weight` + register double-buffer dequant/MMA overlap). Remaining
  perf refinements: split-K for M≤64 long-K, TMA loads, 2-CTA warp specialization.
- ⚠️ **Async-proxy rule:** any kernel writing smem with regular stores that WGMMA then
  reads via descriptors MUST issue `fence.proxy.async.shared::cta` — this was a real
  timing-dependent race (only surfaced under GPU contention).
- **RunPod fallback when the H20 box is busy** (tenant now occupies all 8 GPUs): 1× H100
  SXM via REST API, key in `.env` (`RUNPOD_API_KEY`; also `HF_TOKEN` for gated Llama).
  Create pod → ssh (PUBLIC_KEY env) → `bootstrap.sh` (uv venv, `vllm --torch-backend=cu128`
  to match the image's nvcc 12.8, build sm_90a) → `bench/h100_run.sh`. **Terminate the pod
  when done** (`DELETE /v1/pods/{id}`, ~$3/h).

Build+test on the box (always GPU 6/7):
```
export PATH=/usr/local/cuda/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0a CUDA_VISIBLE_DEVICES=6
source /share/venvs/liquidgemm/bin/activate && cd /share/LiquidGEMM
python setup.py build_ext --inplace && PYTHONPATH=$PWD python -m pytest tests/ -q
```
