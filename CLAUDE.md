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

### ⚠️ Shared box — use GPUs 6 and 7 only
GPUs **0–5 are occupied by another tenant** (≈62 GB + 100% util each). **GPUs 6 and 7 are
free.** Always `export CUDA_VISIBLE_DEVICES=6` (or 7) for builds/benchmarks, and **run
`nvidia-smi` before any benchmark** — a contended GPU silently halves bandwidth (measured
1.77 TB/s under contention vs 3.9 TB/s idle). Never benchmark on 0–5.

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

## Status

Full plan: `/Users/husein.z/.claude/plans/functional-orbiting-kite.md`.

- **D0 done** — scaffold, remote env (`/share/venvs/liquidgemm`), rsync side-loaded, refs cloned.
- **Phase 1 done** — `liquidgemm/quant.py` + `pack.py`, 33 tests green (XOR trick, overflow
  invariants, 4-lane SIMD dequant, quant quality). Verified against QServe's actual scheme.
- **Phase 2 in progress:**
  - ✅ CUDA extension builds on CUDA 13 / torch 2.9 / sm_90a (`setup.py`, `csrc/liquid_gemm/`).
  - ✅ `torch.ops.liquidgemm.dequant_weight` (IMAD+XOR) — bit-exact vs reference (10 tests).
  - ✅ `ops.w4a8_linear_unfused` correctness oracle (dequant → `torch._int_mm` → scale) —
    end-to-end numerics validated at all M incl. decode (20 tests). NOTE: `torch._int_mm`
    needs M>16, so it pads; also it's unfused (extra GMEM traffic) → NOT the deliverable.
  - ⏭️ **NEXT: the fused kernel.** (a) correctness-first tiled int8 GEMM with in-register
    LiquidQuant dequant using `mma.sync.m16n8k32.s8` (mirror QServe's mainloop, swap dequant);
    (b) upgrade to WGMMA + TMA + ImFP (1 Load WG + 2 Compute WGs) for the H20 perf win;
    (c) expose as `torch.ops.liquidgemm.w4a8_gemm`. Verify each step vs `quant.w4a8_matmul`.

Build+test on the box (always GPU 6/7):
```
export PATH=/usr/local/cuda/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0a CUDA_VISIBLE_DEVICES=6
source /share/venvs/liquidgemm/bin/activate && cd /share/LiquidGEMM
python setup.py build_ext --inplace && PYTHONPATH=$PWD python -m pytest tests/ -q
```
