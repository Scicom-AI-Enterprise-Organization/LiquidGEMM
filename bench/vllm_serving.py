"""Serving throughput: LiquidGEMM W4A8 vs bf16 baseline in vLLM (decode-dominated)."""

import argparse
import time

from vllm import LLM, SamplingParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--quant", default="liquidgemm", help="liquidgemm | none")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out-len", type=int, default=256)
    ap.add_argument("--eager", type=int, default=1, help="1=enforce_eager, 0=cuda graphs")
    args = ap.parse_args()

    kw = dict(enforce_eager=bool(args.eager), gpu_memory_utilization=0.6,
              max_model_len=2048, dtype="bfloat16", seed=0)
    if args.quant != "none":
        kw["quantization"] = args.quant
    llm = LLM(args.model, **kw)

    prompt = "Once upon a time"  # short prompt -> decode dominates
    prompts = [prompt] * args.batch
    sp = SamplingParams(max_tokens=args.out_len, temperature=0.0, ignore_eos=True)

    llm.generate(prompts[:1], SamplingParams(max_tokens=8, ignore_eos=True))  # warmup
    t = time.perf_counter()
    outs = llm.generate(prompts, sp)
    dt = time.perf_counter() - t
    gen = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"RESULT quant={args.quant} batch={args.batch} out={args.out_len} "
          f"| {gen} gen-tok in {dt:.2f}s = {gen/dt:.1f} tok/s "
          f"({gen/dt/args.batch:.1f} tok/s/seq)")


if __name__ == "__main__":
    main()
