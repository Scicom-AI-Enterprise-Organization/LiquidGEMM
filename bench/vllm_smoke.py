"""End-to-end smoke test: run a bf16 model in vLLM with quantization='liquidgemm'."""

import argparse

from vllm import LLM, SamplingParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--quant", default="liquidgemm", help="liquidgemm or none")
    args = ap.parse_args()

    kw = dict(enforce_eager=True, gpu_memory_utilization=0.55, max_model_len=2048,
              dtype="bfloat16")
    if args.quant != "none":
        kw["quantization"] = args.quant
    llm = LLM(args.model, **kw)

    prompts = [
        "The capital of France is",
        "Q: What is 17 * 23? A:",
        "def fibonacci(n):",
    ]
    sp = SamplingParams(max_tokens=48, temperature=0.0)
    for o in llm.generate(prompts, sp):
        print("PROMPT:", repr(o.prompt))
        print("OUTPUT:", repr(o.outputs[0].text))
        print("-" * 60)


if __name__ == "__main__":
    main()
