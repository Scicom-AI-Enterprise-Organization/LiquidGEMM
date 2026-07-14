"""Accuracy comparison: LiquidQuant W4 vs standard group-wise INT4, and full W4A8.

WikiText-2 perplexity on a real model, replacing every Linear weight with the
quantized-then-dequantized version (fake-quant). Optionally also fake-quants activations
to per-token INT8 (the "A8" of W4A8). This isolates the *quantization scheme* quality —
the scientific question behind "accuracy vs available W4A8".

Llama-3.1-8B is gated (no HF token on this box), so we use an ungated model (default
Qwen2.5-3B). The relative comparison between weight schemes transfers across models.

    CUDA_VISIBLE_DEVICES=6 python bench/accuracy.py --model Qwen/Qwen2.5-3B
"""

import argparse
import copy
import importlib.machinery
import sys
import types

# The NGC container's torchaudio is ABI-mismatched against its torch 2.9 nightly
# (undefined symbol: torch_library_impl) and transformers imports it eagerly. Replace it
# with an empty, properly-spec'd stand-in before importing transformers so availability
# checks (find_spec) succeed and nothing loads the broken .so.
_fake_ta = types.ModuleType("torchaudio")
_fake_ta.__version__ = "0.0.0"
_fake_ta.__spec__ = importlib.machinery.ModuleSpec("torchaudio", loader=None)
_fake_ta.__getattr__ = lambda _name: None
sys.modules.setdefault("torchaudio", _fake_ta)

import torch
import transformers
from transformers import AutoConfig, AutoTokenizer
from datasets import load_dataset

from liquidgemm import quant


def rtn_int4_group(W, g, sym=True):
    """Standard round-to-nearest group-wise INT4 (the common W4 weight scheme)."""
    O, I = W.shape
    if I % g:
        g = I
    Wg = W.float().view(O, I // g, g)
    if sym:
        s = (Wg.abs().amax(-1, keepdim=True) / 7.0).clamp_min(1e-8)
        deq = torch.round(Wg / s).clamp(-8, 7) * s
    else:
        mn = Wg.amin(-1, keepdim=True)
        mx = Wg.amax(-1, keepdim=True)
        s = ((mx - mn) / 15.0).clamp_min(1e-8)
        z = torch.round(-mn / s)
        deq = (torch.round(Wg / s + z).clamp(0, 15) - z) * s
    return deq.view(O, I).to(W.dtype)


def liquidquant_dequant(W, g=64):
    qw = quant.quantize_weight(W.float(), group_size=g)
    return quant.dequantize(qw).to(W.dtype)


def _act_quant_pre_hook(mod, inp):
    """Fake per-token INT8 activation quant (the A8 in W4A8)."""
    x = inp[0]
    s = (x.abs().amax(-1, keepdim=True) / 127.0).clamp_min(1e-8)
    xq = torch.round(x / s).clamp(-127, 127) * s
    return (xq.to(x.dtype),) + tuple(inp[1:])


def target_linears(model):
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and "lm_head" not in name and mod.weight.shape[1] % 64 == 0:
            yield name, mod


@torch.no_grad()
def apply_scheme(model, orig_state, scheme, act8):
    model.load_state_dict(orig_state)             # restore bf16 weights
    for h in getattr(model, "_lq_hooks", []):
        h.remove()
    model._lq_hooks = []
    if scheme == "bf16":
        return
    for name, mod in target_linears(model):
        W = mod.weight.data
        if scheme == "liquidquant":
            mod.weight.data = liquidquant_dequant(W, 64)
        elif scheme == "rtn_int4_g64":
            mod.weight.data = rtn_int4_group(W, 64, sym=True)
        elif scheme == "rtn_int4_g128":
            mod.weight.data = rtn_int4_group(W, 128, sym=True)
        else:
            raise ValueError(scheme)
        if act8:
            model._lq_hooks.append(mod.register_forward_pre_hook(_act_quant_pre_hook))


@torch.no_grad()
def eval_ppl(model, enc, seqlen=2048, max_samples=80):
    n = min(max_samples, enc.shape[1] // seqlen)
    nll = 0.0
    tok = 0
    for i in range(n):
        batch = enc[:, i * seqlen:(i + 1) * seqlen].to(model.device)
        out = model(batch, labels=batch)
        nll += out.loss.float().item() * (seqlen - 1)
        tok += seqlen - 1
    return torch.exp(torch.tensor(nll / tok)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--max-samples", type=int, default=80)
    args = ap.parse_args()

    print(f"model={args.model} seqlen={args.seqlen} samples={args.max_samples}")
    tok = AutoTokenizer.from_pretrained(args.model)
    # Load the concrete architecture class directly (avoids AutoModel's mapping probe,
    # which eagerly imports audio model classes -> the broken torchaudio).
    cfg = AutoConfig.from_pretrained(args.model)
    model_cls = getattr(transformers, cfg.architectures[0])
    model = model_cls.from_pretrained(args.model, dtype=torch.bfloat16).cuda()
    model.eval()
    orig_state = copy.deepcopy(model.state_dict())

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    enc = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids

    runs = [
        ("bf16", "bf16", False),
        ("liquidquant W4 (weight-only)", "liquidquant", False),
        ("RTN int4 g64 W4 (weight-only)", "rtn_int4_g64", False),
        ("RTN int4 g128 W4 (weight-only)", "rtn_int4_g128", False),
        ("liquidquant W4A8 (+int8 act)", "liquidquant", True),
        ("RTN int4 g64 W4A8 (+int8 act)", "rtn_int4_g64", True),
    ]
    print(f"\n{'scheme':<34} {'wikitext2 ppl':>14}")
    print("-" * 50)
    for label, scheme, act8 in runs:
        apply_scheme(model, orig_state, scheme, act8)
        ppl = eval_ppl(model, enc, args.seqlen, args.max_samples)
        print(f"{label:<34} {ppl:>14.4f}", flush=True)


if __name__ == "__main__":
    main()
