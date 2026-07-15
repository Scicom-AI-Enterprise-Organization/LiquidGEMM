"""Out-of-tree vLLM quantization plugin for LiquidGEMM W4A8 (no vLLM fork required).

Importing this module registers a `"liquidgemm"` quantization method via vLLM's
`register_quantization_config` hook. Use it with an ordinary bf16 checkpoint:

    from vllm import LLM
    import liquidgemm.vllm_plugin            # registers "liquidgemm"
    llm = LLM("Qwen/Qwen2.5-3B", quantization="liquidgemm", enforce_eager=True)

On load, every Linear's bf16 weight is quantized on-device with LiquidQuant (two-level
W4, group 64) and stored 4-bit; `apply()` per-token INT8-quantizes activations and calls
`torch.ops.liquidgemm.w4a8_gemm`. `enforce_eager=True` is recommended (the custom op has
no torch.compile/CUDA-graph meta registration yet).
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import set_weight_attrs

from vllm import _custom_ops as vllm_ops

from .quant import quantize_weight
from .pack import pack_nibbles
from . import ops  # noqa: F401  (registers torch.ops.liquidgemm.*)


def register():
    """vLLM general-plugin entry point. Importing this module already runs the
    @register_quantization_config decorator below, so this is just the load hook."""
    return None


@register_quantization_config("liquidgemm")
class LiquidGemmConfig(QuantizationConfig):
    def __init__(self, group_size: int = 64, w4: bool = False):
        super().__init__()
        import os
        self.group_size = group_size
        # Default (w4=False): store LiquidQuant weights as INT8 and run them through vLLM's
        #   production CUTLASS INT8 GEMM (cutlass_scaled_mm) + fused per-token int8 quant.
        #   Fast at all M (matches vLLM W8A8), CUDA-graph-safe, ~2x weight-memory, and keeps
        #   LiquidQuant's accuracy. This is the throughput/production path.
        # w4=True: store 4-bit and run the custom fused w4a8_gemm op (~4x memory, best for
        #   pure decode / memory-constrained; dp4a is slow for M>16). Opt-in.
        env = os.environ.get("LIQUIDGEMM_W4")
        self.w4 = (env == "1") if env is not None else w4

    @classmethod
    def get_name(cls) -> str:
        return "liquidgemm"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "LiquidGemmConfig":
        return cls(group_size=int(config.get("group_size", 64)),
                   w4=bool(config.get("w4", True)))

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        if isinstance(layer, LinearBase):
            # Only quantize Linear layers whose in-features are group-aligned; otherwise
            # leave them in bf16 (kernel requires K % group_size == 0).
            return LiquidGemmLinearMethod(self)
        return None


class LiquidGemmLinearMethod(LinearMethodBase):
    def __init__(self, cfg: LiquidGemmConfig):
        self.cfg = cfg

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # Create a plain bf16 weight so vLLM's loader fills it; quantize post-load.
        weight_loader = extra_weight_attrs.pop("weight_loader")
        weight = ModelWeightParameter(
            data=torch.empty(sum(output_partition_sizes), input_size_per_partition,
                             dtype=params_dtype),
            input_dim=1, output_dim=0, weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)
        layer.input_size_per_partition = input_size_per_partition

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        W = layer.weight.data  # [out, in], bf16, on device
        g = self.cfg.group_size
        # RS-WGMMA needs K%128 and N%64; int8 path needs K%group. Fall back to bf16 else.
        aligned = (W.shape[1] % 128 == 0 and W.shape[0] % 64 == 0) if self.cfg.w4 \
            else (W.shape[1] % g == 0)
        if not aligned:
            layer.lq_enabled = False
            layer.weight = torch.nn.Parameter(W, requires_grad=False)
            return
        qw = quantize_weight(W.float(), group_size=g)
        dev = W.device
        layer.lq_enabled = True
        layer.lq_N, layer.lq_K = qw.N, qw.K
        if self.cfg.w4:
            # True 4-bit storage (~4x memory win): RS-WGMMA fragment-order pack.
            # In-register dequant feeds INT8 WGMMA directly (the LiquidGEMM datapath).
            layer.register_buffer("lq_s1", qw.s1.to(dev))              # [N]
            layer.register_buffer("lq_rs", ops.repack_rs_weight(qw).to(dev))
            layer.register_buffer("lq_s_u8", qw.s_u8.to(dev))
            layer.register_buffer("lq_offset_a", qw.offset_a.to(dev))
        else:
            # INT8 storage (== W4 values, ~2x memory) for vLLM's CUTLASS INT8 GEMM. cutlass
            # wants the B operand column-major [K,N] == a contiguous [N,K] viewed with .t().
            w_i8 = torch.ops.liquidgemm.dequant_weight(
                qw.qweight_u4.to(dev), qw.s_u8.to(dev), qw.offset_a.to(dev), qw.N, qw.K, g)
            layer.register_buffer("lq_w_i8", w_i8.contiguous())        # [N, K]
            layer.register_buffer("lq_s1", qw.s1.to(dev).view(qw.N, 1).contiguous())  # scale_b [N,1]
        del layer.weight  # free the bf16 copy

    def apply(self, layer: torch.nn.Module, x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not getattr(layer, "lq_enabled", False):
            y = torch.nn.functional.linear(x, layer.weight)
            return y if bias is None else y + bias
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).contiguous()
        N = layer.lq_N

        if self.cfg.w4:
            # True 4-bit weights via RS-WGMMA (in-register dequant -> INT8 tensor cores),
            # fully fused: quant -> RS grid -> one reduce+scale+cast kernel. The op pads
            # tokens to its 64-token CTA tile internally (compile/graph-safe).
            x_i8, ascale = torch.ops.liquidgemm.quant_per_token(x2)
            odt = {torch.bfloat16: 0, torch.float16: 1, torch.float32: 2}.get(x.dtype, 0)
            y = torch.ops.liquidgemm.w4a8_wgmma_rs_fused(
                x_i8, layer.lq_rs, layer.lq_s_u8, layer.lq_offset_a, ascale, layer.lq_s1,
                N, layer.lq_K, self.cfg.group_size, odt)
        else:
            # Production path: vLLM's fused per-token INT8 quant + CUTLASS INT8 GEMM
            # (cutlass_scaled_mm). Fast at all M, CUDA-graph-safe, fused scale+bias epilogue.
            x_q, x_s, _ = vllm_ops.scaled_int8_quant(x2)          # [M,K] int8, [M,1] fp32
            y = vllm_ops.cutlass_scaled_mm(
                x_q, layer.lq_w_i8.t(), scale_a=x_s, scale_b=layer.lq_s1,
                out_dtype=x.dtype, bias=bias)                     # bias fused
            return y.reshape(*shape[:-1], N)
        y = y.reshape(*shape[:-1], N)
        return y if bias is None else y + bias
