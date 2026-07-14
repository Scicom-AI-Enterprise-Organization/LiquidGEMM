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

from .quant import quantize_weight
from .pack import pack_nibbles
from . import ops  # noqa: F401  (registers torch.ops.liquidgemm.*)


def register():
    """vLLM general-plugin entry point. Importing this module already runs the
    @register_quantization_config decorator below, so this is just the load hook."""
    return None


@register_quantization_config("liquidgemm")
class LiquidGemmConfig(QuantizationConfig):
    def __init__(self, group_size: int = 64, keep_w4: bool = False):
        super().__init__()
        self.group_size = group_size
        # keep_w4=True also stores 4-bit weights and uses the fused GEMV for M<=16 decode
        # (4x weight-memory, best decode bandwidth). Default False: store INT8 only and use
        # cuBLASLt INT8 tensor cores at all M (2x memory, best throughput at concurrency).
        self.keep_w4 = keep_w4

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
                   keep_w4=bool(config.get("keep_w4", False)))

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
        if W.shape[1] % g != 0:
            # non-group-aligned (e.g. some vision/proj layers) -> keep bf16.
            layer.lq_enabled = False
            layer.weight = torch.nn.Parameter(W, requires_grad=False)
            return
        qw = quantize_weight(W.float(), group_size=g)
        dev = W.device
        layer.lq_enabled = True
        layer.lq_N, layer.lq_K = qw.N, qw.K
        # LiquidQuant reconstructed INT8 weights (== W4 values), stored [K, N] for cuBLASLt.
        w_i8 = torch.ops.liquidgemm.dequant_weight(
            qw.qweight_u4.to(dev), qw.s_u8.to(dev), qw.offset_a.to(dev),
            qw.N, qw.K, g)                        # [N, K] int8
        layer.register_buffer("lq_w_i8_t", w_i8.t().contiguous())  # [K, N]
        layer.register_buffer("lq_s1", qw.s1.to(dev))
        if self.cfg.keep_w4:
            # optional: also keep 4-bit packed for the memory-optimal decode GEMV path.
            layer.register_buffer("lq_packed", pack_nibbles(qw.qweight_u4).to(dev))
            layer.register_buffer("lq_s_u8", qw.s_u8.to(dev))
            layer.register_buffer("lq_offset_a", qw.offset_a.to(dev))
        del layer.weight  # free the bf16 copy

    def apply(self, layer: torch.nn.Module, x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not getattr(layer, "lq_enabled", False):
            y = torch.nn.functional.linear(x, layer.weight)
            return y if bias is None else y + bias
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).contiguous()
        M, K, N = x2.shape[0], layer.lq_K, layer.lq_N
        odt = {torch.bfloat16: 0, torch.float16: 1, torch.float32: 2}.get(x.dtype, 0)
        x_i8, ascale = torch.ops.liquidgemm.quant_per_token(x2)   # 1 fused kernel

        if self.cfg.keep_w4 and M <= 16:
            # Memory-bound decode: fused 4-bit GEMV (streams 4-bit weights from GMEM).
            y = torch.ops.liquidgemm.w4a8_gemm(
                x_i8, layer.lq_packed, layer.lq_s_u8, layer.lq_offset_a, layer.lq_s1,
                ascale, N, K, self.cfg.group_size).to(x.dtype)
        else:
            # Concurrency / prefill: INT8 tensor cores (cuBLASLt) + fused scale epilogue.
            # torch._int_mm requires M>16; pad small M (low-concurrency decode).
            if M <= 16:
                x_i8 = torch.cat([x_i8, x_i8.new_zeros(17 - M, K)], 0)
            acc = torch._int_mm(x_i8, layer.lq_w_i8_t)            # [M(+pad), N] int32
            acc = acc[:M]
            y = torch.ops.liquidgemm.scale_epilogue(acc, ascale, layer.lq_s1, odt)  # 1 kernel
        y = y.reshape(*shape[:-1], N)
        return y if bias is None else y + bias
