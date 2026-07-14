"""LiquidGEMM: hardware-efficient W4A8-INT8 GEMM for Hopper (arXiv:2509.01229).

Phase 1 exposes the LiquidQuant reference quantizer and weight packing (pure PyTorch).
Later phases add the CUDA kernel behind ``torch.ops.liquidgemm.w4a8_gemm``.
"""

__version__ = "0.0.1"
