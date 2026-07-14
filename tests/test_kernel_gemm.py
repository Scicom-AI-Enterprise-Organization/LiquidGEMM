"""W4A8 GEMM correctness vs the Phase-1 int8-domain reference and vs full precision."""

import pytest
import torch

from liquidgemm import quant

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

try:
    from liquidgemm import ops
    _HAVE_EXT = True
except Exception:
    _HAVE_EXT = False

# (name, N, K) — real Llama-3.1-8B out/in features (down uses K=14336).
LLAMA = [("qkv", 6144, 4096), ("o", 4096, 4096), ("gate_up", 28672, 4096), ("down", 4096, 14336)]
M_SWEEP = [1, 4, 16, 64, 256]


@pytest.mark.skipif(not _HAVE_EXT, reason="liquidgemm._C not built")
@pytest.mark.parametrize("name,N,K", LLAMA)
@pytest.mark.parametrize("M", M_SWEEP)
def test_unfused_matches_reference_and_fp(name, N, K, M):
    torch.manual_seed(0)
    W = torch.randn(N, K) * 0.05
    X = torch.randn(M, K)
    qw = quant.quantize_weight(W, group_size=64)
    x_i8, ascale = quant.quantize_activation(X)

    # exact int accumulation must match the reference
    y_ref = quant.w4a8_matmul(x_i8, ascale, qw)            # [M, N] float32 (cpu)
    q_hat = quant.dequantize_i8(qw).to(torch.int32)
    acc_ref = (x_i8.to(torch.int32) @ q_hat.t())            # [M, N] int32 (cpu)

    y_gpu, acc_gpu = ops.w4a8_linear_unfused(x_i8, ascale, qw, out_dtype=torch.float32)
    assert torch.equal(acc_gpu.cpu(), acc_ref), "int32 accumulation diverged from reference"
    torch.testing.assert_close(y_gpu.cpu(), y_ref, rtol=1e-3, atol=1e-3)

    # sanity vs full precision (quantization error only)
    y_fp = X @ W.t()
    cos = torch.nn.functional.cosine_similarity(y_gpu.cpu().flatten(), y_fp.flatten(), dim=0)
    assert cos > 0.99, f"[{name} M={M}] cosine vs fp too low: {cos:.4f}"


@pytest.mark.skipif(not _HAVE_EXT, reason="liquidgemm._C not built")
@pytest.mark.parametrize("name,N,K", LLAMA)
@pytest.mark.parametrize("M", M_SWEEP)
def test_fused_dp4a_matches_reference(name, N, K, M):
    """Fused dp4a kernel must match the int8-domain reference (exact int accumulation)."""
    torch.manual_seed(0)
    W = torch.randn(N, K) * 0.05
    X = torch.randn(M, K)
    qw = quant.quantize_weight(W, group_size=64)
    x_i8, ascale = quant.quantize_activation(X)

    y_ref = quant.w4a8_matmul(x_i8, ascale, qw)                # [M, N] float32 (cpu)
    y_gpu = ops.w4a8_gemm(x_i8, ascale, qw, out_dtype=torch.float32).cpu()
    # fp32 accumulate of exact ints -> should match reference to float round-off.
    torch.testing.assert_close(y_gpu, y_ref, rtol=2e-4, atol=2e-4)

    y_fp = X @ W.t()
    cos = torch.nn.functional.cosine_similarity(y_gpu.flatten(), y_fp.flatten(), dim=0)
    assert cos > 0.99, f"[{name} M={M}] fused cosine vs fp too low: {cos:.4f}"
