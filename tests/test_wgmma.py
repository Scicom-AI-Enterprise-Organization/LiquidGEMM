"""Stage 1: real Hopper WGMMA int8 tensor-core GEMM must match torch._int_mm bit-exactly."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

try:
    import liquidgemm.ops  # noqa: F401  registers torch.ops.liquidgemm.*
    _HAVE = True
except Exception:
    _HAVE = False

SHAPES = [(128, 512, 4096), (256, 256, 512), (128, 6144, 4096), (512, 4096, 14336)]


@pytest.mark.skipif(not _HAVE, reason="extension not built")
@pytest.mark.parametrize("M,N,K", SHAPES)
def test_wgmma_i8_matches_int_mm(M, N, K):
    torch.manual_seed(0)
    a = torch.randint(-127, 127, (M, K), device="cuda", dtype=torch.int8)
    b = torch.randint(-127, 127, (N, K), device="cuda", dtype=torch.int8)
    got = torch.ops.liquidgemm.wgmma_i8_gemm(a, b)
    ref = torch._int_mm(a, b.t().contiguous())
    assert torch.equal(got, ref), f"WGMMA int8 mismatch at {M}x{N}x{K}"
