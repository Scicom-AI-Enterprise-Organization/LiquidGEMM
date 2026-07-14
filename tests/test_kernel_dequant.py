"""CUDA dequant kernel must reproduce the Phase-1 reference bit-for-bit."""

import pytest
import torch

from liquidgemm import quant

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

try:
    from liquidgemm import ops
    _HAVE_EXT = True
except Exception as e:  # extension not built
    _HAVE_EXT = False
    _IMPORT_ERR = e

SHAPES = [(1536, 4096), (1024, 4096), (2048, 4096), (1024, 14336), (6144, 4096)]


@pytest.mark.skipif(not _HAVE_EXT, reason="liquidgemm._C not built")
@pytest.mark.parametrize("N,K", SHAPES)
@pytest.mark.parametrize("g", [64, 128])
def test_cuda_dequant_matches_reference(N, K, g):
    torch.manual_seed(0)
    W = torch.randn(N, K)
    qw = quant.quantize_weight(W, group_size=g)
    ref = quant.dequantize_i8(qw).cuda()      # [N, K] int8 (XOR path)
    got = ops.dequant_weight_i8(qw)           # CUDA kernel
    assert got.dtype == torch.int8 and got.shape == (N, K)
    assert torch.equal(got, ref), f"CUDA dequant mismatch at N={N},K={K},g={g}"
