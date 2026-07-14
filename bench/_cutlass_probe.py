import torch, traceback
from vllm import _custom_ops as ops
from liquidgemm import quant
torch.manual_seed(0)
M, N, K = 30, 512, 4096
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.5
W = torch.randn(N, K) * 0.02
qw = quant.quantize_weight(W, 64)
w_i8 = quant.dequantize_i8(qw).cuda()          # [N,K] int8
s1 = qw.s1.cuda()
xq, xs, _ = ops.scaled_int8_quant(x.contiguous())
print("xq", tuple(xq.shape), xq.dtype, "xs", tuple(xs.shape), xs.dtype)
ref = torch._int_mm(xq, w_i8.t().contiguous()).float() * xs.float() * s1[None, :]
for name, b in [("wt_view", w_i8.t()), ("wt_contig", w_i8.t().contiguous())]:
    for sbn, sb in [("[N,1]", s1.view(N, 1)), ("[1,N]", s1.view(1, N))]:
        try:
            y = ops.cutlass_scaled_mm(xq, b, scale_a=xs, scale_b=sb, out_dtype=torch.bfloat16)
            rel = ((y.float() - ref).norm() / ref.norm()).item()
            print(f"b={name} scale_b={sbn}: rel={rel:.5f} shape={tuple(y.shape)}")
        except Exception as e:
            print(f"b={name} scale_b={sbn}: ERR {repr(e)[:100]}")
