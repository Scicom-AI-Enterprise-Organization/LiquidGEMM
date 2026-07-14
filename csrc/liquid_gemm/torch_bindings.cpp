// Torch op registration for LiquidGEMM. Importing liquidgemm._C registers these under
// torch.ops.liquidgemm.*.
#include <torch/extension.h>
#include <torch/library.h>

namespace liquidgemm {

torch::Tensor dequant_weight(torch::Tensor qweight, torch::Tensor s_u8,
                             torch::Tensor offset_a, int64_t N, int64_t K,
                             int64_t group_size);

torch::Tensor w4a8_gemm(torch::Tensor X, torch::Tensor qweight, torch::Tensor s_u8,
                        torch::Tensor offset_a, torch::Tensor s1, torch::Tensor ascale,
                        int64_t N, int64_t K, int64_t group_size);

std::vector<torch::Tensor> quant_per_token(torch::Tensor x);

torch::Tensor scale_epilogue(torch::Tensor acc, torch::Tensor ascale, torch::Tensor s1,
                             int64_t out_dtype);

}  // namespace liquidgemm

TORCH_LIBRARY(liquidgemm, m) {
  m.def(
      "dequant_weight(Tensor qweight, Tensor s_u8, Tensor offset_a, int N, int K, "
      "int group_size) -> Tensor");
  m.def(
      "w4a8_gemm(Tensor X, Tensor qweight, Tensor s_u8, Tensor offset_a, Tensor s1, "
      "Tensor ascale, int N, int K, int group_size) -> Tensor");
  m.def("quant_per_token(Tensor x) -> Tensor[]");
  m.def("scale_epilogue(Tensor acc, Tensor ascale, Tensor s1, int out_dtype) -> Tensor");
}

TORCH_LIBRARY_IMPL(liquidgemm, CUDA, m) {
  m.impl("dequant_weight", &liquidgemm::dequant_weight);
  m.impl("w4a8_gemm", &liquidgemm::w4a8_gemm);
  m.impl("quant_per_token", &liquidgemm::quant_per_token);
  m.impl("scale_epilogue", &liquidgemm::scale_epilogue);
}

// Empty module so `import liquidgemm._C` succeeds; the TORCH_LIBRARY static initializers
// above register the ops when this shared object is dlopen'd during import.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
