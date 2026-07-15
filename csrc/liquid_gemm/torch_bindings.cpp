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

torch::Tensor wgmma_i8_gemm(torch::Tensor a, torch::Tensor b);

torch::Tensor w4a8_wgmma(torch::Tensor x_i8, torch::Tensor packed, torch::Tensor s_u8,
                         torch::Tensor off_a, int64_t N, int64_t K, int64_t group_size);

torch::Tensor wgmma_rs_a_coords();

torch::Tensor w4a8_wgmma_rs(torch::Tensor x_i8, torch::Tensor w, torch::Tensor spack,
                            torch::Tensor s_u8, torch::Tensor off_a, int64_t N, int64_t K,
                            int64_t group_size, bool packed);

torch::Tensor w4a8_wgmma_rs_fused(torch::Tensor x_i8, torch::Tensor w, torch::Tensor s_u8,
                                  torch::Tensor off_a, torch::Tensor ascale,
                                  torch::Tensor s1, int64_t N, int64_t K,
                                  int64_t group_size, int64_t out_dtype);

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
  m.def("wgmma_i8_gemm(Tensor a, Tensor b) -> Tensor");
  m.def("w4a8_wgmma(Tensor x_i8, Tensor packed, Tensor s_u8, Tensor off_a, int N, int K, int group_size) -> Tensor");
  // no Tensor args -> register with an inline catch-all kernel (cannot dispatch by device)
  m.def("wgmma_rs_a_coords", &liquidgemm::wgmma_rs_a_coords);
  m.def("w4a8_wgmma_rs(Tensor x_i8, Tensor w, Tensor spack, Tensor s_u8, Tensor off_a, int N, int K, int group_size, bool packed) -> Tensor");
  m.def("w4a8_wgmma_rs_fused(Tensor x_i8, Tensor w, Tensor s_u8, Tensor off_a, Tensor ascale, Tensor s1, int N, int K, int group_size, int out_dtype) -> Tensor");
}

TORCH_LIBRARY_IMPL(liquidgemm, CUDA, m) {
  m.impl("dequant_weight", &liquidgemm::dequant_weight);
  m.impl("w4a8_gemm", &liquidgemm::w4a8_gemm);
  m.impl("quant_per_token", &liquidgemm::quant_per_token);
  m.impl("scale_epilogue", &liquidgemm::scale_epilogue);
  m.impl("wgmma_i8_gemm", &liquidgemm::wgmma_i8_gemm);
  m.impl("w4a8_wgmma", &liquidgemm::w4a8_wgmma);
  m.impl("w4a8_wgmma_rs", &liquidgemm::w4a8_wgmma_rs);
  m.impl("w4a8_wgmma_rs_fused", &liquidgemm::w4a8_wgmma_rs_fused);
}

// Empty module so `import liquidgemm._C` succeeds; the TORCH_LIBRARY static initializers
// above register the ops when this shared object is dlopen'd during import.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
