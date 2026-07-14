// Torch op registration for LiquidGEMM. Importing liquidgemm._C registers these under
// torch.ops.liquidgemm.*.
#include <torch/extension.h>
#include <torch/library.h>

namespace liquidgemm {

torch::Tensor dequant_weight(torch::Tensor qweight, torch::Tensor s_u8,
                             torch::Tensor offset_a, int64_t N, int64_t K,
                             int64_t group_size);

}  // namespace liquidgemm

TORCH_LIBRARY(liquidgemm, m) {
  m.def(
      "dequant_weight(Tensor qweight, Tensor s_u8, Tensor offset_a, int N, int K, "
      "int group_size) -> Tensor");
}

TORCH_LIBRARY_IMPL(liquidgemm, CUDA, m) {
  m.impl("dequant_weight", &liquidgemm::dequant_weight);
}

// Empty module so `import liquidgemm._C` succeeds; the TORCH_LIBRARY static initializers
// above register the ops when this shared object is dlopen'd during import.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
