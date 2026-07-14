"""Build the LiquidGEMM CUDA extension (torch.ops.liquidgemm.*).

Build in place on the remote H20 box:
    export PATH=/usr/local/cuda/bin:$PATH
    export TORCH_CUDA_ARCH_LIST="9.0a"          # Hopper: WGMMA/TMA need the 'a' variant
    source /share/venvs/liquidgemm/bin/activate
    python setup.py build_ext --inplace
"""

import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CUTLASS_DIR = os.environ.get("CUTLASS_DIR", os.path.abspath("third_party/cutlass"))

setup(
    name="liquidgemm-ext",
    ext_modules=[
        CUDAExtension(
            name="liquidgemm._C",
            sources=[
                "csrc/liquid_gemm/torch_bindings.cpp",
                "csrc/liquid_gemm/dequant.cu",
                "csrc/liquid_gemm/w4a8_gemm.cu",
                "csrc/liquid_gemm/epilogue.cu",
                "csrc/liquid_gemm/w4a8_wgmma.cu",
            ],
            include_dirs=[os.path.join(CUTLASS_DIR, "include")],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
