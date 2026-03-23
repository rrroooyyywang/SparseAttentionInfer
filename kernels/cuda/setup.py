import os
from pathlib import Path
from setuptools import setup, Extension
from torch.utils import cpp_extension

# PROJECT_NAME = "SparseAttentionInfer"
EXT_NAME = "build.SparseAttentionExtension"
PROJECT_DIR = Path(__file__).parent.resolve()


# nvcc flags
nvcc_flgas = []
ld_flags = []

nvcc_flgas += [
    "-O3",
    "-DNDEBUG",
    "-std=c++17",
    # "--generate-code=arch=compute_86,code=sm_86",
]
ld_flags += ["cuda"]

# sources
C_SOURCES = [
    str(PROJECT_DIR / "src/csrc/bind.cu"),
]

# setup
include_dirs = []


setup(
    ext_modules=[
        cpp_extension.CUDAExtension(
            name=EXT_NAME,
            sources=C_SOURCES,
            include_dirs=include_dirs,
            extra_compile_args={"nvcc": nvcc_flgas},
            libraries=ld_flags,
        )
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)