#include <torch/extension.h>
#include <stdio.h>

// Includes for any additional CUDA kernels or utilities can go here
#include "helloWorldKernel.cu"


// ── op registration ──────────────────────────────────────────────────────────
// Declare the op schema (must match ops.py)
TORCH_LIBRARY(cuda_sparse_attention, m) {
    m.def("cuda_helloworld() -> Tensor");
}

// No tensor inputs → dispatcher can't infer device.
// CompositeExplicitAutograd means "works for all backends";
// the kernel itself always runs on CUDA regardless.
TORCH_LIBRARY_IMPL(cuda_sparse_attention, CompositeExplicitAutograd, m) {
    m.impl("cuda_helloworld", &cuda_helloworld_impl);
}

// Required by Python's import machinery (PyInit_<name>).
// Empty because all ops are registered via TORCH_LIBRARY above.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
