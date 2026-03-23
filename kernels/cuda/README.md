## Folder structure

```
kernels/cuda/
├── setup.py                        # builds the CUDA extension (.so)
├── __init__.py                     # package entry point; loads the .so
├── build/                          # compiled .so is placed here (--inplace)
│   └── SparseAttentionExtension.cpython-311-*.so
├── src/
│   ├── csrc/
│   │   ├── bind.cu                 # TORCH_LIBRARY registration + PYBIND11_MODULE
│   │   └── <name>Kernel.cu        # one file per CUDA kernel (no main())
│   └── torch_wrappers/
│       └── ops.py                  # Python-facing wrappers for each op
└── test/
    └── test_<name>.py              # plain Python test (run with `python`)
```

---

## How the pieces connect

```
<name>Kernel.cu         raw CUDA __global__ kernel + host wrapper function
      ↓  #include
bind.cu                 TORCH_LIBRARY   → declares op schema & namespace
                        TORCH_LIBRARY_IMPL → binds C++ impl to dispatcher
                        PYBIND11_MODULE → satisfies Python's PyInit_* requirement
      ↓  build_ext --inplace
build/SparseAttentionExtension.so
      ↓  from .build import SparseAttentionExtension   (__init__.py)
torch.ops.cuda_sparse_attention.<op_name>   registered in dispatcher
      ↓  ops.py
Python function         + register_fake (torch.compile support)
                        + register_autograd (training support)
```

---

## Build

> Requires: CUDA 12.8 toolkit, PyTorch built with CUDA 12.8.

```bash
# from kernels/cuda/
CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH \
    python setup.py build_ext --inplace
```

The compiled `.so` is placed in `build/` because `EXT_NAME = "build.SparseAttentionExtension"` in `setup.py` — the dot maps to a subdirectory when `--inplace` is used.

---

## Adding a new kernel — step by step

### 1. Write the kernel — `src/csrc/<name>Kernel.cu`

```cuda
// No main(). Only __global__ kernel + host wrapper.
__global__ void myKernel(/* args */) { ... }

torch::Tensor my_op_impl(/* args */) {
    myKernel<<<blocks, threads>>>(/* args */);
    cudaDeviceSynchronize();
    return ...;
}
```

### 2. Register in `src/csrc/bind.cu`

```cpp
#include "<name>Kernel.cu"

// Declare schema — must match ops.py
TORCH_LIBRARY_FRAGMENT(cuda_sparse_attention, m) {
    m.def("my_op(Tensor q, Tensor k) -> Tensor");
}

// Bind implementation
// Use CUDA when op has tensor inputs (dispatcher can infer device from tensors).
// Use CompositeExplicitAutograd when op has NO tensor inputs.
TORCH_LIBRARY_IMPL(cuda_sparse_attention, CUDA, m) {
    m.impl("my_op", &my_op_impl);
}
```

> **Namespace rule:** `cuda_sparse_attention` is our namespace. Never use `cuda` —
> PyTorch already owns that namespace (`TORCH_LIBRARY(cuda, ...)` in torch internals)
> and a second `TORCH_LIBRARY(cuda, ...)` will crash at load time.
> Use `TORCH_LIBRARY_FRAGMENT` when adding to an existing namespace across multiple files.

### 3. Wrap in `src/torch_wrappers/ops.py`

```python
def my_op(q: Tensor, k: Tensor) -> Tensor:
    return torch.ops.cuda_sparse_attention.my_op.default(q, k)

@torch.library.register_fake("cuda_sparse_attention::my_op")
def _(q, k):
    # Describe output shape/dtype for torch.compile / tracing
    return torch.empty_like(q)
```

### 4. Write a test — `test/test_<name>.py`

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # project root

import torch
import kernels.cuda  # loads .so, registers ops

def test_my_op():
    q = torch.randn(1, 4, 16, 64, device="cuda")
    k = torch.randn(1, 4, 16, 64, device="cuda")
    out = torch.ops.cuda_sparse_attention.my_op(q, k)
    assert out.shape == q.shape

if __name__ == "__main__":
    test_my_op()
    print("PASSED")
```

Run with:
```bash
python test/test_<name>.py
```

---

## Dispatcher key cheat-sheet

| Situation | `TORCH_LIBRARY_IMPL` key |
|---|---|
| Op takes tensor inputs (normal case) | `CUDA` |
| Op takes no tensor inputs | `CompositeExplicitAutograd` |
| Op must work on CPU and CUDA | register both `CPU` and `CUDA` impls |
