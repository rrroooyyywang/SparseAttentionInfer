"""
kernels
==============================
Home for custom Triton and CUDA sparse attention kernels (Phase 5).

Directory layout
----------------
kernels/
├── triton/          ← Triton-JIT kernels (.py)
│   └── *.py
└── cuda/            ← CUDA extension kernels (.cu / .cpp + Python binding)
    ├── *.cu
    ├── *.cpp
    └── *.py         ← Python binding / load_inline wrapper

How to plug a kernel into the benchmark harness
------------------------------------------------
1. Write your kernel in kernels/triton/ or kernels/cuda/.
2. Create an AttentionBackend wrapper in sparse_attention_bench/attention/
   (see attention/triton_backend.py for a template).
3. Register it in sparse_attention_bench/attention/__init__.py:

       from sparse_attention_bench.attention.triton_backend import TritonTopkBackend
       _REGISTRY["triton_topk"] = TritonTopkBackend

4. Use it in a sweep YAML:

       patterns:
         - type: topk
           backend: triton_topk
           topk: [32, 64, 128]

The benchmark runner will measure pattern-build time and kernel time
separately, and compare accuracy against the dense SDPA baseline automatically.
"""
