"""
CUDA sparse attention kernels.

Recommended approach: use torch.utils.cpp_extension.load_inline() or
build a proper C++ extension with setup.py / CMakeLists.txt.

Each binding file should expose one callable with the signature:

    def cuda_<name>_attn(
        q: torch.Tensor,   # [B, H, T_q, D]  contiguous, fp16 / bf16
        k: torch.Tensor,   # [B, H, T_k, D]
        v: torch.Tensor,   # [B, H, T_k, D]
        **kernel_args,
    ) -> torch.Tensor:     # [B, H, T_q, D]

Then wrap it as an AttentionBackend in sparse_attention_bench/attention/
so it integrates with the benchmark runner and YAML sweep configs.

File naming convention
----------------------
  block_sparse_attn.cu    ← CUDA kernel
  block_sparse_attn.cpp   ← pybind11 / torch extension binding
  block_sparse_attn.py    ← Python wrapper that calls load() and exposes the callable
"""
