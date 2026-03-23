"""
Test: cuda_helloworld kernel bound to PyTorch.

Build first (from kernels/cuda/):
    CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH python setup.py build_ext --inplace

Run:
    python test/test_helloworld.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # SparseAttentionInfer/

import torch
import kernels.cuda  # noqa: F401 — loads .so and registers TORCH_LIBRARY ops


def test_helloworld_returns_tensor():
    out = torch.ops.cuda_sparse_attention.cuda_helloworld()
    assert isinstance(out, torch.Tensor), f"expected Tensor, got {type(out)}"
    assert out.shape == torch.Size([1]), f"expected shape [1], got {out.shape}"
    assert out.dtype == torch.float32, f"expected float32, got {out.dtype}"
    print(f"  output: {out}")


def test_helloworld_via_wrapper():
    from kernels.cuda.src.torch_wrappers.ops import cuda_helloworld
    out = cuda_helloworld()
    assert isinstance(out, torch.Tensor), f"expected Tensor, got {type(out)}"
    print(f"  output: {out}")


def test_helloworld_no_crash():
    for _ in range(2):
        torch.ops.cuda_sparse_attention.cuda_helloworld()
    torch.cuda.synchronize()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device available")
        sys.exit(0)

    tests = [
        test_helloworld_returns_tensor,
        test_helloworld_via_wrapper,
        test_helloworld_no_crash,
    ]

    passed = 0
    for t in tests:
        try:
            print(f"running {t.__name__} ...")
            t()
            print(f"  PASSED")
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")

    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
