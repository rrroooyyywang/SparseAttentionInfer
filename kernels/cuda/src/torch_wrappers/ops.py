import torch
from torch import Tensor

__all__ = ["cuda_helloworld","cuda_bigbird_sparse_attention"]

def cuda_helloworld() -> Tensor:
    """Performs a * b + c in an efficient fused kernel"""
    return torch.ops.cuda_sparse_attention.cuda_helloworld.default()

# Registers a FakeTensor kernel (aka "meta kernel", "abstract impl")
# that describes what the properties of the output Tensor are given
# the properties of the input Tensor. The FakeTensor kernel is necessary
# for the op to work performantly with torch.compile.
@torch.library.register_fake("cuda_sparse_attention::cuda_helloworld")
def _():
    return torch.empty(1)  # shape and dtype don't matter since the kernel doesn't take any inputs


def _backward(ctx, grad):
    ...


def _setup_context(ctx, inputs, output):
    ...

# This adds training support for the operator. You must provide us
# the backward formula for the operator and a `setup_context` function
# to save values to be used in the backward.
torch.library.register_autograd(
    "cuda_sparse_attention::cuda_helloworld", _backward, setup_context=_setup_context)



# def cuda_bigbird_sparse_attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
#     """Performs a * b + c in an efficient fused kernel"""
#     return torch.ops.cuda.cuda_bigbird_sparse_attention.default(q, k, v)

# Registers a FakeTensor kernel (aka "meta kernel", "abstract impl")
# that describes what the properties of the output Tensor are given
# the properties of the input Tensor. The FakeTensor kernel is necessary
# for the op to work performantly with torch.compile.
# @torch.library.register_fake("cuda_sparse_attention::cuda_bigbird_sparse_attention")
# def _(q, k, v):
#     torch._check(q.dim() == 4)
#     torch._check(k.dim() == 4)
#     torch._check(v.dim() == 4)

#     torch._check(q.device == k.device)
#     torch._check(q.device == v.device)

#     torch._check(q.dtype == k.dtype)
#     torch._check(q.dtype == v.dtype)

#     torch._check(q.shape[0] == k.shape[0] == v.shape[0])  # batch
#     torch._check(q.shape[1] == k.shape[1] == v.shape[1])  # heads

#     torch._check(q.shape[3] == k.shape[3])                # q/k head_dim
#     torch._check(k.shape[2] == v.shape[2])                # kv seq len

#     torch._check(v.shape[3] == q.shape[3])

#     return torch.empty_like(q)


# def _backward(ctx, grad):
#     ...


# def _setup_context(ctx, inputs, output):
#     ...

# # This adds training support for the operator. You must provide us
# # the backward formula for the operator and a `setup_context` function
# # to save values to be used in the backward.
# torch.library.register_autograd(
#     "cuda_sparse_attention::cuda_bigbird_sparse_attention", _backward, setup_context=_setup_context)