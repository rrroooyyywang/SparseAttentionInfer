"""
Template: wrapping a Triton kernel as an AttentionBackend.

Copy this file, rename it, implement _call_kernel(), then register
the class in sparse_attentions/attention/__init__.py.
"""
import torch

from sparse_attentions.attention.base import AttentionBackend
from sparse_attentions.attention.masked_sdpa import MaskedSdpaBackend
from sparse_attentions.patterns.base import PatternMetadata


class TritonTopkBackend(AttentionBackend):
    """
    Example wrapper for a Triton top-k sparse attention kernel.

    Falls back to MaskedSdpaBackend if Triton is not available,
    so the harness can still run for correctness checks on CPU.
    """

    def __init__(self):
        self._triton_available = self._check_triton()
        self._fallback = MaskedSdpaBackend()

    @staticmethod
    def _check_triton() -> bool:
        try:
            import triton  # noqa: F401
            return True
        except ImportError:
            return False

    def forward(
        self,
        q: torch.Tensor,        # [B, H, T_q, D]
        k: torch.Tensor,        # [B, H, T_k, D]
        v: torch.Tensor,        # [B, H, T_k, D]
        pattern: PatternMetadata,
    ) -> torch.Tensor:
        if not self._triton_available:
            # Correctness fallback — lets you run sweeps on CPU before the kernel is ready
            return self._fallback.forward(q, k, v, pattern)

        # ── Replace the lines below with your real Triton kernel call ──────────
        # from sparse_attentions.kernels.triton.topk_sparse_attn import triton_topk_attn
        # return triton_topk_attn(q, k, v, top_k=pattern.topk, causal=True)
        raise NotImplementedError(
            "Triton kernel not implemented yet. "
            "Write your kernel in kernels/triton/ and call it here."
        )
