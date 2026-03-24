"""Dense causal attention using torch.nn.functional.scaled_dot_product_attention."""
import torch
import torch.nn.functional as F

from sparse_attentions.attention.base import AttentionBackend
from sparse_attentions.patterns.base import PatternMetadata


class DenseSdpaBackend(AttentionBackend):
    """
    Dense scaled dot-product attention.
    Uses PyTorch's fused SDPA (FlashAttention when available).
    Ignores pattern metadata — always runs full causal attention.
    """

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pattern: PatternMetadata,
    ) -> torch.Tensor:
        # is_causal=True uses the fused causal kernel path
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
