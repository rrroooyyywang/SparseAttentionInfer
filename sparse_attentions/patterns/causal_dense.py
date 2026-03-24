"""Dense causal attention pattern (baseline)."""
import torch

from sparse_attentions.patterns.base import PatternMetadata, SparsePattern


class DenseCausalPattern(SparsePattern):
    """Full causal attention — attends to all previous positions."""

    def build(self, q: torch.Tensor, k: torch.Tensor, causal: bool = True) -> PatternMetadata:
        return PatternMetadata(kind="dense", keep_ratio=1.0)
