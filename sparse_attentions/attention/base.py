"""Abstract base class for attention backends."""
from abc import ABC, abstractmethod

import torch

from sparse_attentions.patterns.base import PatternMetadata


class AttentionBackend(ABC):
    """
    Interface for attention computation backends.

    Separates *how to compute* attention from *which positions can attend*
    (the latter is handled by SparsePattern).

    forward() receives pre-projected Q/K/V tensors; it does not own
    QKV projections or output projections — those stay in AttentionLayer.
    """

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,        # [B, H, T_q, D]
        k: torch.Tensor,        # [B, H, T_k, D]
        v: torch.Tensor,        # [B, H, T_k, D]
        pattern: PatternMetadata,
    ) -> torch.Tensor:
        """
        Compute attention output.
        Returns: [B, H, T_q, D]
        """
        ...

    @property
    def actual_backend(self) -> str:
        """
        Name of the kernel/path that was actually executed on the last forward() call.
        Backends that can fall back to a different implementation should override this.
        """
        return type(self).__name__
