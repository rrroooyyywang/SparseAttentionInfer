"""Base classes for sparse patterns."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch


@dataclass
class PatternMetadata:
    """
    Output of SparsePattern.build().

    - kind="dense"      → no masking, full causal attention
    - kind="mask"       → pre-built boolean mask in `mask` [H, T, T], True = KEEP
    - kind="topk"       → data-dependent; backend applies top-k selection using `topk`
    - kind="local"      → pre-built boolean mask; semantically a sliding window
    """
    kind: str                              # "dense" | "mask" | "topk" | "local"
    mask: torch.Tensor | None = None       # [H, T, T] bool, True=keep
    topk: int | None = None               # for kind="topk"
    keep_ratio: float = 1.0              # estimated fraction of attention kept
    build_time_ms: float = 0.0           # time to build this pattern (filled by runner)


class SparsePattern(ABC):
    """Interface for sparse attention patterns.

    Separates *which positions can attend* from *how to compute* attention.
    The build() method must be pure (no side effects on q/k/v) and fast
    enough to include in timing measurements.
    """

    @abstractmethod
    def build(self, q: torch.Tensor, k: torch.Tensor, causal: bool = True) -> PatternMetadata:
        """
        Args:
            q: [B, H, T_q, D]
            k: [B, H, T_k, D]
            causal: whether to enforce causal masking on top of the pattern
        Returns:
            PatternMetadata describing the allowed attention connections.
        """
        ...
