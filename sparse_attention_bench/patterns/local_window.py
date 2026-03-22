"""Sliding local window attention pattern."""
import torch

from sparse_attention_bench.metrics.accuracy import causal_mask
from sparse_attention_bench.patterns.base import PatternMetadata, SparsePattern


class LocalWindowPattern(SparsePattern):
    """
    Each query attends to the W most recent positions (including itself).
    The mask is pre-built and cached, making it compatible with both
    MaskedSdpaBackend and GatherSparseBackend.
    """

    def __init__(self, window_size: int):
        self.window_size = window_size
        self._mask_cache: dict = {}

    def build(self, q: torch.Tensor, k: torch.Tensor, causal: bool = True) -> PatternMetadata:
        T_q = q.size(2)
        T_k = k.size(2)
        H = q.size(1)
        device = q.device
        device_key = f"{device.type}:{device.index}"
        cache_key = (T_q, T_k, H, device_key, self.window_size, causal)

        if cache_key not in self._mask_cache:
            self._mask_cache[cache_key] = self._build_mask(T_q, T_k, H, device, causal)

        mask = self._mask_cache[cache_key]
        keep_ratio = self._estimate_keep_ratio(T_q, T_k)
        return PatternMetadata(kind="local", mask=mask, keep_ratio=keep_ratio)

    def _build_mask(
        self, T_q: int, T_k: int, H: int, device: torch.device, causal: bool
    ) -> torch.Tensor:
        """Build [H, T_q, T_k] boolean mask where True = attend."""
        W = self.window_size
        rows = torch.arange(T_q, device=device).unsqueeze(1)   # [T_q, 1]
        cols = torch.arange(T_k, device=device).unsqueeze(0)   # [1, T_k]
        # Sliding window: key must be within [row - W + 1, row]
        in_window = (cols >= rows - W + 1) & (cols <= rows)
        if causal:
            in_window = in_window & (cols <= rows)
        # Same mask for all heads
        return in_window.unsqueeze(0).expand(H, -1, -1)  # [H, T_q, T_k]

    def _estimate_keep_ratio(self, T_q: int, T_k: int) -> float:
        W = self.window_size
        if T_q == 0 or T_k == 0:
            return 0.0
        total = T_q * T_k
        kept = sum(min(W, i + 1) for i in range(T_q))
        return min(1.0, kept / total)
