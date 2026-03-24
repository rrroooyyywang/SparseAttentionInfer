"""Sliding local window attention pattern."""
import torch

from sparse_attention_bench.patterns.base import (
    PatternMetadata,
    SparsePattern,
    build_block_pairs_from_mask,
)


class LocalWindowPattern(SparsePattern):
    """
    Each query attends to the W most recent positions (including itself).
    The mask is pre-built and cached, making it compatible with both
    MaskedSdpaBackend and GatherSparseBackend.

    Pass block_size (power-of-2, >= 16) to also build the CSR block-pair
    schedule required by TritonUniversalBackend.
    """

    def __init__(self, window_size: int, block_size: int | None = None):
        self.window_size = window_size
        self.block_size = block_size   # None → no block-sparse schedule built
        self._mask_cache: dict = {}
        self._csr_cache: dict = {}

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

        block_pairs, block_pair_offsets = None, None
        if self.block_size is not None:
            csr_key = (T_q, T_k, H, device_key, self.window_size, causal, self.block_size)
            if csr_key not in self._csr_cache:
                bp, bpo = build_block_pairs_from_mask(mask, self.block_size)
                self._csr_cache[csr_key] = (bp, bpo)
            block_pairs, block_pair_offsets = self._csr_cache[csr_key]

        return PatternMetadata(
            kind="local",
            mask=mask,
            keep_ratio=keep_ratio,
            block_size=self.block_size,
            block_pairs=block_pairs,
            block_pair_offsets=block_pair_offsets,
        )

    def _build_mask(
        self, T_q: int, T_k: int, H: int, device: torch.device, causal: bool
    ) -> torch.Tensor:
        """Build [H, T_q, T_k] boolean mask where True = attend."""
        W = self.window_size
        query_offset = max(0, T_k - T_q)
        rows = (query_offset + torch.arange(T_q, device=device)).unsqueeze(1)   # [T_q, 1]
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
        query_offset = max(0, T_k - T_q)
        kept = sum(min(W, query_offset + i + 1) for i in range(T_q))
        return min(1.0, kept / total)
