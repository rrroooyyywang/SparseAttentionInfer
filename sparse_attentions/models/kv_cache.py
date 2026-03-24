"""Simple KV cache for decoder autoregressive inference."""
from __future__ import annotations

import torch


class KVCache:
    """
    Per-layer KV cache for autoregressive decoding.

    Usage:
        cache = KVCache(num_layers=2)
        # prefill: initialize from a full sequence
        cache.init(k_list, v_list)
        # decode: append new K/V and return the full cache
        k_full, v_full = cache.append(layer_idx, k_new, v_new)
    """

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self._k: list[torch.Tensor | None] = [None] * num_layers
        self._v: list[torch.Tensor | None] = [None] * num_layers

    def init(self, k_list: list[torch.Tensor], v_list: list[torch.Tensor]) -> None:
        assert len(k_list) == self.num_layers
        assert len(v_list) == self.num_layers
        self._k = list(k_list)
        self._v = list(v_list)

    def clone(self) -> "KVCache":
        """Deep-copy cached tensors so decode benchmarks can reuse a fixed prefix."""
        cloned = KVCache(num_layers=self.num_layers)
        cloned._k = [None if k is None else k.clone() for k in self._k]
        cloned._v = [None if v is None else v.clone() for v in self._v]
        return cloned

    def append(
        self,
        layer_idx: int,
        k_new: torch.Tensor,   # [B, H, 1, D]
        v_new: torch.Tensor,   # [B, H, 1, D]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new K/V step and return full [B, H, T+1, D] tensors."""
        if self._k[layer_idx] is None:
            self._k[layer_idx] = k_new
            self._v[layer_idx] = v_new
        else:
            self._k[layer_idx] = torch.cat([self._k[layer_idx], k_new], dim=2)
            self._v[layer_idx] = torch.cat([self._v[layer_idx], v_new], dim=2)
        return self._k[layer_idx], self._v[layer_idx]

    def seq_len(self, layer_idx: int = 0) -> int:
        if self._k[layer_idx] is None:
            return 0
        return self._k[layer_idx].size(2)

    def clear(self) -> None:
        self._k = [None] * self.num_layers
        self._v = [None] * self.num_layers
