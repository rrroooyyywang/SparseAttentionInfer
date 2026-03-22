"""
AttentionLayer: QKV projection + sparse pattern + attention backend + output projection.

This is the building block for DecoderBlock and bench_decoder.py.
It is parameterized by a SparsePattern and an AttentionBackend, making it
fully pluggable.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from sparse_attention_bench.attention.base import AttentionBackend
from sparse_attention_bench.patterns.base import SparsePattern
from sparse_attention_bench.models.kv_cache import KVCache


class AttentionLayer(nn.Module):
    """
    Multi-head attention with pluggable pattern and backend.

    forward(x) → output tensor, same shape as x.
    forward(x, kv_cache, layer_idx) → output + updated cache for decode mode.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        pattern: SparsePattern,
        backend: AttentionBackend,
        causal: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.causal = causal

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)

        self.pattern = pattern
        self.backend = backend

    def forward(
        self,
        x: torch.Tensor,                    # [B, T, C]
        kv_cache: KVCache | None = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            k, v = kv_cache.append(layer_idx, k, v)

        pattern_meta = self.pattern.build(q, k, causal=self.causal)
        out = self.backend.forward(q, k, v, pattern_meta)  # [B, H, T, D]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)
