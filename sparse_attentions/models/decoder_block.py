"""
Decoder block building blocks for end-to-end benchmarking.

DecoderBlock: RMSNorm + AttentionLayer + RMSNorm + MLP + residual connections.
ToyDecoder:   Stack of DecoderBlocks with token/position embeddings.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from sparse_attentions.attention.base import AttentionBackend
from sparse_attentions.models.attention_layer import AttentionLayer
from sparse_attentions.models.kv_cache import KVCache
from sparse_attentions.patterns.base import SparsePattern


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


class MLP(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: int = 4, bias: bool = False):
        super().__init__()
        hidden = d_model * mlp_ratio
        self.fc1 = nn.Linear(d_model, hidden, bias=bias)
        self.fc2 = nn.Linear(hidden, d_model, bias=bias)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class DecoderBlock(nn.Module):
    """
    Single transformer decoder block:
        x → RMSNorm → AttentionLayer → residual
          → RMSNorm → MLP → residual
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        pattern: SparsePattern,
        backend: AttentionBackend,
        mlp_ratio: int = 4,
        causal: bool = True,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = AttentionLayer(
            d_model=d_model,
            num_heads=num_heads,
            pattern=pattern,
            backend=backend,
            causal=causal,
        )
        self.norm2 = RMSNorm(d_model)
        self.mlp = MLP(d_model=d_model, mlp_ratio=mlp_ratio)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: KVCache | None = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), kv_cache=kv_cache, layer_idx=layer_idx)
        x = x + self.mlp(self.norm2(x))
        return x


class ToyDecoder(nn.Module):
    """
    Minimal toy decoder for end-to-end benchmarking.
    Each layer uses the same pattern and backend (configurable per-layer in future).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        pattern: SparsePattern,
        backend: AttentionBackend,
        max_seq_len: int = 4096,
        mlp_ratio: int = 4,
        causal: bool = True,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            DecoderBlock(
                d_model=d_model, num_heads=num_heads,
                pattern=pattern, backend=backend,
                mlp_ratio=mlp_ratio, causal=causal,
            )
            for _ in range(num_layers)
        ])
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # weight tying

    def forward(
        self,
        input_ids: torch.Tensor,          # [B, T]
        kv_cache: KVCache | None = None,
        pos_offset: int = 0,
    ) -> torch.Tensor:
        B, T = input_ids.shape
        device = input_ids.device
        pos = torch.arange(pos_offset, pos_offset + T, device=device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        for i, blk in enumerate(self.blocks):
            x = blk(x, kv_cache=kv_cache, layer_idx=i)
        return self.lm_head(self.norm_f(x))
