"""Dense and sparse decoder models used by the proxy profiler."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from sparse_attentions.patterns.bigbird_pattern import BigBirdPattern
from sparse_attentions.patterns.topk_pattern import (
    feature_keep_from_attention_keep,
    sparsify_last_dim_topk,
)
from sparse_attentions.utils import causal_mask

if TYPE_CHECKING:
    from sparse_attention_bench.analytical.config import DecoderConfig


class DenseSelfAttention(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(causal_mask(T, x.device), float("-inf"))
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class SparseSelfAttention(nn.Module):
    """
    Sparse attention: feature-dim sparsification + row-wise top-k or BigBird mask.
    The forward still forms dense scores before sparsifying (not a real sparse kernel).
    """
    def __init__(self, cfg: DecoderConfig, top_k: int):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.sparse_mode = cfg.sparse_mode
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.top_k = top_k
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        self._bigbird = BigBirdPattern(top_k=top_k, n_heads=cfg.n_heads) if cfg.sparse_mode == "bigbird" else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k_keep = min(self.top_k, T)

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if self.sparse_mode == "top-k":
            feature_keep = feature_keep_from_attention_keep(self.head_dim, k_keep / T)
            q = sparsify_last_dim_topk(q, keep_k=feature_keep)
            k = sparsify_last_dim_topk(k, keep_k=feature_keep)
            v = sparsify_last_dim_topk(v, keep_k=feature_keep)
        elif self.sparse_mode != "bigbird":
            raise ValueError(f"Unsupported sparse mode: {self.sparse_mode}")

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(causal_mask(T, x.device), float("-inf"))

        if self.sparse_mode == "top-k":
            topk_vals, topk_idx = torch.topk(scores, k=k_keep, dim=-1)
            sparse_scores = torch.full_like(scores, float("-inf"))
            sparse_scores.scatter_(-1, topk_idx, topk_vals)
        else:  # bigbird
            bigbird_mask = self._bigbird._get_mask(T, T, x.device)
            sparse_scores = scores.masked_fill(~bigbird_mask.unsqueeze(0), float("-inf"))

        attn = self.dropout(F.softmax(sparse_scores, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class FeedForward(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        hidden = cfg.d_model * cfg.mlp_ratio
        self.fc1 = nn.Linear(cfg.d_model, hidden)
        self.fc2 = nn.Linear(hidden, cfg.d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class DecoderBlock(nn.Module):
    def __init__(self, cfg: DecoderConfig, attn: nn.Module):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = attn
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class DenseDecoder(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([DecoderBlock(cfg, DenseSelfAttention(cfg)) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # weight tying

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        return self.lm_head(self.ln_f(x))


class SparseDecoder(nn.Module):
    def __init__(self, cfg: DecoderConfig, top_k: int):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([
            DecoderBlock(cfg, SparseSelfAttention(cfg, top_k=top_k))
            for _ in range(cfg.n_layers)
        ])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        return self.lm_head(self.ln_f(x))


def build_sparse_from_dense(dense_model: DenseDecoder, cfg: DecoderConfig, top_k: int) -> SparseDecoder:
    sparse_model = SparseDecoder(cfg, top_k=top_k).to(next(dense_model.parameters()).device)
    sparse_model.load_state_dict(dense_model.state_dict(), strict=True)
    return sparse_model
