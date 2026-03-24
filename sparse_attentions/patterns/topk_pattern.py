"""Top-k sparsity math helpers and TopKPattern class."""
import math

import torch

from sparse_attentions.patterns.base import PatternMetadata, SparsePattern


# ── Math helpers (used by proxy estimator and pattern class) ──────────────────

def sparsify_last_dim_topk(x: torch.Tensor, keep_k: int) -> torch.Tensor:
    """Zero out all but the top-k entries on the last dimension."""
    if keep_k >= x.size(-1):
        return x
    _, idx = torch.topk(x.abs(), k=keep_k, dim=-1)
    out = torch.zeros_like(x)
    out.scatter_(-1, idx, x.gather(-1, idx))
    return out


def feature_keep_from_attention_keep(head_dim: int, attention_keep_ratio: float) -> int:
    return min(head_dim, max(1, math.ceil(head_dim * attention_keep_ratio)))


def causal_token_pairs(seq_len: int) -> int:
    return seq_len * (seq_len + 1) // 2


def effective_topk_attention_keep_ratio(seq_len: int, top_k: int) -> float:
    if seq_len <= 0:
        return 0.0
    total_kept = sum(min(top_k, pos + 1) for pos in range(seq_len))
    return min(1.0, total_kept / max(1, causal_token_pairs(seq_len)))


def decode_topk_attention_keep_ratio(seq_len: int, top_k: int) -> float:
    if seq_len <= 0:
        return 0.0
    return min(top_k, seq_len) / seq_len


# ── Pattern class ─────────────────────────────────────────────────────────────

class TopKPattern(SparsePattern):
    """
    Data-dependent top-k pattern.
    build() returns metadata with kind="topk"; the backend applies the
    top-k masking after computing raw QK scores.
    """
    def __init__(self, top_k: int):
        self.top_k = top_k

    def build(self, q: torch.Tensor, k: torch.Tensor, causal: bool = True) -> PatternMetadata:
        # Pattern metadata only carries parameters; the backend handles selection.
        T_q = q.size(2)
        T_k = k.size(2)
        top_k = min(self.top_k, T_k)
        if T_q == T_k:
            keep_ratio = effective_topk_attention_keep_ratio(T_k, top_k)
        else:
            keep_ratio = decode_topk_attention_keep_ratio(T_k, top_k)
        return PatternMetadata(kind="topk", topk=self.top_k, keep_ratio=keep_ratio)
