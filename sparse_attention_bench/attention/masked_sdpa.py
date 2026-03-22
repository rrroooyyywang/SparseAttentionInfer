"""Masked SDPA backend: applies sparse pattern mask then runs SDPA."""
import math

import torch
import torch.nn.functional as F

from sparse_attention_bench.attention.base import AttentionBackend
from sparse_attention_bench.metrics.accuracy import causal_mask
from sparse_attention_bench.patterns.base import PatternMetadata


class MaskedSdpaBackend(AttentionBackend):
    """
    Applies the pattern mask (boolean) to raw QK scores, then runs softmax + AV.

    Supports three pattern kinds:
    - "dense": full causal attention via is_causal=True
    - "mask": pre-built [H, T, T] boolean mask (True = KEEP)
    - "topk": compute full QK, select top-k per row, then softmax + AV
    - "local": same as "mask"
    """

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pattern: PatternMetadata,
    ) -> torch.Tensor:
        if pattern.kind == "dense":
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)

        B, H, T_q, D = q.shape
        T_k = k.size(2)
        scale = 1.0 / math.sqrt(D)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, T_q, T_k]

        if pattern.kind in ("mask", "local"):
            # pattern.mask: [H, T, T], True = keep
            keep_mask = pattern.mask  # [H, T_q, T_k]
            # Apply causal mask on top of pattern mask
            c_mask = causal_mask(T_k, q.device)  # [1, 1, T_k, T_k]
            if T_q == 1:
                # decode: only last row of causal mask matters
                causal_keep = ~c_mask[:, :, -1:, :]  # [1, 1, 1, T_k]
            else:
                causal_keep = ~c_mask  # [1, 1, T_q, T_k]
            combined = keep_mask.unsqueeze(0) & causal_keep
            scores = scores.masked_fill(~combined, float("-inf"))

        elif pattern.kind == "topk":
            top_k = min(pattern.topk, T_k)
            # First apply causal mask
            c_mask = causal_mask(T_k, q.device)
            if T_q == 1:
                scores = scores.masked_fill(c_mask[:, :, -1:, :], float("-inf"))
            else:
                scores = scores.masked_fill(c_mask, float("-inf"))
            # Then keep top-k per row
            topk_vals, topk_idx = torch.topk(scores, k=top_k, dim=-1)
            sparse_scores = torch.full_like(scores, float("-inf"))
            sparse_scores.scatter_(-1, topk_idx, topk_vals)
            scores = sparse_scores

        attn = F.softmax(scores, dim=-1)
        # Replace NaN rows (all -inf → softmax = NaN) with 0
        attn = torch.nan_to_num(attn, nan=0.0)
        return torch.matmul(attn, v)
