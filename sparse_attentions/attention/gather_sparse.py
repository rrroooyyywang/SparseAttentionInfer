"""
Gather-sparse attention backend.

For patterns with pre-determined, fixed-width connectivity (e.g. local window),
gather only the relevant K/V positions for each query, then compute a smaller
GEMM instead of applying a mask to the full score matrix.

This is more efficient than masked SDPA for large seq_len with small window,
because the dense QK matmul is avoided entirely.

Limitation: requires a fixed number of K positions per query (uniform gather width).
This works for local window; top-k is non-uniform so falls back to masked_sdpa.
"""
import math

import torch
import torch.nn.functional as F

from sparse_attentions.attention.base import AttentionBackend
from sparse_attentions.attention.masked_sdpa import MaskedSdpaBackend
from sparse_attentions.patterns.base import PatternMetadata


class GatherSparseBackend(AttentionBackend):
    """
    Gather-based sparse attention.

    For "local" patterns: gathers a fixed window of K/V positions per query,
    computes a [B, H, T_q, W] matmul, applies softmax, computes [B, H, T_q, D].

    Falls back to MaskedSdpaBackend for "mask", "topk", and "dense" patterns.
    """

    def __init__(self):
        self._fallback = MaskedSdpaBackend()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pattern: PatternMetadata,
    ) -> torch.Tensor:
        if pattern.kind != "local" or pattern.mask is None:
            return self._fallback.forward(q, k, v, pattern)

        return self._gather_forward(q, k, v, pattern)

    def _gather_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pattern: PatternMetadata,
    ) -> torch.Tensor:
        B, H, T_q, D = q.shape
        T_k = k.size(2)
        scale = 1.0 / math.sqrt(D)
        device = q.device

        # pattern.mask: [H, T_q, T_k] bool, True = keep
        # For local window, each query has the same number of valid positions.
        # Find window width W = max number of True positions per row.
        mask = pattern.mask  # [H, T_q, T_k]
        # Use first head as representative (local window is head-independent)
        row_mask = mask[0]   # [T_q, T_k]

        # Build gather indices: for each query position, collect the T_k indices
        # that are True. Pad with 0 for rows with fewer than W positions.
        counts = row_mask.sum(dim=-1)  # [T_q]
        W = int(counts.max().item())
        if W == 0:
            return torch.zeros(B, H, T_q, D, device=device, dtype=q.dtype)

        # Build padded gather index [T_q, W]
        gather_idx = torch.zeros(T_q, W, dtype=torch.long, device=device)
        for i in range(T_q):
            valid = row_mask[i].nonzero(as_tuple=True)[0]
            n = valid.size(0)
            gather_idx[i, :n] = valid
            if n < W:
                gather_idx[i, n:] = valid[-1] if n > 0 else 0  # pad with last valid

        # Expand for B and H: [B, H, T_q, W]
        idx = gather_idx.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)
        # Expand for D: [B, H, T_q, W, D]
        idx_d = idx.unsqueeze(-1).expand(-1, -1, -1, -1, D)

        # Gather K and V
        k_exp = k.unsqueeze(3).expand(-1, -1, -1, W, -1)  # won't work directly
        # k: [B, H, T_k, D] → gather along T_k dim
        k_gathered = k.gather(2, idx_d.view(B, H, T_q * W, D)).view(B, H, T_q, W, D)
        v_gathered = v.gather(2, idx_d.view(B, H, T_q * W, D)).view(B, H, T_q, W, D)

        # Scores: q [B,H,T_q,D] × k_gathered [B,H,T_q,W,D]
        # → [B, H, T_q, W]
        scores = torch.einsum("bhqd,bhqwd->bhqw", q, k_gathered) * scale

        # Mask out padded positions
        valid_mask = torch.zeros(T_q, W, dtype=torch.bool, device=device)
        for i in range(T_q):
            n = int(counts[i].item())
            valid_mask[i, :n] = True
        scores = scores.masked_fill(~valid_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)

        # Output: attn [B,H,T_q,W] × v_gathered [B,H,T_q,W,D] → [B,H,T_q,D]
        out = torch.einsum("bhqw,bhqwd->bhqd", attn, v_gathered)
        return out
