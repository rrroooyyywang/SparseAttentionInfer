"""Base classes for sparse patterns."""
from __future__ import annotations

import math
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
    # ── block-sparse layout for Triton kernels ─────────────────────────────────
    block_size: int | None = None          # power-of-2 block size used by kv_block_list
    kv_block_list: torch.Tensor | None = None  # [H, NQB, MAX_KV] int32, -1=padding (legacy)
    # ── CSR block-pair format for universal_sparse_attn ────────────────────────
    # Padding-free: block_pairs[h, offsets[h,qb] : offsets[h,qb+1]] are the KB
    # indices that query block qb in head h attends to.
    block_pairs: torch.Tensor | None = None         # [H, max_pairs] int32
    block_pair_offsets: torch.Tensor | None = None  # [H, NQB+1] int32


def build_block_pairs_from_mask(
    mask: torch.Tensor,   # [H, T_q, T_k] bool, True = attend
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert any boolean attention mask to CSR block-pair format.

    Coarsens the token-level mask to block level: a block pair (qb, kb) is
    included iff at least one token in mask[h, qb*BS:(qb+1)*BS, kb*BS:(kb+1)*BS]
    is True.

    Returns:
        block_pairs        : [H, max_pairs] int32  — flat sorted KB indices per head
        block_pair_offsets : [H, NQB+1]    int32  — CSR row pointers
    """
    import torch.nn.functional as F

    H, T_q, T_k = mask.shape
    BS = block_size
    NQB = math.ceil(T_q / BS)
    NKB = math.ceil(T_k / BS)
    device = mask.device

    # Coarsen to block level: [H, NQB, NKB] bool
    pad_q = NQB * BS - T_q
    pad_k = NKB * BS - T_k
    m = F.pad(mask.float(), (0, pad_k, 0, pad_q))   # [H, NQB*BS, NKB*BS]
    m = m.reshape(H, NQB, BS, NKB, BS)
    block_active = m.any(dim=(2, 4))                  # [H, NQB, NKB] bool

    all_pairs: list[list[int]] = []
    all_offsets: list[list[int]] = []
    max_pairs = 0

    for h in range(H):
        offsets_h = [0]
        pairs_h: list[int] = []
        for qb in range(NQB):
            kb_list = block_active[h, qb].nonzero(as_tuple=False).view(-1).tolist()
            pairs_h.extend(int(x) for x in kb_list)
            offsets_h.append(len(pairs_h))
        all_pairs.append(pairs_h)
        all_offsets.append(offsets_h)
        max_pairs = max(max_pairs, len(pairs_h))

    pairs_t   = torch.zeros(H, max(1, max_pairs), dtype=torch.int32, device=device)
    offsets_t = torch.zeros(H, NQB + 1,           dtype=torch.int32, device=device)
    for h in range(H):
        n = len(all_pairs[h])
        if n > 0:
            pairs_t[h, :n] = torch.tensor(all_pairs[h], dtype=torch.int32, device=device)
        offsets_t[h] = torch.tensor(all_offsets[h], dtype=torch.int32, device=device)

    return pairs_t, offsets_t


def kv_block_list_to_pairs(
    kv_block_list: torch.Tensor,  # [H, NQB, MAX_KV] int32, -1=padding
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert the legacy padded kv_block_list to CSR block-pair format.

    Useful as a migration bridge: patterns that still build kv_block_list can
    call this to populate block_pairs / block_pair_offsets without duplicating
    logic.

    Returns:
        block_pairs        : [H, max_pairs] int32
        block_pair_offsets : [H, NQB+1]    int32
    """
    H, NQB, _ = kv_block_list.shape
    device = kv_block_list.device
    valid = kv_block_list >= 0

    all_pairs: list[list[int]] = []
    all_offsets: list[list[int]] = []
    max_pairs = 0

    for h in range(H):
        offsets_h = [0]
        pairs_h: list[int] = []
        for qb in range(NQB):
            kb_vals = kv_block_list[h, qb][valid[h, qb]].tolist()
            pairs_h.extend(int(x) for x in kb_vals)
            offsets_h.append(len(pairs_h))
        all_pairs.append(pairs_h)
        all_offsets.append(offsets_h)
        max_pairs = max(max_pairs, len(pairs_h))

    pairs_t   = torch.zeros(H, max(1, max_pairs), dtype=torch.int32, device=device)
    offsets_t = torch.zeros(H, NQB + 1,           dtype=torch.int32, device=device)
    for h in range(H):
        n = len(all_pairs[h])
        if n > 0:
            pairs_t[h, :n] = torch.tensor(all_pairs[h], dtype=torch.int32, device=device)
        offsets_t[h] = torch.tensor(all_offsets[h], dtype=torch.int32, device=device)

    return pairs_t, offsets_t


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
