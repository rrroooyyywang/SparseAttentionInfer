"""BigBird sparsity math helpers and BigBirdPattern class."""
import math

import torch

from sparse_attention_bench.metrics.accuracy import causal_mask
from sparse_attention_bench.patterns.base import PatternMetadata, SparsePattern
from sparse_attention_bench.patterns.topk_pattern import causal_token_pairs


# ── Math helpers ──────────────────────────────────────────────────────────────

def bigbird_layout_from_topk(seq_len: int, top_k: int) -> tuple[int, int, int, int]:
    if seq_len <= 1:
        return 1, 1, 1, 0
    block_size = max(1, min(seq_len, int(math.sqrt(max(1, top_k)))))
    block_budget = max(1, math.ceil(top_k / block_size))
    global_blocks = 1
    sliding_blocks = min(2, block_budget)
    random_blocks = max(0, block_budget - global_blocks - sliding_blocks)
    return block_size, global_blocks, sliding_blocks, random_blocks


def select_bigbird_random_block_ids(
    query_block: int,
    head_idx: int,
    num_blocks: int,
    global_blocks: int,
    sliding_blocks: int,
    random_blocks: int,
) -> list[int]:
    if random_blocks <= 0 or query_block <= 0:
        return []
    local_start = max(global_blocks, query_block - sliding_blocks + 1)
    candidates = [b for b in range(global_blocks, query_block) if b < local_start]
    if not candidates:
        return []
    if len(candidates) <= random_blocks:
        return candidates
    offset = (head_idx * 131 + query_block * 17 + num_blocks * 7) % len(candidates)
    rotated = candidates[offset:] + candidates[:offset]
    return rotated[:random_blocks]


def estimate_bigbird_attention_keep_ratio(seq_len: int, top_k: int) -> float:
    if seq_len <= 0:
        return 0.0
    if seq_len == 1:
        return 1.0
    block_size, global_blocks, sliding_blocks, random_blocks = bigbird_layout_from_topk(seq_len, top_k)
    num_blocks = math.ceil(seq_len / block_size)
    total_visible = 0
    for query_pos in range(seq_len):
        query_block = query_pos // block_size
        visible = set(range(global_blocks))
        local_start = max(0, query_block - sliding_blocks + 1)
        visible.update(range(local_start, query_block + 1))
        visible.update(select_bigbird_random_block_ids(
            query_block, 0, num_blocks, global_blocks, sliding_blocks, random_blocks
        ))
        row_visible = sum(
            max(0, min(seq_len, (b + 1) * block_size) - b * block_size -
                max(0, min(seq_len, (b + 1) * block_size) - (query_pos + 1)))
            for b in visible
        )
        # simpler: count tokens in visible blocks up to causal limit
        row_count = 0
        for b in visible:
            t_start = b * block_size
            t_end = min(seq_len, (b + 1) * block_size)
            row_count += max(0, min(t_end, query_pos + 1) - t_start)
        total_visible += min(row_count, query_pos + 1)
    return min(1.0, total_visible / max(1, causal_token_pairs(seq_len)))


def estimate_bigbird_decode_keep_ratio(seq_len: int, top_k: int) -> float:
    if seq_len <= 0:
        return 0.0
    if seq_len == 1:
        return 1.0
    block_size, global_blocks, sliding_blocks, random_blocks = bigbird_layout_from_topk(seq_len, top_k)
    num_blocks = math.ceil(seq_len / block_size)
    query_pos = seq_len - 1
    query_block = query_pos // block_size
    visible = set(range(global_blocks))
    local_start = max(0, query_block - sliding_blocks + 1)
    visible.update(range(local_start, query_block + 1))
    visible.update(select_bigbird_random_block_ids(
        query_block, 0, num_blocks, global_blocks, sliding_blocks, random_blocks
    ))
    visible_tokens = sum(
        max(0, min(seq_len, (b + 1) * block_size) - b * block_size)
        for b in visible
    )
    return min(1.0, visible_tokens / seq_len)


# ── Pattern class ─────────────────────────────────────────────────────────────

class BigBirdPattern(SparsePattern):
    """Structured BigBird sparsity: global + sliding-window + random blocks."""

    def __init__(self, top_k: int, n_heads: int):
        self.top_k = top_k
        self.n_heads = n_heads
        self._mask_cache: dict = {}

    def build(self, q: torch.Tensor, k: torch.Tensor, causal: bool = True) -> PatternMetadata:
        T = q.size(2)
        device = q.device
        mask = self._get_mask(T, device)
        keep_ratio = estimate_bigbird_attention_keep_ratio(T, min(self.top_k, T))
        return PatternMetadata(kind="mask", mask=mask, keep_ratio=keep_ratio)

    def _get_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        device_key = f"{device.type}:{device.index}"
        cache_key = (seq_len, device_key, self.top_k)
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]

        block_size, global_blocks, sliding_blocks, random_blocks = bigbird_layout_from_topk(
            seq_len, self.top_k
        )
        num_blocks = math.ceil(seq_len / block_size)
        block_mask = torch.zeros(self.n_heads, num_blocks, num_blocks, dtype=torch.bool, device=device)

        for h in range(self.n_heads):
            for qb in range(num_blocks):
                block_mask[h, qb, :global_blocks] = True
                local_start = max(0, qb - sliding_blocks + 1)
                block_mask[h, qb, local_start:qb + 1] = True
                rand_ids = select_bigbird_random_block_ids(
                    qb, h, num_blocks, global_blocks, sliding_blocks, random_blocks
                )
                if rand_ids:
                    block_mask[h, qb, torch.tensor(rand_ids, device=device)] = True

        t2b = torch.arange(seq_len, device=device) // block_size
        token_mask = block_mask[:, t2b][:, :, t2b]
        causal_keep = ~causal_mask(seq_len, device).squeeze(0).squeeze(0)
        token_mask = token_mask & causal_keep.unsqueeze(0)
        diag = torch.arange(seq_len, device=device)
        token_mask[:, diag, diag] = True

        self._mask_cache[cache_key] = token_mask
        return token_mask
