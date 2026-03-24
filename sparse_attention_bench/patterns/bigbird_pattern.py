"""BigBird sparsity math helpers and BigBirdPattern class."""
import math

import torch

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


# ── Triton block-size helper ───────────────────────────────────────────────────

def _next_pow2_min16(x: int) -> int:
    """Smallest power of 2 that is both >= x and >= 16 (Triton tl.dot minimum)."""
    p = max(16, x)
    result = 1
    while result < p:
        result <<= 1
    return result


# ── Pattern class ─────────────────────────────────────────────────────────────

class BigBirdPattern(SparsePattern):
    """Structured BigBird sparsity: global + sliding-window + random blocks."""

    def __init__(self, top_k: int, n_heads: int):
        self.top_k = top_k
        self.n_heads = n_heads
        self._mask_cache: dict = {}
        self._kv_cache: dict = {}
        self._keep_ratio_cache: dict = {}

    def build(self, q: torch.Tensor, k: torch.Tensor, causal: bool = True) -> PatternMetadata:
        T_q = q.size(2)
        T_k = k.size(2)
        device = q.device
        mask = self._get_mask(T_q, T_k, device)
        keep_ratio = self._get_keep_ratio(T_q, T_k)
        kv_block_list, triton_bs = self._get_kv_block_list(T_q, T_k, device)
        return PatternMetadata(
            kind="mask",
            mask=mask,
            keep_ratio=keep_ratio,
            block_size=triton_bs,
            kv_block_list=kv_block_list,
        )

    def _get_keep_ratio(self, T_q: int, T_k: int) -> float:
        cache_key = (T_q, T_k, self.top_k)
        if cache_key not in self._keep_ratio_cache:
            if T_q == T_k:
                keep_ratio = estimate_bigbird_attention_keep_ratio(T_k, min(self.top_k, T_k))
            else:
                keep_ratio = estimate_bigbird_decode_keep_ratio(T_k, min(self.top_k, T_k))
            self._keep_ratio_cache[cache_key] = keep_ratio
        return self._keep_ratio_cache[cache_key]

    def _get_mask(self, T_q: int, T_k: int, device: torch.device) -> torch.Tensor:
        device_key = f"{device.type}:{device.index}"
        cache_key = (T_q, T_k, device_key, self.top_k)
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]

        block_size, global_blocks, sliding_blocks, random_blocks = bigbird_layout_from_topk(
            T_k, self.top_k
        )
        num_blocks = math.ceil(T_k / block_size)
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

        query_offset = max(0, T_k - T_q)
        q_positions = query_offset + torch.arange(T_q, device=device)
        q2b = q_positions // block_size
        k2b = torch.arange(T_k, device=device) // block_size
        token_mask = block_mask[:, q2b][:, :, k2b]

        q_pos = q_positions.unsqueeze(1)
        k_pos = torch.arange(T_k, device=device).unsqueeze(0)
        causal_keep = k_pos <= q_pos
        token_mask = token_mask & causal_keep.unsqueeze(0)
        diag_q = torch.arange(T_q, device=device)
        token_mask[:, diag_q, q_positions] = True

        self._mask_cache[cache_key] = token_mask
        return token_mask

    def _get_kv_block_list(
        self,
        T_q: int,
        T_k: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, int]:
        """
        Build the KV block schedule used by the Triton kernel.

        Works for both prefill (T_q == T_k) and decode (T_q < T_k), where the
        queries correspond to the most recent tokens in the KV sequence.

        Returns:
            kv_block_list : [H, num_query_blocks, MAX_KV] int32
                            Sorted KV block indices for each (head, query_block).
                            Entries past the last valid slot are -1 (padding).
            triton_bs     : int — power-of-2 block size (>= 16) used to build
                            this schedule.  Must match the BLOCK_SIZE passed to
                            bigbird_sparse_attn().
        """
        device_key = f"{device.type}:{device.index}"
        cache_key = (T_q, T_k, device_key, self.top_k)
        if cache_key in self._kv_cache:
            return self._kv_cache[cache_key]

        # Use T_k to determine block layout (that is the context length)
        orig_bs, global_blocks, sliding_blocks, random_blocks = bigbird_layout_from_topk(
            T_k, self.top_k
        )
        triton_bs = _next_pow2_min16(orig_bs)
        num_query_blocks = math.ceil(T_q / triton_bs)
        num_kv_blocks    = math.ceil(T_k / triton_bs)
        # Upper bound on unique KV blocks per (head, query_block)
        max_kv = max(1, global_blocks + sliding_blocks + random_blocks)

        kv_block_list = torch.full(
            (self.n_heads, num_query_blocks, max_kv),
            fill_value=-1,
            dtype=torch.int32,
            device=device,
        )

        for h in range(self.n_heads):
            for qb_local in range(num_query_blocks):
                query_offset = max(0, T_k - T_q)
                query_block_offset = query_offset // triton_bs
                qb_abs = query_block_offset + qb_local

                allowed: set[int] = set(range(global_blocks))
                local_start = max(0, qb_abs - sliding_blocks + 1)
                # Sliding window capped at num_kv_blocks (query block itself is not a KV block)
                allowed.update(range(local_start, min(qb_abs + 1, num_kv_blocks)))
                rand_ids = select_bigbird_random_block_ids(
                    qb_abs, h, num_kv_blocks,
                    global_blocks, sliding_blocks, random_blocks,
                )
                allowed.update(rand_ids)
                for slot, kb in enumerate(sorted(allowed)[:max_kv]):
                    kv_block_list[h, qb_local, slot] = kb

        result = (kv_block_list, triton_bs)
        self._kv_cache[cache_key] = result
        return result


# ── BigBird2: first AND last key blocks are global ────────────────────────────

class BigBird2Pattern(BigBirdPattern):
    """BigBird where the last query block attends to all key blocks (full row).

    Analogous to how token-0 is a global KEY (every query attends to it),
    here the last query block is a global QUERY: it attends to every key
    position within the causal limit (which for the final token is the full
    context).  All other rows follow normal BigBird sparsity.
    """

    def _get_mask(self, T_q: int, T_k: int, device: torch.device) -> torch.Tensor:
        device_key = f"{device.type}:{device.index}"
        cache_key = (T_q, T_k, device_key, self.top_k)
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]

        block_size, global_blocks, sliding_blocks, random_blocks = bigbird_layout_from_topk(
            T_k, self.top_k
        )
        num_blocks = math.ceil(T_k / block_size)

        # In a decode step T_q < T_k; the "last query block" is the only query block.
        query_offset = max(0, T_k - T_q)
        num_query_blocks = math.ceil(T_q / block_size)
        last_query_block_local = num_query_blocks - 1             # local index in query
        block_mask = torch.zeros(self.n_heads, num_query_blocks, num_blocks, dtype=torch.bool, device=device)

        for h in range(self.n_heads):
            for qb_local in range(num_query_blocks):
                qb_abs = (query_offset // block_size) + qb_local
                if qb_local == last_query_block_local:
                    # BigBird2: last query block attends to all key blocks
                    block_mask[h, qb_local, :] = True
                else:
                    block_mask[h, qb_local, :global_blocks] = True
                    local_start = max(0, qb_abs - sliding_blocks + 1)
                    block_mask[h, qb_local, local_start:qb_abs + 1] = True
                    rand_ids = select_bigbird_random_block_ids(
                        qb_abs, h, num_blocks, global_blocks, sliding_blocks, random_blocks
                    )
                    if rand_ids:
                        block_mask[h, qb_local, torch.tensor(rand_ids, device=device)] = True

        q_positions = query_offset + torch.arange(T_q, device=device)
        q2b_local = torch.arange(T_q, device=device) // block_size   # local query-block index
        k2b = torch.arange(T_k, device=device) // block_size
        token_mask = block_mask[:, q2b_local][:, :, k2b]

        q_pos = q_positions.unsqueeze(1)
        k_pos = torch.arange(T_k, device=device).unsqueeze(0)
        causal_keep = k_pos <= q_pos
        token_mask = token_mask & causal_keep.unsqueeze(0)

        diag_q = torch.arange(T_q, device=device)
        token_mask[:, diag_q, q_positions] = True

        # Compute keep_ratio from actual mask
        ratio_key = (T_q, T_k, self.top_k)
        if ratio_key not in self._keep_ratio_cache:
            self._keep_ratio_cache[ratio_key] = token_mask.float().mean().item()

        self._mask_cache[cache_key] = token_mask
        return token_mask

    def _get_kv_block_list(
        self,
        T_q: int,
        T_k: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, int]:
        device_key = f"{device.type}:{device.index}"
        cache_key = (T_q, T_k, device_key, self.top_k)
        if cache_key in self._kv_cache:
            return self._kv_cache[cache_key]

        orig_bs, global_blocks, sliding_blocks, random_blocks = bigbird_layout_from_topk(
            T_k, self.top_k
        )
        triton_bs = _next_pow2_min16(orig_bs)
        num_query_blocks = math.ceil(T_q / triton_bs)
        num_kv_blocks = math.ceil(T_k / triton_bs)
        last_query_block_local = num_query_blocks - 1
        # Last query block needs all KV blocks; others follow normal budget
        max_kv = max(num_kv_blocks, global_blocks + sliding_blocks + random_blocks)

        kv_block_list = torch.full(
            (self.n_heads, num_query_blocks, max_kv),
            fill_value=-1,
            dtype=torch.int32,
            device=device,
        )

        for h in range(self.n_heads):
            for qb_local in range(num_query_blocks):
                query_offset = max(0, T_k - T_q)
                query_block_offset = query_offset // triton_bs
                qb_abs = query_block_offset + qb_local

                if qb_local == last_query_block_local:
                    # BigBird2: last query block attends all KV blocks
                    allowed: set[int] = set(range(num_kv_blocks))
                else:
                    allowed = set(range(global_blocks))
                    local_start = max(0, qb_abs - sliding_blocks + 1)
                    allowed.update(range(local_start, min(qb_abs + 1, num_kv_blocks)))
                    rand_ids = select_bigbird_random_block_ids(
                        qb_abs, h, num_kv_blocks,
                        global_blocks, sliding_blocks, random_blocks,
                    )
                    allowed.update(rand_ids)
                for slot, kb in enumerate(sorted(allowed)[:max_kv]):
                    kv_block_list[h, qb_local, slot] = kb

        result = (kv_block_list, triton_bs)
        self._kv_cache[cache_key] = result
        return result
