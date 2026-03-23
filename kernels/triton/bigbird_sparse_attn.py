"""
BigBird block-sparse attention — Triton JIT kernel.

BigBird sparsity structure (per query block):
  - Global blocks  : first `global_blocks` KV blocks (attended by every query)
  - Sliding window : `sliding_blocks` KV blocks ending at the query block (causal)
  - Random blocks  : `random_blocks` KV blocks drawn pseudo-randomly per head

The kernel uses an online-softmax (Flash Attention style) loop that only loads
the allowed KV blocks, avoiding computation for masked-out positions entirely.

Public entry point
------------------
    from kernels.triton.bigbird_sparse_attn import bigbird_sparse_attn

    out = bigbird_sparse_attn(q, k, v, kv_block_list, block_size)

Inputs
------
    q, k, v        : [B, H, T, D]  fp16 or bf16, GPU, contiguous
    kv_block_list  : [H, NQB, MAX_KV]  int32, GPU — block schedule built by
                     BigBirdPattern._get_kv_block_list(); -1 = padding
    block_size     : int, power-of-2, >= 16

Returns
-------
    out : [B, H, T, D] same dtype as q
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


# ── Triton JIT kernel ──────────────────────────────────────────────────────────

@triton.jit
def _bigbird_sparse_attn_kernel(
    # ── tensor pointers ──────────────────────────────────────────────────────
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    # KV block schedule [B*H, NQB, MAX_KV] int32, -1 = padding
    KV_LIST_ptr,
    # ── strides for Q / K / V / Out — each reshaped to [B*H, T, D] ─────────
    stride_qbh, stride_qt,
    stride_kbh, stride_kt,
    stride_vbh, stride_vt,
    stride_obh, stride_ot,
    # ── strides for KV_LIST [B*H, NQB, MAX_KV] ───────────────────────────
    stride_lbh, stride_lqb, stride_lkv,
    # ── scalar params ────────────────────────────────────────────────────────
    T_Q,            # runtime int — number of query tokens
    T_K,            # runtime int — number of key/value tokens (may differ from T_Q in decode)
    Q_OFFSET,       # runtime int — absolute sequence offset of the first query token
                    #   prefill: 0  (query and KV share the same coordinate space)
                    #   decode:  T_K (new token sits after all T_K cached tokens)
    sm_scale,       # 1 / sqrt(D_HEAD) — runtime float
    # ── compile-time constants ───────────────────────────────────────────────
    BLOCK_SIZE: tl.constexpr,   # token block size; power-of-2, >= 16
    D_HEAD:     tl.constexpr,   # head dimension; power-of-2
    MAX_KV:     tl.constexpr,   # padded KV-block slots per query block
):
    """
    Grid: (B*H, num_query_blocks)

    Each program handles one (batch*head, query_block) pair.
    Iterates over the pre-scheduled KV blocks using online softmax to
    accumulate the attention output without materialising the full matrix.
    """
    # ── program IDs ──────────────────────────────────────────────────────────
    bh_idx = tl.program_id(0)   # flattened batch * head index
    qb_idx = tl.program_id(1)   # query block index

    # ── query token offsets ───────────────────────────────────────────────────
    q_start  = qb_idx * BLOCK_SIZE
    offs_t   = tl.arange(0, BLOCK_SIZE)     # [BS]  intra-block token offsets
    offs_d   = tl.arange(0, D_HEAD)          # [D]   head-dim offsets
    q_offs   = q_start + offs_t             # [BS]  positions within the query tensor
    q_valid  = q_offs < T_Q                 # [BS]  bool — within query sequence
    # Absolute positions in the full sequence (for causal check)
    q_abs    = Q_OFFSET + q_offs            # decode: Q_OFFSET=T_K so q_abs >= all k_abs

    # ── load Q block [BS, D] ─────────────────────────────────────────────────
    q_ptrs = (Q_ptr
              + bh_idx * stride_qbh
              + q_offs[:, None] * stride_qt   # [BS, 1]
              + offs_d[None, :])              # [1, D]  (innermost stride = 1)
    q = tl.load(q_ptrs, mask=q_valid[:, None], other=0.0)  # [BS, D]

    # ── online-softmax state (float32 accumulators) ──────────────────────────
    m_i = tl.full([BLOCK_SIZE], float("-inf"), dtype=tl.float32)  # row max
    l_i = tl.zeros([BLOCK_SIZE], dtype=tl.float32)                 # exp sum
    acc = tl.zeros([BLOCK_SIZE, D_HEAD], dtype=tl.float32)         # output acc

    # ── iterate over scheduled KV blocks ─────────────────────────────────────
    list_base = bh_idx * stride_lbh + qb_idx * stride_lqb

    for step in range(MAX_KV):
        # Load KV block index (-1 means padding → skip)
        kb_id    = tl.load(KV_LIST_ptr + list_base + step * stride_lkv)
        is_valid = kb_id >= 0
        # Replace -1 with 0 so pointer arithmetic stays in-bounds
        safe_kb  = tl.where(is_valid, kb_id, 0)

        # Absolute key token positions for this block
        k_start = safe_kb * BLOCK_SIZE
        k_offs  = k_start + offs_t             # [BS]  absolute key positions
        k_valid = k_offs < T_K                 # [BS]  bool — within KV sequence

        # Load K block [BS, D] — zeroed if block is invalid or out-of-range
        k_ptrs = (K_ptr
                  + bh_idx * stride_kbh
                  + k_offs[:, None] * stride_kt
                  + offs_d[None, :])
        k = tl.load(k_ptrs, mask=k_valid[:, None] & is_valid, other=0.0)

        # Load V block [BS, D]
        v_ptrs = (V_ptr
                  + bh_idx * stride_vbh
                  + k_offs[:, None] * stride_vt
                  + offs_d[None, :])
        v = tl.load(v_ptrs, mask=k_valid[:, None] & is_valid, other=0.0)

        # ── QK attention scores [BS, BS] ─────────────────────────────────────
        # fp16/bf16 x fp16/bf16 → fp32 accumulation inside tl.dot
        qk = tl.dot(q, tl.trans(k)) * sm_scale   # [BS, BS], float32

        # ── causal + validity mask ────────────────────────────────────────────
        causal_ok = q_abs[:, None] >= k_offs[None, :]    # [BS, BS]  absolute positions
        tok_ok    = q_valid[:, None] & k_valid[None, :]  # [BS, BS]
        # is_valid is scalar — broadcasts to [BS, BS]
        attend    = causal_ok & tok_ok & is_valid
        qk        = tl.where(attend, qk, float("-inf"))

        # ── online softmax update ─────────────────────────────────────────────
        m_ij  = tl.max(qk, axis=1)                     # [BS]  row max this block
        m_new = tl.maximum(m_i, m_ij)                  # [BS]  updated row max
        alpha = tl.exp(m_i - m_new)                    # [BS]  rescale factor
        p     = tl.exp(qk - m_new[:, None])            # [BS, BS]  unnorm probs

        l_i = alpha * l_i + tl.sum(p, axis=1)          # [BS]
        # cast p to input dtype for the PV dot (avoids float32 dot)
        acc = alpha[:, None] * acc + tl.dot(p.to(q.dtype), v)  # [BS, D], fp32
        m_i = m_new

    # ── normalise ─────────────────────────────────────────────────────────────
    # Guard against l_i == 0 (all-padding rows — only possible for OOB tokens)
    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    out = (acc / l_safe[:, None]).to(q.dtype)  # [BS, D]

    # ── store output ──────────────────────────────────────────────────────────
    o_ptrs = (Out_ptr
              + bh_idx * stride_obh
              + q_offs[:, None] * stride_ot
              + offs_d[None, :])
    tl.store(o_ptrs, out, mask=q_valid[:, None])


# ── Python entry point ─────────────────────────────────────────────────────────

def bigbird_sparse_attn(
    q: torch.Tensor,              # [B, H, T, D]  fp16 or bf16, contiguous, CUDA
    k: torch.Tensor,
    v: torch.Tensor,
    kv_block_list: torch.Tensor,  # [H, NQB, MAX_KV]  int32, CUDA
    block_size: int,              # power-of-2, >= 16 — Triton BLOCK_SIZE
) -> torch.Tensor:
    """
    Block-sparse BigBird attention via Triton.

    The KV block schedule (`kv_block_list`) must have been built with the
    same `block_size` as passed here (BigBirdPattern._get_kv_block_list does
    this automatically).

    Returns output tensor of the same shape and dtype as `q`.
    """
    B, H, T_q, D = q.shape
    T_k = k.size(2)
    BH     = B * H
    NQB    = kv_block_list.size(1)
    MAX_KV = kv_block_list.size(2)

    assert q.is_cuda, "bigbird_sparse_attn requires CUDA tensors"
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert q.dtype in (torch.float16, torch.bfloat16), \
        f"Expected fp16 or bf16, got {q.dtype}"
    assert block_size >= 16 and (block_size & (block_size - 1)) == 0, \
        "block_size must be a power of 2 and >= 16"

    sm_scale = 1.0 / math.sqrt(D)

    # In decode mode the query sits after all T_k cached tokens, so its absolute
    # sequence position is T_k.  In prefill T_q == T_k and offset is 0.
    Q_OFFSET = T_k if T_q < T_k else 0

    # ── Reshape to [B*H, T, D] for simpler kernel indexing ────────────────────
    q_  = q.reshape(BH, T_q, D)
    k_  = k.reshape(BH, T_k, D)   # T_k, not T_q — fixes the decode crash
    v_  = v.reshape(BH, T_k, D)
    out_ = torch.empty_like(q_)

    # ── Expand kv_block_list from [H, NQB, MAX_KV] to [B*H, NQB, MAX_KV] ─────
    # All items in the batch share the same BigBird block schedule per head.
    kv_list_bh = (
        kv_block_list
        .unsqueeze(0)                         # [1, H, NQB, MAX_KV]
        .expand(B, -1, -1, -1)               # [B, H, NQB, MAX_KV]
        .reshape(BH, NQB, MAX_KV)            # [B*H, NQB, MAX_KV]
        .contiguous()
    )

    grid = (BH, NQB)
    _bigbird_sparse_attn_kernel[grid](
        q_, k_, v_, out_,
        kv_list_bh,
        # Q / K / V / Out strides (innermost D-stride is 1, not passed)
        q_.stride(0),  q_.stride(1),
        k_.stride(0),  k_.stride(1),
        v_.stride(0),  v_.stride(1),
        out_.stride(0), out_.stride(1),
        # kv_block_list strides
        kv_list_bh.stride(0), kv_list_bh.stride(1), kv_list_bh.stride(2),
        # scalar params
        T_Q=T_q, T_K=T_k, Q_OFFSET=Q_OFFSET,
        sm_scale=sm_scale,
        # compile-time constants
        BLOCK_SIZE=block_size,
        D_HEAD=D,
        MAX_KV=MAX_KV,
    )
    return out_.reshape(B, H, T_q, D)
