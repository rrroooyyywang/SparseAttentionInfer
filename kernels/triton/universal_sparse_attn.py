"""
Universal block-sparse attention — Triton JIT kernel.

Accepts any sparsity pattern encoded as a CSR block-pair schedule built by
SparsePattern.build() (see sparse_attention_bench/patterns/base.py).

Public entry point
------------------
    from kernels.triton.universal_sparse_attn import universal_sparse_attn

    out = universal_sparse_attn(q, k, v, block_pairs, block_pair_offsets, block_size)

Inputs
------
    q, k, v              : [B, H, T, D]  fp16 or bf16, CUDA, contiguous
    block_pairs          : [H, max_pairs]  int32, CUDA
                           Flat sorted KB indices per head.  For head h, query
                           block qb attends to KB indices
                               block_pairs[h, block_pair_offsets[h,qb] :
                                              block_pair_offsets[h,qb+1]]
    block_pair_offsets   : [H, NQB+1]  int32, CUDA — CSR row pointers
    block_size           : int, power-of-2, >= 16

Returns
-------
    out : [B, H, T, D] same dtype as q

Tensor core usage
-----------------
Two GEMMs per block pair use Tensor Cores (fp16/bf16 + block_size >= 16):
    QK   : [BS, D] @ [D, BS]  →  [BS, BS]   via tl.dot(q, tl.trans(k))
    PV   : [BS, BS] @ [BS, D]  →  [BS, D]   via tl.dot(p, v)
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


# ── Triton JIT kernel ──────────────────────────────────────────────────────────

@triton.jit
def _universal_sparse_attn_kernel(
    # ── tensor pointers ──────────────────────────────────────────────────────
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    # CSR block-pair schedule
    PAIRS_ptr,    # [H, max_pairs] int32 — flat KB indices per head
    OFFSETS_ptr,  # [H, NQB+1]    int32 — CSR row pointers
    # ── strides for Q / K / V / Out, each reshaped to [B*H, T, D] ──────────
    stride_qbh, stride_qt,
    stride_kbh, stride_kt,
    stride_vbh, stride_vt,
    stride_obh, stride_ot,
    # ── strides for PAIRS [H, max_pairs] and OFFSETS [H, NQB+1] ────────────
    stride_pairs_h,    # = max_pairs
    stride_offsets_h,  # = NQB + 1
    # ── scalar runtime params ─────────────────────────────────────────────
    H,          # number of heads (int, not constexpr — avoids recompile per head count)
    T_Q,        # number of query tokens
    T_K,        # number of key/value tokens
    Q_OFFSET,   # 0 for prefill; T_K for decode (absolute position of first query token)
    sm_scale,   # 1 / sqrt(D_HEAD)
    # ── compile-time constants ────────────────────────────────────────────
    BLOCK_SIZE: tl.constexpr,   # token block size; power-of-2, >= 16
    D_HEAD:     tl.constexpr,   # head dimension; power-of-2
):
    """
    Grid: (B*H, NQB)

    Each program owns one (batch*head, query_block) pair.
    It reads its exact KB list from the CSR schedule and accumulates the
    attention output via online softmax — no padding, no -1 checks.
    """
    bh_idx = tl.program_id(0)   # flattened batch * head index
    qb_idx = tl.program_id(1)   # query block index

    # Recover head index (scalar modulo — not in any inner loop)
    h_idx = bh_idx % H

    # ── query token offsets ───────────────────────────────────────────────
    q_start = qb_idx * BLOCK_SIZE
    offs_t  = tl.arange(0, BLOCK_SIZE)   # [BS]  intra-block token offsets
    offs_d  = tl.arange(0, D_HEAD)       # [D]   head-dim offsets
    q_offs  = q_start + offs_t           # [BS]  positions within the query tensor
    q_valid = q_offs < T_Q
    q_abs   = Q_OFFSET + q_offs          # absolute positions for causal check

    # ── load Q block [BS, D] ─────────────────────────────────────────────
    q_ptrs = (Q_ptr
              + bh_idx * stride_qbh
              + q_offs[:, None] * stride_qt
              + offs_d[None, :])
    q = tl.load(q_ptrs, mask=q_valid[:, None], other=0.0)   # [BS, D]

    # ── online-softmax state (float32 accumulators) ───────────────────────
    m_i = tl.full([BLOCK_SIZE], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    acc = tl.zeros([BLOCK_SIZE, D_HEAD], dtype=tl.float32)

    # ── CSR range for (h_idx, qb_idx) — exact pair count, no padding ─────
    off_base = h_idx * stride_offsets_h + qb_idx
    start = tl.load(OFFSETS_ptr + off_base)
    end   = tl.load(OFFSETS_ptr + off_base + 1)

    pairs_base = h_idx * stride_pairs_h

    # ── iterate over scheduled KV blocks (variable length, no -1 checks) ─
    for step in range(start, end):
        kb_idx  = tl.load(PAIRS_ptr + pairs_base + step)
        k_start = kb_idx * BLOCK_SIZE
        k_offs  = k_start + offs_t       # [BS]  absolute key token positions
        k_valid = k_offs < T_K

        # Load K block [BS, D]
        k_ptrs = (K_ptr
                  + bh_idx * stride_kbh
                  + k_offs[:, None] * stride_kt
                  + offs_d[None, :])
        k = tl.load(k_ptrs, mask=k_valid[:, None], other=0.0)

        # Load V block [BS, D]
        v_ptrs = (V_ptr
                  + bh_idx * stride_vbh
                  + k_offs[:, None] * stride_vt
                  + offs_d[None, :])
        v = tl.load(v_ptrs, mask=k_valid[:, None], other=0.0)

        # ── QK GEMM [BS, D] @ [D, BS] → [BS, BS]  (Tensor Core) ─────────
        qk = tl.dot(q, tl.trans(k)) * sm_scale   # [BS, BS], float32

        # ── causal + bounds mask ──────────────────────────────────────────
        causal_ok = q_abs[:, None] >= k_offs[None, :]   # [BS, BS]
        tok_ok    = q_valid[:, None] & k_valid[None, :]  # [BS, BS]
        qk        = tl.where(causal_ok & tok_ok, qk, float("-inf"))

        # ── online softmax update ─────────────────────────────────────────
        m_ij  = tl.max(qk, axis=1)                    # [BS]
        m_new = tl.maximum(m_i, m_ij)                 # [BS]
        alpha = tl.exp(m_i - m_new)                   # [BS]
        p     = tl.exp(qk - m_new[:, None])           # [BS, BS]

        l_i = alpha * l_i + tl.sum(p, axis=1)         # [BS]
        # PV GEMM [BS, BS] @ [BS, D] → [BS, D]  (Tensor Core)
        acc = alpha[:, None] * acc + tl.dot(p.to(q.dtype), v)  # [BS, D], fp32
        m_i = m_new

    # ── normalise ─────────────────────────────────────────────────────────
    # l_i == 0 only for fully-padding rows (OOB tokens) — write zeros safely
    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    out = (acc / l_safe[:, None]).to(q.dtype)   # [BS, D]

    # ── store output ──────────────────────────────────────────────────────
    o_ptrs = (Out_ptr
              + bh_idx * stride_obh
              + q_offs[:, None] * stride_ot
              + offs_d[None, :])
    tl.store(o_ptrs, out, mask=q_valid[:, None])


# ── Python entry point ─────────────────────────────────────────────────────────

def universal_sparse_attn(
    q: torch.Tensor,                      # [B, H, T_q, D]  fp16 or bf16, CUDA, contiguous
    k: torch.Tensor,                      # [B, H, T_k, D]
    v: torch.Tensor,                      # [B, H, T_k, D]
    block_pairs: torch.Tensor,            # [H, max_pairs]  int32, CUDA
    block_pair_offsets: torch.Tensor,     # [H, NQB+1]      int32, CUDA
    block_size: int,                      # power-of-2, >= 16
) -> torch.Tensor:
    """
    Universal block-sparse attention via Triton.

    The block schedule (block_pairs / block_pair_offsets) is in CSR format and
    must have been built with the same block_size.  Any SparsePattern that
    populates PatternMetadata.block_pairs / block_pair_offsets works out of
    the box — BigBirdPattern, LocalWindowPattern, or any custom pattern.

    Returns output tensor of the same shape and dtype as q.
    """
    B, H, T_q, D = q.shape
    T_k = k.size(2)
    BH  = B * H
    NQB = block_pair_offsets.size(1) - 1  # derived from offsets shape

    assert q.is_cuda, "universal_sparse_attn requires CUDA tensors"
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert q.dtype in (torch.float16, torch.bfloat16), \
        f"Expected fp16 or bf16, got {q.dtype}"
    assert block_size >= 16 and (block_size & (block_size - 1)) == 0, \
        "block_size must be a power of 2 and >= 16"
    assert block_pairs.dtype == torch.int32, "block_pairs must be int32"
    assert block_pair_offsets.dtype == torch.int32, "block_pair_offsets must be int32"
    assert block_pairs.size(0) == H, \
        f"block_pairs head dim {block_pairs.size(0)} != H={H}"
    assert block_pair_offsets.size(0) == H, \
        f"block_pair_offsets head dim {block_pair_offsets.size(0)} != H={H}"

    sm_scale = 1.0 / math.sqrt(D)

    # In decode mode the query sits after all T_k cached tokens.
    # In prefill T_q == T_k and offset is 0.
    Q_OFFSET = T_k if T_q < T_k else 0

    # ── Reshape to [B*H, T, D] for simpler kernel indexing ────────────────
    q_  = q.reshape(BH, T_q, D)
    k_  = k.reshape(BH, T_k, D)
    v_  = v.reshape(BH, T_k, D)
    out_ = torch.empty_like(q_)

    # PAIRS and OFFSETS stay at [H, ...] — kernel recovers h_idx = bh_idx % H.
    # This avoids a B-fold expansion on every call.
    pairs_c   = block_pairs.contiguous()
    offsets_c = block_pair_offsets.contiguous()

    grid = (BH, NQB)
    _universal_sparse_attn_kernel[grid](
        q_, k_, v_, out_,
        pairs_c, offsets_c,
        # Q / K / V / Out strides (innermost D-stride = 1, not passed)
        q_.stride(0),   q_.stride(1),
        k_.stride(0),   k_.stride(1),
        v_.stride(0),   v_.stride(1),
        out_.stride(0), out_.stride(1),
        # PAIRS / OFFSETS strides
        pairs_c.stride(0),    # stride_pairs_h   = max_pairs
        offsets_c.stride(0),  # stride_offsets_h = NQB+1
        # scalar params
        H=H, T_Q=T_q, T_K=T_k,
        Q_OFFSET=Q_OFFSET,
        sm_scale=sm_scale,
        # compile-time constants
        BLOCK_SIZE=block_size,
        D_HEAD=D,
    )
    return out_.reshape(B, H, T_q, D)
