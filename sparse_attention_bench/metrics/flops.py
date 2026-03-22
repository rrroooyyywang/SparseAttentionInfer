"""Theoretical FLOPs and arithmetic intensity calculations for attention."""


def attention_flops(
    batch_size: int,
    num_heads: int,
    seq_len_q: int,
    seq_len_k: int,
    head_dim: int,
    keep_ratio: float = 1.0,
) -> dict:
    """
    Compute theoretical FLOPs for one attention layer (QK + softmax + AV).

    Args:
        keep_ratio: fraction of attention scores that are computed/kept (for sparse).

    Returns dict with:
        dense_qk_flops, dense_av_flops, dense_total_flops,
        sparse_qk_flops, sparse_av_flops, sparse_total_flops,
        flop_reduction
    """
    effective_pairs = seq_len_q * seq_len_k * keep_ratio
    dense_qk = 2 * batch_size * num_heads * seq_len_q * seq_len_k * head_dim
    dense_av = 2 * batch_size * num_heads * seq_len_q * seq_len_k * head_dim
    dense_softmax = 5 * batch_size * num_heads * seq_len_q * seq_len_k  # exp + sum + div
    dense_total = dense_qk + dense_av + dense_softmax

    sparse_qk = dense_qk * keep_ratio
    sparse_av = dense_av * keep_ratio
    sparse_softmax = dense_softmax * keep_ratio
    sparse_total = sparse_qk + sparse_av + sparse_softmax

    return {
        "dense_qk_flops": dense_qk,
        "dense_av_flops": dense_av,
        "dense_softmax_flops": dense_softmax,
        "dense_total_flops": dense_total,
        "sparse_qk_flops": sparse_qk,
        "sparse_av_flops": sparse_av,
        "sparse_softmax_flops": sparse_softmax,
        "sparse_total_flops": sparse_total,
        "flop_reduction": 1.0 - sparse_total / dense_total if dense_total > 0 else 0.0,
    }


def arithmetic_intensity(flops: float, bytes_accessed: float) -> float:
    """FLOPs per byte — roofline x-axis."""
    if bytes_accessed <= 0:
        return 0.0
    return flops / bytes_accessed


def attention_bytes(
    batch_size: int,
    num_heads: int,
    seq_len_q: int,
    seq_len_k: int,
    head_dim: int,
    activation_bytes: int = 2,
    score_bytes: int = 4,
    keep_ratio: float = 1.0,
) -> dict:
    """Estimate memory traffic for attention (HBM reads + writes)."""
    q_bytes = batch_size * num_heads * seq_len_q * head_dim * activation_bytes
    k_bytes = batch_size * num_heads * seq_len_k * head_dim * activation_bytes
    v_bytes = k_bytes
    score_bytes_total = batch_size * num_heads * seq_len_q * seq_len_k * score_bytes * keep_ratio
    out_bytes = q_bytes
    total = q_bytes + k_bytes + v_bytes + score_bytes_total + out_bytes
    return {
        "q_bytes": q_bytes,
        "k_bytes": k_bytes,
        "v_bytes": v_bytes,
        "score_bytes": score_bytes_total,
        "output_bytes": out_bytes,
        "total_bytes": total,
    }
