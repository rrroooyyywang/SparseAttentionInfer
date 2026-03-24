"""GPU efficiency estimator (roofline proxy)."""
import math

from sparse_attention_bench.analytical.config import DecoderConfig, NvidiaGpuHeuristic
from sparse_attention_bench.analytical.gpu_profiles import validate_execution_phase
from sparse_attention_bench.analytical.roofline import (
    adjust_cacheable_bytes_for_l2,
    roofline_op_time_us,
)
from sparse_attentions.patterns.bigbird_pattern import (
    estimate_bigbird_attention_keep_ratio,
    estimate_bigbird_decode_keep_ratio,
)
from sparse_attentions.patterns.topk_pattern import (
    causal_token_pairs,
    effective_topk_attention_keep_ratio,
    decode_topk_attention_keep_ratio,
    feature_keep_from_attention_keep,
)


def estimate_decoder_sparse_gpu_efficiency(
    cfg: DecoderConfig,
    batch_size: int,
    seq_len: int,
    top_k: int,
    sparse_mode: str = "top-k",
    gpu: NvidiaGpuHeuristic | None = None,
    phase: str = "prefill",
) -> dict:
    gpu = gpu or NvidiaGpuHeuristic()
    phase = validate_execution_phase(phase)

    keep_ratio = min(top_k, seq_len) / seq_len
    d_model = cfg.d_model
    hidden = cfg.d_model * cfg.mlp_ratio
    n_heads = cfg.n_heads
    head_dim = cfg.d_model // cfg.n_heads
    projected_tokens = batch_size * seq_len if phase == "prefill" else batch_size
    activation_elems = projected_tokens * d_model
    kv_cache_elems = batch_size * seq_len * d_model
    output_elems = projected_tokens * d_model
    query_elems = activation_elems
    query_bytes = query_elems * gpu.activation_bytes
    kv_cache_bytes = kv_cache_elems * gpu.activation_bytes
    output_bytes = output_elems * gpu.activation_bytes

    if phase == "prefill":
        dense_score_elems = batch_size * n_heads * causal_token_pairs(seq_len)
    else:
        dense_score_elems = batch_size * n_heads * seq_len

    # Dense FLOPs
    dense_qk_cost = 2.0 * dense_score_elems * head_dim
    dense_av_cost = dense_qk_cost
    dense_softmax_cost = gpu.softmax_flops_per_element * dense_score_elems
    dense_proj_cost = 8.0 * projected_tokens * d_model * d_model
    dense_mlp_cost = 4.0 * projected_tokens * d_model * hidden

    proj_linear_bytes = (
        activation_elems * gpu.activation_bytes
        + d_model * d_model * gpu.weight_bytes
        + activation_elems * gpu.activation_bytes
    )
    dense_proj_bytes = 4.0 * proj_linear_bytes
    dense_mlp_bytes = (
        activation_elems * gpu.activation_bytes
        + d_model * hidden * gpu.weight_bytes
        + projected_tokens * hidden * gpu.activation_bytes
        + projected_tokens * hidden * gpu.activation_bytes
        + hidden * d_model * gpu.weight_bytes
        + output_elems * gpu.activation_bytes
    )
    effective_dense_k_bytes, dense_qk_l2_hit_rate = adjust_cacheable_bytes_for_l2(
        kv_cache_bytes, gpu, phase
    )
    effective_dense_v_bytes, dense_av_l2_hit_rate = adjust_cacheable_bytes_for_l2(
        kv_cache_bytes, gpu, phase
    )
    dense_qk_bytes = query_bytes + effective_dense_k_bytes + dense_score_elems * gpu.score_bytes
    dense_softmax_bytes = 2.0 * dense_score_elems * gpu.score_bytes
    dense_av_bytes = dense_score_elems * gpu.score_bytes + effective_dense_v_bytes + output_bytes

    def _proj_block_time() -> float:
        return 4.0 * roofline_op_time_us(
            flops=2.0 * projected_tokens * d_model * d_model,
            num_bytes=proj_linear_bytes,
            compute_tflops=gpu.dense_tensor_tflops,
            gpu=gpu,
        )

    def _mlp_time() -> float:
        return (
            roofline_op_time_us(
                flops=2.0 * projected_tokens * d_model * hidden,
                num_bytes=(
                    activation_elems * gpu.activation_bytes
                    + d_model * hidden * gpu.weight_bytes
                    + projected_tokens * hidden * gpu.activation_bytes
                ),
                compute_tflops=gpu.dense_tensor_tflops, gpu=gpu,
            )
            + roofline_op_time_us(
                flops=2.0 * projected_tokens * hidden * d_model,
                num_bytes=(
                    projected_tokens * hidden * gpu.activation_bytes
                    + hidden * d_model * gpu.weight_bytes
                    + output_elems * gpu.activation_bytes
                ),
                compute_tflops=gpu.dense_tensor_tflops, gpu=gpu,
            )
        )

    dense_total_time_us = cfg.n_layers * (
        _proj_block_time()
        + _mlp_time()
        + roofline_op_time_us(dense_qk_cost, dense_qk_bytes, gpu.dense_tensor_tflops, gpu)
        + roofline_op_time_us(dense_softmax_cost, dense_softmax_bytes, gpu.vector_tflops, gpu)
        + roofline_op_time_us(dense_av_cost, dense_av_bytes, gpu.dense_tensor_tflops, gpu)
    )
    dense_total_cost = cfg.n_layers * (
        dense_proj_cost + dense_mlp_cost + dense_qk_cost + dense_softmax_cost + dense_av_cost
    )

    sparse_topk_bytes = 0.0
    selection_cost = 0.0
    selection_time_us = 0.0

    if sparse_mode == "top-k":
        feature_keep_ratio = feature_keep_from_attention_keep(head_dim, keep_ratio) / head_dim
        if phase == "prefill":
            attention_keep_ratio = effective_topk_attention_keep_ratio(seq_len, min(top_k, seq_len))
        else:
            attention_keep_ratio = decode_topk_attention_keep_ratio(seq_len, min(top_k, seq_len))
        sparse_qk_cost = dense_qk_cost * (feature_keep_ratio ** 2)
        sparse_av_cost = dense_av_cost * (attention_keep_ratio * feature_keep_ratio)
        sparse_softmax_cost = dense_softmax_cost * attention_keep_ratio

        sparse_query_bytes = query_elems * feature_keep_ratio * (gpu.activation_bytes + gpu.index_bytes)
        sparse_kv_cache_bytes = kv_cache_elems * feature_keep_ratio * (gpu.activation_bytes + gpu.index_bytes)
        sparse_score_elems = dense_score_elems * attention_keep_ratio
        effective_sparse_k_bytes, sparse_qk_l2_hit_rate = adjust_cacheable_bytes_for_l2(
            sparse_kv_cache_bytes, gpu, phase
        )
        effective_sparse_v_bytes, sparse_av_l2_hit_rate = adjust_cacheable_bytes_for_l2(
            sparse_kv_cache_bytes, gpu, phase
        )
        sparse_qk_bytes = (
            sparse_query_bytes
            + effective_sparse_k_bytes
            + sparse_score_elems * (gpu.score_bytes + gpu.index_bytes)
        )
        sparse_topk_bytes = (
            dense_score_elems * gpu.score_bytes
            + sparse_score_elems * (gpu.score_bytes + gpu.index_bytes)
        )
        sparse_softmax_bytes = 2.0 * sparse_score_elems * gpu.score_bytes
        sparse_av_bytes = (
            sparse_score_elems * (gpu.score_bytes + gpu.index_bytes)
            + effective_sparse_v_bytes
            + output_bytes
        )
        selection_cost = (
            dense_score_elems
            * gpu.topk_compare_flops_per_element
            * max(1.0, math.log2(max(2, seq_len)))
        )
        selection_time_us = roofline_op_time_us(selection_cost, sparse_topk_bytes, gpu.vector_tflops, gpu)
        sparse_attention_time_us = (
            roofline_op_time_us(sparse_qk_cost, sparse_qk_bytes, gpu.dynamic_sparse_tflops, gpu)
            + selection_time_us
            + roofline_op_time_us(sparse_softmax_cost, sparse_softmax_bytes, gpu.vector_tflops, gpu)
            + roofline_op_time_us(sparse_av_cost, sparse_av_bytes, gpu.dynamic_sparse_tflops, gpu)
        )
    elif sparse_mode == "bigbird":
        feature_keep_ratio = 1.0
        if phase == "prefill":
            attention_keep_ratio = estimate_bigbird_attention_keep_ratio(seq_len, min(top_k, seq_len))
        else:
            attention_keep_ratio = estimate_bigbird_decode_keep_ratio(seq_len, min(top_k, seq_len))
        sparse_qk_cost = dense_qk_cost * attention_keep_ratio
        sparse_av_cost = dense_av_cost * attention_keep_ratio
        sparse_softmax_cost = dense_softmax_cost * attention_keep_ratio
        sparse_score_elems = dense_score_elems * attention_keep_ratio
        effective_bigbird_k_bytes, sparse_qk_l2_hit_rate = adjust_cacheable_bytes_for_l2(
            kv_cache_bytes, gpu, phase
        )
        effective_bigbird_v_bytes, sparse_av_l2_hit_rate = adjust_cacheable_bytes_for_l2(
            kv_cache_bytes, gpu, phase
        )
        sparse_qk_bytes = query_bytes + effective_bigbird_k_bytes + sparse_score_elems * gpu.score_bytes
        sparse_softmax_bytes = 2.0 * sparse_score_elems * gpu.score_bytes
        sparse_av_bytes = sparse_score_elems * gpu.score_bytes + effective_bigbird_v_bytes + output_bytes
        sparse_attention_time_us = (
            roofline_op_time_us(sparse_qk_cost, sparse_qk_bytes, gpu.block_sparse_tensor_tflops, gpu)
            + roofline_op_time_us(sparse_softmax_cost, sparse_softmax_bytes, gpu.vector_tflops, gpu)
            + roofline_op_time_us(sparse_av_cost, sparse_av_bytes, gpu.block_sparse_tensor_tflops, gpu)
        )
    else:
        raise ValueError(f"Unsupported sparse mode: {sparse_mode}")

    sparse_total_cost = cfg.n_layers * (
        dense_proj_cost + dense_mlp_cost + sparse_qk_cost + sparse_softmax_cost + sparse_av_cost + selection_cost
    )
    sparse_total_time_us = cfg.n_layers * (
        _proj_block_time()
        + _mlp_time()
        + sparse_attention_time_us
    )
    sparse_total_bytes = cfg.n_layers * (
        dense_proj_bytes + dense_mlp_bytes + sparse_qk_bytes + sparse_softmax_bytes + sparse_av_bytes + sparse_topk_bytes
    )

    return {
        "phase": phase,
        "sparse_mode": sparse_mode,
        "keep_ratio": keep_ratio,
        "attention_keep_ratio": attention_keep_ratio,
        "feature_keep_ratio": feature_keep_ratio,
        "dense_attention_l2_hit_rate": 0.5 * (dense_qk_l2_hit_rate + dense_av_l2_hit_rate),
        "sparse_attention_l2_hit_rate": 0.5 * (sparse_qk_l2_hit_rate + sparse_av_l2_hit_rate),
        "attention_workload_reduction": 1.0 - (
            (sparse_qk_cost + sparse_softmax_cost + sparse_av_cost + selection_cost)
            / (dense_qk_cost + dense_softmax_cost + dense_av_cost)
        ),
        "runtime_proxy_reduction": 1.0 - sparse_total_time_us / dense_total_time_us,
        "gpu_speedup_est": dense_total_time_us / sparse_total_time_us,
        "dense_time_us": dense_total_time_us,
        "sparse_time_us": sparse_total_time_us,
        "selection_time_us": cfg.n_layers * selection_time_us,
        "dense_total_flops": dense_total_cost,
        "sparse_total_flops": sparse_total_cost,
        "sparse_total_bytes": sparse_total_bytes,
        "dense_total_bytes": cfg.n_layers * (
            dense_proj_bytes + dense_mlp_bytes + dense_qk_bytes + dense_softmax_bytes + dense_av_bytes
        ),
        "sparse_attention_time_us": cfg.n_layers * sparse_attention_time_us,
    }
