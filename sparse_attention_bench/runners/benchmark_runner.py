"""Single-case benchmark runner: pattern build + attention compute + metrics."""
from __future__ import annotations

import torch

from sparse_attention_bench.attention import get_backend
from sparse_attention_bench.config import ExperimentConfig
from sparse_attention_bench.metrics.accuracy import (
    cosine_sim, mean_kl_divergence, relative_error,
)
from sparse_attention_bench.metrics.latency import measure_latency
from sparse_attention_bench.metrics.memory import measure_peak_memory_mb
from sparse_attention_bench.patterns.base import PatternMetadata


def _get_pattern(cfg: ExperimentConfig):
    """Factory: return the right SparsePattern instance for the config."""
    from sparse_attention_bench.patterns.causal_dense import DenseCausalPattern
    from sparse_attention_bench.patterns.topk_pattern import TopKPattern
    from sparse_attention_bench.patterns.bigbird_pattern import BigBirdPattern
    from sparse_attention_bench.patterns.local_window import LocalWindowPattern

    if cfg.pattern_type == "dense":
        return DenseCausalPattern()
    if cfg.pattern_type == "topk":
        assert cfg.topk is not None, "topk must be set for pattern_type='topk'"
        return TopKPattern(top_k=cfg.topk)
    if cfg.pattern_type == "bigbird":
        assert cfg.topk is not None, "topk must be set for pattern_type='bigbird'"
        return BigBirdPattern(top_k=cfg.topk, n_heads=cfg.num_heads)
    if cfg.pattern_type == "local_window":
        assert cfg.window_size is not None, "window_size must be set for pattern_type='local_window'"
        return LocalWindowPattern(window_size=cfg.window_size)
    raise ValueError(f"Unknown pattern_type: {cfg.pattern_type!r}")


class BenchmarkRunner:
    """
    Run one ExperimentConfig and return a result dict with:
    - latency stats (pattern build + attention compute, separately and combined)
    - peak memory
    - accuracy metrics vs dense baseline
    - keep_ratio and sparsity_ratio
    """

    def run(self, cfg: ExperimentConfig) -> dict:
        q, k, v = cfg.make_qkv()

        pattern_obj = _get_pattern(cfg)
        backend = get_backend(cfg.backend)
        dense_backend = get_backend("dense_sdpa")

        # Dense reference (no_grad, for accuracy comparison only)
        with torch.no_grad():
            dense_out = dense_backend.forward(q, k, v, PatternMetadata(kind="dense"))

        # ── Pattern build timing ───────────────────────────────────────────────
        pattern_latency = measure_latency(
            fn=lambda: pattern_obj.build(q, k, causal=cfg.causal),
            num_iters=cfg.num_iters,
            num_warmup=cfg.num_warmup,
            device=cfg.device,
        )

        # Build pattern once for attention timing
        with torch.no_grad():
            pattern = pattern_obj.build(q, k, causal=cfg.causal)

        # ── Attention compute timing ───────────────────────────────────────────
        attn_latency = measure_latency(
            fn=lambda: backend.forward(q, k, v, pattern),
            num_iters=cfg.num_iters,
            num_warmup=cfg.num_warmup,
            device=cfg.device,
        )

        # ── Combined timing (build + compute) ─────────────────────────────────
        def _combined():
            p = pattern_obj.build(q, k, causal=cfg.causal)
            backend.forward(q, k, v, p)

        total_latency = measure_latency(
            fn=_combined,
            num_iters=cfg.num_iters,
            num_warmup=cfg.num_warmup,
            device=cfg.device,
        )

        # ── Memory ────────────────────────────────────────────────────────────
        def _full_run():
            p = pattern_obj.build(q, k, causal=cfg.causal)
            return backend.forward(q, k, v, p)

        peak_mem_mb = measure_peak_memory_mb(_full_run, device=cfg.device)

        # ── Accuracy ──────────────────────────────────────────────────────────
        with torch.no_grad():
            sparse_out = backend.forward(q, k, v, pattern)

        # Promote to float32 for metric computation
        d_f32 = dense_out.float()
        s_f32 = sparse_out.float()

        keep_ratio = pattern.keep_ratio

        # Classify whether the backend is a proxy (dense masked) or a real sparse kernel
        _PROXY_BACKENDS = {"masked_sdpa", "gather_sparse"}
        timing_source = "proxy" if cfg.backend in _PROXY_BACKENDS else "real_cuda"
        actual_backend = backend.actual_backend

        return {
            "config": cfg.as_dict(),
            "timing_source": timing_source,
            "actual_backend": actual_backend,
            # Pattern build
            "pattern_build_time_ms_mean": pattern_latency["mean_ms"],
            "pattern_build_time_ms_p50": pattern_latency["p50_ms"],
            "pattern_build_time_ms_p95": pattern_latency["p95_ms"],
            # Attention compute
            "attention_time_ms_mean": attn_latency["mean_ms"],
            "attention_time_ms_p50": attn_latency["p50_ms"],
            "attention_time_ms_p95": attn_latency["p95_ms"],
            # Total
            "total_time_ms_mean": total_latency["mean_ms"],
            "total_time_ms_p50": total_latency["p50_ms"],
            "total_time_ms_p95": total_latency["p95_ms"],
            # Memory
            "peak_memory_mb": peak_mem_mb,
            # Accuracy
            "rel_err": relative_error(s_f32, d_f32),
            "cosine_sim": cosine_sim(s_f32, d_f32),
            "kl_divergence": mean_kl_divergence(d_f32, s_f32),
            "keep_ratio": keep_ratio,
            "sparsity_ratio": 1.0 - keep_ratio,
        }
