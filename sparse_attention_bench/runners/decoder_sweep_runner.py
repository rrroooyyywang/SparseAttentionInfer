"""
End-to-end decoder sweep runner.

Runs a full ToyDecoder (embedding + N×DecoderBlock + LM head) for every config
in a YAML sweep, producing flat result rows suitable for the shared plotter.

Unlike the layer-level sweep_runner, accuracy metrics here are computed on
LM-head logits [B, T, vocab_size], so top1_match_rate is meaningful.

Both sparse and dense decoders share the same randomly-initialised weights
(copied via state_dict) so accuracy differences reflect only the attention
sparsity, not weight randomness.

Usage:
    python -m sparse_attention_bench.runners.decoder_sweep_runner \\
        --config sparse_attention_bench/experiment_configs/sweep_seq_len_base_vs_bigbird.yaml \\
        --tag e2e_bigbird --plot
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from sparse_attention_bench.attention import get_backend
from sparse_attention_bench.config import ExperimentConfig
from sparse_attention_bench.metrics.accuracy import (
    cosine_sim, mean_kl_divergence, relative_error, top1_match_rate,
)
from sparse_attention_bench.metrics.latency import measure_latency
from sparse_attention_bench.metrics.memory import measure_peak_memory_mb
from sparse_attention_bench.models.decoder_block import ToyDecoder
from sparse_attention_bench.models.kv_cache import KVCache
from sparse_attention_bench.paths import OUTPUTS_DIR
from sparse_attention_bench.patterns.causal_dense import DenseCausalPattern
from sparse_attention_bench.runners.benchmark_runner import _get_pattern
from sparse_attention_bench.runners.sweep_runner import build_configs_from_yaml, _load_yaml


# ── Model helpers ──────────────────────────────────────────────────────────────

def _build_toy_decoder(
    cfg: ExperimentConfig,
    num_layers: int,
    vocab_size: int,
    d_model: int,
    pattern,
    backend_name: str,
) -> ToyDecoder:
    backend = get_backend(backend_name)
    model = ToyDecoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=cfg.num_heads,
        num_layers=num_layers,
        pattern=pattern,
        backend=backend,
        max_seq_len=max(cfg.seq_len + 1, 4096),
        causal=cfg.causal,
    )
    return model.to(device=cfg.device, dtype=cfg.torch_dtype()).eval()


# ── Single-config runner ───────────────────────────────────────────────────────

def run_one(
    cfg: ExperimentConfig,
    num_layers: int = 2,
    vocab_size: int = 5000,
    d_model: int = 256,
) -> dict:
    """
    Run one config end-to-end and return a flat result dict.

    Dense and sparse decoders share weights (state_dict copy) so accuracy
    metrics reflect only the attention sparsity.

    Prefill : input_ids [B, seq_len] → logits [B, seq_len, vocab_size]
    Decode  : prefill KV cache with seq_len tokens, then single-token step
              → logits [B, 1, vocab_size]
    """
    device = torch.device(cfg.device)

    # ── Build dense decoder (reference) ───────────────────────────────────────
    dense_dec = _build_toy_decoder(
        cfg, num_layers, vocab_size, d_model,
        pattern=DenseCausalPattern(),
        backend_name="dense_sdpa",
    )

    # ── Build sparse decoder, copy weights from dense ──────────────────────────
    sparse_pattern = _get_pattern(cfg)
    sparse_dec = _build_toy_decoder(
        cfg, num_layers, vocab_size, d_model,
        pattern=sparse_pattern,
        backend_name=cfg.backend,
    )
    sparse_dec.load_state_dict(dense_dec.state_dict())
    sparse_dec.eval()

    _PROXY_BACKENDS = {"masked_sdpa", "gather_sparse"}
    timing_source = "proxy" if cfg.backend in _PROXY_BACKENDS else "real_cuda"

    if cfg.mode == "prefill":
        input_ids = torch.randint(0, vocab_size, (cfg.batch_size, cfg.seq_len), device=device)

        with torch.no_grad():
            dense_logits = dense_dec(input_ids)    # [B, seq_len, vocab_size]
            sparse_logits = sparse_dec(input_ids)  # [B, seq_len, vocab_size]

        # Read after first forward so _last_actual_backend is populated
        actual_backend = sparse_dec.blocks[0].attn.backend.actual_backend

        latency = measure_latency(
            fn=lambda: sparse_dec(input_ids),
            num_iters=cfg.num_iters,
            num_warmup=cfg.num_warmup,
            device=cfg.device,
        )
        peak_mem = measure_peak_memory_mb(lambda: sparse_dec(input_ids), device=cfg.device)

        d_f32 = dense_logits.float()
        s_f32 = sparse_logits.float()

        return {
            "config": cfg.as_dict(),
            "timing_source": timing_source,
            "actual_backend": actual_backend,
            "total_time_ms_mean": latency["mean_ms"],
            "total_time_ms_p50": latency["p50_ms"],
            "total_time_ms_p95": latency["p95_ms"],
            "peak_memory_mb": peak_mem,
            "rel_err": relative_error(s_f32, d_f32),
            "cosine_sim": cosine_sim(s_f32, d_f32),
            "kl_divergence": mean_kl_divergence(d_f32, s_f32),
            "top1_match_rate": top1_match_rate(s_f32, d_f32),
        }

    else:  # decode
        kv_len = cfg.seq_len

        # Pre-fill KV caches with identical inputs so the cached state is the same
        prefill_ids = torch.randint(0, vocab_size, (cfg.batch_size, kv_len), device=device)
        dense_kv  = KVCache(num_layers=num_layers)
        sparse_kv = KVCache(num_layers=num_layers)
        with torch.no_grad():
            dense_dec(prefill_ids,  kv_cache=dense_kv)
            sparse_dec(prefill_ids, kv_cache=sparse_kv)

        new_token = torch.randint(0, vocab_size, (cfg.batch_size, 1), device=device)
        pos_offset = kv_len

        with torch.no_grad():
            dense_logits  = dense_dec(new_token,  pos_offset=pos_offset)  # [B, 1, vocab_size]
            sparse_logits = sparse_dec(new_token, pos_offset=pos_offset)  # [B, 1, vocab_size]

        # Read after first forward so _last_actual_backend is populated
        actual_backend = sparse_dec.blocks[0].attn.backend.actual_backend

        latency = measure_latency(
            fn=lambda: sparse_dec(new_token, pos_offset=pos_offset),
            num_iters=cfg.num_iters,
            num_warmup=cfg.num_warmup,
            device=cfg.device,
        )
        peak_mem = measure_peak_memory_mb(
            lambda: sparse_dec(new_token, pos_offset=pos_offset), device=cfg.device
        )

        d_f32 = dense_logits.float()
        s_f32 = sparse_logits.float()

        return {
            "config": cfg.as_dict(),
            "timing_source": timing_source,
            "actual_backend": actual_backend,
            "total_time_ms_mean": latency["mean_ms"],
            "total_time_ms_p50": latency["p50_ms"],
            "total_time_ms_p95": latency["p95_ms"],
            "peak_memory_mb": peak_mem,
            "rel_err": relative_error(s_f32, d_f32),
            "cosine_sim": cosine_sim(s_f32, d_f32),
            "kl_divergence": mean_kl_divergence(d_f32, s_f32),
            "top1_match_rate": top1_match_rate(s_f32, d_f32),
        }


# ── Sweep orchestration ────────────────────────────────────────────────────────

def run_decoder_sweep(
    configs: list[ExperimentConfig],
    num_layers: int = 2,
    vocab_size: int = 5000,
    d_model: int = 256,
    verbose: bool = True,
) -> list[dict]:
    results = []
    for i, cfg in enumerate(configs):
        tag = (
            f"[{i+1}/{len(configs)}] "
            f"{cfg.pattern_type}/{cfg.backend} "
            f"seq={cfg.seq_len} mode={cfg.mode}"
        )
        if verbose:
            print(tag, flush=True)
        try:
            result = run_one(cfg, num_layers=num_layers, vocab_size=vocab_size, d_model=d_model)
            result["_status"] = "ok"
        except Exception as e:
            result = {"config": cfg.as_dict(), "_status": "error", "_error": str(e)}
            if verbose:
                print(f"  ERROR: {e}")
        results.append(result)
        if verbose and result["_status"] == "ok":
            print(
                f"  total={result['total_time_ms_mean']:.2f}ms  "
                f"actual_backend={result['actual_backend']}  "
                f"top1={result['top1_match_rate']:.2%}  "
                f"kl={result['kl_divergence']:.4f}  "
                f"rel_err={result['rel_err']:.4f}"
            )
    return results


# ── Save + plot ────────────────────────────────────────────────────────────────

def save_results(results: list[dict], output_dir: Path, tag: str) -> None:
    json_dir = output_dir / "json"
    csv_dir  = output_dir / "csv"
    json_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    json_path = json_dir / f"{tag}.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"Saved JSON: {json_path}")

    rows = []
    for r in results:
        row = dict(r.get("config", {}))
        row.update({k: v for k, v in r.items() if k != "config"})
        rows.append(row)

    if rows:
        csv_path = csv_dir / f"{tag}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved CSV:  {csv_path}")


def plot_results(results: list[dict], output_dir: Path, tag: str) -> None:
    from sparse_attention_bench.analytical.plotting import plot_sweep_latency
    import matplotlib.pyplot as plt

    ok_results = [r for r in results if r.get("_status") == "ok"]
    if not ok_results:
        print("No successful results to plot.")
        return

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    flat = []
    for r in ok_results:
        row = dict(r.get("config", {}))
        row.update({k: v for k, v in r.items() if k != "config"})
        flat.append(row)

    metrics = [
        ("total_time_ms_mean",  "Total Latency (ms)"),
        ("peak_memory_mb",      "Peak Memory (MB)"),
        ("top1_match_rate",     "Top-1 Match Rate vs Dense"),
        ("kl_divergence",       "KL Divergence vs Dense"),
        ("rel_err",             "Relative Error vs Dense"),
        ("cosine_sim",          "Cosine Similarity vs Dense"),
    ]

    for y_key, y_label in metrics:
        if not any(y_key in r for r in flat):
            continue
        fig = plot_sweep_latency(flat, x_key="seq_len", y_key=y_key)
        for ax in fig.axes:
            ax.set_ylabel(y_label)
        fig_path = figures_dir / f"{tag}_{y_key}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot: {fig_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="End-to-end decoder sweep benchmark.")
    parser.add_argument("--config", type=str, required=True, help="YAML sweep config.")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--tag", type=str, default="decoder_sweep")
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=5000)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    data = _load_yaml(args.config)
    configs = build_configs_from_yaml(data)
    print(f"Loaded {len(configs)} configs from {args.config}")

    results = run_decoder_sweep(
        configs,
        num_layers=args.num_layers,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        verbose=not args.quiet,
    )

    out_dir = Path(args.output_dir) if args.output_dir else OUTPUTS_DIR
    save_results(results, out_dir, tag=args.tag)

    if args.plot:
        plot_results(results, out_dir, tag=args.tag)


if __name__ == "__main__":
    main()
