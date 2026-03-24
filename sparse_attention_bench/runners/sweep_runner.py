"""
Sweep runner: load a YAML sweep config, iterate over all combinations,
collect results, and save to CSV/JSON.

Usage:
    python -m sparse_attention_bench.runners.sweep_runner --config sparse_attention_bench/experiment_configs/sweep_seq_len.yaml
"""
import argparse
import csv
import itertools
import json
from pathlib import Path

import torch

from sparse_attention_bench.config import ExperimentConfig
from sparse_attention_bench.paths import OUTPUTS_DIR
from sparse_attention_bench.runners.benchmark_runner import BenchmarkRunner


def _load_yaml(path: str) -> dict:
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required: pip install pyyaml")
    with open(path) as f:
        return yaml.safe_load(f)


def _ensure_list(v):
    return v if isinstance(v, list) else [v]


def build_configs_from_yaml(data: dict) -> list[ExperimentConfig]:
    """
    YAML format (all list values are swept; scalars are fixed):

        experiment:
          batch_size: 1
          num_heads: 8
          head_dim: 64
          seq_len: [128, 512, 2048]
          dtype: fp16
          device: cuda
          mode: [prefill, decode]
          patterns:
            - type: dense
              backend: dense_sdpa
            - type: topk
              backend: masked_sdpa
              topk: [32, 64]
            - type: local_window
              backend: masked_sdpa
              window_size: [64, 128]
          num_warmup: 20
          num_iters: 100
    """
    sweep = data.get("experiment", data)
    batch_sizes = _ensure_list(sweep.get("batch_size", 1))
    num_heads_list = _ensure_list(sweep.get("num_heads", 8))
    head_dims = _ensure_list(sweep.get("head_dim", 64))
    seq_lens = _ensure_list(sweep.get("seq_len", [512]))
    dtypes = _ensure_list(sweep.get("dtype", "fp16"))
    devices = _ensure_list(sweep.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    modes = _ensure_list(sweep.get("mode", "prefill"))
    num_warmup = sweep.get("num_warmup", 20)
    num_iters = sweep.get("num_iters", 100)
    causal = sweep.get("causal", True)

    # Parse patterns block
    patterns_spec = sweep.get("patterns", [{"type": "dense", "backend": "dense_sdpa"}])
    pattern_cfgs = []
    for p in patterns_spec:
        p_type = p["type"]
        backend = p.get("backend", "masked_sdpa")
        topks = _ensure_list(p.get("topk", [None])) if "topk" in p else [None]
        window_sizes = _ensure_list(p.get("window_size", [None])) if "window_size" in p else [None]
        block_sizes = _ensure_list(p.get("block_size", [None])) if "block_size" in p else [None]
        for topk, ws, bs in itertools.product(topks, window_sizes, block_sizes):
            pattern_cfgs.append({
                "pattern_type": p_type,
                "backend": backend,
                "topk": topk,
                "window_size": ws,
                "block_size": bs,
            })

    configs = []
    for (batch, heads, hdim, seq, dtype, device, mode, pat) in itertools.product(
        batch_sizes, num_heads_list, head_dims, seq_lens, dtypes, devices, modes, pattern_cfgs
    ):
        configs.append(ExperimentConfig(
            batch_size=batch,
            num_heads=heads,
            head_dim=hdim,
            seq_len=seq,
            dtype=dtype,
            device=device,
            mode=mode,
            pattern_type=pat["pattern_type"],
            backend=pat["backend"],
            causal=causal,
            topk=pat["topk"],
            window_size=pat["window_size"],
            block_size=pat["block_size"],
            num_warmup=num_warmup,
            num_iters=num_iters,
        ))
    return configs


def run_sweep(configs: list[ExperimentConfig], verbose: bool = True) -> list[dict]:
    runner = BenchmarkRunner()
    results = []
    for i, cfg in enumerate(configs):
        tag = (
            f"[{i+1}/{len(configs)}] "
            f"{cfg.pattern_type}/{cfg.backend} "
            f"seq={cfg.seq_len} mode={cfg.mode} dtype={cfg.dtype}"
        )
        if verbose:
            print(tag, flush=True)
        try:
            result = runner.run(cfg)
            result["_status"] = "ok"
        except Exception as e:
            result = {"config": cfg.as_dict(), "_status": "error", "_error": str(e)}
            if verbose:
                print(f"  ERROR: {e}")
        results.append(result)
        if verbose:
            if result["_status"] == "ok":
                print(
                    f"  total={result['total_time_ms_mean']:.2f}ms "
                    f"(build={result['pattern_build_time_ms_mean']:.2f}ms "
                    f"attn={result['attention_time_ms_mean']:.2f}ms) "
                    f"actual_backend={result['actual_backend']} "
                    f"rel_err={result['rel_err']:.4f} "
                    f"keep={result['keep_ratio']:.2%}"
                )
    return results


def save_results(results: list[dict], output_dir: Path, tag: str = "sweep"):
    """Save results to output_dir/json/<tag>.json and output_dir/csv/<tag>.csv."""
    json_dir = output_dir / "json"
    csv_dir = output_dir / "csv"
    json_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    json_path = json_dir / f"{tag}.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"Saved JSON: {json_path}")

    # Flatten for CSV (config dict merged into top-level)
    rows = []
    for r in results:
        row = {}
        cfg = r.get("config", {})
        row.update({f"cfg_{k}": v for k, v in cfg.items()})
        row.update({k: v for k, v in r.items() if k != "config"})
        rows.append(row)

    if rows:
        csv_path = csv_dir / f"{tag}.csv"
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved CSV:  {csv_path}")


def plot_sweep_results(results: list[dict], out_dir: Path, tag: str) -> None:
    """Generate latency plots from sweep results and save to out_dir/figures/."""
    from sparse_attention_bench.analytical.plotting import plot_sweep_latency
    import matplotlib.pyplot as plt

    ok_results = [r for r in results if r.get("_status") == "ok"]
    if not ok_results:
        print("No successful results to plot.")
        return

    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("total_time_ms_mean", "Total Latency (ms)"),
        ("attention_time_ms_mean", "Attention Latency (ms)"),
        ("pattern_build_time_ms_mean", "Pattern Build Time (ms)"),
        ("peak_memory_mb", "Peak Memory (MB)"),
        ("rel_err", "Relative Error vs Dense"),
        ("cosine_sim", "Cosine Similarity vs Dense"),
        ("kl_divergence", "KL Divergence vs Dense"),
        ("keep_ratio", "Keep Ratio"),
    ]
    # Flatten config dict into top-level for the plotter
    flat_results = []
    for r in ok_results:
        row = dict(r.get("config", {}))
        row.update({k: v for k, v in r.items() if k != "config"})
        flat_results.append(row)

    for y_key, y_label in metrics:
        if not any(y_key in r for r in flat_results):
            continue
        fig = plot_sweep_latency(flat_results, x_key="seq_len", y_key=y_key)
        fig.axes[0].set_ylabel(y_label)
        fig_path = figures_dir / f"{tag}_{y_key}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot: {fig_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep runner for attention benchmarks.")
    parser.add_argument("--config", type=str, required=True, help="YAML sweep config.")
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save results. Default: outputs/ at project root.",
    )
    parser.add_argument("--tag", type=str, default="sweep", help="Output filename prefix.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--plot", action="store_true", help="Generate latency plots after sweep.")
    return parser.parse_args()


def main():
    args = parse_args()
    data = _load_yaml(args.config)
    configs = build_configs_from_yaml(data)
    print(f"Loaded {len(configs)} experiment configs from {args.config}")

    results = run_sweep(configs, verbose=not args.quiet)

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = OUTPUTS_DIR

    save_results(results, out_dir, tag=args.tag)

    if args.plot:
        plot_sweep_results(results, out_dir, tag=args.tag)


if __name__ == "__main__":
    main()
