"""
Single attention layer benchmark.

Usage:
    python -m sparse_attention_bench.benchmarks.bench_layer \\
        --pattern topk --topk 32 --seq-len 512 --num-heads 8 --head-dim 64

    python -m sparse_attention_bench.benchmarks.bench_layer --config sparse_attention_bench/experiment_configs/topk.yaml

Outputs a JSON result to stdout and optionally to --output-dir.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

from sparse_attention_bench.config import ExperimentConfig
from sparse_attention_bench.runners.benchmark_runner import BenchmarkRunner


def parse_args():
    parser = argparse.ArgumentParser(description="Layer-level attention benchmark with real CUDA timing.")
    # Config file (overrides everything if provided)
    parser.add_argument("--config", type=str, default=None, help="YAML config file path.")
    # Direct args
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mode", type=str, default="prefill", choices=["prefill", "decode"])
    parser.add_argument(
        "--pattern", type=str, default="dense",
        choices=["dense", "topk", "bigbird", "local_window"],
    )
    parser.add_argument(
        "--backend", type=str, default="dense_sdpa",
        choices=["dense_sdpa", "masked_sdpa", "gather_sparse"],
    )
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--num-warmup", type=int, default=20)
    parser.add_argument("--num-iters", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-print", action="store_true", help="Suppress JSON output to stdout.")
    return parser.parse_args()


def _load_yaml_config(path: str) -> dict:
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required to load config files: pip install pyyaml")
    with open(path) as f:
        return yaml.safe_load(f)


def _cfg_from_args(args) -> ExperimentConfig:
    if args.config:
        data = _load_yaml_config(args.config)
        cfg_data = data.get("experiment", data)
        return ExperimentConfig(
            batch_size=cfg_data.get("batch_size", 1),
            num_heads=cfg_data.get("num_heads", 8),
            head_dim=cfg_data.get("head_dim", 64),
            seq_len=cfg_data.get("seq_len", 512),
            dtype=cfg_data.get("dtype", "fp16"),
            device=cfg_data.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
            mode=cfg_data.get("mode", "prefill"),
            pattern_type=cfg_data.get("pattern_type", "dense"),
            backend=cfg_data.get("backend", "dense_sdpa"),
            causal=cfg_data.get("causal", True),
            topk=cfg_data.get("topk", None),
            window_size=cfg_data.get("window_size", None),
            block_size=cfg_data.get("block_size", None),
            num_warmup=cfg_data.get("num_warmup", 20),
            num_iters=cfg_data.get("num_iters", 100),
        )
    return ExperimentConfig(
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        seq_len=args.seq_len,
        dtype=args.dtype,
        device=args.device,
        mode=args.mode,
        pattern_type=args.pattern,
        backend=args.backend,
        topk=args.topk,
        window_size=args.window_size,
        num_warmup=args.num_warmup,
        num_iters=args.num_iters,
    )


def main():
    args = parse_args()
    cfg = _cfg_from_args(args)
    runner = BenchmarkRunner()
    result = runner.run(cfg)

    if not args.no_print:
        print(json.dumps(result, indent=2))

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = (
            f"{cfg.pattern_type}_{cfg.backend}_"
            f"seq{cfg.seq_len}_{cfg.mode}_{cfg.dtype}"
        )
        out_path = out_dir / f"bench_layer_{tag}.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"Saved to: {out_path}", file=sys.stderr)

    return result


if __name__ == "__main__":
    main()
