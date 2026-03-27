from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from sparse_llm.common.benchmark.perplexity import (
    benchmark_runtime as benchmark_perplexity_runtime,
)
from sparse_llm.common.benchmark.utils import cleanup_cuda_state
from sparse_llm.common.io.json_io import write_json
from sparse_llm.qwen3.adapter import get_qwen3_adapter
from sparse_llm.qwen3.benchmark.generation import (
    DEFAULT_DECODE_STEPS,
    DEFAULT_TEMPERATURE,
    DEFAULT_WARMUP_ITERS,
    benchmark_runtime as benchmark_generation_runtime,
)
from sparse_llm.qwen3.integrations.runtime import _build_hf_pretrained_kwargs


EXPERIMENT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = EXPERIMENT_ROOT / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
DEFAULT_RESULTS_JSON = str(METRICS_DIR / "qwen3_layer_block_sparsity.json")
QWEN3_ADAPTER = get_qwen3_adapter()


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run grouped sparse-attention experiments where only one contiguous layer "
            "block is assigned a target sparsity and all other layers stay at zero sparsity."
        )
    )
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--tokenizer-name-or-path", type=str, default=None)
    parser.add_argument(
        "--allow-download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether Hugging Face downloads are allowed when files are missing locally.",
    )
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom modeling/tokenizer code from the model repository.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="triton_universal",
        choices=[
            "auto",
            "dense_sdpa",
            "masked_sdpa",
            "gather_sparse",
            "triton_bigbird",
            "triton_universal",
        ],
    )
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument(
        "--text",
        type=str,
        default="Sparse attention kernels should run on real tokenized text.",
    )
    parser.add_argument(
        "--text-file",
        type=str,
        default=None,
        help="Optional UTF-8 text file to use as the prompt/context. If omitted, --text is used.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-iters", type=int, default=DEFAULT_WARMUP_ITERS)
    parser.add_argument("--benchmark-decode-steps", type=int, default=DEFAULT_DECODE_STEPS)
    parser.add_argument(
        "--decode-temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
    )
    parser.add_argument(
        "--ppl-text",
        type=str,
        default=None,
        help="Optional raw text or path used for perplexity evaluation. When omitted, WikiText is loaded.",
    )
    parser.add_argument("--ppl-text-file", type=str, default=None)
    parser.add_argument("--ppl-dataset-name", type=str, default="wikitext")
    parser.add_argument("--ppl-dataset-config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--ppl-dataset-split", type=str, default="test")
    parser.add_argument("--ppl-dataset-text-key", type=str, default="text")
    parser.add_argument("--ppl-max-length", type=int, default=1024)
    parser.add_argument(
        "--ppl-stride",
        type=int,
        default=None,
        help="Sliding-window stride for perplexity evaluation. Defaults to half of --ppl-max-length.",
    )
    parser.add_argument("--ppl-max-samples", type=int, default=None)
    parser.add_argument("--ppl-batch-size", type=int, default=1)
    parser.add_argument(
        "--prebuild-patterns",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fast-benchmark",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--respect-eos-in-benchmark",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--layer-span",
        type=int,
        default=6,
        help="Number of consecutive layers in each experiment block.",
    )
    parser.add_argument(
        "--target-sparsity",
        type=float,
        default=0.2,
        help="Sparsity assigned to the active layer block; all other layers use 0.0.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=DEFAULT_RESULTS_JSON,
        help="Summary JSON written after the full sweep completes.",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default=None,
        help="Optional CSV path for experiment summaries. Defaults next to --output-json.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.layer_span <= 0:
        raise ValueError("--layer-span must be a positive integer.")
    if not 0.0 <= args.target_sparsity < 1.0:
        raise ValueError("--target-sparsity must satisfy 0 <= sparsity < 1.")


def _load_model_shape(args: argparse.Namespace) -> tuple[int, int]:
    hf_kwargs = _build_hf_pretrained_kwargs(args)
    config = Qwen3Config.from_pretrained(
        args.model_name_or_path,
        **hf_kwargs,
    )
    return int(config.num_hidden_layers), int(config.num_key_value_heads)


def _default_summary_csv_path(output_json_path: str) -> str:
    output_path = Path(output_json_path)
    return str(output_path.with_name(f"{output_path.stem}_summary.csv"))


def _default_runs_dir(output_json_path: str) -> Path:
    output_path = Path(output_json_path)
    return output_path.with_name(f"{output_path.stem}_runs")


def _slug_float(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _build_layer_block_experiments(
    *,
    num_layers: int,
    num_groups: int,
    layer_span: int,
    target_sparsity: float,
) -> list[dict[str, Any]]:
    experiments: list[dict[str, Any]] = []
    for experiment_idx, start in enumerate(range(0, num_layers, layer_span)):
        stop = min(start + layer_span, num_layers)
        layer_group_sparsities = []
        for layer_idx in range(num_layers):
            layer_sparsity = target_sparsity if start <= layer_idx < stop else 0.0
            layer_group_sparsities.append([layer_sparsity] * num_groups)
        mean_sparsity = sum(
            value for row in layer_group_sparsities for value in row
        ) / max(1, num_layers * num_groups)
        label = f"exp_{experiment_idx:02d}_layers_{start}_{stop - 1}_s{_slug_float(target_sparsity)}"
        experiments.append(
            {
                "experiment_idx": experiment_idx,
                "label": label,
                "layer_start": start,
                "layer_end": stop - 1,
                "layer_count": stop - start,
                "target_sparsity": target_sparsity,
                "mean_sparsity": mean_sparsity,
                "layer_group_sparsities": layer_group_sparsities,
            }
        )
    return experiments


def _sparse_args_for_experiment(
    base_args: argparse.Namespace,
    *,
    layer_group_sparsities: list[list[float]],
) -> argparse.Namespace:
    run_args = copy.deepcopy(base_args)
    run_args.pattern = "bigbird"
    run_args.keep_ratio = None
    run_args.group_sparsities = None
    run_args.layer_group_sparsities = tuple(
        tuple(row) for row in layer_group_sparsities
    )
    return run_args


def _baseline_args(base_args: argparse.Namespace) -> argparse.Namespace:
    run_args = copy.deepcopy(base_args)
    run_args.pattern = "bigbird"
    run_args.keep_ratio = None
    run_args.group_sparsities = None
    run_args.layer_group_sparsities = None
    return run_args


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def _short_generation_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "prefill_time_s": metrics.get("prefill_time_s"),
        "decode_tokens_per_second": metrics.get("decode_tokens_per_second"),
        "backend_actual": metrics.get("backend_actual"),
        "pattern": metrics.get("pattern"),
    }


def _short_perplexity_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "perplexity": metrics.get("perplexity"),
        "tokens_per_second": metrics.get("tokens_per_second"),
        "avg_sparsity": metrics.get("avg_sparsity"),
        "avg_keep_ratio": metrics.get("avg_keep_ratio"),
        "backend_actual": metrics.get("backend_actual"),
        "pattern": metrics.get("pattern"),
    }


def _summarize_experiment(
    *,
    experiment: dict[str, Any],
    dense_generation: dict[str, Any],
    dense_perplexity: dict[str, Any],
    sparse_generation: dict[str, Any],
    sparse_perplexity: dict[str, Any],
    raw_metrics_path: Path,
) -> dict[str, Any]:
    dense_prefill_time = dense_generation.get("prefill_time_s")
    sparse_prefill_time = sparse_generation.get("prefill_time_s")
    dense_decode_tps = dense_generation.get("decode_tokens_per_second")
    sparse_decode_tps = sparse_generation.get("decode_tokens_per_second")
    dense_ppl = dense_perplexity.get("perplexity")
    sparse_ppl = sparse_perplexity.get("perplexity")
    dense_ppl_tps = dense_perplexity.get("tokens_per_second")
    sparse_ppl_tps = sparse_perplexity.get("tokens_per_second")

    ppl_ratio = _safe_ratio(sparse_ppl, dense_ppl)
    baseline_over_ppl = _safe_ratio(dense_ppl, sparse_ppl)
    ppl_delta = (
        None
        if dense_ppl is None or sparse_ppl is None
        else float(sparse_ppl) - float(dense_ppl)
    )

    return {
        "experiment_idx": experiment["experiment_idx"],
        "label": experiment["label"],
        "layer_start": experiment["layer_start"],
        "layer_end": experiment["layer_end"],
        "layer_count": experiment["layer_count"],
        "target_sparsity": experiment["target_sparsity"],
        "mean_sparsity": experiment["mean_sparsity"],
        "generation_prefill_speedup": _safe_ratio(dense_prefill_time, sparse_prefill_time),
        "generation_decode_speedup": _safe_ratio(sparse_decode_tps, dense_decode_tps),
        "generation_dense_prefill_time_s": dense_prefill_time,
        "generation_sparse_prefill_time_s": sparse_prefill_time,
        "generation_dense_decode_tokens_per_second": dense_decode_tps,
        "generation_sparse_decode_tokens_per_second": sparse_decode_tps,
        "perplexity_dense": dense_ppl,
        "perplexity_sparse": sparse_ppl,
        "ppl_ratio": ppl_ratio,
        "baseline_over_ppl": baseline_over_ppl,
        "ppl_delta": ppl_delta,
        "perplexity_eval_speedup": _safe_ratio(sparse_ppl_tps, dense_ppl_tps),
        "perplexity_dense_tokens_per_second": dense_ppl_tps,
        "perplexity_sparse_tokens_per_second": sparse_ppl_tps,
        "avg_keep_ratio": sparse_perplexity.get("avg_keep_ratio"),
        "avg_sparsity": sparse_perplexity.get("avg_sparsity"),
        "generation_backend_actual": sparse_generation.get("backend_actual"),
        "perplexity_backend_actual": sparse_perplexity.get("backend_actual"),
        "raw_metrics_path": str(raw_metrics_path),
    }


def _write_summary_csv(rows: list[dict[str, Any]], output_csv_path: str) -> None:
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run_dense_baselines(
    args: argparse.Namespace,
    *,
    baseline_generation_path: Path,
    baseline_perplexity_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dense_args = _baseline_args(args)
    dense_generation = benchmark_generation_runtime(
        runtime_name="dense",
        args=dense_args,
        warmup_iters=args.warmup_iters,
        decode_steps=args.benchmark_decode_steps,
        temperature=args.decode_temperature,
        prebuild_patterns=False,
        fast_benchmark=False,
        respect_eos_in_benchmark=args.respect_eos_in_benchmark,
    )
    write_json(dense_generation, str(baseline_generation_path))
    cleanup_cuda_state()

    dense_perplexity = benchmark_perplexity_runtime(
        QWEN3_ADAPTER,
        runtime_name="dense",
        args=dense_args,
    )
    write_json(dense_perplexity, str(baseline_perplexity_path))
    cleanup_cuda_state()
    return dense_generation, dense_perplexity


def _run_single_experiment(
    args: argparse.Namespace,
    *,
    experiment: dict[str, Any],
    dense_generation: dict[str, Any],
    dense_perplexity: dict[str, Any],
    raw_metrics_path: Path,
    baseline_generation_path: Path,
    baseline_perplexity_path: Path,
) -> dict[str, Any]:
    sparse_args = _sparse_args_for_experiment(
        args,
        layer_group_sparsities=experiment["layer_group_sparsities"],
    )
    sparse_generation = benchmark_generation_runtime(
        runtime_name="sparse",
        args=sparse_args,
        warmup_iters=args.warmup_iters,
        decode_steps=args.benchmark_decode_steps,
        temperature=args.decode_temperature,
        prebuild_patterns=args.prebuild_patterns,
        fast_benchmark=args.fast_benchmark,
        respect_eos_in_benchmark=args.respect_eos_in_benchmark,
    )
    cleanup_cuda_state()

    sparse_perplexity = benchmark_perplexity_runtime(
        QWEN3_ADAPTER,
        runtime_name="sparse",
        args=sparse_args,
    )
    cleanup_cuda_state()

    summary = _summarize_experiment(
        experiment=experiment,
        dense_generation=dense_generation,
        dense_perplexity=dense_perplexity,
        sparse_generation=sparse_generation,
        sparse_perplexity=sparse_perplexity,
        raw_metrics_path=raw_metrics_path,
    )
    raw_payload = {
        "experiment_type": "qwen3_layer_block_sparsity",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment,
        "dense_generation_baseline_path": str(baseline_generation_path),
        "dense_perplexity_baseline_path": str(baseline_perplexity_path),
        "summary": summary,
        "sparse_generation": sparse_generation,
        "sparse_perplexity": sparse_perplexity,
    }
    write_json(raw_payload, str(raw_metrics_path))
    return summary


def main() -> None:
    parser = _build_argparser()
    args = parser.parse_args()
    _validate_args(args)

    output_json_path = Path(args.output_json)
    summary_csv_path = Path(
        args.summary_csv
        if args.summary_csv is not None
        else _default_summary_csv_path(args.output_json)
    )
    runs_dir = _default_runs_dir(args.output_json)
    runs_dir.mkdir(parents=True, exist_ok=True)

    num_layers, num_groups = _load_model_shape(args)
    experiments = _build_layer_block_experiments(
        num_layers=num_layers,
        num_groups=num_groups,
        layer_span=args.layer_span,
        target_sparsity=args.target_sparsity,
    )

    baseline_generation_path = runs_dir / "dense_generation_baseline.json"
    baseline_perplexity_path = runs_dir / "dense_perplexity_baseline.json"

    print(
        f"[baseline] model={args.model_name_or_path} num_layers={num_layers} "
        f"num_groups={num_groups}"
    )
    dense_generation, dense_perplexity = _run_dense_baselines(
        args,
        baseline_generation_path=baseline_generation_path,
        baseline_perplexity_path=baseline_perplexity_path,
    )

    summary_rows: list[dict[str, Any]] = []
    for experiment in experiments:
        print(
            "[experiment] "
            f"{experiment['label']} layers={experiment['layer_start']}-{experiment['layer_end']} "
            f"sparsity={experiment['target_sparsity']}"
        )
        raw_metrics_path = runs_dir / f"{experiment['label']}.json"
        summary = _run_single_experiment(
            args,
            experiment=experiment,
            dense_generation=dense_generation,
            dense_perplexity=dense_perplexity,
            raw_metrics_path=raw_metrics_path,
            baseline_generation_path=baseline_generation_path,
            baseline_perplexity_path=baseline_perplexity_path,
        )
        summary_rows.append(summary)
        print(
            "[done] "
            f"{experiment['label']} decode_speedup={summary['generation_decode_speedup']} "
            f"ppl_ratio={summary['ppl_ratio']}"
        )

    _write_summary_csv(summary_rows, str(summary_csv_path))
    payload = {
        "experiment_type": "qwen3_layer_block_sparsity",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_name_or_path": args.model_name_or_path,
        "tokenizer_name_or_path": args.tokenizer_name_or_path or args.model_name_or_path,
        "num_layers": num_layers,
        "num_groups": num_groups,
        "layer_span": args.layer_span,
        "target_sparsity": args.target_sparsity,
        "experiment_count": len(summary_rows),
        "output_json": str(output_json_path),
        "summary_csv": str(summary_csv_path),
        "raw_runs_dir": str(runs_dir),
        "dense_baseline_paths": {
            "generation": str(baseline_generation_path),
            "perplexity": str(baseline_perplexity_path),
        },
        "dense_baseline_summary": {
            "generation": _short_generation_summary(dense_generation),
            "perplexity": _short_perplexity_summary(dense_perplexity),
        },
        "benchmark_settings": {
            "pattern": "bigbird",
            "backend": args.backend,
            "window_size": args.window_size,
            "block_size": args.block_size,
            "top_k": args.top_k,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "warmup_iters": args.warmup_iters,
            "benchmark_decode_steps": args.benchmark_decode_steps,
            "decode_temperature": args.decode_temperature,
            "ppl_max_length": args.ppl_max_length,
            "ppl_stride": args.ppl_stride,
            "ppl_max_samples": args.ppl_max_samples,
            "ppl_batch_size": args.ppl_batch_size,
            "prebuild_patterns": args.prebuild_patterns,
            "fast_benchmark": args.fast_benchmark,
            "respect_eos_in_benchmark": args.respect_eos_in_benchmark,
        },
        "experiments": summary_rows,
    }
    write_json(payload, str(output_json_path))
    print(f"[saved] summary_json={output_json_path}")
    print(f"[saved] summary_csv={summary_csv_path}")


if __name__ == "__main__":
    main()
