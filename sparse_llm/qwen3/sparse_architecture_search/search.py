import argparse
import copy
import importlib.util
from pathlib import Path
from typing import Optional

from sparse_llm.common.sparse_architecture_search import (
    BayesianSearchStrategy,
    ParetoSpeedVsQualityObjective,
    RandomSearchStrategy,
    WeightedScalarObjective,
    load_search_results,
    pareto_front_indices,
    plot_search_results,
    run_search,
    write_search_results,
)
from sparse_llm.qwen3.sparse_architecture_search.config import GQSparseATuningConfig
from sparse_llm.qwen3.search_adapter import get_qwen3_search_adapter


MODEL_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = MODEL_ROOT / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PLOTS_DIR = OUTPUTS_DIR / "plots"

DEFAULT_RESULTS_JSON = str(METRICS_DIR / "gq_sparse_a_tuning_results.json")
DEFAULT_USER_CONFIG_PATH = Path(__file__).with_name("user_config.py")
DEFAULT_USER_CONFIG_CLASS_NAME = "UserGQSparseATuningConfig"
QWEN3_SEARCH_ADAPTER = get_qwen3_search_adapter()


def _grid_to_cli_default(values: Optional[tuple[float, ...]]) -> Optional[str]:
    if values is None:
        return None
    return ",".join(str(value) for value in values)


def _load_auto_config() -> tuple[GQSparseATuningConfig, Optional[Path]]:
    if not DEFAULT_USER_CONFIG_PATH.is_file():
        return GQSparseATuningConfig(), None

    spec = importlib.util.spec_from_file_location(
        "user_config",
        DEFAULT_USER_CONFIG_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load config module from {DEFAULT_USER_CONFIG_PATH}.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    config_cls = getattr(module, DEFAULT_USER_CONFIG_CLASS_NAME, None)
    if config_cls is None:
        raise RuntimeError(
            f"{DEFAULT_USER_CONFIG_PATH} must define {DEFAULT_USER_CONFIG_CLASS_NAME}."
        )
    config = config_cls()
    if not isinstance(config, GQSparseATuningConfig):
        raise TypeError(
            f"{DEFAULT_USER_CONFIG_CLASS_NAME} must inherit from GQSparseATuningConfig."
        )
    return config, DEFAULT_USER_CONFIG_PATH


def _default_plot_path(results_json_path: str) -> str:
    metrics_path = Path(results_json_path)
    return str(PLOTS_DIR / f"{metrics_path.stem}_plot.png")


def _build_argparser(config: GQSparseATuningConfig) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sparse-architecture-search tuner for Qwen3 grouped-query sparse attention.",
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default=config.model_name_or_path,
        required=config.model_name_or_path is None,
    )
    parser.add_argument(
        "--tokenizer-name-or-path",
        type=str,
        default=config.tokenizer_name_or_path,
    )
    parser.add_argument(
        "--allow-download",
        action=argparse.BooleanOptionalAction,
        default=config.allow_download,
        help="Whether Hugging Face downloads are allowed when files are missing locally.",
    )
    parser.add_argument("--cache-dir", type=str, default=config.cache_dir)
    parser.add_argument("--revision", type=str, default=config.revision)
    parser.add_argument("--hf-token", type=str, default=config.hf_token)
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=config.trust_remote_code,
    )
    parser.add_argument("--device", type=str, default=config.device)
    parser.add_argument(
        "--dtype",
        type=str,
        default=config.dtype,
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=config.backend,
        choices=[
            "auto",
            "dense_sdpa",
            "masked_sdpa",
            "gather_sparse",
            "triton_bigbird",
            "triton_universal",
        ],
    )
    parser.add_argument("--window-size", type=int, default=config.window_size)
    parser.add_argument("--block-size", type=int, default=config.block_size)
    parser.add_argument("--top-k", type=int, default=config.top_k)
    parser.add_argument("--seed", type=int, default=config.seed)
    parser.add_argument(
        "--strategy",
        type=str,
        default=config.strategy,
        choices=["random", "bayesian"],
    )
    parser.add_argument(
        "--objective",
        type=str,
        default=config.objective,
        choices=["pareto", "weighted_scalar"],
    )
    parser.add_argument(
        "--objective-speed-weight",
        type=float,
        default=config.objective_speed_weight,
    )
    parser.add_argument(
        "--objective-quality-weight",
        type=float,
        default=config.objective_quality_weight,
    )
    parser.add_argument(
        "--objective-quality-penalty",
        type=float,
        default=config.objective_quality_penalty,
    )
    parser.add_argument(
        "--objective-max-ppl-ratio",
        type=float,
        default=config.objective_max_ppl_ratio,
    )
    parser.add_argument(
        "--bayes-startup-trials",
        type=int,
        default=config.bayes_startup_trials,
    )
    parser.add_argument(
        "--bayes-multivariate",
        action=argparse.BooleanOptionalAction,
        default=config.bayes_multivariate,
    )
    parser.add_argument(
        "--bayes-group",
        action=argparse.BooleanOptionalAction,
        default=config.bayes_group,
    )
    parser.add_argument("--num-samples", type=int, default=config.num_samples)
    parser.add_argument(
        "--sampling-grid",
        type=str,
        default=_grid_to_cli_default(config.sampling_grid),
        help="Comma-separated candidate sparsity values used by random search.",
    )
    parser.add_argument(
        "--tail-sampling-grid",
        type=str,
        default=_grid_to_cli_default(config.tail_sampling_grid),
        help="Optional comma-separated sparsity values used only for layers after --prefix-dense-layers.",
    )
    parser.add_argument(
        "--prefix-dense-layers",
        type=int,
        default=config.prefix_dense_layers,
        help="Number of earliest layers forced to all-zero sparsity.",
    )
    parser.add_argument(
        "--layer-share-span",
        type=int,
        default=config.layer_share_span,
        help="How many consecutive layers share one sampled group-sparsity vector. "
        "Use 0 to share one vector across all layers; use 1 for per-layer search.",
    )
    parser.add_argument("--dense-repeats", type=int, default=config.dense_repeats)
    parser.add_argument("--sparse-repeats", type=int, default=config.sparse_repeats)
    parser.add_argument("--results-json", type=str, default=config.results_json)
    parser.add_argument("--plot-path", type=str, default=config.plot_path)
    parser.add_argument(
        "--append-results",
        action=argparse.BooleanOptionalAction,
        default=config.append_results,
    )
    parser.add_argument("--plot-only-json", type=str, default=config.plot_only_json)
    parser.add_argument("--trial-label", type=str, default=config.trial_label)
    parser.add_argument(
        "--ppl-text",
        type=str,
        default=config.ppl_text,
        help="Optional raw text or path used for perplexity evaluation. When omitted, WikiText is loaded.",
    )
    parser.add_argument("--ppl-text-file", type=str, default=config.ppl_text_file)
    parser.add_argument("--ppl-dataset-name", type=str, default=config.ppl_dataset_name)
    parser.add_argument("--ppl-dataset-config", type=str, default=config.ppl_dataset_config)
    parser.add_argument("--ppl-dataset-split", type=str, default=config.ppl_dataset_split)
    parser.add_argument("--ppl-dataset-text-key", type=str, default=config.ppl_dataset_text_key)
    parser.add_argument("--ppl-max-length", type=int, default=config.ppl_max_length)
    parser.add_argument("--ppl-stride", type=int, default=config.ppl_stride)
    parser.add_argument("--ppl-max-samples", type=int, default=config.ppl_max_samples)
    parser.add_argument("--ppl-batch-size", type=int, default=config.ppl_batch_size)
    parser.add_argument(
        "--prebuild-patterns",
        action=argparse.BooleanOptionalAction,
        default=config.prebuild_patterns,
    )
    parser.add_argument(
        "--fast-benchmark",
        action=argparse.BooleanOptionalAction,
        default=config.fast_benchmark,
        help="Reduce sparse benchmark telemetry during perplexity evaluation.",
    )
    return parser


def _build_strategy(args: argparse.Namespace):
    if args.strategy == "random":
        return RandomSearchStrategy(seed=args.seed, max_trials=args.num_samples)
    return BayesianSearchStrategy(
        seed=args.seed,
        max_trials=args.num_samples,
        startup_trials=args.bayes_startup_trials,
        multivariate=args.bayes_multivariate,
        group=args.bayes_group and args.bayes_multivariate,
    )


def _build_objective(args: argparse.Namespace):
    objective_name = args.objective
    if objective_name is None:
        objective_name = "weighted_scalar" if args.strategy == "bayesian" else "pareto"

    if args.strategy == "bayesian" and objective_name == "pareto":
        raise ValueError(
            "Bayesian search requires a scalar objective. "
            "Use `--objective weighted_scalar` or leave `--objective` unset."
        )

    if objective_name == "pareto":
        return ParetoSpeedVsQualityObjective()

    return WeightedScalarObjective(
        speed_weight=args.objective_speed_weight,
        quality_weight=args.objective_quality_weight,
        quality_penalty=args.objective_quality_penalty,
        max_ppl_ratio=args.objective_max_ppl_ratio,
    )


def _plot_title(args: argparse.Namespace) -> str:
    if args.strategy == "bayesian":
        return "Group-Query Sparse Attention Architecture Search (Bayesian)"
    return "Group-Query Sparse Attention Architecture Search (Random)"


def main() -> None:
    tuning_config, config_path = _load_auto_config()
    parser = _build_argparser(tuning_config)
    args = parser.parse_args()
    args.fixed_layer_head_sparsities = copy.deepcopy(
        tuning_config.fixed_layer_head_sparsities
    )

    if args.plot_only_json is not None:
        payload = load_search_results(args.plot_only_json)
        if payload is None:
            raise FileNotFoundError(f"Could not find results JSON: {args.plot_only_json}")
        payload["pareto_front_indices"] = pareto_front_indices(payload.get("trials", []))
        output_plot_path = args.plot_path or payload.get("plot_path") or _default_plot_path(
            args.plot_only_json
        )
        saved_plot_path = plot_search_results(
            payload,
            output_plot_path,
            title=_plot_title(args),
        )
        payload["plot_path"] = saved_plot_path
        write_search_results(payload, args.plot_only_json)
        print(f"Plot saved to: {saved_plot_path}")
        return

    strategy = _build_strategy(args)
    objective = _build_objective(args)
    existing_payload = load_search_results(args.results_json) if args.append_results else None
    payload = run_search(
        search_adapter=QWEN3_SEARCH_ADAPTER,
        strategy=strategy,
        objective=objective,
        args=args,
        output_json_path=args.results_json,
        existing_payload=existing_payload,
        extra_payload_metadata={
            "auto_config_path": None if config_path is None else str(config_path),
            "strategy_cli": args.strategy,
            "objective_cli": args.objective,
        },
    )

    plot_path = args.plot_path or _default_plot_path(args.results_json)
    saved_plot_path = plot_search_results(
        payload,
        plot_path,
        title=_plot_title(args),
    )
    payload["plot_path"] = saved_plot_path
    write_search_results(payload, args.results_json)

    print(f"Results JSON: {args.results_json}")
    print(f"Plot saved to: {saved_plot_path}")


if __name__ == "__main__":
    main()
