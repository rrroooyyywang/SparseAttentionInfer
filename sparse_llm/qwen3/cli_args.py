import argparse

from sparse_llm.qwen3.adapter import get_qwen3_adapter
from sparse_llm.qwen3.benchmark.generation import (
    DEFAULT_DECODE_STEPS,
    DEFAULT_TEMPERATURE,
    DEFAULT_WARMUP_ITERS,
    benchmark_dense_and_sparse_to_json,
    benchmark_dense_to_json,
    benchmark_sparse_to_json,
)
from sparse_llm.qwen3.benchmark.perplexity import (
    benchmark_dense_and_sparse_perplexity_to_json,
    benchmark_dense_perplexity_to_json,
    benchmark_sparse_perplexity_to_json,
)
from sparse_llm.qwen3.benchmark.smoke import run_smoke_test
from sparse_llm.qwen3.metrics_io import DEFAULT_METRICS_JSON
from sparse_llm.qwen3.plotting.metrics_plot import plot_metrics_json


QWEN3_ADAPTER = get_qwen3_adapter()


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke test for HF Qwen3 sparse-kernel attention integration."
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--tokenizer-name-or-path", type=str, default=None)
    parser.add_argument(
        "--allow-download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether Hugging Face downloads are allowed when files are missing locally.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Optional Hugging Face cache directory for configs, tokenizers, and weights.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Model/tokenizer revision to load from Hugging Face Hub.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Optional Hugging Face access token for gated models.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom modeling/tokenizer code from the model repository.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="local",
        choices=["local", "bigbird"],
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
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
    parser.add_argument("--keep-ratio", type=float, default=None)
    parser.add_argument(
        "--group-sparsities",
        type=str,
        default=None,
        help="Comma-separated sparsity values, one per KV group, for BigBird keep-ratio mode.",
    )
    parser.add_argument(
        "--layer-group-sparsities",
        type=str,
        default=None,
        help="JSON string or JSON file path containing per-layer group sparsity lists. Use null to fall back to the global setting on a layer.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument(
        "--text",
        type=str,
        default="Sparse attention kernels should run on real tokenized text.",
    )
    parser.add_argument(
        "--text-file",
        type=str,
        default=None,
        help="Optional UTF-8 text file to use as the prompt/context. If omitted, --text is used. If --text itself points to a file, that file is used automatically.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--run-mode",
        type=str,
        default="smoke",
        choices=[
            "smoke",
            "benchmark_sparse",
            "benchmark_dense",
            "benchmark_both",
            "benchmark_sparse_ppl",
            "benchmark_dense_ppl",
            "benchmark_both_ppl",
        ],
    )
    parser.add_argument("--metrics-json", type=str, default=DEFAULT_METRICS_JSON)
    parser.add_argument("--warmup-iters", type=int, default=DEFAULT_WARMUP_ITERS)
    parser.add_argument("--benchmark-decode-steps", type=int, default=DEFAULT_DECODE_STEPS)
    parser.add_argument(
        "--ppl-text",
        type=str,
        default=None,
        help="Optional raw text or path used for perplexity evaluation. When omitted, WikiText is loaded.",
    )
    parser.add_argument(
        "--ppl-text-file",
        type=str,
        default=None,
        help="Optional UTF-8 text file used for perplexity evaluation.",
    )
    parser.add_argument(
        "--ppl-dataset-name",
        type=str,
        default="wikitext",
        help="Dataset name used for perplexity evaluation when --ppl-text/--ppl-text-file are not provided.",
    )
    parser.add_argument(
        "--ppl-dataset-config",
        type=str,
        default="wikitext-2-raw-v1",
        help="Dataset config used for perplexity evaluation.",
    )
    parser.add_argument(
        "--ppl-dataset-split",
        type=str,
        default="test",
        help="Dataset split used for perplexity evaluation.",
    )
    parser.add_argument(
        "--ppl-dataset-text-key",
        type=str,
        default="text",
        help="Column name containing raw text in the perplexity dataset.",
    )
    parser.add_argument(
        "--ppl-max-length",
        type=int,
        default=1024,
        help="Context window length for perplexity evaluation.",
    )
    parser.add_argument(
        "--ppl-stride",
        type=int,
        default=None,
        help="Sliding-window stride for perplexity evaluation. Defaults to half of --ppl-max-length.",
    )
    parser.add_argument(
        "--ppl-max-samples",
        type=int,
        default=None,
        help="Optional cap on the number of perplexity windows to score.",
    )
    parser.add_argument(
        "--ppl-batch-size",
        type=int,
        default=1,
        help="Batch size for perplexity evaluation windows of the same length.",
    )
    parser.add_argument(
        "--prebuild-patterns",
        action=argparse.BooleanOptionalAction,
        default=False,
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
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser


def main() -> None:
    parser = _build_argparser()
    args = parser.parse_args()
    QWEN3_ADAPTER.normalize_args(args)
    if args.run_mode == "smoke":
        run_smoke_test(args)
        return

    if args.run_mode == "benchmark_sparse":
        metrics = benchmark_sparse_to_json(
            args,
            output_json_path=args.metrics_json,
            warmup_iters=args.warmup_iters,
            decode_steps=args.benchmark_decode_steps,
            temperature=DEFAULT_TEMPERATURE,
            prebuild_patterns=args.prebuild_patterns,
            fast_benchmark=args.fast_benchmark,
            respect_eos_in_benchmark=args.respect_eos_in_benchmark,
        )
    elif args.run_mode == "benchmark_dense":
        metrics = benchmark_dense_to_json(
            args,
            output_json_path=args.metrics_json,
            warmup_iters=args.warmup_iters,
            decode_steps=args.benchmark_decode_steps,
            temperature=DEFAULT_TEMPERATURE,
            prebuild_patterns=False,
            fast_benchmark=False,
            respect_eos_in_benchmark=args.respect_eos_in_benchmark,
        )
    elif args.run_mode == "benchmark_both":
        metrics = benchmark_dense_and_sparse_to_json(
            args,
            output_json_path=args.metrics_json,
            warmup_iters=args.warmup_iters,
            decode_steps=args.benchmark_decode_steps,
            temperature=DEFAULT_TEMPERATURE,
            prebuild_patterns=args.prebuild_patterns,
            fast_benchmark=args.fast_benchmark,
            respect_eos_in_benchmark=args.respect_eos_in_benchmark,
        )
    elif args.run_mode == "benchmark_sparse_ppl":
        metrics = benchmark_sparse_perplexity_to_json(
            args,
            output_json_path=args.metrics_json,
        )
    elif args.run_mode == "benchmark_dense_ppl":
        metrics = benchmark_dense_perplexity_to_json(
            args,
            output_json_path=args.metrics_json,
        )
    else:
        metrics = benchmark_dense_and_sparse_perplexity_to_json(
            args,
            output_json_path=args.metrics_json,
        )

    print(f"metrics_json={args.metrics_json}")
    if args.plot:
        plot_path = plot_metrics_json(args.metrics_json)
        print(f"plot_path={plot_path}")
    if isinstance(metrics, dict) and "generated_text" in metrics:
        print(f"generated_text={metrics['generated_text']}")
