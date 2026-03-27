import argparse

from sparse_llm.common.benchmark.perplexity import (
    benchmark_dense_and_sparse_perplexity_to_json as _benchmark_dense_and_sparse_perplexity_to_json,
    benchmark_dense_perplexity_to_json as _benchmark_dense_perplexity_to_json,
    benchmark_runtime as _benchmark_runtime,
    benchmark_sparse_perplexity_to_json as _benchmark_sparse_perplexity_to_json,
)
from sparse_llm.qwen3.adapter import get_qwen3_adapter
from sparse_llm.qwen3.metrics_io import DEFAULT_METRICS_JSON


QWEN3_ADAPTER = get_qwen3_adapter()


def _benchmark_perplexity_model(
    *,
    runtime_name: str,
    args: argparse.Namespace,
    sparse: bool,
) -> dict:
    expected_sparse = runtime_name == "sparse"
    if sparse != expected_sparse:
        raise ValueError(
            "The legacy `sparse` flag must agree with `runtime_name`. "
            f"Received runtime_name={runtime_name!r}, sparse={sparse!r}."
        )
    return _benchmark_runtime(
        QWEN3_ADAPTER,
        runtime_name=runtime_name,
        args=args,
    )


def benchmark_sparse_perplexity_to_json(
    args: argparse.Namespace,
    output_json_path: str = DEFAULT_METRICS_JSON,
) -> dict:
    return _benchmark_sparse_perplexity_to_json(
        QWEN3_ADAPTER,
        args,
        output_json_path,
    )


def benchmark_dense_perplexity_to_json(
    args: argparse.Namespace,
    output_json_path: str = DEFAULT_METRICS_JSON,
) -> dict:
    return _benchmark_dense_perplexity_to_json(
        QWEN3_ADAPTER,
        args,
        output_json_path,
    )


def benchmark_dense_and_sparse_perplexity_to_json(
    args: argparse.Namespace,
    output_json_path: str = DEFAULT_METRICS_JSON,
) -> dict:
    return _benchmark_dense_and_sparse_perplexity_to_json(
        QWEN3_ADAPTER,
        args,
        output_json_path,
    )
