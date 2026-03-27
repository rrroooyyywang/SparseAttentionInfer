import argparse

from sparse_llm.common.benchmark.generation import (
    DEFAULT_DECODE_STEPS,
    DEFAULT_TEMPERATURE,
    DEFAULT_WARMUP_ITERS,
    benchmark_dense_and_sparse_to_json as _benchmark_dense_and_sparse_to_json,
    benchmark_dense_to_json as _benchmark_dense_to_json,
    benchmark_runtime as _benchmark_runtime,
    benchmark_sparse_to_json as _benchmark_sparse_to_json,
)
from sparse_llm.qwen3.adapter import get_qwen3_adapter
from sparse_llm.qwen3.metrics_io import DEFAULT_METRICS_JSON


QWEN3_ADAPTER = get_qwen3_adapter()


def benchmark_runtime(
    *,
    runtime_name: str,
    args: argparse.Namespace,
    warmup_iters: int = DEFAULT_WARMUP_ITERS,
    decode_steps: int = DEFAULT_DECODE_STEPS,
    temperature: float = DEFAULT_TEMPERATURE,
    prebuild_patterns: bool = False,
    fast_benchmark: bool = False,
    respect_eos_in_benchmark: bool = False,
) -> dict:
    return _benchmark_runtime(
        QWEN3_ADAPTER,
        runtime_name=runtime_name,
        args=args,
        warmup_iters=warmup_iters,
        decode_steps=decode_steps,
        temperature=temperature,
        prebuild_patterns=prebuild_patterns,
        fast_benchmark=fast_benchmark,
        respect_eos_in_benchmark=respect_eos_in_benchmark,
    )


def benchmark_sparse_to_json(
    args: argparse.Namespace,
    output_json_path: str = DEFAULT_METRICS_JSON,
    *,
    warmup_iters: int = DEFAULT_WARMUP_ITERS,
    decode_steps: int = DEFAULT_DECODE_STEPS,
    temperature: float = DEFAULT_TEMPERATURE,
    prebuild_patterns: bool = False,
    fast_benchmark: bool = False,
    respect_eos_in_benchmark: bool = False,
) -> dict:
    return _benchmark_sparse_to_json(
        QWEN3_ADAPTER,
        args,
        output_json_path,
        warmup_iters=warmup_iters,
        decode_steps=decode_steps,
        temperature=temperature,
        prebuild_patterns=prebuild_patterns,
        fast_benchmark=fast_benchmark,
        respect_eos_in_benchmark=respect_eos_in_benchmark,
    )


def benchmark_dense_to_json(
    args: argparse.Namespace,
    output_json_path: str = DEFAULT_METRICS_JSON,
    *,
    warmup_iters: int = DEFAULT_WARMUP_ITERS,
    decode_steps: int = DEFAULT_DECODE_STEPS,
    temperature: float = DEFAULT_TEMPERATURE,
    prebuild_patterns: bool = False,
    fast_benchmark: bool = False,
    respect_eos_in_benchmark: bool = False,
) -> dict:
    return _benchmark_dense_to_json(
        QWEN3_ADAPTER,
        args,
        output_json_path,
        warmup_iters=warmup_iters,
        decode_steps=decode_steps,
        temperature=temperature,
        prebuild_patterns=prebuild_patterns,
        fast_benchmark=fast_benchmark,
        respect_eos_in_benchmark=respect_eos_in_benchmark,
    )


def benchmark_dense_and_sparse_to_json(
    args: argparse.Namespace,
    output_json_path: str = DEFAULT_METRICS_JSON,
    *,
    warmup_iters: int = DEFAULT_WARMUP_ITERS,
    decode_steps: int = DEFAULT_DECODE_STEPS,
    temperature: float = DEFAULT_TEMPERATURE,
    prebuild_patterns: bool = False,
    fast_benchmark: bool = False,
    respect_eos_in_benchmark: bool = False,
) -> dict:
    return _benchmark_dense_and_sparse_to_json(
        QWEN3_ADAPTER,
        args,
        output_json_path,
        warmup_iters=warmup_iters,
        decode_steps=decode_steps,
        temperature=temperature,
        prebuild_patterns=prebuild_patterns,
        fast_benchmark=fast_benchmark,
        respect_eos_in_benchmark=respect_eos_in_benchmark,
    )
