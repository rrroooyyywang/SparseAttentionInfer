import argparse
import time
from typing import Optional

import torch

from sparse_llm.common.benchmark.contracts import BenchmarkAdapter
from sparse_llm.common.benchmark.utils import (
    cleanup_cuda_state,
    sample_next_token,
    synchronize_device,
)
from sparse_llm.common.io.json_io import write_json


DEFAULT_DECODE_STEPS = 50
DEFAULT_TEMPERATURE = 0.8
DEFAULT_WARMUP_ITERS = 2


def _run_prefill_timed(
    adapter: BenchmarkAdapter,
    *,
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
):
    device = input_ids.device
    synchronize_device(device)
    start = time.perf_counter()
    prefill_result = adapter.run_prefill(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    synchronize_device(device)
    elapsed_s = time.perf_counter() - start
    return prefill_result, elapsed_s


def _run_decode_timed(
    adapter: BenchmarkAdapter,
    *,
    model,
    past_key_values,
    next_token_logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    decode_steps: int,
    temperature: float,
    eos_token_id: Optional[int] = None,
) -> tuple[torch.Tensor, list[int], list[float], list[float], list[float]]:
    generated_ids = input_ids
    current_attention_mask = attention_mask
    sampled_token_ids: list[int] = []
    keep_ratios: list[float] = []
    sparsities: list[float] = []
    decode_step_times_s: list[float] = []
    device = input_ids.device

    next_token = sample_next_token(
        next_token_logits,
        temperature=temperature,
    )
    generated_ids = torch.cat([generated_ids, next_token], dim=-1)
    current_attention_mask = torch.cat(
        [
            current_attention_mask,
            torch.ones(
                (current_attention_mask.size(0), 1),
                dtype=current_attention_mask.dtype,
                device=current_attention_mask.device,
            ),
        ],
        dim=-1,
    )
    sampled_token_ids.append(int(next_token[0, 0].item()))
    next_input_ids = next_token

    if eos_token_id is not None and bool((next_token == eos_token_id).all()):
        return (
            generated_ids,
            sampled_token_ids,
            keep_ratios,
            sparsities,
            decode_step_times_s,
        )

    for _ in range(max(0, decode_steps - 1)):
        synchronize_device(device)
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model(
                input_ids=next_input_ids,
                attention_mask=current_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=False,
                logits_to_keep=1,
            )
        synchronize_device(device)
        decode_step_times_s.append(time.perf_counter() - start)

        next_token = sample_next_token(
            outputs.logits[:, -1, :],
            temperature=temperature,
        )
        past_key_values = outputs.past_key_values
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        current_attention_mask = torch.cat(
            [
                current_attention_mask,
                torch.ones(
                    (current_attention_mask.size(0), 1),
                    dtype=current_attention_mask.dtype,
                    device=current_attention_mask.device,
                ),
            ],
            dim=-1,
        )
        next_input_ids = next_token
        sampled_token_ids.append(int(next_token[0, 0].item()))
        observations = adapter.collect_runtime_observations(model)
        pattern_stats = observations.get("pattern_stats")
        if pattern_stats is not None:
            keep_ratios.append(float(pattern_stats["keep_ratio"]))
            sparsities.append(float(pattern_stats["sparsity"]))

        if eos_token_id is not None and bool((next_token == eos_token_id).all()):
            break

    return (
        generated_ids,
        sampled_token_ids,
        keep_ratios,
        sparsities,
        decode_step_times_s,
    )


def benchmark_runtime(
    adapter: BenchmarkAdapter,
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
    runtime_bundle = adapter.build_runtime(
        args=args,
        runtime_name=runtime_name,
        mode="generation",
    )
    device = runtime_bundle.device
    tokenizer = runtime_bundle.tokenizer
    model = runtime_bundle.model
    input_ids = runtime_bundle.input_ids
    attention_mask = runtime_bundle.attention_mask
    prompt_source = runtime_bundle.prompt_source
    if input_ids is None or attention_mask is None or prompt_source is None:
        raise ValueError("Generation runtime must provide input_ids, attention_mask, and prompt_source.")

    prebuild_info = {
        "enabled": False,
        "prefill_pattern_built": False,
        "decode_patterns_built": 0,
        "layer_count": 0,
    }
    benchmark_mode_info = {
        "fast_benchmark": False,
        "configured_layers": 0,
    }
    if runtime_name == "sparse":
        benchmark_mode_info = adapter.configure_sparse_benchmark_mode(
            model,
            fast_benchmark=fast_benchmark,
        )
    if prebuild_patterns:
        if runtime_name != "sparse":
            raise ValueError("Pattern prebuild is only supported for sparse runtime.")
        prebuild_info = adapter.prebuild_generation_patterns(
            model,
            prompt_len=input_ids.size(1),
            decode_steps=decode_steps,
            batch_size=input_ids.size(0),
        )

    for _ in range(max(0, warmup_iters)):
        warmup_prefill = adapter.run_prefill(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        _ = adapter.run_decode(
            model=model,
            past_key_values=warmup_prefill.outputs.past_key_values,
            next_token_logits=warmup_prefill.outputs.logits[:, -1, :],
            input_ids=input_ids,
            attention_mask=attention_mask,
            decode_steps=decode_steps,
            temperature=temperature,
            eos_token_id=tokenizer.eos_token_id if respect_eos_in_benchmark else None,
        )

    prefill_result, prefill_time_s = _run_prefill_timed(
        adapter,
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    generated_ids, sampled_token_ids, keep_ratios, sparsities, decode_step_times_s = (
        _run_decode_timed(
            adapter,
            model=model,
            past_key_values=prefill_result.outputs.past_key_values,
            next_token_logits=prefill_result.outputs.logits[:, -1, :],
            input_ids=input_ids,
            attention_mask=attention_mask,
            decode_steps=decode_steps,
            temperature=temperature,
            eos_token_id=tokenizer.eos_token_id if respect_eos_in_benchmark else None,
        )
    )

    total_decode_time_s = float(sum(decode_step_times_s))
    measured_decode_steps = len(decode_step_times_s)
    measured_decode_tokens = measured_decode_steps * input_ids.size(0)
    decode_tokens_per_second = (
        measured_decode_tokens / total_decode_time_s if total_decode_time_s > 0 else None
    )
    generated_text = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    metrics = {
        "runtime_name": runtime_name,
        "model_name_or_path": args.model_name_or_path,
        "tokenizer_name_or_path": args.tokenizer_name_or_path or args.model_name_or_path,
        "device": str(device),
        "dtype": str(model.dtype),
        "batch_size": input_ids.size(0),
        "prompt_source": prompt_source,
        "prompt": tokenizer.decode(
            input_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ),
        "prompt_token_ids": input_ids[0].tolist(),
        "warmup_iters": warmup_iters,
        "fast_benchmark": fast_benchmark,
        "respect_eos_in_benchmark": respect_eos_in_benchmark,
        "benchmark_mode_info": benchmark_mode_info,
        "prebuild_patterns": prebuild_patterns,
        "prebuild_info": prebuild_info,
        "decode_temperature": temperature,
        "decode_steps_requested": decode_steps,
        "generated_token_count_per_sequence": len(sampled_token_ids),
        "generated_token_count_total": len(sampled_token_ids) * input_ids.size(0),
        "measured_decode_steps": measured_decode_steps,
        "measured_decode_tokens": measured_decode_tokens,
        "prefill_time_s": prefill_time_s,
        "first_token_time_s": prefill_time_s,
        "decode_step_times_s": decode_step_times_s,
        "decode_total_time_s": total_decode_time_s,
        "decode_tokens_per_second": decode_tokens_per_second,
        "loss": prefill_result.loss,
        "logits_shape": prefill_result.logits_shape,
        "backend_actual": prefill_result.backend_actual,
        "pattern_stats": prefill_result.pattern_stats,
        "decode_keep_ratios": keep_ratios,
        "decode_sparsities": sparsities,
        "generated_text": generated_text,
    }

    runtime_metadata = adapter.resolve_runtime_metadata(
        args=args,
        model=model,
        runtime_name=runtime_name,
    )
    metrics["backend_requested"] = runtime_metadata.backend_requested
    metrics["pattern"] = runtime_metadata.pattern
    if metrics["backend_actual"] is None:
        metrics["backend_actual"] = runtime_metadata.backend_actual

    return metrics


def _benchmark_to_json(
    adapter: BenchmarkAdapter,
    args: argparse.Namespace,
    output_json_path: str,
    *,
    runtime_names: tuple[str, ...],
    warmup_iters: int = DEFAULT_WARMUP_ITERS,
    decode_steps: int = DEFAULT_DECODE_STEPS,
    temperature: float = DEFAULT_TEMPERATURE,
    prebuild_patterns: bool = False,
    fast_benchmark: bool = False,
    respect_eos_in_benchmark: bool = False,
) -> dict:
    results: dict[str, dict] = {}
    for index, runtime_name in enumerate(runtime_names):
        results[runtime_name] = benchmark_runtime(
            adapter,
            runtime_name=runtime_name,
            args=args,
            warmup_iters=warmup_iters,
            decode_steps=decode_steps,
            temperature=temperature,
            prebuild_patterns=prebuild_patterns if runtime_name == "sparse" else False,
            fast_benchmark=fast_benchmark if runtime_name == "sparse" else False,
            respect_eos_in_benchmark=respect_eos_in_benchmark,
        )
        if index < len(runtime_names) - 1:
            cleanup_cuda_state()

    metrics = results[runtime_names[0]] if len(runtime_names) == 1 else results
    write_json(metrics, output_json_path)
    return metrics


def benchmark_sparse_to_json(
    adapter: BenchmarkAdapter,
    args: argparse.Namespace,
    output_json_path: str,
    *,
    warmup_iters: int = DEFAULT_WARMUP_ITERS,
    decode_steps: int = DEFAULT_DECODE_STEPS,
    temperature: float = DEFAULT_TEMPERATURE,
    prebuild_patterns: bool = False,
    fast_benchmark: bool = False,
    respect_eos_in_benchmark: bool = False,
) -> dict:
    return _benchmark_to_json(
        adapter,
        args,
        output_json_path,
        runtime_names=("sparse",),
        warmup_iters=warmup_iters,
        decode_steps=decode_steps,
        temperature=temperature,
        prebuild_patterns=prebuild_patterns,
        fast_benchmark=fast_benchmark,
        respect_eos_in_benchmark=respect_eos_in_benchmark,
    )


def benchmark_dense_to_json(
    adapter: BenchmarkAdapter,
    args: argparse.Namespace,
    output_json_path: str,
    *,
    warmup_iters: int = DEFAULT_WARMUP_ITERS,
    decode_steps: int = DEFAULT_DECODE_STEPS,
    temperature: float = DEFAULT_TEMPERATURE,
    prebuild_patterns: bool = False,
    fast_benchmark: bool = False,
    respect_eos_in_benchmark: bool = False,
) -> dict:
    return _benchmark_to_json(
        adapter,
        args,
        output_json_path,
        runtime_names=("dense",),
        warmup_iters=warmup_iters,
        decode_steps=decode_steps,
        temperature=temperature,
        prebuild_patterns=False,
        fast_benchmark=False,
        respect_eos_in_benchmark=respect_eos_in_benchmark,
    )


def benchmark_dense_and_sparse_to_json(
    adapter: BenchmarkAdapter,
    args: argparse.Namespace,
    output_json_path: str,
    *,
    warmup_iters: int = DEFAULT_WARMUP_ITERS,
    decode_steps: int = DEFAULT_DECODE_STEPS,
    temperature: float = DEFAULT_TEMPERATURE,
    prebuild_patterns: bool = False,
    fast_benchmark: bool = False,
    respect_eos_in_benchmark: bool = False,
) -> dict:
    return _benchmark_to_json(
        adapter,
        args,
        output_json_path,
        runtime_names=("dense", "sparse"),
        warmup_iters=warmup_iters,
        decode_steps=decode_steps,
        temperature=temperature,
        prebuild_patterns=prebuild_patterns,
        fast_benchmark=fast_benchmark,
        respect_eos_in_benchmark=respect_eos_in_benchmark,
    )
