import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from sparse_llm.common.benchmark.contracts import BenchmarkAdapter
from sparse_llm.common.benchmark.utils import cleanup_cuda_state, synchronize_device
from sparse_llm.common.io.json_io import write_json


def _load_wikitext_corpus(args: argparse.Namespace) -> tuple[str, str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Perplexity benchmarking on WikiText requires the `datasets` package. "
            "Install it with `uv add datasets` or add `datasets` to your environment."
        ) from exc

    dataset_kwargs = {
        "path": args.ppl_dataset_name,
        "name": args.ppl_dataset_config,
        "split": args.ppl_dataset_split,
    }
    if args.cache_dir is not None:
        dataset_kwargs["cache_dir"] = args.cache_dir
    if args.revision is not None:
        dataset_kwargs["revision"] = args.revision
    if args.hf_token is not None:
        dataset_kwargs["token"] = args.hf_token

    dataset = load_dataset(**dataset_kwargs)
    text_key = args.ppl_dataset_text_key
    if text_key not in dataset.column_names:
        raise KeyError(
            f"Dataset column {text_key!r} was not found in {args.ppl_dataset_name}/"
            f"{args.ppl_dataset_config} split {args.ppl_dataset_split!r}. "
            f"Available columns: {dataset.column_names}"
        )

    texts = [str(text) for text in dataset[text_key] if str(text).strip()]
    if not texts:
        raise ValueError("Loaded WikiText split is empty after filtering blank rows.")

    corpus = "\n\n".join(texts)
    source = f"{args.ppl_dataset_name}/{args.ppl_dataset_config}:{args.ppl_dataset_split}"
    return corpus, source


def _resolve_ppl_text(args: argparse.Namespace) -> tuple[str, str]:
    if args.ppl_text_file is not None:
        path = Path(args.ppl_text_file)
        return path.read_text(encoding="utf-8"), str(path)

    if args.ppl_text is not None:
        maybe_path = Path(args.ppl_text)
        if maybe_path.is_file():
            return maybe_path.read_text(encoding="utf-8"), str(maybe_path)
        return args.ppl_text, "cli"

    return _load_wikitext_corpus(args)


def _build_ppl_windows(
    token_ids: torch.Tensor,
    *,
    max_length: int,
    stride: int,
    max_samples: int | None = None,
) -> list[dict]:
    if token_ids.dim() != 1:
        raise ValueError("Expected a flat 1D token tensor for perplexity evaluation.")
    if token_ids.numel() < 2:
        raise ValueError("Need at least 2 tokens to compute perplexity.")
    if max_length < 2:
        raise ValueError("ppl_max_length must be at least 2.")
    if stride <= 0:
        raise ValueError("ppl_stride must be positive.")

    windows: list[dict] = []
    prev_end = 0
    total_tokens = token_ids.numel()

    for begin in range(0, total_tokens - 1, stride):
        end = min(begin + max_length, total_tokens)
        if end - begin < 2:
            break

        score_from = max(begin + 1, prev_end)
        if score_from >= end:
            prev_end = end
            if end == total_tokens:
                break
            continue

        input_ids = token_ids[begin:end].clone()
        labels = input_ids.clone()
        labels[: score_from - begin] = -100
        scored_tokens = int((labels[1:] != -100).sum().item())
        if scored_tokens <= 0:
            prev_end = end
            if end == total_tokens:
                break
            continue

        windows.append(
            {
                "input_ids": input_ids,
                "labels": labels,
                "scored_tokens": scored_tokens,
                "length": input_ids.numel(),
            }
        )
        prev_end = end
        if max_samples is not None and len(windows) >= max_samples:
            break
        if end == total_tokens:
            break

    if not windows:
        raise ValueError("No valid perplexity windows were built from the provided text.")

    return windows


def _iter_same_length_batches(examples: list[dict], batch_size: int):
    idx = 0
    while idx < len(examples):
        current_len = examples[idx]["length"]
        batch = [examples[idx]]
        idx += 1
        while (
            idx < len(examples)
            and len(batch) < batch_size
            and examples[idx]["length"] == current_len
        ):
            batch.append(examples[idx])
            idx += 1
        yield batch


def _collate_ppl_batch(
    batch: list[dict],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    input_ids = torch.stack([item["input_ids"] for item in batch], dim=0).to(device)
    labels = torch.stack([item["labels"] for item in batch], dim=0).to(device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    scored_tokens = sum(int(item["scored_tokens"]) for item in batch)
    return input_ids, attention_mask, labels, scored_tokens


def benchmark_runtime(
    adapter: BenchmarkAdapter,
    *,
    runtime_name: str,
    args: argparse.Namespace,
) -> dict:
    runtime_bundle = adapter.build_runtime(
        args=args,
        runtime_name=runtime_name,
        mode="perplexity",
    )
    device = runtime_bundle.device
    tokenizer = runtime_bundle.tokenizer
    model = runtime_bundle.model

    benchmark_mode_info = None
    prebuild_info = None
    fast_benchmark = bool(getattr(args, "fast_benchmark", False))
    if runtime_name == "sparse":
        benchmark_mode_info = adapter.configure_sparse_benchmark_mode(
            model,
            fast_benchmark=fast_benchmark,
        )

    ppl_text, corpus_source = _resolve_ppl_text(args)
    encoded = tokenizer(
        ppl_text,
        return_tensors="pt",
        padding=False,
        truncation=False,
    )
    token_ids = encoded["input_ids"][0].cpu()
    ppl_max_length = args.ppl_max_length
    ppl_stride = args.ppl_stride if args.ppl_stride is not None else max(1, ppl_max_length // 2)
    windows = _build_ppl_windows(
        token_ids,
        max_length=ppl_max_length,
        stride=ppl_stride,
        max_samples=args.ppl_max_samples,
    )

    if runtime_name == "sparse" and args.prebuild_patterns:
        prebuild_info = adapter.prebuild_perplexity_patterns(
            model,
            sequence_lengths=[int(window["length"]) for window in windows],
            batch_size=args.ppl_batch_size,
        )

    total_nll = 0.0
    total_scored_tokens = 0
    total_eval_tokens = 0
    weighted_keep_ratio = 0.0
    weighted_sparsity = 0.0
    backend_actual = "dense_hf"
    forward_time_s = 0.0
    postprocess_time_s = 0.0

    synchronize_device(device)
    start = time.perf_counter()
    for batch in _iter_same_length_batches(windows, args.ppl_batch_size):
        batch_start = time.perf_counter()
        input_ids, attention_mask, labels, scored_tokens = _collate_ppl_batch(batch, device)
        synchronize_device(device)
        forward_start = time.perf_counter()
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_attentions=False,
            )
        synchronize_device(device)
        forward_elapsed_s = time.perf_counter() - forward_start
        forward_time_s += forward_elapsed_s

        shift_logits = outputs.logits[:, :-1, :].float().contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss_sum = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        total_nll += float(loss_sum.item())
        total_scored_tokens += scored_tokens
        total_eval_tokens += int(input_ids.numel())

        if runtime_name == "sparse":
            observations = adapter.collect_runtime_observations(model)
            if observations.get("backend_actual") is not None:
                backend_actual = str(observations["backend_actual"])
            pattern_stats = observations.get("pattern_stats")
            if pattern_stats is not None:
                weighted_keep_ratio += float(pattern_stats["keep_ratio"]) * scored_tokens
                weighted_sparsity += float(pattern_stats["sparsity"]) * scored_tokens
        synchronize_device(device)
        postprocess_time_s += time.perf_counter() - batch_start - forward_elapsed_s
    synchronize_device(device)
    eval_time_s = time.perf_counter() - start

    avg_nll = total_nll / total_scored_tokens
    perplexity = math.exp(avg_nll)
    forward_tokens_per_second = (
        total_scored_tokens / forward_time_s if forward_time_s > 0 else None
    )
    tokens_per_second = total_scored_tokens / eval_time_s if eval_time_s > 0 else None

    metrics = {
        "runtime_name": runtime_name,
        "benchmark_type": "perplexity",
        "model_name_or_path": args.model_name_or_path,
        "tokenizer_name_or_path": args.tokenizer_name_or_path or args.model_name_or_path,
        "device": str(device),
        "dtype": str(model.dtype),
        "corpus_source": corpus_source,
        "dataset_name": None if args.ppl_text or args.ppl_text_file else args.ppl_dataset_name,
        "dataset_config": None if args.ppl_text or args.ppl_text_file else args.ppl_dataset_config,
        "dataset_split": None if args.ppl_text or args.ppl_text_file else args.ppl_dataset_split,
        "corpus_char_count": len(ppl_text),
        "tokenized_corpus_length": int(token_ids.numel()),
        "window_count": len(windows),
        "window_length": ppl_max_length,
        "window_stride": ppl_stride,
        "ppl_batch_size": args.ppl_batch_size,
        "scored_token_count": total_scored_tokens,
        "evaluated_token_count": total_eval_tokens,
        "avg_nll": avg_nll,
        "perplexity": perplexity,
        "fast_benchmark": bool(runtime_name == "sparse" and fast_benchmark),
        "forward_time_s": forward_time_s,
        "postprocess_time_s": postprocess_time_s,
        "eval_time_s": eval_time_s,
        "forward_tokens_per_second": forward_tokens_per_second,
        "eval_tokens_per_second": tokens_per_second,
        "tokens_per_second": tokens_per_second,
        "prebuild_patterns": bool(runtime_name == "sparse" and args.prebuild_patterns),
        "prebuild_info": prebuild_info,
        "benchmark_mode_info": benchmark_mode_info,
        "backend_actual": backend_actual,
    }

    runtime_metadata = adapter.resolve_runtime_metadata(
        args=args,
        model=model,
        runtime_name=runtime_name,
    )
    if runtime_name == "sparse" and total_scored_tokens > 0:
        metrics["avg_keep_ratio"] = weighted_keep_ratio / total_scored_tokens
        metrics["avg_sparsity"] = weighted_sparsity / total_scored_tokens
    else:
        metrics["avg_keep_ratio"] = None
        metrics["avg_sparsity"] = None
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
) -> dict:
    results: dict[str, dict] = {}
    for index, runtime_name in enumerate(runtime_names):
        results[runtime_name] = benchmark_runtime(
            adapter,
            runtime_name=runtime_name,
            args=args,
        )
        if index < len(runtime_names) - 1:
            cleanup_cuda_state()

    metrics = results[runtime_names[0]] if len(runtime_names) == 1 else results
    write_json(metrics, output_json_path)
    return metrics


def benchmark_sparse_perplexity_to_json(
    adapter: BenchmarkAdapter,
    args: argparse.Namespace,
    output_json_path: str,
) -> dict:
    return _benchmark_to_json(
        adapter,
        args,
        output_json_path,
        runtime_names=("sparse",),
    )


def benchmark_dense_perplexity_to_json(
    adapter: BenchmarkAdapter,
    args: argparse.Namespace,
    output_json_path: str,
) -> dict:
    return _benchmark_to_json(
        adapter,
        args,
        output_json_path,
        runtime_names=("dense",),
    )


def benchmark_dense_and_sparse_perplexity_to_json(
    adapter: BenchmarkAdapter,
    args: argparse.Namespace,
    output_json_path: str,
) -> dict:
    return _benchmark_to_json(
        adapter,
        args,
        output_json_path,
        runtime_names=("dense", "sparse"),
    )
