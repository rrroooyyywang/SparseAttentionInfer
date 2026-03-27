from __future__ import annotations

import argparse
import copy
from typing import Any, Optional

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from sparse_llm.common.sparse_architecture_search.contracts import (
    LayerTarget,
    SearchContext,
)
from sparse_llm.common.sparse_architecture_search.search_space import CallableSearchSpace
from sparse_llm.qwen3.adapter import Qwen3BenchmarkAdapter, get_qwen3_adapter
from sparse_llm.qwen3.integrations.runtime import _build_hf_pretrained_kwargs


TRITON_BACKENDS = {"triton_universal", "triton_bigbird"}


def _parse_grid(raw_value: str | tuple[float, ...]) -> tuple[float, ...]:
    if isinstance(raw_value, tuple):
        values = raw_value
    else:
        values = tuple(
            float(chunk.strip())
            for chunk in raw_value.split(",")
            if chunk.strip()
        )
    if not values:
        raise ValueError("--sampling-grid must contain at least one sparsity value.")
    if any(not 0.0 <= value < 1.0 for value in values):
        raise ValueError("--sampling-grid values must satisfy 0 <= sparsity < 1.")
    return values


def _load_model_shape(args: argparse.Namespace) -> tuple[int, int, int]:
    hf_kwargs = _build_hf_pretrained_kwargs(args)
    config = Qwen3Config.from_pretrained(
        args.model_name_or_path,
        **hf_kwargs,
    )
    return (
        config.num_hidden_layers,
        config.num_key_value_heads,
        config.num_attention_heads,
    )


def _normalize_fixed_layer_group_sparsities(
    fixed_layer_head_sparsities: Optional[list[list[float]]],
    *,
    num_layers: int,
    num_groups: int,
    num_heads: int,
) -> Optional[list[list[float]]]:
    if fixed_layer_head_sparsities is None:
        return None
    if len(fixed_layer_head_sparsities) != num_layers:
        raise ValueError(
            "fixed_layer_head_sparsities must have one row per model layer: "
            f"expected {num_layers}, got {len(fixed_layer_head_sparsities)}."
        )

    if num_heads % num_groups != 0:
        raise ValueError(
            f"num_attention_heads={num_heads} must be divisible by num_key_value_heads={num_groups}."
        )
    group_size = num_heads // num_groups
    layer_group_sparsities: list[list[float]] = []

    for layer_idx, row in enumerate(fixed_layer_head_sparsities):
        if len(row) == num_groups:
            normalized_row = [float(value) for value in row]
        elif len(row) == num_heads:
            normalized_row = []
            for group_idx in range(num_groups):
                start = group_idx * group_size
                chunk = [float(value) for value in row[start : start + group_size]]
                if max(chunk) - min(chunk) > 1e-8:
                    raise ValueError(
                        "fixed_layer_head_sparsities row "
                        f"{layer_idx} varies within GQA group {group_idx}. "
                        "Current sparse kernel supports one sparsity per KV group."
                    )
                normalized_row.append(chunk[0])
        else:
            raise ValueError(
                "Each row of fixed_layer_head_sparsities must have length equal to either "
                f"num_key_value_heads ({num_groups}) or num_attention_heads ({num_heads}); "
                f"row {layer_idx} has length {len(row)}."
            )

        if any(not 0.0 <= value < 1.0 for value in normalized_row):
            raise ValueError(
                f"Layer {layer_idx} contains invalid sparsity values; expected 0 <= sparsity < 1."
            )
        layer_group_sparsities.append(normalized_row)

    return layer_group_sparsities


def _effective_layer_share_span(
    requested_span: int,
    num_layers: int,
) -> int:
    if requested_span <= 0:
        return num_layers
    return min(requested_span, num_layers)


def _sample_candidate(
    *,
    rng,
    grid: tuple[float, ...],
    prefix_dense_layers: int,
    num_groups: int,
    num_layers: int,
    layer_share_span: int,
) -> dict[str, Any]:
    prefix_dense_layers = min(max(0, prefix_dense_layers), num_layers)
    remaining_layers = num_layers - prefix_dense_layers

    if prefix_dense_layers == 0 and layer_share_span >= num_layers:
        group_sparsities = [rng.choice(grid) for _ in range(num_groups)]
        return {
            "group_sparsities": group_sparsities,
            "layer_group_sparsities": None,
            "layer_share_span": layer_share_span,
            "prefix_dense_layers": prefix_dense_layers,
            "mean_sparsity": sum(group_sparsities) / len(group_sparsities),
        }

    if remaining_layers <= 0:
        layer_group_sparsities = [[0.0] * num_groups for _ in range(num_layers)]
        return {
            "group_sparsities": None,
            "layer_group_sparsities": layer_group_sparsities,
            "layer_share_span": layer_share_span,
            "prefix_dense_layers": prefix_dense_layers,
            "mean_sparsity": 0.0,
        }

    effective_tail_span = remaining_layers if layer_share_span >= num_layers else max(1, layer_share_span)
    num_layer_blocks = -(-remaining_layers // effective_tail_span)
    block_group_sparsities = [
        tuple(rng.choice(grid) for _ in range(num_groups))
        for _ in range(num_layer_blocks)
    ]
    layer_group_sparsities: list[list[float]] = []
    for layer_idx in range(num_layers):
        if layer_idx < prefix_dense_layers:
            layer_group_sparsities.append([0.0] * num_groups)
            continue
        tail_idx = layer_idx - prefix_dense_layers
        layer_group_sparsities.append(
            list(block_group_sparsities[tail_idx // effective_tail_span])
        )
    flat_values = [
        value
        for layer_values in layer_group_sparsities
        for value in layer_values
    ]
    return {
        "group_sparsities": None,
        "layer_group_sparsities": layer_group_sparsities,
        "layer_share_span": layer_share_span,
        "prefix_dense_layers": prefix_dense_layers,
        "mean_sparsity": sum(flat_values) / len(flat_values),
    }


class Qwen3SearchAdapter:
    name = "qwen3_gq_sparse"
    benchmark_adapter: Qwen3BenchmarkAdapter

    def __init__(self, benchmark_adapter: Qwen3BenchmarkAdapter) -> None:
        self.benchmark_adapter = benchmark_adapter

    def register_search_args(self, parser) -> None:
        del parser
        return None

    def normalize_search_args(self, args: argparse.Namespace) -> None:
        args.pattern = "bigbird"
        args.group_sparsities = None
        args.layer_group_sparsities = None
        if hasattr(args, "sampling_grid"):
            args.sampling_grid = _parse_grid(args.sampling_grid)
        if hasattr(args, "tail_sampling_grid") and args.tail_sampling_grid is not None:
            args.tail_sampling_grid = _parse_grid(args.tail_sampling_grid)

    def build_search_context(self, args: argparse.Namespace) -> SearchContext:
        num_layers, num_groups, num_heads = _load_model_shape(args)
        targets = [
            LayerTarget(
                target_id=f"layer_{layer_idx}.self_attn",
                layer_index=layer_idx,
                module_path=f"model.layers.{layer_idx}.self_attn",
                role="self_attention",
                num_attention_heads=num_heads,
                num_key_value_heads=num_groups,
                tags=("decoder", "attention"),
            )
            for layer_idx in range(num_layers)
        ]
        return SearchContext(
            model_name_or_path=args.model_name_or_path,
            targets=targets,
            extra={
                "num_layers": num_layers,
                "num_groups": num_groups,
                "num_heads": num_heads,
            },
        )

    def build_default_search_space(
        self,
        args: argparse.Namespace,
        context: SearchContext,
    ) -> CallableSearchSpace:
        num_layers = int(context.extra["num_layers"])
        num_groups = int(context.extra["num_groups"])
        sampling_grid = tuple(args.sampling_grid)
        tail_sampling_grid = (
            sampling_grid
            if args.tail_sampling_grid is None
            else tuple(args.tail_sampling_grid)
        )
        effective_span = _effective_layer_share_span(args.layer_share_span, num_layers)

        def _sampler(rng) -> dict[str, Any]:
            return _sample_candidate(
                rng=rng,
                grid=tail_sampling_grid,
                prefix_dense_layers=args.prefix_dense_layers,
                num_groups=num_groups,
                num_layers=num_layers,
                layer_share_span=effective_span,
            )

        def _optuna_sampler(trial) -> dict[str, Any]:
            prefix_dense_layers = min(max(0, args.prefix_dense_layers), num_layers)
            remaining_layers = num_layers - prefix_dense_layers

            if prefix_dense_layers == 0 and effective_span >= num_layers:
                group_sparsities = [
                    float(
                        trial.suggest_categorical(
                            f"group_sparsity_{group_idx}",
                            list(tail_sampling_grid),
                        )
                    )
                    for group_idx in range(num_groups)
                ]
                return {
                    "group_sparsities": group_sparsities,
                    "layer_group_sparsities": None,
                    "layer_share_span": effective_span,
                    "prefix_dense_layers": prefix_dense_layers,
                    "mean_sparsity": sum(group_sparsities) / len(group_sparsities),
                }

            if remaining_layers <= 0:
                layer_group_sparsities = [[0.0] * num_groups for _ in range(num_layers)]
                return {
                    "group_sparsities": None,
                    "layer_group_sparsities": layer_group_sparsities,
                    "layer_share_span": effective_span,
                    "prefix_dense_layers": prefix_dense_layers,
                    "mean_sparsity": 0.0,
                }

            num_layer_blocks = -(-remaining_layers // effective_span)
            block_group_sparsities = []
            for block_idx in range(num_layer_blocks):
                block_group_sparsities.append(
                    tuple(
                        float(
                            trial.suggest_categorical(
                                f"block_{block_idx}_group_{group_idx}_sparsity",
                                list(tail_sampling_grid),
                            )
                        )
                        for group_idx in range(num_groups)
                    )
                )

            layer_group_sparsities: list[list[float]] = []
            for layer_idx in range(num_layers):
                if layer_idx < prefix_dense_layers:
                    layer_group_sparsities.append([0.0] * num_groups)
                    continue
                tail_idx = layer_idx - prefix_dense_layers
                layer_group_sparsities.append(
                    list(block_group_sparsities[tail_idx // effective_span])
                )
            flat_values = [
                value
                for layer_values in layer_group_sparsities
                for value in layer_values
            ]
            return {
                "group_sparsities": None,
                "layer_group_sparsities": layer_group_sparsities,
                "layer_share_span": effective_span,
                "prefix_dense_layers": prefix_dense_layers,
                "mean_sparsity": sum(flat_values) / len(flat_values),
            }

        return CallableSearchSpace(
            name="qwen3_group_sparsity",
            sampler=_sampler,
            optuna_sampler=_optuna_sampler,
            spec={
                "sampling_grid": list(sampling_grid),
                "tail_sampling_grid": list(tail_sampling_grid),
                "prefix_dense_layers": args.prefix_dense_layers,
                "layer_share_span": effective_span,
                "num_layers": num_layers,
                "num_groups": num_groups,
            },
        )

    def build_payload_metadata(
        self,
        args: argparse.Namespace,
        context: SearchContext,
        search_space: CallableSearchSpace,
    ) -> dict[str, Any]:
        fixed_layer_head_sparsities = getattr(args, "fixed_layer_head_sparsities", None)
        return {
            "model_name_or_path": args.model_name_or_path,
            "tokenizer_name_or_path": args.tokenizer_name_or_path or args.model_name_or_path,
            "num_layers": context.extra["num_layers"],
            "num_groups": context.extra["num_groups"],
            "num_heads": context.extra["num_heads"],
            "sampling_grid": list(args.sampling_grid),
            "tail_sampling_grid": search_space.describe()["tail_sampling_grid"],
            "prefix_dense_layers": args.prefix_dense_layers,
            "layer_share_span": search_space.describe()["layer_share_span"],
            "num_samples_requested": args.num_samples,
            "fixed_layer_head_sparsities": fixed_layer_head_sparsities,
            "fixed_layer_group_sparsities": self._fixed_layer_group_sparsities(args, context),
            "dense_repeats": args.dense_repeats,
            "sparse_repeats": args.sparse_repeats,
            "prebuild_patterns": bool(args.prebuild_patterns),
            "fast_benchmark": bool(getattr(args, "fast_benchmark", False)),
        }

    def get_fixed_candidate(
        self,
        args: argparse.Namespace,
        context: SearchContext,
    ) -> Optional[dict[str, Any]]:
        layer_group_sparsities = self._fixed_layer_group_sparsities(args, context)
        if layer_group_sparsities is None:
            return None
        mean_sparsity = sum(
            value
            for row in layer_group_sparsities
            for value in row
        ) / max(1, int(context.extra["num_layers"]) * int(context.extra["num_groups"]))
        return {
            "group_sparsities": None,
            "layer_group_sparsities": layer_group_sparsities,
            "prefix_dense_layers": args.prefix_dense_layers,
            "layer_share_span": _effective_layer_share_span(
                args.layer_share_span,
                int(context.extra["num_layers"]),
            ),
            "mean_sparsity": mean_sparsity,
        }

    def validate_candidate(
        self,
        args: argparse.Namespace,
        context: SearchContext,
        candidate: dict[str, Any],
    ) -> None:
        del args
        num_layers = int(context.extra["num_layers"])
        num_groups = int(context.extra["num_groups"])
        group_sparsities = candidate.get("group_sparsities")
        layer_group_sparsities = candidate.get("layer_group_sparsities")
        if group_sparsities is not None:
            if len(group_sparsities) != num_groups:
                raise ValueError(
                    f"group_sparsities must have length {num_groups}, got {len(group_sparsities)}."
                )
        if layer_group_sparsities is not None:
            if len(layer_group_sparsities) != num_layers:
                raise ValueError(
                    "layer_group_sparsities must have one row per model layer: "
                    f"expected {num_layers}, got {len(layer_group_sparsities)}."
                )
            for row in layer_group_sparsities:
                if len(row) != num_groups:
                    raise ValueError(
                        f"Each layer row must contain {num_groups} group sparsities."
                    )

    def materialize_candidate(
        self,
        args: argparse.Namespace,
        context: SearchContext,
        candidate: dict[str, Any],
    ) -> dict[str, Any]:
        del args
        group_sparsities = candidate.get("group_sparsities")
        layer_group_sparsities = candidate.get("layer_group_sparsities")
        summary = {
            "mean_sparsity": candidate.get("mean_sparsity"),
            "prefix_dense_layers": candidate.get("prefix_dense_layers"),
            "layer_share_span": candidate.get("layer_share_span"),
        }
        if summary["mean_sparsity"] is None and layer_group_sparsities is not None:
            flat_values = [
                value
                for row in layer_group_sparsities
                for value in row
            ]
            summary["mean_sparsity"] = sum(flat_values) / max(1, len(flat_values))
        placement = [
            {
                "target_id": target.target_id,
                "enabled": True,
                "kernel": "qwen3_sparse_attention",
            }
            for target in context.targets
        ]
        return {
            "runtime_overrides": {
                "pattern": "bigbird",
                "group_sparsities": group_sparsities,
                "layer_group_sparsities": layer_group_sparsities,
            },
            "placement": placement,
            "summary": summary,
        }

    def candidate_signature(
        self,
        context: SearchContext,
        materialized_candidate: dict[str, Any],
    ) -> tuple[Any, ...]:
        del context
        runtime_overrides = materialized_candidate["runtime_overrides"]
        group_sparsities = runtime_overrides.get("group_sparsities")
        if group_sparsities is not None:
            return ("global", tuple(group_sparsities))
        return (
            "layered",
            tuple(
                tuple(layer_values)
                for layer_values in runtime_overrides["layer_group_sparsities"]
            ),
        )

    def apply_candidate_to_args(
        self,
        args: argparse.Namespace,
        context: SearchContext,
        materialized_candidate: dict[str, Any],
    ) -> argparse.Namespace:
        del context
        run_args = copy.deepcopy(args)
        runtime_overrides = materialized_candidate["runtime_overrides"]
        run_args.pattern = runtime_overrides["pattern"]
        run_args.keep_ratio = None
        run_args.group_sparsities = (
            None
            if runtime_overrides["group_sparsities"] is None
            else tuple(runtime_overrides["group_sparsities"])
        )
        run_args.layer_group_sparsities = (
            None
            if runtime_overrides["layer_group_sparsities"] is None
            else tuple(tuple(values) for values in runtime_overrides["layer_group_sparsities"])
        )
        return run_args

    def validate_trial_metrics(
        self,
        args: argparse.Namespace,
        context: SearchContext,
        runtime_name: str,
        materialized_candidate: Optional[dict[str, Any]],
        metrics: dict[str, Any],
    ) -> None:
        del args, context, materialized_candidate
        if runtime_name != "sparse":
            return
        actual_backend = metrics.get("backend_actual")
        if actual_backend not in TRITON_BACKENDS:
            raise RuntimeError(
                "Sparse tuning requires the Triton kernel, but the actual backend was "
                f"{actual_backend!r}."
            )

    def _fixed_layer_group_sparsities(
        self,
        args: argparse.Namespace,
        context: SearchContext,
    ) -> Optional[list[list[float]]]:
        return _normalize_fixed_layer_group_sparsities(
            getattr(args, "fixed_layer_head_sparsities", None),
            num_layers=int(context.extra["num_layers"]),
            num_groups=int(context.extra["num_groups"]),
            num_heads=int(context.extra["num_heads"]),
        )


_QWEN3_SEARCH_ADAPTER = Qwen3SearchAdapter(get_qwen3_adapter())


def get_qwen3_search_adapter() -> Qwen3SearchAdapter:
    return _QWEN3_SEARCH_ADAPTER
