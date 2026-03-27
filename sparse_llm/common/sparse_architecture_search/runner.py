from __future__ import annotations

import argparse
import copy
import statistics
from typing import Any, Callable, Optional

from sparse_llm.common.benchmark import benchmark_perplexity_runtime
from sparse_llm.common.benchmark.utils import cleanup_cuda_state
from sparse_llm.common.sparse_architecture_search.contracts import (
    SearchAdapter,
    SearchObjective,
    SearchStrategy,
)
from sparse_llm.common.sparse_architecture_search.results import (
    collect_successful_signatures,
    next_trial_index,
    pareto_front_indices,
    write_search_results,
)


DEFAULT_MAX_DUPLICATE_PROPOSALS = 512


def _median_metric(metrics_runs: list[dict[str, Any]], key: str) -> float | None:
    values = [float(run[key]) for run in metrics_runs if run.get(key) is not None]
    if not values:
        return None
    return statistics.median(values)


def _mean_metric(metrics_runs: list[dict[str, Any]], key: str) -> float | None:
    values = [float(run[key]) for run in metrics_runs if run.get(key) is not None]
    if not values:
        return None
    return statistics.fmean(values)


def aggregate_repeated_metrics(
    metrics_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    if not metrics_runs:
        raise ValueError("metrics_runs must contain at least one run.")

    if len(metrics_runs) == 1:
        aggregated = copy.deepcopy(metrics_runs[0])
        aggregated["repeat_count"] = 1
        aggregated["raw_runs"] = metrics_runs
        return aggregated

    aggregated = copy.deepcopy(metrics_runs[0])
    aggregated["repeat_count"] = len(metrics_runs)
    aggregated["raw_runs"] = metrics_runs
    aggregated["backend_actual_runs"] = [run.get("backend_actual") for run in metrics_runs]
    aggregated["backend_requested_runs"] = [
        run.get("backend_requested") for run in metrics_runs
    ]

    mean_metric_keys = (
        "perplexity",
        "avg_nll",
        "avg_keep_ratio",
        "avg_sparsity",
    )
    median_metric_keys = (
        "forward_time_s",
        "postprocess_time_s",
        "eval_time_s",
        "forward_tokens_per_second",
        "eval_tokens_per_second",
        "tokens_per_second",
    )

    for key in mean_metric_keys:
        if key in aggregated:
            aggregated[key] = _mean_metric(metrics_runs, key)
    for key in median_metric_keys:
        if key in aggregated:
            aggregated[key] = _median_metric(metrics_runs, key)

    return aggregated


def _default_evaluate_runtime(
    benchmark_adapter,
    *,
    runtime_name: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return benchmark_perplexity_runtime(
        benchmark_adapter,
        runtime_name=runtime_name,
        args=args,
    )


def _build_context_summary(search_adapter: SearchAdapter, context) -> dict[str, Any]:
    return {
        "model_name_or_path": context.model_name_or_path,
        "target_count": len(context.targets),
        "targets": [
            {
                "target_id": target.target_id,
                "layer_index": target.layer_index,
                "module_path": target.module_path,
                "role": target.role,
                "tags": list(target.tags),
            }
            for target in context.targets
        ],
        **context.extra,
    }


def _run_repeated_evaluation(
    *,
    search_adapter: SearchAdapter,
    args: argparse.Namespace,
    context,
    runtime_name: str,
    repeats: int,
    evaluate_runtime: Callable[..., dict[str, Any]],
    aggregate_metrics: Callable[[list[dict[str, Any]]], dict[str, Any]],
    materialized_candidate: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    metrics_runs: list[dict[str, Any]] = []
    for repeat_idx in range(repeats):
        run_args = copy.deepcopy(args)
        if materialized_candidate is not None:
            run_args = search_adapter.apply_candidate_to_args(
                run_args,
                context,
                materialized_candidate,
            )
        metrics = evaluate_runtime(
            search_adapter.benchmark_adapter,
            runtime_name=runtime_name,
            args=run_args,
        )
        search_adapter.validate_trial_metrics(
            args=args,
            context=context,
            runtime_name=runtime_name,
            materialized_candidate=materialized_candidate,
            metrics=metrics,
        )
        metrics["repeat_idx"] = repeat_idx
        metrics_runs.append(metrics)
        cleanup_cuda_state()
    return aggregate_metrics(metrics_runs)


def _build_trial_record(
    *,
    trial_idx: int,
    candidate: dict[str, Any],
    materialized_candidate: dict[str, Any],
    candidate_signature: tuple[Any, ...],
    metrics: Optional[dict[str, Any]] = None,
    objective_result: Optional[dict[str, Any]] = None,
    error: Optional[str] = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "trial_idx": trial_idx,
        "candidate": candidate,
        "materialized_candidate": materialized_candidate,
        "candidate_signature": candidate_signature,
    }
    if error is not None:
        record["status"] = "error"
        record["error"] = error
        return record

    record["status"] = "ok"
    record["metrics"] = metrics
    record["objective"] = objective_result
    if objective_result is not None:
        reported_metrics = objective_result.get("reported_metrics", {})
        record.update(reported_metrics)
    return record


def run_search(
    *,
    search_adapter: SearchAdapter,
    strategy: SearchStrategy,
    objective: SearchObjective,
    args: argparse.Namespace,
    output_json_path: str,
    existing_payload: Optional[dict[str, Any]] = None,
    evaluate_runtime: Optional[Callable[..., dict[str, Any]]] = None,
    aggregate_metrics_fn: Callable[[list[dict[str, Any]]], dict[str, Any]] = aggregate_repeated_metrics,
    evaluator_name: str = "benchmark_perplexity",
    extra_payload_metadata: Optional[dict[str, Any]] = None,
    max_duplicate_proposals: int = DEFAULT_MAX_DUPLICATE_PROPOSALS,
) -> dict[str, Any]:
    evaluate_runtime = evaluate_runtime or _default_evaluate_runtime

    search_adapter.normalize_search_args(args)
    context = search_adapter.build_search_context(args)
    search_space = search_adapter.build_default_search_space(args, context)

    payload: dict[str, Any] = existing_payload if existing_payload is not None else {}
    payload.update(
        {
            "benchmark_adapter": search_adapter.benchmark_adapter.name,
            "search_adapter": search_adapter.name,
            "strategy_name": strategy.name,
            "objective_name": objective.name,
            "evaluator_name": evaluator_name,
            "search_space_spec": search_space.describe(),
            "search_context_summary": _build_context_summary(search_adapter, context),
        }
    )
    payload.update(search_adapter.build_payload_metadata(args, context, search_space))
    if extra_payload_metadata is not None:
        payload.update(extra_payload_metadata)
    payload.setdefault("baseline", None)
    payload.setdefault("trials", [])
    payload.setdefault("pareto_front_indices", [])
    payload.setdefault("strategy_state", {})
    write_search_results(payload, output_json_path)

    baseline = payload.get("baseline")
    if baseline is None:
        baseline = _run_repeated_evaluation(
            search_adapter=search_adapter,
            args=args,
            context=context,
            runtime_name="dense",
            repeats=int(getattr(args, "dense_repeats", 1)),
            evaluate_runtime=evaluate_runtime,
            aggregate_metrics=aggregate_metrics_fn,
        )
        payload["baseline"] = baseline
        write_search_results(payload, output_json_path)

    fixed_candidate = search_adapter.get_fixed_candidate(args, context)
    if fixed_candidate is not None:
        materialized_candidate = search_adapter.materialize_candidate(
            args,
            context,
            fixed_candidate,
        )
        candidate_signature = search_adapter.candidate_signature(
            context,
            materialized_candidate,
        )
        trial_idx = next_trial_index(payload["trials"])
        try:
            sparse_metrics = _run_repeated_evaluation(
                search_adapter=search_adapter,
                args=args,
                context=context,
                runtime_name="sparse",
                repeats=int(getattr(args, "sparse_repeats", 1)),
                evaluate_runtime=evaluate_runtime,
                aggregate_metrics=aggregate_metrics_fn,
                materialized_candidate=materialized_candidate,
            )
            objective_result = objective.evaluate(baseline, sparse_metrics)
            trial_record = _build_trial_record(
                trial_idx=trial_idx,
                candidate=fixed_candidate,
                materialized_candidate=materialized_candidate,
                candidate_signature=candidate_signature,
                metrics=sparse_metrics,
                objective_result=objective_result,
            )
        except Exception as exc:
            trial_record = _build_trial_record(
                trial_idx=trial_idx,
                candidate=fixed_candidate,
                materialized_candidate=materialized_candidate,
                candidate_signature=candidate_signature,
                error=f"{type(exc).__name__}: {exc}",
            )
        if getattr(args, "trial_label", None) is not None:
            trial_record["trial_label"] = args.trial_label
        payload["search_type"] = "fixed_eval"
        payload["trials"].append(trial_record)
        payload["pareto_front_indices"] = pareto_front_indices(payload["trials"])
        payload["strategy_state"] = {}
        write_search_results(payload, output_json_path)
        return payload

    seen_signatures = collect_successful_signatures(payload["trials"])
    strategy.initialize(
        args=args,
        context=context,
        search_space=search_space,
        history=payload["trials"],
        state=payload.get("strategy_state"),
    )
    payload["search_type"] = strategy.name

    while not strategy.should_stop():
        candidate = None
        materialized_candidate = None
        candidate_signature = None
        attempts = 0
        while attempts < max_duplicate_proposals:
            proposed_candidate = strategy.propose()
            if proposed_candidate is None:
                break
            search_adapter.validate_candidate(args, context, proposed_candidate)
            proposed_materialized = search_adapter.materialize_candidate(
                args,
                context,
                proposed_candidate,
            )
            proposed_signature = search_adapter.candidate_signature(
                context,
                proposed_materialized,
            )
            if proposed_signature not in seen_signatures:
                candidate = proposed_candidate
                materialized_candidate = proposed_materialized
                candidate_signature = proposed_signature
                seen_signatures.add(proposed_signature)
                break
            reject_proposal = getattr(strategy, "reject_proposal", None)
            if callable(reject_proposal):
                reject_proposal(reason="duplicate")
            attempts += 1

        if candidate is None or materialized_candidate is None or candidate_signature is None:
            break

        trial_idx = next_trial_index(payload["trials"])
        try:
            sparse_metrics = _run_repeated_evaluation(
                search_adapter=search_adapter,
                args=args,
                context=context,
                runtime_name="sparse",
                repeats=int(getattr(args, "sparse_repeats", 1)),
                evaluate_runtime=evaluate_runtime,
                aggregate_metrics=aggregate_metrics_fn,
                materialized_candidate=materialized_candidate,
            )
            objective_result = objective.evaluate(baseline, sparse_metrics)
            trial_record = _build_trial_record(
                trial_idx=trial_idx,
                candidate=candidate,
                materialized_candidate=materialized_candidate,
                candidate_signature=candidate_signature,
                metrics=sparse_metrics,
                objective_result=objective_result,
            )
        except Exception as exc:
            trial_record = _build_trial_record(
                trial_idx=trial_idx,
                candidate=candidate,
                materialized_candidate=materialized_candidate,
                candidate_signature=candidate_signature,
                error=f"{type(exc).__name__}: {exc}",
            )
        if getattr(args, "trial_label", None) is not None:
            trial_record["trial_label"] = args.trial_label

        strategy.observe(trial_record)
        payload["trials"].append(trial_record)
        payload["pareto_front_indices"] = pareto_front_indices(payload["trials"])
        payload["strategy_state"] = strategy.export_state()
        write_search_results(payload, output_json_path)

    return payload
