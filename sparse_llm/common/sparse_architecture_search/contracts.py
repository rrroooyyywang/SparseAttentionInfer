from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, TypedDict

from sparse_llm.common.benchmark.contracts import BenchmarkAdapter


@dataclass
class LayerTarget:
    target_id: str
    layer_index: int
    module_path: str
    role: str
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None
    tags: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchContext:
    model_name_or_path: str
    targets: list[LayerTarget]
    extra: dict[str, Any] = field(default_factory=dict)


class SearchSpace(Protocol):
    name: str

    def sample(self, rng: random.Random) -> dict[str, Any]: ...

    def describe(self) -> dict[str, Any]: ...


class SearchAdapter(Protocol):
    name: str
    benchmark_adapter: BenchmarkAdapter

    def register_search_args(self, parser: argparse.ArgumentParser) -> None: ...

    def normalize_search_args(self, args: argparse.Namespace) -> None: ...

    def build_search_context(self, args: argparse.Namespace) -> SearchContext: ...

    def build_default_search_space(
        self,
        args: argparse.Namespace,
        context: SearchContext,
    ) -> SearchSpace: ...

    def build_payload_metadata(
        self,
        args: argparse.Namespace,
        context: SearchContext,
        search_space: SearchSpace,
    ) -> dict[str, Any]: ...

    def get_fixed_candidate(
        self,
        args: argparse.Namespace,
        context: SearchContext,
    ) -> Optional[dict[str, Any]]: ...

    def validate_candidate(
        self,
        args: argparse.Namespace,
        context: SearchContext,
        candidate: dict[str, Any],
    ) -> None: ...

    def materialize_candidate(
        self,
        args: argparse.Namespace,
        context: SearchContext,
        candidate: dict[str, Any],
    ) -> dict[str, Any]: ...

    def candidate_signature(
        self,
        context: SearchContext,
        materialized_candidate: dict[str, Any],
    ) -> tuple[Any, ...]: ...

    def apply_candidate_to_args(
        self,
        args: argparse.Namespace,
        context: SearchContext,
        materialized_candidate: dict[str, Any],
    ) -> argparse.Namespace: ...

    def validate_trial_metrics(
        self,
        args: argparse.Namespace,
        context: SearchContext,
        runtime_name: str,
        materialized_candidate: Optional[dict[str, Any]],
        metrics: dict[str, Any],
    ) -> None: ...


class SearchStrategy(Protocol):
    name: str

    def register_strategy_args(self, parser: argparse.ArgumentParser) -> None: ...

    def initialize(
        self,
        args: argparse.Namespace,
        context: SearchContext,
        search_space: SearchSpace,
        history: list[dict[str, Any]],
        state: Optional[dict[str, Any]] = None,
    ) -> None: ...

    def propose(self) -> Optional[dict[str, Any]]: ...

    def observe(self, trial_result: dict[str, Any]) -> None: ...

    def should_stop(self) -> bool: ...

    def export_state(self) -> dict[str, Any]: ...


class SearchObjective(Protocol):
    name: str

    def evaluate(
        self,
        baseline_metrics: dict[str, Any],
        trial_metrics: dict[str, Any],
    ) -> dict[str, Any]: ...


class SearchTrialRecord(TypedDict, total=False):
    trial_idx: int
    status: str
    candidate: dict[str, Any]
    materialized_candidate: dict[str, Any]
    candidate_signature: Any
    metrics: dict[str, Any]
    objective: dict[str, Any]
    error: str


class SearchRunPayload(TypedDict, total=False):
    benchmark_adapter: str
    search_adapter: str
    strategy_name: str
    strategy_state: dict[str, Any]
    objective_name: str
    evaluator_name: str
    search_space_spec: dict[str, Any]
    search_context_summary: dict[str, Any]
    baseline: dict[str, Any]
    trials: list[SearchTrialRecord]
    pareto_front_indices: list[int]
