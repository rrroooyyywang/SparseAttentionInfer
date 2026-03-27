from __future__ import annotations

import math
from typing import Any


def _optional_float(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    return float(value)


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


class ParetoSpeedVsQualityObjective:
    name = "pareto_speed_vs_quality"

    def evaluate(
        self,
        baseline_metrics: dict[str, Any],
        trial_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        baseline_ppl = float(baseline_metrics["perplexity"])
        trial_ppl = float(trial_metrics["perplexity"])
        baseline_tps = float(baseline_metrics["tokens_per_second"])
        trial_tps = float(trial_metrics["tokens_per_second"])
        baseline_forward_tps = _optional_float(
            baseline_metrics,
            "forward_tokens_per_second",
        )
        trial_forward_tps = _optional_float(
            trial_metrics,
            "forward_tokens_per_second",
        )

        ppl_ratio = None if baseline_ppl == 0 else trial_ppl / baseline_ppl
        baseline_over_ppl = None if trial_ppl == 0 else baseline_ppl / trial_ppl
        speedup = None if baseline_tps == 0 else trial_tps / baseline_tps
        forward_speedup = _safe_ratio(trial_forward_tps, baseline_forward_tps)
        one_over_speedup = None if speedup in (None, 0) else 1.0 / speedup

        return {
            "score": None,
            "constraints": {},
            "reported_metrics": {
                "baseline_over_ppl": baseline_over_ppl,
                "speedup": speedup,
                "eval_speedup": speedup,
                "forward_speedup": forward_speedup,
                "ppl_ratio": ppl_ratio,
                "one_over_speedup": one_over_speedup,
            },
            "pareto_coords": {
                "x": baseline_over_ppl,
                "y": speedup,
            },
        }


class WeightedScalarObjective:
    name = "weighted_scalar"

    def __init__(
        self,
        *,
        speed_weight: float = 1.0,
        quality_weight: float = 1.0,
        quality_penalty: float = 20.0,
        max_ppl_ratio: float = 1.1,
    ) -> None:
        self.speed_weight = float(speed_weight)
        self.quality_weight = float(quality_weight)
        self.quality_penalty = float(quality_penalty)
        self.max_ppl_ratio = float(max_ppl_ratio)

    def evaluate(
        self,
        baseline_metrics: dict[str, Any],
        trial_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        baseline_ppl = float(baseline_metrics["perplexity"])
        trial_ppl = float(trial_metrics["perplexity"])
        baseline_tps = float(baseline_metrics["tokens_per_second"])
        trial_tps = float(trial_metrics["tokens_per_second"])
        baseline_forward_tps = _optional_float(
            baseline_metrics,
            "forward_tokens_per_second",
        )
        trial_forward_tps = _optional_float(
            trial_metrics,
            "forward_tokens_per_second",
        )

        ppl_ratio = None if baseline_ppl == 0 else trial_ppl / baseline_ppl
        baseline_over_ppl = None if trial_ppl == 0 else baseline_ppl / trial_ppl
        speedup = None if baseline_tps == 0 else trial_tps / baseline_tps
        forward_speedup = _safe_ratio(trial_forward_tps, baseline_forward_tps)
        one_over_speedup = None if speedup in (None, 0) else 1.0 / speedup

        speed_term = 0.0 if speedup in (None, 0) else self.speed_weight * math.log(speedup)
        quality_term = (
            0.0
            if baseline_over_ppl in (None, 0)
            else self.quality_weight * math.log(baseline_over_ppl)
        )
        constraint_gap = 0.0 if ppl_ratio is None else max(0.0, ppl_ratio - self.max_ppl_ratio)
        penalty_term = self.quality_penalty * constraint_gap
        score = speed_term + quality_term - penalty_term

        return {
            "score": score,
            "constraints": {
                "max_ppl_ratio": self.max_ppl_ratio,
                "constraint_gap": constraint_gap,
            },
            "reported_metrics": {
                "baseline_over_ppl": baseline_over_ppl,
                "speedup": speedup,
                "eval_speedup": speedup,
                "forward_speedup": forward_speedup,
                "ppl_ratio": ppl_ratio,
                "one_over_speedup": one_over_speedup,
                "score": score,
            },
            "pareto_coords": {
                "x": baseline_over_ppl,
                "y": speedup,
            },
        }
