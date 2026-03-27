from __future__ import annotations

from typing import Any, Optional

from sparse_llm.common.io.json_io import load_json, write_json


def write_search_results(payload: dict[str, Any], output_path: str) -> None:
    write_json(payload, output_path)


def load_search_results(output_path: str) -> Optional[dict[str, Any]]:
    try:
        return load_json(output_path)
    except FileNotFoundError:
        return None


def _freeze_json_value(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_freeze_json_value(item) for item in value)
    if isinstance(value, dict):
        return tuple(
            sorted((key, _freeze_json_value(item)) for key, item in value.items())
        )
    return value


def next_trial_index(trials: list[dict[str, Any]]) -> int:
    if not trials:
        return 0
    return 1 + max(int(trial.get("trial_idx", -1)) for trial in trials)


def collect_successful_signatures(trials: list[dict[str, Any]]) -> set[Any]:
    signatures: set[Any] = set()
    for trial in trials:
        if trial.get("status") != "ok":
            continue
        signature = trial.get("candidate_signature")
        if signature is None:
            continue
        signatures.add(_freeze_json_value(signature))
    return signatures


def trial_coords(trial: dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
    objective = trial.get("objective")
    if isinstance(objective, dict):
        pareto_coords = objective.get("pareto_coords")
        if isinstance(pareto_coords, dict):
            x_value = pareto_coords.get("x")
            y_value = pareto_coords.get("y")
            return x_value, y_value

    x_value = trial.get("baseline_over_ppl")
    if x_value is None and trial.get("ppl_ratio") is not None:
        ppl_ratio = float(trial["ppl_ratio"])
        x_value = None if ppl_ratio == 0 else 1.0 / ppl_ratio

    y_value = trial.get("speedup")
    if y_value is None and trial.get("one_over_speedup") is not None:
        one_over_speedup = float(trial["one_over_speedup"])
        y_value = None if one_over_speedup == 0 else 1.0 / one_over_speedup

    return x_value, y_value


def pareto_front_indices(trials: list[dict[str, Any]]) -> list[int]:
    valid_trials = [
        (index, trial)
        for index, trial in enumerate(trials)
        if trial.get("status") == "ok" and trial_coords(trial) != (None, None)
    ]
    pareto: list[int] = []
    for index, trial in valid_trials:
        x_i, y_i = trial_coords(trial)
        if x_i is None or y_i is None:
            continue
        dominated = False
        for other_index, other in valid_trials:
            if other_index == index:
                continue
            x_j, y_j = trial_coords(other)
            if x_j is None or y_j is None:
                continue
            if (x_j >= x_i and y_j >= y_i) and (x_j > x_i or y_j > y_i):
                dominated = True
                break
        if not dominated:
            pareto.append(index)
    pareto.sort(key=lambda idx: trial_coords(trials[idx]))
    return pareto
