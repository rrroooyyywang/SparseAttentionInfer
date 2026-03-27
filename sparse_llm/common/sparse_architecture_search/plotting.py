from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from sparse_llm.common.sparse_architecture_search.results import trial_coords


def _trial_color_value(trial: dict[str, Any]) -> float:
    materialized = trial.get("materialized_candidate")
    if isinstance(materialized, dict):
        summary = materialized.get("summary")
        if isinstance(summary, dict) and summary.get("mean_sparsity") is not None:
            return float(summary["mean_sparsity"])
    candidate = trial.get("candidate")
    if isinstance(candidate, dict) and candidate.get("mean_sparsity") is not None:
        return float(candidate["mean_sparsity"])
    return 0.0


def plot_search_results(
    payload: dict[str, Any],
    output_plot_path: str,
    *,
    title: str = "Sparse Architecture Search",
) -> str:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    trials = payload.get("trials", [])
    valid_trials = [
        trial
        for trial in trials
        if trial.get("status") == "ok" and trial_coords(trial) != (None, None)
    ]
    if not valid_trials:
        raise ValueError("No successful trials were found; cannot plot tuning results.")

    x_values = [trial_coords(trial)[0] for trial in valid_trials]
    y_values = [trial_coords(trial)[1] for trial in valid_trials]
    colors = [_trial_color_value(trial) for trial in valid_trials]

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        x_values,
        y_values,
        c=colors,
        cmap="viridis",
        s=60,
        alpha=0.9,
        edgecolors="black",
        linewidths=0.4,
    )
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Mean sampled sparsity")

    ax.scatter(
        [1.0],
        [1.0],
        marker="*",
        s=220,
        color="#E45756",
        edgecolors="black",
        linewidths=0.8,
        label="Dense baseline",
        zorder=5,
    )

    pareto_indices = payload.get("pareto_front_indices", [])
    if pareto_indices:
        pareto_trials = [
            trials[idx]
            for idx in pareto_indices
            if trials[idx].get("status") == "ok" and trial_coords(trials[idx]) != (None, None)
        ]
        if pareto_trials:
            ax.plot(
                [trial_coords(trial)[0] for trial in pareto_trials],
                [trial_coords(trial)[1] for trial in pareto_trials],
                color="#4C78A8",
                linewidth=1.5,
                label="Pareto front",
            )
            for trial in pareto_trials:
                ax.annotate(
                    f"#{trial['trial_idx']}",
                    trial_coords(trial),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=8,
                )

    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1.0)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
    ax.set_xlabel("baseline PPL / sample PPL")
    ax.set_ylabel("speedup = sample tok/s / baseline tok/s")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)

    plot_path = Path(output_plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return str(plot_path)
