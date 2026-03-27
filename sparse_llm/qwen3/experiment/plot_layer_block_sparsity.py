from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from sparse_llm.common.io.json_io import load_json, write_json


EXPERIMENT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = EXPERIMENT_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
DEFAULT_RESULTS_JSON = str(OUTPUTS_DIR / "metrics" / "qwen3_layer_block_sparsity.json")


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Qwen3 layer-block sparsity experiments as speedup vs ppl-ratio points "
            "without numeric point labels."
        )
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default=DEFAULT_RESULTS_JSON,
        help="Summary JSON produced by the layer-block sparsity experiment runner.",
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default=None,
        help="Optional output figure path. Defaults next to the experiment plots directory.",
    )
    parser.add_argument(
        "--speedup-key",
        type=str,
        default="generation_decode_speedup",
        choices=[
            "generation_decode_speedup",
            "generation_prefill_speedup",
            "perplexity_eval_speedup",
        ],
        help="Which speedup field to place on the x-axis.",
    )
    parser.add_argument(
        "--ppl-ratio-key",
        type=str,
        default="ppl_ratio",
        choices=["ppl_ratio"],
        help="Which perplexity-ratio field to place on the y-axis.",
    )
    parser.add_argument(
        "--swap-axes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Place ppl ratio on the x-axis and speedup on the y-axis.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Qwen3 Layer-Block Sparsity Trend",
    )
    parser.add_argument(
        "--connect-points",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to connect points in experiment order.",
    )
    parser.add_argument(
        "--annotate-layers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Annotate each point with its layer range.",
    )
    return parser


def _default_plot_path(summary_json_path: str, speedup_key: str) -> str:
    summary_path = Path(summary_json_path)
    return str(PLOTS_DIR / f"{summary_path.stem}_{speedup_key}_vs_ppl_ratio.png")


def _speedup_axis_label(speedup_key: str) -> str:
    if speedup_key == "generation_prefill_speedup":
        return "Speedup (generation prefill)"
    if speedup_key == "perplexity_eval_speedup":
        return "Speedup (perplexity eval)"
    return "Speedup (generation decode)"


def _valid_experiments(
    experiments: list[dict[str, Any]],
    *,
    speedup_key: str,
    ppl_ratio_key: str,
) -> list[dict[str, Any]]:
    valid: list[dict[str, Any]] = []
    for experiment in experiments:
        if experiment.get(speedup_key) is None or experiment.get(ppl_ratio_key) is None:
            continue
        valid.append(experiment)
    if not valid:
        raise ValueError(
            f"No experiment rows contained both {speedup_key!r} and {ppl_ratio_key!r}."
        )
    return valid


def plot_summary_json(
    summary_json_path: str,
    *,
    output_plot_path: str | None = None,
    speedup_key: str = "generation_decode_speedup",
    ppl_ratio_key: str = "ppl_ratio",
    title: str = "Qwen3 Layer-Block Sparsity Trend",
    connect_points: bool = False,
    swap_axes: bool = True,
    annotate_layers: bool = True,
) -> str:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    payload = load_json(summary_json_path)
    experiments = payload.get("experiments", [])
    if not isinstance(experiments, list):
        raise ValueError("Expected `experiments` to be a list in the summary JSON.")
    rows = _valid_experiments(
        experiments,
        speedup_key=speedup_key,
        ppl_ratio_key=ppl_ratio_key,
    )

    speedup_values = [float(row[speedup_key]) for row in rows]
    ppl_ratio_values = [float(row[ppl_ratio_key]) for row in rows]
    if swap_axes:
        x_values = ppl_ratio_values
        y_values = speedup_values
        x_label = "PPL ratio (sparse / dense)"
        y_label = _speedup_axis_label(speedup_key)
    else:
        x_values = speedup_values
        y_values = ppl_ratio_values
        x_label = _speedup_axis_label(speedup_key)
        y_label = "PPL ratio (sparse / dense)"

    if output_plot_path is None:
        output_plot_path = _default_plot_path(summary_json_path, speedup_key)
    output_path = Path(output_plot_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    if connect_points and len(rows) > 1:
        ax.plot(
            x_values,
            y_values,
            color="#9BB3C9",
            linewidth=1.5,
            alpha=0.9,
            zorder=1,
        )

    ax.scatter(
        x_values,
        y_values,
        s=90,
        color="#1F6AA5",
        edgecolors="white",
        linewidths=0.8,
        alpha=0.95,
        zorder=2,
    )

    if annotate_layers:
        for row, x_value, y_value in zip(rows, x_values, y_values):
            layer_start = row.get("layer_start")
            layer_end = row.get("layer_end")
            label = (
                f"layers {layer_start}-{layer_end}"
                if layer_start is not None and layer_end is not None
                else str(row.get("label", ""))
            )
            ax.annotate(
                label,
                (x_value, y_value),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=9,
                color="#284B63",
            )

    ax.axvline(1.0, color="#C9CED6", linestyle="--", linewidth=1.0, zorder=0)
    ax.axhline(1.0, color="#C9CED6", linestyle="--", linewidth=1.0, zorder=0)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.25)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    x_margin = max(0.01, (max(x_values) - min(x_values)) * 0.15 if len(x_values) > 1 else 0.05)
    y_margin = max(0.005, (max(y_values) - min(y_values)) * 0.2 if len(y_values) > 1 else 0.02)
    ax.set_xlim(min(x_values) - x_margin, max(x_values) + x_margin)
    ax.set_ylim(min(y_values) - y_margin, max(y_values) + y_margin)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    payload["plot_path"] = str(output_path)
    payload["plot_config"] = {
        "speedup_key": speedup_key,
        "ppl_ratio_key": ppl_ratio_key,
        "title": title,
        "connect_points": connect_points,
        "swap_axes": swap_axes,
        "annotate_layers": annotate_layers,
    }
    write_json(payload, summary_json_path)
    return str(output_path)


def main() -> None:
    parser = _build_argparser()
    args = parser.parse_args()
    plot_path = plot_summary_json(
        args.summary_json,
        output_plot_path=args.output_plot,
        speedup_key=args.speedup_key,
        ppl_ratio_key=args.ppl_ratio_key,
        title=args.title,
        connect_points=args.connect_points,
        swap_axes=args.swap_axes,
        annotate_layers=args.annotate_layers,
    )
    print(f"[saved] plot_path={plot_path}")


if __name__ == "__main__":
    main()
