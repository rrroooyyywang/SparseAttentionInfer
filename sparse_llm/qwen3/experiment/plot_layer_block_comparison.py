from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from sparse_llm.common.io.json_io import load_json


EXPERIMENT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = EXPERIMENT_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
DEFAULT_OUTPUT_PLOT = str(PLOTS_DIR / "qwen3_layer_block_comparison.png")


def _discover_summary_jsons() -> list[str]:
    metrics_dir = OUTPUTS_DIR / "metrics"
    summary_paths: list[tuple[int, Path]] = []
    for path in metrics_dir.glob("qwen3_layer_block_sparsity*.json"):
        if path.stem == "qwen3_layer_block_sparsity":
            summary_paths.append((1, path))
            continue
        prefix = "qwen3_layer_block_sparsity_"
        if not path.stem.startswith(prefix):
            continue
        suffix = path.stem[len(prefix):]
        if suffix.isdigit():
            summary_paths.append((int(suffix), path))
    return [str(path) for _, path in sorted(summary_paths, key=lambda item: item[0])]


DEFAULT_SUMMARY_JSONS = _discover_summary_jsons()


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare multiple Qwen3 layer-block sparsity experiment summaries in one figure "
            "with separate colors and a red dashed split between early and late layers."
        )
    )
    parser.add_argument(
        "--summary-json",
        dest="summary_jsons",
        action="append",
        default=None,
        help=(
            "Summary json to include. Repeat this flag to compare specific files. "
            "If omitted, all qwen3_layer_block_sparsity*.json files in outputs/metrics are used."
        ),
    )
    parser.add_argument(
        "--label",
        dest="labels",
        action="append",
        default=None,
        help="Legend label. Repeat in the same order as --summary-json.",
    )
    parser.add_argument("--output-plot", type=str, default=DEFAULT_OUTPUT_PLOT)
    parser.add_argument("--speedup-key", type=str, default="generation_decode_speedup")
    parser.add_argument("--ppl-ratio-key", type=str, default="ppl_ratio")
    parser.add_argument(
        "--split-after-layer",
        type=int,
        default=17,
        help="Draw a red dashed vertical line after the x-category whose ending layer matches this value.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Qwen3 Layer-Block Comparison",
    )
    return parser


def _default_label(payload: dict[str, Any], fallback_path: str) -> str:
    target_sparsity = payload.get("target_sparsity")
    if target_sparsity is not None:
        return f"sparsity={float(target_sparsity):g}"
    return Path(fallback_path).stem


def _load_rows(summary_json_path: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = load_json(summary_json_path)
    rows = payload.get("experiments", [])
    if not isinstance(rows, list):
        raise ValueError(f"`experiments` is missing or not a list in {summary_json_path}.")
    rows = sorted(rows, key=lambda row: int(row["layer_start"]))
    return payload, rows


def _layer_label(row: dict[str, Any]) -> str:
    return f"{int(row['layer_start'])}-{int(row['layer_end'])}"


def _row_map(rows: list[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    return {
        (int(row["layer_start"]), int(row["layer_end"])): row
        for row in rows
    }


def _offsets(num_series: int) -> list[float]:
    if num_series <= 1:
        return [0.0]
    step = 0.18
    midpoint = (num_series - 1) / 2.0
    return [(index - midpoint) * step for index in range(num_series)]


def plot_comparison(
    summary_json_paths: list[str],
    *,
    output_plot: str,
    speedup_key: str,
    ppl_ratio_key: str,
    split_after_layer: int,
    title: str,
    labels: list[str] | None = None,
) -> str:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    payloads_and_rows: list[tuple[str, str, dict[str, Any], list[dict[str, Any]]]] = []
    labels = labels or []
    if len(labels) > len(summary_json_paths):
        raise ValueError("Received more --label values than --summary-json values.")

    for index, summary_json_path in enumerate(summary_json_paths):
        payload, rows = _load_rows(summary_json_path)
        payloads_and_rows.append(
            (
                summary_json_path,
                labels[index] if index < len(labels) and labels[index] else _default_label(payload, summary_json_path),
                payload,
                rows,
            )
        )

    if len(payloads_and_rows) < 2:
        raise ValueError("At least two summary json files are required for comparison.")

    ordered_keys = sorted({
        key
        for _, _, _, rows in payloads_and_rows
        for key in _row_map(rows).keys()
    }, key=lambda item: item[0])
    x_positions = list(range(len(ordered_keys)))
    x_labels = [f"{start}-{end}" for start, end in ordered_keys]

    def _series(
        row_map: dict[tuple[int, int], dict[str, Any]],
        metric_key: str,
    ) -> list[float]:
        values: list[float] = []
        for key in ordered_keys:
            row = row_map.get(key)
            if row is None or row.get(metric_key) is None:
                values.append(float("nan"))
            else:
                values.append(float(row[metric_key]))
        return values

    speedup_series: list[list[float]] = []
    ppl_series: list[list[float]] = []
    legend_labels: list[str] = []
    for _, label, _, rows in payloads_and_rows:
        row_map = _row_map(rows)
        speedup_series.append(_series(row_map, speedup_key))
        ppl_series.append(_series(row_map, ppl_ratio_key))
        legend_labels.append(label)

    separator_x = None
    for idx, (_, layer_end) in enumerate(ordered_keys):
        if int(layer_end) == split_after_layer:
            separator_x = idx + 0.5
            break

    output_path = Path(output_plot)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    colors = (
        "#1F6AA5",
        "#D97706",
        "#2E8B57",
        "#8B5CF6",
        "#C2410C",
        "#0F766E",
    )
    offsets = _offsets(len(speedup_series))

    for axis, series_collection, ylabel in (
        (axes[0], speedup_series, "Generation decode speedup"),
        (axes[1], ppl_series, "PPL ratio (sparse / dense)"),
    ):
        for index, values in enumerate(series_collection):
            axis.scatter(
                [x + offsets[index] for x in x_positions],
                values,
                color=colors[index % len(colors)],
                s=75,
                label=legend_labels[index],
                zorder=3,
            )
        axis.axhline(1.0, color="#C9CED6", linestyle="--", linewidth=1.0, zorder=1)
        if separator_x is not None:
            axis.axvline(separator_x, color="red", linestyle="--", linewidth=1.2, zorder=2)
        axis.grid(True, linestyle="--", linewidth=0.7, alpha=0.25)
        axis.set_ylabel(ylabel)

    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels(x_labels)
    axes[1].set_xlabel("Active layer block")
    axes[0].legend(loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def main() -> None:
    parser = _build_argparser()
    args = parser.parse_args()
    summary_json_paths = args.summary_jsons or DEFAULT_SUMMARY_JSONS
    plot_path = plot_comparison(
        summary_json_paths,
        output_plot=args.output_plot,
        speedup_key=args.speedup_key,
        ppl_ratio_key=args.ppl_ratio_key,
        split_after_layer=args.split_after_layer,
        title=args.title,
        labels=args.labels,
    )
    print(f"[saved] plot_path={plot_path}")


if __name__ == "__main__":
    main()
