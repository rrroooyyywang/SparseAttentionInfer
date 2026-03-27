import os
from typing import Optional

from sparse_llm.qwen3.metrics_io import default_plot_path, load_metrics_json


def plot_metrics_json(
    metrics_json_path: str,
    output_plot_path: Optional[str] = None,
) -> str:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    payload = load_metrics_json(metrics_json_path)

    if output_plot_path is None:
        output_plot_path = default_plot_path(metrics_json_path)

    if "dense" in payload and "sparse" in payload:
        series = [payload["dense"], payload["sparse"]]
        labels = ["Dense", "Sparse"]
        colors = ["#4C78A8", "#E45756"]
        benchmark_type = payload["dense"].get("benchmark_type", "generation")
        title = "Qwen3 Dense vs Sparse Benchmark"
    else:
        series = [payload]
        label = payload.get("runtime_name", "Run").capitalize()
        labels = [label]
        colors = ["#4C78A8"]
        benchmark_type = payload.get("benchmark_type", "generation")
        title = f"Qwen3 {label} Benchmark"

    def _safe_number(value) -> float:
        return 0.0 if value is None else float(value)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    if benchmark_type == "perplexity":
        perplexities = [_safe_number(item["perplexity"]) for item in series]
        axes[0].bar(labels, perplexities, color=colors[: len(series)])
        axes[0].set_title("Perplexity")
        axes[0].set_ylabel("ppl")
        for i, value in enumerate(perplexities):
            axes[0].text(i, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

        avg_nlls = [_safe_number(item["avg_nll"]) for item in series]
        axes[1].bar(labels, avg_nlls, color=colors[: len(series)])
        axes[1].set_title("Average NLL")
        axes[1].set_ylabel("nats/token")
        for i, value in enumerate(avg_nlls):
            axes[1].text(i, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

        token_tps = [_safe_number(item["tokens_per_second"]) for item in series]
        axes[2].bar(labels, token_tps, color=colors[: len(series)])
        axes[2].set_title("Evaluation Throughput")
        axes[2].set_ylabel("tokens/s")
        for i, (raw_value, plot_value) in enumerate(
            zip([item["tokens_per_second"] for item in series], token_tps)
        ):
            label_text = "n/a" if raw_value is None else f"{plot_value:.1f}"
            axes[2].text(i, plot_value, label_text, ha="center", va="bottom", fontsize=9)

        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(output_plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return output_plot_path

    first_token_ms = [item["first_token_time_s"] * 1000.0 for item in series]
    axes[0].bar(labels, first_token_ms, color=colors[: len(series)])
    axes[0].set_title("First Token Time")
    axes[0].set_ylabel("ms")
    for i, value in enumerate(first_token_ms):
        axes[0].text(i, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    decode_tps = [_safe_number(item["decode_tokens_per_second"]) for item in series]
    axes[1].bar(labels, decode_tps, color=colors[: len(series)])
    axes[1].set_title("Decode Throughput")
    axes[1].set_ylabel("tokens/s")
    for i, (raw_value, plot_value) in enumerate(
        zip([item["decode_tokens_per_second"] for item in series], decode_tps)
    ):
        label_text = "n/a" if raw_value is None else f"{plot_value:.1f}"
        axes[1].text(i, plot_value, label_text, ha="center", va="bottom", fontsize=9)

    for label, color, item in zip(labels, colors, series):
        step_times_ms = [x * 1000.0 for x in item["decode_step_times_s"]]
        axes[2].plot(
            range(1, len(step_times_ms) + 1),
            step_times_ms,
            label=label,
            color=color,
        )
    axes[2].set_title("Per-step Decode Latency")
    axes[2].set_xlabel("Decode step")
    axes[2].set_ylabel("ms")
    if any(item["decode_step_times_s"] for item in series):
        axes[2].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_plot_path
