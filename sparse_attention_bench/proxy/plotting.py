"""Plotting utilities for proxy profiler results."""
import itertools

import torch


def _auto_linestyles(sparse_modes):
    styles = ["-", "--", "-.", ":"]
    return {mode: styles[i % len(styles)] for i, mode in enumerate(sparse_modes)}


def plot_accuracy_summary(
    seq_lens,
    percentage_list,
    sparse_modes,
    rel_error_curves,
    kl_div_curves,
    top1_match_curves,
    profile_name: str,
):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    linestyles = _auto_linestyles(sparse_modes)

    for sparse_mode in sparse_modes:
        for percentage in percentage_list:
            rel_mean = torch.tensor(rel_error_curves[sparse_mode][percentage]["mean"], dtype=torch.float64)
            rel_std = torch.tensor(rel_error_curves[sparse_mode][percentage]["std"], dtype=torch.float64)
            kl_mean = torch.tensor(kl_div_curves[sparse_mode][percentage]["mean"], dtype=torch.float64)
            kl_std = torch.tensor(kl_div_curves[sparse_mode][percentage]["std"], dtype=torch.float64)
            top1_mean = torch.tensor(top1_match_curves[sparse_mode][percentage]["mean"], dtype=torch.float64)
            top1_std = torch.tensor(top1_match_curves[sparse_mode][percentage]["std"], dtype=torch.float64)
            label = f"{sparse_mode}, keep={percentage:.0%}"
            ls = linestyles[sparse_mode]

            axes[0].plot(seq_lens, rel_mean.tolist(), marker="o", linestyle=ls, label=label)
            axes[0].fill_between(
                seq_lens,
                torch.clamp(rel_mean - rel_std, min=1e-12).tolist(),
                torch.clamp(rel_mean + rel_std, min=1e-12).tolist(),
                alpha=0.10,
            )
            axes[1].plot(seq_lens, kl_mean.tolist(), marker="o", linestyle=ls, label=label)
            axes[1].fill_between(
                seq_lens,
                torch.clamp(kl_mean - kl_std, min=0.0).tolist(),
                (kl_mean + kl_std).tolist(),
                alpha=0.10,
            )
            axes[2].plot(seq_lens, top1_mean.tolist(), marker="o", linestyle=ls, label=label)
            axes[2].fill_between(
                seq_lens,
                torch.clamp(top1_mean - top1_std, min=0.0, max=1.0).tolist(),
                torch.clamp(top1_mean + top1_std, min=0.0, max=1.0).tolist(),
                alpha=0.10,
            )

    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Sequence Length")
    axes[0].set_ylabel("Relative Error")
    axes[0].set_title("Relative Error vs Sequence Length")
    axes[0].grid(True, which="both", linestyle="--", alpha=0.4)

    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("Sequence Length")
    axes[1].set_ylabel("KL Divergence")
    axes[1].set_title("KL Divergence")
    axes[1].grid(True, which="both", linestyle="--", alpha=0.4)
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    axes[2].set_xscale("log", base=2)
    axes[2].set_xlabel("Sequence Length")
    axes[2].set_ylabel("Top-1 Match Rate")
    axes[2].set_title("Top-1 Match Rate")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].grid(True, which="both", linestyle="--", alpha=0.4)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)))
    fig.suptitle(f"Sparse Decoder Accuracy Summary ({profile_name}, prefill eval)")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return fig


def plot_phase_speedup(
    seq_lens,
    percentage_list,
    sparse_modes,
    speedup_curves,
    phase: str,
    profile_name: str,
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    linestyles = _auto_linestyles(sparse_modes)

    for sparse_mode in sparse_modes:
        for percentage in percentage_list:
            speed_mean = torch.tensor(speedup_curves[sparse_mode][percentage]["mean"], dtype=torch.float64)
            speed_std = torch.tensor(speedup_curves[sparse_mode][percentage]["std"], dtype=torch.float64)
            label = f"{sparse_mode}, keep={percentage:.0%}"
            ax.plot(seq_lens, speed_mean.tolist(), marker="o", linestyle=linestyles[sparse_mode], label=label)
            ax.fill_between(
                seq_lens,
                torch.clamp(speed_mean - speed_std, min=0.0).tolist(),
                (speed_mean + speed_std).tolist(),
                alpha=0.10,
            )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Estimated GPU Speedup (x)")
    ax.set_title(f"Estimated NVIDIA GPU Speedup ({phase}, {profile_name})")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    return fig


def _config_group_key(r: dict) -> tuple:
    """Stable grouping key: does not include seq_len or keep_ratio."""
    return (
        r.get("pattern_type", ""),
        r.get("topk"),
        r.get("window_size"),
        r.get("block_size"),
        r.get("mode", ""),
    )


def _config_label(r: dict) -> str:
    pattern = r.get("pattern_type", "?")
    topk = r.get("topk")
    window_size = r.get("window_size")
    block_size = r.get("block_size")
    if topk is not None:
        return f"{pattern} k={topk}"
    if window_size is not None:
        return f"{pattern} w={window_size}"
    if block_size is not None:
        return f"{pattern} b={block_size}"
    return pattern


def plot_sweep_latency(results: list[dict], x_key: str = "seq_len", y_key: str = "total_time_ms_mean"):
    """
    Plot latency curves from sweep results.

    For each pattern type and mode:
    - Draws a mean line across all hyperparameter variants (e.g. all topk values).
    - Shades the band from min to max to show the range of configurations.
    - Creates one subplot per mode (prefill / decode).
    """
    import matplotlib.pyplot as plt

    # Determine modes and pattern types
    modes = sorted({r.get("mode", "") for r in results if r.get("mode")}) or [""]
    pattern_types = sorted({r.get("pattern_type", "") for r in results})

    # For each (pattern_type, mode, x_key value): collect all y values across variants
    # Structure: {(pattern, mode): {x_val: [y1, y2, ...]}}
    buckets: dict[tuple, dict] = {}
    for r in results:
        if y_key not in r or x_key not in r:
            continue
        key = (r.get("pattern_type", ""), r.get("mode", ""))
        x_val = r[x_key]
        buckets.setdefault(key, {}).setdefault(x_val, []).append(r[y_key])

    # One subplot per mode
    n_modes = len(modes)
    fig, axes = plt.subplots(1, n_modes, figsize=(7 * n_modes, 5), squeeze=False)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for col, mode in enumerate(modes):
        ax = axes[0][col]
        for idx, pattern in enumerate(pattern_types):
            per_x = buckets.get((pattern, mode))
            if not per_x:
                continue

            xs = sorted(per_x)
            means = [sum(per_x[x]) / len(per_x[x]) for x in xs]
            mins  = [min(per_x[x]) for x in xs]
            maxs  = [max(per_x[x]) for x in xs]
            color = colors[idx % len(colors)]

            ax.plot(xs, means, marker="o", label=pattern, color=color)
            ax.fill_between(xs, mins, maxs, alpha=0.20, color=color)

        ax.set_xscale("log", base=2)
        ax.set_xlabel(x_key)
        ax.set_ylabel(y_key)
        ax.set_title(f"{y_key} — {mode}" if mode else y_key)
        ax.legend(loc="upper left")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

    fig.tight_layout()
    return fig
