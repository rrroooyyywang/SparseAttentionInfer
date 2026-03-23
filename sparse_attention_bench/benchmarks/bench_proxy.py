"""
Proxy profiler CLI — drop-in replacement for profiling/sparse_decoder_prof.py
using the modular package.

Usage:
    python -m sparse_attention_bench.benchmarks.bench_proxy --gpu-profile rtx_4090 --no-show
    python -m sparse_attention_bench.benchmarks.bench_proxy --list-gpu-profiles
"""
import argparse
from pathlib import Path

from sparse_attention_bench.paths import OUTPUTS_DIR
from sparse_attention_bench.analytical.config import DecoderConfig
from sparse_attention_bench.analytical.evaluator import (
    collect_accuracy_curves,
    collect_phase_speedup_curves,
)
from sparse_attention_bench.analytical.gpu_profiles import (
    DEFAULT_GPU_PROFILE_PATH,
    load_gpu_heuristic,
    load_gpu_profile_catalog,
)
from sparse_attention_bench.analytical.plotting import plot_accuracy_summary, plot_phase_speedup
from sparse_attention_bench.analytical.utils import (
    TerminalProgressBar,
    build_proxy_profile_payload,
    save_proxy_profile_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Sparse decoder proxy profiler (modular).")
    parser.add_argument("--gpu-profile", type=str, default=None)
    parser.add_argument("--gpu-profile-path", type=str, default=str(DEFAULT_GPU_PROFILE_PATH))
    parser.add_argument("--list-gpu-profiles", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save outputs. Default: outputs/ at project root.",
    )
    # Sweep params (expose what was hardcoded in original)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--num-trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    default_profile_name, available_profiles = load_gpu_profile_catalog(args.gpu_profile_path)

    if args.list_gpu_profiles:
        print(f"GPU profile file: {args.gpu_profile_path}")
        print(f"default_profile: {default_profile_name}")
        print("-" * 72)
        for name in sorted(available_profiles):
            desc = available_profiles[name].get("description", "")
            print(f"{name:<16s} {desc}")
        raise SystemExit(0)

    gpu_heuristic = load_gpu_heuristic(
        profile_name=args.gpu_profile, profile_path=args.gpu_profile_path
    )

    seq_lens = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    percentage_list = [0.001, 0.05, 0.1, 0.5, 0.8, 0.9, 0.99]
    sparse_modes = ["top-k", "bigbird"]
    phases = ["prefill", "decode"]
    num_trials = args.num_trials
    batch_size = args.batch_size
    seed = args.seed
    verbose = not args.quiet

    if verbose:
        print("Note: gpu_speedup_est is a heuristic proxy estimate, not a real kernel benchmark.")
        print(f"profile     : {gpu_heuristic.profile_name}")
        print(f"description : {gpu_heuristic.description}")
        print(f"profile_toml: {args.gpu_profile_path}")
        print("=" * 72)

    total_trials = len(seq_lens) * len(sparse_modes) * num_trials
    progress_bar = TerminalProgressBar(total=total_trials, unit="trials", enabled=True)
    progress_bar.render()

    rel_error_curves, kl_div_curves, top1_match_curves, accuracy_records = collect_accuracy_curves(
        seq_lens=seq_lens, percentage_list=percentage_list, sparse_modes=sparse_modes,
        batch_size=batch_size, seed=seed, num_trials=num_trials, gpu=gpu_heuristic,
        phase="prefill", verbose=verbose, progress_bar=progress_bar,
    )
    progress_bar.close()

    speed_cfg = DecoderConfig(max_seq_len=max(seq_lens))
    phase_speedup_curves, speedup_records = collect_phase_speedup_curves(
        cfg=speed_cfg, batch_size=batch_size, seq_lens=seq_lens,
        percentage_list=percentage_list, sparse_modes=sparse_modes,
        phases=phases, gpu=gpu_heuristic,
    )

    accuracy_fig = plot_accuracy_summary(
        seq_lens=seq_lens, percentage_list=percentage_list, sparse_modes=sparse_modes,
        rel_error_curves=rel_error_curves, kl_div_curves=kl_div_curves,
        top1_match_curves=top1_match_curves, profile_name=gpu_heuristic.profile_name,
    )
    speed_figures = {
        phase: plot_phase_speedup(
            seq_lens=seq_lens, percentage_list=percentage_list, sparse_modes=sparse_modes,
            speedup_curves=phase_speedup_curves[phase], phase=phase,
            profile_name=gpu_heuristic.profile_name,
        )
        for phase in phases
    }

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = OUTPUTS_DIR / "proxy"
    output_dir.mkdir(parents=True, exist_ok=True)

    mode_tag = "-".join(sparse_modes)
    profile_tag = gpu_heuristic.profile_name.replace("/", "-")

    accuracy_output_path = output_dir / (
        f"sparse_decoder_accuracy_{mode_tag}_{profile_tag}_"
        f"seq{min(seq_lens)}-{max(seq_lens)}_trial{num_trials}.png"
    )
    accuracy_fig.savefig(accuracy_output_path, dpi=200, bbox_inches="tight")
    print(f"Saved accuracy plot to: {accuracy_output_path}")

    speedup_output_paths = {}
    for phase, figure in speed_figures.items():
        speed_output_path = output_dir / (
            f"sparse_decoder_speedup_{phase}_{mode_tag}_{profile_tag}_"
            f"seq{min(seq_lens)}-{max(seq_lens)}.png"
        )
        figure.savefig(speed_output_path, dpi=200, bbox_inches="tight")
        speedup_output_paths[phase] = speed_output_path
        print(f"Saved {phase} speedup plot to: {speed_output_path}")

    json_output_path = output_dir / (
        f"sparse_decoder_proxy_profile_{mode_tag}_{profile_tag}_"
        f"seq{min(seq_lens)}-{max(seq_lens)}_trial{num_trials}.json"
    )
    payload = build_proxy_profile_payload(
        gpu=gpu_heuristic, gpu_profile_path=args.gpu_profile_path,
        batch_size=batch_size, seed=seed, num_trials=num_trials,
        seq_lens=seq_lens, percentage_list=percentage_list,
        sparse_modes=sparse_modes, phases=phases,
        accuracy_records=accuracy_records, speedup_records=speedup_records,
        artifacts={
            "accuracy_plot": accuracy_output_path,
            "speedup_plots": speedup_output_paths,
        },
    )
    save_proxy_profile_json(json_output_path, payload)
    print(f"Saved proxy profile JSON to: {json_output_path}")

    import matplotlib.pyplot as plt
    if args.no_show:
        plt.close(accuracy_fig)
        for fig in speed_figures.values():
            plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
