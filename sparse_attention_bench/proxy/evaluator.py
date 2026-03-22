"""Accuracy evaluation and sweep orchestration for the proxy profiler."""
import torch

from sparse_attention_bench.metrics.accuracy import (
    relative_error, cosine_sim, top1_match_rate, mean_kl_divergence, summarize_scalar,
)
from sparse_attention_bench.models.proxy_models import DenseDecoder, build_sparse_from_dense
from sparse_attention_bench.proxy.config import DecoderConfig, NvidiaGpuHeuristic
from sparse_attention_bench.proxy.estimator import estimate_decoder_sparse_gpu_efficiency
from sparse_attention_bench.proxy.gpu_profiles import validate_execution_phase
from sparse_attention_bench.proxy.utils import (
    TerminalProgressBar, build_accuracy_record, build_speedup_record,
)


@torch.no_grad()
def evaluate_sparse_decoder_once(
    batch_size: int = 4,
    seq_len: int = 64,
    top_k_list=(4, 8, 16, 32, 64),
    sparse_mode: str = "top-k",
    seed: int = 42,
    gpu: NvidiaGpuHeuristic | None = None,
    phase: str = "prefill",
    device: str | None = None,
):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cfg = DecoderConfig(max_seq_len=max(seq_len, 128), sparse_mode=sparse_mode)
    if device is not None:
        cfg.device = device
    dev = torch.device(cfg.device)
    gpu_h = gpu or NvidiaGpuHeuristic()
    phase = validate_execution_phase(phase)

    dense_model = DenseDecoder(cfg).to(dev).eval()
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=dev)
    dense_logits = dense_model(input_ids)

    results = []
    for top_k in top_k_list:
        sparse_model = build_sparse_from_dense(dense_model, cfg, top_k=top_k).eval()
        sparse_logits = sparse_model(input_ids)

        speed_est = estimate_decoder_sparse_gpu_efficiency(
            cfg=cfg, batch_size=batch_size, seq_len=seq_len,
            top_k=top_k, sparse_mode=sparse_mode, gpu=gpu_h, phase=phase,
        )
        results.append({
            "phase": phase,
            "sparse_mode": sparse_mode,
            "top_k": top_k,
            "keep_ratio": speed_est["keep_ratio"],
            "rel_err": relative_error(sparse_logits, dense_logits),
            "kl_div": mean_kl_divergence(dense_logits, sparse_logits),
            "cos_sim": cosine_sim(sparse_logits, dense_logits),
            "top1_match": top1_match_rate(sparse_logits, dense_logits),
            "top1_mismatch_rate": 1.0 - top1_match_rate(sparse_logits, dense_logits),
            **speed_est,
        })
        del sparse_model, sparse_logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


@torch.no_grad()
def evaluate_sparse_decoder(
    batch_size: int = 4,
    seq_len: int = 64,
    top_k_list=(4, 8, 16, 32, 64),
    sparse_mode: str = "top-k",
    seed: int = 42,
    num_trials: int = 5,
    gpu: NvidiaGpuHeuristic | None = None,
    phase: str = "prefill",
    verbose: bool = True,
    progress_bar: TerminalProgressBar | None = None,
    device: str | None = None,
):
    cfg = DecoderConfig(max_seq_len=max(seq_len, 128), sparse_mode=sparse_mode)
    if device is not None:
        cfg.device = device
    dev = torch.device(cfg.device)
    gpu_h = gpu or NvidiaGpuHeuristic()
    phase = validate_execution_phase(phase)

    if verbose:
        lines = [
            f"device      : {dev}", f"phase       : {phase}",
            f"gpu_profile : {gpu_h.profile_name}", f"sparse_mode : {sparse_mode}",
            f"batch_size  : {batch_size}", f"seq_len     : {seq_len}",
            f"d_model     : {cfg.d_model}", f"n_heads     : {cfg.n_heads}",
            f"n_layers    : {cfg.n_layers}", f"num_trials  : {num_trials}",
            "-" * 72,
        ]
        msg = "\n".join(lines)
        (progress_bar.write(msg) if progress_bar else print(msg))

    trial_results = []
    for trial_idx in range(num_trials):
        trial_results.append(
            evaluate_sparse_decoder_once(
                batch_size=batch_size, seq_len=seq_len, top_k_list=top_k_list,
                sparse_mode=sparse_mode, seed=seed + trial_idx, gpu=gpu_h,
                phase=phase, device=cfg.device,
            )
        )
        if progress_bar is not None:
            progress_bar.update(
                description=(
                    f"phase={phase} | mode={sparse_mode} | seq_len={seq_len} "
                    f"| trial {trial_idx + 1}/{num_trials}"
                )
            )

    results = []
    for metric_idx, top_k in enumerate(top_k_list):
        samples = [trial_results[t][metric_idx] for t in range(num_trials)]
        summary = {
            "phase": phase, "sparse_mode": sparse_mode, "top_k": top_k,
            "keep_ratio": samples[0]["keep_ratio"],
            "attention_keep_ratio": samples[0]["attention_keep_ratio"],
            "feature_keep_ratio": samples[0]["feature_keep_ratio"],
            "rel_err": summarize_scalar([s["rel_err"] for s in samples]),
            "kl_div": summarize_scalar([s["kl_div"] for s in samples]),
            "cos_sim": summarize_scalar([s["cos_sim"] for s in samples]),
            "top1_match": summarize_scalar([s["top1_match"] for s in samples]),
            "top1_mismatch_rate": summarize_scalar([s["top1_mismatch_rate"] for s in samples]),
            "gpu_speedup_est": summarize_scalar([s["gpu_speedup_est"] for s in samples]),
            "dense_time_us": summarize_scalar([s["dense_time_us"] for s in samples]),
            "sparse_time_us": summarize_scalar([s["sparse_time_us"] for s in samples]),
            "selection_time_us": summarize_scalar([s["selection_time_us"] for s in samples]),
            "attention_workload_reduction": summarize_scalar(
                [s["attention_workload_reduction"] for s in samples]
            ),
            "runtime_proxy_reduction": summarize_scalar(
                [s["runtime_proxy_reduction"] for s in samples]
            ),
            "num_trials": num_trials,
        }
        if verbose:
            msg = (
                f"phase={phase:<7s} | mode={sparse_mode:<7s} | top_k={top_k:>4d} | "
                f"keep_ratio={summary['keep_ratio']:>6.2%} | "
                f"attn_keep={summary['attention_keep_ratio']:>6.2%} | "
                f"feature_keep={summary['feature_keep_ratio']:>6.2%} | "
                f"rel_err={summary['rel_err']['mean']:>10.6f}±{summary['rel_err']['std']:.2e} | "
                f"kl={summary['kl_div']['mean']:>10.6f}±{summary['kl_div']['std']:.2e} | "
                f"top1_match={summary['top1_match']['mean']:>8.4%}±{summary['top1_match']['std']:.2e} | "
                f"gpu_speedup_est={summary['gpu_speedup_est']['mean']:>6.2f}x"
            )
            (progress_bar.write(msg) if progress_bar else print(msg))
        results.append(summary)
    return results


def init_curve_bank(sparse_modes, percentage_list) -> dict:
    return {
        mode: {p: {"mean": [], "std": []} for p in percentage_list}
        for mode in sparse_modes
    }


def collect_accuracy_curves(
    seq_lens,
    percentage_list,
    sparse_modes,
    batch_size: int,
    seed: int,
    num_trials: int,
    gpu: NvidiaGpuHeuristic,
    phase: str = "prefill",
    verbose: bool = True,
    progress_bar: TerminalProgressBar | None = None,
    cpu_seq_len_threshold: int = 4096,
):
    rel_error_curves = init_curve_bank(sparse_modes, percentage_list)
    kl_div_curves = init_curve_bank(sparse_modes, percentage_list)
    top1_match_curves = init_curve_bank(sparse_modes, percentage_list)
    accuracy_records = []

    for seq_len in seq_lens:
        if torch.cuda.is_available() and seq_len >= cpu_seq_len_threshold:
            run_device = "cpu"
            msg = f"[seq_len={seq_len}] Falling back to CPU to avoid GPU OOM."
            if verbose:
                (progress_bar.write(msg) if progress_bar else print(msg))
        else:
            run_device = None

        k_list = [max(1, int(seq_len * p)) for p in percentage_list]
        for sparse_mode in sparse_modes:
            results = evaluate_sparse_decoder(
                batch_size=batch_size, seq_len=seq_len, top_k_list=k_list,
                sparse_mode=sparse_mode, seed=seed, num_trials=num_trials,
                gpu=gpu, phase=phase, verbose=verbose,
                progress_bar=progress_bar, device=run_device,
            )
            for percentage, metrics in zip(percentage_list, results):
                rel_error_curves[sparse_mode][percentage]["mean"].append(metrics["rel_err"]["mean"])
                rel_error_curves[sparse_mode][percentage]["std"].append(metrics["rel_err"]["std"])
                kl_div_curves[sparse_mode][percentage]["mean"].append(metrics["kl_div"]["mean"])
                kl_div_curves[sparse_mode][percentage]["std"].append(metrics["kl_div"]["std"])
                top1_match_curves[sparse_mode][percentage]["mean"].append(metrics["top1_match"]["mean"])
                top1_match_curves[sparse_mode][percentage]["std"].append(metrics["top1_match"]["std"])
                accuracy_records.append(
                    build_accuracy_record(seq_len=seq_len, requested_keep_ratio=percentage, metrics=metrics)
                )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return rel_error_curves, kl_div_curves, top1_match_curves, accuracy_records


def collect_phase_speedup_curves(
    cfg: DecoderConfig,
    batch_size: int,
    seq_lens,
    percentage_list,
    sparse_modes,
    phases,
    gpu: NvidiaGpuHeuristic,
):
    phase_speedup_curves = {
        phase: init_curve_bank(sparse_modes, percentage_list) for phase in phases
    }
    speedup_records = []

    for phase in phases:
        for seq_len in seq_lens:
            for sparse_mode in sparse_modes:
                for percentage in percentage_list:
                    top_k = max(1, int(seq_len * percentage))
                    speed_est = estimate_decoder_sparse_gpu_efficiency(
                        cfg=cfg, batch_size=batch_size, seq_len=seq_len,
                        top_k=top_k, sparse_mode=sparse_mode, gpu=gpu, phase=phase,
                    )
                    phase_speedup_curves[phase][sparse_mode][percentage]["mean"].append(
                        speed_est["gpu_speedup_est"]
                    )
                    phase_speedup_curves[phase][sparse_mode][percentage]["std"].append(0.0)
                    speedup_records.append(
                        build_speedup_record(
                            seq_len=seq_len, requested_keep_ratio=percentage,
                            top_k=top_k, metrics=speed_est,
                        )
                    )
    return phase_speedup_curves, speedup_records
