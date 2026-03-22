import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


# ============================================================
# Config
# ============================================================

@dataclass
class DecoderConfig:
    vocab_size: int = 5000
    max_seq_len: int = 128
    sparse_mode: str = "top-k"
    d_model: int = 256
    n_heads: int = 8
    mlp_ratio: int = 4
    n_layers: int = 2
    dropout: float = 0.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class NvidiaGpuHeuristic:
    """
    Lightweight roofline-style proxy for recent NVIDIA GPUs.
    The defaults are representative, not measured on a specific card.
    They are meant to distinguish:
    1. dense Tensor Core GEMMs;
    2. structured/block-sparse kernels;
    3. dynamic sparse kernels with index overhead;
    4. vector/memory-bound ops such as softmax and top-k selection.
    """
    profile_name: str = "default"
    description: str = "Generic recent NVIDIA GPU proxy"
    dense_tensor_tflops: float = 125.0
    block_sparse_tensor_tflops: float = 62.0
    dynamic_sparse_tflops: float = 18.0
    vector_tflops: float = 20.0
    memory_bandwidth_gbps: float = 900.0
    memory_bandwidth_efficiency: float = 1.0
    kernel_launch_overhead_us: float = 3.5
    activation_bytes: int = 2
    weight_bytes: int = 2
    score_bytes: int = 4
    index_bytes: int = 4
    softmax_flops_per_element: float = 8.0
    topk_compare_flops_per_element: float = 2.0
    sm_count: int = 0
    tensor_cores: int = 0
    boost_clock_mhz: float = 0.0
    peak_fp32_tflops: float = 0.0
    peak_fp16_tensor_tflops: float = 0.0
    peak_fp16_tensor_sparse_tflops: float = 0.0
    l1_shared_kb_total: int = 0
    l2_cache_kb: int = 0
    register_file_kb_total: int = 0
    l2_residency_fraction: float = 0.0
    prefill_l2_hit_rate_max: float = 0.0
    decode_l2_hit_rate_max: float = 0.0
    l2_hit_bandwidth_multiplier: float = 1.0


DEFAULT_GPU_PROFILE_PATH = Path(__file__).resolve().with_name("gpu_profiles.toml")
VALID_EXECUTION_PHASES = ("prefill", "decode")


# ============================================================
# Utilities
# ============================================================

def load_gpu_profile_catalog(profile_path: str | Path = DEFAULT_GPU_PROFILE_PATH) -> tuple[str, dict]:
    profile_path = Path(profile_path)
    if not profile_path.exists():
        raise FileNotFoundError(f"GPU profile TOML not found: {profile_path}")

    with profile_path.open("rb") as handle:
        payload = tomllib.load(handle)

    profiles = payload.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError(f"No [profiles.*] entries found in {profile_path}")

    default_profile = payload.get("default_profile", next(iter(profiles)))
    if default_profile not in profiles:
        raise ValueError(
            f"default_profile={default_profile!r} not found in {profile_path}"
        )
    return default_profile, profiles


def load_gpu_heuristic(
    profile_name: str | None = None,
    profile_path: str | Path = DEFAULT_GPU_PROFILE_PATH,
) -> NvidiaGpuHeuristic:
    default_profile, profiles = load_gpu_profile_catalog(profile_path)
    selected_profile = profile_name or default_profile
    if selected_profile not in profiles:
        available = ", ".join(sorted(profiles))
        raise ValueError(
            f"Unknown GPU profile {selected_profile!r}. Available profiles: {available}"
        )

    profile_data = dict(profiles[selected_profile])
    description = profile_data.pop("description", "")
    return NvidiaGpuHeuristic(
        profile_name=selected_profile,
        description=description or f"GPU profile {selected_profile}",
        **profile_data,
    )


def validate_execution_phase(phase: str) -> str:
    if phase not in VALID_EXECUTION_PHASES:
        valid = ", ".join(VALID_EXECUTION_PHASES)
        raise ValueError(f"Unsupported execution phase: {phase}. Expected one of: {valid}")
    return phase


def iso_utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def relative_path_str(path: str | Path, base_dir: str | Path | None = None) -> str:
    base_dir = Path.cwd() if base_dir is None else Path(base_dir)
    path_obj = Path(path)
    if not path_obj.is_absolute():
        return path_obj.as_posix()
    return Path(os.path.relpath(path_obj, start=base_dir)).as_posix()


def normalize_metric_value(value, base_dir: str | Path | None = None):
    if isinstance(value, dict):
        return {
            key: normalize_metric_value(sub_value, base_dir=base_dir)
            for key, sub_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [normalize_metric_value(item, base_dir=base_dir) for item in value]
    if isinstance(value, Path):
        return relative_path_str(value, base_dir=base_dir)
    return value


def build_accuracy_record(
    seq_len: int,
    requested_keep_ratio: float,
    metrics: dict,
) -> dict:
    record = {
        "seq_len": seq_len,
        "requested_keep_ratio": requested_keep_ratio,
    }
    record.update(normalize_metric_value(metrics))
    return record


def build_speedup_record(
    seq_len: int,
    requested_keep_ratio: float,
    top_k: int,
    metrics: dict,
) -> dict:
    record = {
        "seq_len": seq_len,
        "top_k": top_k,
        "requested_keep_ratio": requested_keep_ratio,
    }
    record.update(normalize_metric_value(metrics))
    return record


def build_proxy_profile_payload(
    gpu: NvidiaGpuHeuristic,
    gpu_profile_path: str | Path,
    batch_size: int,
    seed: int,
    num_trials: int,
    seq_lens,
    percentage_list,
    sparse_modes,
    phases,
    accuracy_records,
    speedup_records,
    artifacts: dict,
) -> dict:
    path_base = Path.cwd()
    return {
        "schema_version": 1,
        "kind": "sparse_decoder_proxy_profile",
        "generated_at_utc": iso_utc_now(),
        "generator": "profiling/sparse_decoder_prof.py",
        "gpu_profile": {
            "name": gpu.profile_name,
            "description": gpu.description,
            "source_toml": relative_path_str(gpu_profile_path, base_dir=path_base),
            "parameters": normalize_metric_value(asdict(gpu), base_dir=path_base),
        },
        "experiment": {
            "batch_size": batch_size,
            "seed": seed,
            "num_trials": num_trials,
            "seq_lens": list(seq_lens),
            "requested_keep_ratios": list(percentage_list),
            "sparse_modes": list(sparse_modes),
            "phases": list(phases),
            "accuracy_phase": "prefill",
            "paths_relative_to": ".",
        },
        "artifacts": normalize_metric_value(artifacts, base_dir=path_base),
        "accuracy_records": normalize_metric_value(accuracy_records, base_dir=path_base),
        "speedup_records": normalize_metric_value(speedup_records, base_dir=path_base),
        "notes": [
            "These results are software proxy estimates, not real sparse CUDA kernel benchmarks.",
            "The current PyTorch implementation still forms dense attention scores before sparsifying.",
            "Use this JSON as a baseline for later real-kernel benchmarking and calibration.",
        ],
    }


def save_proxy_profile_json(output_path: str | Path, payload: dict) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=False),
        encoding="utf-8",
    )


class TerminalProgressBar:
    def __init__(self, total: int, unit: str = "trials", enabled: bool = True, width: int = 28):
        self.total = max(1, total)
        self.unit = unit
        self.enabled = enabled
        self.width = width
        self.completed = 0
        self.description = ""
        self.start_time = time.perf_counter()
        self._last_line_length = 0

    def update(self, step: int = 1, description: str | None = None) -> None:
        self.completed = min(self.total, self.completed + step)
        if description is not None:
            self.description = description
        self.render()

    def render(self) -> None:
        if not self.enabled:
            return

        ratio = self.completed / self.total
        filled = min(self.width, int(self.width * ratio))
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = max(time.perf_counter() - self.start_time, 1e-9)
        rate = self.completed / elapsed
        line = (
            f"\r[{bar}] {self.completed:>4d}/{self.total:<4d} {self.unit} "
            f"| {rate:>6.2f} {self.unit}/s"
        )
        if self.description:
            line += f" | {self.description}"

        print(line.ljust(self._last_line_length), end="", flush=True)
        self._last_line_length = len(line)

    def write(self, message: str) -> None:
        if not self.enabled:
            print(message)
            return

        clear_line = "\r" + (" " * self._last_line_length) + "\r"
        print(clear_line + message)
        self.render()

    def close(self) -> None:
        if not self.enabled:
            return
        self.render()
        print()

def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    # [1, 1, T, T]
    mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
        diagonal=1
    )
    return mask.unsqueeze(0).unsqueeze(0)


def relative_error(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    return (torch.norm(x - y) / (torch.norm(y) + eps)).item()


def cosine_sim(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    x_flat = x.reshape(-1)
    y_flat = y.reshape(-1)
    return F.cosine_similarity(x_flat, y_flat, dim=0, eps=eps).item()


def top1_match_rate(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
    # logits: [B, T, V]
    pred_a = logits_a.argmax(dim=-1)
    pred_b = logits_b.argmax(dim=-1)
    return (pred_a == pred_b).float().mean().item()


def mean_kl_divergence(reference_logits: torch.Tensor, approx_logits: torch.Tensor) -> float:
    """
    Average token-wise KL divergence: KL(reference || approx).
    """
    ref_log_probs = F.log_softmax(reference_logits.double(), dim=-1)
    approx_log_probs = F.log_softmax(approx_logits.double(), dim=-1)
    ref_probs = ref_log_probs.exp()
    token_kl = (ref_probs * (ref_log_probs - approx_log_probs)).sum(dim=-1)
    kl_value = token_kl.mean().item()
    return max(kl_value, 0.0)


def summarize_scalar(values) -> dict:
    values_t = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": values_t.mean().item(),
        "std": values_t.std(unbiased=False).item(),
    }


def sparsify_last_dim_topk(x: torch.Tensor, keep_k: int) -> torch.Tensor:
    """
    Keep only the top-k entries on the last dimension and zero out the rest.
    This simulates feature-sparse Q/K/V so both attention matmuls become sparse.
    """
    if keep_k >= x.size(-1):
        return x

    _, topk_idx = torch.topk(x.abs(), k=keep_k, dim=-1)
    sparse_x = torch.zeros_like(x)
    sparse_x.scatter_(-1, topk_idx, x.gather(-1, topk_idx))
    return sparse_x


def feature_keep_from_attention_keep(head_dim: int, attention_keep_ratio: float) -> int:
    return min(head_dim, max(1, math.ceil(head_dim * attention_keep_ratio)))


def causal_token_pairs(seq_len: int) -> int:
    return seq_len * (seq_len + 1) // 2


def effective_topk_attention_keep_ratio(seq_len: int, top_k: int) -> float:
    if seq_len <= 0:
        return 0.0

    total_kept = sum(min(top_k, pos + 1) for pos in range(seq_len))
    return min(1.0, total_kept / max(1, causal_token_pairs(seq_len)))


def decode_topk_attention_keep_ratio(seq_len: int, top_k: int) -> float:
    if seq_len <= 0:
        return 0.0
    return min(top_k, seq_len) / seq_len


def bigbird_layout_from_topk(seq_len: int, top_k: int) -> tuple[int, int, int, int]:
    if seq_len <= 1:
        return 1, 1, 1, 0

    block_size = max(1, min(seq_len, int(math.sqrt(max(1, top_k)))))
    block_budget = max(1, math.ceil(top_k / block_size))

    global_blocks = 1
    sliding_blocks = min(2, block_budget)
    random_blocks = max(0, block_budget - global_blocks - sliding_blocks)
    return block_size, global_blocks, sliding_blocks, random_blocks


def select_bigbird_random_block_ids(
    query_block: int,
    head_idx: int,
    num_blocks: int,
    global_blocks: int,
    sliding_blocks: int,
    random_blocks: int,
) -> list[int]:
    if random_blocks <= 0 or query_block <= 0:
        return []

    local_start = max(global_blocks, query_block - sliding_blocks + 1)
    candidates = list(range(global_blocks, query_block))
    candidates = [block_id for block_id in candidates if block_id < local_start]
    if not candidates:
        return []

    if len(candidates) <= random_blocks:
        return candidates

    offset = (head_idx * 131 + query_block * 17 + num_blocks * 7) % len(candidates)
    rotated = candidates[offset:] + candidates[:offset]
    return rotated[:random_blocks]


def estimate_bigbird_attention_keep_ratio(seq_len: int, top_k: int) -> float:
    if seq_len <= 0:
        return 0.0
    if seq_len == 1:
        return 1.0

    block_size, global_blocks, sliding_blocks, random_blocks = bigbird_layout_from_topk(seq_len, top_k)
    num_blocks = math.ceil(seq_len / block_size)
    total_visible = 0

    for query_pos in range(seq_len):
        query_block = query_pos // block_size
        visible_blocks = set(range(global_blocks))

        local_start = max(0, query_block - sliding_blocks + 1)
        visible_blocks.update(range(local_start, query_block + 1))
        visible_blocks.update(
            select_bigbird_random_block_ids(
                query_block=query_block,
                head_idx=0,
                num_blocks=num_blocks,
                global_blocks=global_blocks,
                sliding_blocks=sliding_blocks,
                random_blocks=random_blocks,
            )
        )

        row_visible = 0
        for block_id in visible_blocks:
            token_start = block_id * block_size
            token_end = min(seq_len, (block_id + 1) * block_size)
            row_visible += max(0, min(token_end, query_pos + 1) - token_start)

        total_visible += min(row_visible, query_pos + 1)

    return min(1.0, total_visible / max(1, causal_token_pairs(seq_len)))


def estimate_bigbird_decode_keep_ratio(seq_len: int, top_k: int) -> float:
    if seq_len <= 0:
        return 0.0
    if seq_len == 1:
        return 1.0

    block_size, global_blocks, sliding_blocks, random_blocks = bigbird_layout_from_topk(seq_len, top_k)
    num_blocks = math.ceil(seq_len / block_size)
    query_pos = seq_len - 1
    query_block = query_pos // block_size

    visible_blocks = set(range(global_blocks))
    local_start = max(0, query_block - sliding_blocks + 1)
    visible_blocks.update(range(local_start, query_block + 1))
    visible_blocks.update(
        select_bigbird_random_block_ids(
            query_block=query_block,
            head_idx=0,
            num_blocks=num_blocks,
            global_blocks=global_blocks,
            sliding_blocks=sliding_blocks,
            random_blocks=random_blocks,
        )
    )

    visible_tokens = 0
    for block_id in visible_blocks:
        token_start = block_id * block_size
        token_end = min(seq_len, (block_id + 1) * block_size)
        visible_tokens += max(0, min(token_end, query_pos + 1) - token_start)

    return min(1.0, visible_tokens / seq_len)


def flops_to_us(flops: float, compute_tflops: float) -> float:
    if flops <= 0.0 or compute_tflops <= 0.0:
        return 0.0
    return flops / (compute_tflops * 1e12) * 1e6


def bytes_to_us(num_bytes: float, bandwidth_gbps: float) -> float:
    if num_bytes <= 0.0 or bandwidth_gbps <= 0.0:
        return 0.0
    return num_bytes / (bandwidth_gbps * 1e9) * 1e6


def effective_memory_bandwidth_gbps(gpu: NvidiaGpuHeuristic) -> float:
    return gpu.memory_bandwidth_gbps * gpu.memory_bandwidth_efficiency


def estimate_l2_hit_rate(
    gpu: NvidiaGpuHeuristic,
    working_set_bytes: float,
    phase: str,
) -> float:
    if gpu.l2_cache_kb <= 0 or working_set_bytes <= 0.0:
        return 0.0

    phase = validate_execution_phase(phase)
    l2_budget_bytes = gpu.l2_cache_kb * 1024 * gpu.l2_residency_fraction
    if l2_budget_bytes <= 0.0:
        return 0.0

    fit_ratio = min(1.0, l2_budget_bytes / working_set_bytes)
    hit_cap = gpu.prefill_l2_hit_rate_max if phase == "prefill" else gpu.decode_l2_hit_rate_max
    return min(hit_cap, hit_cap * math.sqrt(fit_ratio))


def adjust_cacheable_bytes_for_l2(
    cacheable_bytes: float,
    gpu: NvidiaGpuHeuristic,
    phase: str,
) -> tuple[float, float]:
    if cacheable_bytes <= 0.0:
        return 0.0, 0.0

    hit_rate = estimate_l2_hit_rate(gpu=gpu, working_set_bytes=cacheable_bytes, phase=phase)
    if hit_rate <= 0.0 or gpu.l2_hit_bandwidth_multiplier <= 1.0:
        return cacheable_bytes, hit_rate

    effective_bytes = cacheable_bytes * (
        (1.0 - hit_rate) + hit_rate / gpu.l2_hit_bandwidth_multiplier
    )
    return effective_bytes, hit_rate


def roofline_op_time_us(
    flops: float,
    num_bytes: float,
    compute_tflops: float,
    gpu: NvidiaGpuHeuristic,
    launches: int = 1,
) -> float:
    if flops <= 0.0 and num_bytes <= 0.0:
        return 0.0

    compute_time_us = flops_to_us(flops, compute_tflops)
    memory_time_us = bytes_to_us(num_bytes, effective_memory_bandwidth_gbps(gpu))
    return max(compute_time_us, memory_time_us) + launches * gpu.kernel_launch_overhead_us


def estimate_decoder_sparse_gpu_efficiency(
    cfg: DecoderConfig,
    batch_size: int,
    seq_len: int,
    top_k: int,
    sparse_mode: str = "top-k",
    gpu: NvidiaGpuHeuristic | None = None,
    phase: str = "prefill",
):
    """
    Estimate GPU speedup for the decoder stack using a roofline-style proxy.

    Assumptions:
    1. Q/K/V/O projections and MLP stay dense and use Tensor Core friendly GEMMs.
    2. Softmax and top-k are bandwidth-sensitive vector ops.
    3. Dynamic top-k sparsity suffers from lower effective throughput because of
       irregular accesses and metadata.
    4. BigBird-like sparsity is more GPU-friendly than dynamic top-k because its
       connectivity is structured.
    5. `prefill` models a full-sequence forward pass, while `decode` models a
       single autoregressive step that reuses the KV cache and only projects the
       newest token.

    Note:
    The current PyTorch implementation still forms dense attention scores before top-k,
    so this estimate is a model of a hypothetical sparse CUDA kernel. It should be read as
    a runtime proxy, not the real runtime of the code in this file as-is.
    """
    gpu = gpu or NvidiaGpuHeuristic()
    phase = validate_execution_phase(phase)

    keep_ratio = min(top_k, seq_len) / seq_len
    d_model = cfg.d_model
    hidden = cfg.d_model * cfg.mlp_ratio
    n_heads = cfg.n_heads
    head_dim = cfg.d_model // cfg.n_heads
    projected_tokens = batch_size * seq_len if phase == "prefill" else batch_size
    activation_elems = projected_tokens * d_model
    kv_cache_elems = batch_size * seq_len * d_model
    output_elems = projected_tokens * d_model
    query_elems = activation_elems
    query_bytes = query_elems * gpu.activation_bytes
    kv_cache_bytes = kv_cache_elems * gpu.activation_bytes
    output_bytes = output_elems * gpu.activation_bytes
    if phase == "prefill":
        dense_score_elems = batch_size * n_heads * causal_token_pairs(seq_len)
    else:
        dense_score_elems = batch_size * n_heads * seq_len

    # Dense FLOPs
    dense_qk_cost = 2.0 * dense_score_elems * head_dim
    dense_av_cost = dense_qk_cost
    dense_softmax_cost = gpu.softmax_flops_per_element * dense_score_elems
    dense_proj_cost = 8.0 * projected_tokens * d_model * d_model
    dense_mlp_cost = 4.0 * projected_tokens * d_model * hidden

    # Dense memory traffic
    proj_linear_bytes = (
        activation_elems * gpu.activation_bytes
        + d_model * d_model * gpu.weight_bytes
        + activation_elems * gpu.activation_bytes
    )
    dense_proj_bytes = 4.0 * proj_linear_bytes
    dense_mlp_bytes = (
        activation_elems * gpu.activation_bytes
        + d_model * hidden * gpu.weight_bytes
        + projected_tokens * hidden * gpu.activation_bytes
        + projected_tokens * hidden * gpu.activation_bytes
        + hidden * d_model * gpu.weight_bytes
        + output_elems * gpu.activation_bytes
    )
    effective_dense_k_bytes, dense_qk_l2_hit_rate = adjust_cacheable_bytes_for_l2(
        cacheable_bytes=kv_cache_bytes,
        gpu=gpu,
        phase=phase,
    )
    effective_dense_v_bytes, dense_av_l2_hit_rate = adjust_cacheable_bytes_for_l2(
        cacheable_bytes=kv_cache_bytes,
        gpu=gpu,
        phase=phase,
    )
    dense_qk_bytes = query_bytes + effective_dense_k_bytes + dense_score_elems * gpu.score_bytes
    dense_softmax_bytes = 2.0 * dense_score_elems * gpu.score_bytes
    dense_av_bytes = dense_score_elems * gpu.score_bytes + effective_dense_v_bytes + output_bytes

    dense_total_cost = cfg.n_layers * (
        dense_proj_cost + dense_mlp_cost + dense_qk_cost + dense_softmax_cost + dense_av_cost
    )
    dense_total_time_us = cfg.n_layers * (
        4.0 * roofline_op_time_us(
            flops=2.0 * projected_tokens * d_model * d_model,
            num_bytes=proj_linear_bytes,
            compute_tflops=gpu.dense_tensor_tflops,
            gpu=gpu,
        )
        + roofline_op_time_us(
            flops=2.0 * projected_tokens * d_model * hidden,
            num_bytes=(
                activation_elems * gpu.activation_bytes
                + d_model * hidden * gpu.weight_bytes
                + projected_tokens * hidden * gpu.activation_bytes
            ),
            compute_tflops=gpu.dense_tensor_tflops,
            gpu=gpu,
        )
        + roofline_op_time_us(
            flops=2.0 * projected_tokens * hidden * d_model,
            num_bytes=(
                projected_tokens * hidden * gpu.activation_bytes
                + hidden * d_model * gpu.weight_bytes
                + output_elems * gpu.activation_bytes
            ),
            compute_tflops=gpu.dense_tensor_tflops,
            gpu=gpu,
        )
        + roofline_op_time_us(
            flops=dense_qk_cost,
            num_bytes=dense_qk_bytes,
            compute_tflops=gpu.dense_tensor_tflops,
            gpu=gpu,
        )
        + roofline_op_time_us(
            flops=dense_softmax_cost,
            num_bytes=dense_softmax_bytes,
            compute_tflops=gpu.vector_tflops,
            gpu=gpu,
        )
        + roofline_op_time_us(
            flops=dense_av_cost,
            num_bytes=dense_av_bytes,
            compute_tflops=gpu.dense_tensor_tflops,
            gpu=gpu,
        )
    )

    sparse_topk_bytes = 0.0
    selection_cost = 0.0
    selection_time_us = 0.0

    if sparse_mode == "top-k":
        feature_keep_ratio = feature_keep_from_attention_keep(head_dim, keep_ratio) / head_dim
        if phase == "prefill":
            attention_keep_ratio = effective_topk_attention_keep_ratio(seq_len, min(top_k, seq_len))
        else:
            attention_keep_ratio = decode_topk_attention_keep_ratio(seq_len, min(top_k, seq_len))
        sparse_qk_cost = dense_qk_cost * (feature_keep_ratio ** 2)
        sparse_av_cost = dense_av_cost * (attention_keep_ratio * feature_keep_ratio)
        sparse_softmax_cost = dense_softmax_cost * attention_keep_ratio

        sparse_query_elems = query_elems * feature_keep_ratio
        sparse_kv_elems = kv_cache_elems * feature_keep_ratio
        sparse_score_elems = dense_score_elems * attention_keep_ratio
        sparse_query_bytes = sparse_query_elems * (gpu.activation_bytes + gpu.index_bytes)
        sparse_kv_cache_bytes = sparse_kv_elems * (gpu.activation_bytes + gpu.index_bytes)
        effective_sparse_k_bytes, sparse_qk_l2_hit_rate = adjust_cacheable_bytes_for_l2(
            cacheable_bytes=sparse_kv_cache_bytes,
            gpu=gpu,
            phase=phase,
        )
        effective_sparse_v_bytes, sparse_av_l2_hit_rate = adjust_cacheable_bytes_for_l2(
            cacheable_bytes=sparse_kv_cache_bytes,
            gpu=gpu,
            phase=phase,
        )
        sparse_qk_bytes = (
            sparse_query_bytes
            + effective_sparse_k_bytes
            + sparse_score_elems * (gpu.score_bytes + gpu.index_bytes)
        )
        sparse_topk_bytes = (
            dense_score_elems * gpu.score_bytes
            + sparse_score_elems * (gpu.score_bytes + gpu.index_bytes)
        )
        sparse_softmax_bytes = 2.0 * sparse_score_elems * gpu.score_bytes
        sparse_av_bytes = (
            sparse_score_elems * (gpu.score_bytes + gpu.index_bytes)
            + effective_sparse_v_bytes
            + output_bytes
        )
        selection_cost = (
            dense_score_elems
            * gpu.topk_compare_flops_per_element
            * max(1.0, math.log2(max(2, seq_len)))
        )
        selection_time_us = roofline_op_time_us(
            flops=selection_cost,
            num_bytes=sparse_topk_bytes,
            compute_tflops=gpu.vector_tflops,
            gpu=gpu,
        )

        sparse_attention_time_us = (
            roofline_op_time_us(
                flops=sparse_qk_cost,
                num_bytes=sparse_qk_bytes,
                compute_tflops=gpu.dynamic_sparse_tflops,
                gpu=gpu,
            )
            + roofline_op_time_us(
                flops=selection_cost,
                num_bytes=sparse_topk_bytes,
                compute_tflops=gpu.vector_tflops,
                gpu=gpu,
            )
            + roofline_op_time_us(
                flops=sparse_softmax_cost,
                num_bytes=sparse_softmax_bytes,
                compute_tflops=gpu.vector_tflops,
                gpu=gpu,
            )
            + roofline_op_time_us(
                flops=sparse_av_cost,
                num_bytes=sparse_av_bytes,
                compute_tflops=gpu.dynamic_sparse_tflops,
                gpu=gpu,
            )
        )
    elif sparse_mode == "bigbird":
        feature_keep_ratio = 1.0
        if phase == "prefill":
            attention_keep_ratio = estimate_bigbird_attention_keep_ratio(seq_len, min(top_k, seq_len))
        else:
            attention_keep_ratio = estimate_bigbird_decode_keep_ratio(seq_len, min(top_k, seq_len))
        sparse_qk_cost = dense_qk_cost * attention_keep_ratio
        sparse_av_cost = dense_av_cost * attention_keep_ratio
        sparse_softmax_cost = dense_softmax_cost * attention_keep_ratio

        sparse_score_elems = dense_score_elems * attention_keep_ratio
        effective_bigbird_k_bytes, sparse_qk_l2_hit_rate = adjust_cacheable_bytes_for_l2(
            cacheable_bytes=kv_cache_bytes,
            gpu=gpu,
            phase=phase,
        )
        effective_bigbird_v_bytes, sparse_av_l2_hit_rate = adjust_cacheable_bytes_for_l2(
            cacheable_bytes=kv_cache_bytes,
            gpu=gpu,
            phase=phase,
        )
        sparse_qk_bytes = (
            query_bytes
            + effective_bigbird_k_bytes
            + sparse_score_elems * gpu.score_bytes
        )
        sparse_softmax_bytes = 2.0 * sparse_score_elems * gpu.score_bytes
        sparse_av_bytes = (
            sparse_score_elems * gpu.score_bytes
            + effective_bigbird_v_bytes
            + output_bytes
        )

        sparse_attention_time_us = (
            roofline_op_time_us(
                flops=sparse_qk_cost,
                num_bytes=sparse_qk_bytes,
                compute_tflops=gpu.block_sparse_tensor_tflops,
                gpu=gpu,
            )
            + roofline_op_time_us(
                flops=sparse_softmax_cost,
                num_bytes=sparse_softmax_bytes,
                compute_tflops=gpu.vector_tflops,
                gpu=gpu,
            )
            + roofline_op_time_us(
                flops=sparse_av_cost,
                num_bytes=sparse_av_bytes,
                compute_tflops=gpu.block_sparse_tensor_tflops,
                gpu=gpu,
            )
        )
        selection_cost = 0.0
        selection_time_us = 0.0
    else:
        raise ValueError(f"Unsupported sparse mode: {sparse_mode}")

    sparse_total_cost = cfg.n_layers * (
        dense_proj_cost + dense_mlp_cost + sparse_qk_cost + sparse_softmax_cost + sparse_av_cost + selection_cost
    )
    sparse_total_time_us = cfg.n_layers * (
        4.0 * roofline_op_time_us(
            flops=2.0 * projected_tokens * d_model * d_model,
            num_bytes=proj_linear_bytes,
            compute_tflops=gpu.dense_tensor_tflops,
            gpu=gpu,
        )
        + roofline_op_time_us(
            flops=2.0 * projected_tokens * d_model * hidden,
            num_bytes=(
                activation_elems * gpu.activation_bytes
                + d_model * hidden * gpu.weight_bytes
                + projected_tokens * hidden * gpu.activation_bytes
            ),
            compute_tflops=gpu.dense_tensor_tflops,
            gpu=gpu,
        )
        + roofline_op_time_us(
            flops=2.0 * projected_tokens * hidden * d_model,
            num_bytes=(
                projected_tokens * hidden * gpu.activation_bytes
                + hidden * d_model * gpu.weight_bytes
                + output_elems * gpu.activation_bytes
            ),
            compute_tflops=gpu.dense_tensor_tflops,
            gpu=gpu,
        )
        + sparse_attention_time_us
    )
    sparse_total_bytes = cfg.n_layers * (
        dense_proj_bytes + dense_mlp_bytes + sparse_qk_bytes + sparse_softmax_bytes + sparse_av_bytes + sparse_topk_bytes
    )

    return {
        "phase": phase,
        "sparse_mode": sparse_mode,
        "keep_ratio": keep_ratio,
        "attention_keep_ratio": attention_keep_ratio,
        "feature_keep_ratio": feature_keep_ratio,
        "dense_attention_l2_hit_rate": 0.5 * (dense_qk_l2_hit_rate + dense_av_l2_hit_rate),
        "sparse_attention_l2_hit_rate": 0.5 * (sparse_qk_l2_hit_rate + sparse_av_l2_hit_rate),
        "attention_workload_reduction": 1.0 - (
            (sparse_qk_cost + sparse_softmax_cost + sparse_av_cost + selection_cost)
            / (dense_qk_cost + dense_softmax_cost + dense_av_cost)
        ),
        "runtime_proxy_reduction": 1.0 - sparse_total_time_us / dense_total_time_us,
        "gpu_speedup_est": dense_total_time_us / sparse_total_time_us,
        "dense_time_us": dense_total_time_us,
        "sparse_time_us": sparse_total_time_us,
        "selection_time_us": cfg.n_layers * selection_time_us,
        "dense_total_flops": dense_total_cost,
        "sparse_total_flops": sparse_total_cost,
        "sparse_total_bytes": sparse_total_bytes,
        "dense_total_bytes": cfg.n_layers * (
            dense_proj_bytes + dense_mlp_bytes + dense_qk_bytes + dense_softmax_bytes + dense_av_bytes
        ),
        "sparse_attention_time_us": cfg.n_layers * sparse_attention_time_us,
    }


# ============================================================
# Attention Modules
# ============================================================

class DenseSelfAttention(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads

        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B,H,T,D]
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)     # [B,H,T,T]
        mask = causal_mask(T, x.device)
        scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)                                                   # [B,H,T,D]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class SparseSelfAttention(nn.Module):
    """
    Sparse attention approximation:
    1. sparsify Q/K/V on the feature dimension, so both matmuls are sparse;
    2. sparsify attention scores before softmax, either with row-wise top-k
       or a BigBird-style structured sparse pattern.
    """
    def __init__(self, cfg: DecoderConfig, top_k: int):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.sparse_mode = cfg.sparse_mode
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.top_k = top_k

        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        self.dropout = nn.Dropout(cfg.dropout)
        self._bigbird_mask_cache: dict[tuple[int, str, int | None], torch.Tensor] = {}

    def _bigbird_layout(self, seq_len: int) -> tuple[int, int, int, int]:
        return bigbird_layout_from_topk(seq_len, self.top_k)

    def _select_bigbird_random_blocks(
        self,
        query_block: int,
        head_idx: int,
        num_blocks: int,
        global_blocks: int,
        sliding_blocks: int,
        random_blocks: int,
        device: torch.device,
    ) -> torch.Tensor:
        if random_blocks <= 0 or query_block <= 0:
            return torch.empty(0, dtype=torch.long, device=device)
        random_block_ids = select_bigbird_random_block_ids(
            query_block=query_block,
            head_idx=head_idx,
            num_blocks=num_blocks,
            global_blocks=global_blocks,
            sliding_blocks=sliding_blocks,
            random_blocks=random_blocks,
        )
        if not random_block_ids:
            return torch.empty(0, dtype=torch.long, device=device)
        return torch.tensor(random_block_ids, dtype=torch.long, device=device)

    def _build_bigbird_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        device_key = f"{device.type}:{device.index}"
        cache_key = (seq_len, device_key, self.top_k)
        cached_mask = self._bigbird_mask_cache.get(cache_key)
        if cached_mask is not None:
            return cached_mask

        block_size, global_blocks, sliding_blocks, random_blocks = self._bigbird_layout(seq_len)
        num_blocks = math.ceil(seq_len / block_size)
        block_mask = torch.zeros(
            self.n_heads,
            num_blocks,
            num_blocks,
            dtype=torch.bool,
            device=device,
        )

        for head_idx in range(self.n_heads):
            for query_block in range(num_blocks):
                block_mask[head_idx, query_block, :global_blocks] = True

                local_start = max(0, query_block - sliding_blocks + 1)
                block_mask[head_idx, query_block, local_start:query_block + 1] = True

                random_block_ids = self._select_bigbird_random_blocks(
                    query_block=query_block,
                    head_idx=head_idx,
                    num_blocks=num_blocks,
                    global_blocks=global_blocks,
                    sliding_blocks=sliding_blocks,
                    random_blocks=random_blocks,
                    device=device,
                )
                if random_block_ids.numel() > 0:
                    block_mask[head_idx, query_block, random_block_ids] = True

        token_to_block = torch.arange(seq_len, device=device) // block_size
        token_mask = block_mask[:, token_to_block][:, :, token_to_block]
        causal = ~causal_mask(seq_len, device).squeeze(0).squeeze(0)
        token_mask = token_mask & causal.unsqueeze(0)
        token_mask[:, torch.arange(seq_len, device=device), torch.arange(seq_len, device=device)] = True

        self._bigbird_mask_cache[cache_key] = token_mask
        return token_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        B, T, C = x.shape
        k_keep = min(self.top_k, T)

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)   # [B,H,T,D]
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if self.sparse_mode == "top-k":
            feature_keep = feature_keep_from_attention_keep(self.head_dim, k_keep / T)
            q = sparsify_last_dim_topk(q, keep_k=feature_keep)
            k = sparsify_last_dim_topk(k, keep_k=feature_keep)
            v = sparsify_last_dim_topk(v, keep_k=feature_keep)
        elif self.sparse_mode == "bigbird":
            pass
        else:
            raise ValueError(f"Unsupported sparse mode: {self.sparse_mode}")

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)      # [B,H,T,T]
        mask = causal_mask(T, x.device)
        scores = scores.masked_fill(mask, float("-inf"))

        if self.sparse_mode == "top-k":
            topk_vals, topk_idx = torch.topk(scores, k=k_keep, dim=-1)                 # [B,H,T,K]
            sparse_scores = torch.full_like(scores, float("-inf"))
            sparse_scores.scatter_(-1, topk_idx, topk_vals)
        elif self.sparse_mode == "bigbird":
            bigbird_mask = self._build_bigbird_mask(T, x.device)
            sparse_scores = scores.masked_fill(~bigbird_mask.unsqueeze(0), float("-inf"))

        attn = F.softmax(sparse_scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)                                                    # [B,H,T,D]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


# ============================================================
# Decoder Block
# ============================================================

class FeedForward(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        hidden = cfg.d_model * cfg.mlp_ratio
        self.fc1 = nn.Linear(cfg.d_model, hidden)
        self.fc2 = nn.Linear(hidden, cfg.d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class DecoderBlock(nn.Module):
    def __init__(self, cfg: DecoderConfig, attn: nn.Module):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = attn
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ============================================================
# Dense Decoder / Sparse Decoder
# ============================================================

class DenseDecoder(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)

        self.blocks = nn.ModuleList([
            DecoderBlock(cfg, DenseSelfAttention(cfg))
            for _ in range(cfg.n_layers)
        ])

        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Common practice: weight tying
        self.lm_head.weight = self.token_emb.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [B, T]
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)                    # [1, T]

        x = self.token_emb(input_ids) + self.pos_emb(pos)

        for blk in self.blocks:
            x = blk(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)                                                       # [B, T, V]
        return logits


class SparseDecoder(nn.Module):
    def __init__(self, cfg: DecoderConfig, top_k: int):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)

        self.blocks = nn.ModuleList([
            DecoderBlock(cfg, SparseSelfAttention(cfg, top_k=top_k))
            for _ in range(cfg.n_layers)
        ])

        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)

        x = self.token_emb(input_ids) + self.pos_emb(pos)

        for blk in self.blocks:
            x = blk(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


# ============================================================
# Weight Copy
# ============================================================

def build_sparse_from_dense(dense_model: DenseDecoder, cfg: DecoderConfig, top_k: int) -> SparseDecoder:
    sparse_model = SparseDecoder(cfg, top_k=top_k).to(next(dense_model.parameters()).device)
    sparse_model.load_state_dict(dense_model.state_dict(), strict=True)
    return sparse_model


# ============================================================
# Accuracy Test
# ============================================================

@torch.no_grad()
def evaluate_sparse_decoder_once(
    batch_size: int = 4,
    seq_len: int = 64,
    top_k_list = (4, 8, 16, 32, 64),
    sparse_mode: str = "top-k",
    seed: int = 42,
    gpu: NvidiaGpuHeuristic | None = None,
    phase: str = "prefill",
):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cfg = DecoderConfig(max_seq_len=max(seq_len, 128), sparse_mode=sparse_mode)
    device = torch.device(cfg.device)
    gpu_heuristic = gpu or NvidiaGpuHeuristic()
    phase = validate_execution_phase(phase)

    dense_model = DenseDecoder(cfg).to(device).eval()

    # Random token input
    input_ids = torch.randint(
        low=0,
        high=cfg.vocab_size,
        size=(batch_size, seq_len),
        device=device
    )

    dense_logits = dense_model(input_ids)

    results = []
    for top_k in top_k_list:
        sparse_model = build_sparse_from_dense(dense_model, cfg, top_k=top_k).eval()
        sparse_logits = sparse_model(input_ids)

        rel_err = relative_error(sparse_logits, dense_logits)
        kl_div = mean_kl_divergence(dense_logits, sparse_logits)
        cos = cosine_sim(sparse_logits, dense_logits)
        match = top1_match_rate(sparse_logits, dense_logits)

        speed_est = estimate_decoder_sparse_gpu_efficiency(
            cfg=cfg,
            batch_size=batch_size,
            seq_len=seq_len,
            top_k=top_k,
            sparse_mode=sparse_mode,
            gpu=gpu_heuristic,
            phase=phase,
        )
        keep_ratio = speed_est["keep_ratio"]
        results.append({
            "phase": phase,
            "sparse_mode": sparse_mode,
            "top_k": top_k,
            "keep_ratio": keep_ratio,
            "rel_err": rel_err,
            "kl_div": kl_div,
            "cos_sim": cos,
            "top1_match": match,
            "top1_mismatch_rate": 1.0 - match,
            **speed_est,
        })
    return results


@torch.no_grad()
def evaluate_sparse_decoder(
    batch_size: int = 4,
    seq_len: int = 64,
    top_k_list = (4, 8, 16, 32, 64),
    sparse_mode: str = "top-k",
    seed: int = 42,
    num_trials: int = 5,
    gpu: NvidiaGpuHeuristic | None = None,
    phase: str = "prefill",
    verbose: bool = True,
    progress_bar: TerminalProgressBar | None = None,
):
    cfg = DecoderConfig(max_seq_len=max(seq_len, 128), sparse_mode=sparse_mode)
    device = torch.device(cfg.device)
    gpu_heuristic = gpu or NvidiaGpuHeuristic()
    phase = validate_execution_phase(phase)

    if verbose:
        lines = [
            f"device      : {device}",
            f"phase       : {phase}",
            f"gpu_profile : {gpu_heuristic.profile_name}",
            f"sparse_mode : {sparse_mode}",
            f"batch_size  : {batch_size}",
            f"seq_len     : {seq_len}",
            f"d_model     : {cfg.d_model}",
            f"n_heads     : {cfg.n_heads}",
            f"n_layers    : {cfg.n_layers}",
            f"num_trials  : {num_trials}",
            "-" * 72,
        ]
        message = "\n".join(lines)
        if progress_bar is not None:
            progress_bar.write(message)
        else:
            print(message)

    trial_results = []
    for trial_idx in range(num_trials):
        trial_seed = seed + trial_idx
        trial_results.append(
            evaluate_sparse_decoder_once(
                batch_size=batch_size,
                seq_len=seq_len,
                top_k_list=top_k_list,
                sparse_mode=sparse_mode,
                seed=trial_seed,
                gpu=gpu_heuristic,
                phase=phase,
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
        samples = [trial_results[trial_idx][metric_idx] for trial_idx in range(num_trials)]
        summary = {
            "phase": phase,
            "sparse_mode": sparse_mode,
            "top_k": top_k,
            "keep_ratio": samples[0]["keep_ratio"],
            "attention_keep_ratio": samples[0]["attention_keep_ratio"],
            "feature_keep_ratio": samples[0]["feature_keep_ratio"],
            "rel_err": summarize_scalar([sample["rel_err"] for sample in samples]),
            "kl_div": summarize_scalar([sample["kl_div"] for sample in samples]),
            "cos_sim": summarize_scalar([sample["cos_sim"] for sample in samples]),
            "top1_match": summarize_scalar([sample["top1_match"] for sample in samples]),
            "top1_mismatch_rate": summarize_scalar([sample["top1_mismatch_rate"] for sample in samples]),
            "gpu_speedup_est": summarize_scalar([sample["gpu_speedup_est"] for sample in samples]),
            "dense_time_us": summarize_scalar([sample["dense_time_us"] for sample in samples]),
            "sparse_time_us": summarize_scalar([sample["sparse_time_us"] for sample in samples]),
            "selection_time_us": summarize_scalar([sample["selection_time_us"] for sample in samples]),
            "attention_workload_reduction": summarize_scalar(
                [sample["attention_workload_reduction"] for sample in samples]
            ),
            "runtime_proxy_reduction": summarize_scalar(
                [sample["runtime_proxy_reduction"] for sample in samples]
            ),
            "num_trials": num_trials,
        }

        if verbose:
            message = (
                f"phase={phase:<7s} | mode={sparse_mode:<7s} | top_k={top_k:>4d} | keep_ratio={summary['keep_ratio']:>6.2%} | "
                f"attn_keep={summary['attention_keep_ratio']:>6.2%} | "
                f"feature_keep={summary['feature_keep_ratio']:>6.2%} | "
                f"rel_err={summary['rel_err']['mean']:>10.6f}±{summary['rel_err']['std']:.2e} | "
                f"kl={summary['kl_div']['mean']:>10.6f}±{summary['kl_div']['std']:.2e} | "
                f"top1_match={summary['top1_match']['mean']:>8.4%}±{summary['top1_match']['std']:.2e} | "
                f"gpu_speedup_est={summary['gpu_speedup_est']['mean']:>6.2f}x"
            )
            if progress_bar is not None:
                progress_bar.write(message)
            else:
                print(message)
        results.append(summary)

    return results


def init_curve_bank(sparse_modes, percentage_list) -> dict:
    return {
        sparse_mode: {percentage: {"mean": [], "std": []} for percentage in percentage_list}
        for sparse_mode in sparse_modes
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
):
    rel_error_curves = init_curve_bank(sparse_modes, percentage_list)
    kl_div_curves = init_curve_bank(sparse_modes, percentage_list)
    top1_match_curves = init_curve_bank(sparse_modes, percentage_list)
    accuracy_records = []

    for seq_len in seq_lens:
        k_list = [max(1, int(seq_len * p)) for p in percentage_list]
        for sparse_mode in sparse_modes:
            results = evaluate_sparse_decoder(
                batch_size=batch_size,
                seq_len=seq_len,
                top_k_list=k_list,
                sparse_mode=sparse_mode,
                seed=seed,
                num_trials=num_trials,
                gpu=gpu,
                phase=phase,
                verbose=verbose,
                progress_bar=progress_bar,
            )

            for percentage, metrics in zip(percentage_list, results):
                rel_error_curves[sparse_mode][percentage]["mean"].append(metrics["rel_err"]["mean"])
                rel_error_curves[sparse_mode][percentage]["std"].append(metrics["rel_err"]["std"])
                kl_div_curves[sparse_mode][percentage]["mean"].append(metrics["kl_div"]["mean"])
                kl_div_curves[sparse_mode][percentage]["std"].append(metrics["kl_div"]["std"])
                top1_match_curves[sparse_mode][percentage]["mean"].append(metrics["top1_match"]["mean"])
                top1_match_curves[sparse_mode][percentage]["std"].append(metrics["top1_match"]["std"])
                accuracy_records.append(
                    build_accuracy_record(
                        seq_len=seq_len,
                        requested_keep_ratio=percentage,
                        metrics=metrics,
                    )
                )

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
        phase: init_curve_bank(sparse_modes, percentage_list)
        for phase in phases
    }
    speedup_records = []

    for phase in phases:
        for seq_len in seq_lens:
            for sparse_mode in sparse_modes:
                for percentage in percentage_list:
                    top_k = max(1, int(seq_len * percentage))
                    speed_est = estimate_decoder_sparse_gpu_efficiency(
                        cfg=cfg,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        top_k=top_k,
                        sparse_mode=sparse_mode,
                        gpu=gpu,
                        phase=phase,
                    )
                    phase_speedup_curves[phase][sparse_mode][percentage]["mean"].append(
                        speed_est["gpu_speedup_est"]
                    )
                    phase_speedup_curves[phase][sparse_mode][percentage]["std"].append(0.0)
                    speedup_records.append(
                        build_speedup_record(
                            seq_len=seq_len,
                            requested_keep_ratio=percentage,
                            top_k=top_k,
                            metrics=speed_est,
                        )
                    )

    return phase_speedup_curves, speedup_records


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
    linestyles = {"top-k": "-", "bigbird": "--"}

    for sparse_mode in sparse_modes:
        for percentage in percentage_list:
            rel_mean = torch.tensor(rel_error_curves[sparse_mode][percentage]["mean"], dtype=torch.float64)
            rel_std = torch.tensor(rel_error_curves[sparse_mode][percentage]["std"], dtype=torch.float64)
            kl_mean = torch.tensor(kl_div_curves[sparse_mode][percentage]["mean"], dtype=torch.float64)
            kl_std = torch.tensor(kl_div_curves[sparse_mode][percentage]["std"], dtype=torch.float64)
            top1_mean = torch.tensor(top1_match_curves[sparse_mode][percentage]["mean"], dtype=torch.float64)
            top1_std = torch.tensor(top1_match_curves[sparse_mode][percentage]["std"], dtype=torch.float64)
            label = f"{sparse_mode}, keep_ratio={percentage:.0%}"

            axes[0].plot(
                seq_lens,
                rel_mean.tolist(),
                marker="o",
                linestyle=linestyles[sparse_mode],
                label=label,
            )
            axes[0].fill_between(
                seq_lens,
                torch.clamp(rel_mean - rel_std, min=1e-12).tolist(),
                torch.clamp(rel_mean + rel_std, min=1e-12).tolist(),
                alpha=0.10,
            )

            axes[1].plot(
                seq_lens,
                kl_mean.tolist(),
                marker="o",
                linestyle=linestyles[sparse_mode],
                label=label,
            )
            axes[1].fill_between(
                seq_lens,
                torch.clamp(kl_mean - kl_std, min=0.0).tolist(),
                (kl_mean + kl_std).tolist(),
                alpha=0.10,
            )

            axes[2].plot(
                seq_lens,
                top1_mean.tolist(),
                marker="o",
                linestyle=linestyles[sparse_mode],
                label=label,
            )
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
    linestyles = {"top-k": "-", "bigbird": "--"}

    for sparse_mode in sparse_modes:
        for percentage in percentage_list:
            speed_mean = torch.tensor(speedup_curves[sparse_mode][percentage]["mean"], dtype=torch.float64)
            speed_std = torch.tensor(speedup_curves[sparse_mode][percentage]["std"], dtype=torch.float64)
            label = f"{sparse_mode}, keep_ratio={percentage:.0%}"

            ax.plot(
                seq_lens,
                speed_mean.tolist(),
                marker="o",
                linestyle=linestyles[sparse_mode],
                label=label,
            )
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


def parse_args():
    parser = argparse.ArgumentParser(description="Sparse decoder accuracy and GPU speed proxy profiler.")
    parser.add_argument(
        "--gpu-profile",
        type=str,
        default=None,
        help="GPU profile name defined in the TOML file.",
    )
    parser.add_argument(
        "--gpu-profile-path",
        type=str,
        default=str(DEFAULT_GPU_PROFILE_PATH),
        help="Path to the GPU profile TOML file.",
    )
    parser.add_argument(
        "--list-gpu-profiles",
        action="store_true",
        help="List the available GPU profiles from the TOML file and exit.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open matplotlib windows after saving figures.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the detailed terminal logs while keeping the progress bar and final artifact paths.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    default_profile_name, available_profiles = load_gpu_profile_catalog(args.gpu_profile_path)

    if args.list_gpu_profiles:
        print(f"GPU profile file: {args.gpu_profile_path}")
        print(f"default_profile: {default_profile_name}")
        print("-" * 72)
        for profile_name in sorted(available_profiles):
            description = available_profiles[profile_name].get("description", "")
            print(f"{profile_name:<16s} {description}")
        raise SystemExit(0)

    gpu_heuristic = load_gpu_heuristic(
        profile_name=args.gpu_profile,
        profile_path=args.gpu_profile_path,
    )

    seq_lens = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    percentage_list = [0.001, 0.05, 0.1, 0.5, 0.8, 0.9, 0.99]
    sparse_modes = ["top-k", "bigbird"]
    phases = ["prefill", "decode"]
    num_trials = 5
    batch_size = 10
    seed = 42
    verbose = not args.quiet

    if verbose:
        print("Note: gpu_speedup_est is a heuristic for a custom sparse CUDA kernel on NVIDIA GPUs.")
        print("      The current implementation still computes dense scores before sparsifying,")
        print("      so the actual runtime of this script will not reach that speedup directly.")
        print(f"profile     : {gpu_heuristic.profile_name}")
        print(f"description : {gpu_heuristic.description}")
        print(f"profile_toml: {args.gpu_profile_path}")
        print("=" * 72)

    total_accuracy_trials = len(seq_lens) * len(sparse_modes) * num_trials
    progress_bar = TerminalProgressBar(total=total_accuracy_trials, unit="trials", enabled=True)
    progress_bar.render()

    rel_error_curves, kl_div_curves, top1_match_curves, accuracy_records = collect_accuracy_curves(
        seq_lens=seq_lens,
        percentage_list=percentage_list,
        sparse_modes=sparse_modes,
        batch_size=batch_size,
        seed=seed,
        num_trials=num_trials,
        gpu=gpu_heuristic,
        phase="prefill",
        verbose=verbose,
        progress_bar=progress_bar,
    )
    progress_bar.close()

    speed_cfg = DecoderConfig(max_seq_len=max(seq_lens))
    phase_speedup_curves, speedup_records = collect_phase_speedup_curves(
        cfg=speed_cfg,
        batch_size=batch_size,
        seq_lens=seq_lens,
        percentage_list=percentage_list,
        sparse_modes=sparse_modes,
        phases=phases,
        gpu=gpu_heuristic,
    )

    accuracy_fig = plot_accuracy_summary(
        seq_lens=seq_lens,
        percentage_list=percentage_list,
        sparse_modes=sparse_modes,
        rel_error_curves=rel_error_curves,
        kl_div_curves=kl_div_curves,
        top1_match_curves=top1_match_curves,
        profile_name=gpu_heuristic.profile_name,
    )
    speed_figures = {
        phase: plot_phase_speedup(
            seq_lens=seq_lens,
            percentage_list=percentage_list,
            sparse_modes=sparse_modes,
            speedup_curves=phase_speedup_curves[phase],
            phase=phase,
            profile_name=gpu_heuristic.profile_name,
        )
        for phase in phases
    }

    output_dir = Path(__file__).resolve().parent / "out"
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
        gpu=gpu_heuristic,
        gpu_profile_path=args.gpu_profile_path,
        batch_size=batch_size,
        seed=seed,
        num_trials=num_trials,
        seq_lens=seq_lens,
        percentage_list=percentage_list,
        sparse_modes=sparse_modes,
        phases=phases,
        accuracy_records=accuracy_records,
        speedup_records=speedup_records,
        artifacts={
            "accuracy_plot": accuracy_output_path,
            "speedup_plots": speedup_output_paths,
        },
    )
    save_proxy_profile_json(json_output_path, payload)
    print(f"Saved proxy profile JSON to: {json_output_path}")

    if args.no_show:
        import matplotlib.pyplot as plt

        plt.close(accuracy_fig)
        for figure in speed_figures.values():
            plt.close(figure)
    else:
        import matplotlib.pyplot as plt

        plt.show()
    
