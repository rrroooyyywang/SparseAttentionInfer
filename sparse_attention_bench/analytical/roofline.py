"""Roofline model math helpers."""
import math

from sparse_attention_bench.analytical.config import NvidiaGpuHeuristic
from sparse_attention_bench.analytical.gpu_profiles import validate_execution_phase


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


def estimate_l2_hit_rate(gpu: NvidiaGpuHeuristic, working_set_bytes: float, phase: str) -> float:
    if gpu.l2_cache_kb <= 0 or working_set_bytes <= 0.0:
        return 0.0
    phase = validate_execution_phase(phase)
    l2_budget = gpu.l2_cache_kb * 1024 * gpu.l2_residency_fraction
    if l2_budget <= 0.0:
        return 0.0
    fit_ratio = min(1.0, l2_budget / working_set_bytes)
    hit_cap = gpu.prefill_l2_hit_rate_max if phase == "prefill" else gpu.decode_l2_hit_rate_max
    return min(hit_cap, hit_cap * math.sqrt(fit_ratio))


def adjust_cacheable_bytes_for_l2(
    cacheable_bytes: float, gpu: NvidiaGpuHeuristic, phase: str
) -> tuple[float, float]:
    if cacheable_bytes <= 0.0:
        return 0.0, 0.0
    hit_rate = estimate_l2_hit_rate(gpu=gpu, working_set_bytes=cacheable_bytes, phase=phase)
    if hit_rate <= 0.0 or gpu.l2_hit_bandwidth_multiplier <= 1.0:
        return cacheable_bytes, hit_rate
    effective = cacheable_bytes * ((1.0 - hit_rate) + hit_rate / gpu.l2_hit_bandwidth_multiplier)
    return effective, hit_rate


def roofline_op_time_us(
    flops: float,
    num_bytes: float,
    compute_tflops: float,
    gpu: NvidiaGpuHeuristic,
    launches: int = 1,
) -> float:
    if flops <= 0.0 and num_bytes <= 0.0:
        return 0.0
    compute_time = flops_to_us(flops, compute_tflops)
    memory_time = bytes_to_us(num_bytes, effective_memory_bandwidth_gbps(gpu))
    return max(compute_time, memory_time) + launches * gpu.kernel_launch_overhead_us
