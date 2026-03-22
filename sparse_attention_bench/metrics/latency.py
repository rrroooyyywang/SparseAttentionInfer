"""CUDA event timing and latency statistics."""
from __future__ import annotations

import time
from typing import Callable

import torch


def _cpu_timer_ms(fn: Callable, num_iters: int, num_warmup: int) -> list[float]:
    for _ in range(num_warmup):
        fn()
    times = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def _cuda_timer_ms(fn: Callable, num_iters: int, num_warmup: int) -> list[float]:
    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(num_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times


def measure_latency(
    fn: Callable,
    num_iters: int = 100,
    num_warmup: int = 20,
    device: str = "cpu",
) -> dict:
    """
    Measure latency of fn() and return statistics.

    Returns dict with keys:
        mean_ms, std_ms, p50_ms, p95_ms, min_ms, max_ms
    """
    if "cuda" in device and torch.cuda.is_available():
        times = _cuda_timer_ms(fn, num_iters, num_warmup)
    else:
        times = _cpu_timer_ms(fn, num_iters, num_warmup)

    t = torch.tensor(times, dtype=torch.float64)
    return {
        "mean_ms": t.mean().item(),
        "std_ms": t.std(unbiased=False).item(),
        "p50_ms": t.quantile(0.50).item(),
        "p95_ms": t.quantile(0.95).item(),
        "min_ms": t.min().item(),
        "max_ms": t.max().item(),
    }
