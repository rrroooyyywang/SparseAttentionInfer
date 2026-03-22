"""GPU memory measurement utilities."""
from __future__ import annotations

from typing import Callable

import torch


def measure_peak_memory_mb(fn: Callable, device: str = "cpu") -> float:
    """
    Run fn() and return the peak GPU memory allocated during the call (in MB).
    Returns 0.0 on CPU.
    """
    if "cuda" not in device or not torch.cuda.is_available():
        fn()
        return 0.0

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def current_memory_mb(device: str = "cpu") -> float:
    if "cuda" not in device or not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / (1024 ** 2)
