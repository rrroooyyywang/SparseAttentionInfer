"""Experiment configuration dataclass used by all benchmarks."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class ExperimentConfig:
    """
    Full specification for one benchmark run.

    Pattern params (topk, keep_ratio, window_size, block_size) are optional;
    only the ones relevant to pattern_type need to be set.
    """
    batch_size: int = 1
    num_heads: int = 8
    head_dim: int = 64
    seq_len: int = 512
    dtype: str = "fp16"          # "fp32" | "fp16" | "bf16"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mode: str = "prefill"        # "prefill" | "decode"
    pattern_type: str = "dense"  # "dense" | "topk" | "bigbird" | "local_window"
    backend: str = "dense_sdpa"  # "dense_sdpa" | "masked_sdpa" | "gather_sparse"
    causal: bool = True
    # Pattern params
    topk: int | None = None
    keep_ratio: float | None = None
    window_size: int | None = None
    block_size: int | None = None
    # Measurement params
    num_warmup: int = 20
    num_iters: int = 100

    def __post_init__(self) -> None:
        self.validate_supported()

    def validate_supported(self) -> None:
        if not self.causal:
            raise ValueError(
                "This codebase currently supports causal decoder attention only; "
                "set causal=True."
            )

    def torch_dtype(self) -> torch.dtype:
        return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[self.dtype]

    def make_qkv(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create random Q/K/V tensors matching this config."""
        dev = torch.device(self.device)
        dt = self.torch_dtype()
        if self.mode == "prefill":
            q_shape = (self.batch_size, self.num_heads, self.seq_len, self.head_dim)
        else:  # decode: single new token attending to KV cache
            q_shape = (self.batch_size, self.num_heads, 1, self.head_dim)
        kv_shape = (self.batch_size, self.num_heads, self.seq_len, self.head_dim)
        q = torch.randn(q_shape, device=dev, dtype=dt)
        k = torch.randn(kv_shape, device=dev, dtype=dt)
        v = torch.randn(kv_shape, device=dev, dtype=dt)
        return q, k, v

    def as_dict(self) -> dict:
        return {
            "batch_size": self.batch_size,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "seq_len": self.seq_len,
            "dtype": self.dtype,
            "device": self.device,
            "mode": self.mode,
            "pattern_type": self.pattern_type,
            "backend": self.backend,
            "causal": self.causal,
            "topk": self.topk,
            "keep_ratio": self.keep_ratio,
            "window_size": self.window_size,
            "block_size": self.block_size,
        }
