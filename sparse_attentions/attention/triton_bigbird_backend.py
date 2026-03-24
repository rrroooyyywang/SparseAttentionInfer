"""
TritonBigBirdBackend: dispatches BigBird attention to the Triton block-sparse
kernel when Triton is available, with a transparent fallback to MaskedSdpaBackend.

Requirements for Triton path:
  - CUDA tensor
  - fp16 or bf16 dtype  (tl.dot constraint)
  - pattern.kv_block_list is not None  (BigBirdPattern was used)
  - pattern.block_size is not None

Everything else falls back to MaskedSdpaBackend, so this backend can be used
in CPU sweeps, correctness checks, and mixed-dtype experiments without error.
"""
import torch

from sparse_attentions.attention.base import AttentionBackend
from sparse_attentions.attention.masked_sdpa import MaskedSdpaBackend
from sparse_attentions.patterns.base import PatternMetadata


class TritonBigBirdBackend(AttentionBackend):
    """Triton block-sparse BigBird attention with masked-SDPA fallback."""

    def __init__(self):
        self._triton_available = self._check_triton()
        self._fallback = MaskedSdpaBackend()
        # Track which (block_size, D_HEAD, MAX_KV) combos have already been compiled.
        # The first forward() call for each new combo triggers a silent warm-up run
        # so that Triton JIT compilation never contaminates timed benchmark iterations.
        self._compiled_keys: set[tuple] = set()
        self._last_actual_backend: str = "unknown"

    @property
    def actual_backend(self) -> str:
        return self._last_actual_backend

    @staticmethod
    def _check_triton() -> bool:
        try:
            import triton  # noqa: F401
            return True
        except ImportError:
            return False

    def forward(
        self,
        q: torch.Tensor,        # [B, H, T_q, D]
        k: torch.Tensor,        # [B, H, T_k, D]
        v: torch.Tensor,        # [B, H, T_k, D]
        pattern: PatternMetadata,
    ) -> torch.Tensor:
        if (
            self._triton_available
            and q.is_cuda
            and pattern.kv_block_list is not None
            and pattern.block_size is not None
            and q.dtype in (torch.float16, torch.bfloat16)
        ):
            self._ensure_compiled(q, k, v, pattern)
            self._last_actual_backend = "triton_bigbird"
            return self._triton_forward(q, k, v, pattern)
        self._last_actual_backend = "masked_sdpa_fallback"
        return self._fallback.forward(q, k, v, pattern)

    def _ensure_compiled(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pattern: PatternMetadata,
    ) -> None:
        """
        Force Triton JIT compilation for this (block_size, D_HEAD, MAX_KV) combo
        before any timed call is made.  Subsequent calls with the same key are
        no-ops (the compiled kernel is already in Triton's in-process cache).
        """
        key = (pattern.block_size, q.shape[-1], pattern.kv_block_list.shape[-1])
        if key in self._compiled_keys:
            return
        # Silent compile run — not timed by the benchmark framework
        with torch.no_grad():
            self._triton_forward(q, k, v, pattern)
        torch.cuda.synchronize()
        self._compiled_keys.add(key)

    @staticmethod
    def _triton_forward(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pattern: PatternMetadata,
    ) -> torch.Tensor:
        # Deferred import so the class can be loaded on CPU without Triton
        try:
            from kernels.triton.bigbird_sparse_attn import bigbird_sparse_attn
            return bigbird_sparse_attn(
                q, k, v,
                kv_block_list=pattern.kv_block_list,
                block_size=pattern.block_size,
            )
        except ImportError as e:
            raise RuntimeError(e)
