"""
TritonUniversalBackend: dispatches any block-sparse pattern to the universal
Triton kernel (universal_sparse_attn) when Triton is available, with a
transparent fallback to MaskedSdpaBackend.

Requirements for Triton path:
  - CUDA tensor
  - fp16 or bf16 dtype  (tl.dot Tensor Core constraint)
  - pattern.block_pairs is not None         (CSR schedule present)
  - pattern.block_pair_offsets is not None
  - pattern.block_size is not None

Works with any SparsePattern that populates the CSR block-pair fields:
BigBirdPattern, BigBird2Pattern, LocalWindowPattern (with block_size),
or any custom pattern that calls build_block_pairs_from_mask().
"""
import torch

from sparse_attentions.attention.base import AttentionBackend
from sparse_attentions.attention.masked_sdpa import MaskedSdpaBackend
from sparse_attentions.patterns.base import PatternMetadata


class TritonUniversalBackend(AttentionBackend):
    """Universal Triton block-sparse attention with masked-SDPA fallback."""

    def __init__(self):
        self._triton_available = self._check_triton()
        self._fallback = MaskedSdpaBackend()
        # Track (block_size, D_HEAD) combos that have already been JIT-compiled
        # so that the first timed call is never contaminated by compilation.
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
            and pattern.block_pairs is not None
            and pattern.block_pair_offsets is not None
            and pattern.block_size is not None
            and q.dtype in (torch.float16, torch.bfloat16)
        ):
            self._ensure_compiled(q, k, v, pattern)
            self._last_actual_backend = "triton_universal"
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
        Force Triton JIT compilation for this (block_size, D_HEAD) combo
        before any timed call.  Subsequent calls with the same key are no-ops.
        """
        key = (pattern.block_size, q.shape[-1])
        if key in self._compiled_keys:
            return
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
        try:
            from kernels.triton.universal_sparse_attn import universal_sparse_attn
            return universal_sparse_attn(
                q, k, v,
                block_pairs=pattern.block_pairs,
                block_pair_offsets=pattern.block_pair_offsets,
                block_size=pattern.block_size,
            )
        except ImportError as e:
            raise RuntimeError(e)
