from sparse_attentions.attention.base import AttentionBackend
from sparse_attentions.attention.dense_sdpa import DenseSdpaBackend
from sparse_attentions.attention.masked_sdpa import MaskedSdpaBackend
from sparse_attentions.attention.gather_sparse import GatherSparseBackend
from sparse_attentions.attention.triton_bigbird_backend import TritonBigBirdBackend
from sparse_attentions.attention.triton_universal_backend import TritonUniversalBackend

_REGISTRY: dict[str, type[AttentionBackend]] = {
    "dense_sdpa":       DenseSdpaBackend,
    "masked_sdpa":      MaskedSdpaBackend,
    "gather_sparse":    GatherSparseBackend,
    "triton_bigbird":   TritonBigBirdBackend,
    "triton_universal": TritonUniversalBackend,
}


def get_backend(name: str, **kwargs) -> AttentionBackend:
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown backend {name!r}. Available: {available}")
    return _REGISTRY[name](**kwargs)


__all__ = [
    "AttentionBackend",
    "DenseSdpaBackend",
    "MaskedSdpaBackend",
    "GatherSparseBackend",
    "TritonBigBirdBackend",
    "TritonUniversalBackend",
    "get_backend",
]
