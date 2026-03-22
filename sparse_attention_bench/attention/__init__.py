from sparse_attention_bench.attention.base import AttentionBackend
from sparse_attention_bench.attention.dense_sdpa import DenseSdpaBackend
from sparse_attention_bench.attention.masked_sdpa import MaskedSdpaBackend
from sparse_attention_bench.attention.gather_sparse import GatherSparseBackend

_REGISTRY: dict[str, type[AttentionBackend]] = {
    "dense_sdpa": DenseSdpaBackend,
    "masked_sdpa": MaskedSdpaBackend,
    "gather_sparse": GatherSparseBackend,
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
    "get_backend",
]
