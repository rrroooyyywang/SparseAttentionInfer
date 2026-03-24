"""Core sparse attention patterns, backends, and toy model components."""

__all__ = ["get_backend"]


def __getattr__(name: str):
    if name == "get_backend":
        from sparse_attentions.attention import get_backend

        return get_backend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
