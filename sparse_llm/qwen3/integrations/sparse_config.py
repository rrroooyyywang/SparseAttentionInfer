import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from sparse_llm.qwen3.integrations.modeling_sparse_qwen3 import (
        Qwen3SparseKernelAttention,
    )


def _get_config_attr(config, name: str, default):
    return getattr(config, name, default)


def _normalize_group_sparsities(
    values,
    *,
    name: str,
) -> Optional[tuple[float, ...]]:
    if values is None:
        return None
    normalized = tuple(float(value) for value in values)
    if any(not 0.0 <= value < 1.0 for value in normalized):
        raise ValueError(f"{name} values must satisfy 0 <= sparsity < 1.")
    return normalized


def _normalize_layer_group_sparsities(
    values,
    *,
    name: str,
) -> Optional[tuple[Optional[tuple[float, ...]], ...]]:
    if values is None:
        return None
    normalized_layers: list[Optional[tuple[float, ...]]] = []
    for layer_idx, layer_values in enumerate(values):
        if layer_values is None:
            normalized_layers.append(None)
            continue
        normalized_layers.append(
            _normalize_group_sparsities(
                layer_values,
                name=f"{name}[{layer_idx}]",
            )
        )
    return tuple(normalized_layers)


def _materialize_layer_group_sparsities(
    values: Optional[tuple[Optional[tuple[float, ...]], ...]],
) -> Optional[list[Optional[list[float]]]]:
    if values is None:
        return None
    return [
        None if layer_values is None else list(layer_values)
        for layer_values in values
    ]


def _parse_group_sparsities_arg(raw_value: Optional[str]) -> Optional[tuple[float, ...]]:
    if raw_value is None:
        return None
    text = raw_value.strip()
    if not text:
        return None
    return _normalize_group_sparsities(
        [chunk.strip() for chunk in text.split(",") if chunk.strip()],
        name="--group-sparsities",
    )


def _parse_layer_group_sparsities_arg(
    raw_value: Optional[str],
) -> Optional[tuple[Optional[tuple[float, ...]], ...]]:
    if raw_value is None:
        return None
    maybe_path = Path(raw_value)
    if maybe_path.is_file():
        payload = maybe_path.read_text(encoding="utf-8")
    else:
        payload = raw_value
    parsed = json.loads(payload)
    if not isinstance(parsed, list):
        raise ValueError("--layer-group-sparsities must decode to a JSON list.")
    return _normalize_layer_group_sparsities(
        parsed,
        name="--layer-group-sparsities",
    )


def _ensure_attn_implementation(config) -> None:
    if not hasattr(config, "_attn_implementation"):
        config._attn_implementation = "eager"


def _resolve_backend_candidates(
    pattern_name: str,
    backend_name: str,
    keep_ratio: Optional[float] = None,
    group_sparsities=None,
) -> list[str]:
    if backend_name != "auto":
        return [backend_name]
    if pattern_name == "bigbird":
        if keep_ratio is not None or group_sparsities is not None:
            return ["triton_universal"]
        return ["triton_universal", "triton_bigbird"]
    return ["triton_universal"]


def _resolve_backend_name(
    pattern_name: str,
    backend_name: str,
    keep_ratio: Optional[float] = None,
    group_sparsities=None,
) -> str:
    return _resolve_backend_candidates(
        pattern_name,
        backend_name,
        keep_ratio,
        group_sparsities,
    )[0]


def _describe_pattern(attn: "Qwen3SparseKernelAttention") -> str:
    if attn.pattern_name == "bigbird":
        if attn.group_sparsities is not None:
            resolved_layouts = getattr(attn.sparse_pattern, "last_group_layouts", None)
            resolved = f", layouts={resolved_layouts}" if resolved_layouts is not None else ""
            return (
                f"BigBirdGroupSparsity(group_sparsities={list(attn.group_sparsities)}"
                f"{resolved})"
            )
        if attn.keep_ratio is not None:
            resolved_layout = getattr(attn.sparse_pattern, "last_layout", None)
            resolved = f", layout={resolved_layout}" if resolved_layout is not None else ""
            return f"BigBirdKeepRatio(keep_ratio={attn.keep_ratio:.4f}{resolved})"
        return f"BigBird(top_k={attn.top_k})"
    return f"LocalWindow(window_size={attn.window_size}, block_size={attn.block_size})"


def _is_plain_causal_mask(
    attention_mask: Optional[torch.Tensor],
    q_len: int,
    k_len: int,
) -> bool:
    if attention_mask is None:
        return True

    if attention_mask.dim() != 4:
        return False

    if attention_mask.shape[-2] != q_len or attention_mask.shape[-1] < k_len:
        return False

    mask = attention_mask[..., :k_len]
    query_offset = max(0, k_len - q_len)
    q_positions = query_offset + torch.arange(q_len, device=mask.device)
    k_positions = torch.arange(k_len, device=mask.device)
    causal_keep = k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)
    keep_violations = (mask < 0) & causal_keep.view(1, 1, q_len, k_len)
    return not bool(keep_violations.any().item())


def validate_sparse_kernel_runtime(
    *,
    backend_name: str,
    device: torch.device,
    dtype: torch.dtype,
    block_size: Optional[int],
    has_block_list: bool = False,
) -> tuple[bool, Optional[str]]:
    if backend_name not in ("triton_universal", "triton_bigbird"):
        return True, None

    if device.type != "cuda":
        return False, f"{backend_name} requires CUDA; current device is not CUDA."

    if dtype not in (torch.float16, torch.bfloat16):
        return (
            False,
            f"{backend_name} requires float16 or bfloat16 inputs; "
            f"current dtype is {dtype}.",
        )

    if block_size is None:
        return (
            False,
            f"{backend_name} requires a block-sparse schedule; block_size is None.",
        )

    if backend_name == "triton_bigbird" and not has_block_list:
        return (
            False,
            "triton_bigbird requires BigBird block metadata; current pattern does not provide kv_block_list.",
        )

    return True, None


def _compute_pattern_stats(pattern, q_len: int, k_len: int) -> dict[str, float]:
    stats = {
        "keep_ratio": float(pattern.keep_ratio),
        "sparsity": 1.0 - float(pattern.keep_ratio),
    }

    if pattern.mask is None:
        return stats

    if pattern.block_pairs is not None or pattern.kv_block_list is not None:
        return stats

    exact_keep_ratio = float(pattern.mask.float().mean().item())
    stats["keep_ratio"] = exact_keep_ratio
    stats["sparsity"] = 1.0 - exact_keep_ratio

    return stats
