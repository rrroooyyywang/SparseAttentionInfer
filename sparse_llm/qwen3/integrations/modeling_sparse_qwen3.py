import sys
import types
import warnings
from enum import Enum
from typing import Optional

import torch
from torch import nn
from transformers.cache_utils import Cache


def _install_torchvision_stub() -> None:
    if "torchvision" in sys.modules:
        return

    class _InterpolationMode(Enum):
        NEAREST = "nearest"
        BOX = "box"
        BILINEAR = "bilinear"
        HAMMING = "hamming"
        BICUBIC = "bicubic"
        LANCZOS = "lanczos"

    tv = types.ModuleType("torchvision")
    tv.__path__ = []

    transforms = types.ModuleType("torchvision.transforms")
    transforms.InterpolationMode = _InterpolationMode

    transforms_functional = types.ModuleType("torchvision.transforms.functional")
    io = types.ModuleType("torchvision.io")
    datasets = types.ModuleType("torchvision.datasets")
    models = types.ModuleType("torchvision.models")
    utils = types.ModuleType("torchvision.utils")
    meta = types.ModuleType("torchvision._meta_registrations")

    extension = types.ModuleType("torchvision.extension")
    extension._has_ops = lambda: False

    ops = types.ModuleType("torchvision.ops")
    ops_misc = types.ModuleType("torchvision.ops.misc")
    ops_misc.FrozenBatchNorm2d = nn.BatchNorm2d
    ops.misc = ops_misc

    tv.transforms = transforms
    tv.io = io
    tv.datasets = datasets
    tv.models = models
    tv.ops = ops
    tv.utils = utils
    tv.extension = extension
    tv._meta_registrations = meta

    sys.modules["torchvision"] = tv
    sys.modules["torchvision.transforms"] = transforms
    sys.modules["torchvision.transforms.functional"] = transforms_functional
    sys.modules["torchvision.io"] = io
    sys.modules["torchvision.datasets"] = datasets
    sys.modules["torchvision.models"] = models
    sys.modules["torchvision.ops"] = ops
    sys.modules["torchvision.ops.misc"] = ops_misc
    sys.modules["torchvision.utils"] = utils
    sys.modules["torchvision.extension"] = extension
    sys.modules["torchvision._meta_registrations"] = meta


#_install_torchvision_stub()

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3ForCausalLM,
    Qwen3Model,
    apply_rotary_pos_emb,
    repeat_kv,
)

from sparse_attentions.attention import get_backend
from sparse_attentions.patterns import BigBirdKeepRatioPattern, BigBirdPattern, LocalWindowPattern
from sparse_llm.qwen3.integrations.sparse_config import (
    _compute_pattern_stats,
    _ensure_attn_implementation,
    _get_config_attr,
    _is_plain_causal_mask,
    _materialize_layer_group_sparsities,
    _normalize_group_sparsities,
    _normalize_layer_group_sparsities,
    _resolve_backend_candidates,
    validate_sparse_kernel_runtime,
)


class Qwen3SparseKernelAttention(Qwen3Attention):
    """
    HF Qwen3 attention with the dense attention core replaced by a sparse backend.
    """

    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        *,
        backend_name: Optional[str] = None,
        window_size: Optional[int] = None,
        block_size: Optional[int] = None,
        group_sparsities=None,
    ):
        super().__init__(config, layer_idx)
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        pattern_name = _get_config_attr(config, "sparse_attention_pattern", "local")
        raw_backend_name = backend_name or _get_config_attr(
            config, "sparse_attention_backend", "auto"
        )
        window_size = window_size or _get_config_attr(
            config, "sparse_attention_window_size", 512
        )
        block_size = block_size or _get_config_attr(
            config, "sparse_attention_block_size", 64
        )
        top_k = _get_config_attr(config, "sparse_attention_top_k", 128)
        keep_ratio = _get_config_attr(config, "sparse_attention_keep_ratio", None)
        resolved_group_sparsities = _normalize_group_sparsities(
            group_sparsities,
            name=f"layer_{layer_idx}_group_sparsities",
        )
        if resolved_group_sparsities is None:
            config_layer_group_sparsities = _normalize_layer_group_sparsities(
                _get_config_attr(config, "sparse_attention_layer_group_sparsities", None),
                name="config.sparse_attention_layer_group_sparsities",
            )
            if (
                config_layer_group_sparsities is not None
                and layer_idx < len(config_layer_group_sparsities)
            ):
                resolved_group_sparsities = config_layer_group_sparsities[layer_idx]
        if resolved_group_sparsities is None:
            resolved_group_sparsities = _normalize_group_sparsities(
                _get_config_attr(config, "sparse_attention_group_sparsities", None),
                name="config.sparse_attention_group_sparsities",
            )
        backend_candidates = _resolve_backend_candidates(
            pattern_name,
            raw_backend_name,
            keep_ratio,
            resolved_group_sparsities,
        )

        self.pattern_name = pattern_name
        self.backend_candidates = backend_candidates
        self.backend_name = backend_candidates[0]
        self.window_size = window_size
        self.block_size = block_size
        self.top_k = top_k
        self.keep_ratio = None if resolved_group_sparsities is not None else keep_ratio
        self.group_sparsities = resolved_group_sparsities

        self.sparse_backends = {
            name: get_backend(name) for name in self.backend_candidates
        }
        self.sparse_backend = self.sparse_backends[self.backend_name]
        if self.pattern_name == "bigbird":
            if self.group_sparsities is not None:
                self.sparse_pattern = BigBirdKeepRatioPattern(
                    group_sparsities=self.group_sparsities,
                    n_heads=config.num_attention_heads,
                    n_kv_heads=config.num_key_value_heads,
                )
            elif self.keep_ratio is not None:
                self.sparse_pattern = BigBirdKeepRatioPattern(
                    keep_ratio=self.keep_ratio,
                    n_heads=config.num_attention_heads,
                    n_kv_heads=config.num_key_value_heads,
                )
            else:
                self.sparse_pattern = BigBirdPattern(
                    top_k=self.top_k,
                    n_heads=config.num_attention_heads,
                )
        else:
            self.sparse_pattern = LocalWindowPattern(
                window_size=self.window_size,
                block_size=self.block_size,
            )
        self.last_pattern_stats: Optional[dict[str, float]] = None
        self.validate_causal_mask = True
        self.collect_pattern_stats = True
        self.lock_runtime_backend = False
        self._locked_backend_name: Optional[str] = None
        self._fallback_warning_emitted = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        output_attentions = kwargs.get("output_attentions", False)
        if output_attentions:
            raise NotImplementedError(
                "Sparse kernel path does not yet return attention weights."
            )

        input_shape = hidden_states.shape[:-1]
        bsz, q_len = input_shape
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(
            1, 2
        )
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(
            1, 2
        )
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

        if self.validate_causal_mask and not _is_plain_causal_mask(
            attention_mask,
            q_len=q_len,
            k_len=key_states.size(2),
        ):
            raise NotImplementedError(
                "Sparse kernel path currently supports plain causal masks only. "
                "Padding-aware/custom attention masks are not wired up yet."
            )

        pattern = self.sparse_pattern.build(query_states, key_states, causal=True)
        if self.collect_pattern_stats:
            self.last_pattern_stats = _compute_pattern_stats(
                pattern,
                q_len=q_len,
                k_len=key_states.size(2),
            )
        else:
            self.last_pattern_stats = None

        selected_backend_name = self.backend_candidates[0]
        selected_backend = self.sparse_backends[selected_backend_name]
        can_use_kernel = False
        reason = None
        if self.lock_runtime_backend and self._locked_backend_name is not None:
            selected_backend_name = self._locked_backend_name
            selected_backend = self.sparse_backends[selected_backend_name]
            can_use_kernel = True
        else:
            for candidate in self.backend_candidates:
                candidate_ok, candidate_reason = validate_sparse_kernel_runtime(
                    backend_name=candidate,
                    device=query_states.device,
                    dtype=query_states.dtype,
                    block_size=pattern.block_size,
                    has_block_list=pattern.kv_block_list is not None,
                )
                if candidate_ok:
                    selected_backend_name = candidate
                    selected_backend = self.sparse_backends[candidate]
                    can_use_kernel = True
                    reason = None
                    break
                if reason is None:
                    reason = candidate_reason
            if self.lock_runtime_backend:
                self._locked_backend_name = selected_backend_name

        self.backend_name = selected_backend_name
        self.sparse_backend = selected_backend
        if (
            not can_use_kernel
            and self.backend_name in ("triton_universal", "triton_bigbird")
            and not self._fallback_warning_emitted
        ):
            warnings.warn(
                "Sparse attention will fall back instead of using the Triton kernel. "
                f"Reason: {reason}",
                stacklevel=2,
            )
            self._fallback_warning_emitted = True
        attn_output = selected_backend.forward(
            query_states,
            key_states,
            value_states,
            pattern,
        )

        expected_shape = (bsz, self.num_heads, q_len, self.head_dim)
        if attn_output.size() != expected_shape:
            raise ValueError(
                f"`attn_output` should be of size {expected_shape}, but is {attn_output.size()}."
            )

        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None


def build_sparse_attention_from_dense(
    attention: nn.Module,
    *,
    backend_name: str = "triton_universal",
    window_size: int = 512,
    block_size: int = 64,
    target_dtype: Optional[torch.dtype] = None,
    group_sparsities=None,
) -> Qwen3SparseKernelAttention:
    if not hasattr(attention, "config") or not hasattr(attention, "layer_idx"):
        raise TypeError(
            "Expected a Qwen3 attention module with `config` and `layer_idx` attributes."
        )

    sparse_attention = Qwen3SparseKernelAttention(
        attention.config,
        layer_idx=attention.layer_idx,
        backend_name=backend_name,
        window_size=window_size,
        block_size=block_size,
        group_sparsities=group_sparsities,
    )
    sparse_attention.load_state_dict(attention.state_dict(), strict=False)
    sparse_attention.to(device=attention.q_proj.weight.device)
    if target_dtype is not None:
        sparse_attention.to(dtype=target_dtype)
    return sparse_attention


def replace_qwen3_attention_with_sparse(
    model: nn.Module,
    *,
    backend_name: str = "triton_universal",
    window_size: int = 512,
    block_size: int = 64,
    target_dtype: Optional[torch.dtype] = None,
    group_sparsities=None,
    layer_group_sparsities=None,
) -> nn.Module:
    if not hasattr(model, "layers"):
        raise TypeError("Expected a Qwen3 decoder model with a `.layers` attribute.")

    normalized_group_sparsities = _normalize_group_sparsities(
        group_sparsities,
        name="group_sparsities",
    )
    if normalized_group_sparsities is None:
        normalized_group_sparsities = _normalize_group_sparsities(
            _get_config_attr(model.config, "sparse_attention_group_sparsities", None),
            name="config.sparse_attention_group_sparsities",
        )

    normalized_layer_group_sparsities = _normalize_layer_group_sparsities(
        layer_group_sparsities,
        name="layer_group_sparsities",
    )
    if normalized_layer_group_sparsities is None:
        normalized_layer_group_sparsities = _normalize_layer_group_sparsities(
            _get_config_attr(model.config, "sparse_attention_layer_group_sparsities", None),
            name="config.sparse_attention_layer_group_sparsities",
        )
    if (
        normalized_layer_group_sparsities is not None
        and len(normalized_layer_group_sparsities) > len(model.layers)
    ):
        raise ValueError(
            "layer_group_sparsities has more entries than there are decoder layers."
        )

    for layer in model.layers:
        layer_group_values = None
        if normalized_layer_group_sparsities is not None and layer.self_attn.layer_idx < len(
            normalized_layer_group_sparsities
        ):
            layer_group_values = normalized_layer_group_sparsities[layer.self_attn.layer_idx]
        effective_group_sparsities = (
            layer_group_values
            if layer_group_values is not None
            else normalized_group_sparsities
        )
        layer.self_attn = build_sparse_attention_from_dense(
            layer.self_attn,
            backend_name=backend_name,
            window_size=window_size,
            block_size=block_size,
            target_dtype=target_dtype,
            group_sparsities=effective_group_sparsities,
        )
    return model


def _apply_sparse_config_to_hf_config(
    config: Qwen3Config,
    *,
    group_sparsities=None,
    layer_group_sparsities=None,
) -> None:
    _ensure_attn_implementation(config)
    if group_sparsities is not None:
        config.sparse_attention_group_sparsities = list(
            _normalize_group_sparsities(
                group_sparsities,
                name="group_sparsities",
            )
        )
    if layer_group_sparsities is not None:
        config.sparse_attention_layer_group_sparsities = _materialize_layer_group_sparsities(
            _normalize_layer_group_sparsities(
                layer_group_sparsities,
                name="layer_group_sparsities",
            )
        )


def _replace_sparse_attention_from_config(
    model: nn.Module,
    config: Qwen3Config,
) -> None:
    replace_qwen3_attention_with_sparse(
        model,
        backend_name=_get_config_attr(
            config, "sparse_attention_backend", "triton_universal"
        ),
        window_size=_get_config_attr(config, "sparse_attention_window_size", 512),
        block_size=_get_config_attr(config, "sparse_attention_block_size", 64),
        target_dtype=_get_config_attr(config, "sparse_attention_target_dtype", None),
        group_sparsities=_get_config_attr(
            config, "sparse_attention_group_sparsities", None
        ),
        layer_group_sparsities=_get_config_attr(
            config, "sparse_attention_layer_group_sparsities", None
        ),
    )


class Qwen3SparseKernelModel(Qwen3Model):
    def __init__(
        self,
        config: Qwen3Config,
        *,
        group_sparsities=None,
        layer_group_sparsities=None,
    ):
        _apply_sparse_config_to_hf_config(
            config,
            group_sparsities=group_sparsities,
            layer_group_sparsities=layer_group_sparsities,
        )
        super().__init__(config)
        _replace_sparse_attention_from_config(self, config)


class Qwen3SparseKernelForCausalLM(Qwen3ForCausalLM):
    def __init__(
        self,
        config: Qwen3Config,
        *,
        group_sparsities=None,
        layer_group_sparsities=None,
    ):
        _apply_sparse_config_to_hf_config(
            config,
            group_sparsities=group_sparsities,
            layer_group_sparsities=layer_group_sparsities,
        )
        super().__init__(config)
        _replace_sparse_attention_from_config(self.model, config)
