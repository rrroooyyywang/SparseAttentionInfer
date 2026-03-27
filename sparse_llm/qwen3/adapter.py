from __future__ import annotations

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from sparse_llm.common.benchmark.contracts import (
    BaseHFCausalLMBenchmarkAdapter,
    BenchmarkCapabilities,
    BenchmarkMode,
    RuntimeBundle,
    RuntimeMetadata,
    RuntimeName,
)
from sparse_llm.qwen3.integrations.modeling_sparse_qwen3 import (
    Qwen3SparseKernelAttention,
    Qwen3SparseKernelForCausalLM,
)
from sparse_llm.qwen3.integrations.runtime import (
    _build_generation_runtime,
    _build_runtime,
)
from sparse_llm.qwen3.integrations.sparse_config import (
    _describe_pattern,
    _parse_group_sparsities_arg,
    _parse_layer_group_sparsities_arg,
    _resolve_backend_name,
)


class Qwen3BenchmarkAdapter(BaseHFCausalLMBenchmarkAdapter):
    name = "qwen3"
    capabilities = BenchmarkCapabilities(
        supports_smoke=True,
        supports_generation=True,
        supports_perplexity=True,
        supports_sparse_runtime=True,
        supports_generation_pattern_prebuild=True,
        supports_perplexity_pattern_prebuild=True,
    )

    def normalize_args(self, args) -> None:
        if hasattr(args, "group_sparsities"):
            args.group_sparsities = _parse_group_sparsities_arg(args.group_sparsities)
        if hasattr(args, "layer_group_sparsities"):
            args.layer_group_sparsities = _parse_layer_group_sparsities_arg(
                args.layer_group_sparsities
            )

    def build_runtime(
        self,
        args,
        runtime_name: RuntimeName,
        mode: BenchmarkMode,
    ) -> RuntimeBundle:
        sparse = runtime_name == "sparse"
        model_cls = self._get_model_cls(runtime_name)
        if mode in ("generation", "smoke"):
            device, tokenizer, model, input_ids, attention_mask, prompt_source = (
                _build_generation_runtime(
                    args,
                    model_cls=model_cls,
                    sparse=sparse,
                )
            )
            return RuntimeBundle(
                device=device,
                tokenizer=tokenizer,
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                prompt_source=prompt_source,
            )

        device, tokenizer, model = _build_runtime(
            args,
            model_cls=model_cls,
            sparse=sparse,
        )
        return RuntimeBundle(
            device=device,
            tokenizer=tokenizer,
            model=model,
        )

    def resolve_runtime_metadata(
        self,
        args,
        model,
        runtime_name: RuntimeName,
    ) -> RuntimeMetadata:
        if runtime_name != "sparse":
            return RuntimeMetadata(
                backend_requested="dense_hf",
                pattern="dense_hf",
                backend_actual="dense_hf",
            )

        observations = self.collect_runtime_observations(model)
        return RuntimeMetadata(
            backend_requested=_resolve_backend_name(
                args.pattern,
                args.backend,
                args.keep_ratio,
                args.group_sparsities
                if args.group_sparsities is not None
                else args.layer_group_sparsities,
            ),
            pattern=_describe_pattern(model.model.layers[0].self_attn),
            backend_actual=observations.get("backend_actual"),
        )

    def configure_sparse_benchmark_mode(
        self,
        model,
        fast_benchmark: bool,
    ) -> dict[str, int | bool]:
        configured_layers = 0
        for attn in self._iter_sparse_attention_layers(model):
            attn.validate_causal_mask = not fast_benchmark
            attn.collect_pattern_stats = not fast_benchmark
            attn.lock_runtime_backend = fast_benchmark
            attn._locked_backend_name = None
            attn._fallback_warning_emitted = False
            attn.last_pattern_stats = None
            configured_layers += 1
        return {
            "fast_benchmark": fast_benchmark,
            "configured_layers": configured_layers,
        }

    def prebuild_generation_patterns(
        self,
        model,
        prompt_len: int,
        decode_steps: int,
        batch_size: int,
    ) -> dict[str, int | bool]:
        sparse_attn_layers = self._iter_sparse_attention_layers(model)
        if not sparse_attn_layers:
            return {
                "enabled": False,
                "prefill_pattern_built": False,
                "decode_patterns_built": 0,
                "layer_count": 0,
            }

        decode_patterns_built = 0
        with torch.no_grad():
            for attn in sparse_attn_layers:
                device = attn.q_proj.weight.device
                dtype = attn.q_proj.weight.dtype
                num_heads = attn.num_heads
                head_dim = attn.head_dim
                pattern = attn.sparse_pattern

                prefill_q = torch.empty(
                    (batch_size, num_heads, prompt_len, head_dim),
                    device=device,
                    dtype=dtype,
                )
                prefill_k = torch.empty(
                    (batch_size, num_heads, prompt_len, head_dim),
                    device=device,
                    dtype=dtype,
                )
                pattern.build(prefill_q, prefill_k, causal=True)

                for step in range(1, decode_steps + 1):
                    decode_q = torch.empty(
                        (batch_size, num_heads, 1, head_dim),
                        device=device,
                        dtype=dtype,
                    )
                    decode_k = torch.empty(
                        (batch_size, num_heads, prompt_len + step, head_dim),
                        device=device,
                        dtype=dtype,
                    )
                    pattern.build(decode_q, decode_k, causal=True)
                    decode_patterns_built += 1

        return {
            "enabled": True,
            "prefill_pattern_built": True,
            "decode_patterns_built": decode_patterns_built,
            "layer_count": len(sparse_attn_layers),
        }

    def prebuild_perplexity_patterns(
        self,
        model,
        sequence_lengths: list[int],
        batch_size: int,
    ) -> dict[str, object]:
        sparse_attn_layers = self._iter_sparse_attention_layers(model)
        if not sparse_attn_layers:
            return {
                "enabled": False,
                "patterns_built": 0,
                "sequence_lengths": [],
                "layer_count": 0,
            }

        unique_lengths = sorted(set(int(length) for length in sequence_lengths if int(length) > 0))
        patterns_built = 0
        with torch.no_grad():
            for attn in sparse_attn_layers:
                device = attn.q_proj.weight.device
                dtype = attn.q_proj.weight.dtype
                num_heads = attn.num_heads
                head_dim = attn.head_dim
                pattern = attn.sparse_pattern

                for seq_len in unique_lengths:
                    q = torch.empty(
                        (batch_size, num_heads, seq_len, head_dim),
                        device=device,
                        dtype=dtype,
                    )
                    k = torch.empty(
                        (batch_size, num_heads, seq_len, head_dim),
                        device=device,
                        dtype=dtype,
                    )
                    pattern.build(q, k, causal=True)
                    patterns_built += 1

        return {
            "enabled": True,
            "patterns_built": patterns_built,
            "sequence_lengths": unique_lengths,
            "layer_count": len(sparse_attn_layers),
        }

    def collect_runtime_observations(self, model) -> dict[str, object]:
        attn = model.model.layers[0].self_attn
        backend_actual = "dense_hf"
        pattern_stats = None
        if hasattr(attn, "sparse_backend"):
            backend_actual = attn.sparse_backend.actual_backend
            pattern_stats = getattr(attn, "last_pattern_stats", None)
        return {
            "backend_actual": backend_actual,
            "pattern_stats": pattern_stats,
            "extra": {},
        }

    def _get_model_cls(
        self,
        runtime_name: RuntimeName,
    ) -> type[Qwen3ForCausalLM]:
        if runtime_name == "dense":
            return Qwen3ForCausalLM
        if runtime_name == "sparse":
            return Qwen3SparseKernelForCausalLM
        raise ValueError(f"Unsupported runtime_name: {runtime_name!r}")

    def _iter_sparse_attention_layers(self, model) -> list[Qwen3SparseKernelAttention]:
        return [
            layer.self_attn
            for layer in model.model.layers
            if isinstance(layer.self_attn, Qwen3SparseKernelAttention)
        ]


_QWEN3_BENCHMARK_ADAPTER = Qwen3BenchmarkAdapter()


def get_qwen3_adapter() -> Qwen3BenchmarkAdapter:
    return _QWEN3_BENCHMARK_ADAPTER
