from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import torch


RuntimeName = Literal["dense", "sparse"]
BenchmarkMode = Literal["smoke", "generation", "perplexity"]


@dataclass(frozen=True)
class BenchmarkCapabilities:
    supports_smoke: bool = True
    supports_generation: bool = True
    supports_perplexity: bool = True
    supports_sparse_runtime: bool = True
    supports_generation_pattern_prebuild: bool = False
    supports_perplexity_pattern_prebuild: bool = False


@dataclass
class RuntimeBundle:
    device: Any
    tokenizer: Any
    model: Any
    input_ids: Any | None = None
    attention_mask: Any | None = None
    prompt_source: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeMetadata:
    backend_requested: str
    pattern: str
    backend_actual: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class PrefillResult:
    outputs: Any
    logits_shape: tuple[int, ...] | None = None
    loss: float | None = None
    pattern_stats: dict[str, Any] | None = None
    backend_actual: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecodeResult:
    generated_ids: Any
    sampled_token_ids: list[int]
    keep_ratios: list[float]
    sparsities: list[float]
    decode_step_times_s: list[float] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


class BenchmarkAdapter(Protocol):
    name: str
    capabilities: BenchmarkCapabilities

    def register_cli_args(self, parser) -> None: ...

    def normalize_args(self, args) -> None: ...

    def build_runtime(
        self,
        args,
        runtime_name: RuntimeName,
        mode: BenchmarkMode,
    ) -> RuntimeBundle: ...

    def resolve_runtime_metadata(
        self,
        args,
        model,
        runtime_name: RuntimeName,
    ) -> RuntimeMetadata: ...

    def run_prefill(
        self,
        model,
        input_ids,
        attention_mask,
    ) -> PrefillResult: ...

    def run_decode(
        self,
        model,
        past_key_values,
        next_token_logits,
        input_ids,
        attention_mask,
        decode_steps: int,
        temperature: float,
        eos_token_id: int | None,
    ) -> DecodeResult: ...

    def configure_sparse_benchmark_mode(
        self,
        model,
        fast_benchmark: bool,
    ) -> dict[str, Any]: ...

    def prebuild_generation_patterns(
        self,
        model,
        prompt_len: int,
        decode_steps: int,
        batch_size: int,
    ) -> dict[str, Any]: ...

    def prebuild_perplexity_patterns(
        self,
        model,
        sequence_lengths: list[int],
        batch_size: int,
    ) -> dict[str, Any]: ...

    def collect_runtime_observations(self, model) -> dict[str, Any]: ...


class BaseBenchmarkAdapter(ABC):
    capabilities = BenchmarkCapabilities()

    def register_cli_args(self, parser) -> None:
        return None

    def normalize_args(self, args) -> None:
        return None

    @abstractmethod
    def build_runtime(
        self,
        args,
        runtime_name: RuntimeName,
        mode: BenchmarkMode,
    ) -> RuntimeBundle:
        raise NotImplementedError

    @abstractmethod
    def resolve_runtime_metadata(
        self,
        args,
        model,
        runtime_name: RuntimeName,
    ) -> RuntimeMetadata:
        raise NotImplementedError

    @abstractmethod
    def run_prefill(
        self,
        model,
        input_ids,
        attention_mask,
    ) -> PrefillResult:
        raise NotImplementedError

    @abstractmethod
    def run_decode(
        self,
        model,
        past_key_values,
        next_token_logits,
        input_ids,
        attention_mask,
        decode_steps: int,
        temperature: float,
        eos_token_id: int | None,
    ) -> DecodeResult:
        raise NotImplementedError

    def configure_sparse_benchmark_mode(
        self,
        model,
        fast_benchmark: bool,
    ) -> dict[str, Any]:
        return {
            "fast_benchmark": fast_benchmark,
            "configured_layers": 0,
        }

    def prebuild_generation_patterns(
        self,
        model,
        prompt_len: int,
        decode_steps: int,
        batch_size: int,
    ) -> dict[str, Any]:
        return {
            "enabled": False,
        }

    def prebuild_perplexity_patterns(
        self,
        model,
        sequence_lengths: list[int],
        batch_size: int,
    ) -> dict[str, Any]:
        return {
            "enabled": False,
        }

    def collect_runtime_observations(self, model) -> dict[str, Any]:
        return {
            "backend_actual": None,
            "pattern_stats": None,
            "extra": {},
        }


class BaseHFCausalLMBenchmarkAdapter(BaseBenchmarkAdapter):
    def _sample_next_token(
        self,
        next_token_logits: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        next_token_logits = next_token_logits.float()
        if temperature <= 0:
            return next_token_logits.argmax(dim=-1, keepdim=True)

        scaled_logits = next_token_logits / temperature
        probs = torch.softmax(scaled_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def run_prefill(
        self,
        model,
        input_ids,
        attention_mask,
    ) -> PrefillResult:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                labels=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                output_attentions=False,
            )
        observations = self.collect_runtime_observations(model)
        return PrefillResult(
            outputs=outputs,
            logits_shape=tuple(outputs.logits.shape) if hasattr(outputs, "logits") else None,
            loss=float(outputs.loss) if getattr(outputs, "loss", None) is not None else None,
            pattern_stats=observations.get("pattern_stats"),
            backend_actual=observations.get("backend_actual"),
            extra=dict(observations.get("extra", {})),
        )

    def run_decode(
        self,
        model,
        past_key_values,
        next_token_logits,
        input_ids,
        attention_mask,
        decode_steps: int,
        temperature: float,
        eos_token_id: int | None,
    ) -> DecodeResult:
        generated_ids = input_ids
        current_attention_mask = attention_mask
        sampled_token_ids: list[int] = []
        keep_ratios: list[float] = []
        sparsities: list[float] = []

        next_token = self._sample_next_token(
            next_token_logits,
            temperature=temperature,
        )
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        current_attention_mask = torch.cat(
            [
                current_attention_mask,
                torch.ones(
                    (current_attention_mask.size(0), 1),
                    dtype=current_attention_mask.dtype,
                    device=current_attention_mask.device,
                ),
            ],
            dim=-1,
        )
        sampled_token_ids.append(int(next_token[0, 0].item()))
        next_input_ids = next_token

        if eos_token_id is not None and bool((next_token == eos_token_id).all()):
            return DecodeResult(
                generated_ids=generated_ids,
                sampled_token_ids=sampled_token_ids,
                keep_ratios=keep_ratios,
                sparsities=sparsities,
            )

        for _ in range(max(0, decode_steps - 1)):
            with torch.no_grad():
                outputs = model(
                    input_ids=next_input_ids,
                    attention_mask=current_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_attentions=False,
                    logits_to_keep=1,
                )

            next_token = self._sample_next_token(
                outputs.logits[:, -1, :],
                temperature=temperature,
            )
            past_key_values = outputs.past_key_values
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            current_attention_mask = torch.cat(
                [
                    current_attention_mask,
                    torch.ones(
                        (current_attention_mask.size(0), 1),
                        dtype=current_attention_mask.dtype,
                        device=current_attention_mask.device,
                    ),
                ],
                dim=-1,
            )
            next_input_ids = next_token
            sampled_token_ids.append(int(next_token[0, 0].item()))

            observations = self.collect_runtime_observations(model)
            pattern_stats = observations.get("pattern_stats")
            if pattern_stats is not None:
                keep_ratios.append(float(pattern_stats["keep_ratio"]))
                sparsities.append(float(pattern_stats["sparsity"]))

            if eos_token_id is not None and bool((next_token == eos_token_id).all()):
                break

        return DecodeResult(
            generated_ids=generated_ids,
            sampled_token_ids=sampled_token_ids,
            keep_ratios=keep_ratios,
            sparsities=sparsities,
        )
