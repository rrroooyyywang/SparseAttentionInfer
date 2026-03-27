import argparse

from sparse_llm.common.benchmark.contracts import BenchmarkAdapter
from sparse_llm.common.benchmark.generation import DEFAULT_DECODE_STEPS, DEFAULT_TEMPERATURE


def run_smoke_test(
    adapter: BenchmarkAdapter,
    args: argparse.Namespace,
    *,
    decode_steps: int = DEFAULT_DECODE_STEPS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> None:
    runtime_bundle = adapter.build_runtime(
        args=args,
        runtime_name="sparse",
        mode="smoke",
    )
    device = runtime_bundle.device
    tokenizer = runtime_bundle.tokenizer
    model = runtime_bundle.model
    input_ids = runtime_bundle.input_ids
    attention_mask = runtime_bundle.attention_mask
    prompt_source = runtime_bundle.prompt_source
    if input_ids is None or attention_mask is None or prompt_source is None:
        raise ValueError("Smoke runtime must provide input_ids, attention_mask, and prompt_source.")

    prefill_result = adapter.run_prefill(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    runtime_metadata = adapter.resolve_runtime_metadata(
        args=args,
        model=model,
        runtime_name="sparse",
    )

    pattern_stats = prefill_result.pattern_stats
    prompt_text = tokenizer.decode(
        input_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(f"Sparse {adapter.name.capitalize()} HF smoke test passed.")
    print(f"device={device}")
    print(f"dtype={next(model.parameters()).dtype}")
    print(f"tokenizer={args.tokenizer_name_or_path or args.model_name_or_path}")
    print(f"prompt_source={prompt_source}")
    print(f"batch_size={input_ids.size(0)}")
    print(f"prompt={prompt_text}")
    print(f"tokens={input_ids[0].tolist()}")
    print(f"logits_shape={prefill_result.logits_shape}")
    print(f"loss={prefill_result.loss:.6f}")
    print(f"backend_requested={runtime_metadata.backend_requested}")
    print(f"backend_actual={prefill_result.backend_actual or runtime_metadata.backend_actual}")
    if pattern_stats is not None:
        print(f"keep_ratio={pattern_stats['keep_ratio']:.4%}")
        print(f"sparsity={pattern_stats['sparsity']:.4%}")
    print(f"pattern={runtime_metadata.pattern}")

    if input_ids.size(0) != 1:
        raise ValueError("Decode mode currently supports batch_size=1 only.")
    decode_result = adapter.run_decode(
        model=model,
        past_key_values=prefill_result.outputs.past_key_values,
        next_token_logits=prefill_result.outputs.logits[:, -1, :],
        input_ids=input_ids,
        attention_mask=attention_mask,
        decode_steps=decode_steps,
        temperature=temperature,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated_text = tokenizer.decode(
        decode_result.generated_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(f"decode_temperature={temperature}")
    print(f"decode_keep_ratios={decode_result.keep_ratios}")
    print(f"decode_sparsities={decode_result.sparsities}")
    print(f"generated_text={generated_text}")
