import argparse
import gc
from pathlib import Path

import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from sparse_llm.qwen3.integrations.sparse_config import (
    _materialize_layer_group_sparsities,
)


def _get_target_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def _build_hf_pretrained_kwargs(args: argparse.Namespace) -> dict:
    return {
        "cache_dir": args.cache_dir,
        "revision": args.revision,
        "token": args.hf_token,
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": not args.allow_download,
    }


def _resolve_prompt_text(args: argparse.Namespace) -> tuple[str, str]:
    if args.text_file is not None:
        path = Path(args.text_file)
        return path.read_text(encoding="utf-8"), str(path)

    maybe_path = Path(args.text)
    if maybe_path.is_file():
        return maybe_path.read_text(encoding="utf-8"), str(maybe_path)

    return args.text, "cli"


def _load_pretrained_components(
    args: argparse.Namespace,
    *,
    sparse: bool,
) -> tuple[Qwen3Config, AutoTokenizer]:
    tokenizer_name_or_path = args.tokenizer_name_or_path or args.model_name_or_path
    hf_kwargs = _build_hf_pretrained_kwargs(args)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        **hf_kwargs,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = Qwen3Config.from_pretrained(
        args.model_name_or_path,
        **hf_kwargs,
    )
    config._attn_implementation = "eager"
    if sparse:
        config.sparse_attention_pattern = args.pattern
        config.sparse_attention_backend = args.backend
        config.sparse_attention_window_size = args.window_size
        config.sparse_attention_block_size = args.block_size
        config.sparse_attention_top_k = args.top_k
        config.sparse_attention_keep_ratio = args.keep_ratio
        config.sparse_attention_group_sparsities = (
            None if args.group_sparsities is None else list(args.group_sparsities)
        )
        config.sparse_attention_layer_group_sparsities = _materialize_layer_group_sparsities(
            args.layer_group_sparsities
        )
        config.sparse_attention_target_dtype = None
    return config, tokenizer


def _build_runtime(
    args: argparse.Namespace,
    *,
    model_cls: type[Qwen3ForCausalLM],
    sparse: bool,
):
    torch.manual_seed(args.seed)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA GPU is available.")

    config, tokenizer = _load_pretrained_components(args, sparse=sparse)
    hf_kwargs = _build_hf_pretrained_kwargs(args)
    model = model_cls.from_pretrained(
        args.model_name_or_path,
        config=config,
        **hf_kwargs,
    ).to(device)
    model = model.to(_get_target_dtype(args.dtype))
    model.eval()

    return device, tokenizer, model


def _build_generation_runtime(
    args: argparse.Namespace,
    *,
    model_cls: type[Qwen3ForCausalLM],
    sparse: bool,
):
    device, tokenizer, model = _build_runtime(
        args,
        model_cls=model_cls,
        sparse=sparse,
    )

    prompt_text, prompt_source = _resolve_prompt_text(args)
    texts = [prompt_text] * args.batch_size
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=args.seq_len,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    return device, tokenizer, model, input_ids, attention_mask, prompt_source


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cleanup_cuda_state() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
