"""
End-to-end decoder block benchmark.

Measures the full forward pass of a ToyDecoder (embedding + N×DecoderBlock + LM head)
for both prefill and decode modes, with and without KV cache.

Usage:
    python -m sparse_attention_bench.benchmarks.bench_decoder \\
        --pattern topk --topk 64 --seq-len 512 --num-layers 2

    python -m sparse_attention_bench.benchmarks.bench_decoder \\
        --pattern local_window --window-size 128 --seq-len 2048 --num-layers 4 --mode decode
"""
import argparse
import json
import sys
from pathlib import Path

import torch

from sparse_attention_bench.attention import get_backend
from sparse_attention_bench.config import ExperimentConfig
from sparse_attention_bench.metrics.accuracy import relative_error, cosine_sim
from sparse_attention_bench.metrics.latency import measure_latency
from sparse_attention_bench.metrics.memory import measure_peak_memory_mb
from sparse_attention_bench.models.decoder_block import ToyDecoder
from sparse_attention_bench.models.kv_cache import KVCache
from sparse_attention_bench.patterns.base import SparsePattern
from sparse_attention_bench.runners.benchmark_runner import _get_pattern


def _build_decoder(
    cfg: ExperimentConfig,
    num_layers: int,
    pattern: SparsePattern,
    backend_name: str,
    vocab_size: int = 5000,
    d_model: int = 256,
) -> ToyDecoder:
    backend = get_backend(backend_name)
    model = ToyDecoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=cfg.num_heads,
        num_layers=num_layers,
        pattern=pattern,
        backend=backend,
        max_seq_len=max(cfg.seq_len + 1, 4096),
        causal=cfg.causal,
    )
    return model.to(cfg.device).eval()


def run_decoder_bench(
    cfg: ExperimentConfig,
    num_layers: int = 2,
    vocab_size: int = 5000,
    d_model: int = 256,
    kv_lens: list[int] | None = None,
) -> dict:
    """
    Benchmark ToyDecoder for the given config.

    For prefill: single forward pass with input_ids of shape [B, seq_len].
    For decode:  single-token step with pre-filled KV cache at each kv_len.
    """
    device = torch.device(cfg.device)
    dt = cfg.torch_dtype()

    pattern = _get_pattern(cfg)
    dense_pattern_obj = __import__(
        "sparse_attention_bench.patterns.causal_dense", fromlist=["DenseCausalPattern"]
    ).DenseCausalPattern()

    # Build sparse and dense decoders
    sparse_dec = _build_decoder(cfg, num_layers, pattern, cfg.backend, vocab_size, d_model)
    dense_dec = _build_decoder(cfg, num_layers, dense_pattern_obj, "dense_sdpa", vocab_size, d_model)

    results = {"config": cfg.as_dict(), "num_layers": num_layers, "d_model": d_model, "vocab_size": vocab_size}

    if cfg.mode == "prefill":
        input_ids = torch.randint(0, vocab_size, (cfg.batch_size, cfg.seq_len), device=device)

        with torch.no_grad():
            dense_out = dense_dec(input_ids)

        latency = measure_latency(
            fn=lambda: sparse_dec(input_ids),
            num_iters=cfg.num_iters,
            num_warmup=cfg.num_warmup,
            device=cfg.device,
        )
        peak_mem = measure_peak_memory_mb(lambda: sparse_dec(input_ids), device=cfg.device)

        with torch.no_grad():
            sparse_out = sparse_dec(input_ids)

        results["prefill"] = {
            "seq_len": cfg.seq_len,
            "total_time_ms_mean": latency["mean_ms"],
            "total_time_ms_p50": latency["p50_ms"],
            "total_time_ms_p95": latency["p95_ms"],
            "peak_memory_mb": peak_mem,
            "rel_err": relative_error(sparse_out.float(), dense_out.float()),
            "cosine_sim": cosine_sim(sparse_out.float(), dense_out.float()),
        }

    else:  # decode
        decode_kv_lens = kv_lens or [128, 512, 2048]
        results["decode"] = []

        for kv_len in decode_kv_lens:
            if kv_len > cfg.seq_len:
                continue

            # Pre-fill KV cache with random activations
            kv_cache = KVCache(num_layers=num_layers)
            # Simulate prefill by running a full forward to populate the cache
            prefill_ids = torch.randint(0, vocab_size, (cfg.batch_size, kv_len), device=device)
            with torch.no_grad():
                sparse_dec(prefill_ids, kv_cache=kv_cache)

            # Single decode step
            new_token = torch.randint(0, vocab_size, (cfg.batch_size, 1), device=device)
            pos_offset = kv_len

            latency = measure_latency(
                fn=lambda: sparse_dec(new_token, pos_offset=pos_offset),
                num_iters=cfg.num_iters,
                num_warmup=cfg.num_warmup,
                device=cfg.device,
            )
            peak_mem = measure_peak_memory_mb(
                lambda: sparse_dec(new_token, pos_offset=pos_offset), device=cfg.device
            )

            results["decode"].append({
                "kv_len": kv_len,
                "total_time_ms_mean": latency["mean_ms"],
                "total_time_ms_p50": latency["p50_ms"],
                "total_time_ms_p95": latency["p95_ms"],
                "peak_memory_mb": peak_mem,
            })

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-end decoder block benchmark.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=5000)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mode", type=str, default="prefill", choices=["prefill", "decode"])
    parser.add_argument(
        "--pattern", type=str, default="dense",
        choices=["dense", "topk", "bigbird", "local_window"],
    )
    parser.add_argument(
        "--backend", type=str, default="dense_sdpa",
        choices=["dense_sdpa", "masked_sdpa", "gather_sparse"],
    )
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--kv-lens", type=int, nargs="+", default=[128, 512, 2048],
                        help="KV cache lengths to test in decode mode.")
    parser.add_argument("--num-warmup", type=int, default=20)
    parser.add_argument("--num-iters", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-print", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = ExperimentConfig(
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        seq_len=args.seq_len,
        dtype=args.dtype,
        device=args.device,
        mode=args.mode,
        pattern_type=args.pattern,
        backend=args.backend,
        topk=args.topk,
        window_size=args.window_size,
        num_warmup=args.num_warmup,
        num_iters=args.num_iters,
    )

    result = run_decoder_bench(
        cfg=cfg,
        num_layers=args.num_layers,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        kv_lens=args.kv_lens,
    )

    if not args.no_print:
        print(json.dumps(result, indent=2))

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = f"{cfg.pattern_type}_{cfg.backend}_seq{cfg.seq_len}_{cfg.mode}_{cfg.dtype}"
        out_path = out_dir / f"bench_decoder_{tag}.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"Saved to: {out_path}", file=sys.stderr)

    return result


if __name__ == "__main__":
    main()
