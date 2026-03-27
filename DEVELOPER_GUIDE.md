# Developer Guide

---

## Recommended Workflow

It is recommended to proceed in the following order instead of jumping straight into writing CUDA kernels or integrating with a large model:

1. Start by prototyping patterns and backends in `sparse_attentions/`.
2. Use `sparse_attention_bench/` for single-layer and ToyDecoder validation.
3. Once you confirm the direction is worth pursuing, move on to Triton or CUDA kernels in `kernels/`.
4. Then connect the new path into `sparse_llm/qwen3/` for real-model benchmarking.
5. Only after that should you move on to sparse architecture search.

This makes it much easier to identify whether a problem comes from:

- pattern semantics
- backend implementation
- kernel execution path
- benchmark configuration
- Qwen3 integration

---

## 1. Current Repository Structure

```text
SparseAttentionInfer/
├── pyproject.toml
├── README.md
├── DEVELOPER_GUIDE.md
│
├── sparse_attentions/
│   ├── attention/
│   ├── patterns/
│   ├── models/
│   └── utils.py
│
├── sparse_attention_bench/
│   ├── config.py
│   ├── paths.py
│   ├── benchmarks/
│   ├── runners/
│   ├── metrics/
│   ├── analytical/
│   ├── experiment_configs/
│   └── outputs/
│
├── sparse_llm/
│   ├── common/
│   └── qwen3/
│
└── kernels/
    ├── triton/
    └── cuda/
```

### 1.1 Responsibility Boundaries

- `sparse_attentions/`
  A reusable low-level library. It defines patterns, attention backends, and ToyDecoder-related models, and does not load Hugging Face models.
- `sparse_attention_bench/`
  The synthetic benchmark layer. It handles random-tensor benchmarks, ToyDecoder benchmarks, sweeps, and the proxy profiler.
- `sparse_llm/`
  The real-model layer. `qwen3/` is currently the primary and most complete integration path.
- `kernels/`
  Triton and CUDA kernel implementations.

### 1.2 What You Should Read First

It is recommended to read the following files before you start changing anything:

1. `sparse_attentions/patterns/base.py`
2. `sparse_attentions/attention/base.py`
3. `sparse_attention_bench/runners/benchmark_runner.py`
4. `sparse_attention_bench/runners/sweep_runner.py`
5. `sparse_llm/common/benchmark/contracts.py`
6. `sparse_llm/qwen3/adapter.py`
7. `sparse_llm/qwen3/integrations/modeling_sparse_qwen3.py`
8. `sparse_llm/qwen3/search_adapter.py`

---

## 2. Environment and Dependencies

The current dependency source of truth is `pyproject.toml`:

- Python `>=3.11`
- `torch==2.7.1`
- `triton`
- `matplotlib`
- `pyyaml`
- `datasets`
- `transformers`
- `optuna`

The repository documentation also assumes CUDA `12.8`. If you want to run Triton or CUDA paths, that is the default version you should prepare for.

Recommended:

```bash
uv sync
```

If you are not using `uv`, make sure the dependencies in `pyproject.toml` are installed by some other means.

---

## 3. Core Sparse Attention Abstractions

The core design of this repository is to separate the attention connectivity pattern from the computation backend:

```text
SparsePattern.build(q, k, causal=True) -> PatternMetadata
AttentionBackend.forward(q, k, v, pattern) -> output
```

### 3.1 The Pattern Layer

The abstract definitions live in `sparse_attentions/patterns/base.py`:

- `SparsePattern`
- `PatternMetadata`

The current `PatternMetadata.kind` values are:

- `"dense"`
- `"mask"`
- `"topk"`
- `"local"`

In addition to `mask/topk/keep_ratio`, `PatternMetadata` now also supports block-sparse metadata needed by kernels:

- `block_size`
- `kv_block_list`
- `block_pairs`
- `block_pair_offsets`

This means whether a pattern can drive a Triton kernel depends not only on `kind`, but also on whether it provides the correct block-sparse scheduling metadata.

### 3.2 Current Built-in Patterns

The current implementations live in `sparse_attentions/patterns/`:

- `DenseCausalPattern`
- `TopKPattern`
- `LocalWindowPattern`
- `BigBirdPattern`
- `BigBirdKeepRatioPattern`
- `BigBird2Pattern`

A few practical notes:

- `TopKPattern` only describes `topk`; the actual top-k selection is performed by the backend.
- `LocalWindowPattern` generates CSR block-pair metadata when `block_size` is provided, so it can drive `triton_universal`.
- `BigBirdPattern` and `BigBirdKeepRatioPattern` both build block-sparse schedules, so they can be used with both proxy backends and Triton backends.
- `BigBirdKeepRatioPattern` also supports `group_sparsities`, which is currently used by the Qwen3 search path.

### 3.3 The Backend Layer

The abstraction lives in `sparse_attentions/attention/base.py`, and the registry is in `sparse_attentions/attention/__init__.py`.

The current registry names are:

- `dense_sdpa`
- `masked_sdpa`
- `gather_sparse`
- `triton_bigbird`
- `triton_universal`

Their rough roles are:

- `dense_sdpa`
  The PyTorch SDPA dense baseline.
- `masked_sdpa`
  The general baseline path. It supports dense, mask, top-k, and local-window patterns.
- `gather_sparse`
  Mainly optimizes fixed-window patterns; if the conditions are not met, it falls back to `masked_sdpa`.
- `triton_bigbird`
  A BigBird-specific Triton backend that depends on `kv_block_list`.
- `triton_universal`
  A universal Triton block-sparse backend that depends on `block_pairs + block_pair_offsets`.

### 3.4 `actual_backend` Matters

Many backend names are only the requested path and do not guarantee that the corresponding kernel was actually executed.

For example:

- requesting `triton_bigbird` may actually end up using `masked_sdpa_fallback`
- requesting `triton_universal` may actually end up using `masked_sdpa_fallback`

So when reading benchmark results, always check `actual_backend` rather than only the configured `backend`.

---

## 4. Recommended Development Stages

### 4.1 Phase 0: Understand the Repository Before Editing

Look at these directories first:

1. `sparse_attentions/patterns/`
2. `sparse_attentions/attention/`
3. `sparse_attention_bench/runners/`
4. `sparse_attention_bench/experiment_configs/`
5. `kernels/triton/` and `kernels/cuda/`
6. `sparse_llm/qwen3/`

### 4.2 Phase 1: Prototype the Pattern / Backend First

The goal here is to validate the sparse strategy itself and the attention computation path before you get stuck in kernel details.

Recommended steps:

1. First decide whether your sparse strategy is closer to `mask`, `topk`, or block-sparse.
2. Add the pattern in `sparse_attentions/patterns/`.
3. Add a backend in `sparse_attentions/attention/`, or reuse an existing one first.
4. Register the pattern in `_get_pattern()` inside `sparse_attention_bench/runners/benchmark_runner.py`.
5. If you want to pass parameters directly from the CLI, then also extend the `choices` in `bench_layer.py` and `bench_decoder.py`.

The safest starting point is usually:

- pattern: start from `local_window.py` or `bigbird_pattern.py`
- backend: reuse `masked_sdpa` first

Single-case validation example:

```bash
uv run python -m sparse_attention_bench.benchmarks.bench_layer \
  --pattern local_window \
  --backend masked_sdpa \
  --window-size 128 \
  --seq-len 512 \
  --dtype fp16 \
  --device cuda
```

### 4.3 Phase 2: Use the Benchmark Harness for Systematic Comparison

The goal is to measure speed, memory, and accuracy across different `seq_len`, `mode`, and `dtype` settings.

Prefer using `sparse_attention_bench/runners/sweep_runner.py`.

Common sweeps:

1. `sparse_attention_bench/experiment_configs/sweep_seq_len.yaml`
   Basic sweep across dense / topk / local_window / bigbird.
2. `sparse_attention_bench/experiment_configs/sweep_seq_len_base_vs_bigbird.yaml`
   Compare `masked_sdpa`, `triton_bigbird`, and `triton_universal`.
3. `sparse_attention_bench/experiment_configs/sweep_decoder_bigbird_keep_ratio.yaml`
   Compare BigBird `keep_ratio` and `topk`-related configurations.

Example:

```bash
uv run python -m sparse_attention_bench.runners.sweep_runner \
  --config sparse_attention_bench/experiment_configs/sweep_seq_len.yaml \
  --tag sweep \
  --plot
```

### 4.4 Phase 3: Validate at the ToyDecoder Level

Do not only look at single-layer attention latency. Also validate the behavior of a full decoder block / ToyDecoder.

Entry points:

1. `sparse_attention_bench/benchmarks/bench_decoder.py`
2. `sparse_attention_bench/runners/decoder_sweep_runner.py`

Example:

```bash
uv run python -m sparse_attention_bench.runners.decoder_sweep_runner \
  --config sparse_attention_bench/experiment_configs/sweep_seq_len_base_vs_bigbird.yaml \
  --tag e2e_bigbird \
  --plot
```

Before moving to the next stage, it is recommended to confirm:

1. The single-layer benchmark shows the method is not obviously broken.
2. `top1_match_rate`, `kl_divergence`, and `rel_err` on ToyDecoder are still acceptable.
3. The bottleneck is actually in the attention kernel rather than pattern construction or Python/framework overhead.

### 4.5 Phase 4: Only Then Write Triton / CUDA Kernels

The goal of this phase is to replace proxy backends with real kernels instead of continuing to rely on `masked_sdpa` for "fake speedups."

There are two implementation paths:

1. Triton kernels: `kernels/triton/`
2. CUDA extensions: `kernels/cuda/`

Recommended order:

1. First make the pattern produce the metadata the kernel actually needs.
2. Then write the backend wrapper in `sparse_attentions/attention/`.
3. Finally register the backend in `sparse_attentions/attention/__init__.py`.

### 4.6 Phase 5: Connect the New Path into Qwen3

Key files:

1. `sparse_llm/qwen3/integrations/modeling_sparse_qwen3.py`
2. `sparse_llm/qwen3/integrations/sparse_config.py`
3. `sparse_llm/qwen3/adapter.py`
4. `sparse_llm/qwen3/cli_args.py`

If you add a new backend, the wiring usually looks like this:

1. Register its name in `sparse_attentions/attention/__init__.py`.
2. Add it to the `--backend` choices in `sparse_llm/qwen3/cli_args.py`.
3. If it participates in `auto` selection, update `_resolve_backend_candidates()`.
4. If it introduces new runtime constraints, update `validate_sparse_kernel_runtime()`.
5. If it depends on new pattern metadata, update the corresponding construction logic in `modeling_sparse_qwen3.py`.

### 4.7 Phase 6: Do Sparse Architecture Search Last

This stage is currently mainly for Qwen3.

Recommended order:

1. Set the default model, backend, search budget, and output path in `user_config.py`.
2. Start with a small search space.
3. Confirm the runtime is actually using a Triton kernel before trusting the search results.

At the moment, `Qwen3SearchAdapter.validate_trial_metrics()` requires `backend_actual` for sparse trials to really be a Triton path rather than a fallback.

---

## 5. Current State of `sparse_attention_bench/`

### 5.1 `ExperimentConfig`

`ExperimentConfig` in `sparse_attention_bench/config.py` is the core input object for layer benchmarks and sweeps.

Key fields:

- `batch_size`
- `num_heads`
- `head_dim`
- `seq_len`
- `dtype`
- `device`
- `mode`
- `pattern_type`
- `backend`
- `topk`
- `keep_ratio`
- `window_size`
- `block_size`
- `num_warmup`
- `num_iters`

Only causal decoder attention is currently supported; `causal=False` will fail immediately.

### 5.2 `benchmark_runner.py`

`sparse_attention_bench/runners/benchmark_runner.py` does the following:

1. Generates random Q/K/V
2. Builds the pattern
3. Runs the dense reference output
4. Measures pattern build, attention compute, and combined latency separately
5. Measures peak memory
6. Computes accuracy metrics

The current `_get_pattern()` supports:

- `dense`
- `topk`
- `bigbird`
- `bigbird2`
- `local_window`

One practical mismatch to be aware of:

- `benchmark_runner` already supports `bigbird2`
- but the direct CLI in `bench_layer.py` and `bench_decoder.py` does not yet expose `bigbird2` in `choices`

### 5.3 Current Benchmark Entry Points

#### `bench_layer.py`

A single-layer attention benchmark, useful for quickly checking a specific pattern/backend combination.

```bash
uv run python -m sparse_attention_bench.benchmarks.bench_layer \
  --pattern topk \
  --topk 64 \
  --backend masked_sdpa \
  --seq-len 512 \
  --device cuda
```

#### `sweep_runner.py`

Runs YAML sweeps for layer benchmarks and supports matrix expansion through `patterns:`.

```bash
uv run python -m sparse_attention_bench.runners.sweep_runner \
  --config sparse_attention_bench/experiment_configs/sweep_seq_len.yaml \
  --tag sweep_seq_len \
  --plot
```

#### `decoder_sweep_runner.py`

Runs end-to-end sweeps on ToyDecoder. The dense and sparse decoders share the same randomly initialized weights, so this is better suited for comparing accuracy drift.

```bash
uv run python -m sparse_attention_bench.runners.decoder_sweep_runner \
  --config sparse_attention_bench/experiment_configs/sweep_seq_len_base_vs_bigbird.yaml \
  --tag bigbird_compare \
  --plot
```

#### `bench_decoder.py`

A single end-to-end ToyDecoder benchmark, useful for quick smoke tests.

#### `bench_proxy.py`

A proxy profiler based on GPU parameter tables. This is not a real kernel benchmark; it is better suited for answering whether something looks promising in theory or heuristically.

```bash
uv run python -m sparse_attention_bench.benchmarks.bench_proxy \
  --gpu-profile rtx_4090 \
  --no-show
```

#### `bench_matmul.py`

A toy sparse GEMM benchmark at the matrix level, mainly for theoretical MACs and approximation error.

### 5.4 Be Careful: There Are Two YAML Formats

This is one of the easiest places to get confused in this repository.

There are currently two different YAML styles:

1. sweep YAML
   Read by `sweep_runner.py`, using `patterns:` list expansion.
2. single-experiment YAML
   Read by `bench_layer.py --config`, where fields should map directly to a single `ExperimentConfig`.

Most of the existing files in `sparse_attention_bench/experiment_configs/` are of the first type, i.e. sweep YAML.

For example, this:

```yaml
experiment:
  batch_size: 1
  num_heads: 8
  head_dim: 64
  seq_len: 512
  dtype: fp16
  device: cuda
  mode: prefill
  patterns:
    - type: topk
      backend: masked_sdpa
      topk: 64
```

is meant for `sweep_runner.py`, not for `bench_layer --config`.

If you want a single-experiment YAML for `bench_layer --config`, it should look like this instead:

```yaml
experiment:
  batch_size: 1
  num_heads: 8
  head_dim: 64
  seq_len: 512
  dtype: fp16
  device: cuda
  mode: prefill
  pattern_type: topk
  backend: masked_sdpa
  topk: 64
  causal: true
  num_warmup: 20
  num_iters: 100
```

Do not mix these two formats.

### 5.5 Direct CLI and Underlying Capability Are Not Fully Aligned

The direct CLI options exposed by `bench_layer.py` and `bench_decoder.py` are narrower than what the underlying registry and runner actually support.

For example:

- `triton_universal` is not exposed in the backend choices of `bench_layer.py`
- `bigbird2` is not exposed in the pattern choices of `bench_layer.py` / `bench_decoder.py`
- `keep_ratio` is also better handled through sweep configs than through the direct CLI

When you run into those cases, prefer using a sweep runner.

---

## 6. The Real-Model Layer: `sparse_llm/`

You can think of `sparse_llm/` as two layers:

1. `sparse_llm/common/`
   Model-agnostic benchmark/search infrastructure.
2. `sparse_llm/<model>/`
   Model-specific adaptation.

### 6.1 The Common Benchmark Layer

`sparse_llm/common/benchmark/` is responsible for:

- generation benchmark
- perplexity benchmark
- smoke benchmark
- runtime metadata collection

The core abstractions are in `sparse_llm/common/benchmark/contracts.py`:

- `BenchmarkAdapter`
- `RuntimeBundle`
- `RuntimeMetadata`
- `PrefillResult`
- `DecodeResult`

When integrating a new model into the common benchmark layer, you should implement its own `BenchmarkAdapter`.

### 6.2 Qwen3 Is Currently the Most Complete Model Integration

`sparse_llm/qwen3/` currently includes:

- CLI
- dense/sparse generation benchmarks
- dense/sparse perplexity benchmarks
- sparse architecture search

Key files:

- `sparse_llm/qwen3/cli.py`
- `sparse_llm/qwen3/cli_args.py`
- `sparse_llm/qwen3/adapter.py`
- `sparse_llm/qwen3/integrations/runtime.py`
- `sparse_llm/qwen3/integrations/modeling_sparse_qwen3.py`
- `sparse_llm/qwen3/integrations/sparse_config.py`

### 6.3 Sparse Configurations Currently Supported by Qwen3

From `cli_args.py` and `modeling_sparse_qwen3.py`, the current focus is:

- `pattern`: `local`, `bigbird`
- `backend`: `auto`, `dense_sdpa`, `masked_sdpa`, `gather_sparse`, `triton_bigbird`, `triton_universal`

BigBird-related parameters:

- `--top-k`
- `--keep-ratio`
- `--group-sparsities`
- `--layer-group-sparsities`

Where:

- `group_sparsities` is the global sparsity per KV group
- `layer_group_sparsities` is the per-layer override
- if both are provided, the layer-level configuration takes precedence

### 6.4 How `backend=auto` Is Resolved Now

In `sparse_llm/qwen3/integrations/sparse_config.py`:

- when `pattern=bigbird` and `keep_ratio` or `group_sparsities` is used, `auto` prefers `triton_universal`
- when `pattern=bigbird` uses the traditional `top_k` BigBird path, `auto` candidates are `["triton_universal", "triton_bigbird"]`
- when `pattern=local`, `auto` candidates are `["triton_universal"]`

Whether the runtime can actually use a Triton kernel still depends on runtime validation.

### 6.5 Current Limitations of the Qwen3 Sparse Path

These limitations are already hard-coded in the current implementation:

- only plain causal masks are currently supported
- padding-aware/custom attention masks are not fully wired up yet
- Triton backends require CUDA
- Triton backends require `float16` or `bfloat16`
- Triton backends also require block-sparse scheduling metadata from the pattern

If these conditions are not satisfied, the path falls back to masked SDPA.

### 6.6 Current Usable Qwen3 Commands

#### smoke

```bash
uv run python -m sparse_llm.qwen3.cli \
  --model-name-or-path Qwen/Qwen3-4B-Instruct-2507 \
  --run-mode smoke \
  --pattern bigbird \
  --backend auto \
  --top-k 128 \
  --dtype bfloat16
```

#### generation benchmark

```bash
uv run python -m sparse_llm.qwen3.cli \
  --model-name-or-path Qwen/Qwen3-4B-Instruct-2507 \
  --run-mode benchmark_both \
  --pattern bigbird \
  --backend triton_universal \
  --keep-ratio 0.5 \
  --prebuild-patterns \
  --fast-benchmark \
  --plot
```

#### perplexity benchmark

```bash
uv run python -m sparse_llm.qwen3.cli \
  --model-name-or-path Qwen/Qwen3-4B-Instruct-2507 \
  --run-mode benchmark_both_ppl \
  --pattern bigbird \
  --backend triton_universal \
  --group-sparsities 0.2,0.2,0.4,0.4,0.6,0.6,0.8,0.8 \
  --ppl-max-length 1024 \
  --prebuild-patterns \
  --fast-benchmark \
  --plot
```

If you want a strictly offline run that uses only local weights and tokenizers, explicitly add:

```bash
--no-allow-download
```

## 7. Current State of Sparse Architecture Search

`sparse_llm/common/sparse_architecture_search/` has now become a shared search framework.

Its core responsibilities are:

- defining `SearchAdapter`
- defining `SearchStrategy`
- defining `SearchObjective`
- managing the baseline + sparse trial execution loop
- recording trial results
- computing the Pareto front and plotting results

### 7.1 How Qwen3 Search Is Organized

The Qwen3-specific code lives in:

- `sparse_llm/qwen3/search_adapter.py`
- `sparse_llm/qwen3/sparse_architecture_search/config.py`
- `sparse_llm/qwen3/sparse_architecture_search/search.py`
- `sparse_llm/qwen3/sparse_architecture_search/user_config.py`

The current search target is `self_attn` inside decoder layers, with a focus on group sparsity settings for grouped-query sparse attention.

### 7.2 The Current Search Space

From `search_adapter.py`, the current search mainly revolves around:

- `sampling_grid`
- `tail_sampling_grid`
- `prefix_dense_layers`
- `layer_share_span`
- `group_sparsities`
- `layer_group_sparsities`

This means it is not a full search over arbitrary patterns and arbitrary backends. It is much more specifically focused on:

- `Qwen3`
- grouped-query sparse attention
- the BigBird keep-ratio / group-sparsity path

### 7.3 Current Search Strategies and Objectives

Currently integrated:

- `RandomSearchStrategy`
- `BayesianSearchStrategy`
- `ParetoSpeedVsQualityObjective`
- `WeightedScalarObjective`

Bayesian search depends on `optuna`, which is already included in the repository dependencies.

### 7.4 Current Usable Search Commands

If `user_config.py` is already filled in, the simplest run is:

```bash
uv run python -m sparse_llm.qwen3.sparse_architecture_search.search
```

A run with explicit parameters:

```bash
uv run python -m sparse_llm.qwen3.sparse_architecture_search.search \
  --model-name-or-path Qwen/Qwen3-4B-Instruct-2507 \
  --backend triton_universal \
  --strategy bayesian \
  --objective weighted_scalar \
  --num-samples 20 \
  --sampling-grid 0.0,0.1,0.2,0.4,0.6,0.8 \
  --layer-share-span 6
```

To redraw an existing result plot only:

```bash
uv run python -m sparse_llm.qwen3.sparse_architecture_search.search \
  --plot-only-json sparse_llm/qwen3/outputs/metrics/qwen3_bayesian_search_default.json
```

---

## 8. The Correct Place to Add a New Pattern

### 8.1 Low-Level Implementation

Create a new file in `sparse_attentions/patterns/` and implement `SparsePattern.build()`.

The minimal skeleton below supports both prefill and decode. The key detail is that when `T_q != T_k`, you must account for `query_offset`; you cannot just assume the query row indices start from 0.

```python
import torch

from sparse_attentions.patterns.base import (
    PatternMetadata,
    SparsePattern,
    build_block_pairs_from_mask,
)


class MyPattern(SparsePattern):
    def __init__(self, window_size: int, block_size: int | None = None):
        self.window_size = window_size
        self.block_size = block_size

    def build(self, q: torch.Tensor, k: torch.Tensor, causal: bool = True) -> PatternMetadata:
        h = q.size(1)
        t_q = q.size(2)
        t_k = k.size(2)
        device = q.device

        query_offset = max(0, t_k - t_q)
        rows = (query_offset + torch.arange(t_q, device=device)).unsqueeze(1)
        cols = torch.arange(t_k, device=device).unsqueeze(0)

        mask = cols >= rows - self.window_size + 1
        if causal:
            mask = mask & (cols <= rows)

        mask = mask.unsqueeze(0).expand(h, -1, -1)

        block_pairs = None
        block_pair_offsets = None
        if self.block_size is not None:
            block_pairs, block_pair_offsets = build_block_pairs_from_mask(mask, self.block_size)

        return PatternMetadata(
            kind="local",
            mask=mask,
            keep_ratio=float(mask.float().mean().item()),
            block_size=self.block_size,
            block_pairs=block_pairs,
            block_pair_offsets=block_pair_offsets,
        )
```

Notes:

- If your pattern is not a fixed local-window semantic pattern but a more general boolean mask, then `kind` should be `"mask"` instead.
- If you want `gather_sparse` to optimize it, it is best to keep the semantics aligned with `"local"`.

### 8.2 Export and Registration

At minimum, update:

- `sparse_attentions/patterns/__init__.py`
- `sparse_attention_bench/runners/benchmark_runner.py`

If you also want the direct CLI to expose it, continue with:

- `sparse_attention_bench/benchmarks/bench_layer.py`
- `sparse_attention_bench/benchmarks/bench_decoder.py`
- the corresponding YAML configs

### 8.3 If You Want to Feed It into `triton_universal`

Do not only return a token-level `mask`. You also need to provide:

- `block_size`
- `block_pairs`
- `block_pair_offsets`

The simplest way is usually to reuse `build_block_pairs_from_mask()`.

### 8.4 If You Want to Use It with Qwen3

Besides the benchmark runner, you also need to wire it into:

- `sparse_llm/qwen3/integrations/modeling_sparse_qwen3.py`
- `sparse_llm/qwen3/cli_args.py`
- `sparse_llm/qwen3/integrations/sparse_config.py`

Otherwise it will only work in the toy benchmark path and not in the real-model path.

---

## 9. The Correct Place to Add a New Backend

### 9.1 Low-Level Implementation

Add a new `AttentionBackend` subclass in `sparse_attentions/attention/`.

### 9.2 Registration

Update `_REGISTRY` in `sparse_attentions/attention/__init__.py`.

### 9.3 If the Backend Can Fall Back

It is recommended to track `actual_backend` like the existing Triton backends do, so benchmark results can distinguish:

- what the user requested
- what actually ran

### 9.4 If You Want Qwen3 to Select This Backend

You also need to update:

- `sparse_llm/qwen3/cli_args.py`
- `sparse_llm/qwen3/integrations/sparse_config.py`

Especially:

- CLI choices
- the candidate resolution for `backend=auto`
- runtime legality checks

---

## 10. If You Add a New Real-Model Integration, Follow the Qwen3 Structure

If you want to add full support for a new Hugging Face model, the recommended structure is:

```text
sparse_llm/<model>/
├── cli.py
├── cli_args.py
├── adapter.py
├── integrations/
└── sparse_architecture_search/
```

You will typically need:

- a `BenchmarkAdapter`
- a sparse runtime/modeling integration
- generation/perplexity CLI entry points
- and, if needed, a `SearchAdapter`

What should not go into `common`:

- how the model layers are counted
- which layers are searchable
- how GQA/group sparsity maps to concrete config
- which backends/patterns a specific model accepts

Those concerns should stay in `sparse_llm/<model>/`.

---

## 11. Kernel Development Paths

### 11.1 Triton

Triton kernels live in `kernels/triton/`.

Currently available:

- `bigbird_sparse_attn.py`
- `universal_sparse_attn.py`

They are called by:

- `TritonBigBirdBackend`
- `TritonUniversalBackend`

### 11.2 CUDA Extension

The CUDA extension lives in `kernels/cuda/`, which includes:

- `setup.py`
- `src/csrc/bind.cu`
- `src/csrc/helloWorldKernel.cu`
- `src/torch_wrappers/ops.py`
- `test/test_helloworld.py`

This directory is currently more of a native CUDA extension template/example path than a central dependency of the main benchmark flow.

If you want to continue developing new native CUDA ops here, start with:

- `kernels/cuda/README.md`

Build example:

```bash
cd kernels/cuda
CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH \
  python setup.py build_ext --inplace
python test/test_helloworld.py
```

If you are using a `uv` environment, you can also run:

```bash
uv run python setup.py build_ext --inplace
uv run python test/test_helloworld.py
```

Do not hardcode `.venv` paths in the documentation.

---
