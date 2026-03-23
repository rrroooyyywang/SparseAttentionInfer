# Developer Guide: Benchmarking Sparse Attention Algorithms

This guide walks you through adding your own sparse attention algorithm to the benchmark harness and running all available benchmark types.

---

## Table of Contents

1. [Repo Layout](#1-repo-layout)
2. [Core Concepts](#2-core-concepts)
3. [Adding a Custom Sparse Pattern](#3-adding-a-custom-sparse-pattern)
4. [Adding a Custom Compute Backend (Optional)](#4-adding-a-custom-compute-backend-optional)
5. [Writing a Real Sparse Kernel (Triton / CUDA)](#5-writing-a-real-sparse-kernel-triton--cuda)
6. [Running Benchmarks](#6-running-benchmarks)
7. [Understanding the Outputs](#7-understanding-the-outputs)
8. [Quick Reference](#8-quick-reference)

---

## 1. Repo Layout

```
SparseAttentionInfer/
├── sparse_attention_bench/             ← benchmark harness (pure Python)
│   ├── patterns/                       ← WHERE YOU ADD YOUR SPARSE PATTERN
│   │   ├── base.py                     ← SparsePattern ABC + PatternMetadata
│   │   ├── causal_dense.py             ← Built-in: full causal dense
│   │   ├── topk_pattern.py             ← Built-in: row-wise top-k
│   │   ├── bigbird_pattern.py          ← Built-in: BigBird structured sparse
│   │   └── local_window.py             ← Built-in: sliding local window
│   │
│   ├── attention/                      ← Compute backends (how attention is computed)
│   │   ├── base.py                     ← AttentionBackend ABC
│   │   ├── dense_sdpa.py               ← PyTorch scaled_dot_product_attention
│   │   ├── masked_sdpa.py              ← SDPA with boolean mask / top-k selection
│   │   ├── gather_sparse.py            ← Gather-based sparse (local patterns)
│   │   ├── triton_backend.py           ← Template for Triton kernel backends
│   │   └── __init__.py                 ← Backend registry + get_backend()
│   │
│   ├── runners/
│   │   ├── benchmark_runner.py         ← Core timing engine (pattern build + attention)
│   │   └── sweep_runner.py             ← YAML-driven multi-config sweep
│   │
│   ├── benchmarks/
│   │   ├── bench_layer.py              ← Single attention layer benchmark CLI
│   │   ├── bench_decoder.py            ← Full decoder block benchmark CLI
│   │   ├── bench_proxy.py              ← GPU heuristic / accuracy proxy CLI
│   │   └── bench_matmul.py             ← Sparse GEMM accuracy benchmark CLI
│   │
│   ├── metrics/
│   │   ├── accuracy.py                 ← relative_error, cosine_sim, KL divergence
│   │   ├── latency.py                  ← CUDA-event timing
│   │   ├── memory.py                   ← Peak memory measurement
│   │   ├── flops.py                    ← Theoretical FLOPs / arithmetic intensity
│   │   └── sparse_matmul.py            ← Sparse GEMM metrics
│   │
│   ├── models/                         ← Pluggable decoder models (for bench_decoder)
│   │   ├── attention_layer.py
│   │   ├── decoder_block.py
│   │   ├── kv_cache.py
│   │   └── proxy_models.py
│   │
│   ├── proxy/                          ← Roofline-based GPU speedup estimator
│   │   ├── estimator.py
│   │   ├── evaluator.py
│   │   ├── gpu_profiles.py
│   │   ├── roofline.py
│   │   └── plotting.py
│   │
│   ├── experiment_configs/             ← YAML experiment configs
│   │   ├── sweep_seq_len.yaml
│   │   ├── bigbird.yaml
│   │   ├── topk.yaml
│   │   ├── local_window.yaml
│   │   └── dense.yaml
│   │
│   ├── outputs/                        ← Generated results (gitignored)
│   │   ├── json/                       ← Sweep JSON results
│   │   ├── csv/                        ← Sweep CSV results
│   │   └── figures/                    ← Latency plots
│   │
│   └── config.py                       ← ExperimentConfig dataclass
│
├── kernels/                            ← real Triton / CUDA sparse kernels
│   ├── triton/                         ← @triton.jit kernel files (.py)
│   └── cuda/                           ← CUDA extension (setup.py build)
│       ├── setup.py                    ← builds the .so via CUDAExtension
│       ├── __init__.py                 ← loads the .so; imports ops wrappers
│       ├── README.md                   ← CUDA kernel development guide
│       ├── build/                      ← compiled SparseAttentionExtension.so
│       ├── src/csrc/
│       │   ├── bind.cu                 ← TORCH_LIBRARY registration + PYBIND11_MODULE
│       │   └── helloWorldKernel.cu     ← example kernel (no main())
│       ├── src/torch_wrappers/
│       │   └── ops.py                  ← Python-facing op wrappers + register_fake
│       └── test/
│           └── test_helloworld.py      ← plain Python test
│
├── profiling/                          ← standalone profiling scripts + outputs
│   ├── sparse_prof.py
│   ├── sparse_decoder_prof.py
│   ├── gpu_profiles.toml
│   └── out/                            ← profiling plots + JSON
│
├── requirements.txt
├── pyproject.toml
├── DEVELOPER_GUIDE.md                  ← this file
└── README.md
```

---

## 2. Core Concepts

The harness separates two orthogonal concerns:

```
SparsePattern.build(q, k)  →  PatternMetadata
        ↓ describes WHICH positions can attend
AttentionBackend.forward(q, k, v, pattern)  →  output tensor
        ↓ describes HOW to compute attention
```

**`PatternMetadata`** carries four `kind` values:

| `kind`    | What it means | Backend behaviour |
|-----------|---------------|-------------------|
| `"dense"` | Full causal attention | Backend ignores mask |
| `"mask"`  | Pre-built `[H, T, T]` bool mask (True = attend) | Backend applies mask before softmax |
| `"local"` | Same as mask, semantically a sliding window | `GatherSparseBackend` can use gather |
| `"topk"`  | Data-dependent; only `topk` int is set | Backend selects top-k from raw scores |

---

## 3. Adding a Custom Sparse Pattern

This is the **only file you need to write**. The rest of the harness picks it up automatically.

### Step 1 — Create `sparse_attention_bench/patterns/my_pattern.py`

```python
import torch
from sparse_attention_bench.patterns.base import PatternMetadata, SparsePattern


class MyPattern(SparsePattern):
    """
    Replace this docstring with what makes your pattern special.
    """

    def __init__(self, my_param: int):
        self.my_param = my_param
        # Optional: add a mask cache to avoid rebuilding each forward pass
        self._cache: dict = {}

    def build(self, q: torch.Tensor, k: torch.Tensor, causal: bool = True) -> PatternMetadata:
        """
        Args:
            q: [B, H, T_q, D]
            k: [B, H, T_k, D]
            causal: enforce upper-triangular masking on top of your pattern
        Returns:
            PatternMetadata
        """
        B, H, T_q, D = q.shape
        T_k = k.size(2)
        device = q.device

        # --- Build your boolean attention mask ---
        # True  = this (query, key) pair is ALLOWED to attend
        # False = masked out (set to -inf before softmax)

        mask = self._build_mask(H, T_q, T_k, device, causal)

        # Estimate what fraction of attention scores are kept (for reporting)
        keep_ratio = mask.float().mean().item()

        return PatternMetadata(kind="mask", mask=mask, keep_ratio=keep_ratio)

    def _build_mask(
        self, H: int, T_q: int, T_k: int,
        device: torch.device, causal: bool,
    ) -> torch.Tensor:
        # --- Replace this with your actual mask logic ---
        # Example: attend to every other key position
        cols = torch.arange(T_k, device=device)
        row_mask = (cols % 2 == 0).unsqueeze(0).unsqueeze(0)     # [1, 1, T_k]
        mask = row_mask.expand(H, T_q, T_k).clone()

        if causal:
            rows = torch.arange(T_q, device=device)
            causal_ok = cols.unsqueeze(0) <= rows.unsqueeze(1)    # [T_q, T_k]
            mask = mask & causal_ok.unsqueeze(0)

        return mask   # [H, T_q, T_k] bool
```

> **Tips:**
> - Keep `build()` fast — it is timed separately from the attention compute.
> - Cache the mask if it only depends on `(T_q, T_k, H, device)` and your parameters.
>   See `LocalWindowPattern._mask_cache` for an example.
> - If your pattern is data-dependent (needs QK scores to decide which keys to keep),
>   use `kind="topk"` and set `topk=<int>`. The `MaskedSdpaBackend` handles the
>   actual top-k selection from raw scores. See `TopKPattern` for an example.

### Step 2 — Register it in the benchmark runner

Open [`sparse_attention_bench/runners/benchmark_runner.py`](sparse_attention_bench/runners/benchmark_runner.py) and add your pattern to `_get_pattern()`:

```python
# At the top of the file, add your import:
from sparse_attention_bench.patterns.my_pattern import MyPattern

def _get_pattern(cfg: ExperimentConfig):
    ...
    # Add this block:
    if cfg.pattern_type == "my_pattern":
        assert cfg.block_size is not None, "block_size must be set for my_pattern"
        return MyPattern(my_param=cfg.block_size)
    ...
```

> `ExperimentConfig` has spare fields you can reuse: `topk`, `window_size`, `block_size`.
> Pick whichever matches your parameter semantically.

### Step 3 — Done. Run a quick single-layer smoke test:

```bash
python -m sparse_attention_bench.benchmarks.bench_layer \
    --pattern my_pattern \
    --block-size 64 \
    --seq-len 512 \
    --device cpu
```

---

## 4. Adding a Custom Compute Backend (Optional)

Most patterns work fine with `masked_sdpa`. Only implement a custom backend if you have a **real sparse CUDA kernel** or want a fundamentally different compute path.

### Create `sparse_attention_bench/attention/my_backend.py`

```python
import torch
from sparse_attention_bench.attention.base import AttentionBackend
from sparse_attention_bench.patterns.base import PatternMetadata


class MyBackend(AttentionBackend):
    def forward(
        self,
        q: torch.Tensor,        # [B, H, T_q, D]
        k: torch.Tensor,        # [B, H, T_k, D]
        v: torch.Tensor,        # [B, H, T_k, D]
        pattern: PatternMetadata,
    ) -> torch.Tensor:
        # Your custom kernel / compute logic here
        ...
        return output   # [B, H, T_q, D]
```

### Register it in `sparse_attention_bench/attention/__init__.py`

```python
from sparse_attention_bench.attention.my_backend import MyBackend

_REGISTRY: dict[str, type[AttentionBackend]] = {
    "dense_sdpa":    DenseSdpaBackend,
    "masked_sdpa":   MaskedSdpaBackend,
    "gather_sparse": GatherSparseBackend,
    "my_backend":    MyBackend,          # ← add this line
}
```

---

## 5. Writing a Real Sparse Kernel (Triton / CUDA)

Sections 3 and 4 let you benchmark algorithms that still run through PyTorch ops
(masked SDPA, gather/scatter). When you are ready to write a **real sparse kernel**
that avoids computing masked-out scores entirely, use the `kernels/` folder.

---

### 5a. Triton kernel

Put a `@triton.jit` function in `kernels/triton/your_kernel.py`.
The Python entry point must match this signature:

```python
def triton_<name>_attn(
    q: torch.Tensor,   # [B, H, T_q, D]  fp16 / bf16, contiguous
    k: torch.Tensor,   # [B, H, T_k, D]
    v: torch.Tensor,   # [B, H, T_k, D]
    **kernel_args,     # e.g. top_k=64, causal=True
) -> torch.Tensor:     # [B, H, T_q, D]
```

---

### 5b. CUDA kernel — step by step

The `kernels/cuda/` folder is a pre-built PyTorch extension. All CUDA ops share
one compiled `.so` and one namespace (`cuda_sparse_attention`).
See [`kernels/cuda/README.md`](kernels/cuda/README.md) for the full reference.

#### Step 1 — Write the kernel — `kernels/cuda/src/csrc/<name>Kernel.cu`

```cuda
// No main(). Only __global__ kernel + a host wrapper that returns a Tensor.
#include <torch/extension.h>

__global__ void myKernel(/* args */) { ... }

torch::Tensor my_op_impl(torch::Tensor q, torch::Tensor k) {
    // launch kernel, synchronize, return output tensor
}
```

#### Step 2 — Register in `kernels/cuda/src/csrc/bind.cu`

```cpp
#include "<name>Kernel.cu"

TORCH_LIBRARY_FRAGMENT(cuda_sparse_attention, m) {
    m.def("my_op(Tensor q, Tensor k) -> Tensor");
}

// Use CUDA when op has tensor inputs (dispatcher infers device from tensors).
// Use CompositeExplicitAutograd when op has NO tensor inputs.
TORCH_LIBRARY_IMPL(cuda_sparse_attention, CUDA, m) {
    m.impl("my_op", &my_op_impl);
}
```

> **Namespace:** always use `cuda_sparse_attention`. Never use `cuda` — PyTorch
> already owns that namespace and a second `TORCH_LIBRARY(cuda, …)` will crash at
> load time. Use `TORCH_LIBRARY_FRAGMENT` (not `TORCH_LIBRARY`) when adding ops
> to the shared namespace across multiple files.

#### Step 3 — Add a Python wrapper — `kernels/cuda/src/torch_wrappers/ops.py`

```python
def my_op(q: Tensor, k: Tensor) -> Tensor:
    return torch.ops.cuda_sparse_attention.my_op.default(q, k)

@torch.library.register_fake("cuda_sparse_attention::my_op")
def _(q, k):
    return torch.empty_like(q)   # describe output shape for torch.compile
```

#### Step 4 — Build the extension

```bash
# from kernels/cuda/
CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH \
    python setup.py build_ext --inplace
```

The `.so` is placed in `kernels/cuda/build/`. Loading `kernels.cuda` anywhere
in Python triggers `__init__.py`, which imports the `.so` and registers all ops.

#### Step 5 — Write a test — `kernels/cuda/test/test_<name>.py`

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # project root

import torch
import kernels.cuda  # loads .so, registers all cuda_sparse_attention ops

def test_my_op():
    q = torch.randn(1, 4, 16, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 4, 16, 64, device="cuda", dtype=torch.float16)
    out = torch.ops.cuda_sparse_attention.my_op(q, k)
    assert out.shape == q.shape

if __name__ == "__main__":
    test_my_op()
    print("PASSED")
```

```bash
python kernels/cuda/test/test_<name>.py
```

---

### Step 6 — Wrap as an AttentionBackend (both Triton and CUDA)

```python
# sparse_attention_bench/attention/my_kernel_backend.py
from sparse_attention_bench.attention.base import AttentionBackend
import kernels.cuda  # noqa: F401

class MyKernelBackend(AttentionBackend):
    def forward(self, q, k, v, pattern):
        return torch.ops.cuda_sparse_attention.my_op(q, k, v)
```

Register it in `attention/__init__.py`:

```python
from sparse_attention_bench.attention.my_kernel_backend import MyKernelBackend
_REGISTRY["my_kernel"] = MyKernelBackend
```

The backend receives a `PatternMetadata` object with:

| Field | Type | Set when |
|---|---|---|
| `kind` | `str` | always — `"dense"`, `"mask"`, `"topk"`, `"local"` |
| `mask` | `Tensor [H,T,T]` bool | `kind="mask"` or `"local"` |
| `topk` | `int` | `kind="topk"` |
| `keep_ratio` | `float` | always — useful for reporting |

Then use it in a sweep YAML:

```yaml
patterns:
  - type: topk
    backend: my_kernel    # your real kernel
    topk: [32, 64, 128]
  - type: topk
    backend: masked_sdpa  # PyTorch baseline for comparison
    topk: [32, 64, 128]
```

The benchmark runner automatically measures and compares:
- **Pattern build time** vs **kernel compute time** separately
- Accuracy vs dense SDPA baseline (`rel_err`, `cosine_sim`)
- Peak memory

---

## 6. Running Benchmarks

### 6.1 Single attention layer — quick check

```bash
# With your new pattern
python -m sparse_attention_bench.benchmarks.bench_layer \
    --pattern my_pattern \
    --block-size 64 \
    --seq-len 512 \
    --num-heads 8 \
    --head-dim 64 \
    --dtype fp16 \
    --device cuda

# Built-in patterns for comparison
python -m sparse_attention_bench.benchmarks.bench_layer --pattern dense --seq-len 512
python -m sparse_attention_bench.benchmarks.bench_layer --pattern topk  --topk 64 --seq-len 512
```

Output (JSON to stdout): latency mean/p50/p95, peak memory, relative error vs dense.

### 6.2 Sweep across seq_len / sparsity — with plots

**Create a YAML config** (e.g. `sparse_attention_bench/experiment_configs/my_sweep.yaml`):

```yaml
experiment:
  batch_size: 1
  num_heads: 8
  head_dim: 64
  seq_len: [128, 512, 1024, 2048, 4096]
  dtype: fp16
  device: cuda
  mode: [prefill, decode]
  causal: true

  patterns:
    - type: dense
      backend: dense_sdpa

    - type: my_pattern          # your new algorithm
      backend: masked_sdpa
      block_size: [32, 64, 128] # swept automatically

    - type: topk                # comparison baseline
      backend: masked_sdpa
      topk: [64, 128]

  num_warmup: 20
  num_iters: 100
```

**Run the sweep with graph output:**

```bash
python -m sparse_attention_bench.runners.sweep_runner \
    --config sparse_attention_bench/experiment_configs/my_sweep.yaml \
    --tag my_sweep \
    --plot
```

Outputs:
- `sparse_attention_bench/outputs/json/my_sweep.json`
- `sparse_attention_bench/outputs/csv/my_sweep.csv`
- `sparse_attention_bench/outputs/figures/my_sweep_total_time_ms_mean.png`
- `sparse_attention_bench/outputs/figures/my_sweep_attention_time_ms_mean.png`
- `sparse_attention_bench/outputs/figures/my_sweep_pattern_build_time_ms_mean.png`

### 6.3 Full decoder block benchmark

Tests your pattern inside a stacked transformer decoder (embedding → N×DecoderBlock → LM head):

```bash
# Prefill: measure full forward pass at one sequence length
python -m sparse_attention_bench.benchmarks.bench_decoder \
    --pattern my_pattern \
    --block-size 64 \
    --seq-len 512 \
    --num-layers 4 \
    --mode prefill

# Decode: single new token step with pre-filled KV cache
python -m sparse_attention_bench.benchmarks.bench_decoder \
    --pattern my_pattern \
    --block-size 64 \
    --seq-len 4096 \
    --num-layers 4 \
    --mode decode \
    --kv-lens 128 512 2048
```

### 6.4 Proxy benchmark — heuristic accuracy + GPU speedup curves

No real GPU required. Estimates accuracy degradation and theoretical GPU speedup across seq_lens for `top-k` and `bigbird` using a roofline model:

```bash
python -m sparse_attention_bench.benchmarks.bench_proxy \
    --gpu-profile rtx_4090 \
    --num-trials 10 \
    --no-show \
    --output-dir outputs/proxy

# List available GPU profiles (rtx_4090, a100_sxm80, h100_sxm80, l40s, generic_ampere)
python -m sparse_attention_bench.benchmarks.bench_proxy --list-gpu-profiles
```

Outputs 3 PNG plots to `sparse_attention_bench/outputs/proxy/`: relative error, KL divergence, top-1 match rate, plus prefill/decode speedup curves.

---

## 7. Understanding the Outputs

### Latency fields (from bench_layer / sweep)

| Field | Description |
|---|---|
| `pattern_build_time_ms_mean` | Time to build the sparse mask (your `build()` method) |
| `attention_time_ms_mean` | Time for the attention kernel (`backend.forward()`) |
| `total_time_ms_mean` | End-to-end: build + compute |
| `peak_memory_mb` | GPU peak memory during one full forward |

### Accuracy fields

| Field | Description |
|---|---|
| `rel_err` | `‖sparse_out − dense_out‖ / ‖dense_out‖` — lower is better |
| `cosine_sim` | Cosine similarity between sparse and dense outputs — higher is better |
| `keep_ratio` | Fraction of attention scores not masked out |
| `sparsity_ratio` | `1 − keep_ratio` |

### Timing is measured correctly

- **GPU**: CUDA events (`torch.cuda.Event(enable_timing=True)`) bracket each call; synchronised after.
- **CPU**: `time.perf_counter` with explicit sync before/after.
- Pattern build time and attention compute time are measured **independently**, so you can see whether your overhead is in mask construction or in the kernel.

---

## 8. Quick Reference

```bash
# Install dependencies
pip install -r requirements.txt

# Smoke test (no GPU needed)
python -m sparse_attention_bench.benchmarks.bench_layer \
    --pattern dense --seq-len 128 --device cpu

# Add your pattern → register in benchmark_runner.py → run:
python -m sparse_attention_bench.benchmarks.bench_layer \
    --pattern my_pattern --block-size 64 --seq-len 512 --device cuda

# Sweep with plots
python -m sparse_attention_bench.runners.sweep_runner \
    --config sparse_attention_bench/experiment_configs/my_sweep.yaml --tag my_sweep --plot

# Proxy heuristic plots (no GPU needed)
python -m sparse_attention_bench.benchmarks.bench_proxy \
    --gpu-profile rtx_4090 --no-show
```

### Available built-in patterns

| `--pattern` | Key parameter | `kind` returned |
|---|---|---|
| `dense` | — | `dense` |
| `topk` | `--topk K` | `topk` |
| `bigbird` | `--topk K` | `mask` |
| `local_window` | `--window-size W` | `local` |

### Available backends

| `--backend` | Best for |
|---|---|
| `dense_sdpa` | Dense baseline; uses PyTorch fused SDPA |
| `masked_sdpa` | Any pattern with a pre-built mask or top-k |
| `gather_sparse` | Local/window patterns; gathers K/V for a smaller GEMM |
