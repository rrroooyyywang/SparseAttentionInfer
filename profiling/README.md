# Profiling

This directory contains two kinds of profiling / proxy profiling scripts:

- `sparse_prof.py`
  A matrix-level toy experiment that uses random matrices and global `top-k` sparsification to observe approximation accuracy and changes in theoretical MACs.
- `sparse_decoder_prof.py`
  A decoder-level proxy profiler that compares dense and sparse model outputs using random tokens, while also using a phase-aware GPU proxy model to estimate speedups for the `prefill` and `decode` phases.
- `gpu_profiles.toml`
  GPU architecture parameter configuration. `sparse_decoder_prof.py` reads profiles from this file instead of hardcoding hardware parameters in Python.
- `out/`
  Directory where generated figures are saved after running the scripts.

## Environment

It is recommended to run these scripts in the project environment:

```bash
uv run python3 profiling/sparse_prof.py
uv run python3 profiling/sparse_decoder_prof.py --no-show
```

If you are not using `uv`, make sure your current Python environment at least has `torch` and `matplotlib`.

## 1. Run `sparse_prof.py`

This is the simplest matrix-level profiling script.

```bash
uv run python3 profiling/sparse_prof.py
```

Current default behavior:

- Only runs `single_profile(100)`
- Uses `m=n=512`
- Evaluates one-sided sparse matrix multiplication
- Prints average accuracy, estimated MACs, and theoretical speedup

If you want to test two-sided sparsity or change the matrix size, you need to modify the function entry points and constants directly in [sparse_prof.py](/Users/yuemingwang/Doc/IC/Year4-Term2/ADLS/Sparse/profiling/sparse_prof.py).

## 2. Run `sparse_decoder_prof.py`

First, list the available GPU profiles:

```bash
uv run python3 profiling/sparse_decoder_prof.py --list-gpu-profiles
```

Run with a specific profile:

```bash
uv run python3 profiling/sparse_decoder_prof.py --gpu-profile generic_ampere --no-show
uv run python3 profiling/sparse_decoder_prof.py --gpu-profile rtx_4090 --no-show
uv run python3 profiling/sparse_decoder_prof.py --gpu-profile a100_sxm80 --no-show
uv run python3 profiling/sparse_decoder_prof.py --gpu-profile rtx_4090 --quiet --no-show
```

If you have your own parameter file, you can also specify it explicitly:

```bash
uv run python3 profiling/sparse_decoder_prof.py \
  --gpu-profile my_gpu \
  --gpu-profile-path profiling/gpu_profiles.toml \
  --no-show
```

The script does the following:

1. Runs dense / sparse decoder output comparisons under `prefill` semantics and records `rel_err`, `kl_div`, and `top1_match`
2. Uses a GPU proxy model to estimate `gpu_speedup_est` for both the `prefill` and `decode` phases
3. Saves one accuracy figure and two speedup figures to `profiling/out/`
4. Saves a structured JSON file to `profiling/out/` for later comparison with real kernel benchmarks

Typical output filenames:

- `sparse_decoder_accuracy_*.png`
- `sparse_decoder_speedup_prefill_*.png`
- `sparse_decoder_speedup_decode_*.png`
- `sparse_decoder_proxy_profile_*.json`

The JSON includes:

- The GPU profile parameters used for the run
- Sweep configuration such as `seq_lens`, `keep_ratio`, and `batch_size`
- `accuracy_records`
  Accuracy summaries for each `seq_len / sparse_mode / top_k`
- `speedup_records`
  Proxy timings and speedup estimates for each `phase / seq_len / sparse_mode / top_k`
- Paths to the generated figures

Path fields in the JSON are saved as paths relative to the current workspace, not absolute local machine paths.

Terminal behavior:

- By default, the script prints the current configuration and a summary for each `seq_len / sparse_mode`
- It shows a global progress bar that updates according to the actual executed accuracy trials
- The progress bar includes `trials/s`
- If you only want the progress bar and final output file paths, add `--quiet`

## 3. Modify GPU profiles

GPU profiles are stored in [gpu_profiles.toml](/Users/yuemingwang/Doc/IC/Year4-Term2/ADLS/Sparse/profiling/gpu_profiles.toml).

Each profile corresponds to a `[profiles.<name>]` table, for example:

```toml
[profiles.rtx_4090]
description = "NVIDIA RTX 4090 proxy using official Ada Lovelace specs plus L2-aware heuristics"
dense_tensor_tflops = 165.2
block_sparse_tensor_tflops = 82.0
dynamic_sparse_tflops = 24.0
vector_tflops = 26.4
memory_bandwidth_gbps = 1008.0
memory_bandwidth_efficiency = 0.88
kernel_launch_overhead_us = 3.4
activation_bytes = 2
weight_bytes = 2
score_bytes = 4
index_bytes = 4
softmax_flops_per_element = 8.0
topk_compare_flops_per_element = 2.0
sm_count = 128
tensor_cores = 512
boost_clock_mhz = 2520
l2_cache_kb = 73728
decode_l2_hit_rate_max = 0.70
```

Key fields you may want to tune:

- `dense_tensor_tflops`
  Effective throughput proxy for dense Tensor Core GEMM
- `block_sparse_tensor_tflops`
  Effective throughput proxy for structured / block-sparse attention kernels
- `dynamic_sparse_tflops`
  Effective throughput proxy for dynamic top-k sparse kernels
- `vector_tflops`
  Throughput proxy for vector operators such as softmax, top-k, and scatter
- `memory_bandwidth_gbps`
  Bandwidth proxy
- `kernel_launch_overhead_us`
  Kernel launch overhead proxy

## Limitations

These profiling results should always be interpreted together with the following limitations.

### Limitations of `sparse_prof.py`

- It is not a real sparse kernel benchmark. The script first sparsifies matrices and then still calls dense `torch.mm`.
- It reports theoretical MACs / theoretical speedup, not CUDA event timing or wall-clock runtime.
- The sparsity pattern is "global top-k after flattening", which differs from common attention sparsity patterns such as row-wise top-k, block sparse, or BigBird.
- Inputs are random matrices and do not represent real model activation distributions.

### Limitations of `sparse_decoder_prof.py`

- This is not a real performance test of native PyTorch sparse attention; it is a runtime proxy.
- The current forward implementation still forms dense attention scores before applying `top-k` or masking, so real runtime will not directly match the speedup shown in the figures.
- The `prefill` / `decode` plots depend on heuristic parameters in `gpu_profiles.toml`, not measurements from Nsight, CUDA profilers, or actual kernel benchmarks.
- The current `decode` model represents the proxy cost of a "single-token autoregressive step + KV cache reads", and does not include all real overheads in a serving system.
- It does not model kernel fusion, stream overlap, allocator behavior, framework scheduling overhead, paged KV cache, tensor parallelism, or pipeline parallelism.
- The cost of `top-k` enters the model through an approximate formula, not instruction-level measurement of a specific CUDA implementation.
- The speedup reported for `BigBird` is likewise a structured sparsity proxy and does not represent measured performance from any specific library on any specific GPU.

### Limitations of the data and task setup

- The accuracy experiments use randomly initialized decoder weights and random tokens, not a trained language model.
- Therefore, `rel_err`, `kl_div`, and `top1_match` are better interpreted as perturbations relative to the dense baseline, not as direct predictors of real task quality.
- The `seq_lens`, `percentage_list`, `batch_size`, and `num_trials` in `__main__` are currently hardcoded in the script rather than exposed as a full CLI.
- If you run the full `sparse_decoder_prof.py` on CPU, the accuracy sweep may be slow because it really executes dense and sparse model forwards.

## How to interpret the results

- `sparse_prof.py`
  Best suited for answering: "In a highly simplified matrix model, how do theoretical MACs scale after sparsification?"
- `sparse_decoder_prof.py`
  Best suited for answering: "In the context of decoder attention, given a set of GPU architecture proxy parameters, what do the theoretical speed trends and accuracy perturbations for `prefill` / `decode` look like?"
- If you need to report real performance, you should additionally collect measured data from CUDA events, `torch.profiler`, Nsight Systems, or Nsight Compute.
