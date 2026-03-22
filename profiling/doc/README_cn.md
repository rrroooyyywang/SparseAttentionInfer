# Profiling

这个目录放的是两类 profiling / proxy profiling 脚本：

- `sparse_prof.py`
  一个矩阵级别的 toy experiment，用随机矩阵和全局 `top-k` 稀疏化，观察近似精度和理论 MACs 变化。
- `sparse_decoder_prof.py`
  一个 decoder 级别的 proxy profiler，用随机 token 和 dense/sparse 模型输出对比精度，同时用一个 phase-aware 的 GPU 代理模型估计 `prefill` / `decode` 两阶段的速度收益。
- `gpu_profiles.toml`
  GPU 架构参数配置。`sparse_decoder_prof.py` 会从这里读取 profile，而不是把硬件参数硬编码在 Python 里。
- `out/`
  运行脚本后保存图片的目录。

## 环境

推荐用项目环境运行：

```bash
uv run python3 profiling/sparse_prof.py
uv run python3 profiling/sparse_decoder_prof.py --no-show
```

如果你不用 `uv`，至少要保证当前 Python 环境里有 `torch` 和 `matplotlib`。

## 1. 运行 `sparse_prof.py`

这个脚本是最简单的矩阵级 profiling。

```bash
uv run python3 profiling/sparse_prof.py
```

当前默认行为：

- 只会运行 `single_profile(100)`
- 使用 `m=n=512`
- 对单边稀疏矩阵乘法做实验
- 打印平均精度、估计 MACs、理论 speedup

如果你想跑双边稀疏或改矩阵大小，需要直接修改 [sparse_prof.py](/Users/yuemingwang/Doc/IC/Year4-Term2/ADLS/Sparse/profiling/sparse_prof.py) 里的函数入口和常量。

## 2. 运行 `sparse_decoder_prof.py`

先看有哪些 GPU profile：

```bash
uv run python3 profiling/sparse_decoder_prof.py --list-gpu-profiles
```

用某个 profile 运行：

```bash
uv run python3 profiling/sparse_decoder_prof.py --gpu-profile generic_ampere --no-show
uv run python3 profiling/sparse_decoder_prof.py --gpu-profile rtx_4090 --no-show
uv run python3 profiling/sparse_decoder_prof.py --gpu-profile a100_sxm80 --no-show
uv run python3 profiling/sparse_decoder_prof.py --gpu-profile rtx_4090 --quiet --no-show
```

如果你有自己的参数文件，也可以指定：

```bash
uv run python3 profiling/sparse_decoder_prof.py \
  --gpu-profile my_gpu \
  --gpu-profile-path profiling/gpu_profiles.toml \
  --no-show
```

脚本会做这些事：

1. 在 `prefill` 语义下跑 dense / sparse decoder 输出对比，统计 `rel_err`、`kl_div`、`top1_match`
2. 用 GPU proxy model 估计 `prefill` 和 `decode` 两阶段的 `gpu_speedup_est`
3. 保存一张 accuracy 图和两张 speedup 图到 `profiling/out/`
4. 保存一份结构化 JSON 到 `profiling/out/`，用于后续和真实 kernel benchmark 对比

典型输出文件名：

- `sparse_decoder_accuracy_*.png`
- `sparse_decoder_speedup_prefill_*.png`
- `sparse_decoder_speedup_decode_*.png`
- `sparse_decoder_proxy_profile_*.json`

JSON 里会包含：

- 本次运行使用的 GPU profile 参数
- 实验 sweep 配置，例如 `seq_lens`、`keep_ratio`、`batch_size`
- `accuracy_records`
  每个 `seq_len / sparse_mode / top_k` 下的精度摘要
- `speedup_records`
  每个 `phase / seq_len / sparse_mode / top_k` 下的代理时间和 speedup 估计
- 生成的图片路径

JSON 里的路径字段会保存为相对当前工作区的相对路径，不会写入本机绝对路径。

终端体验：

- 默认会打印当前配置、每个 `seq_len / sparse_mode` 的摘要
- 会显示一个全局进度条，按真实执行的 accuracy trial 更新
- 进度条里会显示 `trials/s`
- 如果你只想保留进度条和最终输出文件路径，可以加 `--quiet`

## 3. 修改 GPU profile

GPU profile 放在 [gpu_profiles.toml](/Users/yuemingwang/Doc/IC/Year4-Term2/ADLS/Sparse/profiling/gpu_profiles.toml)。

一个 profile 对应一个 `[profiles.<name>]` 表，例如：

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

可以改的核心字段：

- `dense_tensor_tflops`
  dense Tensor Core GEMM 的有效吞吐代理
- `block_sparse_tensor_tflops`
  结构化 / block-sparse attention kernel 的有效吞吐代理
- `dynamic_sparse_tflops`
  动态 top-k 稀疏 kernel 的有效吞吐代理
- `vector_tflops`
  softmax、top-k、scatter 这类向量算子的吞吐代理
- `memory_bandwidth_gbps`
  带宽代理
- `kernel_launch_overhead_us`
  kernel launch 开销代理

## 限制

这些 profiling 结果一定要结合下面的限制一起看。

### `sparse_prof.py` 的限制

- 它不是实际 sparse kernel benchmark。脚本先把矩阵稀疏化，再调用 dense `torch.mm`。
- 它报告的是理论 MACs / 理论 speedup，不是 CUDA event 或 wall-clock runtime。
- 稀疏模式是“全局 flatten 之后的 top-k”，这和 attention 里常见的 row-wise top-k、block sparse、BigBird pattern 都不一样。
- 输入是随机矩阵，不代表真实模型激活分布。

### `sparse_decoder_prof.py` 的限制

- 这不是 PyTorch 原生 sparse attention 的真实性能测试，而是 runtime proxy。
- 当前前向实现仍然会先形成 dense attention scores，再做 `top-k` 或 mask，所以真实运行时间不会直接等于图里的 speedup。
- `prefill` / `decode` 图依赖 `gpu_profiles.toml` 里的启发式参数，不是来自 Nsight、CUDA profiler 或实测 kernel benchmark。
- `decode` 现在建模的是“单 token 自回归步骤 + KV cache 读取”的代理成本，不包含 serving 系统里的所有真实开销。
- 没有建模 kernel fusion、stream overlap、allocator 行为、框架调度开销、分页 KV cache、tensor parallel、pipeline parallel。
- `top-k` 的代价是通过近似公式进入模型的，不是对某个具体 CUDA 实现逐指令测出来的。
- `BigBird` 的速度收益同样是结构化稀疏代理，不代表某个现成库在某张卡上的实测值。

### 数据与任务本身的限制

- accuracy 部分用的是随机初始化的 decoder 权重和随机 token，不是训练好的语言模型。
- 所以 `rel_err`、`kl_div`、`top1_match` 更适合看“稀疏近似相对 dense baseline 的扰动”，不适合直接外推到真实任务质量。
- `__main__` 里的 `seq_lens`、`percentage_list`、`batch_size`、`num_trials` 目前是写死在脚本里的，不是完整 CLI。
- 如果你在 CPU 上跑完整 `sparse_decoder_prof.py`，accuracy sweep 会比较慢，因为它真的会执行 dense/sparse 模型前向。

## 结果该怎么解读

- `sparse_prof.py`
  更适合回答“在一个非常简化的矩阵模型里，稀疏化后理论 MACs 怎么缩放”。
- `sparse_decoder_prof.py`
  更适合回答“在 decoder attention 这个语境里，给定一种 GPU 架构代理参数，`prefill` / `decode` 的理论速度趋势和精度扰动大概是什么样子”。
- 如果你要报告真实性能，最好另外补上 CUDA event、`torch.profiler`、Nsight Systems 或 Nsight Compute 的实测数据。
