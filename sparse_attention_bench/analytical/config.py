from dataclasses import dataclass

import torch


@dataclass
class DecoderConfig:
    vocab_size: int = 5000
    max_seq_len: int = 128
    sparse_mode: str = "top-k"
    d_model: int = 256
    n_heads: int = 8
    mlp_ratio: int = 4
    n_layers: int = 2
    dropout: float = 0.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class NvidiaGpuHeuristic:
    """
    Lightweight roofline-style proxy for recent NVIDIA GPUs.
    """
    profile_name: str = "default"
    description: str = "Generic recent NVIDIA GPU proxy"
    dense_tensor_tflops: float = 125.0
    block_sparse_tensor_tflops: float = 62.0
    dynamic_sparse_tflops: float = 18.0
    vector_tflops: float = 20.0
    memory_bandwidth_gbps: float = 900.0
    memory_bandwidth_efficiency: float = 1.0
    kernel_launch_overhead_us: float = 3.5
    activation_bytes: int = 2
    weight_bytes: int = 2
    score_bytes: int = 4
    index_bytes: int = 4
    softmax_flops_per_element: float = 8.0
    topk_compare_flops_per_element: float = 2.0
    sm_count: int = 0
    tensor_cores: int = 0
    boost_clock_mhz: float = 0.0
    peak_fp32_tflops: float = 0.0
    peak_fp16_tensor_tflops: float = 0.0
    peak_fp16_tensor_sparse_tflops: float = 0.0
    l1_shared_kb_total: int = 0
    l2_cache_kb: int = 0
    register_file_kb_total: int = 0
    l2_residency_fraction: float = 0.0
    prefill_l2_hit_rate_max: float = 0.0
    decode_l2_hit_rate_max: float = 0.0
    l2_hit_bandwidth_multiplier: float = 1.0


VALID_EXECUTION_PHASES = ("prefill", "decode")
