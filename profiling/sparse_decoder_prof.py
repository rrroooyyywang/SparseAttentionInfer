import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Config
# ============================================================

@dataclass
class DecoderConfig:
    vocab_size: int = 5000
    max_seq_len: int = 128
    d_model: int = 256
    n_heads: int = 8
    mlp_ratio: int = 4
    n_layers: int = 2
    dropout: float = 0.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class NvidiaGpuHeuristic:
    """
    Heuristic throughput factors relative to dense Tensor Core GEMM.
    These coefficients are not measured hardware values. They are only used to
    map the ideal change in operator workload to an effective speedup that is
    closer to unstructured sparse kernels on NVIDIA GPUs.
    """
    dense_efficiency: float = 1.0
    pair_sparse_efficiency: float = 0.35
    score_value_sparse_efficiency: float = 0.40


# ============================================================
# Utilities
# ============================================================

def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    # [1, 1, T, T]
    mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
        diagonal=1
    )
    return mask.unsqueeze(0).unsqueeze(0)


def relative_error(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    return (torch.norm(x - y) / (torch.norm(y) + eps)).item()


def cosine_sim(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    x_flat = x.reshape(-1)
    y_flat = y.reshape(-1)
    return F.cosine_similarity(x_flat, y_flat, dim=0, eps=eps).item()


def top1_match_rate(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
    # logits: [B, T, V]
    pred_a = logits_a.argmax(dim=-1)
    pred_b = logits_b.argmax(dim=-1)
    return (pred_a == pred_b).float().mean().item()


def mean_kl_divergence(reference_logits: torch.Tensor, approx_logits: torch.Tensor) -> float:
    """
    Average token-wise KL divergence: KL(reference || approx).
    """
    ref_log_probs = F.log_softmax(reference_logits.double(), dim=-1)
    approx_log_probs = F.log_softmax(approx_logits.double(), dim=-1)
    ref_probs = ref_log_probs.exp()
    token_kl = (ref_probs * (ref_log_probs - approx_log_probs)).sum(dim=-1)
    kl_value = token_kl.mean().item()
    return max(kl_value, 0.0)


def summarize_scalar(values) -> dict:
    values_t = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": values_t.mean().item(),
        "std": values_t.std(unbiased=False).item(),
    }


def sparsify_last_dim_topk(x: torch.Tensor, keep_k: int) -> torch.Tensor:
    """
    Keep only the top-k entries on the last dimension and zero out the rest.
    This simulates feature-sparse Q/K/V so both attention matmuls become sparse.
    """
    if keep_k >= x.size(-1):
        return x

    _, topk_idx = torch.topk(x.abs(), k=keep_k, dim=-1)
    sparse_x = torch.zeros_like(x)
    sparse_x.scatter_(-1, topk_idx, x.gather(-1, topk_idx))
    return sparse_x


def feature_keep_from_attention_keep(head_dim: int, attention_keep_ratio: float) -> int:
    return min(head_dim, max(1, math.ceil(head_dim * attention_keep_ratio)))


def estimate_decoder_sparse_gpu_efficiency(
    cfg: DecoderConfig,
    batch_size: int,
    seq_len: int,
    top_k: int,
    gpu: NvidiaGpuHeuristic | None = None,
):
    """
    Estimate GPU speedup for a decoder block using a runtime proxy.

    Assumptions:
    1. QK^T uses sparse Q/K features, so its effective workload scales with
       feature_keep_ratio^2.
    2. Attn @ V uses a sparse attention matrix times sparse V,
       so its effective workload scales with
       attention_keep_ratio * feature_keep_ratio.
    3. Q/K/V/O projections and MLP stay dense.

    Note:
    The current PyTorch implementation still forms dense attention scores before top-k,
    so this estimate is a model of a hypothetical sparse CUDA kernel. It should be read as
    a runtime proxy, not the real runtime of the code in this file as-is.
    """
    gpu = gpu or NvidiaGpuHeuristic()

    keep_ratio = min(top_k, seq_len) / seq_len
    d_model = cfg.d_model
    hidden = cfg.d_model * cfg.mlp_ratio
    head_dim = cfg.d_model // cfg.n_heads
    feature_keep_ratio = feature_keep_from_attention_keep(head_dim, keep_ratio) / head_dim

    dense_qk_cost = batch_size * seq_len * seq_len * d_model
    dense_av_cost = dense_qk_cost
    dense_proj_cost = 4 * batch_size * seq_len * d_model * d_model
    dense_mlp_cost = 2 * batch_size * seq_len * d_model * hidden

    dense_total_cost = cfg.n_layers * (
        dense_proj_cost + dense_mlp_cost + dense_qk_cost + dense_av_cost
    )

    sparse_qk_cost = dense_qk_cost * (feature_keep_ratio ** 2)
    sparse_av_cost = dense_av_cost * (keep_ratio * feature_keep_ratio)
    sparse_total_cost = cfg.n_layers * (
        dense_proj_cost + dense_mlp_cost + sparse_qk_cost + sparse_av_cost
    )

    dense_time_units = dense_total_cost / gpu.dense_efficiency
    sparse_time_units = cfg.n_layers * (
        dense_proj_cost / gpu.dense_efficiency
        + dense_mlp_cost / gpu.dense_efficiency
        + sparse_qk_cost / gpu.pair_sparse_efficiency
        + sparse_av_cost / gpu.score_value_sparse_efficiency
    )

    return {
        "keep_ratio": keep_ratio,
        "feature_keep_ratio": feature_keep_ratio,
        "attention_workload_reduction": 1.0 - (sparse_qk_cost + sparse_av_cost) / (dense_qk_cost + dense_av_cost),
        "runtime_proxy_reduction": 1.0 - sparse_total_cost / dense_total_cost,
        "gpu_speedup_est": dense_time_units / sparse_time_units,
    }


# ============================================================
# Attention Modules
# ============================================================

class DenseSelfAttention(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads

        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B,H,T,D]
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)     # [B,H,T,T]
        mask = causal_mask(T, x.device)
        scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)                                                   # [B,H,T,D]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class SparseSelfAttention(nn.Module):
    """
    Dual sparse attention approximation:
    1. sparsify Q/K/V on the feature dimension, so both matmuls are sparse;
    2. sparsify attention scores row-wise before softmax.
    """
    def __init__(self, cfg: DecoderConfig, top_k: int):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.top_k = top_k

        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        B, T, C = x.shape
        k_keep = min(self.top_k, T)
        feature_keep = feature_keep_from_attention_keep(self.head_dim, k_keep / T)

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)   # [B,H,T,D]
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Tie feature sparsity to attention sparsity so QK^T and Attn@V are both sparse.
        q = sparsify_last_dim_topk(q, keep_k=feature_keep)
        k = sparsify_last_dim_topk(k, keep_k=feature_keep)
        v = sparsify_last_dim_topk(v, keep_k=feature_keep)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)      # [B,H,T,T]

        # causal mask
        mask = causal_mask(T, x.device)
        scores = scores.masked_fill(mask, float("-inf"))

        # row-wise top-k
        # Apply top-k on the last dimension to keep the most important k keys
        # for each query position.
        topk_vals, topk_idx = torch.topk(scores, k=k_keep, dim=-1)                     # [B,H,T,K]

        sparse_scores = torch.full_like(scores, float("-inf"))
        sparse_scores.scatter_(-1, topk_idx, topk_vals)

        attn = F.softmax(sparse_scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)                                                    # [B,H,T,D]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


# ============================================================
# Decoder Block
# ============================================================

class FeedForward(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        hidden = cfg.d_model * cfg.mlp_ratio
        self.fc1 = nn.Linear(cfg.d_model, hidden)
        self.fc2 = nn.Linear(hidden, cfg.d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class DecoderBlock(nn.Module):
    def __init__(self, cfg: DecoderConfig, attn: nn.Module):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = attn
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ============================================================
# Dense Decoder / Sparse Decoder
# ============================================================

class DenseDecoder(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)

        self.blocks = nn.ModuleList([
            DecoderBlock(cfg, DenseSelfAttention(cfg))
            for _ in range(cfg.n_layers)
        ])

        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Common practice: weight tying
        self.lm_head.weight = self.token_emb.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [B, T]
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)                    # [1, T]

        x = self.token_emb(input_ids) + self.pos_emb(pos)

        for blk in self.blocks:
            x = blk(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)                                                       # [B, T, V]
        return logits


class SparseDecoder(nn.Module):
    def __init__(self, cfg: DecoderConfig, top_k: int):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)

        self.blocks = nn.ModuleList([
            DecoderBlock(cfg, SparseSelfAttention(cfg, top_k=top_k))
            for _ in range(cfg.n_layers)
        ])

        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)

        x = self.token_emb(input_ids) + self.pos_emb(pos)

        for blk in self.blocks:
            x = blk(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


# ============================================================
# Weight Copy
# ============================================================

def build_sparse_from_dense(dense_model: DenseDecoder, cfg: DecoderConfig, top_k: int) -> SparseDecoder:
    sparse_model = SparseDecoder(cfg, top_k=top_k).to(next(dense_model.parameters()).device)
    sparse_model.load_state_dict(dense_model.state_dict(), strict=True)
    return sparse_model


# ============================================================
# Accuracy Test
# ============================================================

@torch.no_grad()
def evaluate_sparse_decoder_once(
    batch_size: int = 4,
    seq_len: int = 64,
    top_k_list = (4, 8, 16, 32, 64),
    seed: int = 42,
):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cfg = DecoderConfig(max_seq_len=max(seq_len, 128))
    device = torch.device(cfg.device)
    gpu_heuristic = NvidiaGpuHeuristic()

    dense_model = DenseDecoder(cfg).to(device).eval()

    # Random token input
    input_ids = torch.randint(
        low=0,
        high=cfg.vocab_size,
        size=(batch_size, seq_len),
        device=device
    )

    dense_logits = dense_model(input_ids)

    results = []
    for top_k in top_k_list:
        sparse_model = build_sparse_from_dense(dense_model, cfg, top_k=top_k).eval()
        sparse_logits = sparse_model(input_ids)

        rel_err = relative_error(sparse_logits, dense_logits)
        kl_div = mean_kl_divergence(dense_logits, sparse_logits)
        cos = cosine_sim(sparse_logits, dense_logits)
        match = top1_match_rate(sparse_logits, dense_logits)

        speed_est = estimate_decoder_sparse_gpu_efficiency(
            cfg=cfg,
            batch_size=batch_size,
            seq_len=seq_len,
            top_k=top_k,
            gpu=gpu_heuristic,
        )
        keep_ratio = speed_est["keep_ratio"]
        results.append({
            "top_k": top_k,
            "keep_ratio": keep_ratio,
            "rel_err": rel_err,
            "kl_div": kl_div,
            "cos_sim": cos,
            "top1_match": match,
            "top1_mismatch_rate": 1.0 - match,
            **speed_est,
        })
    return results


@torch.no_grad()
def evaluate_sparse_decoder(
    batch_size: int = 4,
    seq_len: int = 64,
    top_k_list = (4, 8, 16, 32, 64),
    seed: int = 42,
    num_trials: int = 5,
):
    cfg = DecoderConfig(max_seq_len=max(seq_len, 128))
    device = torch.device(cfg.device)

    print(f"device      : {device}")
    print(f"batch_size  : {batch_size}")
    print(f"seq_len     : {seq_len}")
    print(f"d_model     : {cfg.d_model}")
    print(f"n_heads     : {cfg.n_heads}")
    print(f"n_layers    : {cfg.n_layers}")
    print(f"num_trials  : {num_trials}")
    print("-" * 72)

    trial_results = []
    for trial_idx in range(num_trials):
        trial_seed = seed + trial_idx
        trial_results.append(
            evaluate_sparse_decoder_once(
                batch_size=batch_size,
                seq_len=seq_len,
                top_k_list=top_k_list,
                seed=trial_seed,
            )
        )

    results = []
    for metric_idx, top_k in enumerate(top_k_list):
        samples = [trial_results[trial_idx][metric_idx] for trial_idx in range(num_trials)]
        summary = {
            "top_k": top_k,
            "keep_ratio": samples[0]["keep_ratio"],
            "feature_keep_ratio": samples[0]["feature_keep_ratio"],
            "rel_err": summarize_scalar([sample["rel_err"] for sample in samples]),
            "kl_div": summarize_scalar([sample["kl_div"] for sample in samples]),
            "cos_sim": summarize_scalar([sample["cos_sim"] for sample in samples]),
            "top1_match": summarize_scalar([sample["top1_match"] for sample in samples]),
            "top1_mismatch_rate": summarize_scalar([sample["top1_mismatch_rate"] for sample in samples]),
            "gpu_speedup_est": summarize_scalar([sample["gpu_speedup_est"] for sample in samples]),
            "attention_workload_reduction": summarize_scalar(
                [sample["attention_workload_reduction"] for sample in samples]
            ),
            "runtime_proxy_reduction": summarize_scalar(
                [sample["runtime_proxy_reduction"] for sample in samples]
            ),
            "num_trials": num_trials,
        }

        print(
            f"top_k={top_k:>4d} | keep_ratio={summary['keep_ratio']:>6.2%} | "
            f"feature_keep={summary['feature_keep_ratio']:>6.2%} | "
            f"rel_err={summary['rel_err']['mean']:>10.6f}±{summary['rel_err']['std']:.2e} | "
            f"kl={summary['kl_div']['mean']:>10.6f}±{summary['kl_div']['std']:.2e} | "
            f"top1_match={summary['top1_match']['mean']:>8.4%}±{summary['top1_match']['std']:.2e} | "
            f"gpu_speedup_est={summary['gpu_speedup_est']['mean']:>6.2f}x"
        )
        results.append(summary)

    return results

if __name__ == "__main__":
    seq_lens = [16, 32, 64, 128, 256, 512, 1024, 2048]
    percentage_list = [0.001, 0.01, 0.05, 0.1, 0.5, 0.8, 0.9, 0.99]
    num_trials = 10
    rel_error_curves = {percentage: {"mean": [], "std": []} for percentage in percentage_list}
    kl_div_curves = {percentage: {"mean": [], "std": []} for percentage in percentage_list}
    top1_match_curves = {percentage: {"mean": [], "std": []} for percentage in percentage_list}
    speedup_curves = {percentage: {"mean": [], "std": []} for percentage in percentage_list}

    print("Note: gpu_speedup_est is a heuristic for a custom sparse CUDA kernel on NVIDIA GPUs.")
    print("      The current implementation still computes dense scores before top-k,")
    print("      so the actual runtime of this script will not reach that speedup directly.")
    print("=" * 72)

    for seq_len in seq_lens:
        k_list = [max(1, int(seq_len * p)) for p in percentage_list]
        results = evaluate_sparse_decoder(
            batch_size=4,
            seq_len=seq_len,
            top_k_list=k_list,
            seed=42,
            num_trials=num_trials,
        )

        for percentage, metrics in zip(percentage_list, results):
            rel_error_curves[percentage]["mean"].append(metrics["rel_err"]["mean"])
            rel_error_curves[percentage]["std"].append(metrics["rel_err"]["std"])
            kl_div_curves[percentage]["mean"].append(metrics["kl_div"]["mean"])
            kl_div_curves[percentage]["std"].append(metrics["kl_div"]["std"])
            top1_match_curves[percentage]["mean"].append(metrics["top1_match"]["mean"])
            top1_match_curves[percentage]["std"].append(metrics["top1_match"]["std"])
            speedup_curves[percentage]["mean"].append(metrics["gpu_speedup_est"]["mean"])
            speedup_curves[percentage]["std"].append(metrics["gpu_speedup_est"]["std"])

    # plot the result
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.reshape(-1)
    for percentage in percentage_list:
        rel_mean = torch.tensor(rel_error_curves[percentage]["mean"], dtype=torch.float64)
        rel_std = torch.tensor(rel_error_curves[percentage]["std"], dtype=torch.float64)
        kl_mean = torch.tensor(kl_div_curves[percentage]["mean"], dtype=torch.float64)
        kl_std = torch.tensor(kl_div_curves[percentage]["std"], dtype=torch.float64)
        speed_mean = torch.tensor(speedup_curves[percentage]["mean"], dtype=torch.float64)
        speed_std = torch.tensor(speedup_curves[percentage]["std"], dtype=torch.float64)
        top1_mean = torch.tensor(top1_match_curves[percentage]["mean"], dtype=torch.float64)
        top1_std = torch.tensor(top1_match_curves[percentage]["std"], dtype=torch.float64)

        axes[0].plot(
            seq_lens,
            rel_mean.tolist(),
            marker="o",
            label=f"keep_ratio={percentage:.0%}",
        )
        axes[0].fill_between(
            seq_lens,
            torch.clamp(rel_mean - rel_std, min=1e-12).tolist(),
            torch.clamp(rel_mean + rel_std, min=1e-12).tolist(),
            alpha=0.15,
        )

        axes[1].plot(
            seq_lens,
            kl_mean.tolist(),
            marker="o",
            label=f"keep_ratio={percentage:.0%}",
        )
        axes[1].fill_between(
            seq_lens,
            torch.clamp(kl_mean - kl_std, min=0.0).tolist(),
            (kl_mean + kl_std).tolist(),
            alpha=0.15,
        )

        axes[2].plot(
            seq_lens,
            speed_mean.tolist(),
            marker="o",
            label=f"keep_ratio={percentage:.0%}",
        )
        axes[2].fill_between(
            seq_lens,
            torch.clamp(speed_mean - speed_std, min=0.0).tolist(),
            (speed_mean + speed_std).tolist(),
            alpha=0.15,
        )

        axes[3].plot(
            seq_lens,
            top1_mean.tolist(),
            marker="o",
            label=f"keep_ratio={percentage:.0%}",
        )
    axes[3].fill_between(
        seq_lens,
        torch.clamp(top1_mean - top1_std, min=0.0, max=1.0).tolist(),
        torch.clamp(top1_mean + top1_std, min=0.0, max=1.0).tolist(),
        alpha=0.15,
    )

    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Sequence Length")
    axes[0].set_ylabel("Relative Error")
    axes[0].set_title("Relative Error vs Sequence Length")
    axes[0].grid(True, which="both", linestyle="--", alpha=0.4)

    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("Sequence Length")
    axes[1].set_ylabel("KL Divergence")
    axes[1].set_title("KL Divergence")
    axes[1].grid(True, which="both", linestyle="--", alpha=0.4)
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    axes[2].set_xscale("log", base=2)
    axes[2].set_xlabel("Sequence Length")
    axes[2].set_ylabel("Estimated GPU Speedup (x)")
    axes[2].set_title("Estimated NVIDIA GPU Speedup")
    axes[2].grid(True, which="both", linestyle="--", alpha=0.4)

    axes[3].set_xscale("log", base=2)
    axes[3].set_xlabel("Sequence Length")
    axes[3].set_ylabel("Top-1 Match Rate")
    axes[3].set_title("Top-1 Match Rate")
    axes[3].set_ylim(0.0, 1.0)
    axes[3].grid(True, which="both", linestyle="--", alpha=0.4)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(3, len(percentage_list)))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    plt.show()
    
