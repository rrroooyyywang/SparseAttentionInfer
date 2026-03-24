"""Visualise BigBird attention mask and attention weights as heatmaps.

Usage (from repo root):
    python -m sparse_attention_bench.playground.bigbird_heatmap
    python -m sparse_attention_bench.playground.bigbird_heatmap --seq_len 128 --top_k 32 --head 0
"""
import argparse
import sys

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Make sure the repo root is on the path when run as a script
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sparse_attentions.patterns.bigbird_pattern import BigBirdPattern, BigBird2Pattern


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_masked_attention(
    q: torch.Tensor,   # [B, H, T, D]
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,  # [H, T, T] bool  (True = keep)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (output [B,H,T,D], attn_weights [B,H,T,T])."""
    scale = q.size(-1) ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale   # [B,H,T,T]

    # mask: True = attend, False = -inf
    attn_mask = mask.unsqueeze(0).float()                    # [1,H,T,T]
    attn_mask = attn_mask.masked_fill(attn_mask == 0, float("-inf"))
    attn_mask = attn_mask.masked_fill(attn_mask == 1, 0.0)

    scores = scores + attn_mask
    attn_weights = torch.softmax(scores, dim=-1)             # [B,H,T,T]
    # Replace NaN rows (all-masked queries) with 0
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
    output = torch.matmul(attn_weights, v)                   # [B,H,T,D]
    return output, attn_weights


def plot_heatmaps(
    mask: torch.Tensor,                 # [H, T_q, T_k] bool
    attn_weights: torch.Tensor,         # [H, T_q, T_k] float  (masked)
    orig_attn_weights: torch.Tensor,    # [H, T_q, T_k] float  (dense causal)
    head: int,
    seq_len: int,
    top_k: int,
    n_heads: int,
    pattern_name: str = "bigbird",
    save_path: str | None = None,
) -> None:
    mask_np = mask[head].cpu().float().numpy()              # [T_q, T_k]
    attn_np = attn_weights[head].cpu().float().numpy()      # [T_q, T_k]
    orig_np = orig_attn_weights[head].cpu().float().numpy() # [T_q, T_k]

    fig = plt.figure(figsize=(20, 6))
    fig.suptitle(
        f"{pattern_name}  |  seq_len={seq_len}  top_k={top_k}"
        f"  n_heads={n_heads}  head={head}",
        fontsize=13,
    )

    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

    # ── Left: Mask ────────────────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    im0 = ax0.imshow(mask_np, origin="upper", aspect="auto", cmap="Blues",
                     vmin=0, vmax=1, interpolation="nearest")
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
    ax0.set_title("Attention Mask\n(1 = attend, 0 = masked)", fontsize=11)
    ax0.set_xlabel("Key position")
    ax0.set_ylabel("Query position")

    keep_pct = mask_np.mean() * 100
    ax0.text(
        0.02, 0.97, f"density: {keep_pct:.1f}%",
        transform=ax0.transAxes, fontsize=9,
        va="top", ha="left", color="darkblue",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
    )

    # ── Middle: Original (dense causal) attention weights ─────────────────────
    ax1 = fig.add_subplot(gs[1])
    im1 = ax1.imshow(orig_np, origin="upper", aspect="auto", cmap="hot_r",
                     interpolation="nearest")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    ax1.set_title("Original Attention Weights\n(dense causal softmax)", fontsize=11)
    ax1.set_xlabel("Key position")
    ax1.set_ylabel("Query position")

    # ── Right: Masked attention weights ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[2])
    im2 = ax2.imshow(attn_np, origin="upper", aspect="auto", cmap="hot_r",
                     interpolation="nearest")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    ax2.set_title(f"{pattern_name} Attention Weights\n(softmax over masked scores)", fontsize=11)
    ax2.set_xlabel("Key position")
    ax2.set_ylabel("Query position")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BigBird heatmap visualisation")
    parser.add_argument("--pattern", type=str, default="bigbird",
                        choices=["bigbird", "bigbird2"],
                        help="Pattern variant: bigbird (token-0 global) or "
                             "bigbird2 (token-0 + last-token global) (default: bigbird)")
    parser.add_argument("--seq_len", type=int, default=64,
                        help="Sequence length (default: 64)")
    parser.add_argument("--top_k",  type=int, default=16,
                        help="BigBird top_k budget (default: 16)")
    parser.add_argument("--n_heads", type=int, default=4,
                        help="Number of attention heads (default: 4)")
    parser.add_argument("--head",   type=int, default=0,
                        help="Which head to plot (default: 0)")
    parser.add_argument("--d_head", type=int, default=32,
                        help="Head dimension (default: 32)")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--save",   type=str, default=None,
                        help="Optional path to save the figure (e.g. mask.png)")
    args = parser.parse_args()

    assert 0 <= args.head < args.n_heads, \
        f"--head must be in [0, n_heads-1={args.n_heads-1}]"

    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    # Build random Q, K, V  [B=1, H, T, D]
    shape = (1, args.n_heads, args.seq_len, args.d_head)
    q = torch.randn(*shape, device=device)
    k = torch.randn(*shape, device=device)
    v = torch.randn(*shape, device=device)

    # Generate mask  →  [H, T_q, T_k]
    _PATTERN_CLS = {"bigbird": BigBirdPattern, "bigbird2": BigBird2Pattern}
    pattern_cls = _PATTERN_CLS[args.pattern]
    pattern = pattern_cls(top_k=args.top_k, n_heads=args.n_heads)
    meta = pattern.build(q, k, causal=True)
    mask = meta.mask  # [H, T, T]

    # Compute sparse (masked) attention weights
    _, attn_weights = compute_masked_attention(q, k, v, mask)
    attn_weights = attn_weights.squeeze(0)  # [H, T, T]

    # Compute original dense causal attention weights
    causal_mask = torch.ones(args.seq_len, args.seq_len, dtype=torch.bool, device=device).tril()
    causal_mask = causal_mask.unsqueeze(0).expand(args.n_heads, -1, -1)  # [H, T, T]
    _, orig_attn_weights = compute_masked_attention(q, k, v, causal_mask)
    orig_attn_weights = orig_attn_weights.squeeze(0)  # [H, T, T]

    print(f"Pattern         : {args.pattern}")
    print(f"Sequence length : {args.seq_len}")
    print(f"top_k budget    : {args.top_k}")
    print(f"n_heads         : {args.n_heads}")
    print(f"Mask density    : {mask.float().mean().item()*100:.1f}%")
    print(f"Plotting head   : {args.head}")

    plot_heatmaps(
        mask=mask,
        attn_weights=attn_weights,
        orig_attn_weights=orig_attn_weights,
        head=args.head,
        seq_len=args.seq_len,
        top_k=args.top_k,
        n_heads=args.n_heads,
        pattern_name=args.pattern,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
