import torch
import torch.nn.functional as F


def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Upper-triangular causal mask: True = masked (forbidden). Shape [1, 1, T, T]."""
    mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
        diagonal=1,
    )
    return mask.unsqueeze(0).unsqueeze(0)


def relative_error(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    return (torch.norm(x - y) / (torch.norm(y) + eps)).item()


def cosine_sim(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    return F.cosine_similarity(x.reshape(-1), y.reshape(-1), dim=0, eps=eps).item()


def top1_match_rate(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
    """Fraction of token positions where argmax agrees. logits: [B, T, V]."""
    return (logits_a.argmax(dim=-1) == logits_b.argmax(dim=-1)).float().mean().item()


def mean_kl_divergence(reference_logits: torch.Tensor, approx_logits: torch.Tensor) -> float:
    """Average token-wise KL(reference || approx)."""
    ref_log = F.log_softmax(reference_logits.double(), dim=-1)
    app_log = F.log_softmax(approx_logits.double(), dim=-1)
    token_kl = (ref_log.exp() * (ref_log - app_log)).sum(dim=-1)
    return max(token_kl.mean().item(), 0.0)


def summarize_scalar(values) -> dict:
    t = torch.tensor(values, dtype=torch.float64)
    return {"mean": t.mean().item(), "std": t.std(unbiased=False).item()}
