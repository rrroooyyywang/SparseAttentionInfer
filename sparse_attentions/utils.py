"""Small shared utilities for sparse attention components."""

import torch


def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Upper-triangular causal mask: True = masked (forbidden). Shape [1, 1, T, T]."""
    mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
        diagonal=1,
    )
    return mask.unsqueeze(0).unsqueeze(0)
