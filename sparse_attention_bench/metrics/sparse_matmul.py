"""
Sparse matrix multiplication utilities and MACs estimation.

Migrated from profiling/sparse_prof.py.
"""
import torch


def global_top_k_sparse_matrix(M: torch.Tensor, k: int) -> torch.Tensor:
    """Zero out all but the top-k absolute-value elements of M."""
    flat = M.flatten()
    topk_values, topk_indices = torch.topk(flat.abs(), k)
    sparse_flat = torch.zeros_like(flat)
    sparse_flat[topk_indices] = flat[topk_indices]
    return sparse_flat.view_as(M)


def sparse_matrix_multiply(A: torch.Tensor, B: torch.Tensor, k: int) -> torch.Tensor:
    """Multiply two globally top-k sparse versions of A and B (A @ B^T)."""
    sparse_A = global_top_k_sparse_matrix(A, k)
    sparse_B = global_top_k_sparse_matrix(B, k)
    return torch.mm(sparse_A, sparse_B.t())


def single_sparse_matrix_multiply(A: torch.Tensor, B: torch.Tensor, k: int) -> torch.Tensor:
    """Multiply one sparse A against a dense B (A @ B^T), only A is sparsified."""
    sparse_A = global_top_k_sparse_matrix(A, k)
    return torch.mm(sparse_A, B.t())


def estimate_dense_macs(m: int, n: int) -> int:
    """MACs for A[m, n] @ B[n, m] -> output[m, m]."""
    return m * m * n


def estimate_single_sparse_macs(m: int, n: int, k: int) -> tuple[float, float]:
    """
    Estimate MACs when only A is globally top-k sparse.

    Returns:
        (sparse_macs, density_a)
    """
    density_a = k / (m * n)
    dense_macs = estimate_dense_macs(m, n)
    sparse_macs = dense_macs * density_a
    return sparse_macs, density_a


def estimate_double_sparse_macs(m: int, n: int, k: int) -> tuple[float, float, float, float]:
    """
    Estimate MACs when both A and B are globally top-k sparse.

    Returns:
        (sparse_macs_optimistic, sparse_macs_overlap, density_a, density_b)

        - optimistic: cost driven by A nonzeros only (upper-bound speedup).
        - overlap: only overlapping nonzeros contribute (lower-bound speedup).
    """
    density_a = k / (m * n)
    density_b = k / (m * n)
    dense_macs = estimate_dense_macs(m, n)
    sparse_macs_optimistic = dense_macs * density_a
    sparse_macs_overlap = dense_macs * density_a * density_b
    return sparse_macs_optimistic, sparse_macs_overlap, density_a, density_b
