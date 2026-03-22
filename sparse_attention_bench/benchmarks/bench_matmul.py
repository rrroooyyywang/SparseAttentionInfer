"""
Sparse matrix multiply benchmark — accuracy and theoretical speedup.

Migrated from profiling/sparse_prof.py.

Usage:
    python -m sparse_attention_bench.benchmarks.bench_matmul
    python -m sparse_attention_bench.benchmarks.bench_matmul --mode single --m 512 --n 512 --runs 100
    python -m sparse_attention_bench.benchmarks.bench_matmul --mode double --m 1000 --n 1000
"""
import argparse

import torch

from sparse_attention_bench.metrics.sparse_matmul import (
    estimate_dense_macs,
    estimate_double_sparse_macs,
    estimate_single_sparse_macs,
    sparse_matrix_multiply,
    single_sparse_matrix_multiply,
)


def run_single_sparse_profile(m: int, n: int, top_percent: list[float], runs: int) -> None:
    """Benchmark single-sparse GEMM (only A is sparse) over a range of sparsity levels."""
    dense_macs = estimate_dense_macs(m, n)
    print(f"Dense MACs: {dense_macs:e}")

    for percent in top_percent:
        k = int(m * n * percent)
        acc_list = []
        for _ in range(runs):
            A = torch.rand(m, n)
            B = torch.rand(m, n)
            result = single_sparse_matrix_multiply(A, B, k)
            golden = torch.mm(A, B.t())
            acc = 1 - (torch.norm(result - golden) / torch.norm(golden))
            acc_list.append(acc.item())

        avg_acc = sum(acc_list) / len(acc_list)
        sparse_macs, density_a = estimate_single_sparse_macs(m, n, k)
        print(f"Top {percent * 100:.1f}% elements:")
        print(f"  Average Accuracy         = {avg_acc:.6f}")
        print(f"  Density A                = {density_a:.6f}")
        print(f"  Dense MACs               = {dense_macs:e}")
        print(f"  Sparse MACs (estimated)  = {sparse_macs:e}")
        print(f"  Theoretical speedup      = {dense_macs / sparse_macs:.2f}x")


def run_double_sparse_profile(m: int, n: int, top_percent: list[float], runs: int) -> None:
    """Benchmark double-sparse GEMM (both A and B are sparse) over a range of sparsity levels."""
    dense_macs = estimate_dense_macs(m, n)
    print(f"Dense MACs: {dense_macs:e}")

    for percent in top_percent:
        k = int(m * n * percent)
        acc_list = []
        for _ in range(runs):
            A = torch.rand(m, n)
            B = torch.rand(m, n)
            result = sparse_matrix_multiply(A, B, k)
            golden = torch.mm(A, B.t())
            acc = 1 - (torch.norm(result - golden) / torch.norm(golden))
            acc_list.append(acc.item())

        avg_acc = sum(acc_list) / len(acc_list)
        sparse_macs_opt, sparse_macs_overlap, density_a, density_b = estimate_double_sparse_macs(m, n, k)
        print(f"Top {percent * 100:.1f}% elements:")
        print(f"  Average Accuracy                = {avg_acc:.6f}")
        print(f"  Density A / B                   = {density_a:.6f} / {density_b:.6f}")
        print(f"  Dense MACs                      = {dense_macs:e}")
        print(f"  Sparse MACs (optimistic)        = {sparse_macs_opt:e}")
        print(f"  Sparse MACs (overlap estimate)  = {sparse_macs_overlap:e}")
        print(f"  Theoretical speedup (optimistic)= {dense_macs / sparse_macs_opt:.2f}x")
        print(f"  Theoretical speedup (overlap)   = {dense_macs / sparse_macs_overlap:.2f}x")


def parse_args():
    parser = argparse.ArgumentParser(description="Sparse GEMM accuracy and MACs benchmark.")
    parser.add_argument("--mode", type=str, default="single", choices=["single", "double"],
                        help="'single': only A is sparse; 'double': both A and B are sparse.")
    parser.add_argument("--m", type=int, default=512, help="Matrix rows.")
    parser.add_argument("--n", type=int, default=512, help="Matrix columns.")
    parser.add_argument("--runs", type=int, default=100, help="Number of random trials per sparsity level.")
    parser.add_argument(
        "--top-percent", type=float, nargs="+",
        default=[0.1, 0.5, 0.8, 0.9, 0.95, 0.99],
        help="Keep-ratio values to sweep (fractions, e.g. 0.1 = top 10%%).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "single":
        run_single_sparse_profile(args.m, args.n, args.top_percent, args.runs)
    else:
        run_double_sparse_profile(args.m, args.n, args.top_percent, args.runs)


if __name__ == "__main__":
    main()
