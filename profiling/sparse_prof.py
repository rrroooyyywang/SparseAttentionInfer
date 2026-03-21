import torch

def rand_two_matrix(m, n):
    A = torch.rand(m, n)
    B = torch.rand(m, n)
    return A, B

def global_top_k_sparse_matrix(M, k):
    flat = M.flatten()
    topk_values, topk_indices = torch.topk(flat, k)

    sparse_flat = torch.zeros_like(flat)
    sparse_flat[topk_indices] = topk_values

    return sparse_flat.view_as(M)

def sparse_matrix_multiply(A, B, k):
    sparse_A = global_top_k_sparse_matrix(A, k)
    sparse_B = global_top_k_sparse_matrix(B, k)
    result = torch.mm(sparse_A, sparse_B.t())
    return result

def single_sparse_matrix_multiply(A, B, k):
    sparse_A = global_top_k_sparse_matrix(A, k)
    result = torch.mm(sparse_A, B.t())
    return result

def estimate_dense_macs(m, n):
    # A[m, n] @ B[n, m] -> output[m, m]
    return m * m * n

def estimate_single_sparse_macs(m, n, k):
    # only A is sparse
    density_a = k / (m * n)
    dense_macs = estimate_dense_macs(m, n)
    sparse_macs = dense_macs * density_a
    return sparse_macs, density_a

def estimate_double_sparse_macs(m, n, k):
    # both A and B are sparse
    density_a = k / (m * n)
    density_b = k / (m * n)
    dense_macs = estimate_dense_macs(m, n)

    # optimistic estimate: cost driven by A nonzeros only
    sparse_macs_optimistic = dense_macs * density_a

    # overlap estimate: only overlapping nonzeros contribute
    sparse_macs_overlap = dense_macs * density_a * density_b

    return sparse_macs_optimistic, sparse_macs_overlap, density_a, density_b

def profile(runs=100):
    m, n = 1000, 1000
    top_percent = [0.1, 0.5, 0.8, 0.9, 0.95, 0.99]
    k_list = [int(m * n * p) for p in top_percent]

    dense_macs = estimate_dense_macs(m, n)
    all_accuracy = []

    print(f"Dense MACs: {dense_macs:e}")

    for percent, k in zip(top_percent, k_list):
        acc_list = []

        for _ in range(runs):
            A, B = rand_two_matrix(m, n)
            result = sparse_matrix_multiply(A, B, k)
            golden = torch.mm(A, B.t())
            acc = 1 - (torch.norm(result - golden) / torch.norm(golden))
            acc_list.append(acc.item())

        avg_acc = sum(acc_list) / len(acc_list)
        all_accuracy.append(avg_acc)

        sparse_macs_opt, sparse_macs_overlap, density_a, density_b = estimate_double_sparse_macs(m, n, k)

        print(f"Top {percent*100:.1f}% elements:")
        print(f"  Average Accuracy                = {avg_acc:.6f}")
        print(f"  Density A / B                   = {density_a:.6f} / {density_b:.6f}")
        print(f"  Dense MACs                      = {dense_macs:e}")
        print(f"  Sparse MACs (optimistic)        = {sparse_macs_opt:e}")
        print(f"  Sparse MACs (overlap estimate)  = {sparse_macs_overlap:e}")
        print(f"  Theoretical speedup (optimistic)= {dense_macs / sparse_macs_opt:.2f}x")
        print(f"  Theoretical speedup (overlap)   = {dense_macs / sparse_macs_overlap:.2f}x")

    return all_accuracy, k_list

def single_profile(runs=100):
    m, n = 512, 512
    top_percent = [0.1, 0.5, 0.8, 0.9, 0.95, 0.99]
    k_list = [int(m * n * p) for p in top_percent]

    dense_macs = estimate_dense_macs(m, n)
    all_accuracy = []

    print(f"Dense MACs: {dense_macs:e}")

    for percent, k in zip(top_percent, k_list):
        acc_list = []

        for _ in range(runs):
            A, B = rand_two_matrix(m, n)
            result = single_sparse_matrix_multiply(A, B, k)
            golden = torch.mm(A, B.t())
            acc = 1 - (torch.norm(result - golden) / torch.norm(golden))
            acc_list.append(acc.item())

        avg_acc = sum(acc_list) / len(acc_list)
        all_accuracy.append(avg_acc)

        sparse_macs, density_a = estimate_single_sparse_macs(m, n, k)

        print(f"Top {percent*100:.1f}% elements:")
        print(f"  Average Accuracy         = {avg_acc:.6f}")
        print(f"  Density A                = {density_a:.6f}")
        print(f"  Dense MACs               = {dense_macs:e}")
        print(f"  Sparse MACs (estimated)  = {sparse_macs:e}")
        print(f"  Theoretical speedup      = {dense_macs / sparse_macs:.2f}x")

    return all_accuracy, k_list

if __name__ == "__main__":
    # accuracy, k = profile(100)
    single_accuracy, single_k = single_profile(100)