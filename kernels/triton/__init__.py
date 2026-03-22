"""
Triton sparse attention kernels.

Each kernel file should expose one callable with the signature:

    def triton_<name>_attn(
        q: torch.Tensor,   # [B, H, T_q, D]  fp16 / bf16
        k: torch.Tensor,   # [B, H, T_k, D]
        v: torch.Tensor,   # [B, H, T_k, D]
        **kernel_args,     # e.g. top_k, window_size, block_size, causal
    ) -> torch.Tensor:     # [B, H, T_q, D]

Then wrap it as an AttentionBackend in sparse_attention_bench/attention/
so it integrates with the benchmark runner and YAML sweep configs.

Useful references
-----------------
- Triton flash-attention tutorial:
  https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
- PatternMetadata fields available to the backend:
  kind, mask [H,T,T], topk (int), keep_ratio (float)
"""
