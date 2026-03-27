# Qwen3 Layer-Block Sparsity Experiment

This experiment sweeps contiguous layer blocks and assigns the same sparsity to
all KV groups inside the active block while keeping every other layer at `0.0`
sparsity.

For a 36-layer Qwen3 model with `--layer-span 6 --target-sparsity 0.2`, the
script will run:

- layers `0-5` at `0.2`, all other layers at `0.0`
- layers `6-11` at `0.2`, all other layers at `0.0`
- layers `12-17` at `0.2`, all other layers at `0.0`
- layers `18-23` at `0.2`, all other layers at `0.0`
- layers `24-29` at `0.2`, all other layers at `0.0`
- layers `30-35` at `0.2`, all other layers at `0.0`

It writes:

- one dense generation baseline JSON
- one dense perplexity baseline JSON
- one raw JSON per experiment block
- one summary JSON
- one summary CSV

Example:

```bash
uv run python -m sparse_llm.qwen3.experiment.layer_block_sparsity \
  --model-name-or-path Qwen/Qwen3-4B-Instruct-2507 \
  --tokenizer-name-or-path Qwen/Qwen3-4B-Instruct-2507 \
  --device cuda \
  --dtype bfloat16 \
  --backend triton_universal \
  --layer-span 6 \
  --target-sparsity 0.2 \
  --seq-len 256 \
  --benchmark-decode-steps 256 \
  --ppl-max-length 1024 \
  --ppl-batch-size 1 \
  --output-json sparse_llm/qwen3/experiment/outputs/metrics/qwen3_layer_block_sparsity.json
```

Important:

- The script fixes the sparse pattern to BigBird and applies per-layer
  `layer_group_sparsities`.
- `speedup` is summarized from generation decode throughput and perplexity
  evaluation throughput.
- `ppl` degradation is summarized with `ppl_ratio`, `baseline_over_ppl`, and
  `ppl_delta`.

Plot the trend after the run:

```bash
uv run python -m sparse_llm.qwen3.experiment.plot_layer_block_sparsity \
  --summary-json sparse_llm/qwen3/experiment/outputs/metrics/qwen3_layer_block_sparsity.json
```
