## Useful shell commands:

```
TRITON_PRINT_AUTOTUNING=1\
  python decoder_sweep_runner.py \
    --config ../experiment_configs/sweep_seq_len_base_vs_bigbird.yaml \
    --tag bigbird_vs_bigbird2 \
    --plot \
    --quiet
```