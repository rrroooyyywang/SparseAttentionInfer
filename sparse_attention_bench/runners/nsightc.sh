PYTHON_BIN="python"

CUDNN_LIB=$("$PYTHON_BIN" -c "import nvidia.cudnn; print(next(iter(nvidia.cudnn.__path__)) + '/lib')")

export LD_LIBRARY_PATH="$CUDNN_LIB:$LD_LIBRARY_PATH"

echo "[INFO] PYTHON_BIN=$PYTHON_BIN"
echo "[INFO] CUDNN_LIB=$CUDNN_LIB"
echo "[INFO] LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

# Profile your benchmark script, capturing the Triton kernel
TRITON_PRINT_AUTOTUNING=1 ncu \
  -f \
  --metrics sm__utilization.avg.pct_of_peak_sustained_active,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,dram__bytes.sum \
  --target-processes all \
  --launch-count 20 \
  -o profile_report \
  $PYTHON_BIN decoder_sweep_runner.py \
    --config ../experiment_configs/sweep_seq_len_base_vs_bigbird.yaml \
    --tag bigbird_vs_bigbird2 \
    --plot \
    --quiet
