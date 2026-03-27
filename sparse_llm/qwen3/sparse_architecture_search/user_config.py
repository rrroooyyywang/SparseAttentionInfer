from sparse_llm.qwen3.sparse_architecture_search.config import GQSparseATuningConfig


class UserGQSparseATuningConfig(GQSparseATuningConfig):
    def __init__(self):
        super().__init__(
            model_name_or_path="Qwen/Qwen3-4B-Instruct-2507",
            tokenizer_name_or_path="Qwen/Qwen3-4B-Instruct-2507",
            device="cuda",
            dtype="bfloat16",

            strategy="bayesian",
            objective="weighted_scalar",
            objective_speed_weight=1.0,
            objective_quality_weight=1.0,
            objective_quality_penalty=20.0,
            objective_max_ppl_ratio=1.1,
            bayes_startup_trials=8,
            bayes_multivariate=True,
            bayes_group=True,
            num_samples=20,
            sampling_grid=(0.0, 0.1, 0.2, 0.4, 0.6, 0.8),
            tail_sampling_grid=None,

            prefix_dense_layers=0,
            layer_share_span=6,
            dense_repeats=1,
            sparse_repeats=1,
            append_results=True,
            plot_only_json=None,
            trial_label="default_qwen3_bayesian_search",
            ppl_batch_size=1,
            ppl_max_length=1024,
            prebuild_patterns=True,
            fast_benchmark=True,
            results_json="sparse_llm/qwen3/outputs/metrics/qwen3_bayesian_search_default.json",
        )


"""
第一次使用贝叶斯搜索前需要先让环境装好 optuna:
uv sync

然后运行:
uv run python -m sparse_llm.qwen3.sparse_architecture_search.search

只重画结果图:
uv run python -m sparse_llm.qwen3.sparse_architecture_search.search \
  --plot-only-json sparse_llm/qwen3/outputs/metrics/qwen3_bayesian_search_default.json
"""
