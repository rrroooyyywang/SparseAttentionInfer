from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


MODEL_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = MODEL_ROOT / "outputs" / "metrics"


@dataclass
class GQSparseATuningConfig:
    model_name_or_path: Optional[str] = None
    tokenizer_name_or_path: Optional[str] = None
    allow_download: bool = True
    cache_dir: Optional[str] = None
    revision: Optional[str] = None
    hf_token: Optional[str] = None
    trust_remote_code: bool = False
    device: Optional[str] = None
    dtype: str = "bfloat16"
    backend: str = "triton_universal"
    window_size: int = 32
    block_size: int = 16
    top_k: int = 128
    seed: int = 0
    strategy: str = "random"
    objective: Optional[str] = None
    objective_speed_weight: float = 1.0
    objective_quality_weight: float = 1.0
    objective_quality_penalty: float = 20.0
    objective_max_ppl_ratio: float = 1.03
    bayes_startup_trials: int = 8
    bayes_multivariate: bool = True
    bayes_group: bool = True
    num_samples: int = 20
    sampling_grid: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8)
    tail_sampling_grid: Optional[tuple[float, ...]] = None
    prefix_dense_layers: int = 0
    layer_share_span: int = 0
    dense_repeats: int = 1
    sparse_repeats: int = 1
    results_json: str = str(METRICS_DIR / "gq_sparse_a_tuning_results.json")
    plot_path: Optional[str] = None
    ppl_text: Optional[str] = None
    ppl_text_file: Optional[str] = None
    ppl_dataset_name: str = "wikitext"
    ppl_dataset_config: str = "wikitext-2-raw-v1"
    ppl_dataset_split: str = "test"
    ppl_dataset_text_key: str = "text"
    ppl_max_length: int = 1024
    ppl_stride: Optional[int] = None
    ppl_max_samples: Optional[int] = None
    ppl_batch_size: int = 1
    prebuild_patterns: bool = True
    fast_benchmark: bool = False
    append_results: bool = True
    plot_only_json: Optional[str] = None
    trial_label: Optional[str] = None
    fixed_layer_head_sparsities: Optional[list[list[float]]] = field(default=None)
