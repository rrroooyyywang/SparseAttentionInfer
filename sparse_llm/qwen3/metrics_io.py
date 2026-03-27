import json
from pathlib import Path
from typing import Any


MODEL_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = MODEL_ROOT / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PLOTS_DIR = OUTPUTS_DIR / "plots"
DEFAULT_METRICS_JSON = str(METRICS_DIR / "qwen3_prefill_decode_metrics.json")


def default_plot_path(metrics_json_path: str) -> str:
    metrics_path = Path(metrics_json_path)
    return str(PLOTS_DIR / f"{metrics_path.stem}_plot.png")


def load_metrics_json(metrics_json_path: str) -> Any:
    with open(metrics_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_metrics_json(payload: Any, output_json_path: str) -> None:
    output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
