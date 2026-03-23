import json
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from sparse_attention_bench.analytical.config import NvidiaGpuHeuristic


def iso_utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def relative_path_str(path: str | Path, base_dir: str | Path | None = None) -> str:
    base_dir = Path.cwd() if base_dir is None else Path(base_dir)
    path_obj = Path(path)
    if not path_obj.is_absolute():
        return path_obj.as_posix()
    return Path(os.path.relpath(path_obj, start=base_dir)).as_posix()


def normalize_metric_value(value, base_dir: str | Path | None = None):
    if isinstance(value, dict):
        return {k: normalize_metric_value(v, base_dir=base_dir) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_metric_value(i, base_dir=base_dir) for i in value]
    if isinstance(value, Path):
        return relative_path_str(value, base_dir=base_dir)
    return value


def build_accuracy_record(seq_len: int, requested_keep_ratio: float, metrics: dict) -> dict:
    record = {"seq_len": seq_len, "requested_keep_ratio": requested_keep_ratio}
    record.update(normalize_metric_value(metrics))
    return record


def build_speedup_record(
    seq_len: int, requested_keep_ratio: float, top_k: int, metrics: dict
) -> dict:
    record = {"seq_len": seq_len, "top_k": top_k, "requested_keep_ratio": requested_keep_ratio}
    record.update(normalize_metric_value(metrics))
    return record


def build_proxy_profile_payload(
    gpu: NvidiaGpuHeuristic,
    gpu_profile_path: str | Path,
    batch_size: int,
    seed: int,
    num_trials: int,
    seq_lens,
    percentage_list,
    sparse_modes,
    phases,
    accuracy_records,
    speedup_records,
    artifacts: dict,
) -> dict:
    path_base = Path.cwd()
    return {
        "schema_version": 1,
        "kind": "sparse_decoder_proxy_profile",
        "generated_at_utc": iso_utc_now(),
        "generator": "sparse_attention_bench/benchmarks/bench_proxy.py",
        "gpu_profile": {
            "name": gpu.profile_name,
            "description": gpu.description,
            "source_toml": relative_path_str(gpu_profile_path, base_dir=path_base),
            "parameters": normalize_metric_value(asdict(gpu), base_dir=path_base),
        },
        "experiment": {
            "batch_size": batch_size,
            "seed": seed,
            "num_trials": num_trials,
            "seq_lens": list(seq_lens),
            "requested_keep_ratios": list(percentage_list),
            "sparse_modes": list(sparse_modes),
            "phases": list(phases),
            "accuracy_phase": "prefill",
            "paths_relative_to": ".",
        },
        "artifacts": normalize_metric_value(artifacts, base_dir=path_base),
        "accuracy_records": normalize_metric_value(accuracy_records, base_dir=path_base),
        "speedup_records": normalize_metric_value(speedup_records, base_dir=path_base),
        "notes": [
            "These results are software proxy estimates, not real sparse CUDA kernel benchmarks.",
            "The current PyTorch implementation still forms dense attention scores before sparsifying.",
            "Use this JSON as a baseline for later real-kernel benchmarking and calibration.",
        ],
    }


def save_proxy_profile_json(output_path: str | Path, payload: dict) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


class TerminalProgressBar:
    def __init__(self, total: int, unit: str = "trials", enabled: bool = True, width: int = 28):
        self.total = max(1, total)
        self.unit = unit
        self.enabled = enabled
        self.width = width
        self.completed = 0
        self.description = ""
        self.start_time = time.perf_counter()
        self._last_line_length = 0

    def update(self, step: int = 1, description: str | None = None) -> None:
        self.completed = min(self.total, self.completed + step)
        if description is not None:
            self.description = description
        self.render()

    def render(self) -> None:
        if not self.enabled:
            return
        ratio = self.completed / self.total
        filled = min(self.width, int(self.width * ratio))
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = max(time.perf_counter() - self.start_time, 1e-9)
        rate = self.completed / elapsed
        line = (
            f"\r[{bar}] {self.completed:>4d}/{self.total:<4d} {self.unit} "
            f"| {rate:>6.2f} {self.unit}/s"
        )
        if self.description:
            line += f" | {self.description}"
        print(line.ljust(self._last_line_length), end="", flush=True)
        self._last_line_length = len(line)

    def write(self, message: str) -> None:
        if not self.enabled:
            print(message)
            return
        clear_line = "\r" + (" " * self._last_line_length) + "\r"
        print(clear_line + message)
        self.render()

    def close(self) -> None:
        if not self.enabled:
            return
        self.render()
        print()
