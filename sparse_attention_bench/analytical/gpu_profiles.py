from pathlib import Path

from sparse_attention_bench.paths import PROFILING_DIR
from sparse_attention_bench.analytical.config import NvidiaGpuHeuristic, VALID_EXECUTION_PHASES

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

DEFAULT_GPU_PROFILE_PATH = PROFILING_DIR / "gpu_profiles.toml"


def load_gpu_profile_catalog(profile_path: str | Path = DEFAULT_GPU_PROFILE_PATH) -> tuple[str, dict]:
    profile_path = Path(profile_path)
    if not profile_path.exists():
        raise FileNotFoundError(f"GPU profile TOML not found: {profile_path}")
    with profile_path.open("rb") as handle:
        payload = tomllib.load(handle)
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError(f"No [profiles.*] entries found in {profile_path}")
    default_profile = payload.get("default_profile", next(iter(profiles)))
    if default_profile not in profiles:
        raise ValueError(f"default_profile={default_profile!r} not found in {profile_path}")
    return default_profile, profiles


def load_gpu_heuristic(
    profile_name: str | None = None,
    profile_path: str | Path = DEFAULT_GPU_PROFILE_PATH,
) -> NvidiaGpuHeuristic:
    default_profile, profiles = load_gpu_profile_catalog(profile_path)
    selected = profile_name or default_profile
    if selected not in profiles:
        available = ", ".join(sorted(profiles))
        raise ValueError(f"Unknown GPU profile {selected!r}. Available: {available}")
    data = dict(profiles[selected])
    description = data.pop("description", "")
    return NvidiaGpuHeuristic(
        profile_name=selected,
        description=description or f"GPU profile {selected}",
        **data,
    )


def validate_execution_phase(phase: str) -> str:
    if phase not in VALID_EXECUTION_PHASES:
        valid = ", ".join(VALID_EXECUTION_PHASES)
        raise ValueError(f"Unsupported phase: {phase!r}. Expected one of: {valid}")
    return phase
