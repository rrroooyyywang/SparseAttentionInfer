"""
Central path definitions for the sparse_attention_bench package.

All paths are derived from this file's location, which is stable
regardless of where scripts are run from.
"""
from pathlib import Path

# Directory containing this file: <repo>/sparse_attention_bench/
PACKAGE_DIR: Path = Path(__file__).resolve().parent

# Repository root: <repo>/
PROJECT_ROOT: Path = PACKAGE_DIR.parent

# Default output directory for all benchmarks
OUTPUTS_DIR: Path = PACKAGE_DIR / "outputs"

# Location of GPU hardware profile definitions
PROFILING_DIR: Path = PROJECT_ROOT / "profiling"
