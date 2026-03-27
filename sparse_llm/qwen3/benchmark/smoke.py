import argparse

from sparse_llm.common.benchmark.generation import DEFAULT_DECODE_STEPS, DEFAULT_TEMPERATURE
from sparse_llm.common.benchmark.smoke import run_smoke_test as _run_smoke_test
from sparse_llm.qwen3.adapter import get_qwen3_adapter


QWEN3_ADAPTER = get_qwen3_adapter()


def run_smoke_test(args: argparse.Namespace) -> None:
    _run_smoke_test(
        QWEN3_ADAPTER,
        args,
        decode_steps=DEFAULT_DECODE_STEPS,
        temperature=DEFAULT_TEMPERATURE,
    )
