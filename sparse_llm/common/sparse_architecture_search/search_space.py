from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class CallableSearchSpace:
    name: str
    sampler: Callable[[random.Random], dict[str, Any]]
    optuna_sampler: Optional[Callable[[Any], dict[str, Any]]] = None
    spec: dict[str, Any] = field(default_factory=dict)

    def sample(self, rng: random.Random) -> dict[str, Any]:
        return self.sampler(rng)

    def sample_optuna(self, trial: Any) -> dict[str, Any]:
        if self.optuna_sampler is None:
            raise TypeError(
                f"Search space {self.name!r} does not provide an Optuna sampler."
            )
        return self.optuna_sampler(trial)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            **self.spec,
        }
