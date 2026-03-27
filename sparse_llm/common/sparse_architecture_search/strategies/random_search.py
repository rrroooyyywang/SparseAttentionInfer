from __future__ import annotations

import random
from typing import Any, Optional


class RandomSearchStrategy:
    name = "random_search"

    def __init__(
        self,
        *,
        seed: Optional[int] = None,
        max_trials: Optional[int] = None,
    ) -> None:
        self._seed = seed
        self._max_trials = max_trials
        self._rng: Optional[random.Random] = None
        self._search_space = None
        self._completed_trials = 0

    def register_strategy_args(self, parser) -> None:
        return None

    def initialize(
        self,
        args,
        context,
        search_space,
        history: list[dict[str, Any]],
        state: Optional[dict[str, Any]] = None,
    ) -> None:
        del context
        self._search_space = search_space
        self._seed = args.seed if self._seed is None else self._seed
        self._max_trials = args.num_samples if self._max_trials is None else self._max_trials
        self._completed_trials = sum(
            1
            for trial in history
            if trial.get("status") in {"ok", "error"}
        )
        if state is not None and state.get("completed_trials") is not None:
            self._completed_trials = max(
                self._completed_trials,
                int(state["completed_trials"]),
            )
        self._rng = random.Random(self._seed)

    def propose(self) -> Optional[dict[str, Any]]:
        if self._rng is None or self._search_space is None or self.should_stop():
            return None
        return self._search_space.sample(self._rng)

    def observe(self, trial_result: dict[str, Any]) -> None:
        del trial_result
        self._completed_trials += 1

    def should_stop(self) -> bool:
        if self._max_trials is None:
            return False
        return self._completed_trials >= self._max_trials

    def export_state(self) -> dict[str, Any]:
        return {
            "seed": self._seed,
            "completed_trials": self._completed_trials,
        }
