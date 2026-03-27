from __future__ import annotations

from typing import Any, Optional


class BayesianSearchStrategy:
    name = "bayesian_search"

    def __init__(
        self,
        *,
        seed: Optional[int] = None,
        max_trials: Optional[int] = None,
        startup_trials: int = 8,
        multivariate: bool = True,
        group: bool = True,
    ) -> None:
        self._seed = seed
        self._max_trials = max_trials
        self._startup_trials = int(startup_trials)
        self._multivariate = bool(multivariate)
        self._group = bool(group)
        self._optuna = None
        self._study = None
        self._search_space = None
        self._active_trial = None
        self._completed_trials = 0
        self._serialized_trials: list[dict[str, Any]] = []

    def register_strategy_args(self, parser) -> None:
        del parser
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
        try:
            import optuna
            from optuna.distributions import json_to_distribution
            from optuna.trial import TrialState
        except ImportError as exc:
            raise ImportError(
                "BayesianSearchStrategy requires the `optuna` package. "
                "Install dependencies with `uv sync` or run without `--no-sync` first."
            ) from exc

        self._optuna = optuna
        self._json_to_distribution = json_to_distribution
        self._trial_state = TrialState
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

        sampler = optuna.samplers.TPESampler(
            seed=self._seed,
            n_startup_trials=self._startup_trials,
            multivariate=self._multivariate,
            group=self._group,
        )
        self._study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
        )
        self._active_trial = None
        self._serialized_trials = []
        if state is not None:
            for serialized in state.get("optuna_trials", []):
                self._replay_trial(serialized)

    def propose(self) -> Optional[dict[str, Any]]:
        if self._study is None or self._search_space is None or self.should_stop():
            return None
        if not hasattr(self._search_space, "sample_optuna"):
            raise TypeError(
                "BayesianSearchStrategy requires the search space to expose `sample_optuna(trial)`."
            )
        trial = self._study.ask()
        candidate = self._search_space.sample_optuna(trial)
        self._active_trial = trial
        return candidate

    def observe(self, trial_result: dict[str, Any]) -> None:
        if self._active_trial is None:
            raise RuntimeError("observe() called without an active Optuna trial.")

        if trial_result.get("status") == "ok":
            objective = trial_result.get("objective", {})
            score = objective.get("score")
            if score is None:
                raise ValueError(
                    "BayesianSearchStrategy requires objective.evaluate(...) to return a numeric `score`."
                )
            self._study.tell(self._active_trial, float(score))
            self._serialized_trials.append(
                self._serialize_trial(
                    self._active_trial,
                    value=float(score),
                    state="COMPLETE",
                )
            )
        else:
            self._study.tell(self._active_trial, state=self._trial_state.FAIL)
            self._serialized_trials.append(
                self._serialize_trial(
                    self._active_trial,
                    value=None,
                    state="FAIL",
                )
            )

        self._completed_trials += 1
        self._active_trial = None

    def reject_proposal(self, *, reason: str) -> None:
        if self._active_trial is None:
            return
        if reason == "duplicate":
            self._study.tell(self._active_trial, state=self._trial_state.PRUNED)
        else:
            self._study.tell(self._active_trial, state=self._trial_state.FAIL)
        self._active_trial = None

    def should_stop(self) -> bool:
        if self._max_trials is None:
            return False
        return self._completed_trials >= self._max_trials

    def export_state(self) -> dict[str, Any]:
        return {
            "seed": self._seed,
            "completed_trials": self._completed_trials,
            "startup_trials": self._startup_trials,
            "multivariate": self._multivariate,
            "group": self._group,
            "optuna_trials": self._serialized_trials,
        }

    def _serialize_trial(
        self,
        trial,
        *,
        value: Optional[float],
        state: str,
    ) -> dict[str, Any]:
        return {
            "params": dict(trial.params),
            "distributions": {
                name: self._optuna.distributions.distribution_to_json(distribution)
                for name, distribution in trial.distributions.items()
            },
            "value": value,
            "state": state,
        }

    def _replay_trial(self, serialized_trial: dict[str, Any]) -> None:
        distributions = {
            name: self._json_to_distribution(payload)
            for name, payload in serialized_trial.get("distributions", {}).items()
        }
        params = serialized_trial.get("params", {})
        state_name = serialized_trial.get("state", "COMPLETE")
        trial_state = getattr(self._trial_state, state_name, self._trial_state.COMPLETE)
        create_kwargs: dict[str, Any] = {
            "params": params,
            "distributions": distributions,
            "state": trial_state,
        }
        if serialized_trial.get("value") is not None and trial_state == self._trial_state.COMPLETE:
            create_kwargs["value"] = float(serialized_trial["value"])
        frozen_trial = self._optuna.trial.create_trial(**create_kwargs)
        self._study.add_trial(frozen_trial)
        self._serialized_trials.append(serialized_trial)
