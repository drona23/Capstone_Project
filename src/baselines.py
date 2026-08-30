"""Feasible reference policies used by the experiment suite."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from .scheduling import ScheduleCandidate, schedule_with_policy


def _immediate_local_selector(
    job: pd.Series,
    candidates: list[ScheduleCandidate],
) -> ScheduleCandidate:
    """Prefer the origin city, then the earliest feasible placement."""
    origin = str(job["origin_city"])
    return min(
        candidates,
        key=lambda item: (
            item.city != origin,
            item.start_time,
            item.dc_id,
        ),
    )


def _carbon_greedy_selector(
    _job: pd.Series,
    candidates: list[ScheduleCandidate],
) -> ScheduleCandidate:
    """Minimize predicted operational carbon only."""
    return min(
        candidates,
        key=lambda item: (item.expected_co2_kg, item.start_time, item.dc_id),
    )


def _random_selector(seed: int) -> Callable:
    rng = np.random.default_rng(seed)

    def select(
        _job: pd.Series,
        candidates: list[ScheduleCandidate],
    ) -> ScheduleCandidate:
        return candidates[int(rng.integers(0, len(candidates)))]

    return select


def run_baseline(
    strategy: str,
    env_df: pd.DataFrame,
    jobs_df: pd.DataFrame,
    dc_df: pd.DataFrame,
    seed: int = 0,
) -> pd.DataFrame:
    """Run a named baseline under the same capacity and deadline constraints."""
    selectors = {
        "immediate_local": _immediate_local_selector,
        "carbon_greedy": _carbon_greedy_selector,
        "random": _random_selector(seed),
    }
    if strategy not in selectors:
        choices = ", ".join(sorted(selectors))
        raise ValueError(f"Unknown baseline {strategy!r}. Choose one of: {choices}")

    return schedule_with_policy(
        env_df=env_df,
        jobs_df=jobs_df,
        dc_df=dc_df,
        selector=selectors[strategy],
    )
