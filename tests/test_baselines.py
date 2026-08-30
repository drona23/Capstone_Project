from __future__ import annotations

import pandas as pd

from src.baselines import run_baseline


def test_immediate_local_prefers_origin(environment, dc_config, jobs) -> None:
    result = run_baseline("immediate_local", environment, jobs.iloc[[0]], dc_config)
    assert result.loc[0, "assigned_city"] == "Alpha"


def test_all_baselines_respect_deadlines(environment, dc_config, jobs) -> None:
    for strategy in ("immediate_local", "carbon_greedy", "random"):
        result = run_baseline(strategy, environment, jobs, dc_config, seed=7)
        scheduled = result[result["scheduled"].eq(1)]
        assert (scheduled["scheduled_end"] <= pd.to_datetime(scheduled["deadline"])).all()


def test_random_baseline_is_reproducible(environment, dc_config, jobs) -> None:
    first = run_baseline("random", environment, jobs, dc_config, seed=12)
    second = run_baseline("random", environment, jobs, dc_config, seed=12)
    assert first["assigned_dc_id"].tolist() == second["assigned_dc_id"].tolist()
    assert first["scheduled_start"].tolist() == second["scheduled_start"].tolist()
