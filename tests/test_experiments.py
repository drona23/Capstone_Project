from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from src.experiments import run_experiments


@pytest.mark.integration
def test_end_to_end_artifacts_are_created(tmp_path) -> None:
    paths = run_experiments(
        results_dir=tmp_path / "results",
        figures_dir=tmp_path / "figures",
        seeds=2,
        jobs=24,
        hours=72,
    )
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths.values())
    summary = pd.read_csv(paths["paired_summary"])
    assert set(summary["baseline"]) == {"immediate_local", "carbon_greedy", "random"}
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert metadata["claim_scope"] == "simulation only"

    checksum_dir = paths["checksums"].parent
    for line in paths["checksums"].read_text(encoding="utf-8").splitlines():
        expected, relative_path = line.split("  ", maxsplit=1)
        artifact = checksum_dir / relative_path
        assert artifact.is_file()
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == expected
