"""Run the complete offline benchmark and write auditable research artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .baselines import run_baseline
from .forecast_robustness import run_forecast_robustness
from .scheduling import paired_comparison, schedule_jobs, summarize_schedule
from .synthetic_benchmark import generate_benchmark

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
BASELINES = ("immediate_local", "carbon_greedy", "random")
PAIRED_METRICS = (
    "co2_reduction_pct",
    "water_reduction_pct",
    "scarcity_weighted_water_reduction_pct",
    "baseline_coverage",
    "proposed_coverage",
    "coverage_gap_percentage_points",
    "common_jobs",
    "common_it_energy_kwh",
)


def _mean_ci95(values: pd.Series) -> tuple[float, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    mean = float(numeric.mean())
    ci = (
        float(1.96 * numeric.std(ddof=1) / math.sqrt(len(numeric)))
        if len(numeric) > 1
        else 0.0
    )
    return mean, ci


def _aggregate_paired(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for baseline, group in frame.groupby("baseline"):
        row: dict[str, float | str] = {"baseline": str(baseline)}
        for metric in PAIRED_METRICS:
            mean, ci = _mean_ci95(group[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95"] = ci
        row["seeds"] = float(len(group))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("baseline").reset_index(drop=True)


def _plot_benchmark(summary: pd.DataFrame, destination: Path) -> None:
    labels = [value.replace("_", " ").title() for value in summary["baseline"]]
    positions = np.arange(len(labels))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].bar(
        positions - width / 2,
        summary["co2_reduction_pct_mean"],
        width,
        yerr=summary["co2_reduction_pct_ci95"],
        capsize=4,
        label="Operational CO2e",
        color="#2f6b3b",
    )
    axes[0].bar(
        positions + width / 2,
        summary["scarcity_weighted_water_reduction_pct_mean"],
        width,
        yerr=summary["scarcity_weighted_water_reduction_pct_ci95"],
        capsize=4,
        label="Scarcity-weighted water",
        color="#3478a8",
    )
    axes[0].axhline(0, color="#222222", linewidth=0.8)
    axes[0].set_ylabel("Paired-job reduction (%)")
    axes[0].set_title("Proposed policy relative to each baseline")
    axes[0].set_xticks(positions, labels, rotation=15, ha="right")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].errorbar(
        positions,
        100 * summary["baseline_coverage_mean"],
        yerr=100 * summary["baseline_coverage_ci95"],
        marker="o",
        linewidth=2,
        capsize=4,
        label="Baseline",
        color="#777777",
    )
    axes[1].errorbar(
        positions,
        100 * summary["proposed_coverage_mean"],
        yerr=100 * summary["proposed_coverage_ci95"],
        marker="o",
        linewidth=2,
        capsize=4,
        label="Proposed",
        color="#7b3f98",
    )
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("Jobs completed before deadline (%)")
    axes[1].set_title("Coverage is reported separately")
    axes[1].set_xticks(positions, labels, rotation=15, ha="right")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("Synthetic benchmark, mean and approximate 95% CI across seeds")
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_robustness(frame: pd.DataFrame, destination: Path) -> None:
    x = 100 * frame["noise_level"].to_numpy(dtype=float)
    y = frame["carbon_delta_pct_mean"].to_numpy(dtype=float)
    ci = frame["carbon_delta_pct_ci95"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(x, y, marker="o", linewidth=2.2, color="#9a4d1f")
    ax.fill_between(x, y - ci, y + ci, alpha=0.20, color="#9a4d1f")
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_xlabel("Forecast noise standard deviation (% of city mean)")
    ax.set_ylabel("CO2e delta vs perfect-forecast policy (%)")
    ax.set_title("Carbon-only forecast stress test on paired completed jobs")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_example_schedule(frame: pd.DataFrame, destination: Path) -> None:
    display = frame.loc[
        frame["scheduled"].eq(1),
        [
            "job_id",
            "origin_city",
            "assigned_city",
            "scheduled_start",
            "expected_co2_kg",
            "expected_water_liters",
        ],
    ].head(8).copy()
    display["scheduled_start"] = pd.to_datetime(display["scheduled_start"]).dt.strftime(
        "%Y-%m-%d %H:%M"
    )
    display["expected_co2_kg"] = display["expected_co2_kg"].map(lambda value: f"{value:.2f}")
    display["expected_water_liters"] = display["expected_water_liters"].map(
        lambda value: f"{value:.2f}"
    )
    display.columns = [
        "Job",
        "Origin",
        "Assigned zone",
        "UTC start",
        "CO2e (kg)",
        "Water (L)",
    ]
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.axis("off")
    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    for column in range(len(display.columns)):
        table[(0, column)].set_facecolor("#dbe9df")
        table[(0, column)].set_text_props(weight="bold")
    ax.set_title("Example proposed schedule, synthetic seed 42", pad=18, fontsize=14)
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_checksums(paths: dict[str, Path], destination: Path) -> None:
    lines = []
    for label, path in sorted(paths.items()):
        if label == "checksums":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        relative_path = os.path.relpath(path, start=destination.parent)
        lines.append(f"{digest}  {relative_path}")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiments(
    results_dir: Path = DEFAULT_RESULTS_DIR,
    figures_dir: Path = DEFAULT_FIGURES_DIR,
    seeds: int = 10,
    jobs: int = 180,
    hours: int = 168,
) -> dict[str, Path]:
    """Run all policies, paired comparisons, and forecast-error stress tests."""
    if seeds < 2:
        raise ValueError("Use at least two seeds to estimate uncertainty.")
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    policy_rows: list[dict[str, float | str]] = []
    paired_rows: list[dict[str, float | str]] = []
    example_schedule: pd.DataFrame | None = None

    for seed in range(42, 42 + seeds):
        env, jobs_df, dc_df = generate_benchmark(seed=seed, jobs=jobs, hours=hours)
        proposed = schedule_jobs(env, jobs_df, dc_df)
        if example_schedule is None:
            example_schedule = proposed.head(25).copy()
        proposed_summary = summarize_schedule(proposed)
        policy_rows.append({"seed": seed, "policy": "proposed", **proposed_summary})

        for baseline_name in BASELINES:
            baseline = run_baseline(
                baseline_name, env, jobs_df, dc_df, seed=seed
            )
            policy_rows.append(
                {
                    "seed": seed,
                    "policy": baseline_name,
                    **summarize_schedule(baseline),
                }
            )
            paired_rows.append(
                {
                    "seed": seed,
                    "baseline": baseline_name,
                    **paired_comparison(baseline, proposed),
                }
            )

    policy_frame = pd.DataFrame(policy_rows)
    paired_frame = pd.DataFrame(paired_rows)
    paired_summary = _aggregate_paired(paired_frame)

    env, jobs_df, dc_df = generate_benchmark(seed=42, jobs=jobs, hours=hours)
    robustness = run_forecast_robustness(
        env,
        dc_df,
        jobs_df,
        repetitions=max(8, seeds),
        seed=42,
    )

    paths = {
        "policy_by_seed": results_dir / "policy_summary_by_seed.csv",
        "paired_by_seed": results_dir / "paired_comparisons_by_seed.csv",
        "paired_summary": results_dir / "benchmark_summary.csv",
        "robustness": results_dir / "forecast_robustness.csv",
        "example_schedule": results_dir / "example_schedule.csv",
        "metadata": results_dir / "run_metadata.json",
        "benchmark_figure": figures_dir / "benchmark_overview.png",
        "robustness_figure": figures_dir / "forecast_robustness.png",
        "schedule_figure": figures_dir / "example_schedule.png",
        "checksums": results_dir / "SHA256SUMS",
    }
    policy_frame.to_csv(paths["policy_by_seed"], index=False, float_format="%.8f")
    paired_frame.to_csv(paths["paired_by_seed"], index=False, float_format="%.8f")
    paired_summary.to_csv(paths["paired_summary"], index=False, float_format="%.8f")
    robustness.to_csv(paths["robustness"], index=False, float_format="%.8f")
    assert example_schedule is not None
    example_schedule.to_csv(paths["example_schedule"], index=False, float_format="%.8f")
    paths["metadata"].write_text(
        json.dumps(
            {
                "benchmark": "synthetic-v1",
                "seed_start": 42,
                "seed_count": seeds,
                "jobs_per_seed": jobs,
                "hours_per_seed": hours,
                "weights": {"carbon": 0.55, "water": 0.35, "relocation": 0.10},
                "forecast_repetitions": max(8, seeds),
                "claim_scope": "simulation only",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _plot_benchmark(paired_summary, paths["benchmark_figure"])
    _plot_robustness(robustness, paths["robustness_figure"])
    _plot_example_schedule(example_schedule, paths["schedule_figure"])
    _write_checksums(paths, paths["checksums"])
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--jobs", type=int, default=180)
    parser.add_argument("--hours", type=int, default=168)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    args = parser.parse_args()
    paths = run_experiments(
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
        seeds=args.seeds,
        jobs=args.jobs,
        hours=args.hours,
    )
    print("Research artifacts written:")
    for label, path in paths.items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
