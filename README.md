# Carbon and Water Aware Compute Scheduling

This repository is a research-only artifact for studying when and where to run
deadline-flexible compute jobs. It contains no web interface, API service, or
deployment stack. The focus is a reproducible scheduling experiment with clear
assumptions, paired baselines, uncertainty tests, and traceable outputs.

## Research question

Can a scheduler reduce operational carbon emissions and onsite water use for
non-preemptive compute jobs by shifting them across time and location, while
preserving job deadlines and data center capacity constraints?

The problem matters because U.S. data centers consumed an estimated 176 TWh in
2023, and the demand range projected for 2028 is substantially higher
[[R1](docs/REFERENCES.md#r1)]. Grid carbon intensity and cooling water use also
vary by time and location [[R2](docs/REFERENCES.md#r2),
[R4](docs/REFERENCES.md#r4)].

## Intended beneficiaries

| Organization type | Representative examples | Decision supported |
|---|---|---|
| Cloud and hyperscale operators | AWS, Google Cloud, Microsoft Azure | Placement of deferrable batch, analytics, training, and maintenance jobs |
| AI infrastructure teams | Model training and batch inference platforms | Carbon and water tradeoff analysis before a run is admitted |
| Research computing institutions | Universities, national laboratories, HPC centers | Queue scheduling under deadlines and site power limits |
| Colocation and enterprise operators | Multi-site data center portfolios | Scenario analysis for regional load shifting |
| Utilities and sustainability researchers | Grid operators, energy institutes, public agencies | Study of flexible computing demand and forecast sensitivity |

These are representative beneficiaries. The repository does not claim a
deployment, partnership, or validation by any named organization.

## What this artifact contributes

Prior work already established carbon-aware scheduling, water-aware scheduling,
and joint carbon-water optimization. This project does not claim to invent those
ideas. Its contribution is a compact evaluation scaffold that combines:

- deadline and hourly capacity feasibility;
- physical water use reported separately from scarcity-weighted water impact;
- per-job normalization before combining quantities with different units;
- paired-job comparisons, so a policy cannot appear cleaner only by completing
  fewer jobs;
- full execution-window carbon accounting with data center PUE;
- forecast-error stress tests against a perfect-forecast reference policy;
- deterministic synthetic inputs that run offline and in continuous integration.

See [the related-work comparison](docs/RELATED_WORK.md) for a claim-by-claim
positioning against published research.

## Reproduced results

The committed results use 10 seeds, 180 jobs per seed, four synthetic zones, and
one week of hourly signals per seed. All policies completed every submitted job
in this benchmark. The values below are paired comparisons over exactly the same
completed jobs.

| Comparison | Operational CO2e | Physical water | Scarcity-weighted water |
|---|---:|---:|---:|
| Proposed vs immediate local | 23.94% lower, CI 1.92 | 28.60% lower, CI 1.89 | 20.08% lower, CI 1.28 |
| Proposed vs random feasible | 22.31% lower, CI 2.18 | 29.50% lower, CI 2.81 | 20.78% lower, CI 1.87 |
| Proposed vs carbon greedy | 46.88% higher, CI 2.11 | 52.75% lower, CI 1.47 | 39.17% lower, CI 1.17 |

The carbon-greedy comparison is important. A joint policy does not dominate a
single-objective carbon policy. It accepts more carbon in this synthetic case to
reduce water impact. That is a tradeoff, not a universal improvement.

![Benchmark results](docs/figures/benchmark_overview.png)

At a forecast noise standard deviation equal to 20% of each city's mean carbon
intensity, the carbon-only policy used 3.65% more operational CO2e than the
perfect-forecast reference, with an approximate 95% confidence interval of 0.54
percentage points. At 50% noise, the delta was 21.42%, with an interval of 1.71
percentage points.

![Forecast robustness](docs/figures/forecast_robustness.png)

These results are simulation evidence only. They are not estimates of savings
for a real company or facility. Full interpretation is in
[docs/RESULTS.md](docs/RESULTS.md).

## Reproduce the artifact

Python 3.11 or later is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pytest
python -m src.experiments
```

The final command regenerates:

- `results/benchmark_summary.csv`
- `results/paired_comparisons_by_seed.csv`
- `results/policy_summary_by_seed.csv`
- `results/forecast_robustness.csv`
- `results/example_schedule.csv`
- `results/run_metadata.json`
- `results/SHA256SUMS`
- the three PNG figures in `docs/figures/`

## Repository map

| Path | Purpose |
|---|---|
| `src/scheduling.py` | Feasibility checks, candidate accounting, normalized objective, paired metrics |
| `src/baselines.py` | Immediate-local, carbon-greedy, and seeded-random feasible baselines |
| `src/synthetic_benchmark.py` | Deterministic offline environment, jobs, and data center configuration |
| `src/forecast_robustness.py` | Full-window realized carbon evaluation under forecast error |
| `src/experiments.py` | Multi-seed runner, CSV export, metadata, and figures |
| `src/fetch_eia.py` | Optional U.S. EIA generation-mix adapter |
| `src/fetch_entso_e.py` | Optional ENTSO-E generation-mix adapter with sub-hourly aggregation |
| `src/workload_loader.py` | Optional converter for the public Azure VM packing trace |
| `tests/` | Unit and offline integration tests |
| `docs/` | Problem statement, methodology, related work, provenance, limitations, and results |

## Read next

- [Research problem and use cases](docs/RESEARCH_PROBLEM.md)
- [Methodology and equations](docs/METHODOLOGY.md)
- [Related work and project difference](docs/RELATED_WORK.md)
- [Data provenance and canonical schemas](docs/DATA_PROVENANCE.md)
- [Results and output guide](docs/RESULTS.md)
- [Threats to validity](docs/THREATS_TO_VALIDITY.md)
- [Reproducibility checklist](docs/REPRODUCIBILITY.md)
- [Verified references](docs/REFERENCES.md)

## Research boundaries

This artifact models operational electricity emissions and onsite cooling water.
It does not model embodied hardware emissions, electricity-generation water,
network transfer energy, data sovereignty, hardware compatibility, electricity
price, marginal emissions, job preemption, or production failure modes. Those
boundaries are deliberate and are documented as future research directions.

## Citation and reuse

Use `CITATION.cff` for the software citation and `references.bib` for the
literature and data sources. Replace the placeholder contributor entry in
`CITATION.cff` with the author's preferred name and ORCID before using the
repository in an application or publication.

No software license has been selected. Copyright therefore remains with the
repository owner, and reuse permission is not granted by default. Choosing a
license is an owner decision that should be completed before public reuse or
external contribution.
