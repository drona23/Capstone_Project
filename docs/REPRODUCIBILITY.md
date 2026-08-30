# Reproducibility Checklist

## Environment

- Python 3.11 or later
- Dependency versions pinned in `requirements.txt` and `requirements-dev.txt`
- No API key or network access required for the default benchmark
- Fixed seed start, seed count, jobs, horizon, and weights in run metadata
- Headless figure generation for CI and remote systems

## Clean run

```bash
git clone https://github.com/drona23/Personal_Project.git
cd Personal_Project
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m ruff check src tests
python -m pytest
python -m src.experiments
```

## Expected checks

The test suite covers:

- UTC conversion;
- exact deadline completion and impossible deadlines;
- hourly capacity enforcement;
- score-weight validation;
- paired-job comparison when coverage differs;
- immediate-local and seeded-random baseline behavior;
- full-window realized carbon accounting with PUE;
- zero-noise forecast identity;
- EIA fuel-code parsing;
- ENTSO-E 15-minute energy aggregation;
- end-to-end artifact creation.

## Output verification

After the full run, confirm:

```bash
python -m pytest
python -m ruff check src tests
git diff --exit-code -- results docs/figures
cd results && shasum -a 256 -c SHA256SUMS
```

The final command succeeds when the regenerated tracked artifacts match the
committed files. PNG bytes can vary if operating-system font rendering differs,
even with the same Matplotlib version. In that case, compare the CSV files first
and inspect the figures visually.

## Parameterized run

```bash
python -m src.experiments \
  --seeds 30 \
  --jobs 500 \
  --hours 336 \
  --results-dir scratch/results \
  --figures-dir scratch/figures
```

Use a scratch path for exploratory runs so the reviewed benchmark remains
unchanged.

## Continuous integration

`.github/workflows/ci.yml` installs the pinned environment, runs Ruff and pytest,
and executes a smaller offline smoke benchmark. No private dataset or developer
home-directory fallback is used.

## Reporting checklist for a paper extension

- Commit the experiment configuration.
- Save raw and derived data checksums.
- Record source versions and retrieval times.
- Separate average from marginal carbon intensity.
- Define onsite and offsite water boundaries.
- Report every policy's coverage.
- Use a paired functional unit for impact reductions.
- Publish seed-level outputs, not only aggregate charts.
- Include negative and null results.
- Document all exclusions and failed runs.
- Replace the placeholder author metadata in `CITATION.cff`.
- Select and add a software license before inviting external reuse.
