# Contributing

Changes should preserve the research boundary and reproducibility guarantees.

Before submitting a change:

```bash
python -m ruff check src tests
python -m pytest
python -m src.experiments
```

For a method or metric change, include:

- a test that fails under the previous behavior;
- an update to `docs/METHODOLOGY.md`;
- regenerated seed-level and aggregate outputs;
- an update to `docs/RESULTS.md` if interpretation changes;
- a citation for any borrowed model, factor, or dataset;
- a clear statement of new limitations.

Do not add a dashboard or service layer to this branch. Keep application code in
a separate repository or branch so the research artifact remains small and
auditable.
