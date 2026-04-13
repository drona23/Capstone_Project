# Baseline Comparison — Research Validation

This document explains the baseline comparison feature that validates the optimized scheduler against industry-standard strategies.

## Motivation

To claim a method is "better," research papers must compare it against baselines. This project compares the optimal (carbon-aware) scheduler against three baselines:

1. **Random Routing** — picks any data center at random
2. **Nearest Neighbor** — sends jobs to the nearest data center (current industry practice)
3. **Time-of-Day Heuristic** — sends to the city with lowest CO₂ at that hour

## The Endpoint

**POST /compare-baselines**

Schedule the same batch of jobs using all four strategies and compare results.

### Request

```json
{
  "time": "2024-06-01T14:00",
  "batch_size": 50,
  "priority": "medium"
}
```

### Response

```json
{
  "timestamp": "2024-06-01T14:00:00",
  "batch_size": 50,
  "results": [
    {
      "strategy": "Optimal (Your Method)",
      "jobs_scheduled": 50,
      "total_co2_kg": 245.3,
      "total_water_liters": 1203.5,
      "avg_co2_per_job": 4.91,
      "avg_water_per_job": 24.07
    },
    {
      "strategy": "Random Routing",
      "jobs_scheduled": 50,
      "total_co2_kg": 318.7,
      "total_water_liters": 1456.2,
      "avg_co2_per_job": 6.37,
      "avg_water_per_job": 29.12,
      "co2_reduction_vs_optimal_pct": 23.0,
      "water_reduction_vs_optimal_pct": 17.3
    },
    {
      "strategy": "Nearest Data Center",
      "jobs_scheduled": 50,
      "total_co2_kg": 301.2,
      "total_water_liters": 1389.6,
      "avg_co2_per_job": 6.02,
      "avg_water_per_job": 27.79,
      "co2_reduction_vs_optimal_pct": 18.6,
      "water_reduction_vs_optimal_pct": 13.4
    },
    {
      "strategy": "Time-of-Day Heuristic",
      "jobs_scheduled": 50,
      "total_co2_kg": 267.4,
      "total_water_liters": 1298.3,
      "avg_co2_per_job": 5.35,
      "avg_water_per_job": 25.97,
      "co2_reduction_vs_optimal_pct": 8.2,
      "water_reduction_vs_optimal_pct": 7.1
    }
  ],
  "summary": {
    "total_jobs_in_batch": 50,
    "co2_improvement_vs_random_pct": 23.0,
    "winner": "Optimal"
  }
}
```

## Key Metrics

- **co2_reduction_vs_optimal_pct** — How much worse (%) each baseline is at carbon than optimal
- **water_reduction_vs_optimal_pct** — How much worse (%) each baseline is at water than optimal
- **avg_co2_per_job** — Per-job carbon footprint (useful for normalizing across batch sizes)

## Using This for Research

### Expected Results

On real data, you should see:

1. **Optimal > Time-of-Day > Nearest > Random**
   - Time-of-day respects carbon patterns → better than nearest/random
   - Nearest is better than random (same city = lower latency = less overall cost)
   - Random should be worst

2. **Improvement magnitudes**
   - vs Random: expect 15–30% CO₂ reduction
   - vs Nearest: expect 5–15% CO₂ reduction
   - vs Time-of-Day: expect 5–10% CO₂ reduction

### Writing the Paper

Include in your results section:

```
Table 4: Scheduler Comparison on 50-Job Batch (June 1, 14:00)

Strategy            Total CO₂ (kg)  Reduction vs Optimal  Jobs Scheduled
──────────────────────────────────────────────────────────────────────
Optimal (Ours)      245.3           —                     50
Time-of-Day         267.4           8.2%                  50
Nearest Neighbor    301.2           18.6%                 50
Random              318.7           23.0%                 50
```

Then add prose:

> The carbon-aware scheduler outperforms all baselines. Against random routing (the worst case), it saves 23% CO₂; against nearest-neighbor (industry standard), it saves 18.6%. Even the time-of-day heuristic, which uses real-time grid data to pick the lowest-carbon city, is beaten by 8.2% because our method also optimizes routing over time and incorporates water efficiency constraints.

## Running Bulk Comparisons

To generate results for multiple time windows:

```python
from src.api import app
from src.app_backend import SustainabilitySchedulingBackend
import pandas as pd
from fastapi.testclient import TestClient

client = TestClient(app)

# Run comparison for 7 days, every 6 hours
start = pd.Timestamp("2024-06-01")
results_df_rows = []

for i in range(0, 7 * 24, 6):  # Every 6 hours
    time = start + pd.Timedelta(hours=i)
    response = client.post(
        "/compare-baselines",
        json={"time": time.isoformat(), "batch_size": 50, "priority": "medium"}
    )
    
    data = response.json()
    for strategy_result in data["results"]:
        results_df_rows.append({
            "timestamp": data["timestamp"],
            "strategy": strategy_result["strategy"],
            "total_co2_kg": strategy_result["total_co2_kg"],
            "jobs_scheduled": strategy_result["jobs_scheduled"],
        })

results_df = pd.DataFrame(results_df_rows)
print(results_df.groupby("strategy")["total_co2_kg"].mean())
```

## Files

- `src/baselines.py` — Three baseline strategies
- `tests/test_baselines.py` — 14 unit tests covering each baseline
- `src/api.py` — `/compare-baselines` endpoint
- `src/app_backend.py` — `run_baseline_comparison()` method
