# Making AI Less Thirsty

> Predicting how much carbon and water a data center will use — then automatically sending AI jobs to the cleanest one available.

---

## The Problem in Plain English

When you ask an AI model a question, a computer somewhere runs the calculation. That computer uses electricity and water to stay cool. The problem:

- **Different power sources produce different carbon** — a data center running on solar emits far less CO₂ than one running on coal
- **Carbon output changes by the hour** — at night in Texas, wind covers most of the grid; midday on a cloudy day, coal picks up the slack
- **Nobody is optimising for this** — jobs get sent to the nearest or cheapest server, not the cleanest one

This project builds a system that watches those changes in real time and routes computing jobs toward whichever data center is cleanest *right now*.

---

## What It Actually Does

```
Step 1 — Collect data
  Download hourly electricity generation (EIA) and weather (Open-Meteo)
  for 28 U.S. data center cities. Free government data, no API keys needed.

Step 2 — Predict the future
  Train machine learning models that forecast CO₂ intensity and water
  usage for each city 48 hours ahead. Uses XGBoost + Prophet + hybrid.

Step 3 — Schedule jobs
  Given a batch of AI workloads and the 48-hour forecast, find the
  routing plan that minimises carbon + water while respecting deadlines.

Step 4 — Show the reasoning
  For any routing decision, explain which features (temperature, time of
  day, fuel mix) pushed the prediction up or down — using SHAP values.

Step 5 — Visualise it
  Interactive U.S. map showing routing flows, environmental trade-offs,
  and a baseline comparison so you can see the improvement.
```

---

## Results (from real model output)

### CO₂ varies 3.2× across cities

| City | Avg CO₂ Intensity |
|---|---|
| Albany, NY (cleanest) | 0.206 kg CO₂/kWh |
| Dallas, TX | 0.320 kg CO₂/kWh |
| Dona Ana County, NM (dirtiest) | 0.667 kg CO₂/kWh |

**What this means:** Sending the same job to Albany instead of Dona Ana County produces one-third the carbon. The scheduler exploits this gap automatically.

### Water usage also varies by city

| City | WUE (litres per kWh) |
|---|---|
| Northern Indiana (most efficient) | 1.117 |
| Shackelford County, TX (least efficient) | 1.186 |

### Scheduler performance (50-job test)

| Jobs submitted | Jobs scheduled | Coverage |
|---|---|---|
| 50 | 50 | 100% |

---

## How the System Is Built

```
┌─────────────────────────────────────────────────────────────────────┐
│  DATA LAYER                                                          │
│  EIA (electricity grid) + Open-Meteo (weather) + EPA (emissions)    │
│  → build_dataset.py → one clean CSV with everything merged          │
│                                                                      │
│  Azure VM Trace (2.2M real cloud jobs)                              │
│  → workload_loader.py → realistic job queue for testing             │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  ML LAYER                                                            │
│  For each city, for each hour, predict:                             │
│    • CO₂ intensity (how dirty is the electricity right now?)        │
│    • WUE (how much water does cooling use?)                         │
│                                                                      │
│  Three models trained per target:                                   │
│    Prophet  — captures time patterns (rush hour, seasonal)          │
│    XGBoost  — learns from weather + fuel mix + time features        │
│    Hybrid   — Prophet + XGBoost residual (best of both)             │
│                                                                      │
│  tune.py   — Optuna auto-searches for best XGBoost settings        │
│  explain.py — SHAP shows WHY a prediction is high or low           │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  SCHEDULER + API + DASHBOARD                                         │
│  scheduling.py — scores every (job, city, time) combination:        │
│    score = α × carbon + β × water + γ × distance                   │
│  α, β, γ are sliders you control — trade off green vs fast          │
│                                                                      │
│  FastAPI backend  — serves predictions and routing decisions        │
│  React frontend   — interactive U.S. map, batch mode, SHAP chart   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Weather Pipeline Integration

This project connects to a **live weather data pipeline** built in the companion `weather_pipeline` project (Apache Airflow + PostgreSQL).

When the weather pipeline is running, Capstone can pull **real-time hourly weather** for all 28 cities instead of using historical CSV data:

```python
from src.weather_loader import load_from_pipeline_db

# Returns live weather from the last 48 hours
df = load_from_pipeline_db(hours=48)
```

The pipeline runs as a Docker service on `localhost:15432`. If it's not running, the system falls back to the historical Meteostat dataset automatically.

---

## Quick Start

### Option A — Docker (recommended, no setup required)

```bash
git clone <this-repo>
cd Capstone_Research
docker compose up --build
```

Open in your browser:
- **`http://localhost:3000`** — the interactive routing dashboard
- **`http://localhost:8000/docs`** — the API with a built-in test interface

### Option B — Run locally

```bash
# Install Python dependencies
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Add the dataset (required)
# Place base_data_with_metrics.parquet or sample_dataset.xlsx in data/

# Train the ML models
python3 -m src.main --data-path data/sample_dataset.xlsx

# Start the API
python3 -m uvicorn src.api:app --reload

# In a second terminal, start the frontend
cd frontend && npm install && npm run dev
# Open http://localhost:5173
```

---

## Tuning the Models (Optional)

The default XGBoost settings work well. To automatically find better ones:

```bash
# Search 40 parameter combinations per model (~5 minutes)
python3 -m src.tune

# Re-train using the best parameters found above
python3 -m src.main
```

Results are saved to `models/best_params.json`. Training picks them up automatically next time.

---

## Running Tests

```bash
python3 -m pytest tests/ -v
# Expected: 32 passed, 1 skipped
```

Tests cover three areas:
- **Unit tests** — the scheduler's scoring formula and job builder
- **API tests** — every endpoint with valid and invalid inputs
- **SHAP tests** — the explainer's output shape and sorting (skipped if shap not installed)

---

## API Endpoints

| Endpoint | What it does |
|---|---|
| `GET /health` | Check the server is running |
| `GET /context` | Get available cities, time range, and default settings |
| `POST /simulate` | Get a routing recommendation for one job |
| `POST /simulate-batch` | Schedule 50–200 jobs across data centers |
| `GET /explain` | Explain why a city got a particular CO₂ or WUE prediction |

### Example: get a routing recommendation

```bash
curl -X POST http://localhost:8000/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "priority": "medium",
    "time": "2024-06-01T14:00",
    "alpha": 1.0,
    "beta": 1.0,
    "gamma": 1.0
  }'
```

### Example: explain a prediction

```bash
curl "http://localhost:8000/explain?city=Dallas&target=co2&time=2024-06-01T14:00"
```

Returns which features (temperature, time of day, fuel mix) pushed the CO₂ prediction up or down, and by how much.

---

## The Scheduling Formula

The scheduler picks the city that minimises this score for each job:

```
score = α × (CO₂ intensity × server efficiency)
      + β × (water usage × local water scarcity)
      + γ × (distance penalty — 0 if same city, 1 if different)
```

You control α, β, and γ through sliders in the dashboard:
- **α = 2, β = 0, γ = 0** → pure carbon optimisation, ignore water and distance
- **α = 0, β = 0, γ = 2** → always send to nearest data center
- **α = 1, β = 1, γ = 1** → balanced trade-off (default)

---

## Project Files

```
Capstone_Research/
│
├── src/                     Python source code
│   ├── fetch_eia.py         Downloads electricity grid data from EIA
│   ├── fetch_weather.py     Downloads weather data from Open-Meteo
│   ├── build_dataset.py     Merges EIA + weather into one dataset
│   ├── preprocessing.py     Cleans data and builds ML features
│   ├── train.py             Trains XGBoost + Prophet + Hybrid models
│   ├── tune.py              Finds best XGBoost settings using Optuna
│   ├── explain.py           Explains predictions using SHAP
│   ├── scheduling.py        Routes jobs to optimal data centers
│   ├── api.py               FastAPI web server
│   ├── app_backend.py       Backend logic used by the API
│   ├── weather_loader.py    Loads weather data (CSV or live pipeline)
│   └── workload_loader.py   Converts Azure VM trace into job records
│
├── frontend/                React web dashboard
│   ├── src/components/      Map, control panel, insight panel
│   ├── nginx.conf           Routes API calls to backend in Docker
│   └── Dockerfile           Builds frontend for Docker
│
├── tests/                   Automated tests
│   ├── test_scheduling.py   Tests the scoring formula
│   ├── test_api.py          Tests every API endpoint
│   └── test_explain.py      Tests the SHAP explainer output
│
├── models/                  Saved ML models (not in git — too large)
│   ├── co2_model.pkl        Carbon intensity model bundle
│   ├── wue_model.pkl        Water usage model bundle
│   └── best_params.json     Optuna tuning results (created by tune.py)
│
├── data/                    Datasets (not in git — too large)
│   ├── processed/           Model forecasts and schedule outputs
│   ├── reference/           City coordinates
│   └── templates/           Data center configuration template
│
├── Dockerfile               Builds the Python backend for Docker
├── docker-compose.yml       Starts backend + frontend together
├── requirements.txt         Python package versions
└── pytest.ini               Test configuration
```

---

## Data Sources (all free)

| Source | What it provides |
|---|---|
| [EIA Open Data](https://www.eia.gov/opendata/) | Hourly electricity generation mix for 7 US grid regions |
| [Open-Meteo](https://open-meteo.com/) | Historical hourly weather — no API key needed |
| [EPA eGRID](https://www.epa.gov/egrid) | How much CO₂ each fuel type produces |
| [Azure Public Dataset](https://github.com/Azure/AzurePublicDataset) | 2.2 million real Microsoft VM job records |

---

## Tech Stack

| What | How |
|---|---|
| ML models | XGBoost, Prophet, scikit-learn |
| Hyperparameter search | Optuna (Bayesian, not random) |
| Model explanations | SHAP (TreeExplainer) |
| API | FastAPI + Uvicorn |
| Dashboard | React 19, TypeScript, MapLibre GL |
| Containers | Docker Compose |
| Tests | pytest |

---

*For data source details and gaps, see [docs/data_requirements.md](docs/data_requirements.md)*
*For the Azure trace setup, see [docs/workload_trace_guide.md](docs/workload_trace_guide.md)*
