# Making AI Less Thirsty

Predicting data center water use and carbon impact with machine learning.

## Problem Statement

Modern AI workloads increase pressure on data center infrastructure by driving
higher compute demand, cooling demand, electricity consumption, and water use.
As model training and inference scale, operators need better tools to estimate
the environmental cost of where and when workloads run.

This project investigates whether machine learning can be used to predict two
important sustainability indicators:

- water usage effectiveness (`WUE_total`)
- carbon intensity derived from `co2_kg / total_gen_kwh`

The broader goal is to support more sustainable workload planning by combining
environmental, temporal, and power-generation features into a predictive
pipeline.

## Dataset Description

The project uses a structured dataset of data center and grid-related signals
stored in `data/sample_dataset.xlsx`.

The loader also supports `.csv`, `.parquet`, and `.zip` archives that contain a
master dataset such as `base_data_with_metrics.parquet`.

The dataset is expected to contain fields such as:

- `TIMESTAMP`
- `WUE_total`
- `co2_kg`
- `total_gen_kwh`
- `total_gen_mwh`
- `EGRIDREGION`
- weather variables such as temperature and humidity
- electricity generation mix features such as `COAL`, `HYDRO`, `NATURALGAS`,
  `NUCLEAR`, `SOLAR`, and `WIND`

These features allow the project to study how operational conditions, time, and
regional energy mix influence water and carbon outcomes.

## Methodology

### Data Preprocessing

The preprocessing pipeline is implemented in [src/preprocessing.py](src/preprocessing.py).
The current workflow:

- converts `TIMESTAMP` into a canonical hourly city time series
- derives cyclic time features such as hour, day of week, month, and
  day-of-year encodings
- computes carbon intensity as `co2_kg / total_gen_kwh`
- normalizes weather signals such as temperature, humidity, and wind speed
- derives fuel-share features from generation mix columns when available
- scales drought stress into a `scarcity_index`
- creates leakage-safe exogenous feature profiles for future horizons and
  evaluation windows
- removes invalid infinite values and drops rows with missing target values
  before training

Data loading is handled by [src/data_loader.py](src/data_loader.py), which
standardizes dataset resolution through `data/sample_dataset.xlsx` and can also
load `.csv`, `.parquet`, and `.zip` inputs.

### Modeling

The training pipeline is implemented in [src/train.py](src/train.py).
The end-to-end orchestration entrypoint is [src/main.py](src/main.py).

The forecasting pipeline trains two target families:

- `WUE_total`
- carbon intensity

For each target, the system trains:

- a per-city Prophet model that captures temporal trend and seasonality
- a direct XGBoost regressor using:
  - time features
  - weather features
  - fuel mix shares
  - water scarcity index
  - one-hot encoded city indicators
- a hybrid model where:
  - Prophet produces the baseline forecast
  - XGBoost learns the residual error
  - final prediction = `Prophet + XGBoost residual`

The pipeline can also optionally merge hourly weather features before training,
including temperature, dew point, humidity, precipitation, wind, pressure, and
derived weather stress indicators.

Training uses a strict time-based split with the final 24-48 hours held out for
evaluation. To avoid leakage, the evaluation horizon and future inference
horizon both use exogenous feature profiles derived only from historical
training data rather than from future observations.

Trained artifacts are saved to:

- `models/co2_model.pkl`
- `models/wue_model.pkl`
- `models/prophet_models/`
- `data/processed/city_forecasts_24h.csv` or `city_forecasts_48h.csv`

### Evaluation

Evaluation utilities are defined in [src/evaluate.py](src/evaluate.py).

The current project reports standard regression metrics:

- RMSE
- MAE
- R2

These metrics are reported separately for Prophet, direct XGBoost, and the
hybrid model for both WUE and carbon intensity. The pipeline also compares the
XGBoost and hybrid models against the Prophet baseline to quantify improvement
in RMSE.

## Results

Results are generated when the forecasting pipeline is executed on the research
dataset.

Planned result reporting includes:

- Prophet vs XGBoost vs hybrid model performance for both prediction targets
- 24-48 hour city-level forecast tables
- feature importance and regional trend interpretation
- discussion of operational recommendations based on predicted impact

## How to Run the Project

### 1. Install Requirements

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Add the Dataset

Place the dataset file at:

```bash
data/sample_dataset.xlsx
```

### 3. Run the Training Pipeline

```bash
python3 -m src.main --data-path data/sample_dataset.xlsx
```

Optional example with custom settings:

```bash
python3 -m src.main \
  --data-path data/sample_dataset.xlsx \
  --history-days 365 \
  --test-horizon-hours 48 \
  --forecast-horizon-hours 48 \
  --random-state 7
```

Optional example with external weather enrichment:

```bash
python3 -m src.main --data-path data/sample_dataset.xlsx --weather-path ../weather_zip_city_2019_2023_meteostat
```

Optional example using the downloaded master archive directly:

```bash
python3 -m src.main --data-path "/Users/drona23/Downloads/Archive 3.zip"
```

When training completes, the pipeline writes:

- XGBoost model bundles to `models/co2_model.pkl` and `models/wue_model.pkl`
- per-city Prophet artifacts under `models/prophet_models/`
- next-horizon forecasts under `data/processed/`

### 4. Build Workload Jobs from the Azure Trace

```bash
python3 -m src.workload_loader \
  --input-path data/external/azure_packing/packing_trace_zone_a_v1.sqlite \
  --output-path data/processed/azure_jobs_sample.csv \
  --limit 5000 \
  --start-datetime "2019-01-01 00:00:00"
```

### 5. Run the Scheduling Pipeline

```bash
python3 -m src.scheduling \
  --data-path "/Users/drona23/Downloads/Archive 3.zip" \
  --jobs-path data/processed/azure_jobs_sample.csv \
  --dc-config-path data/templates/dc_config_template.csv \
  --job-limit 100 \
  --output-path data/processed/schedule_results.csv
```

### 6. Explore the Notebook

The exploratory analysis notebook is located at:

- [notebooks/eda.ipynb](notebooks/eda.ipynb)

It can be used to inspect distributions, temporal trends, regional differences,
and feature relationships before formal model training.

### 7. Launch the API

```bash
python3 -m uvicorn src.api:app --reload
```

The repository now exposes the scheduling simulation through FastAPI rather than
shipping a built-in Streamlit interface.

Core API endpoints:

- `GET /health`
- `GET /context`
- `POST /simulate`

The external visualization layer can call `POST /simulate` with:

- `priority`
- `alpha`
- `beta`
- `gamma`
- `time`
- `latency_sensitivity`
- `workload_size`
- `origin_city`

and receive candidate routes with carbon, water, latency, score, and route
metadata for presentation or product integration.

## Project Structure

```text
Capstone_Research/
├── data/
│   ├── reference/
│   │   └── city_coordinates.csv
│   ├── processed/
│   │   ├── city_forecasts_24h.csv
│   │   └── city_forecasts_48h.csv
│   ├── sample_dataset.xlsx
│   └── templates/
│       └── dc_config_template.csv
├── models/
│   ├── co2_model.pkl
│   ├── wue_model.pkl
│   └── prophet_models/
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── __init__.py
│   ├── app_backend.py
│   ├── api.py
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── main.py
│   ├── preprocessing.py
│   ├── runtime.py
│   ├── scheduling.py
│   ├── train.py
│   ├── weather_loader.py
│   └── workload_loader.py
├── docs/
│   ├── data_requirements.md
│   └── workload_trace_guide.md
├── requirements.txt
├── .gitignore
└── README.md
```

## Future Work

The project can be extended in several important directions:

- optimize the current baseline through hyperparameter tuning
- extend the hybrid XGBoost + Prophet forecasting pipeline with more tuning and
  diagnostics
- incorporate feature selection and model explainability methods
- expand evaluation with cross-validation and temporal holdout strategies
- build a recommendation layer for workload shifting and greener scheduling
- package the workflow into a reproducible cloud pipeline on AWS
- move data ingestion, training, and reporting into an automated MLOps workflow

## Research Framing

This repository is structured as a research-oriented machine learning project:
it combines exploratory analysis, modular preprocessing, baseline modeling, and
reproducible evaluation into a foundation that can support deeper experimental
work on sustainable AI infrastructure.

For the exact list of remaining data gaps and source links, see
[docs/data_requirements.md](docs/data_requirements.md).

For the recommended public workload trace and conversion workflow, see
[docs/workload_trace_guide.md](docs/workload_trace_guide.md).

For the first runnable scheduler and the starter data-center configuration
template, use [src/scheduling.py](src/scheduling.py) and
[data/templates/dc_config_template.csv](data/templates/dc_config_template.csv).
