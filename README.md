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

- converts `TIMESTAMP` into datetime features such as hour, day of week, and
  month
- computes carbon intensity as `co2_kg / total_gen_kwh`
- derives fuel-share features from generation mix columns when available
- keeps numeric modeling features and removes invalid infinite values
- drops rows with missing target values before training

Data loading is handled by [src/data_loader.py](src/data_loader.py), which
standardizes dataset resolution through `data/sample_dataset.xlsx` and can also
load `.csv`, `.parquet`, and `.zip` inputs.

### Modeling

The training pipeline is implemented in [src/train.py](src/train.py).
The end-to-end orchestration entrypoint is [src/main.py](src/main.py).

The current baseline approach trains two separate regression models:

- one model for `WUE_total`
- one model for carbon intensity

Both models use:

- median imputation for missing numeric values
- `RandomForestRegressor` as the baseline estimator
- a train/test split for out-of-sample evaluation

The pipeline can also optionally merge hourly weather features before training,
including temperature, dew point, humidity, precipitation, wind, pressure, and
derived weather stress indicators.

The dependency stack also includes `xgboost` and `prophet` so the project can
expand into stronger gradient-boosted baselines and time-aware forecasting
experiments in later iterations.

### Evaluation

Evaluation utilities are defined in [src/evaluate.py](src/evaluate.py).

The current project reports standard regression metrics:

- RMSE
- MAE
- R2

These metrics are printed separately for the water-use model and the
carbon-intensity model.

## Results

Results are currently a placeholder until the finalized dataset is loaded and
the training pipeline is executed on the full research data.

Planned result reporting includes:

- baseline model performance for both prediction targets
- comparison across candidate models such as Random Forest and XGBoost
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
python3 -m src.main --data-path data/sample_dataset.xlsx --test-size 0.25 --random-state 7
```

Optional example with external weather enrichment:

```bash
python3 -m src.main --data-path data/sample_dataset.xlsx --weather-path ../weather_zip_city_2019_2023_meteostat
```

Optional example using the downloaded master archive directly:

```bash
python3 -m src.main --data-path "/Users/drona23/Downloads/Archive 3.zip"
```

### 4. Explore the Notebook

The exploratory analysis notebook is located at:

- [notebooks/eda.ipynb](notebooks/eda.ipynb)

It can be used to inspect distributions, temporal trends, regional differences,
and feature relationships before formal model training.

## Project Structure

```text
Capstone_Research/
├── data/
│   └── sample_dataset.xlsx
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── main.py
│   ├── train.py
│   ├── evaluate.py
│   └── weather_loader.py
├── docs/
│   └── data_requirements.md
├── requirements.txt
├── .gitignore
└── README.md
```

## Future Work

The project can be extended in several important directions:

- optimize the current baseline through hyperparameter tuning
- add XGBoost and Prophet experiments for stronger predictive benchmarks
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
