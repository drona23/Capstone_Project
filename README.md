# Capstone Project

Making AI Less Thirsty: predicting data center water and carbon impact using
machine learning.

## Project Structure

```
Capstone_Research/
├── data/
│   └── sample_dataset.xlsx
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluate.py
├── requirements.txt
├── .gitignore
└── README.md
```

## What Each Module Does

- `src/data_loader.py`: resolves dataset paths and loads Excel data.
- `src/preprocessing.py`: builds modeling features and targets (`WUE_total` and
  carbon intensity).
- `src/train.py`: trains RandomForest models and reports evaluation metrics.
- `src/evaluate.py`: reusable regression metrics (RMSE, MAE, R2).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Place your dataset at `data/sample_dataset.xlsx`.

The provided file path is now standardized so both scripts and notebooks can
use the same location.

## Run Training

From the project root:

```bash
python3 -m src.train --data-path data/sample_dataset.xlsx
```

Optional flags:

```bash
python3 -m src.train --test-size 0.25 --random-state 7 --no-save-models
```

## Notes

- If `data/sample_dataset.xlsx` is empty or missing, training will fail with a
  clear message.
- The notebook now lives in `notebooks/eda.ipynb` and points to
  `../data/sample_dataset.xlsx`.
