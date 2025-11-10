# Capstone_Project
**Making AI Less Thirsty: Predicting Data Center Water & Carbon Impact Using Machine Learning**
This project looks at how the regions of data centers use water and produce carbon emissions, and how machine learning can help reduce that impact. As AI systems grow, data centers need more cooling and power, which leads to higher water usage and more CO₂ output.
The goal of this project is to build models that can predict these sustainability metrics and give practical suggestions on how to lower them. Instead of just showing numbers, the system explains when the impact might rise and what steps can be taken to keep it under control.

**Features**:
Predicts Water Usage Effectiveness (WUE)
Predicts CO₂ emissions
Shows environmental trends across different regions
Compares baseline and tuned models
Uses hyperparameter tuning to boost performance
Gives recommendations to reduce water and carbon impact
Includes visualizations for trends, model results, and patterns
Works smoothly with large Parquet data files

**Technologies Used**
Python
Pandas, NumPy
Scikit-learn (RandomForest, XGBoost, etc.)
Matplotlib, Seaborn
Parquet file handling
GitHub Codespaces

**Dataset**
The dataset contains environmental and workload information such as:
Timestamp, city, zip code, region
Temperature, humidity
WUE and CO₂ emissions
Power and workload-related fields
Energy grid region (EGRIDREGION)
The data was cleaned by removing outliers, filling missing values, encoding categorical features, and preparing it for ML models.

**Model Pipeline**
Load and clean data
Create new features (time-based, region-based, environmental)
Split into training and testing sets
Train baseline models
Perform hyperparameter tuning
Measure performance using RMSE, MAE, and R²
Generate recommendations based on predictions

**Recommendation Engine**
When the model predicts a rise in WUE or CO₂, it suggests steps like:
Moving workloads to cleaner or cooler regions
Scheduling heavy tasks when temperatures are lower
Adjusting cooling strategies
Choosing energy grids with lower carbon intensity
This makes the project a helpful guide for running more sustainable data centers.

