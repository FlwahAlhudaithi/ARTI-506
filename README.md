# ARTI-506 Project — MediSight G3 (Assignment 2)

A data science pipeline for analyzing hospital admission patterns across Saudi Arabian medical facilities using machine learning.

## Overview
This notebook covers a full end-to-end ML pipeline applied to a synthetic hospital dataset, including:
- **Data Overview & Loading** — Exploring patient admission records across multiple hospitals (Mecca, Dammam, Medina)
- **Preprocessing** — Handling missing values, removing duplicates, and feature engineering
- **EDA** — 11 visualizations covering seasonal trends, age/gender breakdowns, risk scores, condition distributions, and city-level analysis
- **Outlier Handling** — Detection and treatment
- **Advanced Preprocessing** — Skewness correction (log transform), correlation analysis, and categorical encoding
- **Feature Selection** — XGBoost-based importance ranking
- **Modeling (Assignment 2)**
  - **Model 1** — XGBoost with data leakage (`patient_risk_score` included) → artificially high performance
  - **Model 2** — XGBoost with leakage fixed → realistic evaluation

## Dataset
Synthetic hospital dataset with features: admission date, hospital name, condition type, patient demographics, severity level, length of stay, comorbidities, diagnosis codes, medication dosage, and emergency visits.

## Tech Stack
`Python` · `XGBoost` · `Pandas` · `Matplotlib` · `Seaborn` · `Scikit-learn` · `Google Colab`

## Course
ARTI-506 · Imam Abdulrahman Bin Faisal University · Group G3
