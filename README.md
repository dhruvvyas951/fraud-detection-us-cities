# Bank Transaction Fraud Detection Across U.S. Cities Using Machine Learning

A complete data science pipeline for detecting fraudulent bank transactions across 43 U.S. cities using machine learning, featuring geographic fraud hotspot analysis, statistical hypothesis testing, and comparative model evaluation.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [How to Run](#how-to-run)
- [Pipeline Stages](#pipeline-stages)
- [Key Findings](#key-findings)
- [Model Performance](#model-performance)
- [Fraud Hotspot Map](#fraud-hotspot-map)
- [Technologies Used](#technologies-used)
- [Research Paper](#research-paper)
- [License](#license)

## Project Overview

Financial fraud in banking costs the industry billions of dollars annually. This project builds a machine learning-based fraud detection system that analyzes transaction patterns across geographic locations, customer demographics, and behavioral indicators.

**Contributions:**
1. A multi-criteria fraud labeling methodology for unlabeled transaction datasets
2. Geographic fraud hotspot analysis identifying high-risk U.S. cities
3. Comparative evaluation of three ML classifiers with class imbalance handling
4. Interactive fraud hotspot visualization using Folium mapping

## Dataset

**Source:** [Bank Transaction Dataset for Fraud Detection](https://www.kaggle.com/datasets/valakhorasani/bank-transaction-dataset-for-fraud-detection) (Kaggle)  

| Property | Value |
|----------|-------|
| Records | 2,512 transactions |
| Features | 16 columns |
| Cities | 43 U.S. cities |
| Channels | ATM, Branch, Online |
| Occupations | Doctor, Engineer, Student, Retired |

### Features

| Feature | Type | Description |
|---------|------|-------------|
| TransactionAmount | Float | $0.26 - $1,919.11 |
| Location | Categorical | 43 U.S. city names |
| Channel | Categorical | ATM / Branch / Online |
| CustomerAge | Integer | 18 - 80 years |
| CustomerOccupation | Categorical | Doctor, Engineer, Student, Retired |
| TransactionDuration | Integer | Duration in seconds |
| LoginAttempts | Integer | 1 - 5 attempts |
| AccountBalance | Float | Post-transaction balance |
| TransactionDate | Datetime | Transaction timestamp |

### Fraud Label Engineering

The dataset lacks an explicit fraud label. We engineer a multi-criteria anomaly flag — a transaction is labeled **fraudulent** if **any** of the following conditions is met:

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Login Attempts | >= 2 | Repeated failed authentication |
| Transaction Amount | > 95th percentile ($878.18) | Unusually large transaction |
| Transaction Duration | < 5th percentile (24 sec) | Suspiciously fast execution |

**Result:** 350 fraudulent transactions (13.93% fraud rate)

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

### Step 1: Download the IPYNB file

### Step 2: Download the Dataset

The dataset is included. If you need to re-download it:

1. Go to [Kaggle Dataset Page](https://www.kaggle.com/datasets/valakhorasani/bank-transaction-dataset-for-fraud-detection)
2. Download `bank_transactions_data_2.csv`

### Run on Google Colab

1. Upload `Bank_Transaction_Fraud_Detection_Notebook.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Upload `bank_transactions_data_2.csv` to the Colab session
3. Update the CSV path in the first code cell to match your upload location
4. Run All Cells

## Key Findings

### Geographic Fraud Hotspots

| Rank | City | Fraud Rate | Fraud Cases |
|------|------|-----------|-------------|
| 1 | Las Vegas | 21.82% | 12 / 55 |
| 2 | Miami | 20.31% | 13 / 64 |
| 3 | Fort Worth | 20.00% | 14 / 70 |
| 4 | Detroit | 19.05% | 12 / 63 |
| 5 | Austin | 18.64% | 11 / 59 |

**Safest cities:** Fresno (5.0%), Atlanta (8.2%), Mesa (8.2%)

### Statistical Tests Summary

| Test | Variables | p-value | Significant? |
|------|-----------|---------|-------------|
| Two-sample t-test | Amount vs Fraud | 0.7421 | No |
| Chi-square | Channel vs Fraud | 0.6815 | No |
| Chi-square | Occupation vs Fraud | > 0.05 | No |
| Mann-Whitney U | Duration vs Fraud | < 0.05 | **Yes** |

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.5646 | 0.0465 | 0.4167 | 0.0837 | 0.4545 |
| Random Forest | 0.9523 | **1.0000** | 0.7286 | **0.8430** | 0.8575 |
| **Gradient Boosting** | **0.9583** | 0.9623 | 0.7286 | 0.8293 | **0.8840** |

**Best model:** Gradient Boosting (AUC-ROC = 0.8840, Accuracy = 95.83%)  
**Highest precision:** Random Forest (1.0 — zero false positives)

## Fraud Hotspot Map

The pipeline generates an interactive Folium map saved to `outputs/fraud_hotspot_map.html`. Open it in any browser to explore:

- **Circle markers** colored by fraud risk level (green/yellow/orange/red)
- **Circle size** proportional to fraud case count
- **Heatmap overlay** showing fraud density concentration
- **Click** any city for detailed fraud statistics

## Technologies Used

| Category | Technologies |
|----------|-------------|
| Language | Python 3.8+ |
| Data Processing | pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Folium |
| Machine Learning | scikit-learn (Logistic Regression, Random Forest, Gradient Boosting) |
| Statistics | SciPy (t-test, chi-square, Mann-Whitney U) |
| Notebook | Jupyter / Google Colab |
