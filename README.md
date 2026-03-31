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

## Project Structure

```
fraud-detection-project/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── LICENSE                    # MIT License
├── .gitignore                 # Git ignore rules
│
├── data/
│   └── bank_transactions_data_2.csv   # Raw dataset (from Kaggle)
│
├── notebooks/
│   └── Bank_Transaction_Fraud_Detection_Notebook.ipynb  # Jupyter notebook (full pipeline)
│
├── src/
│   └── main.py                # Standalone Python pipeline script
│
├── images/                    # Generated EDA plots (auto-created by pipeline)
│   ├── 01_fraud_distribution.png
│   ├── 02_fraud_by_location.png
│   ├── 03_fraud_by_channel_type.png
│   ├── 04_fraud_by_occupation_age.png
│   ├── 05_transaction_amount.png
│   ├── 06_temporal_patterns.png
│   ├── 07_correlation_heatmap.png
│   ├── 08_confusion_matrices.png
│   ├── 09_roc_curves.png
│   └── 10_feature_importance.png
│
└── outputs/
    ├── Fraud_Detection_IEEE_Paper.docx  # IEEE-format research paper
    └── fraud_hotspot_map.html           # Interactive Folium map (auto-generated)
```

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/<your-username>/fraud-detection-bank-transactions.git
cd fraud-detection-bank-transactions
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download the Dataset

The dataset is included in `data/`. If you need to re-download it:

1. Go to [Kaggle Dataset Page](https://www.kaggle.com/datasets/valakhorasani/bank-transaction-dataset-for-fraud-detection)
2. Download `bank_transactions_data_2.csv`
3. Place it in the `data/` directory

## How to Run

### Option A: Run the Python Pipeline (Recommended)

```bash
python src/main.py
```

This runs the entire 7-stage pipeline and saves all plots to `images/` and the fraud hotspot map to `outputs/fraud_hotspot_map.html`.

### Option B: Run the Jupyter Notebook

```bash
# Install Jupyter if not already installed
pip install jupyter

# Launch notebook
jupyter notebook notebooks/Bank_Transaction_Fraud_Detection_Notebook.ipynb
```

### Option C: Run on Google Colab

1. Upload `Bank_Transaction_Fraud_Detection_Notebook.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Upload `bank_transactions_data_2.csv` to the Colab session
3. Add this cell at the top and run it:
   ```python
   !pip install folium
   ```
4. Update the CSV path in the first code cell to match your upload location
5. Run All Cells

## Pipeline Stages

| Stage | Description | Key Outputs |
|-------|-------------|-------------|
| 1. Data Loading | Load CSV, inspect shape/types/nulls | Dataset overview |
| 2. Data Cleaning | Datetime conversion, fraud label engineering | IsFraud column (13.93% fraud rate) |
| 3. EDA | 8 visualization sections including fraud hotspot map | 10 plots + interactive map |
| 4. Statistical Testing | 4 hypothesis tests (t-test, chi-square, Mann-Whitney) | p-values and conclusions |
| 5. Feature Engineering | Label encoding, scaling, temporal features, 80/20 split | 12-feature training set |
| 6. ML Modeling | Logistic Regression, Random Forest, Gradient Boosting | Model predictions |
| 7. Evaluation | Confusion matrices, ROC curves, feature importance | Performance comparison |

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

## Research Paper

The IEEE-formatted research paper is available at `outputs/Fraud_Detection_IEEE_Paper.docx`. It follows the standard structure:

1. Introduction
2. Related Work
3. Dataset and Methodology
4. Results and Analysis
5. Discussion
6. Limitations
7. Conclusion and Future Work

## Future Work

- Apply SMOTE/ADASYN for advanced oversampling
- Evaluate deep learning models (LSTM for sequential transactions)
- Incorporate IP address geolocation cross-referencing
- Test on larger real-world datasets with confirmed fraud labels
- Develop a real-time fraud scoring API

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements

- Dataset by [Vala Khorasani](https://www.kaggle.com/datasets/valakhorasani/bank-transaction-dataset-for-fraud-detection) (Kaggle, Apache 2.0)
- Built with [scikit-learn](https://scikit-learn.org/), [Folium](https://python-visualization.github.io/folium/), and [Seaborn](https://seaborn.pydata.org/)
