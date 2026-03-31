"""
Bank Transaction Fraud Detection Across U.S. Cities
====================================================
Main pipeline script that runs the full data science workflow:
  1. Data Loading & Inspection
  2. Data Cleaning & Preprocessing
  3. Exploratory Data Analysis (EDA) with visualizations
  4. Statistical Hypothesis Testing
  5. Feature Engineering
  6. ML Modeling (Logistic Regression, Random Forest, Gradient Boosting)
  7. Evaluation & Fraud Hotspot Map Generation

Usage:
    python src/main.py

Outputs saved to: outputs/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import sys

from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score,
                             precision_score, recall_score, f1_score)

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

# ─── Paths ───────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "bank_transactions_data_2.csv")
IMG_DIR = os.path.join(ROOT, "images")
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


def save_fig(name):
    """Save current figure to images/ directory."""
    path = os.path.join(IMG_DIR, f"{name}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: images/{name}.png")


# ═══════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("  STEP 1: Data Loading & Inspection")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"Dataset Shape: {df.shape}")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print(f"\nColumn Types:\n{df.dtypes}")
print(f"\nMissing Values: {df.isnull().sum().sum()}")
print(f"\nUnique values per column:")
for col in df.columns:
    print(f"  {col:30s}: {df[col].nunique():>6}")


# ═══════════════════════════════════════════════════════════════
# 2. DATA CLEANING & PREPROCESSING
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 2: Data Cleaning & Preprocessing")
print("=" * 60)

df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
df['PreviousTransactionDate'] = pd.to_datetime(df['PreviousTransactionDate'])
print(f"Duplicate rows: {df.duplicated().sum()}")

# Multi-criteria fraud label engineering
q95_amount = df['TransactionAmount'].quantile(0.95)
q05_duration = df['TransactionDuration'].quantile(0.05)

print(f"\nFraud Labeling Thresholds:")
print(f"  High Amount (>95th pctl):   > ${q95_amount:.2f}")
print(f"  Short Duration (<5th pctl): < {q05_duration} seconds")
print(f"  High Login Attempts:        >= 2")

df['IsFraud'] = (
    (df['LoginAttempts'] >= 2) |
    (df['TransactionAmount'] > q95_amount) |
    (df['TransactionDuration'] < q05_duration)
).astype(int)

print(f"\nFraud Distribution:\n{df['IsFraud'].value_counts()}")
print(f"Fraud Rate: {df['IsFraud'].mean()*100:.2f}%")

# Time features
df['TransactionHour'] = df['TransactionDate'].dt.hour
df['TransactionDay'] = df['TransactionDate'].dt.dayofweek
df['TransactionMonth'] = df['TransactionDate'].dt.month
df['DaysSincePrevTxn'] = (df['PreviousTransactionDate'] - df['TransactionDate']).dt.days.abs()
print("Engineered features: TransactionHour, TransactionDay, TransactionMonth, DaysSincePrevTxn")


# ═══════════════════════════════════════════════════════════════
# 3. EXPLORATORY DATA ANALYSIS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 3: Exploratory Data Analysis")
print("=" * 60)

# 3.1 Fraud Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
labels = ['Legitimate', 'Fraudulent']
sizes = df['IsFraud'].value_counts().values
colors = ['#2ecc71', '#e74c3c']
axes[0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 13}, explode=(0, 0.08))
axes[0].set_title('Fraud vs Legitimate Transactions', fontsize=14, fontweight='bold')
sns.countplot(x='IsFraud', data=df, palette=colors, ax=axes[1])
axes[1].set_xticklabels(['Legitimate (0)', 'Fraudulent (1)'])
axes[1].set_title('Transaction Count by Fraud Status', fontsize=14, fontweight='bold')
for p in axes[1].patches:
    axes[1].annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width()/2., p.get_height()),
                     ha='center', va='bottom', fontsize=13, fontweight='bold')
plt.tight_layout()
save_fig("01_fraud_distribution")

# 3.2 Fraud by Location
fraud_by_location = df.groupby('Location')['IsFraud'].agg(['sum', 'count', 'mean'])
fraud_by_location.columns = ['FraudCount', 'TotalTxns', 'FraudRate']
fraud_by_location = fraud_by_location.sort_values('FraudRate', ascending=False).head(15)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fraud_by_location['FraudCount'].plot(kind='barh', ax=axes[0], color='#e74c3c', edgecolor='black')
axes[0].set_title('Fraud Count by City (Top 15)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Number of Fraudulent Transactions')
fraud_by_location['FraudRate'].plot(kind='barh', ax=axes[1], color='#e67e22', edgecolor='black')
axes[1].set_title('Fraud Rate by City (Top 15)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Fraud Rate')
plt.tight_layout()
save_fig("02_fraud_by_location")

# 3.3 Fraud by Channel & Transaction Type
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
df.groupby('Channel')['IsFraud'].mean().sort_values(ascending=False).plot(
    kind='bar', ax=axes[0], color=['#3498db', '#e74c3c', '#2ecc71'], edgecolor='black')
axes[0].set_title('Fraud Rate by Channel', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Fraud Rate')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
df.groupby('TransactionType')['IsFraud'].mean().sort_values(ascending=False).plot(
    kind='bar', ax=axes[1], color=['#9b59b6', '#1abc9c'], edgecolor='black')
axes[1].set_title('Fraud Rate by Transaction Type', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Fraud Rate')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
plt.tight_layout()
save_fig("03_fraud_by_channel_type")

# 3.4 Fraud by Occupation & Age
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
df.groupby('CustomerOccupation')['IsFraud'].mean().sort_values(ascending=False).plot(
    kind='bar', ax=axes[0], color='#e67e22', edgecolor='black')
axes[0].set_title('Fraud Rate by Occupation', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Fraud Rate')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
axes[1].hist(df[df['IsFraud']==0]['CustomerAge'], bins=20, alpha=0.6, label='Legitimate', color='#2ecc71')
axes[1].hist(df[df['IsFraud']==1]['CustomerAge'], bins=20, alpha=0.6, label='Fraudulent', color='#e74c3c')
axes[1].set_title('Age Distribution: Fraud vs Legitimate', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Customer Age')
axes[1].legend()
plt.tight_layout()
save_fig("04_fraud_by_occupation_age")

# 3.5 Transaction Amount
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(x='IsFraud', y='TransactionAmount', data=df, palette=['#2ecc71', '#e74c3c'], ax=axes[0])
axes[0].set_xticklabels(['Legitimate', 'Fraudulent'])
axes[0].set_title('Transaction Amount: Fraud vs Legitimate', fontsize=14, fontweight='bold')
df[df['IsFraud']==0]['TransactionAmount'].plot(kind='kde', ax=axes[1], label='Legitimate', color='#2ecc71', linewidth=2)
df[df['IsFraud']==1]['TransactionAmount'].plot(kind='kde', ax=axes[1], label='Fraudulent', color='#e74c3c', linewidth=2)
axes[1].set_title('Transaction Amount Distribution', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Transaction Amount ($)')
axes[1].legend()
plt.tight_layout()
save_fig("05_transaction_amount")

# 3.6 Temporal Patterns
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
df.groupby('TransactionHour')['IsFraud'].mean().plot(kind='bar', ax=axes[0], color='#3498db', edgecolor='black')
axes[0].set_title('Fraud Rate by Hour of Day', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Hour')
axes[0].set_ylabel('Fraud Rate')
day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
day_fraud = df.groupby('TransactionDay')['IsFraud'].mean()
day_fraud.index = [day_names[i] for i in day_fraud.index]
day_fraud.plot(kind='bar', ax=axes[1], color='#9b59b6', edgecolor='black')
axes[1].set_title('Fraud Rate by Day of Week', fontsize=14, fontweight='bold')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
plt.tight_layout()
save_fig("06_temporal_patterns")

# 3.7 Correlation Heatmap
numeric_cols = ['TransactionAmount', 'CustomerAge', 'TransactionDuration',
                'LoginAttempts', 'AccountBalance', 'TransactionHour',
                'TransactionDay', 'DaysSincePrevTxn', 'IsFraud']
plt.figure(figsize=(10, 8))
corr = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, linewidths=0.5, square=True)
plt.title('Correlation Heatmap of Numerical Features', fontsize=14, fontweight='bold')
plt.tight_layout()
save_fig("07_correlation_heatmap")

# 3.8 Fraud Hotspot Map
print("\n  Generating fraud hotspot map...")
try:
    import folium
    from folium.plugins import HeatMap

    city_coords = {
        'Las Vegas':(36.17,-115.14), 'Miami':(25.76,-80.19), 'Fort Worth':(32.75,-97.33),
        'Detroit':(42.33,-83.05), 'Austin':(30.27,-97.74), 'Columbus':(39.96,-82.99),
        'El Paso':(31.76,-106.44), 'Sacramento':(38.58,-121.49), 'San Diego':(32.72,-117.16),
        'Washington':(38.91,-77.04), 'Philadelphia':(39.95,-75.17), 'Virginia Beach':(36.85,-75.98),
        'Dallas':(32.78,-96.80), 'Memphis':(35.15,-90.05), 'New York':(40.71,-74.01),
        'San Jose':(37.34,-121.89), 'Colorado Springs':(38.83,-104.82), 'Oklahoma City':(35.47,-97.52),
        'Albuquerque':(35.08,-106.65), 'Nashville':(36.16,-86.78), 'Phoenix':(33.45,-112.07),
        'Houston':(29.76,-95.37), 'Louisville':(38.25,-85.76), 'San Antonio':(29.42,-98.49),
        'Kansas City':(39.10,-94.58), 'Milwaukee':(43.04,-87.91), 'Omaha':(41.26,-95.94),
        'Tucson':(32.22,-110.93), 'Portland':(45.52,-122.68), 'Raleigh':(35.78,-78.64),
        'Charlotte':(35.23,-80.84), 'Baltimore':(39.29,-76.61), 'Jacksonville':(30.33,-81.66),
        'Seattle':(47.61,-122.33), 'Boston':(42.36,-71.06), 'Denver':(39.74,-104.99),
        'Indianapolis':(39.77,-86.16), 'Los Angeles':(34.05,-118.24), 'Chicago':(41.88,-87.63),
        'San Francisco':(37.77,-122.42), 'Atlanta':(33.75,-84.39), 'Mesa':(33.41,-111.83),
        'Fresno':(36.74,-119.77)
    }

    city_fraud = df.groupby('Location').agg(
        TotalTxns=('IsFraud', 'count'), FraudCount=('IsFraud', 'sum'),
        FraudRate=('IsFraud', 'mean'), AvgAmount=('TransactionAmount', 'mean')
    ).reset_index()

    def get_color(rate):
        if rate >= 0.18: return '#e74c3c'
        elif rate >= 0.14: return '#e67e22'
        elif rate >= 0.10: return '#f1c40f'
        else: return '#2ecc71'

    m = folium.Map(location=[39.5, -98.35], zoom_start=4, tiles='CartoDB positron')
    for _, row in city_fraud.iterrows():
        city = row['Location']
        if city in city_coords:
            lat, lng = city_coords[city]
            color = get_color(row['FraudRate'])
            popup_html = (f"<b>{city}</b><br>Fraud: {row['FraudCount']}/{row['TotalTxns']}"
                          f"<br>Rate: {row['FraudRate']*100:.1f}%")
            folium.CircleMarker(
                location=[lat, lng], radius=max(5, row['FraudCount'] * 1.5),
                popup=popup_html, tooltip=f"{city}: {row['FraudRate']*100:.1f}%",
                color=color, fill=True, fill_color=color, fill_opacity=0.7
            ).add_to(m)

    heat_data = [[city_coords[r['Location']][0], city_coords[r['Location']][1], r['FraudCount']]
                 for _, r in city_fraud.iterrows() if r['Location'] in city_coords]
    HeatMap(heat_data, radius=35, blur=25).add_to(m)

    map_path = os.path.join(OUT_DIR, "fraud_hotspot_map.html")
    m.save(map_path)
    print(f"  Saved: outputs/fraud_hotspot_map.html")
except ImportError:
    print("  [SKIP] folium not installed. Run: pip install folium")


# ═══════════════════════════════════════════════════════════════
# 4. STATISTICAL TESTING
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 4: Statistical Hypothesis Testing")
print("=" * 60)

fraud = df[df['IsFraud'] == 1]
legit = df[df['IsFraud'] == 0]

# Test 1: t-test on Transaction Amount
t_stat, p_val = stats.ttest_ind(fraud['TransactionAmount'], legit['TransactionAmount'])
print(f"\nTest 1: Two-sample t-test (Transaction Amount)")
print(f"  t-statistic: {t_stat:.4f}, p-value: {p_val:.4f}")
print(f"  Result: {'Significant' if p_val < 0.05 else 'Not significant'}")

# Test 2: Chi-square on Channel
chi2, p_val2, dof, _ = stats.chi2_contingency(pd.crosstab(df['Channel'], df['IsFraud']))
print(f"\nTest 2: Chi-square (Channel vs Fraud)")
print(f"  Chi2: {chi2:.4f}, p-value: {p_val2:.4f}, dof: {dof}")
print(f"  Result: {'Significant' if p_val2 < 0.05 else 'Not significant'}")

# Test 3: Chi-square on Occupation
chi2_3, p_val3, dof3, _ = stats.chi2_contingency(pd.crosstab(df['CustomerOccupation'], df['IsFraud']))
print(f"\nTest 3: Chi-square (Occupation vs Fraud)")
print(f"  Chi2: {chi2_3:.4f}, p-value: {p_val3:.4f}, dof: {dof3}")
print(f"  Result: {'Significant' if p_val3 < 0.05 else 'Not significant'}")

# Test 4: Mann-Whitney U on Duration
u_stat, p_val4 = stats.mannwhitneyu(fraud['TransactionDuration'], legit['TransactionDuration'])
print(f"\nTest 4: Mann-Whitney U (Transaction Duration)")
print(f"  U-statistic: {u_stat:.4f}, p-value: {p_val4:.4f}")
print(f"  Result: {'Significant' if p_val4 < 0.05 else 'Not significant'}")


# ═══════════════════════════════════════════════════════════════
# 5. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 5: Feature Engineering & Train-Test Split")
print("=" * 60)

feature_cols = ['TransactionAmount', 'TransactionType', 'Location', 'Channel',
                'CustomerAge', 'CustomerOccupation', 'TransactionDuration',
                'AccountBalance', 'TransactionHour', 'TransactionDay',
                'TransactionMonth', 'DaysSincePrevTxn']

df_model = df[feature_cols + ['IsFraud']].copy()

label_encoders = {}
cat_cols = ['TransactionType', 'Location', 'Channel', 'CustomerOccupation']
for col in cat_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    label_encoders[col] = le

X = df_model.drop('IsFraud', axis=1)
y = df_model['IsFraud']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
num_cols = ['TransactionAmount', 'CustomerAge', 'TransactionDuration',
            'AccountBalance', 'TransactionHour', 'TransactionDay',
            'TransactionMonth', 'DaysSincePrevTxn']
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set:     {X_test.shape[0]} samples")
print(f"Features:     {X_train.shape[1]}")
print(f"Train fraud rate: {y_train.mean()*100:.2f}%")
print(f"Test fraud rate:  {y_test.mean()*100:.2f}%")


# ═══════════════════════════════════════════════════════════════
# 6. ML MODELING
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 6: Machine Learning Modeling")
print("=" * 60)

models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
}

results = {}
for name, model in models.items():
    print(f"\n{'─' * 50}")
    print(f"  {name}")
    print(f"{'─' * 50}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    results[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec,
                     'F1-Score': f1, 'AUC-ROC': auc, 'y_prob': y_prob, 'y_pred': y_pred}

    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))


# ═══════════════════════════════════════════════════════════════
# 7. EVALUATION PLOTS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 7: Generating Evaluation Plots")
print("=" * 60)

# Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, (name, res) in enumerate(results.items()):
    cm = confusion_matrix(y_test, res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Legitimate', 'Fraud'], yticklabels=['Legitimate', 'Fraud'])
    axes[idx].set_title(f'{name}', fontsize=13, fontweight='bold')
    axes[idx].set_ylabel('Actual')
    axes[idx].set_xlabel('Predicted')
plt.suptitle('Confusion Matrices', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
save_fig("08_confusion_matrices")

# ROC Curves
plt.figure(figsize=(10, 7))
colors_roc = ['#3498db', '#e74c3c', '#2ecc71']
for idx, (name, res) in enumerate(results.items()):
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    plt.plot(fpr, tpr, color=colors_roc[idx], linewidth=2,
             label=f"{name} (AUC = {res['AUC-ROC']:.4f})")
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
plt.xlabel('False Positive Rate', fontsize=13)
plt.ylabel('True Positive Rate', fontsize=13)
plt.title('ROC Curves — Model Comparison', fontsize=15, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
save_fig("09_roc_curves")

# Feature Importance
rf_model = models['Random Forest']
importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
importances = importances.sort_values(ascending=True)
plt.figure(figsize=(10, 7))
importances.plot(kind='barh', color='#3498db', edgecolor='black')
plt.title('Feature Importance — Random Forest', fontsize=15, fontweight='bold')
plt.xlabel('Importance Score')
plt.tight_layout()
save_fig("10_feature_importance")

# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  FINAL SUMMARY")
print("=" * 60)

summary = pd.DataFrame({name: {k: v for k, v in res.items() if k not in ['y_prob', 'y_pred']}
                        for name, res in results.items()}).T
print(summary.round(4).to_string())

best = summary['AUC-ROC'].idxmax()
print(f"\nBest Model (AUC-ROC): {best} with AUC = {summary.loc[best, 'AUC-ROC']:.4f}")
print(f"\nAll outputs saved to: images/ and outputs/")
print("Pipeline complete!")
