"""
Customer Churn Prediction - ML Training Pipeline
=================================================
This script trains multiple ML models and saves results for the dashboard.
"""

import pandas as pd
import numpy as np
import json
import os
import warnings
import pickle
from datetime import datetime

# Preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold

# ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# XGBoost and LightGBM
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not installed. Skipping XGBoost model.")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not installed. Skipping LightGBM model.")

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)

warnings.filterwarnings('ignore')

# UPDATED IMPORTS AND SETUP FOR ROBUST EXECUTION
try:
    import joblib
except ImportError:
    import pickle as joblib

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(os.path.dirname(BASE_DIR), 'Dataset')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Create directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("CUSTOMER CHURN PREDICTION - ML TRAINING PIPELINE")
print("=" * 60)

# =====================================================
# 1. LOAD DATA
# =====================================================
print("\n[1/7] Loading Data...")

train_df = pd.read_csv(os.path.join(DATASET_DIR, 'customer_churn_dataset-training-master.csv'))
test_df = pd.read_csv(os.path.join(DATASET_DIR, 'customer_churn_dataset-testing-master.csv'))

print(f"  Training set: {train_df.shape[0]:,} rows, {train_df.shape[1]} columns")
print(f"  Testing set: {test_df.shape[0]:,} rows, {test_df.shape[1]} columns")

# =====================================================
# 2. DATA EXPLORATION & STATS
# =====================================================
print("\n[2/7] Exploring Data...")

# Basic statistics
data_stats = {
    'training_rows': int(train_df.shape[0]),
    'testing_rows': int(test_df.shape[0]),
    'total_rows': int(train_df.shape[0] + test_df.shape[0]),
    'features': int(train_df.shape[1] - 1),
    'churn_rate_train': float(round(train_df['Churn'].mean() * 100, 2)),
    'churn_rate_test': float(round(test_df['Churn'].mean() * 100, 2)),
}

# Feature distributions for dashboard
feature_stats = {}
numeric_cols = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']
categorical_cols = ['Gender', 'Subscription Type', 'Contract Length']

for col in numeric_cols:
    feature_stats[col] = {
        'type': 'numeric',
        'mean': float(round(train_df[col].mean(), 2)),
        'median': float(round(train_df[col].median(), 2)),
        'std': float(round(train_df[col].std(), 2)),
        'min': float(train_df[col].min()),
        'max': float(train_df[col].max()),
        'churn_mean': float(round(train_df[train_df['Churn'] == 1][col].mean(), 2)),
        'no_churn_mean': float(round(train_df[train_df['Churn'] == 0][col].mean(), 2))
    }

for col in categorical_cols:
    value_counts = train_df[col].value_counts().to_dict()
    churn_by_cat = train_df.groupby(col)['Churn'].mean().to_dict()
    feature_stats[col] = {
        'type': 'categorical',
        'distribution': {k: int(v) for k, v in value_counts.items()},
        'churn_rate': {k: float(round(v * 100, 2)) for k, v in churn_by_cat.items()}
    }

# Age distribution for charts
age_bins = [18, 25, 35, 45, 55, 65]
age_labels = ['18-24', '25-34', '35-44', '45-54', '55-65']
train_df['Age_Group'] = pd.cut(train_df['Age'], bins=age_bins, labels=age_labels, include_lowest=True)
age_churn = train_df.groupby('Age_Group')['Churn'].agg(['sum', 'count']).reset_index()
age_churn['churn_rate'] = (age_churn['sum'] / age_churn['count'] * 100).round(2)
age_distribution = age_churn.to_dict('records')

print(f"  Training churn rate: {data_stats['churn_rate_train']}%")
print(f"  Testing churn rate: {data_stats['churn_rate_test']}%")

# =====================================================
# 3. DATA PREPROCESSING
# =====================================================
print("\n[3/7] Preprocessing Data...")

# Drop CustomerID and Age_Group
train_df = train_df.drop(['CustomerID', 'Age_Group'], axis=1)
test_df = test_df.drop(['CustomerID'], axis=1)

# Handle missing values
print(f"  Missing values in training: {train_df.isnull().sum().sum()}")
print(f"  Missing values in testing: {test_df.isnull().sum().sum()}")

# Drop rows with missing target
train_df = train_df.dropna(subset=['Churn'])
test_df = test_df.dropna(subset=['Churn'])

# Fill missing numeric values with median
for col in numeric_cols:
    median_val = train_df[col].median()
    train_df[col] = train_df[col].fillna(median_val)
    test_df[col] = test_df[col].fillna(median_val)

# Fill missing categorical values with mode
for col in categorical_cols:
    mode_val = train_df[col].mode()[0]
    train_df[col] = train_df[col].fillna(mode_val)
    test_df[col] = test_df[col].fillna(mode_val)

print(f"  After cleaning - Training: {train_df.shape[0]:,} rows, Testing: {test_df.shape[0]:,} rows")

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))
    label_encoders[col] = le
    print(f"  Encoded '{col}': {dict(zip(le.classes_, range(len(le.classes_))))}")

# Split features and target
X_train = train_df.drop('Churn', axis=1)
y_train = train_df['Churn']
X_test = test_df.drop('Churn', axis=1)
y_test = test_df['Churn']

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

feature_names = X_train.columns.tolist()
print(f"  Features: {feature_names}")

# Save scaler and encoders
pickle.dump(scaler, open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'wb'))
pickle.dump(label_encoders, open(os.path.join(MODELS_DIR, 'label_encoders.pkl'), 'wb'))
pickle.dump(feature_names, open(os.path.join(MODELS_DIR, 'feature_names.pkl'), 'wb'))

# =====================================================
# 4. DEFINE MODELS
# =====================================================
print("\n[4/7] Initializing Models...")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    'Naive Bayes': GaussianNB(),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
}

if XGBOOST_AVAILABLE:
    models['XGBoost'] = XGBClassifier(n_estimators=100, max_depth=6, random_state=42, use_label_encoder=False, eval_metric='logloss')

if LIGHTGBM_AVAILABLE:
    models['LightGBM'] = LGBMClassifier(n_estimators=100, max_depth=6, random_state=42, verbose=-1)

print(f"  Total models to train: {len(models)}")

# =====================================================
# 5. TRAIN AND EVALUATE MODELS
# =====================================================
print("\n[5/7] Training Models...")

model_results = {}
all_roc_curves = {}
confusion_matrices = {}
best_model_name = None
best_accuracy = 0

for name, model in models.items():
    print(f"\n  Training {name}...", end=" ")
    
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # ROC Curve data
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Feature importance (if available)
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        feature_importance = dict(zip(feature_names, [float(round(x, 4)) for x in importance]))
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
        feature_importance = dict(zip(feature_names, [float(round(x, 4)) for x in importance]))
    
    # Store results
    model_results[name] = {
        'accuracy': float(round(accuracy * 100, 2)),
        'precision': float(round(precision * 100, 2)),
        'recall': float(round(recall * 100, 2)),
        'f1_score': float(round(f1 * 100, 2)),
        'roc_auc': float(round(roc_auc * 100, 2)),
        'feature_importance': feature_importance
    }
    
    # Sample ROC curve points (reduce for JSON)
    sample_indices = np.linspace(0, len(fpr) - 1, min(100, len(fpr))).astype(int)
    all_roc_curves[name] = {
        'fpr': [float(round(fpr[i], 4)) for i in sample_indices],
        'tpr': [float(round(tpr[i], 4)) for i in sample_indices],
        'auc': float(round(roc_auc * 100, 2))
    }
    
    confusion_matrices[name] = {
        'tn': int(cm[0][0]),
        'fp': int(cm[0][1]),
        'fn': int(cm[1][0]),
        'tp': int(cm[1][1])
    }
    
    # Track best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = name
    
    # Save model
    pickle.dump(model, open(os.path.join(MODELS_DIR, f'{name.replace(" ", "_").lower()}.pkl'), 'wb'))
    
    print(f"Accuracy: {accuracy*100:.2f}%, AUC: {roc_auc*100:.2f}%")

print(f"\n  ✓ Best Model: {best_model_name} (Accuracy: {best_accuracy*100:.2f}%)")

# =====================================================
# 6. GENERATE INSIGHTS
# =====================================================
print("\n[6/7] Generating Insights...")

# Sort models by accuracy
model_ranking = sorted(model_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
model_ranking = [{'rank': i+1, 'name': name, **metrics} for i, (name, metrics) in enumerate(model_ranking)]

# Get top feature importance from best model
best_model = pickle.load(open(os.path.join(MODELS_DIR, f'{best_model_name.replace(" ", "_").lower()}.pkl'), 'rb'))
if hasattr(best_model, 'feature_importances_'):
    importance = best_model.feature_importances_
elif hasattr(best_model, 'coef_'):
    importance = np.abs(best_model.coef_[0])
else:
    importance = np.zeros(len(feature_names))

feature_importance_sorted = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
top_features = [{'feature': f, 'importance': float(round(i * 100, 2))} for f, i in feature_importance_sorted]

# Customer segments analysis
original_train = pd.read_csv(os.path.join(DATASET_DIR, 'customer_churn_dataset-training-master.csv'))
segments = {
    'by_subscription': {},
    'by_contract': {},
    'by_gender': {}
}

for sub_type in ['Basic', 'Standard', 'Premium']:
    subset = original_train[original_train['Subscription Type'] == sub_type]
    segments['by_subscription'][sub_type] = {
        'count': int(len(subset)),
        'churn_rate': float(round(subset['Churn'].mean() * 100, 2)),
        'avg_tenure': float(round(subset['Tenure'].mean(), 1)),
        'avg_spend': float(round(subset['Total Spend'].mean(), 2))
    }

for contract in ['Monthly', 'Quarterly', 'Annual']:
    subset = original_train[original_train['Contract Length'] == contract]
    segments['by_contract'][contract] = {
        'count': int(len(subset)),
        'churn_rate': float(round(subset['Churn'].mean() * 100, 2)),
        'avg_tenure': float(round(subset['Tenure'].mean(), 1)),
        'avg_spend': float(round(subset['Total Spend'].mean(), 2))
    }

for gender in ['Male', 'Female']:
    subset = original_train[original_train['Gender'] == gender]
    segments['by_gender'][gender] = {
        'count': int(len(subset)),
        'churn_rate': float(round(subset['Churn'].mean() * 100, 2)),
        'avg_tenure': float(round(subset['Tenure'].mean(), 1)),
        'avg_spend': float(round(subset['Total Spend'].mean(), 2))
    }

# Churn risk factors
churn_factors = []
churned = original_train[original_train['Churn'] == 1]
not_churned = original_train[original_train['Churn'] == 0]

for col in numeric_cols:
    churned_mean = churned[col].mean()
    not_churned_mean = not_churned[col].mean()
    diff_pct = ((churned_mean - not_churned_mean) / not_churned_mean * 100) if not_churned_mean != 0 else 0
    churn_factors.append({
        'feature': col,
        'churned_avg': float(round(churned_mean, 2)),
        'retained_avg': float(round(not_churned_mean, 2)),
        'difference_pct': float(round(diff_pct, 2))
    })

churn_factors = sorted(churn_factors, key=lambda x: abs(x['difference_pct']), reverse=True)

# =====================================================
# 7. SAVE RESULTS
# =====================================================
print("\n[7/7] Saving Results...")

# Main dashboard data
dashboard_data = {
    'generated_at': datetime.now().isoformat(),
    'data_stats': data_stats,
    'feature_stats': feature_stats,
    'model_results': model_results,
    'model_ranking': model_ranking,
    'best_model': {
        'name': best_model_name,
        'accuracy': float(round(best_accuracy * 100, 2))
    },
    'roc_curves': all_roc_curves,
    'confusion_matrices': confusion_matrices,
    'feature_importance': top_features,
    'customer_segments': segments,
    'churn_factors': churn_factors,
    'age_distribution': [{'Age_Group': str(d['Age_Group']), 'churned': int(d['sum']), 'total': int(d['count']), 'churn_rate': float(d['churn_rate'])} for d in age_distribution]
}

# Save to JSON
with open(os.path.join(RESULTS_DIR, 'dashboard_data.json'), 'w') as f:
    json.dump(dashboard_data, f, indent=2)

print(f"  ✓ Saved dashboard_data.json")
print(f"  ✓ Saved {len(models)} trained models")

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print(f"\nResults saved to: {RESULTS_DIR}")
print(f"Models saved to: {MODELS_DIR}")
print(f"\nBest Model: {best_model_name}")
print(f"Best Accuracy: {best_accuracy * 100:.2f}%")
print("\nRun the Node.js backend to view the dashboard!")
