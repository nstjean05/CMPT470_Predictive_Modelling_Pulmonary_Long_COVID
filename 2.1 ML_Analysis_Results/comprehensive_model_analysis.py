"""
Comprehensive Machine Learning Analysis for Long COVID Prediction
Trains Random Forest, Gradient Boosting, and Logistic Regression models
(UPDATED: Added Data Cleaning to handle 'NaN;' strings)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.feature_selection import SelectFromModel
from sklearn.utils.class_weight import compute_sample_weight
import warnings
warnings.filterwarnings('ignore')

# ====================
# DATA LOADING & PREP
# ====================
print("="*70)
print("LONG COVID PREDICTION: COMPREHENSIVE ML ANALYSIS")
print("="*70)

print("\n[1/4] Loading and preparing data...")

# Load the file
df_biomarkers = pd.read_excel("Sheet1.xlsx", sheet_name=0)

# Target and metadata columns
target_col = 'x0_Censor_Complete'
metadata_cols = ['x0_LC_ID', 'x0_Censor_Complete', 'x0_Censor_Cohort_ID', 
                 'x0_Patient_Cluster_Label', 'x0_Patient_Cluster',
                 'x0_Censor_Oral_Steroid', 'x0_Censor_Pit_Adre_Dysfunction',
                 'x0_Censor_Pregnancy']

# Separate features and target
X = df_biomarkers.drop(columns=[c for c in metadata_cols if c in df_biomarkers.columns])
y = df_biomarkers[target_col]

# --- DATA CLEANING STEP (FIX FOR 'NaN;' ERROR) ---
print("✓ Cleaning data (converting strings like 'NaN;' to numeric)...")
# errors='coerce' turns non-numeric strings (like 'NaN;') into actual NaNs
X = X.apply(pd.to_numeric, errors='coerce')

# Handle the resulting missing values (Fill with 0 or Median)
if X.isnull().values.any():
    missing_count = X.isnull().sum().sum()
    print(f"⚠ Fixed {missing_count} non-numeric/missing values by setting them to 0.")
    X = X.fillna(0)
# ------------------------------------------------

feature_names = X.columns
print(f"✓ Loaded {X.shape[1]:,} numeric biomarker features")
print(f"✓ Dataset shape: {X.shape}")
print(f"✓ Long COVID cases: {(y==1).sum()} | Non-cases: {(y==0).sum()}")

# Train/Test Split (Stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ====================
# FEATURE SELECTION
# ====================
print("\n[2/4] Performing L1 Feature Selection (LASSO)...")
selector = SelectFromModel(
    LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', C=0.1, random_state=42)
)

X_train_sel = selector.fit_transform(X_train, y_train)
X_test_sel = selector.transform(X_test)

selected_mask = selector.get_support()
selected_features = feature_names[selected_mask]
print(f"✓ Reduced feature space to {len(selected_features)} highly predictive features")

sample_weights_train = compute_sample_weight(class_weight='balanced', y=y_train)

# ====================
# MODEL TRAINING
# ====================
print("\n[3/4] Training and Evaluating Models...")

results_list = []

rf_model = RandomForestClassifier(
    n_estimators=100, class_weight='balanced', max_depth=5, min_samples_leaf=3, random_state=42
)
rf_model.fit(X_train_sel, y_train)
rf_pred = rf_model.predict(X_test_sel)
rf_pred_proba = rf_model.predict_proba(X_test_sel)[:, 1]

gb_model = GradientBoostingClassifier(
    n_estimators=100, max_depth=3, min_samples_leaf=3, random_state=42
)
gb_model.fit(X_train_sel, y_train, sample_weight=sample_weights_train)
gb_pred = gb_model.predict(X_test_sel)
gb_pred_proba = gb_model.predict_proba(X_test_sel)[:, 1]

lr_model = LogisticRegression(
    max_iter=1000, class_weight='balanced', C=0.1, random_state=42
)
lr_model.fit(X_train_sel, y_train)
lr_pred = lr_model.predict(X_test_sel)
lr_pred_proba = lr_model.predict_proba(X_test_sel)[:, 1]

# Cross Validation (Using 3 folds due to tiny minority class size of 5)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
rf_cv = cross_val_score(rf_model, X_train_sel, y_train, cv=cv, scoring='roc_auc')
gb_cv = cross_val_score(gb_model, X_train_sel, y_train, cv=cv, scoring='roc_auc') 
lr_cv = cross_val_score(lr_model, X_train_sel, y_train, cv=cv, scoring='roc_auc')

models = [
    ('Random Forest', rf_pred, rf_pred_proba, rf_cv, rf_model),
    ('Gradient Boosting', gb_pred, gb_pred_proba, gb_cv, gb_model),
    ('Logistic Regression', lr_pred, lr_pred_proba, lr_cv, lr_model)
]

for name, pred, proba, cv_scores, model in models:
    results_list.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, pred),
        'Precision': precision_score(y_test, pred, zero_division=0),
        'Recall': recall_score(y_test, pred, zero_division=0),
        'F1-Score': f1_score(y_test, pred, zero_division=0),
        'ROC-AUC (Test)': roc_auc_score(y_test, proba) if len(np.unique(y_test)) > 1 else np.nan,
        'ROC-AUC (CV Mean)': cv_scores.mean(),
        'predictions': pred
    })

results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'predictions'} for r in results_list])
results_df.to_csv('model_comparison_results.csv', index=False)

# ====================
# FEATURE IMPORTANCE
# ====================
rf_importance = pd.DataFrame({'Feature': selected_features, 'Importance': rf_model.feature_importances_}).sort_values('Importance', ascending=False)
gb_importance = pd.DataFrame({'Feature': selected_features, 'Importance': gb_model.feature_importances_}).sort_values('Importance', ascending=False)
lr_importance = pd.DataFrame({'Feature': selected_features, 'Coefficient': lr_model.coef_[0]}).sort_values('Coefficient', ascending=False)

rf_importance.to_csv('rf_feature_importance.csv', index=False)
gb_importance.to_csv('gb_feature_importance.csv', index=False)
lr_importance.to_csv('lr_feature_importance.csv', index=False)

print("\n" + "="*70)
print("COMPREHENSIVE MODEL COMPARISON RESULTS")
print("="*70)
print(results_df.to_string(index=False))

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)