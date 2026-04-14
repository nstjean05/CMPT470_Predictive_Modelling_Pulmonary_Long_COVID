"""
Comprehensive Machine Learning Analysis for Long COVID Prediction
Trains Random Forest, Gradient Boosting, and Logistic Regression models
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# ====================
# DATA LOADING & PREP
# ====================
print("="*70)
print("LONG COVID PREDICTION: COMPREHENSIVE ML ANALYSIS")
print("="*70)

print("\n[1/4] Loading and preparing data...")

df_biomarkers = pd.read_excel("Sheet1.xlsx", sheet_name=0)
df_clinical = pd.read_excel("Sheet2.xlsx", sheet_name=1)

# Target and metadata columns
target_col = 'x0_Censor_Complete'
metadata_cols = ['x0_LC_ID', 'x0_Censor_Complete', 'x0_Censor_Cohort_ID', 
                 'x0_Patient_Cluster_Label', 'x0_Patient_Cluster',
                 'x0_Censor_Oral_Steroid', 'x0_Censor_Pit_Adre_Dysfunction',
                 'x0_Censor_Pregnancy', 'x0_Censor_Active_Chemotherapy',
                 'x0_Censor_Active_Malignancy']

feature_cols = [col for col in df_biomarkers.columns if col not in metadata_cols]

# Filter numeric features
numeric_features = []
for col in feature_cols:
    try:
        col_data = df_biomarkers[col].copy()
        col_data = col_data.replace(['NaN;', 'NaN', 'nan', '', ' '], np.nan)
        numeric_col = pd.to_numeric(col_data, errors='coerce')
        conversion_rate = numeric_col.notna().sum() / len(numeric_col)
        if conversion_rate > 0.5:
            numeric_features.append(col)
    except:
        pass

print(f"✓ Loaded {len(numeric_features):,} numeric biomarker features")

# Prepare features and target
X = df_biomarkers[numeric_features].copy()
y = df_biomarkers[target_col].copy()

# Clean data
for col in X.columns:
    X[col] = X[col].apply(lambda x: np.nan if isinstance(x, str) and ('NaN' in x or x.strip() == '') else x)
    X[col] = pd.to_numeric(X[col], errors='coerce')

X = X.astype(float)

# Convert to binary target
y_binary = (y > 0).astype(int)

# Fill missing values
for col in X.columns:
    if X[col].isnull().sum() > 0:
        X[col].fillna(X[col].median(), inplace=True)

# Remove rows with missing target
valid_idx = ~y_binary.isnull()
X = X[valid_idx]
y_binary = y_binary[valid_idx]

print(f"✓ Dataset shape: {X.shape}")
print(f"✓ Long COVID cases: {y_binary.sum()} | Non-cases: {(y_binary==0).sum()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# Apply SMOTE to training set only
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"✓ Train set: {X_train_smote.shape[0]} samples (SMOTE applied)")
print(f"✓ Test set: {X_test.shape[0]} samples")

# Scale features for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

# ====================
# MODEL 1: RANDOM FOREST
# ====================
print("\n[2/4] Training Random Forest Classifier...")

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

rf_model.fit(X_train_smote, y_train_smote)
y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

rf_results = {
    'model': 'Random Forest',
    'accuracy': accuracy_score(y_test, y_pred_rf),
    'precision': precision_score(y_test, y_pred_rf, zero_division=0),
    'recall': recall_score(y_test, y_pred_rf, zero_division=0),
    'f1': f1_score(y_test, y_pred_rf, zero_division=0),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_rf),
    'predictions': y_pred_rf,
    'probabilities': y_pred_proba_rf
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_cv = cross_val_score(rf_model, X, y_binary, cv=cv, scoring='roc_auc')
print(f"✓ Random Forest trained")
print(f"  - Test ROC-AUC: {rf_results['roc_auc']:.4f}")
print(f"  - CV ROC-AUC: {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")

# ====================
# MODEL 2: GRADIENT BOOSTING
# ====================
print("\n[3/4] Training Gradient Boosting Classifier...")

gb_model = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8,
    random_state=42
)

gb_model.fit(X_train_smote, y_train_smote)
y_pred_gb = gb_model.predict(X_test)
y_pred_proba_gb = gb_model.predict_proba(X_test)[:, 1]

gb_results = {
    'model': 'Gradient Boosting',
    'accuracy': accuracy_score(y_test, y_pred_gb),
    'precision': precision_score(y_test, y_pred_gb, zero_division=0),
    'recall': recall_score(y_test, y_pred_gb, zero_division=0),
    'f1': f1_score(y_test, y_pred_gb, zero_division=0),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_gb),
    'predictions': y_pred_gb,
    'probabilities': y_pred_proba_gb
}

gb_cv = cross_val_score(gb_model, X, y_binary, cv=cv, scoring='roc_auc')
print(f"✓ Gradient Boosting trained")
print(f"  - Test ROC-AUC: {gb_results['roc_auc']:.4f}")
print(f"  - CV ROC-AUC: {gb_cv.mean():.4f} ± {gb_cv.std():.4f}")

# ====================
# MODEL 3: LOGISTIC REGRESSION
# ====================
print("\n[4/4] Training Logistic Regression...")

lr_model = LogisticRegression(
    max_iter=5000,
    random_state=42,
    class_weight='balanced'
)

lr_model.fit(X_train_scaled, y_train_smote)
y_pred_lr = lr_model.predict(X_test_scaled)
y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

lr_results = {
    'model': 'Logistic Regression',
    'accuracy': accuracy_score(y_test, y_pred_lr),
    'precision': precision_score(y_test, y_pred_lr, zero_division=0),
    'recall': recall_score(y_test, y_pred_lr, zero_division=0),
    'f1': f1_score(y_test, y_pred_lr, zero_division=0),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_lr),
    'predictions': y_pred_lr,
    'probabilities': y_pred_proba_lr,
    'coefficients': lr_model.coef_[0]
}

lr_cv = cross_val_score(lr_model, X, y_binary, cv=cv, scoring='roc_auc')
print(f"✓ Logistic Regression trained")
print(f"  - Test ROC-AUC: {lr_results['roc_auc']:.4f}")
print(f"  - CV ROC-AUC: {lr_cv.mean():.4f} ± {lr_cv.std():.4f}")

# ====================
# COMPREHENSIVE RESULTS
# ====================
print("\n" + "="*70)
print("COMPREHENSIVE MODEL COMPARISON RESULTS")
print("="*70)

results_list = [rf_results, gb_results, lr_results]
results_df = pd.DataFrame([
    {
        'Model': r['model'],
        'Accuracy': f"{r['accuracy']:.4f}",
        'Precision': f"{r['precision']:.4f}",
        'Recall': f"{r['recall']:.4f}",
        'F1-Score': f"{r['f1']:.4f}",
        'ROC-AUC': f"{r['roc_auc']:.4f}"
    }
    for r in results_list
])

print("\nTest Set Performance Metrics:")
print(results_df.to_string(index=False))

# Save detailed results
results_df.to_csv('model_comparison_results.csv', index=False)
print("✓ Saved model_comparison_results.csv")

# Feature importance from Random Forest and Gradient Boosting
print("\n" + "="*70)
print("TOP 15 BIOMARKERS - RANDOM FOREST")
print("="*70)

rf_importance = pd.DataFrame({
    'Feature': numeric_features,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(rf_importance.head(15).to_string(index=False))
rf_importance.to_csv('rf_feature_importance.csv', index=False)

print("\n" + "="*70)
print("TOP 15 BIOMARKERS - GRADIENT BOOSTING")
print("="*70)

gb_importance = pd.DataFrame({
    'Feature': numeric_features,
    'Importance': gb_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(gb_importance.head(15).to_string(index=False))
gb_importance.to_csv('gb_feature_importance.csv', index=False)

print("\n" + "="*70)
print("TOP 15 BIOMARKERS - LOGISTIC REGRESSION (by absolute coefficient)")
print("="*70)

lr_importance = pd.DataFrame({
    'Feature': numeric_features,
    'Coefficient': lr_model.coef_[0],
    'Abs_Coefficient': np.abs(lr_model.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

print(lr_importance[['Feature', 'Coefficient']].head(15).to_string(index=False))
lr_importance.to_csv('lr_feature_importance.csv', index=False)

# ====================
# DETAILED EVALUATION
# ====================
print("\n" + "="*70)
print("DETAILED MODEL EVALUATION")
print("="*70)

for result in results_list:
    model_name = result['model']
    y_pred = result['predictions']
    
    print(f"\n{model_name}:")
    print(f"  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"    TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=['No Long COVID', 'Long COVID'],
                              zero_division=0))

# Summary statistics
print("\n" + "="*70)
print("CROSS-VALIDATION RESULTS (5-Fold)")
print("="*70)
print(f"Random Forest:      {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")
print(f"Gradient Boosting:  {gb_cv.mean():.4f} ± {gb_cv.std():.4f}")
print(f"Logistic Regression: {lr_cv.mean():.4f} ± {lr_cv.std():.4f}")

# Best model
best_model_idx = np.argmax([r['roc_auc'] for r in results_list])
best_model = results_list[best_model_idx]
print(f"\n✓ Best performing model: {best_model['model']} (ROC-AUC: {best_model['roc_auc']:.4f})")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print(f"Summary files saved:")
print(f"  - model_comparison_results.csv")
print(f"  - rf_feature_importance.csv")
print(f"  - gb_feature_importance.csv")
print(f"  - lr_feature_importance.csv")
