import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load datasets
print("Loading datasets...")
df_biomarkers = pd.read_excel("Sheet1.xlsx", sheet_name=0)
df_clinical = pd.read_excel("Sheet2.xlsx", sheet_name=1)

print(f"Biomarkers shape: {df_biomarkers.shape}")
print(f"Clinical data shape: {df_clinical.shape}")

# Identify the target variable - using x0_Censor_Complete as the outcome
# (0 = recovered/no long COVID, 1+ = long COVID or incomplete recovery)
target_col = 'x0_Censor_Complete'
print(f"\nTarget variable: {target_col}")
print(f"Target distribution:\n{df_biomarkers[target_col].value_counts().sort_index()}")

# Use biomarkers as features (all columns except metadata)
metadata_cols = ['x0_LC_ID', 'x0_Censor_Complete', 'x0_Censor_Cohort_ID', 
                 'x0_Patient_Cluster_Label', 'x0_Patient_Cluster',
                 'x0_Censor_Oral_Steroid', 'x0_Censor_Pit_Adre_Dysfunction',
                 'x0_Censor_Pregnancy', 'x0_Censor_Active_Chemotherapy',
                 'x0_Censor_Active_Malignancy']

# Get feature columns (all columns that are not metadata)
feature_cols = [col for col in df_biomarkers.columns if col not in metadata_cols]

# Filter to only NUMERIC columns (remove any non-numeric features)
numeric_features = []
for col in feature_cols:
    try:
        # Replace string NaN values with actual NaN
        col_data = df_biomarkers[col].copy()
        col_data = col_data.replace(['NaN;', 'NaN', 'nan', '', ' '], np.nan)
        
        # Try to convert to numeric
        numeric_col = pd.to_numeric(col_data, errors='coerce')
        
        # Keep only if conversion was successful for most values
        conversion_rate = numeric_col.notna().sum() / len(numeric_col)
        if conversion_rate > 0.5:  # At least 50% valid numeric values
            numeric_features.append(col)
    except:
        pass

print(f"Number of numeric biomarker features: {len(numeric_features)}")
feature_cols = numeric_features

# Prepare data
X = df_biomarkers[feature_cols].copy()
y = df_biomarkers[target_col].copy()

# Clean up data: replace problematic string values with NaN, then convert to float
for col in X.columns:
    X[col] = X[col].apply(lambda x: np.nan if isinstance(x, str) and ('NaN' in x or x.strip() == '') else x)
    X[col] = pd.to_numeric(X[col], errors='coerce')

X = X.astype(float)

# Convert target to binary: 0 = no long COVID, 1 = long COVID
# Group: 0 stays 0, 1,2 becomes 1 (presence of long COVID)
y_binary = (y > 0).astype(int)
print(f"\nBinary target distribution (0=No Long COVID, 1=Long COVID):")
print(y_binary.value_counts().sort_index())

# Handle missing values - use median imputation for more robustness
print(f"\nMissing values in features: {X.isnull().sum().sum()}")
for col in X.columns:
    if X[col].isnull().sum() > 0:
        median_val = X[col].median()
        X[col].fillna(median_val, inplace=True)

# Remove any rows with missing target
valid_idx = ~y_binary.isnull()
X = X[valid_idx]
y_binary = y_binary[valid_idx]

print(f"Final dataset shape: {X.shape}")
print(f"Final target distribution:\n{y_binary.value_counts()}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")
print(f"Training set - Long COVID cases: {y_train.sum()}")
print(f"Test set - Long COVID cases: {y_test.sum()}")

# ====================
# RANDOM FOREST MODEL
# ====================
print("\n" + "="*60)
print("TRAINING RANDOM FOREST CLASSIFIER")
print("="*60)

# Train Random Forest with optimized parameters
rf_model = RandomForestClassifier(
    n_estimators=200,            # Number of trees
    max_depth=15,                # Maximum tree depth
    min_samples_split=5,         # Minimum samples to split
    min_samples_leaf=2,          # Minimum samples in leaf
    max_features='sqrt',         # Number of features to consider at each split
    random_state=42,
    n_jobs=-1,                   # Use all processors
    class_weight='balanced'      # Handle class imbalance
)

print("Fitting model...")
rf_model.fit(X_train, y_train)
print("✓ Model training complete")

# ====================
# MODEL EVALUATION
# ====================
print("\n" + "="*60)
print("MODEL EVALUATION ON TEST SET")
print("="*60)

# Predictions
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\nAccuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"  True Negatives:  {cm[0, 0]}")
print(f"  False Positives: {cm[0, 1]}")
print(f"  False Negatives: {cm[1, 0]}")
print(f"  True Positives:  {cm[1, 1]}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Long COVID', 'Long COVID']))

# Cross-validation
print("\n" + "="*60)
print("CROSS-VALIDATION (5-Fold)")
print("="*60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_model, X, y_binary, cv=cv, scoring='roc_auc')
print(f"Cross-validation ROC-AUC scores: {[f'{s:.4f}' for s in cv_scores]}")
print(f"Mean CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Feature importance
print("\n" + "="*60)
print("TOP 20 MOST IMPORTANT FEATURES")
print("="*60)

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(20).to_string(index=False))

# Save results
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

# Save feature importance
feature_importance.to_csv('feature_importance.csv', index=False)
print("✓ Saved feature_importance.csv")

# Save predictions
results_df = pd.DataFrame({
    'actual': y_test.values,
    'predicted': y_pred,
    'probability': y_pred_proba
})
results_df.to_csv('predictions.csv', index=False)
print("✓ Saved predictions.csv")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Dataset: 185 samples with {len(feature_cols)} numeric biomarker features")
print(f"Target: {target_col} (predicting Long COVID outcomes)")
print(f"Model: Random Forest (200 trees, max_depth=15)")
print(f"\nPerformance Metrics:")
print(f"  Test ROC-AUC:  {roc_auc:.4f}")
print(f"  Test Accuracy: {accuracy:.4f}")
print(f"  Test Recall:   {recall:.4f}")
print(f"  Test Precision: {precision:.4f}")
print(f"\nTop 5 Important Biomarkers for Long COVID Prediction:")
for idx, row in feature_importance.head(5).iterrows():
    print(f"  {row['feature']}: {row['importance']:.6f}")