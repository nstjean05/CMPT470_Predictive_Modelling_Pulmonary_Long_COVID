import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# ====================
# LOAD DATA
# ====================
print("Loading dataset...")
df = pd.read_excel("Sheet1.xlsx", sheet_name=0)

print(f"Dataset shape: {df.shape}")

# ====================
# ✅ CORRECT TARGET: Long COVID from ID
# ====================
def label_long_covid(id_str):
    if isinstance(id_str, str):
        if id_str.endswith('.C') or id_str.endswith('.CVC'):
            return 0  # Control (No Long COVID)
        else:
            return 1  # Long COVID
    return np.nan

df['Long_COVID'] = df['x0_LC_ID'].apply(label_long_covid)

print("\nTarget distribution (0=No LC, 1=LC):")
print(df['Long_COVID'].value_counts())

# ====================
# REMOVE METADATA / LEAKAGE
# ====================
# 1. Base administrative and derived label leakage
base_metadata_cols = [
    'x0_LC_ID', 
    'Long_COVID',
    'x0_Censor_Complete',
    'x0_Censor_Cohort_ID', 
    'x0_Patient_Cluster_Label', 
    'x0_Patient_Cluster',
    'x0_Censor_Oral_Steroid', 
    'x0_Censor_Pit_Adre_Dysfunction',
    'x0_Censor_Pregnancy', 
    'x0_Censor_Active_Chemotherapy',
    'x0_Censor_Active_Malignancy',
    'x0_Censor_Autoimmune_Pre_Exist',
    'x0_Censor_Immuno_Supress_Med',
    'x0_Censor_IVIG',
    'x0_Censor_Thyroid',
    'x0_Symp_Survey_Long_COVID_Propensity_Score_Optimized',
    'x0_LC_Symptom_totalsympt',
    'x0_LCSI_ID_Label',
    'x1_Description_Cytokine_Label',
    'x1_ELISA_Label',
    'x1_SI_ID_Label'
]

# 2. Dynamically find all symptom survey variables (Clinical Circularity)
survey_cols = [col for col in df.columns if 'x0_Symp_Survey_' in col]

# 3. Combine them all
metadata_cols = base_metadata_cols + survey_cols

feature_cols = [col for col in df.columns if col not in metadata_cols]

# ====================
# KEEP NUMERIC FEATURES ONLY
# ====================
numeric_features = []

for col in feature_cols:
    try:
        col_data = df[col].replace(['NaN;', 'NaN', 'nan', '', ' '], np.nan)
        numeric_col = pd.to_numeric(col_data, errors='coerce')
        # Only keep columns where at least 50% of the data is not null
        if numeric_col.notna().sum() / len(numeric_col) > 0.5:
            numeric_features.append(col)
    except:
        pass

print(f"\nFiltered Metadata & Survey columns: Removed {len(metadata_cols)} potential leakage features.")
print(f"Numeric biological features remaining: {len(numeric_features)}")

X = df[numeric_features].copy()
y = df['Long_COVID'].copy()

# ====================
# CLEAN DATA
# ====================
for col in X.columns:
    X[col] = X[col].replace(['NaN;', 'NaN', 'nan', '', ' '], np.nan)
    X[col] = pd.to_numeric(X[col], errors='coerce')

X = X.astype(float)

# Fill missing values with median
for col in X.columns:
    if X[col].isnull().sum() > 0:
        X[col].fillna(X[col].median(), inplace=True)

# Remove any rows missing the target
valid_idx = ~y.isnull()
X = X[valid_idx]
y = y[valid_idx]

print(f"\nFinal dataset for model: {X.shape}")
print(f"LC cases: {y.sum()} | Controls: {(y==0).sum()}")

# ====================
# TRAIN / TEST SPLIT
# ====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ====================
# RANDOM FOREST MODEL
# ====================
print("\nTraining Random Forest on Biological Features...")

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

rf_model.fit(X_train, y_train)

# ====================
# EVALUATION
# ====================
y_pred = rf_model.predict(X_test)
y_proba = rf_model.predict_proba(X_test)[:, 1]

print("\n=== RESULTS ===")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ====================
# CROSS VALIDATION
# ====================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_model, X, y, cv=cv, scoring='roc_auc')

print("\nCross-validation ROC-AUC:")
print(cv_scores)
print(f"Mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ====================
# FEATURE IMPORTANCE
# ====================
importance = pd.DataFrame({
    'feature': numeric_features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 Biomarkers/Features:")
print(importance.head(15).to_string(index=False))

importance.to_csv("feature_importance_biological.csv", index=False)