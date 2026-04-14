import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# Load datasets
print("Loading datasets...")
df_biomarkers = pd.read_excel("Sheet1.xlsx", sheet_name=0)

# Identify the target variable
target_col = 'x0_Censor_Complete'
print(f"\nTarget variable: {target_col}")

metadata_cols = ['x0_LC_ID', 'x0_Censor_Complete', 'x0_Censor_Cohort_ID', 
                 'x0_Patient_Cluster_Label', 'x0_Patient_Cluster',
                 'x0_Censor_Oral_Steroid', 'x0_Censor_Pit_Adre_Dysfunction',
                 'x0_Censor_Pregnancy']

X = df_biomarkers.drop(columns=[c for c in metadata_cols if c in df_biomarkers.columns])
y = df_biomarkers[target_col]
feature_names = X.columns

print(f"Dataset shape: {X.shape}")

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature Selection to avoid high-dimensional overfitting
print("Performing feature selection...")
selector = SelectFromModel(
    LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', C=0.1, random_state=42)
)
X_train_sel = selector.fit_transform(X_train, y_train)
X_test_sel = selector.transform(X_test)
selected_features = feature_names[selector.get_support()]

print(f"Using {len(selected_features)} robust features for training.")

# Initialize Cost-Sensitive Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100, 
    class_weight='balanced', 
    max_depth=5, 
    min_samples_leaf=3,
    random_state=42
)

rf_model.fit(X_train_sel, y_train)
y_pred = rf_model.predict(X_test_sel)
y_pred_proba = rf_model.predict_proba(X_test_sel)[:, 1]

# Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_model, X_train_sel, y_train, cv=cv, scoring='roc_auc')

print("\n" + "="*60)
print("TOP MOST IMPORTANT FEATURES")
print("="*60)

feature_importance = pd.DataFrame({
    'feature': selected_features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(20).to_string(index=False))

# Save results
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

feature_importance.to_csv('feature_importance.csv', index=False)
print("✓ Saved feature_importance.csv")

results_df = pd.DataFrame({
    'actual': y_test.values,
    'predicted': y_pred,
    'probability': y_pred_proba
})
results_df.to_csv('predictions.csv', index=False)
print("✓ Saved predictions.csv")