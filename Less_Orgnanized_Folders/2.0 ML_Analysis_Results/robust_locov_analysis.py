import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.feature_selection import SelectFromModel
from sklearn.utils.class_weight import compute_sample_weight
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. LOAD & CLEAN DATA
# ==========================================
print("Loading Sheet1 (Biomarkers)...")
df = pd.read_excel("Sheet1.xlsx", sheet_name=0)

target_col = 'x0_Censor_Complete'
metadata_cols = ['x0_LC_ID', 'x0_Censor_Complete', 'x0_Censor_Cohort_ID', 
                 'x0_Patient_Cluster_Label', 'x0_Patient_Cluster',
                 'x0_Censor_Oral_Steroid', 'x0_Censor_Pit_Adre_Dysfunction',
                 'x0_Censor_Pregnancy']

X_df = df.drop(columns=[c for c in metadata_cols if c in df.columns])
y_raw = df[target_col].values

# Binarize Target: 0 -> Healthy, (1 or 2) -> Long COVID
y = (y_raw > 0).astype(int)

# Fix 'NaN;' and missing values
X_df = X_df.apply(pd.to_numeric, errors='coerce').fillna(0)
feature_names = X_df.columns.tolist()
X_values = X_df.values 

print(f"Dataset Ready: {X_values.shape[0]} samples, {X_values.shape[1]} features.")
print(f"Target Distribution: {np.bincount(y)} (0=Healthy, 1=Long COVID)")

# ==========================================
# 2. DEFINE MODELS
# ==========================================
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100, class_weight='balanced', max_depth=3, min_samples_leaf=5, random_state=42
    ),
    "Logistic Regression": LogisticRegression(
        class_weight='balanced', solver='liblinear', C=0.1, random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=2, random_state=42
    )
}

# ==========================================
# 3. LOOCV EXECUTION FOR EACH MODEL
# ==========================================
loo = LeaveOneOut()
model_summaries = {}

for name, model in models.items():
    print(f"\nEvaluating {name} via LOOCV...")
    all_probas = []
    all_preds = []

    for train_index, test_index in loo.split(X_values):
        X_train, X_test = X_values[train_index], X_values[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Feature Selection inside the loop
        selector = SelectFromModel(
            LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', C=0.1, random_state=42)
        )
        X_train_sel = selector.fit_transform(X_train, y_train)
        X_test_sel = selector.transform(X_test)

        # Gradient Boosting needs sample weights manually
        if name == "Gradient Boosting":
            sw = compute_sample_weight(class_weight='balanced', y=y_train)
            model.fit(X_train_sel, y_train, sample_weight=sw)
        else:
            model.fit(X_train_sel, y_train)

        all_probas.append(model.predict_proba(X_test_sel)[:, 1][0])
        all_preds.append(model.predict(X_test_sel)[0])

    # Store results
    model_summaries[name] = {
        'auc': roc_auc_score(y, all_probas),
        'acc': accuracy_score(y, all_preds),
        'f1': f1_score(y, all_preds),
        'matrix': confusion_matrix(y, all_preds),
        'report': classification_report(y, all_preds, target_names=['Healthy', 'Long COVID'])
    }

# ==========================================
# 4. FINAL COMPARATIVE OUTPUT
# ==========================================
print("\n" + "="*70)
print("FINAL MODEL COMPARISON (LOOCV)")
print("="*70)
print(f"{'Model Name':<20} | {'ROC-AUC':<10} | {'Accuracy':<10} | {'F1-Score'}")
print("-" * 70)

for name, stats in model_summaries.items():
    print(f"{name:<20} | {stats['auc']:<10.4f} | {stats['acc']:<10.4f} | {stats['f1']:.4f}")

# Detail for each
for name, stats in model_summaries.items():
    print(f"\n--- DETAILED REPORT: {name} ---")
    print(f"Confusion Matrix:\n{stats['matrix']}")
    print(f"Full Metrics:\n{stats['report']}")