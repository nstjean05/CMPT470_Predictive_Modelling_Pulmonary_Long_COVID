import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')


print("Loading dataset...")
df = pd.read_excel("Sheet1.xlsx", sheet_name=0)
print(f"Dataset shape: {df.shape}")

#Label tagetting
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

#Ignore bad columnts
base_metadata_cols = [
    'x0_LC_ID',
    'Long_COVID',
    'x0_Censor_Complete', 'x0_Censor_Cohort_ID',
    'x0_Patient_Cluster_Label', 'x0_Patient_Cluster',
    'x0_Censor_Oral_Steroid', 'x0_Censor_Pit_Adre_Dysfunction',
    'x0_Censor_Pregnancy', 'x0_Censor_Active_Chemotherapy',
    'x0_Censor_Active_Malignancy', 'x0_Censor_Autoimmune_Pre_Exist',
    'x0_Censor_Immuno_Supress_Med', 'x0_Censor_IVIG', 'x0_Censor_Thyroid',
    'x0_Symp_Survey_Long_COVID_Propensity_Score_Optimized',
    'x0_LC_Symptom_totalsympt', 'x0_LCSI_ID_Label',
    'x1_Description_Cytokine_Label', 'x1_ELISA_Label', 'x1_SI_ID_Label'
]

survey_cols = [col for col in df.columns if 'x0_Symp_Survey_' in col]
metadata_cols = base_metadata_cols + survey_cols
feature_cols = [col for col in df.columns if col not in metadata_cols]


numeric_features = []
for col in feature_cols:
    try:
        col_data = df[col].replace(['NaN;', 'NaN', 'nan', '', ' '], np.nan)
        numeric_col = pd.to_numeric(col_data, errors='coerce')
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
for col in X.columns:
    if X[col].isnull().sum() > 0:
        X[col].fillna(X[col].median(), inplace=True)

valid_idx = ~y.isnull()
X = X[valid_idx]
y = y[valid_idx]

print(f"\nFinal dataset for model: {X.shape}")
print(f"LC cases: {y.sum()} | Controls: {(y==0).sum()}")

#Split between training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#Select correct features
selector = SelectKBest(score_func=f_classif, k=min(100, X_train.shape[1]))  # keep top 100 features
X_train_sel = selector.fit_transform(X_train, y_train)
X_test_sel = selector.transform(X_test)
selected_features = np.array(numeric_features)[selector.get_support()].tolist()
print(f"\nSelected top {len(selected_features)} features for modeling.")

#define models
rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=15,
    min_samples_split=5, min_samples_leaf=2,
    max_features='sqrt', random_state=42,
    n_jobs=-1, class_weight='balanced'
)

lr_model = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(
        penalty='l1', solver='liblinear', class_weight='balanced',
        random_state=42, max_iter=500
    ))
])

gb_model = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.05,
    max_depth=4, min_samples_split=5,
    min_samples_leaf=2, random_state=42
)

models = {
    "RandomForest": rf_model,
    "LogisticRegression": lr_model,
    "GradientBoosting": gb_model
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#start training
results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_sel, y_train)
    y_pred = model.predict(X_test_sel)
    y_proba = model.predict_proba(X_test_sel)[:, 1] if hasattr(model, "predict_proba") else None

    print("\n=== RESULTS ===")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
    if y_proba is not None:
        print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cv_scores = cross_val_score(model, X.values, y.values, cv=cv, scoring='roc_auc')
    print(f"\nCross-validation ROC-AUC scores: {cv_scores}")
    print(f"Mean ± SD: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    results.append((name, cv_scores.mean()))

#rank feature importance
rf_model.fit(X_train_sel, y_train)
importance = pd.DataFrame({
    'feature': selected_features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 Biomarkers/Features:")
print(importance.head(15).to_string(index=False))
importance.to_csv("feature_importance_biological.csv", index=False)

#display results
best_model = max(results, key=lambda x: x[1])
print(f"\nBest Performing Model: {best_model[0]} (Mean ROC-AUC = {best_model[1]:.4f})")
print("\nAnalysis complete.")