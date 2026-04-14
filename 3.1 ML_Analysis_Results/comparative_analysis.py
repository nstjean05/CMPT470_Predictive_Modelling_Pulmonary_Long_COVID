"""
Comparative Analysis: Biomarkers vs. Clinical Features for Long COVID Prediction
Trains models on both Sheet1 (biomarkers) and Sheet2 (clinical) separately, then compares
(UPDATED to use Cost-Sensitive Learning, L1 Regularization, and prevent overfitting)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.utils.class_weight import compute_sample_weight
import warnings
warnings.filterwarnings('ignore')

def train_and_evaluate_models(X, y, dataset_name, random_state=42):
    """Train robust models and return results"""
    
    # Handle class imbalance - stratify
    min_class_count = min(y.value_counts().values)
    use_stratify = True if min_class_count >= 2 else False
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, 
        stratify=(y if use_stratify else None)
    )
    
    # Feature Selection (crucial for biomarkers, harmless for clinical)
    selector = SelectFromModel(
        LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', C=0.2, random_state=random_state)
    )
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)
    
    print(f"[{dataset_name}] Selected {X_train_sel.shape[1]} features from original {X.shape[1]}")

    sample_weights_train = compute_sample_weight(class_weight='balanced', y=y_train)

    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', max_depth=5, min_samples_leaf=3, random_state=random_state),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=3, min_samples_leaf=3, random_state=random_state),
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', C=0.1, random_state=random_state)
    }
    
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    for name, model in models.items():
        # Fit model (GB needs sample_weight passed directly)
        if name == 'Gradient Boosting':
            model.fit(X_train_sel, y_train, sample_weight=sample_weights_train)
        else:
            model.fit(X_train_sel, y_train)
            
        y_pred = model.predict(X_test_sel)
        y_proba = model.predict_proba(X_test_sel)[:, 1] if len(np.unique(y_train)) > 1 else y_pred
        
        # Safe ROC-AUC calc
        if len(np.unique(y_test)) > 1:
            roc = roc_auc_score(y_test, y_proba)
        else:
            roc = np.nan
            
        cv_scores = cross_val_score(model, X_train_sel, y_train, cv=cv, scoring='roc_auc')
            
        results[name] = {
            'Test_ROC_AUC': roc,
            'Test_Accuracy': accuracy_score(y_test, y_pred),
            'CV_ROC_AUC_Mean': cv_scores.mean(),
            'CV_ROC_AUC_Std': cv_scores.std()
        }
        
    return results

# =====================================================================
# Main Execution
# =====================================================================
if __name__ == "__main__":
    print("Loading Datasets...")
    df_bio = pd.read_excel("Sheet1.xlsx", sheet_name=0)
    df_clin = pd.read_excel("Sheet2.xlsx", sheet_name=1)

    target_col = 'x0_Censor_Complete'
    meta_cols = ['x0_LC_ID', 'x0_Censor_Complete', 'x0_Censor_Cohort_ID', 'x0_Patient_Cluster_Label', 'x0_Patient_Cluster', 'x0_Censor_Oral_Steroid', 'x0_Censor_Pit_Adre_Dysfunction', 'x0_Censor_Pregnancy']

    X_bio = df_bio.drop(columns=[c for c in meta_cols if c in df_bio.columns])
    y_bio = df_bio[target_col]
    
    X_clin = df_clin.drop(columns=[c for c in meta_cols if c in df_clin.columns])
    y_clin = df_clin[target_col]

    print("\n--- Evaluating Biomarker Data ---")
    bio_results = train_and_evaluate_models(X_bio, y_bio, "Biomarkers")
    
    print("\n--- Evaluating Clinical Data ---")
    clin_results = train_and_evaluate_models(X_clin, y_clin, "Clinical")

    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS RESULTS (ROBUST PIPELINE)")
    print("="*80)
    
    print("\nModel Performance (Test ROC-AUC):")
    print(f"{'Model':<20} | {'Biomarkers':<15} | {'Clinical Data'}")
    print("-" * 55)
    for model in bio_results.keys():
        b_roc = bio_results[model]['Test_ROC_AUC']
        c_roc = clin_results[model]['Test_ROC_AUC']
        print(f"{model:<20} | {b_roc:<15.4f} | {c_roc:.4f}")