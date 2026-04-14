#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.utils.class_weight import compute_sample_weight


def label_long_covid(lc_id):
    if not isinstance(lc_id, str):
        return np.nan
    lc_id = lc_id.strip()
    if lc_id.endswith(".C") or lc_id.endswith(".CVC"):
        return 0
    if lc_id.startswith("LC."):
        return 1
    return np.nan


def safe_numeric_df(df):
    out = df.copy()
    for col in out.columns:
        out[col] = pd.to_numeric(
            out[col].replace(["NaN;", "NaN", "nan", "", " "], np.nan),
            errors="coerce"
        )
        if out[col].notna().sum() == 0:
            out[col] = 0.0
        else:
            out[col] = out[col].fillna(out[col].median())
    return out.astype(float)


def make_selector(X_train, y_train):
    selector = SelectFromModel(
        LogisticRegression(
            penalty="l1",
            solver="liblinear",
            class_weight="balanced",
            C=0.1,
            random_state=42
        )
    )
    X_train_sel = selector.fit_transform(X_train, y_train)

    if X_train_sel.shape[1] == 0:
        k = min(25, X_train.shape[1])
        selector = SelectKBest(score_func=f_classif, k=k)
        X_train_sel = selector.fit_transform(X_train, y_train)

    return selector, X_train_sel


def main():
    print("Loading Sheet1 (Biomarkers)...")
    df = pd.read_excel("Sheet1.xlsx", sheet_name=0)

    if "x0_LC_ID" not in df.columns:
        raise ValueError("x0_LC_ID not found in Sheet1.xlsx")

    df["Long_COVID"] = df["x0_LC_ID"].apply(label_long_covid)
    df = df[df["Long_COVID"].notna()].copy()
    df["Long_COVID"] = df["Long_COVID"].astype(int)

    metadata_cols = [
        "x0_LC_ID",
        "Long_COVID",
        "x0_Censor_Complete",
        "x0_Censor_Cohort_ID",
        "x0_Patient_Cluster_Label",
        "x0_Patient_Cluster",
        "x0_Censor_Oral_Steroid",
        "x0_Censor_Pit_Adre_Dysfunction",
        "x0_Censor_Pregnancy",
        "x0_Censor_Active_Chemotherapy",
        "x0_Censor_Active_Malignancy",
    ]

    feature_cols = [c for c in df.columns if c not in metadata_cols]
    X = safe_numeric_df(df[feature_cols])
    y = df["Long_COVID"].astype(int).values

    if len(np.unique(y)) < 2:
        raise ValueError("Need both long COVID and control samples after labeling.")

    print(f"Dataset Ready: {X.shape[0]} samples, {X.shape[1]} features.")
    print(f"Target Distribution: {np.bincount(y)} (0=Control, 1=Long COVID)")

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            max_depth=3,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        ),
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            solver="liblinear",
            C=0.1,
            random_state=42,
            max_iter=2000
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            random_state=42
        )
    }

    loo = LeaveOneOut()
    model_summaries = {}

    for name, model in models.items():
        print(f"\nEvaluating {name} via LOOCV...")
        all_probas = []
        all_preds = []

        for train_index, test_index in loo.split(X):
            X_train = X.iloc[train_index].copy()
            X_test = X.iloc[test_index].copy()
            y_train = y[train_index]
            y_test = y[test_index]

            selector, X_train_sel = make_selector(X_train, y_train)
            X_test_sel = selector.transform(X_test)

            if X_train_sel.shape[1] == 0:
                X_train_sel = X_train.values
                X_test_sel = X_test.values

            if name == "Gradient Boosting":
                sw = compute_sample_weight(class_weight="balanced", y=y_train)
                model.fit(X_train_sel, y_train, sample_weight=sw)
            else:
                model.fit(X_train_sel, y_train)

            proba = model.predict_proba(X_test_sel)[:, 1][0]
            pred = model.predict(X_test_sel)[0]

            all_probas.append(proba)
            all_preds.append(pred)

        model_summaries[name] = {
            "auc": roc_auc_score(y, all_probas),
            "acc": accuracy_score(y, all_preds),
            "f1": f1_score(y, all_preds),
            "matrix": confusion_matrix(y, all_preds),
            "report": classification_report(
                y, all_preds, target_names=["Control", "Long COVID"]
            )
        }

    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON (LOOCV)")
    print("=" * 70)
    print(f"{'Model Name':<20} | {'ROC-AUC':<10} | {'Accuracy':<10} | {'F1-Score'}")
    print("-" * 70)

    for name, stats in model_summaries.items():
        print(f"{name:<20} | {stats['auc']:<10.4f} | {stats['acc']:<10.4f} | {stats['f1']:.4f}")

    for name, stats in model_summaries.items():
        print(f"\n--- DETAILED REPORT: {name} ---")
        print(f"Confusion Matrix:\n{stats['matrix']}")
        print(f"Full Metrics:\n{stats['report']}")

    # Final full-data feature importance from RF, using the same corrected label
    selector, X_sel = make_selector(X, y)
    rf_final = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        max_depth=3,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    rf_final.fit(X_sel, y)

    if hasattr(selector, "get_support"):
        selected_features = X.columns[selector.get_support()].tolist()
    else:
        selected_features = X.columns.tolist()

    if len(selected_features) == len(rf_final.feature_importances_):
        feature_importance = pd.DataFrame({
            "feature": selected_features,
            "importance": rf_final.feature_importances_
        }).sort_values("importance", ascending=False)
        feature_importance.to_csv("feature_importance_fixed_2_0.csv", index=False)
        print("\nSaved feature_importance_fixed_2_0.csv")


if __name__ == "__main__":
    main()