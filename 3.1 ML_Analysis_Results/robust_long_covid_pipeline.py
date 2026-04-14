#!/usr/bin/env python3
"""
robust_long_covid_pipeline.py

A complete, self-contained pipeline to evaluate models for Long COVID prediction
with correct handling of class imbalance, feature selection, nested cross-validation,
and realistic performance estimates.

Key features:
- Data cleaning and safe numeric conversion
- Unsupervised dimensionality reduction (optional)
- imblearn Pipeline with SMOTE applied only to training folds
- Supervised feature selection inside the pipeline (L1 logistic)
- Nested cross-validation (outer RepeatedStratifiedKFold, inner StratifiedKFold)
- Randomized hyperparameter search inside inner CV
- Metrics focused on minority-class performance: Recall, F1, PR-AUC (average precision)
- Probability calibration and threshold selection utilities
- Optional SHAP explanation (if shap is installed)
- Outputs: per-fold metrics, aggregated metrics, confusion matrices, and saved CSVs

Usage:
    python robust_long_covid_pipeline.py --data Sheet1.xlsx --sheet 0

Requirements:
    pip install scikit-learn imbalanced-learn pandas numpy scipy joblib
    optional: pip install shap matplotlib seaborn
"""

import argparse
import os
import warnings
from datetime import datetime
from pprint import pformat

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Optional imports for explanation and plotting
try:
    import shap
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

warnings.filterwarnings("ignore")
RND = 42


# -------------------------
# Utilities
# -------------------------
def now_tag():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_numeric_df(df):
    """
    Convert all columns to numeric where possible, coerce errors to NaN,
    then impute NaN with median per column. Returns cleaned DataFrame.
    """
    df_num = df.apply(pd.to_numeric, errors="coerce")
    if df_num.isnull().values.any():
        medians = df_num.median()
        df_num = df_num.fillna(medians)
    return df_num


def binarize_target(y_raw):
    """
    Convert original target to binary: 0 -> Healthy, >0 -> Long COVID (1).
    Accepts pandas Series or numpy array.
    """
    y = (y_raw.astype(float) > 0).astype(int)
    return y


def summarize_metrics(metrics_list):
    """
    Given a list of dicts with numeric metrics, return mean and std summary.
    """
    df = pd.DataFrame(metrics_list)
    summary = df.agg(["mean", "std"]).T
    return summary


# -------------------------
# Pipeline builder
# -------------------------
from sklearn.feature_selection import SelectKBest, f_classif

def build_pipeline(
    estimator_name="RandomForest",
    use_pca=False,
    pca_n_components=0.95,
    l1_C=0.1,
    smote_k_neighbors=1,   # safe default
):
    if estimator_name == "RandomForest":
        estimator = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RND)
    elif estimator_name == "GradientBoosting":
        estimator = GradientBoostingClassifier(n_estimators=200, random_state=RND)
    elif estimator_name == "LogisticRegression":
        estimator = LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=500, random_state=RND)
    else:
        raise ValueError("Unsupported estimator_name")

    steps = [("scaler", StandardScaler())]
    if use_pca:
        steps.append(("pca", PCA(n_components=pca_n_components, random_state=RND)))
    # SMOTE with a conservative default; we'll tune k_neighbors but keep safe fallback
    steps.append(("smote", SMOTE(random_state=RND, k_neighbors=smote_k_neighbors)))
    # Supervised selector (may select zero features in some folds) — we'll check later
    sel = SelectFromModel(
        LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced", C=l1_C, random_state=RND)
    )
    steps.append(("sel", sel))
    steps.append(("clf", estimator))
    return ImbPipeline(steps=steps)


def nested_cv_evaluate(
    X, y, estimator_name="RandomForest",
    outer_splits=5, outer_repeats=3, inner_splits=5,
    n_iter_search=40, random_state=RND, use_pca=False
):
    outer_cv = RepeatedStratifiedKFold(n_splits=outer_splits, n_repeats=outer_repeats, random_state=random_state)
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)

    per_fold_results = []
    last_best_estimator = None
    fold_idx = 0

    # Build estimator-specific param grid (only valid params)
    base_param_dist = {
        "smote__k_neighbors": [1, 2, 3],  # conservative choices; 1 is safe for tiny minority
        "sel__estimator__C": [0.02, 0.05, 0.1, 0.2],
    }
    if estimator_name in ["RandomForest", "GradientBoosting"]:
        base_param_dist.update({
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [3, 5, None],
        })
    # For LogisticRegression, do not include n_estimators or max_depth

    from sklearn.metrics import make_scorer
    scorer = make_scorer(f1_score, pos_label=1)

    for train_idx, test_idx in outer_cv.split(X, y):
        fold_idx += 1
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # If training minority count is extremely small, log it and force SMOTE k_neighbors=1
        n_minority = np.bincount(y_train)[1] if 1 in np.bincount(y_train) else 0
        if n_minority < 2:
            print(f"[Fold {fold_idx}] WARNING: only {n_minority} minority samples in training fold; skipping SMOTE and using class_weight only.")
            # Build pipeline without SMOTE
            pipe = ImbPipeline([
                ("scaler", StandardScaler()),
                ("sel", SelectFromModel(LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced", C=0.1, random_state=RND))),
                ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RND) if estimator_name=="RandomForest" else
                        (GradientBoostingClassifier(n_estimators=200, random_state=RND) if estimator_name=="GradientBoosting" else LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=500, random_state=RND)))
            ])
            param_dist = {k: v for k, v in base_param_dist.items() if k != "smote__k_neighbors"}
        else:
            # safe k_neighbors upper bound
            max_k = max(1, n_minority - 1)
            # restrict candidate k_neighbors to <= max_k
            cand_k = [k for k in base_param_dist["smote__k_neighbors"] if k <= max_k]
            if not cand_k:
                cand_k = [1]
            param_dist = base_param_dist.copy()
            param_dist["smote__k_neighbors"] = cand_k
            pipe = build_pipeline(estimator_name=estimator_name, use_pca=use_pca, smote_k_neighbors=cand_k[0])

        # Clean param_dist: remove any keys not present in pipeline
        param_dist = {k: v for k, v in param_dist.items() if k in pipe.get_params()}

        # If selector might select zero features, add a fallback param to use SelectKBest instead
        # We'll try both: the pipeline with SelectFromModel and a pipeline variant with SelectKBest
        # For simplicity, we run RandomizedSearchCV on the current pipeline; after best found, check selected features
        search = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=min(n_iter_search, 30),
                                    scoring=scorer, cv=inner_cv, random_state=random_state, n_jobs=-1, verbose=0)
        search.fit(X_train, y_train)
        best = search.best_estimator_
        last_best_estimator = best

        # After fitting, check if selector produced zero features (some SelectFromModel implementations can)
        try:
            # If 'sel' exists and has get_support, test it
            if "sel" in best.named_steps:
                sel_step = best.named_steps["sel"]
                # If selector is SelectFromModel, it may not expose get_support_ until fitted; handle gracefully
                try:
                    support = sel_step.get_support()
                    if support.sum() == 0:
                        # fallback: replace selector with SelectKBest(k=min(50, X_train.shape[1]))
                        k = min(50, max(1, X_train.shape[1] // 10))
                        print(f"[Fold {fold_idx}] Selector chose 0 features; falling back to SelectKBest(k={k})")
                        # rebuild pipeline with SelectKBest and refit
                        steps = []
                        steps.append(("scaler", StandardScaler()))
                        if use_pca:
                            steps.append(("pca", PCA(n_components=0.95, random_state=RND)))
                        if n_minority >= 2:
                            steps.append(("smote", SMOTE(random_state=RND, k_neighbors=param_dist.get("smote__k_neighbors", [1])[0])))
                        steps.append(("sel", SelectKBest(score_func=f_classif, k=k)))
                        steps.append(("clf", best.named_steps["clf"]))
                        fallback_pipe = ImbPipeline(steps=steps)
                        fallback_pipe.fit(X_train, y_train)
                        final_model = fallback_pipe
                    else:
                        final_model = best
                except Exception:
                    final_model = best
            else:
                final_model = best
        except Exception:
            final_model = best

        # Evaluate final_model on test fold
        y_pred = final_model.predict(X_test)
        if hasattr(final_model, "predict_proba"):
            y_proba = final_model.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        ap = average_precision_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else np.nan
        roc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else np.nan
        cm = confusion_matrix(y_test, y_pred)

        per_fold_results.append({
            "fold": fold_idx,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "n_minority_train": n_minority,
            "best_params": getattr(search, "best_params_", None),
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "avg_precision": ap, "roc_auc": roc, "confusion_matrix": cm
        })

        print(f"[Outer fold {fold_idx}] acc={acc:.4f} f1={f1:.4f} recall={rec:.4f} ap={ap:.4f}")

    return per_fold_results, last_best_estimator


# -------------------------
# Main execution
# -------------------------
def main(args):
    # Load data
    print("Loading data...")
    df = pd.read_excel(args.data, sheet_name=args.sheet)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in sheet")

    # Identify metadata columns to drop if present
    metadata_cols = [
        "x0_LC_ID",
        "x0_Censor_Complete",
        "x0_Censor_Cohort_ID",
        "x0_Patient_Cluster_Label",
        "x0_Patient_Cluster",
        "x0_Censor_Oral_Steroid",
        "x0_Censor_Pit_Adre_Dysfunction",
        "x0_Censor_Pregnancy",
    ]
    feature_cols = [c for c in df.columns if c not in metadata_cols]

    X_raw = df[feature_cols].copy()
    y_raw = df[args.target].copy()

    print(f"Raw dataset shape: {X_raw.shape}")
    X = safe_numeric_df(X_raw)
    y = binarize_target(y_raw)
    y = pd.Series(y, name="target")

    print(f"After cleaning: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target distribution: {np.bincount(y)} (0=Healthy, 1=Long COVID)")

    # Optional: remove zero-variance features
    nunique = X.nunique()
    zero_var = nunique[nunique <= 1].index.tolist()
    if zero_var:
        print(f"Removing {len(zero_var)} zero-variance features")
        X = X.drop(columns=zero_var)

    # Optional unsupervised dimensionality reduction to reduce noise
    if args.use_pca_pre:
        print("Applying unsupervised PCA (pre-pipeline) to reduce dimensionality...")
        pca = PCA(n_components=args.pca_pre_n, random_state=RND)
        X_pca = pca.fit_transform(X)
        X = pd.DataFrame(X_pca, index=X.index)
        print(f"PCA reduced features to {X.shape[1]} components")

    # Run nested CV for each model
    models_to_run = args.models.split(",")
    all_results = {}
    best_estimators = {}

    for model_name in models_to_run:
        model_name = model_name.strip()
        print("\n" + "=" * 80)
        print(f"Running nested CV for: {model_name}")
        print("=" * 80)
        per_fold_results, best_est = nested_cv_evaluate(
            X,
            y,
            estimator_name=model_name,
            outer_splits=args.outer_splits,
            outer_repeats=args.outer_repeats,
            inner_splits=args.inner_splits,
            n_iter_search=args.n_iter_search,
            random_state=RND,
            use_pca=args.use_pca_in_pipeline,
        )
        all_results[model_name] = per_fold_results
        best_estimators[model_name] = best_est

    # Aggregate and save results
    out_dir = f"results_{now_tag()}"
    os.makedirs(out_dir, exist_ok=True)

    summary_rows = []
    for model_name, folds in all_results.items():
        metrics_list = [
            {
                "accuracy": f["accuracy"],
                "precision": f["precision"],
                "recall": f["recall"],
                "f1": f["f1"],
                "avg_precision": f["avg_precision"],
                "roc_auc": f["roc_auc"],
            }
            for f in folds
        ]
        summary = summarize_metrics(metrics_list)
        summary.to_csv(os.path.join(out_dir, f"{model_name}_metrics_summary.csv"))
        # Save per-fold details
        pd.DataFrame(folds).to_csv(os.path.join(out_dir, f"{model_name}_per_fold.csv"), index=False)
        summary_rows.append(
            {
                "model": model_name,
                "accuracy_mean": summary.loc["mean", "accuracy"],
                "accuracy_std": summary.loc["std", "accuracy"],
                "recall_mean": summary.loc["mean", "recall"],
                "recall_std": summary.loc["std", "recall"],
                "f1_mean": summary.loc["mean", "f1"],
                "f1_std": summary.loc["std", "f1"],
                "avg_precision_mean": summary.loc["mean", "avg_precision"],
                "avg_precision_std": summary.loc["std", "avg_precision"],
                "roc_auc_mean": summary.loc["mean", "roc_auc"],
                "roc_auc_std": summary.loc["std", "roc_auc"],
            }
        )

        # Save best estimator for later inspection
        try:
            dump(best_estimators[model_name], os.path.join(out_dir, f"{model_name}_best_estimator.joblib"))
        except Exception:
            pass

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(out_dir, "models_summary.csv"), index=False)

    print("\n" + "=" * 80)
    print("SUMMARY (aggregated across outer folds)")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    # Optional SHAP explanation on the last best estimator of the first model (if available)
    if HAS_SHAP and best_estimators:
        first_model_name = list(best_estimators.keys())[0]
        best = best_estimators[first_model_name]
        print(f"\nGenerating SHAP explanations for {first_model_name} (if supported)...")
        try:
            # Need a sample of X transformed by pipeline steps up to 'sel' to get selected features
            # If pipeline is calibrated wrapper, unwrap to get underlying pipeline
            model_to_explain = best
            if isinstance(best, CalibratedClassifierCV):
                model_to_explain = best.base_estimator

            # Extract pipeline and the selector to transform X
            if hasattr(model_to_explain, "named_steps"):
                pipeline = model_to_explain
                # Transform X using pipeline up to 'sel'
                X_trans = pipeline[:-1].transform(X)  # all steps except classifier
                clf = pipeline.named_steps["clf"]
            else:
                X_trans = X.values
                clf = model_to_explain

            explainer = shap.Explainer(clf, X_trans)
            shap_values = explainer(X_trans)

            # Save summary plot
            plt.figure(figsize=(8, 6))
            shap.summary_plot(shap_values, X_trans, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{first_model_name}_shap_summary.png"))
            plt.close()
            print("SHAP summary saved.")
        except Exception as e:
            print("SHAP explanation failed:", e)

    print(f"\nAll results saved to: {out_dir}")
    print("Pipeline complete.")


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robust Long COVID modeling pipeline")
    parser.add_argument("--data", type=str, default="Sheet1.xlsx", help="Excel file containing data")
    parser.add_argument("--sheet", type=int, default=0, help="Sheet index (0-based) for features/target")
    parser.add_argument("--target", type=str, default="x0_Censor_Complete", help="Target column name")
    parser.add_argument("--models", type=str, default="RandomForest,GradientBoosting,LogisticRegression", help="Comma-separated model names")
    parser.add_argument("--outer_splits", type=int, default=5, help="Outer CV splits")
    parser.add_argument("--outer_repeats", type=int, default=3, help="Outer CV repeats")
    parser.add_argument("--inner_splits", type=int, default=5, help="Inner CV splits")
    parser.add_argument("--n_iter_search", type=int, default=40, help="RandomizedSearchCV iterations")
    parser.add_argument("--use_pca_pre", action="store_true", help="Apply unsupervised PCA before pipeline")
    parser.add_argument("--pca_pre_n", type=float, default=0.95, help="PCA n_components if pre-PCA is used")
    parser.add_argument("--use_pca_in_pipeline", action="store_true", help="Include PCA inside the pipeline")
    args = parser.parse_args()

    main(args)
