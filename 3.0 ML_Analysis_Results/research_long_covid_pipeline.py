#!/usr/bin/env python3
"""
research_long_covid_pipeline.py

Research-grade pipeline for extremely imbalanced biomarker data.
Designed for small minority class (e.g., 7 positives) and very high dimensionality.

Usage:
    python research_long_covid_pipeline.py --data Sheet1.xlsx --sheet 0

Outputs:
  - results_<timestamp>/ : per-model per-fold CSVs, summary CSV, saved best estimators
"""

import argparse
import os
import warnings
from datetime import datetime
import json

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score, confusion_matrix
)
from sklearn.model_selection import (
    RandomizedSearchCV, RepeatedStratifiedKFold, StratifiedKFold
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")
RND = 42


def now_tag():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_numeric_df(df):
    df_num = df.apply(pd.to_numeric, errors="coerce")
    if df_num.isnull().values.any():
        medians = df_num.median()
        df_num = df_num.fillna(medians)
    return df_num


def binarize_target(y_raw):
    return (y_raw.astype(float) > 0).astype(int)


def build_base_pipeline(estimator_name, use_pca=False, pca_n_components=0.95, smote_k=1, sel_l1_C=0.1, kbest_k=200):
    """
    Build an imblearn pipeline with placeholders. SMOTE k_neighbors is set to smote_k (safe default).
    The pipeline uses a two-stage selection: SelectKBest (univariate) then SelectFromModel(L1).
    """
    # estimator
    if estimator_name == "RandomForest":
        clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RND)
    elif estimator_name == "GradientBoosting":
        clf = GradientBoostingClassifier(n_estimators=200, random_state=RND)
    elif estimator_name == "LogisticRegression":
        clf = LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=1000, random_state=RND)
    else:
        raise ValueError("Unsupported estimator")

    steps = []
    steps.append(("scaler", StandardScaler()))
    if use_pca:
        steps.append(("pca", PCA(n_components=pca_n_components, random_state=RND)))
    # SMOTE placeholder (will be tuned per fold)
    steps.append(("smote", SMOTE(random_state=RND, k_neighbors=smote_k)))
    # Two-stage selection: univariate then L1-based
    steps.append(("kbest", SelectKBest(score_func=f_classif, k=kbest_k)))
    sel_l1 = SelectFromModel(LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced", C=sel_l1_C, random_state=RND))
    steps.append(("sel_l1", sel_l1))
    steps.append(("clf", clf))
    pipe = ImbPipeline(steps=steps)
    return pipe


def safe_param_grid_for_estimator(estimator_name):
    """
    Return a param grid dictionary with only valid keys for the pipeline.
    We'll tune: smote__k_neighbors, kbest__k, sel_l1__estimator__C, clf__n_estimators, clf__max_depth (if applicable)
    """
    base = {
        "smote__k_neighbors": [1, 2, 3],
        "kbest__k": [50, 100, 200],
        "sel_l1__estimator__C": [0.02, 0.05, 0.1, 0.2],
    }
    if estimator_name in ["RandomForest", "GradientBoosting"]:
        base.update({
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [3, 5, None],
        })
    # For LogisticRegression, do not include n_estimators or max_depth
    return base


def nested_cv_pipeline(X, y, estimator_name="RandomForest",
                       outer_splits=3, outer_repeats=5, inner_splits=3,
                       n_iter_search=30, use_pca=False, pca_n_components=0.95):
    """
    Nested CV with conservative outer splits to avoid zero-minority training folds.
    Returns per-fold results and last best estimator.
    """
    outer_cv = RepeatedStratifiedKFold(n_splits=outer_splits, n_repeats=outer_repeats, random_state=RND)
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=RND)

    per_fold_results = []
    last_best = None
    fold = 0

    for train_idx, test_idx in outer_cv.split(X, y):
        fold += 1
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        n_minority = int(np.sum(y_train == 1))
        print(f"[Fold {fold}] train_size={len(train_idx)} test_size={len(test_idx)} minority_in_train={n_minority}")

        # If too few minority samples, skip SMOTE and use class_weight only
        if n_minority < 2:
            print(f"[Fold {fold}] WARNING: <2 minority samples in training fold. Skipping SMOTE and using class_weight only.")
            # Build a pipeline without SMOTE
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("kbest", SelectKBest(score_func=f_classif, k=min(50, max(1, X_train.shape[1]//10)))),
                ("sel_l1", SelectFromModel(LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced", C=0.1, random_state=RND))),
                ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RND) if estimator_name=="RandomForest"
                 else (GradientBoostingClassifier(n_estimators=200, random_state=RND) if estimator_name=="GradientBoosting"
                       else LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=1000, random_state=RND)))
            ])
            param_dist = {k: v for k, v in safe_param_grid_for_estimator(estimator_name).items() if not k.startswith("smote")}
        else:
            # safe max k for SMOTE
            max_k = max(1, n_minority - 1)
            cand_k = [k for k in [1, 2, 3] if k <= max_k]
            if not cand_k:
                cand_k = [1]
            # Build pipeline with initial safe smote_k
            pipe = build_base_pipeline(estimator_name, use_pca=use_pca, pca_n_components=pca_n_components, smote_k=cand_k[0], sel_l1_C=0.1, kbest_k=min(200, X_train.shape[1]))
            # param grid but restrict smote__k_neighbors to cand_k
            param_dist = safe_param_grid_for_estimator(estimator_name)
            param_dist["smote__k_neighbors"] = cand_k

        # Keep only params that exist in the pipeline
        param_dist = {k: v for k, v in param_dist.items() if k in pipe.get_params()}

        # Use F1 (minority) as optimization metric
        from sklearn.metrics import make_scorer
        scorer = make_scorer(f1_score, pos_label=1)

        # Randomized search
        n_iter = min(n_iter_search, max(1, sum(len(v) for v in param_dist.values())))
        search = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=n_iter, scoring=scorer, cv=inner_cv, random_state=RND, n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_estimator_
        last_best = best

        # After fit, check if sel_l1 selected zero features; if so, fallback to SelectKBest and refit
        try:
            if hasattr(best, "named_steps") and "sel_l1" in best.named_steps:
                sel = best.named_steps["sel_l1"]
                try:
                    support = sel.get_support()
                    if support.sum() == 0:
                        k_fb = min(100, max(1, X_train.shape[1] // 10))
                        print(f"[Fold {fold}] sel_l1 selected 0 features; falling back to SelectKBest(k={k_fb}) and refitting")
                        # rebuild pipeline with SelectKBest and same classifier
                        clf = best.named_steps["clf"]
                        steps = []
                        steps.append(("scaler", StandardScaler()))
                        if use_pca:
                            steps.append(("pca", PCA(n_components=pca_n_components, random_state=RND)))
                        if n_minority >= 2:
                            steps.append(("smote", SMOTE(random_state=RND, k_neighbors=param_dist.get("smote__k_neighbors", [1])[0])))
                        steps.append(("kbest", SelectKBest(score_func=f_classif, k=k_fb)))
                        steps.append(("clf", clf))
                        fallback = ImbPipeline(steps=steps) if n_minority >= 2 else Pipeline(steps=steps)
                        fallback.fit(X_train, y_train)
                        final_model = fallback
                    else:
                        final_model = best
                except Exception:
                    final_model = best
            else:
                final_model = best
        except Exception:
            final_model = best

        # Evaluate
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
            "fold": fold,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "n_minority_train": n_minority,
            "best_params": getattr(search, "best_params_", None),
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "avg_precision": ap, "roc_auc": roc, "confusion_matrix": cm.tolist()
        })

        print(f"[Fold {fold}] acc={acc:.4f} f1={f1:.4f} recall={rec:.4f} ap={ap:.4f}")

    return per_fold_results, last_best


def main(args):
    df = pd.read_excel(args.data, sheet_name=args.sheet)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found")

    metadata_cols = [
        "x0_LC_ID", "x0_Censor_Complete", "x0_Censor_Cohort_ID",
        "x0_Patient_Cluster_Label", "x0_Patient_Cluster",
        "x0_Censor_Oral_Steroid", "x0_Censor_Pit_Adre_Dysfunction", "x0_Censor_Pregnancy"
    ]
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    X_raw = df[feature_cols].copy()
    y_raw = df[args.target].copy()

    print("Cleaning numeric features...")
    X = safe_numeric_df(X_raw)
    y = pd.Series(binarize_target(y_raw), name="target")

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features. Target dist: {np.bincount(y)}")

    # Remove zero-variance features
    nunique = X.nunique()
    zero_var = nunique[nunique <= 1].index.tolist()
    if zero_var:
        print(f"Removing {len(zero_var)} zero-variance features")
        X = X.drop(columns=zero_var)

    # Optional pre-PCA to reduce noise (disabled by default)
    if args.pre_pca:
        print("Applying pre-PCA to reduce dimensionality...")
        pca = PCA(n_components=args.pre_pca_n, random_state=RND)
        X_pca = pca.fit_transform(X)
        X = pd.DataFrame(X_pca, index=X.index)
        print(f"PCA reduced features to {X.shape[1]} components")

    models = [m.strip() for m in args.models.split(",")]
    results = {}
    best_estimators = {}

    for model_name in models:
        print("\n" + "="*60)
        print(f"Running nested CV for {model_name}")
        print("="*60)
        per_fold, best = nested_cv_pipeline(
            X, y,
            estimator_name=model_name,
            outer_splits=args.outer_splits,
            outer_repeats=args.outer_repeats,
            inner_splits=args.inner_splits,
            n_iter_search=args.n_iter_search,
            use_pca=args.use_pca_in_pipeline,
            pca_n_components=args.pca_n_components
        )
        results[model_name] = per_fold
        best_estimators[model_name] = best

    # Save results
    out_dir = f"results_{now_tag()}"
    os.makedirs(out_dir, exist_ok=True)
    summary_rows = []
    for model_name, folds in results.items():
        pd.DataFrame(folds).to_csv(os.path.join(out_dir, f"{model_name}_per_fold.csv"), index=False)
        # aggregate
        dfm = pd.DataFrame(folds)
        summary = {
            "model": model_name,
            "accuracy_mean": dfm["accuracy"].mean(),
            "accuracy_std": dfm["accuracy"].std(),
            "recall_mean": dfm["recall"].mean(),
            "recall_std": dfm["recall"].std(),
            "f1_mean": dfm["f1"].mean(),
            "f1_std": dfm["f1"].std(),
            "avg_precision_mean": dfm["avg_precision"].mean(),
            "avg_precision_std": dfm["avg_precision"].std(),
            "roc_auc_mean": dfm["roc_auc"].mean(),
            "roc_auc_std": dfm["roc_auc"].std()
        }
        summary_rows.append(summary)
        # save best estimator
        try:
            dump(best_estimators[model_name], os.path.join(out_dir, f"{model_name}_best.joblib"))
        except Exception:
            pass

    pd.DataFrame(summary_rows).to_csv(os.path.join(out_dir, "models_summary.csv"), index=False)
    print("\nAll results saved to:", out_dir)
    print("Summary:")
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Research-grade Long COVID pipeline")
    parser.add_argument("--data", type=str, default="Sheet1.xlsx")
    parser.add_argument("--sheet", type=int, default=0)
    parser.add_argument("--target", type=str, default="x0_Censor_Complete")
    parser.add_argument("--models", type=str, default="RandomForest,GradientBoosting,LogisticRegression")
    parser.add_argument("--outer_splits", type=int, default=3)
    parser.add_argument("--outer_repeats", type=int, default=5)
    parser.add_argument("--inner_splits", type=int, default=3)
    parser.add_argument("--n_iter_search", type=int, default=20)
    parser.add_argument("--pre_pca", action="store_true", help="Apply unsupervised PCA before pipeline")
    parser.add_argument("--pre_pca_n", type=float, default=0.95)
    parser.add_argument("--use_pca_in_pipeline", action="store_true", help="Include PCA inside pipeline")
    parser.add_argument("--pca_n_components", type=float, default=0.95)
    args = parser.parse_args()
    main(args)
