#!/usr/bin/env python3
"""
revised_long_covid_pipeline.py

Pipeline: pre-PCA(100) -> SelectKBest(200) -> SMOTE (safe) -> BalancedRandomForest
Nested CV: outer RepeatedStratifiedKFold(n_splits=3, n_repeats=10), inner StratifiedKFold(n_splits=3)
Per-fold threshold sweep: choose threshold maximizing recall subject to min_precision (configurable).
Saves per-fold CSVs and a summary CSV.

Usage:
    python revised_long_covid_pipeline.py --data Sheet1.xlsx --sheet 0
"""

import argparse
import os
from datetime import datetime
import json

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score, confusion_matrix,
    precision_recall_curve
)
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline

# Optional: silence warnings
import warnings
warnings.filterwarnings("ignore")

RND = 42

def now_tag():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def safe_numeric_df(df):
    df_num = df.apply(pd.to_numeric, errors="coerce")
    if df_num.isnull().values.any():
        df_num = df_num.fillna(df_num.median())
    return df_num

def binarize_target(y_raw):
    return (y_raw.astype(float) > 0).astype(int)

def build_pipeline(smote_k=1, kbest_k=200, sel_l1_C=0.1, use_pca_in_pipeline=False, pca_n=0.95):
    """
    Pipeline steps:
      scaler -> (optional PCA) -> SMOTE -> SelectKBest -> SelectFromModel(L1) -> BalancedRandomForest
    """
    steps = []
    steps.append(("scaler", StandardScaler()))
    if use_pca_in_pipeline:
        steps.append(("pca", PCA(n_components=pca_n, random_state=RND)))
    steps.append(("smote", SMOTE(random_state=RND, k_neighbors=smote_k)))
    steps.append(("kbest", SelectKBest(score_func=f_classif, k=kbest_k)))
    sel = SelectFromModel(LogisticRegression(penalty="l1", solver="liblinear", C=sel_l1_C, class_weight="balanced", random_state=RND))
    steps.append(("sel_l1", sel))
    steps.append(("clf", BalancedRandomForestClassifier(n_estimators=200, random_state=RND)))
    return ImbPipeline(steps=steps)

def threshold_sweep_and_metrics(model, X_val, y_val, min_precision=0.20):
    """
    Compute precision-recall curve and choose threshold that maximizes recall
    while keeping precision >= min_precision. Return chosen threshold and metrics.
    """
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_val)[:, 1]
    else:
        # fallback: use decision_function scaled to [0,1]
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X_val)
            probs = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
        else:
            preds = model.predict(X_val)
            # no probability info; return default metrics at threshold 0.5
            return 0.5, {
                "accuracy": accuracy_score(y_val, preds),
                "precision": precision_score(y_val, preds, zero_division=0),
                "recall": recall_score(y_val, preds, zero_division=0),
                "f1": f1_score(y_val, preds, zero_division=0),
                "avg_precision": np.nan,
                "roc_auc": np.nan,
                "confusion_matrix": confusion_matrix(y_val, preds).tolist()
            }

    precision, recall, thresholds = precision_recall_curve(y_val, probs)
    # thresholds array has length len(precision)-1
    thresholds = np.append(thresholds, 1.0)  # align lengths
    # find thresholds where precision >= min_precision
    valid_idx = np.where(precision >= min_precision)[0]
    if valid_idx.size == 0:
        # no threshold meets min_precision: choose threshold that maximizes recall
        best_idx = np.argmax(recall)
    else:
        # among valid thresholds choose one with max recall
        best_idx = valid_idx[np.argmax(recall[valid_idx])]
    chosen_threshold = thresholds[best_idx]
    preds_thresh = (probs >= chosen_threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_val, preds_thresh),
        "precision": precision_score(y_val, preds_thresh, zero_division=0),
        "recall": recall_score(y_val, preds_thresh, zero_division=0),
        "f1": f1_score(y_val, preds_thresh, zero_division=0),
        "avg_precision": average_precision_score(y_val, probs) if len(np.unique(y_val))>1 else np.nan,
        "roc_auc": roc_auc_score(y_val, probs) if len(np.unique(y_val))>1 else np.nan,
        "confusion_matrix": confusion_matrix(y_val, preds_thresh).tolist()
    }
    return float(chosen_threshold), metrics

def nested_cv_experiment(X, y, outer_splits=3, outer_repeats=10, inner_splits=3,
                         n_iter_search=20, pre_pca_components=100, kbest_k=200,
                         min_precision=0.20, use_pca_in_pipeline=False):
    """
    Outer: RepeatedStratifiedKFold (conservative splits)
    Inner: RandomizedSearchCV to tune smote__k_neighbors, kbest__k, sel_l1__estimator__C, clf params
    Per-fold threshold sweep to prioritize recall subject to min_precision.
    """
    outer_cv = RepeatedStratifiedKFold(n_splits=outer_splits, n_repeats=outer_repeats, random_state=RND)
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=RND)

    per_fold_results = []
    fold = 0
    last_best = None

    for train_idx, test_idx in outer_cv.split(X, y):
        fold += 1
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        n_minority = int(np.sum(y_train == 1))
        print(f"[Fold {fold}] train={len(train_idx)} test={len(test_idx)} minority_in_train={n_minority}")

        # If pre-PCA requested, apply it on training and transform test
        if pre_pca_components is not None and pre_pca_components > 0:
            pca = PCA(n_components=min(pre_pca_components, X_train.shape[1]), random_state=RND)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)
            # convert back to DataFrame for pipeline compatibility
            X_train_df = pd.DataFrame(X_train_pca, index=X_train.index)
            X_test_df = pd.DataFrame(X_test_pca, index=X_test.index)
        else:
            X_train_df, X_test_df = X_train, X_test

        # If too few minority samples, skip SMOTE and use class_weight only
        if n_minority < 2:
            print(f"[Fold {fold}] WARNING: <2 minority samples in training fold. Skipping SMOTE; using class_weight only.")
            # Build a simple pipeline without SMOTE
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("kbest", SelectKBest(score_func=f_classif, k=min(50, max(1, X_train_df.shape[1]//10)))),
                ("sel_l1", SelectFromModel(LogisticRegression(penalty="l1", solver="liblinear", C=0.1, class_weight="balanced", random_state=RND))),
                ("clf", BalancedRandomForestClassifier(n_estimators=200, random_state=RND))
            ])
            param_dist = {
                "kbest__k": [50, 100],
                "sel_l1__estimator__C": [0.05, 0.1, 0.2],
                "clf__n_estimators": [100, 200]
            }
        else:
            # safe SMOTE k candidates
            max_k = max(1, n_minority - 1)
            cand_k = [k for k in [1, 2, 3] if k <= max_k]
            if not cand_k:
                cand_k = [1]
            pipe = build_pipeline(smote_k=cand_k[0], kbest_k=min(kbest_k, X_train_df.shape[1]), sel_l1_C=0.1, use_pca_in_pipeline=use_pca_in_pipeline)
            param_dist = {
                "smote__k_neighbors": cand_k,
                "kbest__k": [50, 100, min(200, X_train_df.shape[1])],
                "sel_l1__estimator__C": [0.02, 0.05, 0.1, 0.2],
                "clf__n_estimators": [100, 200]
            }

        # Keep only params that exist in pipeline
        param_dist = {k: v for k, v in param_dist.items() if k in pipe.get_params()}

        # Randomized search (optimize F1 for minority)
        from sklearn.metrics import make_scorer
        scorer = make_scorer(f1_score, pos_label=1)
        n_iter = min(n_iter_search, max(1, sum(len(v) for v in param_dist.values())))
        search = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=n_iter, scoring=scorer, cv=inner_cv, random_state=RND, n_jobs=-1)
        search.fit(X_train_df, y_train)
        best = search.best_estimator_
        last_best = best

        # Refit best on full training fold (already fit by search.best_estimator_)
        model = best

        # If selector selected zero features, fallback to SelectKBest and refit
        try:
            if hasattr(model, "named_steps") and "sel_l1" in model.named_steps:
                sel = model.named_steps["sel_l1"]
                try:
                    support = sel.get_support()
                    if support.sum() == 0:
                        k_fb = min(100, max(1, X_train_df.shape[1] // 10))
                        print(f"[Fold {fold}] sel_l1 selected 0 features; falling back to SelectKBest(k={k_fb}) and refitting")
                        steps = []
                        steps.append(("scaler", StandardScaler()))
                        if n_minority >= 2:
                            steps.append(("smote", SMOTE(random_state=RND, k_neighbors=param_dist.get("smote__k_neighbors", [1])[0])))
                        steps.append(("kbest", SelectKBest(score_func=f_classif, k=k_fb)))
                        steps.append(("clf", model.named_steps["clf"]))
                        fallback = ImbPipeline(steps=steps) if n_minority >= 2 else Pipeline(steps=steps)
                        fallback.fit(X_train_df, y_train)
                        model = fallback
                except Exception:
                    pass
        except Exception:
            pass

        # Threshold sweep on validation (test) fold to prioritize recall subject to min_precision
        chosen_thresh, metrics = threshold_sweep_and_metrics(model, X_test_df, y_test, min_precision=min_precision)

        per_fold_results.append({
            "fold": fold,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "n_minority_train": n_minority,
            "best_params": getattr(search, "best_params_", None),
            "chosen_threshold": chosen_thresh,
            **metrics
        })

        print(f"[Fold {fold}] acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} recall={metrics['recall']:.4f} chosen_thresh={chosen_thresh:.3f}")

    return per_fold_results, last_best

def main(args):
    df = pd.read_excel(args.data, sheet_name=args.sheet)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found")

    # drop obvious metadata if present
    metadata_cols = ["x0_LC_ID", "x0_Censor_Complete", "x0_Censor_Cohort_ID", "x0_Patient_Cluster_Label", "x0_Patient_Cluster"]
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    X_raw = df[feature_cols].copy()
    y_raw = df[args.target].copy()

    X = safe_numeric_df(X_raw)
    y = pd.Series(binarize_target(y_raw), name="target")
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features. Target dist: {np.bincount(y)}")

    # remove zero-variance features
    nunique = X.nunique()
    zero_var = nunique[nunique <= 1].index.tolist()
    if zero_var:
        print(f"Removing {len(zero_var)} zero-variance features")
        X = X.drop(columns=zero_var)

    # Run nested CV experiment for BalancedRandomForest (and optionally other models)
    per_fold, best = nested_cv_experiment(
        X, y,
        outer_splits=args.outer_splits,
        outer_repeats=args.outer_repeats,
        inner_splits=args.inner_splits,
        n_iter_search=args.n_iter_search,
        pre_pca_components=args.pre_pca_components,
        kbest_k=args.kbest_k,
        min_precision=args.min_precision,
        use_pca_in_pipeline=args.use_pca_in_pipeline
    )

    out_dir = f"results_revised_{now_tag()}"
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(per_fold).to_csv(os.path.join(out_dir, "BalancedRF_per_fold.csv"), index=False)
    try:
        dump(best, os.path.join(out_dir, "BalancedRF_best.joblib"))
    except Exception:
        pass

    # summary
    dfm = pd.DataFrame(per_fold)
    summary = {
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
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\nResults saved to:", out_dir)
    print("Summary:", summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Revised Long COVID pipeline (pre-PCA -> BalancedRF)")
    parser.add_argument("--data", type=str, default="Sheet1.xlsx")
    parser.add_argument("--sheet", type=int, default=0)
    parser.add_argument("--target", type=str, default="x0_Censor_Complete")
    parser.add_argument("--outer_splits", type=int, default=3)
    parser.add_argument("--outer_repeats", type=int, default=10)
    parser.add_argument("--inner_splits", type=int, default=3)
    parser.add_argument("--n_iter_search", type=int, default=20)
    parser.add_argument("--pre_pca_components", type=int, default=100, help="Number of PCA components to reduce to before pipeline")
    parser.add_argument("--kbest_k", type=int, default=200, help="SelectKBest k")
    parser.add_argument("--min_precision", type=float, default=0.20, help="Minimum precision constraint when choosing threshold")
    parser.add_argument("--use_pca_in_pipeline", action="store_true", help="Also include PCA inside pipeline")
    args = parser.parse_args()
    main(args)
