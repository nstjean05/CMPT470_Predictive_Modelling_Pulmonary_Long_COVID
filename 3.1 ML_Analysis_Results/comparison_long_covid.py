#!/usr/bin/env python3
"""
comparison_long_covid.py

Compare resampling strategies (SMOTE, BorderlineSMOTE, ADASYN, None)
and classifiers (BalancedRandomForest, BalancedBagging, LogisticRegression (class_weight),
optional LightGBM if installed) using nested CV with safe SMOTE usage.

Usage:
    python comparison_long_covid.py --data Sheet1.xlsx --sheet 0

Outputs:
  results_comparison_<timestamp>/ per-combination CSVs and a summary CSV.
"""

import argparse
import os
import json
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score, confusion_matrix
)
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier

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

# Optional LightGBM/XGBoost support
try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

# Build pipeline factory
def build_pipeline(resampler_name, smote_k, kbest_k, sel_l1_C, model_name):
    steps = []
    steps.append(("scaler", StandardScaler()))
    # resampler step (only if not 'none')
    if resampler_name == "smote":
        steps.append(("resample", SMOTE(random_state=RND, k_neighbors=smote_k)))
    elif resampler_name == "borderline":
        steps.append(("resample", BorderlineSMOTE(random_state=RND, k_neighbors=smote_k)))
    elif resampler_name == "adasyn":
        steps.append(("resample", ADASYN(random_state=RND, n_neighbors=max(1, smote_k))))
    # univariate filter
    steps.append(("kbest", SelectKBest(score_func=f_classif, k=kbest_k)))
    # L1 selector
    sel = ("sel_l1", SelectKBest(score_func=f_classif, k=max(10, kbest_k//5)))  # lightweight fallback to avoid zero features
    steps.append(sel)
    # classifier
    if model_name == "BalancedRF":
        clf = ("clf", BalancedRandomForestClassifier(n_estimators=200, random_state=RND))
    elif model_name == "BalancedBagging":
        clf = ("clf", BalancedBaggingClassifier(n_estimators=100, random_state=RND))
    elif model_name == "Logistic":
        clf = ("clf", LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=1000, random_state=RND))
    elif model_name == "LightGBM" and HAS_LGB:
        clf = ("clf", lgb.LGBMClassifier(n_estimators=200, class_weight="balanced", random_state=RND))
    else:
        raise ValueError("Unsupported model or missing LightGBM")
    steps.append(clf)
    return ImbPipeline(steps=steps)

def safe_param_grid(resampler_name, model_name, X_train_shape):
    # safe candidate values; smote k will be constrained per fold
    grid = {
        "kbest__k": [50, 100, min(200, max(50, X_train_shape//10))],
        "sel_l1__k": [max(10, min(50, X_train_shape//20)), max(20, min(100, X_train_shape//10))],
        "resample__k_neighbors": [1, 2, 3] if resampler_name != "none" else [None],
    }
    if model_name in ["BalancedRF", "BalancedBagging"]:
        grid["clf__n_estimators"] = [100, 200]
    if model_name == "LightGBM" and HAS_LGB:
        grid["clf__num_leaves"] = [31, 63]
    return {k: v for k, v in grid.items() if k in ["kbest__k", "sel_l1__k", "resample__k_neighbors", "clf__n_estimators", "clf__num_leaves"]}

def evaluate_combination(X, y, resampler_name, model_name, outer_splits=3, outer_repeats=5, inner_splits=3, n_iter_search=20, pre_pca_components=100):
    outer_cv = RepeatedStratifiedKFold(n_splits=outer_splits, n_repeats=outer_repeats, random_state=RND)
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=RND)
    per_fold = []
    fold = 0
    last_best = None

    for train_idx, test_idx in outer_cv.split(X, y):
        fold += 1
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        n_minority = int(np.sum(y_train == 1))
        # pre-PCA if requested
        if pre_pca_components and pre_pca_components > 0:
            pca = PCA(n_components=min(pre_pca_components, X_train.shape[1]), random_state=RND)
            X_train_pca = pd.DataFrame(pca.fit_transform(X_train), index=X_train.index)
            X_test_pca = pd.DataFrame(pca.transform(X_test), index=X_test.index)
        else:
            X_train_pca, X_test_pca = X_train, X_test

        # determine safe smote_k
        if n_minority < 2 or resampler_name == "none":
            smote_k_candidates = [None]
        else:
            max_k = max(1, n_minority - 1)
            smote_k_candidates = [k for k in [1, 2, 3] if k <= max_k]
            if not smote_k_candidates:
                smote_k_candidates = [1]

        # build pipeline with first safe smote_k (or none)
        smote_k0 = smote_k_candidates[0] if smote_k_candidates[0] is not None else 1
        pipe = build_pipeline(resampler_name if resampler_name!="none" else "none", smote_k0, kbest_k=min(200, X_train_pca.shape[1]), sel_l1_C=0.1, model_name=model_name)

        # param grid
        param_grid = safe_param_grid(resampler_name, model_name, X_train_pca.shape[1])
        # restrict resample__k_neighbors to safe candidates if present
        if "resample__k_neighbors" in param_grid:
            param_grid["resample__k_neighbors"] = smote_k_candidates

        # filter keys to those present in pipeline
        param_grid = {k: v for k, v in param_grid.items() if k in pipe.get_params()}

        # scorer: optimize F1 for minority
        from sklearn.metrics import make_scorer
        scorer = make_scorer(f1_score, pos_label=1)

        # RandomizedSearchCV
        n_iter = min(n_iter_search, max(1, sum(len(v) for v in param_grid.values())))
        search = RandomizedSearchCV(pipe, param_distributions=param_grid, n_iter=n_iter, scoring=scorer, cv=inner_cv, random_state=RND, n_jobs=-1)
        # fit
        try:
            search.fit(X_train_pca, y_train)
            best = search.best_estimator_
        except Exception as e:
            # fallback: fit default pipeline without search
            print(f"[Fold {fold}] search failed: {e}; fitting default pipeline")
            pipe.fit(X_train_pca, y_train)
            best = pipe

        last_best = best

        # evaluate on test
        y_pred = best.predict(X_test_pca)
        if hasattr(best, "predict_proba"):
            y_proba = best.predict_proba(X_test_pca)[:, 1]
        else:
            y_proba = y_pred

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        ap = average_precision_score(y_test, y_proba) if len(np.unique(y_test))>1 else np.nan
        roc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test))>1 else np.nan
        cm = confusion_matrix(y_test, y_pred)

        per_fold.append({
            "fold": fold,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "n_minority_train": n_minority,
            "best_params": getattr(search, "best_params_", None),
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "avg_precision": ap, "roc_auc": roc, "confusion_matrix": cm.tolist()
        })

        print(f"[{resampler_name}/{model_name}] Fold {fold} acc={acc:.4f} recall={rec:.4f} f1={f1:.4f} ap={ap:.4f}")

    return per_fold, last_best

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

    # pre-PCA reduction (optional)
    if args.pre_pca and args.pre_pca_components > 0:
        print(f"Applying pre-PCA to {args.pre_pca_components} components")
        pca = PCA(n_components=min(args.pre_pca_components, X.shape[1]), random_state=RND)
        X_pca = pd.DataFrame(pca.fit_transform(X), index=X.index)
        X = X_pca

    resamplers = ["smote", "borderline", "adasyn", "none"]
    models = ["BalancedRF", "BalancedBagging", "Logistic"]
    if HAS_LGB:
        models.append("LightGBM")

    out_dir = f"results_comparison_{now_tag()}"
    os.makedirs(out_dir, exist_ok=True)
    summary_rows = []

    for res in resamplers:
        for mod in models:
            print("\n" + "="*60)
            print(f"Running combination: resampler={res} model={mod}")
            per_fold, best = evaluate_combination(
                X, y,
                resampler_name=res,
                model_name=mod,
                outer_splits=args.outer_splits,
                outer_repeats=args.outer_repeats,
                inner_splits=args.inner_splits,
                n_iter_search=args.n_iter_search,
                pre_pca_components=args.pre_pca_components if args.pre_pca else 0
            )
            # save per-fold
            combo_name = f"{res}__{mod}"
            pd.DataFrame(per_fold).to_csv(os.path.join(out_dir, f"{combo_name}_per_fold.csv"), index=False)
            try:
                dump(best, os.path.join(out_dir, f"{combo_name}_best.joblib"))
            except Exception:
                pass
            # aggregate
            dfm = pd.DataFrame(per_fold)
            summary = {
                "resampler": res,
                "model": mod,
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

    pd.DataFrame(summary_rows).to_csv(os.path.join(out_dir, "comparison_summary.csv"), index=False)
    print("\nAll comparison results saved to:", out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare resampling and models for Long COVID")
    parser.add_argument("--data", type=str, default="Sheet1.xlsx")
    parser.add_argument("--sheet", type=int, default=0)
    parser.add_argument("--target", type=str, default="x0_Censor_Complete")
    parser.add_argument("--outer_splits", type=int, default=3)
    parser.add_argument("--outer_repeats", type=int, default=5)
    parser.add_argument("--inner_splits", type=int, default=3)
    parser.add_argument("--n_iter_search", type=int, default=20)
    parser.add_argument("--pre_pca", action="store_true", help="Apply pre-PCA before pipeline")
    parser.add_argument("--pre_pca_components", type=int, default=100)
    args = parser.parse_args()
    main(args)
