#!/usr/bin/env python3
"""
focused_comparison_ready.py

Focused comparison: LogisticRegression + {SMOTE, BorderlineSMOTE} vs BalancedRandomForest baseline.
Pre-PCA grid, SelectKBest grid, safe per-fold resampling, nested CV, threshold sweep.

Usage:
    python focused_comparison_ready.py --data Sheet1.xlsx --sheet 0
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
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score, confusion_matrix, precision_recall_curve
)
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline

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

def threshold_sweep(probs, y_true, min_precision=0.20):
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    thresholds = np.append(thresholds, 1.0)
    valid = np.where(precision >= min_precision)[0]
    if valid.size == 0:
        best_idx = np.argmax(recall)
    else:
        best_idx = valid[np.argmax(recall[valid])]
    thr = float(thresholds[best_idx])
    preds = (probs >= thr).astype(int)
    return thr, {
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "avg_precision": average_precision_score(y_true, probs) if len(np.unique(y_true))>1 else np.nan,
        "roc_auc": roc_auc_score(y_true, probs) if len(np.unique(y_true))>1 else np.nan,
        "confusion_matrix": confusion_matrix(y_true, preds).tolist()
    }

def build_pipeline(resampler, smote_k, kbest_k, sel_l1_C, model_name):
    steps = [("scaler", StandardScaler())]
    if resampler == "smote":
        steps.append(("resample", SMOTE(random_state=RND, k_neighbors=smote_k)))
    elif resampler == "borderline":
        steps.append(("resample", BorderlineSMOTE(random_state=RND, k_neighbors=smote_k)))
    steps.append(("kbest", SelectKBest(score_func=f_classif, k=kbest_k)))
    steps.append(("sel_l1", SelectFromModel(LogisticRegression(penalty="l1", solver="liblinear", C=sel_l1_C, class_weight="balanced", random_state=RND))))
    if model_name == "Logistic":
        steps.append(("clf", LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=1000, random_state=RND)))
        return ImbPipeline(steps=steps)
    elif model_name == "BalancedRF":
        steps.append(("clf", BalancedRandomForestClassifier(n_estimators=200, random_state=RND)))
        return ImbPipeline(steps=steps)
    else:
        raise ValueError("Unsupported model")

def safe_param_grid(model_name, X_train_cols):
    grid = {
        "kbest__k": [100, 200, min(500, max(100, X_train_cols//10))],
        "sel_l1__estimator__C": [0.1, 0.5, 1.0],
        "resample__k_neighbors": [1, 2]
    }
    if model_name == "BalancedRF":
        grid["clf__n_estimators"] = [100, 200]
    # filter keys later against pipeline.get_params()
    return grid

def focused_experiment(X, y, resamplers, models, pre_pca_list, outer_splits=3, outer_repeats=5, inner_splits=2, n_iter_search=20, min_precision=0.20):
    outer_cv = RepeatedStratifiedKFold(n_splits=outer_splits, n_repeats=outer_repeats, random_state=RND)
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=RND)

    results = []
    best_models = {}

    for res in resamplers:
        for mod in models:
            combo_name = f"{res}__{mod}"
            per_fold = []
            last_best = None
            print(f"\n=== Running combo: {combo_name} ===")
            for pre_pca in pre_pca_list:
                print(f" pre-PCA components: {pre_pca}")
                fold = 0
                for train_idx, test_idx in outer_cv.split(X, y):
                    fold += 1
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    n_minority = int(np.sum(y_train == 1))
                    # apply pre-PCA if requested
                    if pre_pca and pre_pca > 0:
                        pca = PCA(n_components=min(pre_pca, X_train.shape[1]), random_state=RND)
                        X_train_p = pd.DataFrame(pca.fit_transform(X_train), index=X_train.index)
                        X_test_p = pd.DataFrame(pca.transform(X_test), index=X_test.index)
                    else:
                        X_train_p, X_test_p = X_train, X_test

                    # decide whether to include resampler
                    if n_minority < 2 or res == "none":
                        # build pipeline without resampler
                        pipe = build_pipeline(resampler=None, smote_k=1, kbest_k=min(200, X_train_p.shape[1]), sel_l1_C=0.1, model_name=mod)
                        # remove resample param from grid
                        param_grid = safe_param_grid(mod, X_train_p.shape[1])
                        param_grid = {k: v for k, v in param_grid.items() if not k.startswith("resample")}
                    else:
                        # safe smote k candidates
                        max_k = max(1, n_minority - 1)
                        cand_k = [k for k in [1, 2] if k <= max_k]
                        if not cand_k:
                            cand_k = [1]
                        pipe = build_pipeline(resampler=res, smote_k=cand_k[0], kbest_k=min(200, X_train_p.shape[1]), sel_l1_C=0.1, model_name=mod)
                        param_grid = safe_param_grid(mod, X_train_p.shape[1])
                        param_grid["resample__k_neighbors"] = cand_k

                    # filter param_grid to keys present in pipeline
                    param_grid = {k: v for k, v in param_grid.items() if k in pipe.get_params()}

                    # RandomizedSearchCV with F1 (minority) scorer
                    from sklearn.metrics import make_scorer
                    scorer = make_scorer(f1_score, pos_label=1)
                    n_iter = min(n_iter_search, max(1, sum(len(v) for v in param_grid.values())))
                    search = RandomizedSearchCV(pipe, param_distributions=param_grid, n_iter=n_iter, scoring=scorer, cv=inner_cv, random_state=RND, n_jobs=-1, error_score=np.nan)
                    try:
                        search.fit(X_train_p, y_train)
                        best = search.best_estimator_
                    except Exception as e:
                        # fallback: fit default pipeline
                        print(f"[Fold {fold}] inner search failed: {e}; fitting default pipeline")
                        pipe.fit(X_train_p, y_train)
                        best = pipe

                    last_best = best

                    # Evaluate and threshold sweep
                    if hasattr(best, "predict_proba"):
                        probs = best.predict_proba(X_test_p)[:, 1]
                    else:
                        # fallback: use predict
                        preds = best.predict(X_test_p)
                        probs = preds

                    thr, metrics = threshold_sweep(probs, y_test, min_precision=min_precision)
                    per_fold.append({
                        "combo": combo_name,
                        "pre_pca": pre_pca,
                        "fold": fold,
                        "train_size": len(train_idx),
                        "test_size": len(test_idx),
                        "n_minority_train": n_minority,
                        "best_params": getattr(search, "best_params_", None),
                        "chosen_threshold": thr,
                        **metrics
                    })
                    print(f"[{combo_name}][PCA={pre_pca}] Fold {fold} acc={metrics['accuracy']:.3f} recall={metrics['recall']:.3f} f1={metrics['f1']:.3f} thr={thr:.3f}")

            # aggregate and save
            results.extend(per_fold)
            best_models[combo_name] = last_best

    return results, best_models

def main(args):
    df = pd.read_excel(args.data, sheet_name=args.sheet)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found")
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

    # Focused combos
    resamplers = ["smote", "borderline", "none"]
    models = ["Logistic", "BalancedRF"]
    pre_pca_list = [50, 100, 200]

    results, best_models = focused_experiment(
        X, y,
        resamplers=resamplers,
        models=models,
        pre_pca_list=pre_pca_list,
        outer_splits=args.outer_splits,
        outer_repeats=args.outer_repeats,
        inner_splits=args.inner_splits,
        n_iter_search=args.n_iter_search,
        min_precision=args.min_precision
    )

    out_dir = f"results_focused_{now_tag()}"
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(results).to_csv(os.path.join(out_dir, "focused_per_fold.csv"), index=False)
    # aggregate summary
    dfm = pd.DataFrame(results)
    summary = dfm.groupby(["combo", "pre_pca"]).agg({
        "accuracy": ["mean", "std"],
        "recall": ["mean", "std"],
        "f1": ["mean", "std"],
        "avg_precision": ["mean", "std"],
        "roc_auc": ["mean", "std"]
    })
    summary.to_csv(os.path.join(out_dir, "focused_summary.csv"))
    # save best models
    for k, m in best_models.items():
        try:
            dump(m, os.path.join(out_dir, f"{k}_best.joblib"))
        except Exception:
            pass

    print("\nResults saved to:", out_dir)
    print("Top combos by recall_mean (quick):")
    if not dfm.empty:
        quick = dfm.groupby(["combo", "pre_pca"])["recall"].mean().reset_index().sort_values("recall", ascending=False)
        print(quick.head(10).to_string(index=False))
    else:
        print("No results recorded.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Focused comparison ready to run")
    parser.add_argument("--data", type=str, default="Sheet1.xlsx")
    parser.add_argument("--sheet", type=int, default=0)
    parser.add_argument("--target", type=str, default="x0_Censor_Complete")
    parser.add_argument("--outer_splits", type=int, default=3)
    parser.add_argument("--outer_repeats", type=int, default=5)
    parser.add_argument("--inner_splits", type=int, default=2)
    parser.add_argument("--n_iter_search", type=int, default=20)
    parser.add_argument("--min_precision", type=float, default=0.20)
    args = parser.parse_args()
    main(args)
