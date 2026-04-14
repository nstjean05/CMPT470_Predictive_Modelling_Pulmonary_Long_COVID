#!/usr/bin/env python3
"""
long_covid_analysis.py

Honest analysis for extreme class imbalance (7 positives / 185 samples).
Primary evaluation: LOOCV with probability calibration and Wilcoxon rank-sum test.
Secondary evaluation: Nested CV (outer RepeatedStratifiedKFold) for variance estimation.

Key design decisions vs. previous version:
  - SMOTE removed as primary strategy; with n_minority ~4-5 per fold it interpolates
    between near-duplicates and inflates apparent performance.
  - class_weight='balanced' (and tunable cost ratio) is the main imbalance handler.
  - Variance filtering applied BEFORE any CV split (unsupervised, no leakage).
  - LOOCV is used as primary metric because it uses all 7 positives as test cases.
  - Reports predicted probability distributions (positives vs negatives) and
    Wilcoxon rank-sum p-value — more statistically honest than F1 on 7 samples.
  - One simple default pipeline + optional SMOTE-inside-fold as secondary comparison.

Usage:
    python long_covid_analysis.py --data Sheet1.xlsx --sheet 0
"""

import argparse
import os
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import dump
from scipy.stats import ranksums

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score, confusion_matrix,
    precision_recall_curve, brier_score_loss
)
from sklearn.model_selection import (
    LeaveOneOut, RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

try:
    from imblearn.over_sampling import SMOTE, BorderlineSMOTE
    from imblearn.ensemble import BalancedRandomForestClassifier
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("WARNING: imbalanced-learn not available. SMOTE variants disabled.")

warnings.filterwarnings("ignore")
RND = 42

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def now_tag():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def safe_numeric_df(df):
    df_num = df.apply(pd.to_numeric, errors="coerce")
    if df_num.isnull().values.any():
        df_num = df_num.fillna(df_num.median())
    return df_num

def binarize_target(y_raw):
    return (y_raw.astype(float) > 0).astype(int)

def variance_filter(X, threshold=0.01):
    """
    Remove near-zero-variance features BEFORE any CV split.
    This is unsupervised and does not leak label information.
    """
    sel = VarianceThreshold(threshold=threshold)
    X_filtered = sel.fit_transform(X)
    kept = X.columns[sel.get_support()]
    print(f"  Variance filter: {X.shape[1]} → {X_filtered.shape[1]} features (threshold={threshold})")
    return pd.DataFrame(X_filtered, columns=kept, index=X.index)

# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------

def build_logistic_pipeline(pca_components, kbest_k, l1_C, class_weight_ratio):
    """
    Simple, low-variance pipeline:
    StandardScaler → PCA → SelectKBest → L1 feature selection → Logistic
    No SMOTE. Handles imbalance via class_weight.
    """
    cw = {0: 1, 1: class_weight_ratio}
    steps = [
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=pca_components, random_state=RND)),
        ("kbest", SelectKBest(score_func=f_classif, k=min(kbest_k, pca_components))),
        ("sel_l1", SelectFromModel(
            LogisticRegression(penalty="l1", solver="liblinear", C=l1_C,
                               class_weight="balanced", random_state=RND, max_iter=1000)
        )),
        ("clf", LogisticRegression(
            solver="liblinear", class_weight=cw,
            max_iter=1000, random_state=RND
        )),
    ]
    return Pipeline(steps=steps)

def build_brf_pipeline(pca_components, kbest_k):
    """
    BalancedRandomForest with pre-PCA. No SMOTE.
    """
    if not IMBLEARN_AVAILABLE:
        raise RuntimeError("imblearn required for BalancedRF")
    steps = [
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=pca_components, random_state=RND)),
        ("kbest", SelectKBest(score_func=f_classif, k=min(kbest_k, pca_components))),
        ("clf", BalancedRandomForestClassifier(
            n_estimators=200, random_state=RND, n_jobs=-1
        )),
    ]
    return ImbPipeline(steps=steps)

def build_smote_pipeline(pca_components, kbest_k, l1_C, smote_k):
    """
    SMOTE pipeline for secondary comparison. Used only when n_minority_train >= 3.
    SMOTE is the second step (after scaler, before feature selection) so it
    operates only on training data inside the fold.
    """
    if not IMBLEARN_AVAILABLE:
        raise RuntimeError("imblearn required for SMOTE pipeline")
    steps = [
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=pca_components, random_state=RND)),
        ("resample", SMOTE(random_state=RND, k_neighbors=smote_k)),
        ("kbest", SelectKBest(score_func=f_classif, k=min(kbest_k, pca_components))),
        ("sel_l1", SelectFromModel(
            LogisticRegression(penalty="l1", solver="liblinear", C=l1_C,
                               class_weight="balanced", random_state=RND, max_iter=1000)
        )),
        ("clf", LogisticRegression(
            solver="liblinear", class_weight="balanced",
            max_iter=1000, random_state=RND
        )),
    ]
    return ImbPipeline(steps=steps)

# ---------------------------------------------------------------------------
# LOOCV evaluation (primary)
# ---------------------------------------------------------------------------

def loocv_evaluation(X, y, pipeline, pipeline_name):
    """
    Leave-one-out CV. Returns per-sample predicted probabilities.
    With 185 samples this is fast, and crucially gives 7 positive test cases.

    Key output: probability distribution for positives vs negatives,
    plus Wilcoxon rank-sum test to assess separation.
    """
    print(f"\n--- LOOCV: {pipeline_name} ---")
    loo = LeaveOneOut()
    probs = np.zeros(len(y))
    failed_folds = 0

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        n_minority = int(np.sum(y_train == 1))
        if n_minority < 2:
            # Cannot fit meaningfully; assign 0.5 (uninformative)
            probs[test_idx] = 0.5
            failed_folds += 1
            continue

        # For SMOTE pipeline: clamp k_neighbors to safe value
        if "resample" in [s[0] for s in pipeline.steps]:
            safe_k = max(1, min(2, n_minority - 1))
            try:
                pipeline.set_params(resample__k_neighbors=safe_k)
            except Exception:
                pass

        # Adjust PCA components to training size
        n_train_features = X_train.shape[1]
        for step_name, step_obj in pipeline.steps:
            if isinstance(step_obj, PCA):
                new_n = min(step_obj.n_components, X_train.shape[0] - 1, n_train_features)
                pipeline.set_params(**{f"{step_name}__n_components": new_n})
            if isinstance(step_obj, SelectKBest):
                new_k = min(step_obj.k, n_train_features)
                pipeline.set_params(**{f"{step_name}__k": new_k})

        try:
            pipeline.fit(X_train, y_train)
            probs[test_idx] = pipeline.predict_proba(X_test)[:, 1]
        except Exception as e:
            probs[test_idx] = 0.5
            failed_folds += 1

    if failed_folds > 0:
        print(f"  WARNING: {failed_folds} folds used uninformative fallback (p=0.5)")

    pos_probs = probs[y == 1]
    neg_probs = probs[y == 0]

    # Wilcoxon rank-sum: are positive probabilities significantly higher?
    stat, p_val = ranksums(pos_probs, neg_probs, alternative="greater")

    # Threshold sweep over LOOCV probabilities
    precision, recall_vals, thresholds = precision_recall_curve(y, probs)
    thresholds = np.append(thresholds, 1.0)
    valid = np.where(precision >= 0.20)[0]
    if valid.size > 0:
        best_idx = valid[np.argmax(recall_vals[valid])]
    else:
        best_idx = np.argmax(recall_vals)
    best_thr = float(thresholds[best_idx])
    preds = (probs >= best_thr).astype(int)

    metrics = {
        "pipeline": pipeline_name,
        "roc_auc": float(roc_auc_score(y, probs)),
        "avg_precision": float(average_precision_score(y, probs)),
        "brier_score": float(brier_score_loss(y, probs)),
        "wilcoxon_stat": float(stat),
        "wilcoxon_p": float(p_val),
        "pos_prob_mean": float(pos_probs.mean()),
        "pos_prob_std": float(pos_probs.std()),
        "neg_prob_mean": float(neg_probs.mean()),
        "neg_prob_std": float(neg_probs.std()),
        "best_threshold": best_thr,
        "recall_at_threshold": float(recall_score(y, preds, zero_division=0)),
        "precision_at_threshold": float(precision_score(y, preds, zero_division=0)),
        "f1_at_threshold": float(f1_score(y, preds, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, preds).tolist(),
        "positive_sample_probs": pos_probs.tolist(),
    }

    print(f"  ROC-AUC:       {metrics['roc_auc']:.4f}")
    print(f"  Avg Precision: {metrics['avg_precision']:.4f}")
    print(f"  Brier Score:   {metrics['brier_score']:.4f}  (lower is better; baseline={y.mean()*(1-y.mean()):.3f})")
    print(f"  Wilcoxon p:    {p_val:.4f}  ({'significant at 0.05' if p_val < 0.05 else 'not significant'})")
    print(f"  Pos prob distribution: mean={pos_probs.mean():.3f}, std={pos_probs.std():.3f}")
    print(f"  Neg prob distribution: mean={neg_probs.mean():.3f}, std={neg_probs.std():.3f}")
    print(f"  Per-positive predicted probabilities: {[f'{p:.3f}' for p in pos_probs]}")
    print(f"  Recall @ threshold {best_thr:.3f}: {metrics['recall_at_threshold']:.3f}  (F1={metrics['f1_at_threshold']:.3f})")

    return metrics, probs

# ---------------------------------------------------------------------------
# Nested CV evaluation (secondary, for variance estimation only)
# ---------------------------------------------------------------------------

def nested_cv_evaluation(X, y, pipeline_fn, pipeline_name,
                         outer_splits=3, outer_repeats=5, inner_splits=2):
    """
    Nested CV for variance estimation across folds.
    Uses a pipeline factory function (pipeline_fn) called fresh per fold
    to avoid state leakage between folds.

    Reports recall/F1/AP per fold so instability is visible.
    """
    print(f"\n--- Nested CV: {pipeline_name} ---")
    outer_cv = RepeatedStratifiedKFold(
        n_splits=outer_splits, n_repeats=outer_repeats, random_state=RND
    )
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=RND)

    fold_results = []
    fold = 0

    for train_idx, test_idx in outer_cv.split(X, y):
        fold += 1
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        n_minority = int(np.sum(y_train == 1))

        # Build a fresh pipeline for this fold
        pipe = pipeline_fn(n_minority_train=n_minority, n_train_cols=X_train.shape[1])
        if pipe is None:
            continue

        try:
            pipe.fit(X_train, y_train)
            if hasattr(pipe, "predict_proba"):
                probs = pipe.predict_proba(X_test)[:, 1]
            else:
                probs = pipe.predict(X_test).astype(float)

            if len(np.unique(y_test)) < 2:
                # test fold has no positives — record but flag
                fold_results.append({
                    "fold": fold, "n_minority_train": n_minority,
                    "recall": np.nan, "f1": np.nan, "avg_precision": np.nan,
                    "roc_auc": np.nan, "note": "no_positives_in_test"
                })
                continue

            precision_arr, recall_arr, thresholds = precision_recall_curve(y_test, probs)
            thresholds = np.append(thresholds, 1.0)
            valid = np.where(precision_arr >= 0.20)[0]
            if valid.size > 0:
                best_idx = valid[np.argmax(recall_arr[valid])]
            else:
                best_idx = np.argmax(recall_arr)
            best_thr = float(thresholds[best_idx])
            preds = (probs >= best_thr).astype(int)

            fold_results.append({
                "fold": fold,
                "n_minority_train": n_minority,
                "recall": float(recall_score(y_test, preds, zero_division=0)),
                "precision": float(precision_score(y_test, preds, zero_division=0)),
                "f1": float(f1_score(y_test, preds, zero_division=0)),
                "avg_precision": float(average_precision_score(y_test, probs)),
                "roc_auc": float(roc_auc_score(y_test, probs)),
                "threshold": best_thr,
                "note": "ok"
            })
        except Exception as e:
            fold_results.append({
                "fold": fold, "n_minority_train": n_minority,
                "recall": np.nan, "f1": np.nan, "avg_precision": np.nan,
                "roc_auc": np.nan, "note": f"error: {e}"
            })

    df = pd.DataFrame(fold_results)
    ok = df[df["note"] == "ok"]

    summary = {
        "pipeline": pipeline_name,
        "n_folds_total": len(df),
        "n_folds_with_positives": len(ok),
        "recall_mean": float(ok["recall"].mean()) if len(ok) else np.nan,
        "recall_std": float(ok["recall"].std()) if len(ok) else np.nan,
        "f1_mean": float(ok["f1"].mean()) if len(ok) else np.nan,
        "f1_std": float(ok["f1"].std()) if len(ok) else np.nan,
        "avg_precision_mean": float(ok["avg_precision"].mean()) if len(ok) else np.nan,
        "avg_precision_std": float(ok["avg_precision"].std()) if len(ok) else np.nan,
        "roc_auc_mean": float(ok["roc_auc"].mean()) if len(ok) else np.nan,
        "roc_auc_std": float(ok["roc_auc"].std()) if len(ok) else np.nan,
    }

    print(f"  Folds with test positives: {summary['n_folds_with_positives']}/{summary['n_folds_total']}")
    print(f"  Recall:    {summary['recall_mean']:.3f} ± {summary['recall_std']:.3f}")
    print(f"  F1:        {summary['f1_mean']:.3f} ± {summary['f1_std']:.3f}")
    print(f"  Avg Prec:  {summary['avg_precision_mean']:.3f} ± {summary['avg_precision_std']:.3f}")
    print(f"  ROC-AUC:   {summary['roc_auc_mean']:.3f} ± {summary['roc_auc_std']:.3f}")

    return summary, df

# ---------------------------------------------------------------------------
# Pipeline factory functions for nested CV
# ---------------------------------------------------------------------------

def make_logistic_factory(pca=50, kbest=100, l1_C=0.1, cost_ratio=20):
    def factory(n_minority_train, n_train_cols):
        pca_n = min(pca, n_train_cols, 184)  # hard cap at n_train - 1 = 184
        k = min(kbest, pca_n)
        return build_logistic_pipeline(pca_n, k, l1_C, cost_ratio)
    return factory

def make_brf_factory(pca=50, kbest=50):
    def factory(n_minority_train, n_train_cols):
        if not IMBLEARN_AVAILABLE:
            return None
        pca_n = min(pca, n_train_cols, 184)
        k = min(kbest, pca_n)
        return build_brf_pipeline(pca_n, k)
    return factory

def make_smote_factory(pca=50, kbest=100, l1_C=0.1):
    def factory(n_minority_train, n_train_cols):
        if not IMBLEARN_AVAILABLE or n_minority_train < 3:
            # Fall back to logistic without SMOTE
            return build_logistic_pipeline(
                min(pca, n_train_cols, 184),
                min(kbest, min(pca, n_train_cols, 184)),
                l1_C, 20
            )
        smote_k = max(1, min(2, n_minority_train - 1))
        pca_n = min(pca, n_train_cols, 184)
        k = min(kbest, pca_n)
        return build_smote_pipeline(pca_n, k, l1_C, smote_k)
    return factory

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    # --- Load data ---
    df = pd.read_excel(args.data, sheet_name=args.sheet)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found. Columns: {list(df.columns)}")

    metadata_cols = [
        "x0_LC_ID", "x0_Censor_Complete", "x0_Censor_Cohort_ID",
        "x0_Patient_Cluster_Label", "x0_Patient_Cluster"
    ]
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    X_raw = df[feature_cols].copy()
    y_raw = df[args.target].copy()

    X = safe_numeric_df(X_raw)
    y = pd.Series(binarize_target(y_raw), name="target")

    counts = np.bincount(y)
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {counts[0]} negative, {counts[1]} positive")
    print(f"Imbalance ratio: {counts[0]/counts[1]:.1f}:1")
    print(f"\nNOTE: With {counts[1]} positives, all metrics have very wide confidence")
    print("intervals. Wilcoxon p-value and probability distribution are more")
    print("reliable than F1/recall point estimates.\n")

    # --- Unsupervised pre-filtering (no leakage) ---
    print("=== Pre-filtering (unsupervised, outside CV) ===")
    X = variance_filter(X, threshold=0.01)

    # --- Output directory ---
    out_dir = f"results_honest_{now_tag()}"
    os.makedirs(out_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Define LOOCV pipelines (fixed hyperparameters — not tuned to avoid
    # overfitting on 7 positives)
    # -----------------------------------------------------------------------
    loocv_pipelines = {
        "Logistic_PCA50_cost20": build_logistic_pipeline(
            pca_components=min(50, X.shape[1]),
            kbest_k=50,
            l1_C=0.1,
            class_weight_ratio=20
        ),
        "Logistic_PCA50_cost25": build_logistic_pipeline(
            pca_components=min(50, X.shape[1]),
            kbest_k=50,
            l1_C=0.1,
            class_weight_ratio=25
        ),
        "Logistic_PCA100_cost20": build_logistic_pipeline(
            pca_components=min(100, X.shape[1]),
            kbest_k=100,
            l1_C=0.5,
            class_weight_ratio=20
        ),
    }
    if IMBLEARN_AVAILABLE:
        loocv_pipelines["BalancedRF_PCA50"] = build_brf_pipeline(
            pca_components=min(50, X.shape[1]),
            kbest_k=50
        )
        loocv_pipelines["SMOTE_Logistic_PCA50"] = build_smote_pipeline(
            pca_components=min(50, X.shape[1]),
            kbest_k=50,
            l1_C=0.1,
            smote_k=1
        )

    # -----------------------------------------------------------------------
    # PRIMARY: LOOCV
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("PRIMARY EVALUATION: LOOCV")
    print("="*60)

    loocv_results = []
    all_loocv_probs = {}

    for name, pipe in loocv_pipelines.items():
        metrics, probs = loocv_evaluation(X, y, pipe, name)
        loocv_results.append(metrics)
        all_loocv_probs[name] = probs.tolist()

    loocv_df = pd.DataFrame(loocv_results)
    loocv_df.to_csv(os.path.join(out_dir, "loocv_summary.csv"), index=False)

    # Save per-sample probabilities for all LOOCV pipelines
    prob_df = pd.DataFrame(all_loocv_probs)
    prob_df["true_label"] = y.values
    prob_df.to_csv(os.path.join(out_dir, "loocv_probabilities.csv"), index=False)

    # -----------------------------------------------------------------------
    # SECONDARY: Nested CV (variance estimation only)
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("SECONDARY EVALUATION: Nested CV (variance estimation)")
    print("="*60)

    nested_factories = {
        "Logistic_PCA50_cost20": make_logistic_factory(pca=50, kbest=50, l1_C=0.1, cost_ratio=20),
        "BalancedRF_PCA50": make_brf_factory(pca=50, kbest=50),
        "SMOTE_Logistic_PCA50": make_smote_factory(pca=50, kbest=50, l1_C=0.1),
    }

    nested_summaries = []
    nested_fold_dfs = []

    for name, factory_fn in nested_factories.items():
        summary, fold_df = nested_cv_evaluation(
            X, y, factory_fn, name,
            outer_splits=args.outer_splits,
            outer_repeats=args.outer_repeats,
            inner_splits=args.inner_splits
        )
        nested_summaries.append(summary)
        fold_df["pipeline"] = name
        nested_fold_dfs.append(fold_df)

    pd.DataFrame(nested_summaries).to_csv(
        os.path.join(out_dir, "nested_cv_summary.csv"), index=False
    )
    pd.concat(nested_fold_dfs).to_csv(
        os.path.join(out_dir, "nested_cv_per_fold.csv"), index=False
    )

    # -----------------------------------------------------------------------
    # Final comparison table
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("FINAL COMPARISON (LOOCV — primary)")
    print("="*60)
    compare_cols = [
        "pipeline", "roc_auc", "avg_precision", "brier_score",
        "wilcoxon_p", "pos_prob_mean", "neg_prob_mean",
        "recall_at_threshold", "f1_at_threshold"
    ]
    print(loocv_df[compare_cols].sort_values("avg_precision", ascending=False).to_string(index=False))

    print("\n" + "="*60)
    print("INTERPRETATION NOTES")
    print("="*60)
    print(f"  n_positive = {counts[1]}. No metric is reliable in isolation.")
    print("  Prioritize: wilcoxon_p < 0.05 AND pos_prob_mean >> neg_prob_mean")
    print("  ROC-AUC and avg_precision are more robust than F1 with this sample size.")
    print("  Per-positive probabilities (in loocv_summary.csv) show which patients")
    print("  the model consistently identifies vs. misses.")
    print("  The nested CV std values show variance — if recall_std > recall_mean,")
    print("  the model is not reliably detecting positives across folds.\n")

    # Save full run metadata
    meta = {
        "timestamp": now_tag(),
        "n_samples": int(X.shape[0]),
        "n_features_after_filter": int(X.shape[1]),
        "n_positive": int(counts[1]),
        "n_negative": int(counts[0]),
        "outer_cv_splits": args.outer_splits,
        "outer_cv_repeats": args.outer_repeats,
        "inner_cv_splits": args.inner_splits,
        "target_col": args.target,
    }
    with open(os.path.join(out_dir, "run_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"All outputs saved to: {out_dir}/")
    print("  loocv_summary.csv         — primary results per pipeline")
    print("  loocv_probabilities.csv   — per-sample predicted probs (key diagnostic)")
    print("  nested_cv_summary.csv     — secondary variance estimates")
    print("  nested_cv_per_fold.csv    — per-fold breakdown")
    print("  run_metadata.json         — run configuration")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Honest Long COVID ML analysis")
    parser.add_argument("--data",         type=str,   default="Sheet1.xlsx")
    parser.add_argument("--sheet",        type=int,   default=0)
    parser.add_argument("--target",       type=str,   default="x0_Censor_Complete")
    parser.add_argument("--outer_splits", type=int,   default=3)
    parser.add_argument("--outer_repeats",type=int,   default=5)
    parser.add_argument("--inner_splits", type=int,   default=2)
    args = parser.parse_args()
    main(args)