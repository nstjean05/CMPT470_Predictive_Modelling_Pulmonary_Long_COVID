
#!/usr/bin/env python3
import warnings
warnings.filterwarnings('ignore')

import os
import json
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from imblearn.ensemble import BalancedRandomForestClassifier
    HAS_BRF = True
except Exception:
    HAS_BRF = False

RND = 42
OUTER_SPLITS = 5
OUTER_REPEATS = 3
INNER_SPLITS = 3
N_ITER_SEARCH = 20
TOP_K_MAX = 100
MIN_FEATURES_FALLBACK = 20


def now_tag():
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def label_long_covid(lc_id):
    if not isinstance(lc_id, str):
        return np.nan
    lc_id = lc_id.strip()
    if lc_id.endswith('.C') or lc_id.endswith('.CVC'):
        return 0
    if lc_id.startswith('LC.'):
        return 1
    return np.nan


def clean_numeric_frame(df):
    out = df.copy()
    for col in out.columns:
        out[col] = pd.to_numeric(
            out[col].replace(['NaN;', 'NaN', 'nan', '', ' '], np.nan),
            errors='coerce'
        )
    out = out.loc[:, out.notna().sum() > 0]
    out = out.loc[:, out.nunique(dropna=False) > 1]
    return out.astype(float)


def make_feature_selector(kbest_k, l1_c):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('kbest', SelectKBest(score_func=f_classif, k=kbest_k)),
        ('l1', SelectFromModel(
            LogisticRegression(
                penalty='l1', solver='liblinear', class_weight='balanced',
                C=l1_c, random_state=RND, max_iter=3000
            )
        ))
    ])


def make_model(model_name):
    if model_name == 'RandomForest':
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=RND,
            n_jobs=-1,
        )
    if model_name == 'GradientBoosting':
        return GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=2,
            random_state=RND,
        )
    if model_name == 'LogisticRegression':
        return LogisticRegression(
            penalty='l2',
            solver='liblinear',
            class_weight='balanced',
            C=0.5,
            random_state=RND,
            max_iter=3000,
        )
    if model_name == 'BalancedRF' and HAS_BRF:
        return BalancedRandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            random_state=RND,
            n_jobs=-1,
        )
    raise ValueError(f'Unsupported model: {model_name}')


def param_grid_for(model_name, n_features):
    k_vals = [k for k in [20, 30, 50, 75, 100] if k <= n_features]
    if not k_vals:
        k_vals = [min(MIN_FEATURES_FALLBACK, n_features)]

    grid = {
        'selector__kbest__k': k_vals,
        'selector__l1__estimator__C': [0.02, 0.05, 0.1, 0.2, 0.5],
    }

    if model_name == 'RandomForest':
        grid.update({
            'clf__n_estimators': [100, 200, 400],
            'clf__max_depth': [3, 5, None],
            'clf__min_samples_leaf': [1, 2, 4],
        })
    elif model_name == 'GradientBoosting':
        grid.update({
            'clf__n_estimators': [100, 200, 400],
            'clf__learning_rate': [0.03, 0.05, 0.1],
            'clf__max_depth': [2, 3],
        })
    elif model_name == 'LogisticRegression':
        grid.update({
            'clf__C': [0.1, 0.2, 0.5, 1.0, 2.0],
        })
    elif model_name == 'BalancedRF' and HAS_BRF:
        grid.update({
            'clf__n_estimators': [100, 200, 400],
            'clf__max_depth': [3, 5, None],
        })

    return grid


def threshold_from_train_probs(y_true, probs):
    thresholds = np.unique(np.clip(probs, 0, 1))
    if len(thresholds) == 0:
        return 0.5
    best_thr, best_f1 = 0.5, -1
    for thr in thresholds:
        preds = (probs >= thr).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr


def bootstrap_ci(arr, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    arr = np.asarray(arr, dtype=float)
    if len(arr) == 0:
        return np.nan, np.nan
    stats = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        stats.append(np.mean(sample))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def load_data(path='Sheet1.xlsx'):
    df = pd.read_excel(path, sheet_name=0)
    df['Long_COVID'] = df['x0_LC_ID'].apply(label_long_covid)
    df = df[df['Long_COVID'].notna()].copy()
    df['Long_COVID'] = df['Long_COVID'].astype(int)
    return df


def build_xy(df):
    metadata_cols = [
        'x0_LC_ID', 'Long_COVID', 'x0_Censor_Complete', 'x0_Censor_Cohort_ID',
        'x0_Patient_Cluster_Label', 'x0_Patient_Cluster', 'x0_Censor_Oral_Steroid',
        'x0_Censor_Pit_Adre_Dysfunction', 'x0_Censor_Pregnancy',
        'x0_Censor_Active_Chemotherapy', 'x0_Censor_Active_Malignancy',
        'x0_Censor_Autoimmune_Pre_Exist', 'x0_Censor_Immuno_Supress_Med',
        'x0_Censor_IVIG', 'x0_Censor_Thyroid',
        'x0_Symp_Survey_Long_COVID_Propensity_Score_Optimized',
        'x0_LC_Symptom_totalsympt', 'x0_LCSI_ID_Label',
        'x1_Description_Cytokine_Label', 'x1_ELISA_Label', 'x1_SI_ID_Label'
    ]
    survey_cols = [c for c in df.columns if 'x0_Symp_Survey_' in c]
    drop_cols = set(metadata_cols + survey_cols)

    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = clean_numeric_frame(df[feature_cols]).copy()
    y = df.loc[X.index, 'Long_COVID'].astype(int).reset_index(drop=True)
    X = X.reset_index(drop=True)
    return X, y


def run_nested_cv(X, y, model_name, out_dir):
    outer_cv = RepeatedStratifiedKFold(
        n_splits=OUTER_SPLITS,
        n_repeats=OUTER_REPEATS,
        random_state=RND
    )
    inner_cv = StratifiedKFold(
        n_splits=INNER_SPLITS,
        shuffle=True,
        random_state=RND
    )

    rows = []
    feature_counts = Counter()

    for fold_id, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        n_features = X_train.shape[1]
        kbest_k = min(max(20, n_features // 20), TOP_K_MAX, n_features)
        kbest_k = max(5, kbest_k)

        selector = make_feature_selector(kbest_k=kbest_k, l1_c=0.1)
        clf = make_model(model_name)

        pipe = Pipeline([
            ('selector', selector),
            ('clf', clf),
        ])

        param_grid = param_grid_for(model_name, n_features)

        if model_name in ['GradientBoosting', 'LogisticRegression']:
            param_grid.pop('selector__l1__estimator__C', None)

        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_grid,
            n_iter=min(N_ITER_SEARCH, 20),
            scoring='roc_auc',
            cv=inner_cv,
            random_state=RND,
            n_jobs=-1,
            verbose=0,
            refit=True,
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_

        train_probs = best.predict_proba(X_train)[:, 1]
        thr = threshold_from_train_probs(y_train.values, train_probs)

        test_probs = best.predict_proba(X_test)[:, 1]
        test_pred = (test_probs >= thr).astype(int)

        try:
            roc = roc_auc_score(y_test, test_probs)
        except Exception:
            roc = np.nan

        try:
            ap = average_precision_score(y_test, test_probs)
        except Exception:
            ap = np.nan

        cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        rows.append({
            'fold': fold_id,
            'model': model_name,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'n_features_train': n_features,
            'best_params': json.dumps(search.best_params_, default=str),
            'threshold': thr,
            'accuracy': accuracy_score(y_test, test_pred),
            'balanced_accuracy': balanced_accuracy_score(y_test, test_pred),
            'precision': precision_score(y_test, test_pred, zero_division=0),
            'recall': recall_score(y_test, test_pred, zero_division=0),
            'f1': f1_score(y_test, test_pred, zero_division=0),
            'roc_auc': roc,
            'avg_precision': ap,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp,
        })

        sel = best.named_steps['selector']
        support = sel.named_steps['l1'].get_support()
        chosen = X.columns[sel.named_steps['kbest'].get_support()].tolist()
        chosen = [f for f, keep in zip(chosen, support) if keep]
        feature_counts.update(chosen)

        print(f"[{model_name} | fold {fold_id:02d}] acc={rows[-1]['accuracy']:.4f} "
              f"f1={rows[-1]['f1']:.4f} roc_auc={rows[-1]['roc_auc']:.4f} "
              f"thr={thr:.3f} nfeat={len(chosen)}")

    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(os.path.join(out_dir, f"{model_name}_fold_metrics.csv"), index=False)

    stable = pd.DataFrame(feature_counts.items(), columns=['feature', 'selection_count'])
    stable['selection_frequency'] = stable['selection_count'] / fold_df.shape[0]
    stable = stable.sort_values(['selection_frequency', 'selection_count', 'feature'],
                                ascending=[False, False, True])
    stable.to_csv(os.path.join(out_dir, f"{model_name}_feature_stability.csv"), index=False)

    summary = {
        'model': model_name,
        'folds': int(fold_df.shape[0]),
        'accuracy_mean': fold_df['accuracy'].mean(),
        'accuracy_sd': fold_df['accuracy'].std(),
        'balanced_accuracy_mean': fold_df['balanced_accuracy'].mean(),
        'precision_mean': fold_df['precision'].mean(),
        'recall_mean': fold_df['recall'].mean(),
        'f1_mean': fold_df['f1'].mean(),
        'roc_auc_mean': fold_df['roc_auc'].mean(),
        'avg_precision_mean': fold_df['avg_precision'].mean(),
        'accuracy_ci_low': bootstrap_ci(fold_df['accuracy'].values)[0],
        'accuracy_ci_high': bootstrap_ci(fold_df['accuracy'].values)[1],
        'roc_auc_ci_low': bootstrap_ci(fold_df['roc_auc'].values)[0],
        'roc_auc_ci_high': bootstrap_ci(fold_df['roc_auc'].values)[1],
        'f1_ci_low': bootstrap_ci(fold_df['f1'].values)[0],
        'f1_ci_high': bootstrap_ci(fold_df['f1'].values)[1],
    }

    return fold_df, stable, summary


def final_refit_importance(X, y, model_name, out_dir):
    selector = make_feature_selector(kbest_k=min(50, X.shape[1]), l1_c=0.1)
    clf = make_model(model_name)
    pipe = Pipeline([('selector', selector), ('clf', clf)])
    pipe.fit(X, y)

    sel = pipe.named_steps['selector']
    chosen = X.columns[sel.named_steps['kbest'].get_support()].tolist()
    chosen = [f for f, keep in zip(chosen, sel.named_steps['l1'].get_support()) if keep]

    if hasattr(pipe.named_steps['clf'], 'feature_importances_'):
        imp = pipe.named_steps['clf'].feature_importances_
        out = pd.DataFrame({'feature': chosen, 'importance': imp})
    elif hasattr(pipe.named_steps['clf'], 'coef_'):
        coef = np.abs(pipe.named_steps['clf'].coef_).ravel()
        out = pd.DataFrame({'feature': chosen, 'importance': coef})
    else:
        return pd.DataFrame(columns=['feature', 'importance'])

    out = out.sort_values('importance', ascending=False)
    out.to_csv(os.path.join(out_dir, f"final_{model_name}_feature_importance.csv"), index=False)

    # NEW: Save top 15 biomarkers
    top15 = out.head(15)
    top15.to_csv(os.path.join(out_dir, f"final_{model_name}_top15_biomarkers.csv"), index=False)

    print(f"\nTop 15 biomarkers for {model_name}:")
    print(top15.to_string(index=False))

    return out


def main():
    out_dir = os.path.join('output', f"research_grade_{now_tag()}")
    os.makedirs(out_dir, exist_ok=True)

    print("Loading data...")
    df = load_data('Sheet1.xlsx')
    X, y = build_xy(df)

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target distribution: {np.bincount(y)} (0=Control, 1=Long COVID)")

    models = ['LogisticRegression', 'RandomForest', 'GradientBoosting']
    if HAS_BRF:
        models = ['BalancedRF'] + models

    summaries = []
    all_folds = []

    for model_name in models:
        print("\n" + "=" * 80)
        print(f"Nested CV: {model_name}")
        print("=" * 80)

        fold_df, stable_df, summary = run_nested_cv(X, y, model_name, out_dir)
        summaries.append(summary)
        all_folds.append(fold_df)

        if model_name in ['LogisticRegression', 'RandomForest', 'BalancedRF']:
            final_refit_importance(X, y, model_name, out_dir)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(os.path.join(out_dir, "models_summary.csv"), index=False)

    folds_all = pd.concat(all_folds, ignore_index=True)
    folds_all.to_csv(os.path.join(out_dir, "all_folds_long.csv"), index=False)

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == '__main__':
    main()


