"""
Shared result-loading utilities for the analysis scripts in this directory.

Reads the per-fold HPO result JSONs written by `main.py` for the regression
sweep over 6 illnesses x 3 p-value thresholds (distribution=low, row_ratio=1,
col_ratio=1) and exposes them as (illness, model) -> mean/std/significance
arrays. Used by fig_friedman_test.py, table_friedman_results.py, and
table_residual_results.py -- not run directly.
"""

import json
import os
import glob
import numpy as np
from scipy import stats as scipy_stats

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "../../results_hpo")
ILLNESSES = ["SCZ", "BIP", "MDD", "OCD", "AZ", "ADHD"]
ILLNESS_LABELS = {
    "SCZ": "Schizophrenia", "BIP": "Bipolar Disorder", "MDD": "Depression",
    "OCD": "OCD", "AZ": "Alzheimer's", "ADHD": "ADHD",
}
MODELS = ["linear_regression", "lasso_regression", "ridge_regression", "residual_dnn", "tabpfn", "xgboost"]
MODEL_LABELS = {
    "linear_regression": "Linear", "lasso_regression": "Lasso",
    "ridge_regression": "Ridge", "residual_dnn": "DNN",
    "tabpfn": "TabPFN", "xgboost": "XGBoost",
}
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]

P_VALUES = [0.0001, 0.001, 0.01]
P_LABELS = ["p=0.0001", "p=0.001", "p=0.01"]
DISTRIBUTION = "low"
ROW_RATIO = 1
COL_RATIO = 1
TASK = "regression"


def load_latest_result(model: str, illness: str, p: float) -> dict | None:
    """Latest result with an actual HPO search (config.hpo.run == True).

    Some folders also contain "replay" runs (hpo.run == False, params loaded
    from a cached best-params file rather than re-searched) mixed in among
    the real HPO runs. Those can silently collapse to degenerate fits (e.g.
    Lasso predicting the constant mean, r^2 = 0) and must not be picked over
    a genuine HPO run just because their timestamp is newer.
    """
    pattern = os.path.join(
        RESULTS_DIR,
        f"{model}_{illness}_p{p}_{DISTRIBUTION}_{ROW_RATIO}_{COL_RATIO}_{TASK}",
        "*.json",
    )
    files = sorted(glob.glob(pattern))
    hpo_run_files = []
    for fp in files:
        with open(fp) as f:
            result = json.load(f)
        if result.get("config", {}).get("hpo", {}).get("run") is True:
            hpo_run_files.append(result)
    return hpo_run_files[-1] if hpo_run_files else None


def _fisher_p(ps):
    ps = [p for p in ps if not np.isnan(p) and 0 < p <= 1]
    if not ps: return np.nan
    return float(scipy_stats.chi2.sf(-2 * np.sum(np.log(ps)), df=2 * len(ps)))


def _n_samples(result: dict) -> int | None:
    folds = result["hpo"].get("fold_label_distributions", [])
    if not folds: return None
    return folds[0]["train"]["total"] + folds[0]["test"]["total"]


def collect_metrics(metric: str = "pearson_r2") -> tuple[dict, dict]:
    """metric: "pearson_r2" (squared Pearson correlation, default) or "r2"
    (sklearn coefficient of determination) -- reads hpo["mean_{metric}"]/
    hpo["std_{metric}"]. Significance (fisher_ps) is always computed from
    the per-fold Pearson p-values regardless of metric, since sklearn's R^2
    has no corresponding per-fold p-value in this codebase.
    """
    data = {}
    n_map = {ill: [None] * len(P_VALUES) for ill in ILLNESSES}
    for model in MODELS:
        data[model] = {}
        for illness in ILLNESSES:
            means, stds, fisher_ps = [], [], []
            for p_idx, p in enumerate(P_VALUES):
                result = load_latest_result(model, illness, p)
                if result is None:
                    means.append(np.nan); stds.append(np.nan); fisher_ps.append(np.nan)
                else:
                    hpo = result["hpo"]
                    means.append(hpo.get(f"mean_{metric}", np.nan))
                    stds.append(hpo.get(f"std_{metric}", np.nan))
                    fisher_ps.append(_fisher_p([fm.get("pearson_p", np.nan)
                                                for fm in hpo.get("fold_metrics", [])]))
                    if n_map[illness][p_idx] is None:
                        n_map[illness][p_idx] = _n_samples(result)
            data[model][illness] = {
                "means": np.array(means), "stds": np.array(stds), "fisher_ps": fisher_ps,
            }
    return data, n_map
