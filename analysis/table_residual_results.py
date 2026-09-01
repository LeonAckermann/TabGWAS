"""
Residual-ablation table (paper Table \\ref{tab:residual_results}, appendix):
same layout as table_friedman_results.py, but for the Ridge-residual study
-- results_hpo_residual/, folders suffixed "_residual" (row_ratio/col_ratio
written as "1.0_1.0" there, not "1_1"), and only the three nonlinear models
(DNN, TabPFN, XGBoost) -- Ridge itself is the residual baseline
(cfg["data"]["residual_baseline_model"] = "ridge_regression"), not a row in
this table.

Same hpo.run=True filtering as the main pipeline (see results_loader.py's
load_latest_result). Unlike table_friedman_results.py, rows are never
dropped for missing/incomplete data: all 18 (illness, p) combinations are
shown, and a model with no completed HPO run for a given row is marked NaN
in that cell rather than omitting the row.

Run:
    python analysis/table_residual_results.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from results_loader import ILLNESSES, MODEL_LABELS, P_VALUES, P_LABELS, _fisher_p  # noqa: E402
from table_friedman_results import _sig_star  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "../../results_hpo_residual")
MODELS = ["residual_dnn", "tabpfn", "xgboost"]
DISTRIBUTION, ROW_RATIO, COL_RATIO, TASK = "low", "1.0", "1.0", "regression"


def load_latest_result(model: str, illness: str, p: float) -> dict | None:
    """Latest result with an actual HPO search (config.hpo.run == True).

    Same rationale as results_loader.py's load_latest_result -- this
    results directory can also mix in replay runs.
    """
    pattern = os.path.join(
        RESULTS_DIR,
        f"{model}_{illness}_p{p:g}_{DISTRIBUTION}_{ROW_RATIO}_{COL_RATIO}_{TASK}_residual",
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


def _n_samples(result: dict) -> int | None:
    folds = result["hpo"].get("fold_label_distributions", [])
    if not folds:
        return None
    return folds[0]["train"]["total"] + folds[0]["test"]["total"]


def collect_metrics() -> tuple[dict, dict]:
    data = {}
    n_map = {ill: [None] * len(P_VALUES) for ill in ILLNESSES}
    for model in MODELS:
        data[model] = {}
        for illness in ILLNESSES:
            means, fisher_ps = [], []
            for p_idx, p in enumerate(P_VALUES):
                result = load_latest_result(model, illness, p)
                if result is None:
                    means.append(np.nan); fisher_ps.append(np.nan)
                else:
                    hpo = result["hpo"]
                    means.append(hpo.get("mean_pearson_r2", np.nan))
                    fisher_ps.append(_fisher_p([fm.get("pearson_p", np.nan)
                                                for fm in hpo.get("fold_metrics", [])]))
                    if n_map[illness][p_idx] is None:
                        n_map[illness][p_idx] = _n_samples(result)
            data[model][illness] = {"means": np.array(means), "fisher_ps": fisher_ps}
    return data, n_map


if __name__ == "__main__":
    data, n_map = collect_metrics()
    out_path = Path(__file__).resolve().parent / "residual_results.csv"

    header = ["illness", "p_threshold", "N"] + [MODEL_LABELS[m] for m in MODELS]
    rows, missing = [], []
    for illness in ILLNESSES:
        for p_idx, p in enumerate(P_VALUES):
            means = [data[m][illness]["means"][p_idx] for m in MODELS]
            n = n_map[illness][p_idx]
            row = [illness, P_LABELS[p_idx], n if n is not None else ""]
            for model, mean in zip(MODELS, means):
                if mean != mean:
                    # No completed HPO run at all for this model/threshold.
                    row.append("NaN")
                    missing.append(f"{illness} ({P_LABELS[p_idx]}, {MODEL_LABELS[model]})")
                elif mean == 0.0:
                    # Exactly 0 means every fold's prediction was constant
                    # (the conditional mean), for which Pearson correlation
                    # is undefined -- src/evaluation.py's NaN fallback, not
                    # a genuine near-zero effect. Report as NaN, not 0.
                    row.append("NaN")
                else:
                    fisher_p = data[model][illness]["fisher_ps"][p_idx]
                    row.append(f"{mean:.4f}{_sig_star(fisher_p)}")
            rows.append(row)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Saved: {out_path}")
    print(f"{len(rows)} rows (out of {len(ILLNESSES) * len(P_VALUES)} possible)")
    if missing:
        print(f"  no completed HPO run (shown as NaN): {', '.join(missing)}")
