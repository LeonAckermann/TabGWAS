"""
Main results table (paper Table \\ref{tab:friedman_results}): raw
(illness, p-threshold) x model performance, dumped to CSV.

row_ratio=1, col_ratio=1, distribution=low, task=regression, p in
{0.0001, 0.001, 0.01}. Matches the Friedman test's complete-case block
design (fig_friedman_test.py): an (illness, p) row is dropped if any model
is missing a genuine HPO run there, since that's exactly the subset the
Friedman test runs on.

Each cell is "mean" plus a single significance star at p < 0.01, from
Fisher's method on the 5 fold-level Pearson p-values -- not mean +/- std,
which is too wide for a LaTeX table and not very informative from only
5 folds.

Run:
    python analysis/table_friedman_results.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from results_loader import (  # noqa: E402
    ILLNESSES, MODELS, MODEL_LABELS, P_VALUES, P_LABELS, collect_metrics,
)


def _sig_star(p: float) -> str:
    """Single star at p < 0.01 (Fisher's method on fold p-values)."""
    import numpy as np
    return "" if np.isnan(p) or p >= 0.01 else "*"


if __name__ == "__main__":
    data, n_map = collect_metrics()
    out_path = Path(__file__).resolve().parent / "friedman_results.csv"

    header = ["illness", "p_threshold", "N"] + [MODEL_LABELS[m] for m in MODELS]
    rows, dropped = [], []
    for illness in ILLNESSES:
        for p_idx, p in enumerate(P_VALUES):
            means = [data[m][illness]["means"][p_idx] for m in MODELS]
            if any(mean != mean for mean in means):  # NaN check
                dropped.append(f"{illness} ({P_LABELS[p_idx]})")
                continue
            n = n_map[illness][p_idx]
            row = [illness, P_LABELS[p_idx], n if n is not None else ""]
            for model, mean in zip(MODELS, means):
                if mean == 0.0:
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
    print(f"{len(rows)} complete rows (out of {len(ILLNESSES) * len(P_VALUES)} possible)")
    if dropped:
        print(f"  dropped (incomplete): {', '.join(dropped)}")
