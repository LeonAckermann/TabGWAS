"""
Friedman test + critical-difference diagram (paper Figure \\ref{fig:friedman_test}),
blocked by (illness, p-value threshold).

Each of the 6 illnesses x 3 p-value thresholds is treated as one "dataset"
(18 blocks total, following Demsar 2006). Models are ranked within each
block by mean Pearson r^2 over the 5 outer CV folds; only blocks where
every model has a result are kept (Friedman needs a complete block design).

Statistics and drawing (Friedman/Iman-Davenport test, Nemenyi and
Wilcoxon-Holm post-hoc, critical-difference diagram) live in
critical_difference.py.

Run:
    python analysis/fig_friedman_test.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import studentized_range

sys.path.insert(0, str(Path(__file__).resolve().parent))

from results_loader import (  # noqa: E402
    ILLNESSES, MODELS, MODEL_LABELS, P_VALUES, P_LABELS, collect_metrics,
)
from critical_difference import plot_critical_difference  # noqa: E402


def block_score_matrix(data: dict) -> tuple[np.ndarray, list[str], list[str]]:
    """Complete-case (illness, p) x model score matrix of mean Pearson r^2."""
    labels = [MODEL_LABELS[m] for m in MODELS]
    rows, block_names = [], []
    for illness in ILLNESSES:
        for p_idx, p in enumerate(P_VALUES):
            vals = [data[m][illness]["means"][p_idx] for m in MODELS]
            if any(np.isnan(v) for v in vals):
                continue
            rows.append(vals)
            block_names.append(f"{illness} ({P_LABELS[p_idx]})")
    scores = np.array(rows, float)
    return scores, labels, block_names


def nemenyi_pairwise_p(avg_ranks: np.ndarray, k: int, N: int) -> list[tuple[int, int, float, float]]:
    """Exact two-sided p-value for every model pair, from the studentized
    range distribution -- the same statistic nemenyi_cd() derives its single
    critical distance from, just solved for p instead of thresholded at
    alpha=0.05. Returns (i, j, rank_diff, p) for i < j.
    """
    scale = np.sqrt(k * (k + 1) / (6.0 * N))
    pairs = []
    for i in range(len(avg_ranks)):
        for j in range(i + 1, len(avg_ranks)):
            d = abs(float(avg_ranks[i] - avg_ranks[j]))
            q_obs = d / scale * np.sqrt(2)
            p = float(studentized_range.sf(q_obs, k, np.inf))
            pairs.append((i, j, d, p))
    return pairs


def print_and_save_pairwise(st_nem: dict, st_wh: dict, labels: list[str], out_path: Path) -> None:
    """Pairwise exact p-values: Nemenyi (rank-based) and Wilcoxon-Holm (raw
    + Holm-adjusted, from the raw per-block score differences).
    """
    avg_ranks, k, N = st_nem["avg_ranks"], st_nem["k"], st_nem["N"]
    nem_pairs = nemenyi_pairwise_p(avg_ranks, k, N)
    wh_lookup = {(i, j): (p_raw, p_holm) for i, j, p_raw, p_holm in st_wh["pairs"]}

    rows = []
    for i, j, d, p_nem in nem_pairs:
        p_raw, p_holm = wh_lookup[(i, j)]
        rows.append((labels[i], labels[j], avg_ranks[i], avg_ranks[j], d, p_nem, p_raw, p_holm))
    rows.sort(key=lambda r: r[5])  # ascending Nemenyi p

    print("\nPairwise model comparisons (sorted by Nemenyi p):")
    print(f"  {'Model A':<10} {'Model B':<10} {'rank diff':>9}  {'p (Nemenyi)':>12}  "
          f"{'p (Wilcoxon raw)':>17}  {'p (Wilcoxon Holm)':>18}")
    for a, b, ra, rb, d, p_nem, p_raw, p_holm in rows:
        print(f"  {a:<10} {b:<10} {d:>9.3f}  {p_nem:>12.2e}  {p_raw:>17.2e}  {p_holm:>18.2e}")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model_a", "model_b", "rank_a", "rank_b", "rank_diff",
                          "p_nemenyi", "p_wilcoxon_raw", "p_wilcoxon_holm"])
        for a, b, ra, rb, d, p_nem, p_raw, p_holm in rows:
            writer.writerow([a, b, f"{ra:.4f}", f"{rb:.4f}", f"{d:.4f}",
                              f"{p_nem:.6e}", f"{p_raw:.6e}", f"{p_holm:.6e}"])
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent
    data, _ = collect_metrics()
    scores, labels, blocks = block_score_matrix(data)
    print(f"{scores.shape[0]} complete (illness, p) blocks across "
          f"{scores.shape[1]} models "
          f"(out of {len(ILLNESSES) * len(P_VALUES)} possible)")
    missing = [
        f"{illness} ({P_LABELS[p_idx]})"
        for illness in ILLNESSES
        for p_idx in range(len(P_VALUES))
        if f"{illness} ({P_LABELS[p_idx]})" not in blocks
    ]
    if missing:
        print(f"  dropped (incomplete): {', '.join(missing)}")

    if scores.shape[0] < 3 or scores.shape[1] < 3:
        raise SystemExit("Not enough complete blocks/models to run a Friedman test.")

    st_nem = plot_critical_difference(
        scores, labels, posthoc="nemenyi",
        title="Average rank across illness x p-threshold blocks (mean Pearson $r^2$)",
        out_prefix=out_dir / "friedman_test_nemenyi")
    st_wh = plot_critical_difference(
        scores, labels, posthoc="wilcoxon-holm",
        title="Average rank across illness x p-threshold blocks (mean Pearson $r^2$)",
        out_prefix=out_dir / "friedman_test_wilcoxon_holm")

    print_and_save_pairwise(st_nem, st_wh, labels, out_dir / "friedman_test_pairwise_p.csv")
    print("Done.")
