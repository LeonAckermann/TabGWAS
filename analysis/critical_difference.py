"""
Critical-difference (Demsar) diagram utilities.

Given a (blocks x models) score matrix -- each row one block (e.g. an
illness x p-threshold combination), each column one model, higher = better
-- ranks models within each block (rank 1 = best) and plots the average
rank per model on a single axis. Used by fig_friedman_test.py, which builds
its (illness, p-threshold) block score matrix and calls
:func:`plot_critical_difference` directly; not run standalone.

Statistics, following Demsar (2006):

  1. Friedman test on the rank matrix -- omnibus null "all models have the
     same average rank". Reported as both the chi^2 statistic and the
     Iman-Davenport F correction (the chi^2 form is conservative for small k).
  2. Post-hoc, only if the omnibus test rejects:
     * "nemenyi"       -- two models differ if their average ranks differ by
                          more than CD = q_alpha * sqrt(k(k+1) / 6N).
                          One critical distance for every pair; the CD ruler
                          is drawn above the axis.
     * "wilcoxon-holm" -- pairwise Wilcoxon signed-rank tests on the raw
                          per-block scores with Holm correction. More
                          powerful than Nemenyi and uses the actual score
                          differences rather than ranks only, but there is no
                          single CD, so no ruler is drawn.

  Models joined by a thick horizontal bar are *not* significantly different.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import (f as f_dist, friedmanchisquare, rankdata,
                         studentized_range, wilcoxon)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 13,
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "figure.dpi": 150,
    "savefig.dpi": 400,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    # Keep each label (e.g. "TabPFN (1.11)") as one real <text> object in the
    # SVG, not decomposed into a per-character glyph path -- needed to select/
    # move/edit a whole label as a single unit in a vector editor. Tradeoff:
    # the SVG then depends on the viewer having a matching font installed.
    "svg.fonttype": "none",
})

C_AXIS = "#2B2B2B"
C_BAR  = "#2A9D8F"   # "not significantly different" cliques
C_RULE = "#9AA0A6"   # CD ruler / hairlines


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def rank_matrix(scores):
    """Per-block ranks, 1 = best. Ties get their average rank."""
    return np.vstack([rankdata(-row) for row in np.asarray(scores, float)])


def friedman_test(scores):
    """Friedman chi^2 plus the Iman-Davenport F correction.

    Returns a dict with ``chi2``, ``p_chi2``, ``F``, ``p_F``, ``df1``, ``df2``,
    ``N``, ``k`` and the average ranks. scipy's ``friedmanchisquare`` already
    applies the tie correction, so the chi^2 here is taken from it directly.
    """
    scores = np.asarray(scores, float)
    N, k = scores.shape
    ranks = rank_matrix(scores)
    avg = ranks.mean(axis=0)

    stat, p = friedmanchisquare(*scores.T)
    # Iman-Davenport: F = (N-1) chi2 / (N(k-1) - chi2), F(k-1, (k-1)(N-1)).
    denom = N * (k - 1) - stat
    if denom > 0:
        F = (N - 1) * stat / denom
        df1, df2 = k - 1, (k - 1) * (N - 1)
        p_F = float(f_dist.sf(F, df1, df2))
    else:                                   # chi2 saturated -> F diverges
        F, df1, df2, p_F = np.inf, k - 1, (k - 1) * (N - 1), 0.0
    return {"chi2": float(stat), "p_chi2": float(p), "F": float(F), "p_F": p_F,
            "df1": df1, "df2": df2, "df_chi2": k - 1, "N": N, "k": k,
            "avg_ranks": avg}


def nemenyi_cd(k, N, alpha=0.05):
    """Nemenyi critical distance for k models over N blocks."""
    q = float(studentized_range.ppf(1 - alpha, k, np.inf)) / np.sqrt(2.0)
    return q * np.sqrt(k * (k + 1) / (6.0 * N))


def _nemenyi_nonsig(avg_ranks, cd):
    """Boolean matrix: True where two models are *not* separated by the CD."""
    d = np.abs(avg_ranks[:, None] - avg_ranks[None, :])
    return d <= cd


def wilcoxon_holm_nonsig(scores, alpha=0.05):
    """Pairwise Wilcoxon signed-rank with Holm correction.

    Returns ``(nonsig, pairs)`` where ``nonsig[i, j]`` is True when the
    Holm-adjusted p-value for that pair is >= alpha, and ``pairs`` lists
    ``(i, j, p_raw, p_holm)`` sorted by raw p.
    """
    scores = np.asarray(scores, float)
    k = scores.shape[1]
    idx, praw = [], []
    for i in range(k):
        for j in range(i + 1, k):
            d = scores[:, i] - scores[:, j]
            if np.allclose(d, 0):
                p = 1.0
            else:
                p = float(wilcoxon(scores[:, i], scores[:, j])[1])
            idx.append((i, j))
            praw.append(p)
    praw = np.asarray(praw)
    order = np.argsort(praw)
    m = len(praw)
    # Holm step-down, with the running max that keeps adjusted p monotone.
    adj = np.empty(m)
    run = 0.0
    for rank, o in enumerate(order):
        run = max(run, (m - rank) * praw[o])
        adj[o] = min(run, 1.0)

    nonsig = np.eye(k, dtype=bool)
    for (i, j), a in zip(idx, adj):
        nonsig[i, j] = nonsig[j, i] = a >= alpha
    pairs = [(idx[o][0], idx[o][1], float(praw[o]), float(adj[o])) for o in order]
    return nonsig, pairs


def _cliques(nonsig, order):
    """Maximal runs of rank-adjacent models that are mutually non-significant.

    ``order`` is the model indices sorted by average rank. A run [a, b] is kept
    only if every pair inside it is non-significant (for Nemenyi that reduces
    to the endpoints, but Wilcoxon-Holm non-significance is not transitive) and
    it is not contained in a longer run. Singletons are dropped -- a bar over
    one model carries no information.
    """
    n = len(order)
    runs = []
    for a in range(n):
        best_b = a
        for b in range(a + 1, n):
            if all(nonsig[order[i], order[j]]
                   for i in range(a, b + 1) for j in range(i + 1, b + 1)):
                best_b = b
            else:
                break
        if best_b > a:
            runs.append((a, best_b))
    return [r for r in runs
            if not any(r != s and s[0] <= r[0] and r[1] <= s[1] for s in runs)]


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
def _draw_cd_diagram(ax, avg_ranks, labels, cliques, cd=None, better="better"):
    """Classic Demsar layout: rank axis on top, best rank on the left."""
    k = len(labels)
    order = np.argsort(avg_ranks)
    lo = float(np.floor(avg_ranks.min()))
    hi = float(np.ceil(avg_ranks.max()))
    span = hi - lo

    n_left = (k + 1) // 2
    row_gap, cb_gap = 0.60, 0.17
    cb_y = [-0.22 - cb_gap * j for j in range(len(cliques))]
    row0 = (min(cb_y) if cb_y else -0.22) - 0.48
    row_y = [row0 - row_gap * i for i in range(max(n_left, k - n_left))]

    ax.set_xlim(hi + 0.42 * span, lo - 0.42 * span)   # reversed: rank 1 at left
    ax.set_ylim(row_y[-1] - 0.45, 1.75 if cd is not None else 1.0)
    ax.axis("off")

    # ---- rank axis --------------------------------------------------------- #
    ax.plot([lo, hi], [0, 0], color=C_AXIS, lw=1.3, zorder=3,
            solid_capstyle="butt")
    ticks = np.arange(lo, hi + 1e-9, 0.5)
    for t in ticks:
        major = abs(t - round(t)) < 1e-9
        ax.plot([t, t], [0, 0.09 if major else 0.05], color=C_AXIS,
                lw=1.1 if major else 0.9, zorder=3)
        if major:
            ax.text(t, 0.15, f"{t:g}", ha="center", va="bottom", fontsize=12,
                    color=C_AXIS)

    # ---- one leader line per model, labels on the nearer side -------------- #
    for pos, mi in enumerate(order):
        left = pos < n_left
        row = row_y[pos if left else k - 1 - pos]
        edge = lo - 0.40 * span if left else hi + 0.40 * span
        r = avg_ranks[mi]
        ax.plot([r, r], [0, row], color=C_AXIS, lw=1.0, zorder=2)
        ax.plot([r, edge], [row, row], color=C_AXIS, lw=1.0, zorder=2)
        ax.text(edge, row + 0.10, f"{labels[mi]}  ({r:.2f})", fontsize=13,
                ha="left" if left else "right", va="bottom", color=C_AXIS)

    # ---- "not significantly different" bars -------------------------------- #
    for y, (a, b) in zip(cb_y, cliques):
        x0, x1 = avg_ranks[order[a]], avg_ranks[order[b]]
        pad = 0.012 * span
        ax.plot([x0 - pad, x1 + pad], [y, y], color=C_BAR, lw=4.5,
                solid_capstyle="round", zorder=4)

    # ---- CD ruler ---------------------------------------------------------- #
    if cd is not None:
        y = 1.05
        x0, x1 = hi, hi - cd          # axis is reversed -> ruler sits top-left
        ax.plot([x0, x1], [y, y], color=C_RULE, lw=1.4, zorder=3)
        for x in (x0, x1):
            ax.plot([x, x], [y - 0.09, y + 0.09], color=C_RULE, lw=1.4, zorder=3)
        ax.text((x0 + x1) / 2, y + 0.16, f"CD = {cd:.2f}", ha="center",
                va="bottom", fontsize=12, color=C_AXIS)

    # Axis is reversed, so lower rank (= better) is on the visual right.
    ax.annotate(better, xy=(lo - 0.30 * span, 0.62), xytext=(lo, 0.62),
                ha="right", va="center", fontsize=12, color=C_RULE,
                arrowprops=dict(arrowstyle="->", color=C_RULE, lw=1.1))


def plot_critical_difference(scores, labels, alpha=0.05, posthoc="nemenyi",
                             out_prefix=None, title=None, verbose=True):
    """Friedman test + post-hoc, rendered as a critical-difference diagram.

    scores : (n_blocks, n_models), higher = better.
    posthoc: "nemenyi" (rank-based, draws a CD ruler) or "wilcoxon-holm"
             (pairwise signed-rank on raw scores, Holm-corrected).

    Returns the stats dict from :func:`friedman_test`, extended with ``cd``,
    ``cliques`` and (for wilcoxon-holm) ``pairs``.
    """
    scores = np.asarray(scores, float)
    st = friedman_test(scores)
    avg = st["avg_ranks"]
    k, N = st["k"], st["N"]
    order = np.argsort(avg)

    reject = st["p_F"] < alpha
    cd = nemenyi_cd(k, N, alpha)
    if not reject:
        # Omnibus not rejected -> no pair may be declared different; one bar.
        cliques, pairs = [(0, k - 1)], None
    elif posthoc == "nemenyi":
        cliques, pairs = _cliques(_nemenyi_nonsig(avg, cd), order), None
    elif posthoc == "wilcoxon-holm":
        nonsig, pairs = wilcoxon_holm_nonsig(scores, alpha)
        cliques = _cliques(nonsig, order)
    else:
        raise ValueError(f"unknown posthoc: {posthoc!r}")
    st["cd"] = cd
    st["cliques"] = [(labels[order[a]], labels[order[b]]) for a, b in cliques]
    st["pairs"] = pairs

    if verbose:
        p_txt = lambda p: f"{p:.2e}" if p < 1e-3 else f"{p:.3f}"
        print(f"  Friedman: chi2({st['df_chi2']}) = {st['chi2']:.1f}, "
              f"p = {p_txt(st['p_chi2'])} | Iman-Davenport "
              f"F({st['df1']},{st['df2']}) = {st['F']:.1f}, p = {p_txt(st['p_F'])} "
              f"[k = {k} models, N = {N} blocks]")
        print("  average ranks: " + ", ".join(
            f"{labels[i]} {avg[i]:.2f}" for i in order))
        if not reject:
            print(f"  omnibus not rejected at alpha = {alpha}; no post-hoc")
        else:
            print(f"  post-hoc {posthoc}"
                  + (f", CD = {cd:.3f}" if posthoc == "nemenyi" else "")
                  + " | not-different groups: "
                  + ("; ".join(f"{a}–{b}" for a, b in st["cliques"])
                     if st["cliques"] else "none (all pairs differ)"))

    show_cd = reject and posthoc == "nemenyi"
    fig_h = (1.9 if show_cd else 1.4) + 0.45 * ((k + 1) // 2) + 0.16 * len(cliques)
    fig, ax = plt.subplots(figsize=(8.0, fig_h))
    _draw_cd_diagram(ax, avg, labels, cliques, cd=cd if show_cd else None)
    if title:
        ax.set_title(title, pad=14)

    # F, not chi^2: the Iman-Davenport correction is what ``reject`` is based on.
    sub = (f"Friedman $F$({st['df1']}, {st['df2']}) = {st['F']:.1f}, "
           + ("$p$ < 1e-16" if st["p_F"] < 1e-16 else f"$p$ = {st['p_F']:.2g}")
           + f"   |   N = {N} blocks, k = {k} models"
           + (f"   |   post-hoc: {'Nemenyi' if posthoc == 'nemenyi' else 'Wilcoxon-Holm'}"
              if reject else "   |   omnibus not rejected"))
    fig.text(0.5, 0.015, sub, ha="center", va="bottom", fontsize=11, color=C_RULE)

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    if out_prefix is not None:
        out_prefix = Path(out_prefix)
        out_prefix.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(out_prefix.with_suffix(".png"), bbox_inches="tight")
        fig.savefig(out_prefix.with_suffix(".svg"), bbox_inches="tight")
        print(f"  saved {out_prefix.with_suffix('.pdf').name} / .png / .svg")
    plt.close(fig)
    return st
