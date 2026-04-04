"""Statistical helpers for ABC-Q experiments: Pareto analysis, Wilcoxon tests, paper table formatting."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats as scipy_stats


def compute_pareto_frontier(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Return non-dominated (BOPs, accuracy) points: lower BOPs and higher accuracy are better.

    Points are sorted by increasing BOPs for plotting.
    """
    pts = list(points)
    nd: List[Tuple[float, float]] = []
    for i, p in enumerate(pts):
        bp, ap = p
        dominated = False
        for j, q in enumerate(pts):
            if i == j:
                continue
            bq, aq = q
            if bq <= bp and aq >= ap and (bq < bp or aq > ap):
                dominated = True
                break
        if not dominated:
            nd.append(p)
    nd.sort(key=lambda t: t[0])
    return nd


def significance_stars(p: float) -> str:
    """Return '' or '*' (p<0.05) or '**' (p<0.01) for tabular reporting."""
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def wilcoxon_paired_pvalue(a: Sequence[float], b: Sequence[float]) -> Tuple[float, str]:
    """Two-sided Wilcoxon signed-rank on paired samples; returns (p-value, star suffix)."""
    try:
        _, p = scipy_stats.wilcoxon(
            np.asarray(a, dtype=np.float64),
            np.asarray(b, dtype=np.float64),
        )
        pf = float(p)
        return pf, significance_stars(pf)
    except (ValueError, RuntimeError):
        return float("nan"), ""


def run_multiple_seeds(
    method_fn: Callable[[int], Any],
    seeds: Sequence[int],
) -> Tuple[float, float, float, float, float, float]:
    """Aggregate a method over seeds: mean/std acc, BOPs (w+a), BOPs (w-only)."""
    accs: List[float] = []
    bwa: List[float] = []
    bwo: List[float] = []
    for s in seeds:
        r = method_fn(int(s))
        accs.append(float(r.accuracy))
        bwa.append(float(r.bops_weight_act))
        bwo.append(float(r.bops_weight_only))
    a = np.array(accs, dtype=np.float64)
    x = np.array(bwa, dtype=np.float64)
    y = np.array(bwo, dtype=np.float64)
    dd = 1 if len(a) > 1 else 0
    return (
        float(a.mean()),
        float(a.std(ddof=dd)),
        float(x.mean()),
        float(x.std(ddof=dd)),
        float(y.mean()),
        float(y.std(ddof=dd)),
    )


def format_acc_percent(mean_acc: float, std_acc: float, n_seeds: int) -> str:
    """Format calibration accuracy as percentage; omit ± when n_seeds < 2."""
    if n_seeds < 2:
        return f"{mean_acc * 100.0:.1f}"
    return f"{mean_acc * 100.0:.1f} ± {std_acc * 100.0:.1f}"


def format_bops_ratio(mean_b: float, std_b: float, n_seeds: int) -> str:
    """Format BOPs ratio; omit ± when n_seeds < 2."""
    if n_seeds < 2:
        return f"{mean_b:.3f}"
    return f"{mean_b:.3f} ± {std_b:.3f}"


def format_pvalue_cell(p: float, stars: str) -> str:
    """Table cell for p-value vs ABC-Q."""
    if not np.isfinite(p):
        return "n/a"
    if p >= 0.05:
        return "ns"
    return f"{p:.3f}{stars}"


def print_paper_table(
    row_order: List[str],
    rows: Dict[str, Tuple[float, float, float, float, float, float]],
    avg_bits_by_method: Dict[str, float],
    wilcoxon_vs_abc: Dict[str, Tuple[float, str]],
    n_seeds: int,
    reference_method: str = "ABC-Q (ours)",
) -> None:
    """Print paper-style table: Method, Accuracy (%), BOPs ratio, Avg bits, Acc/BOPs, p vs ABC-Q.

    Each row tuple is (mean_acc, std_acc, mean_bops_wa, std_bops_wa, _mean_wo, _std_wo); only w+a
    BOPs is shown in the BOPs column. Acc/BOPs uses mean_acc / mean_bops.
    """
    print(
        "\nMethod\t\tAccuracy (%)\tBOPs ratio\tAvg bits\tAcc/BOPs\tp-value vs ABC-Q"
    )
    print("-" * 100)
    for name in row_order:
        if name not in rows:
            continue
        ma, sa, mb, sb, _mwo, _swo = rows[name]
        is_uniform = name.startswith("Uniform")
        ns = 1 if is_uniform else max(int(n_seeds), 1)
        acc_s = format_acc_percent(ma, sa, ns)
        bops_s = format_bops_ratio(mb, sb, ns)
        ab = avg_bits_by_method.get(name, float("nan"))
        bits_s = f"{ab:.2f}" if np.isfinite(ab) else "—"
        acc_over_bops = (ma / mb) if mb > 1e-12 else float("nan")
        acc_bops_s = f"{acc_over_bops:.1f}" if np.isfinite(acc_over_bops) else "—"

        if name == reference_method:
            p_cell = "—"
        else:
            w = wilcoxon_vs_abc.get(name)
            if w is None:
                p_cell = "—"
            else:
                p, st = w
                p_cell = format_pvalue_cell(p, st)

        print(f"{name}\t{acc_s}\t{bops_s}\t{bits_s}\t{acc_bops_s}\t{p_cell}")


def paired_mean_std(values_a: Sequence[float], values_b: Sequence[float]) -> Tuple[float, float]:
    """Mean and sample std of pairwise differences (a - b)."""
    d = np.asarray(values_a, dtype=np.float64) - np.asarray(values_b, dtype=np.float64)
    dd = 1 if len(d) > 1 else 0
    return float(d.mean()), float(d.std(ddof=dd))
