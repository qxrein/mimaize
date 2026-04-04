"""Publication figures for ABC-Q: Pareto, convergence, bit distribution, method comparisons."""

from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from abc_q_stats import compute_pareto_frontier

DEFAULT_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def ensure_results_dir(results_dir: str) -> None:
    """Create the results directory if it does not exist."""
    os.makedirs(results_dir, exist_ok=True)


def plot_pareto_calib_vs_bops(
    path: str,
    uniform_triples: List[Tuple[str, float, float]],
    random_bops_acc: List[Tuple[float, float]],
    abc_bops: float,
    abc_acc: float,
    cma_bops: float | None,
    cma_acc: float | None,
) -> None:
    """Scatter of random search cloud with Pareto frontier; mark uniforms, ABC-Q, CMA-ES."""
    method_pts: List[Tuple[float, float]] = [(b, a) for _, b, a in uniform_triples]
    method_pts.append((abc_bops, abc_acc))
    if cma_bops is not None and cma_acc is not None:
        method_pts.append((cma_bops, cma_acc))
    all_pts = list(random_bops_acc) + method_pts
    frontier = compute_pareto_frontier(all_pts)

    fig, ax = plt.subplots(figsize=(8, 5.2))
    if random_bops_acc:
        dom_x, dom_y = [], []
        nd_x, nd_y = [], []
        rand_n = len(random_bops_acc)
        for i in range(rand_n):
            p = all_pts[i]
            bp, ap = p
            dominated = False
            for j, q in enumerate(all_pts):
                if j == i:
                    continue
                bq, aq = q
                if bq <= bp and aq >= ap and (bq < bp or aq > ap):
                    dominated = True
                    break
            (dom_x if dominated else nd_x).append(bp)
            (dom_y if dominated else nd_y).append(ap)
        if dom_x:
            ax.scatter(dom_x, dom_y, s=14, alpha=0.35, c="0.65", label="Random (dominated)", edgecolors="none")
        if nd_x:
            ax.scatter(nd_x, nd_y, s=18, alpha=0.55, c="#1f77b4", label="Random (non-dom.)", edgecolors="none")
    if len(frontier) >= 2:
        fx, fy = zip(*frontier)
        ax.plot(fx, fy, "k-", alpha=0.45, linewidth=1.2, label="Pareto frontier")
    elif len(frontier) == 1:
        ax.scatter([frontier[0][0]], [frontier[0][1]], s=40, c="k", marker="o", zorder=4, label="Pareto frontier")
    colors_uni = ["#9467bd", "#8c564b", "#7f7f7f"]
    for i, (label, bop, acc) in enumerate(uniform_triples):
        ax.scatter(
            [bop],
            [acc],
            s=85,
            marker="s",
            c=colors_uni[i % len(colors_uni)],
            label=label,
            zorder=5,
            edgecolors="0.2",
            linewidths=0.6,
        )
    ax.scatter(
        [abc_bops],
        [abc_acc],
        s=220,
        marker="*",
        c="#ff7f0e",
        edgecolors="0.1",
        linewidths=0.9,
        label="ABC-Q (ours)",
        zorder=6,
    )
    if cma_bops is not None and cma_acc is not None:
        ax.scatter(
            [cma_bops],
            [cma_acc],
            s=70,
            marker="D",
            c="#2ca02c",
            label="CMA-ES",
            zorder=5,
            edgecolors="0.2",
            linewidths=0.6,
        )
    ax.set_xlabel("BOPs ratio (vs 8+8 baseline)")
    ax.set_ylabel("Calibration accuracy")
    ax.set_title("Calibration accuracy vs. BOPs (Pareto)")
    ax.legend(loc="lower left", fontsize=7.5)
    ax.grid(True, alpha=0.35)
    ax.set_xlim(0.0, None)
    ax.set_ylim(0.0, 1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_bit_distribution(
    path: str,
    bit_config: np.ndarray,
    sensitivity: np.ndarray,
    title: str = "ABC-Q bit-width per layer",
) -> None:
    """Bar chart of assigned bits per layer; color by top-quartile sensitivity prior."""
    n = len(bit_config)
    thr = float(np.quantile(sensitivity, 0.75))
    colors = ["#d62728" if sensitivity[i] >= thr else "#1f77b4" for i in range(n)]
    fig, ax = plt.subplots(figsize=(max(8, n * 0.22), 4))
    x = np.arange(1, n + 1)
    ax.bar(x, bit_config.astype(float), color=colors, edgecolor="0.2", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xlabel("Quantizable layer index")
    ax.set_ylabel("Bit-width")
    ax.set_title(title)
    ax.set_ylim(0, 9)
    leg = [
        Patch(facecolor="#d62728", edgecolor="0.2", label="Sensitive (≥75th pct)"),
        Patch(facecolor="#1f77b4", edgecolor="0.2", label="Robust"),
    ]
    ax.legend(handles=leg, loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_convergence_abc_vs_cma(
    path: str,
    abc_conv: Dict[str, List[float]],
    cma_conv: Dict[str, List[float]],
) -> None:
    """ABC best/mean colony fitness per cycle vs CMA-ES best-so-far per evaluation."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    if abc_conv.get("abc_cycle") and abc_conv.get("abc_best_fitness"):
        cyc = abc_conv["abc_cycle"]
        ax.plot(cyc, abc_conv["abc_best_fitness"], "-", color="#ff7f0e", label="ABC-Q best", linewidth=1.4)
    if abc_conv.get("abc_cycle") and abc_conv.get("abc_mean_fitness"):
        ax.plot(
            abc_conv["abc_cycle"],
            abc_conv["abc_mean_fitness"],
            "--",
            color="#ffbb78",
            alpha=0.9,
            label="ABC-Q mean colony",
            linewidth=1.1,
        )
    if cma_conv.get("cma_eval") and cma_conv.get("cma_best_fitness"):
        ax.plot(
            cma_conv["cma_eval"],
            cma_conv["cma_best_fitness"],
            "-",
            color="#2ca02c",
            label="CMA-ES best-so-far",
            linewidth=1.2,
        )
    ax.set_xlabel("ABC cycle / CMA-ES eval index")
    ax.set_ylabel("Fitness")
    ax.set_title("Convergence: ABC-Q vs CMA-ES")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_methods_accuracy_comparison(
    path: str,
    names: Sequence[str],
    means: Sequence[float],
    stds: Sequence[float],
    title: str = "Calibration accuracy by method",
) -> None:
    """Horizontal bar chart of mean calibration accuracy with error bars (fractions in [0,1])."""
    fig, ax = plt.subplots(figsize=(7, max(4, 0.35 * len(names))))
    y = np.arange(len(names))
    m = np.asarray(means, dtype=np.float64) * 100.0
    s = np.asarray(stds, dtype=np.float64) * 100.0
    ax.barh(y, m, xerr=s, capsize=3, color="steelblue", alpha=0.85, edgecolor="0.2")
    ax.set_yticks(y)
    ax.set_yticklabels(list(names))
    ax.set_xlabel("Calibration accuracy (%)")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_methods_bops_comparison(
    path: str,
    names: Sequence[str],
    means: Sequence[float],
    stds: Sequence[float],
    title: str = "BOPs ratio by method (w+a vs 8+8)",
) -> None:
    """Bar chart of mean BOPs ratio with error bars."""
    fig, ax = plt.subplots(figsize=(7, max(4, 0.35 * len(names))))
    y = np.arange(len(names))
    m = np.asarray(means, dtype=np.float64)
    s = np.asarray(stds, dtype=np.float64)
    ax.barh(y, m, xerr=s, capsize=3, color="darkseagreen", alpha=0.9, edgecolor="0.2")
    ax.set_yticks(y)
    ax.set_yticklabels(list(names))
    ax.set_xlabel("BOPs ratio")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_methods_acc_per_bops(
    path: str,
    names: Sequence[str],
    mean_acc: Sequence[float],
    mean_bops: Sequence[float],
    std_acc: Sequence[float],
    std_bops: Sequence[float],
    title: str = "Efficiency: Acc/BOPs (error bars from delta method approx.)",
) -> None:
    """Scatter of Acc/BOPs per method; error bars use first-order propagation g = a/b."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    a = np.asarray(mean_acc, dtype=np.float64)
    b = np.asarray(mean_bops, dtype=np.float64)
    sa = np.asarray(std_acc, dtype=np.float64)
    sb = np.asarray(std_bops, dtype=np.float64)
    g = a / np.maximum(b, 1e-12)
    sg = np.sqrt((sa / b) ** 2 + (a * sb / (b * b)) ** 2)
    x = np.arange(len(names))
    ax.scatter(x, g, s=120, c="#e6550d", edgecolors="0.2", zorder=3)
    ax.errorbar(x, g, yerr=sg, fmt="none", capsize=4, color="0.3", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(list(names), rotation=25, ha="right")
    ax.set_ylabel("Acc / BOPs ratio")
    ax.set_title(title)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_summary_scatter_methods(
    path: str,
    points: List[Tuple[str, float, float, float, float]],
    title: str = "Mean calibration acc vs BOPs (±1 std)",
) -> None:
    """Scatter each method at (mean BOPs, mean acc) with crosses for std ranges.

    Each tuple is (name, mean_bops, std_bops, mean_acc, std_acc).
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for i, (name, mb, sb, ma, sa) in enumerate(points):
        ax.errorbar(
            mb,
            ma,
            xerr=sb if sb > 0 else None,
            yerr=sa if sa > 0 else None,
            fmt="o",
            capsize=3,
            label=name,
        )
    ax.set_xlabel("BOPs ratio (mean)")
    ax.set_ylabel("Calibration accuracy (mean)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, alpha=0.35)
    ax.set_xlim(0.0, None)
    ax.set_ylim(0.0, 1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_all_paper_figures(
    results_dir: str,
    uniform_triples: List[Tuple[str, float, float]],
    random_bops_acc: List[Tuple[float, float]],
    abc_bops: float,
    abc_acc: float,
    cma_bops: float | None,
    cma_acc: float | None,
    bit_config: np.ndarray,
    sensitivity: np.ndarray,
    abc_conv: Dict[str, List[float]],
    cma_conv: Dict[str, List[float]],
    comparison_rows: Dict[str, Tuple[float, float, float, float, float, float]],
    comparison_order: Sequence[str],
) -> None:
    """Write the standard figure set plus multi-method comparison plots into ``results_dir``."""
    ensure_results_dir(results_dir)
    plot_pareto_calib_vs_bops(
        os.path.join(results_dir, "pareto_calib_vs_bops.png"),
        uniform_triples,
        random_bops_acc,
        abc_bops,
        abc_acc,
        cma_bops,
        cma_acc,
    )
    plot_bit_distribution(
        os.path.join(results_dir, "bit_distribution.png"),
        bit_config,
        sensitivity,
    )
    plot_convergence_abc_vs_cma(
        os.path.join(results_dir, "convergence_abc_vs_cma.png"),
        abc_conv,
        cma_conv,
    )
    names = [k for k in comparison_order if k in comparison_rows]
    means_acc = [comparison_rows[k][0] for k in names]
    stds_acc = [comparison_rows[k][1] for k in names]
    means_b = [comparison_rows[k][2] for k in names]
    stds_b = [comparison_rows[k][3] for k in names]
    plot_methods_accuracy_comparison(
        os.path.join(results_dir, "comparison_accuracy.png"),
        names,
        means_acc,
        stds_acc,
    )
    plot_methods_bops_comparison(
        os.path.join(results_dir, "comparison_bops.png"),
        names,
        means_b,
        stds_b,
    )
    plot_methods_acc_per_bops(
        os.path.join(results_dir, "comparison_acc_per_bops.png"),
        names,
        means_acc,
        means_b,
        stds_acc,
        stds_b,
    )
    scatter_pts = [
        (k, comparison_rows[k][2], comparison_rows[k][3], comparison_rows[k][0], comparison_rows[k][1])
        for k in names
    ]
    plot_summary_scatter_methods(
        os.path.join(results_dir, "comparison_scatter_mean_std.png"),
        scatter_pts,
    )
