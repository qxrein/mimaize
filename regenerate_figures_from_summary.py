#!/usr/bin/env python3
"""Regenerate paper PNGs from experiment_summary.json and layer_bit_freq.csv."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from abc_q_plots import plot_convergence_abc_vs_cma

REPO = Path(__file__).resolve().parent
RESULTS = REPO / "results"

EXPERIMENTS = [
    ("resnet20", "cifar10"),
    ("resnet20", "cifar100"),
    ("mobilenetv2", "cifar10"),
    ("mobilenetv2", "cifar100"),
    ("efficientnetb0", "cifar10"),
]

COMPARE_ORDER = [
    "Uniform 8-bit",
    "Uniform 4-bit",
    "Random Search",
    "CMA-ES (Hansen)",
    "ABC-Q (no prior)",
    "ABC-Q (ours)",
]


def load_summary(backbone: str, dataset: str) -> dict:
    path = RESULTS / backbone / dataset / "experiment_summary.json"
    with path.open() as f:
        return json.load(f)


def plot_bit_distribution(out_dir: Path) -> None:
    csv_path = out_dir / "layer_bit_freq.csv"
    if not csv_path.exists():
        return
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    x = np.arange(len(rows))
    w2 = np.array([float(r["frac_2bit"]) for r in rows])
    w4 = np.array([float(r["frac_4bit"]) for r in rows])
    w8 = np.array([float(r["frac_8bit"]) for r in rows])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x, w2, label="2-bit", color="#c44e52")
    ax.bar(x, w4, bottom=w2, label="4-bit", color="#4c72b0")
    ax.bar(x, w8, bottom=w2 + w4, label="8-bit", color="#55a868")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Fraction of seeds")
    ax.set_title("ABC-Q bit assignment frequency")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_dir / "bit_distribution.png", dpi=160)
    plt.close(fig)


def plot_comparison_accuracy(out_dir: Path, summary: dict) -> None:
    rows = summary["paper_table_rows"]
    order = [k for k in COMPARE_ORDER if k in rows and "Uniform 2-bit" not in k]
    names = order
    means = [100 * rows[k]["mean_acc"] for k in names]
    stds = [100 * rows[k]["std_acc"] for k in names]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, capsize=3, color="#4c72b0", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" (Hansen)", "").replace(" (ours)", "*") for n in names],
                       rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Calibration accuracy (%)")
    ax.set_title(f"{summary['backbone']} / {summary['dataset']}")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_accuracy.png", dpi=160)
    plt.close(fig)


def plot_scatter_mean_std(out_dir: Path, summary: dict) -> None:
    rows = summary["paper_table_rows"]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    colors = {"Random Search": "#888888", "CMA-ES (Hansen)": "#4c72b0",
              "ABC-Q (no prior)": "#dd8452", "ABC-Q (ours)": "#c44e52"}
    for name, style in [
        ("Random Search", "o"),
        ("CMA-ES (Hansen)", "s"),
        ("ABC-Q (no prior)", "^"),
        ("ABC-Q (ours)", "*"),
    ]:
        if name not in rows:
            continue
        r = rows[name]
        ax.errorbar(
            r["mean_bops_wa"], 100 * r["mean_acc"],
            xerr=r["std_bops_wa"], yerr=100 * r["std_acc"],
            fmt=style, capsize=3, label=name.replace(" (Hansen)", ""),
            color=colors.get(name, "black"), markersize=9,
        )
    ax.set_xlabel("BOPs ratio (MAC-weighted)")
    ax.set_ylabel("Calibration accuracy (%)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_scatter_mean_std.png", dpi=160)
    plt.close(fig)


def plot_pareto_stub(out_dir: Path, summary: dict) -> None:
    """Method means only (no per-trial random cloud in summary JSON)."""
    rows = summary["paper_table_rows"]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, c, m in [
        ("Random Search", "#888888", "o"),
        ("CMA-ES (Hansen)", "#4c72b0", "s"),
        ("ABC-Q (ours)", "#c44e52", "*"),
    ]:
        if name not in rows:
            continue
        r = rows[name]
        ax.scatter(r["mean_bops_wa"], 100 * r["mean_acc"], c=c, marker=m, s=120,
                   label=name.replace(" (Hansen)", "").replace(" (ours)", ""))
    ax.set_xlabel("BOPs ratio")
    ax.set_ylabel("Calibration accuracy (%)")
    ax.set_title("Method means (summary JSON)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "pareto_calib_vs_bops.png", dpi=160)
    plt.close(fig)


def plot_convergence(out_dir: Path, summary: dict) -> bool:
    """Plot ABC vs CMA convergence from summary or convergence_abc_cma.json."""
    conv = summary.get("convergence_abc_cma")
    conv_json = out_dir / "convergence_abc_cma.json"
    if conv is None and conv_json.exists():
        with conv_json.open() as f:
            conv = json.load(f)
    if not conv or not conv.get("abc") or not conv.get("cma"):
        return False
    plot_convergence_abc_vs_cma(
        str(out_dir / "convergence_abc_vs_cma.png"),
        conv["abc"],
        conv["cma"],
    )
    return True


def main() -> None:
    for backbone, dataset in EXPERIMENTS:
        out_dir = RESULTS / backbone / dataset
        summary_path = out_dir / "experiment_summary.json"
        if not summary_path.exists():
            print("skip", out_dir)
            continue
        summary = load_summary(backbone, dataset)
        if not summary.get("run_complete"):
            print("incomplete", out_dir)
            continue
        plot_bit_distribution(out_dir)
        plot_comparison_accuracy(out_dir, summary)
        plot_scatter_mean_std(out_dir, summary)
        if dataset == "cifar10" and backbone in ("resnet20", "mobilenetv2", "efficientnetb0"):
            plot_pareto_stub(out_dir, summary)
        if not plot_convergence(out_dir, summary):
            print("  (no convergence data; run regenerate_convergence_plots.py)")
        print("wrote figures in", out_dir)


if __name__ == "__main__":
    main()
