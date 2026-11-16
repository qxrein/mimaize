#!/usr/bin/env python3
"""ABC-Q v2 — aggregate runs.jsonl into LaTeX tables, figures and a plain-text summary.

    python v2_report.py            # writes paper_v2/tables/*.tex, paper_v2/figs/*.pdf, results_v2/summary.txt

Statistics (Derrac et al., 2011; Demšar, 2006):
  * Friedman test across stochastic methods, seeds as blocks.
  * Two-sided Wilcoxon signed-rank (paired by seed) of ABC-Q v2 vs every other stochastic method,
    Holm-corrected within each (backbone, dataset, budget) family.
  * Vargha–Delaney A12 effect size: P(v2 > other) + 0.5 P(tie).
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results_v2")
PAPER = os.path.join(ROOT, "paper_v2")
TAB = os.path.join(PAPER, "tables")
FIG = os.path.join(PAPER, "figs")
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

CONFIGS = [("resnet20", "cifar10"), ("resnet20", "cifar100"), ("mobilenetv2", "cifar10"),
           ("mobilenetv2", "cifar100")]
PRETTY_BB = {"resnet20": "ResNet-20", "mobilenetv2": "MobileNetV2", "efficientnetb0": "EfficientNet-B0"}
PRETTY_DS = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100"}
MAIN = ["random", "ga", "cmaes", "abc_v1", "fisher_mckp", "probe_mckp", "abc_v2"]
PRETTY = {"random": "Random search", "ga": "Genetic algorithm", "cmaes": "CMA-ES",
          "abc_v1": "ABC-Q v1 (original)", "fisher_mckp": "Fisher-MCKP (HAWQ-V2-style)",
          "probe_mckp": "Probe-MCKP (one-shot)", "abc_v2": "\\textbf{ABC-Q v2 (proposed)}",
          "abc_v2_noscreen": "-- surrogate screening", "abc_v2_nolearn": "-- online surrogate refit",
          "abc_v2_noknap": "-- knapsack initialisation", "abc_v2_noshare": "-- information sharing",
          "abc_v2_hc": "single source (hill climber)", "abc_v2_K1": "$K=1$", "abc_v2_K12": "$K=12$"}
STOCH = {"random", "ga", "cmaes", "abc_v1", "abc_v2"}
COLORS = {"random": "#9e9e9e", "ga": "#1f77b4", "cmaes": "#2ca02c", "abc_v1": "#ff7f0e",
          "abc_v2": "#d62728", "probe_mckp": "#9467bd", "fisher_mckp": "#8c564b"}


def load_runs(bb: str, ds: str) -> List[dict]:
    p = os.path.join(RES, bb, ds, "runs.jsonl")
    out = []
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    return out


def load_stats(bb: str, ds: str) -> dict:
    p = os.path.join(RES, bb, ds, "model_stats.json")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)


def a12(x: np.ndarray, y: np.ndarray) -> float:
    gt = sum((xi > yi) for xi in x for yi in y)
    eq = sum((xi == yi) for xi in x for yi in y)
    return (gt + 0.5 * eq) / (len(x) * len(y))


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    items = sorted([(p, k) for k, p in pvals.items() if np.isfinite(p)])
    m = len(items)
    adj, running = {}, 0.0
    for r, (p, k) in enumerate(items):
        running = max(running, min(1.0, (m - r) * p))
        adj[k] = running
    for k, p in pvals.items():
        adj.setdefault(k, float("nan"))
    return adj


def wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    if np.allclose(d, 0):
        return 1.0
    try:
        return float(stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        return float("nan")


def by_method(runs: List[dict], B: float) -> Dict[str, Dict[int, dict]]:
    out: Dict[str, Dict[int, dict]] = defaultdict(dict)
    for r in runs:
        if abs(r["budget"] - B) < 1e-6 and r.get("calib_acc") is not None:
            out[r["method"]][r["seed"]] = r
    return out


def fmt(m: float, s: float = None, bold: bool = False) -> str:
    t = f"{100 * m:.2f}" if s is None else f"{100 * m:.2f} $\\pm$ {100 * s:.2f}"
    return f"\\textbf{{{t}}}" if bold else t


def main_tables(summary: List[str]) -> None:
    rows_all = []
    for bb, ds in CONFIGS:
        runs = load_runs(bb, ds)
        st = load_stats(bb, ds)
        if not runs or not st:
            continue
        budgets = sorted({round(r["budget"], 4) for r in runs})
        for B in budgets:
            M = by_method(runs, B)
            if "abc_v2" not in M:
                continue
            seeds = sorted(M["abc_v2"].keys())
            # Friedman across stochastic methods sharing all seeds
            fr_methods = [m for m in MAIN if m in STOCH and m in M and all(s in M[m] for s in seeds)]
            fr_p = float("nan")
            if len(fr_methods) >= 3 and len(seeds) >= 3:
                fr_p = float(stats.friedmanchisquare(*[[M[m][s]["calib_acc"] for s in seeds]
                                                       for m in fr_methods]).pvalue)
            v2c = np.array([M["abc_v2"][s]["calib_acc"] for s in seeds])
            v2t = np.array([M["abc_v2"][s]["test_acc"] for s in seeds])
            raw_c, raw_t, eff_c = {}, {}, {}
            for m in fr_methods:
                if m == "abc_v2":
                    continue
                oc = np.array([M[m][s]["calib_acc"] for s in seeds])
                ot = np.array([M[m][s]["test_acc"] for s in seeds])
                raw_c[m] = wilcoxon(v2c, oc)
                raw_t[m] = wilcoxon(v2t, ot)
                eff_c[m] = a12(v2c, oc)
            adj_c, adj_t = holm(raw_c), holm(raw_t)
            best_c = max(np.mean([r["calib_acc"] for r in M[m].values()]) for m in M if m in MAIN)
            best_t = max(np.mean([r["test_acc"] for r in M[m].values()]) for m in M if m in MAIN)
            lines = []
            u4 = st["uniform4"]
            lines.append(f"Uniform W4A4 & {fmt(u4['calib'])} & {fmt(u4['test'])} & {u4['bops']:.3f} & 4.00 & -- & -- \\\\")
            for m in MAIN:
                if m not in M:
                    continue
                c = np.array([r["calib_acc"] for r in M[m].values()])
                t = np.array([r["test_acc"] for r in M[m].values()])
                bo = np.mean([r["bops"] for r in M[m].values()])
                ab = np.mean([r["avg_bits_param"] for r in M[m].values()])
                sd = len(c) > 1
                pc = adj_c.get(m)
                pcell = "--" if pc is None or m == "abc_v2" else (f"{pc:.3f}" if pc >= 0.001 else "$<$0.001")
                ecell = "--" if m not in eff_c else f"{eff_c[m]:.2f}"
                lines.append(
                    f"{PRETTY[m]} & {fmt(c.mean(), c.std(ddof=1) if sd else None, np.isclose(c.mean(), best_c))} & "
                    f"{fmt(t.mean(), t.std(ddof=1) if sd else None, np.isclose(t.mean(), best_t))} & {bo:.3f} & {ab:.2f} & {pcell} & {ecell} \\\\")
                rows_all.append({"bb": bb, "ds": ds, "B": B, "method": m, "calib": c.mean(),
                                 "calib_sd": c.std(ddof=1) if sd else 0, "test": t.mean(),
                                 "test_sd": t.std(ddof=1) if sd else 0, "bops": bo, "n": len(c),
                                 "p_calib_holm": adj_c.get(m), "p_test_holm": adj_t.get(m), "A12": eff_c.get(m)})
            name = f"main_{bb}_{ds}_B{int(round(B * 1000)):03d}"
            cap = (f"{PRETTY_BB[bb]} / {PRETTY_DS[ds]}, BOPs budget $B={B:.2f}$ (fraction of W8A8), "
                   f"$E={runs[0]['evals_used'] if runs else 0}$ evaluations, $n={len(seeds)}$ seeds. "
                   f"Float test accuracy {100 * st['float_test']:.2f}\\%. "
                   f"Calib./test: PTQ accuracy (no fine-tuning) on the 1k calibration and 10k test sets. "
                   f"$p$: Holm-corrected two-sided Wilcoxon vs.\\ ABC-Q v2 on calibration accuracy; "
                   f"$A_{{12}}$: Vargha--Delaney effect size (probability that v2 is better). "
                   f"Friedman $p={fr_p:.2g}$.")
            with open(os.path.join(TAB, name + ".tex"), "w") as f:
                f.write("\\begin{table}[t]\n\\centering\n\\caption{" + cap + "}\n\\label{tab:" + name + "}\n")
                f.write("\\footnotesize\\setlength{\\tabcolsep}{3.5pt}\n")
                f.write("\\begin{tabular}{lcccccc}\n\\toprule\n")
                f.write("Method & Calib.\\ acc.\\ (\\%) & Test acc.\\ (\\%) & BOPs & Bits/param & $p$ & $A_{12}$ \\\\\n\\midrule\n")
                f.write("\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
            summary.append(f"\n== {bb}/{ds} B={B} (n={len(seeds)}, Friedman p={fr_p:.3g})")
            for r in [x for x in rows_all if x["bb"] == bb and x["ds"] == ds and x["B"] == B]:
                summary.append(f"  {r['method']:12s} calib {100 * r['calib']:.2f}±{100 * r['calib_sd']:.2f} "
                               f"test {100 * r['test']:.2f}±{100 * r['test_sd']:.2f} bops {r['bops']:.3f} "
                               f"p_c={r['p_calib_holm']} p_t={r['p_test_holm']} A12={r['A12']}")
    with open(os.path.join(RES, "main_rows.json"), "w") as f:
        json.dump(rows_all, f, indent=1, default=float)

    # compact cross-configuration table (test accuracy) for the paper
    if rows_all:
        budgets = sorted({r["B"] for r in rows_all})
        cfgs = [(bb, ds) for bb, ds in CONFIGS if any(r["bb"] == bb and r["ds"] == ds for r in rows_all)]
        with open(os.path.join(TAB, "overview_test.tex"), "w") as f:
            cols = len(cfgs) * len(budgets)
            f.write("\\begin{table*}[t]\n\\centering\n\\caption{Test accuracy (\\%, mean over seeds, PTQ without "
                    "fine-tuning) of the configuration returned by each method under BOPs budget $B$. "
                    "Bold: best per column; $^{\\dagger}$: ABC-Q v2 significantly better (Holm-corrected Wilcoxon on "
                    "test accuracy, $p<0.05$); $^{\\ddagger}$: significantly worse.}\n\\label{tab:overview_test}\n"
                    "\\footnotesize\\setlength{\\tabcolsep}{3pt}\n")
            f.write("\\begin{tabular}{l" + "c" * cols + "}\n\\toprule\n")
            f.write(" & " + " & ".join(f"\\multicolumn{{{len(budgets)}}}{{c}}{{{PRETTY_BB[bb]} / {PRETTY_DS[ds]}}}"
                                       for bb, ds in cfgs) + " \\\\\n")
            f.write("Method & " + " & ".join(f"$B{{=}}{B:.2f}$" for _ in cfgs for B in budgets) + " \\\\\n\\midrule\n")
            for m in MAIN:
                cells = []
                for bb, ds in cfgs:
                    for B in budgets:
                        r = [x for x in rows_all if x["bb"] == bb and x["ds"] == ds and x["B"] == B and x["method"] == m]
                        if not r:
                            cells.append("--")
                            continue
                        r = r[0]
                        col = [x["test"] for x in rows_all if x["bb"] == bb and x["ds"] == ds and x["B"] == B]
                        s = f"{100 * r['test']:.2f}"
                        if np.isclose(r["test"], max(col)):
                            s = f"\\textbf{{{s}}}"
                        p = r.get("p_test_holm")
                        if m != "abc_v2" and p is not None and np.isfinite(p) and p < 0.05:
                            v2 = [x for x in rows_all if x["bb"] == bb and x["ds"] == ds and x["B"] == B and x["method"] == "abc_v2"][0]
                            s += "$^{\\dagger}$" if v2["test"] > r["test"] else "$^{\\ddagger}$"
                        cells.append(s)
                f.write(PRETTY[m] + " & " + " & ".join(cells) + " \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")


def ablation_table(summary: List[str]) -> None:
    variants = ["abc_v2", "abc_v2_noscreen", "abc_v2_nolearn", "abc_v2_noknap", "abc_v2_noshare", "abc_v2_hc",
                "abc_v2_K1", "abc_v2_K12"]
    blocks = []
    for bb, ds in CONFIGS:
        runs = load_runs(bb, ds)
        for B in sorted({round(r["budget"], 4) for r in runs}):
            M = by_method(runs, B)
            if sum(v in M for v in variants[1:]) < 2:
                continue
            seeds = sorted(M["abc_v2"].keys())
            v2 = np.array([M["abc_v2"][s]["calib_acc"] for s in seeds])
            v2t = np.array([M["abc_v2"][s]["test_acc"] for s in seeds])
            raw = {}
            for v in variants[1:]:
                if v in M and all(s in M[v] for s in seeds):
                    raw[v] = wilcoxon(v2, np.array([M[v][s]["calib_acc"] for s in seeds]))
            adj = holm(raw)
            lines = []
            for v in variants:
                if v not in M:
                    continue
                c = np.array([M[v][s]["calib_acc"] for s in seeds if s in M[v]])
                t = np.array([M[v][s]["test_acc"] for s in seeds if s in M[v]])
                p = adj.get(v)
                pcell = "--" if p is None else (f"{p:.3f}" if p >= 0.001 else "$<$0.001")
                lab = "ABC-Q v2 (full)" if v == "abc_v2" else PRETTY[v]
                lines.append(f"{lab} & {fmt(c.mean(), c.std(ddof=1))} & {fmt(t.mean(), t.std(ddof=1))} & "
                             f"{100 * (c.mean() - v2.mean()):+.2f} & {pcell} \\\\")
                summary.append(f"  ablation {bb}/{ds} B={B} {v:18s} calib {100 * c.mean():.2f} test {100 * t.mean():.2f} p={p}")
            blocks.append((f"{PRETTY_BB[bb]} / {PRETTY_DS[ds]}, $B={B:.2f}$", lines))
    if blocks:
        with open(os.path.join(TAB, "ablation.tex"), "w") as f:
            f.write("\\begin{table}[t]\n\\centering\n\\caption{Component ablation of ABC-Q v2 (calibration and test "
                    "accuracy, \\%, mean $\\pm$ std over seeds; $\\Delta$: change in calibration accuracy vs.\\ the full "
                    "method; $p$: Holm-corrected two-sided Wilcoxon).}\n\\label{tab:ablation}\n"
                    "\\footnotesize\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{lcccc}\n\\toprule\n"
                    "Variant & Calib.\\ (\\%) & Test (\\%) & $\\Delta$ (pp) & $p$ \\\\\n")
            for title, lines in blocks:
                f.write("\\midrule\n\\multicolumn{5}{l}{\\emph{" + title + "}} \\\\\n" + "\n".join(lines) + "\n")
            f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def sensitivity_correlation(summary: List[str]) -> None:
    rows = []
    for bb, ds in CONFIGS:
        st = load_stats(bb, ds)
        if not st:
            continue
        d = np.array(st["delta"])
        g = np.array(st["grad_prior"])
        om = np.array(st["fisher_trace"])[:, None] * np.array(st["weight_qerr"])
        r4 = stats.spearmanr(g, d[:, 1]).correlation
        r2 = stats.spearmanr(g, d[:, 0]).correlation
        f4 = stats.spearmanr(om[:, 1], d[:, 1]).correlation
        f2 = stats.spearmanr(om[:, 0], d[:, 0]).correlation
        # top-quartile overlap: v1 'sensitive set' vs truly most damaging layers at 2 bits
        k = int(np.ceil(0.25 * len(g)))
        top_g = set(np.argsort(-g)[:k])
        top_d = set(np.argsort(-d[:, 0])[:k])
        ov = len(top_g & top_d) / k
        rows.append((bb, ds, st["L"], r4, r2, f4, f2, ov))
        summary.append(f"  corr {bb}/{ds}: spearman(grad, Δ4)={r4:.2f} (grad, Δ2)={r2:.2f} "
                       f"(fisherΩ, Δ4)={f4:.2f} (fisherΩ, Δ2)={f2:.2f} top25%-overlap={ov:.2f}")
    if rows:
        with open(os.path.join(TAB, "sens_corr.tex"), "w") as f:
            f.write("\\begin{table}[t]\n\\centering\n\\caption{Do cheap sensitivity proxies predict the measured "
                    "single-layer accuracy drop $\\Delta_i(b)$? Spearman rank correlation $\\rho$ between the v1 "
                    "gradient prior $\\tilde s_i$ (resp.\\ the HAWQ-V2-style Fisher score $\\Omega_i(b)$) and "
                    "$\\Delta_i(b)$, and overlap between the v1 top-25\\% `sensitive' set and the 25\\% of layers "
                    "with the largest measured 2-bit drop.}\n\\label{tab:sens_corr}\n\\footnotesize\n"
                    "\\begin{tabular}{lccccccc}\n\\toprule\n"
                    "Model & $L$ & $\\rho(\\tilde s,\\Delta_4)$ & $\\rho(\\tilde s,\\Delta_2)$ & $\\rho(\\Omega_4,\\Delta_4)$ & "
                    "$\\rho(\\Omega_2,\\Delta_2)$ & Top-25\\% overlap \\\\\n\\midrule\n")
            for bb, ds, L, r4, r2, f4, f2, ov in rows:
                f.write(f"{PRETTY_BB[bb]} / {PRETTY_DS[ds]} & {L} & {r4:.2f} & {r2:.2f} & {f4:.2f} & {f2:.2f} & {ov:.2f} \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def anytime_figure() -> None:
    cfgs = [(bb, ds) for bb, ds in CONFIGS if load_runs(bb, ds)]
    if not cfgs:
        return
    budgets = sorted({round(r["budget"], 4) for bb, ds in cfgs for r in load_runs(bb, ds)})
    fig, axes = plt.subplots(len(budgets), len(cfgs), figsize=(3.2 * len(cfgs), 2.6 * len(budgets)), squeeze=False)
    for ci, (bb, ds) in enumerate(cfgs):
        runs = load_runs(bb, ds)
        st = load_stats(bb, ds)
        for bi, B in enumerate(budgets):
            ax = axes[bi, ci]
            M = by_method(runs, B)
            for m in ["random", "ga", "cmaes", "abc_v1", "abc_v2"]:
                if m not in M:
                    continue
                tr = [np.array(r["trace"], dtype=float) for r in M[m].values()]
                n = min(len(t) for t in tr)
                A = np.stack([t[:n] for t in tr])
                A[A < 0] = np.nan
                x = np.arange(n) * 5
                mu = np.nanmean(A, axis=0)
                ax.plot(x, 100 * mu, color=COLORS[m], lw=1.4, label=PRETTY[m].replace("\\textbf{", "").replace("}", ""))
            for m in ["probe_mckp", "fisher_mckp"]:
                if m in M:
                    v = list(M[m].values())[0]
                    ax.axhline(100 * v["calib_acc"], color=COLORS[m], ls="--", lw=1.0,
                               label=PRETTY[m].replace("\\textbf{", "").replace("}", ""))
            if st:
                ax.axhline(100 * st["uniform4"]["calib"], color="k", ls=":", lw=1.0, label="Uniform W4A4")
            ax.set_title(f"{PRETTY_BB[bb]} / {PRETTY_DS[ds]}, B={B:.2f}", fontsize=8)
            ax.set_xlabel("distinct evaluations", fontsize=7)
            ax.set_ylabel("best calib. acc. (%)", fontsize=7)
            ax.tick_params(labelsize=7)
            vals = [100 * r["calib_acc"] for mm in M.values() for r in mm.values()]
            if vals:
                lo = np.percentile(vals, 5)
                ax.set_ylim(lo - 2, max(vals) + 1)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(FIG, "anytime.pdf"))
    plt.close(fig)


def gap_figure(summary: List[str]) -> None:
    cfgs = [(bb, ds) for bb, ds in CONFIGS if load_runs(bb, ds)]
    if not cfgs:
        return
    fig, axes = plt.subplots(1, len(cfgs), figsize=(3.0 * len(cfgs), 2.8), squeeze=False)
    for ci, (bb, ds) in enumerate(cfgs):
        ax = axes[0, ci]
        runs = [r for r in load_runs(bb, ds) if r.get("calib_acc") is not None and r["method"] in MAIN]
        c = np.array([r["calib_acc"] for r in runs])
        t = np.array([r["test_acc"] for r in runs])
        for m in MAIN:
            sel = [i for i, r in enumerate(runs) if r["method"] == m]
            if sel:
                ax.scatter(100 * c[sel], 100 * t[sel], s=9, color=COLORS[m], alpha=0.7,
                           label=PRETTY[m].replace("\\textbf{", "").replace("}", ""))
        lim = [100 * min(c.min(), t.min()) - 1, 100 * max(c.max(), t.max()) + 1]
        ax.plot(lim, lim, "k:", lw=0.8)
        rho = stats.spearmanr(c, t).correlation
        ax.set_title(f"{PRETTY_BB[bb]} / {PRETTY_DS[ds]} ($\\rho$={rho:.2f})", fontsize=8)
        ax.set_xlabel("calibration acc. (%)", fontsize=7)
        ax.set_ylabel("test acc. (%)", fontsize=7)
        ax.tick_params(labelsize=7)
        summary.append(f"  calib-test spearman {bb}/{ds}: {rho:.3f}; mean gap (calib-test) {100 * np.mean(c - t):.2f} pp")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(os.path.join(FIG, "calib_vs_test.pdf"))
    plt.close(fig)


def bitmap_figure() -> None:
    """Per-layer bit allocation of ABC-Q v2 (fraction of seeds at 2/4/8) with measured Δ_i(2)."""
    cfgs = [(bb, ds) for bb, ds in CONFIGS if load_runs(bb, ds)]
    if not cfgs:
        return
    fig, axes = plt.subplots(len(cfgs), 1, figsize=(7.0, 1.7 * len(cfgs)), squeeze=False)
    for ci, (bb, ds) in enumerate(cfgs):
        ax = axes[ci, 0]
        runs = load_runs(bb, ds)
        st = load_stats(bb, ds)
        B = min({round(r["budget"], 4) for r in runs})
        M = by_method(runs, B)
        if "abc_v2" not in M:
            continue
        C = np.array([r["config"] for r in M["abc_v2"].values()])
        L = C.shape[1]
        x = np.arange(L)
        f2, f4, f8 = (C == 2).mean(0), (C == 4).mean(0), (C == 8).mean(0)
        ax.bar(x, f2, color="#d9d9d9", label="2-bit")
        ax.bar(x, f4, bottom=f2, color="#6baed6", label="4-bit")
        ax.bar(x, f8, bottom=f2 + f4, color="#08519c", label="8-bit")
        ax2 = ax.twinx()
        d2 = np.array(st["delta"])[:, 0]
        ax2.plot(x, 100 * d2, "r.-", lw=0.8, ms=3, label="measured $\\Delta_i(2)$")
        ax2.set_yscale("symlog", linthresh=1.0)
        ax2.tick_params(labelsize=6)
        ax2.set_ylabel("$\\Delta_i(2)$ (pp)", fontsize=7)
        ax.set_xlim(-0.6, L - 0.4)
        ax.set_ylabel("fraction of seeds", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_title(f"{PRETTY_BB[bb]} / {PRETTY_DS[ds]}, B={B:.2f}: ABC-Q v2 allocation (layer index, input→output)", fontsize=8)
    h1, l1 = axes[0, 0].get_legend_handles_labels()
    fig.legend(h1 + [plt.Line2D([], [], color="r", marker=".")], l1 + ["measured $\\Delta_i(2)$"],
               loc="lower center", ncol=4, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(FIG, "bit_allocation.pdf"))
    plt.close(fig)


def wall_table(summary: List[str]) -> None:
    lines = []
    for bb, ds in CONFIGS:
        runs = load_runs(bb, ds)
        st = load_stats(bb, ds)
        if not runs:
            continue
        cells = []
        for m in MAIN:
            w = [r["wall_s"] for r in runs if r["method"] == m]
            cells.append(f"{np.mean(w):.0f}" if w else "--")
        lines.append(f"{PRETTY_BB[bb]} / {PRETTY_DS[ds]} & " + " & ".join(cells) + " \\\\")
        summary.append(f"  wall {bb}/{ds}: " + ", ".join(f"{m}={c}" for m, c in zip(MAIN, cells)) +
                       f"; probe stats {st.get('stats_wall_s', float('nan')):.0f}s")
    if lines:
        with open(os.path.join(TAB, "wall.tex"), "w") as f:
            f.write("\\begin{table}[t]\n\\centering\n\\caption{Mean wall-clock time per search run (s) on an Apple M4 "
                    "laptop (16\\,GB). All stochastic methods use the same number of distinct evaluations, so wall time is "
                    "dominated by forward passes; ABC-Q v2 includes probes and surrogate overhead.}\n\\label{tab:wall}\n"
                    "\\footnotesize\\setlength{\\tabcolsep}{2.5pt}\n\\begin{tabular}{l" + "c" * len(MAIN) + "}\n\\toprule\n"
                    "Model & Random & GA & CMA-ES & v1 & Fisher & Probe & v2 \\\\\n\\midrule\n" + "\n".join(lines) +
                    "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def models_table() -> None:
    lines = []
    for bb, ds in CONFIGS:
        st = load_stats(bb, ds)
        if not st:
            continue
        u = {b: st[f"uniform{b}"] for b in (8, 4, 2)}
        lines.append(f"{PRETTY_BB[bb]} / {PRETTY_DS[ds]} & {st['L']} & {100 * st['float_test']:.2f} & "
                     f"{100 * u[8]['test']:.2f} & {100 * u[4]['test']:.2f} & {100 * u[2]['test']:.2f} \\\\")
    if lines:
        with open(os.path.join(TAB, "models.tex"), "w") as f:
            f.write("\\begin{table}[t]\n\\centering\n\\caption{Float backbones and uniform-precision PTQ references "
                    "(test accuracy, \\%). W$b$A$b$: all weights and layer inputs at $b$ bits (per-channel symmetric "
                    "weights, per-tensor min--max activations). BOPs relative to W8A8: W4A4 $=0.25$, W2A2 $=0.0625$.}\n"
                    "\\label{tab:models}\n\\footnotesize\n\\begin{tabular}{lccccc}\n\\toprule\n"
                    "Model & $L$ & FP32 & W8A8 & W4A4 & W2A2 \\\\\n\\midrule\n" + "\n".join(lines) +
                    "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def qat_table(summary: List[str]) -> None:
    order = ["uniform4", "fisher_mckp", "ga", "cmaes", "abc_v2"]
    names = {"uniform4": "Uniform W4A4", "fisher_mckp": "Fisher-MCKP", "ga": "GA", "cmaes": "CMA-ES",
             "abc_v2": "ABC-Q v2"}
    cfgs = []
    for bb, ds in CONFIGS:
        p = os.path.join(RES, bb, ds, "qat_B250.json")
        if os.path.exists(p):
            with open(p) as f:
                cfgs.append((bb, ds, json.load(f)))
    if not cfgs:
        return
    with open(os.path.join(TAB, "qat.tex"), "w") as f:
        f.write("\\begin{table*}[t]\n\\centering\n\\caption{Test accuracy (\\%) of the configuration found at $B=0.25$ "
                "before (PTQ) and after two epochs of quantization-aware fine-tuning (QAT, STE, network kept "
                "quantized). One configuration per method (median-calibration seed).}\n\\label{tab:qat}\n"
                "\\footnotesize\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{l" + "cc" * len(cfgs) + "}\n\\toprule\n")
        f.write(" & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{PRETTY_BB[bb]} / {PRETTY_DS[ds]}}}" for bb, ds, _ in cfgs)
                + " \\\\\n")
        f.write("Configuration & " + " & ".join("PTQ & QAT" for _ in cfgs) + " \\\\\n\\midrule\n")
        for m in order:
            cells = []
            for bb, ds, d in cfgs:
                if m in d:
                    cells += [f"{100 * d[m]['ptq_test']:.2f}", f"{100 * d[m]['qat_test']:.2f}"]
                else:
                    cells += ["--", "--"]
            f.write(names[m] + " & " + " & ".join(cells) + " \\\\\n")
            summary.append(f"  qat {m}: " + "; ".join(
                f"{bb}/{ds} {100 * d[m]['ptq_test']:.2f}->{100 * d[m]['qat_test']:.2f}" for bb, ds, d in cfgs if m in d))
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")


def main() -> None:
    summary: List[str] = []
    qat_table(summary)
    models_table()
    main_tables(summary)
    ablation_table(summary)
    sensitivity_correlation(summary)
    anytime_figure()
    gap_figure(summary)
    bitmap_figure()
    wall_table(summary)
    # scale every tabular to the text width (avoids overfull boxes in the manuscript)
    for fn in os.listdir(TAB):
        p = os.path.join(TAB, fn)
        t = open(p).read()
        if "\\resizebox" not in t and "\\begin{tabular}" in t:
            t = t.replace("\\begin{tabular}", "\\resizebox{\\linewidth}{!}{\\begin{tabular}")
            t = t.replace("\\end{tabular}", "\\end{tabular}}")
            open(p, "w").write(t)
    txt = "\n".join(summary)
    with open(os.path.join(RES, "summary.txt"), "w") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
