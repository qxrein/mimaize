#!/usr/bin/env python3
"""Tables for the ImageNet and LLM extensions -> paper_v2/tables/ext_*.tex, results_ext/summary.txt."""

from __future__ import annotations

import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results_ext")
TAB = os.path.join(ROOT, "paper_v2", "tables")
os.makedirs(TAB, exist_ok=True)

LLMS = [("opt125m", "OPT-125M"), ("opt350m", "OPT-350M"), ("qwen0.5b", "Qwen2.5-0.5B"), ("llama3.2-1b", "Llama-3.2-1B")]
IMGS = [("mobilenetv2", "MobileNetV2", 0.6), ("resnet18", "ResNet-18", 0.5)]
METHODS = [("ga", "Genetic algorithm"), ("cmaes", "CMA-ES"), ("fisher_mckp", "Fisher-MCKP (HAWQ-V2-style)"),
           ("probe_mckp", "Probe-MCKP (one-shot)"), ("abc_v2", "\\textbf{ABC-Q (proposed)}")]


def load(task, model):
    d = os.path.join(RES, task, model)
    st_p, rn_p = os.path.join(d, "model_stats.json"), os.path.join(d, "runs.jsonl")
    if not (os.path.exists(st_p) and os.path.exists(rn_p)):
        return None, []
    st = json.load(open(st_p))
    runs = [json.loads(l) for l in open(rn_p) if l.strip()]
    return st, runs


def cell(vals, fmt, best=False, sd=True):
    if not vals:
        return "--"
    m = np.mean(vals)
    s = fmt.format(m)
    if sd and len(vals) > 1:
        s += " $\\pm$ " + fmt.format(np.std(vals, ddof=1))
    return f"\\textbf{{{s}}}" if best else s


def llm_table(summary):
    cols = [(k, n) for k, n in LLMS if load("llm", k)[0] is not None]
    if not cols:
        return
    rows = {}
    for k, _ in cols:
        st, runs = load("llm", k)
        r = {"FP16": [st["float_test"]], "Uniform 4-bit (4.0 bits)": [st["uniform4"]["test"]],
             "Uniform 3-bit (3.0 bits)": [st["uniform3"]["test"]]}
        for m, lab in METHODS:
            r[lab] = [x["test"] for x in runs if x["method"] == m and abs(x["budget"] - 0.75) < 1e-6]
        rows[k] = r
        summary.append(f"LLM {k}: " + "; ".join(f"{lab}={np.mean(v):.1f}" for lab, v in r.items() if v))
    labels = list(rows[cols[0][0]].keys())
    with open(os.path.join(TAB, "ext_llm.tex"), "w") as f:
        f.write("\\begin{table*}[t]\n\\centering\n\\caption{Weight-only mixed-precision quantization of LLMs "
                "(bits $\\{2,3,4\\}$ per linear layer, group-wise round-to-nearest, group size 128) at an average of "
                "$3.0$ bits per parameter. WikiText-2 test perplexity (lower is better; first 65{,}536 test tokens, "
                "context 512), mean $\\pm$ std over 3 seeds for stochastic methods. Uniform 4-bit uses $33\\%$ more "
                "memory and is shown for reference.}\n\\label{tab:ext_llm}\n\\footnotesize\n"
                "\\resizebox{\\linewidth}{!}{\\begin{tabular}{l" + "c" * len(cols) + "}\n\\toprule\n")
        f.write("Method & " + " & ".join(n for _, n in cols) + " \\\\\n\\midrule\n")
        for lab in labels:
            cells = []
            for k, _ in cols:
                v = rows[k][lab]
                cand = [np.mean(rows[k][l]) for l in labels if rows[k][l] and l not in ("FP16", "Uniform 4-bit (4.0 bits)")]
                best = bool(v) and lab not in ("FP16", "Uniform 4-bit (4.0 bits)") and np.isclose(np.mean(v), min(cand))
                cells.append(cell(v, "{:.1f}", best))
            f.write(lab + " & " + " & ".join(cells) + " \\\\\n")
            if lab == "Uniform 3-bit (3.0 bits)":
                f.write("\\midrule\n")
        f.write("\\bottomrule\n\\end{tabular}}\n\\end{table*}\n")


def img_table(summary):
    cols = [(k, n, B) for k, n, B in IMGS if load("imagenet", k)[0] is not None]
    if not cols:
        return
    rows = {}
    for k, _, B in cols:
        st, runs = load("imagenet", k)
        r = {"FP32": [st["float_test"]], "Uniform W8A8 ($B=1.0$)": [st["uniform8"]["test"]],
             "Uniform W4A8 ($B=0.5$)": [st["uniform4"]["test"]]}
        for m, lab in METHODS:
            r[lab] = [x["test"] for x in runs if x["method"] == m and abs(x["budget"] - B) < 1e-6]
        rows[k] = r
        summary.append(f"ImageNet {k} B={B}: " + "; ".join(f"{lab}={100 * np.mean(v):.2f}" for lab, v in r.items() if v))
    labels = list(rows[cols[0][0]].keys())
    with open(os.path.join(TAB, "ext_imagenet.tex"), "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\caption{ImageNet-pretrained CNNs, weights mixed in $\\{2,4,8\\}$ bits "
                "with 8-bit activations (W$\\{2,4,8\\}$A8). Top-1 accuracy (\\%) on 9{,}488 ImageNetV2 test images "
                "(512 further images are used for calibration), mean $\\pm$ std over 3 seeds. Budgets: " +
                ", ".join(f"{n} $B={B}$" for _, n, B in cols) + " (fraction of W8A8 BOPs).}\n\\label{tab:ext_imagenet}\n"
                "\\footnotesize\n\\resizebox{\\linewidth}{!}{\\begin{tabular}{l" + "c" * len(cols) + "}\n\\toprule\n")
        f.write("Method & " + " & ".join(f"{n} ($B={B}$)" for _, n, B in cols) + " \\\\\n\\midrule\n")
        for lab in labels:
            cells = []
            for k, _, _ in cols:
                v = [100 * x for x in rows[k][lab]]
                ref = ("FP32", "Uniform W8A8 ($B=1.0$)")
                cand = [100 * np.mean(rows[k][l]) for l in labels if rows[k][l] and l not in ref]
                best = bool(v) and lab not in ref and np.isclose(np.mean(v), max(cand))
                cells.append(cell(v, "{:.2f}", best))
            f.write(lab + " & " + " & ".join(cells) + " \\\\\n")
            if lab.startswith("Uniform W4A8"):
                f.write("\\midrule\n")
        f.write("\\bottomrule\n\\end{tabular}}\n\\end{table}\n")


def sweep_table(summary):
    lines = []
    for k, n, _ in IMGS:
        st, runs = load("imagenet", k)
        if st is None:
            continue
        Bs = sorted({round(x["budget"], 3) for x in runs if x["method"] in ("probe_mckp", "fisher_mckp")})
        for B in Bs:
            p = [x["test"] for x in runs if x["method"] == "probe_mckp" and abs(x["budget"] - B) < 1e-6]
            fz = [x["test"] for x in runs if x["method"] == "fisher_mckp" and abs(x["budget"] - B) < 1e-6]
            lines.append(f"{n} & {B:.2f} & {100 * p[0]:.2f} & {100 * fz[0]:.2f} \\\\")
            summary.append(f"sweep {k} B={B}: probe {100 * p[0]:.2f} fisher {100 * fz[0]:.2f}")
    if lines:
        with open(os.path.join(TAB, "ext_sweep.tex"), "w") as f:
            f.write("\\begin{table}[t]\n\\centering\n\\caption{One-shot allocation on ImageNet (W$\\{2,4,8\\}$A8) across "
                    "budgets: measured probes vs.\\ the Fisher-weighted (HAWQ-V2-style) proxy. Top-1 test accuracy (\\%).}\n"
                    "\\label{tab:ext_sweep}\n\\footnotesize\n\\begin{tabular}{lccc}\n\\toprule\n"
                    "Model & $B$ & Probe-MCKP & Fisher-MCKP \\\\\n\\midrule\n" + "\n".join(lines) +
                    "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def main():
    summary = []
    llm_table(summary)
    img_table(summary)
    sweep_table(summary)
    open(os.path.join(RES, "summary.txt"), "w").write("\n".join(summary) + "\n")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
