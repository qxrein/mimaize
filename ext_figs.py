#!/usr/bin/env python3
"""Figures for the LLM experiments -> paper_v2/figs/llm_gptq_bars.pdf, paper_v2/figs/llm_bitmap.pdf.

Fig 1: WikiText-2 perplexity (context 2048, full test set) of every 3-bit allocation + GPTQ, one panel
       per model; FP16 and uniform 4-bit GPTQ drawn as reference lines. ABC-Q highlighted, baselines gray.
Fig 2: per-layer bit-widths chosen by ABC-Q (GPTQ-aware search, best-calibration seed) for
       Llama-3.2-1B as a block x layer-type map, with the bit-width printed in each cell.
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results_ext")
FIG = os.path.join(ROOT, "paper_v2", "figs")
os.makedirs(FIG, exist_ok=True)

# palette (validated): highlight = categorical slot 1; baselines in neutral ink; ordinal blue ramp for bits
BLUE = "#2a78d6"
GRAY = "#a3a29b"
GRAY_LIGHT = "#cfcec8"
INK = "#2b2b29"
INK_MUTED = "#6b6a64"
GRID = "#e6e5e0"
RAMP = ["#86b6ef", "#2a78d6", "#104281"]  # 2, 3, 4 bits

plt.rcParams.update({"font.size": 8, "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": INK_MUTED,
                     "ytick.color": INK, "axes.titlesize": 9, "axes.titleweight": "bold", "font.family": "sans-serif"})

MODELS = [("opt125m", "OPT-125M"), ("opt350m", "OPT-350M"), ("qwen0.5b", "Qwen2.5-0.5B"),
          ("llama3.2-1b", "Llama-3.2-1B")]
METHODS = [("gptq3", "Uniform 3-bit"), ("ga", "GA"), ("cmaes", "CMA-ES"), ("fisher_mckp", "Fisher-MCKP"),
           ("probe_mckp", "Probe-MCKP"), ("abc_v2", "ABC-Q")]


def fig_bars():
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.8), sharey=True)
    for ax, (k, name) in zip(axes, MODELS):
        d = json.load(open(os.path.join(RES, "gptq", f"{k}.json")))
        vals = [d[m]["ppl_wikitext2_ctx2048_full"] for m, _ in METHODS]
        y = np.arange(len(METHODS))
        cols = [GRAY_LIGHT if m == "gptq3" else (BLUE if m == "abc_v2" else GRAY) for m, _ in METHODS]
        ax.barh(y, vals, height=0.62, color=cols, edgecolor="white", linewidth=0.8)
        for yi, v, (m, _) in zip(y, vals, METHODS):
            ax.text(v + max(vals) * 0.02, yi, f"{v:.1f}", va="center", fontsize=7,
                    color=INK if m == "abc_v2" else INK_MUTED, fontweight="bold" if m == "abc_v2" else "normal")
        fp, g4 = d["fp16"]["ppl_wikitext2_ctx2048_full"], d["gptq4"]["ppl_wikitext2_ctx2048_full"]
        ax.axvline(fp, color=INK, lw=1.0, ls="-")
        ax.axvline(g4, color=INK_MUTED, lw=1.0, ls="--")
        ax.set_xlim(0, max(vals) * 1.22)
        ax.set_title(name, loc="left", color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels([lab for _, lab in METHODS])
        ax.invert_yaxis()
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("perplexity (lower is better)", color=INK_MUTED)
    handles = [plt.Line2D([], [], color=INK, lw=1.0, ls="-", label="FP16 (16 bits)"),
               plt.Line2D([], [], color=INK_MUTED, lw=1.0, ls="--", label="uniform 4-bit GPTQ (4 bits)"),
               plt.Rectangle((0, 0), 1, 1, color=GRAY_LIGHT, label="uniform 3-bit GPTQ"),
               plt.Rectangle((0, 0), 1, 1, color=GRAY, label="baseline allocation + GPTQ (3 bits)"),
               plt.Rectangle((0, 0), 1, 1, color=BLUE, label="ABC-Q + GPTQ (3 bits)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(os.path.join(FIG, "llm_gptq_bars.pdf"))
    plt.close(fig)


def fig_bitmap(model="llama3.2-1b", title="Llama-3.2-1B"):
    st = json.load(open(os.path.join(RES, "llm_gptq", model, "model_stats.json")))
    runs = [json.loads(l) for l in open(os.path.join(RES, "llm_gptq", model, "runs.jsonl"))]
    best = max([r for r in runs if r["method"] == "abc_v2"], key=lambda r: r["calib_score"])
    cfg = np.array(best["config"])
    names = st["layer_names"]
    types = []
    for n in names:
        t = n.split(".", 2)[2]
        if t not in types:
            types.append(t)
    nb = max(int(n.split(".")[1]) for n in names) + 1
    M = np.full((len(types), nb), np.nan)
    for n, b in zip(names, cfg):
        M[types.index(n.split(".", 2)[2]), int(n.split(".")[1])] = b
    npar = np.array(st["n_params"])
    share = {t: npar[[i for i, n in enumerate(names) if n.endswith(t)]].sum() / npar.sum() for t in types}
    fig, ax = plt.subplots(figsize=(7.2, 2.3))
    idx = np.vectorize(lambda v: {2: 0, 3: 1, 4: 2}[int(v)])(M)
    ax.imshow(idx, cmap=ListedColormap(RAMP), vmin=0, vmax=2, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{int(M[i, j])}", ha="center", va="center", fontsize=7,
                    color=INK if M[i, j] == 2 else "white")
    ax.set_xticks(np.arange(nb))
    ax.set_xticklabels([str(j) for j in range(nb)], fontsize=7)
    ax.set_yticks(np.arange(len(types)))
    ax.set_yticklabels([f"{t.split('.')[-1]}  ({100 * share[t]:.0f}% of weights)" for t in types], fontsize=7)
    ax.set_xlabel("transformer block (input → output)", color=INK_MUTED)
    ax.set_xticks(np.arange(-0.5, nb, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(types), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.5)
    ax.tick_params(which="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    avg = float(np.sum(cfg * npar) / npar.sum())
    ax.set_title(f"{title}: bit-widths chosen by ABC-Q + GPTQ (average {avg:.2f} bits/weight)", loc="left",
                 color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=f"{b} bits") for c, b in zip(RAMP, (2, 3, 4))]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"llm_bitmap_{model}.pdf"))
    plt.close(fig)
    counts = {b: int((cfg == b).sum()) for b in (2, 3, 4)}
    by_type = {t.split(".")[-1]: float(np.nanmean(M[i])) for i, t in enumerate(types)}
    return counts, by_type


if __name__ == "__main__":
    fig_bars()
    c, bt = fig_bitmap()
    print("bits count", c, "mean bits by type", {k: round(v, 2) for k, v in bt.items()})
