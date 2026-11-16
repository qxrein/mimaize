#!/usr/bin/env python3
"""ABC-Q v2 — run budget-constrained mixed-precision search experiments.

Example:
    python v2_run.py --backbone resnet20 --dataset cifar10 --budgets 0.25 0.35 \
        --seeds 0 1 2 3 4 5 6 7 8 9 --methods random ga cmaes abc_v1 abc_v2 probe_mckp fisher_mckp

Outputs (under results_v2/<backbone>/<dataset>/):
    model_stats.json           float/uniform accuracies, probes Δ, gradient prior, Fisher trace
    runs.jsonl                 one JSON record per (method, budget, seed) — resumable
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--backbone", required=True)
ap.add_argument("--dataset", required=True)
ap.add_argument("--budgets", type=float, nargs="+", default=[0.25, 0.35])
ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
ap.add_argument("--methods", nargs="+", default=["random", "ga", "cmaes", "abc_v1", "abc_v2",
                                                  "probe_mckp", "fisher_mckp"])
ap.add_argument("--evals", type=int, default=500)
ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
ap.add_argument("--out", default="runs.jsonl")
args = ap.parse_args()

import numpy as np
import tensorflow as tf

if args.device == "cpu":
    tf.config.set_visible_devices([], "GPU")

import abc_q_core as q  # noqa: E402
import v2_search as S  # noqa: E402
from v2_engine import BITS, FastEvaluator  # noqa: E402

q.BACKBONE = args.backbone
q.DATASET = args.dataset
q.NUM_CLASSES = 100 if args.dataset == "cifar100" else 10
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results_v2", args.backbone, args.dataset)

DETERMINISTIC = {"probe_mckp", "fisher_mckp"}


def load() -> FastEvaluator:
    cfg = q.get_experiment_config()
    data = q.load_dataset_splits(cfg)
    model = q.build_model()
    model.load_weights(os.path.join(OUT, "float.weights.h5"))
    return FastEvaluator(model, data["x_calib"], data["y_calib"], data["x_test"], data["y_test"])


def model_stats(ev: FastEvaluator) -> dict:
    path = os.path.join(OUT, "model_stats.json")
    if os.path.exists(path):
        with open(path) as f:
            st = json.load(f)
        st["delta"] = np.array(st["delta"])
        return st
    t0 = time.time()
    st = {"L": ev.L, "layer_names": ev.names, "layer_types": ev.types,
          "macs": ev.macs.tolist(), "n_params": ev.n_params.tolist(),
          "float_calib": ev.calib_acc(None), "float_test": ev.test_acc(None)}
    for b in BITS:
        c = np.full(ev.L, b)
        st[f"uniform{b}"] = {"calib": ev.calib_acc(c), "test": ev.test_acc(c), "bops": ev.bops_ratio(c),
                             "mem": ev.mem_ratio(c)}
    acc8, delta = ev.probe_sensitivity()
    st["acc8"] = acc8
    st["delta"] = delta
    st["grad_prior"] = ev.gradient_prior().tolist()
    st["fisher_trace"] = ev.fisher_trace().tolist()
    st["weight_qerr"] = ev.weight_qerr().tolist()
    st["stats_wall_s"] = time.time() - t0
    with open(path, "w") as f:
        json.dump({**st, "delta": delta.tolist()}, f, indent=2)
    return st


def done_keys(path: str) -> set:
    keys = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    keys.add((r["method"], round(r["budget"], 4), r["seed"]))
                except Exception:
                    pass
    return keys


def run_one(ev: FastEvaluator, st: dict, method: str, B: float, seed: int) -> dict:
    L = ev.L
    cost = ev.layer_costs()
    rng = np.random.RandomState(1000 * seed + 7)
    np.random.seed(1000 * seed + 7)
    oracle = S.Oracle(ev.calib_acc, cost, B, args.evals)
    delta = np.asarray(st["delta"])
    acc8 = float(st["acc8"])

    def preload_probes():
        # probes are deterministic; charge them to this run's budget and expose them to the surrogate
        base = np.full(L, 8, dtype=np.int64)
        items = [(base, acc8)]
        for i in range(L):
            for j, b in enumerate(BITS):
                if b == 8:
                    continue
                c = base.copy()
                c[i] = b
                items.append((c, acc8 - delta[i, j]))
        for c, a in items:
            oracle.cache[c.tobytes()] = a
            oracle.history.append((c, a))
            oracle.n += 1
            oracle.trace.append(oracle.best_acc)

    t0 = time.time()
    base_name = method
    kw = {}
    if method.startswith("abc_v2"):
        base_name = "abc_v2"
        suffix = method[len("abc_v2"):]
        if suffix == "_noscreen":
            kw["screen"] = False
        elif suffix == "_nolearn":
            kw["learn"] = False
        elif suffix == "_noknap":
            kw["knapsack_init"] = False
        elif suffix == "_noshare":
            kw["share"] = False
        elif suffix == "_hc":
            kw.update({"N": 1, "share": False})
        elif suffix == "_K1":
            kw["K"] = 1
        elif suffix == "_K12":
            kw["K"] = 12
        elif suffix:
            raise ValueError(method)

    if base_name == "random":
        S.run_random(oracle, L, rng)
    elif base_name == "ga":
        S.run_ga(oracle, L, rng)
    elif base_name == "cmaes":
        S.run_cmaes(oracle, L, rng, seed)
    elif base_name == "abc_v1":
        S.run_abc_v1(oracle, L, rng, np.asarray(st["grad_prior"]))
    elif base_name == "abc_v2":
        preload_probes()
        S.run_abc_v2(oracle, L, rng, acc8, delta, **kw)
    elif base_name == "probe_mckp":
        preload_probes()
        oracle(S.mckp(delta, cost, B))
    elif base_name == "fisher_mckp":
        omega = np.asarray(st["fisher_trace"])[:, None] * np.asarray(st["weight_qerr"])
        omega = omega - omega[:, 2:3]
        oracle(S.mckp(omega, cost, B))
    else:
        raise ValueError(method)
    wall = time.time() - t0

    best = oracle.best_cfg
    rec = {"method": method, "budget": B, "seed": seed, "evals_used": oracle.n, "wall_s": wall}
    if best is None:
        rec.update({"calib_acc": None, "test_acc": None})
    else:
        rec.update({
            "calib_acc": oracle.best_acc,
            "test_acc": ev.test_acc(best),
            "bops": ev.bops_ratio(best),
            "mem": ev.mem_ratio(best),
            "avg_bits_param": float(np.sum(best * ev.n_params) / np.sum(ev.n_params)),
            "config": best.tolist(),
            "trace": [oracle.trace[k] for k in range(0, len(oracle.trace), 5)] + [oracle.trace[-1]],
        })
    return rec


def main() -> None:
    ev = load()
    st = model_stats(ev)
    print(f"[stats] L={ev.L} float_test={st['float_test']:.4f} acc8={st['acc8']:.4f} "
          f"u4={st['uniform4']['calib']:.4f} u2={st['uniform2']['calib']:.4f}", flush=True)
    path = os.path.join(OUT, args.out)
    done = done_keys(path)
    for B in args.budgets:
        for method in args.methods:
            seeds = [0] if method in DETERMINISTIC else args.seeds
            for seed in seeds:
                if (method, round(B, 4), seed) in done:
                    continue
                rec = run_one(ev, st, method, B, seed)
                with open(path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"[run] B={B} {method:18s} seed={seed} calib={rec['calib_acc']} "
                      f"test={rec['test_acc']} bops={rec.get('bops')} evals={rec['evals_used']} "
                      f"wall={rec['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
