#!/usr/bin/env python3
"""
Master runner for ABC-Q publication experiments.

- Runs each entry script in a fixed order (fresh subprocess per experiment).
- Skip-if-done: if ``RUN_SUCCEEDED`` exists in that experiment's results directory, skip unless
  ``--force`` (resume after crashes without redoing finished jobs).
- ``--dry-run``: sets ``ABC_Q_DRY_RUN=1`` and writes under ``results/_dry_run/<BACKBONE>/<DATASET>/``
  so full runs are not marked complete in production paths.
- After all jobs (or available summaries), writes ``results/all_experiments_summary.csv`` by reading
  each ``experiment_summary.json`` — no hardcoded metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_RESULTS = os.path.join(_REPO_ROOT, "results")

# (script relative to repo, backbone, dataset, num_classes) — N_SEEDS remain in each entry file.
_EXPERIMENTS: List[Tuple[str, str, str, int]] = [
    ("abc_q_cifar10_full.py", "mobilenetv2", "cifar10", 10),
    ("abc_q_cifar10_resnet20.py", "resnet20", "cifar10", 10),
    ("abc_q_efficientnet_cifar10.py", "efficientnetb0", "cifar10", 10),
    ("abc_q_cifar100_mobilenet.py", "mobilenetv2", "cifar100", 100),
    ("abc_q_cifar100_resnet.py", "resnet20", "cifar100", 100),
]

_DONE_NAME = "RUN_SUCCEEDED"
_SUMMARY_NAME = "experiment_summary.json"


def _results_dir(backbone: str, dataset: str, dry_run: bool) -> str:
    if dry_run:
        return os.path.join(_RESULTS, "_dry_run", backbone, dataset)
    return os.path.join(_RESULTS, backbone, dataset)


def _done_path(backbone: str, dataset: str, dry_run: bool) -> str:
    return os.path.join(_results_dir(backbone, dataset, dry_run), _DONE_NAME)


def _summary_path(backbone: str, dataset: str, dry_run: bool) -> str:
    return os.path.join(_results_dir(backbone, dataset, dry_run), _SUMMARY_NAME)


def _run_one(script: str, backbone: str, dataset: str, dry_run: bool, force: bool) -> int:
    done = _done_path(backbone, dataset, dry_run)
    if not force and os.path.isfile(done):
        print(f"[skip] {script} — found {done}")
        return 0

    env = os.environ.copy()
    env["ABC_Q_RESULTS_DIR"] = _results_dir(backbone, dataset, dry_run)
    if dry_run:
        env["ABC_Q_DRY_RUN"] = "1"
    else:
        env.pop("ABC_Q_DRY_RUN", None)

    if force and os.path.isfile(done):
        try:
            os.remove(done)
        except OSError:
            pass

    cmd = [sys.executable, "-u", os.path.join(_REPO_ROOT, script)]
    print(f"[run] {' '.join(cmd)}")
    print(f"       ABC_Q_RESULTS_DIR={env['ABC_Q_RESULTS_DIR']}")
    if dry_run:
        print("       ABC_Q_DRY_RUN=1")
    proc = subprocess.run(cmd, cwd=_REPO_ROOT, env=env)
    return int(proc.returncode)


def _flatten_summary_row(
    script: str,
    backbone: str,
    dataset: str,
    num_classes: int,
    dry_run: bool,
    data: Dict[str, Any],
) -> Dict[str, Any]:
    rows = data.get("paper_table_rows") or {}
    abc = rows.get("ABC-Q (ours)") or {}
    rnd = rows.get("Random Search") or {}
    cma = rows.get("CMA-ES (Hansen)") or {}
    return {
        "script": script,
        "backbone": backbone,
        "dataset": dataset,
        "num_classes": num_classes,
        "n_seeds": data.get("n_seeds", ""),
        "dry_run": dry_run,
        "float_test_accuracy": data.get("float_test_accuracy", ""),
        "final_test_accuracy": data.get("final_test_accuracy", ""),
        "abc_mean_acc": abc.get("mean_acc", ""),
        "abc_std_acc": abc.get("std_acc", ""),
        "abc_mean_bops_wa": abc.get("mean_bops_wa", ""),
        "abc_std_bops_wa": abc.get("std_bops_wa", ""),
        "random_mean_acc": rnd.get("mean_acc", ""),
        "cma_mean_acc": cma.get("mean_acc", ""),
        "run_complete": data.get("run_complete", ""),
    }


def write_aggregate_csv(
    experiments: List[Tuple[str, str, str, int]],
    dry_run: bool,
    out_path: str,
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    flat: List[Dict[str, Any]] = []
    for script, bb, ds, nc in experiments:
        sp = _summary_path(bb, ds, dry_run)
        if not os.path.isfile(sp):
            continue
        try:
            with open(sp, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        flat.append(_flatten_summary_row(script, bb, ds, nc, dry_run, data))

    if not flat:
        print(f"[warn] No {_SUMMARY_NAME} files found to aggregate → {out_path}")
        return

    fieldnames = list(flat[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as cf:
        w = csv.DictWriter(cf, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(flat)
    print(f"[log] Wrote aggregate CSV ({len(flat)} rows): {out_path}")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run all ABC-Q experiments with resume and optional dry-run.")
    p.add_argument("--dry-run", action="store_true", help="Fast smoke test; results under results/_dry_run/")
    p.add_argument("--force", action="store_true", help="Re-run even if RUN_SUCCEEDED exists.")
    p.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated basenames (e.g. abc_q_cifar10_full.py) to run subset.",
    )
    args = p.parse_args(argv)
    dry = bool(args.dry_run)
    only_set = {s.strip() for s in args.only.split(",") if s.strip()}

    exps = _EXPERIMENTS
    if only_set:
        exps = [e for e in exps if e[0] in only_set]
        if not exps:
            print("[error] --only matched no experiments.", file=sys.stderr)
            return 2

    for script, bb, ds, nc in exps:
        rc = _run_one(script, bb, ds, dry, bool(args.force))
        if rc != 0:
            print(f"[fail] {script} exited {rc}; stop. Re-run ``python master_run.py`` to resume.", file=sys.stderr)
            agg = (
                os.path.join(_RESULTS, "all_experiments_summary_dry.csv")
                if dry
                else os.path.join(_RESULTS, "all_experiments_summary.csv")
            )
            write_aggregate_csv(exps, dry, agg)
            return rc

    agg = (
        os.path.join(_RESULTS, "all_experiments_summary_dry.csv")
        if dry
        else os.path.join(_RESULTS, "all_experiments_summary.csv")
    )
    write_aggregate_csv(exps, dry, agg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
