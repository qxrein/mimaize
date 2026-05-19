#!/usr/bin/env python3
"""Re-capture ABC vs CMA convergence plots for completed experiments."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent

# (entry script, backbone, dataset)
CIFAR10_CONVERGENCE = [
    ("abc_q_cifar10_resnet20.py", "resnet20", "cifar10"),
    ("abc_q_cifar10_full.py", "mobilenetv2", "cifar10"),
    ("abc_q_efficientnet_cifar10.py", "efficientnetb0", "cifar10"),
]


def run_capture(entry_script: str) -> None:
    path = REPO / entry_script
    spec = importlib.util.spec_from_file_location("_abc_entry", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import abc_q_core as q

    q.capture_convergence_plot()


def main() -> int:
    for entry, backbone, dataset in CIFAR10_CONVERGENCE:
        ckpt = REPO / "results" / backbone / dataset / "float_checkpoint.weights.h5"
        if not ckpt.exists():
            print(f"skip {backbone}/{dataset}: no checkpoint at {ckpt}")
            continue
        print(f"\n=== {backbone} / {dataset} ===")
        run_capture(entry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
