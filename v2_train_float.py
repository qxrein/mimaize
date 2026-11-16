#!/usr/bin/env python3
"""ABC-Q v2 — train the float (FP32) backbone once per (backbone, dataset).

Usage:
    python v2_train_float.py --backbone resnet20 --dataset cifar10 --epochs 60 --device gpu
    python v2_train_float.py --backbone mobilenetv2 --dataset cifar100 --epochs 60 --device cpu

Notes
-----
* Architecture definitions and the 49k/1k/10k split are reused from ``abc_q_core`` so the
  float models match the ones described in the paper.
* Augmentation (reflect-pad 4 + random 32x32 crop + horizontal flip) is vectorised in NumPy and
  regenerated every epoch; this is ~4x faster than per-image tf.image ops on Apple Silicon.
* Depthwise-heavy nets (MobileNetV2, EfficientNet-B0) are faster on the CPU than on tensorflow-metal.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--backbone", required=True, choices=["resnet20", "mobilenetv2", "efficientnetb0"])
ap.add_argument("--dataset", required=True, choices=["cifar10", "cifar100"])
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

import numpy as np
import tensorflow as tf

if args.device == "cpu":
    tf.config.set_visible_devices([], "GPU")

import abc_q_core as q  # noqa: E402

q.BACKBONE = args.backbone
q.DATASET = args.dataset
q.NUM_CLASSES = 100 if args.dataset == "cifar100" else 10

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results_v2", args.backbone, args.dataset)
os.makedirs(OUT, exist_ok=True)


def normalize(x: np.ndarray) -> np.ndarray:
    return ((x / 255.0 - q.CIFAR_MEAN) / q.CIFAR_STD).astype(np.float32)


def augment(x: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    n = x.shape[0]
    xp = np.pad(x, ((0, 0), (4, 4), (4, 4), (0, 0)), mode="reflect")
    oy = rng.randint(0, 9, size=n)
    ox = rng.randint(0, 9, size=n)
    rows = oy[:, None] + np.arange(32)[None, :]
    cols = ox[:, None] + np.arange(32)[None, :]
    out = xp[np.arange(n)[:, None, None], rows[:, :, None], cols[:, None, :]]
    flip = rng.rand(n) < 0.5
    out[flip] = out[flip, :, ::-1]
    return out


def main() -> None:
    q.set_seed(args.seed)
    cfg = q.get_experiment_config()
    data = q.load_dataset_splits(cfg)
    x_tr = normalize(data["x_train"])
    y_tr = data["y_train"]
    x_te = normalize(data["x_test"])
    y_te = data["y_test"]

    steps = int(np.ceil(len(x_tr) / args.batch))
    sched = tf.keras.optimizers.schedules.CosineDecay(1e-3, decay_steps=steps * args.epochs, alpha=0.02)
    model = q.build_model(learning_rate=sched)
    rng = np.random.RandomState(args.seed)
    log = []
    t0 = time.time()
    for ep in range(args.epochs):
        xa = augment(x_tr, rng)
        h = model.fit(xa, y_tr, batch_size=args.batch, epochs=1, verbose=0, shuffle=True)
        rec = {"epoch": ep + 1, "loss": float(h.history["loss"][0]), "acc": float(h.history["accuracy"][0]),
               "elapsed_s": time.time() - t0}
        if (ep + 1) % 10 == 0 or ep + 1 == args.epochs:
            _, te = model.evaluate(x_te, y_te, batch_size=500, verbose=0)
            rec["test_acc"] = float(te)
        log.append(rec)
        print(json.dumps(rec), flush=True)
    model.save_weights(os.path.join(OUT, "float.weights.h5"))
    _, te = model.evaluate(x_te, y_te, batch_size=500, verbose=0)
    with open(os.path.join(OUT, "float_train_log.json"), "w") as f:
        json.dump({"args": vars(args), "float_test_acc": float(te), "log": log}, f, indent=2)
    print(f"[done] float test acc = {te:.4f} -> {OUT}")


if __name__ == "__main__":
    main()
