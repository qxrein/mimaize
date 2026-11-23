#!/usr/bin/env python3
"""ABC-Q v2 — short quantisation-aware fine-tuning (QAT) of searched configurations.

The v1 pipeline "fine-tuned" by quantising the weights once and then training normally, so the
weights left the quantisation grid after the first optimiser step and activations were not quantised:
the reported post-fine-tuning accuracy was effectively that of a float model. Here the network stays
quantised throughout:

* float *shadow* kernels are updated; on every step they are re-quantised (per-channel symmetric) and
  written into the model, with the straight-through estimator (STE) passing gradients to the shadows;
* layer inputs are fake-quantised with STE using the calibration ranges;
* BatchNorm stays in inference mode (running statistics frozen), biases/BN affine params are trained;
* the reported accuracy is that of the quantised network (weights on the grid, activations quantised).

    python v2_qat.py --backbone resnet20 --dataset cifar10 --budget 0.25 --epochs 2
"""

from __future__ import annotations

import argparse
import json
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument("--backbone", required=True)
ap.add_argument("--dataset", required=True)
ap.add_argument("--budget", type=float, default=0.25)
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--methods", nargs="+", default=["uniform4", "cmaes", "ga", "fisher_mckp", "abc_v2"])
ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
ap.add_argument("--seeds", type=int, nargs="*", default=None,
                help="multi-configuration mode: QAT the configuration of each listed search seed (uniform4: fine-tuning seed); results -> qat_B<budget>_multi.json")
args = ap.parse_args()

import numpy as np
import tensorflow as tf

if args.device == "cpu":
    tf.config.set_visible_devices([], "GPU")

import abc_q_core as q  # noqa: E402
from v2_engine import FastEvaluator, normalize  # noqa: E402

q.BACKBONE = args.backbone
q.DATASET = args.dataset
q.NUM_CLASSES = 100 if args.dataset == "cifar100" else 10
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results_v2", args.backbone, args.dataset)


def augment(x: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    n = x.shape[0]
    xp = np.pad(x, ((0, 0), (4, 4), (4, 4), (0, 0)), mode="reflect")
    oy, ox = rng.randint(0, 9, size=n), rng.randint(0, 9, size=n)
    rows = oy[:, None] + np.arange(32)[None, :]
    cols = ox[:, None] + np.arange(32)[None, :]
    out = xp[np.arange(n)[:, None, None], rows[:, :, None], cols[:, None, :]]
    flip = rng.rand(n) < 0.5
    out[flip] = out[flip, :, ::-1]
    return out


def tf_quant_kernel(w: tf.Tensor, bits: int, ltype: str) -> tf.Tensor:
    axes = [0, 1] if ltype == "depthwise" else ([0, 1, 2] if ltype == "conv" else [0])
    qmax = float(2 ** (bits - 1) - 1)
    amax = tf.reduce_max(tf.abs(w), axis=axes, keepdims=True)
    scale = tf.where(amax > 0, amax / qmax, tf.ones_like(amax))
    return tf.clip_by_value(tf.round(w / scale), -qmax, qmax) * scale


def pick_configs() -> dict:
    runs = []
    with open(os.path.join(OUT, "runs.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if abs(r["budget"] - args.budget) < 1e-6 and r.get("config") is not None:
                runs.append(r)
    out = {}
    for m in args.methods:
        if m == "uniform4":
            continue
        rs = sorted([r for r in runs if r["method"] == m], key=lambda r: r["calib_acc"])
        if rs:
            out[m] = rs[len(rs) // 2]  # median-calibration seed: representative, not cherry-picked
    return out


def main() -> None:
    cfg = q.get_experiment_config()
    data = q.load_dataset_splits(cfg)
    x_tr = normalize(data["x_train"])
    y_tr = data["y_train"].astype(np.int64)
    chosen = pick_configs()
    multi = args.seeds is not None and len(args.seeds) > 0
    suffix = "_multi" if multi else ""
    results_path = os.path.join(OUT, f"qat_B{int(round(args.budget * 1000)):03d}{suffix}.json")
    results = json.load(open(results_path)) if os.path.exists(results_path) else {}
    by_seed = {}
    if multi:
        with open(os.path.join(OUT, "runs.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                if abs(r["budget"] - args.budget) < 1e-6 and r.get("config") is not None:
                    by_seed[(r["method"], r["seed"])] = r
    jobs = [(m, sd) for m in args.methods for sd in (args.seeds if multi else [None])]
    for m, qseed in jobs:
        key = m if qseed is None else f"{m}|s{qseed}"
        if key in results:
            continue
        model = q.build_model()
        model.load_weights(os.path.join(OUT, "float.weights.h5"))
        ev = FastEvaluator(model, data["x_calib"], data["y_calib"], data["x_test"], data["y_test"])
        if m == "uniform4":
            bits = np.full(ev.L, 4, dtype=np.int64)
            src = {"seed": None, "calib_acc": None}
        elif qseed is not None:
            if (m, qseed) not in by_seed and (m, 0) in by_seed and m.endswith("mckp"):
                src = by_seed[(m, 0)]  # deterministic allocator: same configuration, different fine-tuning seed
            elif (m, qseed) not in by_seed:
                continue
            else:
                src = by_seed[(m, qseed)]
            bits = np.array(src["config"], dtype=np.int64)
        else:
            if m not in chosen:
                continue
            bits = np.array(chosen[m]["config"], dtype=np.int64)
            src = chosen[m]
        ptq_test = ev.test_acc(bits)
        # --- QAT
        ev.ste = True
        shadows = [tf.keras.Variable(w, trainable=True, name=f"shadow_{i}") for i, w in enumerate(ev.float_kernels)]
        kref = {id(v) for v in ev.kernel_vars}
        others = [v for v in model.trainable_variables if id(v) not in kref]
        steps_per_epoch = int(np.ceil(len(x_tr) / 128))
        sched = tf.keras.optimizers.schedules.CosineDecay(args.lr, steps_per_epoch * args.epochs, alpha=0.1)
        opt = tf.keras.optimizers.Adam(sched)
        ev.set_config(bits)  # sets act bit-widths / enables act quant
        lce = tf.keras.losses.SparseCategoricalCrossentropy()
        bl = [int(b) for b in bits]

        @tf.function
        def step(x, y):
            for i in range(ev.L):
                ev.kernel_vars[i].assign(tf_quant_kernel(shadows[i], bl[i], ev.types[i]))
            with tf.GradientTape() as tape:
                p = model(x, training=False)
                loss = lce(y, p)
            g = tape.gradient(loss, ev.kernel_vars + others)
            opt.apply_gradients(zip(g, shadows + others))
            return loss

        rng = np.random.RandomState(0 if qseed is None else 100 + qseed)
        t0 = time.time()
        for ep in range(args.epochs):
            xa = augment(x_tr, rng)
            perm = rng.permutation(len(xa))
            for s in range(0, len(xa), 128):
                idx = perm[s:s + 128]
                step(tf.constant(xa[idx]), tf.constant(y_tr[idx]))
        for i in range(ev.L):
            ev.kernel_vars[i].assign(tf_quant_kernel(shadows[i], bl[i], ev.types[i]))
        # quantised-model test accuracy (kernels on grid, activations quantised)
        correct = 0
        for s in range(0, len(ev.xt), 1000):
            pr = ev._fwd(tf.constant(ev.xt[s:s + 1000])).numpy()
            correct += int(np.sum(np.argmax(pr, 1) == ev.yt[s:s + 1000]))
        qat_test = correct / len(ev.xt)
        # sanity: weights really on the grid
        on_grid = all(np.allclose(ev.kernel_vars[i].numpy(),
                                  tf_quant_kernel(ev.kernel_vars[i], bl[i], ev.types[i]).numpy(), atol=1e-6)
                      for i in range(ev.L))
        results[key] = {"method": m, "qat_seed": qseed, "seed": src.get("seed"), "search_calib": src.get("calib_acc"), "ptq_test": ptq_test,
                      "qat_test": qat_test, "epochs": args.epochs, "bops": ev.bops_ratio(bits),
                      "weights_on_grid": bool(on_grid), "wall_s": time.time() - t0}
        print(key, results[key], flush=True)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        tf.keras.backend.clear_session()


if __name__ == "__main__":
    main()
