#!/usr/bin/env python3
"""Quantize a single saved Keras model (per-layer 2/4/8 uniform min–max on Conv/Dense weights).

Modes:
  uniform  — every quantizable layer uses the same bit width (--uniform-bits).
  manual   — supply --bits-config "8,4,4,..." (one integer per quantizable layer, in layer order).
  abc      — run ABC-Q search on the calibration set (same logic as abc_q_cifar10_full).

Calibration data: either --calib-npz (keys x, y) or --cifar-calib N (shuffled CIFAR-10 train).

Images in npz should be CIFAR-like: shape (N,32,32,3), float32 0–255 or uint8 (normalized internally
if you use the same pipeline as the ABC scripts). For other input sizes, use a model trained on that
size and preprocess x in the npz to match what the model expects before saving."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

import abc_q_cifar10_full as abcq


def _load_calib_npz(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load calibration arrays from .npz; expects keys x,y (or x_calib, y_calib)."""
    d = np.load(path)
    if "x" in d.files and "y" in d.files:
        x, y = d["x"], d["y"]
    elif "x_calib" in d.files and "y_calib" in d.files:
        x, y = d["x_calib"], d["y_calib"]
    else:
        raise KeyError(f"{path}: need keys (x,y) or (x_calib, y_calib); got {d.files}")
    y = y.squeeze().astype(np.int32)
    if x.dtype != np.float32:
        x = x.astype(np.float32)
    return x, y


def _cifar_calib(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """First n samples from shuffled CIFAR-10 training set (float32 0–255)."""
    (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
    y_train = y_train.squeeze().astype(np.int32)
    x_train = x_train.astype(np.float32)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(x_train))
    return x_train[perm[:n]], y_train[perm[:n]]


def _calib_accuracy(model: tf.keras.Model, x: np.ndarray, y: np.ndarray, batch: int) -> float:
    """Calibration accuracy using same normalization as ABC code (no train augmentation)."""
    ds = abcq.make_dataset(x, y, batch_size=batch, shuffle=False, augment=False)
    _, acc = model.evaluate(ds, verbose=0)
    return float(acc)


def _create_demo_cifar_model(path: str, seed: int) -> None:
    """Train a small ResNet-20 CIFAR model and save (for trying this script without your own weights)."""
    abcq.set_seed(seed)
    (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
    y_train = y_train.squeeze().astype(np.int32)
    x_train = x_train.astype(np.float32)
    rng = np.random.RandomState(seed)
    sel = rng.permutation(len(x_train))[:5000]
    x_sub, y_sub = x_train[sel], y_train[sel]
    train_ds = abcq.make_dataset(x_sub, y_sub, batch_size=64, shuffle=True, augment=True)
    steps_per_epoch = max(1, len(x_sub) // 64)
    total_steps = steps_per_epoch * 8
    lr = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-3,
        decay_steps=total_steps,
        alpha=0.08,
    )
    model = abcq.build_model(learning_rate=lr)
    model.fit(train_ds, epochs=8, verbose=1)
    model.save(path)
    print(f"Demo model saved to {path} (CIFAR-10, same architecture as ABC-Q).")


def main() -> None:
    p = argparse.ArgumentParser(description="Post-train per-layer weight quantization for one Keras model.")
    p.add_argument(
        "--create-demo-model",
        metavar="PATH",
        help="Only: train a small CIFAR-10 model and save it to PATH (~1–3 min on GPU), then exit.",
    )
    p.add_argument("--model", help="Path to SavedModel / .keras / .h5")
    p.add_argument("--out", help="Where to save quantized model (.keras recommended)")
    p.add_argument(
        "--mode",
        choices=["uniform", "manual", "abc"],
        help="uniform | manual (list) | abc (search)",
    )
    p.add_argument("--uniform-bits", type=int, default=8, choices=[2, 4, 8])
    p.add_argument("--bits-config", type=str, help='manual only: e.g. "8,4,4,8,..."')
    p.add_argument("--calib-npz", type=str, help="Calibration .npz with x,y or x_calib,y_calib")
    p.add_argument("--cifar-calib", type=int, help="Use N random CIFAR-10 train images as calibration")
    p.add_argument("--calib-batch", type=int, default=128)
    p.add_argument("--seed", type=int, default=abcq.SEED)
    p.add_argument("--abc-cycles", type=int, default=12)
    p.add_argument("--abc-bees", type=int, default=10)
    p.add_argument("--abc-scout-limit", type=int, default=5)
    p.add_argument(
        "--finetune-epochs",
        type=int,
        default=0,
        help="Optional QAT-style fine-tune after baking in quantization; needs --finetune-npz or --cifar-train",
    )
    p.add_argument("--finetune-npz", type=str, help="Training npz with x,y for fine-tune")
    p.add_argument("--cifar-train", type=int, default=0, help="Use N CIFAR train images for fine-tune")
    p.add_argument("--finetune-lr", type=float, default=1e-4)
    args = p.parse_args()

    if args.create_demo_model:
        out_demo = Path(args.create_demo_model).expanduser().resolve()
        out_demo.parent.mkdir(parents=True, exist_ok=True)
        _create_demo_cifar_model(str(out_demo), args.seed)
        print(
            f"\nNext: python quantize_one_model.py --model {out_demo} --out q8.keras "
            f"--mode uniform --uniform-bits 8 --cifar-calib 1000"
        )
        return

    if not args.model or not args.out or not args.mode:
        print("Quantization requires --model, --out, and --mode (or use only --create-demo-model PATH).", file=sys.stderr)
        sys.exit(1)

    model_path = Path(args.model).expanduser()
    if not model_path.is_file():
        print(
            f"Model file not found: {model_path.resolve()}\n\n"
            "The example `my_model.keras` was only a placeholder. Use a real path after you save one, e.g.:\n"
            "  model.save('my_model.keras')  # after training in your script\n\n"
            "Or generate a test model compatible with this repo:\n"
            f"  python quantize_one_model.py --create-demo-model {model_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.calib_npz is None and args.cifar_calib is None:
        print("Provide --calib-npz or --cifar-calib N", file=sys.stderr)
        sys.exit(1)
    if args.calib_npz is not None:
        x_calib, y_calib = _load_calib_npz(args.calib_npz)
    else:
        x_calib, y_calib = _cifar_calib(args.cifar_calib, args.seed)

    abcq.set_seed(args.seed)
    model = tf.keras.models.load_model(str(model_path), compile=False)
    num_classes = int(model.output_shape[-1])
    loss = abcq.sparse_cce_with_label_smoothing(num_classes, abcq.LABEL_SMOOTHING)
    model.compile(optimizer=Adam(learning_rate=args.finetune_lr), loss=loss, metrics=["accuracy"])

    quant_idx = abcq.get_quantizable_layers(model)
    if not quant_idx:
        print("No quantizable Conv2D/Dense layers with weights found.", file=sys.stderr)
        sys.exit(1)
    param_counts = abcq.layer_param_counts(model, quant_idx)

    acc_before = _calib_accuracy(model, x_calib, y_calib, args.calib_batch)
    print(f"Float model calibration accuracy: {acc_before:.4f} ({len(quant_idx)} quantizable layers)")

    bits: np.ndarray
    if args.mode == "uniform":
        bits = np.full(len(quant_idx), args.uniform_bits, dtype=np.int32)
        abcq.apply_best_config_permanently(model, quant_idx, bits)
    elif args.mode == "manual":
        if not args.bits_config:
            print("--bits-config required for manual mode", file=sys.stderr)
            sys.exit(1)
        vals = [int(s.strip()) for s in args.bits_config.split(",")]
        if len(vals) != len(quant_idx):
            print(
                f"bits-config length {len(vals)} != number of quantizable layers {len(quant_idx)}",
                file=sys.stderr,
            )
            sys.exit(1)
        if any(b not in (2, 4, 8) for b in vals):
            print("Each bit width must be 2, 4, or 8", file=sys.stderr)
            sys.exit(1)
        bits = np.array(vals, dtype=np.int32)
        abcq.apply_best_config_permanently(model, quant_idx, bits)
    else:
        sens = abcq.compute_layer_sensitivity(model, quant_idx, x_calib, y_calib)
        best, _ = abcq.run_abc_q(
            model=model,
            quantizable_layers=quant_idx,
            x_calib=x_calib,
            y_calib=y_calib,
            param_counts=param_counts,
            sensitivity=sens,
            num_bees=args.abc_bees,
            cycles=args.abc_cycles,
            scout_limit=args.abc_scout_limit,
        )
        bits = best.bit_config
        print(f"ABC best bits: {abcq.format_bits(bits)}")
        print(
            f"ABC calib acc={best.accuracy:.4f} fitness={best.fitness:.4f} "
            f"BOPs={best.bops_ratio:.4f}"
        )
        abcq.apply_best_config_permanently(model, quant_idx, bits)

    acc_q = _calib_accuracy(model, x_calib, y_calib, args.calib_batch)
    print(f"Quantized calibration accuracy: {acc_q:.4f}")

    if args.finetune_epochs > 0:
        if args.finetune_npz:
            x_tr, y_tr = _load_calib_npz(args.finetune_npz)
        elif args.cifar_train > 0:
            (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
            y_train = y_train.squeeze().astype(np.int32)
            x_train = x_train.astype(np.float32)
            rng = np.random.RandomState(args.seed + 1)
            perm = rng.permutation(len(x_train))
            x_tr = x_train[perm[: args.cifar_train]]
            y_tr = y_train[perm[: args.cifar_train]]
        else:
            print("Fine-tune requested: add --finetune-npz or --cifar-train N", file=sys.stderr)
            sys.exit(1)
        train_ds = abcq.make_dataset(x_tr, y_tr, batch_size=64, shuffle=True, augment=True)
        model.fit(train_ds, epochs=args.finetune_epochs, verbose=2)
        acc_ft = _calib_accuracy(model, x_calib, y_calib, args.calib_batch)
        print(f"After fine-tune, calibration accuracy: {acc_ft:.4f}")

    model.save(args.out)
    print(f"Saved quantized model to {args.out}")


if __name__ == "__main__":
    main()
