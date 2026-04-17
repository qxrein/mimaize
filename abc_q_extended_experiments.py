"""Extended ABC-Q experimental runner for CIFAR-10/CIFAR-100 and multiple backbones.

This script adds:
- CIFAR-100 experiments (ResNet-20, ResNet-56, MobileNetV2)
- ResNet-56 backbone on CIFAR-10/CIFAR-100
- EfficientNet-B0 on CIFAR-10 (30-epoch pre-fine-tune before quant search)
- Wall-clock timing CSV/plot
- Extended MobileNetV2 Pareto/convergence/heatmap figures

Outputs are saved under ``results/`` as CSV and PNG.
"""

from __future__ import annotations

import copy
import csv
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

import abc_q_cifar10_full as core
from abc_q_stats import compute_pareto_frontier, print_paper_table, wilcoxon_paired_pvalue

plt.rcParams.update({"font.size": 12})

RESULTS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
A100_OPTIMIZED = os.environ.get("A100_OPTIMIZED", "1") == "1"
USE_MIXED_PRECISION = os.environ.get("USE_MIXED_PRECISION", "1") == "1"
USE_XLA = os.environ.get("USE_XLA", "1") == "1"
ALLOW_TF32 = os.environ.get("ALLOW_TF32", "1") == "1"
CSV_COLUMNS = [
    "method",
    "dataset",
    "model",
    "seed",
    "calib_acc",
    "test_acc",
    "bops_ratio",
    "avg_bits",
    "search_time_s",
    "finetune_time_s",
]


@dataclass
class ExpConfig:
    dataset: str  # cifar10 or cifar100
    model_name: str  # resnet20, resnet56, mobilenetv2, efficientnetb0
    seeds: Sequence[int]
    train_samples: int = 49000
    calib_samples: int = 1000
    test_samples: int = 10000
    pretrain_epochs: int = 100
    finetune_epochs: int = 8
    abc_cycles: int = 20
    num_bees: int = 16
    scout_limit: int = 8
    batch_size: int = 128
    run_sensitivity_ablation: bool = True
    use_sensitivity_prior: bool = True
    random_cloud_samples: int = 500
    data_split_seed: int = 0


def configure_a100_runtime() -> None:
    """Enable common A100 speedups (mixed precision, TF32, XLA) when requested."""
    if not A100_OPTIMIZED:
        return
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass
    if USE_XLA:
        tf.config.optimizer.set_jit(True)
    if USE_MIXED_PRECISION:
        mixed_precision.set_global_policy("mixed_float16")
    try:
        tf.config.experimental.enable_tensor_float_32_execution(ALLOW_TF32)
    except Exception:
        pass


def make_dataset_fast(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    augment: bool = False,
    shuffle_seed: int | None = None,
) -> tf.data.Dataset:
    """Dataset pipeline with non-deterministic execution for throughput."""
    ds = core.make_dataset(x, y, batch_size, shuffle=shuffle, augment=augment, shuffle_seed=shuffle_seed)
    opts = tf.data.Options()
    opts.experimental_deterministic = False
    return ds.with_options(opts)


def make_dirs() -> Dict[str, str]:
    paths = {
        "root": RESULTS_ROOT,
        "cifar100": os.path.join(RESULTS_ROOT, "cifar100"),
        "timing": os.path.join(RESULTS_ROOT, "timing"),
        "pareto": os.path.join(RESULTS_ROOT, "pareto"),
        "convergence": os.path.join(RESULTS_ROOT, "convergence"),
        "heatmap": os.path.join(RESULTS_ROOT, "heatmap"),
        "csv": os.path.join(RESULTS_ROOT, "csv"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def load_dataset(dataset: str, config: ExpConfig) -> Dict[str, np.ndarray]:
    if dataset == "cifar10":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    elif dataset == "cifar100":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode="fine")
    else:
        raise ValueError(dataset)
    y_train = y_train.squeeze().astype(np.int32)
    y_test = y_test.squeeze().astype(np.int32)
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)

    rng = np.random.RandomState(int(config.data_split_seed))
    tr = rng.permutation(len(x_train))
    te = rng.permutation(len(x_test))
    x_train, y_train = x_train[tr], y_train[tr]
    x_test, y_test = x_test[te], y_test[te]

    tn = int(config.train_samples)
    cn = int(config.calib_samples)
    sn = int(config.test_samples)
    return {
        "x_train": x_train[:tn],
        "y_train": y_train[:tn],
        "x_calib": x_train[tn : tn + cn],
        "y_calib": y_train[tn : tn + cn],
        "x_test": x_test[:sn],
        "y_test": y_test[:sn],
    }


def _resnet_block(x: tf.Tensor, filters: int, reg: Any, stride: int = 1) -> tf.Tensor:
    shortcut = x
    out = Conv2D(filters, 3, strides=stride, padding="same", use_bias=False, kernel_regularizer=reg)(x)
    out = BatchNormalization()(out)
    out = Activation("relu")(out)
    out = Conv2D(filters, 3, padding="same", use_bias=False, kernel_regularizer=reg)(out)
    out = BatchNormalization()(out)
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, 1, strides=stride, padding="same", use_bias=False, kernel_regularizer=reg)(
            shortcut
        )
        shortcut = BatchNormalization()(shortcut)
    out = Add()([out, shortcut])
    out = Activation("relu")(out)
    return out


def build_resnet(depth: int, num_classes: int, lr_schedule: Any) -> Model:
    # depth = 6n + 2. For 20 => n=3, for 56 => n=9.
    n = (depth - 2) // 6
    reg = l2(core.WEIGHT_DECAY)
    inputs = Input(shape=(32, 32, 3))
    x = Conv2D(16, 3, padding="same", use_bias=False, kernel_regularizer=reg)(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    for _ in range(n):
        x = _resnet_block(x, 16, reg, stride=1)
    x = _resnet_block(x, 32, reg, stride=2)
    for _ in range(n - 1):
        x = _resnet_block(x, 32, reg, stride=1)
    x = _resnet_block(x, 64, reg, stride=2)
    for _ in range(n - 1):
        x = _resnet_block(x, 64, reg, stride=1)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(core.CLASSIFIER_DROPOUT)(x)
    logits = Dense(num_classes, activation=None, kernel_regularizer=l2(core.WEIGHT_DECAY))(x)
    outputs = Activation("softmax")(logits)
    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=lr_schedule),
        loss=core.sparse_cce_with_label_smoothing(num_classes, core.LABEL_SMOOTHING),
        metrics=["accuracy"],
        jit_compile=USE_XLA,
    )
    return model


def build_mobilenetv2(num_classes: int, lr_schedule: Any) -> Model:
    base = tf.keras.applications.MobileNetV2(
        input_shape=(32, 32, 3),
        include_top=False,
        weights=None,
        alpha=1.0,
        pooling=None,
    )
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(core.CLASSIFIER_DROPOUT)(x)
    logits = Dense(num_classes, activation=None, kernel_regularizer=l2(core.WEIGHT_DECAY))(x)
    outputs = Activation("softmax")(logits)
    model = Model(base.input, outputs)
    model.compile(
        optimizer=Adam(learning_rate=lr_schedule),
        loss=core.sparse_cce_with_label_smoothing(num_classes, core.LABEL_SMOOTHING),
        metrics=["accuracy"],
        jit_compile=USE_XLA,
    )
    return model


def build_efficientnetb0(num_classes: int, lr_schedule: Any) -> Model:
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(32, 32, 3),
        pooling=None,
    )
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.2)(x)
    logits = Dense(num_classes, activation=None, kernel_regularizer=l2(core.WEIGHT_DECAY))(x)
    outputs = Activation("softmax")(logits)
    model = Model(base.input, outputs)
    model.compile(
        optimizer=Adam(learning_rate=lr_schedule),
        loss=core.sparse_cce_with_label_smoothing(num_classes, core.LABEL_SMOOTHING),
        metrics=["accuracy"],
        jit_compile=USE_XLA,
    )
    return model


def build_model(model_name: str, num_classes: int, lr_schedule: Any) -> Model:
    if model_name == "resnet20":
        return build_resnet(20, num_classes, lr_schedule)
    if model_name == "resnet56":
        return build_resnet(56, num_classes, lr_schedule)
    if model_name == "mobilenetv2":
        return build_mobilenetv2(num_classes, lr_schedule)
    if model_name == "efficientnetb0":
        return build_efficientnetb0(num_classes, lr_schedule)
    raise ValueError(model_name)


def avg_bits(bit_config: np.ndarray, param_counts: np.ndarray) -> float:
    return float(np.sum(bit_config.astype(np.float64) * param_counts) / np.sum(param_counts))


def eval_test_after_finetune(
    model_name: str,
    num_classes: int,
    float_weights: List[np.ndarray],
    quant_names: Sequence[str],
    bit_config: np.ndarray,
    ds_train: tf.data.Dataset,
    ds_test: tf.data.Dataset,
    train_size: int,
    batch_size: int,
    finetune_epochs: int,
) -> Tuple[float, float]:
    """Fine-tune quantized model and return (test_acc, finetune_seconds)."""
    m = build_model(model_name, num_classes, lr_schedule=1e-3)
    m.set_weights(copy.deepcopy(float_weights))
    steps = max(1, train_size // batch_size) * finetune_epochs
    ft_lr = tf.keras.optimizers.schedules.CosineDecay(2e-4, decay_steps=steps, alpha=0.15)
    m.compile(
        optimizer=Adam(learning_rate=ft_lr),
        loss=core.sparse_cce_with_label_smoothing(num_classes, core.LABEL_SMOOTHING),
        metrics=["accuracy"],
        jit_compile=USE_XLA,
    )
    core.apply_best_config_permanently(m, quant_names, bit_config)
    t0 = time.perf_counter()
    m.fit(ds_train, epochs=finetune_epochs, verbose=0)
    finetune_t = time.perf_counter() - t0
    _, test_acc = m.evaluate(ds_test, verbose=0)
    return float(test_acc), float(finetune_t)


def run_one_combo(cfg: ExpConfig, paths: Dict[str, str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    dataset, model_name, seeds = cfg.dataset, cfg.model_name, list(cfg.seeds)
    num_classes = 100 if dataset == "cifar100" else 10
    data = load_dataset(dataset, cfg)
    run_batch = int(cfg.batch_size)
    if A100_OPTIMIZED and model_name in {"resnet20", "resnet56"}:
        run_batch = max(run_batch, 256)
    elif A100_OPTIMIZED and model_name in {"mobilenetv2", "efficientnetb0"}:
        run_batch = max(run_batch, 192)
    ds_train = make_dataset_fast(data["x_train"], data["y_train"], run_batch, shuffle=True, augment=True)
    ds_test = make_dataset_fast(data["x_test"], data["y_test"], run_batch, shuffle=False)

    # float pretrain
    steps = max(1, len(data["x_train"]) // run_batch)
    lr = tf.keras.optimizers.schedules.CosineDecay(1e-3, decay_steps=steps * cfg.pretrain_epochs, alpha=0.06)
    core.set_seed(seeds[0])
    model = build_model(model_name, num_classes, lr)
    if model_name == "efficientnetb0":
        model.fit(ds_train, epochs=30, verbose=0)  # mandatory task 3 stage
    model.fit(ds_train, epochs=cfg.pretrain_epochs, verbose=0)
    _, float_test_acc = model.evaluate(ds_test, verbose=0)

    float_weights = copy.deepcopy(model.get_weights())
    quant_names = core.get_quantizable_layer_names(model)
    param_counts = core.layer_param_counts(model, quant_names)
    layer_bops_specs = core.build_layer_bops_specs(model, quant_names)

    # baseline once (uniform)
    act_stats = core.collect_activation_stats(model, quant_names, data["x_calib"], data["y_calib"], batch_size=run_batch)
    ft_ds_train = make_dataset_fast(data["x_train"], data["y_train"], run_batch, shuffle=True, augment=True)
    ft_ds_test = make_dataset_fast(data["x_test"], data["y_test"], run_batch, shuffle=False)

    sensitivity_t0 = time.perf_counter()
    sensitivity = core.compute_layer_sensitivity(model, quant_names, data["x_calib"], data["y_calib"])
    sensitivity_time = time.perf_counter() - sensitivity_t0

    uniform8 = core.evaluate_baseline_config(
        model, quant_names, 8, data["x_calib"], data["y_calib"], param_counts, layer_bops_specs, act_stats, True
    )
    uniform4 = core.evaluate_baseline_config(
        model, quant_names, 4, data["x_calib"], data["y_calib"], param_counts, layer_bops_specs, act_stats, True
    )
    uniform2 = core.evaluate_baseline_config(
        model, quant_names, 2, data["x_calib"], data["y_calib"], param_counts, layer_bops_specs, act_stats, True
    )

    rows: List[Dict[str, Any]] = []
    timing_rows: List[Dict[str, Any]] = []
    conv_per_seed_abc: List[List[float]] = []
    conv_per_seed_cma: List[List[float]] = []
    bit_configs: Dict[str, List[np.ndarray]] = {"ABC-Q (ours)": [], "CMA-ES": [], "Random Search": []}

    for seed in seeds:
        core.set_seed(seed)
        model.set_weights(copy.deepcopy(float_weights))
        conv_abc: Dict[str, List[float]] = {}
        t0 = time.perf_counter()
        abc, budget = core.run_abc_q(
            model,
            quant_names,
            data["x_calib"],
            data["y_calib"],
            param_counts,
            sensitivity,
            cfg.num_bees,
            cfg.abc_cycles,
            cfg.scout_limit,
            layer_bops_specs,
            act_stats,
            True,
            True,
            conv_abc,
        )
        abc_t = time.perf_counter() - t0
        conv_per_seed_abc.append(conv_abc.get("abc_best_fitness", []))

        core.set_seed(seed)
        model.set_weights(copy.deepcopy(float_weights))
        t0 = time.perf_counter()
        rnd = core.run_random_search(
            model,
            quant_names,
            data["x_calib"],
            data["y_calib"],
            param_counts,
            sensitivity,
            budget,
            layer_bops_specs,
            act_stats,
            True,
        )
        rnd_t = time.perf_counter() - t0

        core.set_seed(seed)
        model.set_weights(copy.deepcopy(float_weights))
        conv_cma: Dict[str, List[float]] = {}
        t0 = time.perf_counter()
        cma = core.run_cmaes_baseline(
            model,
            quant_names,
            data["x_calib"],
            data["y_calib"],
            param_counts,
            budget,
            layer_bops_specs,
            act_stats,
            True,
            seed,
            conv_cma,
        )
        cma_t = time.perf_counter() - t0
        conv_per_seed_cma.append(conv_cma.get("cma_best_fitness", []))

        core.set_seed(seed)
        model.set_weights(copy.deepcopy(float_weights))
        abc_np, _ = core.run_abc_q(
            model,
            quant_names,
            data["x_calib"],
            data["y_calib"],
            param_counts,
            sensitivity,
            cfg.num_bees,
            cfg.abc_cycles,
            cfg.scout_limit,
            layer_bops_specs,
            act_stats,
            True,
            False,
            None,
        )

        for method, res, search_t in [
            ("ABC-Q (ours)", abc, abc_t),
            ("ABC-Q (no prior)", abc_np, abc_t),
            ("Random Search", rnd, rnd_t),
            ("CMA-ES", cma, cma_t),
        ]:
            test_acc, ft_t = eval_test_after_finetune(
                model_name,
                num_classes,
                float_weights,
                quant_names,
                res.bit_config,
                ft_ds_train,
                ft_ds_test,
                len(data["x_train"]),
                run_batch,
                cfg.finetune_epochs,
            )
            rows.append(
                {
                    "method": method,
                    "dataset": dataset,
                    "model": model_name,
                    "seed": seed,
                    "calib_acc": float(res.accuracy),
                    "test_acc": float(test_acc),
                    "bops_ratio": float(res.bops_weight_act),
                    "avg_bits": avg_bits(res.bit_config, param_counts),
                    "search_time_s": float(search_t),
                    "finetune_time_s": float(ft_t),
                }
            )
            bit_configs.setdefault(method, []).append(res.bit_config.copy())

        timing_rows.append(
            {
                "method": "timing",
                "dataset": dataset,
                "model": model_name,
                "seed": seed,
                "sensitivity_prior_s": sensitivity_time,
                "abc_search_s": abc_t,
                "cma_search_s": cma_t,
                "random_search_s": rnd_t,
                "fine_tune_s": np.mean([r["finetune_time_s"] for r in rows[-4:]]),
            }
        )

    # add uniform rows with test finetune for first seed only (replicated across seeds)
    for method, res in [("Uniform 8-bit", uniform8), ("Uniform 4-bit", uniform4), ("Uniform 2-bit", uniform2)]:
        for seed in seeds:
            test_acc, ft_t = eval_test_after_finetune(
                model_name,
                num_classes,
                float_weights,
                quant_names,
                res.bit_config,
                ft_ds_train,
                ft_ds_test,
                len(data["x_train"]),
                run_batch,
                cfg.finetune_epochs,
            )
            rows.append(
                {
                    "method": method,
                    "dataset": dataset,
                    "model": model_name,
                    "seed": seed,
                    "calib_acc": float(res.accuracy),
                    "test_acc": float(test_acc),
                    "bops_ratio": float(res.bops_weight_act),
                    "avg_bits": avg_bits(res.bit_config, param_counts),
                    "search_time_s": 0.0,
                    "finetune_time_s": float(ft_t),
                }
            )

    summary = {
        "dataset": dataset,
        "model": model_name,
        "float_test_acc": float(float_test_acc),
        "quant_layers": len(quant_names),
        "rows": rows,
        "timing_rows": timing_rows,
        "conv_abc": conv_per_seed_abc,
        "conv_cma": conv_per_seed_cma,
        "bit_configs": bit_configs,
        "sensitivity": sensitivity,
        "param_counts": param_counts,
        "budget": int(budget),
        "uniform": (uniform8, uniform4, uniform2),
        "last_best": rows[-1] if rows else None,
        "data": data,
        "model_obj": model,
        "quant_names": quant_names,
    }
    return rows, timing_rows, summary


def write_csv(path: str, rows: List[Dict[str, Any]], columns: Sequence[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(columns))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in columns})


def plot_timing_bar(path: str, timing_rows: List[Dict[str, Any]]) -> None:
    key_order = ["sensitivity_prior_s", "abc_search_s", "cma_search_s", "random_search_s", "fine_tune_s"]
    labels = ["Sensitivity", "ABC-Q", "CMA-ES", "Random", "Fine-tune"]
    combos = sorted({(r["dataset"], r["model"]) for r in timing_rows})
    fig, ax = plt.subplots(figsize=(10, max(4, len(combos) * 0.5)))
    y = np.arange(len(combos))
    left = np.zeros(len(combos), dtype=np.float64)
    for k, lab in zip(key_order, labels):
        vals = []
        for d, m in combos:
            xs = [float(r[k]) for r in timing_rows if r["dataset"] == d and r["model"] == m]
            vals.append(float(np.mean(xs)) if xs else 0.0)
        vals_np = np.array(vals, dtype=np.float64)
        ax.barh(y, vals_np, left=left, label=lab)
        left += vals_np
    ax.set_yticks(y)
    ax.set_yticklabels([f"{m}/{d}" for d, m in combos])
    ax.set_xlabel("Wall-clock time (s)")
    ax.set_title("Wall-clock timing by phase")
    ax.legend()
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_extended_pareto(path: str, random_pts: List[Tuple[float, float]], uniform: Tuple[Any, Any, Any], abc: Any, cma: Any) -> None:
    all_pts = list(random_pts) + [
        (uniform[0].bops_weight_act, uniform[0].accuracy),
        (uniform[1].bops_weight_act, uniform[1].accuracy),
        (uniform[2].bops_weight_act, uniform[2].accuracy),
        (abc.bops_weight_act, abc.accuracy),
        (cma.bops_weight_act, cma.accuracy),
    ]
    frontier = compute_pareto_frontier(all_pts)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    rb, ra = zip(*random_pts)
    ax.scatter(rb, ra, s=10, c="0.7", alpha=0.35, edgecolors="none", label="Random configs")
    if frontier:
        fx, fy = zip(*frontier)
        ax.plot(fx, fy, "k-", linewidth=1.5, label="Pareto frontier")
        # dominated region approximation: area below frontier up to max x
        x_ext = np.array(list(fx) + [max(rb)])
        y_ext = np.array(list(fy) + [fy[-1]])
        ax.fill_between(x_ext, 0.0, y_ext, color="0.85", alpha=0.4, label="Dominated region")
    ucols = ["#9467bd", "#8c564b", "#7f7f7f"]
    for i, (name, u) in enumerate(zip(["U8", "U4", "U2"], uniform)):
        ax.scatter([u.bops_weight_act], [u.accuracy], s=90, marker="s", c=ucols[i], label=name)
    ax.scatter([abc.bops_weight_act], [abc.accuracy], s=220, marker="*", c="#ff7f0e", label="ABC-Q")
    ax.scatter([cma.bops_weight_act], [cma.accuracy], s=90, marker="D", c="#1f77b4", label="CMA-ES")
    ax.set_xlabel("BOPs ratio")
    ax.set_ylabel("Calibration accuracy")
    ax.set_title("MobileNetV2/CIFAR-10 Pareto (extended random cloud)")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_convergence_per_seed(path: str, abc_seed_curves: List[List[float]], cma_seed_curves: List[List[float]]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for c in abc_seed_curves:
        if c:
            ax.plot(np.arange(1, len(c) + 1), c, color="#ff7f0e", alpha=0.25, linewidth=1.0)
    for c in cma_seed_curves:
        if c:
            ax.plot(np.arange(1, len(c) + 1), c, color="#1f77b4", alpha=0.25, linewidth=1.0)
    abc_max = max((len(c) for c in abc_seed_curves if c), default=0)
    cma_max = max((len(c) for c in cma_seed_curves if c), default=0)
    if abc_max:
        A = np.array([np.pad(np.array(c), (0, abc_max - len(c)), constant_values=np.nan) for c in abc_seed_curves if c])
        ax.plot(np.arange(1, abc_max + 1), np.nanmean(A, axis=0), color="#ff7f0e", linewidth=2.6, label="ABC mean")
    if cma_max:
        C = np.array([np.pad(np.array(c), (0, cma_max - len(c)), constant_values=np.nan) for c in cma_seed_curves if c])
        ax.plot(np.arange(1, cma_max + 1), np.nanmean(C, axis=0), color="#1f77b4", linewidth=2.6, label="CMA mean")
    ax.set_xlabel("Cycle / Eval index")
    ax.set_ylabel("Best-so-far fitness")
    ax.set_title("MobileNetV2 convergence per seed")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_bitwidth_heatmap(path: str, abc_bits: np.ndarray, cma_bits: np.ndarray, rnd_bits: np.ndarray) -> None:
    mat = np.vstack([abc_bits, cma_bits, rnd_bits]).astype(float)
    fig, ax = plt.subplots(figsize=(max(8, mat.shape[1] * 0.3), 3.2))
    im = ax.imshow(mat, aspect="auto", cmap=plt.get_cmap("RdYlGn"), vmin=2, vmax=8)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["ABC-Q", "CMA-ES", "Random"])
    ax.set_xlabel("Layer index")
    ax.set_title("MobileNetV2 bit-width allocation heatmap")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Bit-width")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def aggregate_bits(bit_list: Sequence[np.ndarray]) -> np.ndarray:
    arr = np.array(bit_list, dtype=np.int32)
    # nearest to mean among {2,4,8}
    mean = np.mean(arr, axis=0)
    choices = np.array([2, 4, 8], dtype=np.float64)
    idx = np.argmin(np.abs(mean[:, None] - choices[None, :]), axis=1)
    return choices[idx].astype(np.int32)


def summarize_stdout(all_rows: List[Dict[str, Any]]) -> None:
    combos = sorted({(r["dataset"], r["model"]) for r in all_rows})
    print("\n=== Final Summary (mean ± std over seeds) ===")
    for dataset, model in combos:
        rs = [r for r in all_rows if r["dataset"] == dataset and r["model"] == model]
        methods = sorted({r["method"] for r in rs})
        print(f"\n[{model} / {dataset}]")
        for m in methods:
            ms = [r for r in rs if r["method"] == m]
            ca = np.array([float(x["calib_acc"]) for x in ms], dtype=np.float64)
            ta = np.array([float(x["test_acc"]) for x in ms], dtype=np.float64)
            bp = np.array([float(x["bops_ratio"]) for x in ms], dtype=np.float64)
            ab = np.array([float(x["avg_bits"]) for x in ms], dtype=np.float64)
            dd = 1 if len(ca) > 1 else 0
            print(
                f"  {m:16s} | calib {ca.mean():.4f}±{ca.std(ddof=dd):.4f} | "
                f"test {ta.mean():.4f}±{ta.std(ddof=dd):.4f} | "
                f"bops {bp.mean():.4f}±{bp.std(ddof=dd):.4f} | bits {ab.mean():.2f}"
            )


def main() -> None:
    configure_a100_runtime()
    print(
        f"A100_OPTIMIZED={A100_OPTIMIZED} MIXED_PRECISION={USE_MIXED_PRECISION} "
        f"XLA={USE_XLA} TF32={ALLOW_TF32}"
    )
    paths = make_dirs()
    combos: List[ExpConfig] = [
        ExpConfig("cifar100", "resnet20", [42, 123, 456, 789, 1024], pretrain_epochs=100),
        ExpConfig("cifar100", "mobilenetv2", [42, 123, 456, 789, 1024], pretrain_epochs=90),
        ExpConfig("cifar100", "resnet56", [42, 123, 456, 789, 1024], pretrain_epochs=120),
        ExpConfig("cifar10", "resnet56", [42, 123, 456, 789, 1024], pretrain_epochs=120),
        ExpConfig("cifar10", "efficientnetb0", [42, 123, 456, 789, 1024], pretrain_epochs=30),
    ]

    all_rows: List[Dict[str, Any]] = []
    all_timing: List[Dict[str, Any]] = []
    mob_summary: Dict[str, Any] | None = None

    for cfg in combos:
        print(f"\nRunning {cfg.model_name} on {cfg.dataset} with {len(cfg.seeds)} seeds...")
        rows, timing_rows, summary = run_one_combo(cfg, paths)
        all_rows.extend(rows)
        all_timing.extend(timing_rows)
        if cfg.model_name == "mobilenetv2" and cfg.dataset == "cifar10":
            mob_summary = summary
        if not (cfg.model_name == "mobilenetv2" and cfg.dataset == "cifar10"):
            tf.keras.backend.clear_session()

    write_csv(os.path.join(paths["csv"], "extended_results.csv"), all_rows, CSV_COLUMNS)
    write_csv(
        os.path.join(paths["timing"], "wall_clock_time.csv"),
        all_timing,
        ["method", "dataset", "model", "seed", "sensitivity_prior_s", "abc_search_s", "cma_search_s", "random_search_s", "fine_tune_s"],
    )
    plot_timing_bar(os.path.join(paths["timing"], "timing_comparison.png"), all_timing)

    # task 5/6/7 on MobileNetV2 CIFAR-10
    if mob_summary is not None:
        data = mob_summary["data"]
        model = mob_summary["model_obj"]
        quant_names = mob_summary["quant_names"]
        param_counts = mob_summary["param_counts"]
        layer_specs = core.build_layer_bops_specs(model, quant_names)
        act_stats = core.collect_activation_stats(model, quant_names, data["x_calib"], data["y_calib"])
        sensitivity = mob_summary["sensitivity"]
        budget = int(mob_summary["budget"])
        random_cloud = core.collect_random_search_points(
            model,
            quant_names,
            data["x_calib"],
            data["y_calib"],
            param_counts,
            sensitivity,
            1000,
            layer_specs,
            act_stats,
            True,
        )
        # use seed-mean representative rows
        rs = [r for r in all_rows if r["dataset"] == "cifar10" and r["model"] == "mobilenetv2"]
        pick = lambda method: max([r for r in rs if r["method"] == method], key=lambda x: float(x["calib_acc"]))
        # recompute best configs from stored bit configs
        abc_bits = aggregate_bits(mob_summary["bit_configs"]["ABC-Q (ours)"])
        cma_bits = aggregate_bits(mob_summary["bit_configs"]["CMA-ES"])
        rnd_bits = aggregate_bits(mob_summary["bit_configs"]["Random Search"])
        ow = {i: copy.deepcopy(model.get_layer(quant_names[i]).get_weights()) for i in range(len(quant_names))}
        abc_acc, abc_bop, _, _ = core.evaluate_bit_config(
            model, quant_names, abc_bits, data["x_calib"], data["y_calib"], param_counts, ow, layer_specs, act_stats, True
        )
        cma_acc, cma_bop, _, _ = core.evaluate_bit_config(
            model, quant_names, cma_bits, data["x_calib"], data["y_calib"], param_counts, ow, layer_specs, act_stats, True
        )
        u8, u4, u2 = mob_summary["uniform"]
        _ = pick  # keep helper for debug
        plot_extended_pareto(
            os.path.join(paths["pareto"], "mobilenetv2_pareto_extended.png"),
            random_cloud,
            (u8, u4, u2),
            type("Tmp", (), {"bops_weight_act": abc_bop, "accuracy": abc_acc}),
            type("Tmp", (), {"bops_weight_act": cma_bop, "accuracy": cma_acc}),
        )
        plot_convergence_per_seed(
            os.path.join(paths["convergence"], "mobilenetv2_convergence_per_seed.png"),
            mob_summary["conv_abc"],
            mob_summary["conv_cma"],
        )
        plot_bitwidth_heatmap(
            os.path.join(paths["heatmap"], "bitwidth_heatmap_mobilenetv2.png"),
            abc_bits,
            cma_bits,
            rnd_bits,
        )

    # print paper-style summary for one canonical combo
    if mob_summary is not None:
        rs = [r for r in all_rows if r["dataset"] == "cifar10" and r["model"] == "mobilenetv2"]
        methods = ["Uniform 8-bit", "Uniform 4-bit", "Uniform 2-bit", "Random Search", "CMA-ES", "ABC-Q (no prior)", "ABC-Q (ours)"]
        rows: Dict[str, Tuple[float, float, float, float, float, float]] = {}
        avg_bits_by: Dict[str, float] = {}
        abc_vals = np.array([float(r["calib_acc"]) for r in rs if r["method"] == "ABC-Q (ours)"], dtype=np.float64)
        for m in methods:
            mr = [r for r in rs if r["method"] == m]
            if not mr:
                continue
            ca = np.array([float(x["calib_acc"]) for x in mr], dtype=np.float64)
            bp = np.array([float(x["bops_ratio"]) for x in mr], dtype=np.float64)
            dd = 1 if len(ca) > 1 else 0
            rows[m] = (float(ca.mean()), float(ca.std(ddof=dd)), float(bp.mean()), float(bp.std(ddof=dd)), 0.0, 0.0)
            avg_bits_by[m] = float(np.mean([float(x["avg_bits"]) for x in mr]))
        wilk: Dict[str, Tuple[float, str]] = {}
        for m in methods:
            if m == "ABC-Q (ours)":
                continue
            mr = [r for r in rs if r["method"] == m]
            if not mr:
                continue
            x = np.array([float(v["calib_acc"]) for v in mr], dtype=np.float64)
            p, st = wilcoxon_paired_pvalue(abc_vals, x)
            wilk[m] = (p, st)
        print_paper_table(methods, rows, avg_bits_by, wilk, n_seeds=5, reference_method="ABC-Q (ours)")

    summarize_stdout(all_rows)
    print("\nSaved CSV/figures under results/")


if __name__ == "__main__":
    main()

