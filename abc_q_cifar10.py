from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.special import softmax
from scipy.stats import multivariate_normal
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


QUICK_TEST = True
SEED = 42
BITS = np.array([2, 4, 8], dtype=np.int32)
LAMBDA_PENALTY = 0.3
MU_PENALTY = 0.1
WEIGHT_DECAY = 3e-4
CLASSIFIER_DROPOUT = 0.25
LABEL_SMOOTHING = 0.05

CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)
CIFAR_PAD = 4


def sparse_cce_with_label_smoothing(num_classes: int, smoothing: float):
    """Sparse integer labels with softmax probs; label smoothing without TF sparse-loss support."""

    def loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        one_hot = tf.one_hot(y_true, depth=num_classes, dtype=tf.float32)
        smooth = one_hot * (1.0 - smoothing) + smoothing / float(num_classes)
        return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(smooth, y_pred))

    return loss_fn


@dataclass
class SearchResult:
    """Container for search result statistics."""

    bit_config: np.ndarray
    fitness: float
    accuracy: float
    bops_ratio: float
    mem_ratio: float


def set_seed(seed: int) -> None:
    """Set deterministic seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_experiment_config(quick_test: bool) -> Dict[str, int]:
    """Return experiment hyperparameters based on QUICK_TEST flag."""
    if quick_test:
        return {
            "train_samples": 4000,
            "calib_samples": 500,
            "test_samples": 1000,
            "pretrain_epochs": 20,
            "finetune_epochs": 3,
            "abc_cycles": 3,
            "num_bees": 5,
            "scout_limit": 2,
            "batch_size": 64,
            "pareto_random_samples": 250,
        }
    return {
        "train_samples": 5000,
        "calib_samples": 1000,
        "test_samples": 2000,
        "pretrain_epochs": 30,
        "finetune_epochs": 3,
        "abc_cycles": 12,
        "num_bees": 10,
        "scout_limit": 5,
        "batch_size": 64,
        "pareto_random_samples": 500,
    }


def load_cifar10_splits(config: Dict[str, int]) -> Dict[str, np.ndarray]:
    """Load CIFAR-10 and create train/calibration/test subsets."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = y_train.squeeze().astype(np.int32)
    y_test = y_test.squeeze().astype(np.int32)

    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)

    rng = np.random.RandomState(SEED)
    tr_perm = rng.permutation(len(x_train))
    x_train, y_train = x_train[tr_perm], y_train[tr_perm]
    te_perm = rng.permutation(len(x_test))
    x_test, y_test = x_test[te_perm], y_test[te_perm]

    train_n = config["train_samples"]
    calib_n = config["calib_samples"]
    test_n = config["test_samples"]

    x_train_small = x_train[:train_n]
    y_train_small = y_train[:train_n]
    x_calib = x_train[train_n : train_n + calib_n]
    y_calib = y_train[train_n : train_n + calib_n]
    x_test_small = x_test[:test_n]
    y_test_small = y_test[:test_n]

    return {
        "x_train": x_train_small,
        "y_train": y_train_small,
        "x_calib": x_calib,
        "y_calib": y_calib,
        "x_test": x_test_small,
        "y_test": y_test_small,
    }


def _normalize_cifar(img: tf.Tensor) -> tf.Tensor:
    """Per-channel normalize CIFAR-10 images to zero mean, unit-ish variance."""
    x = tf.cast(img, tf.float32) / 255.0
    mean = tf.constant(CIFAR_MEAN, dtype=tf.float32)
    std = tf.constant(CIFAR_STD, dtype=tf.float32)
    return (x - mean) / std


def make_dataset(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    augment: bool = False,
) -> tf.data.Dataset:
    """tf.data pipeline: optional train-time crop+flip, then CIFAR normalization."""
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(len(x), seed=SEED, reshuffle_each_iteration=True)

    def _aug(img: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        img = tf.pad(img, [[CIFAR_PAD, CIFAR_PAD], [CIFAR_PAD, CIFAR_PAD], [0, 0]], mode="SYMMETRIC")
        img = tf.image.random_crop(img, [32, 32, 3])
        img = tf.image.random_flip_left_right(img)
        return img, label

    def _prep(img: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return _normalize_cifar(img), label

    if augment:
        ds = ds.map(_aug, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(_prep, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _resnet_block(
    x: tf.Tensor,
    filters: int,
    reg: tf.keras.regularizers.Regularizer,
    stride: int = 1,
) -> tf.Tensor:
    """One residual unit (two convs) for CIFAR-sized ResNet."""
    shortcut = x
    out = Conv2D(
        filters, 3, strides=stride, padding="same", use_bias=False, kernel_regularizer=reg
    )(x)
    out = BatchNormalization()(out)
    out = Activation("relu")(out)
    out = Conv2D(filters, 3, strides=1, padding="same", use_bias=False, kernel_regularizer=reg)(
        out
    )
    out = BatchNormalization()(out)
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv2D(
            filters, 1, strides=stride, padding="same", use_bias=False, kernel_regularizer=reg
        )(shortcut)
        shortcut = BatchNormalization()(shortcut)
    out = Add()([out, shortcut])
    out = Activation("relu")(out)
    return out


def build_model(
    num_classes: int = 10,
    learning_rate: Union[float, tf.keras.optimizers.schedules.LearningRateSchedule] = 1e-3,
) -> Model:
    """ResNet-20 style network for 32x32 CIFAR-10 (train from scratch)."""
    reg = l2(WEIGHT_DECAY)
    inputs = Input(shape=(32, 32, 3))
    x = Conv2D(16, 3, padding="same", use_bias=False, kernel_regularizer=reg)(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    for _ in range(3):
        x = _resnet_block(x, 16, reg, stride=1)
    x = _resnet_block(x, 32, reg, stride=2)
    for _ in range(2):
        x = _resnet_block(x, 32, reg, stride=1)
    x = _resnet_block(x, 64, reg, stride=2)
    for _ in range(2):
        x = _resnet_block(x, 64, reg, stride=1)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(CLASSIFIER_DROPOUT)(x)
    outputs = Dense(
        num_classes,
        activation="softmax",
        kernel_regularizer=l2(WEIGHT_DECAY),
    )(x)
    model = Model(inputs, outputs)
    loss = sparse_cce_with_label_smoothing(num_classes, LABEL_SMOOTHING)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=["accuracy"],
    )
    return model


def get_quantizable_layers(model: Model) -> List[int]:
    """Return indices of quantizable Conv2D and Dense layers."""
    indices = []
    for idx, layer in enumerate(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
            if layer.get_weights():
                indices.append(idx)
    return indices


def layer_param_counts(model: Model, quantizable_layers: Sequence[int]) -> np.ndarray:
    """Compute total parameter count per quantizable layer."""
    counts = []
    for idx in quantizable_layers:
        layer = model.layers[idx]
        layer_params = sum(np.prod(w.shape) for w in layer.get_weights())
        counts.append(float(layer_params))
    return np.array(counts, dtype=np.float64)


def quantize_array_minmax(weights: np.ndarray, bits: int) -> np.ndarray:
    """Uniform min-max quantization for a single weight array."""
    w_min = weights.min()
    w_max = weights.max()
    if np.isclose(w_min, w_max):
        return weights.copy()
    levels = (2 ** bits) - 1
    scale = (w_max - w_min) / levels
    quantized = np.round((weights - w_min) / scale) * scale + w_min
    return quantized.astype(weights.dtype, copy=False)


def apply_bit_config(
    model: Model,
    quantizable_layers: Sequence[int],
    bit_config: np.ndarray,
    original_weights: Dict[int, List[np.ndarray]],
) -> None:
    """Apply quantized weights to model according to bit configuration."""
    for layer_idx, bits in zip(quantizable_layers, bit_config):
        layer = model.layers[layer_idx]
        src_weights = original_weights[layer_idx]
        q_weights = [quantize_array_minmax(w, int(bits)) for w in src_weights]
        layer.set_weights(q_weights)


def restore_original_weights(
    model: Model,
    quantizable_layers: Sequence[int],
    original_weights: Dict[int, List[np.ndarray]],
) -> None:
    """Restore non-quantized original weights for quantizable layers."""
    for layer_idx in quantizable_layers:
        model.layers[layer_idx].set_weights(original_weights[layer_idx])


def evaluate_bit_config(
    model: Model,
    quantizable_layers: Sequence[int],
    bit_config: np.ndarray,
    x_calib: np.ndarray,
    y_calib: np.ndarray,
    param_counts: np.ndarray,
    original_weights: Dict[int, List[np.ndarray]],
) -> Tuple[float, float, float]:
    """Evaluate calibration accuracy and hardware ratios for a bit configuration."""
    apply_bit_config(model, quantizable_layers, bit_config, original_weights)
    calib_ds = make_dataset(x_calib, y_calib, batch_size=128, shuffle=False)
    loss, acc = model.evaluate(calib_ds, verbose=0)
    _ = loss
    restore_original_weights(model, quantizable_layers, original_weights)

    numerator = float(np.sum(bit_config.astype(np.float64) * param_counts))
    denominator = float(np.sum(8.0 * param_counts))
    bops_ratio = numerator / denominator
    mem_ratio = numerator / denominator
    return float(acc), float(bops_ratio), float(mem_ratio)


def compute_fitness(acc: float, bops_ratio: float, mem_ratio: float) -> float:
    """Compute hardware-aware fitness score."""
    return acc - LAMBDA_PENALTY * bops_ratio - MU_PENALTY * mem_ratio


def compute_layer_sensitivity(
    model: Model,
    quantizable_layers: Sequence[int],
    x_calib: np.ndarray,
    y_calib: np.ndarray,
) -> np.ndarray:
    """Compute normalized per-layer gradient magnitude sensitivity prior."""
    xc = x_calib[:128].astype(np.float32)
    x_batch = (xc / 255.0 - CIFAR_MEAN) / CIFAR_STD
    y_batch = y_calib[:128].astype(np.int32)

    with tf.GradientTape() as tape:
        preds = model(x_batch, training=False)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_batch, preds)
        loss = tf.reduce_mean(loss)

    train_vars = model.trainable_variables
    grads = tape.gradient(loss, train_vars)
    grad_map = {id(v): g for v, g in zip(train_vars, grads)}

    sensitivities = []
    for layer_idx in quantizable_layers:
        layer = model.layers[layer_idx]
        mags = []
        for var in layer.trainable_weights:
            grad = grad_map.get(id(var))
            if grad is not None:
                mags.append(float(tf.reduce_mean(tf.abs(grad)).numpy()))
        sensitivities.append(float(np.mean(mags)) if mags else 0.0)

    sens = np.array(sensitivities, dtype=np.float64)
    if np.allclose(sens.sum(), 0.0):
        sens = np.ones_like(sens) / len(sens)
    else:
        sens = sens / sens.sum()
    return sens


def initialize_food_sources(
    num_bees: int,
    num_layers: int,
    sensitivity: np.ndarray,
) -> np.ndarray:
    """Initialize random bit configurations with mild sensitivity protection."""
    configs = np.random.choice(BITS, size=(num_bees, num_layers), replace=True)
    threshold = np.quantile(sensitivity, 0.75)
    sensitive_mask = sensitivity >= threshold
    for i in range(num_bees):
        for d in np.where(sensitive_mask)[0]:
            if configs[i, d] < 4:
                configs[i, d] = np.random.choice([4, 8])
    return configs.astype(np.int32)


def choose_mutation_dimension(sensitivity: np.ndarray) -> int:
    """Sample mutation dimension biased against highly sensitive layers."""
    inv = 1.0 - sensitivity
    inv = np.clip(inv, 1e-8, None)
    inv = inv / inv.sum()
    return int(np.random.choice(len(sensitivity), p=inv))


def mutate_config(config: np.ndarray, sensitivity: np.ndarray) -> np.ndarray:
    """Mutate one dimension of a bit configuration."""
    new_config = config.copy()
    d = choose_mutation_dimension(sensitivity)
    options = [b for b in BITS if b != config[d]]
    new_config[d] = np.random.choice(options)
    return new_config


def format_bits(config: np.ndarray) -> str:
    """Format bit configuration for readable logging."""
    return "[" + ", ".join(str(int(v)) for v in config) + "]"


def run_abc_q(
    model: Model,
    quantizable_layers: Sequence[int],
    x_calib: np.ndarray,
    y_calib: np.ndarray,
    param_counts: np.ndarray,
    sensitivity: np.ndarray,
    num_bees: int,
    cycles: int,
    scout_limit: int,
) -> Tuple[SearchResult, int]:
    """Run ABC-Q optimization loop and return best result and eval count."""
    original_weights = {idx: copy.deepcopy(model.layers[idx].get_weights()) for idx in quantizable_layers}
    num_layers = len(quantizable_layers)
    foods = initialize_food_sources(num_bees, num_layers, sensitivity)
    trials = np.zeros(num_bees, dtype=np.int32)

    scores = np.zeros(num_bees, dtype=np.float64)
    accs = np.zeros(num_bees, dtype=np.float64)
    bops = np.zeros(num_bees, dtype=np.float64)
    mems = np.zeros(num_bees, dtype=np.float64)
    eval_count = 0

    for i in range(num_bees):
        acc, bop, mem = evaluate_bit_config(
            model, quantizable_layers, foods[i], x_calib, y_calib, param_counts, original_weights
        )
        scores[i] = compute_fitness(acc, bop, mem)
        accs[i], bops[i], mems[i] = acc, bop, mem
        eval_count += 1

    best_idx = int(np.argmax(scores))
    best = SearchResult(
        bit_config=foods[best_idx].copy(),
        fitness=float(scores[best_idx]),
        accuracy=float(accs[best_idx]),
        bops_ratio=float(bops[best_idx]),
        mem_ratio=float(mems[best_idx]),
    )

    for cycle in range(1, cycles + 1):
        scouts_triggered = 0

        for i in range(num_bees):
            candidate = mutate_config(foods[i], sensitivity)
            acc, bop, mem = evaluate_bit_config(
                model, quantizable_layers, candidate, x_calib, y_calib, param_counts, original_weights
            )
            fit = compute_fitness(acc, bop, mem)
            eval_count += 1
            if fit > scores[i]:
                foods[i] = candidate
                scores[i] = fit
                accs[i], bops[i], mems[i] = acc, bop, mem
                trials[i] = 0
            else:
                trials[i] += 1

        probs = softmax(scores)
        for _ in range(num_bees):
            i = int(np.random.choice(num_bees, p=probs))
            candidate = mutate_config(foods[i], sensitivity)
            acc, bop, mem = evaluate_bit_config(
                model, quantizable_layers, candidate, x_calib, y_calib, param_counts, original_weights
            )
            fit = compute_fitness(acc, bop, mem)
            eval_count += 1
            if fit > scores[i]:
                foods[i] = candidate
                scores[i] = fit
                accs[i], bops[i], mems[i] = acc, bop, mem
                trials[i] = 0
            else:
                trials[i] += 1

        threshold = np.quantile(sensitivity, 0.75)
        sensitive_mask = sensitivity >= threshold
        for i in range(num_bees):
            if trials[i] >= scout_limit:
                print(f"[Scout] Resetting bee {i} after {int(trials[i])} trials.")
                reset = np.random.choice(BITS, size=num_layers, replace=True).astype(np.int32)
                for d in np.where(sensitive_mask)[0]:
                    if reset[d] < 4:
                        reset[d] = np.random.choice([4, 8])
                acc, bop, mem = evaluate_bit_config(
                    model, quantizable_layers, reset, x_calib, y_calib, param_counts, original_weights
                )
                fit = compute_fitness(acc, bop, mem)
                foods[i] = reset
                scores[i] = fit
                accs[i], bops[i], mems[i] = acc, bop, mem
                trials[i] = 0
                eval_count += 1
                scouts_triggered += 1

        best_idx = int(np.argmax(scores))
        if scores[best_idx] > best.fitness:
            best = SearchResult(
                bit_config=foods[best_idx].copy(),
                fitness=float(scores[best_idx]),
                accuracy=float(accs[best_idx]),
                bops_ratio=float(bops[best_idx]),
                mem_ratio=float(mems[best_idx]),
            )

        print(
            f"[ABC Cycle {cycle}] "
            f"Best bits={format_bits(best.bit_config)} | "
            f"Fitness={best.fitness:.4f} | "
            f"Acc={best.accuracy:.4f} | "
            f"BOPs={best.bops_ratio:.4f} | "
            f"Mem={best.mem_ratio:.4f} | "
            f"Scouts={scouts_triggered}"
        )

    restore_original_weights(model, quantizable_layers, original_weights)
    return best, eval_count


def evaluate_baseline_config(
    model: Model,
    quantizable_layers: Sequence[int],
    bit_value: int,
    x_calib: np.ndarray,
    y_calib: np.ndarray,
    param_counts: np.ndarray,
) -> SearchResult:
    """Evaluate a uniform-bit baseline configuration."""
    bit_config = np.full(len(quantizable_layers), bit_value, dtype=np.int32)
    original_weights = {idx: copy.deepcopy(model.layers[idx].get_weights()) for idx in quantizable_layers}
    acc, bop, mem = evaluate_bit_config(
        model, quantizable_layers, bit_config, x_calib, y_calib, param_counts, original_weights
    )
    return SearchResult(
        bit_config=bit_config,
        fitness=compute_fitness(acc, bop, mem),
        accuracy=acc,
        bops_ratio=bop,
        mem_ratio=mem,
    )


def run_random_search(
    model: Model,
    quantizable_layers: Sequence[int],
    x_calib: np.ndarray,
    y_calib: np.ndarray,
    param_counts: np.ndarray,
    sensitivity: np.ndarray,
    budget: int,
) -> SearchResult:
    """Run random mixed-precision search with same evaluation budget as ABC."""
    original_weights = {idx: copy.deepcopy(model.layers[idx].get_weights()) for idx in quantizable_layers}
    num_layers = len(quantizable_layers)
    best: SearchResult | None = None
    threshold = np.quantile(sensitivity, 0.75)
    sensitive_mask = sensitivity >= threshold

    for _ in range(budget):
        config = np.random.choice(BITS, size=num_layers, replace=True).astype(np.int32)
        for d in np.where(sensitive_mask)[0]:
            if config[d] < 4:
                config[d] = np.random.choice([4, 8])

        acc, bop, mem = evaluate_bit_config(
            model, quantizable_layers, config, x_calib, y_calib, param_counts, original_weights
        )
        candidate = SearchResult(
            bit_config=config,
            fitness=compute_fitness(acc, bop, mem),
            accuracy=acc,
            bops_ratio=bop,
            mem_ratio=mem,
        )
        if best is None or candidate.fitness > best.fitness:
            best = candidate

    restore_original_weights(model, quantizable_layers, original_weights)
    assert best is not None
    return best


def collect_random_search_points(
    model: Model,
    quantizable_layers: Sequence[int],
    x_calib: np.ndarray,
    y_calib: np.ndarray,
    param_counts: np.ndarray,
    sensitivity: np.ndarray,
    n_samples: int,
) -> List[Tuple[float, float]]:
    """Sample random bit configs (same bias as run_random_search); return (bops_ratio, accuracy) each."""
    original_weights = {idx: copy.deepcopy(model.layers[idx].get_weights()) for idx in quantizable_layers}
    num_layers = len(quantizable_layers)
    threshold = np.quantile(sensitivity, 0.75)
    sensitive_mask = sensitivity >= threshold
    points: List[Tuple[float, float]] = []
    for _ in range(n_samples):
        config = np.random.choice(BITS, size=num_layers, replace=True).astype(np.int32)
        for d in np.where(sensitive_mask)[0]:
            if config[d] < 4:
                config[d] = np.random.choice([4, 8])
        acc, bop, mem = evaluate_bit_config(
            model, quantizable_layers, config, x_calib, y_calib, param_counts, original_weights
        )
        _ = mem
        points.append((float(bop), float(acc)))
    restore_original_weights(model, quantizable_layers, original_weights)
    return points


def plot_calib_accuracy_vs_bops(
    path: str,
    uniform_triples: List[Tuple[str, float, float]],
    random_bops_acc: List[Tuple[float, float]],
    abc_bops: float,
    abc_acc: float,
    cma_bops: float | None,
    cma_acc: float | None,
) -> None:
    """Save scatter: x=BOPs ratio, y=calibration accuracy (Pareto-style comparison)."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    if random_bops_acc:
        rb, ra = zip(*random_bops_acc)
        ax.scatter(rb, ra, s=14, alpha=0.28, c="0.45", label=f"Random ({len(random_bops_acc)} pts)", edgecolors="none")
    colors_uni = ["#1f77b4", "#9467bd", "#8c564b"]
    for i, (label, bop, acc) in enumerate(uniform_triples):
        ax.scatter(
            [bop], [acc], s=85, marker="s", c=colors_uni[i % len(colors_uni)], label=label, zorder=5, edgecolors="0.2", linewidths=0.6
        )
    ax.scatter(
        [abc_bops],
        [abc_acc],
        s=200,
        marker="*",
        c="#ff7f0e",
        edgecolors="0.1",
        linewidths=0.9,
        label="ABC-Q (ours)",
        zorder=6,
    )
    if cma_bops is not None and cma_acc is not None:
        ax.scatter(
            [cma_bops], [cma_acc], s=70, marker="D", c="#2ca02c", label="CMA-ES", zorder=5, edgecolors="0.2", linewidths=0.6
        )
    ax.set_xlabel("BOPs ratio (1.0 = uniform 8-bit)")
    ax.set_ylabel("Calibration accuracy")
    ax.set_title("Calibration accuracy vs. compute cost proxy")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.35)
    ax.set_xlim(0.0, None)
    ax.set_ylim(0.0, 1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved plot: {path}")


def bits_from_continuous(z: np.ndarray) -> np.ndarray:
    """Project continuous vector to discrete bit set {2,4,8}."""
    idx = np.clip(np.round(z).astype(np.int32), 0, 2)
    return BITS[idx]


def run_cmaes_baseline(
    model: Model,
    quantizable_layers: Sequence[int],
    x_calib: np.ndarray,
    y_calib: np.ndarray,
    param_counts: np.ndarray,
    budget: int,
) -> SearchResult:
    """Run a lightweight CMA-ES style baseline over relaxed bit indices."""
    original_weights = {idx: copy.deepcopy(model.layers[idx].get_weights()) for idx in quantizable_layers}
    dim = len(quantizable_layers)
    lam = min(12, max(6, dim // 3))
    mu = lam // 2
    sigma = 0.8

    mean = np.random.uniform(0.0, 2.0, size=dim)
    cov = np.eye(dim) * (sigma**2)
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = weights / np.sum(weights)

    best: SearchResult | None = None
    evals = 0

    while evals < budget:
        remaining = min(lam, budget - evals)
        samples = multivariate_normal.rvs(mean=mean, cov=cov, size=remaining)
        if remaining == 1:
            samples = samples[None, :]

        scored = []
        for s in samples:
            s = np.clip(s, 0.0, 2.0)
            config = bits_from_continuous(s)
            acc, bop, mem = evaluate_bit_config(
                model, quantizable_layers, config, x_calib, y_calib, param_counts, original_weights
            )
            fit = compute_fitness(acc, bop, mem)
            scored.append((fit, s, config, acc, bop, mem))
            evals += 1

            candidate = SearchResult(config, fit, acc, bop, mem)
            if best is None or candidate.fitness > best.fitness:
                best = candidate

        scored.sort(key=lambda x: x[0], reverse=True)
        n_elite = min(mu, len(scored))
        elites = scored[:n_elite]
        elite_samples = np.array([e[1] for e in elites])
        w = weights[:n_elite]
        w = w / np.sum(w)
        mean = np.sum(elite_samples * w[:, None], axis=0)

        centered = elite_samples - mean
        cov = np.zeros_like(cov)
        for i in range(n_elite):
            c = centered[i][:, None]
            cov += w[i] * (c @ c.T)
        cov += np.eye(dim) * 1e-5

    restore_original_weights(model, quantizable_layers, original_weights)
    assert best is not None
    return best


def apply_best_config_permanently(
    model: Model,
    quantizable_layers: Sequence[int],
    bit_config: np.ndarray,
) -> None:
    """Permanently quantize model weights using best bit configuration."""
    for layer_idx, bits in zip(quantizable_layers, bit_config):
        layer = model.layers[layer_idx]
        weights = layer.get_weights()
        q_weights = [quantize_array_minmax(w, int(bits)) for w in weights]
        layer.set_weights(q_weights)


def print_results_table(results: Dict[str, SearchResult], param_counts: np.ndarray) -> None:
    """Print final comparison table for all methods."""
    print("\n| Method          | Accuracy | BOPs Ratio | Bits (avg) |")
    print("|-----------------|----------|------------|------------|")
    for name, res in results.items():
        avg_bits = float(np.sum(res.bit_config * param_counts) / np.sum(param_counts))
        print(f"| {name:<15} | {res.accuracy:.4f}   | {res.bops_ratio:.4f}     | {avg_bits:.2f}       |")


def main() -> None:
    """Run the full ABC-Q experiment pipeline on CIFAR-10."""
    set_seed(SEED)
    config = get_experiment_config(QUICK_TEST)
    print(f"Running with QUICK_TEST={QUICK_TEST}: {config}")

    data = load_cifar10_splits(config)
    train_ds = make_dataset(
        data["x_train"], data["y_train"], config["batch_size"], shuffle=True, augment=True
    )
    test_ds = make_dataset(data["x_test"], data["y_test"], config["batch_size"], shuffle=False)

    steps_per_epoch = max(1, len(data["x_train"]) // config["batch_size"])
    total_steps = steps_per_epoch * config["pretrain_epochs"]
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-3,
        decay_steps=total_steps,
        alpha=0.08,
    )
    model = build_model(learning_rate=lr_schedule)
    model.fit(train_ds, epochs=config["pretrain_epochs"], verbose=2)

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    _ = test_loss
    print(f"Pretrained test accuracy: {test_acc:.4f}")

    quantizable_layers = get_quantizable_layers(model)
    print(f"Quantizable layers count: {len(quantizable_layers)}")
    param_counts = layer_param_counts(model, quantizable_layers)

    sensitivity = compute_layer_sensitivity(model, quantizable_layers, data["x_calib"], data["y_calib"])
    print("Layer sensitivity prior computed.")

    uniform8 = evaluate_baseline_config(
        model, quantizable_layers, 8, data["x_calib"], data["y_calib"], param_counts
    )
    uniform4 = evaluate_baseline_config(
        model, quantizable_layers, 4, data["x_calib"], data["y_calib"], param_counts
    )
    uniform2 = evaluate_baseline_config(
        model, quantizable_layers, 2, data["x_calib"], data["y_calib"], param_counts
    )

    abc_best, abc_budget = run_abc_q(
        model=model,
        quantizable_layers=quantizable_layers,
        x_calib=data["x_calib"],
        y_calib=data["y_calib"],
        param_counts=param_counts,
        sensitivity=sensitivity,
        num_bees=config["num_bees"],
        cycles=config["abc_cycles"],
        scout_limit=config["scout_limit"],
    )
    print(f"\nABC best config: {format_bits(abc_best.bit_config)}")
    print(
        f"ABC best fitness={abc_best.fitness:.4f}, "
        f"acc={abc_best.accuracy:.4f}, bops={abc_best.bops_ratio:.4f}, mem={abc_best.mem_ratio:.4f}"
    )

    random_best = run_random_search(
        model,
        quantizable_layers,
        data["x_calib"],
        data["y_calib"],
        param_counts,
        sensitivity,
        budget=abc_budget,
    )
    cma_best = run_cmaes_baseline(
        model,
        quantizable_layers,
        data["x_calib"],
        data["y_calib"],
        param_counts,
        budget=abc_budget,
    )

    n_scatter = int(config.get("pareto_random_samples", max(200, min(500, abc_budget))))
    pareto_pts = collect_random_search_points(
        model,
        quantizable_layers,
        data["x_calib"],
        data["y_calib"],
        param_counts,
        sensitivity,
        n_samples=n_scatter,
    )
    plot_calib_accuracy_vs_bops(
        path="abc_q_pareto_calib.png",
        uniform_triples=[
            ("Uniform 8-bit", uniform8.bops_ratio, uniform8.accuracy),
            ("Uniform 4-bit", uniform4.bops_ratio, uniform4.accuracy),
            ("Uniform 2-bit", uniform2.bops_ratio, uniform2.accuracy),
        ],
        random_bops_acc=pareto_pts,
        abc_bops=abc_best.bops_ratio,
        abc_acc=abc_best.accuracy,
        cma_bops=cma_best.bops_ratio,
        cma_acc=cma_best.accuracy,
    )

    results = {
        "Uniform 8-bit": uniform8,
        "Uniform 4-bit": uniform4,
        "Uniform 2-bit": uniform2,
        "Random Search": random_best,
        "ABC-Q (ours)": abc_best,
        "CMA-ES": cma_best,
    }
    print_results_table(results, param_counts)

    final_model = build_model()
    final_model.set_weights(model.get_weights())
    fine_loss = sparse_cce_with_label_smoothing(10, LABEL_SMOOTHING)
    final_model.compile(
        optimizer=Adam(learning_rate=3e-4),
        loss=fine_loss,
        metrics=["accuracy"],
    )
    apply_best_config_permanently(final_model, quantizable_layers, abc_best.bit_config)
    final_model.fit(train_ds, epochs=config["finetune_epochs"], verbose=2)
    final_loss, final_acc = final_model.evaluate(test_ds, verbose=0)
    _ = final_loss
    print(f"\nFinal test accuracy after quantization + fine-tune: {final_acc:.4f}")


if __name__ == "__main__":
    main()
