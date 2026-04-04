"""
ABC-Q on CIFAR-10: Artificial Bee Colony over per-layer bit-widths with calibration fitness.

RESEARCH_NOTES — design choices
-----------------------------
- **Data**: CIFAR-10 train pool is 50k. ``FULL_DATASET=True`` uses 49k disjoint train + 1k disjoint
  calibration + 10k test so calibration statistics are not taken from training minibatches.
- **Pretrain once**: One float checkpoint is trained; all seeds reload it so search variance is
  from stochastic search, not from retraining (faster, paired Wilcoxon is well-defined on calib acc).
- **Calibration fitness**: ABC / baselines optimize calibration accuracy under fake-quant weights
  (and optional activation PTQ), not test accuracy, to avoid test leakage during search.
- **QUANTIZE_ACTIVATIONS**: If True, layer outputs are min–max fake-quantized after conv/dense using
  calibration min/max collected once; bits match weight bits per layer. Classifier uses Dense logits
  + separate softmax so logits are quantized, not probabilities.
- **REAL_BOPS**: If True, BOPs are MAC-style sums of b_w*b_a*ops per layer vs an 8+8 reference.
  If False, a simple parameter-weighted bit ratio is used (legacy proxy) for fitness and reporting.
- **Sensitivity prior**: Normalized gradient magnitudes on calib batch; biases init/scouts/mutation
  toward higher bits on sensitive layers (ablation optional via config).
- **Fitness** (unchanged): f = acc - λ·bops - μ·mem with mem = param-bit budget vs 8-bit params.
- **Baselines**: Uniform 2/4/8, random search (same eval budget as first ABC run), Hansen CMA-ES
  on continuous codes projected to {2,4,8}.
- **Figures / stats**: See ``abc_q_plots.py`` and ``abc_q_stats.py``; images go under ``results/``.
- **Defaults**: ``FULL_DATASET=False``, ``N_SEEDS=1`` for a fast smoke test; set ``FULL_DATASET=True``,
  ``N_SEEDS=3`` for paper runs (~hours). Override output dir with env ``ABC_Q_RESULTS_DIR``.
"""

from __future__ import annotations

import copy
import os
import random
import types
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cma
import numpy as np
import tensorflow as tf
from scipy.special import softmax
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

from abc_q_plots import DEFAULT_RESULTS_DIR, save_all_paper_figures
from abc_q_stats import paired_mean_std, print_paper_table, wilcoxon_paired_pvalue

# --- Publication toggles ---
FULL_DATASET = True
N_SEEDS = 3
SEEDS = [42, 123, 456]
QUANTIZE_ACTIVATIONS = True
REAL_BOPS = True

DATA_SPLIT_SEED = 0
SEED = 42
BITS = np.array([2, 4, 8], dtype=np.int32)
LAMBDA_PENALTY = 0.05
MU_PENALTY = 0.02
WEIGHT_DECAY = 2.5e-4
CLASSIFIER_DROPOUT = 0.2
LABEL_SMOOTHING = 0.04

CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)
CIFAR_PAD = 4


def sparse_cce_with_label_smoothing(num_classes: int, smoothing: float):
    """Label-smoothed sparse categorical cross-entropy (Keras-compatible)."""

    def loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        one_hot = tf.one_hot(y_true, depth=num_classes, dtype=tf.float32)
        smooth = one_hot * (1.0 - smoothing) + smoothing / float(num_classes)
        return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(smooth, y_pred))

    return loss_fn


@dataclass
class SearchResult:
    """One evaluated bit configuration: fitness, accuracies, and BOPs metrics."""

    bit_config: np.ndarray
    fitness: float
    accuracy: float
    bops_ratio: float
    mem_ratio: float
    bops_weight_act: float = 1.0
    bops_weight_only: float = 1.0


def active_seeds() -> List[int]:
    """First ``N_SEEDS`` entries of ``SEEDS``."""
    return [int(s) for s in SEEDS[: max(1, int(N_SEEDS))]]


def set_seed(seed: int) -> None:
    """Fix Python, NumPy, and TensorFlow RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_experiment_config() -> Dict[str, Any]:
    """Training and search hyperparameters from ``FULL_DATASET``."""
    if FULL_DATASET:
        return {
            "train_samples": 49000,
            "calib_samples": 1000,
            "test_samples": 10000,
            "pretrain_epochs": 100,
            "finetune_epochs": 8,
            "abc_cycles": 20,
            "num_bees": 16,
            "scout_limit": 8,
            "batch_size": 128,
            "pareto_random_samples": 500,
            "use_sensitivity_prior": True,
            "run_sensitivity_ablation": True,
        }
    return {
        "train_samples": 5000,
        "calib_samples": 1000,
        "test_samples": 2000,
        "pretrain_epochs": 45,
        "finetune_epochs": 8,
        "abc_cycles": 20,
        "num_bees": 16,
        "scout_limit": 8,
        "batch_size": 64,
        "pareto_random_samples": 500,
        "use_sensitivity_prior": True,
        "run_sensitivity_ablation": True,
    }


def load_cifar10_splits(config: Dict[str, Any], data_split_seed: int = DATA_SPLIT_SEED) -> Dict[str, np.ndarray]:
    """Shuffle CIFAR-10 once, then slice train / calib / test per config."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = y_train.squeeze().astype(np.int32)
    y_test = y_test.squeeze().astype(np.int32)
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)

    rng = np.random.RandomState(int(data_split_seed))
    tr_perm = rng.permutation(len(x_train))
    x_train, y_train = x_train[tr_perm], y_train[tr_perm]
    te_perm = rng.permutation(len(x_test))
    x_test, y_test = x_test[te_perm], y_test[te_perm]

    train_n = int(config["train_samples"])
    calib_n = int(config["calib_samples"])
    test_n = int(config["test_samples"])
    if train_n + calib_n > len(x_train):
        raise ValueError("train_samples + calib_samples exceeds 50k train pool.")

    return {
        "x_train": x_train[:train_n],
        "y_train": y_train[:train_n],
        "x_calib": x_train[train_n : train_n + calib_n],
        "y_calib": y_train[train_n : train_n + calib_n],
        "x_test": x_test[:test_n],
        "y_test": y_test[:test_n],
    }


def _normalize_cifar(img: tf.Tensor) -> tf.Tensor:
    """Normalize uint8 CIFAR images to zero-mean per-channel."""
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
    shuffle_seed: Optional[int] = None,
) -> tf.data.Dataset:
    """Batched tf.data pipeline with optional crop/flip augmentation."""
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        sd = int(shuffle_seed if shuffle_seed is not None else SEED)
        ds = ds.shuffle(len(x), seed=sd, reshuffle_each_iteration=True)

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
    """Pre-activation residual block (two 3x3 convs)."""
    shortcut = x
    out = Conv2D(filters, 3, strides=stride, padding="same", use_bias=False, kernel_regularizer=reg)(x)
    out = BatchNormalization()(out)
    out = Activation("relu")(out)
    out = Conv2D(filters, 3, strides=1, padding="same", use_bias=False, kernel_regularizer=reg)(out)
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
    """ResNet-20–style CIFAR backbone (3+2+2 blocks at 16/32/64 filters) + GAP + dense logits + softmax."""
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
    logits = Dense(num_classes, activation=None, kernel_regularizer=l2(WEIGHT_DECAY))(x)
    outputs = Activation("softmax")(logits)
    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=sparse_cce_with_label_smoothing(num_classes, LABEL_SMOOTHING),
        metrics=["accuracy"],
    )
    return model


def get_quantizable_layers(model: Model) -> List[int]:
    """Layer indices with Conv2D or Dense weights (quantization targets)."""
    indices: List[int] = []
    for idx, layer in enumerate(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)) and layer.get_weights():
            indices.append(idx)
    return indices


def layer_param_counts(model: Model, quantizable_layers: Sequence[int]) -> np.ndarray:
    """Parameter count per quantizable layer (for mem_ratio)."""
    counts: List[float] = []
    for idx in quantizable_layers:
        layer = model.layers[idx]
        counts.append(float(sum(np.prod(w.shape) for w in layer.get_weights())))
    return np.array(counts, dtype=np.float64)


def quantize_array_minmax(weights: np.ndarray, bits: int) -> np.ndarray:
    """Symmetric uniform quant grid in [w_min, w_max]."""
    w_min = weights.min()
    w_max = weights.max()
    if np.isclose(w_min, w_max):
        return weights.copy()
    levels = (2**bits) - 1
    scale = (w_max - w_min) / levels
    quantized = np.round((weights - w_min) / scale) * scale + w_min
    return quantized.astype(weights.dtype, copy=False)


def apply_bit_config(
    model: Model,
    quantizable_layers: Sequence[int],
    bit_config: np.ndarray,
    original_weights: Dict[int, List[np.ndarray]],
) -> None:
    """Write min-max quantized weights from ``original_weights`` into the live model."""
    for layer_idx, bits in zip(quantizable_layers, bit_config):
        layer = model.layers[layer_idx]
        q_weights = [quantize_array_minmax(w, int(bits)) for w in original_weights[layer_idx]]
        layer.set_weights(q_weights)


def restore_original_weights(
    model: Model,
    quantizable_layers: Sequence[int],
    original_weights: Dict[int, List[np.ndarray]],
) -> None:
    """Restore float weights for quantizable layers."""
    for layer_idx in quantizable_layers:
        model.layers[layer_idx].set_weights(original_weights[layer_idx])


def _spatial_hw_from_layer_output(layer: tf.keras.layers.Layer) -> Tuple[int, int]:
    """Read (H, W) from the layer output tensor (Keras 2/3; avoids missing ``output_shape``)."""
    out = getattr(layer, "output", None)
    if out is None:
        raise RuntimeError(f"Layer {getattr(layer, 'name', '?')} has no output tensor.")
    sh = out.shape
    dims = list(sh.as_list()) if hasattr(sh, "as_list") else [int(d) if d is not None else None for d in sh]
    if len(dims) < 4 or dims[1] is None or dims[2] is None:
        raise RuntimeError(f"Need static spatial dims in output shape, got {dims} for {layer.name}.")
    return int(dims[1]), int(dims[2])


def build_layer_bops_specs(model: Model, quantizable_layers: Sequence[int]) -> List[Dict[str, Any]]:
    """Per-layer shapes for MAC-based BOPs. Uses kernel weights + output tensor (Keras 3-safe)."""
    specs: List[Dict[str, Any]] = []
    for idx in quantizable_layers:
        layer = model.layers[idx]
        if isinstance(layer, Conv2D):
            w = layer.get_weights()[0]
            kh, kw, cin, cout = int(w.shape[0]), int(w.shape[1]), int(w.shape[2]), int(w.shape[3])
            oh, ow = _spatial_hw_from_layer_output(layer)
            specs.append({"type": "conv", "kh": kh, "kw": kw, "cin": cin, "cout": cout, "h": oh, "w": ow})
        elif isinstance(layer, Dense):
            w = layer.get_weights()[0]
            in_dim, out_dim = int(w.shape[0]), int(w.shape[1])
            specs.append({"type": "dense", "in": in_dim, "out": out_dim})
        else:
            raise TypeError(layer)
    return specs


def _layer_mac_bops(spec: Dict[str, Any], bw: int, ba: int) -> float:
    bw_i, ba_i = int(bw), int(ba)
    if spec["type"] == "conv":
        return (
            bw_i * ba_i * spec["kh"] * spec["kw"] * spec["cin"] * spec["cout"] * spec["h"] * spec["w"]
        )
    return float(bw_i * ba_i * spec["in"] * spec["out"])


def compute_real_bops(
    layer_specs: Sequence[Dict[str, Any]],
    bit_config: np.ndarray,
    act_bits: np.ndarray,
) -> Tuple[float, float]:
    """Total MAC BOPs and reference total at 8+8 bits."""
    tot = sum(_layer_mac_bops(spec, int(bit_config[i]), int(act_bits[i])) for i, spec in enumerate(layer_specs))
    ref = sum(_layer_mac_bops(spec, 8, 8) for spec in layer_specs)
    return float(tot), float(ref)


def compute_real_bops_ratios(
    layer_specs: Sequence[Dict[str, Any]],
    bit_config: np.ndarray,
    act_bits: np.ndarray,
) -> Tuple[float, float]:
    """Ratios vs 8+8 reference: (w+a, weight-only with b_a=8)."""
    tot, ref = compute_real_bops(layer_specs, bit_config, act_bits)
    tot_wo, _ = compute_real_bops(layer_specs, bit_config, np.full(len(bit_config), 8, dtype=np.int64))
    ref_f = ref if ref > 0 else 1.0
    return float(tot / ref_f), float(tot_wo / ref_f)


def act_bits_for_config(bit_config: np.ndarray, quantize_activations: bool) -> np.ndarray:
    """Per-layer activation bit-widths for BOPs and fake-quant."""
    if quantize_activations:
        return bit_config.astype(np.int64)
    return np.full(len(bit_config), 8, dtype=np.int64)


def simple_param_bops_ratio(bit_config: np.ndarray, param_counts: np.ndarray) -> Tuple[float, float]:
    """Legacy cost: sum(bits*params) / sum(8*params)."""
    num = float(np.sum(bit_config.astype(np.float64) * param_counts))
    den = float(np.sum(8.0 * param_counts))
    r = num / den if den > 0 else 1.0
    return r, r


def quantize_act_tf(x: tf.Tensor, amin: float, amax: float, bits: int) -> tf.Tensor:
    """Min-max fake-quantization for tensors (PTQ activations)."""
    b = int(bits)
    if b >= 16:
        return x
    levels = (2**b) - 1
    if np.isclose(float(amin), float(amax), rtol=0.0, atol=1e-8):
        return x
    scale = (float(amax) - float(amin)) / float(levels + 1e-8)
    aqmin = tf.cast(amin, x.dtype)
    aqmax = tf.cast(amax, x.dtype)
    st = tf.cast(scale, x.dtype)
    clipped = tf.clip_by_value(x, aqmin, aqmax)
    return tf.round((clipped - aqmin) / st) * st + aqmin


def collect_activation_stats(
    model: Model,
    quantizable_layers: Sequence[int],
    x_calib: np.ndarray,
    y_calib: np.ndarray,
    batch_size: int = 128,
) -> Dict[int, Tuple[float, float]]:
    """Global min/max of each quantizable layer output on calibration data (float model)."""
    outs = [model.layers[i].output for i in quantizable_layers]
    multi = Model(inputs=model.input, outputs=outs)
    ds = make_dataset(x_calib, y_calib, batch_size=batch_size, shuffle=False)
    n = len(quantizable_layers)
    mins = np.full(n, np.inf, dtype=np.float64)
    maxs = np.full(n, -np.inf, dtype=np.float64)
    for batch_x, _ in ds:
        acts = multi(batch_x, training=False)
        if not isinstance(acts, (list, tuple)):
            acts = [acts]
        for j, a in enumerate(acts):
            t = a.numpy()
            mins[j] = min(mins[j], float(np.min(t)))
            maxs[j] = max(maxs[j], float(np.max(t)))
    return {quantizable_layers[j]: (float(mins[j]), float(maxs[j])) for j in range(n)}


def _apply_act_quant_patches(
    model: Model,
    quantizable_layers: Sequence[int],
    bit_config: np.ndarray,
    activation_stats: Dict[int, Tuple[float, float]],
    quantize_activations: bool,
) -> List[Tuple[Any, Any]]:
    """Temporarily wrap ``layer.call`` to quantize outputs; return handles to restore."""
    handles: List[Tuple[Any, Any]] = []
    if not quantize_activations or not activation_stats:
        return handles
    for _li, idx in enumerate(quantizable_layers):
        layer = model.layers[idx]
        amin, amax = activation_stats[idx]
        bits = int(bit_config[_li])
        orig_call = layer.call

        def _patched(
            self,
            *args,
            _oc=orig_call,
            _a0=float(amin),
            _a1=float(amax),
            _b=bits,
            **kwargs,
        ):
            out = _oc(*args, **kwargs)
            return quantize_act_tf(out, _a0, _a1, _b)

        handles.append((layer, orig_call))
        layer.call = types.MethodType(_patched, layer)
    return handles


def _restore_act_quant_patches(handles: Sequence[Tuple[Any, Any]]) -> None:
    for layer, orig_call in handles:
        layer.call = orig_call


def evaluate_bit_config(
    model: Model,
    quantizable_layers: Sequence[int],
    bit_config: np.ndarray,
    x_calib: np.ndarray,
    y_calib: np.ndarray,
    param_counts: np.ndarray,
    original_weights: Dict[int, List[np.ndarray]],
    layer_bops_specs: Sequence[Dict[str, Any]],
    activation_stats: Optional[Dict[int, Tuple[float, float]]] = None,
    quantize_activations: bool = False,
) -> Tuple[float, float, float, float]:
    """Return calib accuracy, BOPs (w+a), BOPs (w-only proxy), mem_ratio."""
    ab = act_bits_for_config(bit_config, quantize_activations)
    if REAL_BOPS:
        bops_wa, bops_wo = compute_real_bops_ratios(layer_bops_specs, bit_config, ab)
    else:
        bops_wa, bops_wo = simple_param_bops_ratio(bit_config, param_counts)
    num = float(np.sum(bit_config.astype(np.float64) * param_counts))
    den = float(np.sum(8.0 * param_counts))
    mem_ratio = num / den if den > 0 else 1.0

    handles = _apply_act_quant_patches(
        model, quantizable_layers, bit_config, activation_stats or {}, quantize_activations
    )
    try:
        apply_bit_config(model, quantizable_layers, bit_config, original_weights)
        calib_ds = make_dataset(x_calib, y_calib, batch_size=128, shuffle=False)
        _, acc = model.evaluate(calib_ds, verbose=0)
    finally:
        restore_original_weights(model, quantizable_layers, original_weights)
        _restore_act_quant_patches(handles)

    return float(acc), float(bops_wa), float(bops_wo), float(mem_ratio)


def compute_fitness(acc: float, bops_ratio: float, mem_ratio: float) -> float:
    """Hardware-aware scalar fitness (higher is better)."""
    return acc - LAMBDA_PENALTY * bops_ratio - MU_PENALTY * mem_ratio


def compute_layer_sensitivity(
    model: Model,
    quantizable_layers: Sequence[int],
    x_calib: np.ndarray,
    y_calib: np.ndarray,
) -> np.ndarray:
    """Normalized gradient L1 magnitudes w.r.t. quantizable layer weights (calib mini-batch)."""
    xc = x_calib[:128].astype(np.float32)
    x_batch = (xc / 255.0 - CIFAR_MEAN) / CIFAR_STD
    y_batch = y_calib[:128].astype(np.int32)

    with tf.GradientTape() as tape:
        preds = model(x_batch, training=False)
        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_batch, preds))

    train_vars = model.trainable_variables
    grads = tape.gradient(loss, train_vars)
    grad_map = {id(v): g for v, g in zip(train_vars, grads)}

    sensitivities: List[float] = []
    for layer_idx in quantizable_layers:
        layer = model.layers[layer_idx]
        mags: List[float] = []
        for var in layer.trainable_weights:
            g = grad_map.get(id(var))
            if g is not None:
                mags.append(float(tf.reduce_mean(tf.abs(g)).numpy()))
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
    use_sensitivity_prior: bool = True,
) -> np.ndarray:
    """Random {2,4,8}^L with optional floor of 4 bits on top-quartile sensitive layers."""
    configs = np.random.choice(BITS, size=(num_bees, num_layers), replace=True)
    if not use_sensitivity_prior:
        return configs.astype(np.int32)
    threshold = np.quantile(sensitivity, 0.75)
    sensitive_mask = sensitivity >= threshold
    for i in range(num_bees):
        for d in np.where(sensitive_mask)[0]:
            if configs[i, d] < 4:
                configs[i, d] = np.random.choice([4, 8])
    return configs.astype(np.int32)


def choose_mutation_dimension(sensitivity: np.ndarray, use_sensitivity_prior: bool = True) -> int:
    """Pick layer index: uniform or inverse-sensitivity weighted."""
    n = len(sensitivity)
    if not use_sensitivity_prior:
        return int(np.random.randint(0, n))
    inv = np.clip(1.0 - sensitivity, 1e-8, None)
    inv = inv / inv.sum()
    return int(np.random.choice(n, p=inv))


def mutate_config(
    config: np.ndarray,
    sensitivity: np.ndarray,
    use_sensitivity_prior: bool = True,
) -> np.ndarray:
    """Flip one layer to a different admissible bit-width."""
    new_config = config.copy()
    d = choose_mutation_dimension(sensitivity, use_sensitivity_prior)
    options = [b for b in BITS if b != config[d]]
    new_config[d] = np.random.choice(options)
    return new_config


def format_bits(config: np.ndarray) -> str:
    """Human-readable bit vector."""
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
    layer_bops_specs: Sequence[Dict[str, Any]],
    activation_stats: Optional[Dict[int, Tuple[float, float]]] = None,
    quantize_activations: bool = False,
    use_sensitivity_prior: bool = True,
    convergence: Optional[Dict[str, List[float]]] = None,
) -> Tuple[SearchResult, int]:
    """ABC loop: employed bees, onlookers (roulette), scouts; return best and number of evals."""
    original_weights = {idx: copy.deepcopy(model.layers[idx].get_weights()) for idx in quantizable_layers}
    num_layers = len(quantizable_layers)
    foods = initialize_food_sources(num_bees, num_layers, sensitivity, use_sensitivity_prior)
    trials = np.zeros(num_bees, dtype=np.int32)

    scores = np.zeros(num_bees, dtype=np.float64)
    accs = np.zeros(num_bees, dtype=np.float64)
    bops = np.zeros(num_bees, dtype=np.float64)
    bops_wo = np.zeros(num_bees, dtype=np.float64)
    mems = np.zeros(num_bees, dtype=np.float64)
    eval_count = 0

    def _eval(cfg: np.ndarray) -> Tuple[float, float, float, float]:
        return evaluate_bit_config(
            model,
            quantizable_layers,
            cfg,
            x_calib,
            y_calib,
            param_counts,
            original_weights,
            layer_bops_specs,
            activation_stats,
            quantize_activations,
        )

    for i in range(num_bees):
        acc, bwa, bwo, mem = _eval(foods[i])
        scores[i] = compute_fitness(acc, bwa, mem)
        accs[i], bops[i], bops_wo[i], mems[i] = acc, bwa, bwo, mem
        eval_count += 1

    best_idx = int(np.argmax(scores))
    best = SearchResult(
        bit_config=foods[best_idx].copy(),
        fitness=float(scores[best_idx]),
        accuracy=float(accs[best_idx]),
        bops_ratio=float(bops[best_idx]),
        mem_ratio=float(mems[best_idx]),
        bops_weight_act=float(bops[best_idx]),
        bops_weight_only=float(bops_wo[best_idx]),
    )

    for cycle in range(1, cycles + 1):
        scouts_triggered = 0

        for i in range(num_bees):
            candidate = mutate_config(foods[i], sensitivity, use_sensitivity_prior)
            acc, bwa, bwo, mem = _eval(candidate)
            fit = compute_fitness(acc, bwa, mem)
            eval_count += 1
            if fit > scores[i]:
                foods[i] = candidate
                scores[i] = fit
                accs[i], bops[i], bops_wo[i], mems[i] = acc, bwa, bwo, mem
                trials[i] = 0
            else:
                trials[i] += 1

        probs = softmax(scores)
        for _ in range(num_bees):
            i = int(np.random.choice(num_bees, p=probs))
            candidate = mutate_config(foods[i], sensitivity, use_sensitivity_prior)
            acc, bwa, bwo, mem = _eval(candidate)
            fit = compute_fitness(acc, bwa, mem)
            eval_count += 1
            if fit > scores[i]:
                foods[i] = candidate
                scores[i] = fit
                accs[i], bops[i], bops_wo[i], mems[i] = acc, bwa, bwo, mem
                trials[i] = 0
            else:
                trials[i] += 1

        threshold = np.quantile(sensitivity, 0.75)
        sensitive_mask = sensitivity >= threshold
        for i in range(num_bees):
            if trials[i] >= scout_limit:
                reset = np.random.choice(BITS, size=num_layers, replace=True).astype(np.int32)
                if use_sensitivity_prior:
                    for d in np.where(sensitive_mask)[0]:
                        if reset[d] < 4:
                            reset[d] = np.random.choice([4, 8])
                acc, bwa, bwo, mem = _eval(reset)
                fit = compute_fitness(acc, bwa, mem)
                foods[i] = reset
                scores[i] = fit
                accs[i], bops[i], bops_wo[i], mems[i] = acc, bwa, bwo, mem
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
                bops_weight_act=float(bops[best_idx]),
                bops_weight_only=float(bops_wo[best_idx]),
            )

        if convergence is not None:
            convergence.setdefault("abc_cycle", []).append(float(cycle))
            convergence.setdefault("abc_best_fitness", []).append(float(best.fitness))
            convergence.setdefault("abc_mean_fitness", []).append(float(np.mean(scores)))

        print(
            f"[ABC {cycle}] best fit={best.fitness:.4f} acc={best.accuracy:.4f} "
            f"bops_wa={best.bops_weight_act:.4f} scouts={scouts_triggered}"
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
    layer_bops_specs: Sequence[Dict[str, Any]],
    activation_stats: Optional[Dict[int, Tuple[float, float]]] = None,
    quantize_activations: bool = False,
) -> SearchResult:
    """Uniform bit-width across all quantizable layers."""
    bit_config = np.full(len(quantizable_layers), bit_value, dtype=np.int32)
    ow = {idx: copy.deepcopy(model.layers[idx].get_weights()) for idx in quantizable_layers}
    acc, bwa, bwo, mem = evaluate_bit_config(
        model,
        quantizable_layers,
        bit_config,
        x_calib,
        y_calib,
        param_counts,
        ow,
        layer_bops_specs,
        activation_stats,
        quantize_activations,
    )
    return SearchResult(
        bit_config=bit_config,
        fitness=compute_fitness(acc, bwa, mem),
        accuracy=acc,
        bops_ratio=bwa,
        mem_ratio=mem,
        bops_weight_act=bwa,
        bops_weight_only=bwo,
    )


def run_random_search(
    model: Model,
    quantizable_layers: Sequence[int],
    x_calib: np.ndarray,
    y_calib: np.ndarray,
    param_counts: np.ndarray,
    sensitivity: np.ndarray,
    budget: int,
    layer_bops_specs: Sequence[Dict[str, Any]],
    activation_stats: Optional[Dict[int, Tuple[float, float]]] = None,
    quantize_activations: bool = False,
) -> SearchResult:
    """Uniform random mixed precision with the same sensitive-layer floor as ABC init."""
    original_weights = {idx: copy.deepcopy(model.layers[idx].get_weights()) for idx in quantizable_layers}
    num_layers = len(quantizable_layers)
    best: Optional[SearchResult] = None
    threshold = np.quantile(sensitivity, 0.75)
    sensitive_mask = sensitivity >= threshold

    for _ in range(budget):
        config = np.random.choice(BITS, size=num_layers, replace=True).astype(np.int32)
        for d in np.where(sensitive_mask)[0]:
            if config[d] < 4:
                config[d] = np.random.choice([4, 8])
        acc, bwa, bwo, mem = evaluate_bit_config(
            model,
            quantizable_layers,
            config,
            x_calib,
            y_calib,
            param_counts,
            original_weights,
            layer_bops_specs,
            activation_stats,
            quantize_activations,
        )
        cand = SearchResult(
            bit_config=config,
            fitness=compute_fitness(acc, bwa, mem),
            accuracy=acc,
            bops_ratio=bwa,
            mem_ratio=mem,
            bops_weight_act=bwa,
            bops_weight_only=bwo,
        )
        if best is None or cand.fitness > best.fitness:
            best = cand

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
    layer_bops_specs: Sequence[Dict[str, Any]],
    activation_stats: Optional[Dict[int, Tuple[float, float]]] = None,
    quantize_activations: bool = False,
) -> List[Tuple[float, float]]:
    """Sample random configs; list of (bops_wa, accuracy) for Pareto scatter."""
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
        acc, bwa, _bwo, mem = evaluate_bit_config(
            model,
            quantizable_layers,
            config,
            x_calib,
            y_calib,
            param_counts,
            original_weights,
            layer_bops_specs,
            activation_stats,
            quantize_activations,
        )
        _ = mem
        points.append((float(bwa), float(acc)))
    restore_original_weights(model, quantizable_layers, original_weights)
    return points


def bits_from_continuous(z: np.ndarray) -> np.ndarray:
    """Map continuous [0,2] per dim to {2,4,8}."""
    idx = np.clip(np.round(z).astype(np.int32), 0, 2)
    return BITS[idx]


def run_cmaes_baseline(
    model: Model,
    quantizable_layers: Sequence[int],
    x_calib: np.ndarray,
    y_calib: np.ndarray,
    param_counts: np.ndarray,
    budget: int,
    layer_bops_specs: Sequence[Dict[str, Any]],
    activation_stats: Optional[Dict[int, Tuple[float, float]]] = None,
    quantize_activations: bool = False,
    cma_seed: int = SEED,
    convergence: Optional[Dict[str, List[float]]] = None,
) -> SearchResult:
    """CMA-ES in [0,2]^L with projection to discrete bits; minimize negative fitness."""
    original_weights = {idx: copy.deepcopy(model.layers[idx].get_weights()) for idx in quantizable_layers}
    dim = len(quantizable_layers)
    best: Optional[SearchResult] = None
    n_eval = 0

    def objective(z: np.ndarray) -> float:
        nonlocal best, n_eval
        z = np.clip(np.asarray(z, dtype=np.float64).ravel(), 0.0, 2.0)
        config = bits_from_continuous(z)
        acc, bwa, bwo, mem = evaluate_bit_config(
            model,
            quantizable_layers,
            config,
            x_calib,
            y_calib,
            param_counts,
            original_weights,
            layer_bops_specs,
            activation_stats,
            quantize_activations,
        )
        fit = compute_fitness(acc, bwa, mem)
        cand = SearchResult(
            config.copy(),
            fit,
            acc,
            bwa,
            mem,
            bops_weight_act=bwa,
            bops_weight_only=bwo,
        )
        if best is None or fit > best.fitness:
            best = cand
        n_eval += 1
        if convergence is not None:
            convergence.setdefault("cma_eval", []).append(float(n_eval))
            convergence.setdefault("cma_best_fitness", []).append(float(best.fitness))
        return float(-fit)

    np.random.seed(int(cma_seed))
    opts = {
        "bounds": [list(np.zeros(dim)), list(2.0 * np.ones(dim))],
        "maxfevals": int(budget),
        "verb_disp": 0,
        "verb_log": 0,
        "seed": int(cma_seed),
    }
    es = cma.CMAEvolutionStrategy(np.ones(dim), 0.6, opts)
    es.optimize(objective)

    restore_original_weights(model, quantizable_layers, original_weights)
    assert best is not None
    return best


def apply_best_config_permanently(
    model: Model,
    quantizable_layers: Sequence[int],
    bit_config: np.ndarray,
) -> None:
    """Write min-max quantized weights in-place (weights only; no activation wrappers)."""
    for layer_idx, bits in zip(quantizable_layers, bit_config):
        layer = model.layers[layer_idx]
        weights = layer.get_weights()
        layer.set_weights([quantize_array_minmax(w, int(bits)) for w in weights])


def _agg(xs: Sequence[float]) -> Tuple[float, float]:
    a = np.asarray(xs, dtype=np.float64)
    dd = 1 if len(a) > 1 else 0
    return float(a.mean()), float(a.std(ddof=dd))


def main() -> None:
    """Train float ResNet-20, run searches across seeds, print table, save figures under ``results/``."""
    config = get_experiment_config()
    use_prior_cfg = bool(config.get("use_sensitivity_prior", True))
    run_ablate = bool(config.get("run_sensitivity_ablation", False))
    seeds = active_seeds()
    results_dir = os.environ.get("ABC_Q_RESULTS_DIR", DEFAULT_RESULTS_DIR)
    os.makedirs(results_dir, exist_ok=True)

    print(
        f"FULL_DATASET={FULL_DATASET} N_SEEDS={N_SEEDS} QUANTIZE_ACTIVATIONS={QUANTIZE_ACTIVATIONS} "
        f"REAL_BOPS={REAL_BOPS} seeds={seeds}"
    )

    data = load_cifar10_splits(config)
    train_ds = make_dataset(
        data["x_train"],
        data["y_train"],
        config["batch_size"],
        shuffle=True,
        augment=True,
        shuffle_seed=int(seeds[0]),
    )
    test_ds = make_dataset(data["x_test"], data["y_test"], config["batch_size"], shuffle=False)

    steps_per_epoch = max(1, len(data["x_train"]) // config["batch_size"])
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-3,
        decay_steps=steps_per_epoch * config["pretrain_epochs"],
        alpha=0.06,
    )
    set_seed(int(seeds[0]))
    model = build_model(learning_rate=lr_schedule)
    model.fit(train_ds, epochs=config["pretrain_epochs"], verbose=2)
    _, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"Float test accuracy: {test_acc:.4f}")

    float_weights = copy.deepcopy(model.get_weights())
    quantizable_layers = get_quantizable_layers(model)
    param_counts = layer_param_counts(model, quantizable_layers)
    layer_bops_specs = build_layer_bops_specs(model, quantizable_layers)

    act_stats: Optional[Dict[int, Tuple[float, float]]] = None
    if QUANTIZE_ACTIVATIONS:
        act_stats = collect_activation_stats(model, quantizable_layers, data["x_calib"], data["y_calib"])

    sensitivity = compute_layer_sensitivity(model, quantizable_layers, data["x_calib"], data["y_calib"])

    uniform8 = evaluate_baseline_config(
        model, quantizable_layers, 8, data["x_calib"], data["y_calib"], param_counts,
        layer_bops_specs, act_stats, QUANTIZE_ACTIVATIONS,
    )
    uniform4 = evaluate_baseline_config(
        model, quantizable_layers, 4, data["x_calib"], data["y_calib"], param_counts,
        layer_bops_specs, act_stats, QUANTIZE_ACTIVATIONS,
    )
    uniform2 = evaluate_baseline_config(
        model, quantizable_layers, 2, data["x_calib"], data["y_calib"], param_counts,
        layer_bops_specs, act_stats, QUANTIZE_ACTIVATIONS,
    )

    abc_accs: List[float] = []
    abc_bwa: List[float] = []
    abc_bwo: List[float] = []
    rand_accs: List[float] = []
    rand_bwa: List[float] = []
    rand_bwo: List[float] = []
    cma_accs: List[float] = []
    cma_bwa: List[float] = []
    cma_bwo: List[float] = []
    abc_np_accs: List[float] = []
    abc_np_bwa: List[float] = []
    abc_np_bwo: List[float] = []

    last_abc: Optional[SearchResult] = None
    last_cma: Optional[SearchResult] = None
    last_rand: Optional[SearchResult] = None
    last_abc_noprior: Optional[SearchResult] = None
    pareto_pts: List[Tuple[float, float]] = []
    abc_conv_plot: Dict[str, List[float]] = {}
    cma_conv_plot: Dict[str, List[float]] = {}
    shared_budget: Optional[int] = None

    for si, s in enumerate(seeds):
        set_seed(s)
        model.set_weights(copy.deepcopy(float_weights))

        conv_abc: Dict[str, List[float]] = {}
        if run_ablate:
            abc_best, abc_budget = run_abc_q(
                model, quantizable_layers, data["x_calib"], data["y_calib"], param_counts, sensitivity,
                config["num_bees"], config["abc_cycles"], config["scout_limit"],
                layer_bops_specs, act_stats, QUANTIZE_ACTIVATIONS, True, conv_abc,
            )
        else:
            abc_best, abc_budget = run_abc_q(
                model, quantizable_layers, data["x_calib"], data["y_calib"], param_counts, sensitivity,
                config["num_bees"], config["abc_cycles"], config["scout_limit"],
                layer_bops_specs, act_stats, QUANTIZE_ACTIVATIONS, use_prior_cfg, conv_abc,
            )

        if shared_budget is None:
            shared_budget = int(abc_budget)

        abc_accs.append(float(abc_best.accuracy))
        abc_bwa.append(float(abc_best.bops_weight_act))
        abc_bwo.append(float(abc_best.bops_weight_only))

        if run_ablate:
            set_seed(s)
            model.set_weights(copy.deepcopy(float_weights))
            abc_ablate, _ = run_abc_q(
                model, quantizable_layers, data["x_calib"], data["y_calib"], param_counts, sensitivity,
                config["num_bees"], config["abc_cycles"], config["scout_limit"],
                layer_bops_specs, act_stats, QUANTIZE_ACTIVATIONS, False, None,
            )
            abc_np_accs.append(float(abc_ablate.accuracy))
            abc_np_bwa.append(float(abc_ablate.bops_weight_act))
            abc_np_bwo.append(float(abc_ablate.bops_weight_only))
            if si == len(seeds) - 1:
                last_abc_noprior = abc_ablate

        set_seed(s)
        model.set_weights(copy.deepcopy(float_weights))
        random_best = run_random_search(
            model, quantizable_layers, data["x_calib"], data["y_calib"], param_counts, sensitivity,
            int(shared_budget), layer_bops_specs, act_stats, QUANTIZE_ACTIVATIONS,
        )
        rand_accs.append(float(random_best.accuracy))
        rand_bwa.append(float(random_best.bops_weight_act))
        rand_bwo.append(float(random_best.bops_weight_only))

        set_seed(s)
        model.set_weights(copy.deepcopy(float_weights))
        conv_cma: Dict[str, List[float]] = {}
        cma_best = run_cmaes_baseline(
            model, quantizable_layers, data["x_calib"], data["y_calib"], param_counts,
            int(shared_budget), layer_bops_specs, act_stats, QUANTIZE_ACTIVATIONS, s, conv_cma,
        )
        cma_accs.append(float(cma_best.accuracy))
        cma_bwa.append(float(cma_best.bops_weight_act))
        cma_bwo.append(float(cma_best.bops_weight_only))

        if si == len(seeds) - 1:
            last_abc = abc_best
            last_cma = cma_best
            last_rand = random_best
            abc_conv_plot = conv_abc
            cma_conv_plot = conv_cma
            n_sc = int(config.get("pareto_random_samples", max(200, min(500, int(shared_budget)))))
            pareto_pts = collect_random_search_points(
                model, quantizable_layers, data["x_calib"], data["y_calib"], param_counts, sensitivity,
                n_sc, layer_bops_specs, act_stats, QUANTIZE_ACTIVATIONS,
            )

    assert last_abc is not None and last_cma is not None and last_rand is not None and shared_budget is not None

    ma, sa = _agg(abc_accs)
    mb, sb = _agg(abc_bwa)
    mwo, swo = _agg(abc_bwo)
    rma, rsa = _agg(rand_accs)
    rmb, rsb = _agg(rand_bwa)
    rmwo, rswo = _agg(rand_bwo)
    cma_m, cma_s = _agg(cma_accs)
    cmb, csb = _agg(cma_bwa)
    cmwo, cswo = _agg(cma_bwo)

    p_rand, st_rand = wilcoxon_paired_pvalue(abc_accs, rand_accs)
    p_cma, st_cma = wilcoxon_paired_pvalue(abc_accs, cma_accs)
    wilcoxon_vs_abc: Dict[str, Tuple[float, str]] = {
        "Random Search": (p_rand, st_rand),
        "CMA-ES (Hansen)": (p_cma, st_cma),
    }

    rows: Dict[str, Tuple[float, float, float, float, float, float]] = {
        "Uniform 8-bit": (float(uniform8.accuracy), 0.0, float(uniform8.bops_weight_act), 0.0, 0.0, 0.0),
        "Uniform 4-bit": (float(uniform4.accuracy), 0.0, float(uniform4.bops_weight_act), 0.0, 0.0, 0.0),
        "Uniform 2-bit": (float(uniform2.accuracy), 0.0, float(uniform2.bops_weight_act), 0.0, 0.0, 0.0),
        "Random Search": (rma, rsa, rmb, rsb, rmwo, rswo),
        "CMA-ES (Hansen)": (cma_m, cma_s, cmb, csb, cmwo, cswo),
        "ABC-Q (ours)": (ma, sa, mb, sb, mwo, swo),
    }

    row_order = [
        "Uniform 8-bit",
        "Uniform 4-bit",
        "Uniform 2-bit",
        "Random Search",
        "CMA-ES (Hansen)",
    ]
    sens_line: Optional[str] = None
    if run_ablate and last_abc_noprior is not None:
        np_ma, np_sa = _agg(abc_np_accs)
        np_mb, np_sb = _agg(abc_np_bwa)
        np_mwo, np_swo = _agg(abc_np_bwo)
        rows["ABC-Q (no prior)"] = (np_ma, np_sa, np_mb, np_sb, np_mwo, np_swo)
        row_order.append("ABC-Q (no prior)")
        p_np, st_np = wilcoxon_paired_pvalue(abc_accs, abc_np_accs)
        wilcoxon_vs_abc["ABC-Q (no prior)"] = (p_np, st_np)
        gm, gs = paired_mean_std(abc_accs, abc_np_accs)
        sens_line = f"Sensitivity prior gain (calib acc, pp): {gm * 100:.2f} ± {gs * 100:.2f} (n={len(seeds)})."
    row_order.append("ABC-Q (ours)")

    pc = float(np.sum(param_counts))

    def avg_bits(res: SearchResult) -> float:
        return float(np.sum(res.bit_config.astype(np.float64) * param_counts) / pc)

    avg_bits_by_method: Dict[str, float] = {
        "Uniform 8-bit": 8.0,
        "Uniform 4-bit": 4.0,
        "Uniform 2-bit": 2.0,
        "Random Search": avg_bits(last_rand),
        "CMA-ES (Hansen)": avg_bits(last_cma),
        "ABC-Q (ours)": avg_bits(last_abc),
    }
    if run_ablate and last_abc_noprior is not None:
        avg_bits_by_method["ABC-Q (no prior)"] = avg_bits(last_abc_noprior)

    print(f"\nLast-seed ABC bits: {format_bits(last_abc.bit_config)}")
    print_paper_table(row_order, rows, avg_bits_by_method, wilcoxon_vs_abc, len(seeds))
    if sens_line:
        print(sens_line)
    print(f"Wilcoxon vs ABC-Q: Random p={p_rand:.4f}{st_rand}; CMA-ES p={p_cma:.4f}{st_cma}.")

    save_all_paper_figures(
        results_dir,
        [
            ("Uniform 8-bit", uniform8.bops_weight_act, uniform8.accuracy),
            ("Uniform 4-bit", uniform4.bops_weight_act, uniform4.accuracy),
            ("Uniform 2-bit", uniform2.bops_weight_act, uniform2.accuracy),
        ],
        pareto_pts,
        float(last_abc.bops_weight_act),
        float(last_abc.accuracy),
        float(last_cma.bops_weight_act),
        float(last_cma.accuracy),
        last_abc.bit_config,
        sensitivity,
        abc_conv_plot,
        cma_conv_plot,
        rows,
        row_order,
    )
    print(f"Figures saved under: {results_dir}")

    final_model = build_model()
    final_model.set_weights(copy.deepcopy(float_weights))
    ft_steps = max(1, len(data["x_train"]) // config["batch_size"]) * config["finetune_epochs"]
    ft_lr = tf.keras.optimizers.schedules.CosineDecay(2e-4, decay_steps=ft_steps, alpha=0.15)
    final_model.compile(
        optimizer=Adam(learning_rate=ft_lr),
        loss=sparse_cce_with_label_smoothing(10, LABEL_SMOOTHING),
        metrics=["accuracy"],
    )
    apply_best_config_permanently(final_model, quantizable_layers, last_abc.bit_config)
    final_model.fit(train_ds, epochs=config["finetune_epochs"], verbose=2)
    _, final_acc = final_model.evaluate(test_ds, verbose=0)
    print(f"Test accuracy after quant + fine-tune (last-seed ABC): {final_acc:.4f}")


if __name__ == "__main__":
    main()
