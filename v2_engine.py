"""ABC-Q v2 — fast, retrace-free mixed-precision evaluator (TensorFlow / Keras 3).

Why a new evaluator
-------------------
The v1 pipeline (``abc_q_core.evaluate_bit_config``) re-quantised every layer with NumPy, called
``set_weights`` for all layers and ran ``model.evaluate`` for every candidate. Activation fake-quant
was injected by monkey-patching ``layer.call`` with the bit-width baked into a Python closure; because
Keras caches its compiled test function, those closures can be frozen at the first trace.

This module fixes both issues:

* **Weights** — per-channel *symmetric* uniform quantisation of each kernel (bias kept in float, as
  int32-accumulator deployments do). Quantised kernels for every candidate bit-width are computed
  once and cached as device tensors; switching a configuration only re-assigns kernels whose
  bit-width changed.
* **Activations** — the *input* of every quantisable layer is min–max fake-quantised with the same
  bit-width as that layer's weights (so BOPs = b_w * b_a * MACs is consistent with what is
  simulated). Range statistics come from the calibration set (float model). Bit-widths live in
  ``tf.Variable``s read inside the traced graph, so the compiled forward pass is traced once and
  every configuration change is honoured.
"""

from __future__ import annotations

import os
import types
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, DepthwiseConv2D

import abc_q_core as q

BITS = (2, 4, 8)


def normalize(x: np.ndarray) -> np.ndarray:
    return ((x / 255.0 - q.CIFAR_MEAN) / q.CIFAR_STD).astype(np.float32)


def quantize_kernel_per_channel(w: np.ndarray, bits: int, layer_type: str) -> np.ndarray:
    """Symmetric per-output-channel uniform quantiser; 2-bit is ternary {-s, 0, s}."""
    if layer_type == "depthwise":  # (kh, kw, cin, dm): one scale per (cin, dm)
        axes = (0, 1)
    elif layer_type == "conv":  # (kh, kw, cin, cout)
        axes = (0, 1, 2)
    else:  # dense (in, out)
        axes = (0,)
    qmax = float(2 ** (bits - 1) - 1)
    amax = np.max(np.abs(w), axis=axes, keepdims=True)
    scale = np.where(amax > 0, amax / qmax, 1.0)
    return (np.clip(np.round(w / scale), -qmax, qmax) * scale).astype(np.float32)


def fake_quant_act(x: tf.Tensor, lo: tf.Variable, hi: tf.Variable, nlev: tf.Variable,
                   ste: bool = False) -> tf.Tensor:
    """Asymmetric min-max fake quantisation with ``nlev = 2^b - 1`` steps (optionally STE)."""
    s = tf.maximum((hi - lo) / nlev, 1e-8)
    xc = tf.clip_by_value(x, lo, hi)
    xq = tf.round((xc - lo) / s) * s + lo
    if ste:
        return xc + tf.stop_gradient(xq - xc)
    return xq


class FastEvaluator:
    """Evaluate calibration / test accuracy of a bit configuration in one compiled forward pass."""

    def __init__(self, model: tf.keras.Model, x_calib: np.ndarray, y_calib: np.ndarray,
                 x_test: np.ndarray, y_test: np.ndarray, quantize_activations: bool = True):
        self.model = model
        self.names: List[str] = q.get_quantizable_layer_names(model)
        self.L = len(self.names)
        self.layers = [model.get_layer(n) for n in self.names]
        self.types: List[str] = []
        for lyr in self.layers:
            if isinstance(lyr, DepthwiseConv2D):
                self.types.append("depthwise")
            elif isinstance(lyr, Conv2D):
                self.types.append("conv")
            elif isinstance(lyr, Dense):
                self.types.append("dense")
            else:
                raise TypeError(lyr)
        self.xc = tf.constant(normalize(x_calib))
        self.yc = np.asarray(y_calib).astype(np.int64)
        self.xt = normalize(x_test)
        self.yt = np.asarray(y_test).astype(np.int64)
        self.quantize_activations = quantize_activations

        # float kernels + cached quantised kernels
        self.kernel_vars = [lyr.kernel for lyr in self.layers]
        self.float_kernels = [v.numpy().copy() for v in self.kernel_vars]
        self.qkernels: Dict[Tuple[int, int], tf.Tensor] = {}
        for i, w in enumerate(self.float_kernels):
            for b in BITS:
                self.qkernels[(i, b)] = tf.constant(quantize_kernel_per_channel(w, b, self.types[i]))
        self.n_params = np.array([w.size for w in self.float_kernels], dtype=np.float64)

        # BOPs specs (MACs per layer)
        specs = q.build_layer_bops_specs(model, self.names)
        self.macs = np.array([q._layer_mac_bops(s, 1, 1) for s in specs], dtype=np.float64)
        self.ref_bops = float(np.sum(64.0 * self.macs))

        # activation quantisers on layer inputs
        self.lo = [tf.Variable(0.0, trainable=False, dtype=tf.float32) for _ in range(self.L)]
        self.hi = [tf.Variable(1.0, trainable=False, dtype=tf.float32) for _ in range(self.L)]
        self.nlev = [tf.Variable(255.0, trainable=False, dtype=tf.float32) for _ in range(self.L)]
        self.act_on = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.ste = False
        self._orig_calls = [lyr.call for lyr in self.layers]
        self._collect_input_ranges()
        self._install_quant_patches()

        self._fwd = tf.function(lambda x: self.model(x, training=False), jit_compile=os.environ.get("ABCQ_XLA", "0") == "1")
        self.current: Optional[np.ndarray] = None
        self.n_evals = 0

    # ------------------------------------------------------------------ setup
    def _collect_input_ranges(self) -> None:
        mins = np.full(self.L, np.inf)
        maxs = np.full(self.L, -np.inf)
        rec: Dict[int, List[float]] = {}

        for i, lyr in enumerate(self.layers):
            oc = self._orig_calls[i]

            def _rec(self_l, inputs, *a, _i=i, _oc=oc, **k):
                rec.setdefault(_i, []).append((float(tf.reduce_min(inputs)), float(tf.reduce_max(inputs))))
                return _oc(inputs, *a, **k)

            lyr.call = types.MethodType(_rec, lyr)
        for s in range(0, int(self.xc.shape[0]), 250):
            self.model(self.xc[s:s + 250], training=False)
        for i, lyr in enumerate(self.layers):
            lyr.call = self._orig_calls[i]
            arr = np.array(rec[i])
            mins[i], maxs[i] = arr[:, 0].min(), arr[:, 1].max()
            self.lo[i].assign(float(mins[i]))
            self.hi[i].assign(float(maxs[i]) if maxs[i] > mins[i] else float(mins[i]) + 1e-3)

    def _install_quant_patches(self) -> None:
        ev = self
        for i, lyr in enumerate(self.layers):
            oc = self._orig_calls[i]

            def _qcall(self_l, inputs, *a, _i=i, _oc=oc, **k):
                xq = fake_quant_act(inputs, ev.lo[_i], ev.hi[_i], ev.nlev[_i], ste=ev.ste)
                x = ev.act_on * xq + (1.0 - ev.act_on) * inputs
                return _oc(x, *a, **k)

            lyr.call = types.MethodType(_qcall, lyr)

    # ------------------------------------------------------------------ config
    def set_config(self, cfg: Optional[np.ndarray]) -> None:
        """Apply a bit configuration (None = float model)."""
        if cfg is None:
            for v, w in zip(self.kernel_vars, self.float_kernels):
                v.assign(w)
            self.act_on.assign(0.0)
            self.current = None
            return
        cfg = np.asarray(cfg, dtype=np.int64)
        for i in range(self.L):
            b = int(cfg[i])
            if self.current is None or int(self.current[i]) != b:
                self.kernel_vars[i].assign(self.qkernels[(i, b)])
                self.nlev[i].assign(float(2 ** b - 1))
        self.act_on.assign(1.0 if self.quantize_activations else 0.0)
        self.current = cfg.copy()

    def bops_ratio(self, cfg: np.ndarray) -> float:
        b = np.asarray(cfg, dtype=np.float64)
        ba = b if self.quantize_activations else np.full_like(b, 8.0)
        return float(np.sum(b * ba * self.macs) / self.ref_bops)

    def layer_costs(self) -> np.ndarray:
        """(L, 3) BOPs cost (fraction of W8A8 total) of each layer at each bit-width in BITS."""
        out = np.zeros((self.L, len(BITS)))
        for j, b in enumerate(BITS):
            ba = b if self.quantize_activations else 8
            out[:, j] = b * ba * self.macs / self.ref_bops
        return out

    def mem_ratio(self, cfg: np.ndarray) -> float:
        b = np.asarray(cfg, dtype=np.float64)
        return float(np.sum(b * self.n_params) / np.sum(8.0 * self.n_params))

    # ------------------------------------------------------------------ eval
    def calib_acc(self, cfg: Optional[np.ndarray]) -> float:
        self.set_config(cfg)
        p = self._fwd(self.xc).numpy()
        self.n_evals += 1
        return float(np.mean(np.argmax(p, axis=1) == self.yc))

    def test_acc(self, cfg: Optional[np.ndarray], bs: int = 1000) -> float:
        self.set_config(cfg)
        correct = 0
        for s in range(0, len(self.xt), bs):
            p = self._fwd(tf.constant(self.xt[s:s + bs])).numpy()
            correct += int(np.sum(np.argmax(p, axis=1) == self.yt[s:s + bs]))
        return correct / len(self.xt)

    # ------------------------------------------------------------------ sensitivity
    def probe_sensitivity(self) -> Tuple[float, np.ndarray]:
        """Measured single-layer accuracy drop Δ[i, j] when layer i alone is at BITS[j] (others 8)."""
        base = np.full(self.L, 8, dtype=np.int64)
        acc8 = self.calib_acc(base)
        delta = np.zeros((self.L, len(BITS)))
        for i in range(self.L):
            for j, b in enumerate(BITS):
                if b == 8:
                    continue
                c = base.copy()
                c[i] = b
                delta[i, j] = acc8 - self.calib_acc(c)
        return acc8, delta

    def gradient_prior(self, n: int = 128) -> np.ndarray:
        """v1 prior: normalised mean |dL/dW| per layer on n calibration images (float model)."""
        self.set_config(None)
        x = self.xc[:n]
        y = self.yc[:n]
        with tf.GradientTape() as tape:
            p = self.model(x, training=False)
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, p))
        g = tape.gradient(loss, self.kernel_vars)
        s = np.array([float(tf.reduce_mean(tf.abs(gi))) for gi in g])
        return s / s.sum()

    def fisher_trace(self, n_batches: int = 8, bs: int = 125) -> np.ndarray:
        """Empirical-Fisher trace proxy per layer: mean over mini-batches of ||dL/dW||^2."""
        self.set_config(None)
        tr = np.zeros(self.L)
        for k in range(n_batches):
            x = self.xc[k * bs:(k + 1) * bs]
            y = self.yc[k * bs:(k + 1) * bs]
            with tf.GradientTape() as tape:
                p = self.model(x, training=False)
                loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, p))
            g = tape.gradient(loss, self.kernel_vars)
            tr += np.array([float(tf.reduce_sum(tf.square(gi))) for gi in g])
        return tr / n_batches

    def weight_qerr(self) -> np.ndarray:
        """(L, 3) squared weight quantisation error ||Q_b(W) - W||^2."""
        out = np.zeros((self.L, len(BITS)))
        for i, w in enumerate(self.float_kernels):
            for j, b in enumerate(BITS):
                out[i, j] = float(np.sum((self.qkernels[(i, b)].numpy() - w) ** 2))
        return out
