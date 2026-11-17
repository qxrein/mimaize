"""PyTorch evaluator for the ImageNet and LLM extensions of ABC-Q.

Same design as the TensorFlow evaluator used for CIFAR:
* quantised weights of every (layer, bit-width) pair are computed once and cached on the device;
  switching configurations only swaps ``module.weight.data`` pointers for layers whose bit-width changed;
* optional activation fake-quantisation of every quantisable layer's *input* (same bit-width as the
  layer's weights), with ranges collected on the calibration set and bit-widths held in device tensors.

Weight quantisers
* ``per_channel_symmetric`` — CNNs: one scale per output channel, symmetric grid (2-bit = ternary);
* ``group_asymmetric`` — LLMs: round-to-nearest with min/max per group of 128 input weights (the
  standard RTN baseline of weight-only LLM quantisation).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


def per_channel_symmetric(w: torch.Tensor, bits: int) -> torch.Tensor:
    wf = w.float()
    qmax = float(2 ** (bits - 1) - 1)
    amax = wf.abs().reshape(wf.shape[0], -1).amax(dim=1).clamp_min(1e-12)
    shape = [-1] + [1] * (wf.dim() - 1)
    scale = (amax / qmax).reshape(shape)
    return (torch.clamp(torch.round(wf / scale), -qmax, qmax) * scale).to(w.dtype)


def group_asymmetric(w: torch.Tensor, bits: int, group: int = 128) -> torch.Tensor:
    out_f, in_f = w.shape
    assert in_f % group == 0, (in_f, group)
    wf = w.float().reshape(out_f, in_f // group, group)
    lo = wf.amin(dim=2, keepdim=True)
    hi = wf.amax(dim=2, keepdim=True)
    nlev = float(2 ** bits - 1)
    scale = ((hi - lo) / nlev).clamp_min(1e-12)
    q = torch.clamp(torch.round((wf - lo) / scale), 0, nlev) * scale + lo
    return q.reshape(out_f, in_f).to(w.dtype)


class TorchQuantEvaluator:
    def __init__(self, model: nn.Module, layers: Sequence[Tuple[str, nn.Module]], bits: Sequence[int],
                 weight_quant: Callable[[torch.Tensor, int], torch.Tensor],
                 score_fn: Callable[[], float], act_quant: bool = False,
                 act_calib_fn: Optional[Callable[[], None]] = None, cache: bool = True,
                 act_bits: Optional[int] = None):
        self.model = model
        self.names = [n for n, _ in layers]
        self.mods = [m for _, m in layers]
        self.L = len(self.mods)
        self.bits = list(bits)
        self.score_fn = score_fn
        self.float_w = [m.weight.data for m in self.mods]
        self.weight_quant = weight_quant
        self.use_cache = cache
        self.cache: Dict[Tuple[int, int], torch.Tensor] = {}
        if cache:  # memory-light mode (LLMs) quantises on the fly when a layer's bit-width changes
            for i, w in enumerate(self.float_w):
                for b in self.bits:
                    self.cache[(i, b)] = weight_quant(w, b)
        self.n_params = np.array([w.numel() for w in self.float_w], dtype=np.float64)
        self.current: Optional[np.ndarray] = None
        self.act_quant = act_quant
        self.act_bits = act_bits  # if set, activations use this fixed bit-width (e.g. W{2,4,8}A8)
        self.n_evals = 0
        self._hooks = []
        if act_quant:
            dev = self.float_w[0].device
            self.lo = torch.zeros(self.L, device=dev)
            self.hi = torch.ones(self.L, device=dev)
            self.nlev = torch.full((self.L,), 255.0, device=dev)
            self.act_on = False
            self._collect_ranges(act_calib_fn)
            for i, m in enumerate(self.mods):
                self._hooks.append(m.register_forward_pre_hook(self._make_qhook(i)))

    # ------------------------------------------------------------- activations
    def _collect_ranges(self, run_calib: Callable[[], None]) -> None:
        mins = [float("inf")] * self.L
        maxs = [float("-inf")] * self.L
        hs = []
        for i, m in enumerate(self.mods):
            def rec(mod, inp, _i=i):
                x = inp[0]
                mins[_i] = min(mins[_i], float(x.min()))
                maxs[_i] = max(maxs[_i], float(x.max()))
            hs.append(m.register_forward_pre_hook(rec))
        run_calib()
        for h in hs:
            h.remove()
        self.lo = torch.tensor(mins, device=self.lo.device)
        self.hi = torch.tensor([mx if mx > mn else mn + 1e-3 for mn, mx in zip(mins, maxs)], device=self.lo.device)

    def _refresh_act_tensors(self) -> None:
        dt = self.float_w[0].dtype
        self._lo_t = self.lo.to(dt)
        self._hi_t = self.hi.to(dt)
        self._scale_t = ((self.hi - self.lo) / self.nlev).clamp_min(1e-6).to(dt)

    def _make_qhook(self, i: int):
        def hook(mod, inp):
            if not self.act_on:
                return None
            x = inp[0]
            lo, s = self._lo_t[i], self._scale_t[i]  # precomputed, already in the activation dtype
            xq = torch.round((torch.clamp(x, lo, self._hi_t[i]) - lo) / s) * s + lo
            return (xq,) + tuple(inp[1:])
        return hook

    # ------------------------------------------------------------- configs
    def set_config(self, cfg: Optional[np.ndarray]) -> None:
        if cfg is None:
            for m, w in zip(self.mods, self.float_w):
                m.weight.data = w
            self.current = None
            if self.act_quant:
                self.act_on = False
            return
        cfg = np.asarray(cfg, dtype=np.int64)
        for i in range(self.L):
            b = int(cfg[i])
            if self.current is None or int(self.current[i]) != b:
                self.mods[i].weight.data = self.qweight(i, b)
        if self.act_quant:
            ab = [self.act_bits or int(b) for b in cfg]
            self.nlev = torch.tensor([float(2 ** a - 1) for a in ab], device=self.lo.device)
            self._refresh_act_tensors()
            self.act_on = True
        self.current = cfg.copy()

    @torch.no_grad()
    def qweight(self, i: int, b: int) -> torch.Tensor:
        if self.use_cache:
            return self.cache[(i, b)]
        return self.weight_quant(self.float_w[i], b)

    @torch.no_grad()
    def score(self, cfg: Optional[np.ndarray]) -> float:
        self.set_config(cfg)
        self.n_evals += 1
        v = float(self.score_fn())
        if self.n_evals % 10 == 0 and torch.backends.mps.is_available():
            torch.mps.empty_cache()  # the MPS caching allocator otherwise grows to many GB over hundreds of evals
        return v

    def probe(self, top: int) -> Tuple[float, np.ndarray]:
        """Δ[i, j] = score(all top) - score(layer i at bits[j], rest top)."""
        base = np.full(self.L, top, dtype=np.int64)
        s0 = self.score(base)
        delta = np.zeros((self.L, len(self.bits)))
        for i in range(self.L):
            for j, b in enumerate(self.bits):
                if b == top:
                    continue
                c = base.copy()
                c[i] = b
                delta[i, j] = s0 - self.score(c)
        return s0, delta

    def weight_qerr(self) -> np.ndarray:
        out = np.zeros((self.L, len(self.bits)))
        for i, w in enumerate(self.float_w):
            wf = w.float()
            for j, b in enumerate(self.bits):
                out[i, j] = float(((self.qweight(i, b).float() - wf) ** 2).sum())
        return out
