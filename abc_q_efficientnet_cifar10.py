"""ABC-Q CIFAR-10 — EfficientNet-B0. Implementation: ``abc_q_core``."""

import abc_q_core as _q

_q.BACKBONE = "efficientnetb0"
_q.DATASET = "cifar10"
_q.NUM_CLASSES = 10
_q.N_SEEDS = 5
_q.SEEDS = [42, 123, 456, 789, 1024]
_q.FULL_DATASET = True
_q.QUANTIZE_ACTIVATIONS = True
_q.REAL_BOPS = True

if __name__ == "__main__":
    _q.main()
