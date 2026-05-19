"""ABC-Q CIFAR-100 — ResNet-20. Implementation: ``abc_q_core``."""

import abc_q_core as _q

_q.BACKBONE = "resnet20"
_q.DATASET = "cifar100"
_q.NUM_CLASSES = 100
_q.N_SEEDS = 3
_q.SEEDS = [42, 123, 456]
_q.FULL_DATASET = True
_q.QUANTIZE_ACTIVATIONS = True
_q.REAL_BOPS = True

if __name__ == "__main__":
    _q.main()
