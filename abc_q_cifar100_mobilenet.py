"""ABC-Q CIFAR-100 — MobileNetV2. Implementation: ``abc_q_core``."""

import abc_q_core as _q

_q.BACKBONE = "mobilenetv2"
_q.DATASET = "cifar100"
_q.NUM_CLASSES = 100
_q.N_SEEDS = 10
_q.SEEDS = [42, 123, 456, 789, 1024, 2048, 3141, 2718, 9001, 4242]
_q.FULL_DATASET = True
_q.QUANTIZE_ACTIVATIONS = True
_q.REAL_BOPS = True

if __name__ == "__main__":
    _q.main()
