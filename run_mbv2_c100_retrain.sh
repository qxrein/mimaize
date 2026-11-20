#!/usr/bin/env bash
# Retrain MobileNetV2/CIFAR-100 for 120 epochs (was 40) and rerun all of its CIFAR experiments.
set -u
cd "$(dirname "$0")"
PY=.venv-tf/bin/python
export TF_CPP_MIN_LOG_LEVEL=2
D=results_v2/logs
$PY -u v2_train_float.py --backbone mobilenetv2 --dataset cifar100 --epochs 120 --device cpu > $D/train_mobilenetv2_cifar100_120ep.log 2>&1
$PY -u v2_run.py --backbone mobilenetv2 --dataset cifar100 --budgets 0.25 0.35 --seeds 0 1 2 3 4 5 6 7 8 9 \
    --methods random ga cmaes abc_v1 abc_v2 probe_mckp fisher_mckp --evals 300 --device cpu > $D/run_mobilenetv2_cifar100_120ep.log 2>&1
$PY -u v2_qat.py --backbone mobilenetv2 --dataset cifar100 --budget 0.25 --epochs 2 --device cpu > $D/qat_mobilenetv2_cifar100_120ep.log 2>&1
echo "MBV2_C100_RETRAIN DONE" >> $D/pipeline.log
