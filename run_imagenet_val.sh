#!/usr/bin/env bash
# ImageNet experiments on the original ILSVRC-2012 validation set (512 calibration / 49,488 test images).
# Needs accepted access to ILSVRC/imagenet-1k on Hugging Face and ~15 GB free disk. Resumable.
set -u
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
mkdir -p results_ext/logs
L=results_ext/logs
$PY -u ext_download_imagenet.py >> $L/imagenet_val_download.log 2>&1 || { echo "download failed, see $L/imagenet_val_download.log"; exit 1; }
M="ga cmaes abc_v2 probe_mckp fisher_mckp"
S="0 1 2 3 4"
$PY -u ext_run.py --task imagenet --imagenet_set val --model mobilenetv2 --budgets 0.6 --seeds $S --methods $M --evals 300 >> $L/imagenet_val_mobilenetv2.log 2>&1
$PY -u ext_run.py --task imagenet --imagenet_set val --model resnet18 --budgets 0.5 --seeds $S --methods $M --evals 300 >> $L/imagenet_val_resnet18.log 2>&1
# one-shot budget sweep (deterministic, cheap)
$PY -u ext_run.py --task imagenet --imagenet_set val --model mobilenetv2 --budgets 0.7 0.8 0.9 --methods probe_mckp fisher_mckp >> $L/imagenet_val_mobilenetv2.log 2>&1
$PY -u ext_run.py --task imagenet --imagenet_set val --model resnet18 --budgets 0.4 0.6 0.7 --methods probe_mckp fisher_mckp >> $L/imagenet_val_resnet18.log 2>&1
echo "IMAGENET VAL DONE" >> $L/pipeline.log
