#!/usr/bin/env bash
# QAT with 3 configurations per method (search seeds 0,1,2; uniform W4A4: 3 fine-tuning seeds), 3 epochs,
# on all four CIFAR tasks at B=0.25. Resumable; results -> results_v2/<bb>/<ds>/qat_B250_multi.json
set -u
cd "$(dirname "$0")"
PY=.venv-tf/bin/python
export TF_CPP_MIN_LOG_LEVEL=2
L=results_v2/logs
for job in "resnet20 cifar10 gpu" "resnet20 cifar100 gpu" "mobilenetv2 cifar10 cpu" "mobilenetv2 cifar100 cpu"; do
  set -- $job
  $PY -u v2_qat.py --backbone $1 --dataset $2 --budget 0.25 --epochs 3 --device $3 --seeds 0 1 2 \
      >> $L/qat_multi_$1_$2.log 2>&1
done
echo "QAT MULTI DONE" >> $L/pipeline.log
