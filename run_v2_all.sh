#!/usr/bin/env bash
# ABC-Q v2: full reproduction pipeline (train float -> search -> ablation -> QAT -> report).
# Two chains run in parallel on Apple Silicon: ResNet-20 on the Metal GPU, MobileNetV2 on the CPU
# (depthwise convolutions are much faster on the CPU with tensorflow-metal). Every step is resumable.
set -u
cd "$(dirname "$0")"
PY=.venv-tf/bin/python  # TensorFlow env (numpy<2); PyTorch experiments use .venv
export TF_CPP_MIN_LOG_LEVEL=2
mkdir -p results_v2/logs
E=300
SEEDS10="0 1 2 3 4 5 6 7 8 9"
SEEDS5="0 1 2 3 4"
MAIN="random ga cmaes abc_v1 abc_v2 probe_mckp fisher_mckp"
ABL="abc_v2 abc_v2_noscreen abc_v2_nolearn abc_v2_noknap abc_v2_noshare abc_v2_hc"

chain() {  # $1 backbone, $2 device, $3 intra-op threads
  local bb=$1 dev=$2 th=$3
  export TF_NUM_INTRAOP_THREADS=$th TF_NUM_INTEROP_THREADS=2
  for ds in cifar10 cifar100; do
    [ -f results_v2/$bb/$ds/float.weights.h5 ] || \
      $PY -u v2_train_float.py --backbone $bb --dataset $ds --epochs 40 --device $dev \
        > results_v2/logs/train_${bb}_${ds}.log 2>&1
  done
  for ds in cifar10 cifar100; do
    $PY -u v2_run.py --backbone $bb --dataset $ds --budgets 0.25 --seeds $SEEDS10 --methods $MAIN \
      --evals $E --device $dev >> results_v2/logs/run_${bb}_${ds}.log 2>&1
  done
  for ds in cifar10 cifar100; do
    $PY -u v2_run.py --backbone $bb --dataset $ds --budgets 0.35 --seeds $SEEDS5 --methods $MAIN \
      --evals $E --device $dev >> results_v2/logs/run_${bb}_${ds}.log 2>&1
  done
  local abl_ds=cifar100; [ "$bb" = mobilenetv2 ] && abl_ds=cifar10
  $PY -u v2_run.py --backbone $bb --dataset $abl_ds --budgets 0.25 --seeds $SEEDS10 --methods $ABL \
    --evals $E --device $dev >> results_v2/logs/run_${bb}_${abl_ds}.log 2>&1
  for ds in cifar10 cifar100; do
    $PY -u v2_qat.py --backbone $bb --dataset $ds --budget 0.25 --epochs 2 --device $dev \
      >> results_v2/logs/qat_${bb}_${ds}.log 2>&1
  done
}

chain resnet20 gpu 3 &
chain mobilenetv2 cpu 6 &
wait
$PY v2_report.py > results_v2/logs/report.log 2>&1
echo "ALL DONE"
