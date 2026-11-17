#!/usr/bin/env bash
# ABC-Q extensions: ImageNet-pretrained CNNs + LLM weight-only quantisation (PyTorch, Apple GPU).
# Runs strictly one job at a time (16 GB laptop). Every job is resumable (skips finished runs).
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5
mkdir -p results_ext/logs
SEEDS="0 1 2"
METHODS="ga cmaes abc_v2 probe_mckp fisher_mckp"

llm() {  # $1 model, $2 number of quantisable layers L  (E = 2L+1 probes + 200 search evaluations)
  local E=$(( 2 * $2 + 201 ))
  $PY -u ext_run.py --task llm --model $1 --budgets 0.75 --seeds $SEEDS --methods $METHODS --evals $E \
    >> results_ext/logs/llm_$1.log 2>&1
}
img() {  # $1 model, $2 BOPs budget (W{2,4,8}A8, fraction of W8A8)
  $PY -u ext_run.py --task imagenet --model $1 --budgets $2 --seeds $SEEDS --methods $METHODS --evals 300 \
    >> results_ext/logs/imagenet_$1.log 2>&1
}

llm opt125m 72
img mobilenetv2 0.6
llm qwen0.5b 168
img resnet18 0.5
llm llama3.2-1b 112
llm opt350m 144
echo "EXT ALL DONE" >> results_ext/logs/pipeline.log
