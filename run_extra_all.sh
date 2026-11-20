#!/usr/bin/env bash
# Downstream evaluation of the GPTQ models: C4 perplexity + six zero-shot tasks (ARC-e/c, HellaSwag, PIQA,
# WinoGrande, BoolQ). Uses the allocations already found by the GPTQ-aware search (results_ext/llm_gptq).
# Resumable; results -> results_ext/gptq/<model>_extra.json
#
#   ./run_extra_all.sh               # all four models
#   ./run_extra_all.sh llama3.2-1b   # one model
set -u
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
export TOKENIZERS_PARALLELISM=false
mkdir -p results_ext/logs
MODELS=${*:-"opt125m opt350m qwen0.5b llama3.2-1b"}
for m in $MODELS; do
  echo "=== $m: GPTQ + C4 + zero-shot ==="
  $PY -u ext_gptq.py --model $m --alloc_src llm_gptq --seqlen 2048 --extra \
      --allocs fp16 gptq3 gptq4 probe_mckp abc_v2 >> results_ext/logs/extra_$m.log 2>&1
done
echo "EXTRA ALL DONE" >> results_ext/logs/pipeline.log
