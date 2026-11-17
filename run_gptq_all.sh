#!/usr/bin/env bash
# ABC-Q + GPTQ: GPTQ-aware bit-allocation search and final sequential-GPTQ evaluation for the four LLMs.
# Works on CUDA (RTX 3050 6 GB tested target), Apple MPS or CPU. Resumable: finished runs are skipped.
#
#   ./run_gptq_all.sh                 # all four models
#   ./run_gptq_all.sh opt125m         # one model
set -u
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
export TOKENIZERS_PARALLELISM=false PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5
mkdir -p results_ext/logs
MODELS=${*:-"opt125m opt350m qwen0.5b llama3.2-1b"}
declare -A NL=([opt125m]=72 [opt350m]=144 [qwen0.5b]=168 [llama3.2-1b]=112)

for m in $MODELS; do
  E=$(( 2 * ${NL[$m]} + 201 ))
  echo "=== $m: GPTQ-aware search (E=$E) ==="
  $PY -u ext_run.py --task llm --quant gptq --model $m --budgets 0.75 --seeds 0 1 2 \
      --methods abc_v2 probe_mckp fisher_mckp ga cmaes --evals $E >> results_ext/logs/llm_gptq_$m.log 2>&1
  echo "=== $m: final sequential GPTQ + WikiText-2 (ctx 2048, full test set) ==="
  $PY -u ext_gptq.py --model $m --alloc_src llm_gptq --seqlen 2048 \
      --allocs fp16 rtn3 rtn4 gptq3 gptq4 probe_mckp fisher_mckp ga cmaes abc_v2 \
      >> results_ext/logs/gptq_final_$m.log 2>&1
done
echo "GPTQ ALL DONE" >> results_ext/logs/pipeline.log
