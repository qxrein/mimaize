#!/usr/bin/env bash
# Two additional search seeds (3, 4) for every LLM, GPTQ-aware and round-to-nearest searches, then
# re-evaluate only the final GPTQ / downstream results whose selected configuration changed. Resumable.
set -u
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
export TOKENIZERS_PARALLELISM=false
mkdir -p results_ext/logs
declare -A NL=([opt125m]=72 [opt350m]=144 [qwen0.5b]=168 [llama3.2-1b]=112)
M="ga cmaes abc_v2"
for m in opt125m opt350m qwen0.5b llama3.2-1b; do
  E=$(( 2 * ${NL[$m]} + 201 ))
  echo "=== $m: seeds 3 4 (GPTQ-aware search) ==="
  $PY -u ext_run.py --task llm --quant gptq --model $m --budgets 0.75 --seeds 3 4 --methods $M --evals $E \
      >> results_ext/logs/llm_gptq_$m.log 2>&1
  echo "=== $m: seeds 3 4 (round-to-nearest search) ==="
  $PY -u ext_run.py --task llm --model $m --budgets 0.75 --seeds 3 4 --methods $M --evals $E \
      >> results_ext/logs/llm_$m.log 2>&1
done
$PY ext_refresh.py | tee -a results_ext/logs/refresh.log
for m in opt125m opt350m qwen0.5b llama3.2-1b; do
  echo "=== $m: re-evaluate changed selections ==="
  $PY -u ext_gptq.py --model $m --alloc_src llm_gptq --seqlen 2048 \
      --allocs fp16 rtn3 rtn4 gptq3 gptq4 probe_mckp fisher_mckp ga cmaes abc_v2 >> results_ext/logs/gptq_final_$m.log 2>&1
  $PY -u ext_gptq.py --model $m --alloc_src llm_gptq --seqlen 2048 --extra \
      --allocs fp16 gptq3 gptq4 probe_mckp abc_v2 >> results_ext/logs/extra_$m.log 2>&1
done
echo "SEEDS ALL DONE" >> results_ext/logs/pipeline.log
