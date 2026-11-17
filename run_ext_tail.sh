#!/usr/bin/env bash
# Remaining extension jobs (sequential): Qwen2.5-0.5B then Llama-3.2-1B.
set -u
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5
M="ga cmaes abc_v2 probe_mckp fisher_mckp"
.venv/bin/python -u ext_run.py --task llm --model qwen0.5b --budgets 0.75 --seeds 0 1 2 --methods $M --evals 537 >> results_ext/logs/llm_qwen0.5b.log 2>&1
.venv/bin/python -u ext_run.py --task llm --model llama3.2-1b --budgets 0.75 --seeds 0 1 2 --methods $M --evals 425 >> results_ext/logs/llm_llama3.2-1b.log 2>&1
echo "TAIL DONE" >> results_ext/logs/pipeline.log
