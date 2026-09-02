#!/usr/bin/env bash
# Run the full TimeCAP LLM experiment on SiliconFlow models.
# Usage:
#   export SILICONFLOW_API_KEY=sk-...          # your key stays in YOUR shell; not stored here
#   bash run_siliconflow.sh                    # full run (contextualize + predicts) for all MODELS
#   bash run_siliconflow.sh --smoke            # tiny 3-call smoke per model first (verify id + key)
#
# Resumable: safe to Ctrl-C and re-run (a manifest skips finished work).
# If a model id is wrong, edit MODELS below (confirm exact ids from /me/models or:
#   curl https://api.siliconflow.cn/v1/models -H "Authorization: Bearer $SILICONFLOW_API_KEY" | python3 -m json.tool)
set -uo pipefail
: "${SILICONFLOW_API_KEY:?Set it first:  export SILICONFLOW_API_KEY=sk-...}"

cd "$(dirname "$0")"
PY=/home/hanlinux/venvs/timecap/bin/python
CH=siliconflow
WORKERS=4          # conservative for a third-party gateway; raise if no rate-limit errors
DATASETS="weather_ny weather_sf weather_hs finance_sp500 finance_nikkei healthcare_mortality healthcare_positive"

# >>> Confirm these exact ids against your /me/models and edit if needed <<<
MODELS=(
  "Qwen/Qwen3.6-27B"
  "moonshotai/Kimi-K2.6"
  "deepseek-ai/DeepSeek-V4-Pro"
)

SMOKE=""; [ "${1:-}" = "--smoke" ] && SMOKE="--limit 3"

for MODEL in "${MODELS[@]}"; do
  TAG=$(echo "$MODEL" | tr '/' '_')
  OUT="pipeline_out/$TAG"
  echo "############################ MODEL: $MODEL  ->  $OUT ############################"

  echo "===== PHASE 1: contextualize ====="
  for ds in $DATASETS; do
    echo "--- contextualize $ds ---"
    $PY -m pipeline.run_llm --task contextualize --channel $CH --model "$MODEL" \
        --dataset "$ds" --out-root "$OUT" --workers $WORKERS $SMOKE 2>&1 | grep -E "run stats|ABORT|token summary"
  done

  echo "===== PHASE 2: predictions ====="
  for ds in $DATASETS; do
    domain=${ds%%_*}
    for task in predict_time predict_text predict_in_context; do
      echo "--- $task $ds ---"
      $PY -m pipeline.run_llm --task "$task" --channel $CH --model "$MODEL" \
          --dataset "$ds" --out-root "$OUT" --summary-dir "$OUT/$domain/gpt_summary" \
          --workers $WORKERS $SMOKE 2>&1 | grep -E "run stats|ABORT"
    done
  done

  echo "===== COST ($MODEL) ====="; $PY -m pipeline.report --channel $CH --model "$MODEL"
  echo "===== METRICS ($MODEL) ====="; $PY -m pipeline.score --out-root "$OUT"
done
echo "############################ ALL MODELS DONE ############################"
