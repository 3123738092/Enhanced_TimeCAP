#!/usr/bin/env bash
# Run DeepSeek-R1 (thinking) full at workers=5, resumable. Reads
# SILICONFLOW_API_KEY from env (NOT stored here). Detached via setsid.
set -uo pipefail
: "${SILICONFLOW_API_KEY:?SILICONFLOW_API_KEY must be set}"
cd /mnt/d/桌面/TimeCAP
PY=/home/hanlinux/venvs/timecap/bin/python
CH=siliconflow; M="deepseek-ai/DeepSeek-R1"; OUT="pipeline_out/deepseek-ai_DeepSeek-R1"
C="--channel $CH --model $M --out-root $OUT --workers 5 --max-retries 10 --max-failures 500"
DATASETS="weather_ny weather_sf weather_hs finance_sp500 finance_nikkei healthcare_mortality healthcare_positive"
echo "### R1 workers=5 START $(date -u) ###"
echo "== PHASE 1 contextualize =="
for ds in $DATASETS; do
  echo "-- ctx $ds --"; $PY -m pipeline.run_llm --task contextualize --dataset "$ds" $C 2>&1 | grep -E "run stats|ABORT"
done
echo "== PHASE 2 predictions =="
for ds in $DATASETS; do
  domain=${ds%%_*}
  for task in predict_time predict_text predict_in_context; do
    echo "-- $task $ds --"; $PY -m pipeline.run_llm --task "$task" --dataset "$ds" --summary-dir "$OUT/$domain/gpt_summary" $C 2>&1 | grep -E "run stats|ABORT"
  done
done
echo "== R1 COST =="; $PY -m pipeline.report --channel $CH --model "$M"
echo "== R1 METRICS =="; $PY -m pipeline.score --out-root "$OUT"
echo "### R1 DONE $(date -u) ###"
