#!/usr/bin/env bash
# Self-contained chain: wait for the Qwen run to finish, then run DeepSeek-V3
# (non-thinking) full, then DeepSeek-R1 (thinking) full. Reads SILICONFLOW_API_KEY
# from the environment (NOT stored here). Launched detached (setsid) so it keeps
# going even if the Claude Code session closes — only needs the computer/WSL on.
set -uo pipefail
: "${SILICONFLOW_API_KEY:?SILICONFLOW_API_KEY must be set in the environment}"

cd /mnt/d/桌面/TimeCAP
PY=/home/hanlinux/venvs/timecap/bin/python
CH=siliconflow
DATASETS="weather_ny weather_sf weather_hs finance_sp500 finance_nikkei healthcare_mortality healthcare_positive"
QWEN_OUT="/tmp/claude-1000/-mnt-c-Users-31237/e9ff8397-fb69-4766-ab5b-4c054efc4fd9/tasks/bg1bma9t2.output"

qwen_ok () {  # count successful Qwen calls in the logs
  $PY - <<'PY'
import glob,json
n=0
for p in glob.glob("llm_runs/*.jsonl"):
    if p.endswith(".manifest.jsonl"): continue
    for l in open(p,encoding="utf-8"):
        l=l.strip()
        if l and '"Qwen/Qwen2.5-72B-Instruct"' in l and '"ok": true' in l: n+=1
print(n)
PY
}

echo "[chain] waiting for Qwen to finish ..."
WAITED=0
while true; do
  grep -q "DONE" "$QWEN_OUT" 2>/dev/null && { echo "[chain] Qwen DONE marker seen"; break; }
  [ "$(qwen_ok)" -ge 13030 ] && { echo "[chain] Qwen ~complete by count"; break; }
  [ "$WAITED" -ge 43200 ] && { echo "[chain] waited 12h, proceeding"; break; }  # safety cap
  sleep 120; WAITED=$((WAITED+120))
done
sleep 30

for MODEL in "deepseek-ai/DeepSeek-V3" "deepseek-ai/DeepSeek-R1"; do
  TAG=$(echo "$MODEL" | tr '/' '_'); OUT="pipeline_out/$TAG"
  COMMON="--channel $CH --model $MODEL --out-root $OUT --workers 3 --max-retries 10 --max-failures 500"
  echo "############ $(date -u +%H:%M) START $MODEL ############"
  echo "== PHASE 1 contextualize =="
  for ds in $DATASETS; do
    echo "-- ctx $ds --"; $PY -m pipeline.run_llm --task contextualize --dataset "$ds" $COMMON 2>&1 | grep -E "run stats|ABORT"
  done
  echo "== PHASE 2 predictions =="
  for ds in $DATASETS; do
    domain=${ds%%_*}
    for task in predict_time predict_text predict_in_context; do
      echo "-- $task $ds --"; $PY -m pipeline.run_llm --task "$task" --dataset "$ds" --summary-dir "$OUT/$domain/gpt_summary" $COMMON 2>&1 | grep -E "run stats|ABORT"
    done
  done
  echo "== COST $MODEL =="; $PY -m pipeline.report --channel $CH --model "$MODEL"
  echo "== METRICS $MODEL =="; $PY -m pipeline.score --out-root "$OUT"
done
echo "############ CHAIN DONE ($(date -u)) ############"
