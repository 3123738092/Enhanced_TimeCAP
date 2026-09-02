#!/usr/bin/env bash
# TimeCAP — OpenAI 能力梯度实验:gpt-5-mini -> gpt-4o -> gpt-5(普通同步调用)
# 安全:本脚本只引用环境变量 $OPENAI_API_KEY,绝不含 key 值。
# 用法(在**已 export OPENAI_API_KEY 的 shell** 里,后台跑):
#   setsid bash run_openai_gradient.sh > openai_gradient.log 2>&1 &
# 冒烟(近乎免费,每格只跑前 3 条,单模型):
#   LIMIT=3 MODELS_ONLY=gpt-5-mini bash run_openai_gradient.sh 2>&1 | tail -40
# 断点续跑:manifest 记录已完成的调用,重跑同一命令会跳过已完成部分。
set -u
cd /mnt/d/桌面/TimeCAP || { echo "cd fail"; exit 1; }
PY=/home/hanlinux/venvs/timecap/bin/python
[ -x "$PY" ] || PY=python3

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "ERROR: OPENAI_API_KEY 未设置。请先在同一 shell 执行:export OPENAI_API_KEY=sk-..."
  exit 1
fi

MODELS=(gpt-5-mini gpt-4o gpt-5)                 # 由便宜到贵:先冒烟再上大的
[ -n "${MODELS_ONLY:-}" ] && read -ra MODELS <<< "${MODELS_ONLY}"   # 可传空格分隔的子集
ALL=(weather_ny weather_sf weather_hs finance_sp500 finance_nikkei healthcare_mortality healthcare_positive)
WORKERS="${WORKERS:-4}"
LIMIT_ARG=""
[ -n "${LIMIT:-}" ] && LIMIT_ARG="--limit ${LIMIT}"  # 冒烟:LIMIT=3 只跑每格前 3 条

domain_of () { case "$1" in weather_*) echo weather;; finance_*) echo finance;; healthcare_*) echo healthcare;; esac; }

run_cell () {  # model task dataset [extra flags...]
  local M="$1" T="$2" D="$3"; shift 3
  echo "---- $M | $T | $D  $(date +%H:%M:%S) ----"
  "$PY" -m pipeline.run_llm --channel openai --model "$M" --task "$T" --dataset "$D" \
     --out-root "pipeline_out/$M" --workers "$WORKERS" --max-retries 8 --max-failures 300 $LIMIT_ARG "$@" 2>&1
}

for M in "${MODELS[@]}"; do
  echo "######################## MODEL $M  start $(date) ########################"
  # 1) contextualize —— 生成该模型自己的摘要(finance_nikkei 与 sp500 共享,自动跳过)
  for D in "${ALL[@]}"; do run_cell "$M" contextualize "$D"; done
  # 2) predict_time —— 直接读原始数字,不需摘要
  for D in "${ALL[@]}"; do run_cell "$M" predict_time "$D"; done
  # 3) predict_text + predict_in_context —— 用该模型自己的摘要
  for D in "${ALL[@]}"; do
    DOM=$(domain_of "$D"); SUM="pipeline_out/$M/$DOM/gpt_summary"
    run_cell "$M" predict_text       "$D" --summary-dir "$SUM"
    run_cell "$M" predict_in_context "$D" --summary-dir "$SUM"
  done
  echo "######################## MODEL $M  done  $(date) ########################"
done
echo "[ALL OPENAI GRADIENT DONE]"
