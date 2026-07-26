#!/bin/bash
# GPT-2-XL exploratory pair (2026-07-15): 2 local RG cells via run_merging_width.sh.
# Predictions doc ADDENDUM 2 frozen before launch. Kill only by PID from
# engine/chain_gainwave3_20260715.pid.
set -u
cd "$(dirname "$0")/.." || exit 2
PIDFILE=engine/chain_gainwave3_20260715.pid
LOG=engine/chain_gainwave3_20260715.log
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
if [ -f "$PIDFILE" ]; then
  oldpid=$(cat "$PIDFILE" 2>/dev/null)
  if [ -n "$oldpid" ] && kill -0 "$oldpid" 2>/dev/null; then
    echo "REFUSE: chain already running (pid $oldpid)" >&2; exit 7
  fi
fi
echo $$ > "$PIDFILE"
log "================ GAINWAVE3 (gpt2-xl pair) START pid=$$ ================"
run_cell(){ local L="$1"
  local table="results/merging/RG_operating_curve_table_gpt2xl_L${L}.json"
  [ -f "$table" ] && { log "SKIP gpt2xl L$L: table exists"; return 0; }
  log "RUN gpt2xl L$L"
  MODEL_DIR=data/models/gpt2-xl MODEL_TAG=gpt2xl LAYER="$L" RG_GROUP_SIZES=2,3,5,10,20 BUDGET_MIN=85 ./run_merging_width.sh >> "$LOG" 2>&1
  local rc=$?; log "DONE gpt2xl L$L rc=$rc"; return $rc
}
fails=0
run_cell 36 || fails=$((fails+1))
run_cell 24 || fails=$((fails+1))
log "================ GAINWAVE3 CHAIN END fails=$fails ================"
touch "engine/chain_gainwave3_20260715.done"
exit "$fails"
