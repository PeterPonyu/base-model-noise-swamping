#!/bin/bash
# Depth-contrast extension chain (2026-07-15): 3 local 50%-depth RG cells via the reviewed
# run_merging_width.sh (explicit LAYER override, logged loudly by the driver by design).
# Predictions frozen in docs/plans/PREDICTIONS-GAIN-WAVE-2026-07-15.md ADDENDUM before
# launch. Kill only by PID from engine/chain_gainwave2_20260715.pid.
set -u
cd "$(dirname "$0")/.." || exit 2
PIDFILE=engine/chain_gainwave2_20260715.pid
LOG=engine/chain_gainwave2_20260715.log
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }

if [ -f "$PIDFILE" ]; then
  oldpid=$(cat "$PIDFILE" 2>/dev/null)
  if [ -n "$oldpid" ] && kill -0 "$oldpid" 2>/dev/null; then
    echo "REFUSE: chain already running (pid $oldpid)" >&2
    exit 7
  fi
fi
echo $$ > "$PIDFILE"
log "================ GAINWAVE2 (depth-contrast) START pid=$$ ================"

run_cell(){ # dir tag layer
  local d="$1" t="$2" L="$3"
  local table="results/merging/RG_operating_curve_table_${t}_L${L}.json"
  if [ -f "$table" ]; then
    log "SKIP $t L$L: table exists"
    return 0
  fi
  log "RUN $t L$L ($d)"
  MODEL_DIR="$d" MODEL_TAG="$t" LAYER="$L" RG_GROUP_SIZES=2,3,5,10,20 BUDGET_MIN=85 ./run_merging_width.sh >> "$LOG" 2>&1
  local rc=$?
  log "DONE $t L$L rc=$rc"
  return $rc
}

fails=0
run_cell data/models/Phi-3.5-mini phi35   16 || fails=$((fails+1))
run_cell data/models/Qwen2.5-3B   qwen3b  18 || fails=$((fails+1))
run_cell data/models/gemma-2-2b   gemma2b 13 || fails=$((fails+1))

log "================ GAINWAVE2 CHAIN END fails=$fails ================"
touch "engine/chain_gainwave2_20260715.done"
exit "$fails"
