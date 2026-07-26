#!/bin/bash
# Serial gain-wave chain (2026-07-15): 4 local RG cells on the 5090 via the reviewed
# run_merging_width.sh (which carries its own preflight + GPU idle gate util<25&&mem<1500
# x3 + refuse-guard). Predictions frozen in docs/plans/PREDICTIONS-GAIN-WAVE-2026-07-15.md
# BEFORE this launch. Kill only by PID from engine/chain_gainwave_20260715.pid (kill -0,
# never pgrep). Idempotent: skips a cell whose canonical table already exists.
set -u
cd "$(dirname "$0")/.." || exit 2
PIDFILE=engine/chain_gainwave_20260715.pid
LOG=engine/chain_gainwave_20260715.log
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }

# double-launch refuse guard (ops lesson 2026-07-15: pidfile-alive => refuse)
if [ -f "$PIDFILE" ]; then
  oldpid=$(cat "$PIDFILE" 2>/dev/null)
  if [ -n "$oldpid" ] && kill -0 "$oldpid" 2>/dev/null; then
    echo "REFUSE: chain already running (pid $oldpid)" >&2
    exit 7
  fi
fi
echo $$ > "$PIDFILE"
log "================ GAINWAVE CHAIN START pid=$$ ================"

run_cell(){ # dir tag
  local d="$1" t="$2"
  local table
  table=$(DRYRUN=1 MODEL_DIR="$d" MODEL_TAG="$t" RG_GROUP_SIZES=2,3,5,10,20 ./run_merging_width.sh 2>/dev/null | sed -n 's/^DRYRUN tag=.* -> //p')
  if [ -n "$table" ] && [ -f "$table" ]; then
    log "SKIP $t: table exists ($table)"
    return 0
  fi
  log "RUN $t ($d)"
  MODEL_DIR="$d" MODEL_TAG="$t" RG_GROUP_SIZES=2,3,5,10,20 BUDGET_MIN=85 ./run_merging_width.sh >> "$LOG" 2>&1
  local rc=$?
  log "DONE $t rc=$rc"
  return $rc
}

fails=0
run_cell data/models/gemma-2-2b   gemma2b || fails=$((fails+1))
run_cell data/models/Llama-3.2-3B llama3b || fails=$((fails+1))
run_cell data/models/Qwen2.5-3B   qwen3b  || fails=$((fails+1))
run_cell data/models/Phi-3.5-mini phi35   || fails=$((fails+1))

log "================ GAINWAVE CHAIN END fails=$fails ================"
touch "engine/chain_gainwave_20260715.done"
exit "$fails"
