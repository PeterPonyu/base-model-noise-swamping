#!/bin/bash
# E1 editor-generality + dataset-generality wave (2026-07-16), local 5090, serial.
# Cells per PREREG-FED-EDITORS-2026-07-16.md: {llama1b L12, qwen15b L21} x {memit, alpha}
# on cf, + zsRE generality cells (rome at both cells). Driver carries idle gate + the
# reviewer-mandated ΔW-fidelity GPU smoke gate per (model,editor,dataset) tag.
# Kill only by PID from engine/chain_e1_editors_20260716.pid. Idempotent via the
# driver's Phase-0a.3 skip-if-valid.
set -u
cd "$(dirname "$0")/.." || exit 2
PIDFILE=engine/chain_e1_editors_20260716.pid
LOG=engine/chain_e1_editors_20260716.log
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }

if [ -f "$PIDFILE" ]; then
  oldpid=$(cat "$PIDFILE" 2>/dev/null)
  if [ -n "$oldpid" ] && kill -0 "$oldpid" 2>/dev/null; then
    echo "REFUSE: chain already running (pid $oldpid)" >&2
    exit 7
  fi
fi
echo $$ > "$PIDFILE"
log "================ E1 EDITORS CHAIN START pid=$$ ================"

run_cell(){ # dir tag editor dataset
  local d="$1" t="$2" e="$3" ds="$4"
  log "RUN $t $e $ds"
  MODEL_DIR="$d" MODEL_TAG="$t" EDITOR="$e" DATASET="$ds" \
    RG_GROUP_SIZES=2,3,5,10,20 BUDGET_MIN=110 ./run_merging_editors.sh >> "$LOG" 2>&1
  local rc=$?
  log "DONE $t $e $ds rc=$rc"
  return $rc
}

fails=0
# editor generality (cf primary, prereg cells)
run_cell data/models/Llama-3.2-1B llama1b memit cf   || fails=$((fails+1))
run_cell data/models/Llama-3.2-1B llama1b alpha cf   || fails=$((fails+1))
run_cell data/models/Qwen2.5-1.5B qwen15b memit cf   || fails=$((fails+1))
run_cell data/models/Qwen2.5-1.5B qwen15b alpha cf   || fails=$((fails+1))
# dataset generality (zsRE, rome anchor editor)
run_cell data/models/Llama-3.2-1B llama1b rome zsre  || fails=$((fails+1))
run_cell data/models/Qwen2.5-1.5B qwen15b rome zsre  || fails=$((fails+1))

log "================ E1 EDITORS CHAIN END fails=$fails ================"
touch "engine/chain_e1_editors_20260716.done"
exit "$fails"
