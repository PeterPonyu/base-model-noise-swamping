#!/bin/bash
# run_frame_a_bc_real_20260721.sh — REAL MIX_B + MIX_C wave (replaces the quarantined
# synthetic-relabel cells). Serial: smoke gate -> MIX_B -> MIX_C. One GPU, setsid,
# systemd-inhibit, PID-by-file. Kill-by-PID only.
set -u
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$H" || exit 2
PY=${PY:-/home/zeyufu/miniconda3/envs/dl/bin/python3}
PIDFILE=engine/run_frame_a_bc_real.pid
LOG=engine/run_frame_a_bc_real.log
mkdir -p engine results/frame_a/cells
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; then
  echo "REFUSE: already running (pid $(cat "$PIDFILE"))" >&2; exit 7
fi
echo $$ > "$PIDFILE"
log "======== FRAME_A BC REAL START pid=$$ ========"
log "LID-OPEN REMINDER: keep lid open (nvidia_uvm wedge)."

ENVP="env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1"

# ---- step 0: fresh smoke gate (M4 convention; writes engine/SMOKE_PASS.ok) ----
log "SMOKE start"
if $ENVP $PY -m experiments.frame_a.run_stream --run --real --smoke \
    --model_dir data/models/Llama-3.2-1B >> "$LOG" 2>&1; then
  log "SMOKE PASS"
else
  log "ABORT: smoke failed rc=$?"; exit 4
fi

# ---- step 1: MIX_B (11 policies x 3 seeds = 33 cells; ~2-3 GPU-h) ----
log "MIX_B start"
if $ENVP $PY -m experiments.frame_a.run_stream --run --real \
    --mixes MIX_B --model_dir data/models/Llama-3.2-1B >> "$LOG" 2>&1; then
  log "MIX_B done"
else
  log "ABORT: MIX_B failed rc=$?"; exit 5
fi

# ---- step 2: MIX_C (33 cells + p2 structural file; ~2-3 GPU-h) ----
log "MIX_C start"
if $ENVP $PY -m experiments.frame_a.run_stream --run --real \
    --mixes MIX_C --model_dir data/models/Llama-3.2-1B >> "$LOG" 2>&1; then
  log "MIX_C done"
else
  log "ABORT: MIX_C failed rc=$?"; exit 6
fi

log "======== FRAME_A BC REAL COMPLETE ========"
echo "FRAME_A_BC_REAL_DONE" >> "$LOG"
exit 0
