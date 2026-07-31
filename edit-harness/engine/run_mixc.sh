#!/bin/bash
# run_mixc.sh — 运行完整的 MIX_C (33 cells)
set -u
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$H/.." || exit 2
PY=${PY:-/home/zeyufu/miniconda3/envs/dl/bin/python3}
PIDFILE=engine/run_mixc.pid
LOG=engine/run_mixc.log
mkdir -p engine

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; then
  echo "REFUSE: already running (pid $(cat "$PIDFILE"))" >&2; exit 7
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

log "======== RUN MIX_C START pid=$$ ========"
log "LID-OPEN REMINDER: keep lid open (nvidia_uvm wedge)."

ENVP="env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1"

log "MIX_C start (33 cells + p2 structural file)"
# setsid: the cell runs in its OWN session/process group, so a SIGTERM aimed at
# this wrapper (or the monitor supervising it) does NOT propagate to the cell —
# the 07-29 4x-SIGTERM incident: python ran in a `| tee` pipeline inside the
# wrapper's process group and died with it, losing ~29 GPU-h mid-stream.
# Side fix: rc now comes from `wait` (real python rc), not from `tee`.
CHILD_PIDFILE=engine/run_mixc.child.pid
setsid $ENVP $PY -m experiments.frame_a.run_stream --run --real \
    --mixes MIX_C --model_dir data/models/Llama-3.2-1B >> "$LOG" 2>&1 &
CHILD=$!
echo "$CHILD" > "$CHILD_PIDFILE"
trap 'log "WRAPPER caught TERM/INT — setsid cell pid '"$CHILD"' stays alive; relaunch resumes landed cells"; exit 143' TERM INT
wait "$CHILD"
rc=$?
trap - TERM INT
rm -f "$CHILD_PIDFILE"
if [ $rc -eq 0 ]; then
  # 验证细胞数量
  count=$(ls results/frame_a/cells/cell_llama-3.2-1b_real_MIX_C_*.json 2>/dev/null | wc -l)
  log "MIX_C done: $count/33 cells present"
  if [ "$count" -eq 33 ]; then
    log "VERIFIED: MIX_C complete"
  else
    log "WARNING: expected 33 cells, found $count"
  fi
else
  log "FAILED: MIX_C rc=$rc"
fi

log "======== RUN MIX_C END rc=$rc ========"
exit $rc
