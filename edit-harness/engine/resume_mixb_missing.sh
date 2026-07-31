#!/bin/bash
# resume_mixb_missing.sh — 补完 MIX_B 缺失的 2 个 s2 细胞
set -u
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$H/.." || exit 2
PY=${PY:-/home/zeyufu/miniconda3/envs/dl/bin/python3}
PIDFILE=engine/resume_mixb_missing.pid
LOG=engine/resume_mixb_missing.log
mkdir -p engine

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; then
  echo "REFUSE: already running (pid $(cat "$PIDFILE"))" >&2; exit 7
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

log "======== RESUME MIX_B MISSING START pid=$$ ========"

ENVP="env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1"

# 只运行 ft_merge 和 random 的 seed 2
log "Running ft_merge and random for seed 2 only"
# setsid: cell in its OWN session/pgroup — wrapper SIGTERM cannot kill it
# (07-29 4x-SIGTERM incident pattern; see run_mixc.sh). rc from `wait`, not tee.
CHILD_PIDFILE=engine/resume_mixb_missing.child.pid
setsid $ENVP $PY -m experiments.frame_a.run_stream --run --real \
    --mixes MIX_B --policies ft_merge,random \
    --model_dir data/models/Llama-3.2-1B >> "$LOG" 2>&1 &
CHILD=$!
echo "$CHILD" > "$CHILD_PIDFILE"
trap 'log "WRAPPER caught TERM/INT — setsid cell pid '"$CHILD"' stays alive; relaunch resumes"; exit 143' TERM INT
wait "$CHILD"
rc=$?
trap - TERM INT
rm -f "$CHILD_PIDFILE"
if [ $rc -eq 0 ]; then
  log "SUCCESS: MIX_B missing cells completed"
  # 验证文件存在
  if [ -f results/frame_a/cells/cell_llama-3.2-1b_real_MIX_B_ft_merge_s2.json ] && \
     [ -f results/frame_a/cells/cell_llama-3.2-1b_real_MIX_B_random_s2.json ]; then
    log "VERIFIED: both missing cells now present"
  else
    log "WARNING: script rc=0 but cells not found"
  fi
else
  log "FAILED: rc=$rc"
fi

log "======== RESUME MIX_B MISSING END rc=$rc ========"
exit $rc
