#!/bin/bash
# monitor_and_chain.sh — 监控 MIX_B 恢复 -> 启动 MIX_C -> 启动后续链
set -u
cd "$(dirname "$0")/.."
LOG=engine/monitor_and_chain.log
PIDFILE=engine/monitor_and_chain.pid

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; then
  echo "REFUSE: monitor already running" >&2; exit 1
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

log "======== MONITOR AND CHAIN START ========"

# Stage 1: 等待 MIX_B 恢复完成
log "Stage 1: Waiting for MIX_B recovery to complete..."
while true; do
  if [ ! -f engine/resume_mixb_missing.pid ]; then
    log "MIX_B recovery pidfile gone, checking results..."
    break
  fi
  pid=$(cat engine/resume_mixb_missing.pid 2>/dev/null || echo "")
  if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
    log "MIX_B recovery process dead, checking results..."
    break
  fi
  log "MIX_B recovery still running (pid $pid)"
  sleep 60
done

# 验证 MIX_B 是否完整
if [ -f results/frame_a/cells/cell_llama-3.2-1b_real_MIX_B_ft_merge_s2.json ] && \
   [ -f results/frame_a/cells/cell_llama-3.2-1b_real_MIX_B_random_s2.json ]; then
  log "✓ MIX_B complete (33/33)"
else
  log "✗ MIX_B still incomplete, aborting chain"
  echo "MIX_B incomplete" > engine/CHAIN_ABORT.txt
  exit 1
fi

# Stage 2: GPU idle gate
log "Stage 2: Waiting for GPU idle..."
consec=0
while [ "$consec" -lt 3 ]; do
  line=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  util=$(echo "$line" | awk -F, '{gsub(/[^0-9]/,"",$1); print $1}')
  mem=$(echo "$line" | awk -F, '{gsub(/[^0-9]/,"",$2); print $2}')
  if [ -n "${util:-}" ] && [ -n "${mem:-}" ] && [ "$util" -lt 25 ] && [ "$mem" -lt 1500 ]; then
    consec=$((consec+1))
  else
    consec=0
  fi
  log "GPU idle check: util=${util:-NA} mem=${mem:-NA} consec=${consec}/3"
  [ "$consec" -lt 3 ] && sleep 30
done

# Stage 3: 启动 MIX_C
log "Stage 3: Launching MIX_C..."
nohup ./engine/run_mixc.sh > engine/run_mixc.nohup.log 2>&1 &
sleep 2
mixc_pid=$(cat engine/run_mixc.pid 2>/dev/null || echo "")
if [ -z "$mixc_pid" ] || ! kill -0 "$mixc_pid" 2>/dev/null; then
  log "✗ MIX_C failed to start"
  echo "MIX_C launch failed" > engine/CHAIN_ABORT.txt
  exit 2
fi
log "✓ MIX_C launched (pid $mixc_pid)"

# Stage 4: 等待 MIX_C 完成
log "Stage 4: Waiting for MIX_C to complete..."
while kill -0 "$mixc_pid" 2>/dev/null; do
  count=$(ls results/frame_a/cells/cell_llama-3.2-1b_real_MIX_C_*.json 2>/dev/null | wc -l)
  log "MIX_C progress: $count/33 cells"
  sleep 300  # 每5分钟检查一次
done

# 验证 MIX_C
count=$(ls results/frame_a/cells/cell_llama-3.2-1b_real_MIX_C_*.json 2>/dev/null | wc -l)
if [ "$count" -eq 33 ]; then
  log "✓ MIX_C complete (33/33)"
else
  log "✗ MIX_C incomplete ($count/33), aborting subsequent chain"
  echo "MIX_C incomplete $count/33" > engine/CHAIN_ABORT.txt
  exit 3
fi

# Stage 5: GPU idle gate again
log "Stage 5: Final GPU idle gate..."
consec=0
while [ "$consec" -lt 3 ]; do
  line=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  util=$(echo "$line" | awk -F, '{gsub(/[^0-9]/,"",$1); print $1}')
  mem=$(echo "$line" | awk -F, '{gsub(/[^0-9]/,"",$2); print $2}')
  if [ -n "${util:-}" ] && [ -n "${mem:-}" ] && [ "$util" -lt 25 ] && [ "$mem" -lt 1500 ]; then
    consec=$((consec+1))
  else
    consec=0
  fi
  log "GPU idle check: util=${util:-NA} mem=${mem:-NA} consec=${consec}/3"
  [ "$consec" -lt 3 ] && sleep 30
done

# Stage 6: 启动后续链
log "Stage 6: Launching chain_after_bc_drain..."
nohup ./engine/chain_after_bc_drain_20260726.sh > engine/chain_after_bc_drain.nohup.log 2>&1 &
sleep 2
chain_pid=$(cat engine/chain_after_bc_drain_20260726.pid 2>/dev/null || echo "")
if [ -z "$chain_pid" ] || ! kill -0 "$chain_pid" 2>/dev/null; then
  log "✗ chain_after_bc_drain failed to start"
  echo "chain launch failed" > engine/CHAIN_ABORT.txt
  exit 4
fi
log "✓ chain_after_bc_drain launched (pid $chain_pid)"
log "Monitor complete. Chain is now running independently."
log "======== MONITOR AND CHAIN END ========"
exit 0
