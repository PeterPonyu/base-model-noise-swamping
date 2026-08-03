#!/bin/bash
# run_mixc.sh — 运行完整的 MIX_C (33 cells)
set -u
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$H/.." || exit 2
PY=${PY:-/home/zeyufu/miniconda3/envs/dl/bin/python3}
PIDFILE=${PIDFILE:-engine/run_mixc.pid}
CHILD_PIDFILE=${CHILD_PIDFILE:-engine/run_mixc.child.pid}
CHECKPOINT=${CHECKPOINT:-engine/run_mixc.checkpoint}
LOG=${LOG:-engine/run_mixc.log}
H18_POLL_SEC=${H18_POLL_SEC:-5}
mkdir -p engine

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
read_pid(){
  [ -f "$1" ] || return 1
  tr -dc 0-9 < "$1"
}
pid_alive(){
  [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null
}
mixc_worker_alive(){
  local pid="${1:-}" cmd
  pid_alive "$pid" || return 1
  cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null) || return 1
  case "$cmd" in
    *experiments.frame_a.run_stream*--mixes*MIX_C*) return 0 ;;
    *) return 1 ;;
  esac
}
cleanup_monitor_pidfile(){
  local recorded
  recorded=$(read_pid "$PIDFILE" 2>/dev/null || true)
  [ "$recorded" = "$$" ] && rm -f "$PIDFILE"
}
checkpoint_signal(){
  local signal="$1" child alive=no count=0 cell
  child=$(read_pid "$CHILD_PIDFILE" 2>/dev/null || true)
  mixc_worker_alive "$child" && alive=yes
  for cell in results/frame_a/cells/cell_llama-3.2-1b_real_MIX_C_*.json; do
    [ -f "$cell" ] && count=$((count+1))
  done
  printf 'event=signal signal=%s time=%s monitor_pid=%s worker_pid=%s worker_alive=%s completed_cells=%s/33\n' \
    "$signal" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$$" "${child:-unknown}" "$alive" "$count" \
    >> "$CHECKPOINT"
  log "CHECKPOINT signal=$signal completed=$count/33 worker=${child:-unknown} alive=$alive; relaunch resumes landed cells"
}
on_signal(){
  local signal="$1" rc="$2"
  trap - TERM INT HUP
  checkpoint_signal "$signal"
  exit "$rc"
}

old_monitor=$(read_pid "$PIDFILE" 2>/dev/null || true)
if pid_alive "$old_monitor"; then
  echo "REFUSE: already running (pid $old_monitor)" >&2; exit 7
fi
printf '%s\n' "$$" > "$PIDFILE"
trap cleanup_monitor_pidfile EXIT
trap 'on_signal TERM 143' TERM
trap 'on_signal INT 130' INT
trap 'on_signal HUP 129' HUP

log "======== RUN MIX_C START pid=$$ ========"
log "LID-OPEN REMINDER: keep lid open (nvidia_uvm wedge)."

# A monitor may be relaunched after receiving a signal while its detached cell is
# still running. Follow that worker instead of starting a duplicate; when it exits,
# the normal idempotent run below skips every cell it already checkpointed to JSON.
CHILD=$(read_pid "$CHILD_PIDFILE" 2>/dev/null || true)
if mixc_worker_alive "$CHILD"; then
  log "REATTACH: detached MIX_C worker pid=$CHILD is still alive"
  while mixc_worker_alive "$CHILD"; do sleep "$H18_POLL_SEC"; done
  recorded=$(read_pid "$CHILD_PIDFILE" 2>/dev/null || true)
  [ "$recorded" = "$CHILD" ] && rm -f "$CHILD_PIDFILE"
  log "REATTACH: worker pid=$CHILD ended; resuming from landed cell JSONs"
elif [ -n "$CHILD" ]; then
  log "STALE: removing non-MIX_C child pid receipt $CHILD_PIDFILE (pid=$CHILD)"
  rm -f "$CHILD_PIDFILE"
fi

ENVP=(env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1)

log "MIX_C start (33 cells + p2 structural file)"
# Keep the worker in its own session/process group. `setsid --wait` preserves the
# Python exit status; the inner shell writes its PID before exec so the receipt
# names Python itself even when util-linux setsid must fork.
rm -f "$CHILD_PIDFILE"
setsid --wait bash -c '
  child_pidfile=$1
  shift
  printf "%s\n" "$$" > "$child_pidfile"
  exec "$@"
' h18-mixc-worker "$CHILD_PIDFILE" "${ENVP[@]}" "$PY" \
    -m experiments.frame_a.run_stream --run --real \
    --mixes MIX_C --model_dir data/models/Llama-3.2-1B >> "$LOG" 2>&1 &
WAITER=$!
CHILD=
for _ in 1 2 3 4 5 6 7 8 9 10; do
  CHILD=$(read_pid "$CHILD_PIDFILE" 2>/dev/null || true)
  [ -n "$CHILD" ] && break
  pid_alive "$WAITER" || break
  sleep 0.1
done
if [ -z "$CHILD" ]; then
  wait "$WAITER"; rc=$?
  log "FAILED: detached MIX_C worker did not publish $CHILD_PIDFILE (rc=$rc)"
  exit "$rc"
fi
log "MIX_C worker started pid=$CHILD session=$CHILD"
wait "$WAITER"
rc=$?
recorded=$(read_pid "$CHILD_PIDFILE" 2>/dev/null || true)
[ "$recorded" = "$CHILD" ] && rm -f "$CHILD_PIDFILE"
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
