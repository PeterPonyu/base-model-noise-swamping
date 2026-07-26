#!/bin/bash
# stop_deep_queue.sh — cleanly preempt the deep queue AND its in-flight GPU job.
# Kill by recorded PID/PGID only (NEVER pgrep/pkill a pattern — self-match rule).
# The queue is launched with setsid, so its PID == its process-group ID; killing the
# group takes down the runner, the timeout wrapper, and the in-flight python job.
# Resume later with the SAME launch command — the idempotent skip (json+npz both
# present, json written last+atomically) re-runs only the truncated cell.
set -u
H=/home/zeyufu/Desktop/idea-feasibility-analysis/edit-harness
PIDFILE=$H/${1:-engine/deep_until1900.pid}
[ -f "$PIDFILE" ] || { echo "no pid file ($PIDFILE) — queue not running?"; exit 1; }
PID=$(cat "$PIDFILE")
if ! kill -0 "$PID" 2>/dev/null; then echo "pid $PID already dead — GPU is free"; exit 0; fi
# Collect descendants FIRST (GNU timeout puts its child in a NEW process group,
# so the group-kill below misses the in-flight job — learned 2026-07-02 02:30).
# Recursive walk by PID only; never pgrep patterns.
descendants(){ local p; for p in $(ps -o pid= --ppid "$1" 2>/dev/null); do echo "$p"; descendants "$p"; done; }
DESC=$(descendants "$PID")
echo "stopping deep queue: TERM to process group -$PID (descendants: $(echo $DESC | tr '\n' ' '))"
kill -TERM -- -"$PID" 2>/dev/null
for d in $DESC; do kill -TERM "$d" 2>/dev/null; done
for i in 1 2 3 4 5 6 7 8 9 10; do
  kill -0 "$PID" 2>/dev/null || { echo "stopped cleanly (${i}s)"; break; }
  sleep 1
done
if kill -0 "$PID" 2>/dev/null; then
  echo "still alive after 10s — KILL to process group"
  kill -KILL -- -"$PID" 2>/dev/null
  sleep 1
fi
# escalate on any surviving descendant (in-flight GPU job under its own pgid)
for d in $DESC; do
  if kill -0 "$d" 2>/dev/null; then
    echo "descendant $d survived TERM — KILL"
    kill -KILL "$d" 2>/dev/null
  fi
done
sleep 1
kill -0 "$PID" 2>/dev/null && { echo "FAILED to stop $PID"; exit 1; }
for d in $DESC; do kill -0 "$d" 2>/dev/null && { echo "FAILED to stop descendant $d"; exit 1; }; done
echo "deep queue stopped. GPU state:"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "RESUME LATER: cd $H && nohup setsid ./run_deep_until1900.sh >> engine/deep_until1900.nohup.log 2>&1 &"
