#!/bin/bash
# gpu_watchdog.sh -- polls nvidia-smi every 60s while the P3 Ollama sweep runs.
# If an ollama/llama-server/llama-cpp process appears on the GPU, kills the
# sweep + ollama server BY PID (never pgrep/pkill by pattern -- workspace
# standing rule) and appends an ABORT line to the sweep log, then exits.
#
# Does NOT touch the review-frozen run_ipi.py / transport.py / models.py --
# this is an external safety net only.
#
# Usage: gpu_watchdog.sh <ollama_pid> <sweep_pid> <sweep_log_path>

set -u
OLLAMA_PID="$1"
SWEEP_PID="$2"
SWEEP_LOG="$3"
SELF_PID=$$
INTERVAL=60

echo "[watchdog $SELF_PID] started $(date -Iseconds); watching ollama_pid=$OLLAMA_PID sweep_pid=$SWEEP_PID" >> "$SWEEP_LOG"

while true; do
    sleep "$INTERVAL"

    # If the sweep has already finished on its own, stand down cleanly.
    if ! kill -0 "$SWEEP_PID" 2>/dev/null; then
        echo "[watchdog $SELF_PID] sweep_pid=$SWEEP_PID no longer alive; watchdog exiting $(date -Iseconds)" >> "$SWEEP_LOG"
        exit 0
    fi

    offenders=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null \
        | grep -iE "ollama|llama-server|llama-cpp" || true)

    if [ -n "$offenders" ]; then
        echo "[watchdog $SELF_PID] ABORT $(date -Iseconds): GPU offender(s) detected:" >> "$SWEEP_LOG"
        echo "$offenders" | sed 's/^/[watchdog]   /' >> "$SWEEP_LOG"

        kill -0 "$SWEEP_PID" 2>/dev/null && kill "$SWEEP_PID" \
            && echo "[watchdog $SELF_PID] killed sweep_pid=$SWEEP_PID" >> "$SWEEP_LOG"
        kill -0 "$OLLAMA_PID" 2>/dev/null && kill "$OLLAMA_PID" \
            && echo "[watchdog $SELF_PID] killed ollama_pid=$OLLAMA_PID" >> "$SWEEP_LOG"

        echo "[watchdog $SELF_PID] ABORT complete $(date -Iseconds); watchdog self-exiting" >> "$SWEEP_LOG"
        exit 1
    fi
done
