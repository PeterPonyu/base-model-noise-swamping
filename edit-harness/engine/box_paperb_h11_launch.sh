#!/usr/bin/env bash
# Launch Paper B H11 missing cells wave on-box after prepare completes.
set -u

H="${HARNESS:-/root/edit-harness}"
DRYRUN="${DRYRUN:-0}"
cd "$H" || exit 1
[ -f engine/box_env.sh ] && source engine/box_env.sh

host=$(hostname)
WAVE="paperb-h11-missing"
DRIVER="$H/run_paperb_h11_missing.sh"
READY="engine/BOX_READY_paperb_h11_missing.ok"

log() { echo "[paperb-h11-launch] $*"; }

# Verify READY receipt and driver sha256
[ -f "$READY" ] || { log "ABORT: wave not READY ($READY missing)"; exit 8; }
recorded=$(sed -n 's/^driver_sha256=//p' "$READY" | tail -1)
current=$(sha256sum "$DRIVER" | cut -d' ' -f1)
[ -n "$recorded" ] && [ "$recorded" = "$current" ] || {
  log "ABORT: READY receipt is stale; rerun box_paperb_h11_prepare.sh check"
  exit 9
}

# Launch helper (with setsid for clean detachment)
launch() {
  name="$1"
  pidfile="$2"
  shift 2

  if [ -f "$pidfile" ]; then
    old=$(tr -dc 0-9 < "$pidfile")
    if [ -n "$old" ] && kill -0 "$old" 2>/dev/null; then
      log "ABORT: $name already running as PID $old"
      return 3
    fi
  fi

  log "LAUNCH $name: $*"
  [ "$DRYRUN" = 1 ] && return 0

  setsid -f env "$@" </dev/null
  for _ in 1 2 3 4 5; do
    if [ -f "$pidfile" ]; then
      p=$(tr -dc 0-9 < "$pidfile")
      if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then
        log "STARTED $name PID=$p"
        return 0
      fi
    fi
    sleep 2
  done
  log "ABORT: $name did not create a live pidfile: $pidfile"
  return 4
}

# Launch dual-card shards
launch_fail=0

launch paperb-h11-card0 engine/run_paperb_h11_missing_card0.pid \
  H="$H" \
  WAVE_BOX="$host" \
  SHARD=card0 \
  GPU_ID=0 \
  BUDGET_MIN="${BUDGET_MIN:-300}" \
  JOB_CAP_MIN="${JOB_CAP_MIN:-120}" \
  bash -c './run_paperb_h11_missing.sh >engine/paperb_h11_missing_card0.nohup.log 2>&1' \
  || launch_fail=1

launch paperb-h11-card1 engine/run_paperb_h11_missing_card1.pid \
  H="$H" \
  WAVE_BOX="$host" \
  SHARD=card1 \
  GPU_ID=1 \
  BUDGET_MIN="${BUDGET_MIN:-300}" \
  JOB_CAP_MIN="${JOB_CAP_MIN:-120}" \
  bash -c './run_paperb_h11_missing.sh >engine/paperb_h11_missing_card1.nohup.log 2>&1' \
  || launch_fail=1

[ "$launch_fail" -eq 0 ] || { log "Launch failed"; exit 4; }

log "Both shards launched successfully"
exit 0
