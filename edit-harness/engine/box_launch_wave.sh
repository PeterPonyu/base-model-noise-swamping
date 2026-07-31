#!/usr/bin/env bash
# Launch a prepared wave on-box after box_prepare_wave.sh check writes BOX_READY.
set -u
WAVE="${1:-}"; DRYRUN="${DRYRUN:-0}"; H="${HARNESS:-/root/edit-harness}"
cd "$H" || exit 1
[ -f engine/box_env.sh ] && source engine/box_env.sh
host=$(hostname)
require_current_ready(){
  wave="$1"; driver="$2"; ready="engine/BOX_READY_${wave}.ok"
  [ -f "$ready" ] || { echo "ABORT: wave not READY" >&2; exit 8; }
  recorded=$(sed -n 's/^driver_sha256=//p' "$ready" | tail -1)
  current=$(sha256sum "$driver" | cut -d' ' -f1)
  [ -n "$recorded" ] && [ "$recorded" = "$current" ] || {
    echo "ABORT: READY receipt is stale for $driver; rerun box_prepare_wave.sh $wave check" >&2
    exit 9
  }
}
launch(){
  name="$1"; pidfile="$2"; shift 2
  if [ -f "$pidfile" ]; then
    old=$(tr -dc 0-9 < "$pidfile")
    if [ -n "$old" ] && kill -0 "$old" 2>/dev/null; then
      echo "ABORT: $name already running as PID $old" >&2; return 3
    fi
  fi
  echo "LAUNCH $name: $*"
  [ "$DRYRUN" = 1 ] && return 0
  setsid -f env "$@" </dev/null
  for _ in 1 2 3 4 5; do
    if [ -f "$pidfile" ]; then
      p=$(tr -dc 0-9 < "$pidfile")
      if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then
        echo "STARTED $name PID=$p"; return 0
      fi
    fi
    sleep 2
  done
  echo "ABORT: $name did not create a live pidfile: $pidfile" >&2
  return 4
}
case "$WAVE" in
  deletion-wave1)
    require_current_ready deletion-wave1 "$H/run_deletion_wave1.sh"
    launch_fail=0
    launch deletion-card0 engine/run_deletion_wave1_card0.pid \
      H="$H" WAVE_BOX="$host" SHARD=card0 GPU_ID=0 BUDGET_MIN="${BUDGET_MIN:-540}" JOB_CAP_MIN="${JOB_CAP_MIN:-100}" \
      bash -c './run_deletion_wave1.sh >engine/deletion_wave1_card0.nohup.log 2>&1' || launch_fail=1
    launch deletion-card1 engine/run_deletion_wave1_card1.pid \
      H="$H" WAVE_BOX="$host" SHARD=card1 GPU_ID=1 BUDGET_MIN="${BUDGET_MIN:-540}" JOB_CAP_MIN="${JOB_CAP_MIN:-100}" \
      bash -c './run_deletion_wave1.sh >engine/deletion_wave1_card1.nohup.log 2>&1' || launch_fail=1
    [ "$launch_fail" -eq 0 ] || exit 4
    ;;
  deletion-wave2)
    require_current_ready deletion-wave2 "$H/run_deletion_wave2.sh"
    launch deletion-wave2 engine/run_deletion_wave2.pid \
      H="$H" WAVE_BOX="$host" GPU_ID=0 BUDGET_MIN="${BUDGET_MIN:-1260}" JOB_CAP_MIN="${JOB_CAP_MIN:-150}" \
      bash -c './run_deletion_wave2.sh >engine/deletion_wave2.nohup.log 2>&1'
    ;;
  paperb-curve)
    require_current_ready paperb-curve "$H/run_paperb_curve_cloud.sh"
    launch paperb-curve engine/run_paperb_curve_cloud.pid \
      H="$H" WAVE_BOX="$host" GPU_ID=0 BUDGET_MIN="${BUDGET_MIN:-300}" JOB_CAP_MIN="${JOB_CAP_MIN:-150}" \
      bash -c './run_paperb_curve_cloud.sh >engine/paperb_curve_cloud.nohup.log 2>&1'
    ;;
  d2-prospective)
    require_current_ready d2-prospective "$H/run_d2_prospective_cloud.sh"
    launch d2-prospective engine/run_d2_prospective_cloud.pid \
      H="$H" WAVE_BOX="$host" BUDGET_MIN="${BUDGET_MIN:-240}" \
      bash -c './run_d2_prospective_cloud.sh >engine/d2_prospective_cloud.nohup.log 2>&1'
    ;;
  *) echo "usage: $0 {deletion-wave1|deletion-wave2|paperb-curve|d2-prospective}" >&2; exit 2 ;;
esac
