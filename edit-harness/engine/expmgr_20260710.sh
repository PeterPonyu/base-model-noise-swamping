#!/usr/bin/env bash
# expmgr_20260710.sh — fully-automatic experiment manager for the next 8 hours
# (user-directed 2026-07-10 ~12:20 EDT). Runs ALONGSIDE the live lane chains; it
# never replaces them — it adds the four things they cannot do for themselves:
#
#   1. WEDGE AUTO-KILL (Lane A): the 10:50 incident — a cell spinning one CPU
#      thread with the GPU idle and its log frozen — was killed manually today.
#      This automates exactly that intervention: if the ACTIVE Lane-A cell's log
#      is >=35 min stale AND GPU util <10% for 3 consecutive 5-min polls, TERM
#      (then KILL) the cell's python, found ONLY by walking /proc descendants of
#      the driver PID (never pgrep/pkill a pattern). The driver logs FAIL and
#      moves on; the supervisor's stage-1 retry gives the cell a second chance.
#      Lane B is NOT wedge-managed here: run_p3_gpu.sh has its own watchdog and
#      100-min per-job caps.
#   2. DOSSIER REFRESH: whenever the Lane-A driver is not running and the 3-seed
#      aggregate tables are newer than the last dossier run, rerun
#      experiments/revision_dossier.py (stdlib, CPU) so PENDING cells resolve.
#   3. P2 FILLER: when the supervisor chain is gone (Lane B finished/aborted),
#      no thermal-defer note exists, the GPU is idle, and >=60 min remain in the
#      window: bridge the P2 specs (EXPLICIT --source; defaults would double-
#      execute Lane B's queue) and start `fission_engine.runner --once` (its own
#      GPU gating + file lock). Launched at most once.
#   4. END REPORT: engine/EXPMGR_REPORT_20260710.txt with the full timeline.
#
# Launch: cd edit-harness && nohup ./engine/expmgr_20260710.sh >> engine/expmgr_20260710.nohup.log 2>&1 &
# Stop:   kill by PID from engine/expmgr_20260710.pid (NEVER pkill -f).
# Env:    WINDOW_MIN (default 480), DRYRUN=1 (log decisions, take no actions),
#         POLL_S (default 300), ENABLE_P2 (default 1).
set -u
H=/home/zeyufu/Desktop/idea-feasibility-analysis/edit-harness
ROOT=/home/zeyufu/Desktop/idea-feasibility-analysis
cd "$H" || exit 2
LOG=engine/expmgr_20260710.log
PIDFILE=engine/expmgr_20260710.pid
REPORT=engine/EXPMGR_REPORT_20260710.txt
WINDOW_MIN="${WINDOW_MIN:-480}"
POLL_S="${POLL_S:-300}"
DRYRUN="${DRYRUN:-0}"
ENABLE_P2="${ENABLE_P2:-1}"
WEDGE_STALE_S=2100          # cell log frozen >= 35 min
WEDGE_POLLS=3               # AND GPU util<10% for 3 consecutive polls
P2_MARKER=engine/expmgr_p2_launched.ok
DOSSIER_STAMP=engine/expmgr_dossier_last
echo "$$" > "$PIDFILE"
log(){ echo "[expmgr $(date '+%F %T')] $*" | tee -a "$LOG"; }
finish(){ rm -f "$PIDFILE"; }
trap finish EXIT
T0=$(date +%s)
left_min(){ echo $(( WINDOW_MIN - ( $(date +%s) - T0 ) / 60 )); }
log "START pid=$$ window=${WINDOW_MIN}m poll=${POLL_S}s DRYRUN=${DRYRUN} ENABLE_P2=${ENABLE_P2}"

alive_from(){ # alive_from <pidfile> <identity-substr> -> echoes pid if alive AND identity matches
  # The identity check (review HIGH): pidfiles can outlive their process, and the kernel
  # recycles PIDs — without matching /proc/<pid>/cmdline, the wedge-kill descendant walk
  # could be rooted at a FOREIGN process. Mirrors chain_lanes stage-0's guard.
  local p ident; [ -f "$1" ] || return 0
  p="$(cat "$1" 2>/dev/null)"
  { [ -n "$p" ] && kill -0 "$p" 2>/dev/null; } || return 0
  ident="$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null)"
  case "$ident" in *"$2"*) echo "$p";; *) ;; esac
}

descendants(){ # all descendant PIDs of $1 (BFS over /proc children; PID-tree only)
  local q="$1" out="" p c
  while [ -n "$q" ]; do
    p="${q%% *}"; [ "$q" = "$p" ] && q="" || q="${q#* }"
    for c in $(cat "/proc/$p/task/"*/children 2>/dev/null); do
      out="$out $c"; q="$q $c"; q="${q# }"
    done
  done
  echo "$out"
}

active_cell_log(){ # prints the log path of the currently-RUNning Lane A cell, if any
  awk '$3=="RUN"{tag=$4; lg=$NF; act=1} ($3=="done"||$3=="FAIL"){if($4==tag)act=0} END{if(act&&lg!="")print lg}' \
      engine/run_lanea_seeds.log 2>/dev/null
}

gpu_util(){ nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc 0-9; }
gpu_mem(){  nvidia-smi --query-gpu=memory.used     --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc 0-9; }

lowutil_consec=0
wedge_kills=0

wedge_check(){
  local drv cell now mt stale util
  drv="$(alive_from engine/run_lanea_seeds.pid run_laneA_seeds.sh)"
  if [ -z "$drv" ]; then lowutil_consec=0; return; fi
  cell="$(active_cell_log)"
  if [ -z "$cell" ] || [ ! -f "$cell" ]; then lowutil_consec=0; return; fi
  now=$(date +%s); mt=$(stat -c %Y "$cell" 2>/dev/null || echo "$now"); stale=$(( now - mt ))
  util="$(gpu_util)"; util="${util:-100}"
  if [ "$stale" -ge "$WEDGE_STALE_S" ] && [ "$util" -lt 10 ]; then
    lowutil_consec=$((lowutil_consec+1))
    log "WEDGE-SUSPECT: $cell stale ${stale}s, util ${util}% (consec ${lowutil_consec}/${WEDGE_POLLS})"
  else
    lowutil_consec=0
    return
  fi
  [ "$lowutil_consec" -lt "$WEDGE_POLLS" ] && return
  lowutil_consec=0
  # find the cell python: python3 descendants of the DRIVER pid only (PID-tree, no patterns)
  local victims="" c comm
  for c in $(descendants "$drv"); do
    comm="$(cat "/proc/$c/comm" 2>/dev/null)"
    case "$comm" in python*) victims="$victims $c";; esac
  done
  if [ -z "${victims# }" ]; then log "WEDGE: no python descendant of driver $drv found — skipping kill"; return; fi
  if [ "$DRYRUN" -eq 1 ]; then log "DRYRUN WEDGE: would TERM${victims} (descendants of driver $drv)"; return; fi
  log "WEDGE-KILL: TERM${victims} (cell $cell frozen; descendants of driver $drv)"
  kill -TERM $victims 2>/dev/null
  sleep 30
  local remain=""
  for c in $victims; do kill -0 "$c" 2>/dev/null && remain="$remain $c"; done
  if [ -n "${remain# }" ]; then log "WEDGE-KILL: escalating KILL${remain}"; kill -KILL $remain 2>/dev/null; fi
  wedge_kills=$((wedge_kills+1))
}

dossier_check(){
  local drv newest t last f
  drv="$(alive_from engine/run_lanea_seeds.pid run_laneA_seeds.sh)"
  [ -n "$drv" ] && return          # only when the driver is not mid-run
  newest=0
  for f in results/C4_causal_instruct_table_3seed.json results/C4_causal_8b_table_3seed.json \
           results/C4_causal_mquake_table_3seed_probesrc.json results/C3_mquake_alpha_L12_3seed.json \
           results/RIPPLE_depth_profile.json; do
    [ -f "$f" ] && { t=$(stat -c %Y "$f"); [ "$t" -gt "$newest" ] && newest="$t"; }
  done
  [ "$newest" -eq 0 ] && return
  last=0; [ -f "$DOSSIER_STAMP" ] && last=$(cat "$DOSSIER_STAMP" 2>/dev/null || echo 0)
  [ "$newest" -le "$last" ] && return
  if [ "$DRYRUN" -eq 1 ]; then log "DRYRUN DOSSIER: would refresh (tables newer than last run)"; return; fi
  log "DOSSIER: aggregates refreshed — rerunning revision_dossier.py"
  python3 experiments/revision_dossier.py --results_dir results --out results/REVISION_DOSSIER.json \
      >> engine/expmgr_dossier.log 2>&1 \
    && { date +%s > "$DOSSIER_STAMP"; log "DOSSIER: $(tail -1 engine/expmgr_dossier.log)"; } \
    || log "DOSSIER: FAILED rc=$? (see engine/expmgr_dossier.log)"
}

p2_check(){
  [ "$ENABLE_P2" -eq 1 ] || return
  [ -f "$P2_MARKER" ] && return
  [ -n "$(alive_from engine/chain_lanes_20260710.pid chain_lanes_20260710.sh)" ] && return   # supervisor still owns the card
  [ -f engine/LANEB_THERMAL_DEFER.txt ] && { log "P2: thermal-defer note present — GPU suspect, NOT launching filler"; : > "$P2_MARKER"; return; }
  [ "$(left_min)" -lt 60 ] && { log "P2: <60m left in window — not starting filler"; : > "$P2_MARKER"; return; }
  local util mem; util="$(gpu_util)"; mem="$(gpu_mem)"
  { [ -n "$util" ] && [ -n "$mem" ] && [ "$util" -lt 25 ] && [ "$mem" -lt 1500 ]; } || return  # try again next poll
  if [ "$DRYRUN" -eq 1 ]; then log "DRYRUN P2: would collect+launch engine runner"; : > "$P2_MARKER"; return; fi
  log "P2: supervisor gone, GPU idle, $(left_min)m left — bridging specs + starting engine runner"
  ( cd "$ROOT" \
    && python3 -m fission_engine.collect_branch_jobs --source "branches/p2_prerl_diag/queue/*.json" \
    && nohup python3 -m fission_engine.runner --once >> "$H/engine/expmgr_p2_runner.nohup.log" 2>&1 & \
    echo $! > "$H/engine/expmgr_p2_runner.pid" ) >> "$LOG" 2>&1
  : > "$P2_MARKER"
  log "P2: engine runner launched pid=$(cat engine/expmgr_p2_runner.pid 2>/dev/null)"
}

all_quiet(){ # 0 when nothing managed is running anymore
  [ -n "$(alive_from engine/chain_lanes_20260710.pid chain_lanes_20260710.sh)" ] && return 1
  [ -n "$(alive_from engine/run_lanea_seeds.pid run_laneA_seeds.sh)" ] && return 1
  [ -n "$(alive_from engine/chain_laneA_20260710.pid chain_laneA_20260710.sh)" ] && return 1
  [ -n "$(alive_from engine/expmgr_p2_runner.pid fission_engine.runner)" ] && return 1
  # P2 not yet attempted (marker absent, filler enabled) -> not quiet yet
  [ "$ENABLE_P2" -eq 1 ] && [ ! -f "$P2_MARKER" ] && return 1
  return 0
}

write_report(){
  {
    echo "EXPMGR REPORT $(date '+%F %T')  (window ${WINDOW_MIN}m, started $(date -d "@$T0" '+%T'))"
    echo "wedge auto-kills this window: ${wedge_kills}"
    echo ""
    echo "--- Lane A (run_lanea_seeds_report.txt) ---"
    head -1 engine/run_lanea_seeds_report.txt 2>/dev/null || echo "(no report yet)"
    echo ""
    echo "--- supervisor (chain_lanes) tail ---"
    tail -6 engine/chain_lanes_20260710.log 2>/dev/null
    echo ""
    echo "--- Lane B (P3) ---"
    if [ -f "$ROOT/branches/p3_agent_ipi/results/P3_GPU_report.json" ]; then
      python3 -c "
import json
r = json.load(open('$ROOT/branches/p3_agent_ipi/results/P3_GPU_report.json'))
for g in r.get('defense_gates', []): print('  defense:', g)
for g in r.get('grid_contrasts', []): print('  grid   :', g)
" 2>/dev/null || echo "  (report unparseable)"
    else echo "  (no P3_GPU_report.json yet)"; fi
    echo ""
    echo "--- dossier ---"
    tail -2 engine/expmgr_dossier.log 2>/dev/null || echo "  (never ran)"
    echo ""
    echo "--- P2 filler ---"
    if [ -f engine/expmgr_p2_runner.pid ]; then
      echo "  launched; queue state:"
      ls "$ROOT/fission-engine/queue/"*.json 2>/dev/null | head -3
      echo "  done:   $(ls "$ROOT/fission-engine/queue/done/"*.json 2>/dev/null | wc -l)"
      echo "  failed: $(ls "$ROOT/fission-engine/queue/failed/"*.json 2>/dev/null | wc -l)"
    else echo "  not launched (marker: $([ -f "$P2_MARKER" ] && echo present || echo absent))"; fi
    echo ""
    echo "--- manager event log ---"
    grep -E "WEDGE|DOSSIER|P2:|START|END" "$LOG" 2>/dev/null | tail -30
  } > "$REPORT"
  log "report written: $REPORT"
}

# ---------------------------------------------------------------- main loop
while :; do
  if [ "$(left_min)" -le 0 ]; then log "END: 8h window elapsed"; break; fi
  wedge_check
  dossier_check
  p2_check
  if all_quiet; then log "END: everything managed has drained (window had $(left_min)m left)"; break; fi
  sleep "$POLL_S"
done
write_report
log "DONE"
