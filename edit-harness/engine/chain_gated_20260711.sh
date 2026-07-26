#!/usr/bin/env bash
# chain_gated_20260711.sh — gate-file-armed supervisor: drains the LIVE run_revins.sh
# (B6 revision-insurance driver, ~4.3 GPU-h, pid in engine/run_revins.pid), runs the CPU
# revision-dossier post-pass, then polls for two USER-APPROVED GPU workloads and launches
# them serially, never both at once:
#   (a) P2 GRPO confirmatory wave  — branches/p2_prerl_diag/run_p2_grpo.sh, gated on
#       engine/GRPO_GO.ok (already smoke-tested GREEN; ~9-24 GPU-h).
#   (b) P3 wave-3 lineage_arm tier — branches/p3_agent_ipi/run_wave3.sh, gated on
#       branches/p3_agent_ipi/WAVE3_GO.ok. Being built by a parallel agent at authoring
#       time — this script only checks it exists before invoking it and self-gates
#       further on its own prereg/models/GO markers; we do not touch its internals.
# Neither GO marker exists at authoring time — both launches are USER-GATED. This script
# is built + verified only; the orchestrator arms it after review (not launched here).
#
# NEW file — does not edit run_revins.sh, chain_lanes_20260710.sh, run_p2_grpo.sh, or any
# existing driver. Own namespace only: chain_gated_20260711.{log,pid}, chain_gated_report.txt,
# chain_gated_grpo.{pid,done}, chain_gated_wave3.{pid,done}. (CHAIN_THERMAL_DEFER.txt is the
# one exception, named to mirror the LANEB_THERMAL_DEFER.txt convention from
# chain_lanes_20260710.sh.)
#
# REVIEW FIXES (2026-07-11, applied after hostile review APPROVE-WITH-FIXES):
#   1. Durable launch state: chain_gated_grpo.done / chain_gated_wave3.done are written
#      immediately after each rc capture (rc value is the file's content) and re-read at
#      startup (hydrate_launch_state, called before anything else runs, incl. under DRYRUN).
#      Without this, a supervisor RESTART while GRPO_GO.ok is still on disk would re-dispatch
#      an already-finished job — the in-memory GRPO_LAUNCHED/WAVE3_LAUNCHED flags reset on
#      restart, but the .done marker survives it.
#   2. Stage-0 DONE detection is unanchored `grep RUN_REVINS_DONE` (was `^RUN_REVINS_DONE$`);
#      the "last DONE line after last START line" ordering check (revins_marker_done) already
#      disambiguates real completions from prior-run leftovers, so the exact-line anchor only
#      added a silent-hang risk if run_revins.sh ever timestamps/prefixes that line.
#   3. Stage-2's 24h deadline is PAUSED for the duration of any dispatch_grpo/dispatch_wave3
#      call (idle-gate + thermal-gate + the job's own wait), via a DISPATCH_ELAPSED accumulator
#      added back onto the deadline each poll. Otherwise a 20h approved GRPO run would eat the
#      entire 24h window and strand an already-approved wave3 behind it. The 24h budget governs
#      time spent waiting for a NEW gate marker to appear, not time spent executing on one
#      that already arrived.
#   4. Stage-0 fast-fail: if the revins pid is observed dead and no ordering-valid DONE
#      marker appears within a 15-minute grace window from that observation, exit 3
#      immediately ("revins died without DONE") instead of polling the full 12h timeout.
#
# Standing rules honored:
#   - wait by PID (kill -0), NEVER pgrep/pkill a pattern (watcher cmdlines self-match).
#   - GPU idle-gate on util<25 AND mem<1500MiB (never zero-compute-apps — persistent CUDA
#     contexts, e.g. mcp_litchron.server, never clear; memory: gpu-idle-gate-not-zero-
#     compute-apps).
#   - the idle gate only reads aggregate nvidia-smi counters and never inspects or signals
#     any specific PID, so it can never touch a foreign workspace's job — "waiting" is its
#     only possible interaction with one.
#   - thermal burn gate before any multi-hour launch (memory: gpu-60w-thermal-cap-reboot-fix).
#   - no heredoc through `conda run` (stdin-swallow hazard) — the burn uses the interpreter
#     path directly.
#
# Stages:
#   0. wait for run_revins.sh to drain (pid dead AND RUN_REVINS_DONE in its log, ordering-
#      checked against the last START line). Poll 60s, 12h overall timeout => log + exit 3;
#      fast-fails at exit 3 sooner if the pid dies and no DONE shows up within 15m (fix 4).
#   1. unconditional CPU post-pass: experiments/revision_dossier.py -> results/REVISION_DOSSIER.json.
#   2. gated dispatch loop, poll 300s, 24h deadline (from script start, PAUSED during any
#      dispatch call — fix 3) on the POLLING itself — once a job is actually launched we
#      block on it by PID with no deadline, since an approved multi-hour job should run to
#      completion, not be abandoned mid-flight:
#        a. GRPO_GO.ok present & GRPO not yet launched (per chain_gated_grpo.done, fix 1) ->
#           idle-gate + thermal-gate -> nohup run_p2_grpo.sh, pid -> chain_gated_grpo.pid,
#           wait, capture rc -> chain_gated_grpo.done.
#        b. else WAVE3_GO.ok present & wave3 not yet launched (per chain_gated_wave3.done) ->
#           same gates -> nohup run_wave3.sh, pid -> chain_gated_wave3.pid, wait, capture rc
#           -> chain_gated_wave3.done.
#        c. each branch is fully synchronous (gate -> launch -> wait -> rc) before the loop
#           re-polls, so GRPO and wave3 can structurally never run concurrently; if both
#           markers are already present the first poll always takes GRPO.
#   3. exit report: engine/chain_gated_report.txt (what ran, rcs, what's still gated).
#
# Launch:  cd edit-harness && nohup ./engine/chain_gated_20260711.sh \
#            >> engine/chain_gated_20260711.nohup.log 2>&1 &
# Stop:    kill by PID from engine/chain_gated_20260711.pid (never pkill -f).
# Dry-run: DRYRUN=1 ./engine/chain_gated_20260711.sh — prints the staged plan + current gate
#          states and exits 0; does not wait, launch, or write any pidfile.
set -u

ROOT=/home/zeyufu/Desktop/idea-feasibility-analysis
H="$ROOT/edit-harness"
P2="$ROOT/branches/p2_prerl_diag"
P3="$ROOT/branches/p3_agent_ipi"
PY=/home/zeyufu/miniconda3/envs/dl/bin/python3
LOG="$H/engine/chain_gated_20260711.log"
PIDFILE="$H/engine/chain_gated_20260711.pid"
REPORT="$H/engine/chain_gated_report.txt"
POLL_SEC=300
START_EPOCH=$(date +%s)

log(){ echo "[chain-gated $(date '+%F %T')] $*" | tee -a "$LOG"; }

# run_revins.log is append-only ACROSS RESTARTS of run_revins.sh (observed live: a dry-run
# pass wrote RUN_REVINS_DONE, then a fresh real run started seconds later and is still
# mid-flight) — a bare `grep -q RUN_REVINS_DONE` on the whole file would false-positive on
# a stale marker from an EARLIER invocation while the current one is still running. Require
# the last RUN_REVINS_DONE line to come after the last "RUN_REVINS START" line instead.
revins_marker_done(){
  local revlog="$1" last_start last_done
  last_start="$(grep -n "RUN_REVINS START" "$revlog" 2>/dev/null | tail -1 | cut -d: -f1)"
  # unanchored (review fix 2): the ordering check above already disambiguates a real
  # completion from an earlier invocation's leftover marker, so an exact-line anchor here
  # only risks a silent hang if run_revins.sh ever prefixes/timestamps this line.
  last_done="$(grep -n "RUN_REVINS_DONE" "$revlog" 2>/dev/null | tail -1 | cut -d: -f1)"
  [ -n "$last_done" ] && [ -n "$last_start" ] && [ "$last_done" -gt "$last_start" ]
}

# ---------------------------------------------------------------- state (set -u needs init)
STAGE0_STATUS="not reached"
STAGE1_RC=""
STAGE1_SUMMARY="not reached"
GRPO_LAUNCHED=0
GRPO_RC=""
WAVE3_LAUNCHED=0
WAVE3_RC=""
DISPATCH_ELAPSED=0   # seconds spent inside dispatch_* calls; added back onto the stage-2 deadline (review fix 3)

# review fix 1: durable launch state survives a supervisor restart. Called before anything
# else (incl. under DRYRUN, so the dry-run report reflects reality) so a restart while
# GRPO_GO.ok/WAVE3_GO.ok are still on disk does not re-dispatch an already-finished job.
hydrate_launch_state(){
  if [ -f "$H/engine/chain_gated_grpo.done" ]; then
    GRPO_LAUNCHED=1
    GRPO_RC="$(cat "$H/engine/chain_gated_grpo.done" 2>/dev/null)"
    log "startup: engine/chain_gated_grpo.done present (rc=${GRPO_RC}) — GRPO already resolved, will not re-dispatch"
  fi
  if [ -f "$H/engine/chain_gated_wave3.done" ]; then
    WAVE3_LAUNCHED=1
    WAVE3_RC="$(cat "$H/engine/chain_gated_wave3.done" 2>/dev/null)"
    log "startup: engine/chain_gated_wave3.done present (rc=${WAVE3_RC}) — wave3 already resolved, will not re-dispatch"
  fi
}
hydrate_launch_state

# ---------------------------------------------------------------- DRYRUN: plan + gate states only
if [ "${DRYRUN:-0}" = "1" ]; then
  revins_pid="$(cat "$H/engine/run_revins.pid" 2>/dev/null || true)"
  if [ -n "$revins_pid" ] && kill -0 "$revins_pid" 2>/dev/null; then revins_alive="ALIVE"; else revins_alive="DEAD"; fi
  revins_done="ABSENT"; revins_marker_done "$H/engine/run_revins.log" && revins_done="PRESENT (for the latest run_revins.sh invocation)"
  grpo_go="ABSENT"; [ -f "$H/engine/GRPO_GO.ok" ] && grpo_go="PRESENT"
  wave3_go="ABSENT"; [ -f "$P3/WAVE3_GO.ok" ] && wave3_go="PRESENT"
  wave3_script="ABSENT"; [ -f "$P3/run_wave3.sh" ] && wave3_script="PRESENT"
  {
    echo "=== chain_gated_20260711.sh DRYRUN plan ==="
    echo "STAGE 0: wait for run_revins.sh to drain"
    echo "  pidfile=$H/engine/run_revins.pid pid=${revins_pid:-none} (${revins_alive})"
    echo "  marker=RUN_REVINS_DONE in $H/engine/run_revins.log: ${revins_done}"
    echo "  poll=60s timeout=12h (from script start) => exit 3 on breach"
    echo ""
    echo "STAGE 1: unconditional CPU post-pass"
    echo "  $PY experiments/revision_dossier.py --results_dir results --out results/REVISION_DOSSIER.json (run from $H)"
    echo ""
    echo "STAGE 2: gated dispatch loop, poll=${POLL_SEC}s, deadline=24h from script start PAUSED during any"
    echo "         dispatch_* call (DISPATCH_ELAPSED accumulator added back onto the deadline each poll)"
    echo "  a. GRPO: engine/GRPO_GO.ok = ${grpo_go} (launched=${GRPO_LAUNCHED} rc=${GRPO_RC:-n/a}, durable via chain_gated_grpo.done)"
    echo "           -> idle-gate(util<25,mem<1500MiB x3) + thermal-gate(GEMM>=120W)"
    echo "           -> nohup $P2/run_p2_grpo.sh, pid -> engine/chain_gated_grpo.pid, wait, capture rc -> chain_gated_grpo.done"
    echo "  b. wave3: $P3/WAVE3_GO.ok = ${wave3_go} (run_wave3.sh present: ${wave3_script}, launched=${WAVE3_LAUNCHED} rc=${WAVE3_RC:-n/a}, durable via chain_gated_wave3.done)"
    echo "           -> idle-gate + thermal-gate -> nohup $P3/run_wave3.sh, pid -> engine/chain_gated_wave3.pid, wait, capture rc -> chain_gated_wave3.done"
    echo "  c. serial only: each branch is synchronous gate->launch->wait->rc; GRPO takes priority if both markers present"
    echo "  thermal-defer (<120W peak): write engine/CHAIN_THERMAL_DEFER.txt, exit 5"
    echo "  torch import failure at thermal-gate: SKIPPED-WARN (logged), gate passes without a power reading"
    echo ""
    echo "STAGE 0 fast-fail: if the revins pid is observed dead with no ordering-valid DONE marker"
    echo "  within a 15-minute grace window, exit 3 immediately instead of polling the full 12h"
    echo ""
    echo "EXIT REPORT: engine/chain_gated_report.txt ; transitions logged to engine/chain_gated_20260711.log"
    echo ""
    echo "=== current state (informational only; nothing launched by DRYRUN) ==="
    echo "revins: pid=${revins_pid:-none} ${revins_alive}, RUN_REVINS_DONE=${revins_done}"
    echo "GRPO_GO.ok: ${grpo_go}  (durable state: launched=${GRPO_LAUNCHED} rc=${GRPO_RC:-n/a})"
    echo "WAVE3_GO.ok: ${wave3_go} (run_wave3.sh: ${wave3_script})  (durable state: launched=${WAVE3_LAUNCHED} rc=${WAVE3_RC:-n/a})"
  } | tee -a "$LOG"
  log "DRYRUN complete — no waiting, no launches, no pidfile written"
  exit 0
fi

# ---------------------------------------------------------------- real run: pidfile + trap
echo "$$" > "$PIDFILE"
finish(){ rm -f "$PIDFILE"; }
trap finish EXIT
log "START pid=$$ (revins pidfile=$H/engine/run_revins.pid, GRPO gate=$H/engine/GRPO_GO.ok, wave3 gate=$P3/WAVE3_GO.ok)"

write_report(){
  local extra="${1:-}"
  {
    echo "chain_gated_20260711 report — generated $(date '+%F %T')"
    echo "script start: $(date -d "@$START_EPOCH" '+%F %T' 2>/dev/null || echo "$START_EPOCH")"
    echo ""
    echo "stage0 (revins drain): ${STAGE0_STATUS}"
    echo "stage1 (revision_dossier.py): rc=${STAGE1_RC:-n/a} :: ${STAGE1_SUMMARY}"
    echo ""
    echo "GRPO (P2 confirmatory wave):"
    echo "  GRPO_GO.ok: $([ -f "$H/engine/GRPO_GO.ok" ] && echo present || echo absent)"
    echo "  launched: ${GRPO_LAUNCHED}  rc: ${GRPO_RC:-n/a}"
    echo ""
    echo "wave3 (P3 lineage_arm tier):"
    echo "  WAVE3_GO.ok: $([ -f "$P3/WAVE3_GO.ok" ] && echo present || echo absent)"
    echo "  run_wave3.sh present: $([ -f "$P3/run_wave3.sh" ] && echo yes || echo no)"
    echo "  launched: ${WAVE3_LAUNCHED}  rc: ${WAVE3_RC:-n/a}"
    echo ""
    [ -n "$extra" ] && echo "note: $extra"
  } > "$REPORT"
  log "report written: $REPORT"
}

# ---------------------------------------------------------------- stage 0: wait for revins
stage0_wait_revins(){
  local pidfile="$H/engine/run_revins.pid" revlog="$H/engine/run_revins.log"
  local deadline=$((START_EPOCH + 12*3600))
  local grace=900   # review fix 4: 15-minute fast-fail grace window after pid-death is observed
  local dead_since=""
  log "stage0: waiting for run_revins.sh to drain (poll 60s, timeout 12h; fast-fail if pid dies without DONE within 15m)"
  while :; do
    local now; now=$(date +%s)
    if [ "$now" -ge "$deadline" ]; then
      log "stage0: ABORT — revins did not drain within 12h of script start"
      STAGE0_STATUS="TIMEOUT after 12h — revins pidfile/log did not show drained"
      write_report "aborted at stage0 (revins drain timeout)"
      exit 3
    fi
    local alive=0 rp
    if [ -f "$pidfile" ]; then
      rp="$(cat "$pidfile" 2>/dev/null)"
      { [ -n "$rp" ] && kill -0 "$rp" 2>/dev/null; } && alive=1
    fi
    local marker=0
    revins_marker_done "$revlog" && marker=1
    if [ "$alive" -eq 0 ] && [ "$marker" -eq 1 ]; then
      log "stage0: revins drained (pid dead, RUN_REVINS_DONE present after the last START line)"
      STAGE0_STATUS="drained OK at $(date '+%F %T')"
      return 0
    fi
    if [ "$alive" -eq 0 ]; then
      if [ -z "$dead_since" ]; then
        dead_since="$now"
        log "stage0: pid observed dead without a valid DONE marker — starting 15m fast-fail grace window (covers a legitimate restart racing the check)"
      elif [ $((now - dead_since)) -ge "$grace" ]; then
        log "stage0: FAST-FAIL — revins pid dead >=15m with no ordering-valid RUN_REVINS_DONE; treating as crashed/killed, not a slow drain"
        STAGE0_STATUS="FAST-FAIL: revins died without DONE (dead since $(date -d "@$dead_since" '+%F %T' 2>/dev/null || echo "$dead_since"))"
        write_report "aborted at stage0 (revins died without DONE, 15m grace expired)"
        exit 3
      fi
    else
      dead_since=""   # pidfile shows a live pid again (e.g. revins restarted) — cancel the grace countdown
    fi
    log "stage0: still waiting (pid_alive=${alive} done_marker=${marker}${dead_since:+ dead_since=$(date -d "@$dead_since" '+%T' 2>/dev/null || echo "$dead_since")})"
    sleep 60
  done
}

# ---------------------------------------------------------------- stage 1: CPU dossier post-pass
run_stage1_dossier(){
  log "stage1: CPU post-pass — revision_dossier.py (unconditional, no GPU)"
  local out rc
  out="$(cd "$H" && "$PY" experiments/revision_dossier.py --results_dir results --out results/REVISION_DOSSIER.json 2>&1)"
  rc=$?
  echo "$out" >> "$LOG"
  local summary
  summary="$(echo "$out" | grep '^\[revision_dossier\] .*stable=' | tail -1)"
  STAGE1_RC=$rc
  if [ -n "$summary" ]; then
    STAGE1_SUMMARY="$summary"
  else
    STAGE1_SUMMARY="rc=${rc} (no summary line found in output — see full output in $LOG)"
  fi
  log "stage1: rc=${rc} :: ${STAGE1_SUMMARY}"
}

# ---------------------------------------------------------------- gates (shared by both dispatch branches)
idle_gate(){
  local idle_ok=0 tries=0
  log "gate: idle-gate start (util<25 && mem<1500MiB, need 3 consecutive passes 60s apart)"
  while [ "$idle_ok" -lt 3 ]; do
    tries=$((tries+1))
    if [ "$tries" -gt 180 ]; then
      log "gate: idle-gate ABORT — GPU never idled within ~3h"
      return 1
    fi
    local line util mem
    line="$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)"
    util="$(echo "$line" | awk -F, '{gsub(/[^0-9]/,"",$1);print $1}')"
    mem="$(echo "$line"  | awk -F, '{gsub(/[^0-9]/,"",$2);print $2}')"
    if [ -n "$util" ] && [ -n "$mem" ] && [ "$util" -lt 25 ] && [ "$mem" -lt 1500 ]; then
      idle_ok=$((idle_ok+1)); log "gate: idle PASS ${idle_ok}/3 (util=${util}% mem=${mem}MiB)"
    else
      idle_ok=0; log "gate: idle busy (util=${util:-?}% mem=${mem:-?}MiB), resetting"
    fi
    [ "$idle_ok" -lt 3 ] && sleep 60
  done
  return 0
}

# returns 0=pass, 1=transient failure (caller retries later); exits 5 itself on a real defer.
thermal_gate(){
  log "gate: thermal-gate start (card must already be idle-gated; GEMM burn requires >=120W peak)"
  if ! timeout 20 "$PY" -c "import torch" >/dev/null 2>&1; then
    log "gate: thermal-gate SKIPPED-WARN — torch import failed under $PY; proceeding without a power reading"
    return 0
  fi
  timeout --signal=KILL 90 "$PY" - <<'PYEOF' >> "$LOG" 2>&1 &
import torch, time
a = torch.randn(8192, 8192, device='cuda', dtype=torch.float16)
b = torch.randn(8192, 8192, device='cuda', dtype=torch.float16)
t0 = time.time()
while time.time() - t0 < 25:
    a @ b
torch.cuda.synchronize()
PYEOF
  local burn_pid=$! peak=0 w
  sleep 6
  for _ in 1 2 3 4; do
    w="$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null | head -1 | cut -d. -f1 | tr -dc 0-9)"
    [ -n "$w" ] && [ "$w" -gt "$peak" ] && peak="$w"
    sleep 5
  done
  wait "$burn_pid" 2>/dev/null; local burn_rc=$?
  log "gate: thermal-gate peak=${peak}W burn_rc=${burn_rc}"
  if [ "$burn_rc" -ne 0 ]; then
    log "gate: thermal-gate burn command failed/timed out (rc=${burn_rc}, not an import error) — transient, will retry next poll cycle"
    return 1
  fi
  if [ "$peak" -lt 120 ]; then
    {
      echo "CHAIN-THERMAL-DEFER $(date '+%F %T'): GEMM burn peaked at ${peak}W (<120W)."
      echo "The 60W SW-thermal cap has likely re-appeared (memory: gpu-60w-thermal-cap-reboot-fix)."
      echo "REBOOT the box, then relaunch:"
      echo "  cd $H && nohup ./engine/chain_gated_20260711.sh >> engine/chain_gated_20260711.nohup.log 2>&1 &"
      echo "(idempotent: revins/stage1 fast-pass if already done; unresolved GO markers keep gating as before)"
    } | tee -a "$LOG" > "$H/engine/CHAIN_THERMAL_DEFER.txt"
    write_report "exited at thermal gate: peak ${peak}W < 120W, THERMAL-DEFER written"
    exit 5
  fi
  return 0
}

# ---------------------------------------------------------------- dispatch branches
dispatch_grpo(){
  log "dispatch: GRPO_GO.ok present -> gating P2 GRPO confirmatory wave"
  idle_gate || { log "dispatch: GRPO idle-gate did not pass — retry next poll cycle"; return 1; }
  thermal_gate; local trc=$?
  [ "$trc" -eq 1 ] && return 1
  log "dispatch: launching run_p2_grpo.sh (from $P2)"
  cd "$P2" || { log "dispatch: ABORT cannot cd to $P2"; return 1; }
  nohup ./run_p2_grpo.sh >> run_p2_grpo.nohup.log 2>&1 &
  local gp=$!
  echo "$gp" > "$H/engine/chain_gated_grpo.pid"
  cd "$H" || exit 2
  log "dispatch: GRPO launched pid=${gp} (own pidfile $P2/run_p2_grpo.pid; supervisor copy engine/chain_gated_grpo.pid)"
  wait "$gp"; local grc=$?
  echo "$grc" > "$H/engine/chain_gated_grpo.done"   # review fix 1: durable marker, written immediately after rc capture
  GRPO_RC=$grc
  GRPO_LAUNCHED=1
  log "dispatch: GRPO finished rc=${grc} (durable marker engine/chain_gated_grpo.done written)"
  return 0
}

dispatch_wave3(){
  log "dispatch: WAVE3_GO.ok present -> gating P3 wave-3"
  if [ ! -f "$P3/run_wave3.sh" ]; then
    log "dispatch: WARN — $P3/run_wave3.sh does not exist yet (built by a parallel agent); retry next poll cycle"
    return 1
  fi
  idle_gate || { log "dispatch: wave3 idle-gate did not pass — retry next poll cycle"; return 1; }
  thermal_gate; local trc=$?
  [ "$trc" -eq 1 ] && return 1
  log "dispatch: launching run_wave3.sh (from $P3; it self-gates further on its own prereg/models/GO markers)"
  cd "$P3" || { log "dispatch: ABORT cannot cd to $P3"; return 1; }
  chmod +x run_wave3.sh 2>/dev/null
  nohup ./run_wave3.sh >> run_wave3.nohup.log 2>&1 &
  local wp=$!
  echo "$wp" > "$H/engine/chain_gated_wave3.pid"
  cd "$H" || exit 2
  log "dispatch: wave3 launched pid=${wp} (supervisor copy engine/chain_gated_wave3.pid)"
  wait "$wp"; local wrc=$?
  echo "$wrc" > "$H/engine/chain_gated_wave3.done"   # review fix 1: durable marker, written immediately after rc capture
  WAVE3_RC=$wrc
  WAVE3_LAUNCHED=1
  log "dispatch: wave3 finished rc=${wrc} (durable marker engine/chain_gated_wave3.done written)"
  return 0
}

# ---------------------------------------------------------------- stage 2: gated dispatch loop
stage2_dispatch_loop(){
  log "stage2: entering gated dispatch loop (poll ${POLL_SEC}s, deadline 24h from script start; review fix 3:"
  log "        the deadline is PAUSED for the duration of any dispatch_* call — DISPATCH_ELAPSED is added back"
  log "        onto it each poll, so an approved multi-hour job never starves a later approved one of window)"
  while :; do
    local now; now=$(date +%s)
    local deadline=$((START_EPOCH + 24*3600 + DISPATCH_ELAPSED))
    if [ "$now" -ge "$deadline" ]; then
      log "stage2: deadline reached (24h from script start + ${DISPATCH_ELAPSED}s paused for dispatch time) — stopping poll loop"
      return 0
    fi
    if [ "$GRPO_LAUNCHED" -eq 1 ] && [ "$WAVE3_LAUNCHED" -eq 1 ]; then
      log "stage2: both slots resolved (GRPO rc=${GRPO_RC}, wave3 rc=${WAVE3_RC}) — nothing left to gate on"
      return 0
    fi
    if [ "$GRPO_LAUNCHED" -eq 0 ] && [ -f "$H/engine/GRPO_GO.ok" ]; then
      local t0; t0=$(date +%s)
      if dispatch_grpo; then
        DISPATCH_ELAPSED=$((DISPATCH_ELAPSED + $(date +%s) - t0))
        continue
      else
        DISPATCH_ELAPSED=$((DISPATCH_ELAPSED + $(date +%s) - t0))
        sleep "$POLL_SEC"; continue
      fi
    fi
    # dispatch_grpo is fully synchronous (gate->launch->wait->rc) before returning, so GRPO
    # can never be "running" when this branch is reached — serial ordering falls out for free.
    if [ "$WAVE3_LAUNCHED" -eq 0 ] && [ -f "$P3/WAVE3_GO.ok" ]; then
      local t1; t1=$(date +%s)
      if dispatch_wave3; then
        DISPATCH_ELAPSED=$((DISPATCH_ELAPSED + $(date +%s) - t1))
        continue
      else
        DISPATCH_ELAPSED=$((DISPATCH_ELAPSED + $(date +%s) - t1))
        sleep "$POLL_SEC"; continue
      fi
    fi
    log "stage2: waiting — GRPO_GO.ok=$([ -f "$H/engine/GRPO_GO.ok" ] && echo present || echo absent)(launched=${GRPO_LAUNCHED}) WAVE3_GO.ok=$([ -f "$P3/WAVE3_GO.ok" ] && echo present || echo absent)(launched=${WAVE3_LAUNCHED}) dispatch_elapsed=${DISPATCH_ELAPSED}s"
    sleep "$POLL_SEC"
  done
}

# ---------------------------------------------------------------- main
stage0_wait_revins
run_stage1_dossier
stage2_dispatch_loop
write_report "normal exit"
log "ALL DONE"
