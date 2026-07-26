#!/bin/bash
# chain_local_20260716.sh — LOCAL-FIRST master sequencer (2026-07-16). Runs, in strict serial
# order on the single local 5090, every pending experiment whose user gate is already open at
# the moment this chain reaches it, and skips (never blocks on) whatever isn't ratified yet:
#
#   (a) wait for the LIVE MIX_A wave to drain (results/frame_a/cells/cell_*_MIX_A_*.json reaching
#       33 = 3 seeds x 11 policies, per experiments/frame_a/config.py's SEEDS=(0,1,2) and
#       run_stream.py's POLICIES tuple — OR the recorded MIX_A pid going dead, whichever first).
#   (b) print (NOT apply) the engine/PATCH-smoke-marker-ordering-20260716.md reminder — that
#       patch touches experiments/frame_a/run_stream.py, which the LIVE wave imports; applying
#       code while a queue imports it is a standing hazard in this workspace (see memory
#       "live-file-edit-hazard-under-running-queue"). A human applies it by hand, then re-runs
#       SMOKE=1 ./run_frame_a_wave1.sh before trusting any FUTURE frame_a real-wave launch.
#   (c) run_esr_probe_gpt2xl.sh (self-contained: own idle gate + timeout).
#   (d) IF engine/FRAMEA_LOCAL_BC.ok exists: run MIX_B then MIX_C locally, same invocation
#       pattern as the live MIX_A wave. Gated on a file (not auto-detected) because resume is
#       only filename-idempotent for CELLS THIS MACHINE ALREADY WROTE — if a cloud box is also
#       computing MIX_B/MIX_C, a local re-run duplicates GPU-hours for cells that will land from
#       the box anyway; the gate file is the human's signal that the box is down and local is the
#       only source.
#   (e) IF engine/RE_GO.ok exists AND its content is exactly "solo" or "base": launch R-E
#       (experiments/prospective_admission.py) with that --ns_reference value, per
#       docs/plans/PREREG-PROSPECTIVE-ADMISSION-DRAFT-2026-07-16.md's Launch section.
#   (f) IF engine/PAPERB_GO.ok exists: run ./run_paperb_phase1.sh.
#
# Every real-GPU step is setsid-launched with its OWN pidfile+log under engine/ (so a human can
# `kill -0`/kill an individual stuck step without touching the chain), and the chain only
# advances past a step once that step's own process has exited with rc=0 — one GPU, fully serial,
# never two steps live at once. Kill-by-PID ONLY (kill -0 to check liveness; never
# pgrep/pkill -f a pattern — a watcher's own command line contains the script names it would
# match, causing self-match deadlock/self-kill; see workspace memory).
#
# LID-OPEN REMINDER: keep the laptop lid OPEN for the entire duration this chain runs (closing it
# wedges nvidia_uvm under GPU load — memory "gpu-lid-close-nvidia-uvm-wedge").
#
# BUILD-ONLY as authored 2026-07-16: bash -n clean; NOT launched by the author. Author only
# verified individual referenced scripts/gate-files exist or are correctly treated as absent.
set -u
H="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$H" || exit 2
PY=${PY:-python3}
PIDFILE=engine/chain_local_20260716.pid
LOG=engine/chain_local_20260716.log
mkdir -p engine results/frame_a/cells
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
say(){ echo "$*"; log "$*"; }

# ---------------------------------------------------------------- double-launch refuse guard
if [ -f "$PIDFILE" ]; then
  oldpid=$(cat "$PIDFILE" 2>/dev/null)
  if [ -n "$oldpid" ] && kill -0 "$oldpid" 2>/dev/null; then
    echo "REFUSE: chain_local_20260716 already running (pid $oldpid)" >&2
    exit 7
  fi
fi
echo $$ > "$PIDFILE"
say "================ CHAIN_LOCAL_20260716 START pid=$$ ================"
say "LID-OPEN REMINDER: keep the laptop lid open for the whole run (nvidia_uvm wedge on close)."

ENVP="env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1"

# ---------------------------------------------------------------- shared helpers
# gpu_idle_gate: util<25 && mem<1500 x3 consecutive polls, pinned to THIS card via `nvidia-smi -i`
# (nvidia-smi ignores CUDA_VISIBLE_DEVICES otherwise — memory "gpu-idle-gate-not-zero-compute-apps"
# + the run_merging_editors.sh/run_esr_probe_gpt2xl.sh convention this mirrors).
gpu_idle_gate(){
  local consec=0 GPU_ID
  GPU_ID=${CUDA_VISIBLE_DEVICES%%,*}; GPU_ID=${GPU_ID:-0}
  while [ "$consec" -lt 3 ]; do
    local line util mem
    line=$(nvidia-smi -i "$GPU_ID" --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    util=$(echo "$line" | awk -F, '{gsub(/[^0-9]/,"",$1); print $1}')
    mem=$(echo "$line"  | awk -F, '{gsub(/[^0-9]/,"",$2); print $2}')
    if [ -n "$util" ] && [ -n "$mem" ] && [ "$util" -lt 25 ] && [ "$mem" -lt 1500 ]; then
      consec=$((consec+1))
    else
      consec=0
    fi
    log "gpu poll util=${util:-NA} mem=${mem:-NA} consec=${consec}/3"
    [ "$consec" -lt 3 ] && sleep 20
  done
}

# run_step NAME cmd... : setsid-launch `cmd...`, own pidfile+log under engine/, BLOCK (serial —
# one GPU) until it exits, return its rc. The chain only proceeds past a step once this returns 0.
run_step(){
  local name="$1"; shift
  local pidfile="engine/chain_local_20260716_${name}.pid"
  local steplog="engine/chain_local_20260716_${name}.log"
  say "STEP ${name} START: $* (log ${steplog}, pidfile ${pidfile})"
  setsid "$@" >> "$steplog" 2>&1 &
  local pid=$!
  echo "$pid" > "$pidfile"
  wait "$pid"
  local rc=$?
  say "STEP ${name} END rc=${rc} (pid ${pid})"
  return $rc
}

abort_on_fail(){
  local name="$1" rc="$2"
  if [ "$rc" -ne 0 ]; then
    say "ABORT: step ${name} failed rc=${rc} — chain halts (serial gate, no further steps run)"
    say "================ CHAIN_LOCAL_20260716 END (ABORTED at ${name}) ================"
    exit "$rc"
  fi
}

# ================================================================ (a) wait for MIX_A drain
say "---- (a) waiting for the live MIX_A wave (target 33 cells = 3 seeds x 11 policies) ----"
# The wave can be alive under EITHER pidfile: run_frame_a_wave1.pid (the 2026-07-18 relaunch
# path — run_frame_a_wave1.sh writes ONLY this) or frame_a_mixa_local.pid (the older direct
# launch). Watching only the latter saw a stale dead pid and proceeded while the wave was
# still running — fixed 2026-07-18 (hostile review MAJOR-1).
MIXA_PIDFILES=("engine/run_frame_a_wave1.pid" "engine/frame_a_mixa_local.pid")
mixa_alive(){
  local pf mpid
  for pf in "${MIXA_PIDFILES[@]}"; do
    [ -f "$pf" ] || continue
    mpid=$(cat "$pf" 2>/dev/null | tr -d '[:space:]')
    if [ -n "$mpid" ] && kill -0 "$mpid" 2>/dev/null; then return 0; fi
  done
  return 1
}
while true; do
  n=$(ls results/frame_a/cells/cell_*_MIX_A_*.json 2>/dev/null | wc -l)
  if [ "$n" -ge 33 ]; then
    say "(a) MIX_A cell count reached ${n}/33 — drained"
    break
  fi
  if ! mixa_alive; then
    say "(a) WARN: no live MIX_A pid in ${MIXA_PIDFILES[*]} with only ${n}/33 cells present — "\
"proceeding anyway per the chain's OR-condition (a future MIX_A re-run would fill the gap; "\
"not this chain's job to force it)."
    break
  fi
  log "(a) waiting: ${n}/33 MIX_A cells, wave alive"
  sleep 60
done

# ================================================================ (b) manual-patch reminder (NOT applied)
say "---- (b) REMINDER (manual step, NOT auto-applied) ----"
say "  engine/PATCH-smoke-marker-ordering-20260716.md describes a fix to the SMOKE_PASS.ok"
say "  marker-write ordering in experiments/frame_a/run_stream.py — a file the (just-drained)"
say "  live wave imports, so this chain deliberately does NOT edit it for you. Apply the patch"
say "  by hand when convenient, then re-run: SMOKE=1 ./run_frame_a_wave1.sh"
say "  (this changes the frame_a code checksum, so the NEXT real-wave launch needs a fresh smoke"
say "  pass by design). Continuing the chain now WITHOUT applying the patch."

# ================================================================ (c) ESR probe (gpt2-xl)
say "---- (c) run_esr_probe_gpt2xl.sh ----"
if [ -f "results/esr_probe_gpt2xl/esr_by_layer.json" ]; then
  say "(c) SKIP: results/esr_probe_gpt2xl/esr_by_layer.json already present"
else
  run_step esr_probe_gpt2xl ./run_esr_probe_gpt2xl.sh
  rc=$?; abort_on_fail esr_probe_gpt2xl "$rc"
fi

# ================================================================ (d) MIX_B / MIX_C (gated)
if [ -f "engine/FRAMEA_LOCAL_BC.ok" ]; then
  say "---- (d) engine/FRAMEA_LOCAL_BC.ok present — running MIX_B then MIX_C locally ----"
  run_step frame_a_mixb $ENVP $PY -m experiments.frame_a.run_stream --run --real \
    --mixes MIX_B --model_dir data/models/Llama-3.2-1B
  rc=$?; abort_on_fail frame_a_mixb "$rc"
  run_step frame_a_mixc $ENVP $PY -m experiments.frame_a.run_stream --run --real \
    --mixes MIX_C --model_dir data/models/Llama-3.2-1B
  rc=$?; abort_on_fail frame_a_mixc "$rc"
else
  say "---- (d) SKIP: engine/FRAMEA_LOCAL_BC.ok absent (MIX_B/MIX_C not gated open) ----"
fi

# ================================================================ (e) R-E prospective admission (gated)
if [ -f "engine/RE_GO.ok" ]; then
  re_val=$(tr -d '[:space:]' < engine/RE_GO.ok)
  case "$re_val" in
    solo|base)
      say "---- (e) engine/RE_GO.ok = '${re_val}' — running R-E prospective_admission ----"
      gpu_idle_gate
      run_step prospective_admission $ENVP $PY experiments/prospective_admission.py \
        --model data/models/Llama-3.2-1B --layer 12 --data data/counterfact.json \
        --n_pool 100 --budget 0.25 --group_size 5 --n_retention 200 --n_random_draws 3 \
        --ns_reference "$re_val" --seeds 0,1,2 --steps 20 --lr 0.1 --device cuda \
        --out_dir results/prospective_admission
      rc=$?; abort_on_fail prospective_admission "$rc"
      ;;
    *)
      say "---- (e) SKIP: engine/RE_GO.ok present but content '${re_val}' is not 'solo' or 'base' ----"
      ;;
  esac
else
  say "---- (e) SKIP: engine/RE_GO.ok absent ----"
fi

# ================================================================ (f) Paper B Phase-1 (gated)
if [ -f "engine/PAPERB_GO.ok" ]; then
  if [ -f "./run_paperb_phase1.sh" ]; then
    say "---- (f) engine/PAPERB_GO.ok present — running run_paperb_phase1.sh ----"
    run_step paperb_phase1 ./run_paperb_phase1.sh
    rc=$?; abort_on_fail paperb_phase1 "$rc"
  else
    say "---- (f) WARN: engine/PAPERB_GO.ok present but run_paperb_phase1.sh does not exist yet"\
" (Paper B Phase-1 prereg/driver not landed at chain-authoring time) — SKIPPING ----"
  fi
else
  say "---- (f) SKIP: engine/PAPERB_GO.ok absent ----"
fi

say "================ CHAIN_LOCAL_20260716 END (all gated/available steps complete) ================"
touch engine/chain_local_20260716.done
exit 0
