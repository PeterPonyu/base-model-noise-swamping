#!/bin/bash
# cloud/run_cloud_wave.sh — ONE-COMMAND launcher for the AutoDL dual-4090 driver-shard
# wave (2026-07-08, re-sharded post wave-review). Run ON the cloud box, from anywhere
# (this script cd's to its own repo root), AFTER cloud/setup_autodl.sh has downloaded
# models and both GPUs are attached.
#
# DESIGN: static DRIVER-level sharding, NOT seed-sharding, NOT a dynamic atomic queue.
# Two workers, one per physical card, each running a DISJOINT subset of the 6 drivers
# pinned via CUDA_VISIBLE_DEVICES. Every driver already sweeps its own seeds 0/1/2
# internally (see e.g. run_mquake_law.sh's gate_llama1b_rome_mquake_L8_s{0,1,2} rows) —
# since each driver now runs on exactly ONE card, that full 3-seed sweep happens once,
# not duplicated, and no two workers ever write the same --out/npz path concurrently
# (the wave-review B2 fix: the old design had BOTH cards run ALL 6 drivers via a
# SEED_OVERRIDE knob that no driver actually read, so both cards executed identical
# --out paths in parallel — killgate's npz write is non-atomic, so that was a live
# corruption risk, not just wasted compute). Full runbook + shard map: cloud/README.md.
#
# SEED_OVERRIDE is exported for forward-compat but is INERT against every driver in
# this repo (none read it) — do not rely on it; driver-level sharding is what actually
# prevents concurrent-write collisions now.
set -u
H="$(cd "$(dirname "$0")/.." && pwd)"
cd "$H" || exit 1
mkdir -p cloud/logs

# Driver -> card assignment, balanced by each driver's own SCIENCE+SMOKE minute
# estimates summed from its run_row calls (card0 723 min / card1 748 min, ~3% apart).
# card0 deliberately carries the heaviest single driver (run_8bcausal.sh, the 8B-model
# cells) paired with two light ones; card1 carries the two mid-weight MQuAKE drivers
# (mquake_law + mquake_t, ~350 min each) balanced by the smallest driver (glue_seq).
# DRIVERS (no _CARDn suffix) is a back-compat/testing override: if set, it applies to
# BOTH cards uniformly (this is what cloud/selftest.sh uses with the fake driver) —
# DRIVERS_CARD0/DRIVERS_CARD1 take priority over it when set individually.
DRIVERS_CARD0=${DRIVERS_CARD0:-${DRIVERS:-"run_8bcausal.sh run_ripple.sh run_cfplus.sh"}}
DRIVERS_CARD1=${DRIVERS_CARD1:-${DRIVERS:-"run_mquake_law.sh run_mquake_t.sh run_glue_seq.sh"}}
SEED_CARD0=${SEED_CARD0:-1}   # inert (see header) — kept only as a forward-compat knob
SEED_CARD1=${SEED_CARD1:-2}   # inert (see header) — kept only as a forward-compat knob
DRYRUN=${DRYRUN:-0}
# B4 fix (2026-07-08): every driver hardcodes a local conda path for PY; on the box
# neither that path nor the local repo's H exists. CLOUD_PY, once exported, is honored
# by the 3 WP2 drivers + run_neox20b.sh directly, and by the 3 chain-locked drivers once
# `setup_autodl.sh patch-drivers` has run on the box (see cloud/patch_h_py.sed). Default
# to the box's own python3 on PATH; override with CLOUD_PY=<path> if it's not on PATH.
CLOUD_PY=${CLOUD_PY:-$(command -v python3 2>/dev/null || echo python3)}

log(){ echo "[run_cloud_wave $(date '+%F %T')] $*"; }

# OPTIONAL polish (wave-review): DRIVERS applies to BOTH cards uniformly — safe for
# cloud/selftest.sh's fake driver, but if pointed at real science drivers it reintroduces
# the exact B2 collision this rework fixed (both cards executing the same --out paths
# concurrently). Warn once at load time so it's not a silent footgun.
if [ -n "${DRIVERS:-}" ]; then
  case "$DRIVERS" in
    *selftest_fake_driver.sh*) : ;;
    *) log "WARNING: DRIVERS='${DRIVERS}' is set — applies to BOTH cards uniformly. If" \
            "these are real science drivers (not the selftest fake driver), both cards" \
            "will run the SAME driver(s) concurrently on the SAME --out paths — the B2" \
            "collision bug this rework fixed. Prefer DRIVERS_CARD0/DRIVERS_CARD1 instead." ;;
  esac
fi

launch_worker(){    # launch_worker <card> <seed> <driver-list-string>
  local card="$1" seed="$2"; shift 2
  local drivers="$*"
  local logf="cloud/logs/card${card}.log" pidf="cloud/logs/card${card}.pid"
  log "worker card=${card} drivers=[${drivers}] -> ${logf}"
  (
    export CUDA_VISIBLE_DEVICES="$card"
    export SEED_OVERRIDE="$seed"          # inert against today's drivers — see header
    export SKIP_IDLE_GATE=1               # dedicated box: safe default per README
    export IDLE_GATE_DEVICE="$card"       # honored only by gpu_idle_lib.sh-sourcing drivers
    export CLOUD_PY                       # B4 fix — box python, see header
    echo "[card${card}] START $(date '+%F %T') drivers=[${drivers}] pid=$$" >> "$logf"
    local d rc
    for d in $drivers; do
      if [ ! -f "$d" ]; then
        echo "[card${card}] SKIP ${d} (not present in repo root — WP2/WP3 pending?)" >> "$logf"
        continue
      fi
      echo "[card${card}] >>> ${d} START $(date '+%T')" >> "$logf"
      if [ "$DRYRUN" -eq 1 ]; then
        echo "[card${card}] DRYRUN would run: bash ${d}" >> "$logf"
      else
        bash "$d" >> "$logf" 2>&1
        rc=$?
        echo "[card${card}] <<< ${d} DONE rc=${rc} $(date '+%T')" >> "$logf"
      fi
    done
    echo "[card${card}] ALL DONE $(date '+%F %T')" >> "$logf"
  ) &
  echo $! > "$pidf"
  log "worker card=${card} pid=$(cat "$pidf")"
}

wait_worker(){       # wait_worker <card> — waits by PID, NEVER pgrep/pkill
  local card="$1"
  local pidf="cloud/logs/card${card}.pid"
  [ -f "$pidf" ] || return 0
  local pid; pid=$(cat "$pidf")
  while kill -0 "$pid" 2>/dev/null; do sleep 30; done
}

case "${1:-both}" in
  both)
    launch_worker 0 "$SEED_CARD0" $DRIVERS_CARD0
    launch_worker 1 "$SEED_CARD1" $DRIVERS_CARD1
    log "both workers launched — driver sets are disjoint (see DRIVERS_CARD0/DRIVERS_CARD1"
    log "above) so they no longer need to start in lockstep for gate safety; SKIP_IDLE_GATE=1"
    log "is still exported as the dedicated-box default (see cloud/README.md)"
    log "tail progress: tail -f cloud/logs/card0.log cloud/logs/card1.log"
    log "wait for both:  bash cloud/run_cloud_wave.sh wait"
    ;;
  card0) launch_worker 0 "${2:-$SEED_CARD0}" ${3:-$DRIVERS_CARD0} ;;
  card1) launch_worker 1 "${2:-$SEED_CARD1}" ${3:-$DRIVERS_CARD1} ;;
  wait) wait_worker 0; wait_worker 1; log "both workers finished" ;;
  tp20b)
    # Tensor-parallel 20B phase — spans BOTH cards together. Run AFTER the driver-shard
    # phase above fully finishes (it needs both 4090s free, not one each).
    if [ -f run_neox20b.sh ]; then
      log "TP 20B phase starting (log cloud/logs/tp20b.log)"
      export CUDA_VISIBLE_DEVICES="0,1"
      if [ "$DRYRUN" -eq 1 ]; then
        log "DRYRUN would run: bash run_neox20b.sh"
      else
        bash run_neox20b.sh >> cloud/logs/tp20b.log 2>&1 &
        echo $! > cloud/logs/tp20b.pid
        log "tp20b pid=$(cat cloud/logs/tp20b.pid)"
      fi
    else
      log "run_neox20b.sh not present yet (WP3 pending) — nothing to launch"
    fi
    ;;
  *) echo "usage: bash cloud/run_cloud_wave.sh {both|card0 [seed] [drivers]|card1 [seed] [drivers]|wait|tp20b}"; exit 1 ;;
esac
