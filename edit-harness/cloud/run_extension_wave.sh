#!/bin/bash
# cloud/run_extension_wave.sh — ONE-COMMAND launcher for the 2026-07-11 extension wave
# (Track 1: 7-9B family-transfer battery, run_family_transfer.sh; Track 2: cross-arch
# causal seed gap-fills, run_extension_causal_seeds.sh). Mirrors cloud/run_cloud_wave.sh's
# driver-shard launcher EXACTLY (same launch_worker/wait_worker shape, same
# CUDA_VISIBLE_DEVICES pinning, same wait-by-PID contract) — a DELIBERATELY separate log/
# pid namespace (cloud/logs_ext/, not cloud/logs/) so this wave can coexist on the same
# box as a run_cloud_wave.sh invocation without colliding on card0.log/card0.pid.
#
# DESIGN: driver-level sharding across 2 cards for the per-card phase (Track 1 models +
# Track 2's single-GPU pythia rows), THEN a dual-card tensor-parallel phase for Track 2's
# neox20b rows (mirrors run_cloud_wave.sh's tp20b split: per-card work must fully drain
# before a TP phase claims both cards — see the `wait` subcommand below).
#
# Card assignment (balanced from run_family_transfer.sh/run_extension_causal_seeds.sh's
# own per-row minute estimates — see cloud/EXTENSION-WAVE-RUNBOOK.md for the full table):
#   card0: run_family_transfer.sh (FAMILY_MODELS="mistral7b gemma9b", the 2 heaviest
#          models, ~735min s0-only floor) — no Track-2 rows, so it can go straight to the
#          7-9B battery.
#   card1: run_family_transfer.sh (FAMILY_MODELS="qwen7b llama8binst", ~625min s0-only
#          floor) + run_extension_causal_seeds.sh (TRACK2_SCOPE=pythia, ~240min) —
#          Track-2's pythia rows are cheap and single-GPU, so they ride along on the
#          lighter card (~865min combined, ~130min/~15% heavier than card0 — same order
#          of imbalance run_cloud_wave.sh's own README accepts, "~3% apart" was a
#          different 6-driver mix; these two drivers' cost estimates carry real
#          uncertainty, see the runbook).
#   tp2 phase (AFTER both cards drain): run_extension_causal_seeds.sh TRACK2_SCOPE=neox,
#          CUDA_VISIBLE_DEVICES=0,1 together (~300min for the default Trimmed 1 gap-fill
#          seed; ~600min for 2 seeds under an explicit NEOX_SEEDS="1 2" Full opt-in — see
#          the runbook's cost-tier table; NEOX_SEEDS is read ONLY here, it is INERT if
#          set on the `both`/card0/card1 commands below).
#
# Pythia rows deliberately run on ONLY ONE card (card1) — running run_extension_causal_
# seeds.sh with TRACK2_SCOPE=pythia (or =all) on BOTH cards would re-execute the SAME
# --out paths concurrently, the exact B2 collision bug cloud/run_cloud_wave.sh's 07-08
# rework fixed for the original wave. Do not add pythia/all scope to card0's driver list.
#
# PHASE-0 (before the fan-out, see `both` below): run_family_transfer.sh runs on BOTH
# cards, and its Phase A (re-)derives the SHARED engine/r3_equiv_bf16.ok marker via a
# FIXED --out path if it's absent/stale — unlike the pythia/neox split above, driver-
# level sharding does NOT protect this one, since it's the SAME driver racing itself on
# both cards. `phase0_equiv_gate` (below) derives it once, single-card, before either
# worker starts — see that function's own header for the full rationale.
set -u
H="$(cd "$(dirname "$0")/.." && pwd)"
SELF="$H/cloud/$(basename "$0")"   # absolute self-path, resolved BEFORE cd — the `full`
# subcommand below re-invokes this script recursively via "$SELF"; a bare "$0" would break
# under some invocation styles once cwd has changed (e.g. `cd cloud && ./run_extension_
# wave.sh full` — after `cd "$H"` two lines down, a relative $0 no longer resolves).
cd "$H" || exit 1
mkdir -p cloud/logs_ext

DRIVERS_CARD0=${DRIVERS_CARD0:-"run_family_transfer.sh"}
DRIVERS_CARD1=${DRIVERS_CARD1:-"run_family_transfer.sh run_extension_causal_seeds.sh"}
FAMILY_MODELS_CARD0=${FAMILY_MODELS_CARD0:-"mistral7b gemma9b"}
FAMILY_MODELS_CARD1=${FAMILY_MODELS_CARD1:-"qwen7b llama8binst"}
TRACK2_SCOPE_CARD1=${TRACK2_SCOPE_CARD1:-pythia}
BUDGET_MIN_FAMILY=${BUDGET_MIN_FAMILY:-650}     # per-card, per run_family_transfer.sh invocation
BUDGET_MIN_EXTCAUSAL=${BUDGET_MIN_EXTCAUSAL:-300}  # card1's pythia-scope invocation (Track 2)
BUDGET_MIN_TP2=${BUDGET_MIN_TP2:-650}           # tp2 phase (neox20b, both cards, ~600min for 2 seeds)
DRYRUN=${DRYRUN:-0}
CLOUD_PY=${CLOUD_PY:-$(command -v python3 2>/dev/null || echo python3)}

log(){ echo "[run_extension_wave $(date '+%F %T')] $*"; }

launch_worker(){    # launch_worker <card> <driver-list-string>
  local card="$1"; shift
  local drivers="$*"
  local logf="cloud/logs_ext/card${card}.log" pidf="cloud/logs_ext/card${card}.pid"
  log "worker card=${card} drivers=[${drivers}] -> ${logf}"
  (
    export CUDA_VISIBLE_DEVICES="$card"
    export SKIP_IDLE_GATE=1
    export IDLE_GATE_DEVICE="$card"
    export CLOUD_PY
    export BUDGET_MIN="$BUDGET_MIN_FAMILY"
    if [ "$card" = "0" ]; then export FAMILY_MODELS="$FAMILY_MODELS_CARD0"
    else export FAMILY_MODELS="$FAMILY_MODELS_CARD1"; fi
    echo "[card${card}] START $(date '+%F %T') drivers=[${drivers}] pid=$$" >> "$logf"
    local d rc
    for d in $drivers; do
      if [ ! -f "$d" ]; then
        echo "[card${card}] SKIP ${d} (not present in repo root)" >> "$logf"
        continue
      fi
      # run_extension_causal_seeds.sh gets its OWN budget + scope (pythia only, card1 —
      # see header for why this never runs on card0)
      if [ "$d" = "run_extension_causal_seeds.sh" ]; then
        if [ "$card" != "1" ]; then
          echo "[card${card}] SKIP ${d} (Track-2 pythia scope is card1-only by design — see header)" >> "$logf"
          continue
        fi
        export TRACK2_SCOPE="$TRACK2_SCOPE_CARD1"
        export BUDGET_MIN="$BUDGET_MIN_EXTCAUSAL"
      fi
      echo "[card${card}] >>> ${d} START $(date '+%T')" >> "$logf"
      if [ "$DRYRUN" -eq 1 ]; then
        echo "[card${card}] DRYRUN would run: bash ${d} (FAMILY_MODELS=${FAMILY_MODELS:-} TRACK2_SCOPE=${TRACK2_SCOPE:-} BUDGET_MIN=${BUDGET_MIN})" >> "$logf"
      else
        bash "$d" >> "$logf" 2>&1
        rc=$?
        echo "[card${card}] <<< ${d} DONE rc=${rc} $(date '+%T')" >> "$logf"
      fi
      # restore family budget for any subsequent driver in this card's list
      export BUDGET_MIN="$BUDGET_MIN_FAMILY"
    done
    echo "[card${card}] ALL DONE $(date '+%F %T')" >> "$logf"
  ) &
  echo $! > "$pidf"
  log "worker card=${card} pid=$(cat "$pidf")"
}

wait_worker(){       # wait_worker <card> — waits by PID, NEVER pgrep/pkill
  local card="$1"
  local pidf="cloud/logs_ext/card${card}.pid"
  [ -f "$pidf" ] || return 0
  local pid; pid=$(cat "$pidf")
  while kill -0 "$pid" 2>/dev/null; do sleep 30; done
}

# phase0_equiv_gate — MUST run to completion BEFORE the card0/card1 fan-out. Both
# DRIVERS_CARD0 and DRIVERS_CARD1 include run_family_transfer.sh, and that driver's
# Phase A (re-)derives the SHARED, model-independent engine/r3_equiv_bf16.ok marker via
# a FIXED --out path (results/equiv_llama1b_bf16_L12_s0.json + its .npz) whenever the
# marker is absent/stale — no lock, no per-card path. Launching both cards straight into
# `both` on a fresh box (marker absent) would race that TOCTOU check and concurrently
# write the SAME npz — killgate's npz write is non-atomic, so this is a real corruption
# risk, not just wasted compute (the same class of B2 collision run_cloud_wave.sh's
# 07-08 rework fixed via disjoint driver sharding; here it's the SAME driver on both
# cards, so sharding alone doesn't help — the fix is to derive the shared marker once,
# single-card, before either worker starts). Mirrors the freshness check inside
# run_family_transfer.sh itself so this is a no-op (instant return) whenever the marker
# is already fresh — including on a box that already ran run_8bcausal.sh/run_neox20b.sh
# locally, which derive the identical marker.
phase0_equiv_gate(){
  if [ -f engine/r3_equiv_bf16.ok ] && [ -f results/matrices/equiv_llama1b_bf16_L12_s0.npz ] \
     && [ "$(stat -c %Y results/matrices/equiv_llama1b_bf16_L12_s0.npz)" -ge "$(stat -c %Y experiments/killgate_keygeom.py)" ] \
     && [ "$(stat -c %Y results/matrices/equiv_llama1b_bf16_L12_s0.npz)" -ge "$(stat -c %Y editors/arch_compat.py)" ]; then
    log "Phase-0 equiv gate: engine/r3_equiv_bf16.ok already fresh — skipping single-card derivation"
    return 0
  fi
  if [ ! -f run_family_transfer.sh ]; then
    log "Phase-0 equiv gate: run_family_transfer.sh not present — nothing to derive, both cards' Phase A will each CONFIG-skip safely"
    return 0
  fi
  # Named check for the 2 implicit inputs this step needs (see EXTENSION-WAVE-RUNBOOK.md
  # "Also required" section) — a git-clone box (results/ and data/models/ gitignored)
  # would otherwise hard-abort inside run_family_transfer.sh's own preflight with a less
  # specific message; naming them here up front makes a mis-provisioned box obvious.
  [ -d data/models/Llama-3.2-1B ] || log "Phase-0 equiv gate: WARNING data/models/Llama-3.2-1B absent — was this box provisioned by rsync (not git-clone)? Phase A will hard-abort its own preflight."
  [ -f results/matrices/gate_llama1b_rome_cf_L12_s0.npz ] || log "Phase-0 equiv gate: WARNING results/matrices/gate_llama1b_rome_cf_L12_s0.npz absent — was this box provisioned by rsync (not git-clone)? Phase A will hard-abort its own preflight."
  if [ "$DRYRUN" -eq 1 ]; then
    log "Phase-0 equiv gate: DRYRUN — would run: CUDA_VISIBLE_DEVICES=0 EQUIV_GATE_ONLY=1 bash run_family_transfer.sh (single-card, before fan-out)"
    return 0
  fi
  log "Phase-0 equiv gate: marker absent/stale — deriving ONCE on card0 before fan-out (log cloud/logs_ext/phase0_equiv.log)"
  (
    export CUDA_VISIBLE_DEVICES=0
    export SKIP_IDLE_GATE=1
    export IDLE_GATE_DEVICE=0
    export CLOUD_PY
    export EQUIV_GATE_ONLY=1
    bash run_family_transfer.sh
  ) >> cloud/logs_ext/phase0_equiv.log 2>&1
  local rc=$?
  if [ -f engine/r3_equiv_bf16.ok ]; then
    log "Phase-0 equiv gate: DONE (rc=${rc}), marker written — fan-out proceeds, both cards' Phase A will skip it"
  else
    log "Phase-0 equiv gate: DONE (rc=${rc}) but marker still ABSENT (gate FAILED) — fan-out proceeds anyway; both cards' bf16 rows CONFIG-skip cleanly (non-fatal)"
  fi
}

# zero-new-results guard for the `full` subcommand (mirrors run_enhance_4090.sh's
# count_round_results/shutdown pattern) — pure bash arithmetic, no `bc` dependency
count_round_results(){
  local a b
  a=$(ls results/gate_mistral7b_*.json results/gate_qwen7b_*.json results/gate_gemma9b_*.json \
      results/gate_llama8binst_*.json results/g4_mistral7b_*.json results/g4_qwen7b_*.json \
      results/g4_gemma9b_*.json results/g4_llama8binst_*.json 2>/dev/null | wc -l)
  b=$(ls results/g4_pythia14b_alphaHO_cf_L6_s1.json results/g4_pythia14b_alphaHO_cf_L6_s2.json \
      results/g4_pythia28b_alphaHO_cf_L8_s1.json results/g4_pythia28b_alphaHO_cf_L8_s2.json \
      results/g4_neox20b_alphaHO_cf_L16_s1.json results/g4_neox20b_alphaHO_cf_L16_s2.json 2>/dev/null | wc -l)
  echo $(( a + b ))
}

case "${1:-both}" in
  both)
    phase0_equiv_gate
    launch_worker 0 $DRIVERS_CARD0
    launch_worker 1 $DRIVERS_CARD1
    log "both workers launched (Track 1 family battery, sharded card0={${FAMILY_MODELS_CARD0}} /"
    log "card1={${FAMILY_MODELS_CARD1}}; Track-2 pythia rows ride on card1)"
    log "tail progress: tail -f cloud/logs_ext/card0.log cloud/logs_ext/card1.log"
    log "wait for both:  bash cloud/run_extension_wave.sh wait"
    log "AFTER wait completes, run the dual-card TP phase: bash cloud/run_extension_wave.sh tp2"
    ;;
  card0) launch_worker 0 "${2:-$DRIVERS_CARD0}" ;;
  card1) launch_worker 1 "${2:-$DRIVERS_CARD1}" ;;
  wait) wait_worker 0; wait_worker 1; log "both workers finished" ;;
  tp2)
    # Track-2 neox20b causal seeds s1/s2 — spans BOTH cards (tensor-parallel), mirrors
    # run_cloud_wave.sh's tp20b. Run AFTER `wait` above (needs both 4090s free).
    if [ -f run_extension_causal_seeds.sh ]; then
      log "tp2 phase starting (neox20b L16 s1/s2, log cloud/logs_ext/tp2.log)"
      (
        export CUDA_VISIBLE_DEVICES="0,1"
        export TRACK2_SCOPE=neox
        export BUDGET_MIN="$BUDGET_MIN_TP2"
        export CLOUD_PY
        if [ "$DRYRUN" -eq 1 ]; then
          echo "DRYRUN would run: TRACK2_SCOPE=neox bash run_extension_causal_seeds.sh" >> cloud/logs_ext/tp2.log
        else
          bash run_extension_causal_seeds.sh >> cloud/logs_ext/tp2.log 2>&1
        fi
      ) &
      echo $! > cloud/logs_ext/tp2.pid
      log "tp2 pid=$(cat cloud/logs_ext/tp2.pid)"
    else
      log "run_extension_causal_seeds.sh not present — nothing to launch"
    fi
    ;;
  wait_tp2)
    [ -f cloud/logs_ext/tp2.pid ] || { log "no tp2.pid — tp2 phase not launched?"; exit 0; }
    tp2_pid=$(cat cloud/logs_ext/tp2.pid)
    while kill -0 "$tp2_pid" 2>/dev/null; do sleep 30; done
    log "tp2 phase finished"
    ;;
  full)
    # one-command: both -> wait -> tp2 -> wait_tp2 -> zero-new-results shutdown guard
    # (pairs with cloud/failsafe_extension.sh's independent hard power-off timer; cancel
    # either with `touch /root/NO_SHUTDOWN` before this reaches its own shutdown check)
    n0=$(count_round_results)
    log "round-result census (pre-existing files before this run, e.g. rsync'd from a prior round): ${n0}"
    "$SELF" both
    "$SELF" wait
    "$SELF" tp2
    "$SELF" wait_tp2
    n1=$(count_round_results)
    n=$(( n1 - n0 ))
    log "round-result census (NEW gap-fill files produced this run): ${n} (before=${n0} after=${n1})"
    if [ "$n" -le 0 ]; then
      log "ZERO NEW RESULTS — NOT shutting down; box stays up for diagnosis"
      exit 1
    fi
    log "results present — powering off in 300s (cancel: touch /root/NO_SHUTDOWN)"
    sync
    sleep 300
    if [ -f /root/NO_SHUTDOWN ]; then log "shutdown CANCELLED by /root/NO_SHUTDOWN"; exit 0; fi
    if [ "$DRYRUN" -eq 1 ]; then log "DRYRUN — would shutdown -h now"; exit 0; fi
    shutdown -h now
    ;;
  *) echo "usage: bash cloud/run_extension_wave.sh {both|card0 [drivers]|card1 [drivers]|wait|tp2|wait_tp2|full}"; exit 1 ;;
esac
