#!/usr/bin/env bash
# run_frame_a_wave1.sh — Frame-A (Paper A) FIRST-WAVE launch driver.
#
# Wave-1 scope (PREREG-FRAME-A-STREAM-2026-07-16, rev.4): STREAM-v1 @ 500 updates, all 5 arms,
# 3 mixes × 3 seeds, the frozen P1–P4 gate, on the LOCAL 1B slice (+ 3B recalibration when the
# card window opens). House pattern mirrored from run_revins.sh: CPU preflight, MANDATORY
# self-test gate, GPU idle gate (util<25 && mem<1500 ×3), DRYRUN plan mode, per-tag logs + a
# single pidfile, kill-by-PID only (never pgrep/pkill -f — self-match), BUDGET_MIN window.
#
# BUILD-ONLY status: authored + `bash -n` + DRYRUN + SYNTH end-to-end validated; NOT launched.
# Modes:
#   DRYRUN=1                → print the planned cells + gate steps, touch no GPU, exit.
#   SYNTH=1                 → run the fully-wired SYNTHETIC-model pipeline on CPU (builds cells +
#                             emits the P1–P4 verdict). Proves the orchestration end-to-end with
#                             no GPU. This is what a build-only validation runs.
#   SMOKE=1                 → LAUNCH GATE: CPU gates → GPU idle gate → a REAL 5-update micro-stream
#                             on the 1B model that fires all four asserts (a)-(d); exits. Run this
#                             FIRST (after the hostile review) before the full real wave.
#   (default, real wave)    → CPU self-test gate → GPU idle gate → `run_stream --run --real` per
#                             model (loads the model, MEASURED collateral, per-cell base restore).
set -u
H=/home/zeyufu/Desktop/idea-feasibility-analysis/edit-harness
cd "$H" || exit 1
PY=/home/zeyufu/miniconda3/envs/dl/bin/python3
LOG=engine/run_frame_a_wave1.log
BUDGET_MIN=${BUDGET_MIN:-600}
DRYRUN=${DRYRUN:-0}
SYNTH=${SYNTH:-0}
SMOKE=${SMOKE:-0}
MODELS=${MODELS:-"llama-3.2-1b"}      # add "llama-3.2-3b" when the Pro-6000 card window opens.
MODEL_DIR=${MODEL_DIR:-data/models/Llama-3.2-1B}   # on-disk model for the real wave / smoke.
SYNTH_MODEL=$(echo "$MODELS" | awk '{print $1}')   # provenance/model tag for the SYNTH validation.
mkdir -p engine results/frame_a/cells results/frame_a/cells_synth
echo $$ > engine/run_frame_a_wave1.pid
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
say(){ echo "$*"; log "$*"; }
say "================ RUN_FRAME_A_WAVE1 START (pid $$, budget ${BUDGET_MIN}m, DRYRUN=$DRYRUN SYNTH=$SYNTH SMOKE=$SMOKE) ================"

# ---------------------------------------------------------------- Phase 0a: CPU preflight (code/asset presence only)
pf_fail=0
pf(){ if eval "$2"; then log "preflight OK: $1"; else say "PREFLIGHT-FAIL: $1"; pf_fail=1; fi; }
pf "python env (torch,numpy)"        "$PY -c 'import torch,numpy' 2>/dev/null"
pf "frame_a config importable"       "$PY -c 'from experiments.frame_a import config' 2>/dev/null"
pf "stream_builder present"          "[ -f experiments/frame_a/stream_builder.py ]"
pf "router present"                  "[ -f experiments/frame_a/router.py ]"
pf "cost_harness present"            "[ -f experiments/frame_a/cost_harness.py ]"
pf "scorer/analyze present"          "[ -f experiments/frame_a/scorer/analyze_frame_a.py ]"
pf "run_stream present"              "[ -f experiments/frame_a/run_stream.py ]"
pf "arms package present"            "[ -f experiments/frame_a/arms/base.py ]"
pf "real backends present"           "[ -f experiments/frame_a/arms/real_backends.py ]"
pf "editors present"                 "[ -f editors/rome_native.py ] && [ -f editors/grace_editor.py ] && [ -f editors/memit.py ] && [ -f editors/alphaedit.py ]"
pf "datasets present"                "[ -f data/counterfact.json ] && [ -f data/zsre_eval.json ] && [ -f data/mquake_cf3k.json ] && [ -f data/rippleedits/popular.json ]"
pf "gt_damage L12 cells present"     "ls results/matrices/gate_llama1b_rome_cf_L12_s*.npz >/dev/null 2>&1"
pf "model Llama-3.2-1B"              "[ -d data/models/Llama-3.2-1B ]"
if [ "$pf_fail" -ne 0 ]; then say "ABORT: CPU preflight failed"; exit 3; fi

# ---------------------------------------------------------------- Phase 0b: MANDATORY self-test gate (CPU; blocks the wave)
say "running mandatory self-test gate (experiments.frame_a.selftest) ..."
if ! $PY -m experiments.frame_a.selftest >> "$LOG" 2>&1; then
  say "ABORT: self-test gate RED — wave BLOCKED (see $LOG)"; exit 4
fi
say "self-test gate GREEN"
say "running namespacing/provenance gate (experiments.frame_a.selftest_namespacing) ..."
if ! $PY -m experiments.frame_a.selftest_namespacing >> "$LOG" 2>&1; then
  say "ABORT: namespacing gate RED — wave BLOCKED (see $LOG)"; exit 4
fi
say "namespacing gate GREEN"
say "running dead-arm gate (experiments.frame_a.check_dead_arms --refuse_verdict_if_fail) ..."
# The dead-arm gate protects against re-occurrence of the 2026-07-19 FT-arm bug: routes
# flowed into FT, the gate never fired, the merge never ran, and Q landed at the 0.3 floor
# for always_ft / ft_merge / cost_only / random. This gate scans existing cells for that exact
# signature (>=K routed to FT but install_gpu_s=0.0) and refuses to publish a clean verdict
# until every cell passes. On a fresh run the cells dir is empty so the gate is vacuously green.
if ! $PY -m experiments.frame_a.check_dead_arms --refuse_verdict_if_fail >> "$LOG" 2>&1; then
  say "ABORT: dead-arm gate RED — at least one cell shows the FT-no-op signature; rerun contaminated cells before publishing a verdict (see $LOG)"
  exit 7
fi
say "dead-arm gate GREEN"

# ---------------------------------------------------------------- Phase 1: plan the cells
plan_cells(){
  for M in $MODELS; do
    for mix in MIX_A MIX_B MIX_C; do
      for s in 0 1 2; do echo "cell ${M} ${mix} s${s} (11 policies)"; done
    done
  done
}
say "planned cells:"; plan_cells | while read -r l; do say "  $l"; done

if [ "$DRYRUN" -eq 1 ]; then
  say "DRYRUN=1 — full real-wave plan (no GPU touched):"
  say "  LAUNCH GATE (run first):"
  say "    $PY -m experiments.frame_a.run_stream --run --real --smoke --model_dir $MODEL_DIR"
  say "    → real 5-update micro-stream on the 1B; asserts (a)build (b)build (c)ΔW-parity (d)live-key_cos must all FIRE+PASS"
  say "  FULL WAVE (per model, after the gate passes):"
  for M in $MODELS; do
    say "    $PY -m experiments.frame_a.run_stream --run --real --model $M --model_dir $MODEL_DIR --out_cells results/frame_a/cells"
    say "      → 9 stream instances × 11 policies; fp32 edits; per-cell base-weight RESTORE; measured A_upd/A_loc/cost"
  done
  say "  ANALYZE (per model): analyze_frame_a --cells_dir results/frame_a/cells --expect_provenance real --expect_model <M>"
  say "DRYRUN complete."; exit 0
fi

# ---------------------------------------------------------------- SYNTH mode: full CPU pipeline (no GPU gate)
if [ "$SYNTH" -eq 1 ]; then
  say "SYNTH=1 — running the synthetic-model pipeline end-to-end (CPU) ..."
  # SYNTH cells live in their OWN dir (provenance=synth) so they can never mix with real cells;
  # analyze is pinned to the SYNTH model+provenance (MAJOR-2 guard refuses any stray cell).
  $PY -m experiments.frame_a.run_stream --run --synthetic --model "$SYNTH_MODEL" \
    --out_cells results/frame_a/cells_synth >> "$LOG" 2>&1 \
    || { say "ABORT: synthetic wave failed"; exit 5; }
  $PY -m experiments.frame_a.scorer.analyze_frame_a --cells_dir results/frame_a/cells_synth \
    --expect_model "$SYNTH_MODEL" --expect_provenance synth \
    --out results/frame_a/frame_a_verdict_synth.json >> "$LOG" 2>&1 \
    || { say "ABORT: analyze failed"; exit 6; }
  V=$($PY -c "import json;print(json.load(open('results/frame_a/frame_a_verdict_synth.json'))['VERDICT'])")
  say "SYNTH pipeline done — VERDICT=$V (synthetic fixture; real verdict comes from the GPU wave)"
  exit 0
fi

# ---------------------------------------------------------------- Phase 2: GPU idle gate (real wave only)
gate_t0=$(date +%s); consec=0
while [ "$consec" -lt 3 ]; do
  line=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  util=$(echo "$line" | awk -F, '{gsub(/[^0-9]/,"",$1); print $1}')
  mem=$(echo "$line"  | awk -F, '{gsub(/[^0-9]/,"",$2); print $2}')
  if [ -n "$util" ] && [ -n "$mem" ] && [ "$util" -lt 25 ] && [ "$mem" -lt 1500 ]; then
    consec=$((consec+1))
  else
    consec=0
    if [ $(( $(date +%s) - gate_t0 )) -gt 1800 ]; then say "ABORT: GPU busy >30min at gate"; exit 2; fi
  fi
  log "gpu poll util=${util:-NA} mem=${mem:-NA} consec=${consec}/3"
  [ "$consec" -lt 3 ] && sleep 30
done
say "GPU idle — window opens now"
T0=$(date +%s)
FAILED_MODELS=""

# ---------------------------------------------------------------- SMOKE mode: the LAUNCH GATE (real 1B micro-stream)
if [ "$SMOKE" -eq 1 ]; then
  say "SMOKE=1 — real 5-update micro-stream on ${MODEL_DIR} (launch gate: asserts (a)-(d)) ..."
  if $PY -m experiments.frame_a.run_stream --run --real --smoke --model_dir "$MODEL_DIR" >> "$LOG" 2>&1; then
    say "SMOKE PASS — all four asserts (a)-(d) fired+passed on the micro-stream; cleared for the full wave"
    exit 0
  else
    say "ABORT: SMOKE FAILED — an assert (a)-(d) did not fire/pass (see $LOG); DO NOT launch the wave"; exit 7
  fi
fi

# ---------------------------------------------------------------- M4: real wave requires a FRESH smoke marker
if ! $PY -m experiments.frame_a.run_stream --check_smoke_marker --model_dir "$MODEL_DIR" >> "$LOG" 2>&1; then
  say "ABORT: no FRESH SMOKE_PASS.ok for ${MODEL_DIR} (missing / stale / frame_a code changed since smoke)."
  say "       Run the launch gate first:  SMOKE=1 MODEL_DIR=${MODEL_DIR} bash run_frame_a_wave1.sh"; exit 8
fi
say "smoke marker FRESH (model_dir + frame_a code checksum match) — real wave cleared"

# ---------------------------------------------------------------- Phase 3: real wave per model
elapsed_min(){ echo $(( ( $(date +%s) - T0 ) / 60 )); }
for M in $MODELS; do
  now=$(elapsed_min)
  if [ "$now" -ge "$BUDGET_MIN" ]; then say "BUDGET-STOP before ${M} (elapsed ${now}m)"; break; fi
  say "REAL wave: model=${M} (elapsed ${now}m / ${BUDGET_MIN}m)"
  timeout --signal=TERM --kill-after=60 "$(( (BUDGET_MIN - now) * 60 ))s" \
    $PY -m experiments.frame_a.run_stream --run --real --model "$M" --model_dir "$MODEL_DIR" \
      --out_cells results/frame_a/cells >> "engine/frame_a_${M}.log" 2>&1 \
    || { say "RUNNER-FAIL: model ${M} returned nonzero (see engine/frame_a_${M}.log) — verdict for it SKIPPED"; FAILED_MODELS="$FAILED_MODELS $M"; }
done

# ---------------------------------------------------------------- Phase 4: analyze + verdict (per model)
# One verdict per model, each pinned to provenance=real (MAJOR-2 guard: a stale synthetic or
# other-model cell in the dir aborts the analyze loudly rather than being scored as real).
for M in $MODELS; do
  case " $FAILED_MODELS " in *" $M "*)
    say "verdict for ${M} SKIPPED (runner failed; partial grid must not be scored)"; continue;;
  esac
  $PY -m experiments.frame_a.scorer.analyze_frame_a --cells_dir results/frame_a/cells \
    --expect_model "$M" --expect_provenance real \
    --out "results/frame_a/frame_a_verdict_${M}.json" >> "$LOG" 2>&1 \
    && say "verdict written: results/frame_a/frame_a_verdict_${M}.json" \
    || say "NOTE: analyze for ${M} skipped/failed (no real cells yet, or mixed-set refusal — see $LOG)"
done
if [ -n "$FAILED_MODELS" ]; then
  say "================ RUN_FRAME_A_WAVE1 END (RUNNER FAILURES:${FAILED_MODELS}) ================"
  exit 7
fi
say "================ RUN_FRAME_A_WAVE1 END ================"
