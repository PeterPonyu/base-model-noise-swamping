#!/usr/bin/env bash
# run_deletion_phaseL.sh — deletion-predictor PHASE L (LOCAL RTX 5090, ¥0).
#
# Question: does pre-edit key geometry predict per-fact DELETION-edit collateral damage
# (and deletion success) the way it predicts insertion damage? On disk we have only 2
# families (Llama-1B, Qwen-1.5B). Phase L takes that to 4-5 using models already local,
# for free, BEFORE any cloud spend is considered.
#
# PREREG: docs/plans/PREREG-DELETION-PREDICTOR-2026-07-26.md (gates G-D1/G-D2 frozen there).
# SCOOP CHECK: PreUnlearn arXiv:2606.18473 is set-level/text-feature/gradient-unlearning —
# differentiable, but we MUST ship the text-feature baseline arm (experiments/
# deletion_text_baseline.py) as the decisive comparison and never claim "first".
#
# LAYER CHOICE IS DICTATED BY DISK, NOT BY THE DESIGN DOC. u1_deletion_gate.py needs a
# MATCHED INSERTION npz at the SAME layer to compute its variance receipt var(S_del)/
# var(S_ins). Verified 2026-07-26 in results/matrices/:
#     gemma2b   gate_gemma2b_rome_cf_L13_s{0,1,2}   -> deletion at L13
#     phi35     gate_phi35_rome_cf_L16_s{0,1,2}     -> deletion at L16
#     qwen3b    gate_qwen3b_rome_cf_L18_s{0,1,2}    -> deletion at L18
#     qwen15b   gate_qwen15b_rome_cf_L{14,17,21,24} -> deletion at L21 (75% depth)
#     gpt2xl    NO insertion cells                  -> would need BOTH arms; deferred
# The design doc's L19/L24/L36/L27 would each need a fresh insertion cell too (doubling
# cost); using the layers that already have twins is the same science for half the GPU.
#
# COST: 12 deletion cells x ~25-35 min = ~5-7 GPU-h. Zero yuan. Runs AFTER the Frame-A wave.
# Launch:  cd edit-harness && nohup ./run_deletion_phaseL.sh > engine/deletion_phaseL.nohup.log 2>&1 &
# Stop:    kill by PID from engine/run_deletion_phaseL.pid (NEVER pgrep/pkill).

set -u
cd "$(dirname "$0")"
mkdir -p engine results/deletion_phaseL

PREREG="${PREREG:-../docs/plans/PREREG-DELETION-PREDICTOR-2026-07-26.md}"
if [ ! -f "$PREREG" ] || ! grep -qx 'STATUS: RATIFIED' "$PREREG"; then
  echo "ABORT: deletion prereg is not ratified: $PREREG" >&2
  exit 5
fi

PY="${PY:-python}"
KG="experiments/killgate_keygeom.py"
GATE="experiments/u1_deletion_gate.py"
DATA="data/counterfact.json"
CF="--dataset counterfact --data $DATA"
COMMON="--n_edits 200 --n_probes 500 --steps 20 --save_matrices"
SMK="--n_edits 3 --n_probes 8 --steps 2"
ENVP="env -u ALL_PROXY -u all_proxy"
JOB_CAP_MIN="${JOB_CAP_MIN:-60}"
BUDGET_MIN="${BUDGET_MIN:-480}"
DRYRUN="${DRYRUN:-0}"

PIDFILE="engine/run_deletion_phaseL.pid"
LOG="engine/deletion_phaseL.log"
if [ -f "$PIDFILE" ]; then
  old=$(cat "$PIDFILE" 2>/dev/null || true)
  if [ -n "$old" ] && kill -0 "$old" 2>/dev/null; then echo "already running as $old"; exit 3; fi
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
log "======== DELETION PHASE L START pid=$$ budget=${BUDGET_MIN}m ========"
log "LID-OPEN REMINDER: keep lid open (nvidia_uvm wedge)."
if [ "$DRYRUN" = 1 ]; then
  log "DRYRUN: 12 deletion cells (gemma2b L13, phi35 L16, qwen3b L18, qwen15b L21)"
  exit 0
fi

# ---------------------------------------------------------------- preflight
pf() { d="$1"; shift; if eval "$@" >/dev/null 2>&1; then log "PF ok: $d"; else log "PF FAIL: $d"; exit 4; fi; }
pf "killgate"            "test -f $KG"
pf "deletion gate"       "test -f $GATE"
pf "CounterFact data"    "test -f $DATA"
pf "delete mode wired"   "grep -q -- '--edit_mode' $KG"
pf "refusal variant"     "grep -q -- 'refusal' $KG"
pf "text baseline"       "test -f experiments/deletion_text_baseline.py"
# every model in the grid must exist BEFORE we start (07-13 lesson: smoke must cover all)
for m in gemma-2-2b Phi-3.5-mini Qwen2.5-3B Qwen2.5-1.5B; do
  pf "model $m" "test -d data/models/$m"
done
# and every matched insertion twin must exist, or the variance receipt is impossible
for tw in gate_gemma2b_rome_cf_L13_s0 gate_phi35_rome_cf_L16_s0 \
          gate_qwen3b_rome_cf_L18_s0 gate_qwen15b_rome_cf_L21_s0; do
  pf "insertion twin $tw" "test -f results/matrices/${tw}.npz"
done

# ---------------------------------------------------------------- GPU idle gate
consec=0; tries=0
while [ "$consec" -lt 3 ]; do
  [ "$tries" -ge 60 ] && { log "ABORT: GPU never idle after 30min"; exit 9; }
  tries=$((tries+1))
  line=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  util=$(echo "$line" | awk -F, '{gsub(/[^0-9]/,"",$1); print $1}')
  mem=$(echo "$line"  | awk -F, '{gsub(/[^0-9]/,"",$2); print $2}')
  if [ -n "${util:-}" ] && [ -n "${mem:-}" ] && [ "$util" -lt 25 ] && [ "$mem" -lt 1500 ]; then
    consec=$((consec+1)); else consec=0; fi
  log "gpu poll util=${util:-NA} mem=${mem:-NA} consec=${consec}/3"
  [ "$consec" -lt 3 ] && sleep 30
done
log "GPU idle — window opens"

START=$(date +%s); FAILS=0; COMPLETED=0; EXPECTED=16
left() { echo $(( BUDGET_MIN - ( $(date +%s) - START ) / 60 )); }

run_row() {   # TAG EST OUT CMD...
  tag="$1"; est="$2"; out="$3"; shift 3
  [ -e "$out" ] && { log "SKIP $tag (exists)"; COMPLETED=$((COMPLETED+1)); return 0; }
  [ "$(left)" -lt "$est" ] && { log "BUDGET-STOP before $tag"; return 9; }
  [ "$FAILS" -ge 2 ] && { log "ABORT-AFTER-2-FAILS before $tag"; return 8; }
  log "RUN $tag (est ${est}m)"
  timeout --signal=TERM --kill-after=60 "$(( JOB_CAP_MIN * 60 ))s" bash -c "$*" \
    >> "engine/delL_${tag}.log" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ] && [ -e "$out" ]; then log "DONE $tag"; COMPLETED=$((COMPLETED+1)); else log "FAIL $tag rc=$rc"; FAILS=$((FAILS+1)); fi
  return 0
}

# ---------------------------------------------------------------- smoke: EVERY model first
log "--- smoke gate (every model in the grid) ---"
mkdir -p results/smoke_delL
smoke_one() {  # tag model layer
  run_row "smoke_$1" 6 "results/smoke_delL/$1.json" \
    "$ENVP $PY $KG --model data/models/$2 --editor rome --edit_mode delete \
     --delete_variant refusal $CF $SMK --lr 0.1 --layer $3 --seed 0 \
     --out results/smoke_delL/$1.json"
}
smoke_one gemma2b  gemma-2-2b     13
smoke_one phi35    Phi-3.5-mini   16
smoke_one qwen3b   Qwen2.5-3B     18
smoke_one qwen15b  Qwen2.5-1.5B   21
if [ "$FAILS" -gt 0 ]; then log "ABORT: smoke failures — do not run science on an unproven grid"; exit 5; fi
log "smoke 4/4 PASS"

# ---------------------------------------------------------------- science: deletion cells
# tag convention mirrors the existing u1e0_* cells so analyses find them.
cell() {  # tag model layer seed
  run_row "$1_s$4" 35 "results/u1e0_$1_delete_refusal_L$3_s$4.json" \
    "$ENVP $PY $KG --model data/models/$2 --editor rome --edit_mode delete \
     --delete_variant refusal $CF $COMMON --lr 0.1 --layer $3 --seed $4 \
     --out results/u1e0_$1_delete_refusal_L$3_s$4.json"
}
for s in 0 1 2; do cell gemma2b  gemma-2-2b   13 "$s" || break; done
for s in 0 1 2; do cell phi35    Phi-3.5-mini 16 "$s" || break; done
for s in 0 1 2; do cell qwen3b   Qwen2.5-3B   18 "$s" || break; done
for s in 0 1 2; do cell qwen15b  Qwen2.5-1.5B 21 "$s" || break; done

# ---------------------------------------------------------------- CPU tail: gates + baseline
log "--- CPU tail: prereg gate per family (needs the matched insertion twin) ---"
gate_one() {  # tag layer insertion-tag
  for s in 0 1 2; do
    d="results/matrices/u1e0_$1_delete_refusal_L$2_s${s}.npz"
    i="results/matrices/$3_s${s}.npz"
    o="results/deletion_phaseL/GATE_$1_L$2_s${s}.json"
    [ -e "$o" ] && continue
    if [ -e "$d" ] && [ -e "$i" ]; then
      $PY $GATE --del_npz "$d" --ins_npz "$i" --out "$o" >> "$LOG" 2>&1 \
        && log "gate $1 s$s -> $o" || log "gate $1 s$s FAILED"
    else
      log "gate $1 s$s SKIPPED (missing $( [ -e "$d" ] || echo del )$( [ -e "$i" ] || echo ' ins' ))"
    fi
  done
}
gate_one gemma2b 13 gate_gemma2b_rome_cf_L13
gate_one phi35   16 gate_phi35_rome_cf_L16
gate_one qwen3b  18 gate_qwen3b_rome_cf_L18
gate_one qwen15b 21 gate_qwen15b_rome_cf_L21

log "--- CPU tail: PreUnlearn-style text-feature baseline (BINDING amendment) ---"
$PY experiments/deletion_text_baseline.py --phaseL \
  --out results/deletion_phaseL/TEXT_BASELINE_phaseL.json >> "$LOG" 2>&1 \
  && log "text baseline DONE" || log "text baseline FAILED (see $LOG)"

$PY experiments/deletion_phase_readout.py >> "$LOG" 2>&1
readout_rc=$?
[ "$readout_rc" -eq 0 ] && log "Phase-L G-D1/G-D2/text receipts PASS" \
  || log "Phase-L gates did not all pass (rc=$readout_rc); cloud Wave 1 remains locked"

log "======== PHASE L END fails=$FAILS completed=$COMPLETED/$EXPECTED budget_left=$(left)m ========"
log "NEXT: read results/deletion_phaseL/GATE_*.json — G-D1 needs >=3 families decidable"
log "      and G-D2 needs var(S_del)/var(S_ins) >= 0.1. If both pass across 4 families,"
log "      the cloud waves are OPTIONAL generality insurance, not a requirement."
[ "$FAILS" -eq 0 ] && [ "$COMPLETED" -eq "$EXPECTED" ] && [ "$readout_rc" -ne 3 ]
