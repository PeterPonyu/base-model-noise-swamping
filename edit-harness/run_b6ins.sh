#!/usr/bin/env bash
# run_b6ins.sh — B6@TETCI revision-insurance queue (2026-07-26).
#
# Closes the two dossier gaps a hostile review predicted reviewers will hit:
#   Cell H:  alpha-HOLDOUT projector cells at L10/L14 x s0/1/2.
#            The submitted paper's "holdout numbers are primary" currently rests on
#            L8/L12 only (results/matrices/g4_llama1b_alphaHO_cf_L{8,12}_s*.npz);
#            L14 — the norm-growth-dominant layer carrying the most interesting
#            claim — has no holdout cell. ~22-25 GPU-min per cell (anchor: the
#            run_8h.sh alphaHO rows ran 22m), 6 cells = ~140 GPU-min.
#   Cell V:  --save_vectors ROME dumps at L8/L12. VERIFIED 2026-07-26: these already
#            exist on disk (vectors_qv_llama1b_rome_cf_L{8,10,12,14}_s{0,1,2}.npz,
#            u5 wave) — the rows below will idempotent-SKIP. Kept in the queue as
#            self-healing regeneration in case any dump is later invalidated.
#            The CPU sham control (experiments/sham_projector_control.py --all)
#            consumes them directly; it does not need this queue at all when the
#            dumps are present.
#   Cell S:  GPU-LEVEL SHAM — now REQUIRED, not optional. The CPU first-order control
#            (experiments/sham_projector_control.py) was REJECTED 2026-07-26: its
#            denominator rescale cancels the projection identically (residual 1.8e-15
#            for any P), so it carried no projector information. See
#            submissions/ieee/revision/SHAM-CONTROL-READOUT-20260726.md.
#            Design: rerun AlphaEdit through the model with a rank-matched RANDOM projector,
#            comparing within-probe rho(key-cos, damage-removed) against the real holdout
#            cell at the same layer/seed. IMPLEMENTED as --alpha_proj_source sham via
#            experiments/patches/alpha_sham_projector_20260726.patch (dry-run verified;
#            APPLY ONLY AFTER THE FRAME-A WAVE DRAINS — frame_a lazy-imports killgate).
#            Queued below at L8/L12 x s0/1/2, ~25 GPU-min per cell = ~150 GPU-min.
#            Rows self-skip until the patch is applied (preflight below gates them).
#
# Total queue: ~260 GPU-min (+ margin). Idempotent: every row skips if its output
# exists. DO NOT run while another queue holds the GPU — the idle gate blocks.
# Launch:  cd edit-harness && nohup ./run_b6ins.sh > engine/run_b6ins.nohup.log 2>&1 &
# Stop:    kill by PID from engine/run_b6ins.pid (NEVER pgrep/pkill patterns).

set -u
cd "$(dirname "$0")"
mkdir -p engine results/sham_control

PY="${PY:-python}"
KG="experiments/killgate_keygeom.py"
MODEL="data/models/Llama-3.2-1B"
CF="--dataset counterfact"
# COMMON mirrors the science rows in run_revins.sh / archived run_8h.sh
COMMON="--n_edits 200 --n_probes 500 --steps 20 --save_matrices"
JOB_CAP_MIN="${JOB_CAP_MIN:-45}"
BUDGET_MIN="${BUDGET_MIN:-330}"

echo $$ > engine/run_b6ins.pid
LOG="engine/run_b6ins.log"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
log "======== B6INS START pid=$$ budget=${BUDGET_MIN}m cap=${JOB_CAP_MIN}m ========"
log "LID-OPEN REMINDER: keep lid open (nvidia_uvm wedge)."

# ---------------------------------------------------------------- preflight
pf() { desc="$1"; shift; if eval "$@" >/dev/null 2>&1; then log "PF ok: $desc"; else log "PF FAIL: $desc"; exit 4; fi; }
pf "killgate exists" "test -f $KG"
pf "model dir" "test -d $MODEL"
pf "alpha_proj_source flag" "grep -q -- '--alpha_proj_source' $KG"
pf "save_vectors flag" "grep -q -- '--save_vectors' $KG"
# (no sham-control preflight: that script was rejected 2026-07-26 and is not invoked)

# ---------------------------------------------------------------- GPU idle gate
# util<25 && mem<1500, 3 consecutive polls 30s apart (pattern from run_revins.sh;
# NEVER zero-compute-apps — mcp daemons hold permanent CUDA contexts).
consec=0
while [ "$consec" -lt 3 ]; do
  line=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  util=$(echo "$line" | awk -F, '{gsub(/[^0-9]/,"",$1); print $1}')
  mem=$(echo "$line"  | awk -F, '{gsub(/[^0-9]/,"",$2); print $2}')
  if [ -n "${util:-}" ] && [ -n "${mem:-}" ] && [ "$util" -lt 25 ] && [ "$mem" -lt 1500 ]; then
    consec=$((consec+1))
  else
    consec=0
  fi
  log "gpu poll util=${util:-NA} mem=${mem:-NA} consec=${consec}/3"
  [ "$consec" -lt 3 ] && sleep 30
done
log "GPU idle — window opens"

START_EPOCH=$(date +%s)
FAILS=0
budget_left() { echo $(( BUDGET_MIN - ( $(date +%s) - START_EPOCH ) / 60 )); }

# run_row TAG EST_MIN OUT_FILE CMD...
run_row() {
  tag="$1"; est="$2"; out="$3"; shift 3
  if [ -e "$out" ]; then log "SKIP $tag (exists: $out)"; return 0; fi
  if [ "$(budget_left)" -lt "$est" ]; then log "BUDGET-STOP before $tag ($(budget_left)m left < ${est}m est)"; return 9; fi
  if [ "$FAILS" -ge 2 ]; then log "ABORT-AFTER-2-FAILS before $tag"; return 8; fi
  log "RUN $tag (est ${est}m, cap ${JOB_CAP_MIN}m)"
  timeout --signal=TERM --kill-after=60 "$(( JOB_CAP_MIN * 60 ))s" \
    bash -c "$*" >> "engine/b6ins_${tag}.log" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ] && [ -e "$out" ]; then
    log "DONE $tag rc=0"
  else
    log "FAIL $tag rc=$rc (out present: $([ -e "$out" ] && echo yes || echo no))"
    FAILS=$((FAILS+1))
  fi
  return 0
}

# ---------------------------------------------------------------- Cell H: alphaHO L10/L14
for L in 10 14; do
  for s in 0 1 2; do
    run_row "alphaHO_L${L}_s${s}" 25 "results/g4_llama1b_alphaHO_cf_L${L}_s${s}.json" \
      "$PY $KG --model $MODEL --editor alpha $CF $COMMON --lr 0.1 --layer $L --seed $s \
       --alpha_proj_source holdout --holdout_frac 1.0 \
       --out results/g4_llama1b_alphaHO_cf_L${L}_s${s}.json" || break 2
  done
done

# ---------------------------------------------------------------- Cell V: ROME vector dumps L8/L12
for L in 8 12; do
  for s in 0 1 2; do
    run_row "vecdump_L${L}_s${s}" 22 "results/vectors/vectors_qv_llama1b_rome_cf_L${L}_s${s}.npz" \
      "$PY $KG --model $MODEL --editor rome $CF $COMMON --lr 0.1 --layer $L --seed $s \
       --save_vectors \
       --out results/qv_llama1b_rome_cf_L${L}_s${s}.json" || break 2
  done
done

# ---------------------------------------------------------------- Cell S: GPU-level sham
# Runs ONLY if the sham patch has been applied (post-drain). Until then the guard logs and
# skips, so this driver is safe to launch at any time.
if $PY -c "import sys; sys.path.insert(0,'experiments'); import re; \
    src=open('experiments/killgate_keygeom.py').read(); \
    sys.exit(0 if '\"sham\"' in src and 'alpha_proj_source == \"sham\"' in src else 1)" 2>/dev/null; then
  for L in 8 12; do
    for s in 0 1 2; do
      run_row "alphaSHAM_L${L}_s${s}" 25 "results/g4_llama1b_alphaSHAM_cf_L${L}_s${s}.json" \
        "$PY $KG --model $MODEL --editor alpha $CF $COMMON --lr 0.1 --layer $L --seed $s \
         --alpha_proj_source sham --holdout_frac 1.0 \
         --out results/g4_llama1b_alphaSHAM_cf_L${L}_s${s}.json" || break 2
    done
  done
else
  log "Cell S SKIPPED: --alpha_proj_source sham not present in killgate_keygeom.py"
  log "  apply experiments/patches/alpha_sham_projector_20260726.patch after the Frame-A wave drains"
fi

# ---------------------------------------------------------------- CPU tail: REMOVED
# The CPU sham control was rejected on 2026-07-26 (degenerate proxy — the rescale cancels
# the projection identically). Running it would only regenerate withdrawn numbers, so the
# tail is deliberately gone. The replacement is Cell S (GPU-level sham), which still needs
# a killgate flag to inject a rank-matched random projector — implement before queueing.
log "CPU sham tail intentionally REMOVED (proxy rejected 2026-07-26; see revision/SHAM-CONTROL-READOUT-20260726.md)"

log "======== B6INS END fails=$FAILS budget_left=$(budget_left)m ========"
rm -f engine/run_b6ins.pid
