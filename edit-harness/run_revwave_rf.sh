#!/bin/bash
# run_revwave_rf.sh — R-F revision-wave cells: 2 new low-gain AlphaEdit federation cells
# (gpt2-xl L36 alpha cf, Phi-3.5-mini L24 alpha cf), 3 seeds each, g=2..20, n=200.
#
# WHY these two: both land in the "low-gain deep layer" regime the 07-15 gain-wave flagged as
# constructive-merging territory (docs/findings/findings-RG-SIGNED-REANALYSIS-2026-07-15.md /
# PREDICTIONS-GAIN-WAVE-2026-07-15.md — gpt2-xl L36 was the MOST extreme low-gain cell at 1.5B-
# scale for ROME; this wave asks whether the SAME two-regime law replicates through AlphaEdit at
# those exact (model, layer) pairs). Blind-referee framing: a second editor at an architecture/
# depth combination already flagged as interesting, not a fresh direction.
#
# THIS DRIVER ADDS NO NEW PYTHON — it is a thin sequential wrapper around the ALREADY-REVIEWED
# run_merging_editors.sh (same preflight, GPU-idle gate util<25&&mem<1500 x3, CPU --selftest smoke
# gate, real-model ΔW-fidelity gate, refuse-clobber, PID-by-file / kill -0 only), invoked once per
# cell with EDITOR=alpha DATASET=cf and LAYER=auto75 (the driver computes floor(n_layers*0.75)
# from each model's own config.json: gpt2-xl has 48 layers -> L36; Phi-3.5-mini(-instruct) has 32
# layers -> L24 — both match the deliverable's stated layers, so no LAYER override is needed).
# RG_SEEDS/RG_GROUP_SIZES/N_EDITS are left at run_merging_editors.sh's own defaults
# (0,1,2 / 2,3,5,10,20 / 200), which already match this deliverable's spec exactly.
#
# BUILD-ONLY as authored 2026-07-16: CPU-validated only (bash -n on this file AND
# run_merging_editors.sh, DRYRUN=1 for both cells below); NOT launched by the author.
#
# gpt2-xl needs arch_compat's Conv1D->Linear graft (editors/arch_compat.py) — already exercised
# by existing gpt2-xl cells elsewhere in this harness (e.g. run_gptj.sh's peers); no new code
# path. Phi-3.5-mini is native-Llama-shaped (down_proj already present) — "native" arch_compat
# path, also nothing new.
set -u
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$H" || exit 1

# ---------------------------------------------------------------- required env (model locations)
# On the AutoDL box these live under /root/autodl-tmp/models/. The exact Phi-3.5 directory name is
# UNCONFIRMED on that box (local dev copy here is named "Phi-3.5-mini", not "-instruct" — see
# docs/plans/REVWAVE-BUILD-NOTES-2026-07-16.md open risk) — override via PHI35_DIR if it differs.
GPT2XL_DIR=${GPT2XL_DIR:-/root/autodl-tmp/models/gpt2-xl}
PHI35_DIR=${PHI35_DIR:-/root/autodl-tmp/models/Phi-3.5-mini-instruct}
BUDGET_MIN=${BUDGET_MIN:-150}
EST_MIN=${EST_MIN:-45}
DRYRUN=${DRYRUN:-0}

LOG="engine/run_revwave_rf.log"
mkdir -p engine results/merging_editors
echo $$ > engine/run_revwave_rf.pid
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "======== RUN_REVWAVE_RF START pid=$$ gpt2xl=${GPT2XL_DIR} phi35=${PHI35_DIR} budget=${BUDGET_MIN}m/cell ========"

pf_fail=0
pf(){ if eval "$2"; then log "preflight OK: $1"; else log "PREFLIGHT-FAIL: $1"; pf_fail=1; fi; }
pf "run_merging_editors.sh present" "[ -f run_merging_editors.sh ]"
pf "merging_editors.py present" "[ -f experiments/merging_editors.py ]"
if [ "$DRYRUN" -ne 1 ]; then
  pf "GPT2XL_DIR exists (${GPT2XL_DIR})" "[ -d '$GPT2XL_DIR' ]"
  pf "GPT2XL_DIR has config.json" "[ -f '${GPT2XL_DIR}/config.json' ]"
  pf "PHI35_DIR exists (${PHI35_DIR})" "[ -d '$PHI35_DIR' ]"
  pf "PHI35_DIR has config.json" "[ -f '${PHI35_DIR}/config.json' ]"
fi
if [ "$pf_fail" -ne 0 ]; then log "ABORT: preflight failed"; exit 3; fi

run_cell(){
  local model_dir="$1" model_tag="$2"
  log "---- cell ${model_tag} (MODEL_DIR=${model_dir}) ----"
  MODEL_DIR="$model_dir" MODEL_TAG="$model_tag" EDITOR=alpha DATASET=cf LAYER=auto75 \
    BUDGET_MIN="$BUDGET_MIN" EST_MIN="$EST_MIN" DRYRUN="$DRYRUN" \
    ./run_merging_editors.sh
  rc=$?
  log "---- cell ${model_tag} rc=${rc} ----"
  return $rc
}

n_fail=0
run_cell "$GPT2XL_DIR" "gpt2xl"  || n_fail=$((n_fail+1))
run_cell "$PHI35_DIR"  "phi35"  || n_fail=$((n_fail+1))

{
  echo "RUN_REVWAVE_RF REPORT $(date '+%F %T')  cells_failed=${n_fail}/2"
  grep -E 'RUN_REVWAVE_RF|---- cell|VERDICT|rg post|REFUSE|ABORT' "$LOG" | tail -60
} > engine/run_revwave_rf_report.txt
log "======== RUN_REVWAVE_RF COMPLETE (cells_failed=${n_fail}/2) ========"
exit $([ "$n_fail" -eq 0 ] && echo 0 || echo 1)
