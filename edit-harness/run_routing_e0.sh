#!/bin/bash
# run_routing_e0.sh — D4/T3.2 weight-vs-memory ROUTING E0 driver (LOCAL 5090).
# Template = run_merging_rg.sh (VERBATIM skeleton: preflight, CPU self-test smoke gate, GPU-idle
# gate util<25&&mem<1500 x3, budget clock from WORK start, DRYRUN, pid file, per-cell timeout,
# kill -0 only — NEVER pgrep/pkill -f a pattern). ROUTING-namespaced (own pid/log/markers).
# Analysis plan (retrospective, NOT a prereg): docs/plans/ANALYSIS-D4-ROUTING-E0-20260714.md.
#
# HONEST GPU ESTIMATE. The PRIMARY result is CPU-only: the aligned L12 editor cells
# (egl_llama1b_{rome,alpha,grace}_cf_L12_s{0,1,2}.npz) already exist on disk, so on this box the
# GPU "ensure" step FAST-SKIPS every cell (~0 GPU) and the run collapses to the CPU analysis
# (~seconds). GPU is spent ONLY to (re)generate a MISSING/INVALID primary cell (killgate ~91s
# per 200-edit cell, 07-12 timing) or, if L14_ARM=1, the 3 GRACE L14 cells (~5 GPU-min). Worst
# case (all 9 primary cells missing + L14 arm) ~= 18-20 GPU-min. BUDGET_MIN padded well over.
# ROME/GRACE value-opt stays fp32 (editor rule; this driver never passes --model_dtype bf16).
#
# GPU cells, each gated behind the CPU self-test smoke:
#   SMOKE (CPU, ~2s) — routing_e0.py --selftest: exact reduction/oracle/regret accounting on
#     synthetic fixtures. Gate marker: engine/routing_e0_selftest.ok.
#   ENSURE (GPU, idempotent) — for editor in {rome,alpha,grace} x seed {0,1,2} at L12: if the
#     matrix npz exists AND validates (COS/damage_logit/edit_ok present, right shape) SKIP; else
#     regenerate via killgate_keygeom.py --egl (the exact run_u6.sh SCIENCE-row command).
#   L14_ARM=1 (optional GPU secondary) — also ensure egl_llama1b_grace_cf_L14_s{0,1,2} so the
#     L14 DESCRIPTIVE layer can be computed (ROME/alpha L14 already exist). NOT part of the gate.
#   ANALYZE (CPU) — routing_e0.py --layers 12[,14] --seeds 0,1,2 -> results/analysis/D4_routing_e0.json.
set -u
H=/home/zeyufu/Desktop/idea-feasibility-analysis/edit-harness
cd "$H" || exit 1
PY=/home/zeyufu/miniconda3/envs/dl/bin/python3
LOG=engine/run_routing_e0.log
BUDGET_MIN=${BUDGET_MIN:-60}
SEEDS=${SEEDS:-0 1 2}
L14_ARM=${L14_ARM:-0}
DRYRUN=${DRYRUN:-0}
mkdir -p engine results/matrices results/analysis
echo $$ > engine/run_routing_e0.pid
[ -f engine/routing_e0_round_start ] || stat -c %Y engine/run_routing_e0.pid > engine/routing_e0_round_start
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "================ RUN_ROUTING_E0 START (pid $$, budget ${BUDGET_MIN}m, seeds '${SEEDS}', L14_ARM=${L14_ARM}) ================"

# killgate invocation identical to run_u6.sh SCIENCE rows (fp32 value-opt; --egl grid; save matrices)
KG="experiments/killgate_keygeom.py"
CF="--dataset counterfact --data data/counterfact.json"
COMMON="--n_edits 200 --n_probes 500 --steps 20 --save_matrices --matrix_dir results/matrices"
ENVP="env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1"

# ---------------------------------------------------------------- Phase 0a: CPU pre-flight (HARD)
pf_fail=0
pf(){ if eval "$2"; then log "preflight OK: $1"; else log "PREFLIGHT-FAIL: $1"; pf_fail=1; fi; }
pf "python env (torch+numpy)" "$PY -c 'import torch, numpy' 2>/dev/null"
pf "routing_e0.py present" "[ -f experiments/routing_e0.py ]"
pf "routing_e0 imports d3 predictor" "grep -q 'import d3_benefit_predictor' experiments/routing_e0.py"
pf "d3_benefit_predictor.py present" "[ -f experiments/d3_benefit_predictor.py ]"
pf "killgate present" "[ -f $KG ]"
pf "killgate has --egl flag" "grep -q -- '--egl' $KG"
pf "killgate wires editor grace" "grep -qE -- '\"grace\"' $KG"
pf "grace_editor present" "[ -f editors/grace_editor.py ]"
pf "rome_native present" "[ -f editors/rome_native.py ]"
pf "alphaedit present" "[ -f editors/alphaedit.py ]"
pf "arch_compat present" "[ -f editors/arch_compat.py ]"
pf "counterfact.json" "[ -f data/counterfact.json ]"
pf "model Llama-3.2-1B" "[ -d data/models/Llama-3.2-1B ]"
pf "disk >=10GB free" "[ \$(df --output=avail -BG /home | tail -1 | tr -dc 0-9) -ge 10 ]"
if [ "$pf_fail" -ne 0 ]; then log "ABORT: preflight failed"; exit 3; fi

# ---------------------------------------------------------------- Phase 0b: CPU self-test smoke gate
# Skipped on DRYRUN so a plan-only invocation leaves results/ byte-untouched (run_merging_rg precedent).
if [ "$DRYRUN" -ne 1 ]; then
  rm -f engine/routing_e0_selftest.ok
  log "SMOKE routing_e0 --selftest (CPU, ~2s) -> engine/routing_e0_selftest.log"
  if (cd experiments && $PY routing_e0.py --selftest) > engine/routing_e0_selftest.log 2>&1; then
    if grep -q "ALL CHECKS PASSED" engine/routing_e0_selftest.log; then
      : > engine/routing_e0_selftest.ok
      log "SMOKE OK: self-test passed (reduction/oracle/regret accounting)"
    else
      log "ABORT: self-test ran but did not report ALL CHECKS PASSED"; exit 4
    fi
  else
    log "ABORT: self-test failed (see engine/routing_e0_selftest.log)"; exit 4
  fi
fi

# ---------------------------------------------------------------- Phase 0c: GPU idle gate
if [ "$DRYRUN" -eq 1 ]; then
  log "DRYRUN=1 -- skipping self-test + GPU idle gate, printing the ensure+analyze plan without executing"
else
  gate_t0=$(date +%s); consec=0
  while [ "$consec" -lt 3 ]; do
    line=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    util=$(echo "$line" | awk -F, '{gsub(/[^0-9]/,"",$1); print $1}')
    mem=$(echo "$line"  | awk -F, '{gsub(/[^0-9]/,"",$2); print $2}')
    if [ -n "$util" ] && [ -n "$mem" ] && [ "$util" -lt 25 ] && [ "$mem" -lt 1500 ]; then
      consec=$((consec+1))
    else
      consec=0
      if [ $(( $(date +%s) - gate_t0 )) -gt 1800 ]; then log "ABORT: GPU busy >30min at gate"; exit 2; fi
    fi
    log "gpu poll util=${util:-NA} mem=${mem:-NA} consec=${consec}/3"
    [ "$consec" -lt 3 ] && sleep 30
  done
  log "GPU idle -- window opens now"
fi
T0=$(date +%s)
elapsed_min(){ echo $(( ( $(date +%s) - T0 ) / 60 )); }
n_done=0; n_fail=0; n_skip=0; n_regen=0

# validate one editor matrix cell (schema the router reads)
validate_cell(){
  $PY - "$1" <<'EOF'
import sys, numpy as np
try:
    a = np.load(sys.argv[1])
except Exception as e:
    print(f"VALIDATE-FAIL unreadable: {e}"); sys.exit(1)
need = {"COS", "damage_logit", "edit_ok", "pre_p", "norm_growth"}
missing = need - set(a.files)
if missing:
    print(f"VALIDATE-FAIL missing {missing}"); sys.exit(1)
if a["COS"].shape != a["damage_logit"].shape or a["COS"].ndim != 2:
    print("VALIDATE-FAIL bad shape"); sys.exit(1)
print(f"VALIDATE-OK shape={a['COS'].shape}")
EOF
}

# ensure one editor/layer/seed cell exists+valid; regenerate via killgate if not. EST padded.
ensure_cell(){
  local editor="$1" L="$2" s="$3"
  local tag="egl_llama1b_${editor}_cf_L${L}_s${s}"
  local npz="results/matrices/${tag}.npz"
  local outj="results/${tag}.json"
  local est=3
  if [ "$DRYRUN" -eq 1 ]; then
    echo "DRYRUN ensure ${tag} (skip-if-valid; regen est ${est}m)"
    log "DRYRUN ensure ${tag}: $ENVP $PY $KG --model data/models/Llama-3.2-1B --editor ${editor} --egl $CF $COMMON --lr 0.1 --layer ${L} --seed ${s} --out ${outj}"
    return 0
  fi
  if [ -f "$npz" ] && validate_cell "$npz" | grep -q VALIDATE-OK; then
    log "skip ${tag} (exists, validated)"; n_skip=$((n_skip+1)); return 0
  fi
  local now; now=$(elapsed_min)
  if [ $(( now + est + 2 )) -gt "$BUDGET_MIN" ]; then
    log "BUDGET-SKIP ${tag} (elapsed ${now}m + est ${est}m > ${BUDGET_MIN}m)"; n_skip=$((n_skip+1)); return 0
  fi
  local cap=$(( est * 60 * 3 + 600 ))
  log "REGEN ${tag} (est ${est}m, cap ${cap}s, elapsed ${now}m) -> engine/routing_e0_regen.log"
  local t; t=$(date +%s)
  timeout --signal=TERM --kill-after=60 "${cap}s" bash -c \
    "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor ${editor} --egl $CF $COMMON --lr 0.1 --layer ${L} --seed ${s} --out ${outj}" \
    >> engine/routing_e0_regen.log 2>&1
  local rc=$?; local dt=$(( $(date +%s) - t ))
  if [ "$rc" -eq 0 ] && [ -f "$npz" ] && validate_cell "$npz" | grep -q VALIDATE-OK; then
    log "REGEN-OK ${tag} (${dt}s)"; n_regen=$((n_regen+1)); n_done=$((n_done+1))
  else
    log "REGEN-FAIL ${tag} (rc ${rc}, ${dt}s)"; n_fail=$((n_fail+1))
  fi
}

# ---------------------------------------------------------------- Ensure primary L12 cells (rome/alpha/grace x seeds)
for ed in rome alpha grace; do
  for s in $SEEDS; do ensure_cell "$ed" 12 "$s"; done
done
# ---------------------------------------------------------------- Optional L14 GRACE cells (descriptive secondary)
LAYERS_ARG="12"
if [ "$L14_ARM" -eq 1 ]; then
  for s in $SEEDS; do ensure_cell rome 14 "$s"; ensure_cell alpha 14 "$s"; ensure_cell grace 14 "$s"; done
  LAYERS_ARG="12,14"
fi

# ---------------------------------------------------------------- ANALYZE (CPU): the sanctioned routing report
SEEDS_CSV=$(echo "$SEEDS" | tr ' ' ',')
OUTJSON="results/analysis/D4_routing_e0.json"
if [ "$DRYRUN" -eq 1 ]; then
  echo "DRYRUN analyze: $PY experiments/routing_e0.py --layers ${LAYERS_ARG} --seeds ${SEEDS_CSV} --weight_editor rome --out ${OUTJSON}"
  log "DRYRUN analyze cmd printed"
elif [ "$n_fail" -gt 0 ]; then
  log "SKIP analyze: ${n_fail} cell(s) failed to regenerate -- report would be partial"
else
  log "ANALYZE routing_e0 (layers ${LAYERS_ARG}, seeds ${SEEDS_CSV}, weight=rome) -> ${OUTJSON}"
  if (cd experiments && $PY routing_e0.py --layers "${LAYERS_ARG}" --seeds "${SEEDS_CSV}" --weight_editor rome \
        --out "$H/${OUTJSON}") >> engine/routing_e0_analyze.log 2>&1; then
    # secondary AlphaEdit-weight arm (H2): routing matters less when the weight editor is low-damage
    (cd experiments && $PY routing_e0.py --layers 12 --seeds "${SEEDS_CSV}" --weight_editor alpha \
        --out "$H/results/analysis/D4_routing_e0_alpha.json") >> engine/routing_e0_analyze.log 2>&1
    log "ANALYZE-OK -> ${OUTJSON} (+ D4_routing_e0_alpha.json)"; n_done=$((n_done+1))
  else
    log "ANALYZE-FAIL (see engine/routing_e0_analyze.log)"; n_fail=$((n_fail+1))
  fi
fi

# ---------------------------------------------------------------- Post-run report
if [ "$DRYRUN" -ne 1 ] && [ -f "$OUTJSON" ]; then
  log "---------------- POST-RUN (CPU) ----------------"
  $PY - "$OUTJSON" >> "$LOG" 2>&1 <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
g = d.get("primary_gate", {})
print(f"[routing post] VERDICT={g.get('VERDICT')} mean_eta_raw={g.get('mean_eta_geometry_raw')} "
      f"mean_eta_random={g.get('mean_eta_random_expected')} margin={g.get('margin_geometry_minus_random')} "
      f"perseed_pass={g.get('seeds_eta_raw>=perseed_floor')}/3 beats_random={g.get('seeds_geometry_beats_random')}/3")
EOF
  log "post: parsed ${OUTJSON}"
fi

{
  echo "RUN_ROUTING_E0 REPORT $(date '+%F %T')  cells: ${n_done} ok / ${n_regen} regenerated / ${n_fail} fail / ${n_skip} skipped  elapsed $(elapsed_min)m/${BUDGET_MIN}m"
  grep -E 'REGEN|skip |BUDGET-SKIP|ANALYZE|SMOKE|VERDICT|routing post|gpu poll|ABORT' "$LOG" | tail -60
} > engine/run_routing_e0_report.txt
log "================ RUN_ROUTING_E0 COMPLETE (${n_done} ok / ${n_fail} fail / ${n_skip} skip) ================"
echo "RUN_ROUTING_E0_DONE" >> "$LOG"
