#!/bin/bash
# run_quant_smoke.sh — Direction #1 quantization-survival SMOKE driver.
# Template = run_merging_editors.sh / run_merging_width.sh (verbatim skeleton: preflight,
# GPU-idle gate util<25&&mem<1500 x3, CPU self-test smoke gate, budget, DRYRUN, refuse-guard,
# pid-by-file / kill -0 only). Runs experiments/quant_survival_smoke.py --run for ONE
# (model, layer, seed) cell. See docs/plans/PREREG-QUANT-SMOKE-2026-07-16.md.
#
# BUILD-ONLY as authored 2026-07-16: authored under a no-GPU-runs, no-network mandate. Verified
# CPU-side only (bash -n, --selftest, DRYRUN=1). NOT launched by the author. GPU budget when
# launched: <= 40 min. Respects the standard idle gate (the 5090 is busy with the E1 chain).
#
# ROME value-opt stays fp32 (merging_m0._load_edit_model always loads dtype=torch.float32).
set -u
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$H" || exit 1
PY=${PY:-/home/zeyufu/miniconda3/envs/dl/bin/python3}

MODEL_DIR=${MODEL_DIR:-data/models/Llama-3.2-1B}
MODEL_TAG=${MODEL_TAG:-llama1b}
LAYER=${LAYER:-12}
SEED=${SEED:-0}
N_EDITS=${N_EDITS:-50}
N_PROBES=${N_PROBES:-40}
SCHEMES=${SCHEMES:-nf4,int8}
BUDGET_MIN=${BUDGET_MIN:-40}
EST_MIN=${EST_MIN:-30}
DRYRUN=${DRYRUN:-0}

TAG="${MODEL_TAG}_L${LAYER}_s${SEED}"
LOG="engine/run_quant_smoke_${TAG}.log"
OUT_DIR="results/quant_smoke/${TAG}"
TABLE="${OUT_DIR}/QS_smoke_table.json"
RAW="${OUT_DIR}/QS_smoke_raw.npz"
mkdir -p engine results/quant_smoke "${OUT_DIR}" results/quant_smoke/selftest
echo $$ > "engine/run_quant_smoke_${TAG}.pid"
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "======== RUN_QUANT_SMOKE START tag=${TAG} model=${MODEL_DIR} L=${LAYER} seed=${SEED} n_edits=${N_EDITS} n_probes=${N_PROBES} schemes=${SCHEMES} pid=$$ budget=${BUDGET_MIN}m ========"

# ---------------------------------------------------------------- Phase 0a: CPU pre-flight
pf_fail=0
pf(){ if eval "$2"; then log "preflight OK: $1"; else log "PREFLIGHT-FAIL: $1"; pf_fail=1; fi; }
pf "python env (torch+numpy)" "$PY -c 'import torch, numpy' 2>/dev/null"
pf "quant_survival_smoke.py present" "[ -f experiments/quant_survival_smoke.py ]"
pf "has --run flag" "grep -q -- '\"--run\"' experiments/quant_survival_smoke.py"
pf "merging_m0.py present (imported)" "[ -f experiments/merging_m0.py ]"
pf "rome_native editor present" "[ -f editors/rome_native.py ]"
pf "metrics.py present" "[ -f metrics.py ]"
pf "arch_compat present" "[ -f editors/arch_compat.py ]"
pf "counterfact.json" "[ -f data/counterfact.json ]"
pf "MODEL_DIR exists (${MODEL_DIR})" "[ -d '$MODEL_DIR' ]"
pf "MODEL_DIR has config.json" "[ -f '$MODEL_DIR/config.json' ]"
pf "disk >=5GB free" "[ \$(df --output=avail -BG . | tail -1 | tr -dc 0-9) -ge 5 ]"
if [ "$pf_fail" -ne 0 ]; then log "ABORT: preflight failed"; exit 3; fi

# ---------------------------------------------------------------- Phase 0a.2: refuse-guard (no clobber of a valid table)
if [ "$DRYRUN" -ne 1 ] && [ -f "$TABLE" ]; then
  if $PY - "$TABLE" <<'EOF' 2>/dev/null | grep -q VALIDATE-OK
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    print("VALIDATE-FAIL"); sys.exit(0)
print("VALIDATE-OK" if (d.get("arms") and d.get("mechanism_tie")) else "VALIDATE-FAIL")
EOF
  then
    log "REFUSE: valid table already at ${TABLE} — nothing to do (delete ${OUT_DIR} to force a re-run)"
    echo "REFUSE: valid table already exists at ${TABLE}" >&2
    exit 0
  fi
fi

# ---------------------------------------------------------------- Phase 0b: CPU self-test smoke gate
SELFTEST_OK="engine/quant_smoke_selftest_${TAG}.ok"
if [ "$DRYRUN" -ne 1 ]; then
  rm -f "$SELFTEST_OK"
  SELFTEST_LOG="engine/quant_smoke_selftest_${TAG}.log"
  log "SMOKE quant_survival_smoke --selftest (CPU, ~10s) -> ${SELFTEST_LOG}"
  if $PY experiments/quant_survival_smoke.py --selftest > "$SELFTEST_LOG" 2>&1; then
    if grep -q "ALL CHECKS PASSED" "$SELFTEST_LOG"; then
      : > "$SELFTEST_OK"
      log "SMOKE OK: self-test passed (INT8+NF4 codec bounds + tiny e2e)"
    else
      log "ABORT: self-test ran but did not report ALL CHECKS PASSED"; exit 4
    fi
  else
    log "ABORT: self-test failed (see ${SELFTEST_LOG})"; exit 4
  fi
fi

# ---------------------------------------------------------------- Phase 0c: GPU idle gate
if [ "$DRYRUN" -eq 1 ]; then
  log "DRYRUN=1 -- skipping self-test + GPU idle gate, printing the plan without executing"
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

ENVP="env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1"
CMD="$ENVP $PY experiments/quant_survival_smoke.py --run --model ${MODEL_DIR} --data data/counterfact.json --n_edits ${N_EDITS} --n_probes ${N_PROBES} --layer ${LAYER} --seed ${SEED} --steps 20 --lr 0.1 --schemes ${SCHEMES} --device cuda --out_dir ${OUT_DIR} --table_out ${TABLE}"

validate(){
  $PY - "$1" "$2" <<'EOF'
import json, sys, numpy as np
table, raw = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(table))
except Exception as e:
    print(f"VALIDATE-FAIL table unparseable: {e}"); sys.exit(1)
if not (d.get("arms") and d.get("mechanism_tie")):
    print("VALIDATE-FAIL missing arms/mechanism_tie"); sys.exit(1)
try:
    a = np.load(raw)
except Exception as e:
    print(f"VALIDATE-FAIL raw npz unreadable: {e}"); sys.exit(1)
need = {"COS", "damage_fp32", "edit_ok_fp32"}
if need - set(a.files):
    print(f"VALIDATE-FAIL raw npz missing {need - set(a.files)}"); sys.exit(1)
print(f"VALIDATE-OK arms={list(d['arms'])} rho_fp32={d['mechanism_tie']['rho_keycos_damage_fp32_pooled']}")
EOF
}

if [ "$DRYRUN" -eq 1 ]; then
  echo "DRYRUN tag=${TAG} model=${MODEL_DIR} L=${LAYER} seed=${SEED} n_edits=${N_EDITS} n_probes=${N_PROBES} schemes=${SCHEMES} est=${EST_MIN}m -> ${TABLE}"
  echo "DRYRUN raw npz: ${RAW}"
  echo "DRYRUN cmd: ${CMD}"
  log "DRYRUN tag=${TAG} est=${EST_MIN}m cmd: ${CMD}"
else
  now=$(elapsed_min)
  if [ $(( now + EST_MIN + 2 )) -gt "$BUDGET_MIN" ]; then
    log "BUDGET-SKIP tag=${TAG} (elapsed ${now}m + est ${EST_MIN}m > ${BUDGET_MIN}m)"
  elif [ -f "$TABLE" ] && [ -f "$RAW" ] && validate "$TABLE" "$RAW" | grep -q VALIDATE-OK; then
    log "skip tag=${TAG} (exists, validated)"
  else
    cap=$(( EST_MIN * 60 * 3 + 600 ))
    log "RUN tag=${TAG} (est ${EST_MIN}m, cap ${cap}s, elapsed ${now}m) -> engine/quant_smoke_${TAG}_run.log"
    t=$(date +%s)
    timeout --signal=TERM --kill-after=60 "${cap}s" bash -c "$CMD" >> "engine/quant_smoke_${TAG}_run.log" 2>&1
    rc=$?; dt=$(( $(date +%s) - t ))
    if [ "$rc" -eq 0 ] && [ -f "$TABLE" ] && [ -f "$RAW" ]; then
      vres=$(validate "$TABLE" "$RAW")
      if echo "$vres" | grep -q VALIDATE-FAIL; then
        mv "$TABLE" "$TABLE.INVALID" 2>/dev/null
        log "FAIL tag=${TAG} (${dt}s) OUTPUT-INVALID: ${vres}"
      else
        log "done tag=${TAG} (${dt}s) ${vres}"
      fi
    else
      log "FAIL tag=${TAG} (rc ${rc}, ${dt}s)"
    fi
  fi
fi

# ---------------------------------------------------------------- Post-run report
if [ "$DRYRUN" -ne 1 ] && [ -f "$TABLE" ]; then
  log "---------------- POST-RUN (CPU) tag=${TAG} ----------------"
  $PY - "$TABLE" >> "$LOG" 2>&1 <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
m = d["mechanism_tie"]; e = d["esr"]; fp = d["frozen_prediction_readout"]
print(f"[qs post] fp32 rho(key-cos,damage) pooled={m['rho_keycos_damage_fp32_pooled']} "
      f"within={m['rho_keycos_damage_fp32_within_probe']} mean_esr_fp32={e['mean_esr_fp32']}")
for n, a in d["arms"].items():
    print(f"[qs post]   {n}: esr={a['mean_esr']} surv={a['esr_survival_given_fp32_worked']} "
          f"rho(cos,dmg)={a['rho_keycos_damage_pooled']} Δrho={a['delta_rho_vs_fp32_pooled']} "
          f"rank_surv={a['rho_damage_fp32_vs_arm_rank_survival']}")
print(f"[qs post] readout p1(nf4-full esr surv>0.9)={fp['p1_esr_survival_nf4_full_gt_0.9']} "
      f"p3 esr-gap(nf4)={fp['p3_edited_vs_full_esr_gap_nf4']}")
EOF
  log "post: parsed ${TABLE}"
fi

{
  echo "RUN_QUANT_SMOKE REPORT tag=${TAG} model=${MODEL_DIR} L=${LAYER} seed=${SEED} $(date '+%F %T')  elapsed $(elapsed_min)m/${BUDGET_MIN}m"
  grep -E 'RUN |done |FAIL |SKIP|BUDGET-SKIP|REFUSE|ABORT|SMOKE|VERDICT|qs post|gpu poll' "$LOG" | tail -60
} > "engine/run_quant_smoke_${TAG}_report.txt"
log "======== RUN_QUANT_SMOKE COMPLETE tag=${TAG} ========"
echo "RUN_QUANT_SMOKE_DONE tag=${TAG}" >> "$LOG"
