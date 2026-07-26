#!/bin/bash
# run_paperb_phase1.sh — Paper B, Track-1 Phase-1 local driver (¥0, real bitsandbytes kernels).
# Template = run_quant_smoke.sh (verbatim skeleton: CPU preflight, refuse-guard on a valid table,
# CPU self-test smoke gate writing an .ok, GPU-idle gate util<25&&mem<1500 x3 pinned to THIS card
# via `nvidia-smi -i $GPU_ID` (nvidia-smi ignores CUDA_VISIBLE_DEVICES — the corrected convention
# from run_merging_editors.sh / chain_local_20260716.sh::gpu_idle_gate), 30-min busy-abort,
# per-cell timeout cap, BUDGET_MIN/EST_MIN/DRYRUN knobs, per-cell validate() on table+npz,
# pid-by-file / kill -0 only, post-run report grep). Runs experiments/quant_survival_phase1.py
# --run over the frozen Track-1 grid. See docs/plans/{PREREG,DESIGN}-PAPERB-QUANTSURVIVAL-*.md.
#
# GATING: this driver is invoked by engine/chain_local_20260716.sh step (f) ONLY when
# engine/PAPERB_GO.ok exists (the user's ratification signal). The driver itself does NOT re-check
# that gate file — the chain owns it — so a human running this by hand implies the same intent.
#
# BUILD-ONLY as authored 2026-07-16: authored under a no-GPU-runs, no-network mandate. Verified
# CPU-side only (bash -n, DRYRUN=1, quant_survival_phase1.py --selftest). NOT launched by the
# author. Real bnb NF4-dq + INT8 kernels; NO GPTQ/AWQ installs (Track 1.5 SKIPs cleanly, see below).
#
# ROME value-opt stays fp32 (quant_survival_phase1 loads dtype=torch.float32; the editors keep
# their own fp32 value-opt casts). Damage = signed within-probe damage_logit; AUROC BANNED.
set -u
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$H" || exit 1
PY=${PY:-/home/zeyufu/miniconda3/envs/dl/bin/python3}

# ---------------------------------------------------------------- knobs
N_EDITS=${N_EDITS:-200}
N_PROBES=${N_PROBES:-200}
SCHEMES=${SCHEMES:-nf4dq,int8}
CODEC=${CODEC:-real}
FULLMODEL_CACHE=${FULLMODEL_CACHE:-auto}
GEN_CHECK_N=${GEN_CHECK_N:-40}
BUDGET_MIN=${BUDGET_MIN:-900}
DRYRUN=${DRYRUN:-0}
# per-model EST defaults (design §6: ~40 min for 1B/1.5B with the cache; ~90 min for 3B)
EST_MIN_SMALL=${EST_MIN_SMALL:-40}
EST_MIN_3B=${EST_MIN_3B:-90}
# SHARD: for dual-GPU boxes. "small" = 1B+1.5B cells; "3b" = 3B cells; "all" = everything.
SHARD=${SHARD:-all}

LOG="engine/run_paperb_phase1.log"
mkdir -p engine results/quant_survival results/quant_survival/selftest
echo $$ > "engine/run_paperb_phase1.pid"
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "======== RUN_PAPERB_PHASE1 START pid=$$ budget=${BUDGET_MIN}m codec=${CODEC} schemes=${SCHEMES} n_edits=${N_EDITS} cache=${FULLMODEL_CACHE} ========"

# ---------------------------------------------------------------- Phase 0a: CPU pre-flight
pf_fail=0
pf(){ if eval "$2"; then log "preflight OK: $1"; else log "PREFLIGHT-FAIL: $1"; pf_fail=1; fi; }
pf "python env (torch+numpy)" "$PY -c 'import torch, numpy' 2>/dev/null"
pf "bitsandbytes importable" "$PY -c 'import bitsandbytes' 2>/dev/null"
pf "quant_survival_phase1.py present" "[ -f experiments/quant_survival_phase1.py ]"
pf "has --run flag" "grep -q -- '\"--run\"' experiments/quant_survival_phase1.py"
pf "quant_survival_smoke.py present (imported)" "[ -f experiments/quant_survival_smoke.py ]"
pf "merging_m0.py present (imported)" "[ -f experiments/merging_m0.py ]"
pf "rome_native editor present" "[ -f editors/rome_native.py ]"
pf "memit editor present" "[ -f editors/memit.py ]"
pf "alphaedit editor present" "[ -f editors/alphaedit.py ]"
pf "metrics.py present" "[ -f metrics.py ]"
pf "arch_compat present" "[ -f editors/arch_compat.py ]"
pf "counterfact.json" "[ -f data/counterfact.json ]"
pf "Llama-3.2-1B on disk" "[ -f data/models/Llama-3.2-1B/config.json ]"
pf "Llama-3.2-3B on disk (mandatory C2 cell)" "[ -f data/models/Llama-3.2-3B/config.json ]"
pf "Qwen2.5-1.5B on disk" "[ -f data/models/Qwen2.5-1.5B/config.json ]"
pf "disk >=15GB free" "[ \$(df --output=avail -BG . | tail -1 | tr -dc 0-9) -ge 15 ]"
if [ "$pf_fail" -ne 0 ]; then log "ABORT: preflight failed"; echo "ABORT: preflight failed (see $LOG)" >&2; exit 3; fi

# ---------------------------------------------------------------- Phase 0b: CPU self-test smoke gate
SELFTEST_OK="engine/paperb_phase1_selftest.ok"
if [ "$DRYRUN" -ne 1 ]; then
  rm -f "$SELFTEST_OK"
  SELFTEST_LOG="engine/paperb_phase1_selftest.log"
  log "SMOKE quant_survival_phase1 --selftest (CPU, ~30s) -> ${SELFTEST_LOG}"
  if CUDA_VISIBLE_DEVICES="" $PY experiments/quant_survival_phase1.py --selftest > "$SELFTEST_LOG" 2>&1; then
    if grep -q "ALL CHECKS PASSED" "$SELFTEST_LOG"; then
      : > "$SELFTEST_OK"
      log "SMOKE OK: self-test passed (bin-width + sim codecs + stats + disjointness + tiny e2e)"
    else
      log "ABORT: self-test ran but did not report ALL CHECKS PASSED"; exit 4
    fi
  else
    log "ABORT: self-test failed (see ${SELFTEST_LOG})"; exit 4
  fi
fi

# ---------------------------------------------------------------- Phase 0c: GPU idle gate (pinned)
if [ "$DRYRUN" -eq 1 ]; then
  log "DRYRUN=1 -- skipping self-test + GPU idle gate, printing the full cell plan without executing"
else
  gate_t0=$(date +%s); consec=0
  GPU_ID=${CUDA_VISIBLE_DEVICES%%,*}; GPU_ID=${GPU_ID:-0}
  while [ "$consec" -lt 3 ]; do
    line=$(nvidia-smi -i "$GPU_ID" --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    util=$(echo "$line" | awk -F, '{gsub(/[^0-9]/,"",$1); print $1}')
    mem=$(echo "$line"  | awk -F, '{gsub(/[^0-9]/,"",$2); print $2}')
    if [ -n "$util" ] && [ -n "$mem" ] && [ "$util" -lt 25 ] && [ "$mem" -lt 1500 ]; then
      consec=$((consec+1))
    else
      consec=0
      if [ $(( $(date +%s) - gate_t0 )) -gt 1800 ]; then log "ABORT: GPU busy >30min at gate"; echo "ABORT: GPU busy >30min" >&2; exit 2; fi
    fi
    log "gpu poll (card ${GPU_ID}) util=${util:-NA} mem=${mem:-NA} consec=${consec}/3"
    [ "$consec" -lt 3 ] && sleep 30
  done
  log "GPU idle (card ${GPU_ID}) -- window opens now"
fi
T0=$(date +%s)
elapsed_min(){ echo $(( ( $(date +%s) - T0 ) / 60 )); }

ENVP="env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1"

# ---------------------------------------------------------------- per-cell validate (table + npz)
validate(){
  $PY - "$1" "$2" <<'EOF'
import json, sys, numpy as np
table, raw = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(table))
except Exception as e:
    print(f"VALIDATE-FAIL table unparseable: {e}"); sys.exit(1)
if not (d.get("arms") and d.get("mechanism_tie") and d.get("bin_width_mechanism_C3")):
    print("VALIDATE-FAIL missing arms/mechanism_tie/C3"); sys.exit(1)
try:
    a = np.load(raw)
except Exception as e:
    print(f"VALIDATE-FAIL raw npz unreadable: {e}"); sys.exit(1)
need = {"COS", "damage_fp32", "edit_ok_fp32"}
if need - set(a.files):
    print(f"VALIDATE-FAIL raw npz missing {need - set(a.files)}"); sys.exit(1)
print(f"VALIDATE-OK arms={list(d['arms'])} rho_fp32_within={d['mechanism_tie']['rho_keycos_damage_fp32_within_probe']}")
EOF
}

# ---------------------------------------------------------------- one cell
# run_cell MODEL_DIR MODEL_TAG LAYER EDITOR SEED EST_MIN
run_cell(){
  local MODEL_DIR="$1" MODEL_TAG="$2" LAYER="$3" EDITOR="$4" SEED="$5" EST_MIN="$6"
  local TAG="${MODEL_TAG}_${EDITOR}_L${LAYER}_s${SEED}"
  local OUT_DIR="results/quant_survival/${TAG}"
  local TABLE="${OUT_DIR}/QS_phase1_table.json"
  local RAW="${OUT_DIR}/QS_phase1_raw.npz"
  mkdir -p "$OUT_DIR"
  local CMD="$ENVP $PY experiments/quant_survival_phase1.py --run --model ${MODEL_DIR} --data data/counterfact.json --editor ${EDITOR} --n_edits ${N_EDITS} --n_probes ${N_PROBES} --layer ${LAYER} --seed ${SEED} --steps 20 --lr 0.1 --schemes ${SCHEMES} --codec ${CODEC} --fullmodel_cache ${FULLMODEL_CACHE} --gen_check_n ${GEN_CHECK_N} --device cuda --out_dir ${OUT_DIR} --table_out ${TABLE}"

  if [ "$DRYRUN" -eq 1 ]; then
    echo "DRYRUN cell tag=${TAG} model=${MODEL_DIR} L=${LAYER} editor=${EDITOR} seed=${SEED} est=${EST_MIN}m -> ${TABLE}"
    echo "DRYRUN   cmd: ${CMD}"
    log "DRYRUN cell tag=${TAG} est=${EST_MIN}m cmd: ${CMD}"
    return 0
  fi

  # refuse-guard / skip-if-valid
  if [ -f "$TABLE" ] && [ -f "$RAW" ] && validate "$TABLE" "$RAW" | grep -q VALIDATE-OK; then
    log "skip cell tag=${TAG} (exists, validated)"
    return 0
  fi
  local now; now=$(elapsed_min)
  if [ $(( now + EST_MIN + 2 )) -gt "$BUDGET_MIN" ]; then
    log "BUDGET-SKIP cell tag=${TAG} (elapsed ${now}m + est ${EST_MIN}m > ${BUDGET_MIN}m)"
    return 0
  fi
  local cap=$(( EST_MIN * 60 * 3 + 1200 ))
  log "RUN cell tag=${TAG} (est ${EST_MIN}m, cap ${cap}s, elapsed ${now}m) -> engine/paperb_${TAG}_run.log"
  local t; t=$(date +%s)
  timeout --signal=TERM --kill-after=60 "${cap}s" bash -c "$CMD" >> "engine/paperb_${TAG}_run.log" 2>&1
  local rc=$?; local dt=$(( $(date +%s) - t ))
  if [ "$rc" -eq 0 ] && [ -f "$TABLE" ] && [ -f "$RAW" ]; then
    local vres; vres=$(validate "$TABLE" "$RAW")
    if echo "$vres" | grep -q VALIDATE-FAIL; then
      mv "$TABLE" "$TABLE.INVALID" 2>/dev/null
      log "FAIL cell tag=${TAG} (${dt}s) OUTPUT-INVALID: ${vres}"
    else
      log "done cell tag=${TAG} (${dt}s) ${vres}"
    fi
  else
    log "FAIL cell tag=${TAG} (rc ${rc}, ${dt}s)"
  fi
}

# ---------------------------------------------------------------- Track 1 grid (frozen priority)
# ROME FIRST (carries C2 — the geometry-survival headline); budget-skips drop from the TAIL, never
# the head. Llama-3.2-3B L24 is the MANDATORY 2nd validated-fp32-law cell (prereg §1/K1).
# MEMIT / AlphaEdit are C1/C3-only (their fp32 geometry tie is dead), measured identically.
# SHARD selects which subset of the grid runs on this GPU:
#   SHARD=small → Llama-1B + Qwen-1.5B (Card 0); SHARD=3b → Llama-3B only (Card 1); SHARD=all → all.
declare -a MODELS_DIR=("data/models/Llama-3.2-1B" "data/models/Llama-3.2-3B" "data/models/Qwen2.5-1.5B")
declare -a MODELS_TAG=("llama1b" "llama3b" "qwen15b")
declare -a MODELS_LAY=("12" "24" "21")
declare -a MODELS_EST=("$EST_MIN_SMALL" "$EST_MIN_3B" "$EST_MIN_SMALL")
SEEDS=(0 1 2)

if [ "$SHARD" = "small" ]; then
  MODEL_INDICES=(0 2)
elif [ "$SHARD" = "3b" ]; then
  MODEL_INDICES=(1)
else
  MODEL_INDICES=(0 1 2)
fi

for EDITOR in rome memit alpha; do
  for mi in "${MODEL_INDICES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      run_cell "${MODELS_DIR[$mi]}" "${MODELS_TAG[$mi]}" "${MODELS_LAY[$mi]}" "$EDITOR" "$SEED" "${MODELS_EST[$mi]}"
    done
  done
done

# ---------------------------------------------------------------- fp32 seed-spread (prereg §3, binding)
# After the three ROME llama1b seeds land, the ±0.15 Δρ tolerance must be justified against the fp32
# within-probe rho seed-to-seed spread (else "survival within ±0.15" is indistinguishable from seed
# noise). Compute + print it; do NOT edit the prereg here (a widen is a dated pre-data amendment).
if [ "$DRYRUN" -ne 1 ]; then
  $PY - <<'EOF' >> "$LOG" 2>&1
import json, glob, os
rhos = []
for s in (0, 1, 2):
    p = f"results/quant_survival/llama1b_rome_L12_s{s}/QS_phase1_table.json"
    if os.path.isfile(p):
        d = json.load(open(p))
        r = d.get("mechanism_tie", {}).get("rho_keycos_damage_fp32_within_probe")
        if r is not None:
            rhos.append((s, r))
if len(rhos) >= 2:
    vals = [r for _, r in rhos]
    spread = max(vals) - min(vals)
    print(f"FP32-SPREAD: {spread:.4f} (rho_within by seed: {rhos}) — the ±0.15 tolerance band must be >= this")
else:
    print(f"FP32-SPREAD: N/A (only {len(rhos)} ROME llama1b seed table(s) present)")
EOF
  log "fp32 seed-spread computed (grep FP32-SPREAD in ${LOG})"
fi

# ---------------------------------------------------------------- Track 1.5 (AFTER Track 1) — ASK-FIRST
# GPTQ-4bit + AWQ-4bit frozen-scale isolated (llama1b L12 ROME n=50 seed 0). Runs ONLY if BOTH
# AutoGPTQ and AutoAWQ import AND the frozen-scale group-wise codec is available. This driver NEVER
# installs anything (standing download rule; prereg §7 ask-first). Otherwise it logs a clean SKIP.
if [ "$DRYRUN" -ne 1 ]; then
  gptq_ok=$($PY -c "import importlib.util as u; print(1 if u.find_spec('auto_gptq') else 0)" 2>/dev/null)
  awq_ok=$($PY -c "import importlib.util as u; print(1 if u.find_spec('awq') else 0)" 2>/dev/null)
  if [ "${gptq_ok:-0}" = "1" ] && [ "${awq_ok:-0}" = "1" ]; then
    # both present, but the frozen-scale GPTQ/AWQ codec is a SEPARATE build not in
    # quant_survival_phase1.py (it implements only bnb int8/nf4dq). Do not fabricate a run.
    log "TRACK15-SKIP: AutoGPTQ+AutoAWQ present, but the frozen-scale group-wise codec is not yet implemented in quant_survival_phase1.py (build task, prereg §1.5/§7 — ask-first)"
    echo "TRACK15-SKIP: gptq/awq codec not yet built (prereg §1.5 build task)"
  else
    log "TRACK15-SKIP: autogptq(${gptq_ok:-0})/autoawq(${awq_ok:-0}) not both installed (ask-first — see prereg §7); Phase-1 Track-1 stands on bnb NF4-dq+INT8"
    echo "TRACK15-SKIP: autogptq/autoawq not installed (ask-first — see prereg §7)"
  fi
fi

# ---------------------------------------------------------------- Post-run report
{
  echo "RUN_PAPERB_PHASE1 REPORT $(date '+%F %T')  elapsed $(elapsed_min)m/${BUDGET_MIN}m  codec=${CODEC}"
  grep -E 'RUN cell|done cell|FAIL cell|BUDGET-SKIP|skip cell|REFUSE|ABORT|SMOKE|FP32-SPREAD|TRACK15|gpu poll' "$LOG" | tail -80
} > "engine/run_paperb_phase1_report.txt"
log "======== RUN_PAPERB_PHASE1 COMPLETE ========"
echo "RUN_PAPERB_PHASE1_DONE" >> "$LOG"
