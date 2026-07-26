#!/bin/bash
# run_transplant_e0b.sh — D3 / T1.3-E0b generational edit TRANSPLANT driver (Pro-6000 on-box).
# Prereg: docs/plans/PREREG-D3-TRANSPLANT-E0B-20260714.md. Template = run_merging_rg.sh +
# run_family_transfer.sh (preflight, CPU self-test smoke gate, GPU-idle gate util<25&&mem<1500
# x3, budget clock from WORK start, DRYRUN, skip-if-exists, atomic writes, kill -0 PID only,
# per-cell timeout). BUILD-ONLY as authored 2026-07-14: verified CPU-side (bash -n, --selftest,
# DRYRUN); NOT launched. The 14B models live on the Pro-6000 (box 29246) disk.
#
# WHAT IT DOES: transplants ROME residual (value) edits from Qwen2.5-14B (donor) onto
# Qwen3-14B-Base (recipient) via orthogonal Procrustes on shared-vocab token embeddings,
# re-deriving each key natively (see prereg §1 — THE CORE DESIGN DECISION). Two matched-depth
# layer pairs from the E0a VALID band ONLY (deep layers esr-collapse in both):
#     PRIMARY   donor L24 -> recipient L20   (50%   depth)
#     SECONDARY donor L30 -> recipient L25   (62.5% depth)
# Each pair is a DONOR phase (dump per-edit residuals + anchor embeddings to a disk bank) then
# a RECIPIENT phase (fit Procrustes, install every condition, gate). fp32 value-opt only.
#
# VRAM / WHY TWO PHASES: two 14B fp32 models (~56GB each) do NOT co-reside in the Pro-6000's
# 96GB, so the phases load SEQUENTIALLY with the donor bank on disk between them (the python
# frees the donor before the recipient loads). This is not an optimization — it is required.
# WAVE_BOX=pro6000 is asserted below precisely because this cannot run on the local 5090 (24GB).
#
# HONEST GPU ESTIMATE ~2-4 GPU-h for both pairs, single seed s0 (matches the plan's D3 budget).
# Basis: 1B 200-edit ROME battery ~91s; 14B fp32 value-opt ~30-60 min/phase for 200 edits;
# per-edit efficacy across 8 conditions is 1 forward each. Row ests below stay padded so a
# thermally-throttled card still lands the PRIMARY pair (ordered first).
#
# ROME value-opt stays fp32 (editor's rule; this driver never requests bf16 value-opt).
set -u
H="$(cd "$(dirname "$0")" && pwd)"
cd "$H" || exit 1
PY=/home/zeyufu/miniconda3/envs/dl/bin/python3
PY="${CLOUD_PY:-$PY}"
LOG=engine/run_transplant_e0b.log
BUDGET_MIN=${BUDGET_MIN:-300}
DONOR=${DONOR:-data/models/Qwen2.5-14B}
RECIP=${RECIP:-data/models/Qwen3-14B-Base}
WAVE_BOX=${WAVE_BOX:-}
DRYRUN=${DRYRUN:-0}
mkdir -p engine results/transplant results/transplant/selftest
echo $$ > engine/run_transplant_e0b.pid
[ -f engine/transplant_round_start ] || stat -c %Y engine/run_transplant_e0b.pid > engine/transplant_round_start
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "================ RUN_TRANSPLANT_E0B START (pid $$, budget ${BUDGET_MIN}m, donor=${DONOR}, recip=${RECIP}) ================"

# ---------------------------------------------------------------- Phase 0z: WAVE_BOX guard (HARD)
# This driver loads two 14B fp32 models sequentially — impossible on the 24GB local 5090. Refuse
# to run anywhere but the Pro-6000 unless the operator explicitly overrides. DRYRUN is exempt
# (plan-only invocations must work anywhere for CPU validation).
if [ "$DRYRUN" -ne 1 ]; then
  if [ "$WAVE_BOX" != "pro6000" ]; then
    log "ABORT: WAVE_BOX='${WAVE_BOX}' != 'pro6000' — two 14B fp32 models need the 96GB Pro-6000;"
    log "       set WAVE_BOX=pro6000 to run here (or DRYRUN=1 for CPU plan-only validation)."
    echo "ABORT: set WAVE_BOX=pro6000 (this needs the 96GB box); or DRYRUN=1 for validation." >&2
    exit 6
  fi
fi

# ---------------------------------------------------------------- Phase 0a: CPU pre-flight (HARD)
pf_fail=0
pf(){ if eval "$2"; then log "preflight OK: $1"; else log "PREFLIGHT-FAIL: $1"; pf_fail=1; fi; }
pf "python env (torch+numpy)" "$PY -c 'import torch, numpy' 2>/dev/null"
pf "gen_transplant.py present" "[ -f experiments/gen_transplant.py ]"
pf "gen_transplant has --selftest" "grep -q -- '--selftest' experiments/gen_transplant.py"
pf "rome_native editor present" "[ -f editors/rome_native.py ]"
pf "killgate_keygeom present" "[ -f experiments/killgate_keygeom.py ]"
pf "metrics.py present" "[ -f metrics.py ]"
pf "analyze_matrices.py present" "[ -f experiments/analyze_matrices.py ]"
pf "counterfact.json" "[ -f data/counterfact.json ]"
if [ "$pf_fail" -ne 0 ]; then log "ABORT: preflight failed"; exit 3; fi

# ---------------------------------------------------------------- Phase 0a2: model presence (SOFT)
# Models live on the box disk and may not be present on a fresh boot — a missing model makes
# every cell CONFIG-skip cleanly (the driver still completes), mirroring run_family_transfer.sh.
MODELS_READY=1
for m in "$DONOR" "$RECIP"; do
  if [ -d "$m" ]; then log "model present: $m"; else
    log "MODEL-ABSENT: $m — every cell CONFIG-skips (download/stage on box first)"; MODELS_READY=0; fi
done

# ---------------------------------------------------------------- Phase 0b: CPU self-test smoke gate
# Skipped on DRYRUN so a plan-only invocation leaves results/ byte-untouched (run_merging_rg
# precedent). The GPU cells are skipped on DRYRUN anyway.
if [ "$DRYRUN" -ne 1 ]; then
  rm -f engine/transplant_selftest.ok
  log "SMOKE gen_transplant --selftest (CPU) -> engine/transplant_selftest.log"
  if $PY experiments/gen_transplant.py --selftest > engine/transplant_selftest.log 2>&1 \
     && grep -q "ALL CHECKS PASSED" engine/transplant_selftest.log; then
    : > engine/transplant_selftest.ok
    log "SMOKE OK: self-test passed (rotation recovery + install identity + random-null + predictor)"
  else
    log "ABORT: self-test failed (see engine/transplant_selftest.log)"; exit 4
  fi
fi

# ---------------------------------------------------------------- Phase 0c: GPU idle gate
if [ "$DRYRUN" -eq 1 ]; then
  log "DRYRUN=1 — skipping GPU idle gate, printing the transplant plan without executing"
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
n_done=0; n_fail=0; n_skip=0

# validate a recipient-phase result JSON + its npz (verdict present + gate keys + npz shapes)
validate_recip(){
  $PY - "$1" "$2" <<'EOF'
import json, sys, numpy as np
j, z = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(j))
except Exception as e:
    print(f"VALIDATE-FAIL json unparseable: {e}"); sys.exit(1)
if d.get("verdict") not in ("PASS", "PARTIAL_PASS_no_predictor", "KILL_K1_not_alignment_specific",
                            "KILL_K2_transplant_fails", "INCONCLUSIVE",
                            "INCONCLUSIVE_LAYER_INVALID"):
    print(f"VALIDATE-FAIL bad verdict: {d.get('verdict')!r}"); sys.exit(1)
for k in ("esr", "gates", "predictor_ca_vs_success"):
    if k not in d: print(f"VALIDATE-FAIL missing {k}"); sys.exit(1)
try:
    a = np.load(z)
except Exception as e:
    print(f"VALIDATE-FAIL npz unreadable: {e}"); sys.exit(1)
need = {"succ_native", "succ_identity", "succ_procrustes", "succ_random", "ca", "donor_edit_ok"}
missing = need - set(a.files)
if missing: print(f"VALIDATE-FAIL npz missing {missing}"); sys.exit(1)
print(f"VALIDATE-OK verdict={d['verdict']} esr_proc={d['esr'].get('procrustes')}")
EOF
}

# donor phase produces a bank npz; validate it exists + carries the expected keys
validate_bank(){
  $PY - "$1" <<'EOF'
import sys, numpy as np, json
try:
    a = np.load(sys.argv[1], allow_pickle=False)
    need = {"r_donor", "donor_edit_ok", "donor_anchor_emb", "meta"}
    miss = need - set(a.files)
    if miss: print(f"VALIDATE-FAIL bank missing {miss}"); sys.exit(1)
    json.loads(str(a["meta"]))
    print("VALIDATE-OK bank")
except Exception as e:
    print(f"VALIDATE-FAIL bank: {e}"); sys.exit(1)
EOF
}

# run_cell <kind> <tag> <est_min> <outfile> <validate_fn> <cmd...>
run_cell(){
  local kind="$1" tag="$2" est="$3" out="$4" vfn="$5"; shift 5; local cmd="$*"
  if [ "$DRYRUN" -eq 1 ]; then
    echo "DRYRUN ${tag} [${kind}] est=${est}m out=${out}"
    echo "DRYRUN cmd: ${cmd}"
    log "DRYRUN ${tag} [${kind}] est=${est}m cmd: ${cmd}"
    return 0
  fi
  if [ "$MODELS_READY" -ne 1 ]; then log "CONFIG-SKIP ${tag} (models absent)"; n_skip=$((n_skip+1)); return 0; fi
  local now; now=$(elapsed_min)
  if [ $(( now + est + 2 )) -gt "$BUDGET_MIN" ]; then
    log "BUDGET-SKIP ${tag} (elapsed ${now}m + est ${est}m > ${BUDGET_MIN}m)"; n_skip=$((n_skip+1)); return 1; fi
  if [ -f "$out" ] && $vfn "$out" "${out%.json}.npz" 2>/dev/null | grep -q VALIDATE-OK; then
    log "skip ${tag} (exists, validated)"; n_done=$((n_done+1)); return 0; fi
  local cap=$(( est * 60 * 3 + 1200 ))
  log "RUN ${tag} [${kind}] (est ${est}m, cap ${cap}s, elapsed ${now}m) -> engine/transplant_${tag}.log"
  local t rc dt
  t=$(date +%s)
  timeout --signal=TERM --kill-after=60 "${cap}s" bash -c "$cmd" >> "engine/transplant_${tag}.log" 2>&1 </dev/null
  rc=$?; dt=$(( $(date +%s) - t ))
  if [ "$rc" -eq 0 ] && [ -f "$out" ]; then
    local v; v=$($vfn "$out" "${out%.json}.npz")
    if echo "$v" | grep -q VALIDATE-FAIL; then
      mv "$out" "$out.INVALID" 2>/dev/null
      log "FAIL ${tag} (${dt}s) OUTPUT-INVALID: ${v}"; n_fail=$((n_fail+1)); return 1
    fi
    log "done ${tag} (${dt}s) ${v}"; n_done=$((n_done+1)); return 0
  fi
  log "FAIL ${tag} (rc ${rc}, ${dt}s)"; n_fail=$((n_fail+1)); return 1
}

# ---------------------------------------------------------------- the two matched-depth pairs
# "Ld Lr est_donor est_recip"  (PRIMARY first so a short window still lands it)
# "Ld Lr est_donor est_recip collateral"  — PRIMARY first (collateral=1: run the prereg §3.6
# tertiary collateral-law sweep on the primary pair only; its est is bumped to cover the N x Mc
# probe sweep). Secondary pair skips it (collateral=0) to stay lean.
PAIRS=("24 20 60 100 1" "30 25 60 80 0")
COMMON="--data data/counterfact.json --n_edits 200 --n_probes 500 --seed 0 --steps 20 --lr 0.1 --device cuda"

for spec in "${PAIRS[@]}"; do
  read -r Ld Lr est_d est_r collat <<< "$spec"
  COLLAT_FLAG=""; [ "$collat" = "1" ] && COLLAT_FLAG="--collateral --n_collateral_probes 200"
  bank="results/transplant/donor_bank_L${Ld}.npz"
  out="results/transplant/D3_transplant_E0b_L${Ld}to${Lr}_s0.json"

  # donor phase -> bank (skip if a valid bank already exists)
  if [ "$DRYRUN" -eq 1 ]; then
    echo "DRYRUN donor_L${Ld} est=${est_d}m bank=${bank}"
    log "DRYRUN donor_L${Ld} cmd: $ENVP $PY experiments/gen_transplant.py --phase donor --donor $DONOR --layer_donor $Ld $COMMON --donor_bank $bank"
  elif [ "$MODELS_READY" -ne 1 ]; then
    log "CONFIG-SKIP donor_L${Ld} (models absent)"; n_skip=$((n_skip+1))
  elif [ -f "$bank" ] && validate_bank "$bank" | grep -q VALIDATE-OK; then
    log "skip donor_L${Ld} (bank exists, validated)"; n_done=$((n_done+1))
  else
    now=$(elapsed_min)
    if [ $(( now + est_d + 2 )) -gt "$BUDGET_MIN" ]; then
      log "BUDGET-SKIP donor_L${Ld} (elapsed ${now}m + est ${est_d}m > ${BUDGET_MIN}m)"; n_skip=$((n_skip+1))
    else
      cap=$(( est_d * 60 * 3 + 1200 ))
      log "RUN donor_L${Ld} (est ${est_d}m, cap ${cap}s, elapsed ${now}m) -> engine/transplant_donor_L${Ld}.log"
      t=$(date +%s)
      timeout --signal=TERM --kill-after=60 "${cap}s" bash -c \
        "$ENVP $PY experiments/gen_transplant.py --phase donor --donor $DONOR --layer_donor $Ld $COMMON --donor_bank $bank" \
        >> "engine/transplant_donor_L${Ld}.log" 2>&1 </dev/null
      rc=$?; dt=$(( $(date +%s) - t ))
      if [ "$rc" -eq 0 ] && [ -f "$bank" ] && validate_bank "$bank" | grep -q VALIDATE-OK; then
        log "done donor_L${Ld} (${dt}s)"; n_done=$((n_done+1))
      else
        log "FAIL donor_L${Ld} (rc ${rc}, ${dt}s) — recipient phase for this pair will CONFIG-skip"; n_fail=$((n_fail+1))
      fi
    fi
  fi

  # recipient phase -> result (needs the bank; gen_transplant hard-aborts if bank absent)
  if [ "$DRYRUN" -eq 1 ]; then
    echo "DRYRUN recip_L${Ld}to${Lr} est=${est_r}m out=${out} collateral=${collat}"
    log "DRYRUN recip_L${Ld}to${Lr} cmd: $ENVP $PY experiments/gen_transplant.py --phase recipient --donor $DONOR --recipient $RECIP --layer_recip $Lr $COMMON $COLLAT_FLAG --donor_bank $bank --out $out"
  elif [ "$MODELS_READY" -ne 1 ]; then
    log "CONFIG-SKIP recip_L${Ld}to${Lr} (models absent)"; n_skip=$((n_skip+1))
  elif [ ! -f "$bank" ]; then
    log "CONFIG-SKIP recip_L${Ld}to${Lr} (donor bank ${bank} absent — donor phase skipped/failed)"; n_skip=$((n_skip+1))
  else
    run_cell SCIENCE "recip_L${Ld}to${Lr}" "$est_r" "$out" validate_recip \
      "$ENVP $PY experiments/gen_transplant.py --phase recipient --donor $DONOR --recipient $RECIP --layer_recip $Lr $COMMON $COLLAT_FLAG --donor_bank $bank --out $out"
  fi
  log "PROGRESS pair L${Ld}to${Lr} done — jobs ${n_done} done / ${n_fail} fail / ${n_skip} skip, elapsed $(elapsed_min)m/${BUDGET_MIN}m"
done

# ---------------------------------------------------------------- post-run report
if [ "$DRYRUN" -ne 1 ]; then
  for spec in "${PAIRS[@]}"; do
    read -r Ld Lr _ _ <<< "$spec"
    out="results/transplant/D3_transplant_E0b_L${Ld}to${Lr}_s0.json"
    [ -f "$out" ] || continue
    $PY - "$out" >> "$LOG" 2>&1 <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
e = d["esr"]; g = d["gates"]; p = d["predictor_ca_vs_success"]
print(f"[e0b post] {d['layer_donor']}->{d['layer_recip']} verdict={d['verdict']} "
      f"esr(proc={e['procrustes']} rand={e['random_mean']}+-{e['random_std']} "
      f"native={e['native']} ident={e['identity']}) "
      f"G1={g['G1_alignment_specific']} G2={g['G2_nontrivial_absolute']} "
      f"G3={g['G3_geometry_predicts']} pred_rho={p['spearman']} p={p['perm_p']}")
EOF
  done
fi
{
  echo "RUN_TRANSPLANT_E0B REPORT $(date '+%F %T')  jobs: ${n_done} done / ${n_fail} fail / ${n_skip} skipped  elapsed $(elapsed_min)m/${BUDGET_MIN}m"
  grep -E 'RUN |done |FAIL |SKIP|ABORT|SMOKE|e0b post|gpu poll|PROGRESS' "$LOG" | tail -60
} > engine/run_transplant_e0b_report.txt
log "================ RUN_TRANSPLANT_E0B COMPLETE (${n_done} done / ${n_fail} fail / ${n_skip} skip) ================"
echo "RUN_TRANSPLANT_E0B_DONE" >> "$LOG"
