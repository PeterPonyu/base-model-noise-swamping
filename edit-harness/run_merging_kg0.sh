#!/bin/bash
# run_merging_kg0.sh — Merging law M0 kill-gate driver (direction 2.4). Template = run_revins.sh
# (verbatim skeleton: preflight, GPU-idle gate util<25&&mem<1500 x3, run_row/validate,
# BUDGET_MIN, DRYRUN), MERGING-namespaced (own pid/log/markers — never reuses revins/u6/
# instruct/gptj/8bcausal names). BUILD-ONLY as authored 2026-07-11: the local GPU is busy
# with P2/P3 lanes at authoring time, so this driver is verified CPU-side only (bash -n,
# DRYRUN, and merging_m0.py --selftest) and NOT launched by the authoring agent.
#
# ONE GPU science row (~75 GPU-min, 1 seed), gated behind a CPU self-test smoke:
#
#   SMOKE (CPU, ~5s) — merging_m0.py --selftest: asserts the exact cross-term identity
#     ΔW_b @ k_a == r_b(k_b·k_a)/||k_b||^2 to fp64 tol AND runs the full phase-2 analysis on
#     synthetic negligible/strong fixtures, checking the verdict logic fires (KILL vs PASS).
#     Writes only under results/merging/selftest/. Gate marker: engine/merging_selftest.ok.
#
#   PHASE1 (GPU) — merging_m0.py phase 1: 200 independent L12 s0 ROME edits on Llama-3.2-1B
#     with saved per-edit (k, r, S, denom) + solo efficacy; THREE merge regimes measured on
#     the model (natural_pairwise 45 pairs / natural_group 200 / enriched_conflict 45 pairs);
#     merging itself = CPU tensor addition of the saved rank-one factors. Auto-runs phase 2 and
#     writes results/merging/M0_killgate_table.json with the explicit KILL/PASS/MIXED verdict.
#     Vectors persist under results/merging/Llama-3.2-1B_L12_s0/ so phase 2 can be re-run
#     STANDALONE on CPU later:  merging_m0.py --phase2_dir results/merging/Llama-3.2-1B_L12_s0
#
# ROME value-opt stays fp32 (editor's own rule; this driver never passes --model_dtype bf16).
# GPU-HOUR ESTIMATE: 200 edits x ~20 value-opt steps dominate (~75 GPU-min per spec 2.4 M0);
# the 3 regimes add ~330 single forwards (cheap). Total ~= 75 GPU-min = ~1.3 GPU-hours.
set -u
H=/home/zeyufu/Desktop/idea-feasibility-analysis/edit-harness
cd "$H" || exit 1
PY=/home/zeyufu/miniconda3/envs/dl/bin/python3
LOG=engine/run_merging_kg0.log
BUDGET_MIN=${BUDGET_MIN:-150}
mkdir -p engine results/merging results/merging/selftest
echo $$ > engine/run_merging_kg0.pid
[ -f engine/merging_round_start ] || stat -c %Y engine/run_merging_kg0.pid > engine/merging_round_start
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "================ RUN_MERGING_KG0 START (pid $$, budget ${BUDGET_MIN}m) ================"

# ---------------------------------------------------------------- Phase 0a: CPU pre-flight (HARD: code/tool/data presence)
pf_fail=0
pf(){ if eval "$2"; then log "preflight OK: $1"; else log "PREFLIGHT-FAIL: $1"; pf_fail=1; fi; }
pf "python env (torch+numpy)" "$PY -c 'import torch, numpy' 2>/dev/null"
pf "merging_m0.py present" "[ -f experiments/merging_m0.py ]"
pf "rome_native editor present" "[ -f editors/rome_native.py ]"
pf "metrics.py present" "[ -f metrics.py ]"
pf "arch_compat present" "[ -f editors/arch_compat.py ]"
pf "counterfact.json" "[ -f data/counterfact.json ]"
pf "model Llama-3.2-1B" "[ -d data/models/Llama-3.2-1B ]"
pf "disk >=10GB free" "[ \$(df --output=avail -BG /home | tail -1 | tr -dc 0-9) -ge 10 ]"
if [ "$pf_fail" -ne 0 ]; then log "ABORT: preflight failed"; exit 3; fi

# ---------------------------------------------------------------- Phase 0b: CPU self-test smoke gate
# Skipped on DRYRUN so a plan-only invocation leaves results/ byte-untouched (run_revins.sh
# MEDIUM-1 precedent); the GPU row is skipped on DRYRUN anyway, so the .ok marker is not needed.
DRYRUN=${DRYRUN:-0}
if [ "$DRYRUN" -ne 1 ]; then
  rm -f engine/merging_selftest.ok
  log "SMOKE merging_m0 --selftest (CPU, ~5s) -> engine/merging_selftest.log"
  if $PY experiments/merging_m0.py --selftest > engine/merging_selftest.log 2>&1; then
    if grep -q "ALL CHECKS PASSED" engine/merging_selftest.log; then
      : > engine/merging_selftest.ok
      log "SMOKE OK: self-test passed (identity + negligible->KILL + strong->PASS)"
    else
      log "ABORT: self-test ran but did not report ALL CHECKS PASSED"; exit 4
    fi
  else
    log "ABORT: self-test failed (see engine/merging_selftest.log)"; exit 4
  fi
fi

# ---------------------------------------------------------------- Phase 0c: GPU idle gate
if [ "$DRYRUN" -eq 1 ]; then
  log "DRYRUN=1 -- skipping self-test + GPU idle gate, printing the phase-1 plan without executing"
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

# ---------------------------------------------------------------- helpers (run_revins template, trimmed to one row-type)
ENVP="env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1"
n_done=0; n_fail=0; n_skip=0

# validate the phase-1 outputs: the killgate table + the persisted vectors npz
validate_m0(){
  $PY - "$1" "$2" <<'EOF'
import json, sys, numpy as np
table, vnpz = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(table))
except Exception as e:
    print(f"VALIDATE-FAIL table unparseable: {e}"); sys.exit(1)
v = d.get("verdict", {})
if v.get("overall") not in ("KILL", "PASS", "MIXED"):
    print(f"VALIDATE-FAIL no valid verdict.overall: {v.get('overall')!r}"); sys.exit(1)
if not d.get("regimes"):
    print("VALIDATE-FAIL no regimes block"); sys.exit(1)
try:
    a = np.load(vnpz)
except Exception as e:
    print(f"VALIDATE-FAIL vectors npz unreadable: {e}"); sys.exit(1)
need = {"K", "R", "denom", "S", "key_norm", "logit_solo", "argmax_ok_solo"}
missing = need - set(a.files)
if missing:
    print(f"VALIDATE-FAIL vectors npz missing {missing}"); sys.exit(1)
if np.isnan(a["logit_solo"].astype(float)).all():
    print("VALIDATE-FAIL logit_solo all-NaN"); sys.exit(1)
print(f"VALIDATE-OK verdict={v.get('overall')}")
EOF
}

# ---------------------------------------------------------------- PHASE1 (GPU): the single science row
TABLE="results/merging/M0_killgate_table.json"
RUN_DIR="results/merging/Llama-3.2-1B_L12_s0"
VNPZ="${RUN_DIR}/phase1_vectors.npz"
EST=90   # padded over the 75 GPU-min spec estimate
CMD="$ENVP $PY experiments/merging_m0.py --model data/models/Llama-3.2-1B --data data/counterfact.json --n_edits 200 --layer 12 --seed 0 --steps 20 --lr 0.1 --device cuda --pair_pool 10 --group_size 200 --n_enriched 45 --out_dir results/merging --table_out ${TABLE}"

if [ "$DRYRUN" -eq 1 ]; then
  echo "DRYRUN phase1_m0 est=${EST}m -> ${TABLE}"
  echo "DRYRUN cmd: ${CMD}"
  log "DRYRUN phase1_m0 est=${EST}m cmd: ${CMD}"
else
  now=$(elapsed_min)
  if [ $(( now + EST + 2 )) -gt "$BUDGET_MIN" ]; then
    log "BUDGET-SKIP phase1_m0 (elapsed ${now}m + est ${EST}m > ${BUDGET_MIN}m)"; n_skip=$((n_skip+1))
  elif [ -f "$TABLE" ] && [ -f "$VNPZ" ] && validate_m0 "$TABLE" "$VNPZ" | grep -q VALIDATE-OK; then
    log "skip phase1_m0 (exists, validated)"; n_done=$((n_done+1))
  else
    cap=$(( EST * 60 * 3 + 1200 ))
    log "RUN phase1_m0 (est ${EST}m, cap ${cap}s, elapsed ${now}m) -> engine/merging_phase1.log"
    t=$(date +%s)
    timeout --signal=TERM --kill-after=60 "${cap}s" bash -c "$CMD" >> engine/merging_phase1.log 2>&1
    rc=$?; dt=$(( $(date +%s) - t ))
    if [ "$rc" -eq 0 ] && [ -f "$TABLE" ] && [ -f "$VNPZ" ]; then
      vres=$(validate_m0 "$TABLE" "$VNPZ")
      if echo "$vres" | grep -q VALIDATE-FAIL; then
        mv "$TABLE" "$TABLE.INVALID" 2>/dev/null
        log "FAIL phase1_m0 (${dt}s) OUTPUT-INVALID: ${vres}"; n_fail=$((n_fail+1))
      else
        log "done phase1_m0 (${dt}s) ${vres}"; n_done=$((n_done+1))
      fi
    else
      log "FAIL phase1_m0 (rc ${rc}, ${dt}s)"; n_fail=$((n_fail+1))
    fi
  fi
fi

# ---------------------------------------------------------------- Post-run report
if [ "$DRYRUN" -ne 1 ] && [ -f "$TABLE" ]; then
  log "---------------- POST-RUN (CPU) ----------------"
  $PY - "$TABLE" >> "$LOG" 2>&1 <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
v = d["verdict"]
print(f"[merging post] VERDICT overall={v['overall']} "
      f"c1={v['criterion_1_negligible_even_enriched']} "
      f"c2={v['criterion_2_rho_where_nonnegligible']} "
      f"c3(geom-partial)={v['criterion_3_geometry_partial']} "
      f"[decided_on={v.get('criterion_3_decided_on')}]")
for name, st in d["regimes"].items():
    print(f"[merging post]   {name}: med|drop|={st['median_abs_drop_logit']} "
          f"rho(Icos,drop)={st['rho_I_cos_drop']} "
          f"partial(Icos,drop|Imag)={st['partial_rho_geom']} "
          f"sat={st['saturated']} c3elig={st['c3_eligible']}")
EOF
  log "post: parsed ${TABLE}"
fi

{
  echo "RUN_MERGING_KG0 REPORT $(date '+%F %T')  jobs: ${n_done} done / ${n_fail} fail / ${n_skip} skipped  elapsed $(elapsed_min)m/${BUDGET_MIN}m"
  grep -E 'RUN |done |FAIL |SKIP|ABORT|SMOKE|VERDICT|merging post|gpu poll' "$LOG" | tail -60
} > engine/run_merging_kg0_report.txt
log "================ RUN_MERGING_KG0 COMPLETE (${n_done} done / ${n_fail} fail / ${n_skip} skip) ================"
echo "RUN_MERGING_KG0_DONE" >> "$LOG"
