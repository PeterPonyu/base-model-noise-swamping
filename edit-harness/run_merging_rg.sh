#!/bin/bash
# run_merging_rg.sh — Merging-law RG operating-curve driver (RE-SCOPED gate, science ruling
# 2026-07-12). Template = run_merging_kg0.sh / run_revins.sh (verbatim skeleton: preflight,
# GPU-idle gate util<25&&mem<1500 x3, CPU self-test smoke gate, budget, DRYRUN), MERGING-RG-
# namespaced (own pid/log/markers — never reuses kg0/revins/u6/instruct/gptj names).
# BUILD-ONLY as authored 2026-07-12: the local GPU is busy with an HL0 completion; this driver
# is verified CPU-side only (bash -n, DRYRUN, and merging_m0.py --selftest) and NOT launched.
#
# WHY RG (not M1): the M0 raw-rho-delta c3 metric was mis-specified (I_cos & I_mag share the
# ||k_a||*sum(S) scaffold) and the c3 decision at the 200-way group was VACUOUS (saturated
# outcome). The proper geometry discriminant is the PARTIAL rho(I_cos, drop | I_mag). This gate
# sweeps the federation size to find where geometry is testable and predictive.
#
# ONE GPU science row, gated behind a CPU self-test smoke:
#
#   SMOKE (CPU, ~5s) — merging_m0.py --selftest: exact-identity assertion + M0 partial-metric/
#     coherence fixtures (negligible/strong/group_fails/hidden_geometry) + RG pass/kill bundles.
#     Writes only under results/merging/selftest/. Gate marker: engine/merging_rg_selftest.ok.
#
#   RG (GPU) — merging_m0.py --rg: operating curve over group sizes {2,3,5,10,20,50,100} x seeds
#     {0,1,2} at L12 on Llama-3.2-1B. REUSES the existing per-seed solo vectors where present
#     (results/merging/Llama-3.2-1B_L12_s{seed}/phase1_vectors.npz) and computes fresh 200-edit
#     phase-1 passes only where absent. Per-(g,seed) stats use the PARTIAL metric; the
#     pre-registered pass rule (encoded verbatim in the RG table JSON) is: PASS iff partial
#     rho(I_cos, drop | I_mag) >= 0.15 in >= 2 non-negligible, c2-passing group sizes across >= 2
#     seeds; KILL if partial < 0.15 everywhere testable or collapses under own-magnitude
#     partialling. Writes results/merging/RG_operating_curve_table.json + a bundle under
#     results/merging/Llama-3.2-1B_L12_RG/. Standalone CPU re-analysis later:
#       merging_m0.py --rg_phase2_dir results/merging/Llama-3.2-1B_L12_RG
#
# ROME value-opt stays fp32 (editor's own rule; this driver never passes --model_dtype bf16).
# HONEST GPU ESTIMATE: the REAL M0 s0 run measured 91s total (200 edits + all regime merges) on
# this 5090 — NOT the padded 75-min-per-seed figure. So: s0 solo vectors already exist (0 edit
# cost, reused); s1+s2 need fresh 200-edit phase-1 (~90s each ~= 3 min); RG tiled-merge
# measurement across 7 group sizes x 3 seeds is ~1400 cheap forwards/seed (~1-2 min total).
# Realistic total ~= 5-8 GPU-min; the ruling's "~1 GPU-h" was a conservative pad. BUDGET_MIN and
# the row est below stay generously padded so a slow/thermally-throttled card still lands it.
set -u
H=/home/zeyufu/Desktop/idea-feasibility-analysis/edit-harness
cd "$H" || exit 1
PY=/home/zeyufu/miniconda3/envs/dl/bin/python3
LOG=engine/run_merging_rg.log
BUDGET_MIN=${BUDGET_MIN:-90}
RG_SEEDS=${RG_SEEDS:-0,1,2}
RG_GROUP_SIZES=${RG_GROUP_SIZES:-2,3,5,10,20,50,100}
mkdir -p engine results/merging results/merging/selftest
echo $$ > engine/run_merging_rg.pid
[ -f engine/merging_rg_round_start ] || stat -c %Y engine/run_merging_rg.pid > engine/merging_rg_round_start
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "================ RUN_MERGING_RG START (pid $$, budget ${BUDGET_MIN}m, seeds ${RG_SEEDS}, g ${RG_GROUP_SIZES}) ================"

# ---------------------------------------------------------------- Phase 0a: CPU pre-flight (HARD: code/tool/data presence)
pf_fail=0
pf(){ if eval "$2"; then log "preflight OK: $1"; else log "PREFLIGHT-FAIL: $1"; pf_fail=1; fi; }
pf "python env (torch+numpy)" "$PY -c 'import torch, numpy' 2>/dev/null"
pf "merging_m0.py present" "[ -f experiments/merging_m0.py ]"
pf "merging_m0 has --rg flag" "grep -q -- '\"--rg\"' experiments/merging_m0.py"
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
  rm -f engine/merging_rg_selftest.ok
  log "SMOKE merging_m0 --selftest (CPU, ~5s) -> engine/merging_rg_selftest.log"
  if $PY experiments/merging_m0.py --selftest > engine/merging_rg_selftest.log 2>&1; then
    if grep -q "ALL CHECKS PASSED" engine/merging_rg_selftest.log; then
      : > engine/merging_rg_selftest.ok
      log "SMOKE OK: self-test passed (identity + M0 partial/coherence fixtures + RG pass/kill)"
    else
      log "ABORT: self-test ran but did not report ALL CHECKS PASSED"; exit 4
    fi
  else
    log "ABORT: self-test failed (see engine/merging_rg_selftest.log)"; exit 4
  fi
fi

# ---------------------------------------------------------------- Phase 0c: GPU idle gate
if [ "$DRYRUN" -eq 1 ]; then
  log "DRYRUN=1 -- skipping self-test + GPU idle gate, printing the RG plan without executing"
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

# validate the RG outputs: the RG table (pre-reg pass rule + a verdict) + the bundle npz
validate_rg(){
  $PY - "$1" "$2" <<'EOF'
import json, sys, numpy as np
table, meas = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(table))
except Exception as e:
    print(f"VALIDATE-FAIL table unparseable: {e}"); sys.exit(1)
v = d.get("verdict", {})
if v.get("overall") not in ("PASS", "KILL", "MIXED", "INCONCLUSIVE"):
    print(f"VALIDATE-FAIL bad verdict.overall: {v.get('overall')!r}"); sys.exit(1)
if not d.get("per_g_summary"):
    print("VALIDATE-FAIL no per_g_summary"); sys.exit(1)
if not d.get("pass_rule"):
    print("VALIDATE-FAIL no pass_rule recorded"); sys.exit(1)
try:
    a = np.load(meas)
except Exception as e:
    print(f"VALIDATE-FAIL measurements npz unreadable: {e}"); sys.exit(1)
need = {"obs_seed", "obs_g", "obs_edit", "obs_logit_post", "mem_seed", "mem_edit"}
missing = need - set(a.files)
if missing:
    print(f"VALIDATE-FAIL measurements npz missing {missing}"); sys.exit(1)
print(f"VALIDATE-OK verdict={v.get('overall')} qualifying_g={v.get('qualifying_group_sizes')}")
EOF
}

# ---------------------------------------------------------------- RG (GPU): the single science row
# RG_LAYER env knob (2026-07-12, depth-generality tier after the L12 RG PASS): default 12
# keeps the original artifact names so the completed L12 run stays idempotent; other layers
# get layer-tagged table/dir and can never clobber the confirmed L12 artifacts.
RG_LAYER="${RG_LAYER:-12}"
RG_DIR="results/merging/Llama-3.2-1B_L${RG_LAYER}_RG"
if [ "$RG_LAYER" = "12" ]; then
  TABLE="results/merging/RG_operating_curve_table.json"
else
  TABLE="results/merging/RG_operating_curve_table_L${RG_LAYER}.json"
fi
MEAS="${RG_DIR}/rg_measurements.npz"
EST=${EST:-20}   # generously padded over the ~5-8 GPU-min realistic basis (M0 s0 = 91s measured)
CMD="$ENVP $PY experiments/merging_m0.py --rg --model data/models/Llama-3.2-1B --data data/counterfact.json --n_edits 200 --layer ${RG_LAYER} --steps 20 --lr 0.1 --device cuda --rg_seeds ${RG_SEEDS} --rg_group_sizes ${RG_GROUP_SIZES} --out_dir results/merging --table_out ${TABLE}"

if [ "$DRYRUN" -eq 1 ]; then
  echo "DRYRUN rg est=${EST}m -> ${TABLE}"
  echo "DRYRUN cmd: ${CMD}"
  log "DRYRUN rg est=${EST}m cmd: ${CMD}"
else
  now=$(elapsed_min)
  if [ $(( now + EST + 2 )) -gt "$BUDGET_MIN" ]; then
    log "BUDGET-SKIP rg (elapsed ${now}m + est ${EST}m > ${BUDGET_MIN}m)"; n_skip=$((n_skip+1))
  elif [ -f "$TABLE" ] && [ -f "$MEAS" ] && validate_rg "$TABLE" "$MEAS" | grep -q VALIDATE-OK; then
    log "skip rg (exists, validated)"; n_done=$((n_done+1))
  else
    cap=$(( EST * 60 * 3 + 1200 ))
    log "RUN rg (est ${EST}m, cap ${cap}s, elapsed ${now}m) -> engine/merging_rg_run.log"
    t=$(date +%s)
    timeout --signal=TERM --kill-after=60 "${cap}s" bash -c "$CMD" >> engine/merging_rg_run.log 2>&1
    rc=$?; dt=$(( $(date +%s) - t ))
    if [ "$rc" -eq 0 ] && [ -f "$TABLE" ] && [ -f "$MEAS" ]; then
      vres=$(validate_rg "$TABLE" "$MEAS")
      if echo "$vres" | grep -q VALIDATE-FAIL; then
        mv "$TABLE" "$TABLE.INVALID" 2>/dev/null
        log "FAIL rg (${dt}s) OUTPUT-INVALID: ${vres}"; n_fail=$((n_fail+1))
      else
        log "done rg (${dt}s) ${vres}"; n_done=$((n_done+1))
      fi
    else
      log "FAIL rg (rc ${rc}, ${dt}s)"; n_fail=$((n_fail+1))
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
print(f"[rg post] VERDICT overall={v['overall']} qualifying_g={v['qualifying_group_sizes']} "
      f"testable_cells={v['n_testable_cells']}")
b = d.get("scoped_federation_boundary", {})
print(f"[rg post] scoped-federation boundary: per_seed={b.get('per_seed_largest_nonsaturated_g')} "
      f"conservative={b.get('conservative_min_across_seeds')}")
for g, s in d["per_g_summary"].items():
    print(f"[rg post]   g={g}: partial_by_seed={s['partial_by_seed']} "
          f"pass_geometry={s['n_seeds_pass_geometry']} qualifies={s['qualifies']}")
EOF
  log "post: parsed ${TABLE}"
fi

{
  echo "RUN_MERGING_RG REPORT $(date '+%F %T')  jobs: ${n_done} done / ${n_fail} fail / ${n_skip} skipped  elapsed $(elapsed_min)m/${BUDGET_MIN}m"
  grep -E 'RUN |done |FAIL |SKIP|ABORT|SMOKE|VERDICT|rg post|gpu poll' "$LOG" | tail -60
} > engine/run_merging_rg_report.txt
log "================ RUN_MERGING_RG COMPLETE (${n_done} done / ${n_fail} fail / ${n_skip} skip) ================"
echo "RUN_MERGING_RG_DONE" >> "$LOG"
