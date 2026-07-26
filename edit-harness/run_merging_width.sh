#!/bin/bash
# run_merging_width.sh — D2 width-series merging RG driver (family held fixed vs width).
# Template = run_merging_rg.sh (verbatim skeleton: preflight, GPU-idle gate util<25&&mem<1500
# x3, CPU self-test smoke gate, budget, DRYRUN, pid-by-file / kill -0 only). NEW here: every
# knob that run_merging_rg.sh hard-codes to Llama-3.2-1B is env-parameterized (MODEL_DIR,
# MODEL_TAG, LAYER, RG_SEEDS, RG_GROUP_SIZES), and H/PY are resolved dynamically instead of
# hard-coded absolute paths so this driver runs BOTH locally (this repo, this 5090) and
# on-box (e.g. the Pro-6000 cloud wave, where PY=/root/miniconda3/bin/python and the repo
# lives under /root/...). See docs/plans/PREREG-D2-WIDTH-RG-20260714.md for the prereg this
# implements (layer rule, group-size/seed choices, width-ordering hypothesis, VRAM table).
#
# BUILD-ONLY as authored 2026-07-14: authored by the D2 build agent under a no-GPU-runs, no-
# network mandate. Verified CPU-side only (bash -n, merging_m0.py --selftest, DRYRUN=1 for the
# local Qwen2.5-1.5B config) and NOT launched by the authoring agent.
#
# NAMESPACING (never collides with run_merging_rg.sh, run_merging_kg0.sh, or another
# MODEL_TAG's own run of THIS driver):
#   - driver-side pid/log/markers are suffixed by MODEL_TAG: engine/run_merging_width_<tag>.*
#   - the RG results table is ALWAYS written with an explicit --table_out
#     (results/merging/RG_operating_curve_table_<tag>_L<layer>.json) — this driver never
#     relies on merging_m0.py's default top-level table_out path (that path belongs to the
#     canonical Llama-3.2-1B L12 gate; existing cloud/run_wave_pro6000.sh's merging_worker()
#     omits --table_out and is a latent collision hazard on that path — noted in the prereg,
#     not touched here).
#   - the RG bundle DIRECTORY is derived by merging_m0.py itself from the real model
#     basename (results/merging/<basename(MODEL_DIR)>_L<layer>_RG/), which is already unique
#     per model even when two MODEL_TAGs share a layer number (e.g. Qwen2.5-1.5B and
#     Qwen2.5-7B both land on L21 — see prereg "Qwen2.5-1.5B and Qwen2.5-7B land on the SAME
#     layer index" note); this driver mirrors that derivation ONLY to build the --table_out /
#     validate_rg() MEAS argument, never to pick a colliding directory itself.
#
# ROME value-opt stays fp32 (editor's own rule; merging_m0.py's _load_edit_model always loads
# dtype=torch.float32 — this driver never adds a --model_dtype override, because --rg has none).
set -u
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$H" || exit 1
PY=${PY:-/home/zeyufu/miniconda3/envs/dl/bin/python3}

# ---------------------------------------------------------------- required env
MODEL_DIR=${MODEL_DIR:-}
MODEL_TAG=${MODEL_TAG:-}
if [ -z "$MODEL_DIR" ] || [ -z "$MODEL_TAG" ]; then
  echo "usage: MODEL_DIR=<path to model> MODEL_TAG=<short tag, e.g. qwen15b> [LAYER=auto75] [RG_SEEDS=0,1,2] [RG_GROUP_SIZES=2,3,5,10,20,50,100] [BUDGET_MIN=90] [EST_MIN=30] [DRYRUN=1] $0" >&2
  exit 1
fi

LAYER=${LAYER:-auto75}
RG_SEEDS=${RG_SEEDS:-0,1,2}
RG_GROUP_SIZES=${RG_GROUP_SIZES:-2,3,5,10,20,50,100}
BUDGET_MIN=${BUDGET_MIN:-90}
EST_MIN=${EST_MIN:-30}
DRYRUN=${DRYRUN:-0}

LOG="engine/run_merging_width_${MODEL_TAG}.log"
mkdir -p engine results/merging results/merging/selftest
echo $$ > "engine/run_merging_width_${MODEL_TAG}.pid"
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "================ RUN_MERGING_WIDTH START tag=${MODEL_TAG} model=${MODEL_DIR} pid=$$ budget=${BUDGET_MIN}m seeds=${RG_SEEDS} g=${RG_GROUP_SIZES} ================"

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
pf "MODEL_DIR exists (${MODEL_DIR})" "[ -d '$MODEL_DIR' ]"
pf "MODEL_DIR has config.json" "[ -f '$MODEL_DIR/config.json' ]"
pf "disk >=10GB free" "[ \$(df --output=avail -BG . | tail -1 | tr -dc 0-9) -ge 10 ]"
if [ "$pf_fail" -ne 0 ]; then log "ABORT: preflight failed"; exit 3; fi

# ---------------------------------------------------------------- Phase 0a.2: LAYER RULE (fixed a priori: 75% relative depth)
# floor(num_hidden_layers * 0.75), matching every existing RG cell's convention
# (Llama-3.2-1B L12=floor(16*.75), Mistral-7B L24=floor(32*.75) — see the prereg doc's
# "LAYER RULE" table). LAYER=auto75 (default) computes it from the model's own config.json,
# which is the ONLY layer selection mode this driver expects to be used with routinely; an
# explicit numeric LAYER override is still honored (never hard-aborted) but is logged loudly
# so a mismatch is visible rather than silently baked into a filename.
auto75=$($PY - "$MODEL_DIR" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1].rstrip("/") + "/config.json"))
nl = int(d.get("num_hidden_layers") or d["n_layer"])  # n_layer = GPT-2-family key
print(int(nl * 0.75))
EOF
) || { log "ABORT: could not read num_hidden_layers from ${MODEL_DIR}/config.json"; exit 3; }
if [ "$LAYER" = "auto75" ]; then
  LAYER="$auto75"
  log "LAYER auto-computed from config.json: floor(num_hidden_layers*0.75) = ${LAYER}"
else
  if [ "$LAYER" != "$auto75" ]; then
    log "WARN: explicit LAYER=${LAYER} != the 75%-depth rule's ${auto75} for this model — proceeding with the explicit override, but this is NOT the D2 layer rule"
  else
    log "LAYER=${LAYER} explicit override matches the 75%-depth rule (${auto75}) — consistent"
  fi
fi

# ---------------------------------------------------------------- Phase 0a.3: refuse-guard (review fix, 2026-07-14)
# Llama-3.2-1B @ L{8,12,14} is EXACTLY the existing canonical depth-tier bundle set
# (results/merging/Llama-3.2-1B_L{8,12,14}_RG/ + RG_operating_curve_table[_L{8,14}].json —
# the confirmed L12 gate + the L8/L14 depth-generality tier). D2's model set never needs
# this driver to touch that model: the width series is Qwen2.5-{1.5B,7B,14B} (+ Llama-3.1-8B
# for the second family) and Llama-3.2-1B enters the width-ordering comparison only by
# CITING the already-landed canonical artifacts (see the prereg's "LAYER TABLE", cite-only
# rows), never by re-running them. Refuse loudly rather than let an operator typo
# (MODEL_DIR pointed at the wrong model, or a copy-pasted LAYER) silently re-enter and
# potentially clobber those bundle dirs.
MODEL_BASENAME="$(basename "${MODEL_DIR%/}")"
if [ "$MODEL_BASENAME" = "Llama-3.2-1B" ]; then
  case "$LAYER" in
    8|12|14)
      echo "ABORT: refuse-guard fired — MODEL_DIR resolves to Llama-3.2-1B at LAYER=${LAYER}, which re-enters the EXISTING canonical depth-tier bundle (results/merging/Llama-3.2-1B_L${LAYER}_RG/). This driver's prereg model set never needs this combination — cite the existing artifact instead of rerunning it. If this was intentional, use run_merging_rg.sh (RG_LAYER=${LAYER}) directly, not run_merging_width.sh." >&2
      log "ABORT: refuse-guard fired — MODEL_DIR resolves to Llama-3.2-1B at LAYER=${LAYER} (see PREREG-D2-WIDTH-RG-20260714.md)"
      exit 5
      ;;
  esac
fi

# ---------------------------------------------------------------- Phase 0b: CPU self-test smoke gate
# Skipped on DRYRUN so a plan-only invocation leaves results/ byte-untouched (run_merging_rg.sh
# / run_revins.sh precedent). The self-test itself is model-agnostic (synthetic fixtures, no
# model/GPU) so it is shared logic, but the .ok marker is namespaced by MODEL_TAG anyway to
# avoid two concurrent width-driver invocations (different tags, same box) racing on one file.
SELFTEST_OK="engine/merging_width_selftest_${MODEL_TAG}.ok"
if [ "$DRYRUN" -ne 1 ]; then
  rm -f "$SELFTEST_OK"
  SELFTEST_LOG="engine/merging_width_selftest_${MODEL_TAG}.log"
  log "SMOKE merging_m0 --selftest (CPU, ~5s) -> ${SELFTEST_LOG}"
  if $PY experiments/merging_m0.py --selftest > "$SELFTEST_LOG" 2>&1; then
    if grep -q "ALL CHECKS PASSED" "$SELFTEST_LOG"; then
      : > "$SELFTEST_OK"
      log "SMOKE OK: self-test passed (identity + M0 partial/coherence fixtures + RG pass/kill)"
    else
      log "ABORT: self-test ran but did not report ALL CHECKS PASSED"; exit 4
    fi
  else
    log "ABORT: self-test failed (see ${SELFTEST_LOG})"; exit 4
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

# validate the RG outputs: the RG table (pre-reg pass rule + a verdict) + the bundle npz
# (verbatim copy of run_merging_rg.sh's validate_rg — same schema, same checks)
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

# ---------------------------------------------------------------- RG (GPU): the single science row for this MODEL_TAG
# (MODEL_BASENAME already computed in the Phase 0a.3 refuse-guard above — reused here)
RG_DIR="results/merging/${MODEL_BASENAME}_L${LAYER}_RG"
TABLE="results/merging/RG_operating_curve_table_${MODEL_TAG}_L${LAYER}.json"
MEAS="${RG_DIR}/rg_measurements.npz"
CMD="$ENVP $PY experiments/merging_m0.py --rg --model ${MODEL_DIR} --data data/counterfact.json --n_edits 200 --layer ${LAYER} --steps 20 --lr 0.1 --device cuda --rg_seeds ${RG_SEEDS} --rg_group_sizes ${RG_GROUP_SIZES} --out_dir results/merging --table_out ${TABLE}"

if [ "$DRYRUN" -eq 1 ]; then
  echo "DRYRUN tag=${MODEL_TAG} model=${MODEL_DIR} layer=${LAYER} est=${EST_MIN}m -> ${TABLE}"
  echo "DRYRUN rg_dir (npz bundle, merging_m0-derived): ${RG_DIR}"
  echo "DRYRUN cmd: ${CMD}"
  log "DRYRUN tag=${MODEL_TAG} layer=${LAYER} est=${EST_MIN}m cmd: ${CMD}"
else
  now=$(elapsed_min)
  if [ $(( now + EST_MIN + 2 )) -gt "$BUDGET_MIN" ]; then
    log "BUDGET-SKIP rg tag=${MODEL_TAG} (elapsed ${now}m + est ${EST_MIN}m > ${BUDGET_MIN}m)"
  elif [ -f "$TABLE" ] && [ -f "$MEAS" ] && validate_rg "$TABLE" "$MEAS" | grep -q VALIDATE-OK; then
    log "skip rg tag=${MODEL_TAG} (exists, validated)"
  else
    cap=$(( EST_MIN * 60 * 3 + 1200 ))
    log "RUN rg tag=${MODEL_TAG} (est ${EST_MIN}m, cap ${cap}s, elapsed ${now}m) -> engine/merging_width_${MODEL_TAG}_run.log"
    t=$(date +%s)
    timeout --signal=TERM --kill-after=60 "${cap}s" bash -c "$CMD" >> "engine/merging_width_${MODEL_TAG}_run.log" 2>&1
    rc=$?; dt=$(( $(date +%s) - t ))
    if [ "$rc" -eq 0 ] && [ -f "$TABLE" ] && [ -f "$MEAS" ]; then
      vres=$(validate_rg "$TABLE" "$MEAS")
      if echo "$vres" | grep -q VALIDATE-FAIL; then
        mv "$TABLE" "$TABLE.INVALID" 2>/dev/null
        log "FAIL rg tag=${MODEL_TAG} (${dt}s) OUTPUT-INVALID: ${vres}"
      else
        log "done rg tag=${MODEL_TAG} (${dt}s) ${vres}"
      fi
    else
      log "FAIL rg tag=${MODEL_TAG} (rc ${rc}, ${dt}s)"
    fi
  fi
fi

# ---------------------------------------------------------------- Post-run report
if [ "$DRYRUN" -ne 1 ] && [ -f "$TABLE" ]; then
  log "---------------- POST-RUN (CPU) tag=${MODEL_TAG} ----------------"
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
  echo "RUN_MERGING_WIDTH REPORT tag=${MODEL_TAG} model=${MODEL_DIR} layer=${LAYER} $(date '+%F %T')  elapsed $(elapsed_min)m/${BUDGET_MIN}m"
  grep -E 'RUN |done |FAIL |SKIP|BUDGET-SKIP|ABORT|SMOKE|LAYER|VERDICT|rg post|gpu poll' "$LOG" | tail -60
} > "engine/run_merging_width_${MODEL_TAG}_report.txt"
log "================ RUN_MERGING_WIDTH COMPLETE tag=${MODEL_TAG} ================"
echo "RUN_MERGING_WIDTH_DONE tag=${MODEL_TAG}" >> "$LOG"
