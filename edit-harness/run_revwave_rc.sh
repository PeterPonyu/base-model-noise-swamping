#!/bin/bash
# run_revwave_rc.sh — R-C revision-wave cell: ROME federation RG operating curve on a ~13B
# non-Qwen model at 75% relative depth, 3 seeds, g=2..20, n=200 (experiments/merging_m0.py --rg
# path, NOT merging_editors.py — this is a ROME-only federation cell, mirroring
# run_merging_width.sh's D2-width-series template).
#
# WHY THIS DRIVER IS NEW CODE (not a thin wrapper like R-D/R-F): 13B params in bf16 is ~26GB,
# over a single 24GB 4090D card; this revision adds a --device_map/--model_dtype path to
# experiments/merging_m0.py's _load_edit_model (mirrors experiments/killgate_keygeom.py's
# existing --device_map TP path exactly) + fixes _merge_factors/_measure_merged_groups to build
# their GPU tensors on the EDITED LAYER's own device rather than the input-embedding device (the
# two can differ under sharding — see tp_edit_util.py's new resolve_input_device +
# the pre-existing resolve_layer_device). --device_map is OFF by default and hard-fenced to --rg
# mode only in main() (run_phase1's 3-regime kill-gate path was NOT made device_map-aware — see
# the SystemExit guard in merging_m0.py's main()); this driver's whole purpose is to exercise the
# --rg + --device_map auto + --model_dtype bf16 combination for the first time.
#
# MODEL CHOICE — DO NOT DOWNLOAD, USER DECIDES: candidates are Llama-2-13b-hf (40 layers ->
# L30 = floor(40*0.75)) or OLMo-2-1124-13B; MODEL_DIR/MODEL_TAG are REQUIRED env vars (no
# default model path) so an operator must explicitly point this driver at whichever 13B model
# the user has actually put on the box.
#
# BUDGET: 13B under bf16+device_map=auto solo-captures 200 ROME edits (2 forward+backward-ish
# passes per edit for the value-opt, all bf16-forward/fp32-value-opt) then measures ~5 group
# sizes x 3 seeds of cheap forwards. No 13B GPU timing exists yet in this harness (the only
# other >=13B RG cell, Qwen2.5-14B alpha, ran fp32 single-GPU on a bigger-VRAM box, not this
# bf16+2-card path) — EST_MIN is a generous, UNVERIFIED pad; treat the first real run's
# engine/run_revwave_rc_<tag>_run.log timing as the first honest estimate for this configuration
# (same "trust the measured number over the padded estimate" lesson as every other driver in this
# harness — see run_merging_rg.sh's own honest-GPU-estimate note).
#
# Template = run_merging_width.sh (verbatim skeleton: preflight, GPU-idle gate util<25&&mem<1500
# x3, CPU --selftest smoke gate, budget, DRYRUN, pid-by-file / kill -0 only, refuse-guard against
# re-entering an existing canonical bundle). NEW here vs that template: MODEL_DTYPE/DEVICE_MAP
# env knobs threaded to merging_m0.py's --model_dtype/--device_map flags.
#
# BUILD-ONLY as authored 2026-07-16: CPU-validated only (bash -n, merging_m0.py --selftest,
# DRYRUN=1 — see docs/plans/REVWAVE-BUILD-NOTES-2026-07-16.md for exactly what could and could
# not be exercised without the model on this machine); NOT launched by the author.
set -u
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$H" || exit 1
PY=${PY:-/home/zeyufu/miniconda3/envs/dl/bin/python3}

# ---------------------------------------------------------------- required env
MODEL_DIR=${MODEL_DIR:-}
MODEL_TAG=${MODEL_TAG:-}
if [ -z "$MODEL_DIR" ] || [ -z "$MODEL_TAG" ]; then
  echo "usage: MODEL_DIR=<path to ~13B model, e.g. /root/autodl-tmp/models/Llama-2-13b-hf> MODEL_TAG=<short tag, e.g. llama13b> [LAYER=auto75] [MODEL_DTYPE=bf16] [DEVICE_MAP=auto] [RG_SEEDS=0,1,2] [RG_GROUP_SIZES=2,3,5,10,20] [BUDGET_MIN=240] [EST_MIN=90] [DRYRUN=1] $0" >&2
  exit 1
fi

LAYER=${LAYER:-auto75}
MODEL_DTYPE=${MODEL_DTYPE:-bf16}
DEVICE_MAP=${DEVICE_MAP:-auto}
RG_SEEDS=${RG_SEEDS:-0,1,2}
RG_GROUP_SIZES=${RG_GROUP_SIZES:-2,3,5,10,20}
N_EDITS=${N_EDITS:-200}
BUDGET_MIN=${BUDGET_MIN:-240}
EST_MIN=${EST_MIN:-90}
DRYRUN=${DRYRUN:-0}

case "$MODEL_DTYPE" in
  fp32|bf16) ;;
  *) echo "ABORT: MODEL_DTYPE must be fp32|bf16 (got '$MODEL_DTYPE')" >&2; exit 1;;
esac
case "$DEVICE_MAP" in
  none|auto|balanced|balanced_low_0|sequential) ;;
  *) echo "ABORT: DEVICE_MAP must be none|auto|balanced|balanced_low_0|sequential (got '$DEVICE_MAP')" >&2; exit 1;;
esac
# MINOR (revwave review): fp32 + device_map=auto on ~13B (~52GB > 2x24GB) makes accelerate
# silently offload shards to CPU/disk, where in-place W.add_ edits are unreliable — refuse.
if [ "$MODEL_DTYPE" = "fp32" ] && [ "$DEVICE_MAP" != "none" ]; then
  echo "ABORT: fp32 with device_map sharding on a ~13B model exceeds 2x24GB and triggers" \
       "silent CPU/disk offload (in-place edits unreliable there). Use MODEL_DTYPE=bf16." >&2
  exit 1
fi
if [ "$MODEL_DTYPE" = "fp32" ] && [ "$DEVICE_MAP" = "none" ]; then
  echo "WARN: fp32 + device_map=none on a ~13B model needs ~52GB VRAM on ONE card — this will" \
       "almost certainly OOM on a 24GB 4090D. Re-run with DEVICE_MAP=auto (shards across both" \
       "cards) or MODEL_DTYPE=bf16 (halves weight RAM), or both (this driver's own default)." >&2
fi

TAG="${MODEL_TAG}_rc"
LOG="engine/run_revwave_rc_${TAG}.log"
mkdir -p engine results/merging results/merging/selftest
echo $$ > "engine/run_revwave_rc_${TAG}.pid"
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "======== RUN_REVWAVE_RC START tag=${TAG} model=${MODEL_DIR} dtype=${MODEL_DTYPE} device_map=${DEVICE_MAP} pid=$$ budget=${BUDGET_MIN}m seeds=${RG_SEEDS} g=${RG_GROUP_SIZES} ========"

# ---------------------------------------------------------------- Phase 0a: CPU pre-flight
pf_fail=0
pf(){ if eval "$2"; then log "preflight OK: $1"; else log "PREFLIGHT-FAIL: $1"; pf_fail=1; fi; }
pf "python env (torch+numpy)" "$PY -c 'import torch, numpy' 2>/dev/null"
pf "merging_m0.py present" "[ -f experiments/merging_m0.py ]"
pf "merging_m0 has --rg flag" "grep -q -- '\"--rg\"' experiments/merging_m0.py"
pf "merging_m0 has --device_map flag" "grep -q -- '\"--device_map\"' experiments/merging_m0.py"
pf "rome_native editor present" "[ -f editors/rome_native.py ]"
pf "metrics.py present" "[ -f metrics.py ]"
pf "arch_compat present" "[ -f editors/arch_compat.py ]"
pf "tp_edit_util has resolve_input_device" "grep -q 'def resolve_input_device' tp_edit_util.py"
pf "counterfact.json" "[ -f data/counterfact.json ]"
pf "accelerate importable (device_map)" "$PY -c 'import accelerate' 2>/dev/null || [ '$DEVICE_MAP' = 'none' ]"
if [ "$DRYRUN" -ne 1 ]; then
  pf "MODEL_DIR exists (${MODEL_DIR})" "[ -d '$MODEL_DIR' ]"
  pf "MODEL_DIR has config.json" "[ -f '${MODEL_DIR}/config.json' ]"
fi
pf "disk >=10GB free" "[ \$(df --output=avail -BG . | tail -1 | tr -dc 0-9) -ge 10 ]"
if [ "$pf_fail" -ne 0 ]; then log "ABORT: preflight failed"; exit 3; fi

# ---------------------------------------------------------------- Phase 0a.2: LAYER RULE (75% relative depth)
if [ "$DRYRUN" -eq 1 ] && [ ! -f "${MODEL_DIR}/config.json" ]; then
  log "DRYRUN + no local config.json — skipping the auto75 layer read, using LAYER as given"
  [ "$LAYER" = "auto75" ] && LAYER=30
else
  auto75=$($PY - "$MODEL_DIR" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1].rstrip("/") + "/config.json"))
nl = int(d.get("num_hidden_layers") or d["n_layer"])
print(int(nl * 0.75))
EOF
  ) || { log "ABORT: could not read num_hidden_layers from ${MODEL_DIR}/config.json"; exit 3; }
  if [ "$LAYER" = "auto75" ]; then
    LAYER="$auto75"
    log "LAYER auto-computed from config.json: floor(num_hidden_layers*0.75) = ${LAYER}"
  else
    if [ "$LAYER" != "$auto75" ]; then
      log "WARN: explicit LAYER=${LAYER} != the 75%-depth rule's ${auto75} for this model — proceeding with the override"
    else
      log "LAYER=${LAYER} explicit override matches the 75%-depth rule (${auto75})"
    fi
  fi
fi

RG_DIR="results/merging/$(basename "${MODEL_DIR%/}")_L${LAYER}_RG"
TABLE="results/merging/RG_operating_curve_table_${TAG}_L${LAYER}.json"
MEAS="${RG_DIR}/rg_measurements.npz"

# ---------------------------------------------------------------- Phase 0b: CPU self-test smoke gate
SELFTEST_OK="engine/merging_rc_selftest_${TAG}.ok"
if [ "$DRYRUN" -ne 1 ]; then
  rm -f "$SELFTEST_OK"
  SELFTEST_LOG="engine/merging_rc_selftest_${TAG}.log"
  log "SMOKE merging_m0 --selftest (CPU, ~5s) -> ${SELFTEST_LOG}"
  if $PY experiments/merging_m0.py --selftest > "$SELFTEST_LOG" 2>&1; then
    if grep -q "ALL CHECKS PASSED" "$SELFTEST_LOG"; then
      : > "$SELFTEST_OK"
      log "SMOKE OK: self-test passed"
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

CMD="$ENVP $PY experiments/merging_m0.py --rg --model ${MODEL_DIR} --data data/counterfact.json --n_edits ${N_EDITS} --layer ${LAYER} --steps 20 --lr 0.1 --device cuda --model_dtype ${MODEL_DTYPE} --device_map ${DEVICE_MAP} --rg_seeds ${RG_SEEDS} --rg_group_sizes ${RG_GROUP_SIZES} --out_dir results/merging --table_out ${TABLE}"

if [ "$DRYRUN" -eq 1 ]; then
  echo "DRYRUN tag=${TAG} model=${MODEL_DIR} layer=${LAYER} dtype=${MODEL_DTYPE} device_map=${DEVICE_MAP} est=${EST_MIN}m -> ${TABLE}"
  echo "DRYRUN rg_dir (npz bundle, module-derived): ${RG_DIR}"
  echo "DRYRUN cmd: ${CMD}"
  log "DRYRUN tag=${TAG} est=${EST_MIN}m cmd: ${CMD}"
else
  now=$(elapsed_min)
  if [ $(( now + EST_MIN + 2 )) -gt "$BUDGET_MIN" ]; then
    log "BUDGET-SKIP rg tag=${TAG} (elapsed ${now}m + est ${EST_MIN}m > ${BUDGET_MIN}m)"
  elif [ -f "$TABLE" ] && [ -f "$MEAS" ] && validate_rg "$TABLE" "$MEAS" | grep -q VALIDATE-OK; then
    log "skip rg tag=${TAG} (exists, validated)"
  else
    cap=$(( EST_MIN * 60 * 3 + 1800 ))
    log "RUN rg tag=${TAG} (est ${EST_MIN}m, cap ${cap}s, elapsed ${now}m) -> engine/run_revwave_rc_${TAG}_run.log"
    t=$(date +%s)
    timeout --signal=TERM --kill-after=60 "${cap}s" bash -c "$CMD" >> "engine/run_revwave_rc_${TAG}_run.log" 2>&1
    rc=$?; dt=$(( $(date +%s) - t ))
    if [ "$rc" -eq 0 ] && [ -f "$TABLE" ] && [ -f "$MEAS" ]; then
      vres=$(validate_rg "$TABLE" "$MEAS")
      if echo "$vres" | grep -q VALIDATE-FAIL; then
        mv "$TABLE" "$TABLE.INVALID" 2>/dev/null
        log "FAIL rg tag=${TAG} (${dt}s) OUTPUT-INVALID: ${vres}"
      else
        log "done rg tag=${TAG} (${dt}s) ${vres}"
      fi
    else
      log "FAIL rg tag=${TAG} (rc ${rc}, ${dt}s) — check engine/run_revwave_rc_${TAG}_run.log for OOM/device_map errors"
    fi
  fi
fi

{
  echo "RUN_REVWAVE_RC REPORT tag=${TAG} model=${MODEL_DIR} layer=${LAYER} dtype=${MODEL_DTYPE} device_map=${DEVICE_MAP} $(date '+%F %T')  elapsed $(elapsed_min)m/${BUDGET_MIN}m"
  grep -E 'RUN |done |FAIL |SKIP|BUDGET-SKIP|ABORT|SMOKE|LAYER|VERDICT|gpu poll' "$LOG" | tail -60
} > "engine/run_revwave_rc_${TAG}_report.txt"
log "======== RUN_REVWAVE_RC COMPLETE tag=${TAG} ========"
echo "RUN_REVWAVE_RC_DONE tag=${TAG}" >> "$LOG"
