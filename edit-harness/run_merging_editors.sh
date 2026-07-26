#!/bin/bash
# run_merging_editors.sh — editor-general RG federation driver (ROME / MEMIT / AlphaEdit).
# Template = run_merging_width.sh (verbatim skeleton: preflight, GPU-idle gate util<25&&mem<1500
# x3, CPU self-test smoke gate, budget, DRYRUN, pid-by-file / kill -0 only). Tests whether the
# two-regime federation law is EDITOR-GENERAL by running experiments/merging_editors.py --rg for
# a chosen EDITOR ∈ {rome,memit,alpha}. See docs/plans/PREREG-FED-EDITORS-2026-07-16.md.
#
# BUILD-ONLY as authored 2026-07-16: authored under a no-GPU-runs, no-network mandate. Verified
# CPU-side only (bash -n, merging_editors.py --selftest, DRYRUN=1) and NOT launched by the author.
#
# NAMESPACING (never collides with run_merging_width.sh / run_merging_rg.sh, or another
# MODEL_TAG×EDITOR run of THIS driver):
#   - driver-side pid/log/markers suffixed by ${MODEL_TAG}_${EDITOR}: engine/run_merging_editors_<tag>_<editor>.*
#   - the RG results table is ALWAYS written with an explicit --table_out
#     (results/merging_editors/RG_editors_table_<tag>_<editor>_L<layer>.json) — never the module default.
#   - the RG bundle DIRECTORY is derived by merging_editors.py from basename(MODEL_DIR)+editor+layer
#     (results/merging_editors/<basename>_<editor>_L<layer>_RG/) — unique per model×editor×layer.
#
# ROME/MEMIT/AlphaEdit value-opt stays fp32 (editors' own rule; merging_editors.py's
# _load_edit_model always loads dtype=torch.float32).
set -u
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$H" || exit 1
PY=${PY:-/home/zeyufu/miniconda3/envs/dl/bin/python3}

# ---------------------------------------------------------------- required env
MODEL_DIR=${MODEL_DIR:-}
MODEL_TAG=${MODEL_TAG:-}
EDITOR=${EDITOR:-}
if [ -z "$MODEL_DIR" ] || [ -z "$MODEL_TAG" ] || [ -z "$EDITOR" ]; then
  echo "usage: MODEL_DIR=<path> MODEL_TAG=<tag> EDITOR=<rome|memit|alpha> [LAYER=auto75] [DATASET=cf|zsre] [RG_SEEDS=0,1,2] [RG_GROUP_SIZES=2,3,5,10,20] [MEMIT_COV=identity|generic] [KEEP_RATIO=0.99] [BUDGET_MIN=120] [EST_MIN=40] [DRYRUN=1] $0" >&2
  exit 1
fi
case "$EDITOR" in
  rome|memit|alpha) ;;
  *) echo "ABORT: EDITOR must be one of rome|memit|alpha (got '$EDITOR')" >&2; exit 1;;
esac

LAYER=${LAYER:-auto75}
DATASET=${DATASET:-cf}
RG_SEEDS=${RG_SEEDS:-0,1,2}
RG_GROUP_SIZES=${RG_GROUP_SIZES:-2,3,5,10,20}
MEMIT_COV=${MEMIT_COV:-identity}
KEEP_RATIO=${KEEP_RATIO:-0.99}
N_EDITS=${N_EDITS:-200}
N_HOLDOUT=${N_HOLDOUT:-50}
BUDGET_MIN=${BUDGET_MIN:-120}
EST_MIN=${EST_MIN:-40}
DRYRUN=${DRYRUN:-0}

case "$DATASET" in
  cf)   DATA_FILE="data/counterfact.json";;
  zsre) DATA_FILE="data/zsre_eval.json";;
  *) echo "ABORT: DATASET must be cf|zsre (got '$DATASET')" >&2; exit 1;;
esac

TAG="${MODEL_TAG}_${EDITOR}_${DATASET}"
LOG="engine/run_merging_editors_${TAG}.log"
mkdir -p engine results/merging_editors results/merging_editors/selftest
echo $$ > "engine/run_merging_editors_${TAG}.pid"
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "======== RUN_MERGING_EDITORS START tag=${TAG} model=${MODEL_DIR} editor=${EDITOR} dataset=${DATASET} pid=$$ budget=${BUDGET_MIN}m seeds=${RG_SEEDS} g=${RG_GROUP_SIZES} ========"

# ---------------------------------------------------------------- Phase 0a: CPU pre-flight
pf_fail=0
pf(){ if eval "$2"; then log "preflight OK: $1"; else log "PREFLIGHT-FAIL: $1"; pf_fail=1; fi; }
pf "python env (torch+numpy)" "$PY -c 'import torch, numpy' 2>/dev/null"
pf "merging_editors.py present" "[ -f experiments/merging_editors.py ]"
pf "merging_editors has --rg flag" "grep -q -- '\"--rg\"' experiments/merging_editors.py"
pf "merging_m0.py present (imported)" "[ -f experiments/merging_m0.py ]"
pf "rome_native editor present" "[ -f editors/rome_native.py ]"
pf "memit editor present" "[ -f editors/memit.py ]"
pf "alphaedit editor present" "[ -f editors/alphaedit.py ]"
pf "metrics.py present" "[ -f metrics.py ]"
pf "arch_compat present" "[ -f editors/arch_compat.py ]"
pf "dataset file (${DATA_FILE})" "[ -f '$DATA_FILE' ]"
pf "MODEL_DIR exists (${MODEL_DIR})" "[ -d '$MODEL_DIR' ]"
pf "MODEL_DIR has config.json" "[ -f '$MODEL_DIR/config.json' ]"
pf "disk >=10GB free" "[ \$(df --output=avail -BG . | tail -1 | tr -dc 0-9) -ge 10 ]"
if [ "$pf_fail" -ne 0 ]; then log "ABORT: preflight failed"; exit 3; fi

# ---------------------------------------------------------------- Phase 0a.2: LAYER RULE (75% relative depth)
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

MODEL_BASENAME="$(basename "${MODEL_DIR%/}")"
# R-D (2026-07-16): the bundle dir is keyed on MODEL_BASENAME/EDITOR/DATASET/LAYER only, so a
# --editor memit run at a non-identity MEMIT_COV must get its own directory or it collides with
# (silently clobbers, or silently no-ops past via this script's own refuse-check below) an
# existing identity-cov bundle at the same cell — mirrors experiments/merging_editors.py's
# _cov_variant_suffix exactly (identity => "", so every pre-existing EDITOR=rome/alpha and
# MEMIT_COV=identity invocation resolves to the SAME path as before this change).
COV_SUFFIX=""
if [ "$EDITOR" = "memit" ] && [ "$MEMIT_COV" != "identity" ]; then
  COV_SUFFIX="_${MEMIT_COV}"
fi
RG_DIR="results/merging_editors/${MODEL_BASENAME}${COV_SUFFIX}_${EDITOR}_${DATASET}_L${LAYER}_RG"
TABLE="results/merging_editors/RG_editors_table_${TAG}${COV_SUFFIX}_L${LAYER}.json"
MEAS="${RG_DIR}/rg_measurements.npz"

# ---------------------------------------------------------------- Phase 0a.3: refuse-guard (no clobber)
# The module itself refuses to overwrite an existing bundle (--refuse_clobber default on); mirror
# that at the driver level so an operator typo (wrong MODEL_DIR/EDITOR/LAYER re-entering a landed
# bundle) is refused loudly rather than silently re-run.
if [ "$DRYRUN" -ne 1 ] && [ -d "$RG_DIR" ] && [ -f "$RG_DIR/rg_meta.json" ] && [ -f "$TABLE" ]; then
  # allow re-run only if the existing table fails validation (treated as incomplete)
  if $PY - "$TABLE" "$MEAS" <<'EOF' 2>/dev/null | grep -q VALIDATE-OK
import json, sys, numpy as np
try:
    d = json.load(open(sys.argv[1]))
    a = np.load(sys.argv[2])
except Exception:
    print("VALIDATE-FAIL"); sys.exit(0)
if d.get("verdict", {}).get("overall") in ("PASS","KILL","MIXED","INCONCLUSIVE") and d.get("per_g_summary"):
    print("VALIDATE-OK")
else:
    print("VALIDATE-FAIL")
EOF
  then
    log "REFUSE: ${RG_DIR} already holds a VALID RG bundle+table for tag=${TAG} — nothing to do (delete it to force a re-run)"
    echo "REFUSE: valid bundle already exists at ${RG_DIR} (${TABLE})" >&2
    exit 0
  fi
fi

# ---------------------------------------------------------------- Phase 0b: CPU self-test smoke gate
SELFTEST_OK="engine/merging_editors_selftest_${TAG}.ok"
if [ "$DRYRUN" -ne 1 ]; then
  rm -f "$SELFTEST_OK"
  SELFTEST_LOG="engine/merging_editors_selftest_${TAG}.log"
  log "SMOKE merging_editors --selftest (CPU, ~5s) -> ${SELFTEST_LOG}"
  if $PY experiments/merging_editors.py --selftest > "$SELFTEST_LOG" 2>&1; then
    if grep -q "ALL CHECKS PASSED" "$SELFTEST_LOG"; then
      : > "$SELFTEST_OK"
      log "SMOKE OK: self-test passed (cross-term + additivity + ROME-equivalence + RG pass/kill)"
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
    # pin the idle gate to THIS job's card: nvidia-smi ignores CUDA_VISIBLE_DEVICES
    # (memory: use -i), and `head -1` silently read GPU 0 regardless of pinning —
    # on a multi-GPU box the gate then waits on (or passes because of) the WRONG card.
    GPU_ID=${CUDA_VISIBLE_DEVICES%%,*}; GPU_ID=${GPU_ID:-0}
    line=$(nvidia-smi -i "$GPU_ID" --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
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

# validate the RG outputs: the RG table (pass rule + verdict) + the bundle npz measurements
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
print(f"VALIDATE-OK editor={d.get('editor')} verdict={v.get('overall')} qualifying_g={v.get('qualifying_group_sizes')}")
EOF
}

# ---------------------------------------------------------------- RG (GPU): the single science row for this tag
# --no_refuse_clobber: the DRIVER owns the valid-bundle refuse (Phase 0a.3 preflight + skip-if-valid
# below); the module's own refuse would otherwise wedge a half-written bundle after a timeout/OOM
# (MAJOR-1). The driver only re-invokes when the table is missing/invalid, so overwriting is intended.
CMD="$ENVP $PY experiments/merging_editors.py --rg --no_refuse_clobber --editor ${EDITOR} --dataset ${DATASET} --model ${MODEL_DIR} --data ${DATA_FILE} --n_edits ${N_EDITS} --n_holdout ${N_HOLDOUT} --layer ${LAYER} --steps 20 --lr 0.1 --device cuda --rg_seeds ${RG_SEEDS} --rg_group_sizes ${RG_GROUP_SIZES} --keep_ratio ${KEEP_RATIO} --memit_cov ${MEMIT_COV} --out_dir results/merging_editors --table_out ${TABLE}"

SMOKECMD="$ENVP $PY experiments/merging_editors.py --smoke --editor ${EDITOR} --dataset ${DATASET} --model ${MODEL_DIR} --data ${DATA_FILE} --n_holdout ${N_HOLDOUT} --layer ${LAYER} --steps 20 --lr 0.1 --device cuda --keep_ratio ${KEEP_RATIO} --memit_cov ${MEMIT_COV} --smoke_n 2"

if [ "$DRYRUN" -eq 1 ]; then
  echo "DRYRUN tag=${TAG} model=${MODEL_DIR} editor=${EDITOR} layer=${LAYER} dataset=${DATASET} est=${EST_MIN}m -> ${TABLE}"
  echo "DRYRUN rg_dir (npz bundle, module-derived): ${RG_DIR}"
  echo "DRYRUN smoke cmd: ${SMOKECMD}"
  echo "DRYRUN cmd: ${CMD}"
  log "DRYRUN tag=${TAG} editor=${EDITOR} layer=${LAYER} est=${EST_MIN}m cmd: ${CMD}"
else
  # ------------------------------------------------------------ real-model ΔW-fidelity gate (MINOR-1)
  # first-run GPU closure: the re-derived ΔW must equal the REAL editor's installed ΔW (Frobenius
  # rel-err < 1e-4/layer) before any RG cell is trusted. Cached by a per-tag .ok marker so a
  # resumed driver does not re-pay it. Failure aborts the cell (a fidelity break invalidates the
  # stored factors the whole analysis rests on).
  FID_OK="engine/merging_editors_fidelity_${TAG}.ok"
  if [ ! -f "$FID_OK" ]; then
    log "SMOKE ΔW-fidelity gate (real model, ~1-2 GPU-min) -> engine/merging_editors_${TAG}_smoke.log"
    if timeout --signal=TERM --kill-after=60 1200s bash -c "$SMOKECMD" > "engine/merging_editors_${TAG}_smoke.log" 2>&1 \
       && grep -q "ΔW-FIDELITY PASS" "engine/merging_editors_${TAG}_smoke.log"; then
      : > "$FID_OK"
      log "SMOKE OK: ΔW-fidelity PASS ($(grep 'ΔW-FIDELITY PASS' engine/merging_editors_${TAG}_smoke.log | tail -1))"
    else
      log "ABORT: ΔW-fidelity gate FAILED (see engine/merging_editors_${TAG}_smoke.log)"
      echo "ABORT: ΔW-fidelity gate failed for tag=${TAG}" >&2
      exit 4
    fi
  else
    log "skip ΔW-fidelity gate (marker ${FID_OK} present)"
  fi

  now=$(elapsed_min)
  if [ $(( now + EST_MIN + 2 )) -gt "$BUDGET_MIN" ]; then
    log "BUDGET-SKIP rg tag=${TAG} (elapsed ${now}m + est ${EST_MIN}m > ${BUDGET_MIN}m)"
  elif [ -f "$TABLE" ] && [ -f "$MEAS" ] && validate_rg "$TABLE" "$MEAS" | grep -q VALIDATE-OK; then
    log "skip rg tag=${TAG} (exists, validated)"
  else
    cap=$(( EST_MIN * 60 * 3 + 1200 ))
    log "RUN rg tag=${TAG} (est ${EST_MIN}m, cap ${cap}s, elapsed ${now}m) -> engine/merging_editors_${TAG}_run.log"
    t=$(date +%s)
    timeout --signal=TERM --kill-after=60 "${cap}s" bash -c "$CMD" >> "engine/merging_editors_${TAG}_run.log" 2>&1
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
      log "FAIL rg tag=${TAG} (rc ${rc}, ${dt}s)"
    fi
  fi
fi

# ---------------------------------------------------------------- Post-run report
if [ "$DRYRUN" -ne 1 ] && [ -f "$TABLE" ]; then
  log "---------------- POST-RUN (CPU) tag=${TAG} ----------------"
  $PY - "$TABLE" >> "$LOG" 2>&1 <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
v = d["verdict"]
print(f"[rg post] editor={d.get('editor')} VERDICT overall={v['overall']} "
      f"qualifying_g={v['qualifying_group_sizes']} testable_cells={v['n_testable_cells']}")
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
  echo "RUN_MERGING_EDITORS REPORT tag=${TAG} model=${MODEL_DIR} editor=${EDITOR} layer=${LAYER} $(date '+%F %T')  elapsed $(elapsed_min)m/${BUDGET_MIN}m"
  grep -E 'RUN |done |FAIL |SKIP|BUDGET-SKIP|REFUSE|ABORT|SMOKE|LAYER|VERDICT|rg post|gpu poll' "$LOG" | tail -60
} > "engine/run_merging_editors_${TAG}_report.txt"
log "======== RUN_MERGING_EDITORS COMPLETE tag=${TAG} ========"
echo "RUN_MERGING_EDITORS_DONE tag=${TAG}" >> "$LOG"
