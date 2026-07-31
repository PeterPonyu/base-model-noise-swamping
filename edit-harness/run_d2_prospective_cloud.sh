#!/bin/bash
# run_d2_prospective_cloud.sh — OPTIONAL >=7B confirmation arm for the D2 prospective
# group-formation experiment. Mistral-7B-v0.3 L24, bf16, ~2-3 h on a single 4090D 24GB,
# ¥15-25 including the ~15GB model download. Built 2026-07-26, NOT LAUNCHED, NOT COSTED.
#
# ############################################################################
# # DO NOT BUY THIS SPECULATIVELY. The recommendation is NOT to run this arm. #
# ############################################################################
#
# The LOCAL arm (run_d2_prospective.sh, Llama-3.2-1B L12, ¥0) IS the contribution. This
# script exists so that if — and only if — a referee explicitly asks whether the prospective
# result survives at scale, the answer is a priced, ready, one-command response instead of a
# cold design problem. The prereg (docs/plans/PREREG-D2-PROSPECTIVE-2026-07-26.md, gate G-P4)
# preregisters the local result as a SINGLE-CELL result; running this arm is what would
# license a broader claim, and nothing else does.
#
# KNOWN CAVEAT TO STATE IN THE PAPER, NOT ELIDE: the existing Mistral-7B RG cell in the
# merging tree is fp32; this arm is bf16 on a 24GB card, so it is NOT strictly
# dtype-comparable to that cell. If the referee's question is specifically about numerical
# precision rather than scale, the honest buy is the fp32 Pro-6000 96GB variant instead
# (~¥25-33): set CLOUD_DTYPE=fp32 and MODEL_DIR accordingly on a box with the VRAM for it.
#
# BOX SAFETY (this workspace's own campaign lessons, all of which cost real money once):
#   - WAVE_BOX must be set explicitly and match the box; a default silently runs the wave on
#     the wrong machine.
#   - rsync the harness code to the box FIRST — no box has prospective_admission.py.
#   - Smoke the model before the real cell. A model with no smoke json CONFIG-skips silently
#     and the wave still marks DONE.
#   - $BASHPID, not $$, inside any backgrounded worker.
#   - Budget clock starts AFTER the idle gate.
#   - Teardown enumerates nvidia-smi compute PIDs and identity-checks /proc/<pid>/cmdline
#     before killing: `timeout` children sit in their own process group, survive a group
#     kill, and keep holding VRAM.
#   - Stop by PID from the pidfile. NEVER pgrep/pkill -f a pattern (self-match).
#   - Shut the box down MANUALLY and record actual ¥ against the estimate.
#
# Env: WAVE_BOX (required), MODEL_DIR, PREREG, BUDGET_MIN (default 240), DRYRUN=1, SEEDS,
#      CLOUD_DTYPE (default bf16), CLOUD_PY.
set -u
H=${H:-/root/edit-harness}
cd "$H" || { echo "ABORT: harness not found at $H (rsync the code to the box first)" >&2; exit 1; }
CLOUD_PY=${CLOUD_PY:-python3}
PREREG=${PREREG:-$H/docs/plans/PREREG-D2-PROSPECTIVE-2026-07-26.md}
MODEL_DIR=${MODEL_DIR:-data/models/Mistral-7B-v0.3}
CLOUD_DTYPE=${CLOUD_DTYPE:-bf16}
LAYER=${LAYER:-24}
SEEDS=${SEEDS:-0,1,2}
BUDGET_MIN=${BUDGET_MIN:-240}
DRYRUN=${DRYRUN:-0}
EST=${EST:-60}
LOG=engine/run_d2_prospective_cloud.log
OUT_DIR=results/prospective_admission_cloud
TABLE="${OUT_DIR}/prospective_admission_mistral7b_L${LAYER}_table.json"
SMOKE_DIR=results/smoke_prospadm_cloud
SMOKE_TABLE="${SMOKE_DIR}/prospadm_smoke_table.json"
mkdir -p engine "$OUT_DIR" "$SMOKE_DIR"
echo "$BASHPID" > engine/run_d2_prospective_cloud.pid
trap 'rm -f engine/run_d2_prospective_cloud.pid' EXIT
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "============ RUN_D2_PROSPECTIVE_CLOUD START (pid $$, budget ${BUDGET_MIN}m) ============"

# ---------------------------------------------------------------- Phase 0a: WAVE_BOX identity (HARD)
if [ -z "${WAVE_BOX:-}" ]; then
  echo "ABORT: WAVE_BOX is unset. Set it to this box's id explicitly — a default here is how" >&2
  echo "       a wave ends up running on the wrong machine." >&2
  log "ABORT: WAVE_BOX unset"; exit 6
fi
THIS_BOX=$(hostname)
log "WAVE_BOX=${WAVE_BOX} hostname=${THIS_BOX}"
if [ "${WAVE_BOX_STRICT:-1}" = "1" ] && [ "$WAVE_BOX" != "$THIS_BOX" ]; then
  echo "ABORT: WAVE_BOX=${WAVE_BOX} != hostname=${THIS_BOX}. Refusing to run on a box the" >&2
  echo "       caller did not name. Set WAVE_BOX_STRICT=0 only if the mismatch is understood." >&2
  log "ABORT: WAVE_BOX mismatch"; exit 6
fi

# ---------------------------------------------------------------- Phase 0b: ratification gate (HARD)
# Identical to the local driver's: the cloud arm is the SAME preregistered experiment at a
# different scale, so it is bound by the same ratification. Passing --prereg is not enough;
# the file must carry the user's exact line.
if [ ! -f "$PREREG" ]; then
  echo "ABORT: prereg not found at ${PREREG} (did the rsync include docs/plans/?)" >&2
  log "ABORT: prereg missing"; exit 5
fi
if ! grep -qx 'STATUS: RATIFIED' "$PREREG"; then
  echo "ABORT: ${PREREG} has no line reading exactly 'STATUS: RATIFIED'. Nothing was run." >&2
  log "ABORT: prereg unratified"; exit 5
fi
log "ratification OK"
READY="$H/engine/BOX_READY_d2-prospective.ok"
[ -f "$READY" ] || {
  echo "ABORT: run box_prepare_wave.sh d2-prospective check first" >&2
  log "ABORT: missing BOX_READY_d2-prospective.ok"; exit 8
}
expected_sha=$(sha256sum "$0" | cut -d' ' -f1)
prepare_sha=$(sha256sum "$H/engine/box_prepare_wave.sh" | cut -d' ' -f1)
grep -qx "driver_sha256=$expected_sha" "$READY" || {
  echo "ABORT: stale BOX_READY receipt for a different driver hash" >&2
  log "ABORT: stale BOX_READY driver hash"; exit 8
}
grep -qx "prepare_sha256=$prepare_sha" "$READY" || {
  echo "ABORT: stale BOX_READY receipt for a different prepare hash" >&2
  log "ABORT: stale BOX_READY prepare hash"; exit 8
}

# ---------------------------------------------------------------- Phase 0c: pre-flight (HARD)
pf_fail=0
pf(){ if eval "$2"; then log "preflight OK: $1"; else log "PREFLIGHT-FAIL: $1"; pf_fail=1; fi; }
pf "prospective_admission.py present" "[ -f experiments/prospective_admission.py ]"
pf "merging_m0.py present"            "[ -f experiments/merging_m0.py ]"
pf "egl_metrics.py present"           "[ -f experiments/egl_metrics.py ]"
pf "--ns_reference supported"         "$CLOUD_PY experiments/prospective_admission.py --help 2>&1 | grep -q -- --ns_reference"
pf "--prereg supported"               "$CLOUD_PY experiments/prospective_admission.py --help 2>&1 | grep -q -- --prereg"
pf "--model_dtype supported"          "$CLOUD_PY experiments/prospective_admission.py --help 2>&1 | grep -q -- --model_dtype"
pf "counterfact.json"                 "[ -f data/counterfact.json ]"
pf "model dir ${MODEL_DIR}"           "[ -d ${MODEL_DIR} ]"
pf "nvidia-smi available"             "command -v nvidia-smi >/dev/null"
if [ "$pf_fail" -ne 0 ]; then log "ABORT: preflight failed"; exit 3; fi

# ---------------------------------------------------------------- Phase 0d: CPU self-test
if [ "$DRYRUN" -ne 1 ]; then
  rm -f engine/d2_prospective_cloud_selftest.ok
  if $CLOUD_PY experiments/prospective_admission.py --selftest > engine/d2_prospective_cloud_selftest.log 2>&1 \
     && grep -q "ALL CHECKS PASSED" engine/d2_prospective_cloud_selftest.log; then
    : > engine/d2_prospective_cloud_selftest.ok
    log "SMOKE OK: CPU self-test"
  else
    log "ABORT: CPU self-test failed"; exit 4
  fi
fi

# ---------------------------------------------------------------- Phase 0e: GPU idle gate
if [ "$DRYRUN" -eq 1 ]; then
  log "DRYRUN=1 -- skipping smoke + idle gate + GPU rows"
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
      if [ $(( $(date +%s) - gate_t0 )) -gt 1800 ]; then log "ABORT: GPU busy >30min"; exit 2; fi
    fi
    log "gpu poll util=${util:-NA} mem=${mem:-NA} consec=${consec}/3"
    [ "$consec" -lt 3 ] && sleep 30
  done
  log "GPU idle -- window opens now"
fi
# Budget clock AFTER the gate.
T0=$(date +%s)
elapsed_min(){ echo $(( ( $(date +%s) - T0 ) / 60 )); }
ENVP="env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1"
n_done=0; n_fail=0; n_skip=0

# dtype is always passed EXPLICITLY. Precision is exactly the ambiguity this arm may be
# bought to remove, so it is never left to a default. Phase 0c already preflighted the flag.
case "$CLOUD_DTYPE" in
  bf16|fp32) DT_FLAG="--model_dtype ${CLOUD_DTYPE}" ;;
  *) log "ABORT: CLOUD_DTYPE=${CLOUD_DTYPE} not in {bf16,fp32}"
     echo "ABORT: CLOUD_DTYPE must be bf16 or fp32, got '${CLOUD_DTYPE}'" >&2; exit 7 ;;
esac
log "dtype: ${DT_FLAG}"

# ---------------------------------------------------------------- Phase 1: MODEL SMOKE (mandatory)
# A model that never smoked is a model that CONFIG-skips silently while the wave reports DONE.
# Tiny config (n_pool 10, 1 seed, 1 random draw, 10 retention prompts) — minutes, not hours.
if [ "$DRYRUN" -eq 1 ]; then
  echo "DRYRUN smoke -> ${SMOKE_TABLE}"
else
  SMOKE_CMD="$ENVP $CLOUD_PY experiments/prospective_admission.py \
--model ${MODEL_DIR} --data data/counterfact.json --layer ${LAYER} ${DT_FLAG} \
--n_pool 10 --budget 0.5 --group_size 5 --n_random_draws 1 --n_retention 10 \
--ns_reference solo --prereg ${PREREG} --seeds 0 --steps 20 --lr 0.1 \
--device cuda --out_dir ${SMOKE_DIR} --table_out ${SMOKE_TABLE}"
  log "SMOKE model ${MODEL_DIR} -> ${SMOKE_TABLE}"
  timeout --signal=TERM --kill-after=60 1800s bash -c "$SMOKE_CMD" >> engine/run_d2_prospective_cloud_smoke.log 2>&1
  src=$?
  if [ "$src" -ne 0 ] || [ ! -f "$SMOKE_TABLE" ]; then
    log "ABORT: model smoke FAILED (rc ${src}) — refusing to spend the real cell's GPU-hours"
    echo "ABORT: model smoke failed; see engine/run_d2_prospective_cloud_smoke.log" >&2
    exit 8
  fi
  log "SMOKE OK: ${MODEL_DIR} loads, edits, and writes a valid table"
fi

# ---------------------------------------------------------------- Phase 2: the real cell
validate_prospadm(){
  $CLOUD_PY - "$1" "$2" <<'EOF'
import json, sys
path, seeds_arg = sys.argv[1], sys.argv[2]
want = [int(x) for x in seeds_arg.split(",") if x != ""]
try:
    d = json.load(open(path))
except Exception as e:
    print(f"VALIDATE-FAIL table unparseable: {e}"); sys.exit(1)
if d.get("schema_version") != "prospadm.v1":
    print(f"VALIDATE-FAIL bad schema_version {d.get('schema_version')!r}"); sys.exit(1)
if d.get("ns_reference") != "solo":
    print(f"VALIDATE-FAIL ns_reference={d.get('ns_reference')!r}, expected 'solo'"); sys.exit(1)
reports = d.get("seed_reports") or []
got = [r.get("seed") for r in reports]
if sorted(got) != sorted(want):
    print(f"VALIDATE-FAIL seeds {got} != requested {want}"); sys.exit(1)
need = {"code_sha256","pid","hostname","wall_start","wall_end","elapsed_s","nvidia_smi_sample",
        "stamp_version"}
missing = need - set(d.get("runner_stamp") or {})
if missing:
    print(f"VALIDATE-FAIL runner_stamp missing {sorted(missing)}"); sys.exit(1)
drops = {}
for r in reports:
    pol = r.get("policies") or {}
    for p in ("geometry","magnitude","random"):
        if p not in pol:
            print(f"VALIDATE-FAIL seed {r.get('seed')} missing policy {p}"); sys.exit(1)
        m = (pol[p].get("mean_over_draws") or {}).get("target_logit_drop_mean")
        if m is None:
            print(f"VALIDATE-FAIL seed {r.get('seed')} policy {p}: no drop"); sys.exit(1)
        drops.setdefault(p, []).append(float(m))
stamps = [r.get("seed_wall_end") for r in reports]
if len(reports) > 1 and len(set(stamps)) < len(stamps):
    print(f"VALIDATE-FAIL seed_wall_end not distinct: {stamps}"); sys.exit(1)
if len(reports) > 1:
    for p, vals in drops.items():
        if len(set(vals)) == 1:
            print(f"VALIDATE-FAIL policy {p}: zero cross-seed variance {vals}"); sys.exit(1)
print("VALIDATE-OK seeds=" + ",".join(str(s) for s in sorted(got)))
EOF
}

n_seeds=$(echo "$SEEDS" | tr ',' '\n' | grep -c .)
TOTAL_EST=$(( n_seeds * EST ))
CMD="$ENVP $CLOUD_PY experiments/prospective_admission.py \
--model ${MODEL_DIR} --data data/counterfact.json --layer ${LAYER} ${DT_FLAG} \
--n_pool 100 --budget 0.25 --group_size 5 --n_random_draws 3 --n_retention 200 \
--ns_reference solo --prereg ${PREREG} --seeds ${SEEDS} --steps 20 --lr 0.1 \
--device cuda --out_dir ${OUT_DIR} --table_out ${TABLE}"

if [ "$DRYRUN" -eq 1 ]; then
  echo "DRYRUN cloud cell est=${TOTAL_EST}m -> ${TABLE}"
  echo "DRYRUN cmd: ${CMD}"
  log "DRYRUN cmd: ${CMD}"
else
  now=$(elapsed_min)
  if [ $(( now + TOTAL_EST + 2 )) -gt "$BUDGET_MIN" ]; then
    log "BUDGET-SKIP cloud cell (elapsed ${now}m + est ${TOTAL_EST}m > ${BUDGET_MIN}m)"; n_skip=$((n_skip+1))
  elif [ -f "$TABLE" ] && validate_prospadm "$TABLE" "$SEEDS" | grep -q VALIDATE-OK; then
    log "skip cloud cell (exists, validated)"; n_done=$((n_done+1))
  else
    cap=$(( TOTAL_EST * 60 * 3 + 1800 ))
    log "RUN cloud cell (est ${TOTAL_EST}m, cap ${cap}s)"
    t=$(date +%s)
    timeout --signal=TERM --kill-after=60 "${cap}s" bash -c "$CMD" >> engine/run_d2_prospective_cloud_run.log 2>&1
    rc=$?; dt=$(( $(date +%s) - t ))
    if [ "$rc" -eq 0 ] && [ -f "$TABLE" ]; then
      vres=$(validate_prospadm "$TABLE" "$SEEDS")
      if echo "$vres" | grep -q VALIDATE-FAIL; then
        mv "$TABLE" "${TABLE}.INVALID-$(date +%Y%m%d%H%M%S)" 2>/dev/null
        log "FAIL cloud cell (${dt}s) OUTPUT-INVALID (quarantined): ${vres}"; n_fail=$((n_fail+1))
      else
        log "done cloud cell (${dt}s) ${vres}"; n_done=$((n_done+1))
      fi
    else
      log "FAIL cloud cell (rc ${rc}, ${dt}s)"; n_fail=$((n_fail+1))
    fi
  fi
fi

# ---------------------------------------------------------------- Phase 3: exact pull manifest
# box_pull_down.sh uses rsync --files-from and therefore accepts exact paths only.
cat > engine/PULL_MANIFEST_d2_prospective_cloud.txt <<MANIFEST
${TABLE}
${SMOKE_TABLE}
engine/run_d2_prospective_cloud.log
engine/run_d2_prospective_cloud_report.txt
MANIFEST
log "exact pull manifest -> engine/PULL_MANIFEST_d2_prospective_cloud.txt"


# ---------------------------------------------------------------- Phase 4: teardown advisory
# Advisory only — this script never kills anything it did not start. `timeout` children sit in
# their own process group, survive a group kill, and keep holding VRAM, so the operator must
# identity-check each PID before acting.
if [ "$DRYRUN" -ne 1 ]; then
  log "---- residual GPU compute PIDs (identity-check /proc/<pid>/cmdline before killing) ----"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | while read -r ln; do
    p=$(echo "$ln" | awk -F, '{gsub(/[^0-9]/,"",$1); print $1}')
    [ -n "$p" ] && log "  pid=${p} mem=$(echo "$ln" | awk -F, '{print $2}') cmdline=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-120)"
  done
fi

{
  echo "RUN_D2_PROSPECTIVE_CLOUD REPORT $(date '+%F %T')  ${n_done} done / ${n_fail} fail / ${n_skip} skip  elapsed $(elapsed_min)m/${BUDGET_MIN}m"
  echo "box=${WAVE_BOX} model=${MODEL_DIR} layer=${LAYER} dtype=${CLOUD_DTYPE}"
  grep -E 'RUN |done |FAIL |SKIP|ABORT|SMOKE|ratification|gpu poll|pid=' "$LOG" | tail -60
} > engine/run_d2_prospective_cloud_report.txt
log "============ RUN_D2_PROSPECTIVE_CLOUD COMPLETE (${n_done}/${n_fail}/${n_skip}) ============"
if [ "$n_done" -eq 1 ] && [ "$n_fail" -eq 0 ] && [ "$n_skip" -eq 0 ]; then
  echo "RUN_D2_PROSPECTIVE_CLOUD_DONE" >> "$LOG"
  exit 0
fi
log "PARTIAL: expected exactly 1 validated cell"
exit 11
