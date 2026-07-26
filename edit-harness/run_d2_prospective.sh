#!/bin/bash
# run_d2_prospective.sh — D2 prospective group-formation arm, LOCAL 5090, ¥0 (2026-07-26).
# Template = run_merging_rg.sh (preflight / CPU smoke gate / GPU idle gate / budget-aware
# run-with-timeout / validate / post-run report), PROSPADM-namespaced: its own log, pidfile
# and markers, never reusing merging_rg/revins/u6 names.
#
# SCIENCE: docs/plans/PREREG-D2-PROSPECTIVE-2026-07-26.md (binding) and
# docs/plans/REMOTE-DESIGN-D2-PROSPECTIVE-ARM-2026-07-26.md (rationale). One cell:
# Llama-3.2-1B L12 ROME CounterFact, N=100 pool, budget 25%, g=5, seeds 0/1/2, three
# admission policies (geometry / magnitude / random x3 draws). Estimated 20-35 min/seed,
# ~1.5-2 GPU-h total, retention-dominated (~5,000 retention forwards/seed). The workspace's
# driver estimates have run 5-60x padded before; re-derive true pace after seed 0.
#
# FAILS CLOSED ON RATIFICATION. This driver passes --prereg but CANNOT ratify: the prereg
# must contain a line reading exactly "STATUS: RATIFIED", written by the USER. Phase 0a
# checks for that line and aborts with rc 5 and an explicit message if it is absent, so an
# unratified launch stops here rather than reaching the GPU (the module's own guard is the
# second line of defence, not the first). Do NOT add the line on the user's behalf and do
# NOT patch around either guard.
#
# --ns_reference solo is passed EXPLICITLY per the prereg's decision point D1: neighborhood
# damage measured with edit `a` installed alone, i.e. true federation-added damage. 'base'
# would include each solo edit's own collateral and answers a weaker question.
#
# PROCESS CONTROL: pidfile engine/run_d2_prospective.pid. Stop by PID:
#   kill "$(cat engine/run_d2_prospective.pid)"
# NEVER pgrep/pkill -f a pattern — this script's own command line contains the names any
# such pattern would match (self-match deadlock/self-kill; standing workspace rule).
#
# LID OPEN for the whole run (nvidia_uvm wedge under load).
#
# Env knobs: BUDGET_MIN (default 240), DRYRUN=1 (plan only, results/ byte-untouched),
# SEEDS (default 0,1,2), EST (per-seed minute estimate for budget arithmetic, default 35).
set -u
H=/home/zeyufu/Desktop/idea-feasibility-analysis/edit-harness
cd "$H" || exit 1
PY=/home/zeyufu/miniconda3/envs/dl/bin/python3
PREREG=/home/zeyufu/Desktop/idea-feasibility-analysis/docs/plans/PREREG-D2-PROSPECTIVE-2026-07-26.md
LOG=engine/run_d2_prospective.log
BUDGET_MIN=${BUDGET_MIN:-240}
DRYRUN=${DRYRUN:-0}
SEEDS=${SEEDS:-0,1,2}
EST=${EST:-35}
OUT_DIR=results/prospective_admission
TABLE="${OUT_DIR}/prospective_admission_table.json"
mkdir -p engine "$OUT_DIR"
echo $$ > engine/run_d2_prospective.pid
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "================ RUN_D2_PROSPECTIVE START (pid $$, budget ${BUDGET_MIN}m, seeds ${SEEDS}) ================"

# ---------------------------------------------------------------- Phase 0a: ratification gate (HARD)
# Checked FIRST, before any other work: an unratified prereg must not consume preflight,
# smoke or GPU-gate time. Matches the module's guard exactly (a line equal to the literal
# string, not a substring match, so "STATUS: RATIFIED (pending)" does not pass).
if [ ! -f "$PREREG" ]; then
  log "ABORT: prereg not found at ${PREREG}"
  echo "ABORT: prereg not found at ${PREREG}" >&2
  exit 5
fi
if ! grep -qx 'STATUS: RATIFIED' "$PREREG"; then
  log "ABORT: prereg UNRATIFIED — ${PREREG} has no line reading exactly 'STATUS: RATIFIED'"
  {
    echo "ABORT: the prereg is not ratified."
    echo "  ${PREREG}"
    echo "contains no line reading exactly:  STATUS: RATIFIED"
    echo
    echo "The USER writes that line after reading the prereg and accepting its three resolved"
    echo "decision points. No driver or agent may write it. Nothing has been run; the GPU was"
    echo "not touched."
  } >&2
  exit 5
fi
log "ratification OK: ${PREREG} is marked STATUS: RATIFIED"

# ---------------------------------------------------------------- Phase 0b: CPU pre-flight (HARD)
pf_fail=0
pf(){ if eval "$2"; then log "preflight OK: $1"; else log "PREFLIGHT-FAIL: $1"; pf_fail=1; fi; }
pf "prospective_admission.py present" "[ -f experiments/prospective_admission.py ]"
pf "merging_m0.py present"            "[ -f experiments/merging_m0.py ]"
pf "egl_metrics.py present"           "[ -f experiments/egl_metrics.py ]"
pf "--ns_reference flag supported"    "$PY experiments/prospective_admission.py --help 2>&1 | grep -q -- --ns_reference"
pf "--prereg flag supported"          "$PY experiments/prospective_admission.py --help 2>&1 | grep -q -- --prereg"
pf "counterfact.json"                 "[ -f data/counterfact.json ]"
pf "model Llama-3.2-1B"               "[ -d data/models/Llama-3.2-1B ]"
pf "disk >=5GB free"                  "[ \$(df --output=avail -BG /home | tail -1 | tr -dc 0-9) -ge 5 ]"
if [ "$pf_fail" -ne 0 ]; then log "ABORT: preflight failed"; exit 3; fi

# ---------------------------------------------------------------- Phase 0c: CPU self-test smoke gate
# Skipped on DRYRUN so a plan-only invocation leaves results/ byte-untouched (run_merging_rg.sh
# precedent). The GPU row is skipped on DRYRUN anyway, so the .ok marker is not needed there.
if [ "$DRYRUN" -ne 1 ]; then
  rm -f engine/d2_prospective_selftest.ok
  log "SMOKE prospective_admission --selftest (CPU, ~2s) -> engine/d2_prospective_selftest.log"
  if $PY experiments/prospective_admission.py --selftest > engine/d2_prospective_selftest.log 2>&1; then
    if grep -q "ALL CHECKS PASSED" engine/d2_prospective_selftest.log; then
      : > engine/d2_prospective_selftest.ok
      log "SMOKE OK: pool screening vs brute-force, admission/partition, aggregation, ns_reference dispatch"
    else
      log "ABORT: self-test ran but did not report ALL CHECKS PASSED"; exit 4
    fi
  else
    log "ABORT: self-test failed (see engine/d2_prospective_selftest.log)"; exit 4
  fi
fi

# ---------------------------------------------------------------- Phase 0d: GPU idle gate
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
# Budget clock starts AFTER the idle-gate wait, so gate time is not charged to the science.
T0=$(date +%s)
elapsed_min(){ echo $(( ( $(date +%s) - T0 ) / 60 )); }

ENVP="env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1"
n_done=0; n_fail=0; n_skip=0

# Validate the output table: schema, every requested seed present, the three policies each
# carrying a mean-over-draws block, a compute-time runner_stamp, and non-degenerate results
# (cross-seed variance non-zero on the primary outcome — the Frame-A relabel signature is
# identical numbers across seeds, so a zero-variance table is quarantined, not trusted).
validate_prospadm(){
  $PY - "$1" "$2" <<'EOF'
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
stamp = d.get("runner_stamp") or {}
need = {"code_sha256","pid","hostname","wall_start","wall_end","elapsed_s","nvidia_smi_sample",
        "stamp_version"}
missing = need - set(stamp)
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
            print(f"VALIDATE-FAIL seed {r.get('seed')} policy {p}: no target_logit_drop_mean")
            sys.exit(1)
        drops.setdefault(p, []).append(float(m))
# distinct per-seed wall-clock (a same-second batch write is the relabel signature)
stamps = [r.get("seed_wall_end") for r in reports]
if len(reports) > 1 and len(set(stamps)) < len(stamps):
    print(f"VALIDATE-FAIL seed_wall_end not distinct across seeds: {stamps}"); sys.exit(1)
if len(reports) > 1:
    for p, vals in drops.items():
        if len(set(vals)) == 1:
            print(f"VALIDATE-FAIL policy {p}: identical drop across all seeds {vals} "
                  f"(zero cross-seed variance)"); sys.exit(1)
print("VALIDATE-OK seeds=" + ",".join(str(s) for s in sorted(got)) +
      "  drop_mean geo/mag/rnd=" +
      "/".join(f"{sum(drops[p])/len(drops[p]):.4f}" for p in ("geometry","magnitude","random")))
EOF
}

# ---------------------------------------------------------------- The science row (GPU)
# One invocation covers all seeds (the model is loaded once inside run_admission), so this is
# a single budget-checked row rather than a per-seed loop. n_seeds * EST is the estimate.
n_seeds=$(echo "$SEEDS" | tr ',' '\n' | grep -c .)
TOTAL_EST=$(( n_seeds * EST ))
CMD="$ENVP $PY experiments/prospective_admission.py \
--model data/models/Llama-3.2-1B --data data/counterfact.json --layer 12 \
--n_pool 100 --budget 0.25 --group_size 5 --n_random_draws 3 --n_retention 200 \
--ns_reference solo --prereg ${PREREG} --seeds ${SEEDS} --steps 20 --lr 0.1 \
--device cuda --out_dir ${OUT_DIR} --table_out ${TABLE}"

if [ "$DRYRUN" -eq 1 ]; then
  echo "DRYRUN prospadm est=${TOTAL_EST}m (${n_seeds} seeds x ${EST}m) -> ${TABLE}"
  echo "DRYRUN cmd: ${CMD}"
  log "DRYRUN prospadm est=${TOTAL_EST}m cmd: ${CMD}"
else
  now=$(elapsed_min)
  if [ $(( now + TOTAL_EST + 2 )) -gt "$BUDGET_MIN" ]; then
    log "BUDGET-SKIP prospadm (elapsed ${now}m + est ${TOTAL_EST}m > ${BUDGET_MIN}m)"; n_skip=$((n_skip+1))
  elif [ -f "$TABLE" ] && validate_prospadm "$TABLE" "$SEEDS" | grep -q VALIDATE-OK; then
    log "skip prospadm (exists, validated) — idempotent re-run"; n_done=$((n_done+1))
  else
    cap=$(( TOTAL_EST * 60 * 3 + 1200 ))
    log "RUN prospadm (est ${TOTAL_EST}m, cap ${cap}s, elapsed ${now}m) -> engine/run_d2_prospective_run.log"
    t=$(date +%s)
    timeout --signal=TERM --kill-after=60 "${cap}s" bash -c "$CMD" >> engine/run_d2_prospective_run.log 2>&1
    rc=$?; dt=$(( $(date +%s) - t ))
    if [ "$rc" -eq 0 ] && [ -f "$TABLE" ]; then
      vres=$(validate_prospadm "$TABLE" "$SEEDS")
      if echo "$vres" | grep -q VALIDATE-FAIL; then
        # Quarantine, never repair (post-Frame-A rule).
        mv "$TABLE" "${TABLE}.INVALID-$(date +%Y%m%d%H%M%S)" 2>/dev/null
        log "FAIL prospadm (${dt}s) OUTPUT-INVALID (quarantined): ${vres}"; n_fail=$((n_fail+1))
      else
        log "done prospadm (${dt}s) ${vres}"; n_done=$((n_done+1))
      fi
    else
      log "FAIL prospadm (rc ${rc}, ${dt}s) — see engine/run_d2_prospective_run.log"; n_fail=$((n_fail+1))
    fi
  fi
fi

# ---------------------------------------------------------------- Post-run report (CPU)
# Prints the per-seed policy comparison and evaluates P1/P2 mechanically. This is a READOUT of
# the preregistered rule, not an adjudication: the kill-gates are applied by a human against
# docs/plans/PREREG-D2-PROSPECTIVE-2026-07-26.md before any prose is written.
if [ "$DRYRUN" -ne 1 ] && [ -f "$TABLE" ]; then
  log "---------------- POST-RUN (CPU) ----------------"
  $PY - "$TABLE" >> "$LOG" 2>&1 <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"[prospadm post] cell={d.get('model_tag')} L{d.get('layer')} ns_reference={d.get('ns_reference')} "
      f"budget={d.get('budget')} g={d.get('group_size')}")
p1 = p2 = 0; n = 0
for r in d.get("seed_reports", []):
    pol = r["policies"]; n += 1
    def m(p, k): return (pol[p]["mean_over_draws"] or {}).get(k)
    g, mg, rd = m("geometry","target_logit_drop_mean"), m("magnitude","target_logit_drop_mean"), \
                m("random","target_logit_drop_mean")
    lt_r = g is not None and rd is not None and g < rd
    lt_m = g is not None and mg is not None and g < mg
    p1 += bool(lt_r); p2 += bool(lt_m)
    print(f"[prospadm post] seed {r['seed']}: drop geo={g} mag={mg} rnd={rd}  "
          f"geo<rnd={lt_r} geo<mag={lt_m}")
    print(f"[prospadm post]   esr geo={m('geometry','edit_success_rate')} "
          f"rnd={m('random','edit_success_rate')} | "
          f"ns_induced geo={m('geometry','neighborhood_damage_rate_induced')} "
          f"rnd={m('random','neighborhood_damage_rate_induced')} | "
          f"retention geo={m('geometry','retention_shift_mean_logprob')} "
          f"rnd={m('random','retention_shift_mean_logprob')}")
print(f"[prospadm post] READOUT (not a verdict): P1 geo<rnd in {p1}/{n} seeds; "
      f"P2 geo<mag in {p2}/{n} seeds. Prereg needs >=2/3 for each. P2 is the discriminating "
      f"prediction; ties (difference below the random policy's own draw spread) count AGAINST "
      f"P2. Apply G-P1..G-P4 by hand against the prereg before writing prose.")
st = d.get("runner_stamp", {})
print(f"[prospadm post] stamp code_sha256={str(st.get('code_sha256'))[:16]} host={st.get('hostname')} "
      f"pid={st.get('pid')} wall={st.get('wall_start')}..{st.get('wall_end')} "
      f"elapsed_s={st.get('elapsed_s')} gpu={st.get('nvidia_smi_sample')}")
EOF
  log "post: parsed ${TABLE}"
fi

{
  echo "RUN_D2_PROSPECTIVE REPORT $(date '+%F %T')  jobs: ${n_done} done / ${n_fail} fail / ${n_skip} skipped  elapsed $(elapsed_min)m/${BUDGET_MIN}m"
  grep -E 'RUN |done |FAIL |SKIP|ABORT|SMOKE|ratification|prospadm post|gpu poll' "$LOG" | tail -60
} > engine/run_d2_prospective_report.txt
log "================ RUN_D2_PROSPECTIVE COMPLETE (${n_done} done / ${n_fail} fail / ${n_skip} skip) ================"
echo "RUN_D2_PROSPECTIVE_DONE" >> "$LOG"
