#!/bin/bash
# run_u1.sh — U1 DELETION paper program (2026-07-03). Template = run_r4.sh (verbatim
# preflight/idle-gate/validate/run_row/heartbeat/post-run skeleton), U1-namespaced.
# Context: u1e0_llama1b_delete_refusal_L12_s0 PASSED both informal checks today
#   (C3 within-probe rho=.638; U1-E1 transplant Delta-rho=.61) but the FORMAL prereg
#   scorer (experiments/u1_deletion_gate.py, SxC double-centered + edit-level perm null)
#   has ONLY ever run as a tiny CPU dev-smoke (results/smoke_infra/u1_gate_smoke.json,
#   07-02, on new_rome.npz/u1e0_del_smoke.npz) — never on real data. This program:
#   (A) hardens L12 refusal across seeds, (B) sweeps the layer profile, (C) dissociates
#   the refusal data-layer swap from the eos/suppress objective-layer deletion editor
#   (editors/rome_deletion.py — never exercised at GPU scale, only as tiny CPU smokes
#   on tiny-random-Llama from the same 07-02 session), (D) tests whether AlphaEdit's
#   null-space projection mitigates deletion collateral the way it mitigates rewrite
#   collateral, (E) a STRETCH cross-arch cell (qwen15b) testing whether the deletion-
#   geometry law is Llama-family-specific the way the signed REWRITE law is, and a FILLER
#   QuantEdit-delete arm. See PLAN-U1-PAPER.md.
#
# All rows verified against killgate_keygeom.py's post-parse guards (~:196-238) —
# see the guard-legality table in the authoring report. No row is guarded out.
set -u
H=/home/zeyufu/Desktop/idea-feasibility-analysis/edit-harness
cd "$H" || exit 1
PY=/home/zeyufu/miniconda3/envs/dl/bin/python3
LOG=engine/run_u1.log
BUDGET_MIN=${BUDGET_MIN:-480}
mkdir -p engine results/matrices results/smoke_u1/matrices results/vectors
echo $$ > engine/run_u1.pid
[ -f engine/u1_round_start ] || stat -c %Y engine/run_u1.pid > engine/u1_round_start
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "================ RUN_U1 START (pid $$, budget ${BUDGET_MIN}m) ================"

# ---------------------------------------------------------------- Phase 0a: CPU pre-flight
pf_fail=0
pf(){ if eval "$2"; then log "preflight OK: $1"; else log "PREFLIGHT-FAIL: $1"; pf_fail=1; fi; }
pf "python env" "$PY -c 'import torch, numpy' 2>/dev/null"
pf "counterfact.json" "[ -f data/counterfact.json ]"
pf "edit_mode flag" "grep -q -- '--edit_mode' experiments/killgate_keygeom.py"
pf "delete_variant flag" "grep -q -- '--delete_variant' experiments/killgate_keygeom.py"
pf "save_vectors flag" "grep -q -- '--save_vectors' experiments/killgate_keygeom.py"
pf "rome_deletion editor" "[ -f editors/rome_deletion.py ]"
pf "u1_deletion_gate.py" "[ -f experiments/u1_deletion_gate.py ]"
pf "u1_transplant.py" "[ -f experiments/u1_transplant.py ]"
pf "analyze_matrices.py" "[ -f experiments/analyze_matrices.py ]"
pf "model Llama-3.2-1B" "[ -d data/models/Llama-3.2-1B ]"
pf "model Qwen2.5-1.5B (cross-arch stretch)" "[ -d data/models/Qwen2.5-1.5B ]"
pf "matched insertion L8 s0" "[ -f results/matrices/gate_llama1b_rome_cf_L8_s0.npz ]"
pf "matched insertion L12 s0" "[ -f results/matrices/gate_llama1b_rome_cf_L12_s0.npz ]"
pf "matched insertion L14 s0" "[ -f results/matrices/gate_llama1b_rome_cf_L14_s0.npz ]"
pf "matched alpha-insertion L12 s0" "[ -f results/matrices/g4_llama1b_alpha_cf_L12_s0.npz ]"
pf "matched qwen15b-insertion L14 s0" "[ -f results/matrices/gate_qwen15b_rome_cf_L14_s0.npz ]"
pf "disk >=20GB free" "[ \$(df --output=avail -BG /home | tail -1 | tr -dc 0-9) -ge 20 ]"
# fail-closed marker lifecycle (r3/r4 review HIGH #1) — U1-namespaced
rm -f engine/smoke_u1_*.ok
if [ "$pf_fail" -ne 0 ]; then log "ABORT: preflight failed"; exit 3; fi

# ---------------------------------------------------------------- Phase 0b: GPU idle gate
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
log "GPU idle — window opens now"
T0=$(date +%s)

# ---------------------------------------------------------------- helpers (r4 template, verbatim)
ENVP="env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1"
KG="experiments/killgate_keygeom.py"
CF="--dataset counterfact --data data/counterfact.json"
COMMON="--n_edits 200 --n_probes 500 --steps 20 --save_matrices --matrix_dir results/matrices"
SMK="--n_edits 4 --n_probes 40 --steps 2 --save_matrices --matrix_dir results/smoke_u1/matrices"
elapsed_min(){ echo $(( ( $(date +%s) - T0 ) / 60 )); }

validate(){
  $PY - "$1" "$2" "${3:-full}" <<'EOF'
import json, sys, numpy as np
j, z, mode = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    d = json.load(open(j))
except Exception as e:
    print(f"VALIDATE-FAIL json unparseable: {e}"); sys.exit(1)
try:
    a = np.load(z)
except Exception as e:
    print(f"VALIDATE-FAIL npz unreadable: {e}"); sys.exit(1)
need = {"COS", "damage_logit", "norm_growth", "edit_ok"}
missing = need - set(a.files)
if missing: print(f"VALIDATE-FAIL npz missing {missing}"); sys.exit(1)
for k in need:
    arr = a[k].astype(float)
    if np.isnan(arr).all():
        print(f"VALIDATE-FAIL {k} all-NaN"); sys.exit(1)
if mode == "full":
    esr = d.get("edit_success_rate")
    is_delete = "edit_mode" in a.files and str(a["edit_mode"]) == "delete"
    if is_delete:
        # delete mode: esr = 2x-suppression rate; low/zero is a legitimate NEGATIVE
        # finding, not breakage (r4 review LOW-1) — warn, never fail.
        if esr is not None and esr < 0.9: print(f"VALIDATE-NOTE delete-mode suppression rate={esr}")
    else:
        if esr == 0: print("VALIDATE-FAIL esr==0 at real steps"); sys.exit(1)
        if esr is not None and esr < 0.9: print(f"VALIDATE-WARN esr={esr}<0.9")
print("VALIDATE-OK")
EOF
}

wedge_fail=0; MAXWEDGE=2
n_done=0; n_fail=0; n_skip=0
QUEUE_ABORT=0

run_row(){
  local class="$1" tag="$2" est="$3" needs="$4"; shift 4; local cmd="$*"
  local now; now=$(elapsed_min)
  if [ $(( now + est + 2 )) -gt "$BUDGET_MIN" ]; then
    log "BUDGET-SKIP ${tag} (elapsed ${now}m + est ${est}m > ${BUDGET_MIN}m)"; n_skip=$((n_skip+1)); return; fi
  if [ "$needs" != "-" ]; then
    local mk; for mk in ${needs//,/ }; do
      if [ ! -f "$mk" ]; then log "CONFIG-SKIP ${tag} (missing gate marker ${mk})"; n_skip=$((n_skip+1)); return; fi
    done
  fi
  local outj outn
  outj=$(echo "$cmd" | grep -oE -- '--out [^ ]+' | awk '{print $2}')
  outn="results/matrices/$(basename "${outj%.json}").npz"
  case "$cmd" in *smoke_u1*) outn="results/smoke_u1/matrices/$(basename "${outj%.json}").npz";; esac
  if [ -n "$outj" ] && [ -f "$outj" ] && [ -f "$outn" ]; then
    local pvmode="full"; [ "$class" = "SMOKE" ] && pvmode="smoke"
    if validate "$outj" "$outn" "$pvmode" | grep -q VALIDATE-FAIL; then
      mv "$outj" "$outj.INVALID" 2>/dev/null; mv "$outn" "$outn.INVALID" 2>/dev/null
      log "STALE-INVALID ${tag} — quarantined; re-running"
    else
      log "skip ${tag} (exists, validated)"
      [ "$class" = "SMOKE" ] && : > "engine/smoke_u1_${tag}.ok"
      return
    fi
  fi
  local mdir; mdir=$(echo "$cmd" | grep -oE -- '--model [^ ]+' | awk '{print $2}')
  case "$mdir" in data/*|/*)
    if [ -n "$mdir" ] && [ ! -d "$mdir" ]; then log "CONFIG-SKIP ${tag} (no model ${mdir})"; n_skip=$((n_skip+1)); return; fi;;
  esac
  local cap=$(( est * 60 * 3 + 1200 ))
  log "RUN ${tag} [${class}] (est ${est}m, cap ${cap}s, elapsed ${now}m) -> engine/u1_${tag}.log"
  local t rc dt
  t=$(date +%s)
  timeout --signal=TERM --kill-after=60 "${cap}s" bash -c "$cmd" >> "engine/u1_${tag}.log" 2>&1
  rc=$?; dt=$(( $(date +%s) - t ))
  local vmode="full"; [ "$class" = "SMOKE" ] && vmode="smoke"
  if [ "$rc" -eq 0 ] && [ -n "$outj" ] && [ -f "$outj" ] && [ -f "$outn" ]; then
    local v; v=$(validate "$outj" "$outn" "$vmode")
    if echo "$v" | grep -q VALIDATE-FAIL; then
      mv "$outj" "$outj.INVALID" 2>/dev/null; mv "$outn" "$outn.INVALID" 2>/dev/null
      log "FAIL ${tag} (${dt}s) OUTPUT-INVALID: ${v}"; n_fail=$((n_fail+1))
    else
      log "done ${tag} (${dt}s) ${v}"; n_done=$((n_done+1)); wedge_fail=0
      [ "$class" = "SMOKE" ] && : > "engine/smoke_u1_${tag}.ok"
    fi
  else
    if [ "$rc" -eq 124 ] || [ "$dt" -ge $(( est * 60 / 2 )) ]; then
      wedge_fail=$((wedge_fail+1)); n_fail=$((n_fail+1))
      log "FAIL ${tag} (rc ${rc}, ${dt}s) WEDGE-LIKE consec=${wedge_fail}/${MAXWEDGE}"
      [ "$wedge_fail" -ge "$MAXWEDGE" ] && { log "ABORT: wedge-like failures"; QUEUE_ABORT=1; }
    else
      n_fail=$((n_fail+1))
      log "FAIL ${tag} (rc ${rc}, ${dt}s) FAST/CONFIG — not counted toward wedge abort"
    fi
  fi
}
heartbeat(){ log "PROGRESS jobs=${n_done}done/${n_fail}fail/${n_skip}skip elapsed=$(elapsed_min)m budget_left=$(( BUDGET_MIN - $(elapsed_min) ))m"; }

# ---------------------------------------------------------------- Phase 0c: micro-smokes
# Never-run paths first. eos/suppress exercise editors/rome_deletion.py at GPU scale for
# the first time (prior contact was a tiny CPU smoke on tiny-random-Llama, 07-02). The
# alpha-delete-refusal combo (editor=alpha + edit_mode=delete) has also never run.
# rome+delete+refusal (Block A/B) is NOT smoked here — it already ran to completion today
# (results/u1e0_llama1b_delete_refusal_L12_s0.json) and needs no re-proof.
run_row SMOKE eos_smoke 4 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome --edit_mode delete --delete_variant eos $CF $SMK --lr 0.1 --layer 12 --seed 0 --out results/smoke_u1/eos.json"
run_row SMOKE suppress_smoke 4 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome --edit_mode delete --delete_variant suppress $CF $SMK --lr 0.1 --layer 12 --seed 0 --out results/smoke_u1/suppress.json"
run_row SMOKE alpha_del_ref_smoke 4 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor alpha --edit_mode delete --delete_variant refusal $CF $SMK --lr 0.1 --layer 12 --seed 0 --out results/smoke_u1/alpha_del_ref.json"
# DROPPED 2026-07-03 23:5x: qv_del_smoke + the QuantEdit-delete FILLER row it gated.
# QuantEdit-E0 returned KILL tonight (results/QUANTEDIT_E0.json: transition-rung
# rho -0.39/-0.67 vs gate) — the "future QuantEdit-E5 deletion unlock" those vectors
# would have fed is dead by its own prereg gate. Do not resurrect without a new E0.
heartbeat

# ---------------------------------------------------------------- Block A: seed hardening (L12 refusal)
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE u1e0_llama1b_delete_refusal_L12_s1 30 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome --edit_mode delete --delete_variant refusal $CF $COMMON --lr 0.1 --layer 12 --seed 1 --out results/u1e0_llama1b_delete_refusal_L12_s1.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE u1e0_llama1b_delete_refusal_L12_s2 30 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome --edit_mode delete --delete_variant refusal $CF $COMMON --lr 0.1 --layer 12 --seed 2 --out results/u1e0_llama1b_delete_refusal_L12_s2.json"
heartbeat

# ---------------------------------------------------------------- Block B: layer profile (refusal, s0)
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE u1e0_llama1b_delete_refusal_L8_s0 30 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome --edit_mode delete --delete_variant refusal $CF $COMMON --lr 0.1 --layer 8 --seed 0 --out results/u1e0_llama1b_delete_refusal_L8_s0.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE u1e0_llama1b_delete_refusal_L14_s0 30 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome --edit_mode delete --delete_variant refusal $CF $COMMON --lr 0.1 --layer 14 --seed 0 --out results/u1e0_llama1b_delete_refusal_L14_s0.json"
heartbeat

# ---------------------------------------------------------------- Block C: variant dissociation (eos/suppress, L12 s0)
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE u1e0_llama1b_delete_eos_L12_s0 35 engine/smoke_u1_eos_smoke.ok "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome --edit_mode delete --delete_variant eos $CF $COMMON --lr 0.1 --layer 12 --seed 0 --out results/u1e0_llama1b_delete_eos_L12_s0.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE u1e0_llama1b_delete_suppress_L12_s0 35 engine/smoke_u1_suppress_smoke.ok "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome --edit_mode delete --delete_variant suppress $CF $COMMON --lr 0.1 --layer 12 --seed 0 --out results/u1e0_llama1b_delete_suppress_L12_s0.json"
heartbeat

# ---------------------------------------------------------------- Block D: mitigation (AlphaEdit delete-refusal, L12 s0)
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE u1e0_llama1b_alpha_delete_refusal_L12_s0 35 engine/smoke_u1_alpha_del_ref_smoke.ok "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor alpha --edit_mode delete --delete_variant refusal $CF $COMMON --lr 0.1 --layer 12 --seed 0 --out results/u1e0_llama1b_alpha_delete_refusal_L12_s0.json"
heartbeat

# ---------------------------------------------------------------- Block E (STRETCH): cross-arch deletion
# Is the deletion-geometry law Llama-family-specific the way the signed REWRITE law is
# (crossarch-transfer-verdict-2026-07-02)? Same code path as Block A/B (rome+delete+refusal),
# already proven to completion at GPU scale today on Llama-1B -- only the model/layer differ,
# so no fresh smoke gate is needed (consistent with Block A/B's own needs="-"). L14 matches
# qwen15b's existing rewrite-law cells (gate_qwen15b_rome_cf_L14_s*.npz) so this cell drops
# straight into the same C3-null comparison scaffold without a new layer config.
[ "$QUEUE_ABORT" -eq 0 ] && run_row FILLER u1e0_qwen15b_delete_refusal_L14_s0 35 - "$ENVP $PY $KG --model data/models/Qwen2.5-1.5B --editor rome --edit_mode delete --delete_variant refusal $CF $COMMON --lr 0.1 --layer 14 --seed 0 --out results/u1e0_qwen15b_delete_refusal_L14_s0.json"
heartbeat

# FILLER QuantEdit-delete arm REMOVED (QuantEdit-E0 KILL 2026-07-03 — see smoke note above).
# Post-run C3_u1_filler_qv / u1_gate_qv_L12_s0 loops below are file-existence-guarded and
# simply no-op with the row gone.

# ---------------------------------------------------------------- Post-run (CPU, ALWAYS)
log "---------------- POST-RUN (CPU) ----------------"
$PY - > results/u1_validation.json 2>>"$LOG" <<'EOF'
import json, glob, os, numpy as np
t0 = float(open('engine/u1_round_start').read().strip())
out = []
for j in sorted(glob.glob('results/*.json')):
    base = os.path.basename(j)[:-5]
    if os.path.getmtime(j) < t0: continue
    if not base.startswith(('u1e0_', 'qv_u1_')): continue
    z = 'results/matrices/' + base + '.npz'
    row = {'json': j, 'npz_found': os.path.exists(z)}
    try:
        d = json.load(open(j)); row['json_ok'] = True; row['esr'] = d.get('edit_success_rate')
    except Exception as e:
        row['json_ok'] = False; row['err'] = str(e)
    if row['npz_found']:
        a = np.load(z)
        row['npz_keys_ok'] = {'COS','damage_logit','norm_growth','edit_ok'} <= set(a.files)
        row['all_nan'] = bool(np.isnan(a['COS'].astype(float)).all())
    out.append(row)
print(json.dumps({'n': len(out), 'rows': out}, indent=1))
EOF
log "post: validation sweep -> results/u1_validation.json"

# analyze_matrices C3 groups per block (Block B/C split per-cell: pooling across layers
# or across delete objectives would conflate distinct geometry regimes / mechanisms —
# scientifically wrong, not a bash convenience)
for spec in "C3_u1_blockA_seeds:results/matrices/u1e0_llama1b_delete_refusal_L12_s*.npz" \
            "C3_u1_blockB_L8:results/matrices/u1e0_llama1b_delete_refusal_L8_s0.npz" \
            "C3_u1_blockB_L14:results/matrices/u1e0_llama1b_delete_refusal_L14_s0.npz" \
            "C3_u1_blockC_eos:results/matrices/u1e0_llama1b_delete_eos_L12_s0.npz" \
            "C3_u1_blockC_suppress:results/matrices/u1e0_llama1b_delete_suppress_L12_s0.npz" \
            "C3_u1_blockD_alphadelete:results/matrices/u1e0_llama1b_alpha_delete_refusal_L12_s0.npz" \
            "C3_u1_blockE_qwen15b:results/matrices/u1e0_qwen15b_delete_refusal_L14_s0.npz" \
            "C3_u1_filler_qv:results/matrices/qv_u1_llama1b_delete_refusal_L12_s0.npz"; do
  outn="${spec%%:*}"; glob="${spec#*:}"
  if compgen -G "$glob" >/dev/null; then
    $PY experiments/analyze_matrices.py $glob --metric logit --known --edit_ok \
      --out "results/${outn}_u1.json" >> "$LOG" 2>&1 && log "post: ${outn}_u1 done" || log "post: ${outn}_u1 FAIL"
  fi
done

# u1_deletion_gate.py — the FORMAL prereg SxC-DC scorer, re-run (or run for the first
# time at real scale) against each new/existing deletion cell's matched insertion npz.
# NOTE: L12_s0 is included even though it already "passed" informally (C3 rho, U1-E1
# transplant) — the formal prereg gate itself was never executed on real data before now.
for spec in "refusal_L12_s0:results/matrices/u1e0_llama1b_delete_refusal_L12_s0.npz:results/matrices/gate_llama1b_rome_cf_L12_s0.npz" \
            "refusal_L12_s1:results/matrices/u1e0_llama1b_delete_refusal_L12_s1.npz:results/matrices/gate_llama1b_rome_cf_L12_s0.npz" \
            "refusal_L12_s2:results/matrices/u1e0_llama1b_delete_refusal_L12_s2.npz:results/matrices/gate_llama1b_rome_cf_L12_s0.npz" \
            "refusal_L8_s0:results/matrices/u1e0_llama1b_delete_refusal_L8_s0.npz:results/matrices/gate_llama1b_rome_cf_L8_s0.npz" \
            "refusal_L14_s0:results/matrices/u1e0_llama1b_delete_refusal_L14_s0.npz:results/matrices/gate_llama1b_rome_cf_L14_s0.npz" \
            "eos_L12_s0:results/matrices/u1e0_llama1b_delete_eos_L12_s0.npz:results/matrices/gate_llama1b_rome_cf_L12_s0.npz" \
            "suppress_L12_s0:results/matrices/u1e0_llama1b_delete_suppress_L12_s0.npz:results/matrices/gate_llama1b_rome_cf_L12_s0.npz" \
            "alphadelete_L12_s0:results/matrices/u1e0_llama1b_alpha_delete_refusal_L12_s0.npz:results/matrices/g4_llama1b_alpha_cf_L12_s0.npz" \
            "qwen15b_refusal_L14_s0:results/matrices/u1e0_qwen15b_delete_refusal_L14_s0.npz:results/matrices/gate_qwen15b_rome_cf_L14_s0.npz" \
            "qv_L12_s0:results/matrices/qv_u1_llama1b_delete_refusal_L12_s0.npz:results/matrices/gate_llama1b_rome_cf_L12_s0.npz"; do
  tagn="${spec%%:*}"; rest="${spec#*:}"; del="${rest%%:*}"; ins="${rest#*:}"
  if [ -f "$del" ] && [ -f "$ins" ]; then
    $PY experiments/u1_deletion_gate.py --del_npz "$del" --ins_npz "$ins" --metric logit \
      --out "results/u1_gate_${tagn}.json" >> "$LOG" 2>&1 && log "post: u1_gate_${tagn} done" || log "post: u1_gate_${tagn} FAIL"
  fi
done

# u1_transplant.py U1-E1 gate on EVERY new refusal-variant npz this driver can produce
# (L12_s0 already scored in results/U1_E1_transplant_GATE_L12_s0.json from the r4 run --
# not re-scored here). u1_transplant.py is defined ONLY for the refusal transplant question
# (its loader reimplements the ORIGINAL, pre-swap target_new -- see its module docstring),
# so eos/suppress/qv-vector cells are intentionally excluded from this loop; their falsifiable
# gate is u1_deletion_gate.py above. --n_edits/--n_probes/--seed must match the row that
# produced each npz or the loader/npz alignment check raises.
for spec in "L12_s1:results/matrices/u1e0_llama1b_delete_refusal_L12_s1.npz:1" \
            "L12_s2:results/matrices/u1e0_llama1b_delete_refusal_L12_s2.npz:2" \
            "L8_s0:results/matrices/u1e0_llama1b_delete_refusal_L8_s0.npz:0" \
            "L14_s0:results/matrices/u1e0_llama1b_delete_refusal_L14_s0.npz:0" \
            "alphadelete_L12_s0:results/matrices/u1e0_llama1b_alpha_delete_refusal_L12_s0.npz:0" \
            "qwen15b_L14_s0:results/matrices/u1e0_qwen15b_delete_refusal_L14_s0.npz:0"; do
  name="${spec%%:*}"; rest="${spec#*:}"; npz="${rest%%:*}"; seed="${rest#*:}"
  if [ -f "$npz" ]; then
    $PY experiments/u1_transplant.py --npz "$npz" \
      --dataset counterfact --data data/counterfact.json --n_edits 200 --n_probes 500 --seed "$seed" \
      --known --edit_ok --out "results/U1_E1_transplant_GATE_${name}.json" >> "$LOG" 2>&1 \
      && log "post: U1_E1_transplant_GATE_${name} done" || log "post: U1_E1_transplant_GATE_${name} FAIL"
  fi
done

{
  echo "RUN_U1 REPORT $(date '+%F %T')  jobs: ${n_done} done / ${n_fail} fail / ${n_skip} skipped  elapsed $(elapsed_min)m/${BUDGET_MIN}m  abort=${QUEUE_ABORT}"
  grep -E 'RUN |done |FAIL |SKIP|ABORT|PROGRESS' "$LOG" | tail -50
} > engine/run_u1_report.txt
log "================ RUN_U1 COMPLETE (${n_done} done / ${n_fail} fail / ${n_skip} skip) ================"
echo "RUN_U1_DONE" >> "$LOG"
