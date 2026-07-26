#!/bin/bash
# run_r4.sh — ROUND 4 (2026-07-03): NEW-direction first cells + regime-law seed hardening.
# Template = run_r3.sh (review-hardened). See PLAN-12H-RUN-2026-07-03.md.
# Manifest: micro-smokes (deletion, no_restore — NEVER GPU-run paths) -> Block C new science
# (U1-E0 deletion gate; sequential no-restore streams x2) -> Block B regime seeds (8B/3B).
set -u
H=/home/zeyufu/Desktop/idea-feasibility-analysis/edit-harness
cd "$H" || exit 1
PY=/home/zeyufu/miniconda3/envs/dl/bin/python3
LOG=engine/run_r4.log
BUDGET_MIN=${BUDGET_MIN:-300}
mkdir -p engine results/matrices results/smoke_r4/matrices
echo $$ > engine/run_r4.pid
[ -f engine/r4_round_start ] || stat -c %Y engine/run_r4.pid > engine/r4_round_start
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "================ RUN_R4 START (pid $$, budget ${BUDGET_MIN}m) ================"

# ---------------------------------------------------------------- Phase 0a: CPU pre-flight
pf_fail=0
pf(){ if eval "$2"; then log "preflight OK: $1"; else log "PREFLIGHT-FAIL: $1"; pf_fail=1; fi; }
pf "python env" "$PY -c 'import torch, numpy' 2>/dev/null"
pf "counterfact.json" "[ -f data/counterfact.json ]"
pf "edit_mode flag" "grep -q -- '--edit_mode' experiments/killgate_keygeom.py"
pf "no_restore flag" "grep -q -- '--no_restore' experiments/killgate_keygeom.py"
pf "rome_deletion editor" "[ -f editors/rome_deletion.py ]"
pf "model Llama-3.2-1B" "[ -d data/models/Llama-3.2-1B ]"
pf "disk >=20GB free" "[ \$(df --output=avail -BG /home | tail -1 | tr -dc 0-9) -ge 20 ]"
# fail-closed marker lifecycle (r3 review HIGH #1)
rm -f engine/r4_integrity_8b.ok engine/smoke_r4_*.ok
$PY experiments/tools/integrity_check.py data/models/Llama-3.1-8B --expect_params 8.03e9 >> "$LOG" 2>&1 \
  && { : > engine/r4_integrity_8b.ok; log "integrity OK: Llama-3.1-8B"; } \
  || log "integrity NOT-READY: Llama-3.1-8B (8B rows CONFIG-skip)"
# bf16 equivalence: REUSE r3's comparator result only if BOTH npz exist; re-derive fresh
rm -f engine/r4_equiv_bf16.ok
# Freshness guard (review MEDIUM): the reused equiv npz certifies THAT-era killgate bf16
# behavior; if killgate was edited after the npz was made, the proof is stale -> fail closed.
if [ -f results/matrices/equiv_llama1b_bf16_L12_s0.npz ] && [ -f results/matrices/gate_llama1b_rome_cf_L12_s0.npz ] \
   && [ "$(stat -c %Y results/matrices/equiv_llama1b_bf16_L12_s0.npz)" -ge "$(stat -c %Y experiments/killgate_keygeom.py)" ]; then
  $PY - >> "$LOG" 2>&1 <<'EOF'
import numpy as np, sys
sys.path.insert(0, 'experiments')
from analyze_matrices import within_probe_rhos
def rho(f):
    d = np.load(f); C = d['COS'].astype(float); D = d['damage_logit'].astype(float)
    m = d['edit_ok'].astype(float) > 0; c = d['pre_p'].astype(float) > 0.05
    return float(np.nanmean(within_probe_rhos(C[m][:, c], D[m][:, c])))
d = abs(rho('results/matrices/gate_llama1b_rome_cf_L12_s0.npz')
        - rho('results/matrices/equiv_llama1b_bf16_L12_s0.npz'))
print(f"[r4 equiv-gate] |drho|={d:.4f} bar=0.02")
if d < 0.02:
    open('engine/r4_equiv_bf16.ok', 'w').close(); print("[r4 equiv-gate] PASS")
else:
    print("[r4 equiv-gate] FAIL — 8B rows stay CONFIG-skipped")
EOF
else
  log "equiv npz missing OR older than killgate code (stale proof) — 8B rows will CONFIG-skip"
fi
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

# ---------------------------------------------------------------- helpers (r3 template)
ENVP="env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1"
KG="experiments/killgate_keygeom.py"
CF="--dataset counterfact --data data/counterfact.json"
COMMON="--n_edits 200 --n_probes 500 --steps 20 --save_matrices --matrix_dir results/matrices"
SMK="--n_edits 4 --n_probes 40 --steps 2 --save_matrices --matrix_dir results/smoke_r4/matrices"
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
    # SEQ probe_stride writes intentional NaN rows; all-NaN is still invalid
    if np.isnan(arr).all():
        print(f"VALIDATE-FAIL {k} all-NaN"); sys.exit(1)
# seq cells: the prior-edit overwrite panel is the H1 science — gate its presence (review LOW-2)
if "seq_no_restore" in a.files and int(a["seq_no_restore"]) == 1:
    if "prior_eff" not in a.files or np.isnan(a["prior_eff"].astype(float)).all():
        print("VALIDATE-FAIL seq cell missing/all-NaN prior_eff panel"); sys.exit(1)
if mode == "full":
    esr = d.get("edit_success_rate")
    is_delete = "edit_mode" in a.files and str(a["edit_mode"]) == "delete"
    if is_delete:
        # delete mode: esr = 2x-suppression rate; low/zero is a legitimate NEGATIVE finding,
        # not breakage (review LOW-1) — warn, never fail.
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
  case "$cmd" in *smoke_r4*) outn="results/smoke_r4/matrices/$(basename "${outj%.json}").npz";; esac
  if [ -n "$outj" ] && [ -f "$outj" ] && [ -f "$outn" ]; then
    local pvmode="full"; [ "$class" = "SMOKE" ] && pvmode="smoke"
    if validate "$outj" "$outn" "$pvmode" | grep -q VALIDATE-FAIL; then
      mv "$outj" "$outj.INVALID" 2>/dev/null; mv "$outn" "$outn.INVALID" 2>/dev/null
      log "STALE-INVALID ${tag} — quarantined; re-running"
    else
      log "skip ${tag} (exists, validated)"
      [ "$class" = "SMOKE" ] && : > "engine/smoke_r4_${tag}.ok"
      return
    fi
  fi
  local mdir; mdir=$(echo "$cmd" | grep -oE -- '--model [^ ]+' | awk '{print $2}')
  case "$mdir" in data/*|/*)
    if [ -n "$mdir" ] && [ ! -d "$mdir" ]; then log "CONFIG-SKIP ${tag} (no model ${mdir})"; n_skip=$((n_skip+1)); return; fi;;
  esac
  local cap=$(( est * 60 * 3 + 1200 ))
  log "RUN ${tag} [${class}] (est ${est}m, cap ${cap}s, elapsed ${now}m) -> engine/r4_${tag}.log"
  local t rc dt
  t=$(date +%s)
  timeout --signal=TERM --kill-after=60 "${cap}s" bash -c "$cmd" >> "engine/r4_${tag}.log" 2>&1
  rc=$?; dt=$(( $(date +%s) - t ))
  local vmode="full"; [ "$class" = "SMOKE" ] && vmode="smoke"
  if [ "$rc" -eq 0 ] && [ -n "$outj" ] && [ -f "$outj" ] && [ -f "$outn" ]; then
    local v; v=$(validate "$outj" "$outn" "$vmode")
    if echo "$v" | grep -q VALIDATE-FAIL; then
      mv "$outj" "$outj.INVALID" 2>/dev/null; mv "$outn" "$outn.INVALID" 2>/dev/null
      log "FAIL ${tag} (${dt}s) OUTPUT-INVALID: ${v}"; n_fail=$((n_fail+1))
    else
      log "done ${tag} (${dt}s) ${v}"; n_done=$((n_done+1)); wedge_fail=0
      [ "$class" = "SMOKE" ] && : > "engine/smoke_r4_${tag}.ok"
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

# ---------------------------------------------------------------- Phase 0c: micro-smokes (never-run paths)
run_row SMOKE del_ref 4 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome --edit_mode delete --delete_variant refusal $CF $SMK --lr 0.1 --layer 12 --seed 0 --out results/smoke_r4/del_ref.json"
run_row SMOKE seq_nr 4 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome --no_restore --recheck_every 2 $CF $SMK --lr 0.1 --layer 12 --seed 0 --out results/smoke_r4/seq_nr.json"
heartbeat

# ---------------------------------------------------------------- Block C: NEW-direction science (first)
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE u1e0_llama1b_delete_refusal_L12_s0 30 engine/smoke_r4_del_ref.ok "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome --edit_mode delete --delete_variant refusal $CF $COMMON --lr 0.1 --layer 12 --seed 0 --out results/u1e0_llama1b_delete_refusal_L12_s0.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE seq_llama1b_nr_L12_s0 14 engine/smoke_r4_seq_nr.ok "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome --no_restore --recheck_every 10 --dataset counterfact --data data/counterfact.json --n_edits 50 --n_probes 500 --steps 20 --save_matrices --matrix_dir results/matrices --lr 0.1 --layer 12 --seed 0 --out results/seq_llama1b_nr_L12_s0.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE seq_llama1b_nr_L12_s1 14 engine/smoke_r4_seq_nr.ok "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome --no_restore --recheck_every 10 --dataset counterfact --data data/counterfact.json --n_edits 50 --n_probes 500 --steps 20 --save_matrices --matrix_dir results/matrices --lr 0.1 --layer 12 --seed 1 --out results/seq_llama1b_nr_L12_s1.json"
heartbeat

# ---------------------------------------------------------------- Block B: regime-law seed hardening (filler)
[ "$QUEUE_ABORT" -eq 0 ] && run_row FILLER gate_llama8b_rome_cf_L24_s1 50 engine/r4_integrity_8b.ok,engine/r4_equiv_bf16.ok "$ENVP $PY $KG --model data/models/Llama-3.1-8B --model_dtype bf16 --editor rome $CF $COMMON --lr 0.1 --layer 24 --seed 1 --out results/gate_llama8b_rome_cf_L24_s1.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row FILLER gate_llama8b_rome_cf_L24_s2 50 engine/r4_integrity_8b.ok,engine/r4_equiv_bf16.ok "$ENVP $PY $KG --model data/models/Llama-3.1-8B --model_dtype bf16 --editor rome $CF $COMMON --lr 0.1 --layer 24 --seed 2 --out results/gate_llama8b_rome_cf_L24_s2.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row FILLER gate_llama8b_rome_cf_L16_s1 50 engine/r4_integrity_8b.ok,engine/r4_equiv_bf16.ok "$ENVP $PY $KG --model data/models/Llama-3.1-8B --model_dtype bf16 --editor rome $CF $COMMON --lr 0.1 --layer 16 --seed 1 --out results/gate_llama8b_rome_cf_L16_s1.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row FILLER gate_llama3b_rome_cf_L24_s1 55 - "$ENVP $PY $KG --model data/models/Llama-3.2-3B --editor rome $CF $COMMON --lr 0.1 --layer 24 --seed 1 --out results/gate_llama3b_rome_cf_L24_s1.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row FILLER gate_llama3b_rome_cf_L24_s2 55 - "$ENVP $PY $KG --model data/models/Llama-3.2-3B --editor rome $CF $COMMON --lr 0.1 --layer 24 --seed 2 --out results/gate_llama3b_rome_cf_L24_s2.json"
heartbeat

# ---------------------------------------------------------------- Post-run (CPU, ALWAYS)
log "---------------- POST-RUN (CPU) ----------------"
$PY - > results/r4_validation.json 2>>"$LOG" <<'EOF'
import json, glob, os, numpy as np
t0 = float(open('engine/r4_round_start').read().strip())
out = []
for j in sorted(glob.glob('results/*.json')):
    base = os.path.basename(j)[:-5]
    if os.path.getmtime(j) < t0: continue
    if not base.startswith(('gate_', 'g4_', 'u1e0_', 'seq_')): continue
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
log "post: validation sweep -> results/r4_validation.json"
for spec in "C3_regime_8b_L24:results/matrices/gate_llama8b_rome_cf_L24_s*.npz" \
            "C3_regime_3b_L24:results/matrices/gate_llama3b_rome_cf_L24_s*.npz" \
            "C3_u1e0_delete:results/matrices/u1e0_llama1b_delete_refusal_L12_s*.npz"; do
  outn="${spec%%:*}"; glob="${spec#*:}"
  if compgen -G "$glob" >/dev/null; then
    $PY experiments/analyze_matrices.py $glob --metric logit --known --edit_ok \
      --out "results/${outn}_r4.json" >> "$LOG" 2>&1 && log "post: ${outn}_r4 done" || log "post: ${outn}_r4 FAIL"
  fi
done
{
  echo "RUN_R4 REPORT $(date '+%F %T')  jobs: ${n_done} done / ${n_fail} fail / ${n_skip} skipped  elapsed $(elapsed_min)m/${BUDGET_MIN}m  abort=${QUEUE_ABORT}"
  grep -E 'RUN |done |FAIL |SKIP|ABORT|integrity|equiv-gate|PROGRESS' "$LOG" | tail -50
} > engine/run_r4_report.txt
log "================ RUN_R4 COMPLETE (${n_done} done / ${n_fail} fail / ${n_skip} skip) ================"
echo "RUN_R4_DONE" >> "$LOG"
