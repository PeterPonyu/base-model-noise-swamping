#!/bin/bash
# run_u2.sh — U1 hardening seeds + KL-ladder (2026-07-04). Template = run_u1.sh
# (verbatim preflight/idle-gate/validate/run_row/heartbeat/post-run skeleton), U2-namespaced.
# Program:
#   (A) SEED-HARDEN the two single-seed U1 filler cells flagged in the 07-04 status read:
#       AlphaEdit-delete (u1e0_llama1b_alpha_delete_refusal_L12 s1/s2) and cross-arch
#       Qwen-delete (u1e0_qwen15b_delete_refusal_L14 s1/s2). Both row families ran to
#       completion at GPU scale in run_u1 (s0, VALIDATE-OK) — no never-run code path,
#       so no fresh smoke gate is needed (run_u1 Block E precedent).
#   (B) KL-LADDER (EOD plan Phase 2.5): ft_kl in {0.03, 0.3, 1.0} at L8 s0, turning the
#       editor locality spectrum's FT->KL-FT segment into a dose-response curve.
#       kl=0.1 already exists 3-seed (gate_llama1b_ftkl_cf_L8_s0/1/2); kl=0 is plain FT.
#       Tag digits = weight*100: ftkl003=0.03, ftkl030=0.3, ftkl100=1.0.
# Measured ests (healthy 175W baseline, 07-03/04 runs): alpha-delete 1567s, qwen-delete
# 2111s, ftkl 1609-1823s -> ests 30/38/30 min, caps 3x+20m.
set -u
H=/home/zeyufu/Desktop/idea-feasibility-analysis/edit-harness
cd "$H" || exit 1
PY=/home/zeyufu/miniconda3/envs/dl/bin/python3
LOG=engine/run_u2.log
BUDGET_MIN=${BUDGET_MIN:-300}
mkdir -p engine results/matrices
echo $$ > engine/run_u2.pid
[ -f engine/u2_round_start ] || stat -c %Y engine/run_u2.pid > engine/u2_round_start
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "================ RUN_U2 START (pid $$, budget ${BUDGET_MIN}m) ================"

# ---------------------------------------------------------------- Phase 0a: CPU pre-flight
pf_fail=0
pf(){ if eval "$2"; then log "preflight OK: $1"; else log "PREFLIGHT-FAIL: $1"; pf_fail=1; fi; }
pf "python env" "$PY -c 'import torch, numpy' 2>/dev/null"
pf "counterfact.json" "[ -f data/counterfact.json ]"
pf "edit_mode flag" "grep -q -- '--edit_mode' experiments/killgate_keygeom.py"
pf "delete_variant flag" "grep -q -- '--delete_variant' experiments/killgate_keygeom.py"
pf "ft_kl flag" "grep -q -- '--ft_kl' experiments/killgate_keygeom.py"
pf "rome_deletion editor" "[ -f editors/rome_deletion.py ]"
pf "u1_deletion_gate.py" "[ -f experiments/u1_deletion_gate.py ]"
pf "u1_transplant.py" "[ -f experiments/u1_transplant.py ]"
pf "analyze_matrices.py" "[ -f experiments/analyze_matrices.py ]"
pf "model Llama-3.2-1B" "[ -d data/models/Llama-3.2-1B ]"
pf "model Qwen2.5-1.5B" "[ -d data/models/Qwen2.5-1.5B ]"
pf "matched alpha-insertion L12 s0" "[ -f results/matrices/g4_llama1b_alpha_cf_L12_s0.npz ]"
pf "matched qwen15b-insertion L14 s0" "[ -f results/matrices/gate_qwen15b_rome_cf_L14_s0.npz ]"
pf "disk >=20GB free" "[ \$(df --output=avail -BG /home | tail -1 | tr -dc 0-9) -ge 20 ]"
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

# ---------------------------------------------------------------- helpers (u1 template, verbatim)
ENVP="env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1"
KG="experiments/killgate_keygeom.py"
CF="--dataset counterfact --data data/counterfact.json"
COMMON="--n_edits 200 --n_probes 500 --steps 20 --save_matrices --matrix_dir results/matrices"
elapsed_min(){ echo $(( ( $(date +%s) - T0 ) / 60 )); }

validate(){
  $PY - "$1" "$2" "${3:-full}" <<'EOF'
import json, os, sys, numpy as np
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
    elif "ftkl" in os.path.basename(j):
        # KL-regularized FT: heavy KL can legitimately drive edit success toward zero —
        # that collapse IS the ladder's dose-response endpoint (u2 review MED-1). Warn, never fail.
        if esr is not None and esr < 0.9: print(f"VALIDATE-NOTE kl-ft esr={esr}")
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
  if [ -n "$outj" ] && [ -f "$outj" ] && [ -f "$outn" ]; then
    if validate "$outj" "$outn" full | grep -q VALIDATE-FAIL; then
      mv "$outj" "$outj.INVALID" 2>/dev/null; mv "$outn" "$outn.INVALID" 2>/dev/null
      log "STALE-INVALID ${tag} — quarantined; re-running"
    else
      log "skip ${tag} (exists, validated)"
      return
    fi
  fi
  local mdir; mdir=$(echo "$cmd" | grep -oE -- '--model [^ ]+' | awk '{print $2}')
  case "$mdir" in data/*|/*)
    if [ -n "$mdir" ] && [ ! -d "$mdir" ]; then log "CONFIG-SKIP ${tag} (no model ${mdir})"; n_skip=$((n_skip+1)); return; fi;;
  esac
  local cap=$(( est * 60 * 3 + 1200 ))
  log "RUN ${tag} [${class}] (est ${est}m, cap ${cap}s, elapsed ${now}m) -> engine/u2_${tag}.log"
  local t rc dt
  t=$(date +%s)
  timeout --signal=TERM --kill-after=60 "${cap}s" bash -c "$cmd" >> "engine/u2_${tag}.log" 2>&1
  rc=$?; dt=$(( $(date +%s) - t ))
  if [ "$rc" -eq 0 ] && [ -n "$outj" ] && [ -f "$outj" ] && [ -f "$outn" ]; then
    local v; v=$(validate "$outj" "$outn" full)
    if echo "$v" | grep -q VALIDATE-FAIL; then
      mv "$outj" "$outj.INVALID" 2>/dev/null; mv "$outn" "$outn.INVALID" 2>/dev/null
      log "FAIL ${tag} (${dt}s) OUTPUT-INVALID: ${v}"; n_fail=$((n_fail+1))
    else
      log "done ${tag} (${dt}s) ${v}"; n_done=$((n_done+1)); wedge_fail=0
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

# ---------------------------------------------------------------- Block A: U1 seed hardening
# AlphaEdit-delete seeds (causal-mitigation arm of the U1 deletion section)
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE u1e0_llama1b_alpha_delete_refusal_L12_s1 30 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor alpha --edit_mode delete --delete_variant refusal $CF $COMMON --lr 0.1 --layer 12 --seed 1 --out results/u1e0_llama1b_alpha_delete_refusal_L12_s1.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE u1e0_llama1b_alpha_delete_refusal_L12_s2 30 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor alpha --edit_mode delete --delete_variant refusal $CF $COMMON --lr 0.1 --layer 12 --seed 2 --out results/u1e0_llama1b_alpha_delete_refusal_L12_s2.json"
heartbeat
# Cross-arch Qwen-delete seeds (Llama-scoping control of the U1 deletion section)
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE u1e0_qwen15b_delete_refusal_L14_s1 38 - "$ENVP $PY $KG --model data/models/Qwen2.5-1.5B --editor rome --edit_mode delete --delete_variant refusal $CF $COMMON --lr 0.1 --layer 14 --seed 1 --out results/u1e0_qwen15b_delete_refusal_L14_s1.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE u1e0_qwen15b_delete_refusal_L14_s2 38 - "$ENVP $PY $KG --model data/models/Qwen2.5-1.5B --editor rome --edit_mode delete --delete_variant refusal $CF $COMMON --lr 0.1 --layer 14 --seed 2 --out results/u1e0_qwen15b_delete_refusal_L14_s2.json"
heartbeat

# ---------------------------------------------------------------- Block B: KL-ladder (dose-response)
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE gate_llama1b_ftkl003_cf_L8_s0 30 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor ft $CF $COMMON --ft_lr 5e-3 --ft_kl 0.03 --ft_kl_n 5 --layer 8 --seed 0 --out results/gate_llama1b_ftkl003_cf_L8_s0.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE gate_llama1b_ftkl030_cf_L8_s0 30 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor ft $CF $COMMON --ft_lr 5e-3 --ft_kl 0.3 --ft_kl_n 5 --layer 8 --seed 0 --out results/gate_llama1b_ftkl030_cf_L8_s0.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE gate_llama1b_ftkl100_cf_L8_s0 30 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor ft $CF $COMMON --ft_lr 5e-3 --ft_kl 1.0 --ft_kl_n 5 --layer 8 --seed 0 --out results/gate_llama1b_ftkl100_cf_L8_s0.json"
heartbeat

# ---------------------------------------------------------------- Post-run (CPU, ALWAYS)
log "---------------- POST-RUN (CPU) ----------------"
$PY - > results/u2_validation.json 2>>"$LOG" <<'EOF'
import json, glob, os, numpy as np
t0 = float(open('engine/u2_round_start').read().strip())
out = []
for j in sorted(glob.glob('results/*.json')):
    base = os.path.basename(j)[:-5]
    if os.path.getmtime(j) < t0: continue
    if not base.startswith(('u1e0_', 'gate_llama1b_ftkl')): continue
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
log "post: validation sweep -> results/u2_validation.json"

# analyze_matrices C3 groups. Seed-pooling within a fixed (model, editor, layer, variant)
# cell is the Block-A precedent; KL weights stay per-cell (pooling across regularization
# strengths would average out the dose-response the ladder exists to measure).
for spec in "C3_u1_blockD_alphadelete_seeds:results/matrices/u1e0_llama1b_alpha_delete_refusal_L12_s*.npz" \
            "C3_u1_blockE_qwen15b_seeds:results/matrices/u1e0_qwen15b_delete_refusal_L14_s*.npz" \
            "C3_klladder_003_L8:results/matrices/gate_llama1b_ftkl003_cf_L8_s0.npz" \
            "C3_klladder_030_L8:results/matrices/gate_llama1b_ftkl030_cf_L8_s0.npz" \
            "C3_klladder_100_L8:results/matrices/gate_llama1b_ftkl100_cf_L8_s0.npz"; do
  outn="${spec%%:*}"; glob="${spec#*:}"
  if compgen -G "$glob" >/dev/null; then
    $PY experiments/analyze_matrices.py $glob --metric logit --known --edit_ok \
      --out "results/${outn}_u2.json" >> "$LOG" 2>&1 && log "post: ${outn}_u2 done" || log "post: ${outn}_u2 FAIL"
  fi
done

# u1_deletion_gate.py — formal prereg scorer for the new deletion seeds. Insertion
# reference stays the s0 npz of the matched editor/model/layer (run_u1 precedent:
# refusal s1/s2 scored against gate_..._L12_s0).
for spec in "alphadelete_L12_s1:results/matrices/u1e0_llama1b_alpha_delete_refusal_L12_s1.npz:results/matrices/g4_llama1b_alpha_cf_L12_s0.npz" \
            "alphadelete_L12_s2:results/matrices/u1e0_llama1b_alpha_delete_refusal_L12_s2.npz:results/matrices/g4_llama1b_alpha_cf_L12_s0.npz" \
            "qwen15b_refusal_L14_s1:results/matrices/u1e0_qwen15b_delete_refusal_L14_s1.npz:results/matrices/gate_qwen15b_rome_cf_L14_s0.npz" \
            "qwen15b_refusal_L14_s2:results/matrices/u1e0_qwen15b_delete_refusal_L14_s2.npz:results/matrices/gate_qwen15b_rome_cf_L14_s0.npz"; do
  tagn="${spec%%:*}"; rest="${spec#*:}"; del="${rest%%:*}"; ins="${rest#*:}"
  if [ -f "$del" ] && [ -f "$ins" ]; then
    $PY experiments/u1_deletion_gate.py --del_npz "$del" --ins_npz "$ins" --metric logit \
      --out "results/u1_gate_${tagn}.json" >> "$LOG" 2>&1 && log "post: u1_gate_${tagn} done" || log "post: u1_gate_${tagn} FAIL"
  fi
done

# u1_transplant.py U1-E1 gate on the new refusal-variant npz (refusal cells only — its
# loader is defined for the original pre-swap target_new; run_u1 docstring note applies).
# --seed must match the row that produced each npz.
for spec in "alphadelete_L12_s1:results/matrices/u1e0_llama1b_alpha_delete_refusal_L12_s1.npz:1" \
            "alphadelete_L12_s2:results/matrices/u1e0_llama1b_alpha_delete_refusal_L12_s2.npz:2" \
            "qwen15b_L14_s1:results/matrices/u1e0_qwen15b_delete_refusal_L14_s1.npz:1" \
            "qwen15b_L14_s2:results/matrices/u1e0_qwen15b_delete_refusal_L14_s2.npz:2"; do
  name="${spec%%:*}"; rest="${spec#*:}"; npz="${rest%%:*}"; seed="${rest#*:}"
  if [ -f "$npz" ]; then
    $PY experiments/u1_transplant.py --npz "$npz" \
      --dataset counterfact --data data/counterfact.json --n_edits 200 --n_probes 500 --seed "$seed" \
      --known --edit_ok --out "results/U1_E1_transplant_GATE_${name}.json" >> "$LOG" 2>&1 \
      && log "post: U1_E1_transplant_GATE_${name} done" || log "post: U1_E1_transplant_GATE_${name} FAIL"
  fi
done

{
  echo "RUN_U2 REPORT $(date '+%F %T')  jobs: ${n_done} done / ${n_fail} fail / ${n_skip} skipped  elapsed $(elapsed_min)m/${BUDGET_MIN}m  abort=${QUEUE_ABORT}"
  grep -E 'RUN |done |FAIL |SKIP|ABORT|PROGRESS' "$LOG" | tail -50
} > engine/run_u2_report.txt
log "================ RUN_U2 COMPLETE (${n_done} done / ${n_fail} fail / ${n_skip} skip) ================"
echo "RUN_U2_DONE" >> "$LOG"
