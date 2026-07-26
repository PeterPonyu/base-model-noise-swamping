#!/bin/bash
# run_u5.sh — evening/overnight queue 2026-07-04 (~15:40 -> ~24:00). Template = run_u2/u4
# (verbatim skeleton), U5-namespaced. Launches AFTER run_u4 drains. Fills the GPU stack the
# user asked for ("next 12 hours") with paper-hardening seeds, ordered by review leverage:
#   V. ANISO SEEDS: L14 raw-key banks s1/s2 for BOTH models — the aniso L14 contrast is
#      currently a single-seed two-point comparison, flagged by its own module + reviewer.
#      save_vectors path proven at science scale on llama (run8h) AND qwen (run_u3 today).
#   U. U1 LAYER-PROFILE SEEDS: refusal L8 s1/s2 + L14 s1/s2 — §7's layer profile is s0-only
#      at the flanking layers. Path proven (run_u1 Blocks A/B).
#   K. KL-LADDER s2: 0.03/0.3/1.0 at L8 — completes 3-seed error bars (s0 u2, s1 u4 filler).
#   Q. SEQ FLANK SEEDS: L8/L14 second orderings (s1) to pair u4's s0 singles.
#   Z. zsRE-DELETION (new science, smoke-gated): does deletion-collateral geometry hold off
#      CounterFact? rome+delete proven; zsre proven; the delete+zsre COMBO never ran.
#   F. FILLER: KL-ladder at L12 (0.03/0.1/0.3/1.0 s0 — dose-response at the law's peak
#      layer; ft@L12 and ftkl@L8 both proven, config-only), EGL s1 seeds (rome/memit/alpha).
# Measured ests: qv-vector llama 28 / qwen 36; refusal ~28; ftkl 27-31; seq ~7-10; egl 22-31.
# THERMAL NOTE: continuous load since ~05:00 today; the historical 60W SW-thermal wedge
# appeared after ~19h load (memory: gpu-60w-thermal-cap-reboot-fix). If any cell runs
# >1.4x its healthy twin, check power.draw — <100W at high util = wedge; stop at boundary
# (engine/stop_deep_queue.sh engine/run_u5.pid) and reboot per engine/AFTER_REBOOT.txt.
set -u
H=/home/zeyufu/Desktop/idea-feasibility-analysis/edit-harness
cd "$H" || exit 1
PY=/home/zeyufu/miniconda3/envs/dl/bin/python3
LOG=engine/run_u5.log
BUDGET_MIN=${BUDGET_MIN:-520}
mkdir -p engine results/matrices results/vectors results/smoke_u5/matrices
echo $$ > engine/run_u5.pid
[ -f engine/u5_round_start ] || stat -c %Y engine/run_u5.pid > engine/u5_round_start
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "================ RUN_U5 START (pid $$, budget ${BUDGET_MIN}m) ================"

# ---------------------------------------------------------------- Phase 0a: CPU pre-flight
pf_fail=0
pf(){ if eval "$2"; then log "preflight OK: $1"; else log "PREFLIGHT-FAIL: $1"; pf_fail=1; fi; }
pf "python env" "$PY -c 'import torch, numpy' 2>/dev/null"
pf "counterfact.json" "[ -f data/counterfact.json ]"
pf "zsre_eval.json" "[ -f data/zsre_eval.json ]"
pf "edit_mode flag" "grep -q -- '--edit_mode' experiments/killgate_keygeom.py"
pf "save_vectors flag" "grep -q -- '--save_vectors' experiments/killgate_keygeom.py"
pf "ft_kl flag" "grep -q -- '--ft_kl' experiments/killgate_keygeom.py"
pf "egl flag" "grep -q -- '--egl' experiments/killgate_keygeom.py"
pf "u1_deletion_gate.py" "[ -f experiments/u1_deletion_gate.py ]"
pf "u1_transplant.py" "[ -f experiments/u1_transplant.py ]"
pf "analyze_aniso.py" "[ -f experiments/analyze_aniso.py ]"
pf "model Llama-3.2-1B" "[ -d data/models/Llama-3.2-1B ]"
pf "model Qwen2.5-1.5B" "[ -d data/models/Qwen2.5-1.5B ]"
pf "matched insertion L8 s0" "[ -f results/matrices/gate_llama1b_rome_cf_L8_s0.npz ]"
pf "matched insertion L14 s0" "[ -f results/matrices/gate_llama1b_rome_cf_L14_s0.npz ]"
pf "zsre insertion ref L10 s0" "[ -f results/matrices/gate_llama1b_rome_zsre_L10_s0.npz ]"
pf "disk >=20GB free" "[ \$(df --output=avail -BG /home | tail -1 | tr -dc 0-9) -ge 20 ]"
rm -f engine/smoke_u5_*.ok
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

# ---------------------------------------------------------------- helpers (u2/u4 template, verbatim)
ENVP="env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1"
KG="experiments/killgate_keygeom.py"
CF="--dataset counterfact --data data/counterfact.json"
ZS="--dataset zsre --data data/zsre_eval.json"
COMMON="--n_edits 200 --n_probes 500 --steps 20 --save_matrices --matrix_dir results/matrices"
SMK="--n_edits 4 --n_probes 40 --steps 2 --save_matrices --matrix_dir results/smoke_u5/matrices"
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
    is_seq = "seq_no_restore" in a.files and int(a["seq_no_restore"]) == 1
    if is_seq and "prior_eff" not in a.files:
        print("VALIDATE-FAIL seq npz missing prior_eff"); sys.exit(1)
    if is_delete or is_seq:
        # delete: esr = 2x-suppression; seq: stream esr drifts. Legitimate negatives — warn only.
        if esr is not None and esr < 0.9: print(f"VALIDATE-NOTE soft-esr mode rate={esr}")
    elif "ftkl" in os.path.basename(j):
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
  case "$cmd" in *smoke_u5*) outn="results/smoke_u5/matrices/$(basename "${outj%.json}").npz";; esac
  if [ -n "$outj" ] && [ -f "$outj" ] && [ -f "$outn" ]; then
    local pvmode="full"; [ "$class" = "SMOKE" ] && pvmode="smoke"
    if validate "$outj" "$outn" "$pvmode" | grep -q VALIDATE-FAIL; then
      mv "$outj" "$outj.INVALID" 2>/dev/null; mv "$outn" "$outn.INVALID" 2>/dev/null
      log "STALE-INVALID ${tag} — quarantined; re-running"
    else
      log "skip ${tag} (exists, validated)"
      [ "$class" = "SMOKE" ] && : > "engine/smoke_u5_${tag}.ok"
      return
    fi
  fi
  local mdir; mdir=$(echo "$cmd" | grep -oE -- '--model [^ ]+' | awk '{print $2}')
  case "$mdir" in data/*|/*)
    if [ -n "$mdir" ] && [ ! -d "$mdir" ]; then log "CONFIG-SKIP ${tag} (no model ${mdir})"; n_skip=$((n_skip+1)); return; fi;;
  esac
  local cap=$(( est * 60 * 3 + 1200 ))
  log "RUN ${tag} [${class}] (est ${est}m, cap ${cap}s, elapsed ${now}m) -> engine/u5_${tag}.log"
  local t rc dt
  t=$(date +%s)
  timeout --signal=TERM --kill-after=60 "${cap}s" bash -c "$cmd" >> "engine/u5_${tag}.log" 2>&1
  rc=$?; dt=$(( $(date +%s) - t ))
  local vmode="full"; [ "$class" = "SMOKE" ] && vmode="smoke"
  if [ "$rc" -eq 0 ] && [ -n "$outj" ] && [ -f "$outj" ] && [ -f "$outn" ]; then
    local v; v=$(validate "$outj" "$outn" "$vmode")
    if echo "$v" | grep -q VALIDATE-FAIL; then
      mv "$outj" "$outj.INVALID" 2>/dev/null; mv "$outn" "$outn.INVALID" 2>/dev/null
      log "FAIL ${tag} (${dt}s) OUTPUT-INVALID: ${v}"; n_fail=$((n_fail+1))
    else
      log "done ${tag} (${dt}s) ${v}"; n_done=$((n_done+1)); wedge_fail=0
      [ "$class" = "SMOKE" ] && : > "engine/smoke_u5_${tag}.ok"
      # thermal sentinel: >1.4x est at full steps = possible 60W SW-wedge (see header note)
      if [ "$class" != "SMOKE" ] && [ "$dt" -gt $(( est * 60 * 14 / 10 )) ]; then
        local pw; pw=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null | head -1)
        log "THERMAL-WATCH ${tag} ran ${dt}s > 1.4x est; power.draw now=${pw:-NA}W (wedge if <100 under load)"
      fi
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

# ---------------------------------------------------------------- Phase 0c: micro-smoke
# delete+zsRE combo has never run (rome+delete proven on cf; zsre proven on rewrite).
run_row SMOKE zsre_del_smoke 4 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome --edit_mode delete --delete_variant refusal $ZS $SMK --lr 0.1 --layer 10 --seed 0 --out results/smoke_u5/zsre_del.json"
heartbeat

# ---------------------------------------------------------------- Block Z: zsRE deletion (new science)
# FIRST among science (u5 review MED-1): the never-run zsre-x-delete combo is the queue's
# highest-novelty cell — front-loaded so a thermal slowdown sacrifices redundant seed-
# hardening at the tail, not the marquee result. Depends only on the Phase-0c smoke marker.
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE u1e0_llama1b_delete_refusal_zsre_L10_s0 30 engine/smoke_u5_zsre_del_smoke.ok "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome --edit_mode delete --delete_variant refusal $ZS $COMMON --lr 0.1 --layer 10 --seed 0 --out results/u1e0_llama1b_delete_refusal_zsre_L10_s0.json"
heartbeat

# ---------------------------------------------------------------- Block V: aniso L14 bank seeds
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE qv_llama1b_rome_cf_L14_s1 30 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome --save_vectors $CF $COMMON --lr 0.1 --layer 14 --seed 1 --out results/qv_llama1b_rome_cf_L14_s1.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE qv_llama1b_rome_cf_L14_s2 30 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome --save_vectors $CF $COMMON --lr 0.1 --layer 14 --seed 2 --out results/qv_llama1b_rome_cf_L14_s2.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE qv_qwen15b_rome_cf_L14_s1 38 - "$ENVP $PY $KG --model data/models/Qwen2.5-1.5B --editor rome --save_vectors $CF $COMMON --lr 0.1 --layer 14 --seed 1 --out results/qv_qwen15b_rome_cf_L14_s1.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE qv_qwen15b_rome_cf_L14_s2 38 - "$ENVP $PY $KG --model data/models/Qwen2.5-1.5B --editor rome --save_vectors $CF $COMMON --lr 0.1 --layer 14 --seed 2 --out results/qv_qwen15b_rome_cf_L14_s2.json"
heartbeat

# ---------------------------------------------------------------- Block U: U1 layer-profile seeds
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE u1e0_llama1b_delete_refusal_L8_s1 28 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome --edit_mode delete --delete_variant refusal $CF $COMMON --lr 0.1 --layer 8 --seed 1 --out results/u1e0_llama1b_delete_refusal_L8_s1.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE u1e0_llama1b_delete_refusal_L8_s2 28 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome --edit_mode delete --delete_variant refusal $CF $COMMON --lr 0.1 --layer 8 --seed 2 --out results/u1e0_llama1b_delete_refusal_L8_s2.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE u1e0_llama1b_delete_refusal_L14_s1 28 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome --edit_mode delete --delete_variant refusal $CF $COMMON --lr 0.1 --layer 14 --seed 1 --out results/u1e0_llama1b_delete_refusal_L14_s1.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE u1e0_llama1b_delete_refusal_L14_s2 28 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome --edit_mode delete --delete_variant refusal $CF $COMMON --lr 0.1 --layer 14 --seed 2 --out results/u1e0_llama1b_delete_refusal_L14_s2.json"
heartbeat

# ---------------------------------------------------------------- Block K: KL-ladder s2
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE gate_llama1b_ftkl003_cf_L8_s2 28 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor ft $CF $COMMON --ft_lr 5e-3 --ft_kl 0.03 --ft_kl_n 5 --layer 8 --seed 2 --out results/gate_llama1b_ftkl003_cf_L8_s2.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE gate_llama1b_ftkl030_cf_L8_s2 28 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor ft $CF $COMMON --ft_lr 5e-3 --ft_kl 0.3 --ft_kl_n 5 --layer 8 --seed 2 --out results/gate_llama1b_ftkl030_cf_L8_s2.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE gate_llama1b_ftkl100_cf_L8_s2 28 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor ft $CF $COMMON --ft_lr 5e-3 --ft_kl 1.0 --ft_kl_n 5 --layer 8 --seed 2 --out results/gate_llama1b_ftkl100_cf_L8_s2.json"
heartbeat

# ---------------------------------------------------------------- Block Q: seq flank seeds
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE seq_llama1b_nr_L8_s1 10 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome --no_restore --recheck_every 10 $CF --n_edits 50 --n_probes 500 --steps 20 --save_matrices --matrix_dir results/matrices --lr 0.1 --layer 8 --seed 1 --out results/seq_llama1b_nr_L8_s1.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE seq_llama1b_nr_L14_s1 10 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome --no_restore --recheck_every 10 $CF --n_edits 50 --n_probes 500 --steps 20 --save_matrices --matrix_dir results/matrices --lr 0.1 --layer 14 --seed 1 --out results/seq_llama1b_nr_L14_s1.json"
heartbeat

# ---------------------------------------------------------------- Block F: FILLER (budget-gated)
# KL-ladder at L12 (dose-response at the law's peak layer; ft@L12 + ftkl@L8 both proven)
[ "$QUEUE_ABORT" -eq 0 ] && run_row FILLER gate_llama1b_ftkl003_cf_L12_s0 30 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor ft $CF $COMMON --ft_lr 5e-3 --ft_kl 0.03 --ft_kl_n 5 --layer 12 --seed 0 --out results/gate_llama1b_ftkl003_cf_L12_s0.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row FILLER gate_llama1b_ftkl010_cf_L12_s0 30 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor ft $CF $COMMON --ft_lr 5e-3 --ft_kl 0.1 --ft_kl_n 5 --layer 12 --seed 0 --out results/gate_llama1b_ftkl010_cf_L12_s0.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row FILLER gate_llama1b_ftkl030_cf_L12_s0 30 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor ft $CF $COMMON --ft_lr 5e-3 --ft_kl 0.3 --ft_kl_n 5 --layer 12 --seed 0 --out results/gate_llama1b_ftkl030_cf_L12_s0.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row FILLER gate_llama1b_ftkl100_cf_L12_s0 30 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor ft $CF $COMMON --ft_lr 5e-3 --ft_kl 1.0 --ft_kl_n 5 --layer 12 --seed 0 --out results/gate_llama1b_ftkl100_cf_L12_s0.json"
# EGL seeds (table currently s0-only)
[ "$QUEUE_ABORT" -eq 0 ] && run_row FILLER egl_llama1b_rome_cf_L12_s1 30 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome --egl $CF $COMMON --lr 0.1 --layer 12 --seed 1 --out results/egl_llama1b_rome_cf_L12_s1.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row FILLER egl_llama1b_memit_cf_L12_s1 35 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor memit --egl $CF $COMMON --lr 0.1 --layer 12 --seed 1 --out results/egl_llama1b_memit_cf_L12_s1.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row FILLER egl_llama1b_alpha_cf_L12_s1 30 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor alpha --egl $CF $COMMON --lr 0.1 --layer 12 --seed 1 --out results/egl_llama1b_alpha_cf_L12_s1.json"
heartbeat

# ---------------------------------------------------------------- Post-run (CPU, ALWAYS)
log "---------------- POST-RUN (CPU) ----------------"
$PY - > results/u5_validation.json 2>>"$LOG" <<'EOF'
import json, glob, os, numpy as np
t0 = float(open('engine/u5_round_start').read().strip())
out = []
for j in sorted(glob.glob('results/*.json')):
    base = os.path.basename(j)[:-5]
    if os.path.getmtime(j) < t0: continue
    if not base.startswith(('u1e0_', 'seq_', 'egl_', 'gate_llama1b_', 'qv_')): continue
    if base.endswith('.egl'): continue
    z = 'results/matrices/' + base + '.npz'
    row = {'json': j, 'npz_found': os.path.exists(z)}
    try:
        d = json.load(open(j)); row['json_ok'] = True; row['esr'] = d.get('edit_success_rate')
    except Exception as e:
        row['json_ok'] = False; row['err'] = str(e)
    out.append(row)
print(json.dumps({'n': len(out), 'rows': out}, indent=1))
EOF
log "post: validation sweep -> results/u5_validation.json"

# vector-bank check for the new aniso seeds
$PY - > results/u5_vector_validation.json 2>>"$LOG" <<'EOF'
import json, glob, numpy as np
out = []
for z in sorted(glob.glob('results/vectors/vectors_qv_*_L14_s[12].npz')):
    a = np.load(z)
    out.append({'npz': z, 'K_shape': list(a['K'].shape) if 'K' in a.files else None,
                'vectors_valid': bool(a['vectors_valid']) if 'vectors_valid' in a.files else None,
                'K_all_finite': bool(np.isfinite(a['K']).all()) if 'K' in a.files else None})
print(json.dumps({'n': len(out), 'rows': out}, indent=1))
EOF
log "post: vector-bank sweep -> results/u5_vector_validation.json"

# per-seed aniso contrasts (s1, s2) — descriptive hardening of the single-seed L14 contrast
for s in 1 2; do
  L="results/vectors/vectors_qv_llama1b_rome_cf_L14_s${s}.npz"
  Q="results/vectors/vectors_qv_qwen15b_rome_cf_L14_s${s}.npz"
  if [ -f "$L" ] && [ -f "$Q" ]; then
    $PY experiments/analyze_aniso.py "$L" "$Q" --out "results/ANISO_analysis_L14_s${s}.json" >> "$LOG" 2>&1 \
      && log "post: ANISO_analysis_L14_s${s} done" || log "post: ANISO_analysis_L14_s${s} FAIL"
  fi
done

# C3 pools: U1 layer profiles now (s0,s1,s2); KL rungs pool available seeds; zsre-delete single
for spec in "C3_u1_blockB_L8_seeds:results/matrices/u1e0_llama1b_delete_refusal_L8_s*.npz" \
            "C3_u1_blockB_L14_seeds:results/matrices/u1e0_llama1b_delete_refusal_L14_s*.npz" \
            "C3_klladder_003_L8_seeds:results/matrices/gate_llama1b_ftkl003_cf_L8_s*.npz" \
            "C3_klladder_030_L8_seeds:results/matrices/gate_llama1b_ftkl030_cf_L8_s*.npz" \
            "C3_klladder_100_L8_seeds:results/matrices/gate_llama1b_ftkl100_cf_L8_s*.npz" \
            "C3_klladder_003_L12:results/matrices/gate_llama1b_ftkl003_cf_L12_s0.npz" \
            "C3_klladder_010_L12:results/matrices/gate_llama1b_ftkl010_cf_L12_s0.npz" \
            "C3_klladder_030_L12:results/matrices/gate_llama1b_ftkl030_cf_L12_s0.npz" \
            "C3_klladder_100_L12:results/matrices/gate_llama1b_ftkl100_cf_L12_s0.npz" \
            "C3_u1_zsre_delete_L10:results/matrices/u1e0_llama1b_delete_refusal_zsre_L10_s0.npz"; do
  outn="${spec%%:*}"; glob="${spec#*:}"
  if compgen -G "$glob" >/dev/null; then
    $PY experiments/analyze_matrices.py $glob --metric logit --known --edit_ok \
      --out "results/${outn}_u5.json" >> "$LOG" 2>&1 && log "post: ${outn}_u5 done" || log "post: ${outn}_u5 FAIL"
  fi
done

# u1_deletion_gate for new deletion cells (insertion ref = matched layer/dataset s0)
for spec in "refusal_L8_s1:results/matrices/u1e0_llama1b_delete_refusal_L8_s1.npz:results/matrices/gate_llama1b_rome_cf_L8_s0.npz" \
            "refusal_L8_s2:results/matrices/u1e0_llama1b_delete_refusal_L8_s2.npz:results/matrices/gate_llama1b_rome_cf_L8_s0.npz" \
            "refusal_L14_s1:results/matrices/u1e0_llama1b_delete_refusal_L14_s1.npz:results/matrices/gate_llama1b_rome_cf_L14_s0.npz" \
            "refusal_L14_s2:results/matrices/u1e0_llama1b_delete_refusal_L14_s2.npz:results/matrices/gate_llama1b_rome_cf_L14_s0.npz" \
            "zsre_refusal_L10_s0:results/matrices/u1e0_llama1b_delete_refusal_zsre_L10_s0.npz:results/matrices/gate_llama1b_rome_zsre_L10_s0.npz"; do
  tagn="${spec%%:*}"; rest="${spec#*:}"; del="${rest%%:*}"; ins="${rest#*:}"
  if [ -f "$del" ] && [ -f "$ins" ]; then
    $PY experiments/u1_deletion_gate.py --del_npz "$del" --ins_npz "$ins" --metric logit \
      --out "results/u1_gate_${tagn}.json" >> "$LOG" 2>&1 && log "post: u1_gate_${tagn} done" || log "post: u1_gate_${tagn} FAIL"
  fi
done

{
  echo "RUN_U5 REPORT $(date '+%F %T')  jobs: ${n_done} done / ${n_fail} fail / ${n_skip} skipped  elapsed $(elapsed_min)m/${BUDGET_MIN}m  abort=${QUEUE_ABORT}"
  grep -E 'RUN |done |FAIL |SKIP|ABORT|PROGRESS|THERMAL' "$LOG" | tail -60
} > engine/run_u5_report.txt
log "================ RUN_U5 COMPLETE (${n_done} done / ${n_fail} fail / ${n_skip} skip) ================"
echo "RUN_U5_DONE" >> "$LOG"
