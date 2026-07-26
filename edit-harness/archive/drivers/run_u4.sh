#!/bin/bash
# run_u4.sh — day program 2026-07-04 (~09:30->19:30): ARR-package hardening. Template =
# run_u2.sh (verbatim skeleton), U4-namespaced. Launches AFTER run_u3 (L14 key dumps).
# Blocks, in priority order (science first, budget-gated filler last):
#   A. U1 variant seed-hardening: suppress s1/s2 (the DC-FRAGILE cell — seeds matter most),
#      eos s1 (+s2 filler). Paths GPU-proven at s0 in run_u1 -> no fresh smokes.
#   S. Sequential stream widening: 2 more 50-edit no-restore orderings at L12 (s2/s3).
#      Position-fragility is p=0.167 on s1 — more orderings harden the descriptive claim.
#      Post-run re-runs analyze_sequential on all 4 streams to a NEW file (the reviewed
#      2-stream SEQ_analysis_L12.json is not overwritten).
#   B. Canonical EGL metrics (EOD plan 1.3) on ROME/MEMIT/AlphaEdit at L12 s0 — the ARR
#      reviewer-facing Efficacy/Generality/Locality table. --egl is science-proven on
#      GPT-2-XL (r3 sanity cells); the llama+egl combo never ran -> one micro-smoke gates it.
#   C. P2 first real generation (rebalance G2, reviewed GO 07-04): sample_ckpt.py on
#      Qwen2.5-0.5B, 200 GSM8K x k=8. Special block (dl env, ROOT cwd, own validation).
#      GSM8K verified cached -> 0 download.
#   D. FILLER: MEMIT L10/L14 s0 (editor-spectrum layer profile), KL-ladder s1 seeds,
#      eos s2, seq L8/L14 s0 single streams.
# Measured ests: delete-variant ~26-28m, seq ~7m, memit ~30-39m, ftkl ~27-30m; EGL adds
# neighborhood+paraphrase evals -> conservative 45-50m. P2 0.5B gen: rebalance est 20-40m.
set -u
H=/home/zeyufu/Desktop/idea-feasibility-analysis/edit-harness
cd "$H" || exit 1
PY=/home/zeyufu/miniconda3/envs/dl/bin/python3
LOG=engine/run_u4.log
BUDGET_MIN=${BUDGET_MIN:-600}
mkdir -p engine results/matrices results/smoke_u4/matrices
echo $$ > engine/run_u4.pid
[ -f engine/u4_round_start ] || stat -c %Y engine/run_u4.pid > engine/u4_round_start
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "================ RUN_U4 START (pid $$, budget ${BUDGET_MIN}m) ================"

# ---------------------------------------------------------------- Phase 0a: CPU pre-flight
pf_fail=0
pf(){ if eval "$2"; then log "preflight OK: $1"; else log "PREFLIGHT-FAIL: $1"; pf_fail=1; fi; }
pf "python env" "$PY -c 'import torch, numpy' 2>/dev/null"
pf "counterfact.json" "[ -f data/counterfact.json ]"
pf "edit_mode flag" "grep -q -- '--edit_mode' experiments/killgate_keygeom.py"
pf "no_restore flag" "grep -q -- '--no_restore' experiments/killgate_keygeom.py"
pf "egl flag" "grep -q -- '--egl' experiments/killgate_keygeom.py"
pf "ft_kl flag" "grep -q -- '--ft_kl' experiments/killgate_keygeom.py"
pf "analyze_sequential.py" "[ -f experiments/analyze_sequential.py ]"
pf "u1_deletion_gate.py" "[ -f experiments/u1_deletion_gate.py ]"
pf "model Llama-3.2-1B" "[ -d data/models/Llama-3.2-1B ]"
pf "matched insertion L12 s0" "[ -f results/matrices/gate_llama1b_rome_cf_L12_s0.npz ]"
pf "disk >=20GB free" "[ \$(df --output=avail -BG /home | tail -1 | tr -dc 0-9) -ge 20 ]"
rm -f engine/smoke_u4_*.ok
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

# ---------------------------------------------------------------- helpers (u2 template, verbatim)
ENVP="env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1"
KG="experiments/killgate_keygeom.py"
CF="--dataset counterfact --data data/counterfact.json"
COMMON="--n_edits 200 --n_probes 500 --steps 20 --save_matrices --matrix_dir results/matrices"
SEQC="--n_edits 50 --n_probes 500 --steps 20 --save_matrices --matrix_dir results/matrices"
SMK="--n_edits 4 --n_probes 40 --steps 2 --save_matrices --matrix_dir results/smoke_u4/matrices"
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
        # r4 precedent: a seq npz without the recheck panel is malformed, not a soft negative
        print("VALIDATE-FAIL seq npz missing prior_eff"); sys.exit(1)
    if is_delete or is_seq:
        # delete: esr = 2x-suppression; seq: stream esr drifts with cumulative interference.
        # Low/zero is a legitimate NEGATIVE finding in both, not breakage — warn, never fail.
        if esr is not None and esr < 0.9: print(f"VALIDATE-NOTE soft-esr mode rate={esr}")
    elif "ftkl" in os.path.basename(j):
        # KL-regularized FT: heavy KL legitimately drives esr toward 0 (u2 review MED-1).
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
  case "$cmd" in *smoke_u4*) outn="results/smoke_u4/matrices/$(basename "${outj%.json}").npz";; esac
  if [ -n "$outj" ] && [ -f "$outj" ] && [ -f "$outn" ]; then
    local pvmode="full"; [ "$class" = "SMOKE" ] && pvmode="smoke"
    if validate "$outj" "$outn" "$pvmode" | grep -q VALIDATE-FAIL; then
      mv "$outj" "$outj.INVALID" 2>/dev/null; mv "$outn" "$outn.INVALID" 2>/dev/null
      log "STALE-INVALID ${tag} — quarantined; re-running"
    else
      log "skip ${tag} (exists, validated)"
      [ "$class" = "SMOKE" ] && : > "engine/smoke_u4_${tag}.ok"
      return
    fi
  fi
  local mdir; mdir=$(echo "$cmd" | grep -oE -- '--model [^ ]+' | awk '{print $2}')
  case "$mdir" in data/*|/*)
    if [ -n "$mdir" ] && [ ! -d "$mdir" ]; then log "CONFIG-SKIP ${tag} (no model ${mdir})"; n_skip=$((n_skip+1)); return; fi;;
  esac
  local cap=$(( est * 60 * 3 + 1200 ))
  log "RUN ${tag} [${class}] (est ${est}m, cap ${cap}s, elapsed ${now}m) -> engine/u4_${tag}.log"
  local t rc dt
  t=$(date +%s)
  timeout --signal=TERM --kill-after=60 "${cap}s" bash -c "$cmd" >> "engine/u4_${tag}.log" 2>&1
  rc=$?; dt=$(( $(date +%s) - t ))
  local vmode="full"; [ "$class" = "SMOKE" ] && vmode="smoke"
  if [ "$rc" -eq 0 ] && [ -n "$outj" ] && [ -f "$outj" ] && [ -f "$outn" ]; then
    local v; v=$(validate "$outj" "$outn" "$vmode")
    if echo "$v" | grep -q VALIDATE-FAIL; then
      mv "$outj" "$outj.INVALID" 2>/dev/null; mv "$outn" "$outn.INVALID" 2>/dev/null
      log "FAIL ${tag} (${dt}s) OUTPUT-INVALID: ${v}"; n_fail=$((n_fail+1))
    else
      log "done ${tag} (${dt}s) ${v}"; n_done=$((n_done+1)); wedge_fail=0
      [ "$class" = "SMOKE" ] && : > "engine/smoke_u4_${tag}.ok"
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
# llama + --egl combo has never run (EGL science-proven only on GPT-2-XL sanity cells).
# One smoke per Block-B editor (u4 review LOW-2): memit+egl and alpha+egl are ALSO first-ever.
run_row SMOKE egl_smoke 4 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome --egl $CF $SMK --lr 0.1 --layer 12 --seed 0 --out results/smoke_u4/egl_smoke.json"
run_row SMOKE egl_memit_smoke 4 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor memit --egl $CF $SMK --lr 0.1 --layer 12 --seed 0 --out results/smoke_u4/egl_memit_smoke.json"
run_row SMOKE egl_alpha_smoke 4 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor alpha --egl $CF $SMK --lr 0.1 --layer 12 --seed 0 --out results/smoke_u4/egl_alpha_smoke.json"
heartbeat

# ---------------------------------------------------------------- Block A: U1 variant seed-hardening
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE u1e0_llama1b_delete_suppress_L12_s1 28 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome --edit_mode delete --delete_variant suppress $CF $COMMON --lr 0.1 --layer 12 --seed 1 --out results/u1e0_llama1b_delete_suppress_L12_s1.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE u1e0_llama1b_delete_suppress_L12_s2 28 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome --edit_mode delete --delete_variant suppress $CF $COMMON --lr 0.1 --layer 12 --seed 2 --out results/u1e0_llama1b_delete_suppress_L12_s2.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE u1e0_llama1b_delete_eos_L12_s1 28 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome --edit_mode delete --delete_variant eos $CF $COMMON --lr 0.1 --layer 12 --seed 1 --out results/u1e0_llama1b_delete_eos_L12_s1.json"
heartbeat

# ---------------------------------------------------------------- Block S: sequential stream widening
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE seq_llama1b_nr_L12_s2 10 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome --no_restore --recheck_every 10 $CF $SEQC --lr 0.1 --layer 12 --seed 2 --out results/seq_llama1b_nr_L12_s2.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE seq_llama1b_nr_L12_s3 10 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome --no_restore --recheck_every 10 $CF $SEQC --lr 0.1 --layer 12 --seed 3 --out results/seq_llama1b_nr_L12_s3.json"
heartbeat

# ---------------------------------------------------------------- Block B: canonical EGL (ARR table)
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE egl_llama1b_rome_cf_L12_s0 45 engine/smoke_u4_egl_smoke.ok "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome --egl $CF $COMMON --lr 0.1 --layer 12 --seed 0 --out results/egl_llama1b_rome_cf_L12_s0.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE egl_llama1b_memit_cf_L12_s0 50 engine/smoke_u4_egl_memit_smoke.ok "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor memit --egl $CF $COMMON --lr 0.1 --layer 12 --seed 0 --out results/egl_llama1b_memit_cf_L12_s0.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE egl_llama1b_alpha_cf_L12_s0 50 engine/smoke_u4_egl_alpha_smoke.ok "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor alpha --egl $CF $COMMON --lr 0.1 --layer 12 --seed 0 --out results/egl_llama1b_alpha_cf_L12_s0.json"
heartbeat

# ---------------------------------------------------------------- Block C: P2 first real generation (special)
# Runs from workspace ROOT. env = dl, NOT the queue-job's "dl-rl" — dl-rl was NEVER CREATED
# (u4 review CRITICAL-1; dl verified to carry torch/transformers/datasets). Offline pins per
# review LOW-1 (GSM8K cache verified). P2 checks are BLOCK-LOCAL (review HIGH-1): a P2
# misconfig skips this block only, never the ARR-core blocks. bf16-note: 0.5B fits fp32;
# the dtype override was stamped only into the >=3B jobs.
# Own validation: output JSON must parse with 200 problems x 8 samples.
p2_run(){
  local est=35 tag=p2_gen_qwen05b
  local now; now=$(elapsed_min)
  if [ $(( now + est + 2 )) -gt "$BUDGET_MIN" ]; then log "BUDGET-SKIP ${tag}"; n_skip=$((n_skip+1)); return; fi
  for prereq in "../branches/p2_prerl_diag/sample_ckpt.py" "data/models/Qwen2.5-0.5B" "/home/zeyufu/miniconda3/envs/dl"; do
    if [ ! -e "$prereq" ]; then log "CONFIG-SKIP ${tag} (missing ${prereq})"; n_skip=$((n_skip+1)); return; fi
  done
  local outj=../branches/p2_prerl_diag/samples/Qwen2.5-0.5B.json
  if [ -f "$outj" ]; then log "skip ${tag} (exists)"; return; fi
  local cap=$(( est * 60 * 3 + 1200 ))
  log "RUN ${tag} [SCIENCE-P2] (est ${est}m, cap ${cap}s, elapsed ${now}m) -> engine/u4_${tag}.log"
  local t rc; t=$(date +%s)
  timeout --signal=TERM --kill-after=60 "${cap}s" bash -c "cd .. && env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 conda run -n dl python3 branches/p2_prerl_diag/sample_ckpt.py --model edit-harness/data/models/Qwen2.5-0.5B --dataset openai/gsm8k --config main --split test --n-problems 200 --k 8 --temperature 0.9 --top-p 1.0 --max-new-tokens 640 --seed 0 --out branches/p2_prerl_diag/samples/Qwen2.5-0.5B.json" >> "engine/u4_${tag}.log" 2>&1
  rc=$?; local dt=$(( $(date +%s) - t ))
  local v
  v=$($PY - "$outj" <<'EOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    probs = d.get("problems", [])
    ks = [len(p.get("samples", [])) for p in probs]
    if len(probs) == 200 and all(k == 8 for k in ks): print("VALIDATE-OK")
    else: print(f"VALIDATE-FAIL n_problems={len(probs)} k_min={min(ks) if ks else 0}")
except Exception as e:
    print(f"VALIDATE-FAIL {e}")
EOF
)
  if [ "$rc" -eq 0 ] && echo "$v" | grep -q VALIDATE-OK; then
    log "done ${tag} (${dt}s) ${v}"; n_done=$((n_done+1)); wedge_fail=0
  else
    mv "$outj" "$outj.INVALID" 2>/dev/null
    n_fail=$((n_fail+1)); log "FAIL ${tag} (rc ${rc}, ${dt}s) ${v}"
  fi
}
[ "$QUEUE_ABORT" -eq 0 ] && p2_run
heartbeat

# ---------------------------------------------------------------- Block D: FILLER (budget-gated)
[ "$QUEUE_ABORT" -eq 0 ] && run_row FILLER gate_llama1b_memit_cf_L10_s0 32 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor memit $CF $COMMON --lr 0.1 --layer 10 --seed 0 --out results/gate_llama1b_memit_cf_L10_s0.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row FILLER gate_llama1b_memit_cf_L14_s0 32 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor memit $CF $COMMON --lr 0.1 --layer 14 --seed 0 --out results/gate_llama1b_memit_cf_L14_s0.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row FILLER u1e0_llama1b_delete_eos_L12_s2 28 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome --edit_mode delete --delete_variant eos $CF $COMMON --lr 0.1 --layer 12 --seed 2 --out results/u1e0_llama1b_delete_eos_L12_s2.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row FILLER gate_llama1b_ftkl003_cf_L8_s1 28 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor ft $CF $COMMON --ft_lr 5e-3 --ft_kl 0.03 --ft_kl_n 5 --layer 8 --seed 1 --out results/gate_llama1b_ftkl003_cf_L8_s1.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row FILLER gate_llama1b_ftkl030_cf_L8_s1 28 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor ft $CF $COMMON --ft_lr 5e-3 --ft_kl 0.3 --ft_kl_n 5 --layer 8 --seed 1 --out results/gate_llama1b_ftkl030_cf_L8_s1.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row FILLER gate_llama1b_ftkl100_cf_L8_s1 28 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor ft $CF $COMMON --ft_lr 5e-3 --ft_kl 1.0 --ft_kl_n 5 --layer 8 --seed 1 --out results/gate_llama1b_ftkl100_cf_L8_s1.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row FILLER seq_llama1b_nr_L8_s0 10 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome --no_restore --recheck_every 10 $CF $SEQC --lr 0.1 --layer 8 --seed 0 --out results/seq_llama1b_nr_L8_s0.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row FILLER seq_llama1b_nr_L14_s0 10 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome --no_restore --recheck_every 10 $CF $SEQC --lr 0.1 --layer 14 --seed 0 --out results/seq_llama1b_nr_L14_s0.json"
heartbeat

# ---------------------------------------------------------------- Post-run (CPU, ALWAYS)
log "---------------- POST-RUN (CPU) ----------------"
$PY - > results/u4_validation.json 2>>"$LOG" <<'EOF'
import json, glob, os, numpy as np
t0 = float(open('engine/u4_round_start').read().strip())
out = []
for j in sorted(glob.glob('results/*.json')):
    base = os.path.basename(j)[:-5]
    if os.path.getmtime(j) < t0: continue
    if not base.startswith(('u1e0_', 'seq_', 'egl_', 'gate_llama1b_')): continue
    if base.endswith('.egl'): continue  # EGL sidecar, not a cell (u4 review cosmetic)
    z = 'results/matrices/' + base + '.npz'
    row = {'json': j, 'npz_found': os.path.exists(z)}
    try:
        d = json.load(open(j)); row['json_ok'] = True; row['esr'] = d.get('edit_success_rate')
        row['egl_present'] = any(k.startswith('egl') for k in d)
    except Exception as e:
        row['json_ok'] = False; row['err'] = str(e)
    out.append(row)
print(json.dumps({'n': len(out), 'rows': out}, indent=1))
EOF
log "post: validation sweep -> results/u4_validation.json"

# C3 groups (per-variant seed pools; per-cell for new layers)
for spec in "C3_u1_blockC_suppress_seeds:results/matrices/u1e0_llama1b_delete_suppress_L12_s*.npz" \
            "C3_u1_blockC_eos_seeds:results/matrices/u1e0_llama1b_delete_eos_L12_s*.npz" \
            "C3_memit_L10:results/matrices/gate_llama1b_memit_cf_L10_s0.npz" \
            "C3_memit_L14:results/matrices/gate_llama1b_memit_cf_L14_s0.npz" \
            "C3_klladder_003_L8_seeds:results/matrices/gate_llama1b_ftkl003_cf_L8_s*.npz" \
            "C3_klladder_030_L8_seeds:results/matrices/gate_llama1b_ftkl030_cf_L8_s*.npz" \
            "C3_klladder_100_L8_seeds:results/matrices/gate_llama1b_ftkl100_cf_L8_s*.npz"; do
  outn="${spec%%:*}"; glob="${spec#*:}"
  if compgen -G "$glob" >/dev/null; then
    $PY experiments/analyze_matrices.py $glob --metric logit --known --edit_ok \
      --out "results/${outn}_u4.json" >> "$LOG" 2>&1 && log "post: ${outn}_u4 done" || log "post: ${outn}_u4 FAIL"
  fi
done

# u1_deletion_gate for the new variant seeds (insertion ref = matched rome L12 s0, run_u1 precedent)
for spec in "suppress_L12_s1:results/matrices/u1e0_llama1b_delete_suppress_L12_s1.npz" \
            "suppress_L12_s2:results/matrices/u1e0_llama1b_delete_suppress_L12_s2.npz" \
            "eos_L12_s1:results/matrices/u1e0_llama1b_delete_eos_L12_s1.npz" \
            "eos_L12_s2:results/matrices/u1e0_llama1b_delete_eos_L12_s2.npz"; do
  tagn="${spec%%:*}"; del="${spec#*:}"
  if [ -f "$del" ]; then
    $PY experiments/u1_deletion_gate.py --del_npz "$del" --ins_npz results/matrices/gate_llama1b_rome_cf_L12_s0.npz \
      --metric logit --out "results/u1_gate_${tagn}.json" >> "$LOG" 2>&1 \
      && log "post: u1_gate_${tagn} done" || log "post: u1_gate_${tagn} FAIL"
  fi
done

# 4-stream sequential re-analysis -> NEW file (reviewed 2-stream JSON untouched)
if compgen -G "results/matrices/seq_llama1b_nr_L12_s2.npz" >/dev/null; then
  $PY experiments/analyze_sequential.py results/matrices/seq_llama1b_nr_L12_s*.npz \
    --out results/SEQ_analysis_L12_4stream.json >> "$LOG" 2>&1 \
    && log "post: SEQ_analysis_L12_4stream done" || log "post: SEQ_analysis_L12_4stream FAIL"
fi

{
  echo "RUN_U4 REPORT $(date '+%F %T')  jobs: ${n_done} done / ${n_fail} fail / ${n_skip} skipped  elapsed $(elapsed_min)m/${BUDGET_MIN}m  abort=${QUEUE_ABORT}"
  grep -E 'RUN |done |FAIL |SKIP|ABORT|PROGRESS' "$LOG" | tail -60
} > engine/run_u4_report.txt
log "================ RUN_U4 COMPLETE (${n_done} done / ${n_fail} fail / ${n_skip} skip) ================"
echo "RUN_U4_DONE" >> "$LOG"
