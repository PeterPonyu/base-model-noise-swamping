#!/bin/bash
# run_gptj.sh — GPT-J-6B (EleutherAI/gpt-j-6b, float16 revision) battery (2026-07-06).
# Template = run_8bcausal.sh (verbatim skeleton for the bf16-equivalence-gated large-model
# pattern), GPTJ-namespaced (own pid/log/markers) EXCEPT the bf16 equivalence gate itself,
# which is DELIBERATELY REUSED verbatim (engine/r3_equiv_bf16.ok / equiv_llama1b_bf16_L12_s0
# .npz) per run_8bcausal.sh's own precedent comment: that gate certifies a fact about the
# killgate CODE/MATH (does bf16 loading preserve the geometry law on a model everyone
# already has fp32 numbers for?), not about any one large model, so every large-model driver
# should look for the SAME marker rather than re-deriving a model-specific copy.
#
# ARCH NOTE (the reason this driver needed new code, not just new rows): GPT-J is
# GPT-NeoX-style (parallel attn+MLP, mlp.fc_in/fc_out) and shares GPT-2's top-level
# model.transformer.h layout but NOT GPT-2's Conv1D MLP — editors/arch_compat.py's
# normalize_arch() previously assumed transformer.h implies GPT-2 Conv1D unconditionally
# and would have thrown AttributeError on GPT-J's mlp.fc_out (no .c_proj attribute).
# EXTENDED 2026-07-06: normalize_arch now branches on which attribute block-0's mlp
# actually has (c_proj -> gpt2 Conv1D->Linear conversion; fc_out -> gptj graft-only, no
# conversion since fc_out is ALREADY nn.Linear) and returns "gptj" as a new arch string.
# killgate_keygeom.py's memit fence (`arch == "gpt2"`) was broadened to
# `arch in ("gpt2", "gptj")` — MEMIT's _hidden_at residual-stream hook needs the real
# decoder-layer Module, which the graft replaces with a SimpleNamespace on BOTH families.
# Verified end-to-end (CPU, synthetic tiny GPTJForCausalLM, no download needed): normalize_
# arch grafts fc_out as down_proj, forward pass is unchanged (equivalence proof
# max|Δlogit|=0), and editors/rome_native.py's _capture_key/find_subject_last_token_index
# resolve through the graft with ZERO code changes — same guarantee GPT-2 already had.
# NOT yet verified against the REAL gpt-j-6b weights: the download was still in progress
# on shared storage as of this driver's authoring (data/models/gpt-j-6b had tokenizer
# files only, no pytorch_model.bin) — see the integrity gate below, which is what actually
# admits science rows once the real weights land.
#
# WEIGHT FORMAT NOTE: gpt-j-6b's `float16` revision predates safetensors — it ships
# pytorch_model.bin, not model.safetensors (data/DOWNLOADS-20260706.md item 4, note 2:
# hf_xet hung on this file; HF_HUB_DISABLE_XET=1 fixed it). experiments/tools/
# integrity_check.py was extended 2026-07-06 to also accept pytorch_model*.bin via a
# meta-device torch.load (parses the zip+pickle container without allocating real tensor
# storage — same "headers only" guarantee as its existing safetensors path); validated
# against a synthetic .bin (pass / wrong-param-count fail / truncated-file fail all
# correct) since the real file isn't complete yet.
#
# LAYER BAND (28 layers; proportional to the llama1b L8/L10/L12/L14-of-16 band, i.e. the
# SAME depth fractions {0.5, 0.625, 0.75, 0.875} the harness already uses as its canonical
# probe band): round(0.5*28)=14, round(0.625*28)=18 (17.5 rounds to nearest-even 18),
# round(0.75*28)=21 (the llama1b-L12-equivalent PEAK), round(0.875*28)=24 (24.5 rounds to
# nearest-even 24) — computed with Python's own round() (banker's rounding), not hand
# arithmetic; verified interactively before writing this file. => L14, L18, L21, L24; peak=L21.
#
# GPU COST — FLAGGED, NO GPT-J PRECEDENT ANYWHERE IN THE HARNESS (guessed, not measured,
# same honesty convention as run_8bcausal.sh's own AlphaEdit-at-8B flag): GPT-J shares
# Llama-3.1-8B's hidden_size (4096, same O(d^2) cost per edit/projector-fit) but has fewer
# layers (28 vs 32) and fewer params (6.05B vs 8.03B) — anchoring off run_r3.sh's MEASURED
# ~100min/row for Llama-3.1-8B ROME at COMMON settings (200 edits/500 probes/20 steps) is
# the only available anchor and is probably a mild overestimate. 12 ROME rows (4 layers x
# 3 seeds) at ~100min + 1 EGL row at ~110min + 1 alpha-holdout row at ~120min + smoke ~15min
# => ~1445 GPU-min (~24h) if run to completion — this will NOT fit one BUDGET_MIN window.
# Ordered peak-layer-first (L21) so a partial/interrupted run keeps the highest-value cells;
# BUDGET_MIN below is a single-session default, override via env for a longer window.
set -u
H=/home/zeyufu/Desktop/idea-feasibility-analysis/edit-harness
cd "$H" || exit 1
PY=/home/zeyufu/miniconda3/envs/dl/bin/python3
LOG=engine/run_gptj.log
BUDGET_MIN=${BUDGET_MIN:-600}
mkdir -p engine results/matrices results/smoke_gptj/matrices
echo $$ > engine/run_gptj.pid
[ -f engine/gptj_round_start ] || stat -c %Y engine/run_gptj.pid > engine/gptj_round_start
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "================ RUN_GPTJ START (pid $$, budget ${BUDGET_MIN}m) ================"

# ---------------------------------------------------------------- Phase 0a: CPU pre-flight
pf_fail=0
pf(){ if eval "$2"; then log "preflight OK: $1"; else log "PREFLIGHT-FAIL: $1"; pf_fail=1; fi; }
pf "python env" "$PY -c 'import torch, numpy' 2>/dev/null"
pf "counterfact.json" "[ -f data/counterfact.json ]"
pf "model_dtype flag" "grep -q -- '--model_dtype' experiments/killgate_keygeom.py"
pf "alpha_proj_source flag" "grep -q -- '--alpha_proj_source' experiments/killgate_keygeom.py"
pf "integrity_check.py" "[ -f experiments/tools/integrity_check.py ]"
pf "integrity_check.py bin support" "grep -q -- 'bin_meta_params' experiments/tools/integrity_check.py"
pf "arch_compat gptj branch" "grep -q -- 'family == \"gptj\"' editors/arch_compat.py"
pf "killgate gptj memit fence" "grep -q -- 'arch in (\"gpt2\", \"gptj\")' experiments/killgate_keygeom.py"
pf "equiv comparator fp32 npz" "[ -f results/matrices/gate_llama1b_rome_cf_L12_s0.npz ]"
pf "disk >=25GB free" "[ \$(df --output=avail -BG /home | tail -1 | tr -dc 0-9) -ge 25 ]"
rm -f engine/smoke_gptj_*.ok
if [ "$pf_fail" -ne 0 ]; then log "ABORT: preflight failed"; exit 3; fi

# ---------------------------------------------------------------- Phase 0a2: re-derive engine/gptj_integrity.ok
# NOT a pf() hard gate: the downloader agent may still be filling data/models/gpt-j-6b
# concurrently (tokenizer files only as of this driver's authoring, no weight file yet) —
# every row below is gated on this marker via run_row's `needs` mechanism, so a not-yet-
# complete download produces clean CONFIG-SKIP log lines, never a crash or a hard abort.
rm -f engine/gptj_integrity.ok
if [ -d data/models/gpt-j-6b ]; then
  $PY experiments/tools/integrity_check.py data/models/gpt-j-6b --expect_params 6.05e9 >> "$LOG" 2>&1 \
    && { : > engine/gptj_integrity.ok; log "integrity OK: gpt-j-6b"; } \
    || log "integrity NOT-READY: gpt-j-6b (download incomplete or corrupt — rows CONFIG-skip)"
else
  log "MODEL-ABSENT: data/models/gpt-j-6b directory not found — rows CONFIG-skip"
fi
# engine/r3_equiv_bf16.ok: SHARED marker (see header) — re-derive only if stale/absent,
# exactly like run_8bcausal.sh's own freshness check (predates killgate_keygeom.py => force
# a real re-run rather than trusting a marker from before today's arch_compat changes).
if [ -f engine/r3_equiv_bf16.ok ] && [ -f results/matrices/equiv_llama1b_bf16_L12_s0.npz ] \
   && [ "$(stat -c %Y results/matrices/equiv_llama1b_bf16_L12_s0.npz)" -ge "$(stat -c %Y experiments/killgate_keygeom.py)" ] \
   && [ "$(stat -c %Y results/matrices/equiv_llama1b_bf16_L12_s0.npz)" -ge "$(stat -c %Y editors/arch_compat.py)" ]; then
  log "engine/r3_equiv_bf16.ok fresh (postdates killgate_keygeom.py and arch_compat.py) — reusing, no GPU spend"
else
  rm -f engine/r3_equiv_bf16.ok
  log "engine/r3_equiv_bf16.ok absent/stale — will re-derive in Phase A below (real GPU row)"
fi

# ---------------------------------------------------------------- Phase 0b: GPU idle gate
DRYRUN=${DRYRUN:-0}
if [ "$DRYRUN" -eq 1 ]; then
  log "DRYRUN=1 — skipping GPU idle gate, printing every run_row call without executing"
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
  log "GPU idle — window opens now"
fi
T0=$(date +%s)

# ---------------------------------------------------------------- helpers (u5/r3/8bcausal template, verbatim)
ENVP="env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1"
KG="experiments/killgate_keygeom.py"
CF="--dataset counterfact --data data/counterfact.json"
COMMON="--n_edits 200 --n_probes 500 --steps 20 --save_matrices --matrix_dir results/matrices"
SMK="--n_edits 4 --n_probes 40 --steps 2 --save_matrices --matrix_dir results/smoke_gptj/matrices"
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
  if [ "$DRYRUN" -eq 1 ]; then
    echo "DRYRUN ${tag} [${class}] est=${est}m needs=${needs} cmd: ${cmd}"
    log "DRYRUN ${tag} [${class}] est=${est}m needs=${needs} cmd: ${cmd}"
    return
  fi
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
  case "$cmd" in *smoke_gptj*) outn="results/smoke_gptj/matrices/$(basename "${outj%.json}").npz";; esac
  if [ -n "$outj" ] && [ -f "$outj" ] && [ -f "$outn" ]; then
    local pvmode="full"; [ "$class" = "SMOKE" ] && pvmode="smoke"
    if validate "$outj" "$outn" "$pvmode" | grep -q VALIDATE-FAIL; then
      mv "$outj" "$outj.INVALID" 2>/dev/null; mv "$outn" "$outn.INVALID" 2>/dev/null
      log "STALE-INVALID ${tag} — quarantined; re-running"
    else
      log "skip ${tag} (exists, validated)"
      [ "$class" = "SMOKE" ] && : > "engine/smoke_gptj_${tag}.ok"
      return
    fi
  fi
  local mdir; mdir=$(echo "$cmd" | grep -oE -- '--model [^ ]+' | awk '{print $2}')
  case "$mdir" in data/*|/*)
    if [ -n "$mdir" ] && [ ! -d "$mdir" ]; then log "CONFIG-SKIP ${tag} (no model ${mdir})"; n_skip=$((n_skip+1)); return; fi;;
  esac
  local cap=$(( est * 60 * 3 + 1200 ))
  log "RUN ${tag} [${class}] (est ${est}m, cap ${cap}s, elapsed ${now}m) -> engine/gptj_${tag}.log"
  local t rc dt
  t=$(date +%s)
  timeout --signal=TERM --kill-after=60 "${cap}s" bash -c "$cmd" >> "engine/gptj_${tag}.log" 2>&1
  rc=$?; dt=$(( $(date +%s) - t ))
  local vmode="full"; [ "$class" = "SMOKE" ] && vmode="smoke"
  if [ "$rc" -eq 0 ] && [ -n "$outj" ] && [ -f "$outj" ] && [ -f "$outn" ]; then
    local v; v=$(validate "$outj" "$outn" "$vmode")
    if echo "$v" | grep -q VALIDATE-FAIL; then
      mv "$outj" "$outj.INVALID" 2>/dev/null; mv "$outn" "$outn.INVALID" 2>/dev/null
      log "FAIL ${tag} (${dt}s) OUTPUT-INVALID: ${v}"; n_fail=$((n_fail+1))
    else
      log "done ${tag} (${dt}s) ${v}"; n_done=$((n_done+1)); wedge_fail=0
      [ "$class" = "SMOKE" ] && : > "engine/smoke_gptj_${tag}.ok"
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
# GPT-J + bf16 + the new arch_compat graft has NEVER run end-to-end on real weights.
run_row SMOKE bf16_gptj 15 engine/gptj_integrity.ok "$ENVP $PY $KG --model data/models/gpt-j-6b --model_dtype bf16 --editor rome $CF $SMK --lr 0.1 --layer 21 --seed 0 --out results/smoke_gptj/bf16_gptj.json"
heartbeat

# ---------------------------------------------------------------- Phase A: bf16 EQUIVALENCE GATE (shared marker)
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE equiv_llama1b_bf16_L12_s0 22 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --model_dtype bf16 --editor rome $CF $COMMON --lr 0.1 --layer 12 --seed 0 --out results/equiv_llama1b_bf16_L12_s0.json"
if [ "$DRYRUN" -ne 1 ] && [ -f results/matrices/equiv_llama1b_bf16_L12_s0.npz ] && [ -f results/matrices/gate_llama1b_rome_cf_L12_s0.npz ]; then
  $PY - >> "$LOG" 2>&1 <<'EOF'
import numpy as np, sys, os
sys.path.insert(0, 'experiments')
from analyze_matrices import within_probe_rhos
def rho(f):
    d = np.load(f); C = d['COS'].astype(float); D = d['damage_logit'].astype(float)
    m = d['edit_ok'].astype(float) > 0; c = d['pre_p'].astype(float) > 0.05
    return float(np.nanmean(within_probe_rhos(C[m][:, c], D[m][:, c])))
r_fp32 = rho('results/matrices/gate_llama1b_rome_cf_L12_s0.npz')
r_bf16 = rho('results/matrices/equiv_llama1b_bf16_L12_s0.npz')
d = abs(r_fp32 - r_bf16)
print(f"[gptj equiv-gate] fp32 rho={r_fp32:+.4f} bf16 rho={r_bf16:+.4f} |drho|={d:.4f} bar=0.02")
if d < 0.02:
    open('engine/r3_equiv_bf16.ok', 'w').close()
    print("[gptj equiv-gate] PASS — GPT-J science admitted")
else:
    print("[gptj equiv-gate] FAIL — bf16 rows stay CONFIG-skipped; investigate before any GPT-J claim")
EOF
fi
heartbeat

# ---------------------------------------------------------------- Block G: GPT-J ROME layer band x 3 seeds
# peak-first (L21 = 0.75 depth, the llama1b-L12 equivalent) so a partial/interrupted
# window keeps the highest-value cells; see header for the layer-band derivation.
NEEDS="engine/gptj_integrity.ok,engine/smoke_gptj_bf16_gptj.ok,engine/r3_equiv_bf16.ok"
for s in 0 1 2; do
  [ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE gate_gptj_rome_cf_L21_s${s} 100 "$NEEDS" "$ENVP $PY $KG --model data/models/gpt-j-6b --model_dtype bf16 --editor rome $CF $COMMON --lr 0.1 --layer 21 --seed ${s} --out results/gate_gptj_rome_cf_L21_s${s}.json"
done
heartbeat
for s in 0 1 2; do
  [ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE gate_gptj_rome_cf_L14_s${s} 100 "$NEEDS" "$ENVP $PY $KG --model data/models/gpt-j-6b --model_dtype bf16 --editor rome $CF $COMMON --lr 0.1 --layer 14 --seed ${s} --out results/gate_gptj_rome_cf_L14_s${s}.json"
done
heartbeat
for s in 0 1 2; do
  [ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE gate_gptj_rome_cf_L18_s${s} 100 "$NEEDS" "$ENVP $PY $KG --model data/models/gpt-j-6b --model_dtype bf16 --editor rome $CF $COMMON --lr 0.1 --layer 18 --seed ${s} --out results/gate_gptj_rome_cf_L18_s${s}.json"
done
heartbeat

# ---------------------------------------------------------------- Block E: EGL row (peak layer, s0)
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE egl_gptj_rome_cf_L21_s0 110 "$NEEDS" "$ENVP $PY $KG --model data/models/gpt-j-6b --model_dtype bf16 --editor rome --egl $CF $COMMON --lr 0.1 --layer 21 --seed 0 --out results/egl_gptj_rome_cf_L21_s0.json"
heartbeat

# ---------------------------------------------------------------- Block C: matched alpha-holdout causal pair (peak layer, s0)
# holdout (not "probes") projector source — the HONEST causal protocol, not the
# by-construction reference (memory: c4-alphaedit-projector-circularity.md).
[ "$QUEUE_ABORT" -eq 0 ] && run_row SMOKE alphaHO_gptj 15 "$NEEDS" "$ENVP $PY $KG --model data/models/gpt-j-6b --model_dtype bf16 --editor alpha $CF $SMK --lr 0.1 --layer 21 --seed 0 --alpha_proj_source holdout --holdout_frac 1.0 --out results/smoke_gptj/alphaHO_gptj.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE g4_gptj_alphaHO_cf_L21_s0 120 "$NEEDS,engine/smoke_gptj_alphaHO_gptj.ok" "$ENVP $PY $KG --model data/models/gpt-j-6b --model_dtype bf16 --editor alpha $CF $COMMON --lr 0.1 --layer 21 --seed 0 --alpha_proj_source holdout --holdout_frac 1.0 --out results/g4_gptj_alphaHO_cf_L21_s0.json"
heartbeat

# ---------------------------------------------------------------- Block F: FILLER (budget-gated, runs LAST —
# science (peak-layer band + EGL + causal pair) must be funded first; L24 is the 4th
# layer-band point, lowest marginal value of the SCIENCE rows above, moved here 2026-07-06
# review so a tight budget window can't spend on it before the causal pair above lands).
for s in 0 1 2; do
  [ "$QUEUE_ABORT" -eq 0 ] && run_row FILLER gate_gptj_rome_cf_L24_s${s} 100 "$NEEDS" "$ENVP $PY $KG --model data/models/gpt-j-6b --model_dtype bf16 --editor rome $CF $COMMON --lr 0.1 --layer 24 --seed ${s} --out results/gate_gptj_rome_cf_L24_s${s}.json"
done
heartbeat

# ---------------------------------------------------------------- Post-run (CPU, ALWAYS)
log "---------------- POST-RUN (CPU) ----------------"
$PY - > results/gptj_validation.json 2>>"$LOG" <<'EOF'
import json, glob, os
t0 = float(open('engine/gptj_round_start').read().strip())
out = []
for j in sorted(glob.glob('results/*gptj*.json')):
    base = os.path.basename(j)[:-5]
    z = 'results/matrices/' + base + '.npz'
    row = {'json': j, 'npz_found': os.path.exists(z), 'touched_this_run': os.path.getmtime(j) >= t0}
    try:
        d = json.load(open(j)); row['json_ok'] = True; row['esr'] = d.get('edit_success_rate')
    except Exception as e:
        row['json_ok'] = False; row['err'] = str(e)
    out.append(row)
print(json.dumps({'n': len(out), 'rows': out}, indent=1))
EOF
log "post: validation sweep -> results/gptj_validation.json"

# per-layer 3-seed pooling (mirrors run_u6/run_mquake_law's C3 pattern)
for spec in "C3_gptj_rome_L14:results/matrices/gate_gptj_rome_cf_L14_s*.npz" \
            "C3_gptj_rome_L18:results/matrices/gate_gptj_rome_cf_L18_s*.npz" \
            "C3_gptj_rome_L21:results/matrices/gate_gptj_rome_cf_L21_s*.npz" \
            "C3_gptj_rome_L24:results/matrices/gate_gptj_rome_cf_L24_s*.npz"; do
  outn="${spec%%:*}"; glob="${spec#*:}"
  if compgen -G "$glob" >/dev/null; then
    $PY experiments/analyze_matrices.py $glob --metric logit --known --edit_ok \
      --out "results/${outn}_gptj.json" >> "$LOG" 2>&1 && log "post: ${outn}_gptj done" || log "post: ${outn}_gptj FAIL"
  fi
done

# S x C law-replication table across the 4-layer band (mirrors run_mquake_law's headline table)
if compgen -G "results/matrices/gate_gptj_rome_cf_L*_s*.npz" >/dev/null; then
  $PY experiments/mechanism_sc_table.py \
    --npz 'results/matrices/gate_gptj_rome_cf_L*_s*.npz' \
    --known --edit_ok \
    --out results/GPTJ_mechanism_sc_table.json >> "$LOG" 2>&1 \
    && log "post: GPTJ_mechanism_sc_table done" || log "post: GPTJ_mechanism_sc_table FAIL"
fi

# causal aggregation at the peak layer (rome vs alpha-holdout), mirrors run_8bcausal's block
if [ -f experiments/aggregate_g4_causal.py ] \
   && compgen -G "results/matrices/g4_gptj_alphaHO_cf_L21_s0.npz" >/dev/null; then
  tmp_out="results/.C4_causal_gptj_table.json.tmp"
  $PY experiments/aggregate_g4_causal.py \
    --rome_glob 'results/matrices/gate_gptj_rome_cf_L{L}_s0.npz' \
    --alpha_glob 'results/matrices/g4_gptj_alphaHO_cf_L{L}_s0.npz' \
    --layers 21 --known --edit_ok --proj_source holdout \
    --out "$tmp_out" >> "$LOG" 2>&1 \
    && mv "$tmp_out" results/C4_causal_gptj_table.json \
    && log "post: C4_causal_gptj_table done (atomic)" \
    || { rm -f "$tmp_out"; log "FAIL aggregate_g4_causal gptj"; }
else
  log "skip C4-gptj (aggregate_g4_causal.py or holdout alpha npz missing)"
fi

{
  echo "RUN_GPTJ REPORT $(date '+%F %T')  jobs: ${n_done} done / ${n_fail} fail / ${n_skip} skipped  elapsed $(elapsed_min)m/${BUDGET_MIN}m  abort=${QUEUE_ABORT}"
  grep -E 'RUN |done |FAIL |SKIP|ABORT|PROGRESS|THERMAL|equiv-gate|integrity|MODEL-ABSENT' "$LOG" | tail -80
} > engine/run_gptj_report.txt
log "================ RUN_GPTJ COMPLETE (${n_done} done / ${n_fail} fail / ${n_skip} skip) ================"
echo "RUN_GPTJ_DONE" >> "$LOG"
