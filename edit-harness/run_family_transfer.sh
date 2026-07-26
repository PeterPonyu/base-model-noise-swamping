#!/bin/bash
# run_family_transfer.sh — Track 1 of the 2026-07-11 cloud extension wave: does the
# signed key-geometry->damage law + "S x C beats raw key-cos at all layers" (memory:
# sxc-normalization-fix-20260706.md) TRANSFER to the modern 7-9B instruction-tuned tier?
# Local zoo tops out at Llama-3.1-8B base + GPT-J-6B (ext-scout finding, 2026-07-11) — the
# entire 7-9B family-transfer + instruct-at-8B-scale question is CLOUD-ONLY. Template =
# run_pythia.sh (model_battery skeleton) + run_8bcausal.sh/run_neox20b.sh (bf16 equiv-gate,
# validate(), run_row() with npz validation) — all 4 models here are NATIVE architectures
# per editors/arch_compat.py's structural check (`hasattr(model.model, "layers")` — true
# for Llama/Mistral/Qwen2/Gemma2 alike, verified against the already-working local
# gemma-2-2b precedent), so unlike run_gptj.sh/run_neox20b.sh NO graft/TP plumbing is
# needed here — every row is a single-card bf16 native edit, same shape as run_8bcausal.sh.
#
# MODELS + LAYER BANDS (depth fractions {0.5, 0.75} — the same two points run_8bcausal.sh
# uses for Llama-3.1-8B/32L, L16+L24 — sparse by design: a 4-layer band x 3 seeds at 7-9B
# scale would blow the wave's cost ceiling (see cloud/EXTENSION-WAVE-RUNBOOK.md's estimate
# table); depth fractions computed with Python's round() (banker's rounding), same
# convention as run_neox20b.sh's header):
#   mistral7b   mistralai/Mistral-7B-v0.3            32L hidden4096 -> L16 (0.50), L24 (0.75)
#   qwen7b      Qwen/Qwen2.5-7B                      28L hidden3584 -> L14 (0.50), L21 (0.75)
#   gemma9b     unsloth/gemma-2-9b (ungated mirror)  42L hidden3584 -> L21 (0.50), L32 (0.75,
#               round(31.5)->32 nearest-even)
#   llama8binst unsloth/Meta-Llama-3.1-8B-Instruct    32L hidden4096 -> L16 (0.50), L24 (0.75)
#               (paired with the ALREADY-LOCAL Llama-3.1-8B base at the SAME L16/L24 —
#               base-vs-instruct-at-8B-scale is a clean comparison, mirrors run_instruct.sh's
#               1B-scale pairing)
# EXPECT_PARAMS values below are computed from each repo's published config.json (verified
# live 2026-07-11 via a config.json fetch for all 4 — see cloud/EXTENSION-WAVE-RUNBOOK.md)
# or, for llama8binst, copied from this repo's OWN already-established Llama-3.1-8B base
# value (identical architecture) — NOT measured against real safetensors headers on this
# box; integrity_check.py's 1% band is the real arbiter, this is the same "guessed, not
# measured" honesty convention as every other driver's ASSUMPTION FLAGGED headers.
#
# CARD SHARDING (see cloud/run_extension_wave.sh): FAMILY_MODELS env var subsets which
# model tags this invocation runs — e.g. FAMILY_MODELS="mistral7b gemma9b" for one card,
# "qwen7b llama8binst" for the other. Defaults to all 4 (single-card / local testing).
#
# ROW ORDERING — BREADTH-FIRST ACROSS MODELS, not depth-first within one model: Phase 1
# runs L_peak s0 ROME for every selected model before Phase 2 touches L_peak s0 causal for
# any of them, etc. This answers "does the law transfer AT ALL to 7-9B" with the fewest
# GPU-minutes if the wave gets cut short — the same peak-first-cell-survives-a-partial-
# window philosophy as run_neox20b.sh's Block F, just applied across models instead of
# across layers. BUDGET_MIN (run_row's own budget-skip) is what actually decides how deep
# into Phase 4-6 (seeds s1/s2) any given launch gets — nothing here assumes full funding.
set -u
H="$(cd "$(dirname "$0")" && pwd)"
cd "$H" || exit 1
PY=/home/zeyufu/miniconda3/envs/dl/bin/python3
PY="${CLOUD_PY:-$PY}"
LOG=engine/run_family_transfer.log
BUDGET_MIN=${BUDGET_MIN:-650}
FAMILY_MODELS=${FAMILY_MODELS:-"mistral7b qwen7b gemma9b llama8binst"}
# EQUIV_GATE_ONLY=1: derive engine/r3_equiv_bf16.ok (Phase A below) then exit, before any
# Track-1 row. Used by cloud/run_extension_wave.sh's launcher to derive this SHARED,
# model-independent marker on ONE card before the card0/card1 fan-out — both cards
# otherwise run this same script's Phase A concurrently against the SAME fixed --out
# path (results/equiv_llama1b_bf16_L12_s0.json), a TOCTOU + concurrent-npz-write race
# (see run_extension_wave.sh's phase0_equiv_gate for the full rationale). Default 0 —
# standalone invocations (and anyone who already ran run_8bcausal.sh/run_neox20b.sh)
# are unaffected.
EQUIV_GATE_ONLY=${EQUIV_GATE_ONLY:-0}
mkdir -p engine results/matrices results/smoke_family/matrices
echo $$ > engine/run_family_transfer.pid
[ -f engine/family_round_start ] || stat -c %Y engine/run_family_transfer.pid > engine/family_round_start
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "================ RUN_FAMILY_TRANSFER START (pid $$, budget ${BUDGET_MIN}m, models=[${FAMILY_MODELS}]) ================"

# ---------------------------------------------------------------- Phase 0a: CPU pre-flight
pf_fail=0
pf(){ if eval "$2"; then log "preflight OK: $1"; else log "PREFLIGHT-FAIL: $1"; pf_fail=1; fi; }
pf "python env" "$PY -c 'import torch, numpy' 2>/dev/null"
pf "counterfact.json" "[ -f data/counterfact.json ]"
pf "killgate_keygeom.py" "[ -f experiments/killgate_keygeom.py ]"
pf "model_dtype flag" "grep -q -- '--model_dtype' experiments/killgate_keygeom.py"
pf "alpha_proj_source flag" "grep -q -- '--alpha_proj_source' experiments/killgate_keygeom.py"
pf "integrity_check.py" "[ -f experiments/tools/integrity_check.py ]"
pf "mechanism_sc_table.py" "[ -f experiments/mechanism_sc_table.py ]"
pf "aggregate_g4_causal.py" "[ -f experiments/aggregate_g4_causal.py ]"
pf "equiv comparator fp32 npz" "[ -f results/matrices/gate_llama1b_rome_cf_L12_s0.npz ]"
# EQUIV_GATE_ONLY runs only the 1B equiv-gate row (Phase A) — the 4-model disk-space
# requirement below doesn't apply to it and could false-abort a box still mid-download.
[ "$EQUIV_GATE_ONLY" = "1" ] || pf "disk >=70GB free (4 models bf16-only, ~64GB combined)" "[ \$(df --output=avail -BG /home | tail -1 | tr -dc 0-9) -ge 70 ]"
rm -f engine/smoke_family_*.ok
if [ "$pf_fail" -ne 0 ]; then log "ABORT: CPU preflight failed"; exit 3; fi

# ---------------------------------------------------------------- Phase 0a2: re-derive engine/r3_equiv_bf16.ok (SHARED marker)
# Same block as run_8bcausal.sh/run_neox20b.sh — reused verbatim so a fresh box that has
# already run either of those drivers spends zero extra GPU time here.
if [ -f engine/r3_equiv_bf16.ok ] && [ -f results/matrices/equiv_llama1b_bf16_L12_s0.npz ] \
   && [ "$(stat -c %Y results/matrices/equiv_llama1b_bf16_L12_s0.npz)" -ge "$(stat -c %Y experiments/killgate_keygeom.py)" ] \
   && [ "$(stat -c %Y results/matrices/equiv_llama1b_bf16_L12_s0.npz)" -ge "$(stat -c %Y editors/arch_compat.py)" ]; then
  log "engine/r3_equiv_bf16.ok fresh — reusing, no GPU spend"
else
  rm -f engine/r3_equiv_bf16.ok
  log "engine/r3_equiv_bf16.ok absent/stale — will re-derive in Phase A below (real GPU row)"
fi

# ---------------------------------------------------------------- Phase 0a3: per-model integrity re-derivation (soft gate)
# NOT a pf() hard abort — models may simply not be downloaded yet on a fresh box (see
# cloud/dl_extension_models.py, ask-first). Mirrors run_instruct.sh's Phase 0a2 pattern:
# missing/incomplete model -> loud log, every row for that tag CONFIG-skips cleanly via
# run_row's own `needs` marker check, the driver still completes normally.
model_spec(){   # model_spec <tag> -> "mdir expect_params L_peak L_mid est_rome est_causal"
  case "$1" in
    mistral7b)   echo "data/models/Mistral-7B-v0.3 7.248e9 24 16 100 120" ;;
    qwen7b)      echo "data/models/Qwen2.5-7B 7.61e9 21 14 95 115" ;;
    gemma9b)     echo "data/models/gemma-2-9b 9.2422e9 32 21 130 155" ;;
    llama8binst) echo "data/models/Llama-3.1-8B-Instruct 8.03e9 24 16 100 120" ;;
    *) return 1 ;;
  esac
}
ALL_TAGS="mistral7b qwen7b gemma9b llama8binst"
for tag in $ALL_TAGS; do
  spec=$(model_spec "$tag"); read -r mdir expparams _ <<< "$spec"
  rm -f "engine/family_${tag}_integrity.ok"
  case " $FAMILY_MODELS " in
    *" $tag "*) : ;;
    *) log "SKIP-CARD ${tag}: not in FAMILY_MODELS for this invocation"; continue ;;
  esac
  if [ -d "$mdir" ]; then
    $PY experiments/tools/integrity_check.py "$mdir" --expect_params "$expparams" >> "$LOG" 2>&1 \
      && { : > "engine/family_${tag}_integrity.ok"; log "integrity OK: ${tag}"; } \
      || log "integrity NOT-READY: ${tag} (${mdir} incomplete/corrupt — rows CONFIG-skip)"
  else
    log "MODEL-ABSENT: ${tag} (${mdir}) — every row for this tag CONFIG-skips cleanly (cloud/dl_extension_models.py not yet run, or still in flight?)"
  fi
done

# ---------------------------------------------------------------- Phase 0b: GPU idle gate
DRYRUN=${DRYRUN:-0}
if [ "$DRYRUN" -eq 1 ]; then
  log "DRYRUN=1 — skipping GPU idle gate, printing every run_row call without executing"
else
  source "$H/cloud/gpu_idle_lib.sh"
  idle_gate_wait || { log "ABORT: idle gate failed"; exit 2; }
fi
T0=$(date +%s)

# ---------------------------------------------------------------- helpers (8bcausal/neox20b template, verbatim)
ENVP="env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1"
KG="experiments/killgate_keygeom.py"
CF="--dataset counterfact --data data/counterfact.json"
COMMON="--n_edits 200 --n_probes 500 --steps 20 --save_matrices --matrix_dir results/matrices"
SMK="--n_edits 4 --n_probes 40 --steps 2 --save_matrices --matrix_dir results/smoke_family/matrices"
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
  case "$cmd" in *smoke_family*) outn="results/smoke_family/matrices/$(basename "${outj%.json}").npz";; esac
  if [ -n "$outj" ] && [ -f "$outj" ] && [ -f "$outn" ]; then
    local pvmode="full"; [ "$class" = "SMOKE" ] && pvmode="smoke"
    if validate "$outj" "$outn" "$pvmode" | grep -q VALIDATE-FAIL; then
      mv "$outj" "$outj.INVALID" 2>/dev/null; mv "$outn" "$outn.INVALID" 2>/dev/null
      log "STALE-INVALID ${tag} — quarantined; re-running"
    else
      log "skip ${tag} (exists, validated)"
      [ "$class" = "SMOKE" ] && : > "engine/smoke_family_${tag}.ok"
      return
    fi
  fi
  local mdir; mdir=$(echo "$cmd" | grep -oE -- '--model [^ ]+' | awk '{print $2}')
  case "$mdir" in data/*|/*)
    if [ -n "$mdir" ] && [ ! -d "$mdir" ]; then log "CONFIG-SKIP ${tag} (no model ${mdir})"; n_skip=$((n_skip+1)); return; fi;;
  esac
  local cap=$(( est * 60 * 3 + 1200 ))
  log "RUN ${tag} [${class}] (est ${est}m, cap ${cap}s, elapsed ${now}m) -> engine/family_${tag}.log"
  local t rc dt
  t=$(date +%s)
  timeout --signal=TERM --kill-after=60 "${cap}s" bash -c "$cmd" >> "engine/family_${tag}.log" 2>&1 </dev/null
  rc=$?; dt=$(( $(date +%s) - t ))
  local vmode="full"; [ "$class" = "SMOKE" ] && vmode="smoke"
  if [ "$rc" -eq 0 ] && [ -n "$outj" ] && [ -f "$outj" ] && [ -f "$outn" ]; then
    local v; v=$(validate "$outj" "$outn" "$vmode")
    if echo "$v" | grep -q VALIDATE-FAIL; then
      mv "$outj" "$outj.INVALID" 2>/dev/null; mv "$outn" "$outn.INVALID" 2>/dev/null
      log "FAIL ${tag} (${dt}s) OUTPUT-INVALID: ${v}"; n_fail=$((n_fail+1))
    else
      log "done ${tag} (${dt}s) ${v}"; n_done=$((n_done+1)); wedge_fail=0
      [ "$class" = "SMOKE" ] && : > "engine/smoke_family_${tag}.ok"
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

# ---------------------------------------------------------------- Phase A: bf16 equivalence gate (re-derive if needed)
if [ ! -f engine/r3_equiv_bf16.ok ]; then
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
print(f"[family equiv-gate] fp32 rho={r_fp32:+.4f} bf16 rho={r_bf16:+.4f} |drho|={d:.4f} bar=0.02")
if d < 0.02:
    open('engine/r3_equiv_bf16.ok', 'w').close()
    print("[family equiv-gate] PASS — 7-9B bf16 science admitted")
else:
    print("[family equiv-gate] FAIL — bf16 rows stay CONFIG-skipped; investigate before any family-transfer claim")
EOF
  fi
fi
heartbeat

if [ "$EQUIV_GATE_ONLY" = "1" ]; then
  if [ -f engine/r3_equiv_bf16.ok ]; then
    log "EQUIV_GATE_ONLY=1 — marker present, exiting before Track-1 rows (hoisted single-card Phase-0, see cloud/run_extension_wave.sh)"
  else
    log "EQUIV_GATE_ONLY=1 — marker still ABSENT after Phase A (gate FAILED or DRYRUN) — exiting anyway; fan-out cards will CONFIG-skip bf16 rows cleanly"
  fi
  echo "RUN_FAMILY_TRANSFER_EQUIV_GATE_ONLY_DONE"
  exit 0
fi

# ---------------------------------------------------------------- per-model smoke (cheap insurance before a full COMMON row)
for tag in $ALL_TAGS; do
  case " $FAMILY_MODELS " in *" $tag "*) : ;; *) continue ;; esac
  spec=$(model_spec "$tag"); read -r mdir _ Lpeak Lmid _ <<< "$spec"
  need="engine/family_${tag}_integrity.ok,engine/r3_equiv_bf16.ok"
  [ "$QUEUE_ABORT" -eq 0 ] && run_row SMOKE bf16_${tag} 5 "$need" "$ENVP $PY $KG --model $mdir --model_dtype bf16 --editor rome $CF $SMK --lr 0.1 --layer $Lpeak --seed 0 --out results/smoke_family/bf16_${tag}.json"
done
heartbeat

# ---------------------------------------------------------------- Phase 1: peak layer (0.75 depth) s0 ROME, breadth-first across models
for tag in $ALL_TAGS; do
  case " $FAMILY_MODELS " in *" $tag "*) : ;; *) continue ;; esac
  spec=$(model_spec "$tag"); read -r mdir _ Lpeak _ estrome _ <<< "$spec"
  need="engine/family_${tag}_integrity.ok,engine/r3_equiv_bf16.ok,engine/smoke_family_bf16_${tag}.ok"
  [ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE gate_${tag}_rome_cf_L${Lpeak}_s0 "$estrome" "$need" "$ENVP $PY $KG --model $mdir --model_dtype bf16 --editor rome $CF $COMMON --lr 0.1 --layer $Lpeak --seed 0 --out results/gate_${tag}_rome_cf_L${Lpeak}_s0.json"
done
heartbeat

# ---------------------------------------------------------------- Phase 2: peak layer s0 causal (AlphaEdit, holdout projector), breadth-first
for tag in $ALL_TAGS; do
  case " $FAMILY_MODELS " in *" $tag "*) : ;; *) continue ;; esac
  spec=$(model_spec "$tag"); read -r mdir _ Lpeak _ _ estcausal <<< "$spec"
  need="engine/family_${tag}_integrity.ok,engine/r3_equiv_bf16.ok,engine/smoke_family_bf16_${tag}.ok"
  [ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE g4_${tag}_alphaHO_cf_L${Lpeak}_s0 "$estcausal" "$need" "$ENVP $PY $KG --model $mdir --model_dtype bf16 --editor alpha $CF $COMMON --lr 0.1 --layer $Lpeak --seed 0 --alpha_proj_source holdout --holdout_frac 1.0 --out results/g4_${tag}_alphaHO_cf_L${Lpeak}_s0.json"
done
heartbeat

# ---------------------------------------------------------------- Phase 3: mid layer (0.50 depth) s0 ROME, breadth-first
for tag in $ALL_TAGS; do
  case " $FAMILY_MODELS " in *" $tag "*) : ;; *) continue ;; esac
  spec=$(model_spec "$tag"); read -r mdir _ _ Lmid estrome _ <<< "$spec"
  need="engine/family_${tag}_integrity.ok,engine/r3_equiv_bf16.ok,engine/smoke_family_bf16_${tag}.ok"
  [ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE gate_${tag}_rome_cf_L${Lmid}_s0 "$estrome" "$need" "$ENVP $PY $KG --model $mdir --model_dtype bf16 --editor rome $CF $COMMON --lr 0.1 --layer $Lmid --seed 0 --out results/gate_${tag}_rome_cf_L${Lmid}_s0.json"
done
heartbeat

# ---------------------------------------------------------------- Phase 4/5: seeds s1, s2 at both layers, breadth-first (FILLER — only
# as far as BUDGET_MIN allows; this is where a short window naturally stops)
for s in 1 2; do
  for tag in $ALL_TAGS; do
    case " $FAMILY_MODELS " in *" $tag "*) : ;; *) continue ;; esac
    spec=$(model_spec "$tag"); read -r mdir _ Lpeak Lmid estrome _ <<< "$spec"
    need="engine/family_${tag}_integrity.ok,engine/r3_equiv_bf16.ok,engine/smoke_family_bf16_${tag}.ok"
    [ "$QUEUE_ABORT" -eq 0 ] && run_row FILLER gate_${tag}_rome_cf_L${Lpeak}_s${s} "$estrome" "$need" "$ENVP $PY $KG --model $mdir --model_dtype bf16 --editor rome $CF $COMMON --lr 0.1 --layer $Lpeak --seed ${s} --out results/gate_${tag}_rome_cf_L${Lpeak}_s${s}.json"
    [ "$QUEUE_ABORT" -eq 0 ] && run_row FILLER gate_${tag}_rome_cf_L${Lmid}_s${s} "$estrome" "$need" "$ENVP $PY $KG --model $mdir --model_dtype bf16 --editor rome $CF $COMMON --lr 0.1 --layer $Lmid --seed ${s} --out results/gate_${tag}_rome_cf_L${Lmid}_s${s}.json"
  done
  heartbeat
done

# ---------------------------------------------------------------- Phase 6: causal seeds s1, s2 at peak layer, breadth-first (lowest
# priority — a 2nd/3rd causal seed matters less than getting s0 for every model first)
for s in 1 2; do
  for tag in $ALL_TAGS; do
    case " $FAMILY_MODELS " in *" $tag "*) : ;; *) continue ;; esac
    spec=$(model_spec "$tag"); read -r mdir _ Lpeak _ _ estcausal <<< "$spec"
    need="engine/family_${tag}_integrity.ok,engine/r3_equiv_bf16.ok,engine/smoke_family_bf16_${tag}.ok"
    [ "$QUEUE_ABORT" -eq 0 ] && run_row FILLER g4_${tag}_alphaHO_cf_L${Lpeak}_s${s} "$estcausal" "$need" "$ENVP $PY $KG --model $mdir --model_dtype bf16 --editor alpha $CF $COMMON --lr 0.1 --layer $Lpeak --seed ${s} --alpha_proj_source holdout --holdout_frac 1.0 --out results/g4_${tag}_alphaHO_cf_L${Lpeak}_s${s}.json"
  done
  heartbeat
done

# ---------------------------------------------------------------- post-processing (CPU-cheap, no GPU row): per-model S x C table +
# causal aggregation, mirrors run_neox20b.sh/run_gptj.sh's own post blocks
for tag in $ALL_TAGS; do
  case " $FAMILY_MODELS " in *" $tag "*) : ;; *) continue ;; esac
  spec=$(model_spec "$tag"); read -r _ _ Lpeak Lmid _ _ <<< "$spec"
  if compgen -G "results/matrices/gate_${tag}_rome_cf_L*_s*.npz" >/dev/null; then
    $PY experiments/mechanism_sc_table.py \
      --npz "results/matrices/gate_${tag}_rome_cf_L*_s*.npz" \
      --known --edit_ok \
      --out "results/FAMILY_${tag}_mechanism_sc_table.json" >> "$LOG" 2>&1 \
      && log "post: FAMILY_${tag}_mechanism_sc_table done" || log "post: FAMILY_${tag}_mechanism_sc_table FAIL"
  fi
  if [ -f experiments/aggregate_g4_causal.py ] \
     && compgen -G "results/matrices/g4_${tag}_alphaHO_cf_L${Lpeak}_s0.npz" >/dev/null; then
    tmp_out="results/.C4_causal_${tag}_table.json.tmp"
    $PY experiments/aggregate_g4_causal.py \
      --rome_glob "results/matrices/gate_${tag}_rome_cf_L{L}_s0.npz" \
      --alpha_glob "results/matrices/g4_${tag}_alphaHO_cf_L{L}_s0.npz" \
      --layers "$Lpeak" --known --edit_ok --proj_source holdout \
      --out "$tmp_out" >> "$LOG" 2>&1 \
      && mv "$tmp_out" "results/C4_causal_${tag}_table.json" \
      && log "post: C4_causal_${tag}_table done (atomic)" \
      || { rm -f "$tmp_out"; log "FAIL aggregate_g4_causal ${tag}"; }
  else
    log "skip C4-${tag} (aggregate_g4_causal.py or holdout alpha s0 npz missing)"
  fi
done

log "================ RUN_FAMILY_TRANSFER COMPLETE (${n_done} done / ${n_fail} fail / ${n_skip} skip) ================"
echo "RUN_FAMILY_TRANSFER_DONE"
