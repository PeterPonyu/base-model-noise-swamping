#!/bin/bash
# run_extension_causal_seeds.sh — Track 2 of the 2026-07-11 cloud extension wave: 3-seed
# hardening for the cross-arch causal (AlphaEdit-holdout) cells that are currently
# single-seed s0 only and weak/negative on disk (team-lead brief, 2026-07-11): gptj L21
# rho -0.204, neox20b L16 rho 0.049, pythia14b L6 rho -0.060, pythia28b L8 rho -0.076.
# GPT-J is LOCAL (data/models/gpt-j-6b already on disk) and has its own local driver
# (run_gptj.sh) — NOT duplicated here. The other three (neox20b, pythia14b, pythia28b)
# were cloud-only (data/models/{gpt-neox-20b,pythia-1.4b,pythia-2.8b} do NOT exist on the
# local box — verified 2026-07-11) and must run cloud-side; this driver ADDS seeds s1/s2
# to their existing s0 causal cells without touching run_neox20b.sh/run_pythia.sh (both
# are read-only reference here — see cloud/README.md's chain-locked-drivers caution and
# memory/live-file-edit-hazard-under-running-queue.md; a standalone new file is the safe
# path regardless of whether either is chain-locked on the launch box).
#
# LAYER/LR RECONCILIATION (flag loudly — this is a real deviation, not a typo): the
# on-disk run_neox20b.sh still shows its ORIGINAL plan (L33 peak, lr 0.1) — that cell is
# now results/gate_neox20b_rome_cf_L33_s0.json.DEAD-LR01 (quarantined; see memory
# neox20b-esr-depth-collapse-20260709.md: L33/lr0.1 esr collapses to ~0.01 at 20B scale).
# The battery that actually ran (07-08/09, likely via an archived wave-2 orchestrator not
# present in this checkout — only JSON results were synced back, per memory
# cloud-wave-complete-20260710.md's "JSON-only sync" note) redesigned to a shallow band at
# lr 0.5, and its surviving causal cell is g4_neox20b_alphaHO_cf_L16_s0.json (esr 0.935,
# provenance.lr=0.5, model_dtype_arg=bf16) — confirmed by reading that file directly AND
# by the team-lead brief's own "neox20b L16 rho 0.049" line. This driver targets L16/lr0.5
# to match the REAL cell on disk, not run_neox20b.sh's stale header plan.
# pythia14b/pythia28b have no such discrepancy — run_pythia.sh is unchanged and its
# adaptive selector picked L6/lr0.1 (pythia14b) and L8/lr0.1 (pythia28b), confirmed by
# reading g4_pythia14b_alphaHO_cf_L6_s0.json / g4_pythia28b_alphaHO_cf_L8_s0.json directly.
#
# SCOPE: TRACK2_SCOPE env var selects which half runs (default "all"):
#   pythia  — single-GPU, cheap (~50-70min/row), runs fine sharded onto either card
#             alongside run_family_transfer.sh during the per-card wave phase.
#   neox    — tensor-parallel, needs BOTH cards together (--device_map, ~40GB bf16
#             weights) — like run_neox20b.sh, this must run in a dedicated dual-card
#             phase AFTER the per-card phase releases both GPUs (see cloud/
#             run_extension_wave.sh's `tp2` subcommand, mirrors run_cloud_wave.sh's
#             `tp20b`). Running "neox" scope with only one card attached will wedge at
#             the on-box TP smoke gate (Phase 0a3 below), which is the intended failure
#             mode — it aborts BEFORE any real GPU spend, never a silent single-card
#             fallback that would corrupt the TP comparison.
#   all     — both (only correct for a genuinely dual-card, single-worker invocation;
#             NOT what the sharded wave uses).
set -u
H="$(cd "$(dirname "$0")" && pwd)"
cd "$H" || exit 1
PY=/home/zeyufu/miniconda3/envs/dl/bin/python3
PY="${CLOUD_PY:-$PY}"
LOG=engine/run_extension_causal_seeds.log
BUDGET_MIN=${BUDGET_MIN:-700}
TRACK2_SCOPE=${TRACK2_SCOPE:-all}
# Cost knobs (see cloud/EXTENSION-WAVE-RUNBOOK.md's cost table). neox20b's TP phase is
# the expensive one (~300min/row on 2 cards) — NEOX_SEEDS defaults to "1" (Trimmed, one
# gap-fill seed, ~$13 total) to match the standing ~¥100-class ($12-15) pre-approved
# cloud-wave ceiling (workspace CLAUDE.md, 2026-07-10 policy addendum) by default. Set
# NEOX_SEEDS="1 2" for Full (both gap-fill seeds, ~$16, ~$1-4 over the ceiling) as an
# EXPLICIT opt-in — get a go-ahead before choosing it. NEOX_SEEDS is read only by the
# neox20b rows below (TRACK2_SCOPE=neox/all) — setting it under pythia scope is inert.
PYTHIA_SEEDS=${PYTHIA_SEEDS:-"1 2"}
NEOX_SEEDS=${NEOX_SEEDS:-"1"}
mkdir -p engine results/matrices results/smoke_extcausal/matrices
echo $$ > engine/run_extension_causal_seeds.pid
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "================ RUN_EXTENSION_CAUSAL_SEEDS START (pid $$, budget ${BUDGET_MIN}m, scope=${TRACK2_SCOPE}) ================"

# ---------------------------------------------------------------- Phase 0a: CPU pre-flight
pf_fail=0
pf(){ if eval "$2"; then log "preflight OK: $1"; else log "PREFLIGHT-FAIL: $1"; pf_fail=1; fi; }
pf "python env" "$PY -c 'import torch, numpy' 2>/dev/null"
pf "counterfact.json" "[ -f data/counterfact.json ]"
pf "killgate_keygeom.py" "[ -f experiments/killgate_keygeom.py ]"
pf "alpha_proj_source flag" "grep -q -- '--alpha_proj_source' experiments/killgate_keygeom.py"
pf "integrity_check.py" "[ -f experiments/tools/integrity_check.py ]"
case "$TRACK2_SCOPE" in
  neox|all)
    pf "device_map flag" "grep -q -- '--device_map' experiments/killgate_keygeom.py"
    pf "arch_compat gptneox branch" "grep -q -- 'family = \"gptneox\"' editors/arch_compat.py"
    pf "tp_edit_util.py present" "[ -f tp_edit_util.py ]"
    pf "smoke_neox20b_cpu.py" "[ -f experiments/smoke_neox20b_cpu.py ]"
    pf "smoke_neox20b_tp_onbox.py" "[ -f experiments/smoke_neox20b_tp_onbox.py ]"
    ;;
esac
# NOTE (soft, informational only — NOT a pf() hard abort): the s0 .npz matrices these
# seeds extend may be genuinely absent on THIS box even though the science is real —
# only JSON results were synced back from the 07-08 wave (memory cloud-wave-complete-
# 20260710.md), npz matrices stayed on the old box's disk. A missing s0 npz does not
# block s1/s2 from running (each seed is an independent killgate_keygeom.py invocation);
# it only means the post-processing aggregate_g4_causal.py step below will produce a
# seed-{1,2}-only or seed-{1,2}+existing-json table until s0's npz is recovered/rerun.
# Verified 2026-07-11 on THIS box: gate_neox20b_rome_cf_L16_s{0,1,2}.npz all present;
# g4_neox20b_alphaHO_cf_L16_s0.npz (the causal s0 cell itself) is NOT present locally —
# exactly the case this soft-check exists for.
[ -f results/matrices/gate_neox20b_rome_cf_L16_s0.npz ] \
  && log "info: neox20b L16 rome s0 npz present (paired-cell precedent)" \
  || log "info: neox20b L16 rome s0 npz ABSENT locally (non-blocking — only JSON may have synced)"
[ -f results/matrices/g4_neox20b_alphaHO_cf_L16_s0.npz ] \
  && log "info: neox20b L16 alphaHO s0 npz present" \
  || log "info: neox20b L16 alphaHO s0 npz ABSENT locally (non-blocking — s1/s2 still run; multi-seed aggregation will be seed-partial until recovered)"
case "$TRACK2_SCOPE" in
  pythia|all)
    [ -f results/matrices/g4_pythia14b_alphaHO_cf_L6_s0.npz ] \
      && log "info: pythia14b L6 alphaHO s0 npz present" \
      || log "info: pythia14b L6 alphaHO s0 npz ABSENT locally (non-blocking)"
    [ -f results/matrices/g4_pythia28b_alphaHO_cf_L8_s0.npz ] \
      && log "info: pythia28b L8 alphaHO s0 npz present" \
      || log "info: pythia28b L8 alphaHO s0 npz ABSENT locally (non-blocking)"
    ;;
esac
rm -f engine/smoke_extcausal_*.ok
if [ "$pf_fail" -ne 0 ]; then log "ABORT: CPU preflight failed"; exit 3; fi

# ---------------------------------------------------------------- Phase 0a2: neox20b-only structural + on-box TP smoke (HARD gate,
# same two scripts + same abort-before-spend contract as run_neox20b.sh Phase 0a2/0a3 —
# never skip this for a TP row, see that driver's header for the full rationale)
if [ "$TRACK2_SCOPE" = "neox" ] || [ "$TRACK2_SCOPE" = "all" ]; then
  if $PY experiments/smoke_neox20b_cpu.py >> "$LOG" 2>&1; then
    log "preflight OK: CPU structural smoke ALL_PASS"
  else
    log "ABORT: CPU structural smoke FAILED"; exit 3
  fi
  TP_SMOKE_DEVICES=${TP_SMOKE_DEVICES:-}
  TP_SMOKE_ARGS=(); [ -n "$TP_SMOKE_DEVICES" ] && TP_SMOKE_ARGS=(--devices "$TP_SMOKE_DEVICES")
  if $PY experiments/smoke_neox20b_tp_onbox.py "${TP_SMOKE_ARGS[@]}" >> "$LOG" 2>&1; then
    log "preflight OK: ON-BOX TP smoke ALL_PASS — real 2-card dispatch verified"
  else
    log "ABORT: ON-BOX TP smoke FAILED — this box cannot safely TP-edit NeoX-20B (single card attached under neox scope? see header)"; exit 3
  fi
fi

# ---------------------------------------------------------------- Phase 0a3: per-model integrity re-derivation (soft gate)
rm -f engine/extcausal_neox20b_integrity.ok engine/extcausal_pythia14b_integrity.ok engine/extcausal_pythia28b_integrity.ok
check_integrity(){   # check_integrity <tag> <mdir> <expect_params>
  if [ -d "$2" ]; then
    $PY experiments/tools/integrity_check.py "$2" --expect_params "$3" >> "$LOG" 2>&1 \
      && { : > "engine/extcausal_${1}_integrity.ok"; log "integrity OK: ${1}"; } \
      || log "integrity NOT-READY: ${1} (${2} incomplete/corrupt — rows CONFIG-skip)"
  else
    log "MODEL-ABSENT: ${1} (${2}) — every row for this tag CONFIG-skips cleanly (not on this box — see runbook's download step)"
  fi
}
case "$TRACK2_SCOPE" in
  neox|all) check_integrity neox20b data/models/gpt-neox-20b 20.5546e9 ;;
esac
case "$TRACK2_SCOPE" in
  pythia|all)
    check_integrity pythia14b data/models/pythia-1.4b 1.5153e9
    check_integrity pythia28b data/models/pythia-2.8b 2.9094e9
    ;;
esac

# engine/r3_equiv_bf16.ok — SHARED marker, only needed for the bf16 neox path; reuse if
# fresh (same freshness check as run_8bcausal.sh/run_neox20b.sh/run_family_transfer.sh)
if [ "$TRACK2_SCOPE" = "neox" ] || [ "$TRACK2_SCOPE" = "all" ]; then
  if [ -f engine/r3_equiv_bf16.ok ] && [ -f results/matrices/equiv_llama1b_bf16_L12_s0.npz ] \
     && [ "$(stat -c %Y results/matrices/equiv_llama1b_bf16_L12_s0.npz)" -ge "$(stat -c %Y experiments/killgate_keygeom.py)" ] \
     && [ "$(stat -c %Y results/matrices/equiv_llama1b_bf16_L12_s0.npz)" -ge "$(stat -c %Y editors/arch_compat.py)" ]; then
    log "engine/r3_equiv_bf16.ok fresh — reusing, no GPU spend"
  else
    rm -f engine/r3_equiv_bf16.ok
    log "engine/r3_equiv_bf16.ok absent/stale — will re-derive below (real GPU row)"
  fi
fi

# ---------------------------------------------------------------- Phase 0b: GPU idle gate
DRYRUN=${DRYRUN:-0}
if [ "$DRYRUN" -eq 1 ]; then
  log "DRYRUN=1 — skipping GPU idle gate, printing every run_row call without executing"
else
  source "$H/cloud/gpu_idle_lib.sh"
  idle_gate_wait || { log "ABORT: idle gate failed"; exit 2; }
fi
T0=$(date +%s)

# ---------------------------------------------------------------- helpers (verbatim template)
ENVP="env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1"
KG="experiments/killgate_keygeom.py"
CF="--dataset counterfact --data data/counterfact.json"
DEVICE_MAP=${DEVICE_MAP:-auto}
TP="--device_map ${DEVICE_MAP}"
COMMON="--n_edits 200 --n_probes 500 --steps 20 --save_matrices --matrix_dir results/matrices"
SMK="--n_edits 4 --n_probes 40 --steps 2 --save_matrices --matrix_dir results/smoke_extcausal/matrices"
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
  case "$cmd" in *smoke_extcausal*) outn="results/smoke_extcausal/matrices/$(basename "${outj%.json}").npz";; esac
  if [ -n "$outj" ] && [ -f "$outj" ] && [ -f "$outn" ]; then
    local pvmode="full"; [ "$class" = "SMOKE" ] && pvmode="smoke"
    if validate "$outj" "$outn" "$pvmode" | grep -q VALIDATE-FAIL; then
      mv "$outj" "$outj.INVALID" 2>/dev/null; mv "$outn" "$outn.INVALID" 2>/dev/null
      log "STALE-INVALID ${tag} — quarantined; re-running"
    else
      log "skip ${tag} (exists, validated)"
      [ "$class" = "SMOKE" ] && : > "engine/smoke_extcausal_${tag}.ok"
      return
    fi
  fi
  local mdir; mdir=$(echo "$cmd" | grep -oE -- '--model [^ ]+' | awk '{print $2}')
  case "$mdir" in data/*|/*)
    if [ -n "$mdir" ] && [ ! -d "$mdir" ]; then log "CONFIG-SKIP ${tag} (no model ${mdir})"; n_skip=$((n_skip+1)); return; fi;;
  esac
  local cap=$(( est * 60 * 3 + 1800 ))
  log "RUN ${tag} [${class}] (est ${est}m, cap ${cap}s, elapsed ${now}m) -> engine/extcausal_${tag}.log"
  local t rc dt
  t=$(date +%s)
  timeout --signal=TERM --kill-after=60 "${cap}s" bash -c "$cmd" >> "engine/extcausal_${tag}.log" 2>&1 </dev/null
  rc=$?; dt=$(( $(date +%s) - t ))
  local vmode="full"; [ "$class" = "SMOKE" ] && vmode="smoke"
  if [ "$rc" -eq 0 ] && [ -n "$outj" ] && [ -f "$outj" ] && [ -f "$outn" ]; then
    local v; v=$(validate "$outj" "$outn" "$vmode")
    if echo "$v" | grep -q VALIDATE-FAIL; then
      mv "$outj" "$outj.INVALID" 2>/dev/null; mv "$outn" "$outn.INVALID" 2>/dev/null
      log "FAIL ${tag} (${dt}s) OUTPUT-INVALID: ${v}"; n_fail=$((n_fail+1))
    else
      log "done ${tag} (${dt}s) ${v}"; n_done=$((n_done+1)); wedge_fail=0
      [ "$class" = "SMOKE" ] && : > "engine/smoke_extcausal_${tag}.ok"
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

# ---------------------------------------------------------------- pythia14b / pythia28b: seeds s1, s2 (single-GPU, cheap)
if [ "$TRACK2_SCOPE" = "pythia" ] || [ "$TRACK2_SCOPE" = "all" ]; then
  for s in $PYTHIA_SEEDS; do
    need="engine/extcausal_pythia14b_integrity.ok"
    [ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE g4_pythia14b_alphaHO_cf_L6_s${s} 50 "$need" "$ENVP $PY $KG --model data/models/pythia-1.4b --editor alpha $CF $COMMON --lr 0.1 --layer 6 --seed ${s} --alpha_proj_source holdout --holdout_frac 1.0 --out results/g4_pythia14b_alphaHO_cf_L6_s${s}.json"
  done
  heartbeat
  for s in $PYTHIA_SEEDS; do
    need="engine/extcausal_pythia28b_integrity.ok"
    [ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE g4_pythia28b_alphaHO_cf_L8_s${s} 70 "$need" "$ENVP $PY $KG --model data/models/pythia-2.8b --editor alpha $CF $COMMON --lr 0.1 --layer 8 --seed ${s} --alpha_proj_source holdout --holdout_frac 1.0 --out results/g4_pythia28b_alphaHO_cf_L8_s${s}.json"
  done
  heartbeat
fi

# ---------------------------------------------------------------- neox20b: seeds s1, s2 at L16/lr0.5 (TP, both cards — see header
# reconciliation note for why L16/lr0.5, not run_neox20b.sh's on-disk L33/lr0.1 plan)
if [ "$TRACK2_SCOPE" = "neox" ] || [ "$TRACK2_SCOPE" = "all" ]; then
  NEEDS_NEOX="engine/extcausal_neox20b_integrity.ok,engine/r3_equiv_bf16.ok"
  if [ ! -f engine/r3_equiv_bf16.ok ]; then
    [ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE equiv_llama1b_bf16_L12_s0 22 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --model_dtype bf16 --editor rome $CF $COMMON --lr 0.1 --layer 12 --seed 0 --out results/equiv_llama1b_bf16_L12_s0.json"
    if [ "$DRYRUN" -ne 1 ] && [ -f results/matrices/equiv_llama1b_bf16_L12_s0.npz ] && [ -f results/matrices/gate_llama1b_rome_cf_L12_s0.npz ]; then
      $PY - >> "$LOG" 2>&1 <<'EOF'
import numpy as np, sys
sys.path.insert(0, 'experiments')
from analyze_matrices import within_probe_rhos
def rho(f):
    d = np.load(f); C = d['COS'].astype(float); D = d['damage_logit'].astype(float)
    m = d['edit_ok'].astype(float) > 0; c = d['pre_p'].astype(float) > 0.05
    return float(np.nanmean(within_probe_rhos(C[m][:, c], D[m][:, c])))
r_fp32 = rho('results/matrices/gate_llama1b_rome_cf_L12_s0.npz')
r_bf16 = rho('results/matrices/equiv_llama1b_bf16_L12_s0.npz')
d = abs(r_fp32 - r_bf16)
print(f"[extcausal equiv-gate] fp32 rho={r_fp32:+.4f} bf16 rho={r_bf16:+.4f} |drho|={d:.4f} bar=0.02")
if d < 0.02:
    open('engine/r3_equiv_bf16.ok', 'w').close()
    print("[extcausal equiv-gate] PASS")
else:
    print("[extcausal equiv-gate] FAIL — bf16 neox rows stay CONFIG-skipped")
EOF
    fi
  fi
  # micro-smoke: this exact editor(alpha)+layer(16)+dtype(bf16)+TP combo has not run in
  # THIS driver before (run_neox20b.sh's own smoke was at L33) — cheap insurance
  [ "$QUEUE_ABORT" -eq 0 ] && run_row SMOKE alphaHO_neox20b_L16 20 "$NEEDS_NEOX" "$ENVP $PY $KG --model data/models/gpt-neox-20b --model_dtype bf16 $TP --editor alpha $CF $SMK --lr 0.5 --layer 16 --seed 0 --alpha_proj_source holdout --holdout_frac 1.0 --out results/smoke_extcausal/alphaHO_neox20b_L16.json"
  for s in $NEOX_SEEDS; do
    [ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE g4_neox20b_alphaHO_cf_L16_s${s} 300 "$NEEDS_NEOX,engine/smoke_extcausal_alphaHO_neox20b_L16.ok" "$ENVP $PY $KG --model data/models/gpt-neox-20b --model_dtype bf16 $TP --editor alpha $CF $COMMON --lr 0.5 --layer 16 --seed ${s} --alpha_proj_source holdout --holdout_frac 1.0 --out results/g4_neox20b_alphaHO_cf_L16_s${s}.json"
  done
  heartbeat
fi

# ---------------------------------------------------------------- post-processing: refresh causal aggregation tables now that s1/s2
# exist, for whichever tags this invocation actually touched
if [ "$TRACK2_SCOPE" = "pythia" ] || [ "$TRACK2_SCOPE" = "all" ]; then
  for spec in "pythia14b:6" "pythia28b:8"; do
    tag="${spec%%:*}"; L="${spec##*:}"
    mmap="data/models/pythia-1.4b"; [ "$tag" = "pythia28b" ] && mmap="data/models/pythia-2.8b"
    if [ -f experiments/aggregate_g4_causal.py ] && compgen -G "results/matrices/g4_${tag}_alphaHO_cf_L${L}_s*.npz" >/dev/null; then
      tmp_out="results/.C4_causal_${tag}_table.json.tmp"
      $PY experiments/aggregate_g4_causal.py \
        --rome_glob "results/matrices/gate_${tag}_rome_cf_L{L}_s*.npz" \
        --alpha_glob "results/matrices/g4_${tag}_alphaHO_cf_L{L}_s*.npz" \
        --layers "$L" --known --edit_ok --proj_source holdout \
        --out "$tmp_out" >> "$LOG" 2>&1 \
        && mv "$tmp_out" "results/C4_causal_${tag}_table.json" \
        && log "post: C4_causal_${tag}_table refreshed (atomic, now multi-seed)" \
        || { rm -f "$tmp_out"; log "FAIL aggregate_g4_causal ${tag}"; }
    fi
  done
fi
if [ "$TRACK2_SCOPE" = "neox" ] || [ "$TRACK2_SCOPE" = "all" ]; then
  if [ -f experiments/aggregate_g4_causal.py ] && compgen -G "results/matrices/g4_neox20b_alphaHO_cf_L16_s*.npz" >/dev/null; then
    tmp_out="results/.C4_causal_neox20b_table.json.tmp"
    $PY experiments/aggregate_g4_causal.py \
      --rome_glob "results/matrices/gate_neox20b_rome_cf_L{L}_s*.npz" \
      --alpha_glob "results/matrices/g4_neox20b_alphaHO_cf_L{L}_s*.npz" \
      --layers 16 --known --edit_ok --proj_source holdout \
      --out "$tmp_out" >> "$LOG" 2>&1 \
      && mv "$tmp_out" results/C4_causal_neox20b_table.json \
      && log "post: C4_causal_neox20b_table refreshed (atomic, now multi-seed, L16 not L33 — see header)" \
      || { rm -f "$tmp_out"; log "FAIL aggregate_g4_causal neox20b"; }
  fi
fi

log "================ RUN_EXTENSION_CAUSAL_SEEDS COMPLETE (${n_done} done / ${n_fail} fail / ${n_skip} skip) ================"
echo "RUN_EXTENSION_CAUSAL_SEEDS_DONE"
