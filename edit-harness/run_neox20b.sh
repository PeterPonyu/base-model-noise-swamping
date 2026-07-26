#!/bin/bash
# run_neox20b.sh — GPT-NeoX-20B (EleutherAI/gpt-neox-20b) tensor-parallel ROME/AlphaEdit
# battery (WP3, 2026-07-08). Template = run_gptj.sh (verbatim skeleton for the
# bf16-equivalence-gated large-model pattern), NeoX20B-namespaced (own pid/log/markers),
# PLUS three things run_gptj.sh never needed: (1) a THIRD arch_compat family ("gptneox",
# 44-layer model.gpt_neox.layers, not model.transformer.h — see editors/arch_compat.py's
# module docstring), (2) tensor-parallel loading across 2 GPUs via killgate's new
# --device_map flag (NeoX-20B is ~40GB bf16, too big for one 24GB card), and (3) the
# cloud driver-idle-gate contract (cloud/gpu_idle_lib.sh) since this is meant to run on
# the AutoDL dual-4090 box via cloud/run_cloud_wave.sh's `tp20b` mode, not the single-GPU
# local box.
#
# ARCH NOTE: see editors/arch_compat.py's module docstring for the full "gptneox" family
# story (block-0 mlp has dense_h_to_4h/dense_4h_to_h, dense_4h_to_h already nn.Linear —
# same "no conversion, graft-only" shape as GPT-J's fc_out, just a different top-level
# container). killgate_keygeom.py's memit fence was broadened again:
# `arch in ("gpt2", "gptj", "gptneox")` — MEMIT's _hidden_at residual-stream hook needs
# the real decoder-layer Module, which the graft replaces with a SimpleNamespace on all
# three non-native families; MEMIT is therefore NOT part of this driver's rows.
#
# TP NOTE (the load-bearing part of this build — see tp_edit_util.py for the full
# argument): every editor threads a single `device` string end-to-end. That is correct
# for INPUT-side calls (tokenizer(...).to(device)) under accelerate's --device_map
# dispatch (its forward hooks move activations across shard boundaries automatically),
# but was WRONG for two things that build a tensor from scratch and combine it with the
# EDITED LAYER's own weight — which can sit on a DIFFERENT card than the embedding under
# TP: the ROME/AlphaEdit optimized value `v` (editors/rome_native.py::_optimise_value)
# and the AlphaEdit null-space projector `P` (editors/alphaedit.py::_resolve_projector).
# Both now route through tp_edit_util.resolve_layer_device() instead of the ambient
# `device`. A third, more acute bug: every editor's apply_edit() called a bare
# `model.to(device)` — on an accelerate-dispatched model that COLLAPSES the whole model
# back onto one device on every single edit, silently undoing --device_map sharding on
# the SECOND edit of any cell. tp_edit_util.safe_model_to() guards it (no-ops once
# model.hf_device_map is set). Single-device path (no --device_map, every other driver
# in this repo): every fix here is a byte-identical no-op — verified by
# experiments/smoke_memit_cpu.py (regression, native Llama, unaffected) and
# experiments/smoke_neox20b_cpu.py (T0-T5, this build).
#
# ON-BOX RESIDUAL RISK (read before spending money): this build was authored on a box
# with only ONE GPU (reserved for a concurrent job — no CUDA use permitted), so the
# cross-card behavior of accelerate's REAL dispatch_model was validated with a degenerate
# `--devices cpu,cpu` run of experiments/smoke_neox20b_tp_onbox.py (proves the
# device_map SHAPE + the full edit-call chain via accelerate's actual API, not a mock —
# see that script's docstring) rather than real cuda:0/cuda:1 placement. Phase 0a3 below
# runs that exact script for real, on this box, BEFORE any 20B download/run is attempted
# — if the box doesn't have >=2 CUDA devices, or a real cross-card edit fails, this
# driver aborts immediately (see pf() gate), never reaching the 40GB download or a GPU-
# hour of real science.
#
# LAYER BAND (44 layers; proportional to the llama1b L8/L10/L12/L14-of-16 band, i.e. the
# SAME depth fractions {0.5, 0.625, 0.75, 0.875} used everywhere else in this harness):
# round(0.5*44)=22, round(0.625*44)=28 (27.5 rounds to nearest-even 28),
# round(0.75*44)=33 (exact, the PEAK), round(0.875*44)=38 (38.5 rounds to nearest-even
# 38) — computed with Python's own round() (banker's rounding), verified interactively
# before writing this file. => L22, L28, L33 (peak), L38.
#
# GPU COST — FLAGGED, NO NEOX-20B PRECEDENT ANYWHERE IN THE HARNESS (guessed, not
# measured, same honesty convention as run_gptj.sh/run_8bcausal.sh's own flags): NeoX-20B
# hidden_size=6144 vs Llama-3.1-8B's 4096 — the O(d^2) cost per edit/projector-fit scales
# ~(6144/4096)^2 = 2.25x. Anchoring off run_r3.sh's MEASURED ~100min/row for Llama-3.1-8B
# ROME at COMMON settings (200 edits/500 probes/20 steps) gives a naive ~225min/row, PLUS
# an ENTIRELY UNMEASURED tensor-parallel communication overhead (cross-card activation
# transfer on every forward through the shard boundary, twice per edit — key capture +
# value-opt forward passes) that this build has no way to estimate without the real box.
# Budgeted at 260min/row (225 + ~15% TP-overhead guess) to be conservative. 12 ROME rows
# (4 layers x 3 seeds) at ~260min + 1 alpha-holdout causal row at ~300min + smoke/gate
# overhead => ~3600 GPU-min (~60h) if run to completion — this will NOT fit one
# BUDGET_MIN window by a wide margin. Ordered peak-layer-first (L33) so a partial/
# interrupted window keeps the highest-value cells; BUDGET_MIN below is a single-session
# default, override via env for a longer window (this is exactly the kind of multi-day
# job the AutoDL box, not the local laptop, is for).
set -u
H="$(cd "$(dirname "$0")" && pwd)"
cd "$H" || exit 1
PY=${NEOX_PY:-${CLOUD_PY:-python3}}   # 2026-07-08 B4 fix: portable H; PY chain now also honors run_cloud_wave.sh's CLOUD_PY
LOG=engine/run_neox20b.log
BUDGET_MIN=${BUDGET_MIN:-600}
MODEL_DIR=data/models/gpt-neox-20b
# ASSUMPTION FLAGGED: cloud/setup_autodl.sh's MODEL_20B="EleutherAI/gpt-neox-20b" is
# currently pulled via `hf download` into the HF cache (HF_HOME), not into this repo's
# `data/models/<name>` local-dir convention every other driver uses (download_models.py's
# (repo_id, local_dir) manifest). Reconcile before the real run: either re-download with
# `hf download EleutherAI/gpt-neox-20b --local-dir data/models/gpt-neox-20b`, or symlink
# the resolved HF cache snapshot dir to MODEL_DIR above. This driver assumes the
# reconciled local-dir form (matches integrity_check.py's os.listdir(d) contract, which
# needs a real directory, not an HF repo id needing snapshot-hash resolution).
mkdir -p engine results/matrices results/smoke_neox20b/matrices
echo $$ > engine/run_neox20b.pid
[ -f engine/neox20b_round_start ] || stat -c %Y engine/run_neox20b.pid > engine/neox20b_round_start
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "================ RUN_NEOX20B START (pid $$, budget ${BUDGET_MIN}m) ================"

# ---------------------------------------------------------------- Phase 0a: CPU pre-flight
pf_fail=0
pf(){ if eval "$2"; then log "preflight OK: $1"; else log "PREFLIGHT-FAIL: $1"; pf_fail=1; fi; }
pf "python env" "$PY -c 'import torch, numpy, accelerate' 2>/dev/null"
pf "counterfact.json" "[ -f data/counterfact.json ]"
pf "device_map flag" "grep -q -- '--device_map' experiments/killgate_keygeom.py"
pf "alpha_proj_source flag" "grep -q -- '--alpha_proj_source' experiments/killgate_keygeom.py"
pf "integrity_check.py" "[ -f experiments/tools/integrity_check.py ]"
pf "arch_compat gptneox branch" "grep -q -- 'family = \"gptneox\"' editors/arch_compat.py"
pf "killgate gptneox memit fence" "grep -q -- 'arch in (\"gpt2\", \"gptj\", \"gptneox\")' experiments/killgate_keygeom.py"
pf "tp_edit_util.py present" "[ -f tp_edit_util.py ]"
pf "equiv comparator fp32 npz" "[ -f results/matrices/gate_llama1b_rome_cf_L12_s0.npz ]"
pf "disk >=45GB free" "[ \$(df --output=avail -BG /home | tail -1 | tr -dc 0-9) -ge 45 ]"
rm -f engine/smoke_neox20b_*.ok
if [ "$pf_fail" -ne 0 ]; then log "ABORT: CPU preflight failed"; exit 3; fi

# ---------------------------------------------------------------- Phase 0a2: CPU structural smoke
# T0-T5 of experiments/smoke_neox20b_cpu.py — arch_compat graft/equivalence, key capture
# through the graft, a ROME edit with no NaN, and tp_edit_util's device-resolution logic
# against MOCKED heterogeneous devices. Cheap (~seconds, tiny synthetic model, no
# download) — re-run every launch so a code regression is caught before any GPU spend.
if $PY experiments/smoke_neox20b_cpu.py >> "$LOG" 2>&1; then
  log "preflight OK: CPU structural smoke (smoke_neox20b_cpu.py) ALL_PASS"
else
  log "ABORT: CPU structural smoke FAILED — see $LOG for the per-test breakdown"
  exit 3
fi

# ---------------------------------------------------------------- Phase 0a3: ON-BOX TP smoke
# THE hard gate this driver exists to run before spending anything: a REAL
# accelerate.dispatch_model split of a tiny synthetic 2-layer GPT-NeoX across the box's
# actual GPU cards (see experiments/smoke_neox20b_tp_onbox.py's docstring for exactly
# what this does and does not prove, and this file's own "ON-BOX RESIDUAL RISK" header
# note). <2 CUDA devices, or a real cross-card edit failure, aborts HERE — never reaches
# the 40GB download or a GPU-hour of real science.
TP_SMOKE_DEVICES=${TP_SMOKE_DEVICES:-}   # override for a box with an unusual >2-GPU
# layout, or (author-time only) `cpu,cpu` to exercise this gate's plumbing without a
# second real card — see experiments/smoke_neox20b_tp_onbox.py's own docstring. Unset
# (the production default) lets that script auto-detect cuda:0,cuda:1.
TP_SMOKE_ARGS=(); [ -n "$TP_SMOKE_DEVICES" ] && TP_SMOKE_ARGS=(--devices "$TP_SMOKE_DEVICES")
if $PY experiments/smoke_neox20b_tp_onbox.py "${TP_SMOKE_ARGS[@]}" >> "$LOG" 2>&1; then
  log "preflight OK: ON-BOX TP smoke (smoke_neox20b_tp_onbox.py) ALL_PASS — real 2-card dispatch verified"
else
  log "ABORT: ON-BOX TP smoke FAILED (see $LOG) — this box cannot safely tensor-parallel-edit NeoX-20B; do NOT proceed"
  exit 3
fi

# ---------------------------------------------------------------- Phase 0a4: re-derive engine/neox20b_integrity.ok
# NOT a pf() hard gate: cloud/setup_autodl.sh's --with-20b download phase may still be
# filling data/models/gpt-neox-20b concurrently — every science row below is gated on
# this marker via run_row's `needs` mechanism, so an incomplete download produces clean
# CONFIG-SKIP log lines, never a crash or a hard abort (same pattern as run_gptj.sh).
rm -f engine/neox20b_integrity.ok
if [ -d "$MODEL_DIR" ]; then
  $PY experiments/tools/integrity_check.py "$MODEL_DIR" --expect_params 20.5546e9 >> "$LOG" 2>&1 \
    && { : > engine/neox20b_integrity.ok; log "integrity OK: gpt-neox-20b"; } \
    || log "integrity NOT-READY: gpt-neox-20b (download incomplete/corrupt, or not yet reconciled into $MODEL_DIR — see the MODEL_DIR ASSUMPTION FLAGGED note above — rows CONFIG-skip)"
else
  log "MODEL-ABSENT: $MODEL_DIR directory not found — rows CONFIG-skip"
fi
# engine/r3_equiv_bf16.ok: SHARED marker across every large-model driver (see
# run_gptj.sh/run_8bcausal.sh precedent) — re-derive only if stale/absent.
if [ -f engine/r3_equiv_bf16.ok ] && [ -f results/matrices/equiv_llama1b_bf16_L12_s0.npz ] \
   && [ "$(stat -c %Y results/matrices/equiv_llama1b_bf16_L12_s0.npz)" -ge "$(stat -c %Y experiments/killgate_keygeom.py)" ] \
   && [ "$(stat -c %Y results/matrices/equiv_llama1b_bf16_L12_s0.npz)" -ge "$(stat -c %Y editors/arch_compat.py)" ]; then
  log "engine/r3_equiv_bf16.ok fresh (postdates killgate_keygeom.py and arch_compat.py) — reusing, no GPU spend"
else
  rm -f engine/r3_equiv_bf16.ok
  log "engine/r3_equiv_bf16.ok absent/stale — will re-derive in Phase A below (real GPU row)"
fi

# ---------------------------------------------------------------- Phase 0b: GPU idle gate
# cloud driver-idle-gate contract (cloud/gpu_idle_lib.sh) — NOT the inline single-GPU
# poll every other driver hand-rolls (this driver spans BOTH cards, so an unqualified
# nvidia-smi query without -i is the correct behavior here, unlike a single-card worker).
# SKIP_IDLE_GATE=1 is the expected dedicated-box default (see cloud/run_cloud_wave.sh's
# `tp20b` case, which does not set it — this driver honors whatever the launcher sets).
DRYRUN=${DRYRUN:-0}
if [ "$DRYRUN" -eq 1 ]; then
  log "DRYRUN=1 — skipping GPU idle gate, printing every run_row call without executing"
else
  # shellcheck source=cloud/gpu_idle_lib.sh
  source cloud/gpu_idle_lib.sh
  idle_gate_wait >> "$LOG" 2>&1 || { log "ABORT: idle_gate_wait failed"; exit 2; }
fi
T0=$(date +%s)

# ---------------------------------------------------------------- helpers (gptj/8bcausal template, verbatim)
ENVP="env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1"
KG="experiments/killgate_keygeom.py"
CF="--dataset counterfact --data data/counterfact.json"
DEVICE_MAP=${DEVICE_MAP:-auto}
TP="--device_map ${DEVICE_MAP}"
COMMON="--n_edits 200 --n_probes 500 --steps 20 --save_matrices --matrix_dir results/matrices"
SMK="--n_edits 4 --n_probes 40 --steps 2 --save_matrices --matrix_dir results/smoke_neox20b/matrices"
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
  case "$cmd" in *smoke_neox20b*) outn="results/smoke_neox20b/matrices/$(basename "${outj%.json}").npz";; esac
  if [ -n "$outj" ] && [ -f "$outj" ] && [ -f "$outn" ]; then
    local pvmode="full"; [ "$class" = "SMOKE" ] && pvmode="smoke"
    if validate "$outj" "$outn" "$pvmode" | grep -q VALIDATE-FAIL; then
      mv "$outj" "$outj.INVALID" 2>/dev/null; mv "$outn" "$outn.INVALID" 2>/dev/null
      log "STALE-INVALID ${tag} — quarantined; re-running"
    else
      log "skip ${tag} (exists, validated)"
      [ "$class" = "SMOKE" ] && : > "engine/smoke_neox20b_${tag}.ok"
      return
    fi
  fi
  local mdir; mdir=$(echo "$cmd" | grep -oE -- '--model [^ ]+' | awk '{print $2}')
  case "$mdir" in data/*|/*)
    if [ -n "$mdir" ] && [ ! -d "$mdir" ]; then log "CONFIG-SKIP ${tag} (no model ${mdir})"; n_skip=$((n_skip+1)); return; fi;;
  esac
  local cap=$(( est * 60 * 3 + 1800 ))
  log "RUN ${tag} [${class}] (est ${est}m, cap ${cap}s, elapsed ${now}m) -> engine/neox20b_${tag}.log"
  local t rc dt
  t=$(date +%s)
  timeout --signal=TERM --kill-after=60 "${cap}s" bash -c "$cmd" >> "engine/neox20b_${tag}.log" 2>&1
  rc=$?; dt=$(( $(date +%s) - t ))
  local vmode="full"; [ "$class" = "SMOKE" ] && vmode="smoke"
  if [ "$rc" -eq 0 ] && [ -n "$outj" ] && [ -f "$outj" ] && [ -f "$outn" ]; then
    local v; v=$(validate "$outj" "$outn" "$vmode")
    if echo "$v" | grep -q VALIDATE-FAIL; then
      mv "$outj" "$outj.INVALID" 2>/dev/null; mv "$outn" "$outn.INVALID" 2>/dev/null
      log "FAIL ${tag} (${dt}s) OUTPUT-INVALID: ${v}"; n_fail=$((n_fail+1))
    else
      log "done ${tag} (${dt}s) ${v}"; n_done=$((n_done+1)); wedge_fail=0
      [ "$class" = "SMOKE" ] && : > "engine/smoke_neox20b_${tag}.ok"
      if [ "$class" != "SMOKE" ] && [ "$dt" -gt $(( est * 60 * 14 / 10 )) ]; then
        local pw; pw=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null | head -1)
        log "THERMAL-WATCH ${tag} ran ${dt}s > 1.4x est; power.draw card0 now=${pw:-NA}W (2-card job — this reads card 0 only; wedge if <100 under load)"
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
# GPT-NeoX-20B + bf16 + --device_map + the new arch_compat graft has NEVER run
# end-to-end on real weights.
run_row SMOKE bf16tp_neox20b 20 engine/neox20b_integrity.ok "$ENVP $PY $KG --model $MODEL_DIR --model_dtype bf16 $TP --editor rome $CF $SMK --lr 0.1 --layer 33 --seed 0 --out results/smoke_neox20b/bf16tp_neox20b.json"
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
print(f"[neox20b equiv-gate] fp32 rho={r_fp32:+.4f} bf16 rho={r_bf16:+.4f} |drho|={d:.4f} bar=0.02")
if d < 0.02:
    open('engine/r3_equiv_bf16.ok', 'w').close()
    print("[neox20b equiv-gate] PASS — NeoX-20B science admitted")
else:
    print("[neox20b equiv-gate] FAIL — bf16 rows stay CONFIG-skipped; investigate before any NeoX-20B claim")
EOF
fi
heartbeat

# ---------------------------------------------------------------- Block N: NeoX-20B ROME layer band x 3 seeds
# peak-first (L33 = 0.75 depth, the llama1b-L12 equivalent) so a partial/interrupted
# window keeps the highest-value cells; see header for the layer-band derivation and
# the (unmeasured, flagged) per-row cost estimate.
NEEDS="engine/neox20b_integrity.ok,engine/smoke_neox20b_bf16tp_neox20b.ok,engine/r3_equiv_bf16.ok"
for s in 0 1 2; do
  [ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE gate_neox20b_rome_cf_L33_s${s} 260 "$NEEDS" "$ENVP $PY $KG --model $MODEL_DIR --model_dtype bf16 $TP --editor rome $CF $COMMON --lr 0.1 --layer 33 --seed ${s} --out results/gate_neox20b_rome_cf_L33_s${s}.json"
done
heartbeat

# ---------------------------------------------------------------- Block C: matched alpha-holdout causal pair (peak layer, s0)
# holdout (not "probes") projector source — the HONEST causal protocol, not the
# by-construction reference (memory: c4-alphaedit-projector-circularity.md).
[ "$QUEUE_ABORT" -eq 0 ] && run_row SMOKE alphaHO_neox20b 20 "$NEEDS" "$ENVP $PY $KG --model $MODEL_DIR --model_dtype bf16 $TP --editor alpha $CF $SMK --lr 0.1 --layer 33 --seed 0 --alpha_proj_source holdout --holdout_frac 1.0 --out results/smoke_neox20b/alphaHO_neox20b.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE g4_neox20b_alphaHO_cf_L33_s0 300 "$NEEDS,engine/smoke_neox20b_alphaHO_neox20b.ok" "$ENVP $PY $KG --model $MODEL_DIR --model_dtype bf16 $TP --editor alpha $CF $COMMON --lr 0.1 --layer 33 --seed 0 --alpha_proj_source holdout --holdout_frac 1.0 --out results/g4_neox20b_alphaHO_cf_L33_s0.json"
heartbeat

# ---------------------------------------------------------------- Block F: FILLER layers (budget-gated, runs LAST —
# the peak-layer band + causal pair above must be funded first; L22/L28/L38 are the
# remaining 3 layer-band points, lowest marginal value given the peak already carries
# the headline law-replication claim, mirroring run_gptj.sh's own Block F precedent).
for s in 0 1 2; do
  [ "$QUEUE_ABORT" -eq 0 ] && run_row FILLER gate_neox20b_rome_cf_L28_s${s} 260 "$NEEDS" "$ENVP $PY $KG --model $MODEL_DIR --model_dtype bf16 $TP --editor rome $CF $COMMON --lr 0.1 --layer 28 --seed ${s} --out results/gate_neox20b_rome_cf_L28_s${s}.json"
done
heartbeat
for s in 0 1 2; do
  [ "$QUEUE_ABORT" -eq 0 ] && run_row FILLER gate_neox20b_rome_cf_L22_s${s} 260 "$NEEDS" "$ENVP $PY $KG --model $MODEL_DIR --model_dtype bf16 $TP --editor rome $CF $COMMON --lr 0.1 --layer 22 --seed ${s} --out results/gate_neox20b_rome_cf_L22_s${s}.json"
done
heartbeat
for s in 0 1 2; do
  [ "$QUEUE_ABORT" -eq 0 ] && run_row FILLER gate_neox20b_rome_cf_L38_s${s} 260 "$NEEDS" "$ENVP $PY $KG --model $MODEL_DIR --model_dtype bf16 $TP --editor rome $CF $COMMON --lr 0.1 --layer 38 --seed ${s} --out results/gate_neox20b_rome_cf_L38_s${s}.json"
done
heartbeat

# ---------------------------------------------------------------- Post-run (CPU, ALWAYS)
log "---------------- POST-RUN (CPU) ----------------"
$PY - > results/neox20b_validation.json 2>>"$LOG" <<'EOF'
import json, glob, os
t0 = float(open('engine/neox20b_round_start').read().strip())
out = []
for j in sorted(glob.glob('results/*neox20b*.json')):
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
log "post: validation sweep -> results/neox20b_validation.json"

# per-layer 3-seed pooling (mirrors run_gptj/run_u6/run_mquake_law's C3 pattern)
for spec in "C3_neox20b_rome_L22:results/matrices/gate_neox20b_rome_cf_L22_s*.npz" \
            "C3_neox20b_rome_L28:results/matrices/gate_neox20b_rome_cf_L28_s*.npz" \
            "C3_neox20b_rome_L33:results/matrices/gate_neox20b_rome_cf_L33_s*.npz" \
            "C3_neox20b_rome_L38:results/matrices/gate_neox20b_rome_cf_L38_s*.npz"; do
  outn="${spec%%:*}"; glob="${spec#*:}"
  if compgen -G "$glob" >/dev/null; then
    $PY experiments/analyze_matrices.py $glob --metric logit --known --edit_ok \
      --out "results/${outn}_neox20b.json" >> "$LOG" 2>&1 && log "post: ${outn}_neox20b done" || log "post: ${outn}_neox20b FAIL"
  fi
done

# S x C law-replication table across the 4-layer band (mirrors run_gptj/run_mquake_law's
# headline table)
if compgen -G "results/matrices/gate_neox20b_rome_cf_L*_s*.npz" >/dev/null; then
  $PY experiments/mechanism_sc_table.py \
    --npz 'results/matrices/gate_neox20b_rome_cf_L*_s*.npz' \
    --known --edit_ok \
    --out results/NEOX20B_mechanism_sc_table.json >> "$LOG" 2>&1 \
    && log "post: NEOX20B_mechanism_sc_table done" || log "post: NEOX20B_mechanism_sc_table FAIL"
fi

# causal aggregation at the peak layer (rome vs alpha-holdout), mirrors run_gptj/
# run_8bcausal's block
if [ -f experiments/aggregate_g4_causal.py ] \
   && compgen -G "results/matrices/g4_neox20b_alphaHO_cf_L33_s0.npz" >/dev/null; then
  tmp_out="results/.C4_causal_neox20b_table.json.tmp"
  $PY experiments/aggregate_g4_causal.py \
    --rome_glob 'results/matrices/gate_neox20b_rome_cf_L{L}_s0.npz' \
    --alpha_glob 'results/matrices/g4_neox20b_alphaHO_cf_L{L}_s0.npz' \
    --layers 33 --known --edit_ok --proj_source holdout \
    --out "$tmp_out" >> "$LOG" 2>&1 \
    && mv "$tmp_out" results/C4_causal_neox20b_table.json \
    && log "post: C4_causal_neox20b_table done (atomic)" \
    || { rm -f "$tmp_out"; log "FAIL aggregate_g4_causal neox20b"; }
else
  log "skip C4-neox20b (aggregate_g4_causal.py or holdout alpha npz missing)"
fi

{
  echo "RUN_NEOX20B REPORT $(date '+%F %T')  jobs: ${n_done} done / ${n_fail} fail / ${n_skip} skipped  elapsed $(elapsed_min)m/${BUDGET_MIN}m  abort=${QUEUE_ABORT}"
  grep -E 'RUN |done |FAIL |SKIP|ABORT|PROGRESS|THERMAL|equiv-gate|integrity|MODEL-ABSENT|tp-onbox-smoke|neox20b-smoke' "$LOG" | tail -80
} > engine/run_neox20b_report.txt
log "================ RUN_NEOX20B COMPLETE (${n_done} done / ${n_fail} fail / ${n_skip} skip) ================"
echo "RUN_NEOX20B_DONE" >> "$LOG"
