#!/bin/bash
# run_revins.sh — B6 revision-insurance driver (2026-07-11). Template = run_instruct.sh /
# run_u6.sh (verbatim skeleton: preflight, GPU-idle gate, run_row/validate/heartbeat,
# CPU post-run), REVINS-namespaced (own pid/log/markers — never reuses u6/instruct/gptj/
# 8bcausal/mquake_law/gradsim_true names). B6 is already SUBMITTED to IEEE (memory:
# b6-submitted-ieee-policy-shift-20260710); this driver pre-runs the LOCAL, ZERO-DOWNLOAD
# cells a reviewer is most likely to probe, so a revision request can be answered from
# already-computed artifacts instead of a cold GPU queue. BUILD-ONLY as authored 2026-07-11:
# the local GPU is busy with P2/P3 work at authoring time — this driver is verified CPU-side
# (bash -n, DRYRUN) and NOT launched by the authoring agent.
#
# FOUR CELLS (independent; each CONFIG-skips cleanly on its own missing model/data so the
# driver always completes and reports, even if e.g. MQuAKE or GPT-J turn out to be absent):
#
#   CELL A — MQuAKE holdout-projector causal cell. The on-disk MQuAKE causal table
#     (results/C4_causal_mquake_table_3seed_probesrc.json) uses --alpha_proj_source probes
#     BY DESIGN, which is circular (memory: c4-alphaedit-projector-circularity.md — the
#     projector is fit on the SAME probes whose damage it's then scored against). Adds the
#     HONEST holdout variant at L12 x s0/1/2 (killgate_keygeom.py --alpha_proj_source
#     holdout), matched against the ALREADY-EXISTING gate_llama1b_rome_mquake_L12_s{0,1,2}
#     .npz (run_mquake_law.sh, verified present), aggregated with aggregate_g4_causal.py
#     --proj_source holdout -> results/C4_causal_mquake_holdout_table_3seed.json. Output
#     tags are NEW (g4_llama1b_alphaHO_mquake_L12_s*) — the existing gate_llama1b_alpha_
#     mquake_L12_s* files are the probes-sourced ones and must NOT be overwritten/reused.
#
#   CELL B — GRACE EGL seed parity. Only egl_llama1b_grace_cf_L12_s0 exists today (1 seed).
#     Adds egl_llama1b_grace_cf_L12_s{1,2}, exact invocation mirrored from run_u6.sh's s0
#     row; needs=the EXISTING s0 npz as its own proof-of-concept gate (same "reuse a prior
#     success instead of re-smoking" pattern as Cell D's gptj rows), not a fresh smoke.
#     REVISION 2026-07-11 (review MAJOR-2): this cell ORIGINALLY also planned a 12-row
#     "grace GATE band" (gate_llama1b_grace_cf_L{8,10,12,14}_s{0,1,2}, the same layer x seed
#     band every other editor has) to build a within-probe-rho table for grace, mirroring
#     the editor/architecture dissociation story (memory: b6-spinoffs-d1-d3-and-second-
#     paper.md). DROPPED after inspecting the ALREADY-EXISTING egl_llama1b_grace_cf_L12_s0
#     .npz: damage_logit is IDENTICALLY ZERO across all 200x500 entries (verified via
#     `np.all(D==0)` -> True), not just norm_growth. This is a stronger and more basic
#     degeneracy than the norm_growth≡0 S x C caveat already on record (memory: journal-
#     first-infra-built-20260706.md) -- grace's codebook mechanism means any probe that
#     doesn't cosine-match an edited key sees the base model's UNCHANGED output (ΔW==0,
#     "positions with no match pass W's real output" per editors/grace_editor.py), so
#     collateral damage on unrelated probes is trivially, deterministically zero. A within-
#     probe Spearman(COS, damage) over a constant-zero damage column is undefined (NaN) for
#     every probe -- spending ~5 GPU-hours on the 12-row GATE band would have produced an
#     all-NaN table, not a finding. Post-run below instead computes and reports
#     damage-identically-zero DIRECTLY from whatever grace EGL npz exist (no correlation
#     attempted) -> results/GRACE_damage_report_revins.json. Do not resurrect the GATE band
#     without first re-deriving whether damage is non-degenerate at some other layer/eps_cos.
#
#   CELL C — True-GradSim to 3-seed / multi-layer (tightens A4', currently thin: rank-
#     agreement 0.087 on ONE cell, memory: gradsim-true-result-20260707.md). Adds L12 s1/s2
#     and L8/L10/L14 s0, reusing the ALREADY-EXISTING gate_llama1b_rome_cf_L{8,10,12,14}_s
#     {0,1,2}.npz (verified present, all 12) as the --gate_npz half of gradsim_true.py's
#     EXTERNAL mode -- no killgate probe-damage sweep is re-run.
#     COLLISION HAZARD (flagged, found by inspection, worked around without touching the
#     tool): experiments/mechanism_dump.py names its output `{model_tag}_L{layer}.npz` with
#     NO SEED in the filename (confirmed by reading the source: `tag = model_tag(args.model)`
#     -> `os.path.basename(model_path)`, then `f"{tag}_L{layer}.npz"`). Running it twice at
#     the SAME layer with different seeds would silently overwrite the earlier seed's file --
#     exactly the live-file-overwrite hazard this workspace's memory warns about generically.
#     Worked around via --out_dir, NOT a code change: s0 keeps the existing default
#     results/mechanism/ (unchanged, no seed collision since L8/L10/L14 are new layers there);
#     s1/s2 at L12 go to results/mechanism/s1/ and results/mechanism/s2/ respectively. Real
#     GPU cost measured from engine/*.log on the ONE existing cell: mech_l12_s0 71s,
#     gradsim_true_l12_s0 10s -- both far cheaper than run_gradsim_true.sh's own provisional
#     15m/row estimate; ests below use a padded-but-realistic 3m/2m.
#
#   CELL D — Cross-arch causal seed parity. neox20b/pythia14b/pythia28b causal cells are
#     CLOUD-ONLY models (NOT in data/models/ on this box) -- explicitly OUT OF SCOPE here per
#     the brief; they belong to the cloud-wave builder. GPT-J-6B IS local
#     (data/models/gpt-j-6b, 12GB pytorch_model.bin, integrity-verified previously) and its
#     causal cell (g4_gptj_alphaHO_cf_L21_s0) already exists and is cleanly parameterizable
#     from run_gptj.sh/run_8bcausal.sh's own invocation (just --seed 1/2) -- NOT deferred to
#     a CONFIG-skip stub. Matched ROME references at L21 s1/s2 ALREADY EXIST (run_gptj.sh),
#     so this is 2 new AlphaEdit(holdout) rows -> results/C4_causal_gptj_table_3seed.json.
#     Reuses the SHARED engine/r3_equiv_bf16.ok marker per run_8bcausal.sh's own precedent
#     comment (it certifies a fact about killgate's bf16 code path, not about one driver) --
#     re-derives it if stale (predates killgate_keygeom.py or editors/arch_compat.py), exactly
#     mirroring run_gptj.sh's Phase 0a2 freshness check.
#
# DESIGN DEVIATION FROM SINGLE-PURPOSE DRIVERS (flagged): run_8bcausal.sh/run_mquake_law.sh
# hard-pf-abort the WHOLE driver if a prerequisite reference artifact is missing, because
# those drivers are single-purpose. This driver spans 4 independent cells, so per this
# task's explicit requirement ("every cell must CONFIG-skip cleanly... driver always
# completes + reports"), missing MODELS/DATA/reference-artifacts are all SOFT (needs-marker
# or explicit conditional) gates, scoped to the one cell that needs them -- only missing
# CODE/TOOL support (argparse flags, missing scripts) is a hard pf() abort, since that is a
# build-time correctness issue common to every cell, not a runtime data-availability one.
#
# GPU-HOUR ESTIMATE (header; see run_row `est` args below for the per-row breakdown, derived
# from REAL timings in engine/*.log where a precedent cell exists, else anchored to the
# closest same-scale precedent -- flagged provisional where genuinely novel):
#   Cell A:  smoke ~6m  + 3 x 25m (mquake alphaHO L12, anchored on the existing probes-
#            sourced mquake alpha rows: 1260-1991s measured)                    =  81m
#   Cell D:  2 x 50m (gptj alphaHO L21 s1/s2, anchored on measured g4_llama8b_
#            alphaHO/g4_gptj_alphaHO rows: 2630-3293s)                          = 100m
#   Cell C:  5 x (3m mech + 2m gradsim) + 1m CPU smoke, anchored on MEASURED
#            mech_l12_s0=71s / gradsim_true_l12_s0=10s (padded ~2x for margin) =  26m
#   Cell B:  EGL seed parity 2 x 25m (anchored on measured egl_llama1b_grace_
#            cf_L12_s0=1352s), no smoke needed (reuses the existing s0 npz as
#            proof-of-concept)                                                 =  50m
#   TOTAL ≈ 81 + 100 + 26 + 50 = 257 GPU-minutes ≈ 4.3 GPU-hours if run to completion.
# (REVISED 2026-07-11: the original estimate included a 12-row/~315m grace GATE band that
# was DROPPED -- see Cell B header note above; total fell from ~9.6h to ~4.3h.)
# Ordered highest-value-and-cheapest FIRST (A, D, C, B) so a tight BUDGET_MIN window still
# lands every cell.
set -u
H=/home/zeyufu/Desktop/idea-feasibility-analysis/edit-harness
cd "$H" || exit 1
PY=/home/zeyufu/miniconda3/envs/dl/bin/python3
LOG=engine/run_revins.log
BUDGET_MIN=${BUDGET_MIN:-600}
mkdir -p engine results/matrices results/smoke_revins/matrices results/mechanism
echo $$ > engine/run_revins.pid
[ -f engine/revins_round_start ] || stat -c %Y engine/run_revins.pid > engine/revins_round_start
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "================ RUN_REVINS START (pid $$, budget ${BUDGET_MIN}m) ================"

# ---------------------------------------------------------------- Phase 0a: CPU pre-flight (HARD: code/tool presence only)
pf_fail=0
pf(){ if eval "$2"; then log "preflight OK: $1"; else log "PREFLIGHT-FAIL: $1"; pf_fail=1; fi; }
pf "python env" "$PY -c 'import torch, numpy' 2>/dev/null"
pf "counterfact.json" "[ -f data/counterfact.json ]"
pf "alpha_proj_source flag" "grep -q -- '--alpha_proj_source' experiments/killgate_keygeom.py"
pf "dataset mquake wired into killgate" "grep -q -- 'mquake' experiments/killgate_keygeom.py"
pf "egl flag" "grep -q -- '--egl' experiments/killgate_keygeom.py"
pf "editor grace wired into killgate" "grep -qE -- '\"grace\"' experiments/killgate_keygeom.py"
pf "grace_eps_cos flag" "grep -q -- '--grace_eps_cos' experiments/killgate_keygeom.py"
pf "model_dtype flag" "grep -q -- '--model_dtype' experiments/killgate_keygeom.py"
pf "mechanism_dump.py --save_vectors flag" "grep -q -- '--save_vectors' experiments/mechanism_dump.py"
pf "gradsim_true.py present" "[ -f experiments/gradsim_true.py ]"
pf "aggregate_g4_causal.py --proj_source flag" "grep -q -- '--proj_source' experiments/aggregate_g4_causal.py"
pf "analyze_matrices.py" "[ -f experiments/analyze_matrices.py ]"
pf "mechanism_sc_table.py" "[ -f experiments/mechanism_sc_table.py ]"
pf "integrity_check.py" "[ -f experiments/tools/integrity_check.py ]"
pf "model Llama-3.2-1B" "[ -d data/models/Llama-3.2-1B ]"
pf "disk >=20GB free" "[ \$(df --output=avail -BG /home | tail -1 | tr -dc 0-9) -ge 20 ]"
rm -f engine/smoke_revins_*.ok
if [ "$pf_fail" -ne 0 ]; then log "ABORT: preflight failed"; exit 3; fi

# ---------------------------------------------------------------- Phase 0b: GPU idle gate
DRYRUN=${DRYRUN:-0}
if [ "$DRYRUN" -eq 1 ]; then
  log "DRYRUN=1 -- skipping GPU idle gate, printing every run_row/run_row2 call without executing"
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
T0=$(date +%s)

# ---------------------------------------------------------------- Phase 0a2: soft per-cell readiness gates
# MOVED (review MINOR-2, 2026-07-11) to AFTER the DRYRUN gate above and wrapped in
# [DRYRUN != 1]: this block does real (if cheap) work -- writes/removes marker files under
# engine/, and Cell D's branch execs integrity_check.py as a subprocess -- so a true
# plan-only DRYRUN invocation should not touch disk at all. DRYRUN doesn't need these
# markers anyway: run_row/run_row2 return immediately in their DRYRUN branch, before ever
# evaluating `needs`.
if [ "$DRYRUN" -ne 1 ]; then
  # Cell B: grace triple-gate (marker + file + argparse), exact pattern from run_u6.sh --
  # collapsed into ONE derived marker file so every grace row below can use the ordinary
  # `needs=` mechanism instead of a bespoke wrapper function.
  GRACE_ARGPARSE_READY=0
  grep -qE -- '"grace"' experiments/killgate_keygeom.py 2>/dev/null && GRACE_ARGPARSE_READY=1
  GRACE_FILE_READY=0
  { [ -f grace_editor.py ] || [ -f editors/grace.py ] || [ -f editors/grace_editor.py ]; } && GRACE_FILE_READY=1
  GRACE_MARKER_READY=0
  [ -f engine/grace_ready.ok ] && GRACE_MARKER_READY=1
  log "grace gate: marker=${GRACE_MARKER_READY} file=${GRACE_FILE_READY} argparse=${GRACE_ARGPARSE_READY}"
  if [ "$GRACE_MARKER_READY" -eq 1 ] && [ "$GRACE_FILE_READY" -eq 1 ] && [ "$GRACE_ARGPARSE_READY" -eq 1 ]; then
    : > engine/revins_grace_triple.ok
  else
    rm -f engine/revins_grace_triple.ok
  fi

  # Cell D: re-derive engine/gptj_integrity.ok every launch (header-only, no GPU, cheap) --
  # mirrors run_gptj.sh Phase 0a2 exactly; a missing/incomplete download produces a clean
  # CONFIG-SKIP on every Cell-D row via the `needs` mechanism, never a crash.
  rm -f engine/gptj_integrity.ok
  if [ -d data/models/gpt-j-6b ]; then
    $PY experiments/tools/integrity_check.py data/models/gpt-j-6b --expect_params 6.05e9 >> "$LOG" 2>&1 \
      && { : > engine/gptj_integrity.ok; log "integrity OK: gpt-j-6b"; } \
      || log "integrity NOT-READY: gpt-j-6b (download incomplete or corrupt -- Cell D rows CONFIG-skip)"
  else
    log "MODEL-ABSENT: data/models/gpt-j-6b not found -- Cell D rows CONFIG-skip"
  fi
  # engine/r3_equiv_bf16.ok: SHARED marker across drivers (run_8bcausal.sh/run_gptj.sh
  # precedent) -- reuse if fresh (postdates killgate_keygeom.py AND arch_compat.py), else
  # clear it so Cell D's own Phase-A block below re-derives it with one real GPU row.
  if [ -f engine/r3_equiv_bf16.ok ] && [ -f results/matrices/equiv_llama1b_bf16_L12_s0.npz ] \
     && [ "$(stat -c %Y results/matrices/equiv_llama1b_bf16_L12_s0.npz)" -ge "$(stat -c %Y experiments/killgate_keygeom.py)" ] \
     && [ "$(stat -c %Y results/matrices/equiv_llama1b_bf16_L12_s0.npz)" -ge "$(stat -c %Y editors/arch_compat.py)" ]; then
    log "engine/r3_equiv_bf16.ok fresh -- reusing, no GPU spend"
  else
    rm -f engine/r3_equiv_bf16.ok
    log "engine/r3_equiv_bf16.ok absent/stale -- will re-derive in Cell D Phase A (real GPU row)"
  fi
fi

# ---------------------------------------------------------------- helpers (u6/instruct/gptj template, verbatim)
ENVP="env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1"
KG="experiments/killgate_keygeom.py"
CF="--dataset counterfact --data data/counterfact.json"
MQ="--dataset mquake --data data/mquake_cf3k.json"
COMMON="--n_edits 200 --n_probes 500 --steps 20 --save_matrices --matrix_dir results/matrices"
SMK="--n_edits 4 --n_probes 40 --steps 2 --save_matrices --matrix_dir results/smoke_revins/matrices"
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

# validate2 / run_row2: a SECOND family for experiments/mechanism_dump.py and
# experiments/gradsim_true.py rows (Cell C) -- these do NOT produce the killgate
# COS/damage_logit/norm_growth/edit_ok npz schema `validate()` checks, so they get their
# own lightweight schema-appropriate check instead of being force-fit into `validate()`.
validate2(){
  local f="$1"
  case "$f" in
    *.npz)
      $PY - "$f" <<'EOF'
import sys, numpy as np
f = sys.argv[1]
try:
    d = np.load(f)
except Exception as e:
    print(f"VALIDATE2-FAIL npz unreadable: {e}"); sys.exit(1)
if "key_norm" not in d.files:
    print("VALIDATE2-FAIL npz missing key_norm"); sys.exit(1)
arr = d["key_norm"].astype(float)
if arr.size == 0 or np.isnan(arr).all():
    print("VALIDATE2-FAIL key_norm empty/all-NaN"); sys.exit(1)
print("VALIDATE2-OK")
EOF
      ;;
    *.json)
      $PY - "$f" <<'EOF'
import json, sys
f = sys.argv[1]
try:
    d = json.load(open(f))
except Exception as e:
    print(f"VALIDATE2-FAIL json unparseable: {e}"); sys.exit(1)
if not isinstance(d, dict) or not d:
    print("VALIDATE2-FAIL json empty/not-a-dict"); sys.exit(1)
print("VALIDATE2-OK")
EOF
      ;;
    *) echo "VALIDATE2-FAIL unrecognized extension: $f" ;;
  esac
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
  case "$cmd" in *smoke_revins*) outn="results/smoke_revins/matrices/$(basename "${outj%.json}").npz";; esac
  if [ -n "$outj" ] && [ -f "$outj" ] && [ -f "$outn" ]; then
    local pvmode="full"; [ "$class" = "SMOKE" ] && pvmode="smoke"
    if validate "$outj" "$outn" "$pvmode" | grep -q VALIDATE-FAIL; then
      mv "$outj" "$outj.INVALID" 2>/dev/null; mv "$outn" "$outn.INVALID" 2>/dev/null
      log "STALE-INVALID ${tag} -- quarantined; re-running"
    else
      log "skip ${tag} (exists, validated)"
      [ "$class" = "SMOKE" ] && : > "engine/smoke_revins_${tag}.ok"
      return
    fi
  fi
  local mdir; mdir=$(echo "$cmd" | grep -oE -- '--model [^ ]+' | awk '{print $2}')
  case "$mdir" in data/*|/*)
    if [ -n "$mdir" ] && [ ! -d "$mdir" ]; then log "CONFIG-SKIP ${tag} (no model ${mdir})"; n_skip=$((n_skip+1)); return; fi;;
  esac
  local cap=$(( est * 60 * 3 + 1200 ))
  log "RUN ${tag} [${class}] (est ${est}m, cap ${cap}s, elapsed ${now}m) -> engine/revins_${tag}.log"
  local t rc dt
  t=$(date +%s)
  timeout --signal=TERM --kill-after=60 "${cap}s" bash -c "$cmd" >> "engine/revins_${tag}.log" 2>&1
  rc=$?; dt=$(( $(date +%s) - t ))
  local vmode="full"; [ "$class" = "SMOKE" ] && vmode="smoke"
  if [ "$rc" -eq 0 ] && [ -n "$outj" ] && [ -f "$outj" ] && [ -f "$outn" ]; then
    local v; v=$(validate "$outj" "$outn" "$vmode")
    if echo "$v" | grep -q VALIDATE-FAIL; then
      mv "$outj" "$outj.INVALID" 2>/dev/null; mv "$outn" "$outn.INVALID" 2>/dev/null
      log "FAIL ${tag} (${dt}s) OUTPUT-INVALID: ${v}"; n_fail=$((n_fail+1))
    else
      log "done ${tag} (${dt}s) ${v}"; n_done=$((n_done+1)); wedge_fail=0
      [ "$class" = "SMOKE" ] && : > "engine/smoke_revins_${tag}.ok"
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
      log "FAIL ${tag} (rc ${rc}, ${dt}s) FAST/CONFIG -- not counted toward wedge abort"
    fi
  fi
}

# run_row2: mechanism_dump.py / gradsim_true.py family (Cell C). Same budget/needs/
# idempotent-skip/timeout/wedge contract as run_row, but the caller supplies the exact
# output path to check (`outcheck`) instead of run_row inferring a matrices-npz sibling
# from --out, since these two tools' outputs don't live under results/matrices/ and don't
# share the killgate npz schema.
run_row2(){
  local class="$1" tag="$2" est="$3" needs="$4" outcheck="$5"; shift 5; local cmd="$*"
  if [ "$DRYRUN" -eq 1 ]; then
    echo "DRYRUN ${tag} [${class}] est=${est}m needs=${needs} outcheck=${outcheck} cmd: ${cmd}"
    log "DRYRUN ${tag} [${class}] est=${est}m needs=${needs} outcheck=${outcheck} cmd: ${cmd}"
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
  if [ -n "$outcheck" ] && [ -f "$outcheck" ]; then
    if validate2 "$outcheck" | grep -q VALIDATE2-FAIL; then
      mv "$outcheck" "$outcheck.INVALID" 2>/dev/null
      log "STALE-INVALID2 ${tag} -- quarantined; re-running"
    else
      log "skip ${tag} (exists, validated2)"
      [ "$class" = "SMOKE" ] && : > "engine/smoke_revins_${tag}.ok"
      return
    fi
  fi
  local mdir; mdir=$(echo "$cmd" | grep -oE -- '--model [^ ]+' | awk '{print $2}')
  case "$mdir" in data/*|/*)
    if [ -n "$mdir" ] && [ ! -d "$mdir" ]; then log "CONFIG-SKIP ${tag} (no model ${mdir})"; n_skip=$((n_skip+1)); return; fi;;
  esac
  local cap=$(( est * 60 * 3 + 900 ))
  log "RUN ${tag} [${class}] (est ${est}m, cap ${cap}s, elapsed ${now}m) -> engine/revins_${tag}.log"
  local t rc dt
  t=$(date +%s)
  timeout --signal=TERM --kill-after=60 "${cap}s" bash -c "$cmd" >> "engine/revins_${tag}.log" 2>&1
  rc=$?; dt=$(( $(date +%s) - t ))
  if [ "$rc" -eq 0 ] && { [ -z "$outcheck" ] || [ -f "$outcheck" ]; }; then
    local v="VALIDATE2-OK (no outcheck path given)"
    [ -n "$outcheck" ] && v=$(validate2 "$outcheck")
    if echo "$v" | grep -q VALIDATE2-FAIL; then
      [ -n "$outcheck" ] && mv "$outcheck" "$outcheck.INVALID" 2>/dev/null
      log "FAIL ${tag} (${dt}s) OUTPUT-INVALID: ${v}"; n_fail=$((n_fail+1))
    else
      log "done ${tag} (${dt}s) ${v}"; n_done=$((n_done+1)); wedge_fail=0
      [ "$class" = "SMOKE" ] && : > "engine/smoke_revins_${tag}.ok"
    fi
  else
    if [ "$rc" -eq 124 ] || [ "$dt" -ge $(( est * 60 / 2 )) ]; then
      wedge_fail=$((wedge_fail+1)); n_fail=$((n_fail+1))
      log "FAIL ${tag} (rc ${rc}, ${dt}s) WEDGE-LIKE consec=${wedge_fail}/${MAXWEDGE}"
      [ "$wedge_fail" -ge "$MAXWEDGE" ] && { log "ABORT: wedge-like failures"; QUEUE_ABORT=1; }
    else
      n_fail=$((n_fail+1))
      log "FAIL ${tag} (rc ${rc}, ${dt}s) FAST/CONFIG -- not counted toward wedge abort"
    fi
  fi
}
heartbeat(){ log "PROGRESS jobs=${n_done}done/${n_fail}fail/${n_skip}skip elapsed=$(elapsed_min)m budget_left=$(( BUDGET_MIN - $(elapsed_min) ))m"; }

# ---------------------------------------------------------------- Phase 0c: micro-smokes
# alpha+holdout+mquake is a NEW flag combination never run before. grace+egl was already
# proven by the existing egl_llama1b_grace_cf_L12_s0 cell (Cell B reuses it as its own
# proof-of-concept gate below, review MAJOR-2) -- no separate grace smoke needed now that
# the plain-gate (no-egl) combo is no longer used anywhere in this driver.
run_row SMOKE alphaHO_mquake 6 data/mquake_cf3k.json "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor alpha $MQ $SMK --lr 0.1 --layer 12 --seed 0 --alpha_proj_source holdout --holdout_frac 1.0 --out results/smoke_revins/alphaHO_mquake.json"
run_row2 SMOKE gradsim_true_selfcontained 2 - results/smoke_revins/qwen05b_cpu_smoke.json "$ENVP $PY experiments/gradsim_true.py --model data/models/Qwen2.5-0.5B --layer auto --n_edits 3 --n_probes 8 --steps 2 --device cpu --out results/smoke_revins/qwen05b_cpu_smoke.json"
heartbeat

# ---------------------------------------------------------------- Cell A: MQuAKE holdout-projector causal (L12 x s0/1/2)
NEEDS_A="data/mquake_cf3k.json,engine/smoke_revins_alphaHO_mquake.ok"
for s in 0 1 2; do
  [ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE g4_llama1b_alphaHO_mquake_L12_s${s} 25 "$NEEDS_A" "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor alpha $MQ $COMMON --lr 0.1 --layer 12 --seed ${s} --alpha_proj_source holdout --holdout_frac 1.0 --out results/g4_llama1b_alphaHO_mquake_L12_s${s}.json"
done
heartbeat

# ---------------------------------------------------------------- Cell D: GPT-J causal seed parity (L21 x s1/s2)
# Phase A equivalence-gate re-derivation (only fires if Phase 0a2 above cleared the marker
# as stale/absent -- exact mirror of run_gptj.sh's own Phase A block).
[ "$QUEUE_ABORT" -eq 0 ] && [ ! -f engine/r3_equiv_bf16.ok ] && run_row SCIENCE equiv_llama1b_bf16_L12_s0 22 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --model_dtype bf16 --editor rome $CF $COMMON --lr 0.1 --layer 12 --seed 0 --out results/equiv_llama1b_bf16_L12_s0.json"
if [ "$DRYRUN" -ne 1 ] && [ ! -f engine/r3_equiv_bf16.ok ] \
   && [ -f results/matrices/equiv_llama1b_bf16_L12_s0.npz ] && [ -f results/matrices/gate_llama1b_rome_cf_L12_s0.npz ]; then
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
print(f"[revins equiv-gate] fp32 rho={r_fp32:+.4f} bf16 rho={r_bf16:+.4f} |drho|={d:.4f} bar=0.02")
if d < 0.02:
    open('engine/r3_equiv_bf16.ok', 'w').close()
    print("[revins equiv-gate] PASS -- Cell D admitted")
else:
    print("[revins equiv-gate] FAIL -- Cell D rows stay CONFIG-skipped; investigate before any GPT-J claim")
EOF
fi
NEEDS_D="engine/gptj_integrity.ok,engine/r3_equiv_bf16.ok,results/matrices/g4_gptj_alphaHO_cf_L21_s0.npz"
for s in 1 2; do
  [ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE g4_gptj_alphaHO_cf_L21_s${s} 50 "$NEEDS_D" "$ENVP $PY $KG --model data/models/gpt-j-6b --model_dtype bf16 --editor alpha $CF $COMMON --lr 0.1 --layer 21 --seed ${s} --alpha_proj_source holdout --holdout_frac 1.0 --out results/g4_gptj_alphaHO_cf_L21_s${s}.json"
done
heartbeat

# ---------------------------------------------------------------- Cell C: True-GradSim seed/layer expansion
NEEDS_C="engine/smoke_revins_gradsim_true_selfcontained.ok"
# (layer, seed, out_dir) triples: L12 s1/s2 (new seeds, needs its own out_dir -- see header
# collision-hazard note); L8/L10/L14 s0 (new layers, default out_dir is safe -- s0 there
# has never written to those layer filenames before).
for spec in "12:1:results/mechanism/s1" "12:2:results/mechanism/s2" \
            "8:0:results/mechanism" "10:0:results/mechanism" "14:0:results/mechanism"; do
  L="${spec%%:*}"; rest="${spec#*:}"; s="${rest%%:*}"; outdir="${rest#*:}"
  mechnpz="${outdir}/Llama-3.2-1B_L${L}.npz"
  gatenpz="results/matrices/gate_llama1b_rome_cf_L${L}_s${s}.npz"
  [ "$QUEUE_ABORT" -eq 0 ] && run_row2 SCIENCE mech_L${L}_s${s} 3 "$NEEDS_C" "$mechnpz" "$ENVP $PY experiments/mechanism_dump.py --model data/models/Llama-3.2-1B --data data/counterfact.json --dataset counterfact --n_edits 200 --layer ${L} --seed ${s} --steps 20 --lr 0.1 --device cuda --save_vectors --out_dir ${outdir}"
  [ "$QUEUE_ABORT" -eq 0 ] && run_row2 SCIENCE gradsim_true_L${L}_s${s} 2 "$mechnpz" "results/GRADSIM_TRUE_Llama-3.2-1B_L${L}_s${s}.json" "$ENVP $PY experiments/gradsim_true.py --model data/models/Llama-3.2-1B --layer ${L} --seed ${s} --n_edits 200 --n_probes 500 --gate_npz ${gatenpz} --mech_npz ${mechnpz} --known --edit_ok --device cuda --out results/GRADSIM_TRUE_Llama-3.2-1B_L${L}_s${s}.json"
done
heartbeat

# ---------------------------------------------------------------- Cell B: grace EGL seed parity (L12 x s1/s2)
# needs= the EXISTING s0 npz as this combo's own proof-of-concept gate (same "reuse a prior
# success instead of re-smoking" pattern as Cell D's gptj rows) -- no fresh smoke needed.
# The 12-row grace GATE band originally planned here was DROPPED (review MAJOR-2, see
# header): damage_logit is identically zero in the existing grace EGL npz, so a within-
# probe-rho table over it would be all-NaN, not a finding. See the post-run damage-report
# block below for the direct (non-correlation) report instead.
NEEDS_B="engine/revins_grace_triple.ok,results/matrices/egl_llama1b_grace_cf_L12_s0.npz"
for s in 1 2; do
  [ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE egl_llama1b_grace_cf_L12_s${s} 25 "$NEEDS_B" "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor grace --egl $CF $COMMON --lr 0.1 --layer 12 --seed ${s} --out results/egl_llama1b_grace_cf_L12_s${s}.json"
done
heartbeat

# ---------------------------------------------------------------- Post-run (CPU, only on a real launch)
# Wrapped in [DRYRUN != 1] (review MEDIUM-1, 2026-07-11): every step below either writes to
# results/ or execs a subprocess. A DRYRUN invocation exists purely to print the row plan --
# it should leave the results/ tree byte-for-byte untouched, not just GPU-untouched.
if [ "$DRYRUN" -ne 1 ]; then
log "---------------- POST-RUN (CPU) ----------------"
tmp_val="results/.revins_validation.json.tmp"
if $PY - > "$tmp_val" 2>>"$LOG" <<'EOF'
import json, glob, os
t0 = float(open('engine/revins_round_start').read().strip())
patterns = [
    'results/g4_llama1b_alphaHO_mquake_L12_s*.json',
    'results/g4_gptj_alphaHO_cf_L21_s*.json',
    'results/GRADSIM_TRUE_Llama-3.2-1B_L*_s*.json',
    'results/egl_llama1b_grace_cf_L12_s*.json',
]
seen = set()
out = []
for pat in patterns:
    for j in sorted(glob.glob(pat)):
        if j in seen:
            continue
        seen.add(j)
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
then
  mv "$tmp_val" results/revins_validation.json
  log "post: validation sweep -> results/revins_validation.json"
else
  rm -f "$tmp_val"
  log "FAIL validation sweep"
fi

# Cell A aggregation (review MAJOR-1, 2026-07-11): gate on ALL THREE alpha-holdout seeds
# explicitly present via -f, not "any seed matches the s* glob" -- a partial/aborted run
# (or a DRYRUN artifact, or a re-run interrupted mid-Cell-A) must NOT produce a table named
# "_3seed" with 1 or 2 seeds actually in it. Also covers review MEDIUM-2 (don't aggregate
# partial after QUEUE_ABORT) for free: if QUEUE_ABORT fired mid-Cell-A, at least one seed
# npz is missing and this gate fails closed. The rome-side seeds are pre-existing data
# (run_mquake_law.sh) and already all present, but checked explicitly for symmetry/defense.
if [ -f experiments/aggregate_g4_causal.py ] \
   && [ -f results/matrices/g4_llama1b_alphaHO_mquake_L12_s0.npz ] \
   && [ -f results/matrices/g4_llama1b_alphaHO_mquake_L12_s1.npz ] \
   && [ -f results/matrices/g4_llama1b_alphaHO_mquake_L12_s2.npz ] \
   && [ -f results/matrices/gate_llama1b_rome_mquake_L12_s0.npz ] \
   && [ -f results/matrices/gate_llama1b_rome_mquake_L12_s1.npz ] \
   && [ -f results/matrices/gate_llama1b_rome_mquake_L12_s2.npz ]; then
  tmp_out="results/.C4_causal_mquake_holdout_table_3seed.json.tmp"
  $PY experiments/aggregate_g4_causal.py \
    --rome_glob 'results/matrices/gate_llama1b_rome_mquake_L{L}_s*.npz' \
    --alpha_glob 'results/matrices/g4_llama1b_alphaHO_mquake_L{L}_s*.npz' \
    --layers 12 --known --edit_ok --proj_source holdout \
    --out "$tmp_out" >> "$LOG" 2>&1 \
    && mv "$tmp_out" results/C4_causal_mquake_holdout_table_3seed.json \
    && log "post: C4_causal_mquake_holdout_table_3seed done (atomic, all 3 seeds confirmed)" \
    || { rm -f "$tmp_out"; log "FAIL aggregate_g4_causal mquake-holdout"; }
else
  log "skip C4-mquake-holdout (not all 3 alpha-holdout seeds present yet -- avoiding a misleadingly-named partial-seed _3seed table)"
fi

# Cell D aggregation (same all-seeds-present gate, same reasoning)
if [ -f experiments/aggregate_g4_causal.py ] \
   && [ -f results/matrices/g4_gptj_alphaHO_cf_L21_s0.npz ] \
   && [ -f results/matrices/g4_gptj_alphaHO_cf_L21_s1.npz ] \
   && [ -f results/matrices/g4_gptj_alphaHO_cf_L21_s2.npz ] \
   && [ -f results/matrices/gate_gptj_rome_cf_L21_s0.npz ] \
   && [ -f results/matrices/gate_gptj_rome_cf_L21_s1.npz ] \
   && [ -f results/matrices/gate_gptj_rome_cf_L21_s2.npz ]; then
  tmp_out="results/.C4_causal_gptj_table_3seed.json.tmp"
  $PY experiments/aggregate_g4_causal.py \
    --rome_glob 'results/matrices/gate_gptj_rome_cf_L{L}_s*.npz' \
    --alpha_glob 'results/matrices/g4_gptj_alphaHO_cf_L{L}_s*.npz' \
    --layers 21 --known --edit_ok --proj_source holdout \
    --out "$tmp_out" >> "$LOG" 2>&1 \
    && mv "$tmp_out" results/C4_causal_gptj_table_3seed.json \
    && log "post: C4_causal_gptj_table_3seed done (atomic, all 3 seeds confirmed)" \
    || { rm -f "$tmp_out"; log "FAIL aggregate_g4_causal gptj-3seed"; }
else
  log "skip C4-gptj-3seed (not all 3 alpha-holdout seeds present yet -- avoiding a misleadingly-named partial-seed _3seed table)"
fi

# Cell B: DIRECT damage-report (review MAJOR-2, 2026-07-11) -- NOT a within-probe-rho table.
# Reads whatever egl_llama1b_grace_cf_L12_s*.npz currently exist (1-3 depending on how much
# of Cell B has run) and reports whether collateral damage on unrelated probes is
# identically zero, with the same known+edit_ok filter used everywhere else in this driver.
# Deliberately does NOT call analyze_matrices.py / mechanism_sc_table.py on grace matrices
# -- see header SCIENCE CAVEAT (grace norm_growth≡0 makes S x C degenerate; damage_logic≡0
# makes even the plain within-probe rho undefined/NaN, mirrors run_u6.sh's MEMIT exclusion).
if compgen -G "results/matrices/egl_llama1b_grace_cf_L12_s*.npz" >/dev/null; then
  tmp_out="results/.GRACE_damage_report_revins.json.tmp"
  $PY - "$tmp_out" results/matrices/egl_llama1b_grace_cf_L12_s*.npz >> "$LOG" 2>&1 <<'EOF'
import json, sys, glob
import numpy as np
tmp_out = sys.argv[1]
npz_paths = sorted(sys.argv[2:])
per_seed = []
for p in npz_paths:
    d = np.load(p)
    D = d['damage_logit'].astype(float)
    mask = (d['edit_ok'].astype(float) > 0)[:, None] & (d['pre_p'].astype(float) > 0.05)[None, :]
    Df = D[mask] if mask.any() else D.ravel()
    per_seed.append({
        "npz": p,
        "n_edits": int(D.shape[0]), "n_probes": int(D.shape[1]),
        "n_filtered_pairs": int(Df.size),
        "max_abs_damage": round(float(np.max(np.abs(Df))), 8) if Df.size else None,
        "frac_nonzero": round(float(np.count_nonzero(Df) / Df.size), 8) if Df.size else None,
        "damage_identically_zero": bool(Df.size and np.all(Df == 0)),
    })
overall_zero = bool(per_seed) and all(r["damage_identically_zero"] for r in per_seed)
report = {
    "cell": "B (grace)",
    "statistic": "damage_logit identically-zero check (NOT within-probe Spearman -- see run_revins.sh header)",
    "reason": "grace's ΔW==0 codebook mechanism leaves non-codebook-matched probes bit-identical to "
              "the base model; a within-probe rho over a constant damage column is undefined (NaN).",
    "n_seeds_found": len(per_seed),
    "damage_identically_zero_all_seeds": overall_zero,
    "per_seed": per_seed,
}
with open(tmp_out, "w") as f:
    json.dump(report, f, indent=1)
print(f"[revins grace damage-report] {len(per_seed)} seed(s), all-zero={overall_zero}")
EOF
  if [ -f "$tmp_out" ]; then
    mv "$tmp_out" results/GRACE_damage_report_revins.json
    log "post: GRACE_damage_report_revins done (atomic)"
  else
    log "FAIL GRACE_damage_report_revins"
  fi
else
  log "skip GRACE_damage_report_revins (no egl_llama1b_grace_cf_L12_s*.npz found)"
fi
# Deliberately NOT building a GRACE mechanism_sc_table.json here -- see header SCIENCE
# CAVEAT (grace norm_growth≡0, mirrors run_u6.sh's own MEMIT S x C exclusion).

# ---------------------------------------------------------------- Manifest (atomic tmp+mv)
$PY - <<'EOF' >> "$LOG" 2>&1
import json, os

def check(path):
    if not os.path.exists(path):
        return {"path": path, "present": False, "valid": None}
    if path.endswith(".npz"):
        try:
            import numpy as np
            d = np.load(path)
            ok = len(d.files) > 0 and not all(
                np.isnan(d[k].astype(float)).all() for k in d.files
                if np.issubdtype(d[k].dtype, np.number)
            )
        except Exception:
            ok = False
    else:
        try:
            d = json.load(open(path))
            ok = isinstance(d, dict) and len(d) > 0
        except Exception:
            ok = False
    return {"path": path, "present": True, "valid": bool(ok)}

targets = []
# Cell A
for s in (0, 1, 2):
    targets.append(f"results/g4_llama1b_alphaHO_mquake_L12_s{s}.json")
    targets.append(f"results/matrices/g4_llama1b_alphaHO_mquake_L12_s{s}.npz")
targets.append("results/C4_causal_mquake_holdout_table_3seed.json")
# Cell B
for s in (1, 2):
    targets.append(f"results/egl_llama1b_grace_cf_L12_s{s}.json")
    targets.append(f"results/matrices/egl_llama1b_grace_cf_L12_s{s}.npz")
targets.append("results/GRACE_damage_report_revins.json")
# Cell C
targets.append("results/mechanism/s1/Llama-3.2-1B_L12.npz")
targets.append("results/mechanism/s2/Llama-3.2-1B_L12.npz")
for L in (8, 10, 14):
    targets.append(f"results/mechanism/Llama-3.2-1B_L{L}.npz")
for L, s in ((12, 1), (12, 2), (8, 0), (10, 0), (14, 0)):
    targets.append(f"results/GRADSIM_TRUE_Llama-3.2-1B_L{L}_s{s}.json")
# Cell D
for s in (1, 2):
    targets.append(f"results/g4_gptj_alphaHO_cf_L21_s{s}.json")
    targets.append(f"results/matrices/g4_gptj_alphaHO_cf_L21_s{s}.npz")
targets.append("results/C4_causal_gptj_table_3seed.json")

rows = [check(t) for t in targets]
n_present = sum(1 for r in rows if r["present"])
n_valid = sum(1 for r in rows if r["valid"])
manifest = {
    "generated_by": "run_revins.sh",
    "n_targets": len(rows),
    "n_present": n_present,
    "n_valid": n_valid,
    "rows": rows,
}
tmp = "results/.REVINS_manifest.json.tmp"
with open(tmp, "w") as f:
    json.dump(manifest, f, indent=1)
os.replace(tmp, "results/REVINS_manifest.json")
print(f"[revins manifest] {n_present}/{len(rows)} present, {n_valid}/{len(rows)} valid -> results/REVINS_manifest.json")
EOF
log "post: manifest -> results/REVINS_manifest.json"
fi

{
  echo "RUN_REVINS REPORT $(date '+%F %T')  jobs: ${n_done} done / ${n_fail} fail / ${n_skip} skipped  elapsed $(elapsed_min)m/${BUDGET_MIN}m  abort=${QUEUE_ABORT}"
  grep -E 'RUN |done |FAIL |SKIP|ABORT|PROGRESS|THERMAL|grace gate|equiv-gate|integrity|MODEL-ABSENT|manifest' "$LOG" | tail -100
} > engine/run_revins_report.txt
log "================ RUN_REVINS COMPLETE (${n_done} done / ${n_fail} fail / ${n_skip} skip) ================"
echo "RUN_REVINS_DONE" >> "$LOG"
