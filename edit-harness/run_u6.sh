#!/bin/bash
# run_u6.sh — EGLSEEDS grid + gap-closers (2026-07-05). Template = run_u5.sh (verbatim
# skeleton), U6-namespaced (own pid/log/markers — never reuses u5 names). SHELL-ONLY driver
# authored by a CPU-side agent; NOT launched by that agent. Fills three review-flagged gaps:
#   G. EGLSEEDS: complete the 5-editor (rome/memit/alpha/ft/ft_kl) x 3-seed EGL grid at the
#      canonical EGL layer (L12, per run_u5 Block F). rome/memit/alpha already have s0,s1
#      (run_u5) — only s2 is missing for those three; ft and ft_kl have NO egl rows yet.
#   M. MEMIT SEED PARITY: L10 and L14 are s0-only (L8, L12 already 3-seed). Adds s1/s2.
#   Z. zsRE-DELETION SEEDS: run_u5 produced the first-ever rome+delete+zsre cell at s0 only.
#      Adds s1/s2 (matched zsre insertion refs already exist at s0/s1/s2 — checked in preflight).
#   3. Llama-3.2-3B DELETION: no 3B deletion cell exists anywhere in the harness. Mirrors the
#      1B deletion flags onto the existing 3B regime-law row (L24, matches gate_llama3b_rome_
#      cf_L24_s0.npz from run_r3.sh, which becomes the matched-insertion reference).
#   6. OPTIONAL grace tail row — gated behind BOTH a ready-marker from the parallel EDITOR6
#      build AND a killgate_keygeom.py argparse check (grace is NOT in --editor choices as of
#      2026-07-05; a marker-only gate would still hard-fail at argparse, so this uses an
#      explicit conditional instead of the `needs` marker mechanism, which only stats files).
# GUESSED CONVENTIONS (flagged per instructions — verify before trusting):
#   - ft_kl mid-ladder weight: instructions said use 0.3 if unclear; existing ladder covers
#     0.03/0.3/1.0, so 0.3 (tag "ftkl030") is both the instructed default AND already-canonical.
#   - est minutes for egl_ft (~25m) and egl_ftkl030 (~30m): no ft/ft_kl EGL cell has ever run;
#     derived from plain-ft (21-28m, run_deep_v2/until1900) and ftkl (27-31m, run_u2/u4/u5)
#     rows, since EGL adds ~0 measured overhead for rome/memit/alpha (run_u5 L230-232 ests
#     match their non-egl twins). NOT measured — treat as provisional.
#   - grace row gate checks grace_editor.py, editors/grace.py, AND editors/grace_editor.py
#     (repo convention is editors/<name>.py, e.g. editors/{alphaedit,memit,ft_editor,
#     rome_native}.py; the orchestrator's brief said "grace_editor.py"). VERIFIED 2026-07-05:
#     the parallel EDITOR6 build landed at editors/grace_editor.py and wired --editor grace
#     into killgate_keygeom.py's choices — the gate correctly recognizes it now (argparse
#     check passes; file check passes; only engine/grace_ready.ok is still absent, so the
#     row still CONFIG-skips until that driver's own readiness marker appears).
set -u
H=/home/zeyufu/Desktop/idea-feasibility-analysis/edit-harness
cd "$H" || exit 1
PY=/home/zeyufu/miniconda3/envs/dl/bin/python3
LOG=engine/run_u6.log
BUDGET_MIN=${BUDGET_MIN:-560}
mkdir -p engine results/matrices results/smoke_u6/matrices
echo $$ > engine/run_u6.pid
[ -f engine/u6_round_start ] || stat -c %Y engine/run_u6.pid > engine/u6_round_start
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "================ RUN_U6 START (pid $$, budget ${BUDGET_MIN}m) ================"

# ---------------------------------------------------------------- Phase 0a: CPU pre-flight
pf_fail=0
pf(){ if eval "$2"; then log "preflight OK: $1"; else log "PREFLIGHT-FAIL: $1"; pf_fail=1; fi; }
pf "python env" "$PY -c 'import torch, numpy' 2>/dev/null"
pf "counterfact.json" "[ -f data/counterfact.json ]"
pf "zsre_eval.json" "[ -f data/zsre_eval.json ]"
pf "edit_mode flag" "grep -q -- '--edit_mode' experiments/killgate_keygeom.py"
pf "egl flag" "grep -q -- '--egl' experiments/killgate_keygeom.py"
pf "ft_kl flag" "grep -q -- '--ft_kl' experiments/killgate_keygeom.py"
pf "model Llama-3.2-1B" "[ -d data/models/Llama-3.2-1B ]"
pf "model Llama-3.2-3B" "[ -d data/models/Llama-3.2-3B ]"
pf "u1_deletion_gate.py" "[ -f experiments/u1_deletion_gate.py ]"
pf "analyze_matrices.py" "[ -f experiments/analyze_matrices.py ]"
pf "MEMIT L8 s0 (fixed ref)" "[ -f results/matrices/gate_llama1b_memit_cf_L8_s0.npz ]"
pf "MEMIT L10 s0 (to extend)" "[ -f results/matrices/gate_llama1b_memit_cf_L10_s0.npz ]"
pf "MEMIT L14 s0 (to extend)" "[ -f results/matrices/gate_llama1b_memit_cf_L14_s0.npz ]"
pf "EGL rome s0 (to extend)" "[ -f results/matrices/egl_llama1b_rome_cf_L12_s0.npz ]"
pf "EGL memit s1 (to extend)" "[ -f results/matrices/egl_llama1b_memit_cf_L12_s1.npz ]"
pf "EGL alpha s1 (to extend)" "[ -f results/matrices/egl_llama1b_alpha_cf_L12_s1.npz ]"
pf "zsre-delete s0 (to extend)" "[ -f results/matrices/u1e0_llama1b_delete_refusal_zsre_L10_s0.npz ]"
pf "zsre matched ins s1" "[ -f results/matrices/gate_llama1b_rome_zsre_L10_s1.npz ]"
pf "zsre matched ins s2" "[ -f results/matrices/gate_llama1b_rome_zsre_L10_s2.npz ]"
pf "3B matched ins L24 s0" "[ -f results/matrices/gate_llama3b_rome_cf_L24_s0.npz ]"
pf "disk >=20GB free" "[ \$(df --output=avail -BG /home | tail -1 | tr -dc 0-9) -ge 20 ]"
rm -f engine/smoke_u6_*.ok
if [ "$pf_fail" -ne 0 ]; then log "ABORT: preflight failed"; exit 3; fi

# ---------------------------------------------------------------- Phase 0a2: grace tail-row gate
# NOT a `needs` marker (that mechanism only stats files) — grace also needs killgate argparse
# support, which is a code-content check, not a file-existence check. Evaluated once here.
GRACE_ARGPARSE_READY=0
grep -qE -- '"grace"' experiments/killgate_keygeom.py 2>/dev/null && GRACE_ARGPARSE_READY=1
GRACE_FILE_READY=0
{ [ -f grace_editor.py ] || [ -f editors/grace.py ] || [ -f editors/grace_editor.py ]; } && GRACE_FILE_READY=1
GRACE_MARKER_READY=0
[ -f engine/grace_ready.ok ] && GRACE_MARKER_READY=1
log "grace gate: marker=${GRACE_MARKER_READY} file=${GRACE_FILE_READY} argparse=${GRACE_ARGPARSE_READY}"

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

# ---------------------------------------------------------------- helpers (u2/u4/u5 template, verbatim)
ENVP="env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1"
KG="experiments/killgate_keygeom.py"
CF="--dataset counterfact --data data/counterfact.json"
ZS="--dataset zsre --data data/zsre_eval.json"
COMMON="--n_edits 200 --n_probes 500 --steps 20 --save_matrices --matrix_dir results/matrices"
SMK="--n_edits 4 --n_probes 40 --steps 2 --save_matrices --matrix_dir results/smoke_u6/matrices"
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
  case "$cmd" in *smoke_u6*) outn="results/smoke_u6/matrices/$(basename "${outj%.json}").npz";; esac
  if [ -n "$outj" ] && [ -f "$outj" ] && [ -f "$outn" ]; then
    local pvmode="full"; [ "$class" = "SMOKE" ] && pvmode="smoke"
    if validate "$outj" "$outn" "$pvmode" | grep -q VALIDATE-FAIL; then
      mv "$outj" "$outj.INVALID" 2>/dev/null; mv "$outn" "$outn.INVALID" 2>/dev/null
      log "STALE-INVALID ${tag} — quarantined; re-running"
    else
      log "skip ${tag} (exists, validated)"
      [ "$class" = "SMOKE" ] && : > "engine/smoke_u6_${tag}.ok"
      return
    fi
  fi
  local mdir; mdir=$(echo "$cmd" | grep -oE -- '--model [^ ]+' | awk '{print $2}')
  case "$mdir" in data/*|/*)
    if [ -n "$mdir" ] && [ ! -d "$mdir" ]; then log "CONFIG-SKIP ${tag} (no model ${mdir})"; n_skip=$((n_skip+1)); return; fi;;
  esac
  local cap=$(( est * 60 * 3 + 1200 ))
  log "RUN ${tag} [${class}] (est ${est}m, cap ${cap}s, elapsed ${now}m) -> engine/u6_${tag}.log"
  local t rc dt
  t=$(date +%s)
  timeout --signal=TERM --kill-after=60 "${cap}s" bash -c "$cmd" >> "engine/u6_${tag}.log" 2>&1
  rc=$?; dt=$(( $(date +%s) - t ))
  local vmode="full"; [ "$class" = "SMOKE" ] && vmode="smoke"
  if [ "$rc" -eq 0 ] && [ -n "$outj" ] && [ -f "$outj" ] && [ -f "$outn" ]; then
    local v; v=$(validate "$outj" "$outn" "$vmode")
    if echo "$v" | grep -q VALIDATE-FAIL; then
      mv "$outj" "$outj.INVALID" 2>/dev/null; mv "$outn" "$outn.INVALID" 2>/dev/null
      log "FAIL ${tag} (${dt}s) OUTPUT-INVALID: ${v}"; n_fail=$((n_fail+1))
    else
      log "done ${tag} (${dt}s) ${v}"; n_done=$((n_done+1)); wedge_fail=0
      [ "$class" = "SMOKE" ] && : > "engine/smoke_u6_${tag}.ok"
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

# ---------------------------------------------------------------- Phase 0c: micro-smokes
# ft/ft_kl have never run with --egl combined; 3B has never run with --edit_mode delete.
run_row SMOKE egl_ft_smoke 4 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor ft --egl $CF $SMK --ft_lr 5e-3 --layer 12 --seed 0 --out results/smoke_u6/egl_ft.json"
run_row SMOKE egl_ftkl_smoke 4 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor ft --egl $CF $SMK --ft_lr 5e-3 --ft_kl 0.3 --ft_kl_n 5 --layer 12 --seed 0 --out results/smoke_u6/egl_ftkl.json"
run_row SMOKE llama3b_delete_smoke 6 - "$ENVP $PY $KG --model data/models/Llama-3.2-3B --editor rome --edit_mode delete --delete_variant refusal $CF $SMK --lr 0.1 --layer 24 --seed 0 --out results/smoke_u6/llama3b_delete.json"
heartbeat

# ---------------------------------------------------------------- Block G: EGLSEEDS grid (L12)
# rome/memit/alpha: only s2 is missing (s0/s1 done in run_u5 Block F). ft/ft_kl: all 3 seeds new.
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE egl_llama1b_rome_cf_L12_s2 30 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome --egl $CF $COMMON --lr 0.1 --layer 12 --seed 2 --out results/egl_llama1b_rome_cf_L12_s2.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE egl_llama1b_memit_cf_L12_s2 35 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor memit --egl $CF $COMMON --lr 0.1 --layer 12 --seed 2 --out results/egl_llama1b_memit_cf_L12_s2.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE egl_llama1b_alpha_cf_L12_s2 30 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor alpha --egl $CF $COMMON --lr 0.1 --layer 12 --seed 2 --out results/egl_llama1b_alpha_cf_L12_s2.json"
heartbeat
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE egl_llama1b_ft_cf_L12_s0 25 engine/smoke_u6_egl_ft_smoke.ok "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor ft --egl $CF $COMMON --ft_lr 5e-3 --layer 12 --seed 0 --out results/egl_llama1b_ft_cf_L12_s0.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE egl_llama1b_ft_cf_L12_s1 25 engine/smoke_u6_egl_ft_smoke.ok "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor ft --egl $CF $COMMON --ft_lr 5e-3 --layer 12 --seed 1 --out results/egl_llama1b_ft_cf_L12_s1.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE egl_llama1b_ft_cf_L12_s2 25 engine/smoke_u6_egl_ft_smoke.ok "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor ft --egl $CF $COMMON --ft_lr 5e-3 --layer 12 --seed 2 --out results/egl_llama1b_ft_cf_L12_s2.json"
heartbeat
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE egl_llama1b_ftkl030_cf_L12_s0 30 engine/smoke_u6_egl_ftkl_smoke.ok "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor ft --egl $CF $COMMON --ft_lr 5e-3 --ft_kl 0.3 --ft_kl_n 5 --layer 12 --seed 0 --out results/egl_llama1b_ftkl030_cf_L12_s0.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE egl_llama1b_ftkl030_cf_L12_s1 30 engine/smoke_u6_egl_ftkl_smoke.ok "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor ft --egl $CF $COMMON --ft_lr 5e-3 --ft_kl 0.3 --ft_kl_n 5 --layer 12 --seed 1 --out results/egl_llama1b_ftkl030_cf_L12_s1.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE egl_llama1b_ftkl030_cf_L12_s2 30 engine/smoke_u6_egl_ftkl_smoke.ok "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor ft --egl $CF $COMMON --ft_lr 5e-3 --ft_kl 0.3 --ft_kl_n 5 --layer 12 --seed 2 --out results/egl_llama1b_ftkl030_cf_L12_s2.json"
heartbeat
# optional 6th-editor tail row (see Phase 0a2 gate above) — explicit conditional, not `needs`
if [ "$QUEUE_ABORT" -eq 0 ] && [ "$GRACE_MARKER_READY" -eq 1 ] && [ "$GRACE_FILE_READY" -eq 1 ] && [ "$GRACE_ARGPARSE_READY" -eq 1 ]; then
  run_row SCIENCE egl_llama1b_grace_cf_L12_s0 30 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor grace --egl $CF $COMMON --lr 0.1 --layer 12 --seed 0 --out results/egl_llama1b_grace_cf_L12_s0.json"
else
  log "CONFIG-SKIP egl_llama1b_grace_cf_L12_s0 (grace not ready: marker=${GRACE_MARKER_READY} file=${GRACE_FILE_READY} argparse=${GRACE_ARGPARSE_READY})"
  n_skip=$((n_skip+1))
fi
heartbeat

# ---------------------------------------------------------------- Block M: MEMIT seed parity (L10/L14)
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE gate_llama1b_memit_cf_L10_s1 32 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor memit $CF $COMMON --lr 0.1 --layer 10 --seed 1 --out results/gate_llama1b_memit_cf_L10_s1.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE gate_llama1b_memit_cf_L10_s2 32 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor memit $CF $COMMON --lr 0.1 --layer 10 --seed 2 --out results/gate_llama1b_memit_cf_L10_s2.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE gate_llama1b_memit_cf_L14_s1 32 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor memit $CF $COMMON --lr 0.1 --layer 14 --seed 1 --out results/gate_llama1b_memit_cf_L14_s1.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE gate_llama1b_memit_cf_L14_s2 32 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor memit $CF $COMMON --lr 0.1 --layer 14 --seed 2 --out results/gate_llama1b_memit_cf_L14_s2.json"
heartbeat

# ---------------------------------------------------------------- Block Z: zsRE-deletion seeds (s1/s2)
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE u1e0_llama1b_delete_refusal_zsre_L10_s1 30 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome --edit_mode delete --delete_variant refusal $ZS $COMMON --lr 0.1 --layer 10 --seed 1 --out results/u1e0_llama1b_delete_refusal_zsre_L10_s1.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE u1e0_llama1b_delete_refusal_zsre_L10_s2 30 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome --edit_mode delete --delete_variant refusal $ZS $COMMON --lr 0.1 --layer 10 --seed 2 --out results/u1e0_llama1b_delete_refusal_zsre_L10_s2.json"
heartbeat

# ---------------------------------------------------------------- Block 3: Llama-3.2-3B deletion cell
# Mirrors gate_llama3b_rome_cf_L24_s0 (run_r3.sh B3) with the 1B deletion flags overlaid.
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE u1e0_llama3b_delete_refusal_L24_s0 55 engine/smoke_u6_llama3b_delete_smoke.ok "$ENVP $PY $KG --model data/models/Llama-3.2-3B --editor rome --edit_mode delete --delete_variant refusal $CF $COMMON --lr 0.1 --layer 24 --seed 0 --out results/u1e0_llama3b_delete_refusal_L24_s0.json"
heartbeat

# ---------------------------------------------------------------- Post-run (CPU, ALWAYS)
log "---------------- POST-RUN (CPU) ----------------"
$PY - > results/u6_validation.json 2>>"$LOG" <<'EOF'
import json, glob, os, numpy as np
t0 = float(open('engine/u6_round_start').read().strip())
out = []
for j in sorted(glob.glob('results/*.json')):
    base = os.path.basename(j)[:-5]
    if os.path.getmtime(j) < t0: continue
    if not base.startswith(('egl_llama1b_', 'gate_llama1b_memit_', 'u1e0_llama1b_delete_refusal_zsre_', 'u1e0_llama3b_delete_refusal_')):
        continue
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
log "post: validation sweep -> results/u6_validation.json"

# NOTE: no dedicated cross-seed/cross-editor EGL aggregator exists anywhere in this repo
# (egl_metrics.py has no CLI — it's a library imported by killgate_keygeom.py; the EGL table
# quoted in CLAUDE.md was hand-built). The u6_validation.json sweep above is the only
# post-processing the new egl_* cells get here; a dedicated EGL-by-editor table is a TODO
# for whoever builds it, not fabricated in this driver.

# MEMIT L10/L14 now 3-seed pooled (mirrors run_r3.sh's C3_memit_L8/L12 pattern)
for spec in "C3_memit_L10:results/matrices/gate_llama1b_memit_cf_L10_s*.npz" \
            "C3_memit_L14:results/matrices/gate_llama1b_memit_cf_L14_s*.npz" \
            "C3_u1_zsre_delete_L10:results/matrices/u1e0_llama1b_delete_refusal_zsre_L10_s*.npz"; do
  outn="${spec%%:*}"; glob="${spec#*:}"
  if compgen -G "$glob" >/dev/null; then
    $PY experiments/analyze_matrices.py $glob --metric logit --known --edit_ok \
      --out "results/${outn}_u6.json" >> "$LOG" 2>&1 && log "post: ${outn}_u6 done" || log "post: ${outn}_u6 FAIL"
  fi
done
# Deliberately NOT extending results/C1_mechanism_sc_table.json (mechanism_sc_table.py) with
# MEMIT globs here: memory/findings-MEMIT-SC-RECONCILIATION-2026-07-04.md ruled MEMIT's S x C
# framing DEAD (rho_C 0.019/0.037) — adding MEMIT rows to the S x C table risks resurrecting
# the retracted framing by proximity. MEMIT gets analyze_matrices.py pooling only (above).

# u1_deletion_gate for the new deletion cells (matched-seed insertion refs where available)
for spec in "zsre_refusal_L10_s1:results/matrices/u1e0_llama1b_delete_refusal_zsre_L10_s1.npz:results/matrices/gate_llama1b_rome_zsre_L10_s1.npz" \
            "zsre_refusal_L10_s2:results/matrices/u1e0_llama1b_delete_refusal_zsre_L10_s2.npz:results/matrices/gate_llama1b_rome_zsre_L10_s2.npz" \
            "llama3b_refusal_L24_s0:results/matrices/u1e0_llama3b_delete_refusal_L24_s0.npz:results/matrices/gate_llama3b_rome_cf_L24_s0.npz"; do
  tagn="${spec%%:*}"; rest="${spec#*:}"; del="${rest%%:*}"; ins="${rest#*:}"
  if [ -f "$del" ] && [ -f "$ins" ]; then
    $PY experiments/u1_deletion_gate.py --del_npz "$del" --ins_npz "$ins" --metric logit \
      --out "results/u1_gate_${tagn}.json" >> "$LOG" 2>&1 && log "post: u1_gate_${tagn} done" || log "post: u1_gate_${tagn} FAIL"
  fi
done

{
  echo "RUN_U6 REPORT $(date '+%F %T')  jobs: ${n_done} done / ${n_fail} fail / ${n_skip} skipped  elapsed $(elapsed_min)m/${BUDGET_MIN}m  abort=${QUEUE_ABORT}"
  grep -E 'RUN |done |FAIL |SKIP|ABORT|PROGRESS|THERMAL|grace gate' "$LOG" | tail -60
} > engine/run_u6_report.txt
log "================ RUN_U6 COMPLETE (${n_done} done / ${n_fail} fail / ${n_skip} skip) ================"
echo "RUN_U6_DONE" >> "$LOG"
