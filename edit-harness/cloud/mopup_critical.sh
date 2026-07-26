#!/bin/bash
# cloud/mopup_critical.sh — targeted completion mop-up for either 7-9B/14B/32B wave box
# (2026-07-13). Companion to cloud/run_wave_36039.sh / cloud/run_wave_pro6000.sh, NOT a
# replacement: those run gate-band(all seeds)-THEN-alpha per model, so a budget cut costs
# the alphaHO causal cell (the most valuable one, per docs/plans/ANALYSIS-PLAN-WAVES-
# 20260713.md) and leaves ragged seed coverage. This driver REORDERS the SAME cells to a
# priority that protects the headline gates from a second truncation:
#   P-A: every non-gemma family's alphaHO holdout causal cell(s)          (most at-risk)
#   P-B: seeds s0+s1 of every band layer (>=2-seed within-probe coverage — the gate's floor)
#   P-C: Pro-6000-only cells: llama31_8b fp32/bf16 precision twin L16 s0 + merging RG
#   P-D (budget-permitting only): s2 of the 3-seed bands, THEN gemma9b's WHOLE family
#        (band + causal) — deprioritized to the tail per plan so a second cut can't cost a
#        causal or 2-seed cell.
#
# Layer bands + n_layers are NOT re-hardcoded here (they would drift from the wave drivers
# over time) — spec_36039()/spec_pro6000() are extracted verbatim from spec() in
# run_wave_36039.sh / run_wave_pro6000.sh at run time (see load_spec below), so any edit to
# a driver's bands is picked up automatically. Card sharding (36039: card0 = mistral7b,
# qwen3_8b, gemma9b(late); card1 = qwen25_7b, llama31_8bi) is copied from run_wave_36039.sh's
# header comment, since spec() alone doesn't carry that assignment.
#
# GATES per model: reuses the EXACT smoke-gate (esr>0.9) + CONFIG-skip (num_hidden_layers
# match) checks from the wave drivers, reading the SAME results/smoke36039|smokepro jsons —
# this driver never launches its own smoke job, only reuses what the wave already produced.
#
# OPS (identical shape to the wave drivers): idempotent skip-if-exists on BOTH json AND npz
# (--save_matrices is mandatory — analysis-plan §0 catch: json-only cells are unusable for
# the within-probe gate), --model_dtype bf16 (fp32 only for the precision twin + merging,
# which is fp32-only code), same --out filename conventions, JOB_CAP_MIN per-job timeout
# (default 150), BUDGET_MIN wall-clock cap (default 600, checked before EVERY cell), the
# SAME stop-files (engine/STOP_WAVE36039 / engine/STOP_WAVEPRO — one kill switch covers both
# the wave and this mop-up on a box), $BASHPID pidfiles (never $$ — this runs in backgrounded
# subshells), wait-by-PID only, 2-consecutive-fail stop, stale-terminal-marker self-clear at
# start. Completion markers are engine/MOPUP_<box>_{DONE.ok,PARTIAL.err}; DONE requires the
# ENTIRE P-A+P-B(+P-C) set present on disk — NOT merely "budget not exhausted" (the
# analysis-plan lesson: a silent skip must never fake DONE).
#
# --selftest: enumerates the priority-ordered cell list for BOTH box values against the
# CURRENT on-disk state (skip-if-exists vs would-run) WITHOUT touching killgate, model_gate,
# or the network. Safe to run anywhere, any time.
set -u
H="$(cd "$(dirname "$0")/.." && pwd)"; cd "$H" || exit 1
PY=${CLOUD_PY:-/root/miniconda3/bin/python}
M=${MODELS_DIR:-/root/autodl-tmp/models}
DATA="$H/data/counterfact.json"
RES="$H/results"
ENG="$H/engine"
WAVE_BOX=${WAVE_BOX:-36039}
JOB_CAP_MIN=${JOB_CAP_MIN:-150}
BUDGET_MIN=${BUDGET_MIN:-600}
mkdir -p "$RES" "$ENG" cloud/logs

SELFTEST=0
[ "${1:-}" = "--selftest" ] && SELFTEST=1

case "$WAVE_BOX" in
  36039)   STOP="$ENG/STOP_WAVE36039"; SMK="$RES/smoke36039" ;;
  pro6000) STOP="$ENG/STOP_WAVEPRO";   SMK="$RES/smokepro" ;;
  *) echo "mopup_critical.sh: unknown WAVE_BOX=$WAVE_BOX (want 36039 or pro6000)" >&2; exit 1 ;;
esac

log(){ echo "[mopup_${WAVE_BOX} $(date '+%F %T')] $*" | tee -a "$ENG/mopup_${WAVE_BOX}.log"; }
T0=$(date +%s)
over_budget(){ [ $(( ($(date +%s) - T0) / 60 )) -ge "$BUDGET_MIN" ]; }

# ---- derive spec() from the wave drivers themselves — never re-hardcode bands (see header) ----
load_spec(){ # driver_file new_fn_name -> defines new_fn_name() as a copy of that file's spec()
  eval "$(sed -n '/^spec(){/,/^}/p' "$1" | sed "1s/^spec(){/$2(){/")"
}
load_spec "$H/cloud/run_wave_36039.sh"   spec_36039
load_spec "$H/cloud/run_wave_pro6000.sh" spec_pro6000
# review MINOR #1: turn a latent late-crash (broken sed extraction -> undefined/empty spec
# -> "L_" out paths -> killgate int('') crash mid-run) into an immediate loud abort.
for _f in spec_36039 spec_pro6000; do
  type "$_f" >/dev/null 2>&1 || { echo "[mopup] FATAL: $_f not defined (spec extraction failed)"; exit 3; }
done
read -r _d _nl _b1 _b2 _b3 _b4 <<< "$(spec_36039 mistral7b)"
case "$_nl" in ''|*[!0-9]*) echo "[mopup] FATAL: spec_36039 mistral7b gave non-numeric n_layers '$_nl'"; exit 3 ;; esac

# ==================================================================================
# Priority-ordered cell enumerators. Emit TAB-separated:
#   TIER  CARD  TAG  EDITOR  LAYER  SEED  DTYPE  OUT  EXTRA...
# The emission ORDER *is* the priority order; consumers filter by CARD but must not
# reorder the stream, so each card's own subsequence stays priority-correct.
# ==================================================================================

build_cells_36039(){
  local tag dir nl b1 b2 b3 b4 layer seed
  # sharding copied from run_wave_36039.sh's header comment (spec() doesn't carry it)
  local card0_tags="mistral7b qwen3_8b"     # gemma9b excluded here on purpose -> P-D only
  local card1_tags="qwen25_7b llama31_8bi"

  # ---- P-A: alphaHO holdout causal, s0, at the 2 middle band layers (b2,b3) ----
  for tag in $card0_tags; do
    read -r dir nl b1 b2 b3 b4 <<< "$(spec_36039 "$tag")"
    for layer in "$b2" "$b3"; do
      printf 'A\t0\t%s\talpha\t%s\t0\tbf16\t%s\t--alpha_proj_source holdout\n' \
        "$tag" "$layer" "$RES/g4_${tag}_alphaHO_cf_L${layer}_s0.json"
    done
  done
  for tag in $card1_tags; do
    read -r dir nl b1 b2 b3 b4 <<< "$(spec_36039 "$tag")"
    for layer in "$b2" "$b3"; do
      printf 'A\t1\t%s\talpha\t%s\t0\tbf16\t%s\t--alpha_proj_source holdout\n' \
        "$tag" "$layer" "$RES/g4_${tag}_alphaHO_cf_L${layer}_s0.json"
    done
  done

  # ---- P-B: band layers x seeds {0,1} (rome), the 4 non-gemma families ----
  for tag in $card0_tags; do
    read -r dir nl b1 b2 b3 b4 <<< "$(spec_36039 "$tag")"
    for layer in "$b1" "$b2" "$b3" "$b4"; do
      for seed in 0 1; do
        printf 'B\t0\t%s\trome\t%s\t%s\tbf16\t%s\t\n' \
          "$tag" "$layer" "$seed" "$RES/gate_${tag}_rome_cf_L${layer}_s${seed}.json"
      done
    done
  done
  for tag in $card1_tags; do
    read -r dir nl b1 b2 b3 b4 <<< "$(spec_36039 "$tag")"
    for layer in "$b1" "$b2" "$b3" "$b4"; do
      for seed in 0 1; do
        printf 'B\t1\t%s\trome\t%s\t%s\tbf16\t%s\t\n' \
          "$tag" "$layer" "$seed" "$RES/gate_${tag}_rome_cf_L${layer}_s${seed}.json"
      done
    done
  done

  # ---- P-D: s2 of the 4 non-gemma bands, THEN gemma9b's WHOLE family (band + causal) ----
  for tag in $card0_tags; do
    read -r dir nl b1 b2 b3 b4 <<< "$(spec_36039 "$tag")"
    for layer in "$b1" "$b2" "$b3" "$b4"; do
      printf 'D\t0\t%s\trome\t%s\t2\tbf16\t%s\t\n' \
        "$tag" "$layer" "$RES/gate_${tag}_rome_cf_L${layer}_s2.json"
    done
  done
  for tag in $card1_tags; do
    read -r dir nl b1 b2 b3 b4 <<< "$(spec_36039 "$tag")"
    for layer in "$b1" "$b2" "$b3" "$b4"; do
      printf 'D\t1\t%s\trome\t%s\t2\tbf16\t%s\t\n' \
        "$tag" "$layer" "$RES/gate_${tag}_rome_cf_L${layer}_s2.json"
    done
  done
  read -r dir nl b1 b2 b3 b4 <<< "$(spec_36039 gemma9b)"
  for layer in "$b1" "$b2" "$b3" "$b4"; do
    for seed in 0 1 2; do
      printf 'D\t0\tgemma9b\trome\t%s\t%s\tbf16\t%s\t\n' \
        "$layer" "$seed" "$RES/gate_gemma9b_rome_cf_L${layer}_s${seed}.json"
    done
  done
  for layer in "$b2" "$b3"; do
    printf 'D\t0\tgemma9b\talpha\t%s\t0\tbf16\t%s\t--alpha_proj_source holdout\n' \
      "$layer" "$RES/g4_gemma9b_alphaHO_cf_L${layer}_s0.json"
  done
}

build_cells_pro6000(){
  local tag dir nl b1 b2 b3 b4 layer seed

  # ---- P-A: alphaHO holdout s0 at the 3rd band layer, all 3 families ----
  for tag in qwen25_14b qwen3_14b qwen3_32b; do
    read -r dir nl b1 b2 b3 b4 <<< "$(spec_pro6000 "$tag")"
    printf 'A\t-\t%s\talpha\t%s\t0\tbf16\t%s\t--alpha_proj_source holdout\n' \
      "$tag" "$b3" "$RES/g4_${tag}_alphaHO_cf_L${b3}_s0.json"
  done

  # ---- P-B: 14B pair band x seeds{0,1} (their design is 2-seed); 32B band x seed{0}
  #      only (its own design is s0-only — there is no s1 to add here) ----
  for tag in qwen25_14b qwen3_14b; do
    read -r dir nl b1 b2 b3 b4 <<< "$(spec_pro6000 "$tag")"
    for layer in "$b1" "$b2" "$b3" "$b4"; do
      for seed in 0 1; do
        printf 'B\t-\t%s\trome\t%s\t%s\tbf16\t%s\t\n' \
          "$tag" "$layer" "$seed" "$RES/gate_${tag}_rome_cf_L${layer}_s${seed}.json"
      done
    done
  done
  read -r dir nl b1 b2 b3 b4 <<< "$(spec_pro6000 qwen3_32b)"
  for layer in "$b1" "$b2" "$b3" "$b4"; do
    printf 'B\t-\tqwen3_32b\trome\t%s\t0\tbf16\t%s\t\n' \
      "$layer" "$RES/gate_qwen3_32b_rome_cf_L${layer}_s0.json"
  done

  # ---- P-C: pro6000-only cells — precision twin + merging RG ----
  read -r dir nl b1 b2 b3 b4 <<< "$(spec_pro6000 llama31_8b)"
  printf 'C\t-\tllama31_8b\trome\t16\t0\tfp32\t%s\t\n' "$RES/gate_llama31_8b_rome_cf_L16_s0_fp32.json"
  printf 'C\t-\tllama31_8b\trome\t16\t0\tbf16\t%s\t\n' "$RES/gate_llama31_8b_rome_cf_L16_s0_bf16.json"
  printf 'C\t-\tmerging\tmerging_rg\t24\t-\tfp32\t%s\t\n' "$RES/merging/Mistral-7B-v0.3_L24_RG"

  # ---- P-D: none planned on pro6000 — the 14B pair is 2-seed BY DESIGN (no s2 to
  #      run) and the 32B lane is s0-only BY DESIGN; nothing to deprioritize here. ----
}

# ==================================================================================
# --selftest: dry-run enumeration only, no model_gate, no killgate, no network.
# ==================================================================================
selftest_enumerate(){
  local box=$1 fn=$2
  echo "=================================================================="
  echo "WAVE_BOX=$box  priority-ordered cell list (skip-if-exists vs would-run, live disk state)"
  echo "=================================================================="
  local n=0 tier tcard tag editor layer seed dtype out extra status npz
  while IFS=$'\t' read -r tier tcard tag editor layer seed dtype out extra; do
    n=$((n + 1))
    if [ "$editor" = merging_rg ]; then
      [ -d "$out" ] && status="SKIP-EXISTS" || status="WOULD-RUN"
    else
      npz="$RES/matrices/$(basename "$out" .json).npz"
      if [ -f "$out" ] && [ -f "$npz" ]; then status="SKIP-EXISTS"; else status="WOULD-RUN"; fi
    fi
    printf '%3d  P-%s  card=%-2s  %-12s %-11s L%-4s s%-2s %-4s  [%-11s]  %s\n' \
      "$n" "$tier" "$tcard" "$tag" "$editor" "$layer" "$seed" "$dtype" "$status" "$(basename "$out")"
  done < <("$fn")
  echo "total cells: $n"
  echo
}

if [ "$SELFTEST" -eq 1 ]; then
  selftest_enumerate 36039   build_cells_36039
  selftest_enumerate pro6000 build_cells_pro6000
  exit 0
fi

# ==================================================================================
# Real-run machinery below (not exercised by --selftest).
# ==================================================================================

model_gate(){ # tag dir n_layers [smoke_json_override] -> 0 ok / 1 skip
  local tag=$1 dir=$2 nl=$3 smoke=${4:-}
  [ -d "$dir" ] || { log "CONFIG-skip $tag: model dir absent"; return 1; }
  [ -z "$smoke" ] && smoke="$SMK/smoke_${tag}.json"
  if ! "$PY" - "$smoke" <<'EOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
esr = d.get("esr", d.get("edit_success_rate", 0)) or 0
sys.exit(0 if esr > 0.9 else 1)
EOF
  then log "CONFIG-skip $tag: smoke gate not passed ($smoke)"; return 1; fi
  local cfg_nl
  cfg_nl=$("$PY" - "$dir/config.json" <<'EOF'
import json, sys
print(json.load(open(sys.argv[1])).get("num_hidden_layers", -1))
EOF
) || cfg_nl=-1
  if [ "$cfg_nl" != "$nl" ]; then
    log "CONFIG-skip $tag: num_hidden_layers=$cfg_nl != expected $nl (band would be wrong)"
    return 1
  fi
  return 0
}

run_cell(){ # cvd("" | "0" | "1") dir editor layer seed dtype out logtag [extra...] -> rc 9 = stop/budget sentinel
  local cvd=$1 dir=$2 editor=$3 layer=$4 seed=$5 dtype=$6 out=$7 logtag=$8; shift 8
  local npz="$RES/matrices/$(basename "$out" .json).npz"
  [ -f "$out" ] && [ -f "$npz" ] && { log "skip (exists): $(basename "$out")"; return 0; }
  [ -f "$STOP" ] && { log "STOP-file present — not starting $(basename "$out")"; return 9; }
  over_budget && { log "BUDGET_MIN=$BUDGET_MIN reached — not starting $(basename "$out")"; return 9; }
  log "RUN $(basename "$out")"
  if [ -n "$cvd" ]; then
    CUDA_VISIBLE_DEVICES="$cvd" timeout "$((JOB_CAP_MIN * 60))" "$PY" experiments/killgate_keygeom.py \
      --model "$dir" --data "$DATA" --dataset counterfact \
      --n_edits 200 --n_probes 500 --steps 20 --lr 0.1 \
      --editor "$editor" --layer "$layer" --seed "$seed" --model_dtype "$dtype" \
      --save_matrices "$@" --out "$out" > "cloud/logs/mopup_${WAVE_BOX}_${logtag}.log" 2>&1
  else
    timeout "$((JOB_CAP_MIN * 60))" "$PY" experiments/killgate_keygeom.py \
      --model "$dir" --data "$DATA" --dataset counterfact \
      --n_edits 200 --n_probes 500 --steps 20 --lr 0.1 \
      --editor "$editor" --layer "$layer" --seed "$seed" --model_dtype "$dtype" \
      --save_matrices "$@" --out "$out" > "cloud/logs/mopup_${WAVE_BOX}_${logtag}.log" 2>&1
  fi
  local rc=$?
  [ $rc -ne 0 ] && log "FAIL rc=$rc $(basename "$out")"
  return $rc
}

card_worker_36039(){ # card(0|1)
  local card=$1
  echo "$BASHPID" > "$ENG/mopup_36039_card${card}.pid"
  local -A gate_ok=() gate_dir=()
  local fails=0 rc
  local tier tcard tag editor layer seed dtype out extra dir nl
  while IFS=$'\t' read -r tier tcard tag editor layer seed dtype out extra; do
    [ "$tcard" = "$card" ] || continue
    if [ -z "${gate_ok[$tag]+x}" ]; then
      read -r dir nl _ <<< "$(spec_36039 "$tag")"
      if [ "$tag" = gemma9b ]; then
        model_gate "$tag" "$dir" "$nl" "$SMK/smoke_gemma9b_bf16.json" && gate_ok[$tag]=1 || gate_ok[$tag]=0
      else
        model_gate "$tag" "$dir" "$nl" && gate_ok[$tag]=1 || gate_ok[$tag]=0
      fi
      gate_dir[$tag]=$dir
    fi
    [ "${gate_ok[$tag]}" = 1 ] || continue
    run_cell "$card" "${gate_dir[$tag]}" "$editor" "$layer" "$seed" "$dtype" "$out" \
      "${tag}_${editor}_L${layer}_s${seed}" $extra
    rc=$?
    [ $rc -eq 9 ] && return 0
    if [ $rc -ne 0 ]; then
      fails=$((fails + 1))
      [ $fails -ge 2 ] && { log "card$card 2 consecutive fails — stopping (wedge?)"; return 1; }
    else
      fails=0
    fi
  done < <(build_cells_36039)
  return 0
}

serial_worker_pro6000(){
  echo "$BASHPID" > "$ENG/mopup_pro6000_main.pid"
  local -A gate_ok=() gate_dir=()
  local fails=0 rc
  local tier tcard tag editor layer seed dtype out extra dir nl
  while IFS=$'\t' read -r tier tcard tag editor layer seed dtype out extra; do
    if [ "$editor" = merging_rg ]; then
      [ -f "$STOP" ] && { log "STOP-file present — not starting merging RG"; return 0; }
      if over_budget; then log "BUDGET_MIN reached — skipping merging RG"; continue; fi
      if [ -d "$out" ]; then
        log "skip (exists): merging RG dir"
      else
        # explicit --table_out (2026-07-14 fix, same hazard/fix as run_wave_pro6000.sh's
        # merging_worker(): without it, merging_m0.py --rg defaults to $out_dir_top/
        # RG_operating_curve_table.json — the SAME top-level path the canonical
        # Llama-3.2-1B L12 gate owns. Tag+layer-namespaced, same convention as
        # run_merging_width.sh / the run_wave_pro6000.sh fix.
        local table_out="$RES/merging/RG_operating_curve_table_mistral7b_L${layer}.json"
        log "RUN merging RG Mistral-7B L24 (fp32, seeds 0,1,2, g=2..20)"
        timeout $((JOB_CAP_MIN * 60 * 2)) "$PY" experiments/merging_m0.py --rg \
          --model "$M/Mistral-7B-v0.3" --data "$DATA" --layer "$layer" \
          --rg_seeds 0,1,2 --rg_group_sizes 2,3,5,10,20 \
          --table_out "$table_out" \
          > cloud/logs/mopup_pro6000_merging_rg.log 2>&1
        rc=$?
        if [ $rc -ne 0 ]; then log "merging FAIL rc=$rc"; fails=$((fails + 1)); else fails=0; fi
      fi
      continue
    fi
    if [ -z "${gate_ok[$tag]+x}" ]; then
      read -r dir nl _ <<< "$(spec_pro6000 "$tag")"
      model_gate "$tag" "$dir" "$nl" && gate_ok[$tag]=1 || gate_ok[$tag]=0
      gate_dir[$tag]=$dir
    fi
    [ "${gate_ok[$tag]}" = 1 ] || continue
    run_cell "" "${gate_dir[$tag]}" "$editor" "$layer" "$seed" "$dtype" "$out" \
      "${tag}_${editor}_L${layer}_s${seed}" $extra
    rc=$?
    [ $rc -eq 9 ] && return 0
    if [ $rc -ne 0 ]; then
      fails=$((fails + 1))
      [ $fails -ge 2 ] && { log "2 consecutive fails — stopping (wedge?)"; return 1; }
    else
      fails=0
    fi
  done < <(build_cells_pro6000)
  return 0
}

check_complete(){ # fn -> 0 iff every P-A/P-B/P-C cell is present on disk (P-D excluded by design)
  local fn=$1
  local tier tcard tag editor layer seed dtype out extra npz
  while IFS=$'\t' read -r tier tcard tag editor layer seed dtype out extra; do
    case "$tier" in A | B | C) ;; *) continue ;; esac
    if [ "$editor" = merging_rg ]; then
      [ -d "$out" ] || return 1
    else
      npz="$RES/matrices/$(basename "$out" .json).npz"
      { [ -f "$out" ] && [ -f "$npz" ]; } || return 1
    fi
  done < <("$fn")
  return 0
}

echo $$ > "$ENG/mopup_${WAVE_BOX}_main.pid"
# review MAJOR #4: NEVER overlap the still-running main wave. The mop-up runs the SAME
# (reordered) cells; if a cell is mid-flight (json+npz not both on disk yet) the skip check
# sees nothing and launches a DUPLICATE on the same card -> two processes writing the same
# --save_matrices npz (killgate's npz write; the gate's only input) = truncation/corruption.
# Block on the wave's own main pidfile (36039 -> wave36039_main.pid, pro6000 -> wavepro_main.pid).
case "$WAVE_BOX" in
  36039)   WAVE_MAIN_PIDFILE="$ENG/wave36039_main.pid" ;;
  pro6000) WAVE_MAIN_PIDFILE="$ENG/wavepro_main.pid" ;;
esac
WMP=$(cat "$WAVE_MAIN_PIDFILE" 2>/dev/null || true)
if [ -n "$WMP" ]; then
  while kill -0 "$WMP" 2>/dev/null; do
    log "wave main pid $WMP still running — mop-up waiting (60s) to avoid concurrent npz writes"
    sleep 60
  done
  log "wave main pid $WMP has exited — mop-up proceeding"
fi
# clear stale terminal markers from earlier attempts (the analysis-plan lesson burned on
# the wave drivers: a dead run's PARTIAL.err misleads every watcher of the fresh run)
rm -f "$ENG/MOPUP_${WAVE_BOX}_DONE.ok" "$ENG/MOPUP_${WAVE_BOX}_PARTIAL.err"
log "mopup start: WAVE_BOX=$WAVE_BOX JOB_CAP_MIN=$JOB_CAP_MIN BUDGET_MIN=$BUDGET_MIN"

case "$WAVE_BOX" in
  36039)
    card_worker_36039 0 > "cloud/logs/mopup_36039_card0.log" 2>&1 & W0=$!
    card_worker_36039 1 > "cloud/logs/mopup_36039_card1.log" 2>&1 & W1=$!
    log "workers: card0 pid $W0, card1 pid $W1"
    wait $W0; RC0=$?
    wait $W1; RC1=$?
    log "mopup done: card0 rc=$RC0 card1 rc=$RC1"
    COMPLETE_FN=build_cells_36039
    ;;
  pro6000)
    serial_worker_pro6000 > "cloud/logs/mopup_pro6000_serial.log" 2>&1
    RC0=$?
    log "mopup done: rc=$RC0"
    COMPLETE_FN=build_cells_pro6000
    ;;
esac

if check_complete "$COMPLETE_FN"; then
  touch "$ENG/MOPUP_${WAVE_BOX}_DONE.ok"
  log "MOPUP_${WAVE_BOX}_DONE.ok — full P-A+P-B$( [ "$WAVE_BOX" = pro6000 ] && echo "+P-C" ) set on disk"
else
  touch "$ENG/MOPUP_${WAVE_BOX}_PARTIAL.err"
  log "MOPUP_${WAVE_BOX}_PARTIAL.err — critical set not fully on disk (budget/fail/gate-skip)"
fi
