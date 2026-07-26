#!/bin/bash
# cloud/analyze_families.sh — on-box, read-only CPU analysis daemon for the
# 7-9B/14B/32B/precision/merging waves (docs/plans/ANALYSIS-PLAN-WAVES-20260713.md
# section 6, binding). Runs the pre-registered gate-law / S x C / magnitude /
# causal passes for each model family AS SOON AS its full gate-band npz set
# exists under results/matrices/, so the CPU-bound analysis happens on the box's
# fast CPU instead of after a slow rsync home — only the small output JSONs
# need syncing.
#
# READ-ONLY CONTRACT: only reads results/matrices/*.npz (and, for the merging
# family, results/merging/*/RG_operating_curve_table.json); only writes
# results/analysis/*.json + cloud/logs/ + its own pidfile/stopfile under
# cloud/. Never writes to results/matrices/ or engine/ — safe to run alongside
# the live run_wave_36039.sh / run_wave_pro6000.sh. (The pidfile/stopfile are
# deliberately placed under cloud/, NOT engine/, so this daemon can never
# collide with the wave drivers' own engine/ markers, e.g. STOP_WAVE36039.)
#
# Band layers are NEVER hardcoded independently — they are sliced straight out
# of the two wave drivers' own `spec(){...}` functions (see extract_spec_fn
# below) so this script's expectations cannot drift from what actually ran.
# Per-tag seed counts (pro6000's 14B pair = 2 seeds, 32B = 1 seed), the
# precision-twin layer, and the merging RG directory name are likewise grepped
# out of the driver source rather than re-typed.
#
# KNOWN BLOCKER (not fixed here, out of scope for this daemon): mechanism_sc_
# table.py's TAG_RE assumes a SINGLE-TOKEN model tag between the gate_/g4_
# prefix and the trailing _<editor>_<dataset>_L<layer>_s<seed>.npz — e.g.
# gate_mistral7b_rome_cf_L16_s0.npz parses fine, but gate_qwen25_7b_rome_cf_
# L14_s0.npz does NOT (qwen25_7b is two underscore-joined tokens, so the regex
# never matches and the file is silently dropped to the notes/logged-skip
# path). Of the 8 gate-law families in this wave, only mistral7b and gemma9b
# have single-token tags; the other 6 (qwen25_7b, llama31_8bi, qwen3_8b,
# qwen25_14b, qwen3_14b, qwen3_32b) will come back with an EMPTY "groups" list
# from the documented `mechanism_sc_table.py --npz ...` command. This script
# still runs that exact command (per the analysis plan) but checks the result
# and logs a WARNING when "groups" is empty, so an operator notices instead of
# silently trusting an empty C1_sc_<tag>.json. See the header comment inside
# analyze_gate_family() below.
set -u
H="$(cd "$(dirname "$0")/.." && pwd)"; cd "$H" || exit 1
PY=${CLOUD_PY:-/root/miniconda3/bin/python}
RES="$H/results"
MAT="$RES/matrices"
OUTD="$RES/analysis"
LOG="$H/cloud/logs/analyze_families.log"
PIDFILE="$H/cloud/analyze_families.pid"
STOPFILE="$H/cloud/ANALYZE_FAMILIES_STOP"
DRIVER_36039="$H/cloud/run_wave_36039.sh"
DRIVER_PRO6000="$H/cloud/run_wave_pro6000.sh"
M=${MODELS_DIR:-/root/autodl-tmp/models}

log(){ echo "[analyze_families $(date '+%F %T')] $*" | tee -a "$LOG"; }

# ---------------------------------------------------------------- CLI -----
WATCH=0
INTERVAL_MIN=15
MAX_CYCLES=0     # 0 = unlimited (--watch loops until every tracked family is fresh)
SELFTEST=0
DRYRUN=0
FAMILIES_OVERRIDE=""

usage(){
  cat <<'EOF'
Usage: analyze_families.sh [--watch] [--interval MIN] [--max-cycles N]
                            [--families tag1,tag2,...] [--dry-run] [--selftest]

One-shot (default): analyze whatever tracked families are ready now, then exit.
--watch          poll every --interval minutes (default 15); exits once every
                  tracked family has a fresh analysis, or after --max-cycles
                  cycles (0 = unlimited — pair with --families on each box so
                  this can actually terminate; a family that never lands on
                  THIS box will never go "fresh").
--families CSV   restrict the tracked set (default: all 10 known families
                  across both waves). Use the box-local subset in practice:
                    36039:   mistral7b,qwen25_7b,llama31_8bi,qwen3_8b,gemma9b
                    29246:   qwen25_14b,qwen3_14b,qwen3_32b,llama31_8b_twin,merging_rg
--dry-run        print the commands that would run, without invoking python.
--selftest       CPU-only self-check in a throwaway tempdir; touches nothing
                  under the real repo's results/.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --watch) WATCH=1; shift ;;
    --interval) INTERVAL_MIN=$2; shift 2 ;;
    --max-cycles) MAX_CYCLES=$2; shift 2 ;;
    --families) FAMILIES_OVERRIDE=$2; shift 2 ;;
    --dry-run) DRYRUN=1; shift ;;
    --selftest) SELFTEST=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

ALL_FAMILIES="mistral7b qwen25_7b llama31_8bi qwen3_8b gemma9b qwen25_14b qwen3_14b qwen3_32b llama31_8b_twin merging_rg"
if [ -n "$FAMILIES_OVERRIDE" ]; then
  FAMILIES=$(echo "$FAMILIES_OVERRIDE" | tr ',' ' ')
else
  FAMILIES=$ALL_FAMILIES
fi

# ---------------------------------------------------- driver-derived spec -
# Slice out only the `spec(){ ... }` block (pure — no side effects) from each
# wave driver and eval it under a renamed function, so we get the REAL bands
# without executing the rest of the driver (which mkdir's, backgrounds jobs,
# etc.). This is the "derive from the driver so they can't drift" requirement.
extract_spec_fn(){ sed -n '/^spec(){/,/^}/p' "$1"; }
eval "$(extract_spec_fn "$DRIVER_36039"   | sed 's/^spec(){/spec_36039(){/')"
eval "$(extract_spec_fn "$DRIVER_PRO6000" | sed 's/^spec(){/spec_pro6000(){/')"

# pro6000 per-tag seed list, read from the actual band_worker call sites
# (run_wave_pro6000.sh's P1/P2 stanzas) rather than assumed, so a future driver
# edit to the seed counts can't silently drift from this daemon.
pro6000_seeds_of(){  # tag -> csv seeds, e.g. "0,1"
  grep -oP "band_worker\s+\d+\s+${1}\s+\K[0-9,]+" "$DRIVER_PRO6000" | head -1
}

# precision-twin layer, read from twin8b_worker's run_cell calls (both the
# fp32 and bf16 lines carry the same literal layer — take the first match).
twin_layer(){
  awk '/^twin8b_worker\(\)/,/^}/' "$DRIVER_PRO6000" | grep -oP 'rome \K[0-9]+' | head -1
}

# merging RG out_dir (relative to $RES), read from merging_worker's own
# out_dir= assignment.
merging_reldir(){
  awk '/^merging_worker\(\)/,/^}/' "$DRIVER_PRO6000" | grep -oP 'out_dir="\$RES/\K[^"]+' | head -1
}

# ------------------------------------------------------- family metadata --
family_box(){  # tag -> box class, rc 1 if unknown
  case "$1" in
    mistral7b|qwen25_7b|llama31_8bi|qwen3_8b|gemma9b) echo 36039 ;;
    qwen25_14b|qwen3_14b|qwen3_32b) echo pro6000_band ;;
    llama31_8b_twin) echo pro6000_twin ;;
    merging_rg) echo pro6000_merging ;;
    *) return 1 ;;
  esac
}

band_of(){  # tag -> "nl b1 b2 b3 b4", via the correct driver's real spec()
  local tag=$1 box dir nl b1 b2 b3 b4
  box=$(family_box "$tag") || return 1
  case "$box" in
    36039)       read -r dir nl b1 b2 b3 b4 <<< "$(spec_36039 "$tag")" ;;
    pro6000_band) read -r dir nl b1 b2 b3 b4 <<< "$(spec_pro6000 "$tag")" ;;
    *) return 1 ;;
  esac
  [ -z "${nl:-}" ] && return 1
  echo "$nl $b1 $b2 $b3 $b4"
}

# tag -> newline list of expected npz BASENAMES under $MAT (empty + rc1 if
# the tag/box is not derivable). merging_rg is handled separately (it is a
# directory + table, not a set of npz) — see family_ready/family_fresh.
expected_npz(){
  local tag=$1 box nl b1 b2 b3 b4 l s seeds L
  box=$(family_box "$tag") || return 1
  case "$box" in
    36039)
      read -r nl b1 b2 b3 b4 <<< "$(band_of "$tag")" || return 1
      for l in "$b1" "$b2" "$b3" "$b4"; do
        for s in 0 1 2; do echo "gate_${tag}_rome_cf_L${l}_s${s}.npz"; done
      done
      for l in "$b2" "$b3"; do echo "g4_${tag}_alphaHO_cf_L${l}_s0.npz"; done
      ;;
    pro6000_band)
      read -r nl b1 b2 b3 b4 <<< "$(band_of "$tag")" || return 1
      seeds=$(pro6000_seeds_of "$tag"); [ -z "$seeds" ] && return 1
      for l in "$b1" "$b2" "$b3" "$b4"; do
        for s in ${seeds//,/ }; do echo "gate_${tag}_rome_cf_L${l}_s${s}.npz"; done
      done
      echo "g4_${tag}_alphaHO_cf_L${b3}_s0.npz"
      ;;
    pro6000_twin)
      L=$(twin_layer); [ -z "$L" ] && return 1
      echo "gate_llama31_8b_rome_cf_L${L}_s0_fp32.npz"
      echo "gate_llama31_8b_rome_cf_L${L}_s0_bf16.npz"
      ;;
    *) return 1 ;;
  esac
}

family_ready(){  # tag -> rc0 iff every expected input is present
  local tag=$1 box rd f
  box=$(family_box "$tag") || return 1
  if [ "$box" = pro6000_merging ]; then
    rd=$(merging_reldir); [ -z "$rd" ] && return 1
    [ -f "$RES/$rd/RG_operating_curve_table.json" ]
    return $?
  fi
  local -a files=()
  mapfile -t files < <(expected_npz "$tag")
  [ "${#files[@]}" -eq 0 ] && return 1
  for f in "${files[@]}"; do
    [ -f "$MAT/$f" ] || return 1
  done
  return 0
}

newest_mtime(){  # prints max epoch mtime among given existing files (0 if none)
  local f max=0 t
  for f in "$@"; do
    [ -f "$f" ] || continue
    t=$(stat -c %Y "$f" 2>/dev/null) || continue
    [ "$t" -gt "$max" ] && max=$t
  done
  echo "$max"
}

family_fresh(){  # tag -> rc0 iff <tag>_MANIFEST.json exists and is >= all inputs
  local tag=$1 box rd mtm inmax
  local manifest="$OUTD/${tag}_MANIFEST.json"
  [ -f "$manifest" ] || return 1
  box=$(family_box "$tag") || return 1
  if [ "$box" = pro6000_merging ]; then
    rd=$(merging_reldir); [ -z "$rd" ] && return 1
    inmax=$(newest_mtime "$RES/$rd/RG_operating_curve_table.json")
  else
    local -a files=() abspaths=()
    mapfile -t files < <(expected_npz "$tag")
    for f in "${files[@]}"; do abspaths+=("$MAT/$f"); done
    inmax=$(newest_mtime "${abspaths[@]}")
  fi
  mtm=$(stat -c %Y "$manifest" 2>/dev/null) || return 1
  [ "$mtm" -ge "$inmax" ]
}

# ------------------------------------------------------- command runner --
run_or_echo(){  # desc cmd...
  local desc=$1; shift
  if [ "$DRYRUN" -eq 1 ]; then
    log "[dry-run] $desc"
    printf '%q ' "$@"; echo
    return 0
  fi
  log "RUN $desc"
  "$@"
  local rc=$?
  [ $rc -ne 0 ] && log "FAIL rc=$rc: $desc"
  return $rc
}

# --------------------------------------------------- per-family analysis --
# §6 gate-law family: analyze_matrices (per band layer) + mechanism_sc_table
# (S x C, ROME rows only) + magnitude_table (all 4 layers, one call) +
# aggregate_g4_causal (holdout causal, only if the alphaHO npz exist).
analyze_gate_family(){  # tag box(36039|pro6000_band)
  local tag=$1 box=$2 nl b1 b2 b3 b4 l al have_alpha=1 rc=0
  read -r nl b1 b2 b3 b4 <<< "$(band_of "$tag")" || { log "band_of failed for $tag"; return 1; }

  for l in "$b1" "$b2" "$b3" "$b4"; do
    run_or_echo "gate law $tag L$l" \
      "$PY" experiments/analyze_matrices.py "$MAT/gate_${tag}_rome_cf_L${l}_s*.npz" \
        --known --edit_ok --out "$OUTD/gate_${tag}_L${l}.json" || rc=1
  done

  run_or_echo "S x C table $tag" \
    "$PY" experiments/mechanism_sc_table.py \
      --npz "$MAT/gate_${tag}_rome_cf_L*_s*.npz" \
      --known --edit_ok --out "$OUTD/C1_sc_${tag}.json" || rc=1
  # KNOWN BLOCKER (see file header): TAG_RE requires a single-token model tag.
  # Every tag here containing "_" (all but mistral7b/gemma9b) will parse to
  # zero groups. Detect and warn rather than trust an empty table silently.
  if [ "$DRYRUN" -eq 0 ] && [ -f "$OUTD/C1_sc_${tag}.json" ]; then
    "$PY" - "$OUTD/C1_sc_${tag}.json" "$tag" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
if not d.get("groups"):
    print(f"WARNING[{sys.argv[2]}]: C1_sc_{sys.argv[2]}.json has 0 groups — almost "
          f"certainly the mechanism_sc_table.py TAG_RE single-token-tag limitation "
          f"(model tag {sys.argv[2]!r} contains '_'); see analyze_families.sh header.")
PYEOF
  fi

  local -a famspec=()
  for l in "$b1" "$b2" "$b3" "$b4"; do
    famspec+=(--family "${tag}_L${l}=$MAT/gate_${tag}_rome_cf_L${l}_s*.npz")
  done
  # NOTE: combined into ONE call with 4 repeated --family flags (all 4 band
  # layers -> one combined output), not 4 separate calls each writing the
  # SAME --out path (the plan's one-liner example, run naively 4x per layer,
  # would have each call overwrite the last -> only the final layer would
  # survive on disk). This is the usage the CLI's --family/append design and
  # combined headline table are actually built for.
  run_or_echo "magnitude table $tag" \
    "$PY" experiments/magnitude_table.py --known --edit_ok \
      "${famspec[@]}" --out "$OUTD/C1_magnitude_${tag}.json" || rc=1

  if [ "$box" = 36039 ]; then
    local -a alpha_layers=("$b2" "$b3")
  else
    local -a alpha_layers=("$b3")
  fi
  for al in "${alpha_layers[@]}"; do
    [ -f "$MAT/g4_${tag}_alphaHO_cf_L${al}_s0.npz" ] || have_alpha=0
  done
  if [ "$have_alpha" -eq 1 ]; then
    run_or_echo "causal (alphaHO) $tag" \
      "$PY" experiments/aggregate_g4_causal.py \
        --rome_glob "$MAT/gate_${tag}_rome_cf_L{L}_s*.npz" \
        --alpha_glob "$MAT/g4_${tag}_alphaHO_cf_L{L}_s*.npz" \
        --layers "${alpha_layers[@]}" --known --edit_ok --proj_source holdout \
        --out "$OUTD/C4_causal_${tag}.json" || rc=1
  else
    log "skip causal pass for $tag: alphaHO npz missing at layers ${alpha_layers[*]}"
  fi
  return $rc
}

# §6 Family 4 (precision twin): explicit-npz analyze_matrices, twice — the
# _fp32/_bf16 suffix breaks the default globs used elsewhere.
analyze_twin_family(){
  local L rc=0
  L=$(twin_layer) || return 1
  run_or_echo "precision twin fp32" \
    "$PY" experiments/analyze_matrices.py "$MAT/gate_llama31_8b_rome_cf_L${L}_s0_fp32.npz" \
      --known --edit_ok --out "$OUTD/gate_llama31_8b_L${L}_fp32.json" || rc=1
  run_or_echo "precision twin bf16" \
    "$PY" experiments/analyze_matrices.py "$MAT/gate_llama31_8b_rome_cf_L${L}_s0_bf16.npz" \
      --known --edit_ok --out "$OUTD/gate_llama31_8b_L${L}_bf16.json" || rc=1
  return $rc
}

# §6 Family 5 (merging RG): standalone CPU re-analysis of the on-box RG dir.
analyze_merging_family(){
  local rd rc=0
  rd=$(merging_reldir) || return 1
  # CAUTION in the analysis plan: `--rg_phase2_dir` alone re-derives AND
  # OVERWRITES RG_operating_curve_table.json in place inside the RG dir; the
  # plan says to cp the on-box table aside first so the recompute can be
  # diffed instead of silently replacing it. We use --table_out instead
  # (a real flag on merging_m0.py) to redirect the recompute into
  # results/analysis/ entirely — the on-box original is never touched at
  # all, which satisfies the plan's intent more strictly than "cp aside
  # first" and keeps this daemon's read-only contract w.r.t. results/merging/.
  run_or_echo "merging RG recompute" \
    "$PY" experiments/merging_m0.py --rg_phase2_dir "$RES/$rd" \
      --table_out "$OUTD/merging_rg_recompute_table.json" || rc=1
  return $rc
}

analyze_family(){  # tag -> writes <tag>_MANIFEST.json on completion (unless dry-run)
  local tag=$1 box rc=0
  box=$(family_box "$tag") || { log "unknown family: $tag"; return 1; }
  log "ANALYZE $tag (box=$box)"
  case "$box" in
    36039|pro6000_band) analyze_gate_family "$tag" "$box" || rc=1 ;;
    pro6000_twin)       analyze_twin_family || rc=1 ;;
    pro6000_merging)    analyze_merging_family || rc=1 ;;
  esac
  if [ "$DRYRUN" -eq 0 ]; then
    "$PY" - "$OUTD/${tag}_MANIFEST.json" "$tag" "$box" "$rc" <<'PYEOF'
import json, sys, time
out, tag, box, rc = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
json.dump({"family": tag, "box": box,
           "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "any_pass_failed": rc != 0}, open(out, "w"), indent=2)
PYEOF
    log "wrote $OUTD/${tag}_MANIFEST.json (any_pass_failed=$([ $rc -ne 0 ] && echo true || echo false))"
  fi
  return $rc
}

run_cycle(){  # rc0 iff every tracked family is now fresh (nothing left pending)
  local tag pending=0
  for tag in $FAMILIES; do
    family_fresh "$tag" && continue
    if family_ready "$tag"; then
      analyze_family "$tag"
    else
      pending=1
    fi
  done
  return $pending
}

# ------------------------------------------------------------ selftest ---
# CPU-only, no GPU/network/downloads. Sandboxes every path the shared helpers
# touch into a throwaway tempdir (H/RES/MAT/OUTD/LOG/DRIVER_*), so nothing
# under the real repo's results/ or cloud/ is ever written. Copies the REAL
# driver scripts in so band/seed/layer/dir derivation is exercised against
# the actual source of truth, not a hand-typed stand-in.
run_selftest(){
  local FAIL=0
  pass(){ echo "PASS: $1"; }
  fail(){ echo "FAIL: $1"; FAIL=1; }

  local TMP; TMP=$(mktemp -d "${TMPDIR:-/tmp}/analyze_families_selftest.XXXXXX")
  trap 'rm -rf "$TMP"' RETURN

  mkdir -p "$TMP/cloud" "$TMP/results/matrices" "$TMP/results/analysis"
  cp "$DRIVER_36039" "$DRIVER_PRO6000" "$TMP/cloud/"

  H="$TMP"; RES="$TMP/results"; MAT="$TMP/results/matrices"; OUTD="$TMP/results/analysis"
  LOG="$TMP/cloud/analyze_families_selftest.log"
  DRIVER_36039="$TMP/cloud/run_wave_36039.sh"
  DRIVER_PRO6000="$TMP/cloud/run_wave_pro6000.sh"

  echo "=================================================================="
  echo "(a) band/seed/layer/dir derivation straight from the copied real drivers"
  echo "=================================================================="
  local nl b1 b2 b3 b4
  read -r nl b1 b2 b3 b4 <<< "$(band_of mistral7b)"
  if [ "$nl $b1 $b2 $b3 $b4" = "32 16 20 24 28" ]; then
    pass "mistral7b band derived as nl=32 band=16,20,24,28"
  else
    fail "mistral7b band derivation mismatch: got '$nl $b1 $b2 $b3 $b4'"
  fi
  read -r nl b1 b2 b3 b4 <<< "$(band_of qwen25_14b)"
  if [ "$nl $b1 $b2 $b3 $b4" = "48 24 30 36 42" ]; then
    pass "qwen25_14b band derived as nl=48 band=24,30,36,42"
  else
    fail "qwen25_14b band derivation mismatch: got '$nl $b1 $b2 $b3 $b4'"
  fi
  local seeds; seeds=$(pro6000_seeds_of qwen3_32b)
  if [ "$seeds" = "0" ]; then pass "qwen3_32b seeds derived as '0' (1-seed)"
  else fail "qwen3_32b seed derivation: got '$seeds'"; fi
  local tl; tl=$(twin_layer)
  if [ "$tl" = "16" ]; then pass "precision-twin layer derived as 16"
  else fail "twin layer derivation: got '$tl'"; fi
  local rd; rd=$(merging_reldir)
  if [ "$rd" = "merging/Mistral-7B-v0.3_L24_RG" ]; then pass "merging RG dir derived as $rd"
  else fail "merging dir derivation: got '$rd'"; fi

  echo "=================================================================="
  echo "(b) synthetic npz + readiness detection (positive: mistral7b, full set)"
  echo "=================================================================="
  make_fake_npz(){  # out_path -> writes a schema-correct synthetic killgate npz
    "$PY" - "$1" <<'PYEOF'
import numpy as np, sys
n, m = 8, 8
rng = np.random.default_rng(0)
np.savez(sys.argv[1],
         COS=rng.normal(size=(n, m)),
         damage_logit=rng.normal(size=(n, m)),
         norm_growth=rng.uniform(0.1, 1.0, size=n),
         edit_ok=np.ones(n),
         pre_p=np.full(m, 0.5),
         resid_norm=rng.uniform(0.1, 1.0, size=n))
PYEOF
  }
  local f
  while IFS= read -r f; do make_fake_npz "$MAT/$f"; done < <(expected_npz mistral7b)
  if family_ready mistral7b; then pass "mistral7b READY once all 14 synthetic npz exist"
  else fail "mistral7b not detected READY with a complete synthetic npz set"; fi

  echo "=================================================================="
  echo "(c) readiness detection (negative: gemma9b missing one npz, then completed)"
  echo "=================================================================="
  while IFS= read -r f; do make_fake_npz "$MAT/$f"; done < <(expected_npz gemma9b | head -n 13)
  if family_ready gemma9b; then fail "gemma9b incorrectly READY with 13/14 npz present"
  else pass "gemma9b correctly NOT ready with 13/14 npz present"; fi
  expected_npz gemma9b | tail -n 1 | while IFS= read -r f; do make_fake_npz "$MAT/$f"; done
  if family_ready gemma9b; then pass "gemma9b READY once its 14th npz lands"
  else fail "gemma9b still not READY after completing its npz set"; fi

  echo "=================================================================="
  echo "(d) precision-twin + merging-RG readiness"
  echo "=================================================================="
  while IFS= read -r f; do make_fake_npz "$MAT/$f"; done < <(expected_npz llama31_8b_twin)
  if family_ready llama31_8b_twin; then pass "llama31_8b_twin READY once both fp32/bf16 npz exist"
  else fail "llama31_8b_twin not detected READY"; fi
  mkdir -p "$RES/$rd"
  echo '{}' > "$RES/$rd/RG_operating_curve_table.json"
  if family_ready merging_rg; then pass "merging_rg READY once RG_operating_curve_table.json lands"
  else fail "merging_rg not detected READY"; fi

  echo "=================================================================="
  echo "(e) dry-run command assembly (must NOT invoke the heavy analysis)"
  echo "=================================================================="
  DRYRUN=1
  local dryout; dryout=$(analyze_family mistral7b 2>&1)
  DRYRUN=0
  echo "$dryout" | sed 's/^/  /'
  local needle
  for needle in "analyze_matrices.py" "mechanism_sc_table.py" "magnitude_table.py" \
                "aggregate_g4_causal.py" "gate_mistral7b_rome_cf_L16" "gate_mistral7b_rome_cf_L24" \
                "--proj_source holdout"; do
    if echo "$dryout" | grep -qF -- "$needle"; then pass "dry-run command mentions '$needle'"
    else fail "dry-run command missing '$needle'"; fi
  done
  if [ -f "$OUTD/mistral7b_MANIFEST.json" ]; then
    fail "dry-run wrote a manifest file (it must not)"
  else
    pass "dry-run wrote no manifest"
  fi
  if ls "$OUTD"/gate_mistral7b_L*.json >/dev/null 2>&1; then
    fail "dry-run produced real output json — heavy analysis ran when it should not have"
  else
    pass "dry-run produced no real output json — heavy analysis was not executed"
  fi

  echo "=================================================================="
  echo "(f) idempotency: manifest-vs-npz freshness gate"
  echo "=================================================================="
  : > "$OUTD/mistral7b_MANIFEST.json"
  touch -d "2020-01-01" "$OUTD/mistral7b_MANIFEST.json"
  if family_fresh mistral7b; then fail "a stale (2020) manifest was incorrectly considered fresh"
  else pass "a stale manifest is correctly considered NOT fresh"; fi
  touch "$OUTD/mistral7b_MANIFEST.json"
  if family_fresh mistral7b; then pass "a freshly-touched manifest is correctly considered fresh"
  else fail "a freshly-touched manifest was incorrectly considered stale"; fi

  echo "=================================================================="
  if [ "$FAIL" -eq 0 ]; then echo "SELFTEST: ALL CHECKS PASSED"; else echo "SELFTEST: FAILURES ABOVE"; fi
  rm -rf "$TMP"
  return "$FAIL"
}

# --------------------------------------------------------------- entry ---
if [ "$SELFTEST" -eq 1 ]; then
  run_selftest
  exit $?
fi

mkdir -p "$OUTD" "$H/cloud/logs"
echo $$ > "$PIDFILE"
log "start: families=[$FAMILIES] watch=$WATCH interval=${INTERVAL_MIN}m max_cycles=$MAX_CYCLES dryrun=$DRYRUN"

if [ "$WATCH" -eq 1 ]; then
  cycle_n=0
  while :; do
    if [ -f "$STOPFILE" ]; then log "STOP-file present — exiting"; break; fi
    run_cycle; pending=$?
    cycle_n=$((cycle_n + 1))
    if [ "$pending" -eq 0 ]; then
      log "all tracked families analyzed — exiting watch loop"
      break
    fi
    if [ "$MAX_CYCLES" -gt 0 ] && [ "$cycle_n" -ge "$MAX_CYCLES" ]; then
      log "MAX_CYCLES=$MAX_CYCLES reached with families still pending — exiting watch loop"
      break
    fi
    log "cycle $cycle_n done: families still pending — sleeping ${INTERVAL_MIN}m"
    sleep "$((INTERVAL_MIN * 60))"
  done
else
  run_cycle
  log "one-shot cycle complete"
fi
log "analyze_families.sh exiting"
