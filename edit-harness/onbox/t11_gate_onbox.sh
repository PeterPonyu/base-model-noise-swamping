#!/bin/bash
# t11_gate_onbox.sh — CPU wrapper for the T1.1 depth-dissociation GATE-GRADE analysis
# (docs/plans/PREREG-T11-DEPTH-DISSOCIATION-E0-20260713.md), meant to run during a
# cheap no-card CPU-only boot of box 36039, where the raw-K vector banks (~650MB,
# llama1b L8/10/12/14 x s0/1/2 + qwen15b L14/17/21/24 x s0/1/2) live on the box disk
# and were deliberately never pulled home (project memory: this workspace's T1.1
# task brief, 2026-07-14). This script only READS them — no mutation, no GPU, no
# network — and writes a small gate-report JSON to a pull-friendly path so only
# that (KB-sized) file needs to travel home, not the vector banks themselves.
#
# Usage:
#   VECTOR_ROOT=/path/to/onbox/results/vectors ./onbox/t11_gate_onbox.sh
#   ./onbox/t11_gate_onbox.sh /path/to/onbox/results/vectors        # positional OK too
#   DRYRUN=1 ./onbox/t11_gate_onbox.sh /any/path                    # print cmd, no run
#   ARCH2_COLLATERAL_JSON=/path/to/qwen_C.json ./onbox/t11_gate_onbox.sh /path
#
# VECTOR_ROOT feeds BOTH --arch1_vector_dir and --arch2_vector_dir by default (the
# llama1b and qwen15b banks are expected side by side in one results/vectors/ dir,
# matching the naming already used at home: vectors_qv_<tag>_rome_cf_L<layer>_
# s<seed>.npz). Override ARCH1_VECTOR_DIR / ARCH2_VECTOR_DIR separately if the box
# splits them into different directories.
#
# Never pgrep/pkill a pattern (house rule); this script starts nothing long-running
# and needs no waiting, so it doesn't apply here beyond not doing it.
set -u
H="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # edit-harness/ — computed, not
cd "$H" || exit 1                                       # a hardcoded absolute path

PY="${PY:-/home/zeyufu/miniconda3/envs/dl/bin/python3}"
[ -x "$PY" ] || PY="$(command -v python3)"   # on-box fallback: the laptop's conda
[ -z "${PY:-}" ] && { echo "[t11_gate_onbox] ERROR: no usable python3 found" >&2; exit 2; }
# env path won't exist on a fresh box.

VECTOR_ROOT="${1:-${VECTOR_ROOT:-}}"
if [ -z "$VECTOR_ROOT" ]; then
  echo "[t11_gate_onbox] ERROR: no vector-bank root given." >&2
  echo "  Usage: VECTOR_ROOT=/path/to/results/vectors ./onbox/t11_gate_onbox.sh" >&2
  echo "     or: ./onbox/t11_gate_onbox.sh /path/to/results/vectors" >&2
  exit 2
fi

ARCH1_VECTOR_DIR="${ARCH1_VECTOR_DIR:-$VECTOR_ROOT}"
ARCH2_VECTOR_DIR="${ARCH2_VECTOR_DIR:-$VECTOR_ROOT}"
ARCH1_TAG="${ARCH1_TAG:-llama1b}"
ARCH1_LAYERS="${ARCH1_LAYERS:-8,12,14}"
ARCH2_TAG="${ARCH2_TAG:-qwen15b}"
ARCH2_LAYERS="${ARCH2_LAYERS:-14,17,21,24}"
ARCH2_COLLATERAL_JSON="${ARCH2_COLLATERAL_JSON:-}"
RESULTS_DIR="${RESULTS_DIR:-$H/results}"
OUT="${OUT:-$H/results/analysis/T11_gate_report.json}"
LOG="${LOG:-$H/engine/t11_gate_onbox.log}"
mkdir -p "$(dirname "$OUT")" "$(dirname "$LOG")"

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG" >/dev/null; }

log "================ T11_GATE_ONBOX START (pid $$) ================"
log "VECTOR_ROOT=$VECTOR_ROOT  ARCH1_VECTOR_DIR=$ARCH1_VECTOR_DIR  ARCH2_VECTOR_DIR=$ARCH2_VECTOR_DIR"
log "RESULTS_DIR=$RESULTS_DIR  OUT=$OUT"

CMD=( "$PY" experiments/depth_dissoc_gate.py
  --results_dir "$RESULTS_DIR"
  --arch1_vector_dir "$ARCH1_VECTOR_DIR" --arch1_tag "$ARCH1_TAG" --arch1_layers "$ARCH1_LAYERS"
  --arch2_vector_dir "$ARCH2_VECTOR_DIR" --arch2_tag "$ARCH2_TAG" --arch2_layers "$ARCH2_LAYERS"
  --out "$OUT" )
[ -n "$ARCH2_COLLATERAL_JSON" ] && CMD+=( --arch2_collateral_json "$ARCH2_COLLATERAL_JSON" )

DRYRUN="${DRYRUN:-0}"
if [ "$DRYRUN" -eq 1 ]; then
  log "DRYRUN=1 — not executing. cmd: ${CMD[*]}"
  echo "T11_GATE_ONBOX_SUMMARY DRYRUN cmd: ${CMD[*]}"
  exit 0
fi

log "RUN: ${CMD[*]}"
"${CMD[@]}" >> "$LOG" 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then
  log "FAIL rc=$rc — see $LOG"
  echo "T11_GATE_ONBOX_SUMMARY FAIL rc=$rc (analysis errored; see $LOG)"
  exit "$rc"
fi

# One-line summary pulled straight from the report. TWO regimes (see
# depth_dissoc_gate.py): LITERAL gate (literal_gate_decidable=true) -> verdict is
# the pre-registered PASS / KILL / AMBIGUOUS (bare "PASS" legitimate ONLY here).
# PROXY fallback (literal_gate_decidable=false) -> verdict is PASS_PROXY_TARGET /
# KILL / KILL_FOR_GATE_PURPOSES, NEVER bare "PASS", and literal_gate_verdict is
# UNDECIDABLE_AS_PREREGISTERED. Both printed explicitly so nobody has to open the
# JSON to see which regime fired.
SUMMARY=$("$PY" - "$OUT" <<'PYEOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    g = d.get("gate", {})
    verdict = g.get("verdict", "UNKNOWN")
    target_kind = g.get("target_kind")
    decid = g.get("literal_gate_decidable")
    literal_verdict = g.get("literal_gate_verdict")
    winner = g.get("winning_statistic") or g.get("arch1_winner_candidate")
    robust = g.get("winner_robust_to_signed_C")
    sign_regime = g.get("arch2_C_sign_regime")
    a1d = d.get("arch1", {}).get("decidable")
    a2d = d.get("arch2", {}).get("decidable")
    print(f"verdict={verdict} target_kind={target_kind} "
          f"literal_gate_decidable={decid} literal_gate_verdict={literal_verdict} "
          f"arch2_C_sign_regime={sign_regime} winning_statistic={winner} "
          f"winner_robust_to_signed_C={robust} "
          f"arch1_decidable={a1d} arch2_decidable={a2d}")
    if decid:
        pre = "IS the pre-registered outcome" if verdict in ("PASS","KILL") else "is the literal-gate outcome"
        print(f"NOTE: LITERAL two-family gate DECIDED -- '{verdict}' {pre} "
              f"(target_kind=literal_two_family, arch2_C_sign_regime={sign_regime}).")
        if verdict == "PASS_ABS_CONVENTION":
            print("NOTE: PASS_ABS_CONVENTION is NOT bare PASS -- globally-negative-C "
                  "arch, |C| convention; admissible as pre-registered ONLY after a "
                  "prereg amendment (pending user decision).")
        if verdict == "AMBIGUOUS":
            print("NOTE: AMBIGUOUS = MIXED-sign C (abs unjustified) and/or prereg "
                  "s1.3 raw-sign fragility guard (descriptive-only, not a mechanism claim).")
    elif verdict == "PASS_PROXY_TARGET":
        print("NOTE: PASS_PROXY_TARGET is NOT the pre-registered gate firing "
              "-- see literal_gate_verdict above.")
    lit = d.get("arch2", {}).get("literal", {})
    if not decid and isinstance(lit, dict) and lit.get("merge_window_failure"):
        print("NOTE: literal gate UNDECIDABLE via merge-window failure -- arch-2 "
              "merge has no qualifying g2/g3 cell at any depth (signal outside the "
              "frozen s2.2 small-g window); DISTINCT from merge-data-absent, needs a "
              "prereg amendment. See arch2.literal.reason.")
    if not decid and isinstance(lit, dict) and lit.get("D_profile_buildable"):
        print(f"NOTE: literal D_arch2 BUILDABLE at depths {lit.get('depths_C_and_M')} "
              f"but the gate is not decidable (missing stat banks); falling back to proxy.")
except Exception as e:
    print(f"verdict=UNKNOWN (report unreadable: {e})")
PYEOF
)
log "SUMMARY: $SUMMARY"
echo "T11_GATE_ONBOX_SUMMARY $SUMMARY report=$OUT"
log "================ T11_GATE_ONBOX DONE (rc=$rc) ================"
