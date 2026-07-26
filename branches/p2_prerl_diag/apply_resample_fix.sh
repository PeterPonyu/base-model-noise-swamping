#!/bin/bash
# apply_resample_fix.sh — swaps run_p2_resample.sh.fixed into place as
# run_p2_resample.sh, applying the 2026-07-11 post-drain fix for the 3
# vacuous conda-run-heredoc validation/rollup sites (see run_p2_resample.sh's
# header note, and run_p2_resample.sh.fixed for the diff). MUST be run only
# after the driver has drained — a running bash process re-reads its own
# script file, so swapping it in place under a live process is a burned-in
# workspace hazard. Refuses (does not kill) a still-alive pid; identity is
# NOT re-checked beyond kill -0 here because run_p2_resample.pid is a
# single-purpose pidfile written only by this driver, not a pattern to match.
# Idempotent: safe to re-run; no-op if the fix is already applied.
# Refuses (does not proceed) if the pidfile is ABSENT too -- an absent pidfile means
# liveness cannot be verified, and the safe default is to refuse rather than assume the
# driver is stopped (2026-07-11 hostile-review MINOR-3 fix).
set -u
H=/home/zeyufu/Desktop/idea-feasibility-analysis
B=$H/branches/p2_prerl_diag
LIVE="$B/run_p2_resample.sh"
FIXED="$B/run_p2_resample.sh.fixed"
BACKUP="$B/run_p2_resample.sh.pre-fix-20260711"
PIDFILE="$B/run_p2_resample.pid"
MARKER="resample_usability_rollup.py"   # only present in the fixed script

if [ -f "$LIVE" ] && grep -q "$MARKER" "$LIVE" 2>/dev/null; then
  echo "apply_resample_fix: already applied ('$MARKER' found in $LIVE) — no-op"
  exit 0
fi

if [ ! -f "$PIDFILE" ]; then
  echo "apply_resample_fix: REFUSING to apply — pidfile $PIDFILE not found, so liveness of the driver cannot be verified. If the driver truly never ran (or its pidfile was already cleaned up), confirm that by hand, then re-create $PIDFILE or adjust this script before re-running." >&2
  exit 1
fi

pid=$(cat "$PIDFILE" 2>/dev/null)
if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
  echo "apply_resample_fix: REFUSING to apply — pid $pid (from $PIDFILE) is still alive. Wait for the driver to drain, then re-run." >&2
  exit 1
fi

if [ ! -f "$FIXED" ]; then
  echo "apply_resample_fix: $FIXED not found — nothing to apply" >&2
  exit 1
fi

if [ ! -f "$LIVE" ]; then
  echo "apply_resample_fix: $LIVE not found — nothing to back up / replace" >&2
  exit 1
fi

was_exec=0
[ -x "$LIVE" ] && was_exec=1

cp -p "$LIVE" "$BACKUP"
echo "apply_resample_fix: backed up $LIVE -> $BACKUP"

mv "$FIXED" "$LIVE"
[ "$was_exec" -eq 1 ] && chmod +x "$LIVE"
echo "apply_resample_fix: moved $FIXED -> $LIVE (exec bit $([ "$was_exec" -eq 1 ] && echo preserved || echo left-off))"

if bash -n "$LIVE"; then
  echo "apply_resample_fix: bash -n OK on $LIVE"
else
  echo "apply_resample_fix: bash -n FAILED on $LIVE — restoring backup" >&2
  cp -p "$BACKUP" "$LIVE"
  exit 1
fi

echo "apply_resample_fix: DONE. $LIVE now uses compute_overthinking_gap.py --validate-sample (2 sites) + resample_usability_rollup.py (1 site) instead of the vacuous conda-run heredocs."
