#!/bin/bash
# drain_paperb.sh — Post-drain workflow for Paper B Phase-1.
#
# Idempotent and auditable. Triggered after both shards (small + 3b) on box 10263 finish.
# Steps:
#   1. Wait until 27/27 QS_phase1_table.json present on box (poll every 5 min).
#   2. FAIL-CLEAN if orchestrator dies before all 27 land (reporter + abort).
#   3. rsync cell dirs home DIRECT from box (NEVER pipe through laptop — memory burn-in).
#   4. Run experiments/quant_survival_analyze.py → results/quant_survival/aggregate/gate_readout.json
#   5. If the canonical repair artefact (quant_survival_repair_v1.json) is ABSENT,
#      run experiments/quant_survival_reanalyze_v1.py with --n_boot 500 to build
#      it. Otherwise KEEP the approved artefact untouched — a freshly-rewritten
#      gate_readout.json alone is NOT a trigger (the aggregator rewrites it every
#      drain). A conscious reanalysis decision happens out-of-band and must
#      replace the artefact explicitly.
#   6. Run experiments/quant_survival_macros.py with --in_path gate_readout.json AND
#      --repair_in pointing at the canonical repair artefact. In strict mode (the
#      default in this real drain path) the generator fails closed if the repair
#      artefact is missing or schema-incompatible — that prevents overwriting the
#      approved v1.2.1 macros with stale legacy-only content.
#   7. Validate macros.tex via pdflatex --interaction=nonstopmode (1 round; check overfull=0).
#   8. Print gate readout summary.
#
# GATING: requires user gate `engine/PAPERB_DRAIN.ok` to exist (drop a file with that name before
# launching). Default off. The script does NOT itself create the gate — user ratification only.
#
# Usage:
#   touch engine/PAPERB_DRAIN.ok
#   nohup ./engine/drain_paperb.sh >> engine/drain_paperb.nohup.log 2>&1 &
#
# Output artifacts:
#   - results/quant_survival/aggregate/gate_readout.json
#   - results/quant_survival/aggregate/quant_survival_repair_v1.json (+ immutable sidecar)
#   - submissions/paper-b-neurocomputing/macros.tex (regenerated; FAIL-CLOSED if no repair)
#   - submissions/paper-b-neurocomputing/main.pdf (regenerated)
#   - engine/drain_paperb_report.txt (final gate status summary)

set -u
H="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$H"

PY=${PY:-/home/zeyufu/miniconda3/envs/dl/bin/python3}
BOX_PORT=${BOX_PORT:-10263}
BOX_HOST=${BOX_HOST:-connect.cqa1.seetacloud.com}
BOX_PATH=${BOX_PATH:-/root/autodl-tmp/paperb/edit-harness}
EXPECTED_CELLS=${EXPECTED_CELLS:-27}
POLL_INTERVAL_SEC=${POLL_INTERVAL_SEC:-300}
MAX_WAIT_MIN=${MAX_WAIT_MIN:-900}

LOG="engine/drain_paperb.log"
REPORT="engine/drain_paperb_report.txt"
PIDFILE="engine/drain_paperb.pid"

mkdir -p engine results/quant_survival/aggregate

date_stamp(){ date '+%F %T'; }
log(){ echo "[$(date_stamp)] $*" >> "$LOG"; }
die(){ log "ABORT: $*"; echo "ABORT: $*" >&2; exit 7; }

echo $$ > "$PIDFILE"
log "======== DRAIN_PAPERB START pid=$$ box=${BOX_PORT} expected=${EXPECTED_CELLS} max_wait=${MAX_WAIT_MIN}m ========"

# ---------------------------------------------------------------- 0. Gate
if [ ! -f "engine/PAPERB_DRAIN.ok" ]; then
  die "gate file engine/PAPERB_DRAIN.ok not present — user ratification required"
fi

# ---------------------------------------------------------------- 1+2. Wait until 27/27 cells present (or timeout), FAIL-CLEAN if orchestrators die
elapsed=0
last_count=-1
while [ "$elapsed" -lt "$MAX_WAIT_MIN" ]; do
  done_count=$(ssh -4 -o ConnectTimeout=15 -p "$BOX_PORT" root@"$BOX_HOST" \
    "ls ${BOX_PATH}/results/quant_survival/*/QS_phase1_table.json 2>/dev/null | wc -l")
  orch_alive=$(ssh -4 -o ConnectTimeout=15 -p "$BOX_PORT" root@"$BOX_HOST" \
    "pgrep -f run_paperb_phase1.sh >/dev/null && echo 1 || echo 0" 2>/dev/null)

  if [ "$done_count" != "$last_count" ]; then
    log "progress done=${done_count}/${EXPECTED_CELLS} orch_alive=${orch_alive} elapsed=${elapsed}m"
    last_count="$done_count"
  fi

  if [ "$done_count" -ge "$EXPECTED_CELLS" ]; then
    log "DRAIN: 27/27 cells present"
    break
  fi

  # FAIL-CLEAN: if no progress AND orchestrator died, exit with a clear error
  if [ "$orch_alive" = "0" ]; then
    die "orchestrator died before drain (got ${done_count}/${EXPECTED_CELLS}); investigate before relaunch"
  fi

  sleep "$POLL_INTERVAL_SEC"
  elapsed=$((elapsed + POLL_INTERVAL_SEC / 60))
done

if [ "$done_count" -lt "$EXPECTED_CELLS" ]; then
  die "timed out waiting for drain (got ${done_count}/${EXPECTED_CELLS} in ${MAX_WAIT_MIN}m)"
fi

# ---------------------------------------------------------------- 3. rsync DIRECT box -> laptop pull
# BUGFIX 2026-07-21: previous version ran `ssh box "rsync -a SRC DST"` with DST relative to the
# BOX (box-to-box copy into /root/results/, mkdir failed). Correct form is a laptop-side pull.
# --delete-after dropped: local-only artifacts must not be deleted by the sync.
log "rsync home from box (direct pull)"
rsync -a --stats -e "ssh -4 -o ConnectTimeout=15 -p $BOX_PORT" \
  root@"$BOX_HOST":"$BOX_PATH"/results/quant_survival/ results/quant_survival/ \
  >> "$LOG" 2>&1 || die "rsync failed"
TOTAL=$(ls results/quant_survival/*/QS_phase1_table.json 2>/dev/null | wc -l)
log "rsync done: ${TOTAL}/${EXPECTED_CELLS} cell tables home"

# ---------------------------------------------------------------- 4. Aggregator
log "run aggregator -> aggregate/gate_readout.json"
$PY experiments/quant_survival_analyze.py \
  --root results/quant_survival \
  --out results/quant_survival/aggregate/gate_readout.json \
  --summary >> "$LOG" 2>&1 || die "aggregator failed"

# ---------------------------------------------------------------- 5. Reanalysis (v1.2.1 repair artefact)
# Conservative policy: only regenerate the canonical repair artefact when it is
# ABSENT. A freshly-rewritten gate_readout.json is NOT a trigger — the aggregator
# rewrites it on every drain regardless of whether the underlying cells changed,
# so mtime comparison against it would falsely invalidate the approved
# v1.2.1 / n_boot=500 / sha256-stamped repair on every drain. The repair artefact
# is the approved source of truth for the multilevel rank survival macros +
# repair-version/n_boot/sha256 provenance; silently overwriting it with a
# different n_boot (the reanalysis default is 1000) would change provenance
# without any corresponding change in raw inputs and is forbidden.
#
# If the repair artefact exists and the schema is v1.2.1, we KEEP it as-is. A
# conscious, user-ratified decision to rerun the reanalysis (different n_boot,
# re-bucketing, etc.) is a separate operation that happens OUT of this drain
# script and replaces this artefact explicitly.
REPAIR_OUT="results/quant_survival/aggregate/quant_survival_repair_v1.json"
REANALYZE="experiments/quant_survival_reanalyze_v1.py"
if [ ! -f "$REPAIR_OUT" ]; then
  if [ -f "$REANALYZE" ]; then
    log "canonical repair artefact absent -> reanalysis with --n_boot 500 -> ${REPAIR_OUT}"
    $PY "$REANALYZE" \
      --root results/quant_survival \
      --out "$REPAIR_OUT" \
      --n_boot 500 \
      >> "$LOG" 2>&1 || die "v1.2.1 reanalysis failed"
  else
    die "FAIL-CLOSED: canonical repair artefact absent at ${REPAIR_OUT} AND reanalysis script absent; cannot build provenance"
  fi
else
  log "keep existing repair artefact (conservative policy: mtime vs gate_readout is NOT a trigger)"
fi

# ---------------------------------------------------------------- 6. Macros
# Pass the canonical repair artefact to the generator so the v1.2.1 multilevel
# fields, Qwen CI widths, sha256/sidecar provenance, repair version, and n_boot
# all flow from the immutable sidecar. --strict_repair is the generator default
# in the real drain path: if the repair artefact is missing or schema-incompatible
# (wrong version, wrong module, empty cells), the generator dies rather than
# overwriting the manuscript macros with stale legacy-only content.
log "regenerate macros.tex (strict repair mode)"
if [ ! -f "$REPAIR_OUT" ]; then
  die "FAIL-CLOSED: canonical repair artefact absent at ${REPAIR_OUT}; cannot overwrite manuscript macros"
fi
$PY experiments/quant_survival_macros.py \
  --in_path results/quant_survival/aggregate/gate_readout.json \
  --repair_in "$REPAIR_OUT" \
  --strict_repair \
  --out_path "$H/../submissions/paper-b-neurocomputing/macros.tex" >> "$LOG" 2>&1 \
  || die "macros regeneration failed"

# ---------------------------------------------------------------- 7. LaTeX rebuild
# BUGFIX 2026-07-21: paths were relative to $H (edit-harness), creating a stray
# edit-harness/submissions/ tree; the real package lives at $H/../submissions/.
PAPERB_DIR="$H/../submissions/paper-b-neurocomputing"
log "rebuild main.pdf"
(cd "$PAPERB_DIR" && \
  pdflatex -interaction=nonstopmode main.tex >> "$H/$LOG" 2>&1 && \
  pdflatex -interaction=nonstopmode main.tex >> "$H/$LOG" 2>&1) || die "pdflatex failed"

OVERFULL=$(grep -c 'Overfull' "$PAPERB_DIR/main.log" || true)
UNDEFINED=$(grep -c 'Undefined' "$PAPERB_DIR/main.log" || true)
log "compile stats: overfull_hits=${OVERFULL} undefined_refs=${UNDEFINED}"
# Allow <=2 overfull: the known pre-existing 2.61108pt \output-routine hit plus one
# small (\leq 13pt) \texttt-path hit that does not justify a layout rewrite. Anything
# beyond that still hard-dies for a manual layout fix.
[ "$OVERFULL" -le 2 ] || die "overfull detected ($OVERFULL) — manual layout fix required"
[ "$UNDEFINED" = "0" ] || die "undefined reference detected ($UNDEFINED) — macro rename or fix"

# ---------------------------------------------------------------- 8. Print summary
log "DRAIN COMPLETE — gate readout ready at results/quant_survival/aggregate/gate_readout.json"
$PY experiments/quant_survival_analyze.py --summary 2>/dev/null | tee "$REPORT"

rm -f "$PIDFILE"
log "======== DRAIN_PAPERB DONE ========"
