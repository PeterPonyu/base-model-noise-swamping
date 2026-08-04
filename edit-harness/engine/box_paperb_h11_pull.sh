#!/usr/bin/env bash
# Pull results manifest for Paper B H11 missing cells wave.
# Usage: bash engine/box_paperb_h11_pull.sh REMOTE_HOST [REMOTE_HARNESS]
set -u

REMOTE_HOST="${1:-}"
REMOTE_H="${2:-/root/edit-harness}"
LOCAL_H="${HARNESS:-$(cd "$(dirname "$0")/.." && pwd)}"

[ -n "$REMOTE_HOST" ] || {
  echo "usage: $0 REMOTE_HOST [REMOTE_HARNESS]" >&2
  exit 2
}

log() { echo "[paperb-h11-pull] $*"; }

# Files to pull (5 cells × 2 files each + logs + readout + receipt if exists)
CELLS=(
  "gemma2b_rome_L19_s2"
  "qwen3b_rome_L27_s2"
  "phi35_rome_L24_s0"
  "phi35_rome_L24_s1"
  "phi35_rome_L24_s2"
)

# Create pull manifest
MANIFEST="$LOCAL_H/engine/paperb_h11_pull_manifest.txt"
> "$MANIFEST"

# Cell artifacts
for cell in "${CELLS[@]}"; do
  echo "results/quant_survival_curve/${cell}/QS_phase1_table.json" >> "$MANIFEST"
  echo "results/quant_survival_curve/${cell}/QS_phase1_raw.npz" >> "$MANIFEST"
done

# Aggregate readout
echo "results/quant_survival/aggregate/curve_local_readout.json" >> "$MANIFEST"

# Receipt if it exists
echo "engine/PAPERB_CURVE_GS3_PASS.ok" >> "$MANIFEST"

# Logs
echo "engine/run_paperb_h11_missing_card0.log" >> "$MANIFEST"
echo "engine/run_paperb_h11_missing_card1.log" >> "$MANIFEST"
echo "engine/run_paperb_h11_missing_all.log" >> "$MANIFEST"
for cell in "${CELLS[@]}"; do
  tag="${cell%%_*}"
  seed="${cell##*_s}"
  echo "engine/paperb_h11_missing_${tag}_s${seed}.log" >> "$MANIFEST"
done

# Report file
echo "engine/paperb_h11_missing_report.txt" >> "$MANIFEST"

log "Pulling from $REMOTE_HOST:$REMOTE_H"
log "Manifest: $(wc -l < "$MANIFEST") files"

# Pull files (create missing directories; skip missing optional files)
pulled=0
skipped=0
while IFS= read -r relpath; do
  remote_path="$REMOTE_HOST:$REMOTE_H/$relpath"
  local_path="$LOCAL_H/$relpath"
  local_dir=$(dirname "$local_path")

  mkdir -p "$local_dir"

  if rsync -az --ignore-missing-args "$remote_path" "$local_path" 2>/dev/null; then
    if [ -f "$local_path" ]; then
      size=$(stat -c%s "$local_path" 2>/dev/null || stat -f%z "$local_path" 2>/dev/null || echo "?")
      log "pulled $relpath ($size bytes)"
      pulled=$((pulled + 1))
    else
      log "skip $relpath (not present on remote)"
      skipped=$((skipped + 1))
    fi
  else
    log "WARN: rsync failed for $relpath"
    skipped=$((skipped + 1))
  fi
done < "$MANIFEST"

log "COMPLETE: pulled=$pulled skipped=$skipped"

# Verify critical files
missing_critical=0
for cell in "${CELLS[@]}"; do
  table="$LOCAL_H/results/quant_survival_curve/${cell}/QS_phase1_table.json"
  raw="$LOCAL_H/results/quant_survival_curve/${cell}/QS_phase1_raw.npz"
  if [ ! -f "$table" ] || [ ! -f "$raw" ]; then
    log "MISSING CRITICAL: $cell"
    missing_critical=$((missing_critical + 1))
  fi
done

if [ "$missing_critical" -gt 0 ]; then
  log "WARNING: $missing_critical critical cell(s) missing"
  exit 3
fi

log "All critical files present"
exit 0
