#!/usr/bin/env bash
# Pull results manifest for Paper B H11 missing cells wave.
# Usage: bash engine/box_paperb_h11_pull.sh REMOTE_HOST [REMOTE_HARNESS]
set -u

REMOTE_HOST="${1:-}"
REMOTE_H="${2:-/root/edit-harness-deploy-20260727}"
REMOTE_PORT="${REMOTE_PORT:-36039}"
LOCAL_H="${HARNESS:-$(cd "$(dirname "$0")/.." && pwd)}"

[ -n "$REMOTE_HOST" ] || {
  echo "usage: $0 REMOTE_HOST [REMOTE_HARNESS]" >&2
  exit 2
}

log() { echo "[paperb-h11-pull] $*"; }

# Files to pull (5 cells x 2 files each + logs + readout + receipt if exists)
CELLS=(
  "gemma2b_rome_L19_s2"
  "qwen3b_rome_L27_s2"
  "phi35_rome_L24_s0"
  "phi35_rome_L24_s1"
  "phi35_rome_L24_s2"
)

# Create pull manifest
MANIFEST="$LOCAL_H/engine/paperb_h11_pull_manifest.txt"
PULLED_MANIFEST="$LOCAL_H/engine/paperb_h11_pulled_this_run.txt"
> "$MANIFEST"
> "$PULLED_MANIFEST"

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

log "Pulling from $REMOTE_HOST:$REMOTE_PORT:$REMOTE_H"
log "Manifest: $(wc -l < "$MANIFEST") files"

# Pull files (create missing directories; stage critical files until pair validation)
PY="${PY:-python3}"
pulled=0
staged=0
skipped=0
is_critical() {
  case "$1" in
    results/quant_survival_curve/*/QS_phase1_table.json|results/quant_survival_curve/*/QS_phase1_raw.npz)
      return 0 ;;
    *)
      return 1 ;;
  esac
}

while IFS= read -r relpath; do
  remote_path="$REMOTE_HOST:$REMOTE_H/$relpath"
  local_path="$LOCAL_H/$relpath"
  local_dir=$(dirname "$local_path")
  pull_tmp="${local_path}.pull-tmp-$$"

  mkdir -p "$local_dir"
  rm -f "$pull_tmp"

  if rsync -az \
      -e "ssh -4 -p $REMOTE_PORT -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=30" \
      --timeout=90 \
      "$remote_path" "$pull_tmp" 2>/dev/null; then
    if [ -s "$pull_tmp" ]; then
      if is_critical "$relpath"; then
        log "staged $relpath ($(stat -c%s "$pull_tmp" 2>/dev/null || stat -f%z "$pull_tmp" 2>/dev/null || echo '?') bytes)"
        staged=$((staged + 1))
      else
        mv -f "$pull_tmp" "$local_path"
        size=$(stat -c%s "$local_path" 2>/dev/null || stat -f%z "$local_path" 2>/dev/null || echo "?")
        echo "$relpath" >> "$PULLED_MANIFEST"
        log "pulled $relpath ($size bytes)"
        pulled=$((pulled + 1))
      fi
    else
      rm -f "$pull_tmp"
      log "skip $relpath (not present or empty on remote)"
      skipped=$((skipped + 1))
    fi
  else
    rm -f "$pull_tmp"
    log "WARN: rsync failed for $relpath"
    skipped=$((skipped + 1))
  fi
done < "$MANIFEST"

# Validate and commit each critical table/raw pair only after both files arrived.
# Existing local artifacts remain untouched when the remote pair is incomplete or invalid.
missing_critical=0
for cell in "${CELLS[@]}"; do
  rel_dir="results/quant_survival_curve/${cell}"
  table_rel="$rel_dir/QS_phase1_table.json"
  raw_rel="$rel_dir/QS_phase1_raw.npz"
  table_tmp="$LOCAL_H/$table_rel.pull-tmp-$$"
  raw_tmp="$LOCAL_H/$raw_rel.pull-tmp-$$"
  table_local="$LOCAL_H/$table_rel"
  raw_local="$LOCAL_H/$raw_rel"

  if [ ! -s "$table_tmp" ]; then
    log "MISSING CRITICAL: $table_rel"
    missing_critical=$((missing_critical + 1))
    rm -f "$raw_tmp"
    continue
  fi
  if [ ! -s "$raw_tmp" ]; then
    log "MISSING CRITICAL: $raw_rel"
    missing_critical=$((missing_critical + 1))
    rm -f "$table_tmp"
    continue
  fi

  if ! "$PY" - "$table_tmp" "$raw_tmp" <<'PY'
import json
import sys
import numpy as np

table_path, raw_path = sys.argv[1], sys.argv[2]
d = json.load(open(table_path))
a = np.load(raw_path, allow_pickle=True)
s = d.get("runner_stamp") or {}
need = {"code_sha256", "pid", "hostname", "wall_start", "wall_end", "elapsed_s", "nvidia_smi_sample"}
assert not (need - set(s)), f"Missing runner_stamp fields: {need - set(s)}"
assert json.loads(str(a["runner_stamp_json"].item()))["code_sha256"] == s["code_sha256"], "code_sha256 mismatch"
assert a["COS"].shape == (200, 200), f"COS shape={a['COS'].shape}, expected (200,200)"
assert d["editor"] == "rome", f"editor={d['editor']}, expected rome"
assert d["codec"] == "real", f"codec={d['codec']}, expected real"
PY
  then
    log "INVALID CRITICAL: $cell (remote pair retained in pull temp; local artifact untouched)"
    rm -f "$table_tmp" "$raw_tmp"
    missing_critical=$((missing_critical + 1))
    continue
  fi

  mv -f "$table_tmp" "$table_local"
  mv -f "$raw_tmp" "$raw_local"
  printf '%s\n%s\n' "$table_rel" "$raw_rel" >> "$PULLED_MANIFEST"
  pulled=$((pulled + 2))
  log "validated and pulled $cell"
done

rm -f "$LOCAL_H"/results/quant_survival_curve/*/*.pull-tmp-$$ 2>/dev/null || true

log "COMPLETE: pulled=$pulled staged=$staged skipped=$skipped"

if [ "$missing_critical" -gt 0 ]; then
  log "WARNING: $missing_critical critical cell pair(s) missing or invalid from this pull"
  exit 3
fi

log "All critical files pulled and validated"
exit 0
