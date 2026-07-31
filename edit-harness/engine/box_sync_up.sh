#!/usr/bin/env bash
# box_sync_up.sh — push CODE ONLY to a remote box. Run ON THE LAPTOP.
#
# House rule: RSYNC CODE FIRST, never the whole tree. Model weights (67G) and raw result
# matrices (12G) must never cross the wire — they are downloaded on-box or stay home.
# The 07-14 campaign burned a day on a missing sync filter, so every include/exclude here
# is EXPLICIT and this script has no "sync everything" mode.
#
# Usage:
#   bash engine/box_sync_up.sh <host>            # DRY RUN (default — shows what would move)
#   bash engine/box_sync_up.sh <host> --go       # actually transfer
#   PORT=12345 bash engine/box_sync_up.sh <host> --go
#   DEST=/root/edit-harness bash engine/box_sync_up.sh <host> --go
#
# <host> is anything ssh accepts, e.g. root@connect.cqa1.seetacloud.com

set -u
HOST="${1:-}"
GO="${2:-}"
PORT="${PORT:-22}"
DEST="${DEST:-/root/edit-harness}"
WORKSPACE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

if [ -z "$HOST" ]; then
  echo "usage: $0 <host> [--go]   (dry-run unless --go)"; exit 2
fi
cd "$(dirname "$0")/.." || exit 3     # edit-harness/

DRY="--dry-run"
[ "$GO" = "--go" ] && DRY=""
[ -n "$DRY" ] && echo "=== DRY RUN (add --go to transfer) ==="

# ---------------------------------------------------------------- what moves
# Explicit allow-list. Anything not named here does NOT go.
INCLUDES=(
  --include='experiments/'            --include='experiments/**'
  --include='editors/'                --include='editors/**'
  --include='engine/'                 --include='engine/*.sh'
  --include='cloud/'                  --include='cloud/**'
  --include='onbox/'                  --include='onbox/**'
  --include='run_*.sh'
  --include='requirements-box-waves.txt'
  --include='*.py'
)
# Hard exclusions applied BEFORE the includes can match anything heavy.
EXCLUDES=(
  --exclude='data/'                   # 67G of weights — downloaded on-box
  --exclude='results/'                # raw npz stay home; canonical tables go via --include below
  --exclude='archive/'
  --exclude='**/__pycache__/'         --exclude='*.pyc'
  --exclude='**/.omc/'
  --exclude='*.npz' --exclude='*.pt' --exclude='*.safetensors' --exclude='*.bin' --exclude='*.gguf'
  --exclude='*.pid' --exclude='*.log' --exclude='*.nohup.log'
  --exclude='**/.synthetic-relabel-bak/'
  --exclude='engine/archive/'
  --exclude='figures*/'
)

echo "--- phase 1: code -> $HOST:$DEST"
rsync -az $DRY -e "ssh -p $PORT -o ServerAliveInterval=30" \
  "${EXCLUDES[@]}" "${INCLUDES[@]}" --include='*/' --exclude='*' \
  ./ "$HOST:$DEST/" || { echo "rsync FAILED"; exit 4; }

# ---------------------------------------------------------------- canonical tables
# A few small JSONs the on-box analyses read. Named individually — no globs over results/.
echo "--- phase 2: canonical result tables (small JSONs only)"
TABLES=$(ls results/*.json results/merging/*.json 2>/dev/null | head -200)
if [ -n "$TABLES" ]; then
  tmp=$(mktemp)
  printf '%s\n' $TABLES > "$tmp"
  rsync -az $DRY -e "ssh -p $PORT -o ServerAliveInterval=30" \
    --files-from="$tmp" ./ "$HOST:$DEST/" || echo "table sync FAILED (non-fatal)"
  rm -f "$tmp"
  echo "    ($(printf '%s\n' $TABLES | wc -l) table files)"
else
  echo "    (none found)"
fi

# ---------------------------------------------------------------- named gate inputs
# Deletion Wave 1 reuses these existing insertion twins instead of buying duplicate cells.
# They are small, explicit exceptions to the raw-results exclusion; all other results stay home.
echo "--- phase 2b: matched deletion gate inputs"
TWIN_LIST=$(mktemp)
printf '%s\n' \
  results/matrices/gate_mistral7b_rome_cf_L24_s0.npz \
  results/matrices/gate_mistral7b_rome_cf_L24_s1.npz \
  results/matrices/gate_mistral7b_rome_cf_L24_s2.npz \
  results/matrices/gate_llama8b_rome_cf_L24_s0.npz \
  results/matrices/gate_llama8b_rome_cf_L24_s1.npz \
  results/matrices/gate_llama8b_rome_cf_L24_s2.npz > "$TWIN_LIST"
rsync -az $DRY -e "ssh -p $PORT -o ServerAliveInterval=30" \
  --files-from="$TWIN_LIST" ./ "$HOST:$DEST/" \
  || { echo "matched twin sync FAILED"; rm -f "$TWIN_LIST"; exit 6; }
rm -f "$TWIN_LIST"

# ---------------------------------------------------------------- required small inputs + gate receipts
# CounterFact is 44MB and every prepared wave reads it. Receipts are absent until their
# local prerequisite passes; sync only those that exist, never fabricate one.
echo "--- phase 2c: dataset + available gate receipts"
INPUT_LIST=$(mktemp)
printf '%s\n' data/counterfact.json > "$INPUT_LIST"
for receipt in \
  engine/DELETION_PHASEL_GD1_PASS.ok \
  engine/DELETION_PHASEL_GD2_PASS.ok \
  engine/DELETION_PHASEL_TEXT_PASS.ok \
  engine/DELETION_WAVE1_GD3_PASS.ok \
  engine/PAPERB_CURVE_GS3_PASS.ok; do
  [ -f "$receipt" ] && printf '%s\n' "$receipt" >> "$INPUT_LIST"
done
rsync -az $DRY -e "ssh -p $PORT -o ServerAliveInterval=30" \
  --files-from="$INPUT_LIST" ./ "$HOST:$DEST/" \
  || { echo "dataset/receipt sync FAILED"; rm -f "$INPUT_LIST"; exit 7; }
rm -f "$INPUT_LIST"

# ---------------------------------------------------------------- binding preregistrations
rsync -az $DRY -e "ssh -p $PORT -o ServerAliveInterval=30" \
  --include='/docs/' --include='/docs/plans/' --include='/docs/plans/PREREG-*.md' \
  --exclude='*' "$WORKSPACE_ROOT/" "$HOST:$DEST/" \
  || { echo "prereg sync FAILED"; exit 5; }

echo
if [ -n "$DRY" ]; then
  echo "=== DRY RUN COMPLETE — nothing transferred. Re-run with --go ==="
else
  echo "=== SYNC COMPLETE ==="
  echo "Next on the box:  bash engine/box_bootstrap.sh"
fi
