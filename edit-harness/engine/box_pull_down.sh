#!/usr/bin/env bash
# box_pull_down.sh — pull results home against an EXPLICIT manifest. Run ON THE LAPTOP.
#
# Never invoke cloud/sync_results.sh unmodified: its bare default include
# (matrices/*.npz) pulls every npz it finds, which violates the house policy that raw npz
# stay on-box unless a named analysis needs them. This script requires a manifest file and
# pulls exactly what it lists.
#
# Manifest format: one exact repo-relative path per line; blank lines and #comments ignored.
# Globs are rejected because rsync --files-from treats them as literal filenames.
# Each experiment's driver writes its own manifest.
#
# Usage:
#   bash engine/box_pull_down.sh <host> <manifest>          # DRY RUN
#   bash engine/box_pull_down.sh <host> <manifest> --go
#   PORT=12345 SRC=/root/edit-harness bash engine/box_pull_down.sh ... --go
#
# Never deletes on either side. Verifies every manifest entry arrived and reports misses.

set -u
HOST="${1:-}"; MANIFEST="${2:-}"; GO="${3:-}"
PORT="${PORT:-22}"
SRC="${SRC:-/root/edit-harness}"

if [ -z "$HOST" ] || [ -z "$MANIFEST" ]; then
  echo "usage: $0 <host> <manifest-file> [--go]"; exit 2
fi
[ -f "$MANIFEST" ] || { echo "manifest not found: $MANIFEST"; exit 2; }
cd "$(dirname "$0")/.." || exit 3

DRY="--dry-run"; [ "$GO" = "--go" ] && DRY=""
[ -n "$DRY" ] && echo "=== DRY RUN (add --go to transfer) ==="

# strip comments/blanks into an rsync files-from list
LIST=$(mktemp)
trap 'rm -f "$LIST"' EXIT
grep -vE '^\s*(#|$)' "$MANIFEST" > "$LIST"
if grep -qE '[*?\[]' "$LIST"; then
  echo "manifest contains a glob, but rsync --files-from requires exact paths:" >&2
  grep -nE '[*?\[]' "$LIST" >&2
  exit 4
fi
n=$(wc -l < "$LIST")
echo "--- manifest: $MANIFEST ($n entries)"
echo "--- pulling $HOST:$SRC -> $(pwd)"

rsync -azv $DRY -e "ssh -p $PORT -o ServerAliveInterval=30" \
  --files-from="$LIST" "$HOST:$SRC/" ./ 2>&1 | tail -20
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then
  echo "rsync exited rc=$rc" >&2
  exit "$rc"
fi

# ---------------------------------------------------------------- verify
if [ -z "$DRY" ]; then
  echo
  echo "--- verification (expected vs on disk)"
  miss=0; got=0
  while IFS= read -r p; do
    if [ -e "$p" ]; then
      got=$((got+1))
      if command -v sha256sum >/dev/null 2>&1 && [ -f "$p" ] && [ "$(stat -c%s "$p")" -lt 52428800 ]; then
        echo "  OK   $(sha256sum "$p" | cut -c1-12)  $p"
      else
        echo "  OK   $p"
      fi
    else
      miss=$((miss+1)); echo "  MISS $p"
    fi
  done < "$LIST"
  echo
  echo "=== PULL SUMMARY: $got arrived, $miss missing ==="
  if [ "$miss" -gt 0 ]; then
    echo "!! do NOT shut the box down until the misses are resolved or explained" >&2
    exit 5
  fi
fi
