#!/bin/bash
# cloud/sync_results.sh — run ON LOCAL. Pulls the cloud box's results/ back into the
# local edit-harness/results/, merging by TAG_RE filename. --update means rsync never
# overwrites a destination file with an equal-or-older-mtime source file — belt-and-
# suspenders on top of the filename-uniqueness guarantee (seed-suffixed names mean
# local s0 and cloud s1/s2 files never share a name in the first place).
#
# Two source modes:
#   --host <ip> [--port 22] [--key ~/.ssh/id] [--remote-root /path]   real SSH pull
#   --local-src <path>                                                 local-path pull
#     (a staging dir already copied off the box, or — used by cloud/selftest.sh — a
#     synthetic fake "remote" to CPU-test the merge logic without SSH/network)
#
# Usage:
#   bash cloud/sync_results.sh --host 123.45.67.8 --port 22022 --key ~/.ssh/id_autodl
#   bash cloud/sync_results.sh --host ... --dry-run     # preview only, nothing written
set -u
H="$(cd "$(dirname "$0")/.." && pwd)"
cd "$H" || exit 1

HOST=""; PORT=22; KEY=""; DRY=""; LOCAL_SRC=""
REMOTE_ROOT="/root/edit-harness"   # ASSUMPTION FLAGGED: adjust to the cloud box's actual clone path
while [ $# -gt 0 ]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --key) KEY="$2"; shift 2 ;;
    --remote-root) REMOTE_ROOT="$2"; shift 2 ;;
    --local-src) LOCAL_SRC="$2"; shift 2 ;;
    --dry-run) DRY="--dry-run"; shift ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

log(){ echo "[sync_results $(date '+%F %T')] $*"; }

if [ -n "$LOCAL_SRC" ]; then
  SRC="${LOCAL_SRC%/}/results/"
  log "LOCAL-SRC mode: ${SRC} -> ${H}/results/  ${DRY:+(dry-run)}"
  # shellcheck disable=SC2086
  rsync -avz $DRY --update \
    --include='*.json' --include='*/' --include='matrices/*.npz' --exclude='*' \
    "$SRC" "${H}/results/"
elif [ -n "$HOST" ]; then
  SSH_CMD="ssh -p ${PORT}"
  [ -n "$KEY" ] && SSH_CMD="${SSH_CMD} -i ${KEY}"
  log "SSH mode: ${HOST}:${REMOTE_ROOT}/results/ -> ${H}/results/  ${DRY:+(dry-run)}"
  # shellcheck disable=SC2086
  rsync -avz $DRY --update -e "$SSH_CMD" \
    --include='*.json' --include='*/' --include='matrices/*.npz' --exclude='*' \
    "${HOST}:${REMOTE_ROOT}/results/" "${H}/results/"
else
  echo "usage: sync_results.sh --host <ip> [--port 22] [--key ~/.ssh/id] [--remote-root /root/edit-harness] [--dry-run]"
  echo "   or: sync_results.sh --local-src <path> [--dry-run]   (staging dir / selftest)"
  exit 1
fi
log "sync done"
