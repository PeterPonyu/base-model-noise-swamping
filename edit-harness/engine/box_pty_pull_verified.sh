#!/usr/bin/env bash
# Pull a small terminal artifact from the 2026-07-27 SeetaCloud box whose
# non-PTY exec/SFTP/SCP channels stall. The remote file is emitted as base64
# through a forced PTY, decoded locally, and accepted only on exact SHA256.
set -euo pipefail

# BOX-SPECIFIC: PORT=36039 and the deploy-prefix allow-list below are pinned to the
# 2026-07-27 SeetaCloud instance. For any other box, pass PORT= (and HOST=) explicitly
# and extend the allow-list — do not reuse these defaults blindly.
HOST="${HOST:-root@connect.cqa1.seetacloud.com}"
PORT="${PORT:-36039}"
REMOTE_PATH="${1:-}"
LOCAL_PATH="${2:-}"

if [ -z "$REMOTE_PATH" ] || [ -z "$LOCAL_PATH" ]; then
  echo "usage: $0 REMOTE_PATH LOCAL_PATH" >&2
  exit 2
fi
case "$REMOTE_PATH" in
  /root/edit-harness-deploy-20260727/*|/root/autodl-tmp/framea-*|/root/autodl-tmp/d2-*) ;;
  *) echo "ABORT: remote path is outside an approved deployment prefix: $REMOTE_PATH" >&2; exit 3 ;;
esac

mkdir -p "$(dirname "$LOCAL_PATH")"
tmp_base64=$(mktemp)
tmp_file=$(mktemp)
cleanup(){ rm -f "$tmp_base64" "$tmp_file"; }
trap cleanup EXIT

remote_meta=$(timeout 60 ssh -tt -p "$PORT" -o BatchMode=yes -o ConnectTimeout=10 \
  -o ServerAliveInterval=5 -o ServerAliveCountMax=5 "$HOST" \
  "exec /bin/bash --noprofile --norc -c 'stty -onlcr -echo; test -f $REMOTE_PATH; sha256sum $REMOTE_PATH; wc -c $REMOTE_PATH'" \
  | tr -d '\r')
remote_sha=$(printf '%s\n' "$remote_meta" | awk 'NF>=2 && $1 ~ /^[0-9a-f]{64}$/ {print $1; exit}')
remote_size=$(printf '%s\n' "$remote_meta" | awk 'NF>=2 && $1 ~ /^[0-9]+$/ {print $1; exit}')
[ -n "$remote_sha" ] && [ -n "$remote_size" ] || { echo "ABORT: could not read remote metadata" >&2; exit 4; }

# base64 is text-safe over the PTY. stty -onlcr prevents CR insertion.
timeout 300 ssh -tt -p "$PORT" -o BatchMode=yes -o ConnectTimeout=10 \
  -o ServerAliveInterval=10 -o ServerAliveCountMax=12 "$HOST" \
  "exec /bin/bash --noprofile --norc -c 'stty -onlcr -echo; base64 -w 48000 $REMOTE_PATH'" \
  | tr -d '\r' > "$tmp_base64"
base64 -d "$tmp_base64" > "$tmp_file"
local_sha=$(sha256sum "$tmp_file" | cut -d' ' -f1)
local_size=$(stat -c '%s' "$tmp_file")
[ "$local_sha" = "$remote_sha" ] || { echo "ABORT: sha mismatch local=$local_sha remote=$remote_sha" >&2; exit 5; }
[ "$local_size" = "$remote_size" ] || { echo "ABORT: size mismatch local=$local_size remote=$remote_size" >&2; exit 6; }
mv "$tmp_file" "$LOCAL_PATH"
echo "PULL_OK sha256=$local_sha bytes=$local_size path=$LOCAL_PATH"
