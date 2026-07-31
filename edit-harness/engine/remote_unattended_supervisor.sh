#!/usr/bin/env bash
# Remote unattended terminal-state supervisor. It monitors one explicit PID,
# validates exact result paths, and emits either SUCCESS or FAILED plus a
# SHA256-stamped pull bundle. It never kills processes and never shuts down.
set -euo pipefail

H="${H:?H is required — pass the on-box deploy dir explicitly (e.g. H=/root/edit-harness); no default, a dated-dir default silently monitors the wrong tree}"
PIDFILE="${PIDFILE:?PIDFILE is required}"
MANIFEST="${MANIFEST:?MANIFEST is required}"
TAG="${TAG:?TAG is required}"
TIMEOUT_MIN="${TIMEOUT_MIN:-720}"
STATE_DIR="$H/engine/unattended/$TAG"
mkdir -p "$STATE_DIR"
rm -f "$STATE_DIR/SUCCESS" "$STATE_DIR/FAILED" "$STATE_DIR/result.tgz" "$STATE_DIR/result.tgz.sha256"

fail(){
  rc="$1"; shift
  {
    printf 'state=FAILED\nrc=%s\nat=%s\nreason=%s\n' "$rc" "$(date -u '+%FT%TZ')" "$*"
  } > "$STATE_DIR/FAILED"
  exit "$rc"
}

# Wait for the driver to publish its own pidfile. Never discover by pattern.
deadline=$(( $(date +%s) + 300 ))
while [ ! -s "$PIDFILE" ]; do
  [ "$(date +%s)" -lt "$deadline" ] || fail 10 "pidfile not created within 300 seconds: $PIDFILE"
  sleep 2
done
pid=$(tr -dc '0-9' < "$PIDFILE")
[ -n "$pid" ] || fail 11 "pidfile contains no numeric PID: $PIDFILE"
kill -0 "$pid" 2>/dev/null || fail 12 "pid $pid was not alive after pidfile creation"
printf 'pid=%s\nstarted_monitoring=%s\n' "$pid" "$(date -u '+%FT%TZ')" > "$STATE_DIR/RUNNING"

end=$(( $(date +%s) + TIMEOUT_MIN * 60 ))
while kill -0 "$pid" 2>/dev/null; do
  [ "$(date +%s)" -lt "$end" ] || fail 13 "pid $pid exceeded monitor timeout ${TIMEOUT_MIN}m"
  sleep 30
done

[ -f "$MANIFEST" ] || fail 14 "manifest missing after process exit: $MANIFEST"
missing=0
: > "$STATE_DIR/file_sha256.txt"
while IFS= read -r rel; do
  [ -n "$rel" ] || continue
  case "$rel" in /*|*'..'*) fail 15 "unsafe manifest entry: $rel" ;; esac
  if [ ! -f "$H/$rel" ]; then
    printf 'MISSING %s\n' "$rel" >> "$STATE_DIR/file_sha256.txt"
    missing=$((missing+1))
  else
    sha256sum "$H/$rel" >> "$STATE_DIR/file_sha256.txt"
  fi
done < "$MANIFEST"
[ "$missing" -eq 0 ] || fail 16 "$missing manifest entries missing"

# Preserve relative paths under H in a deterministic pull bundle.
state_rel="${STATE_DIR#"$H"/}"
tar -C "$H" -czf "$STATE_DIR/result.tgz" -T "$MANIFEST" "$state_rel/file_sha256.txt" 2>/dev/null
sha256sum "$STATE_DIR/result.tgz" > "$STATE_DIR/result.tgz.sha256"
{
  printf 'state=SUCCESS\npid=%s\nat=%s\n' "$pid" "$(date -u '+%FT%TZ')"
  cat "$STATE_DIR/result.tgz.sha256"
} > "$STATE_DIR/SUCCESS"
rm -f "$STATE_DIR/RUNNING"
