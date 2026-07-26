#!/usr/bin/env bash
# run_all.sh — consume *.json edit configs from this queue/ dir SERIALLY
# (one GPU job at a time), call runner.py on each, move finished configs to
# queue/done/. Failed configs are moved to queue/failed/ with a .log.
#
# Usage:
#   source ~/Desktop/idea-feasibility-analysis/env.sh   # unset ALL_PROXY, dl env
#   bash edit-harness/queue/run_all.sh
#
# Env knobs:
#   CONDA_ENV   conda env to use            (default: dl)
#   HARNESS     edit-harness root           (default: parent of this script)

set -u

QUEUE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS="${HARNESS:-$(dirname "$QUEUE_DIR")}"
CONDA_ENV="${CONDA_ENV:-dl}"
RUNNER="$HARNESS/runner.py"
DONE_DIR="$QUEUE_DIR/done"
FAILED_DIR="$QUEUE_DIR/failed"

mkdir -p "$DONE_DIR" "$FAILED_DIR"

# ALL_PROXY(socks) breaks HF httpx; drop it for every job (see env.sh).
export ALL_PROXY="" all_proxy=""
unset ALL_PROXY all_proxy

shopt -s nullglob
configs=("$QUEUE_DIR"/*.json)
if [ ${#configs[@]} -eq 0 ]; then
  echo "[run_all] no *.json configs in $QUEUE_DIR — nothing to do."
  exit 0
fi

echo "[run_all] found ${#configs[@]} config(s); running serially with env '$CONDA_ENV'."
n_ok=0
n_fail=0
for cfg in "${configs[@]}"; do
  name="$(basename "$cfg")"
  echo "----------------------------------------------------------------"
  echo "[run_all] >>> $name"
  log="$cfg.log"
  if env -u ALL_PROXY -u all_proxy conda run -n "$CONDA_ENV" python3 "$RUNNER" "$cfg" >"$log" 2>&1; then
    cat "$log"
    mv "$cfg" "$DONE_DIR/$name"
    mv "$log" "$DONE_DIR/$name.log"
    echo "[run_all] OK -> done/$name"
    n_ok=$((n_ok + 1))
  else
    cat "$log"
    mv "$cfg" "$FAILED_DIR/$name"
    mv "$log" "$FAILED_DIR/$name.log"
    echo "[run_all] FAILED -> failed/$name (see failed/$name.log)"
    n_fail=$((n_fail + 1))
  fi
done

echo "================================================================"
echo "[run_all] complete: $n_ok ok, $n_fail failed."
[ "$n_fail" -eq 0 ]
