#!/usr/bin/env bash
# Compatibility entry point. Keep existing launchers working while the canonical
# fresh-box probe lives at engine/box_preflight.sh (H21).
set -u
exec "$(dirname "$0")/box_preflight.sh" "${1:-generic}"
