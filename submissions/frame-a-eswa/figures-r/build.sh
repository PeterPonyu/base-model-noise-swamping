#!/usr/bin/env bash
# build.sh -- standalone wrapper for the Frame-A figure generator.
# CPU-only. No GPU. Used by the Makefile targets and CI.
#
# Usage:
#   ./build.sh                # final mode (fail-closed; emits fig02-04 only if all gates PASS)
#   ./build.sh --preview      # MIX_A preview only (writes ONLY under figures-qa/, watermarked)
#   ./build.sh --preflight    # print preflight gate, do not emit any figure

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
Rscript "$HERE/make_figures_frame_a.R" "$@"