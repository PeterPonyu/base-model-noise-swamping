#!/bin/bash
# cloud/selftest_fake_driver.sh — trivial fake driver used ONLY by cloud/selftest.sh to
# CPU-simulate the seed-shard launcher without real GPUs/models/downloads. Doubles as
# the minimal usage EXAMPLE for gpu_idle_lib.sh (see its header) that WP2/WP3 new
# drivers should copy: source the lib, call idle_gate_wait, honor SEED_OVERRIDE, write
# a TAG_RE-unique output so two seeds/cards never collide on disk.
set -u
H="$(cd "$(dirname "$0")/.." && pwd)"
cd "$H" || exit 1
# shellcheck source=cloud/gpu_idle_lib.sh
source cloud/gpu_idle_lib.sh
idle_gate_wait || exit 2

SEED=${SEED_OVERRIDE:-0}
CARD=${CUDA_VISIBLE_DEVICES:-none}
mkdir -p results/selftest
OUT="results/selftest/fake_driver_card${CARD}_s${SEED}.json"
printf '{"card": "%s", "seed": %s, "ts": "%s"}\n' "$CARD" "$SEED" "$(date -Iseconds)" > "$OUT"
echo "[fake_driver] card=${CARD} seed=${SEED} -> ${OUT}"
