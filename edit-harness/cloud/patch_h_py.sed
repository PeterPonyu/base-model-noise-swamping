# cloud/patch_h_py.sed — box-only H/PY portability patch (2026-07-08 wave-review B4
# fix). Applied ONLY by cloud/setup_autodl.sh's `patch-drivers` phase, ON the AutoDL
# box, to the rsynced copy of the 3 chain-locked drivers (run_ripple.sh,
# run_mquake_law.sh, run_8bcausal.sh) — never to the local repo. Companion to
# cloud/patch_idle_gate.sed (same phase, same on-box guard, applied together).
#
# Every driver hardcodes H=/home/zeyufu/Desktop/idea-feasibility-analysis/edit-harness
# and PY=/home/zeyufu/miniconda3/envs/dl/bin/python3 — neither exists on the AutoDL box
# (repo lands at e.g. /root/edit-harness, image ships its own python), so `cd "$H"`
# fails and every driver exits before doing any science. This patch:
#   - rewrites H to derive from the script's own location (portable, matches how the
#     3 WP2 drivers + run_neox20b.sh were fixed directly, since they aren't locked)
#   - adds a CLOUD_PY override line right after the original PY= line, so the box's own
#     python (exported by run_cloud_wave.sh) is used; the original line is left intact
#     as the fallback default, matching the same pattern used in the unlocked drivers.
#
# Idempotency is handled by the CALLER (setup_autodl.sh grep-guards on the literal
# `PY="${CLOUD_PY:-$PY}"` string before invoking this script), not by this file itself
# — re-running this sed alone on an already-patched file would re-append the PY line
# (the H line is safely idempotent on its own since the substituted text no longer
# matches the original pattern).
s|^H=/home/zeyufu/Desktop/idea-feasibility-analysis/edit-harness$|H="$(cd "$(dirname "$0")" \&\& pwd)"|
\|^PY=/home/zeyufu/miniconda3/envs/dl/bin/python3$|a\
PY="${CLOUD_PY:-$PY}"   # 2026-07-08 B4 fix (patched by cloud/setup_autodl.sh patch-drivers)
