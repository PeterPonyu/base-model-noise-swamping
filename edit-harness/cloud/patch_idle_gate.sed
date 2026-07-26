# cloud/patch_idle_gate.sed — box-only idle-gate bypass patch (2026-07-08 wave-review
# B3 fix). Applied ONLY by cloud/setup_autodl.sh's `patch-drivers` phase, ON the AutoDL
# box, to the rsynced copy of the 3 existing chain-locked drivers (run_ripple.sh,
# run_mquake_law.sh, run_8bcausal.sh) — never to the local repo (see setup_autodl.sh's
# phase_patch_drivers() header for the full story and the idempotency/grep-guard).
#
# Wraps each driver's inline `else` branch (the real idle-gate poll loop, run only when
# DRYRUN!=1) in an outer `if SKIP_IDLE_GATE=1 ... else <original loop> fi`, so
# SKIP_IDLE_GATE=1 (exported by cloud/run_cloud_wave.sh) short-circuits it entirely
# instead of polling unqualified `nvidia-smi | head -1` (always physical GPU0).
#
# Anchors are the two ASCII-only substrings shared verbatim by all 3 drivers' Phase 0b
# blocks (confirmed identical across files before writing this).
/skipping GPU idle gate, printing every run_row call/{
n
a\
if [ "${SKIP_IDLE_GATE:-0}" = "1" ]; then log "SKIP_IDLE_GATE-bypass (patched by cloud/setup_autodl.sh patch-drivers)"; else
}
/log "GPU idle/a\
fi
