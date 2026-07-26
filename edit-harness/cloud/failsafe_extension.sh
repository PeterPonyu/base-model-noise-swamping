#!/bin/bash
# cloud/failsafe_extension.sh — billing failsafe for the 2026-07-11 extension wave: hard
# power-off at 30h no matter what (estimate ~22-24h: ~12h per-card Track-1+pythia phase +
# ~10h dual-card Track-2 neox TP phase — see EXTENSION-WAVE-RUNBOOK.md's cost table).
# Same shape as cloud/failsafe_enhance.sh (that one: 24h cap over a ~16-18h estimate,
# same ~1.3-1.5x buffer ratio applied here). Cancel with: touch /root/NO_SHUTDOWN
sleep 108000
[ -f /root/NO_SHUTDOWN ] && exit 0
echo "[failsafe $(date)] 30h elapsed — forcing shutdown" >> /root/extension_wave.log
shutdown -h now
