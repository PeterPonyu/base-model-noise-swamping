#!/usr/bin/env bash
# Phi-3.5 merging RG rerun on the FIXED tokenizer helper (defect 2026-07-30).
# Reproduces results/merging/RG_operating_curve_table_phi35_L{16,24}.json exactly:
#   n_edits 200, seeds 0,1,2, group sizes 2,3,5,10,20, steps 20, lr 0.1, fp32.
set -u
H=/root/edit-harness-deploy-20260727; cd "$H" || exit 2
PY=/root/autodl-tmp/venvs/ifa-20260727/bin/python
LOG=engine/run_phi_rg_refix.log
PIDFILE=engine/run_phi_rg_refix.pid
echo $$ > "$PIDFILE"; trap 'rm -f "$PIDFILE"' EXIT
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
log "======== PHI RG REFIX START pid=$$ ========"
ENVP="env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1"
rc_all=0
for L in 16 24; do
  TABLE="results/merging/RG_operating_curve_table_phi35_L${L}_REFIX20260730.json"
  if [ -f "$TABLE" ]; then log "SKIP L${L} (table exists)"; continue; fi
  log "--- L${L} start"
  $ENVP $PY experiments/merging_m0.py --rg \
      --model data/models/Phi-3.5-mini --data data/counterfact.json \
      --n_edits 200 --layer "$L" --steps 20 --lr 0.1 --device cuda \
      --rg_seeds 0,1,2 --rg_group_sizes 2,3,5,10,20 \
      --out_dir results/merging --table_out "$TABLE" >> "$LOG" 2>&1
  rc=$?
  log "--- L${L} rc=$rc"
  [ $rc -ne 0 ] && rc_all=$rc
done
log "======== PHI RG REFIX END rc=$rc_all ========"
exit $rc_all
