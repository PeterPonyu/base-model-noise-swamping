#!/usr/bin/env bash
# Phi-3.5 insertion-gate + deletion rerun on the FIXED tokenizer helper (defect 2026-07-30).
# Reproduces gate_phi35_rome_cf_L16_s{0,1,2} and u1e0_phi35_delete_refusal_L16_s{0,1,2}.
set -u
H=/root/edit-harness-deploy-20260727; cd "$H" || exit 2
PY=/root/autodl-tmp/venvs/ifa-20260727/bin/python
LOG=engine/run_phi_b6_refix.log
PIDFILE=engine/run_phi_b6_refix.pid
echo $$ > "$PIDFILE"; trap 'rm -f "$PIDFILE"' EXIT
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
# ---- idle gate: wait for the Phi RG rerun to drain (util<25 && mem<1500, 3x) ----
log "waiting for GPU idle (phi rg rerun may still be running)"
consec=0
while true; do
  line=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  util=$(echo "$line" | cut -d, -f1 | tr -d " "); mem=$(echo "$line" | cut -d, -f2 | tr -d " ")
  if [ "${util:-999}" -lt 25 ] && [ "${mem:-99999}" -lt 1500 ]; then
    consec=$((consec+1))
  else
    consec=0
  fi
  [ "$consec" -ge 3 ] && break
  sleep 30
done
log "GPU idle — window opens"
ENVP="env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1"
CF="--data data/counterfact.json"
rc_all=0
run_row(){  # tag OUT CMD...
  tag="$1"; out="$2"; shift 2
  [ -e "$out" ] && { log "SKIP $tag (exists)"; return 0; }
  log "RUN $tag"
  $* >> "engine/phi_b6_${tag}.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ] && [ -e "$out" ]; then log "DONE $tag"; else log "FAIL $tag rc=$rc"; rc_all=$rc; fi
}
# insertion gate cells: rome rewrite, n_edits 200, n_probes 500, lr 0.1, L16, seeds 0/1/2
for s in 0 1 2; do
  run_row "ins_s$s" "results/matrices/gate_phi35_rome_cf_L16_s${s}.npz" \
    $ENVP $PY experiments/killgate_keygeom.py --model data/models/Phi-3.5-mini --editor rome \
      $CF --n_edits 200 --n_probes 500 --lr 0.1 --layer 16 --seed $s \
      --save_matrices --matrix_dir results/matrices \
      --out results/gate_phi35_rome_cf_L16_s${s}.json
done
# deletion cells: rome delete refusal, same grid
for s in 0 1 2; do
  run_row "del_s$s" "results/u1e0_phi35_delete_refusal_L16_s${s}.json" \
    $ENVP $PY experiments/killgate_keygeom.py --model data/models/Phi-3.5-mini --editor rome \
      --edit_mode delete --delete_variant refusal $CF --n_edits 200 --n_probes 500 \
      --lr 0.1 --layer 16 --seed $s --save_matrices --matrix_dir results/matrices \
      --out results/u1e0_phi35_delete_refusal_L16_s${s}.json
done
log "======== PHI B6 REFIX END rc=$rc_all ========"
exit $rc_all
