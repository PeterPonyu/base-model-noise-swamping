#!/bin/bash
# run_deep_until1900.sh — the DEEP serial GPU queue (post-keystone), hard-stop 19:00 local.
#
# Front-loaded by science-per-minute (see design): E1 Qwen SxC mechanism -> E2 cross-arch
# generality -> E3 FT structural-null / KL-FT (D1) -> optional tail (E4 Llama-3B scale, E5 zsRE).
# Then a pure-CPU analysis pass (C3 nulls, C4 causal, C1 SxC table, D3 routing) that ALWAYS runs.
#
# HARD CONSTRAINTS baked in:
#   * SERIAL only — one GPU job on the card at a time. Never two concurrently.
#   * Wait on the keystone BY PID (kill -0 85833). NEVER pgrep/pkill -f a pattern
#     (watcher command lines self-match -> deadlock / self-kill).
#   * env python DIRECTLY ($PY), never `conda run -n dl`. fp32 is the killgate default.
#   * 0 download: HF_HUB_OFFLINE=1, only local models under data/models.
#   * HARD STOP 19:00 local: never START a job that cannot plausibly finish by 19:00.
#
# LAUNCH (later, when you decide — DO NOT run now, GPU is busy with the keystone):
#   cd /home/zeyufu/Desktop/idea-feasibility-analysis/edit-harness
#   nohup ./run_deep_until1900.sh >> engine/deep_until1900.nohup.log 2>&1 &
#   # PID is recorded to engine/deep_until1900.pid for later `kill -0` monitoring.
#   # Do NOT close the laptop lid (suspend -> nvidia_uvm wedge -> jobs hang at P5/0%).

set -u

H=/home/zeyufu/Desktop/idea-feasibility-analysis/edit-harness
cd "$H" || exit 1
PY=/home/zeyufu/miniconda3/envs/dl/bin/python3
LOG=engine/deep_until1900.log
KEYSTONE_PID=85833

mkdir -p engine results/matrices results
echo $$ > engine/deep_until1900.pid   # for later kill -0 monitoring

log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
have(){ compgen -G "$1" >/dev/null 2>&1; }   # true if the glob matches >=1 file

log "================ DEEP_UNTIL1900 START (pid $$) ================"

# ---------------------------------------------------------------------------
# (1) KEYSTONE ALREADY COMPLETE (g4-extend finished 2026-07-01 11:48, all 12 cells).
#     No PID wait — the prior rethread (85833) was killed after its zero-apps gpu_idle
#     gate jammed on the persistent mcp_litchron CUDA context. Go straight to the
#     GPU-idle gate below, which (unlike that gate) tolerates a small resident context.
# ---------------------------------------------------------------------------
log "keystone g4-extend already complete — skipping PID wait, going to GPU-idle gate"

# ---------------------------------------------------------------------------
# (2) GPU-IDLE GATE (gpuguard style: nvidia-smi, NOT pgrep).
#     Require utilization<10 AND memory.used<1500 MiB for 5 CONSECUTIVE polls
#     (guards the ~12-min transient stall + any trailing keystone teardown).
# ---------------------------------------------------------------------------
log "GPU-idle gate: need util<10 AND mem<1500MiB for 5 consecutive 30s polls"
consec=0
while [ "$consec" -lt 5 ]; do
  line=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  util=$(echo "$line" | awk -F, '{gsub(/[^0-9]/,"",$1); print ($1==""?"":$1)}')
  mem=$(echo "$line" | awk -F, '{gsub(/[^0-9]/,"",$2); print ($2==""?"":$2)}')
  if [ -n "$util" ] && [ -n "$mem" ] && [ "$util" -lt 10 ] && [ "$mem" -lt 1500 ]; then
    consec=$((consec + 1))
  else
    consec=0
  fi
  log "gpu poll util=${util:-NA} mem=${mem:-NA} MiB consec=${consec}/5"
  [ "$consec" -lt 5 ] && sleep 30
done
log "GPU confirmed idle — starting serial queue"

# ---------------------------------------------------------------------------
# 19:00 hard-stop epoch (today, local).
# ---------------------------------------------------------------------------
STOP_EPOCH=$(date -d "$(date +%F) 19:00" +%s)
MARGIN=120   # seconds of safety margin on top of est_minutes
consec_fail=0   # GPU-wedge guard: abort the queue after MAXFAIL consecutive failures
MAXFAIL=2       #   (a lid-close nvidia_uvm wedge hangs EVERY job at P5/0% -> don't thrash)

# ---------------------------------------------------------------------------
# (3)+(4) SERIAL QUEUE. Each row = 'tag|est_minutes|full_cmd'.
#   tag = basename of --out sans .json ; npz derives from the same tag.
#   Committed 7 jobs first (E1a,E1b,E2a,E2b,E2c,E3a,E3b), then the only-if-slack
#   tail (E4 Llama-3B scale, E5 zsRE). Every row passes the SAME skip + 19:00 gate.
#   est_minutes below is PER-EXPANSION (job total / #expansions).
# ---------------------------------------------------------------------------
ENVP="env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1"
KG="experiments/killgate_keygeom.py"
CF="--dataset counterfact --data data/counterfact.json"
COMMON="--n_edits 200 --n_probes 500 --steps 20 --save_matrices --matrix_dir results/matrices"

JOBS=(
  # ---- E1a: Qwen2.5-0.5B rome cf L12 x s0/1/2 (SxC mechanism receipt + clean arch-null) ----
  "gate_qwen05b_rome_cf_L12_s0|8|$ENVP $PY $KG --model data/models/Qwen2.5-0.5B --editor rome $CF $COMMON --lr 0.1 --layer 12 --seed 0 --out results/gate_qwen05b_rome_cf_L12_s0.json"
  "gate_qwen05b_rome_cf_L12_s1|8|$ENVP $PY $KG --model data/models/Qwen2.5-0.5B --editor rome $CF $COMMON --lr 0.1 --layer 12 --seed 1 --out results/gate_qwen05b_rome_cf_L12_s1.json"
  "gate_qwen05b_rome_cf_L12_s2|8|$ENVP $PY $KG --model data/models/Qwen2.5-0.5B --editor rome $CF $COMMON --lr 0.1 --layer 12 --seed 2 --out results/gate_qwen05b_rome_cf_L12_s2.json"
  # ---- E1b: Qwen2.5-1.5B rome cf L14 x s0/1/2 (second Qwen scale, cross-seed null) ----
  "gate_qwen15b_rome_cf_L14_s0|25|$ENVP $PY $KG --model data/models/Qwen2.5-1.5B --editor rome $CF $COMMON --lr 0.1 --layer 14 --seed 0 --out results/gate_qwen15b_rome_cf_L14_s0.json"
  "gate_qwen15b_rome_cf_L14_s1|25|$ENVP $PY $KG --model data/models/Qwen2.5-1.5B --editor rome $CF $COMMON --lr 0.1 --layer 14 --seed 1 --out results/gate_qwen15b_rome_cf_L14_s1.json"
  "gate_qwen15b_rome_cf_L14_s2|25|$ENVP $PY $KG --model data/models/Qwen2.5-1.5B --editor rome $CF $COMMON --lr 0.1 --layer 14 --seed 2 --out results/gate_qwen15b_rome_cf_L14_s2.json"
  # ---- E2a: gemma-2-2b rome cf L13 s0 (cross-arch generality) ----
  "gate_gemma2b_rome_cf_L13_s0|45|$ENVP $PY $KG --model data/models/gemma-2-2b --editor rome $CF $COMMON --lr 0.1 --layer 13 --seed 0 --out results/gate_gemma2b_rome_cf_L13_s0.json"
  # ---- E2b: Phi-3.5-mini rome cf L16 s0 (third architecture) ----
  "gate_phi35_rome_cf_L16_s0|67|$ENVP $PY $KG --model data/models/Phi-3.5-mini --editor rome $CF $COMMON --lr 0.1 --layer 16 --seed 0 --out results/gate_phi35_rome_cf_L16_s0.json"
  # ---- E2c: Qwen2.5-3B rome cf L18 s0 (within-Qwen scale point) ----
  "gate_qwen3b_rome_cf_L18_s0|52|$ENVP $PY $KG --model data/models/Qwen2.5-3B --editor rome $CF $COMMON --lr 0.1 --layer 18 --seed 0 --out results/gate_qwen3b_rome_cf_L18_s0.json"
  # ---- E3a: Llama-3.2-1B FT cf L8/10/12 s0 (G2/G3 editor structural-null on clean metric) ----
  "gate_llama1b_ft_cf_L8_s0|21|$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor ft $CF $COMMON --ft_lr 5e-3 --layer 8 --seed 0 --out results/gate_llama1b_ft_cf_L8_s0.json"
  "gate_llama1b_ft_cf_L10_s0|21|$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor ft $CF $COMMON --ft_lr 5e-3 --layer 10 --seed 0 --out results/gate_llama1b_ft_cf_L10_s0.json"
  "gate_llama1b_ft_cf_L12_s0|21|$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor ft $CF $COMMON --ft_lr 5e-3 --layer 12 --seed 0 --out results/gate_llama1b_ft_cf_L12_s0.json"
  # ---- E3b: Llama-3.2-1B KL-FT cf L8 s0 (D1 locality-regularized control) ----
  "gate_llama1b_ftkl_cf_L8_s0|21|$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor ft $CF $COMMON --ft_lr 5e-3 --ft_kl 0.1 --ft_kl_n 5 --layer 8 --seed 0 --out results/gate_llama1b_ftkl_cf_L8_s0.json"
  # ==== ONLY-IF-SLACK TAIL (not in the committed 347-min sum; same gate + skip) ====
  # ---- E4: Llama-3.2-3B rome cf L14 s0 (within-family scale) ----
  "gate_llama3b_rome_cf_L14_s0|58|$ENVP $PY $KG --model data/models/Llama-3.2-3B --editor rome $CF $COMMON --lr 0.1 --layer 14 --seed 0 --out results/gate_llama3b_rome_cf_L14_s0.json"
  # ---- E5: zsRE Llama-3.2-1B rome L10 x s0/1/2 (dataset generality) ----
  "gate_llama1b_rome_zsre_L10_s0|21|$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome --dataset zsre --data data/zsre_eval.json $COMMON --lr 0.1 --layer 10 --seed 0 --out results/gate_llama1b_rome_zsre_L10_s0.json"
  "gate_llama1b_rome_zsre_L10_s1|21|$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome --dataset zsre --data data/zsre_eval.json $COMMON --lr 0.1 --layer 10 --seed 1 --out results/gate_llama1b_rome_zsre_L10_s1.json"
  "gate_llama1b_rome_zsre_L10_s2|21|$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome --dataset zsre --data data/zsre_eval.json $COMMON --lr 0.1 --layer 10 --seed 2 --out results/gate_llama1b_rome_zsre_L10_s2.json"
  # ==== E6: HONEST C4 causal test — AlphaEdit projector fit on a DISJOINT bank, NOT
  #      the measured probes (retires the "98% removal is by construction" reviewer kill).
  #      L8 (cosine regime) + L12 (peak) x {holdout facts, generic activations}, s0.
  #      Reuses the existing gate_llama1b_rome_cf_L{8,12}_s0 ROME matrices as the paired
  #      baseline; C4-holdout aggregation is added to the analysis pass below. ----
  "g4_llama1b_alphaHO_cf_L8_s0|24|$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor alpha $CF $COMMON --lr 0.1 --layer 8 --seed 0 --alpha_proj_source holdout --holdout_frac 1.0 --out results/g4_llama1b_alphaHO_cf_L8_s0.json"
  "g4_llama1b_alphaHO_cf_L12_s0|24|$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor alpha $CF $COMMON --lr 0.1 --layer 12 --seed 0 --alpha_proj_source holdout --holdout_frac 1.0 --out results/g4_llama1b_alphaHO_cf_L12_s0.json"
  "g4_llama1b_alphaGEN_cf_L12_s0|24|$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor alpha $CF $COMMON --lr 0.1 --layer 12 --seed 0 --alpha_proj_source generic --holdout_frac 1.0 --out results/g4_llama1b_alphaGEN_cf_L12_s0.json"
)

for entry in "${JOBS[@]}"; do
  IFS='|' read -r tag est cmd <<< "$entry"

  # (4) HARD STOP: if it is already >= 19:00, stop the GPU queue entirely.
  now_hm=$(date +%H%M)
  if [ "$((10#$now_hm))" -ge 1900 ]; then
    log "HARD-STOP 19:00 reached (now ${now_hm}) — stopping GPU queue before ${tag}"
    echo "DEEP_QUEUE_DONE" >> "$LOG"
    : > engine/deep_queue_DONE
    break
  fi

  # (3a) IDEMPOTENT SKIP: both the json AND the npz already exist.
  if [ -f "results/${tag}.json" ] && [ -f "results/matrices/${tag}.npz" ]; then
    log "skip ${tag} (exists)"
    continue
  fi

  # (3a') MODEL PRECHECK: a missing local model is a CONFIG skip, NOT a GPU failure.
  #   Without this, a missing model dir makes the job exit fast rc!=0; two such fast
  #   failures would trip the consecutive-failure "GPU wedge" abort and needlessly
  #   kill the whole queue. Extract --model <dir> from the command and require it.
  mdir=$(echo "$cmd" | grep -oE -- '--model [^ ]+' | awk '{print $2}')
  if [ -n "$mdir" ] && [ ! -d "$mdir" ]; then
    log "SKIP-NOMODEL ${tag} (missing model dir ${mdir}) — config skip, not counted as failure"
    continue
  fi

  # (3b) 19:00 fine gate: don't START a job that can't finish (est*60 + margin) by 19:00.
  now_epoch=$(date +%s)
  finish_epoch=$((now_epoch + est * 60 + MARGIN))
  if [ "$finish_epoch" -gt "$STOP_EPOCH" ]; then
    log "SKIP-TIME ${tag} (est ${est}m + margin cannot finish by 19:00)"
    continue
  fi

  # (3c) RUN — serial, per-job log, TIMEOUT-guarded so a wedged/hung job (P5/0% forever)
  #      cannot freeze the whole queue. Cap = 2x est + 15min. Never run two jobs concurrently.
  cap=$(( est * 60 * 2 + 900 ))
  log "RUN ${tag} (est ${est}m, timeout ${cap}s) -> engine/deep_${tag}.log"
  t_start=$(date +%s)
  timeout --signal=TERM --kill-after=60 "${cap}s" bash -c "$cmd" >> "engine/deep_${tag}.log" 2>&1
  rc=$?
  t_elapsed=$(( $(date +%s) - t_start ))
  if [ "$rc" -eq 0 ]; then
    log "done ${tag} (${t_elapsed}s)"
    consec_fail=0
  else
    # Classify the failure. A GPU WEDGE (lid-close nvidia_uvm) hangs a job at P5/0%
    # until the timeout fires (rc=124) or it burns most of its budget — those are the
    # only failures that should trip the abort. A FAST rc!=0 (crash in seconds: bad
    # arg, import error, missing data file) is a CONFIG failure — log and continue
    # WITHOUT counting it, so one broken row never aborts the whole night's queue.
    wedge_like=0
    if [ "$rc" -eq 124 ] || [ "$t_elapsed" -ge $(( est * 60 / 2 )) ]; then
      wedge_like=1
    fi
    if [ "$wedge_like" -eq 1 ]; then
      consec_fail=$((consec_fail + 1))
      log "FAIL ${tag} (exit ${rc}, ${t_elapsed}s; 124=timeout) WEDGE-LIKE consec=${consec_fail}/${MAXFAIL} — continuing"
      if [ "$consec_fail" -ge "$MAXFAIL" ]; then
        log "ABORT: ${consec_fail} consecutive wedge-like failures — likely GPU wedge (nvidia_uvm). Stopping GPU queue, proceeding to analysis pass."
        : > engine/deep_queue_DONE
        break
      fi
    else
      log "FAIL ${tag} (exit ${rc}, ${t_elapsed}s) FAST/CONFIG failure — not counted toward wedge abort, continuing"
    fi
  fi
done

if [ ! -f engine/deep_queue_DONE ]; then
  log "GPU queue exhausted (no hard-stop trigger)"
  : > engine/deep_queue_DONE
fi

# ---------------------------------------------------------------------------
# (5) ANALYSIS PASS — pure CPU, ALWAYS runs (safe even after 19:00, even if the
#     GPU queue was truncated). Each step guards on the npz/script it needs.
# ---------------------------------------------------------------------------
log "---------------- ANALYSIS PASS (CPU only) ----------------"

# C3 per-config within-probe nulls (arch-conditioned dissociation + FT/KL-FT structural null).
# rows: 'outname|gate-glob'
C3_CFGS=(
  "C3_null_qwen05b_L12|results/matrices/gate_qwen05b_rome_cf_L12_s*.npz"
  "C3_null_qwen15b_L14|results/matrices/gate_qwen15b_rome_cf_L14_s*.npz"
  "C3_null_gemma2b_L13|results/matrices/gate_gemma2b_rome_cf_L13_s*.npz"
  "C3_null_phi35_L16|results/matrices/gate_phi35_rome_cf_L16_s*.npz"
  "C3_null_qwen3b_L18|results/matrices/gate_qwen3b_rome_cf_L18_s*.npz"
  "C3_null_ft_L8|results/matrices/gate_llama1b_ft_cf_L8_s*.npz"
  "C3_null_ft_L10|results/matrices/gate_llama1b_ft_cf_L10_s*.npz"
  "C3_null_ft_L12|results/matrices/gate_llama1b_ft_cf_L12_s*.npz"
  "C3_null_ftkl_L8|results/matrices/gate_llama1b_ftkl_cf_L8_s*.npz"
  "C3_null_llama3b_L14|results/matrices/gate_llama3b_rome_cf_L14_s*.npz"
  "C3_null_llama1b_zsre_L10|results/matrices/gate_llama1b_rome_zsre_L10_s*.npz"
)
if [ -f experiments/analyze_matrices.py ]; then
  for c in "${C3_CFGS[@]}"; do
    IFS='|' read -r outn cglob <<< "$c"
    if have "$cglob"; then
      log "analyze_matrices -> results/${outn}.json"
      $PY experiments/analyze_matrices.py $cglob --metric logit --known --edit_ok \
        --out "results/${outn}.json" >> "$LOG" 2>&1 || log "FAIL analyze ${outn}"
    else
      log "skip ${outn} (no npz: ${cglob})"
    fi
  done
else
  log "skip C3 nulls (experiments/analyze_matrices.py missing)"
fi

# C4 causal quartile table (absolute damage-removed by key-cos quartile, all layers).
if [ -f experiments/aggregate_g4_causal.py ] && have "results/matrices/g4_llama1b_alpha_cf_L*_s*.npz"; then
  log "aggregate_g4_causal -> results/C4_causal_table.json"
  $PY experiments/aggregate_g4_causal.py \
    --rome_glob 'results/matrices/gate_llama1b_rome_cf_L{L}_s*.npz' \
    --alpha_glob 'results/matrices/g4_llama1b_alpha_cf_L{L}_s*.npz' \
    --layers 8 10 12 14 --known --edit_ok \
    --out results/C4_causal_table.json >> "$LOG" 2>&1 || log "FAIL aggregate_g4_causal"
else
  log "skip C4 (aggregate_g4_causal.py or alpha npz missing)"
fi

# C4-HONEST: same causal aggregation but projector fit on the DISJOINT holdout bank
# (E6). This is the reviewer-proof version — geometry-tracked damage removal on probes
# the projector never saw. Guarded: only runs if E6 produced a holdout alpha matrix.
if [ -f experiments/aggregate_g4_causal.py ] && have "results/matrices/g4_llama1b_alphaHO_cf_L*_s*.npz"; then
  log "aggregate_g4_causal (holdout projector) -> results/C4_causal_holdout_table.json"
  $PY experiments/aggregate_g4_causal.py \
    --rome_glob 'results/matrices/gate_llama1b_rome_cf_L{L}_s0.npz' \
    --alpha_glob 'results/matrices/g4_llama1b_alphaHO_cf_L{L}_s0.npz' \
    --layers 8 12 --known --edit_ok --proj_source holdout \
    --out results/C4_causal_holdout_table.json >> "$LOG" 2>&1 || log "FAIL aggregate_g4_causal holdout"
else
  log "skip C4-holdout (no holdout alpha npz — E6 did not run)"
fi

# C1 SxC mechanism table (S=mean||v-Wk||, C=mean|cos|, S*|C| within-probe rho).
if [ -f experiments/mechanism_sc_table.py ] && have "results/matrices/gate_llama1b_rome_cf_L*_s*.npz"; then
  log "mechanism_sc_table -> results/C1_mechanism_sc_table.json"
  $PY experiments/mechanism_sc_table.py \
    --npz 'results/matrices/gate_llama1b_rome_cf_L*_s*.npz' \
          'results/matrices/gate_qwen05b_rome_cf_L12_s*.npz' \
          'results/matrices/gate_qwen15b_rome_cf_L14_s*.npz' \
    --known --edit_ok \
    --out results/C1_mechanism_sc_table.json >> "$LOG" 2>&1 || log "FAIL mechanism_sc_table"
else
  log "skip C1 (mechanism_sc_table.py or Llama rome npz missing)"
fi

# D3 geometry-gated routing eval (deployable routing angle for TNNLS/KBS).
if [ -f experiments/geometry_router.py ] && have "results/matrices/gate_*_rome_cf_*_s0.npz"; then
  log "geometry_router -> results/D3_routing_eval.json"
  $PY experiments/geometry_router.py \
    --gate_glob 'results/matrices/gate_*_rome_cf_*_s0.npz' \
    --alpha_glob 'results/matrices/g4_llama1b_alpha_cf_L*_s0.npz' \
    --cos_threshold 0.05 --known \
    --out results/D3_routing_eval.json >> "$LOG" 2>&1 || log "FAIL geometry_router"
else
  log "skip D3 (geometry_router.py or gate npz missing)"
fi

log "================ DEEP_UNTIL1900 COMPLETE ================"
echo "DEEP_UNTIL1900_DONE" >> "$LOG"
