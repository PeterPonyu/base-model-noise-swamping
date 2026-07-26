# edit-harness — knowledge-editing research repo

Single-GPU (RTX 5090 24GB) knowledge-editing experiment bench. **Key constraint: all subprocesses use the direct env-python**
`/home/zeyufu/miniconda3/envs/dl/bin/python3` (do NOT use `conda run` — nested activation
triggers the broken gcc activate.d script in the dl env and fails); downloads go through `curl -L`
(hf_transfer/httpx proxying stalls); editing uses **fp32** (ROME value-optimization NaNs under fp16).

## Directory
```
editors/           Three editors, unified interface apply_edit(model, tok, edit_request, config, device) -> dict
  rome_native.py     ROME rank-one:  ΔW=(v−Wk)kᵀ/‖k‖²; returns residual_norm(=S factor)
  ft_editor.py       FT-L constrained fine-tuning; supports lambda_kl+neighborhood_prompts (D1 control)
  alphaedit.py       AlphaEdit null-space projection:ΔW=(v−Wk)(Pk)ᵀ/(kᵀPk)(G4/D3)
experiments/
  killgate_keygeom.py  Core experiment: capture MLP key → edit → measure per-probe damage; --editor{rome,ft,alpha}
                       --save_matrices saves raw matrices(.npz), --ft_kl/--keep_ratio
  analyze_matrices.py  G1 GATE: within-probe partialled Spearman + column-permutation null
  analyze_g4.py        G4 causal test: ROME vs AlphaEdit damage by cosine quartile
  collate.py           Aggregate all results/*.json into a wide table
  metrics.py           efficacy/generalization/locality/fluency primitives
engine.py            Fission engine: self-waiting layer-sweep → run round by round → wrap up with collate
run_gate.sh          Self-wait engine → run GATE(L8/10/12×seeds)→ analyze_matrices
run_g4.sh            Self-wait gate → run matched AlphaEdit → analyze_g4
download_models.py   Shard-aware curl download, writes a COMPLETE marker
engine_watchdog.sh   cron watchdog (restarts a dead engine; requires the user to install the crontab)
```

## Autonomous pipeline (fully self-advancing end-to-end, no wasted GPU time)
```
engine.py ──→ run_gate.sh ──→ run_g4.sh
(wide sweep)     (GATE verdict)      (causal test)
```
Each stage launches detached, self-waits on the previous stage, and is idempotent (skips if the output JSON already exists).

## Engine robustness (the fission engine's hardening points)
- **Timeout = GPU-wedge signal → stop** (rerun after `sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm`; the engine skips completed work)
- **Fast-fail = bad model → after 2 failures that model is marked bad, skipped, and the engine continues with other models** (one bad architecture no longer drags down the whole round)
- Each job retries 3 times (to ride out GPU handoff transients); per-job log at `engine/jobs/<tag>.log`
- Waits per job for the model's COMPLETE marker (downloads happen in parallel); state.json shows live progress

## Common operations
```
# check progress
cat engine/state.json ; tail engine/engine.log
# wide table
/home/zeyufu/miniconda3/envs/dl/bin/python3 experiments/collate.py
# manually run one config (note: direct python)
env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1 /home/zeyufu/miniconda3/envs/dl/bin/python3 \
  experiments/killgate_keygeom.py --model data/models/Llama-3.2-1B --editor rome \
  --dataset counterfact --data data/counterfact.json --layer 8 --save_matrices --out results/x.json
# resume the engine (after fixing a wedge)
nohup env -u ALL_PROXY -u all_proxy /home/zeyufu/miniconda3/envs/dl/bin/python3 engine.py &
```

## Result artifacts
- `results/sweep_*.json` wide-sweep configs · `results/matrices/*.npz` raw matrices
- `results/GATE_L*.json` GATE verdicts · `results/G4_L*.json` causal verdicts
- `engine/breadth_table.txt` final wide table · `engine/state.json` progress
