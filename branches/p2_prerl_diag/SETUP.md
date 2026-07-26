# P2 env setup — build & patch `dl-rl` (instructions only; NOT executed)

The shared `dl` env must stay untouched: other projects use it, and `trl` /
`unsloth` are currently broken in it (see `trl_mergekit_fix.md`). The RL work
for P2 runs in an isolated **clone**, `dl-rl`. Nothing below has been run by the
scaffolding step — copy/paste when you actually start the GPU phase.

The CPU diagnostic (`diagnostic.py` / `run_diag.py`) needs **only numpy** and
runs in `dl` as-is; `dl-rl` is required solely for the queued `gen` + GRPO jobs.

---

## 1. Clone dl → dl-rl (does not mutate `dl`)

```bash
# ~10 min, ~big disk; a full conda clone, isolated from dl.
conda create --clone dl -n dl-rl
```

## 2. Apply the trl mergekit patch (opt-in, dl-rl ONLY)

Pick ONE of the two fixes from `trl_mergekit_fix.md`. The one-liner (Fix B) is
easiest to script. This edits `dl-rl`'s site-packages, never `dl`'s.

```bash
# locate trl inside the CLONE (note: -n dl-rl)
TRL=$(conda run -n dl-rl python3 -c "import trl,os;print(os.path.dirname(trl.__file__))")
echo "patching $TRL/import_utils.py"

# safety: refuse to run if this resolved to the shared dl env
case "$TRL" in
  *"/envs/dl-rl/"*) : ;;
  *) echo "REFUSING: $TRL is not in dl-rl"; exit 1;;
esac

# back up, then apply Fix B (unpack the (bool, version) tuple -> real bool)
cp "$TRL/import_utils.py" "$TRL/import_utils.py.bak"
python3 - "$TRL/import_utils.py" <<'PY'
import sys, io
p = sys.argv[1]
src = io.open(p).read()
old = '_mergekit_available = _is_package_available("mergekit")'
new = '_mergekit_available = _is_package_available("mergekit", return_version=True)[0]'
assert old in src, "expected line not found — trl version drift; re-check patch"
io.open(p, "w").write(src.replace(old, new))
print("patched:", p)
PY
```

(Do **not** `pip install mergekit` — it downgrades accelerate/hub/safetensors/
pydantic and would break the parity between `dl` and `dl-rl`.)

## 3. Verify the trainers import (dl-rl)

```bash
conda run -n dl-rl python3 - <<'PY'
from trl import GRPOTrainer, DPOTrainer
print("OK", GRPOTrainer.__name__, DPOTrainer.__name__)
PY
# expected:  OK GRPOTrainer DPOTrainer
```

Also sanity-check the P2 config builder wires up (still no training):

```bash
conda run -n dl-rl python3 -c "import grpo_config as g; print('panel', len(g.CHECKPOINT_PANEL)); print(g.GRPOScaffold().max_steps, 'steps matched-budget')"
```

## 4. Run the queued jobs (GPU) — only after steps 1–3 pass

```bash
source ~/Desktop/idea-feasibility-analysis/env.sh   # unset ALL_PROXY etc.
python3 branches/p2_prerl_diag/make_jobs.py         # writes queue/*.json (specs)
# then a serial runner (mirror edit-harness/queue/run_all.sh) executes:
#   gen  jobs  -> conda run -n dl-rl ... sample_ckpt.py   (GPU)
#   diag jobs  -> conda run -n dl    ... run_diag.py       (CPU, numpy-only)
```

`sample_ckpt.py` (the GPU sampler) is the queued work item, not part of this
scaffold; its output JSON must match `run_diag.py`'s input contract
(`{"checkpoint", "problems":[{"problem","samples":[{"text","len","correct"}]}]}`).

## 5. Teardown (optional)

```bash
conda env remove -n dl-rl        # dl is unaffected
```

## Rollback the patch (if ever applied)

```bash
TRL=$(conda run -n dl-rl python3 -c "import trl,os;print(os.path.dirname(trl.__file__))")
mv "$TRL/import_utils.py.bak" "$TRL/import_utils.py"
```

---

## ADDENDUM 2026-07-10 — §4 superseded: the fission-engine IS the P2 runner

Two corrections to the sections above, then the actual launch procedure:

1. **dl-rl is NOT needed for the queued gen/diag jobs.** The 2026-07-04 u4 review
   (CRITICAL-1) re-stamped every queued cmd to `conda run -n dl` (see each job's
   `_env_note`); `dl` carries torch/transformers/datasets and GSM8K is already in
   the local HF cache (`openai___gsm8k`) — zero network. The `dl-rl` clone exists
   (built 2026-07-10, UNPATCHED) but is required only for the GRPO **training**
   step, which stays gated on the pre-RL diagnostic clearing its kill-gate.
2. **No bespoke serial runner needs writing.** `fission_engine.collect_branch_jobs`
   gained a P2 mapping (2026-07-10): `job_id→id`, `spec.out`→gen expected output,
   diag expected output derived as `results/<checkpoint_id>.json`, cwd anchored at
   the WORKSPACE ROOT (P2 cmds are root-relative). The engine runner (GPU-idle
   gating, file lock, idempotent skip, done/failed buckets) is the orchestrator.

### Launch (GPU-filler — ONLY when the Lane A/B chains are NOT holding the card)

```bash
cd ~/Desktop/idea-feasibility-analysis
# 1. bridge the P2 specs into the central engine queue.
#    ALWAYS pass --source explicitly: the collector's DEFAULT sources include
#    branches/p3_agent_ipi/jobs/queue.json, which run_p3_gpu.sh owns — collecting
#    defaults would create a DOUBLE-EXECUTION path for the Lane B jobs.
python3 -m fission_engine.collect_branch_jobs --source "branches/p2_prerl_diag/queue/*.json"

# 2. drain once, serially, GPU-idle-gated (wait-by-PID rules inside the engine):
python3 -m fission_engine.runner --once
```

Idempotency: already-satisfied pairs (e.g. Qwen2.5-0.5B from the 07-04 smoke) are
skipped at collect time; a rerun of either command is safe. Runtime is unbounded
per job (no timeout_s stamped) — expect the 6 remaining gen jobs to be the long
pole (200 problems × k=8 × ≤640 new tokens each; the 3B-class rows are bf16).
