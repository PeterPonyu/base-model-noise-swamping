# P3 wave-2 spec: outgroup x s0 (capability probe) — DRAFT, NOT LAUNCHED — 2026-07-10

Companion file: `wave2_outgroup_DRAFT-NOT-QUEUED.json` (branch root — moved OUT of `jobs/` per review, so no queue reader can ever glob it) (preview of the queue entry;
filename carries NOT-QUEUED deliberately -- it is not read by `run_p3_gpu.sh`, which
hardcodes `jobs/queue.json`). Neither file has been launched. This spec does not modify
`jobs/queue.json` or `PREREG-B2B4-FROZEN-20260710.md` (both frozen/live).

## 1. What wave 2 is, per the frozen prereg

`PREREG-B2B4-FROZEN-20260710.md` sec 1: *"`outgroup` x seed 0 is **wave 2** (deferred, not
queued): it doubles as a capability probe (sec 5 item 4 -- newly added families may lack
chat/tool templates) and should be interpreted with that in mind. It runs only after wave-1
results + audits are inspected."* Wave-1 results + audits ARE now inspected and consolidated
(`results/B4_CONSOLIDATED.json`, `findings-B4-LINEAGE-CONSOLIDATED-20260710.md`) -- the
prereg's stated precondition for wave 2 is satisfied. This spec does not itself authorize
launch; it prepares the launch artifact for the user/orchestrator to trigger.

## 2. Model panel — PULLED FROM EXISTING REVIEWED CODE, not invented

The `outgroup` tier is not an ad hoc list -- it is already defined and frozen in the
review-clean `grid.py` (`TIERS["outgroup"] = _CORE + _OUTGROUP_EXTRA`, verified by direct
import: `grid.tier_names("outgroup")` returns exactly 24 names). No model list needed to be
proposed or guessed.

**24 models = the 11 `core` models (wave 1, already run) + 13 new out-group families**:

```
llama3.2:3b, gemma3:12b-it-q8_0, mistral-nemo:12b-instruct-2407-q8_0, phi4:14b-q8_0,
glm4:9b, yi:9b, granite3.3:8b, exaone3.5:7.8b, falcon3:7b, solar:10.7b,
aya-expanse:8b, internlm2:7b, qwen3.5:9b-q8_0
```

Source: `grid.py` `_OUTGROUP_EXTRA` (lines defining `EXTENDED_DESIGN`/`TIERS`), itself
sourced from `SCOPE-B2-B4-20260710.md` sec 3's local-zoo enumeration (`~/.ollama/models/manifests`,
filesystem-only, no daemon, no inference -- done 2026-07-10 prior to this task). This list
is CONFIRMED, not PROPOSED: it is the exact set `grid.py` will resolve at `--tier outgroup`.
Per `grid.py`'s own docstring, `starcoder2` was deliberately excluded (code model, poor
chat/tool fit) -- not an oversight.

**Capability-probe framing** (per prereg sec 5 item 4 / `SCOPE-B2-B4-20260710.md` open
decision 4): r1-distills + gemma2 are known to not advertise `tools` capability (already
true in wave 1, handled via the prompt-format arm). Several of these 13 new families are
untested on this box for chat/tool-template support -- `supports_tools` is resolved LIVE via
`ollama show` at run time (not the static `supports_tools_hint` in `grid.py`, which is a
mock-backend fallback only). Any family whose Ollama build lacks a working template may:
(a) land in the prompt-format arm (scored via `transport.py`'s fallback path, same as
gemma2 in wave 1), or (b) error out and get nulled by the `error_rate > 0.2` threshold (same
handling as `deepseek-r1:8b` in wave-1 s0). **Both outcomes are informative** -- this run is
explicitly also a capability inventory, not just an ASR sweep, and should be reported as
such regardless of what the lineage-generality question shows.

## 3. Scientific purpose

The wave-1 B4 lineage-vs-architecture contrast (`findings-B4-LINEAGE-CONSOLIDATED-20260710.md`
sec 2) is computed only over the in-group Qwen-lineage pairs (r1-distill vs matched
base-instruct); out-group models are already scored for ASR in wave 1 but their pairwise
similarity is reported separately (`mean_outgroup_corr` in `analyze.contrast()`) and NOT
part of the headline contrast. Wave 2 does not change that contrast's definition -- `core`'s
3 out-group members (llama3.1, gemma2, mistral) stay the out-group baseline. What wave 2
adds is breadth for two purposes: (1) does the ASR pattern (which families are
attackable/resistant) generalize beyond the 3 out-group families already sampled, and (2)
the capability-probe inventory above. It is explicitly NOT expected to add new
architecture-matched lineage pairs (no r1-distill counterpart exists for these 13 families
on this box, per `grid.py`'s design note).

## 4. Runtime estimate

Measured from `logs/run_p3_gpu.log` (2026-07-10 14:20-15:22 run), wall-clock per **full
11-model `core`-panel sweep, 30 scenarios**:

| job | duration | arms | per-arm-sweep |
|---|---|---|---|
| `ipi_grid_core_n30_s0` | 534s | 1 | 534s |
| `ipi_grid_core_n30_s1` | 556s | 1 | 556s |
| `ipi_grid_core_n30_s2` | 539s | 1 | 539s |
| `defense_spotlight_core_s0` | 1068s | 2 (off+on) | 534s |
| `defense_whitelist_core_s0` | 1047s | 2 (off+on) | 523.5s |

**Measured mean: ~539s (~9.0 min) per full 11-model-panel sweep of 30 scenarios**,
consistent across all 5 wave-1 jobs regardless of arm -- this is the "~9 min/sweep"
throughput the team-lead brief referenced, confirmed against the log timestamps (not
estimated).

**Linear extrapolation to 24 models** (assuming per-model marginal cost is roughly constant,
which wave-1's consistency across jobs supports): 539s x (24/11) approx **1176s (~19.6
min)** for the single `outgroup` x seed-0 sweep.

**Caveat, stated explicitly**: `SCOPE-B2-B4-20260710.md` sec 3's a-priori estimate for the
`outgroup` tier (24 models) was **75-120 min**, ~4-6x higher than the linear extrapolation
above, because it anticipated heavier model-load/swap overhead for out-group families and
explicitly says these numbers are "ROUGH -- monitor, do NOT extrapolate." The wave-1 core
panel already contains a similar size mix (1.5B-14B) to the 13 new out-group additions
(avg ~9-10B, comparable to core's ~8.6B average), which is why the linear extrapolation is
plausible -- but wave-1 never exercised a COLD load of any of these 13 new checkpoints, and
first-time loads of previously-unpulled-into-memory large quantized models (gemma3:12b-q8,
mistral-nemo:12b-q8, phi4:14b-q8) could run slower than the already-warm-from-repeated-use
core models. **Recommendation: budget conservatively (60-90 min) and monitor `logs/run_p3_gpu.log`
progress lines rather than trusting either estimate blindly** -- the job-cap
(`JOB_CAP_MIN`, currently defaulting 100 min) already protects against a runaway single job
regardless of which estimate is closer.

## 5. Launch procedure — NOT to be run while the B6 GPU queue holds the card

**Binding precondition**: this workspace's `edit-harness/` GPU work and this P3 work share
the single local 5090. Per `CLAUDE.md`'s 2026-07-10 addenda, Lane A (B6 revision pre-runs)
and Lane B (this P3 work) are sequenced by a supervisor
(`edit-harness/engine/chain_lanes_20260710.sh`) specifically so they never overlap on the
GPU. **Do not launch wave 2 while any `edit-harness` job or the Lane A/B chain holds the
GPU** -- check `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader`
and the chain's own PID files before launching; wait by PID (`kill -0`), never
`pgrep`/`pkill -f` a pattern (per this workspace's standing fission-engine rule).

**Launch** (once the GPU is confirmed free and idle, and the user has confirmed the model
panel above and the runtime budget):

```
cd branches/p3_agent_ipi
python make_jobs.py --kind grid --tier outgroup --seeds 0 --n 30   # appends to jobs/queue.json
nohup ./run_p3_gpu.sh >> logs/run_p3_gpu.nohup.log 2>&1 &
```

This uses the EXISTING `run_p3_gpu.sh` unmodified with a new queue entry (there is no
`QUEUE_FILE` env override in the current script -- it hardcodes `jobs/queue.json` --
verified by reading the script; `make_jobs.py --kind grid` is idempotent and appends only
new run_ids). `wave2_outgroup_DRAFT-NOT-QUEUED.json` (branch root — moved OUT of `jobs/` per review, so no queue reader can ever glob it) above is a preview of exactly what
that `make_jobs.py` invocation will append; it is informational only and is not consumed by
any runner. Post-run, `run_p3_gpu.sh` will automatically run `audit_unmatched.py` over the
new result (per the standing binding condition, prereg sec 5) and fold it into
`results/P3_GPU_report.json`.

## 6. Open items for the user before launch

1. Confirm the 24-model panel above (pulled verbatim from `grid.py`, not edited here).
2. Confirm the runtime budget (recommend 90 min `JOB_CAP_MIN` override for this one job,
   given the load-time uncertainty in sec 4: `JOB_CAP_MIN=90 BUDGET_MIN=120 ./run_p3_gpu.sh`).
3. Confirm the GPU is free (Lane A/B chain not holding it) at launch time.
4. Decide whether wave-2's out-group ASR pattern should be folded into a second
   `consolidate_b4.py`-style pass once results land, or left as a standalone capability-probe
   writeup -- not decided here.
