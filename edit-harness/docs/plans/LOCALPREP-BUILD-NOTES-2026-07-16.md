# Local-first prep package build notes   2026-07-16

Author pass only (per team-lead's task: "hostile review follows; launch NOTHING on GPU").
Goal: every pending experiment can run on the single local 5090 the instant its user gate opens,
no cloud box required. **Nothing was launched on GPU by this build.** The live local Frame-A
MIX_A wave (`experiments/frame_a/run_stream.py` + `arms/*`, pid recorded in
`engine/frame_a_mixa_local.pid`) was left running untouched throughout.

## 1. R-E dual-reference flag — DONE

**Files changed**: `experiments/prospective_admission.py` (module docstring, new `_solo_delta_w`
helper, `_base_ns_for`'s dispatch logic, CLI `--ns_reference`, the `main()` guard, `selftest()`'s
new part (d), the report dict); `docs/plans/PREREG-PROSPECTIVE-ADMISSION-DRAFT-2026-07-16.md`
(new "Decision point: neighborhood-damage reference" subsection + item (b) wording + the Launch
command). **No new driver** — R-E still has no launch script, unchanged from the prior build
(that gap is orthogonal to this task).

**What the flag resolves**: the prereg's original prose for outcome (b) read as "solo-edit
baseline" for the pre-merge neighborhood-NS reference, but the shipped code actually measured it
at the **unedited base model** — an artifact of `merging_m0._compute_solo` restoring every edit
after its own solo pass, so the weights at that point are indistinguishable from base. `--ns_reference
{solo,base}` makes both readings real, flag-selected options:
- `base` (option ii): the prior de facto behavior, unchanged in effect — measure at the unedited
  base model. Includes solo-edit collateral in what counts as "already fine".
- `solo` (option i): install edit `a`'s own solo ΔW alone, measure, restore — true
  federation-added damage. Implemented via a NEW helper `_solo_delta_w(K, R, denom, a, device)`
  that calls the singleton-group case of the **already-imported, unmodified**
  `merging_m0._merge_factors` (`merged = Rt[a].outer(Ktsc[a])`) — bit-identical to the editor's
  own per-edit ΔW (`merging_m0._compute_solo`'s own `recon_rel_err` assertion already guarantees
  this <1e-3 at capture time), so this is genuinely "no new editor/ROME math", matching the
  prereg's reuse contract.

No default: the GPU path now refuses with a distinct, informative `SystemExit` if
`--ns_reference` is omitted ("ratifying means passing one explicitly, not waiting on new code"),
checked BEFORE and WITHOUT weakening the pre-existing prereg-ratification `SystemExit` (which
still fires unconditionally on any non-`--selftest` invocation, exactly as before).

**Validated**:
- `python3 -m py_compile experiments/prospective_admission.py` clean.
- `--selftest` ALL CHECKS PASSED, now including new part (d): `_solo_delta_w`'s singleton-group
  reconstruction checked against a manual `outer(r_a, k_a/denom_a)` recomputation (synthetic
  K/R/denom, CPU); a new `_simulate_ns_reference_dispatch` helper exercises both dispatch
  branches with a fake "measurement" that snapshots the current weight tensor — asserts `base`
  measures the UNTOUCHED base W and never perturbs it, and `solo` measures a PERTURBED
  (edit-a-installed) W then restores it exactly. All existing parts (a)-(c) still pass unchanged.
- Manually verified all four CLI behaviors:
  - no flags → the new `--ns_reference` guard fires (rc=1, clear message).
  - `--ns_reference solo` or `--ns_reference base` → falls through to the still-unconditional
    master ratification guard (rc=1, unchanged message) — confirms the master gate was NOT
    weakened.
  - `--ns_reference bogus` → argparse itself rejects it (rc=2, `choose from solo, base`).
- `merging_m0.py`, `experiments/frame_a/run_stream.py`, and `experiments/frame_a/arms/*.py` were
  NOT modified (confirmed: no Edit/Write calls touched them; mtimes predate this session).

## 2. Paper B Phase-1 local driver — BLOCKED, NOT BUILT

The task pointed at `docs/plans/PREREG-PAPERB-QUANTSURVIVAL-DRAFT-2026-07-16.md` as the ratified-
pending spec to build against (Llama-3.2-1B L12 + Llama-3.2-3B L24, C2 cells, ROME fp32
round-trip INT8+NF4, n 50→200, seeds 0/1/2, + MEMIT/AlphaEdit C1/C3 arms, + Track 1.5 frozen-
calibration GPTQ/AWQ). **That file does not exist** in `docs/plans/`. The only related document
on disk is `docs/plans/PREREG-QUANT-SMOKE-2026-07-16.md`, a materially smaller SMOKE-scale spec
(`experiments/quant_survival_smoke.py` + `run_quant_smoke.sh`): Llama-3.2-1B only, L12 only, ROME
only, n=50/seed 0 only, INT8+NF4 only — no 3B/L24 cell, no MEMIT/AlphaEdit arms, no n=200 or
seeds 1/2, no Track 1.5.

Flagged this to team-lead (SendMessage, before starting file edits) since `paperb-designer` /
`paperb-design-reviewer` are active teammates and the Phase-1 prereg is presumably being
authored/reviewed concurrently. Did not invent the Phase-1 spec myself rather than risk building
a driver against un-ratified, self-authored numbers (n_edits/seeds/arm list) that a hostile
reviewer would then have to untangle from the actual ratified design.

**Consequently**: `run_paperb_phase1.sh` was NOT built. `engine/chain_local_20260716.sh`'s step
(f) is written defensively — it checks for `engine/PAPERB_GO.ok` AND that `run_paperb_phase1.sh`
exists on disk at the moment the chain actually runs; if the gate is open but the script is
still missing, it logs a clear WARN and skips rather than erroring the whole chain (see below).

**If/when the ratified prereg lands**, building the driver should reuse
`experiments/quant_survival_smoke.py`'s already-audited codecs (`int8_roundtrip`,
`nf4_roundtrip`, blockwise absmax, the INT8-half-step/NF4-grid-membership `--selftest` bounds)
and `run_quant_smoke.sh`'s skeleton (GPU-idle gate `util<25 && mem<1500 x3` pinned via
`nvidia-smi -i`, CPU self-test smoke gate, budget/DRYRUN, refuse-on-valid-table, pid-by-file) —
both already exist and are reviewed-clean at SMOKE scope; Phase-1 is a scale-up (n, seeds, arms,
second model/layer) of the same machinery, not a rewrite.

## 3. Master local chain — DONE (steps a-e); step (f) gated-but-inert

**File**: `engine/chain_local_20260716.sh` (new). Sequences, strictly serially on the one local
5090:
- **(a)** waits for the live MIX_A wave: polls `results/frame_a/cells/cell_*_MIX_A_*.json` count
  against 33 (= `len(config.SEEDS)=3` x `len(POLICIES)=11`, confirmed by reading
  `experiments/frame_a/config.py`/`run_stream.py` directly — NOT a guessed number) OR the pid in
  `engine/frame_a_mixa_local.pid` going dead, whichever comes first (matches the task's stated
  OR-condition). At build time: 9/33 cells present, pid 683987 alive.
- **(b)** prints (does **not** apply) the `engine/PATCH-smoke-marker-ordering-20260716.md`
  reminder — that patch touches `experiments/frame_a/run_stream.py`, which the just-drained live
  wave imports; per the task's explicit instruction this stays a manual step.
- **(c)** runs `./run_esr_probe_gpt2xl.sh` (self-contained: has its own idle gate + timeout;
  skipped if `results/esr_probe_gpt2xl/esr_by_layer.json` already exists).
- **(d)** if `engine/FRAMEA_LOCAL_BC.ok` exists: runs MIX_B then MIX_C via the same
  `python3 -m experiments.frame_a.run_stream --run --real --mixes <MIX> --model_dir
  data/models/Llama-3.2-1B` invocation pattern as the live MIX_A process (verified via
  `ps`/`/proc/<pid>/cmdline` against the running job) — else SKIP, logged.
- **(e)** if `engine/RE_GO.ok` exists and its (whitespace-stripped) content is exactly `solo` or
  `base`: runs an inline GPU-idle gate (mirrors `run_esr_probe_gpt2xl.sh`'s pattern — R-E has no
  driver of its own yet) then launches `experiments/prospective_admission.py` with that
  `--ns_reference` value and the exact flags from the prereg's Launch section — else SKIP,
  logged (including the case where the file exists but holds neither value).
- **(f)** if `engine/PAPERB_GO.ok` exists AND `./run_paperb_phase1.sh` exists: runs it — else
  SKIP with a distinguishing message for "gate open but driver missing" vs. "gate not open".

Every real-GPU step (`esr_probe_gpt2xl`, `frame_a_mixb`, `frame_a_mixc`, `prospective_admission`,
`paperb_phase1`) runs through a shared `run_step` helper: `setsid`-launched, its own
`engine/chain_local_20260716_<name>.pid`/`.log`, the chain blocks on `wait "$pid"` before
advancing — so a human can `kill -0`/kill one stuck step without touching the chain, and the
chain is provably serial (one step's process fully exits, rc captured, before the next starts).
`abort_on_fail` halts the whole chain on any non-gated step's nonzero rc (gated-but-absent steps
are SKIPs, not failures, and never abort the chain). A double-launch refuse guard
(`engine/chain_local_20260716.pid`) and a `.done` marker on clean completion match the existing
`chain_gainwave_20260715.sh`/`chain_e1_editors_20260716.sh` house style. Kill-by-PID only
throughout (`kill -0`, never `pgrep`/`pkill -f`). Header carries the lid-open reminder.

**Validated** (no GPU touched):
- `bash -n engine/chain_local_20260716.sh` clean.
- Isolated logic checks (scratch files under the session scratchpad, not the real `engine/`
  paths): the MIX_A glob-count against the actual live state returns 9 (matches a direct `ls`);
  the `RE_GO.ok` whitespace-strip + `case` dispatch correctly matches `solo`/`base` and correctly
  falls through on `bogus`; the pid-liveness check (`kill -0`) correctly reports dead for a
  fabricated high PID and alive for the test shell's own PID.
- The `run_step` helper's mechanics (setsid-launch, pidfile write, `wait`, rc capture) were
  exercised standalone against two throwaway commands (`exit 0` and `exit 5`) in a scratch
  script — both rc values (0 and 5) and both log files came back correct.
- Confirmed `experiments/frame_a/run_stream.py` and `experiments/frame_a/arms/*.py` were not
  touched (no Edit/Write calls issued against them all session; mtimes predate this session).
  Confirmed `experiments/merging_m0.py` likewise untouched (deliverable 1 depends on it staying
  exactly as-is).
- None of `engine/FRAMEA_LOCAL_BC.ok`, `engine/RE_GO.ok`, `engine/PAPERB_GO.ok` exist yet
  (checked directly) — confirming steps (d)/(e)/(f) are correctly inert until a human ratifies
  each one, exactly as designed.

## Open risks / follow-ups

1. **Paper B Phase-1 driver does not exist** — blocking item, see section 2. Chain step (f) is
   safe (skips cleanly) either way, but the deliverable itself is incomplete pending either the
   ratified prereg doc or an explicit team-lead instruction to build against an extrapolated
   scope.
2. R-E (`experiments/prospective_admission.py`) still has no dedicated launch driver (own
   preflight/idle-gate/budget skeleton) — chain step (e) inlines a bare idle-gate + direct module
   invocation instead. This matches the prereg's own Launch section ("a driver mirroring
   run_merging_editors.sh's skeleton should be built alongside ratification, not before it") —
   flagging in case the reviewer wants a standalone driver built now instead of leaving it to the
   chain.
3. The `--ns_reference solo` GPU code path (the actual `full_target_scores` integration around
   the install/restore, as opposed to the pure-tensor mechanics `--selftest` now covers) has
   never executed against a real model — same category of unvalidated surface the original R-E
   build already flagged for `_measure_one_group`/`_neighborhood_damage_rate`.
4. Chain step (a)'s MIX_A wait loop has no timeout/abort — by design (an unusually slow but
   healthy wave shouldn't be killed by this chain), but a human should know a stuck chain here is
   killable only via `engine/chain_local_20260716.pid`, which will show the CHAIN's own pid (the
   wait loop itself doesn't spawn a sub-process), not the MIX_A job's pid.
