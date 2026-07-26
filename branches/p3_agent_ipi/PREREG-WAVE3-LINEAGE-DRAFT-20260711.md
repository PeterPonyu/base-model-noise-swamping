# P3 wave-3 lineage-arm pre-registration — DRAFT — NOT FROZEN; awaiting user freeze + download approval — 2026-07-11

Mirrors `PREREG-B2B4-FROZEN-20260710.md`'s structure. This file is a DRAFT: it may be
edited freely until the user explicitly freezes it (and separately approves the 4 model
downloads in `DOWNLOAD-MANIFEST-WAVE3-20260711.md`). Nothing here has been launched;
`jobs/queue.json` has not been touched.

## 0. What problem this wave fixes (from `docs/plans/DESIGN-P3-LINEAGE-ARM-2026-07-11.md`)

The `core`/`outgroup` architecture-matched pairs are ALL (deepseek-r1 distill) x
(qwen2.5/qwen3 base) — the r1-distills score ASR=0 on every item (constant success vector,
Pearson forced to 0), so `mean_architecture_corr` is a STRUCTURAL 0 and
`lineage_gt_architecture` is trivially true regardless of any real lineage-vs-architecture
effect. Wave-2's 16 out-group models don't fix this: `analyze._pair_class` (analyze.py:84-92)
returns `None` for any pair touching `group=="out"`, so they never enter `architecture` or
`lineage` pairs.

Wave 3 (a) adds 4 new roster rows (`grid.py` `EXTENDED_DESIGN` "Group D", `group="base"`)
and (b) **relabels the known-ATTACKABLE `llama3.1:8b-instruct-q8_0` (measured ASR ~0.10 in
wave-1/2) from `group="out"` to an in-group alternate-lineage ANCHOR** (`group="base"`,
`lineage="llama-instruct"`) so it forms real architecture-matched pairs with the three
Llama-3.1-8B fine-tunes. This gives `mean_architecture_corr` a genuine (non-degenerate)
value anchored on a model we KNOW is attackable, rather than relying solely on three
not-yet-downloaded fine-tunes.

The relabel is **tier-LOCAL** (`grid.py` `TIER_OVERRIDES["lineage_arm"]`): the global
`EXTENDED_DESIGN` `llama3.1` row is untouched (stays `group="out"`), and offline regression
confirms `core`/`outgroup` still resolve `llama3.1` as out-group — so the frozen wave-1/2
`ipi_grid_core_*`/`outgroup` JSONs are byte-unchanged; nothing that already ran is redefined.
No `analyze.py` pairing logic is touched — only roster metadata.

## 1. Tier (decision 1)

**`lineage_arm` x seeds {0,1,2}, n=30 scenarios** — `grid.py` `TIERS["lineage_arm"]` =
`_CORE` (the 11 wave-1 models) + 4 new rows, with `llama3.1:8b-instruct-q8_0` relabeled
in-group by `TIER_OVERRIDES` (see §0):

| tag | lineage | arch_family | match_group | scale | note |
|---|---|---|---|---|---|
| `llama3.1:8b-instruct-q8_0` | llama-instruct | Llama3.1 | Llama3.1/large | large | already local; relabeled out->base for this tier (anchor, ASR ~0.10) |
| `hermes3:8b` | hermes | Llama3.1 | Llama3.1/large | large | download |
| `dolphin3:8b` | dolphin | Llama3.1 | Llama3.1/large | large | download |
| `tulu3:8b` | tulu | Llama3.1 | Llama3.1/large | large | download |
| `openthinker:7b` | openthinker | Qwen2.5 | Qwen2.5/mid | mid | download |

15 models total (llama3.1 was already in `_CORE`; it is relabeled, not added). Offline-verified
(`grid.py --tier lineage_arm --backend mock`) to produce **12** `architecture_matched_pairs`
(up from 4 at `core`):
- **6 in the `Llama3.1/large` cluster** — the four Llama-3.1-8B models (llama3.1 anchor +
  hermes3 + dolphin3 + tulu3), all different lineage: 3 anchored pairs
  (`llama3.1`×{hermes3,dolphin3,tulu3}, each involving the known-attackable anchor) + 3
  among the fine-tunes.
- **2 at `Qwen2.5/mid`** — (`qwen2.5:7b-instruct-q8_0`, `openthinker:7b`) [attackable] and
  (`deepseek-r1:7b`, `openthinker:7b`) [r1 side degenerate].
- **4 pre-existing degenerate r1<->base pairs** (`Qwen2.5/small`, `Qwen2.5/mid`,
  `Qwen2.5/xl`, `Qwen3/large`) — reported, NOT deleted, per the design doc's "honest, not
  p-hacking" principle.

The headline win: `mean_architecture_corr` is now computed over pairs that include the 6
anchored/attackable Llama pairs, so it is a genuine number instead of the structural 0 the
`core`/`outgroup` tiers were forced to.

## 2. Arms / claim under test (decision 2)

The claim tested is narrower than the general "lineage > architecture" headline:
**"among ATTACKABLE (non-zero-ASR) models, architecture-matched pairs of different
post-training lineage do not show higher cross-model IPI-agreement than same-lineage
cross-scale pairs."** The pre-existing r1-distill degenerate pairs remain in the panel and
are reported alongside (they are the ORIGINAL structural-0 problem, not deleted, not
excluded from `architecture_matched_pairs` — excluding them would itself be a form of
p-hacking the design doc explicitly rules out).

No defense arms in wave 3 (B2 is separately settled — kill-gate FAIL, backlogged per
`p3-laneB-results-20260710` memory). This wave is `ipi_grid` only, 3 seeds, same as B4
wave-1's headline arm.

## 3. Gate parameters (decision 3) — CORRECTED, not the B2 defense gate

**Binding correction**: 0.20 / 0.80 / 0.05 in `PREREG-B2B4-FROZEN-20260710.md` sec 3 are the
**B2 DEFENSE** kill-gate (mean-ASR-drop threshold, defense-on vs off) — inapplicable here,
there is no defense arm in wave 3. Wave-3 success is defined directly from
`analyze.contrast()`'s own output fields (`run_ipi.py` / `analyze.py`):

```
lineage_gt_architecture == True                    (analyze.py:202, obs_diff > 0)
AND p_value < 0.05          (within-model item-permutation, n_perm >= 1000; analyze.py:164)
AND audit false-negative rate <= [ceiling TBD by user before freeze — audit_unmatched.py
     reported FN rates historically range 0-11.7% (conservative) on this codebase;
     PROPOSE ceiling = 15% pending user confirmation]
AND >= 1 surviving attackable architecture pair: both members have per_model_asr > 0
     (run_ipi.py "per_model_asr") AND neither member is in "contrast_excluded_models"
     (nulled by ERROR_RATE_THRESHOLD=0.2, run_ipi.py:51,193)
```

If the 3rd and 4th conditions both fail for a given seed (no surviving attackable pair AND
audit FN rate over ceiling), that seed's contrast is UNTRUSTED and must not be quoted
standalone — fold across the 3 seeds only over trusted cells.

**Open before freeze**: the audit FN-rate ceiling above is a placeholder proposal, not a
frozen number — the user must confirm or replace it. `p_value` uses the SAME within-model
item-permutation null as wave-1 (analyze.py's own docstring rationale: across-model
lineage-label permutation is degenerate here for the same reason it was in wave-1 — a
balanced-group global label swap always ties the observed statistic — so within-model item
permutation is the only test with no p-floor).

## 3a. Singleton-lineage suppression behavior + opt-in relaxation (2026-07-11, MAJOR-1 fix)

**Suppression, as it behaves by DEFAULT (`--allow_singleton_lineage_drop` absent/False,
which is the state of every command in sec 4 below unless explicitly edited).** Five of the
`lineage_arm` in-group lineages are SINGLETONS — one model each: `llama-instruct` (the
tier-local anchor relabel of `llama3.1:8b-instruct-q8_0`), `hermes`, `dolphin`, `tulu`,
`openthinker`. `run_ipi.py::_gate_contrast` suppresses the ENTIRE seed's contrast
(`contrast=None`) if ANY in-group lineage — including a singleton — loses all its members
to `error_rate>0.2` (`ERROR_RATE_THRESHOLD`, run_ipi.py:51). Concretely: if `openthinker:7b`
alone errors out on a seed, the whole seed's contrast is suppressed, taking the 6 healthy
Llama3.1/large architecture pairs down with it — NOT just openthinker's own
`Qwen2.5/mid` pair. This is a FAIL-SAFE (it never emits a silently-wrong number) but it is
NOT "openthinker will simply fail to contribute a surviving pair" — that framing in
`DOWNLOAD-MANIFEST-WAVE3-20260711.md` was incorrect as originally written and has been
corrected there (2026-07-11).

**Opt-in relaxation (`--allow_singleton_lineage_drop`, default OFF).** `run_grid.py` and
`make_jobs.py --kind grid` accept `--allow_singleton_lineage_drop` (threaded to
`run_ipi.run` -> `_gate_contrast`). When explicitly passed, AND ONLY when:
  (i) every lost in-group lineage is a SINGLETON (exactly one model in the design panel) —
      a multi-member lineage (`r1-distill`, `base-instruct`) losing all its members still
      unconditionally suppresses, flag or no flag;
  (ii) consequently both multi-member anchor lineages (`r1-distill`, `base-instruct`) still
       have >= 1 surviving member each; and
  (iii) at least one architecture-matched pair among the survivors is ATTACKABLE — both
       members' `per_model_asr > 0`, not merely alive/un-nulled —
the gate drops the dead singleton lineage(s) instead of suppressing, and records exactly
which ones in `contrast["dropped_singleton_lineages"]`. Offline regression
(`run_ipi.py --selftest`) proves: with the flag OFF, nulling only `openthinker` still
suppresses the whole `lineage_arm` contrast byte-identically to pre-fix behavior; with the
flag ON, the same nulled `openthinker` yields a contrast with `dropped_singleton_lineages ==
["openthinker"]` and all 6 Llama3.1/large architecture pairs present.

**This is a conscious per-launch decision, not a default.** None of the queue commands in
sec 4 set this flag; freezing this doc does NOT imply enabling it. If the user wants the
`lineage_arm` launch to survive a degenerate `openthinker:7b` rather than report a fully
suppressed seed, that requires explicitly adding `--allow_singleton_lineage_drop` to the
`make_jobs.py` invocation (or editing the queued job's `cmd`) before launch — a decision to
make, and record here, at freeze time.

## 4. Queue as frozen (what will run, in order, once the user freezes this doc + approves downloads)

1. `ipi_grid_lineage_arm_n30_s0` / `_s1` / `_s2` (3 seeds, the wave-3 headline)

**Exact commands** (NOT run by this build task — offline-verify only was performed):

```
cd branches/p3_agent_ipi
python make_jobs.py --kind grid --tier lineage_arm --seeds 0,1,2 --n 30 --n_perm 1000
nohup env BUDGET_MIN=420 JOB_CAP_MIN=100 ./run_p3_gpu.sh >> logs/run_p3_gpu.nohup.log 2>&1 &
```

Runtime estimate (linear from the wave-2 spec's measured ~539s/11-model-panel-sweep):
15/11 x 539s approx 735s (~12.3 min) per seed x 3 seeds approx **37 min** total, before any
first-load overhead for the 4 new checkpoints (unmeasured — first pull + first load of a
previously-absent model is typically the slowest single step; see
`DOWNLOAD-MANIFEST-WAVE3-20260711.md`). Treat as ROUGH per this workspace's standing
"do not extrapolate" caveat; `JOB_CAP_MIN=100` bounds any single runaway job regardless.

**Binding precondition** (same as wave-2 spec sec 5): do not launch while any `edit-harness`
GPU job or the Lane A/B chain holds the card. Check
`nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader` and wait by PID
(`kill -0`), never `pgrep`/`pkill -f` a pattern.

## 5. Binding post-run conditions (standing, same as wave-1)

- `audit_unmatched.py` on the `ipi_grid_lineage_arm_*` result before ANY ASR/lineage number
  is quoted (run automatically by `run_p3_gpu.sh` post-run).
- Report `per_model_asr` for all 4 new models explicitly, even if this wave's headline
  passes.
- **Singleton-lineage suppression is explicit, not implicit (MAJOR-1 fix, sec 3a):** by
  DEFAULT (no `--allow_singleton_lineage_drop`), a degenerate (all-error, nulled)
  `openthinker:7b` does NOT merely drop its own `Qwen2.5/mid` pair — it is a SINGLETON
  in-group lineage, so its loss suppresses the ENTIRE seed's contrast (`contrast=None`),
  Llama3.1/large replication included. The Llama3.1/large result "stands on its own"
  independent of openthinker ONLY if `--allow_singleton_lineage_drop` was explicitly set
  for that launch (sec 3a). State plainly in any writeup which launch mode actually
  produced the reported numbers — do not assume or imply the flag was on unless the queued
  `cmd` (in `jobs/queue.json`) shows it.
- **Scope note (reviewer MINOR-1):** `mean_lineage_corr` / `mean_architecture_corr`
  (`analyze.contrast`'s pooled means, `run_ipi.py`'s `contrast` field) average over ALL
  surviving pairs in each class — `mean_architecture_corr` in particular pools the 4
  pre-existing degenerate r1-distill-vs-base pairs (constant zero-ASR vectors, correlation
  forced to 0) together with the new attackable Llama3.1/large (and, if not dropped,
  Qwen2.5/mid) pairs. `observed_diff` (and therefore `lineage_gt_architecture`) is a POOLED
  degenerate+attackable contrast, NOT a clean attackable-only one — the r1-distill pairs are
  never excluded from this mean (sec 5, next bullet: they're never filtered out, by design).
  Any writeup MUST describe `observed_diff` as pooling degenerate and attackable pairs
  (disclosed, conservative in the lineage>architecture direction since the degenerate pairs
  drag `mean_architecture_corr` toward 0, widening the gap) — never narrate it as an
  "attackable-only" contrast.
- The r1-distill degenerate pairs stay in every reported `architecture_matched_pairs` table
  — never filtered out to inflate the apparent effect size.
- If `lineage_gt_architecture` flips to `False` under this arm (the pre-registered
  falsification outcome the whole design exists to test for), that is reported as the
  headline result, not treated as a failed run.
