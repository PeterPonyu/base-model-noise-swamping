# Macro source audit — four manuscripts (2026-08-01)

Machine audit of every `\newcommand` in the four submission macro files against the
artifacts their provenance comments name. Plan reference: `docs/plans/PLAN-DEEP-STRENGTHEN-2026-08-01.md` §T1.

- **Auditor**: `submissions/audit_macro_sources.py` (read-only; no manuscript was modified)
- **Run**: `python3 submissions/audit_macro_sources.py --json <out>`
- **Method**: parse `\newcommand{\X}{value}` + its `%` provenance comment → resolve the
  artifact → resolve the field path → round the recomputed value to the macro's *displayed*
  decimals → compare. Anything not mechanically resolvable is `CANNOT-VERIFY`, never guessed.

> **Concurrency caveat.** During this run another workstream was actively modifying and
> deleting files under `edit-harness/results/frame_a/cells/` (6 MIX_B `_s2` cells changed or
> removed, per `git status`). The frame-a section below reflects disk state at run time. The
> auditor's seed-cardinality guard turns an incomplete cell set into CANNOT-VERIFY rather than
> a fabricated mismatch, so the reported frame-a mismatches are sound — but frame-a counts
> should be re-derived once that tree settles. The other three papers read artifacts that were
> not being modified.

## Verdicts

| Paper | Macros | MATCH | MISMATCH | CANNOT-VERIFY | Verdict |
|---|---:|---:|---:|---:|---|
| `ieee` (B6 → TETCI, under review) | 292 | 49 | 1 | 242 | **DEFECTS-1** |
| `d2-neurocomputing` | 78 | 3 | 0 | 75 | **CLEAN** |
| `paper-b-neurocomputing` | 335 | 252 | 0 | 83 | **CLEAN** |
| `frame-a-eswa` (quarantined) | 52 | 2 | 7 | 43 | **QUARANTINED** — 7 reported, none failed |
| **Total** | **757** | **306** | **8** | **443** | |

The single non-quarantined defect is a standard-deviation macro, not a headline result.
No headline statistic in any paper mismatched its artifact.

## MISMATCH list (full, all 8)

### `ieee` — 1 defect

| Macro | Line | Quoted | Artifact value | Source |
|---|---|---|---|---|
| `\gWithinLtenSD` | 31 | `0.005` | `0.0105` | `edit-harness/results/G1_L10_analysis.json::within_probe_std_across_seeds` |

Hand-verified (appendix A1). The L10 within-probe SD is **0.0105**; the macro says `0.005`,
which is neither the artifact field, the population SD (0.010474), nor the sample SD
(0.012828) of the per-seed values `[0.5321, 0.5218, 0.5473]`. Its two siblings are correct
(`\gWithinLeightSD` 0.011 vs 0.0106 ✓, `\gWithinLtwelveSD` 0.019 vs 0.0189 ✓), which isolates
this to a single transcription slip — most consistent with a dropped digit (`0.0105` → `0.005`).
Low blast radius: a dispersion figure, not the reported ρ. **Fix before the TETCI revision
round.** Not fixed here (task is read-only on manuscripts).

### `frame-a-eswa` — 7 reported, NOT failed (quarantined, known-stale-pending-H14)

All seven are `quality.Q` means over the per-seed cell files. They are consistent with the
documented Frame-A contamination incident (synthetic-relabel, 2026-07-21/26) and the
still-pending H14 re-run, so the audit marks them stale-pending rather than failing them.

| Macro | Line | Quoted | Recomputed (mean over 3 cells) | Cell set |
|---|---|---|---|---|
| `\qBoth` | 11 | `0.999` | `0.9383` | MIX_A / both |
| `\qGrace` | 12 | `1.000` | `0.9544` | MIX_A / always_grace |
| `\qEdit` | 13 | `0.804` | `0.8266` | MIX_A / always_edit |
| `\qDamageOnly` | 16 | `0.999` | `0.9383` | MIX_A / damage_only |
| `\qBothB` | 59 | `0.999` | `0.8853` | MIX_B / both |
| `\qGraceB` | 60 | `1.000` | `0.8909` | MIX_B / always_grace |
| `\qEditB` | 61 | `0.804` | `0.8372` | MIX_B / always_edit |

`\qRag` / `\qRagB` (0.604) **do** reproduce exactly, so the cell files and the aggregation
path are sound — the divergence is confined to the router/GRACE/edit arms.

Related structural finding (reported as CANNOT-VERIFY, 6 macros): the declared verdict
artifact `results/frame_a/frame_a_verdict_ftfix.json` **does not exist**; it is on disk only
as `frame_a_verdict_ftfix.json.INVALID-ALLMIXES-20260726`. Every macro sourced to it is
therefore unverifiable by construction until the H14 re-run lands. This is consistent with
the standing "frame-a-eswa main.pdf 不可投" status and is not a new defect.

## CANNOT-VERIFY, grouped by reason

443 total. These are **provenance-format limits, not number errors** — the auditor refuses to
guess a field path.

| Reason | Count | Papers |
|---|---:|---|
| Artifact resolves, but the comment states no machine-parseable field path (prose-only provenance) | 243 | ieee 198, d2 42, frame-a 3 |
| Macro absent from generator output — hand-added to an auto-generated file | 83 | paper-b 83 |
| Non-scalar macro body (string / list / range / CI / label) | 56 | ieee 31, d2 25 |
| No artifact named in the provenance comment | 36 | frame-a 26, d2 8, ieee 2 |
| Named artifact missing on disk | 13 | frame-a 8 (+6 quarantine-only), ieee 5 |
| Field name ambiguous inside the artifact (multiple occurrences) | 3 | ieee (`ng_to_margin_damage_rho` ×2, `mean_esr`) |
| Derived arithmetic does not close from the comment alone | 1 | ieee |
| Layer disagreement between macro and only resolvable artifact | 2 | ieee |

Two structural observations worth acting on:

1. **`paper-b` has 83 hand-added macros in a file whose header says
   `DO NOT hand-edit; regenerate with: python3 experiments/quant_survival_macros.py`.**
   The 252 generated macros regenerate byte-identically (strongest evidence class in this
   audit), but the 83 later additions (the 07-26 revision analyses: editor-ordering
   bootstrap, base-noise swamping) sit outside the generator and so outside its guarantee.
   Folding them into the generator would move 83 macros from CANNOT-VERIFY to MATCH.
2. **`ieee`'s 198 prose-only macros** are the single largest coverage gap. They name the right
   artifact but describe the field in prose ("L12 peak", "3-seed within-probe aggregate").
   Appending `::field.path` to those comments — the convention already used in the §8
   extension block, which is why those macros verify — would make most of them machine-checkable.

## Auditor verification (done before trusting the run)

The auditor was itself hand-checked, and the checking found **five auditor defects**, all
fixed. They are recorded because each was a would-be false positive:

1. **Block-comment leakage.** A source block absorbed unrelated later comment lines,
   mis-attributing `\gResidNorm` (a residual norm, 22.9) to a ρ field. Fixed: absorption now
   stops at any intervening macro or rule line.
2. **Layer crossing.** A `L{8,14}` brace-set source paired the L12 macro `\rewriteRef` to the
   L8 file, reporting `0.602 vs 0.461`. Fixed: a layer-agreement guard refuses when the macro
   and the resolved filename disagree on layer.
3. **Mean/SD inversion.** `\gWithinLeight` (`0.395  % L8 (SD 0.011)`) mentions its *partner's*
   SD, so comment-sniffing promoted it to the std field, reporting `0.395 vs 0.0106`. Fixed:
   the macro **name** is authoritative for mean-vs-dispersion.
4. **Phantom arithmetic.** The arithmetic scanner read model names and dates as expressions
   (`llama-3.2-1b` → `3.2-1`, `20260714` → subtraction), producing bogus reasons on 34
   findings. Fixed: operators must be whitespace-delimited.
5. **Incomplete-glob aggregation.** Frame-A means were computed over however many cell files
   happened to exist, which would fabricate a mismatch from a 2-of-3 seed set. Fixed: the
   declared `s{0,1,2}` cardinality must be present on disk or the macro goes CANNOT-VERIFY.
   (This correctly moved 3 findings out of MISMATCH.)

Expected-mismatch handling was verified: all four pre-declared macros are present and none is
counted as a defect. `\phiSigned` (0.017) actually **verifies** post-refix; `\magPhi`,
`\magPhiSD`, `\magPhiPeek` land in CANNOT-VERIFY (prose-only provenance) and carry the
`expected` flag. 142 findings carry a self-declared stale/pre-refix tag from their own
comments and are excluded from defect counts.

## Appendix A — manual recomputes

Ten entries hand-checked independently of the auditor (4 MATCH, 4 CANNOT-VERIFY, 2 MISMATCH;
random sample, seed 42, plus both mismatch classes forced in).

### A1 — `\gWithinLtenSD` (ieee, MISMATCH) — confirms the defect
```
G1_L10_analysis.json aggregate.within_probe_std_across_seeds = 0.0105
per_seed within_probe_mean                                   = [0.5321, 0.5218, 0.5473]
population stdev = 0.010474      sample stdev = 0.012828
macro                                                        = 0.005
=> no aggregation of the artifact yields 0.005.  DEFECT CONFIRMED.
Siblings, same block, same statistic:
  L8 : artifact 0.0106  macro 0.011  -> MATCH
  L12: artifact 0.0189  macro 0.019  -> MATCH
```

### A2 — `\qBoth` (frame-a, MISMATCH) — confirms, and bounds the scope
```
cells/cell_llama-3.2-1b_real_MIX_A_both_s{0,1,2}.json  quality.Q
  = [0.9502, 0.9134, 0.9513]   mean = 0.9383     macro = 0.999
control (same path, same code):
  always_rag Q = [0.604, 0.604, 0.604]  mean = 0.6040  macro = 0.604  -> MATCH
=> aggregation logic is correct; the router/grace/edit arms are the stale ones.
```

### A3 — `\delRefLeight` (ieee, MATCH)
```
C3_u1_blockB_L8_seeds_u5.json::within_probe_mean_across_seeds = 0.461
macro = 0.461                                                  -> MATCH
(also cross-checks auditor defect #2: the L8 macro correctly binds the L8 file)
```

### A4 — `\instructCpeak` (ieee, MATCH)
```
INSTRUCT_mechanism_sc_table.json::groups[L12].within_probe_rho_C = 0.5586
macro = 0.559  (3dp)                                             -> MATCH
```

### A5 — `\rippleLfourteen` (ieee, MATCH, relaxed path)
```
comment path : profile.L14.rho_ripple_mean
actual path  : rome_depth_profile.L14.rho_ripple_mean = 0.4915470381693712
macro = 0.492 (3dp)                                              -> MATCH
(the declared path omits one container; resolved as a unique subsequence match)
```

### A6 — `\cfplusNSmid` (ieee, MATCH)
```
CFPLUS_aggregate.json::by_editor.alpha.NS_mean = 0.47333333333333333
macro = 0.473                                                    -> MATCH
```

### A7 — `\pRankSurvEightThreshold` + `\pPriFNFMLtwoPt` + `\pSecFNFMBase` (paper-b, MATCH)
```
Verified by regeneration, not comment parsing:
  python3 experiments/quant_survival_macros.py --out_path <tmp>   (rc=0)
  regenerated 0.950 / 0.996 / 0.640  ==  committed 0.950 / 0.996 / 0.640
All 252 generator-emitted macros reproduce; 0 mismatches.
```

### A8 — `\gWithinLtwelve` (ieee, MATCH via canonical field)
```
G1_L12_analysis.json aggregate.within_probe_mean_across_seeds = 0.6018
macro = 0.602                                                    -> MATCH
```

### A9 — CANNOT-VERIFY, prose-only provenance (4 sampled, all correctly classified)
```
\eosRaw   0.616  -> artifact u1_gate_eos_L12_s0.json resolves; comment says
                    "raw key-cos within-probe, 3-seed" with no field path.
\ftRho    0.024  -> gate_llama1b_ft_cf_L8_s0.json resolves; no field path.
\klZeroThreeRhoSD 0.008 -> C3_klladder_003_L8_seeds_u5.json resolves; no field path.
\pWithinNfFMQOneRome 0.879 -> absent from the paper-b generator output (hand-added).
Each is a provenance-format gap, not evidence of a wrong number. Correctly NOT failed.
```

### A10 — frame-a quarantined-artifact check
```
declared : results/frame_a/frame_a_verdict_ftfix.json      -> DOES NOT EXIST
on disk  : frame_a_verdict_ftfix.json.INVALID-ALLMIXES-20260726
=> 6 macros sourced to it are unverifiable until the H14 re-run. Reported, not failed.
```

## Actions

1. **`ieee` / `\gWithinLtenSD`**: `0.005` → `0.0105` (or `0.011` at the block's precision).
   One-token fix; fold into the TETCI revision. Only real defect in the audit.
2. **`frame-a-eswa`**: the 7 `q*` macros and the 6 verdict-sourced macros stay blocked on H14.
   The `always_rag` control reproducing exactly is useful evidence that the harness and
   aggregation are sound.
3. **`paper-b-neurocomputing`**: fold the 83 hand-added 07-26 revision macros into
   `quant_survival_macros.py` so the file's own "do not hand-edit" contract holds.
4. **`ieee`**: append `::field.path` to prose-only comments (the §8 convention) to convert
   ~198 CANNOT-VERIFY into machine-checked macros. Highest-leverage durable improvement.
5. Re-run this auditor after each of the above; it is idempotent and takes ~1 minute.
