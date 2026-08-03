# D2 Zenodo Deposit Repair Log — 2026-08-01

Repairs applied against `DEPOSIT-AUDIT-20260801.md` (verdict **DEFECTS-8 — NOT
SELF-CONTAINED**, 3 PASS / 10 DEFECT rows). Author-side log; a separate reviewer
re-audits after this.

Scope constraints honoured: no `.PHI-PREFIX-STALE` file deleted; no manuscript (`.tex`)
edit; CPU-only throughout; repo originals under `edit-harness/` touched only where the
audit's recommended fix required it (D2, D4) and never in a way that changes the
2026-07-15/16 default semantics.

## Summary of the eight defects

| Defect | Status | Where fixed |
|---|---|---|
| D1 figure pipeline hard-codes absent pre-refix files | **FIXED** | `zenodo-deposit/figures/make_figures.R` |
| D2 operating-map aggregate + consolidator pre-refix | **FIXED** | consolidator (repo + deposit); new `RG_map_evidence_REFIX20260801.json` |
| D3 shipped analysis scripts don't reproduce REFIX names | **FIXED** | 4 deposit scripts repointed |
| D4 signed re-analysis consumes stale Phi tables | **FIXED** | `rg_signed_reanalysis.py` (repo + deposit) + new REFIX aggregate |
| D5 two stale Phi operating tables unquarantined | **FIXED** | renamed `.PHI-PREFIX-STALE` + deterministic source precedence |
| D6 matched-dose generator omitted | **FIXED** | copied into `code/experiments/` |
| D7 prereg copies lack ratification status | **NOT FIXED — reported** | see "Defects not fixed" |
| D8 `CITATION.cff` release metadata incomplete | **NOT FIXED — reported** | see "Defects not fixed" |

Two additional blocking defects were found while verifying and are fixed: a
results-path layout wart that made four shipped scripts abort, and a stale figA
cell-count comment. Both are recorded below.

## Item 1 — Map evidence regenerated with refixed Phi (audit D2, D5; rows 94, 95)

### Code change: `edit-harness/experiments/rg_map_evidence_consolidate.py`

Before: module-level constants, no CLI. Line 22 `OUT = "results/merging/RG_map_evidence_20260716.json"`;
line 25 `gl = json.load(open("results/merging/RG_gain_law_20260715.json"))`; line 34 glob
admitted every `RG_operating_curve_table*.json` and line 40 let **glob order** decide
which of several tables describing the same `(model, layer)` won; line 92 recorded the
stale gain-law path.

After: `argparse` with `--results_dir` / `--gain_law` / `--out`. Defaults are now the
refixed set (`RG_gain_law_MERGED_REFIX20260730.json` in, `RG_map_evidence_REFIX20260801.json`
out); the pre-refix invocation is still reachable verbatim and is documented in the
docstring. Duplicate `(model, layer)` tables are resolved by explicit precedence
(`_rank`: REFIX-tagged > bundle-local `<cell>_RG/RG_operating_curve_table.json` > flat
legacy name) and a top-rank tie whose contents disagree raises `SystemExit` rather than
silently picking one (closes the audit's "make consolidation reject duplicate
`(model, layer)` sources"). Recorded paths are normalized through `_rec()` to
`results/merging/<rel>` so the artifact is byte-stable across absolute/relative
`--results_dir` values. Provenance gained `superseded_duplicates_ignored`,
`source_precedence`, and a refix `note`; `created` moved `2026-07-16` → `2026-08-01`.

### Artifact written

`RG_map_evidence_REFIX20260801.json`, into **both** `edit-harness/results/merging/` and
`zenodo-deposit/results/merging/`. All 22 cell blocks are byte-identical between the two
copies (verified by JSON canonical compare); the only difference is
`provenance.opcurve_files` / `superseded_duplicates_ignored`, because the repo tree also
holds the post-freeze `Llama-2-13b-hf_L30_RG` boundary cell that the deposit does not
ship (23 vs 22 sources). Totals unchanged: `n_cells=22`, `n_subcells=330`,
`total_merge_observations=65868`.

### Phi rows verified to differ from the stale version

The audit's two named values reproduce exactly:

| Quantity | Stale (`RG_map_evidence_20260716.json`) | REFIX (`...REFIX20260801.json`) |
|---|---:|---:|
| Phi-3.5 L16, g=2, s0 median abs drop | 0.5576 | **0.2611** |
| Phi-3.5 L24, g=2, s0 partial rho | −0.1011 | **−0.2745** |

Cell-level movement (source table + gain + representative g):

| Cell | source_table | gain | g=2 med3 | g=20 med3 | g=2 partial rho mean |
|---|---|---:|---:|---:|---:|
| Phi-3.5 L16 | `..._phi35_L16.json` → `..._phi35_L16_REFIX20260730.json` | 16.9271 → 15.6357 | 0.4776 → 0.2649 | 9.1409 → 5.2012 | 0.1634 → 0.0995 |
| Phi-3.5 L24 | `..._phi35_L24.json` → `..._phi35_L24_REFIX20260730.json` | 1.8668 → 3.9101 | 0.0391 → 0.0548 | 1.6633 → 3.6384 | −0.2066 → −0.2453 |

Regime labels are unchanged (L16 stays high-gain, L24 stays low-gain). Every **non-Phi**
cell is identical to the stale aggregate once the `source_table` string is excluded
(checked all 20 cells field-by-field): the only non-Phi change is that provenance now
names the bundle-local table instead of the byte-identical flat alias, a consequence of
the new deterministic precedence.

### Stale copy retained, not deleted

`zenodo-deposit/results/merging/RG_map_evidence_20260716.json` →
`RG_map_evidence_20260716.json.PHI-PREFIX-STALE` (renamed, contents untouched). The repo
copy at `edit-harness/results/merging/RG_map_evidence_20260716.json` is left in place as
the historical artifact.

## Item 2 — Signed re-analysis regenerated with refixed Phi op-tables (audit D4; row 103)

### Code change: `edit-harness/experiments/rg_signed_reanalysis.py` (deposit copy synced)

The audit named lines 171–173, which read the pre-refix Phi reference tables. The script
was patched to take the paths via argv rather than being rewritten, so the 2026-07-15
semantics survive untouched as the default.

| Line (before) | Before | After |
|---|---|---|
| 138–139 | `--out` default `RG_signed_reanalysis_20260715.json` | `--out` default `None`; resolved to the 20260715 name by default, `RG_signed_reanalysis_REFIX20260801.json` under `--refix` |
| 142 | `mg = os.path.join(HARNESS, "results", "merging")` | `mg = args.results_dir` (new flag) |
| 171 | `os.path.join(mg, "RG_operating_curve_table_phi35_L24.json")` | `phi_l24` — `..._phi35_L24_REFIX20260730.json` under `--refix`, or `--phi_l24_table` |
| 173 | `os.path.join(mg, "RG_operating_curve_table_phi35_L16.json")` | `phi_l16` — same routing via `--phi_l16_table` |
| 203–206 | wrote to `args.out` | writes to `out_path` |

New flags: `--refix`, `--results_dir`, `--phi_l16_table`, `--phi_l24_table`. New report
fields: `phi35_reference_tables`, `phi35_refix`, and (under `--refix`) `note_refix`.

Semantics note recorded in the docstring: only the Phi **reference** tables move. Every
bundle, Phi included, is always re-measured from the on-disk `*_RG/` vectors, so
`--refix` changes which canonical numbers the per-cell `reproduction_check` validates
against, not the measurement. That is exactly why the stale tables were a real defect:
under the pre-refix references the two Phi bundles' `reproduction_check` compares refixed
measurements against invalidated canon.

### Artifact written

`RG_signed_reanalysis_REFIX20260801.json` in repo + deposit; the pre-refix
`RG_signed_reanalysis_20260715.json` quarantined in the deposit as
`.PHI-PREFIX-STALE` (repo copy retained in place).

## Item 3 — Deposit generator scripts repointed at REFIX names (audit D1, D3; rows 96–99, 102, 105)

All edits below are to the **deposit** copies under `zenodo-deposit/`; the repo originals
were left alone except for the two scripts item 1 and item 2 required.

Each edited script carries a top-of-file comment block with the full rename map
(old name → REFIX name + date), so a script read in isolation still documents which
artifact it produces.

### `figures/make_figures.R`

| Line (before) | Before | After |
|---|---|---|
| 2 | `# SOURCE (figA): ...RG_gain_law_20260715.json` | `...RG_gain_law_MERGED_REFIX20260730.json` (+ holdout source) |
| 3 | `# SOURCE (figB): ...RG_crossterm_alignment_20260715.json` | `...RG_crossterm_alignment_ALL_REFIX20260801.json` |
| — | (no figC/D/E SOURCE headers) | added, all naming REFIX artifacts |
| 12 | `fromJSON(... "RG_gain_law_20260715.json")` | `... "RG_gain_law_MERGED_REFIX20260730.json"` |
| 15 | `fromJSON(... "RG_crossterm_alignment_20260715.json")` | `... "RG_crossterm_alignment_ALL_REFIX20260801.json"` |
| 115–116 | figC comment + `fromJSON(... "RG_admission_benefit_20260715.json")` | `... "RG_admission_benefit_REFIX20260730.json"` |
| 152–153 | figD comment + `fromJSON(... "RG_map_evidence_20260716.json")` | `... "RG_map_evidence_REFIX20260801.json"` |
| 195 | figE comment naming `RG_map_evidence_20260716.json` | `RG_map_evidence_REFIX20260801.json` |
| 26 | `## figA: gain vs constructive fraction (19 cells)` | `(22 protocol cells)` — see "additional defects" |

figB needed care: the 2026-07-30 alignment refix is **Phi-scoped**
(`RG_crossterm_alignment_phi35_REFIX20260730.json` carries only the 2 Phi bundles) while
figB plots all 19. Rather than have the figure straddle a live file and a quarantined
one, a full-coverage artifact was regenerated from the shipped bundles —
`RG_crossterm_alignment_ALL_REFIX20260801.json` — and figB reads that. The R header
documents the provenance and the verification.

### `code/experiments/rg_gain_law.py`
- line 73 `--out` default `RG_gain_law_20260715.json` → `RG_gain_law_MERGED_REFIX20260730.json`.
- Docstring notes that a fresh run re-derives all 22 rows and restamps `created`, whereas
  the shipped file was assembled by splicing refixed Phi rows into the 2026-07-16 table
  (hence its `note_refix` field).

### `code/experiments/rg_crossterm_alignment.py`
- line 91 `--out` default `RG_crossterm_alignment_20260715.json` → resolved from a new
  `--bundles {phi,all}` switch: `phi` (default) writes the shipped
  `RG_crossterm_alignment_phi35_REFIX20260730.json`, `all` writes
  `RG_crossterm_alignment_ALL_REFIX20260801.json`.
- `experiment` / `note` fields now reflect which scope ran.
- `args.out` → `out_path` at the write site.

### `code/experiments/rg_admission_benefit.py`
- line 12 docstring `RG_gain_law_20260715.json` → `RG_gain_law_MERGED_REFIX20260730.json`.
- line 71 `--out` default `RG_admission_benefit_20260715.json` → `RG_admission_benefit_REFIX20260730.json`.
- line 75 hard-coded gain-law read → new `--gain_law` flag defaulting to the REFIX table.

### `code/experiments/rg_gain_holdout.py`
- line 78 `json.load(open("results/merging/RG_gain_law_20260715.json"))` → module constant
  `GAIN_LAW = "results/merging/RG_gain_law_MERGED_REFIX20260730.json"`.
- line 135 provenance now records `GAIN_LAW`.
- This one was not merely cosmetic: the script asserts (line ~89) that each recomputed
  gain matches the reference within 2%. Against the stale table that assertion **fails**
  on both refixed Phi bundles (L16 15.6357 vs 16.9271; L24 3.9101 vs 1.8668). Noted in
  the docstring.

### Grep verification

A tree-wide scan for the eight superseded filenames, counting a hit only when the name is
**not** immediately followed by `.PHI-PREFIX-STALE`, returns zero live references. Every
remaining occurrence is one of: a rename-map documentation line, a `supersedes:` field, a
README supersession note, or the historical `prereg/LEDGER-PREREG-2026-07-16.md` record
(a frozen pre-registration document, deliberately not rewritten). Details in
"Verification" below.

## Item 4 — Missing generator shipped (audit D6; row 106)

`edit-harness/experiments/rg_matched_dose_spread.py` → `zenodo-deposit/code/experiments/rg_matched_dose_spread.py`
(same relative layout as the other ten experiment scripts).

Three deposit-local adjustments, matching how the sibling scripts were already
anonymized/adapted:

1. Line 2 header `(deposit artifact, H9 of PLAN-GAP-CLOSURE-MASTER-2026-07-31)` →
   `(deposit artifact for the matched-dose spread statistic)`. Internal plan codenames
   must not ship (same policy the other deposit copies follow).
2. `ADDENDUM_CELLS = ["Llama-2-13b-hf_L30_RG"]` is a post-freeze boundary cell that the
   deposit does **not** ship. The upstream loop `for c in PROTOCOL_22 + ADDENDUM_CELLS`
   raises `SystemExit(f"missing bundle {d}")`, so the shipped copy would have aborted on
   every run. Split into a required pass over `PROTOCOL_22` and an optional pass that
   records the addendum only when the bundle is present.
3. Added `--results_dir` with a sibling-`results/` default (see "additional defects").

## Item 5 — README updated (audit rows 94, 95, 103 + rename map)

`zenodo-deposit/README.md`:

- Row "Operating-map table": artifact list now names the per-cell bundle tables plus
  `RG_operating_curve_table_phi35_L{16,24}_REFIX20260730.json` consolidated into
  `RG_map_evidence_REFIX20260801.json`, with the supersession of
  `RG_map_evidence_20260716.json` stated. The old unqualified wildcard
  `RG_operating_curve_table_*.json` was removed — that wildcard was precisely the D5
  ambiguity (it matched stale and refixed Phi tables at once).
- Row "Gate-evidence figure (figE) and dose-response figure (figD)":
  `RG_map_evidence_20260716.json` → `RG_map_evidence_REFIX20260801.json`.
- Row "g-resolved cross-talk figure (figB)": now names
  `RG_crossterm_alignment_ALL_REFIX20260801.json` and the script invocation
  `rg_crossterm_alignment.py --bundles all`, with the Phi/non-Phi equality statement.
- Row "Signed re-analysis": script `rg_signed_reanalysis.py --refix`, artifact
  `RG_signed_reanalysis_REFIX20260801.json`, supersession noted.
- New **2026-08-01 self-containment repair** block after the existing 2026-07-31 refresh
  block: what was regenerated and why, an explicit statement that no result value changed
  beyond the Phi-3.5 rows already superseded on 2026-07-30/31, and the full rename-map
  table (8 rows, old → shipped, dated). Also states that the shipped
  `figures/fig{A..E}` renders are deliberately the 2026-07-16 build (byte-identical to
  the submitted manuscript's figures) and that re-running the R script regenerates them
  from refixed data, moving only the Phi series.
- "Re-running an analysis on the shipped data": the illustrative `rg_gain_law.py` line was
  replaced with the four regeneration commands, and the `code/`-vs-`results/` layout
  caveat is now stated concretely (which scripts self-resolve, which need the one-line
  symlink) instead of the previous vague "point them at this archive's `results/`".

## Additional defects found while verifying (both fixed)

**A1 — shipped scripts abort on the deposit's own layout.** In the deposit `code/` and
`results/` are siblings, but the scripts inherit the harness layout where `results/` sits
inside the code root, so `HARNESS/results/merging` resolves to the non-existent
`code/results/merging`. Observed live: `rg_matched_dose_spread.py` exited
`missing bundle .../code/results/merging/Llama-3.1-8B_L24_RG`, and
`rg_crossterm_alignment.py --bundles all` produced an artifact with all 19 bundles marked
`MISSING_LOCALLY`. Fixed in the four scripts this repair touches
(`rg_matched_dose_spread.py`, `rg_crossterm_alignment.py`, `rg_signed_reanalysis.py`,
`rg_map_evidence_consolidate.py`): each resolves the harness-local `results/merging` when
it exists and otherwise the archive's sibling `results/merging`, and each accepts an
explicit `--results_dir`. The three untouched scripts (`rg_gain_law.py`,
`rg_gain_holdout.py`, `rg_admission_benefit.py`) keep their original resolution and the
README documents the one-line symlink they need.

**A2 — figA cell-count comment stale.** `make_figures.R:26` said "19 cells" while the
shipped gain-law artifact carries 22 and every row has a matching bootstrap-CI entry in
`RG_gain_holdout_20260716.json` (checked: 22/22, no `NA` rows). Comment corrected to
"22 protocol cells". No plotting logic changed.

## Defects not fixed — reported

**D7 — prereg copies lack explicit ratification status.** Not fixed, deliberately. All
five `prereg/*.md` copies describe frozen pre-run designs but none carries
`STATUS: RATIFIED` or `STATUS: DRAFT`. The audit's own recommendation warns: *"Do not
retroactively assert ratification without the underlying record."* Stamping a status line
now would be an author asserting a historical fact the deposit cannot evidence. This needs
either the underlying pre-run ratification record (to cite a real date) or an explicit
user decision, so it is left for the user. Note also that
`prereg/LEDGER-PREREG-2026-07-16.md:24` cites `RG_gain_law_20260715.json` by name; that
line was **not** rewritten, because a frozen pre-registration document must keep naming
the artifact that existed when it was frozen. The README rename map is what connects it
to the shipped file.

**D8 — `CITATION.cff` release metadata incomplete.** Not fixed. It lacks `version`,
`date-released`, `repository-code`, and a DOI, and still says "DOI assigned at deposit".
The audit classes this as acceptable pre-minting and the fix is inherently
deposit-time: the Zenodo DOI does not exist yet. This is a submission-mechanics step for
the user at deposition (add version + date + minted DOI, then validate with a CFF
validator), not a repair that can be completed now. No license-text change is indicated.

## Verification (author-side)

Re-ran the audit's row-by-row checks (existence + name routing) plus reproduction checks
the audit did not run. Numbers were recomputed from the raw bundle npz files rather than
read from aggregates.

### Reproduction fidelity of the regenerated artifacts

- **Gain law from the deposit's own bundles** (all 22 `*_RG/` dirs, 758 s CPU): every row
  reproduces `RG_gain_law_MERGED_REFIX20260730.json`. The only four field differences are
  the `model` strings, where the deposit stores anonymized names
  (e.g. `Mistral-7B-v0.3`) against the repo's capture paths
  (`/root/autodl-tmp/models/Mistral-7B-v0.3`). No numeric difference in any row. This
  confirms the shipped REFIX gain law is reproducible from shipped data.
- **Cross-term alignment, all 19 bundles** (109 s CPU): the 17 non-Phi rows reproduce the
  retained `RG_crossterm_alignment_20260715.json.PHI-PREFIX-STALE` **field-for-field**,
  and the 2 Phi rows reproduce `RG_crossterm_alignment_phi35_REFIX20260730.json`
  field-for-field, while differing from the stale file (e.g. L16 g2_s0
  `frac_cos_align_pos` 1.0 → 0.91, `rho_proj_drop` 0.1816 → 0.2174). This is the evidence
  behind shipping `..._ALL_REFIX20260801.json` as figB's single source.
- **Operating-table precedence:** the flat aliases are byte-identical to their
  bundle-local counterparts for all 20 non-Phi cells (md5-compared), so the new
  precedence rule changes provenance strings only. The two Phi keys are the sole genuine
  three-way disagreements, which is what the deterministic rule and the hard-error tie
  check now handle.
- **Map evidence, run two ways** (from the archive root and from `code/`): byte-identical
  output, confirming the `_rec()` path normalization.

### Signed re-analysis REFIX parity

- 19 bundles, 0 MISSING_LOCALLY, all `reproduction_check: PASS`
- Cells that changed vs the quarantined stale: exactly `Phi-3.5-mini_L16` and `Phi-3.5-mini_L24`
- Non-Phi cells unchanged
- Sample Phi values: L16 g2/s0 `partial_abscos_given_mag` 0.2026 → **0.0858** (stale → REFIX); L24 g2/s0: −0.1011 → **−0.2745**
- 0 absolute paths in the deposit copy

### All-bundles alignment REFIX parity

- 19/19 bundles computed; 17/17 non-Phi cells reproduce the quarantined 20260715 file field-for-field; 2/2 Phi cells reproduce `RG_crossterm_alignment_phi35_REFIX20260730.json` field-for-field, and both differ from the stale
- `rg_dir` fields recorded as `results/merging/<cell>` (no absolute paths)
- figB macros verified from the new artifact: Qwen-14B −0.7340 (macro −0.73), Llama-1B +0.4041 (macro +0.40), Mistral-7B +0.3230 (macro +0.32)

### Matched-dose spread deposit re-verification

Run from `code/` using the deposit's own bundles via the new sibling-resolver default:

| Quantity | Shipped artifact | Re-derived | Match |
|---|---:|---:|---|
| `spread_max_over_min` | 760.4245 | 760.4245 | OK |
| `max_pairwise_matched_dose_ratio` | 613.9281 | 613.9281 | OK |
| Phi-3.5 L16 response in band | 15.2474 | 15.2474 | OK |
| Phi-3.5 L24 response in band | 2.9004 | 2.9004 | OK |
| All 22 cells with numeric diff | — | NONE | OK |
| Addendum (Llama-2-13b) | (absent) | (absent, expected) | OK |

### Additional model-path anonymization

The REFIX map-evidence and signed re-analysis artifacts inherited 4 absolute model paths from the gain-law artifact's per-bundle `model` field (Mistral-7B, Mistral-Nemo-12B, gemma-2-9b, GPT-NeoX-20B). These matched the pattern present in the repo's operating tables but were not present in the deposit's stale aggregate. Patched in both repo and deposit copies to the short anonymized names used by the stale deposit:

| Cell | Before | After |
|---|---|---|
| Mistral-7B-v0.3_L24_RG | `/root/autodl-tmp/models/Mistral-7B-v0.3` | `Mistral-7B-v0.3` |
| Mistral-Nemo-Base-2407_L30_RG | `/root/autodl-tmp/models/Mistral-Nemo-Base-2407` | `Mistral-Nemo-Base-2407` |
| gemma-2-9b_L31_RG | `/root/autodl-tmp/models/gemma-2-9b` | `gemma-2-9b` |
| gpt-neox-20b_L33_RG | `/root/autodl-tmp/models/gpt-neox-20b` | `gpt-neox-20b` |

Zero remaining absolute paths in any of the three new artifacts. The `cells` values are unchanged — these fields are provenance-only.

### Final absolute-path sweep

All three new REFIX result artifacts: 0 leaks (RG_signed_reanalysis, RG_crossterm_alignment_ALL, RG_map_evidence).

### Final functional stale-name check

A pattern-based scan was tightened to catch functional occurrences only (lines containing `fromJSON`, `open(`, `json.load(`, `read.csv`, `--out`, `default=`). One hit appeared in `rg_map_evidence_consolidate.py` line 21, but inspection confirmed it is the docstring example showing the pre-refix legacy invocation:

```
python3 experiments/rg_map_evidence_consolidate.py \
    --gain_law results/merging/RG_gain_law_20260715.json \
    --out      results/merging/RG_map_evidence_20260716.json
```

This is intentional — the docstring documents both the old and the new (default) invocations. No functional stale routing found. All 80 total hits enumerated earlier are rename-map documentation, docstring examples, `supersedes:` provenance fields, or the frozen pre-registration ledger.

### Operational note

Three concurrent BLAS-heavy jobs drove load average to ~120 on 24 cores and slowed each ~5×. When re-running: cap threads (`OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4`) or run sequentially. Single-job wall times: gain law ≈ 13 min, alignment ≈ 2 min, signed ≈ 80 min (contended) / ~25 min (uncontended).

### Row-by-row re-audit

| README row | Audit verdict | Now | Basis |
|---:|---|---|---|
| 94 | DEFECT | PASS | consolidator reads `RG_gain_law_MERGED_REFIX20260730.json`, writes the mapped `RG_map_evidence_REFIX20260801.json`; Phi sources are the REFIX tables; duplicate `(model, layer)` resolved deterministically; stale Phi tables quarantined so the map no longer matches both |
| 95 | DEFECT | PASS | figD line reads `RG_map_evidence_REFIX20260801.json`; figE reuses that object; aggregate contains refixed Phi values |
| 96 | DEFECT | PASS | R reads `RG_gain_law_MERGED_REFIX20260730.json`; `rg_gain_law.py` writes the same mapped name |
| 97 | DEFECT | PASS | R reads `..._ALL_REFIX20260801.json` + REFIX gain law; alignment script writes REFIX names under both `--bundles` modes |
| 98 | DEFECT | PASS | `rg_gain_law.py --out` default is the mapped REFIX artifact |
| 99 | DEFECT | PASS | `rg_gain_holdout.py` reads + records the REFIX gain law (also removes a live assertion failure) |
| 100 | PASS | PASS | untouched; Phi permutation-null files still byte-match the canonical harness copies |
| 101 | PASS | PASS | untouched; `../merging_editors/` traversal still resolves inside the deposit |
| 102 | DEFECT | PASS | R reads `RG_admission_benefit_REFIX20260730.json`; analysis script reads REFIX gain law and writes the REFIX admission name |
| 103 | DEFECT | PASS | `rg_signed_reanalysis.py --refix` routes to the REFIX Phi tables and writes `RG_signed_reanalysis_REFIX20260801.json`; pre-refix aggregate quarantined |
| 104 | PASS | PASS | untouched |
| 105 | DEFECT | PASS | alignment script writes the mapped REFIX filename |
| 106 | DEFECT | PASS | `code/experiments/rg_matched_dose_spread.py` now exists, runs on the deposit layout, and writes `RG_matched_dose_spread_REFIX20260731.json` by default |

Remaining open: **D7** (prereg ratification status — needs the underlying record or a user
decision) and **D8** (`CITATION.cff` version/date/DOI — deposit-time step). Both are
reported above rather than forced.

### No published number moves

The manuscript's `macros.tex` cites two of the superseded artifacts in `% SOURCE`
comments, so every macro sourced from them was re-derived against the regenerated
artifacts. All hold:

| Macro | Value in `macros.tex` | Recomputed | Source |
|---|---|---|---|
| `\nSubcells` | 330 | 330 | `RG_map_evidence_REFIX20260801.json` `n_subcells` |
| `\nObsRome` | 65,868 | 65868 | same artifact, `total_merge_observations` |
| `\fourteenBPartialsGtwo` | −0.432/−0.419/−0.523 | −0.4322/−0.4188/−0.5234 | signed re-analysis, Qwen2.5-14B L36 g2 |
| `\fourteenBConstructiveRange` | 81–88% | 81.3–87.7% | same, `frac_drop_negative` g2..10 3-seed means |
| `\projRhoFourteenB` | −0.73 | −0.7340 | alignment, Qwen2.5-14B L36 g2 3-seed mean |
| `\projRhoLlamaOneB` | +0.40 | +0.4041 | alignment, Llama-3.2-1B L12 |
| `\projRhoMistral` | +0.32 | +0.3230 | alignment, Mistral-7B-v0.3 L24 |

The reason none moves is structural: every macro drawn from the signed re-analysis and
the alignment probe is a **Qwen-14B / Llama-1B / Mistral-7B** quantity, and no non-Phi
bundle was touched by the tokenizer defect. The two map-evidence macros are pure counts
(sub-cells and observations), which the refix leaves at 330 / 65,868. The Phi-specific
macros (`\phiGainShallow` 15.6, `\phiGainDeep` 3.9) were already updated on 2026-07-30
and continue to match the REFIX gain law. Consistent with the README's existing claim
that "no published macro moves" — now verified for the 2026-08-01 regenerations too.

Consequently the `% SOURCE` comments in `macros.tex` still name the pre-refix filenames
for these lines. That is a manuscript-side annotation, and manuscript edits were out of
scope for this repair; the values are correct and the README rename map documents the
correspondence. Flagged for the user as an optional cosmetic follow-up.

### Boundary and containment re-checks

- No reference escapes the deposit except the two documented `../merging_editors/`
  traversals, which resolve inside `zenodo-deposit/`.
- No `.PHI-PREFIX-STALE` file was deleted or modified. Four new ones were created by
  renaming (`RG_map_evidence_20260716.json`, `RG_signed_reanalysis_20260715.json`,
  `RG_operating_curve_table_phi35_L{16,24}.json`), bringing the quarantine set to eight.
  Contents were verified untouched: each quarantined file was compared against its repo
  original, and the only differences anywhere are the deposit's **pre-existing**
  anonymization of `model` / `rg_dir` / `source_table` path strings
  (`/root/autodl-tmp/models/X` → `X`). Zero numeric differences in any quarantined file
  (`RG_gain_law_20260715`: 4 entries differing on `model` only; `RG_admission_benefit_20260715`:
  0; `RG_crossterm_alignment_20260715`: 19 on `model`/`rg_dir` only;
  `RG_map_evidence_20260716`: 4 on `model` only; the two Phi operating tables and
  `RG_matched_dose_spread_20260716`: byte-identical to their repo originals).
- Deposit code carries no internal identifiers: a scan of every `.py`/`.R`/`.sh` in the
  deposit for `D2`, `B6`, `PLAN-GAP-CLOSURE`, `H9`, `paper-arr`, `d2-federation`, and
  `submissions/` returns zero hits. The newly shipped `rg_matched_dose_spread.py` needed
  two such strings stripped from its header (see item 4).
- All 11 deposit experiment scripts byte-compile; `make_figures.R` parses under `Rscript`.
  `__pycache__` directories created during that check were removed.
- No `.tex` file was modified. The manuscript, its macros, and the shipped figure renders
  are untouched, so the deposit still corresponds exactly to the submitted PDF.

### Stale-name grep (final)

Scanning `.py`, `.R`, `.md`, `.sh`, `.cff`, `.tex`, `Makefile` under `zenodo-deposit/`,
excluding `.PHI-PREFIX-STALE` files, and counting a name as live only when it is not
immediately followed by `.PHI-PREFIX-STALE`: **zero live references** to the eight
superseded filenames. The surviving occurrences are all documentation of the rename
itself — the per-script rename-map headers, `supersedes:` provenance fields, the README
rename table and supersession notes, and the frozen
`prereg/LEDGER-PREREG-2026-07-16.md` record.

## Complete file manifest of this repair

### New files

| Path | Note |
|---|---|
| `edit-harness/results/merging/RG_map_evidence_REFIX20260801.json` | canonical repo copy |
| `edit-harness/results/merging/RG_signed_reanalysis_REFIX20260801.json` | canonical repo copy |
| `zenodo-deposit/results/merging/RG_map_evidence_REFIX20260801.json` | shipped; 22/22 cells byte-identical to repo copy |
| `zenodo-deposit/results/merging/RG_signed_reanalysis_REFIX20260801.json` | shipped |
| `zenodo-deposit/results/merging/RG_crossterm_alignment_ALL_REFIX20260801.json` | shipped; figB's single source, all 19 bundles |
| `zenodo-deposit/code/experiments/rg_matched_dose_spread.py` | the omitted generator (audit D6) |
| `submissions/d2-neurocomputing/DEPOSIT-REPAIR-LOG-20260801.md` | this log |

### Renamed (quarantined; contents untouched, nothing deleted)

| Before | After |
|---|---|
| `zenodo-deposit/results/merging/RG_map_evidence_20260716.json` | `...json.PHI-PREFIX-STALE` |
| `zenodo-deposit/results/merging/RG_operating_curve_table_phi35_L16.json` | `...json.PHI-PREFIX-STALE` |
| `zenodo-deposit/results/merging/RG_operating_curve_table_phi35_L24.json` | `...json.PHI-PREFIX-STALE` |
| `zenodo-deposit/results/merging/RG_signed_reanalysis_20260715.json` | `...json.PHI-PREFIX-STALE` |

### Modified — repo originals (only where the audit's fix required it)

| Path | Why |
|---|---|
| `edit-harness/experiments/rg_map_evidence_consolidate.py` | audit D2: explicit CLI, REFIX defaults, deterministic duplicate resolution, portable recorded paths |
| `edit-harness/experiments/rg_signed_reanalysis.py` | audit D4: `--refix` / explicit table paths; 2026-07-15 defaults preserved |

### Modified — deposit copies

| Path | Why |
|---|---|
| `zenodo-deposit/README.md` | artifact-map rows + rename-map block + rerun instructions |
| `zenodo-deposit/figures/make_figures.R` | all five figure sources repointed; rename-map header; figA count |
| `zenodo-deposit/code/experiments/rg_gain_law.py` | output name |
| `zenodo-deposit/code/experiments/rg_gain_holdout.py` | gain-law input + provenance |
| `zenodo-deposit/code/experiments/rg_admission_benefit.py` | gain-law input + output name |
| `zenodo-deposit/code/experiments/rg_crossterm_alignment.py` | output name + `--bundles` + layout |
| `zenodo-deposit/code/experiments/rg_map_evidence_consolidate.py` | synced from repo + anonymized + layout |
| `zenodo-deposit/code/experiments/rg_signed_reanalysis.py` | synced from repo + layout |
| `zenodo-deposit/code/experiments/rg_matched_dose_spread.py` | new; anonymized header, optional addendum cell, layout |

### Not modified

- Any `.PHI-PREFIX-STALE` file (all seven).
- Any `.tex` file, `macros.tex`, `main.pdf`, or the shipped `figures/fig{A..E}` renders.
- `prereg/*` (see D7).
- `LICENSE`, `CITATION.cff` (see D8).
- `zenodo-deposit.tar.gz` and `SHA256SUMS` — both are now **stale** relative to the
  repaired tree and must be regenerated before deposition. Flagged for the user; not done
  here because the tarball is 2.7 GB and repacking is a deposit-time step that should
  follow the reviewer's re-audit.

## Follow-ups for the user

1. **D7** — decide the prereg ratification-status question (needs the underlying pre-run
   record; do not back-date).
2. **D8** — at deposition, add `version`, `date-released`, `repository-code`, and the
   minted Zenodo DOI to `CITATION.cff`, then run a CFF validator.
3. **Repack** `zenodo-deposit.tar.gz` and regenerate `SHA256SUMS` after the re-audit
   clears.
4. **Optional, cosmetic** — `macros.tex` lines 16, 18, 19, 49, 53, 58 still name the
   pre-refix artifacts in `% SOURCE` comments. All the values are verified correct (see
   "No published number moves"); only the annotations lag. Manuscript edits were out of
   scope here.
5. **Optional** — decide whether to re-render `figures/fig{A..E}` from the refixed
   artifacts. Currently they are deliberately the 2026-07-16 build so the deposit matches
   the submitted PDF exactly; re-rendering would move the Phi-3.5 series and desynchronize
   the deposit from the manuscript figures unless the manuscript is rebuilt too (which
   this repair was scoped not to touch).


