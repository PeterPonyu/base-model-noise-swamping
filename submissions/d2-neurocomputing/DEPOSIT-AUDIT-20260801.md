# D2 Zenodo Deposit Self-Containment Audit — 2026-08-01

## Scope and method

Audited tree: `submissions/d2-neurocomputing/zenodo-deposit/`.

The audit treated `results/merging/` as the base for unqualified artifact names in the README map and `results/merging_editors/` as the intentional sibling reached by the two `../merging_editors/` references. A row passes only when:

1. every mapped script exists in the deposit;
2. every mapped artifact or wildcard has at least one in-deposit match;
3. no reference escapes the deposit, except the documented `../merging_editors/` sibling traversal, which must resolve back inside the deposit;
4. the shipped script reads the refixed inputs and writes the mapped artifact name, rather than an absent or quarantined pre-refix name.

Static checks were used; the deposit was not modified and no figure render was run. Local R is available at `/usr/bin/Rscript`.

## Per-row verdicts

| README row | Paper element | Script check | Artifact check | Boundary check | Refix/rebuild check | Verdict |
|---:|---|---|---|---|---|---|
| 94 | Operating-map table | `code/experiments/rg_map_evidence_consolidate.py` and `code/experiments/merging_m0.py` exist | 22 `RG_operating_curve_table_*.json` matches, 22 `*_L*_RG/` bundles, and `RG_map_evidence_20260716.json` exist | All inside deposit | Consolidator line 25 reads absent `results/merging/RG_gain_law_20260715.json`; shipped map evidence still records stale Phi source tables and stale values. The wildcard also includes both unsuffixed stale Phi tables and REFIX tables. | **DEFECT** |
| 95 | Gate-evidence figE and dose-response figD | `figures/make_figures.R` exists | `RG_map_evidence_20260716.json` exists | Inside deposit | R line 153 reads the shipped map evidence for figD; figE reuses that loaded object from its line-195 section. The aggregate contains pre-refix Phi values and points to unsuffixed Phi tables. | **DEFECT** |
| 96 | Gain-screen figA | `figures/make_figures.R` and `code/experiments/rg_gain_law.py` exist | Both mapped REFIX gain law and regenerated holdout JSON exist | Inside deposit | R line 12 reads absent `RG_gain_law_20260715.json`; `rg_gain_law.py` line 73 also writes the old unsuffixed name instead of the mapped REFIX name. | **DEFECT** |
| 97 | g-resolved cross-talk figB | R and `code/experiments/rg_crossterm_alignment.py` exist | Both mapped REFIX JSONs exist | Inside deposit | R lines 12 and 15 read absent old gain/alignment names; the alignment script line 91 writes the old alignment filename. | **DEFECT** |
| 98 | Gain-screen table | `code/experiments/rg_gain_law.py` exists | `RG_gain_law_MERGED_REFIX20260730.json` exists | Inside deposit | Script line 73 writes `RG_gain_law_20260715.json`, not the mapped REFIX artifact. | **DEFECT** |
| 99 | Ordering Spearman and g=2 fractions | `rg_gain_law.py` and `rg_gain_holdout.py` exist | All three mapped JSONs exist | Inside deposit | `rg_gain_holdout.py` line 78 reads absent `RG_gain_law_20260715.json`; line 135 retains that stale provenance. | **DEFECT** |
| 100 | Ordering permutation null | `code/experiments/rg_permutation_null.py` exists | Aggregate JSON and `perm_null_allcells/` exist | Inside deposit | The two deposited Phi null files exactly match the canonical refixed harness copies byte-for-byte. | **PASS** |
| 101 | Editor/dataset generality | `merging_editors.py` and `rg_gain_law_editors.py` exist | 12 `*_RG/RG_editors_table.json` matches plus `RG_gain_law_editors_20260716.json` | Intentional `../merging_editors/` traversal resolves to `zenodo-deposit/results/merging_editors/`, inside deposit | No Phi-refixed dependency found in this row. | **PASS** |
| 102 | Admission-benefit table and figC | `rg_admission_benefit.py` and R exist | `RG_admission_benefit_REFIX20260730.json` exists | Inside deposit | R line 116 reads absent `RG_admission_benefit_20260715.json`; analysis line 75 reads absent old gain law and line 71 writes the old admission filename. | **DEFECT** |
| 103 | Signed re-analysis | `code/experiments/rg_signed_reanalysis.py` exists | `RG_signed_reanalysis_20260715.json` exists | Inside deposit | Script lines 171–173 explicitly read unsuffixed pre-refix Phi L24/L16 operating tables. The mapped aggregate contains Phi entries but no REFIX provenance and was not replaced/quarantined. | **DEFECT** |
| 104 | Kill-gate origin M0 | `code/experiments/merging_m0.py` exists | `M0_killgate_table.json` and 66 `*_s{0,1,2}/` directories exist | Inside deposit | No superseded consolidated Phi filename is named or consumed by this mapped row. | **PASS** |
| 105 | Cross-term/value-direction alignment | `code/experiments/rg_crossterm_alignment.py` exists | `RG_crossterm_alignment_phi35_REFIX20260730.json` exists | Inside deposit | Script line 91 writes the absent old alignment filename, not the mapped REFIX filename. | **DEFECT** |
| 106 | Matched-dose spread | Mapped `code/experiments/rg_matched_dose_spread.py` is **missing** from the deposit | `RG_matched_dose_spread_REFIX20260731.json` exists and parses | Inside deposit | Canonical generator exists only outside the deposit at `edit-harness/experiments/rg_matched_dose_spread.py`. | **DEFECT** |

Summary: **3 PASS, 10 DEFECT rows**. All mapped result artifacts exist, and no unauthorized map reference resolves outside the deposit. The failure is reproducibility/routing, not raw artifact absence, except for the missing matched-dose generator.

## Defect list

### D1 — Figure pipeline hard-codes absent pre-refix files

Exact file: `submissions/d2-neurocomputing/zenodo-deposit/figures/make_figures.R`.

- Lines 2, 12: `RG_gain_law_20260715.json`; only `RG_gain_law_20260715.json.PHI-PREFIX-STALE` and `RG_gain_law_MERGED_REFIX20260730.json` are shipped.
- Lines 3, 15: `RG_crossterm_alignment_20260715.json`; only the `.PHI-PREFIX-STALE` file and `RG_crossterm_alignment_phi35_REFIX20260730.json` are shipped.
- Lines 115–116: `RG_admission_benefit_20260715.json`; only the `.PHI-PREFIX-STALE` file and `RG_admission_benefit_REFIX20260730.json` are shipped.

Impact: `Rscript make_figures.R` fails with missing files. If old names are restored manually, figs A–C silently use pre-refix data. Figs D–E read a stale aggregate (D2).

Recommended fix: point figs A/B/C to the mapped REFIX filenames; rebuild the map aggregate from refixed inputs before using it for figs D/E; update source comments in the R file to the same names.

### D2 — Operating-map aggregate and consolidator remain pre-refix

Exact files:

- `submissions/d2-neurocomputing/zenodo-deposit/code/experiments/rg_map_evidence_consolidate.py`
- `submissions/d2-neurocomputing/zenodo-deposit/results/merging/RG_map_evidence_20260716.json`

The consolidator reads absent `RG_gain_law_20260715.json` at line 25 and records it at line 92. Its broad line-34 wildcard admits both unsuffixed and REFIX Phi operating tables. The shipped aggregate proves it predates the refresh: its Phi cells name `RG_operating_curve_table_phi35_L16.json` and `...L24.json` and contain old values, e.g. L16 g=2 s0 median absolute drop `0.5576` instead of refixed `0.2611`, and L24 g=2 s0 partial rho `-0.1011` instead of refixed `-0.2745`.

Recommended fix: read `RG_gain_law_MERGED_REFIX20260730.json`; explicitly exclude unsuffixed superseded Phi tables or select REFIX/bundle tables deterministically; regenerate `RG_map_evidence_20260716.json` under a REFIX-qualified name; update README rows 94–95 and R inputs.

### D3 — Shipped analysis scripts do not reproduce mapped REFIX names

Exact stale pointers:

- `code/experiments/rg_gain_law.py:73` writes `RG_gain_law_20260715.json`.
- `code/experiments/rg_gain_holdout.py:78,135` reads/records `RG_gain_law_20260715.json`.
- `code/experiments/rg_admission_benefit.py:12,71,75` reads old gain law and writes old admission name.
- `code/experiments/rg_crossterm_alignment.py:91` writes old alignment name.

Recommended fix: update defaults and provenance to the REFIX filenames, or add explicit command-line input/output arguments and show exact REFIX rebuild commands in README.

### D4 — Signed re-analysis still consumes stale Phi tables

Exact file: `submissions/d2-neurocomputing/zenodo-deposit/code/experiments/rg_signed_reanalysis.py:171-173`.

It reads `RG_operating_curve_table_phi35_L24.json` and `...L16.json`, which materially differ from both the REFIX tables and the refreshed per-bundle tables (188 and 184 leaf-value differences, respectively). `RG_signed_reanalysis_20260715.json` still contains Phi rows without REFIX provenance.

Recommended fix: route to `RG_operating_curve_table_phi35_L{16,24}_REFIX20260730.json` or the refreshed bundle-local tables; regenerate the signed aggregate; quarantine the pre-refix aggregate if retained.

### D5 — Two stale Phi operating tables were not quarantined and are matched by the map wildcard

Exact files:

- `results/merging/RG_operating_curve_table_phi35_L16.json`
- `results/merging/RG_operating_curve_table_phi35_L24.json`

Unlike the four consolidated stale files, these retain ordinary `.json` names. README row 94's `RG_operating_curve_table_*.json` therefore matches stale and REFIX versions simultaneously.

Recommended fix: rename the two unsuffixed files with `.PHI-PREFIX-STALE`, narrow the map to canonical/refixed tables, and make consolidation reject duplicate `(model, layer)` sources.

### D6 — Matched-dose generator omitted

Mapped path `submissions/d2-neurocomputing/zenodo-deposit/code/experiments/rg_matched_dose_spread.py` does not exist. The source exists outside the deposit at `edit-harness/experiments/rg_matched_dose_spread.py`, which violates self-containment for README row 106.

Recommended fix: copy the reviewed generator into `code/experiments/`, then verify its defaults consume refreshed bundle data and produce `RG_matched_dose_spread_REFIX20260731.json` (or document an explicit output argument).

### D7 — Preregistration copies lack explicit ratification status

Present files:

- `prereg/LEDGER-PREREG-2026-07-16.md`
- `prereg/PREDICTIONS-GAIN-WAVE-2026-07-15.md`
- `prereg/PREREG-FED-EDITORS-2026-07-16.md`
- `prereg/PREREG-RG-DEPTH-2026-07-12.md`
- `prereg/PREREG-WIDTH-RG-20260714.md`

All five are present and describe frozen/pre-run designs, but **none contains `STATUS: RATIFIED`**; none contains an active `STATUS: DRAFT` marker either. Therefore the requested ratification-status check cannot pass from the deposit copies alone.

Recommended fix: add an explicit, historically supported status line to each deposit copy, or add a signed/dated ratification manifest linking each frozen document to its pre-run timestamp. Do not retroactively assert ratification without the underlying record.

### D8 — `CITATION.cff` is identity-current but release metadata is incomplete

`CITATION.cff` exists, parses as plain YAML structure, names the current D2 paper, author, MIT code license, and split CC BY 4.0 data/prereg terms. `LICENSE` exists and consistently spells out MIT for code/figures plus CC BY 4.0 for results/prereg. Neither is stale with respect to the project identity or licensing.

However, `CITATION.cff` has no `version`, `date-released`, `repository-code`, or DOI/identifier, and still says "DOI assigned at deposit." This is acceptable before DOI minting but not complete final-release metadata.

Recommended fix: at deposition, add version/date and the minted Zenodo DOI (plus repository URL if public), then validate with a CFF validator. No license text change is indicated.

## Refix and parity checks

Four quarantined consolidated files and their replacements are present:

| Superseded file | Replacement |
|---|---|
| `RG_gain_law_20260715.json.PHI-PREFIX-STALE` | `RG_gain_law_MERGED_REFIX20260730.json` |
| `RG_admission_benefit_20260715.json.PHI-PREFIX-STALE` | `RG_admission_benefit_REFIX20260730.json` |
| `RG_crossterm_alignment_20260715.json.PHI-PREFIX-STALE` | `RG_crossterm_alignment_phi35_REFIX20260730.json` |
| `RG_matched_dose_spread_20260716.json.PHI-PREFIX-STALE` | `RG_matched_dose_spread_REFIX20260731.json` |

All replacement JSONs parse. README rows that name these consolidated results use REFIX names or explicitly document supersession. The defect is that the shipped scripts and figure pipeline do not follow the refreshed map.

Phi permutation-null parity against `edit-harness/results/merging/perm_null_allcells/`:

| File | Byte equality | Parsed-content equality |
|---|---:|---:|
| `Phi-3.5-mini_L16_RG.json` | yes | yes |
| `Phi-3.5-mini_L24_RG.json` | yes | yes |

## Final verdict

**DEFECTS-8 — NOT SELF-CONTAINED.**

All named artifacts are present, the intentional editor sibling paths exist inside the deposit, and the refixed Phi permutation-null files are canonical. The deposit nevertheless cannot reproduce its own refixed tables/figures because its R pipeline and several analysis scripts retain pre-refix filenames, one stale aggregate is still mapped into figures, two stale Phi operating tables remain unquarantined, one mapped generator is missing, and prereg copies lack explicit ratification status.
