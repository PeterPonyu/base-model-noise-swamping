# IEEE journal figures — R/ggplot2/tikzDevice set

`make_figures_ieee.R` is a **copy of** `../../../paper-arr/figures-r/make_figures.R`
(the ACL-submission R figure generator), adapted for `IEEEtran` journal (`[journal]`,
double-column) geometry. Run with:

```
Rscript submissions/ieee/figures/make_figures_ieee.R
```

which regenerates `fig{A,B,C,D,E,F}.tex` in this directory.

## Relationship to the ACL version

- **Source of truth is shared.** Every plotted number is read at run time from the
  same canonical JSONs under `../../../edit-harness/results/*.json` — nothing is
  retyped, hardcoded, or recomputed. The full `% SOURCE` provenance block (one line
  per plotted series, citing the exact JSON path + field + value) is emitted at the
  top of each `.tex` file, unchanged in form and content from the ACL version.
- **Data logic, wording, and figure content are identical** to
  `../../../paper-arr/figures-r/make_figures.R`. All plotting functions (`fig_A()`
  through `fig_F()`), the palette, the shared `theme_b6()` theme, and the `emit()`
  routine are byte-identical between the two scripts. A `diff` of the two files shows
  only:
  1. the output-directory resolution (this copy lives one directory level deeper —
     `submissions/ieee/figures/` vs. `figures-r/` — so the relative lookup of
     `edit-harness/results/` has one extra `..`; it resolves to the same directory),
  2. the per-figure `emit()` width/height arguments (see below), and
  3. header/usage comments updated to describe this file's location.
- **No new numbers, no rewording.** This script does not re-derive or alter any of
  the MEMIT-rho_C / signed-law / magnitude / sequential wording constraints that
  govern the paper text — it only lays out the same six figures at IEEE dimensions.

## Dimensions (IEEEtran `journal` class, double column)

| Figure | ACL (paper-arr) | IEEE (this dir) | Column type |
|---|---|---|---|
| figA | 6.30 x 3.20 in | **7.16 x 3.60 in** | full-width (`figure*`, `\textwidth`) |
| figB | 3.03 x 2.20 in | **3.50 x 2.50 in** | single-column (`figure`, `\columnwidth`) |
| figC | 3.03 x 2.00 in | **3.50 x 2.30 in** | single-column |
| figD | 6.30 x 2.00 in | **7.16 x 2.30 in** | full-width |
| figE | 6.30 x 2.70 in | **7.16 x 3.00 in** | full-width |
| figF | 3.03 x 2.00 in | **3.50 x 2.30 in** | single-column |

IEEEtran journal geometry: `\columnwidth` = 3.5in, `\textwidth` = 7.16in. Heights are
scaled up from the ACL (8-page-budget-constrained) versions since the IEEE target
(14 pages total) has more room to breathe.

## Verification performed

- All six `fig{A..F}.tex` files render with bounding boxes matching the requested
  widths/heights (checked via the emitted `\path[use as bounding box, ...] (0,0)
  rectangle (W,H)` line in each file; tikzDevice's internal unit is the TeX point,
  1in = 72.27pt).
- Ran twice (`Rscript make_figures_ieee.R` back to back); all six output files are
  byte-identical across runs (verified via `diff` and `md5sum`) — deterministic, no
  embedded timestamps (the tikzDevice `% Created by tikzDevice` line is stripped by
  `emit()`, same as the ACL version).
- Confirmed no internal filenames (`*.json`/`*.npz`/`*.py`), run/queue codenames
  (`run_u*`, `VALIDATE`, `hostile`, `authoring-pass`), or section codenames
  (`B6`/`E6`/`D3`/`H1`) appear anywhere except inside non-rendered `% SOURCE`
  provenance comments — matching the ACL figures' convention.

## Do not edit in place

Source of truth for the plotting/data logic is
`../../../paper-arr/figures-r/make_figures.R` (READ-ONLY from this workspace). If the
canonical JSONs are refreshed, re-run this script to regenerate; do not hand-edit the
emitted `.tex` files or hand-edit data values in `make_figures_ieee.R`.
