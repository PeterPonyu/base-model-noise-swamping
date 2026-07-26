# paper-arr — build setup

ARR / EMNLP-main LaTeX package for the B6 paper. **Authoring pass** — this
package has NOT been through the hostile submission review that gates
submission.

## Style files (fetch manually — nothing is downloaded automatically)

The official ACL style files are **not vendored** in this repo. Before a real
submission build, place these in `paper-arr/` (from the ACL styles repo,
<https://github.com/acl-org/acl-style-files>):

- `acl.sty`
- `acl_natbib.bst`
- `acldoc.sty` (only if you build the ACL doc; not required for the paper)

`main.tex` loads `acl.sty` **only if present**. If it is absent, it falls back
to a plain `article` layout (Times, 1-inch margins, natbib) so the source
still compiles for local drafting. **The fallback is for drafting only** — for
the submission, drop the fallback branch and use `\usepackage[review]{acl}`
(anonymous/line-numbered) or no option for camera-ready.

## Build

```bash
cd paper-arr
latexmk -pdf main.tex        # or: pdflatex main && bibtex main && pdflatex main x2
```

Figures are referenced from `../edit-harness/figures/F*.pdf` via
`\graphicspath`; they are not copied into `paper-arr/`.

## Layout

- `main.tex` — document shell, package loads, section `\input`s, bibliography.
- `macros.tex` — every quoted number as a `\newcommand`, named by source, with
  the canonical `edit-harness/results/*.json` path in a comment. **All number
  edits happen here** (single-point updates for run_u5 seed refreshes).
- `sections/*.tex` — one file per section.
- `references.bib` — citations (verify final metadata before camera-ready).

## Value provenance

Every quoted value is a `\newcommand` in `macros.tex` with a comment naming its
canonical JSON under `edit-harness/results/`. All run_u5 seed updates are
folded; remaining low-seed cells (MEMIT L10/L14, the L12 KL-ladder,
zsRE-deletion, Llama-3B L14, the generic projector; 2-seed EGL; 2-stream
sequential flank) are flagged as such in-text and in the Limitations section.

## Binding wording (do not paraphrase — review-gated)

- **MEMIT:** quote `rho_C` only, never "MEMIT S×C"; 3-seed C3 means; DEAD
  (<0.10) vs ROME 0.41/0.60. The exact approved sentence is in
  `sections/06_dissociation.tex`.
- **Sequential:** descriptive only; ordering-dependent 10–42% range; no ρ≈0.55;
  no geometry attribution.
- **S×C:** zero-cost GradSim surrogate, never "beats key-cosine" (it loses at
  L8/L10).
- **Scope:** signed law is Llama-family-specific (abstract + intro); magnitude
  law attenuates at 8B.
- **suppress (U1):** two distinct statistics (gate S×C vs raw within-probe),
  never averaged or swapped.
