# IEEE workspace — build, venue toggle, figures

This is the IEEE journal (TNNLS/TASLP) extension package for the B6 paper, built
from the same shared truth as `../../paper-arr/` (ARR/EMNLP submission). See
`../README.md` for the cross-venue rules; this file covers only IEEE-specific
build mechanics.

`paper-arr/` and its `figures-r/` are READ-ONLY sources for this workspace:
copy in, never edit out. `macros.tex` and `references.bib` here are copies of
the ARR originals (macros.tex byte-identical incl. provenance comments;
references.bib append-only for new journal-related-work entries). A future
number refresh always lands in `paper-arr/macros.tex` first and is re-copied
here — never edited independently in this workspace.

## Build

Standard latexmk / pdflatex+bibtex+pdflatex×2 cycle. IEEEtran.cls and
IEEEtran.bst/IEEEtranN.bst are present in this machine's TeX Live tree
(`texlive/texmf-dist/tex/latex/ieeetran/`, `.../bibtex/bst/ieeetran/`) — no
network fetch needed.

```
cd submissions/ieee
latexmk -pdf main.tex
# or manually:
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

`main.tex` is written to compile standalone at any point during authoring:
every section file exists (either real content or a one-line `\section` stub
with a `% STUB — awaiting authoring pass` comment naming its ARR source), so a
partially-authored package always produces a PDF, never a missing-`\input`
error. Check `sections/*.tex` for any file whose first content line still
starts `% STUB` — those are the ones not yet ported.

## Bibliography style — the natbib/IEEEtran fork

The ARR paper (`paper-arr/main.tex`) uses `natbib` (`\citep`/`\citet`) with
`acl_natbib.bst`. This package needs `\citep`/`\citet` calls in ported text to
keep working while satisfying the requirement to build the bibliography with
`\bibliographystyle{IEEEtran}`. `IEEEtran.bst` is a plain numeric bst, not a
natbib-aware one, so `main.tex` does **not** load `natbib`; instead it aliases
`\citep`/`\citet` to plain `\cite` (`\providecommand`, so it is a no-op if a
section ever loads natbib itself). This is the "fallback: rewrite to `\cite`
with plain `IEEEtran.bst`" path named in `tnnls/EXTENSION-PLAN.md`.

If author-year-flavoured behaviour with sorting/compression is wanted instead
(matching the ARR build's citation rendering more closely), switch to the
natbib-native fork:

```latex
\usepackage[numbers,sort&compress]{natbib}
% remove the two \providecommand lines for \citep/\citet
```

and change `\bibliographystyle{IEEEtran}` → `\bibliographystyle{IEEEtranN}`
(also present locally, at
`texlive/texmf-dist/bibtex/bst/ieeetran/IEEEtranN.bst`). Both forks compile
against the same `references.bib`; only the two lines above and the
bibliographystyle line change.

## Venue toggle: TNNLS vs. TASLP

Two `\newif` toggles near the top of `main.tex` select the venue; **exactly
one must be true** (a `\PackageError` fires at compile time if both or
neither are set):

```latex
\tnnlstrue   % active
\taslpfalse
```

Flip to target TASLP instead:

```latex
\tnnlsfalse
\taslptrue
```

What differs between the two builds (all inside existing `\iftnnls...\fi` /
`\iftaslp...\fi` blocks in the section files — there are no forked files):

| | TNNLS | TASLP |
|---|---|---|
| Abstract / intro framing | (nothing extra) | extra language-technology-maintenance framing paragraph in `00_abstract.tex` and `01_intro.tex` |
| STUB-THEOREM (`sections/03_method.tex`) | **mandatory** `proposition` environment — TNNLS's standing rule is nothing ships without a theorem-bearing artifact | **optional** — same content may render as a `remark` instead |
| 8B-causal stub (`sections/05_causal.tex`) | kept prominent (TNNLS wants the scale-evidence chain closed) | present but not emphasized |
| Masthead line | "IEEE Transactions on Neural Networks and Learning Systems" | "IEEE/ACM Transactions on Audio, Speech, and Language Processing" |

The venue decision itself is still open — see `VENUE-NOTE.md`.

## Extension items (not currently in the manuscript body)

Five items back the >=30%-new-material extension-disclosure requirement
(STUB-THEOREM, STUB-EDITOR6, STUB-EGLSEEDS, STUB-DATASET, STUB-8BCAUSAL).
They used to render as bordered gray "EXTENSION (not in the conference
version)" boxes via an in-document `\extstub{label}{design text}` macro with
a `\finalbuildtrue` hard-error gate; both were removed 2026-07-05 so the
compiled PDF carries no visible placeholder/"not yet run" content —
see `EXTENSION-TODO.md` for the full verbatim text of each item, what it
would take to close, and where it used to live in the section files.

**None of the five items are done.** As of this draft, `main.tex` has no
measurable delta versus the ARR conference paper — it is not yet a valid
journal-extension submission under any of TNNLS/KBS/TASLP's disclosure
policy. Closing an item means actually running the experiment / writing the
proof and putting the real result back into the relevant section as
reviewed prose (removing that item's entry from `EXTENSION-TODO.md`), not
re-adding a placeholder box.

## Page budget

Target: **14 pages total** (~13.0pp body + ~1.0pp references; `IEEEtran.bst`
typically renders ~45–55 entries in about a page in double-column). Check the
running page count with `main.log` / `texcount`:

```
texcount -inc main.tex
```

If the body is running long, the two-column IEEEtran journal layout has
materially more room than the ACL two-column page than the 8pp ARR version —
prefer trimming prose in `11_discussion.tex` (the merged Limitations content)
before cutting any of the promoted tables/figures (EGL, generality,
anisotropy, sequential), which the plan explicitly promotes to the body.

## Figure regeneration

Figures are TeX-native pgfplots, ported byte-identical from
`paper-arr/figures-tex/` (figure wrapper + caption) and `paper-arr/figures-r/`
(the actual `axis`/`groupplot` environments with inline coordinate data,
audited against `edit-harness/results/*.json`). This workspace has its own
copies under `figures-tex/` and `figures-r/` so figure work here never
touches the ARR package.

- **Do not hand-edit coordinate data** in `figures-r/*.tex` — it is audited
  against canonical JSON. Only the width/height/float-wrapper lines (the
  `\columnwidth` vs. `\textwidth` sizing, `figure` vs. `figure*`) should
  change when retuning for the IEEE column.
- The plan's one required figure change: **de-merge** `figA_lawtransfer.tex`
  (which the ACL build consolidated from two source figures, F1+F7, to save
  space) back into two separate floats — a layer-law figure and a scale-law
  figure — inside `sections/04_regime.tex`. Reuse the existing coordinate
  blocks in `figures-r/figA.tex` unchanged; split only the `\begin{figure*}`
  wrapper and caption in `figures-tex/`, and if the underlying `figures-r`
  file itself needs splitting into two `axis` blocks, copy the relevant
  `groupplot`/`axis` sub-blocks apart rather than re-deriving them.
- If the canonical numbers in `paper-arr/results/*.json` are ever refreshed,
  regenerate `paper-arr/figures-tex/` and `paper-arr/figures-r/` first (via
  `edit-harness/experiments/make_figures_tex.py` / `figures-r/make_figures.R`
  in the ARR workspace), then re-copy the changed files into this workspace's
  `figures-tex/`/`figures-r/` — never regenerate independently here.

## Known-good local TeX Live packages used

`IEEEtran.cls`, `IEEEtran.bst`, `IEEEtranN.bst`, `natbib`, `xcolor`,
`pgfplots` (+ `groupplots`, `patterns` tikz libraries), `amsthm`, `booktabs`,
`multirow` — all confirmed present via `kpsewhich` in this environment; no
network fetch required to build.
