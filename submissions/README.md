# Submissions — per-venue workspaces (created 2026-07-05)

> One paper core, multiple potential venues, target NOT yet decided. Each venue gets its
> own workspace so drafts never cross-contaminate. SHARED TRUTH lives outside these dirs:
> canonical numbers = `../edit-harness/results/*.json`; figures = `../edit-harness/figures/`
> (regenerate via `experiments/make_figures.py`); binding wording = the three
> `../findings-*.md` docs; number macros pattern = `../paper-arr/macros.tex`.
> RULE: a venue workspace may COPY from shared truth, never the reverse; any new number
> enters shared truth (canonical JSON + review gate) first.

## Workspaces
- `../paper-arr/` — ARR → CCF-B main (EMNLP/NAACL/ACL 2027 via commitment). ACL two-column,
  8 pages. STATUS: **SUBMISSION-READY** (2026-07-05): real acl.sty [review] build, 10pp,
  R/ggplot2 figures (figF in appendix), leak-swept, all reviews passed; form draft at
  ../paper-arr/SUBMISSION-FORM-DRAFT.md; user read-through + OpenReview submission remain. Next ARR deadline: **2026-08-03** (cycle commits from Oct 11;
  note the cycle's listed venue EACL 2027 FAILS the SCIE/CCF filter — commit later to a
  CCF-B venue when its window opens).
- `ieee/` — **BUILT (2026-07-05, sonnet workflow)**: IEEEtran journal package, double-column,
  13pp compiled (target ~14 incl. extension work). Venue fork `\iftnnls`/`\iftaslp` in
  main.tex; 6 disclosed EXTENSION stubs (theorem, 8B causal, 6th editor, EGL seeds,
  dataset breadth); macros byte-identical to paper-arr; IEEE-width R figures. Leak-swept
  (verifier caught 6 caption filename leaks — fixed + re-verified 0). See `ieee/SETUP.md`
  + `ieee/VENUE-NOTE.md`. Authors = placeholder (journals not anonymous — fill at venue
  decision).
- `kbs/` — Knowledge-Based Systems (Elsevier, SCIE Q1). elsarticle format, no page limit,
  ROLLING submission. See `kbs/EXTENSION-PLAN.md` for the gap list.
- `tnnls/` — IEEE TNNLS (SCIE Q1, CCF-B journal). IEEEtran format, rolling. Higher bar:
  theory/artifact expectation. See `tnnls/EXTENSION-PLAN.md`.

## Standing constraints (from CLAUDE.md / venue strategy — apply to ALL venues)
- SCIE-indexed / CCF-ranked ONLY. TMLR, BlackboxNLP, ICBINB, COLM, EACL fail the filter.
- Dual-submission rules: the SAME content cannot be under review at a journal and ARR
  simultaneously. The sanctioned path (VENUE-GAP-ANALYSIS 2026-07-01/02): ARR first,
  journal EXTENSION after with substantial new material (norm: ≥30% new content +
  disclosure of the conference version). A journal-FIRST path is possible instead —
  see the timing comparison in the extension plans — but forfeits the Aug-3 ARR cycle.
- Author/review separate passes; every quoted number verified against canonical JSON.
