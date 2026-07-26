# Extension items removed from the submission manuscript

This doc holds the verbatim planning text that used to render as visible
`\extstub{...}{...}` gray boxes inside `main.pdf`. It was pulled out of the
manuscript body (2026-07-05) so the compiled PDF carries zero placeholder /
"not yet run" content — a real submission asset should not visibly confess
to unfinished work inside the document itself.

**Read this before claiming the manuscript is submission-ready.** None of
the five items below have been completed. Per the standing >=30%-new-material
extension-disclosure requirement for a TNNLS/KBS/TASLP journal-extension
submission (see `VENUE-NOTE.md`), removing this content means the current
`main.tex` has **no measurable delta** versus the ARR conference draft it is
meant to extend — submitting it as-is under an extension-disclosure claim
would misrepresent the paper to the venue. Each item must either be
(a) actually completed and written back into the relevant section as real,
review-gated prose (removing its entry here), or (b) the venue plan
re-scoped to not depend on it, before this package is truly ready to send.

The old in-document mechanism (`\extstub` macro + `\finalbuildtrue` hard-error
gate in `main.tex`) has been removed along with the calls, since with no
stubs left in the body it no longer serves a purpose. If new extension work
is drafted before being fully done, consider re-adding a similar draft-only
gate rather than writing "not yet run" language directly into submission
prose.

---

## STUB-THEOREM — formal S×C-as-gradient-influence statement — CLOSED 2026-07-10

**Closed 2026-07-10.** The formal statement now lives in
`sections/03_method.tex`, subsection "Toward a formal statement"
(`\label{sec:formal}`), ported from the review-passed source
`docs/findings/THEOREM-SXC-DRAFT-2026-07-06.md`. Both venue variants ship: a
`proposition` environment (fuller statement + proof sketch) inside
`\iftnnls`, and a weaker `remark` environment inside `\iftaslp`. The claim is
the cautious "rank estimator" form, not the earlier "identical rather than
merely correlated" overclaim: Proposition 1's read-out reduction is an exact
algebraic identity, but the loss-level relationship is a proportionality with
an edit-varying coefficient, so \SxC{} is stated only as a conditional rank
estimator of first-order influence (under Assumption A4′), not a faithful
surrogate. A true-backprop influence cell now anchors this in the same
subsection: true influence predicts damage while its rank agreement with \SxC{}
is low, which is what motivates the conservative wording. The error-term
derivation for ROME's iterative multi-step value solve is included in the
proof (same rank-one factorization applied to the value-solve gap), but a
numerical bound on that gap for production (non-toy) edits is honestly stated
in the text as deferred to future work — not hidden. Two preamble theorem
environments (`\newtheorem{proposition}` / `\newtheorem{remark}`) are required
in `main.tex` for this to compile; flagged to the orchestrator separately.

---

## STUB-8BCAUSAL — AlphaEdit causal cell at Llama-8B — CLOSED 2026-07-10

**Closed 2026-07-10.** The AlphaEdit causal cell at Llama-8B was run (holdout
projector, matched ROME/AlphaEdit edit sets, known-fact and successful-edit
filters, key-cosine quartile binning; `C4_causal_8b_table.json`, single seed
s0, layers L16 and L24) and folded into `sections/05_causal.tex`, subsection
"Generality: architecture, scale, and instruction tuning"
(`\label{tab:causal-scale}`).

**Landed as a claim-tightening / attenuation result, NOT a confirmation.** The
held-out-projector within-probe rho(key-cos, damage-removed) is weak and
sign-inconsistent across depth (+0.19 at L16, -0.10 at L24), matching the
attenuation of the observational coupling at 8B. The section states explicitly
that the 8B cell is not read as a causal confirmation, and that the large,
positive-signed, monotone-removal form of the law is a small-to-mid-scale
result on current single-seed evidence. The 8B L16 quartile ratio (-68.96,
numerically unstable near the sign change) is deliberately not quoted. The
same subsection also folds the GPT-J-6B and GPT-NeoX-20B causal cells (both
single-seed, holdout projector) and the instruction-tuned 1B twin (clean
positive removal at base magnitudes), which supply the architecture/scale
generality context that frames the 8B boundary.

---

## STUB-EDITOR6 — memory/in-context editor family (GRACE/WISE-class)

**Removed from:** `sections/06_dissociation.tex`, subsection "Extension: a
sixth editor family".

> Adding a memory- or in-context-based editor family (of the class
> exemplified by GRACE- and WISE-style methods, which store edits in an
> auxiliary key-value memory or a codebook rather than in a rank-one or
> full-rank weight perturbation) would add a sixth row to both the
> locality/coupling spectrum table and the canonical edit-quality (EGL)
> table. Because this class of editor does not perturb W at all for
> unrelated probes by construction, the natural hypothesis is a coupling and
> mean damage near the AlphaEdit floor at the low end of the spectrum, but
> the interesting empirical question is whether measured collateral damage
> (via retrieval interference or representation drift, rather than a weight
> perturbation) shows any residual key-geometry structure at all, which
> would extend the closed-form account to a mechanism this paper does not
> currently cover. This is the single largest implementation lift among the
> extensions: it requires new editing code for the memory or codebook
> mechanism, integration with the existing gate-matrix measurement pipeline,
> and a fresh run across the layer band studied elsewhere in the section.

**To close:** implement GRACE- or WISE-style editing in the harness,
integrate with the gate-matrix measurement pipeline, run across the same
layer band as the other editors, add a sixth row to both the
locality/coupling spectrum and EGL tables. Largest implementation lift of
the five items.

---

## STUB-EGLSEEDS — full editor-by-seed EGL grid

**Removed from:** `sections/06_dissociation.tex`, subsection "Extension:
full editor-by-seed EGL grid".

> The canonical edit-quality table reports 2-seed means for three editors on
> one canonical layer. A complete editor-by-seed grid — the same
> Efficacy/Paraphrase/Neighborhood-specificity metrics at 3 or more seeds
> for every editor family in the locality/coupling spectrum table, including
> fine-tuning and KL-regularized fine-tuning, and including the sixth editor
> family described above if it is added — would let the edit-quality view
> carry the same seed-count guarantees as the damage-spectrum numbers
> already reported, and would let per-editor variance on
> neighborhood-specificity be reported directly rather than only as a
> caveat on the mean. This is a moderate additional compute cost relative to
> the existing 2-seed grid.

**To close:** re-run the Efficacy/Paraphrase/Neighborhood-specificity (EGL)
grid at 3+ seeds for every editor family already in the locality/coupling
spectrum table (FT, KL-FT, ROME, MEMIT, AlphaEdit, +STUB-EDITOR6 if done).
Moderate compute cost relative to the current 2-seed grid.

---

## STUB-DATASET (`ext:dataset`) — third evaluation dataset

**Removed from:** `sections/08_generality.tex`, subsection "Extension:
Broader Dataset Coverage".

> A third evaluation dataset, structurally different from both CounterFact
> and zsRE — for example an MQuAKE-class multi-hop or a temporal-reasoning
> benchmark — would extend the dataset-generality argument beyond
> single-hop factual rewrites and test whether the signed law and its
> magnitude counterpart transfer to multi-hop factual structure. Any
> acquisition of a new dataset for this purpose is subject to an ask-first
> download policy and would be executed and reviewed before any numbers
> derived from it enter this manuscript.

**To close:** pick and (ask-first) acquire a structurally different dataset
(e.g. MQuAKE-class multi-hop, or a temporal-reasoning benchmark), run the
signed + magnitude law replication on it, review-gate before any numbers
enter the manuscript.
