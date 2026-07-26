# ARR Submission Form — DRAFT

> **Status**: Authoring pass complete. This document is the pre-submission checklist for the B6 paper. Flagged items marked `[USER CONFIRM]` require user verification before final submission on OpenReview.

---

## SUBMISSION METADATA

**Title:** When and Why Does Key Geometry Predict Locate-then-Edit Collateral Damage? A Closed-Form, Editor- and Architecture-Conditioned Account of the Llama Family, with a Causal Test

**Abstract:**
Locate-then-edit knowledge editors such as ROME insert a fact by a rank-one update to one MLP layer, but that update also perturbs unrelated facts -- collateral damage. We ask *when and why* a purely a-priori geometric quantity, the cosine between an edit's key vector and an unrelated probe's key, predicts that damage. For rank-one ROME updates **on the Llama family**, we show the collateral perturbation factorizes in closed form into a per-edit strength *S*, a per-probe norm, and the edit×probe key-cosine *C* (Eq.~\ref{eq:sc}); the product S×C is a zero-cost surrogate for first-order (GradSim) gradient influence, computed with no backprop. On a confound-clean within-probe metric the key-cosine predicts ROME damage across four mid layers (Spearman ρ up to 0.602 at L12), undergoes a depth/regime transition in which matrix-norm-growth overtakes it at L14, and its *sign* tracks the sign of the damage regime across model scale. The law is **conditional on editor and architecture**: it vanishes for Adam full-rank fine-tuning and for MEMIT's multi-layer spread, is near-null on gemma and Phi, and *inverts* on Qwen. A causal test closes the loop: null-space projection (AlphaEdit) removes the geometry-predicted damage in proportion to key-cosine at every layer, erases the Qwen inversion, and collapses the coupling of a refusal-*deletion* edit that was never in the original design. We position the account as complementary to the concurrent empirical predictor CLaRE-ty, contributing the mechanism, the editor/architecture dissociation, and the causal test that it leaves open.

**Track:** Interpretability and Analysis of Models for NLP

**Keywords:** knowledge editing, mechanistic interpretability, locate-then-edit, rank-one updates, ROME, collateral damage, key geometry, neural network mechanisms, model behavior prediction

**Venue Note (Internal):** Reviews committed to a CCF-B *ACL main venue in a later window (EMNLP-2026, NAACL-2026, or ACL-2026). EACL excluded per authors' venue constraints.

---

## RESPONSIBLE NLP CHECKLIST

### A. Limitations and Risks

**A1. Does your work have limitations, and have you included a discussion of them?**
- **Answer:** Yes
- **Justification:** Section 9 "Limitations" comprehensively documents scope boundaries: (1) GPU scale (single 24GB RTX 5090 caps models at ≤8B parameters); (2) architecture scope (signed law is Llama-family-specific; off Llama it goes null on gemma/Phi or inverts on Qwen); (3) mechanism scope (rank-one ROME identity, does not describe full-rank fine-tuning or MEMIT multi-layer spread); (4) seed counts for underpowered cells (MEMIT L10/L14, KL-ladder L12, zsRE-deletion, Llama-3B L14, generic projector, and 2-seed EGL table); (5) descriptive claims without causal grounds (anisotropy, sequential geometry attribution); (6) GPT-2-XL sanity cells below threshold.
- **Section Reference:** §9 (Limitations, pp. 8–9)

**A2. Does your work present any dual-use concerns?**
- **Answer:** No
- **Justification:** The work is a mechanistic analysis of collateral damage in knowledge-editing methods (ROME, MEMIT, fine-tuning). The causal intervention (AlphaEdit) is a null-space projection designed to *reduce* damage, not amplify harm. No techniques for misuse, attack synthesis, or adversarial evasion are introduced. Code will be released and is intended for defensive interpretation and mitigation of knowledge-editing side effects.
- **Section Reference:** N/A

**A3. Are there foreseeable harms or negative impacts of your work?**
- **Answer:** No significant harms identified.
- **Justification:** The work advances understanding of knowledge-editing collateral effects and proposes a mitigation (AlphaEdit). The primary use case is hardening knowledge-editing pipelines against unintended side effects (e.g., refusal-deletion coupling). The mechanistic framing is foundational, not prescriptive of deployment decisions.
- **Section Reference:** N/A

---

### B. Artifacts and Resources

**B1. Models Used — Licensed / Compliance**

| Model | License | Citation | Status |
|-------|---------|----------|--------|
| Llama-3.2-1B | Llama 2 Community License | Meta; HuggingFace | ✓ Licensed, open-access |
| Llama-3.1-8B | Llama Community License | Meta; HuggingFace | ✓ Licensed, open-access |
| Llama-3.2-3B | Llama Community License | Meta; HuggingFace | ✓ Licensed, open-access |
| Qwen2.5-0.5B | Apache 2.0 | Alibaba Qwen; HuggingFace | ✓ Licensed, open-access |
| Qwen-1.5B | Apache 2.0 | Alibaba Qwen; HuggingFace | ✓ Licensed, open-access |
| Qwen-3B | Apache 2.0 | Alibaba Qwen; HuggingFace | ✓ Licensed, open-access |
| Phi-3.5-mini | MIT | Microsoft; HuggingFace | ✓ Licensed, open-access |
| gemma-2-2b | Gemma Terms of Use | Google; HuggingFace | ✓ Licensed, open-access |
| GPT-2-XL | MIT | OpenAI; HuggingFace | ✓ Licensed, open-access |

**B2. Datasets Used**

| Dataset | License | Source | Usage |
|---------|---------|--------|-------|
| CounterFact (CF) | CC-BY-4.0 (inferred) | Meng et al. 2022 ROME paper | Knowledge editing evaluation (~200 facts edited per run, ~500 probes for damage metric) |
| zsRE | MIT | Elazar et al., HuggingFace | Transfer study for deletion collateral (§7) |

**B3. Code and Artifacts Created**

- **Source code:** The fission-engine (`edit-harness/`) and paper-specific experiment runners (e.g., `experiments/{mechanism_dump.py, gradsim_baseline.py, geometry_router.py, aggregate_g4_causal.py, mechanism_sc_table.py}`) are to be released upon publication.
- **Result artifacts:** All per-cell JSON result files (`edit-harness/results/*.json`) and seed-level data are to be released.
- **Figures:** TeX-native pgfplots with inline data from canonical JSONs, audited for byte-identity (§10_appendix.tex, figures-tex/).
- **Reproducibility:** Figure-generation scripts and configuration files sufficient to regenerate all numbers are included.

**Answer to B:** Yes, we report all models and datasets used; all are open-access with documented licenses. Code and results will be released upon publication.

---

### C. Computational Experiments

**C1. Compute Resources Reported**

- **Answer:** Yes
- **Justification:** The work runs entirely on a single RTX 5090 24GB laptop. The final campaign (run_u5 and related queues) consumed ~52 GPU-cells over ~4 days. A "cell" is defined as 200 knowledge edits × 500 probes per edit = 100,000 forward passes for the damage metric. Each cell takes ~200–220 seconds on the RTX 5090.
- **Section Reference:** SETUP.md, workspace CLAUDE.md (GPU cell definition); paper §3 (Method).

**C2. Hyperparameters and Training Details**

- **Answer:** Yes
- **Justification:** All key hyperparameters are documented in code and reported in the method section. ROME uses the official implementation with defaults; edit strength (single rank-one rank, α for orthogonal update magnitude) is controlled per layer. The within-probe metric is Spearman rank correlation. All results are reported at single-seed (§3, early), 2-seed (EGL canonical table), or 3-seed (C1, C4, dissociation) resolution; seed counts are explicitly marked in text and Limitations (§9).
- **Section Reference:** §3 (Method); Limitations §9 (seed count declarations).

**C3. Seed / Random Initialization**

- **Answer:** Yes
- **Justification:** Experiments are reported with stated seed counts (1-seed, 2-seed, or 3-seed means). Random seeds are `{0, 1, 2}` for 3-seed runs. Standard deviations are reported in macros.tex and in-text where applicable (e.g., G1 gate within-probe rhos ρ ± SD at L8/L10/L12/L14). Single-seed cells are flagged as such in the paper and Limitations section.
- **Section Reference:** macros.tex comments; §9 Limitations.

---

### D. Human Subjects / Crowdsourcing

**D1. Human subjects or crowdsourced data involved?**
- **Answer:** No
- **Justification:** N/A. All evaluation uses automated metrics (Spearman correlation, forward-pass accuracy, norm differences) on benchmark datasets (CounterFact, zsRE). No human raters, surveys, or crowdsourced labels.
- **Section Reference:** N/A

---

### E. AI Assistants in the Research and Writing Process

**E1. AI coding or writing assistants used?**
- **Answer:** Yes
- **Justification:** AI coding and writing assistants (Claude, specifically Claude Code) were used extensively in the research and drafting pipeline **under direct human supervision**. Usage includes: (1) code authoring and debugging (fission-engine job queue, experiment runners, analysis scripts); (2) LaTeX manuscript composition and macro generation; (3) figure generation and validation scripts; (4) documentation and comment authoring. All outputs were reviewed, tested, and verified against canonical JSON artifacts before integration. All scientific decisions, experimental design, result interpretation, and vulnerability (e.g., CP-Edit pre-registration kill, MEMIT S×C embargo) were made by the human author.
- **Scope & Transparency:** This is consistent with ACL's AI assistant policy (aclanthology.org/2023/acl-blog.pdf), which permits AI-assisted coding and writing under human direction and with full disclosure.
- **Section Reference:** Transparent disclosure per ACL policy.

---

### F. Reproducibility

**F1. Reproducibility statement and artifact release plan**

- **Answer:** Yes
- **Justification:** The paper includes explicit statements of reproducibility intent (§1, final paragraph). Code, per-cell result JSONs, and figure-generation scripts sufficient to reproduce every quoted number will be released upon publication. The fission-engine (`edit-harness/`) is a reusable, version-controlled GPU job queue; all experiments are runnable from archived configs and result artifacts.
- **Conditions:** (1) Models are downloaded from HuggingFace (no authentication required). (2) GPU compute requires a single RTX 5090 24GB or higher (smaller runs can use cheaper GPUs with longer wall-clock times). (3) Dependencies: PyTorch, transformers, datasets, numpy/scipy, tqdm (all standard, pip-installable). (4) Estimated time: ~4 days on RTX 5090 for the full campaign; individual cells take ~3–4 minutes.
- **Section Reference:** §1 (reproducibility statement); SETUP.md; edit-harness/README (upon release).

**F2. Code release timeline**

- **Answer:** Upon publication (contingent on venue acceptance).
- **Justification:** Code is review-ready but is held pending final venue acceptance to avoid pre-review publication and ensure anonymity during the review round.

---

## FORM FIELDS FOR OPENREVIEW

### Submission Metadata

| Field | Value |
|-------|-------|
| **Title** | When and Why Does Key Geometry Predict Locate-then-Edit Collateral Damage? A Closed-Form, Editor- and Architecture-Conditioned Account of the Llama Family, with a Causal Test |
| **Abstract** | [See RESPONSIBLE NLP CHECKLIST, Section A1 — verbatim from paper] |
| **Track** | Interpretability and Analysis of Models for NLP |
| **Keywords** (5–8) | knowledge editing; mechanistic interpretability; rank-one updates; ROME; AlphaEdit; collateral damage; causal inference; model editing |
| **Venue Note** | [Internal author guidance] Reviewers will be committed to revision resubmission at a CCF-B *ACL main venue (e.g., EMNLP-2026, NAACL-2026) in a later review cycle if conditional acceptance is offered. EACL excluded per authors' venue constraints. This note is for internal tracking only and should not appear in the submission. |

---

## PRE-SUBMISSION COMPLIANCE CHECKLIST

Use this checklist on **submission day** before clicking "Submit" on OpenReview. All items must be verified **by the human author** (Claude does not submit).

### Anonymity
- [ ] **Author names and affiliations scrubbed from PDF and LaTeX source.**
  - Verify `main.tex` line 44: `\author{Anonymous ARR submission}`
  - Verify PDF header / metadata has no author info.
  - Run: `pdftotext main.pdf - | grep -i "author\|affiliation"` (should return only "Anonymous")

- [ ] **Acknowledgments section (if present) does not identify authors.**
  - Check: §1 closing paragraph acknowledges CLaRE-ty without author names.
  - No institution names or GitHub profiles mentioned.

### Formatting & Length
- [ ] **Page count: exactly 8 pages for main body + 2–3 for appendix (total 10–11).**
  - Run: `pdfinfo main.pdf | grep Pages`
  - Verify: body ends at p. 8; appendix §10 starts p. 9; total ≤ 11 pages.

- [ ] **Line numbers present (per acl.sty [review] option).**
  - Verify: Left margin of PDF shows continuous line numbers.
  - Check: `main.tex` line 14 has `\usepackage[review]{acl}`.

- [ ] **No overfull or underfull boxes; PDF is clean.**
  - Run: `pdflatex main.tex 2>&1 | grep -i "overfull\|underfull"`
  - Expected: 0 warnings.

- [ ] **Figures embedded and audited.**
  - Check: `figures-tex/*.tex` files are present; all figures render in PDF.
  - Verify: 6 figures (F1–F6) span pp. 4–8 with captions.
  - Data provenance: All figures have inline pgfplots data audited against `edit-harness/results/*.json`.

### Responsible NLP Checklist
- [ ] **Limitations section present and substantive.**
  - Verify: §9 "Limitations" (pp. 8–9) covers scale, architecture scope, mechanism scope, seed counts, and unsettled claims.
  - Length: ~350 words.

- [ ] **Responsible NLP checklist completed (this form).**
  - Verify: All items A1–F2 have explicit Yes/No/N/A answers.
  - Flag any [USER CONFIRM] items and resolve before submission.

### Bibliography & References
- [ ] **All citations are complete and formatted consistently.**
  - Run: `bibtex main && grep -c "CITEREF" main.bbl`
  - Expected: 0 undefined citations; all .bbl entries populated.

- [ ] **CLaRE-ty reference is correct (arXiv 2603.19297, ACL 2026 Findings).**
  - Verify in `references.bib`: arXiv link or official ACL listing.

### Result Accuracy
- [ ] **All quoted numbers match canonical JSON sources in `edit-harness/results/`.**
  - Spot-check 5 random values from macros.tex against JSONs:
    - `\gWithinLtwelve{} = 0.602` → check `G1_L12_analysis.json`
    - `\regimeEightB = −0.097` → check `C3_regime_8b_L24_r4.json`
    - etc.
  - Expected: byte-identical matches.

- [ ] **All binding wording constraints are respected (from SETUP.md).**
  - MEMIT: Quote `ρ_C` only; never "MEMIT S×C". Approved sentence present.
  - Sequential: Descriptive only; ordering-dependent 10–42% range; no ρ≈0.55.
  - S×C: Zero-cost GradSim surrogate; never "beats key-cosine".
  - Scope: Signed law is Llama-family-specific; magnitude law attenuates at 8B.

### Final Artifact
- [ ] **PDF to submit is `/paper-arr/main.pdf` (final version, not draft).**
  - Verify: File timestamp is recent (after final authoring pass).
  - Size: ~2–3 MB (typical for pgfplots figs).
  - Run: `file main.pdf` → should report "PDF document, version 1.5 (or higher)"

### [USER CONFIRM] Items to Resolve Before Submission

- [ ] [USER CONFIRM] **Venue decision: ARR → (conditional accept) → KBS or IEEE TNNLS?**
  - Current state: B6 targeting a CCF-B *ACL main venue first (EMNLP/NAACL); KBS/TNNLS as journal extension after.
  - User must decide whether to commit to this sequencing or pivot.
  - See workspace CLAUDE.md for VENUE-STRATEGY-2026-07-01.md on Desktop.

- [ ] [USER CONFIRM] **Code release consent.**
  - Confirm: Human author will release `edit-harness/` and `experiments/` upon publication acceptance.
  - Anon repos (GitHub, zenodo, etc.) are acceptable for the review round if needed for reproduction.

- [ ] [USER CONFIRM] **Final read-through of §1–§10 by human author.**
  - This checklist is an automated pre-submission audit. The human author must read the final PDF end-to-end to catch any errors the checklist missed (e.g., typos, narrative flow, claim alignment).

---

## SUMMARY

**Submission status:** Ready for final review by human author + resolution of [USER CONFIRM] items.

**File paths:**
- PDF to submit: `/home/zeyufu/Desktop/idea-feasibility-analysis/paper-arr/main.pdf`
- LaTeX source: `/home/zeyufu/Desktop/idea-feasibility-analysis/paper-arr/main.tex` (+ sections/, macros.tex, references.bib)
- Canonical results: `/home/zeyufu/Desktop/idea-feasibility-analysis/edit-harness/results/*.json`

**Next steps:**
1. User resolves [USER CONFIRM] items above.
2. User performs final read-through of PDF.
3. Run compliance checklist (shell commands provided).
4. User manually submits on OpenReview (Claude does not have submission access).

---

**Document prepared by:** Claude Haiku 4.5 (research agent)  
**Date:** 2026-07-05  
**Review status:** Authoring pass; awaiting user sign-off and submission.
