# D2 submission package — binding build rules (adopted 2026-07-16)

Adopted from the portfolio-wide review directive of the parallel campaign
(REVIEW-DIRECTIVE-2026-07-16, other workspace) — apply DURING drafting, not as a
post-hoc fix pass. Integrity rails always: zero result-number changes without artifact
provenance; both versions (canonical.md, main.tex) compile/render clean; canonical.md
stays the source of truth.

1. **Tone.** No bold/italic abuse; AI-register phrasing out ("delve", "crucially",
   formulaic intensifiers); prose reads like a person wrote it.
2. **No code leakage in submission prose.** File/function/script/artifact names never
   appear in main-text prose — re-express in natural language. Code names live ONLY in
   the Data/Code Availability statement. (canonical.md MAY carry them in its §A
   provenance block, which is marked non-submission.)
3. **No appendix.** Content merges into the main body within venue limits.
4. **Figures = standalone-compiled PDFs.** R/ggplot2 → tikzDevice sources +
   standalone wrappers + Makefile under `figures-src/`; `main.tex` includes ONLY the
   compiled PDFs (portal-safe, no inline TikZ). Font sizes set per the figure's actual
   column span. NEVER `\resizebox` a tikzpicture. Visual audit of rendered pages after
   every figure change — logs are blind to overlap/clipping (3 such defects already
   caught this way on figA/figB).
5. **Floats.** Span (one- vs two-column) audited per float; placement after first
   reference; order matches first occurrence; every `\ref`/`\cite` resolves.
6. **References.** Every bib entry verified against live sources (DOI/metadata)
   before submission; unverifiable entries flagged, never fabricated.
7. **Venue limits VERIFIED live 2026-07-16** (`VENUE-KBS-FACTS-2026-07-16.md`, URLs +
   dates inside): abstract ≤250 words (ours: 219 expanded); soft 20-page guidance;
   **single-anonymized** (real byline, cite B6 normally); highlights 3–5 × ≤85 chars
   (`highlights.txt`, verified); refs flexible at submission, elsarticle-num final;
   declarations required: CRediT, competing interests (.docx), data availability,
   GenAI-use declaration (before references), funding.
8. **Code archiving: KBS Option C (deposit-or-explain), code explicitly included.**
   Plan: Zenodo deposit with DOI, cited in the Data availability statement — the ONLY
   place code/artifact names may appear (rule 2).
9. **Desk-review mitigation (applied).** Lead abstract/intro/cover-letter with
   knowledge maintenance + decision support (KBS scope), not geometry; one explicit
   scope-relevance sentence in the cover letter.
