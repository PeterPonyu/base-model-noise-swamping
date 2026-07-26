# KBS submission facts — verified LIVE 2026-07-16 (kbs-limits agent, rendered official pages)

PRIMARY SOURCE: https://www.sciencedirect.com/journal/knowledge-based-systems/publish/guide-for-authors
(retrieved 2026-07-16; ScienceDirect 403s plain fetches — verified via rendered browser).
Timeline row: https://www.sciencedirect.com/journal/knowledge-based-systems

1. **Abstract ≤250 words** ("concise and factual abstract which does not exceed 250
   words"); stand-alone, no references. → Our abstract: 219 words (macros expanded), OK.
2. **Length: soft cap** — research papers "preferably no more than 20 double line spaced
   manuscript pages, including tables and figures". Guidance, not a hard gate.
3. **SINGLE anonymized review** → do NOT anonymize; author names/affiliations on the
   title page. (Resolves the companion-citation question: B6 can be cited normally.)
4. **Highlights encouraged, not required**: 3–5 bullets, ≤85 characters each incl.
   spaces, separate editable file with "highlights" in the name. → `highlights.txt`.
5. **Required declarations**: CRediT (14-role); Declaration of competing interests
   (always, via declarations tool, .docx); Data availability statement; **Declaration of
   generative AI and AI-assisted technologies in the manuscript preparation process**
   (required IF AI tools used; section placed before references; template: "During the
   preparation of this work the author(s) used [TOOL] in order to [REASON]. After using
   this tool/service, the author(s) reviewed and edited the content as needed and
   take(s) full responsibility for the content of the published article."; grammar/spell
   tools exempt; AI may not be an author); Funding sources statement.
6. **References: flexible at submission** (any consistent style); journal style is
   numbered [n] in citation order (= elsarticle-num); **DOIs encouraged, not required**;
   LTWA journal abbreviations.
7. **Code/data: Option C (mandatory deposit-or-explain).** Deposit research data —
   explicitly incl. "software, code, models, algorithms" — in a repository, cite with
   link/DOI, or state why sharing is impossible. **GitHub+Zenodo acceptable** (guide's
   own software example cites Zenodo DOI). → Plan: Zenodo deposit, DOI cited in the Data
   availability statement.
8. **LaTeX: elsarticle accepted/encouraged; editable .tex required** (PDF-only source
   not accepted). Double-column permitted only for LaTeX; single-column preprint style
   is the safe first-submission default.
9. **Desk screening**: editor suitability assessment + originality/plagiarism/AI tools.
   Explicit hard gates: editable source, scope fit, declarations present.
   **Aims & scope core**: "knowledge-based and other artificial intelligence
   techniques-based systems… to support human prediction and decision-making through
   data science and computation techniques"; leading topics incl. "Machine learning
   theory, methodology and algorithms," "Knowledge presentation and engineering,"
   "Intelligent decision support systems."
10. **Timeline (official journal insights)**: first decision 3 days; decision after
    review 56 days; acceptance 158 days; online +8 days. CiteScore 13.7, IF 8.0.
    (A third-party blog's conflicting numbers were checked and disregarded.)

## Desk-review mitigation (adopted into the draft)
- Main risk: pattern-matching to "generic ML theory". Mitigation APPLIED: abstract now
  leads with knowledge maintenance/updating in deployed systems (done 2026-07-16);
  intro port must keep the maintenance-workflow lead; cover letter gets one explicit
  KBS-scope relevance sentence (knowledge engineering + intelligent decision support:
  the admission rule IS a decision-support artifact).
- Formatting risk low: single-anonymized, flexible refs, no hard page cap.

## Neurocomputing (fallback), verified same day
Abstract ≤250 words; single anonymized; highlights encouraged 3–5 × ≤85 chars.
Source: https://www.sciencedirect.com/journal/neurocomputing/publish/guide-for-authors

## SUITABILITY DEEP-CHECK (2026-07-16, kbs-suitability agent) — VERDICT: GO
- KBS's demonstrated envelope covers this paper's shape: verified KBS-published
  precedents incl. **Knowledge Neuronal Ensemble (KNE)** (locate-then-edit editing, KBS
  2025, arXiv 2412.20637), DeCO (in-context editing, KBS 2026), DCHD (LLM reliability,
  KBS 2026), + the LLM×KB survey (KBS 2025). Editorial line: "knowledge/reasoning must
  be the protagonist"; wants decision support + ablation isolating the knowledge
  component + deployment context — our admission-rule framing fits.
- **Reference norms (hard data, OpenAlex issn:0950-7051, 2024-06+ sample n=15):
  min 27 / mean ≈75 / max 137. Our 18 = desk-visible anomaly. Target 40–60.**
- Length: soft guideline ≤20 double-spaced pages / typical 6–9k words; our 17pp
  single-column ≈10–13k words — plan a light trim toward ~9k (not a desk gate).
- Special issues: CFP page 403'd to agents — UNVERIFIED; check in-browser before
  submitting; default = regular research-article track.
- Neurocomputing compare: its guide redirects knowledge-based/decision-support work to
  KBS; viable backup only if reshaped mechanism-first. **KBS primary confirmed.**
- Cheap reshapes prescribed (applied same day): explicit gain-screen ablation framing
  (§6), deployment-context paragraph (§6). Full report in session transcript.
