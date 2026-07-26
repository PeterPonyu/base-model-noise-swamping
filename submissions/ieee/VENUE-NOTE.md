# IEEE venue decision — RESOLVED: submitted to IEEE TETCI (2026-07-09/10)

> **RECONCILED 2026-07-11.** The manuscript was submitted to **IEEE TETCI**
> (SCIE Q1, CCF-none; see `portal/` — Full Paper, sole author,
> double-anonymous) and is under review. The same day, the user clarified the
> venue standard: **SCIE-indexed is the filter; CCF rank is NOT required** —
> so TETCI fully satisfies the standard and no CCF-motivated parallel
> TNNLS extension is needed. The TNNLS/TASLP fork notes below are retained as
> *fallback/extension* planning (rejection contingency or a later ≥30%-new
> journal extension), not as an open decision.

This workspace targeted **one manuscript, two possible IEEE homes**, selected
by the `\iftnnls` / `\iftaslp` toggle in `main.tex` (see `SETUP.md` for the
mechanics); the summary below is kept for what each fallback venue would need,
in sync with `../tnnls/EXTENSION-PLAN.md` and `../kbs/EXTENSION-PLAN.md` (KBS
is a third, non-IEEE option tracked separately — Elsevier `elsarticle`, not
scaffolded here).

## IEEE TNNLS (SCIE Q1, CCF-B)

- **Standing rule**: nothing ships to TNNLS without a theorem-bearing
  artifact. STUB-THEOREM (`sections/03_method.tex`) is therefore **mandatory**
  for this fork — it must be closed (proposition + proof + corollary,
  review-gated) before a TNNLS submission, not merely rendered as a draft box.
- **Scale evidence**: the old "≤3B" blocker is lifted (8B triple now exists),
  but STUB-8BCAUSAL (`sections/05_causal.tex` — the never-run AlphaEdit causal
  cell at Llama-8B, ~2–3 GPU-h bf16) is what completes the scale-evidence
  chain end to end. Strongly recommended to close before submitting to this
  fork.
- **Editor breadth + artifact**: same STUB-EDITOR6 / STUB-EGLSEEDS gaps as
  KBS — TNNLS also expects methodological novelty over pure analysis, so
  these help here too, though they are not as load-bearing as the theorem.
- **Format**: `IEEEtran.cls` (`[journal]` mode), ~14pp double-column,
  structured-abstract conventions satisfied by the abstract + IEEEkeywords
  block in `sections/00_abstract.tex`.
- **Timing**: rolling, no deadline. First decision typically ~4–8 months
  (slower than KBS); major-revision rounds common. Longest horizon of the
  three venues under consideration (TNNLS/KBS/dual-submission-compliant
  extension of ARR).

## IEEE/ACM TASLP

- Not named in the original 07-01/07-02 venue strategy docs (which considered
  TNNLS and KBS); added here as the IEEE-family alternative when the
  language-technology framing is the better fit, since the paper's subject —
  factual maintenance of deployed language models — is squarely in-scope for
  a speech-and-language-processing systems journal audience.
- STUB-THEOREM is **optional** under this fork: the proposition may ship as
  a `remark` instead (still describing the same GradSim-equivalence result,
  just without the load-bearing "proposition" framing TNNLS's standing rule
  demands).
- The language-technology-maintenance framing paragraph (in
  `sections/00_abstract.tex` and `sections/01_intro.tex`, inside
  `\iftaslp...\fi`) is the fork-specific addition: it recasts "knowing when a
  factual patch is collaterally expensive before applying it" as a maintenance
  problem for deployed language-technology pipelines, not only a modeling
  curiosity.
- All other content (method, regime, causal, dissociation, deletion,
  generality, anisotropy, sequential, discussion) is shared verbatim with the
  TNNLS fork — there is no separate section content, only the toggle blocks.
- SCIE/CCF status and timing profile for TASLP have not yet been
  independently verified against the standing SCIE-indexed/CCF-ranked
  constraint (see workspace `CLAUDE.md`) — **do this check before committing
  to this fork**; TNNLS's CCF-B / SCIE-Q1 status is already confirmed via the
  07-01 venue strategy work, TASLP's has not been re-verified in this
  workspace.

## What each venue needs, restated as a closing checklist

| Gap | KBS | TNNLS | TASLP |
|---|---|---|---|
| STUB-THEOREM (formal GradSim-equivalence statement) | not required | **mandatory** | optional (may ship as remark) |
| STUB-8BCAUSAL (AlphaEdit causal cell at 8B) | nice-to-have | strongly recommended (completes scale chain) | nice-to-have |
| STUB-EDITOR6 (memory/in-context 6th editor family) | top reviewer-expectation gap | expected | expected |
| STUB-EGLSEEDS (full editor × seed EGL grid) | expected | expected | expected |
| STUB-DATASET (+1 dataset, e.g. MQuAKE-class) | expected | nice-to-have | nice-to-have |
| Deployable-artifact framing (D3 benefit-magnitude predictor) | **primary sanctioned angle** | secondary | secondary |
| Page/format | 30–40pp, `elsarticle`, no limit | ~14pp, `IEEEtran` | ~14pp, `IEEEtran` |
| Extension-disclosure threshold | ≥30% new material (Elsevier policy) | ≥30% new material (IEEE policy, same norm) | same |

## Binding constraint reminder (applies to every fork)

Per workspace `CLAUDE.md` (updated 2026-07-11): venue selection is
**SCIE-indexing only — CCF rank NOT required** (user clarification). TMLR /
BlackboxNLP / ICBINB / COLM / EACL fail this filter (not SCIE) and are out of
scope regardless of fit. The extension path (ARR first, journal
extension after with disclosure) is the sanctioned route; a journal-first path
skipping ARR is possible but forfeits the Aug-3 ARR cycle — see
`../README.md` and the two `EXTENSION-PLAN.md` files for the full timing
trade-off already worked out for KBS vs. TNNLS. This note does not re-litigate
that trade-off; it only tracks what content-level work each IEEE fork still
needs.

## Open user decisions (unchanged from `../README.md` / workspace `CLAUDE.md`)

KBS-vs-TNNLS-vs-TASLP is still with the user. This workspace does not commit
to a venue — it keeps both IEEE forks buildable from one `main.tex` so the
decision can be made (or deferred) without redoing the LaTeX scaffolding.
