# KBS (Knowledge-Based Systems, Elsevier, SCIE Q1) — extension workspace

STATUS (2026-07-11): FALLBACK/EXTENSION ONLY. The paper was submitted to IEEE TETCI
(07-09/10; SCIE-only standard confirmed by user — TETCI acceptable). KBS remains the
fast fallback on rejection, or the home for a later ≥30%-new journal extension. Do NOT
scaffold LaTeX unless one of those triggers fires; elsarticle class when the time comes.

## What KBS requires beyond the current (conference-shaped) package
1. **Deployable artifact framing** — KBS wants a system/method contribution, not analysis
   alone. The sanctioned angle: the D3 reframe as a *benefit-magnitude predictor*
   (geometry → predicted AlphaEdit benefit, 27–71× ratios) packaged as a usable gate.
   Needs: predictor implementation + evaluation protocol + the aniso arm as its feature
   ablation. (This was Phase 3 of the 07-02 EOD plan.)
2. **Editor breadth ≥6 incl. a memory/in-context family** (GRACE/WISE class — new code,
   the largest single lift; also the top reviewer-expectation gap at any Q1 venue).
3. **EGL breadth**: full editor × seed EGL grid (currently rome/memit/alpha, 2-seed).
4. **Dataset breadth**: +1 (MQuAKE-class multi-hop, or temporal). Ask-first download.
5. **Journal-format completeness**: extended related work, computational-cost analysis,
   reproducibility statement + code release plan, 30–40pp manuscript (the 12-section
   density that is a LIABILITY at 8 pages becomes an asset here).
6. If extending the ARR paper: ≥30% new material + explicit disclosure of the conference
   version (Elsevier policy). Items 1–4 comfortably clear that bar.

## Timing profile
- Rolling submission — NO deadline; submit any day. First decision typically ~3–6 months;
  1–2 revision rounds common; camera-ready to online-first is fast after acceptance.
- Journal-FIRST path: skip ARR, spend ~3–5 weeks closing items 1–4, submit directly.
  Pro: single venue, no anonymity-period constraints, no cycle pressure.
  Con: forfeits the Aug-3 ARR shot + its portable reviews; KBS decision horizon is long;
  rejection returns the paper ~5 months later with no conference fallback banked.
- Extension path (sanctioned): ARR Aug 3 → journal extension over the fall. Two shots.

## Estimated GPU/code cost to close the gaps
- Item 1: mostly CPU (predictor + eval on existing matrices) + a few alpha seed cells.
- Item 2: ~2–4 days code (new editor family) + ~1 GPU-day cells.
- Item 3: ~4–6 GPU-h. Item 4: download decision + ~1 GPU-day.
