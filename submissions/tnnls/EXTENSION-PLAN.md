# IEEE TNNLS (SCIE Q1, CCF-B journal) — extension workspace

STATUS (2026-07-11): FALLBACK/EXTENSION ONLY. The paper was submitted to IEEE TETCI
(07-09/10) and the venue standard is now SCIE-only (CCF not required) — TNNLS is no
longer CCF-mandated. This plan applies only to a rejection-contingency resubmission or
a later ≥30%-new journal extension. IEEEtran class when scaffolded.

## What TNNLS requires beyond the current package (harder bar than KBS)
1. **Theory**: the standing rule ("nothing to TNNLS while ... no theorem-bearing
   artifact") still binds. The natural theorem: formalize S×C ≈ first-order influence
   (the GradSim equivalence, currently empirical to ~2 decimals) as a proposition with
   assumptions + proof for the rank-one locate-then-edit update; corollary for the
   within-probe damage bound. This is CPU/paper work but needs a careful pass — the
   empirical equivalence is already banked, the derivation exists in §2; the gap is
   rigor (assumptions, error term for the multi-step value solve).
2. **Scale evidence**: now partially satisfied (8B triple, bf16 equivalence gate) —
   the old "≤3B" blocker is lifted; a 8B causal (AlphaEdit) cell would complete it
   (~2-3 GPU-h, bf16; never run at 8B).
3. Editor breadth + artifact: same items as KBS 1-2 (TNNLS also expects methodological
   novelty over analysis).
4. IEEE format/compliance: IEEEtran, structured abstract conventions, ~14pp double-column.

## Timing profile
- Rolling, no deadline. First decision typically ~4–8 months (slower than KBS);
  major-revision rounds common. Longest total horizon of the three venues.
- Same dual-submission rule: same-content overlap with ARR is prohibited; extension
  with disclosure is the sanctioned path.

## Verdict vs KBS (for the eventual decision)
TNNLS = higher prestige ceiling, requires the theorem + more method framing, slowest.
KBS = artifact-first framing fits the existing D3/benefit angle better, faster decisions.
The 07-01 venue strategy named both; the theorem question is the real fork.
