# TETCI portal — code/data questions: answers + post-acceptance plan (2026-07-10)

## Portal answers (recommended)

**Code Ocean — "Do you have code associated with your manuscript?"**
→ **"Yes, I have code associated with my manuscript"** — but do NOT upload now.
Uploading is explicitly permitted "at submission, revision, or after acceptance";
defer to AFTER ACCEPTANCE because (a) review is double-anonymous and a Code Ocean
capsule is tied to your identity (TETCI's anonymization rule excludes links to
external sites for exactly this reason), and (b) the manuscript deliberately
contains no code link, so nothing dangles.

**Scope dropdowns:**
1. "Main contribution a theoretical and/or algorithmic advancement that expands
   the scope of existing computational/machine intelligence?" → **Yes**
   (closed-form S×C decomposition + formal rank-estimator statement + causal
   test — the paper's spine is analytical, not an application write-up).
2. "Experimental investigation of / survey on the application of CI techniques
   in an innovative/emerging real-world domain?" → **Yes** (large-scale
   experimental investigation of knowledge editing — an emerging LLM-maintenance
   problem; 11 checkpoints, 6 editors, 5+ datasets).
3. "Does the manuscript identify a NEW domain of application for CI?" → **No**
   (knowledge editing exists as a domain; we characterize its mechanism — answer
   honestly, the two Yes answers above carry the classification).

**IEEE DataPort — "Do you have data associated with your manuscript?"**
→ At submission: nothing to enter (no DOI exists yet); plan a post-acceptance
deposit. Do NOT deposit third-party benchmarks (CounterFact, zsRE, MQuAKE(-T),
RippleEdits, GLUE) — they are cited, not ours to redistribute.

## Post-acceptance release inventory (prepare only after decision letter)

**Code Ocean capsule** (from `edit-harness/`):
- `experiments/` (killgate_keygeom.py + analysis scripts incl.
  mechanism_sc_table.py, aggregate_g4_causal.py, mquake_overlap_audit.py,
  esr_band_analysis.py, gradsim_true.py), `editors/` (ROME/MEMIT/AlphaEdit/
  FT/KL-FT/GRACE implementations), a small CPU-runnable demo config
  (Qwen-0.5B smoke cell) so Code Ocean's verification passes without a GPU,
  requirements pin (torch 2.8, transformers 5.13, numpy 2.3 — from the
  provenance fields), and the R figure pipeline
  (submissions/ieee/figures/make_figures_ieee.R).

**IEEE DataPort dataset** (free ≤2TB, DOI issued immediately):
- All canonical `results/*.json` (analysis tables quoted in the paper).
- Per-edit `.npz` matrices (COS/damage/norm_growth per cell) that back every
  reported statistic.
- ⚠️ **CRITICAL**: the cloud-run npz (NeoX-20B, Pythia 1.4B/2.8B, GPT-J,
  Llama-8B causal, instruct, MQuAKE cloud cells) exist ONLY on the two
  powered-off AutoDL instance disks. If a DataPort deposit is wanted, PULL
  THOSE NPZ BEFORE RELEASING THE INSTANCES. Local-run npz are already in
  `edit-harness/results/matrices/`.

**Manuscript hook**: at camera-ready, add one data/code-availability sentence
citing the Code Ocean + DataPort DOIs (goes in with the de-anonymization pass;
see CAMERA-READY-NOTES.md).
