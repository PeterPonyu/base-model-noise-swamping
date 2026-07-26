# B6 harness + science hardening — 2026-07-01 (evening session)

Investigation of the current direction (was the core designed well enough?) + concrete
enhancements grounded in the training/run logs. All changes are 0-download, CPU-validated
where possible, and live GPU-smoke-tested where they touch the editor.

## Verdict on the core design
**Solid, with two reviewer-fatal gaps now closed.** The within-probe partialled-Spearman
methodology, atomic result writes, idempotent skip logic, native fp32 ROME, and the
metric-discipline (signed Spearman not AUROC) are genuinely well built. Two things would have
been killed on review, plus the scheduler had the same bug that already cost 8h.

## What was weak → what changed

### 1. C4 causal test was circular (CRITICAL, science)
The AlphaEdit null-space projector was fit on the **same 500 probe keys** whose damage it then
measured — so "98% removal, concentrated on high-cosine probes" was partly true *by
construction*. Fixed in `killgate_keygeom.py`: `--alpha_proj_source {probes,holdout,generic}`
(+ `--holdout_frac`). `holdout` fits the projector on a **disjoint** fact bank; `generic` on
random non-subject activation keys. Provenance (`alpha_proj_source`, `proj_disjoint`) saved to
npz + JSON. `aggregate_g4_causal.py` gained `--proj_source` to aggregate only the honest runs
→ `C4_causal_holdout_table.json`. E6 jobs staged in the launch script. **The paper's causal
section must report holdout numbers as primary.**

### 2. "Raw key-cosine" headline is beaten by its own baseline (CRITICAL, framing)
Extended G2 gradsim + H2 lex/SBERT from L8-only to **all four layers**. Finding: the
norm-growth×cosine surrogate (= the closed-form **S×C** mechanism) matches/beats raw key-cosine
at *every* layer (L12 peak: 0.602 vs 0.677; L14: 0.301 vs 0.504). This is expected from the
rank-one derivation and *supports* the mechanism — but the paper must lead with **S×C**, not
raw geometry, or a reviewer's own baseline sinks it. H2 stays clean: key-cosine beats
lexical (≤0.05) and SBERT (≤0.12) surface similarity at all layers.

### 3. Statistics were anti-conservative (MAJOR)
`analyze_matrices.py`: the per-column shuffle null destroys cross-column edit structure →
under-states null variance. Added the **strict edit-level (single-row) permutation null**
(edits are the exchangeable unit), **edit-cluster bootstrap CI**, tie-averaged (midrank)
Spearman, and `--n_perm`. Re-ran G1 → `G1_stability_L{8,10,12,14}_v2.json`: **PASS at all
layers** under the strict null (p=0.001 floor, z=6.5–16.4; edit-cluster CIs L8 [0.35,0.45] …
L12 [0.55,0.64] … L14 [0.19,0.34] — all clearly > 0). The headline is *more* defensible now.

### 4. Scheduler robustness (MAJOR, would recur)
- `fission-engine/gpuguard.py` STILL gated on zero-compute-apps — the exact 8h-loss bug.
  Backported the util+mem gate (`is_gpu_idle` → util<10 AND mem<1500MiB), added `gpu_load()`,
  a `consecutive`-streak `wait_for_gpu`, and **fail-closed** on unparseable nvidia-smi.
- `run_deep_until1900.sh`: (a) missing-model **precheck** = config skip, not a failure — a
  missing model dir previously counted toward the 2-strike "GPU wedge" abort and could kill
  the whole night's queue; (b) failure **classification** — only timeouts / long-running
  failures (wedge-like) count toward the abort; a fast crash (bad arg, missing data) is logged
  and skipped; (c) staged **E6** disjoint-projector jobs + C4-holdout aggregation.

## Files touched
- `edit-harness/experiments/killgate_keygeom.py` — projector sources, provenance
- `edit-harness/experiments/analyze_matrices.py` — strict null, cluster CI, midrank, --n_perm
- `edit-harness/experiments/aggregate_g4_causal.py` — --proj_source honest aggregation
- `edit-harness/run_deep_until1900.sh` — precheck, failure classification, E6, C4-holdout
- `fission-engine/gpuguard.py` — util+mem idle gate backport
- New results: `G1_stability_L*_v2.json`, `G2_gradsim_L{10,12,14}.json`, `H2_lexsbert_L{10,12,14}_s0.json`

## Remaining high-value work (ranked, all 1-GPU / 0-download)
1. **Run E6** (holdout/generic AlphaEdit) and report `C4_causal_holdout_table.json` as the
   primary causal result. Highest reviewer-risk item; ~72 GPU-min.
2. **Cross-architecture generality** (E1/E2 already staged): confirm the S×C law on Qwen /
   gemma / Phi. The whole result is single-family (Llama-3.2-1B) today.
3. Re-run C1/C4 tables through the upgraded midrank Spearman for consistency with G1_v2.
4. zsRE dataset generality (E5, staged) — second dataset closes the "counterfact-only" gap.
