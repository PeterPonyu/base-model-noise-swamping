# PORTFOLIO REBALANCE — 2026-07-03

> Corrects a 3-day over-focus on P1/editing back toward the PLAN.md charter breadth.
> Grounded in result JSONs on disk, not plan optimism. Synthetic results are NOT partial-real.
> Charter authority: PLAN.md (2026-06-30, breadth-first, independent of Desktop projects).
> Hardware reality: ONE RTX 5090 24GB laptop GPU, SERIAL, occupied all day today by the B6/P1 queue.

---

## 1. Honest reckoning

Three days after a nine-direction breadth charter, exactly ONE direction (P1/B6) has real,
gate-passing evidence — G1 PASS (within-probe partialled ρ 0.395/0.534/0.602/0.301 at
L8/10/12/14) and C4 causal COMPLETE (AlphaEdit removes ~98% of ROME damage) — while it
monopolizes the single CUDA GPU for a third consecutive day. Of the three "started" fission
branches, P4 FAILED its own pre-registered kill-gate on 3/3 real ETT datasets (gate_pass=false
×5, cleaned cross-ρ −0.267/−0.119/+0.064, PICP 0.24–0.56 vs 0.90), P3's single real Ollama run
is self-declared DEGENERATE (4/9 models HTTP-400 on the tool API → observed_diff=NaN, hypothesis
UNTESTED), and P2 has zero real-model evidence (SYNTH-bias/control only; its own n=2 artifact
warns "correlation is not interpretable"; the sampler script its queue jobs invoke does not
exist). P5–P9 are untouched — no code, no data, no written hypothesis — and the charter's
"must fix torchvision" gate on P5–P6 has been stale since torchvision 0.27.1+cu130 verified.
The charter's discipline line was "one training task at a time," not "one direction at a time":
the CPU/Ollama parallel lanes have been idle while the GPU lane ran, and that is the correctable
failure — not the B6 focus itself, which is the portfolio's only headline asset.

---

## 2. Portfolio state table (evidence from disk, 2026-07-03)

| ID | Direction (current variant) | Evidence state | Viab /10 | GPU? | CPU-now? | Cheapest REAL next result | Realistic venue |
|----|------------------------------|----------------|:---:|:---:|:---:|---------------------------|-----------------|
| P1 | Knowledge editing → B6 key-geometry damage predictor | **live-result** (G1 PASS, C4 complete, 07-02/03 campaigns 3-seed) | 8 | yes | yes | 0 GPU-min: CPU reconciliation of MEMIT S×C discrepancy + S×C-vs-raw-key-cos overclaim fix against existing `results/matrices/*.npz` (~1–2 h) | ARR→EMNLP/NAACL main (CCF-B); KBS/TNNLS (SCIE Q1) journal ext. |
| P2 | CoT/RL → pre-RL length-bias diagnostic (numpy) | **synthetic-only** (SYNTH-bias 1.611 vs control 1.007; n=2 "not interpretable") | 3 | yes (real panel) | no (analysis only, adds nothing) | Write missing `sample_ckpt.py` (~150 L, CPU dev now) + 1 gen job Qwen2.5-0.5B ×200 GSM8K ×k=8 ≈ 20–40 GPU-min queued | None near-term; Neurocomputing Q2 stretch iff full pre/post-GRPO panel ever lands |
| P3 | Agentic IPI attack–defense (local Ollama) | **partial-real, DEGENERATE** (n=30×9; 4 models HTTP-400 → diff=NaN, hypothesis UNTESTED) | 4 | no | yes (start Ollama) | 5 tool-capable models × 30 scenarios, defense-on vs defense-off ASR delta table, ~1–2 h, 0 download | TIFS unrealistic; SCI Q3–Q4 / workshop after transport fix + scale-up |
| P4 | Temporal-UQ conformal committee (ETT, Ollama) | **live-result, NEGATIVE** — kill-gate FAILED 3/3 real datasets (5/5 runs incl. mocks) | 2 | no | yes | Exists and is negative. Only honest new run: real ETTh1 with persisted raw decodes (~1–2 h) — expected to reconfirm the kill | IJF dead; ceiling = negative-result SCI Q3–Q4, most such venues fail the filter |
| P5 | Discriminative vision (MVTec/COCO/DOTA) | **untouched** (no code, no data, no hypothesis) | 4 | yes | no | MVTec-AD bottle (~0.5–1 GB, ask-first) + PatchCore pretrained → AUROC in ~10–15 GPU-min | SCI Q3–Q4 fallback (Visual Computer / SIVP); GRSL Q2 with RS angle |
| P6 | Diffusion instruction-editing (SDXL LoRA) | **untouched**; 06-30 audit verdict NICHE/"drop" | 2 | yes | no | Inference-only InstructPix2Pix + MagicBrush subset (~7–10 GB ask-first, 1–2 GPU-h) — reproduction, not a finding | No SCIE/CCF path on current design; Q3–Q4 only if a mechanism angle survives novelty re-check |
| P8 | World-model distillation (LIBERO BC + future-aux) | **untouched**; claimed reuse repo does NOT exist (stale memory) | 2 | yes | no | LIBERO spatial (~4–8 GB, ask-first) + mujoco/EGL setup (~1–2 d) + 8–12 GPU-h BC vs BC+aux | CoRL/ICRA unrealistic; RAS/Neurocomputing-apps ceiling, positive result only |
| P9 | 3D recon adversarial (VGGT PGD fragility) | **untouched**; base direction scooped (AdvSplat) | 2 | yes | no | VGGT-1B (~5 GB, ask-first) + PGD on bundled scenes, confidence-head rank-predicts degradation, 30–60 GPU-min + ~1 d code | Not Q1-competitive; Q3–Q4 / CCF-C ceiling — same tier P4/P5 reach cheaper |

**Reading:** 1 live-positive (P1), 1 live-negative (P4), 1 degenerate-untested (P3),
1 synthetic-only (P2), 5 untouched (P5–P9). GPU-needing = P1 ext + P5–P9, all queued.
Runnable NOW without CUDA = P1 CPU analysis, P2 dev work, P3 Ollama, (P4 only if resurrected).

---

## 3. THE REBALANCE — two concurrent lanes, starting today

### Lane A — GPU (do NOT displace)
B6/P1 queue continues exactly as staged. It is the only SCIE/CCF headline asset; the 07-02
venue-gap resequencing (ARR/EMNLP main first, MEMIT > zsRE promotion > canonical E/G/L >
sequential stress > GPT-2-XL sanity) stands. Rebalancing means filling the OTHER lane, not
starving this one. Standing rules apply: lid open, wait by PID, idle-gate on util+mem.

### Lane B — CPU/Ollama, runnable NOW in parallel (ordered)

**B1. P1 CPU debt (first, ~1–2 h, 0 GPU-min).** Reconcile the MEMIT S×C discrepancy and fix
the S×C-vs-raw-key-cos overclaim against existing `edit-harness/results/matrices/*.npz` +
the 07-03 validation-sweep false-positive. REAL because it re-analyzes real edit matrices
already on disk. Kill-gate: if reconciliation shows the MEMIT S×C table is wrong rather than
mislabeled, the ARR S×C framing must be rewritten before any new GPU cell.

**B2. P3 first real defense table (~2–3 h wall, 0 download).** Start Ollama
(`~/.local/bin/ollama serve`, add to PATH; force CPU serving — `OLLAMA_NUM_GPU=0` /
CUDA_VISIBLE_DEVICES="" — to avoid VRAM contention with Lane A; the documented litchron
persistent-context incident makes this non-negotiable). Run the 5 tool-API-capable models
(qwen2.5:1.5b/7b, qwen3:8b, llama3.1:8b, mistral:7b) over the existing 30 real ToolACE+BFCL
scenarios WITH vs WITHOUT tool-whitelist/memory-isolation. REAL because: live model calls on
real cached scenario data, both arms, no mock backend — this is the PLAN.md first experiment
and is independent of the degenerate lineage hypothesis. Kill-gate (pre-registered before the
run): defense must cut mean ASR by ≥0.20 absolute with a sign-consistent drop in ≥4/5 models
(permutation p<0.05); else the defense angle joins the lineage angle as untested-thin and P3
drops to backlog until the prompt-format transport is written.

**B3. P2 unblock dev (CPU-only authoring, ~half day).** Write the missing `sample_ckpt.py`
(GPU k-sample GSM8K generator its 7 queued gen jobs already reference) and validate it on
CPU with a 2-problem smoke against Qwen2.5-0.5B weights already local. Produces no result
itself but converts P2 from "cannot run" to "one 20–40 GPU-min queued job from its first REAL
D_within." Kill-gate lives at the panel (see §4); do NOT create the dl-rl env or touch the
GRPO leg now.

**B4. P3 transport fix (if B2 completes early).** Implement prompt-format tool-calling for
non-tool-API models so the r1-distill lineage hypothesis becomes testable at all
(observed_diff currently NaN). CPU/Ollama only.

**Explicitly NOT in Lane B:** P4 re-runs (see §5); P2 re-analysis of synthetic data
(adds nothing); any P5–P9 work (all CUDA-gated + download-gated).

---

## 4. Sequenced derivative map — GPU slots after the B6 queue drains

Order by (headline leverage) → (marginal cost of first real result) → (breadth insurance).
Every slot has a cheap pre-registered kill-gate; the ROADMAP rule stands: no signal in 2–3
days → cut.

| Slot | Job | Cost | Kill-gate |
|------|-----|------|-----------|
| G1 | **P1/ARR remainder** (MEMIT panel completion, sequential stress, GPT-2-XL sanity) | already staged | existing per-cell gates; ships the ARR package |
| G2 | **P2 real panel, descriptive half**: `sample_ckpt.py` on Qwen2.5-0.5B, 200 GSM8K × k=8 × 640 tok | 20–40 GPU-min (1 model), 3–4 GPU-h (7-model panel) | ≥3/7 models must show non-degenerate D_within spread (not all-flag / no-flag) with CI separation; else P2 killed without ever touching GRPO |
| G3 | **P5 stack probe**: MVTec-AD bottle + PatchCore pretrained (needs user OK for ~1 GB download) | 10–15 GPU-min | This proves plumbing only. HARD gate: a written novel hypothesis (few-shot / LoveDA change-detection / efficiency claim) must exist BEFORE any second P5 GPU-hour; bare reproduction of a 99%-AUROC 2022 baseline earns zero slots |
| G4 | **P8 designated second front** (PORTFOLIO's own naming), only post-ARR-submission: LIBERO spatial BC vs BC+future-aux at reduced epochs (needs user OK, ~4–8 GB) | ~1–2 d integration + 8–12 GPU-h | aux head must beat BC by ≥5 pts success with seed×3 non-overlap AND survive the shuffled-future-targets causal ablation; either failure = cut with no consolation paper (its audit says so) |
| G5 | **P9 fragility gate** (only if G3/G4 both die and breadth insurance is still needed): VGGT-1B PGD, 10 scenes | 30–60 GPU-min + ~1 d code, ~5 GB ask-first | confidence head must rank-predict degradation, Spearman perm-p<0.05 across ≥7/10 scenes; fail = total write-off, pre-acknowledged |
| — | **P6**: no slot | — | P6 pre-killed by the 06-30 audit (NICHE/contested/no moat) — needs a written novel angle to earn queue entry |

P2's GRPO validation leg (≥3 RL runs) gets NO slot this cycle: the branch's own n=2 artifact
says the correlation is uninterpretable below n=3, and 3+ GRPO runs on a serial 24 GB GPU
before ARR ships is exactly the sprawl the charter forbids.

---

## 5. KILL / de-scope (by their own gates — CP-Edit/D3 precedent)

- **P4 temporal-UQ: KILLED.** gate_pass=false on ALL 5 runs; 3/3 real ETT datasets negative
  (cleaned ρ −0.267 / −0.119 / +0.064; PICP 0.24–0.56 vs 0.90; interval widths blown to
  1e2–1e149; delta-vs-point CIs all straddle 0; raw decodes discarded → nothing repairable).
  Do not re-run the same design; do not present as viable. Resurrection requires a NEW
  disagreement statistic + persisted decodes + real ETTh1, and any new statistic tested on
  the same 3 negative datasets is post-hoc fishing. IJF target: dead.
- **P3 lineage-vs-architecture claim: DE-SCOPED to untested.** Not null — NaN. The r1-distill
  arm does not exist until the transport fix (B4). P3 survives only as the defense-table
  experiment (B2) with its own fresh gate; TIFS framing retired at this maturity.
- **P2 GRPO/validation leg: DE-SCOPED** (compute-infeasible before ARR; descriptive panel G2
  is the whole near-term branch). The synthetic "pass" is a self-test of injected bias and
  carries zero evidential weight.
- **P6 diffusion editing: KILLED for the near-term queue** per the lab's own 2026-06-30 deep
  audit (NICHE, novelty CONTESTED, moat WEAK, "drop"). Re-entry requires a surviving
  mechanism angle (cross-attention-locality → LoRA leakage) through a fresh novelty check.
- **Standing kills unchanged:** CP-Edit (failed KG-0, do not resurrect), D3 router-as-artifact
  framing (degenerate 12/12; reframed to benefit-magnitude only).

---

## 6. Recommendation — single highest-value breadth action, next 24 h

Start the Ollama server CPU-pinned and run **B2: the P3 five-model attack–defense table**
tonight, in parallel with the untouched B6 GPU queue. It is the only action in the whole
portfolio that converts a non-P1 direction from "no interpretable real evidence" to "one
clean, pre-gated, two-arm real result" at literally zero download, zero CUDA contention, and
~2–3 h wall — P2's first real number needs a GPU slot that doesn't exist today, P4 is dead by
its own gate, and P5–P9 are download- and GPU-gated. If the defense delta passes its gate
(≥0.20 ASR drop, ≥4/5 models, perm p<0.05), the portfolio has a second live direction and the
breadth charter is factually — not rhetorically — restored; if it fails, P3 is honestly
parked and the 24 h cost was two CPU-hours nobody else was using. Fold B1 (P1's MEMIT-S×C
CPU reconciliation) into the same evening: it is the cheapest protection of the one headline
asset and shares zero resources with B2.
