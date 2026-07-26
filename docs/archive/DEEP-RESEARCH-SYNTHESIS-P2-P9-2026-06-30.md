# DEEP-RESEARCH SYNTHESIS — broader local pool P2–P9 (2026-06-30)

> Source: 8-agent Sonnet-5 deep-research workflow (live web/arXiv + on-disk verification of local readiness). P1 (knowledge editing) covered in the sibling `DEEP-RESEARCH-SYNTHESIS-2026-06-30.md`.

## The one cross-cutting insight

The portfolio's real moat is **not any single topic** — it is the workspace's own signature template: **"predict an expensive process's damage/side-effect from a cheap pre-process diagnostic, architecture-conditioned, with a permutation-null gate."** Every strong P-direction below is that template transplanted to a new modality (P2 pre-RL→overthinking, P5 VLM→detector-FP, P6 frozen-attention→LoRA-leakage, P9 VGGT-confidence→fragility). The genuine *asset* moat (51-model Ollama zoo + cached datasets) only truly powers **P2/P3/P4**. P6/P8/P9 are cold starts with **no local-asset advantage** — a strong deprioritization signal for a solo researcher.

## Ranked bets

| Rank | Dir | Verdict | Novelty | Moat | Best sub-seam (one line) |
|---|---|---|---|---|---|
| **1** | **P4** temporal UQ | VIABLE | **OPEN** | strong (free 15-model ensemble) | cross-architecture committee disagreement as calibrated UQ for zero-shot numeric TS forecasting (ETT) vs single-model resampling null |
| **2** | **P3** agent-safety | VIABLE | CONTESTED | strong (lineage design only a many-local-model owner runs free) | lineage-vs-architecture fingerprint of agentic **indirect-prompt-injection** susceptibility |
| **3** | **P2** CoT/RL | VIABLE | CONTESTED | medium (7 checkpoints) | cheap **pre-RL** diagnostic predicts **post-GRPO overthinking gap**, architecture-gated (mirrors the editing line's identity) |
| **5** | **P5** vision | VIABLE | CONTESTED | medium (gemma3-vision) | confound-controlled audit: does a VLM plausibility judgment carry detector-FP signal **beyond** confidence+base-rate |
| — | **P6** diffusion | **NICHE** | CONTESTED | weak | frozen cross-attention locality predicts post-LoRA background leakage — but zero diffusion assets on disk, crowded eval field |
| — | **P8** world-model/sim | **NICHE** | CONTESTED | **none** | compute-matched causal-necessity ablation (real vs shuffled vs noise future-target) on LIBERO — hyper-crowded, cold start |
| — | **P9** 3D recon | **NICHE** | CONTESTED | **none** | VGGT confidence head predicts adversarial fragility — crowded (AdvSplat "first systematic study"), no consolation paper if gate fails |

## Per-direction essentials (with scoop + readiness)

**P4 — cross-architecture committee disagreement as forecasting UQ (the only OPEN verdict).**
- Turn the 51-model zoo into a free deep ensemble; test whether architectural-diversity disagreement beats single-model temperature-resampling disagreement at flagging forecast error on ETT. Pure inference, 0 training, 0 broken deps.
- Nearest misses (checked, don't scoop): QuantSightBench 2604.15859 (single-model self-elicited PIs), DiscoUQ 2603.20975 (agent tasks, not TS), ZooCast 2509.04208 (TS-FM routing, not disagreement-UQ). Window is closing (both landed <3 months ago).
- Readiness: 51 models verified; **no vision Ollama models → AIGC-image-detection is OUT** (zoo is text-only); ETT not cached (~150MB fetch). Caveat: it's the "SCI-safe fallback" — a solid audit paper, not a flagship.

**P3 — lineage-vs-architecture fingerprint of agentic IPI.**
- Local R1-distill checkpoints are **all Qwen-lineage** (`deepseek-r1:8b` = Qwen3-based, not Llama — corrects a naive cross-family design). Design: 3 R1-distills × matched vanilla bases × out-group families, item-level attack-success correlation, permutation test. Zero download, inference-only.
- Scoop watch: 2506.12913 (jailbreak-transfer, chat-only), 2601.03868 (32-model safety, chat-based). CONTESTED + churny → needs fast execution.

**P2 — pre-RL diagnostic → post-GRPO overthinking (best continuity with the editing line).**
- 7 fp16 checkpoints on disk (~30GB, 0 dl). **RESOURCES.md is WRONG: no 8B base exists on disk** (P1 used 0.5–3B); the 7-checkpoint panel is the real asset (and a better fit).
- ⚠️ **Env blocker (verified live):** `trl 0.24.0` GRPOTrainer/DPOTrainer **fail to import** (`No module named 'mergekit'`); `unsloth 2026.3.8` fails to import (transformers 5.12.1 too new). Do **NOT** `pip install mergekit` — it downgrades accelerate/hub/safetensors/pydantic in the **shared `dl` env** other projects use. Fix = lazy-import source patch **or clone `dl`→`dl-rl`**. n=7 is underpowered → effect-size-first.

**P5 — VLM-plausibility-vs-detector-FP audit.**
- **RESOURCES.md STALE: torchvision is NOT broken** (0.27.1+cu130 imports fine). COCO val2017 only ~1.3GB (not 20GB). gemma3-vision local. Kill-gate: ΔAUROC≥0.03 & perm-p<0.05 over the detector's own confidence+base-rate, else redundant (the editing line's "probe-marginal leakage" trap).

**P6 / P8 / P9 — NICHE, deprioritize for now.**
- P6: zero diffusion assets cached; VLM-judge niche crowded (ADIEE/ImgEdit-Judge/GEditBench-v2); only moat is gemma3-judge + paraphrase-ensemble (secondary).
- P8: **cited reuse repo `robotics-embodied-reliability-research` does not exist on disk** (stale-memory pattern again); no sim installed; hyper-crowded (VPP→AtomVLA→FutureVLA→Cosmos→2606.07687); mujoco headless setup risk.
- P9: crowded (AdvSplat + 5 more in ~10 months); VGGT-1B ungated 5GB but everything else cold; **no consolation paper if the kill-gate fails** (unlike B6's Qwen-null).

## Recommended actions

1. **Two clear STRONG candidates to actually start** (both inference-only, exploit the zoo, no broken deps, cheap decisive kill-gate): **P4** (only OPEN) and **P3** (best moat leverage). Either can be *prepped now on CPU/web* (fetch ETT / build the IPI scenario set + scorer) while the GPU sweep runs.
2. **P2 needs an env fix first** — clone `dl`→`dl-rl` (do not mutate shared `dl`); then it's the most methodologically continuous with the editing flagship.
3. **Correct RESOURCES.md** — three verified-false claims: (a) P2 "reuses an 8B base @0 download" (no 8B on disk); (b) P5 "torchvision broken" (it works); (c) P8 reuse-repo exists (it doesn't). Same overstatement/stale-doc pattern flagged in the editing-line and reliability-portfolio audits.
4. **Drop P6/P8/P9 from the near-term queue** — no asset moat, crowded, cold-start cost; revisit only if a specific mechanism-audit angle survives a fast re-check.
