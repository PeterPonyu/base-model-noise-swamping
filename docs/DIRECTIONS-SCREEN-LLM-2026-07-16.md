# Feasibility screen — five LLM/agent research directions for a solo researcher (2026-07-16)

> Research-only screen. Live-web landscape (2025–2026) + honest local-infra inventory. No
> code written, no experiments launched. Purpose: rank five user-supplied directions by
> **feasibility × niche-survival × venue-reach** for THIS researcher (one 24GB laptop GPU +
> rentable single 4090D boxes, SCIE-journal venue standard), and name the one E0 spike to
> run first.
>
> **Provenance note.** Landscape gathered by five parallel Sonnet web-research agents +
> targeted venue searches by the lead. arXiv IDs are passed through as found; several
> load-bearing "gap" papers are very recent preprints (2604.*, 2606.*, 2607.* = Apr–Jul
> 2026) with thin citation records — treat them as *evidence the niche is filling*, not as
> vetted results. Nothing here is fabricated; where a claim rests on a single preprint it is
> flagged.

---

## 0. The one fact that reorders everything: venue class vs. the SCIE standard

All five directions are, as the user phrased them, **conference-genre topics.** The defining
2025–2026 works cited throughout this document are almost entirely arXiv → **CVPR / ICLR /
NeurIPS / ICCV / ICML / ACL / EMNLP** (Vision-R1 → ICLR 2026, R1-VL → ICCV 2025, ViGoRL →
NeurIPS 2025, ACTIVE-o3 → ICML 2026, …). A survey of 26K CV papers found VLM work rose from
16% of abstracts in 2023 to **40% in 2025** — this is the single most crowded region of ML
right now, and it lives in a venue class the user's own standard **forbids as a terminal
venue.**

The portfolio's venue rule (CLAUDE.md 2026-07-11; `ml-reliability-research/docs/records/
VENUE-RELIABILITY-SCAN-2026-07-16.md`): **SCIE-indexing is the hard filter; conference-only
venues — NeurIPS/ICLR/CVPR/ACL/EMNLP/COLM/TMLR — do NOT satisfy it on their own.** That rule
applies here in full force and is the dominant screening variable:

- **A "new-SOTA VLM-R1" contribution has no compliant home.** Its natural venue is a
  conference; journals in this space (IEEE TPAMI/TNNLS/TIP/TMM/TCSVT, Pattern Recognition,
  Information Fusion, Information Sciences, Knowledge-Based Systems, Neurocomputing) reward
  **systematic study / benchmark / reliability-audit / method-with-thorough-evaluation**,
  not a leaderboard delta that a big lab will beat next month.
- **The reliability/certification angle is therefore not a "nice cross-link" — it is the
  venue-survival mechanism.** Turning "hot VLM topic" into "uncertainty-quantified,
  conformal-guaranteed, abstention-calibrated study of a hot VLM topic" is exactly the move
  that (a) converts a conference-race into a journal-shaped paper and (b) reuses the
  portfolio's existing machinery. Confirmed journal appetite: Pattern Recognition and
  Information Sciences already publish conformal/selective-classification work (venue scan
  Q2); Information Fusion is the natural home for multimodal-fusion + uncertainty; a
  data-driven **inductive-conformal calibration of prediction sets in large VLMs** preprint
  already exists (arXiv 2504.17671), and "risk control for MLLMs" reached ICLR 2025 — the
  genre is real and journal-portable.

**Consequence for the ranking:** the winner is whichever direction most naturally yields a
**journal-shaped reliability contribution with an inference-only or near-inference-only E0**,
because (i) that is the only compliant venue path and (ii) it is the only compute profile a
solo can actually de-risk on this hardware.

---

## 1. Honest infra inventory (what actually applies)

**Hardware.** One RTX 5090 Laptop, **24 GB VRAM** (Blackwell CC12.0, driver 580), Core Ultra
9 275HX 24-core, 62 GB RAM, ~579 GB free disk. Rentable single **4090D** boxes at ~¥2/h
(AutoDL); one Pro-6000 wave has been used before. `torch 2.12.1+cu130`, `transformers
5.12.1`, `trl 0.24.0`.

**⚠ Blocking issue for four of five directions.** Per README §0, the **vision stack in the
`dl` env is currently broken** (`torchvision::nms does not exist`, torchvision/torchao vs
torch 2.12.1 mismatch — drags down `peft`/`diffusers`/any object-detection import). *Every
VLM direction here must fix this first (~10 min reinstall).* Pure-text LLM work is
unaffected. This is a real, present tax on directions 1–5 and a reason to prefer the
inference-first spike that stresses it least.

**Local model fleet.** Ollama fleet is **all-text** (qwen2.5/3/3.5, llama3.1/3.2, phi4,
mistral family, gemma2/3, deepseek-r1, glm4, internlm2, yi, aya-expanse, …) — **no VLM in
Ollama, no VLM in the HF cache** (cache is embedding/reranker + text-eval datasets). Any VLM
(Qwen2.5-VL-3B/7B, InternVL, LLaVA-NeXT) is a **fresh download** (ask-before-large-download
rule applies). unsloth 4-bit adapters cached for Llama-3.2-1B, Llama-3.1-8B, Qwen3-8B (text).

**Directly reusable assets.**
- `edit-harness/` + `fission-engine/` — a **GPU-serial job queue** (`queue.py/runner.py/
  gpuguard.py`, gates on nvidia-smi, wait-by-PID discipline). This is a ready-made harness
  for running a serial batch of VLM-inference or QLoRA cells on the single GPU without
  babysitting — reuse it verbatim for any E0 here.
- **Conformal / CRC / abstention machinery** — `ml-reliability-research/reliability-commons/
  relmetrics/conformal.py` (+ tests), plus CRC / risk-coverage / abstention-curve code across
  the geospatial, causal-policy, materials, and structured-data repos. **This is the bridge
  substrate** for every reliability-framed spike below.
- Cached eval/reasoning datasets (gsm8k, bbh, gpqa, mmlu, IFEval, TruthfulQA, function-
  calling suites: BFCL, APIBank, ToolACE, xlam-60k, gorilla APIBench). Text-side only — no
  VQA/visual-reasoning set is cached; those download fresh.
- unsloth **vision-RL** now supports Qwen2.5-VL / Gemma-3 GRPO with large claimed VRAM
  savings (Qwen2.5-VL-7B GRPO demonstrated on 16 GB) — the enabling tech for a *narrow*
  QLoRA-GRPO cell on 24 GB.

**Solo compute envelope (the number every verdict is derived from).** On 24 GB: VLM
**inference** to 7–14B comfortable; **QLoRA/LoRA SFT** on 3–7B VLM comfortable (single-digit
GPU-h for a few-K-example run); **QLoRA-GRPO** on a **3B** VLM feasible for a *narrow*
rule-verifiable task, **7B GRPO at the ragged edge** (multimodal KV-cache inflation) and
likely needing the rented 4090D for headroom. **Out of reach, full stop:** multi-turn
agentic RL of a 7B VLM (documented at 200–1500 GPU-h on 8–32 H100 for DeepEyes/ACTIVE-o3/
UI-TARS-2), any PRM-dataset build at VisualPRM400K scale, any 38B+ RLVR, any "unified
framework" spanning modalities.

---

## 2. Per-direction screens

### Direction 4 — Adaptive CoT + visual reasoning  → **RANK 1**

*(screened out of numeric order because it is the recommendation)*

**(a) Landscape & crowding.** Three sub-fronts, very different heat:
- Adaptive visual **zoom/search + RL** (Chain-of-Focus 2505.15436, DeepEyes, VLM-R3) — hot,
  big-lab-raced, **crowded**.
- **Text-only** adaptive-length / anti-overthinking (REFRAIN 2510.10103 training-free early-
  stop; GRPO-λ 2505.18086; CODA 2603.08659; SAT 2604.07922; LEAD 2605.09806; survey "Don't
  Overthink It" 2508.02120) — **very crowded**.
- **Conformal / calibrated abstention on reasoning** — small, recent: **Pause and Reflect:
  Conformal Aggregation for CoT** (2605.14098) calibrates an abstention rule via conformal
  risk control with a finite-sample bound on the confident-error rate; Conformal Path
  Reasoning (2605.08077, KG-QA). Per-step multimodal confidence exists (MMBoundary 2505.23224)
  but *without* conformal guarantees.
- **The uncontested intersection:** the research agent found **no 2025–2026 paper applying
  conformal prediction / calibrated abstention to *visually-grounded* CoT steps** — i.e.
  using a per-step grounding signal (bbox-IoU variance across resamples, region-attention
  entropy, step log-prob margin) as the conformal nonconformity score to bound the
  "confidently-wrong grounded step" rate. VGR/CoF-style grounding × MMBoundary-style per-step
  confidence × Pause-and-Reflect-style CRC = genuinely open.

**(b) Compute verdict.** **The most feasible of the five: inference-only.** Use a released
grounded-CoT checkpoint (CoF's Qwen2.5-VL-7B, or prompt stock Qwen2.5-VL-7B into
locate→reason→answer), sample k=6–8 rollouts over ~1.5K calibration + ~500 test examples via
batched vLLM; conformal calibration is CPU-only post-hoc statistics. **<10–15 GPU-h, no
training.** Out of reach (and unnecessary): reproducing CoF's ~280 A100-h SFT+RL.

**(c) Infra fit — the best in the set.** The conformal/CRC layer *is* the portfolio's
existing `relmetrics/conformal.py` + risk-coverage/abstention machinery applied to a new
(multimodal, per-step) nonconformity score. The `fission-engine` queue runs the rollout
batch. Only new download: one 7B VLM checkpoint + a grounded-VQA set (MagiC 2507.07297, or
GQA/A-OKVQA region annotations).

**(d) Venue reach — the best in the set.** This is *natively* a journal contribution:
distribution-free selective-prediction with coverage guarantees on a multimodal reasoner.
Compliant homes with demonstrated appetite: **Pattern Recognition** (conformal e-prediction,
rank-based conformal sets — venue scan), **Information Sciences** (conjunction-subspaces
conformal + selective classification), **Information Fusion** (multimodal + UQ),
**Neurocomputing / KBS**. This does not need a conference.

**(e) E0 spike (<15 GPU-h).** *"Conformal abstention on visually-grounded CoT steps."*
Released grounded-CoT VLM → per-step grounding score (≥2 candidates) → split conformal risk
control → risk-coverage curve vs. a plain final-answer-only conformal baseline. **Kill
criteria:** (1) no per-step signal reaches AUROC ≥0.55 for step-correctness → conformal can't
rescue an uninformative score (the "score-separability" precondition of 2605.14098); (2)
hitting a 10% confident-error bound forces abstention on >50% of steps → not separable
enough; (3) grounded-CoT output fails to parse on >30% of a 50-example smoke → swap backbone
before burning hours; (4) grounded per-step conformal fails to beat the response-level
conformal baseline → the visual-grounding angle adds nothing, reroute.

**(f) Cross-link.** *This IS the bridge paper.* It is "certified abstention for reasoning /
conformal risk on CoT steps" instantiated on a VLM — the exact machinery the ml-reliability
portfolio already ships, extended to a hot new surface. Highest strategic leverage.

---

### Direction 2 — Active perception + tool-using agents  → **RANK 2**

**(a) Landscape & crowding.** The core recipe (VLM + GRPO + zoom/crop/search tool) is
**very crowded and big-lab-raced**: OpenAI o3 "thinking with images" (Apr 2025, proprietary),
then DeepEyes (2505.14362), Pixel-Reasoner (2505.15966), ViGoRL (2505.23678), Chain-of-Focus
(2505.15436), ACTIVE-o3 (2505.21457) — **five within one week of May 2025**, a textbook
crowding signal — plus PyVision (2507.07998), SenseNova-MARS (2512.24330), Skywork-R1V4.
Training a better zoom policy is saturated *and* out of reach.
- **Surviving niche = reliability of active perception, and it is thin but starting to
  fill:** abstention ("Knowing When Not to Answer" / MoHoBench 2604.14799 — eval only, not
  connected to the tool setting); tool-call calibration (ToolGate 2606.03054; "LLM Agents
  Already Know When to Call Tools" 2605.09252; uncertainty-aligned tool-calling 2606.06976 —
  all June 2026, all single-lab); faithfulness of visual thinking (2510.23482; DeFacto
  2509.20912 — do the crops the model "looked at" causally drive the answer, or post-hoc
  rationalization?). All measurement/audit, none from OpenAI/Qwen/ByteDance yet.

**(b) Compute verdict.** Feasible **inference-first**: eval existing checkpoints (DeepEyes-7B
released; Qwen2.5-VL-7B with a prompted zoom-tool harness) across V*/HR-Bench/TIR-Bench —
single-digit GPU-h. One light QLoRA "when-to-call / when-to-abstain" adapter (~5–8 GPU-h).
Out of reach: multi-turn agentic RL training of the tool policy (400–1500 GPU-h, 8–32 H100).

**(c) Infra fit.** Same conformal/abstention machinery reused as a **tool-call gate** and
**abstention-after-failed-tool** calibrator. `fission-engine` runs the eval batch. Downloads:
DeepEyes-7B + benchmarks.

**(d) Venue reach.** Journal-shapeable as "uncertainty-calibrated tool-use for multimodal
agents" (Information Fusion / KBS / ESWA / Neurocomputing). Slightly behind Direction 4
because the story is calibration-of-a-decision rather than a clean coverage guarantee, and
the niche is visibly filling (three June-2026 preprints).

**(e) E0 spike (<20 GPU-h).** *Tool-call-calibration audit:* 2–3 open zoom-tool VLMs →
stratified set (needs-zoom / needs-no-zoom / unanswerable) → measure tool-call precision/
recall, abstention rate, and a DeFacto-style crop-swap faithfulness check (inference-only,
~5–8 GPU-h); then one QLoRA calibration adapter (~5–8 GPU-h). **Kill:** (1) can't reproduce
published tool-call behavior from a checkpoint within 2 h → stop; (2) baselines already
>90% correct on the tool-decision split → no exploitable gap; (3) QLoRA lifts tool-call F1
/ abstention <5 pts → fix needs RL/data beyond solo scale, stop.

**(f) Cross-link.** "Certified abstention" applied to *acting* (abstain from a tool call /
from answering after tool failure) — a natural sibling of Direction 4.

---

### Direction 1 — Process-consistent RL for trustworthy multimodal reasoning  → **RANK 3**

**(a) Landscape & crowding.** "Apply R1/GRPO to a VLM" is a **template genre** (MM-Eureka
2503.07365, Vision-R1 2503.06749, VLM-R1 2504.07615, R1-VL/StepGRPO 2503.12937, VisualPRM
2503.10291 + VisualPRM400K, dozens of domain "-R1" spinoffs) — crowded and resource-gated
(you cannot match cluster-scale PRM-data construction).
- **Solo-shaped niches (real):** process-vs-outcome **faithfulness auditing** of GRPO-trained
  CoT (SPD-Faith 2511.08409; GeoFaith 2605.26893; text-only "Linking Process to Outcome"
  2509.26578 — thin); **reward-hacking of visual CoT** under rule-based rewards (acknowledged,
  never cleanly audited on a small model); **small-scale negative-result GRPO studies** —
  direct precedent exists (2607.12640, an 18-run controlled null on GRPO failure in a 4–8B
  VL web agent). Non-math process supervision (spatial/counting/chart faithfulness) also
  under-served.

**(b) Compute verdict.** QLoRA-GRPO on **Qwen2.5-VL-3B** for a narrow rule-verifiable task is
in reach (community R1-V got OOD-counting generalization in ~single-digit GPU-h; budget
<20 GPU-h for a focused spike, 2–4× the A100 wall-clock on one consumer card). **7B GRPO at
the ragged edge; 38B/78B RLVR and PRM-dataset builds out of reach.**

**(c) Infra fit.** unsloth vision-RL + `fission-engine` serial queue. **Higher infra risk
than 4/2:** requires the broken vision stack fixed *and* a working multimodal GRPO loop at
the 24 GB edge — the one direction whose E0 depends on training actually converging.

**(d) Venue reach.** The audit/negative-result framing (not the "-R1" splash) is journal-
shapeable (TNNLS / Neurocomputing / KBS / Pattern Recognition). But it overlaps heavily with
Directions 4 and 5, and the pure-RL versions are conference-native.

**(e) E0 spike (<18 GPU-h).** *"Does GRPO raise accuracy while degrading CoT faithfulness?"*
Qwen2.5-VL-3B + QLoRA-GRPO on a ~500–2K CLEVR-style counting subset (free rule reward) →
faithfulness via scene-graph hallucination scan + counterfactual image-swap, pre vs post RL,
≥2 seeds. **Kill:** (1) reward flat in first 50 steps/3 GPU-h → pipeline broken; (2)
faithfulness Δ <3 pts, CIs overlap → report as null (precedent: 2607.12640); (3) clean
ground-truth check not assemblable in ~2 h → pivot domain; (4) hard stop at 20 GPU-h.

**(f) Cross-link.** Faithfulness/consistency of a *certified* reasoning process — feeds the
same "process ≠ outcome" thesis as Directions 4/5.

---

### Direction 3 — Long-horizon GUI / mobile / embodied agents  → **RANK 4**

**(a) Landscape & crowding.** Foundation-scale GUI agents are a **big-lab arms race**
(UI-TARS 2501.12326 / UI-TARS-2 2509.02544 with fleet-scale multi-turn RL; Aguvis 2412.04454;
OS-Atlas 2410.23218; ShowUI 2411.17465; Qwen3-VL grounding; Claude/OpenAI computer-use) —
uncompetitive solo.
- The obvious reliability niche is **already filling fast:** per-click grounding calibration
  now has SafeGround (2602.02419, finite-sample risk), HyperClick (2510.27266, Brier-
  calibrated confidence), UI-Zoomer (2604.14113), Zoom-Consistency (2604.15376), and a UQ
  benchmark (2606.25760) — **per-click is no longer virgin territory.**
- **Thinner surviving niche:** *trajectory-level* reliability — agent-vs-environment failure
  attribution, **when-to-stop / abstain mid-task**, cheap cross-model diagnosis harnesses
  (DiagEval 2605.17439; GUITester 2601.04500; GUIDE 2604.04399; "When Actions Go Off-Task"
  2602.08995; VeriOS 2509.07553; Adaptive Milestone Reward 2602.11524).

**(b) Compute verdict.** GPU side is cheap (QLoRA grounding fine-tune of 3–7B; eval is VLM
inference). **The real tax is the ENVIRONMENT, not the GPU:** OSWorld needs full desktop VM
snapshots (4–8 GB RAM each, macOS licensing headaches, wall-clock-bound stepping over 369
tasks); AndroidWorld needs an emulator (lighter, one instance fine on a laptop; RL-scale
parallel rollouts need a fleet). This orchestration burden is a **structural solo bottleneck**
and the reason to prefer AndroidWorld-only or a static-log study for any first pass.

**(c) Infra fit.** Conformal/selective-prediction machinery reuses cleanly for grounding
calibration — but that sub-niche is the crowded one. The thin niche (trajectory when-to-stop)
needs env infra the portfolio does not have.

**(d) Venue reach.** Genre is conference-native (OSWorld/AndroidWorld leaderboards). Journal-
shapeable via a reliability/diagnosis framing (KBS / ESWA / Information Fusion take agent-
reliability), but with more friction than 4/2.

**(e) E0 spike (<18 GPU-h).** *Selective-grounding calibration harness* (training-free
first): ScreenSpot-v2/Pro (no env infra) × 2–3 open 7B grounding VLMs → 5-sample coordinate
spread + verbalized confidence + logit entropy as confidence signals → isotonic/conformal
calibration → risk-coverage curves. **Kill:** (1) zero-shot grounding <30% for all models →
testbed too hard, swap; (2) no signal AUROC >0.65 after ~4 GPU-h → uninformative, stop; (3)
QLoRA calibration doesn't improve ECE / drops accuracy >5 pt → keep training-free result;
(4) by ~15 GPU-h no curve beats no-abstention baseline → per-click is saturated (SafeGround/
HyperClick cluster), pivot up to trajectory-level using existing OSWorld/AndroidWorld logs.

**(f) Cross-link.** Certified abstention for *action* under long horizons; strongest embodied
tie to `robotics-embodied-reliability-research`, but weakest venue/infra fit here.

---

### Direction 5 — Unified process modeling for cross-modal R1 post-training  → **RANK 5**

**(a) Landscape & crowding.** **Extremely crowded and survey-saturated** (Awesome-Multimodal-
Reasoning tracks 50+ papers): Vision-R1 (2503.06749), LMM-R1 (2503.07536, explicit text→vision
transfer), Open Vision Reasoner (2507.05255), URSA (2501.04686), CORA (2606.14691,
thinking-answer consistency), VLM self-correction via rollout augmentation (2602.08503), plus
the skeptical text-RLVR spine every multimodal paper *should* cite but mostly doesn't: "Does
RL incentivize reasoning beyond the base model?" (2504.13837, pass@k), "Spurious Rewards"
(2506.10947, GRPO improves under random rewards via clipping bias — Qwen-family-specific),
"Limits of Generalization in RLVR" (2510.27044).
- **The "unified framework spanning reasoning+consistency+self-correction+generalization" is
  precisely what big labs are converging on — a solo cannot own it.** The only solo-shaped
  carve-out is a **narrow, rigorously-controlled negative/replication result**: do the
  spurious-reward and pass@k controls (near-mandatory in text now) hold in the *multimodal*
  setting? Is VLM "self-correction" genuine or a Qwen-clipping-bias artifact? Does text-RL
  actually transfer to vision on the *same* base model under a decontaminated pass@k?

**(b) Compute verdict.** Same 3B QLoRA-GRPO envelope as Direction 1 (<20 GPU-h for the narrow
control study; inference-only behavioral audits of released checkpoints are cheapest). **The
"unified framework" ambition is multi-A100-months — out of reach, full stop.**

**(c) Infra fit.** Same as Direction 1 (unsloth vision-RL + queue), same infra risk, same
broken-vision-stack dependency.

**(d) Venue reach.** A clean multimodal negative-result is journal-publishable (TNNLS /
Neurocomputing), but the framing is a **scope trap**: the interesting-sounding version is
unreachable, and the reachable version is a short controlled-ablation paper that overlaps
Direction 1. Lowest venue-reach-per-unit-risk of the five.

**(e) E0 spike (<20 GPU-h).** *Spurious-reward + pass@k control, transplanted to multimodal:*
Qwen2.5-VL-3B, three short GRPO runs — real-reward multimodal, real-reward text-only (frozen
vision tower, test transfer to held-out vision eval), spurious/random-reward control — +
pass@k (k=8–16) pre/post. **Kill:** (1) results just reproduce 2504.13837's pass@k ceiling
with no modality-specific signal after n≈50–100 → non-novel replication, stop; (2) spurious ≈
real with no vision-specific mechanism → Qwen artifact transplanted, pivot; (3) per-step
wall-clock >90 s → downscale; (4) decontaminated visual eval not assemblable in 2 h → drop
the generalization framing, fall back to inference-only audit.

**(f) Cross-link.** Consistency/self-correction of a certified process — thematically closest
to the portfolio's "process ≠ outcome" thesis, but the ambition/scope mismatch makes it the
riskiest place to start.

---

## 3. Ranking (feasibility × niche-survival × venue-reach)

| Rank | Direction | One-line verdict |
|---|---|---|
| **1** | **D4 — Adaptive CoT + visual reasoning** | **Run this first.** Uncontested intersection (conformal abstention × visually-grounded CoT steps), inference-only <15 GPU-h E0, directly reuses `relmetrics/conformal.py`, natively journal-shaped (Pattern Recognition / Information Sciences / Information Fusion). Best on all three axes. |
| **2** | **D2 — Active perception + tool-using agents** | Training race is saturated & out of reach, but the reliability sub-niche (abstention-from-acting, tool-call calibration, visual-thinking faithfulness) is thin, inference-first, and UQ-journal-shapeable. Slightly behind D4 (calibration story, not a coverage guarantee) and visibly filling. |
| **3** | **D1 — Process-consistent RL** | Real solo niche in CoT-faithfulness / reward-hacking *audits* and small-scale negative results (direct precedent), but the E0 requires a converging QLoRA-GRPO loop at the 24 GB edge (higher infra risk) and overlaps D4/D5. |
| **4** | **D3 — Long-horizon GUI/mobile agents** | Per-click reliability already crowded (SafeGround/HyperClick/UI-Zoomer); the thin niche (trajectory when-to-stop) carries a heavy OSWorld/Android-emulator environment tax that is a structural poor fit for a solo laptop; conference-native genre. |
| **5** | **D5 — Unified cross-modal R1 post-training** | Ambition trap: the "unified framework" is multi-A100-months and big-lab-owned; only a narrow controlled negative-result is solo-reachable, and that overlaps D1. Least journal-shaped, highest scope-mismatch risk. |

**Cross-cutting truth (state it plainly to the user):** none of the five satisfies the
SCIE-journal standard *as a conference-genre topic.* All five must be journal-shaped through
a reliability/certification lens, and the portfolio already owns that lens. The ranking is
essentially "how cleanly does each direction convert into a certified-abstention /
conformal-risk contribution with an inference-only E0" — which is why the two inference-first,
conformal-native directions (D4, D2) top it and the two RL-training-native, big-framework
directions (D5, and the RL half of D1) sit at the bottom.

---

## 4. The single best E0 spike to run first

**D4 — "Conformal abstention on visually-grounded chain-of-thought steps."**

- **Why this one:** it is the only spike that scores top on all three axes simultaneously —
  (i) *feasibility:* inference-only, <15 GPU-h, no training convergence to babysit, minimal
  exposure to the broken vision stack; (ii) *niche survival:* the conformal × per-step-visual-
  grounding intersection is currently uncontested (the text-only conformal-CoT cluster and the
  multimodal per-step-confidence cluster exist *separately*); (iii) *venue reach:* it is
  natively a distribution-free selective-prediction paper for journals that demonstrably buy
  this genre.
- **Why now:** it doubles as a **bridge paper** — a direct extension of the ml-reliability
  portfolio's certified-abstention / conformal-risk machinery onto the hottest surface in ML,
  using code the portfolio already ships. A positive result seeds a paper; a null result is
  still a clean "score-separability fails for grounded VLM steps" finding.
- **First concrete step (before any GPU):** fix the `torchvision` env (~10 min), then stand
  up the released grounded-CoT checkpoint + one grounded-VQA set behind the `fission-engine`
  queue and run the 50-example parse smoke — the cheapest gate (kill-criterion 3) before
  spending the rollout budget.

**Kill the spike (and reconsider D2 as the fallback) if:** no per-step grounding signal
reaches AUROC ≥0.55 for step-correctness on the calibration set — that means conformal has
no separable score to work with on this surface, and the effort should move to D2's
tool-call-calibration audit, which tests the same abstention thesis on a different (decision-
level) signal.
