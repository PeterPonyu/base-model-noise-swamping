# Locally-feasible directions —— independent action plan

> Independent evaluation: based only on (1) Vaughn's idea list (2) measured local hardware. Does not reference any existing projects on the Desktop.
> Machine: RTX 5090 Laptop **24 GB single card** · 24 cores · 62 GB RAM · 579 GB free disk · torch 2.12.1+cu130.
> The only hard constraint: **24 GB single GPU, no multi-GPU, no real hardware/vehicle.**

---

## Judging principle (derived directly from hardware, no subjectivity mixed in)
A direction is "locally feasible" ⟺ it satisfies all of:
1. **Training scale** fits within 24 GB (≤14B LoRA/QLoRA, or full fine-tuning of a smaller model);
2. **Evaluation does not depend on real hardware / real vehicles** (simulation or public datasets can substitute);
3. **Datasets are publicly available** (not dependent on restricted/proprietary data).

If any condition is not met → excluded. Below lists only directions that **pass all three**.

---

## Tier 1: local sweet spot, ready to start immediately (pure-text/small-model algorithms, 0 vision-environment dependency)

This tier is not affected by torchvision environment issues and can run today.

### P1. Knowledge editing / lifelong editing series (CCF-A caliber, one topic can fission into a whole group)
- **What it is**: single/batch/lifelong knowledge editing on ≤7B LLMs (locate-then-edit weights or external memory), studying generalization, retention, forgetting, and stopping criteria.
- **Why it's locally sufficient**: editing operates on GPT-J-6B / Llama-3-8B scale models, QLoRA + single-edit peak memory < 20 GB. No training needed, mostly inference + a small amount of gradient computation.
- **Public data**: `zsRE`, `CounterFact`, `MQuAKE` (multi-hop), `WikiBio`. All directly downloadable from HF.
- **First experiment (achievable in 1–2 days)**: reproduce the ROME/MEMIT baseline → measure "chain-consistency collapse after multi-fact editing" on MQuAKE → propose an improvement such as an "editing stopping criterion" or "routed memory".
- **Fissionable sub-topics**: long-text editing, multi-fact editing, multimodal editing, RL editing strategies, editing theory/stopping criteria —— all sharing the same code skeleton.

### P2. Efficient Reasoning / CoT compression + post-training RL
- **What it is**: shortening/pruning CoT tokens, early-exit, adaptive inference budget; or post-training alignment via DPO/GRPO.
- **Why it's locally sufficient**: 7–8B model GRPO/DPO with trl can run on a single card (QLoRA + vLLM inference sampling fits within 24 GB, small batch).
- **Public data**: `GSM8K`, `MATH`, `Big-Bench-Hard`; preference data `UltraFeedback`.
- **First experiment**: measure the "CoT length vs. accuracy" curve → train a length-penalized GRPO, proving tokens↓40% while accuracy doesn't drop.

### P3. Agent / multi-agent / agent security / long-term memory
- **What it is**: agent orchestration, jailbreak/injection attack-defense, long-term memory mechanisms, multi-agent collaboration protocols.
- **Why it's locally sufficient**: the core is orchestration + evaluation + small-model fine-tuning, with low compute demand; running a local Ollama 7–14B as the agent backend is more than enough.
- **Public data/benchmarks**: `AgentBench`, `ToolBench`, `AdvBench` (security), `τ-bench`.
- **First experiment**: measure the injection success rate of a local 7B agent on AdvBench → add a layer of "memory isolation/tool whitelisting" defense, produce an attack-defense comparison table.

### P4. Multimodal knowledge-graph reasoning / LLM recommendation / time-series forecasting / AIGC detection
- **What it is**: retrieval-augmented KG reasoning, LLM recommendation, LLM+Transformer time series, AIGC-generated content detection.
- **Why it's locally sufficient**: embedding encoding + medium-sized model inference, memory-friendly.
- **Public data**: `FB15k-237`/`WN18RR` (KG), `Amazon`/`MovieLens` (recommendation), `Monash TS`/`ETT` (time series), `GenImage`/`DFD` (AIGC detection).
- **Lowest barrier to entry, suited for a "guaranteed SCI Q3–Q4" fallback.**

---

## Tier 2: locally feasible, but **must fix torchvision first** (vision/diffusion)

> Must run before starting: reinstall torchvision matching torch 2.12.1+cu130, verify `torchvision.ops.nms` doesn't error.

### P5. Discriminative vision (detection / segmentation / classification / tracking / remote sensing)
- **What it is**: object detection, fine-grained classification, action/sign-language recognition, industrial defect detection, multi-object tracking, weakly-supervised 3D segmentation, remote-sensing interpretation/fusion/change detection.
- **Why it's locally sufficient**: ResNet/ViT/DETR/D-FINE/SAM fine-tuning, standard single-card 24 GB configuration.
- **Public data**: `COCO`, `ImageNet`, `MOT17`, `DOTA`/`LoveDA` (remote sensing), `MVTec-AD` (industrial).
- **Solid, good cost-effectiveness for SCI Q3–Q4.**

### P6. Diffusion controllable generation / image editing (LoRA side)
- **What it is**: ControlNet-style controllable generation, instruction-aligned image editing.
- **Why it's locally sufficient**: SD1.5/SDXL LoRA fine-tuning fits on a single card (FLUX needs quantization + offload, at the limit).
- **Public data**: `InstructPix2Pix`, `MagicBrush`, `COCO`.
- **First experiment**: fine-tune SDXL + an editing dataset, evaluate "instruction alignment degree".

---

## Tier 3: locally feasible as an "algorithm-side / simulation-evaluation version" (world models / VLA, avoiding real hardware)

### P8. World-model distillation: future-supervised training, no rendering at inference
- **What it is**: distill the ability to "predict the future" into an action policy —— use future supervision during training to learn planning, output actions directly at inference time (Value-of-Imagination / Decision-Influential Abstraction).
- **Why it's exactly locally sufficient**: its selling point is precisely "no video/latent-trajectory generation at inference time," which conveniently avoids the most expensive local cost, video generation. Training a compact policy on **offline demo data** fits within 24 GB.
- **Key substitution: replace real hardware with simulation benchmarks** —— `LIBERO` / `ManiSkill2` / `CALVIN` offline datasets + simulation evaluation. Real-hardware evaluation goes into future work; the paper submits as-is.
- **First experiment**: run a BC baseline on LIBERO → add a "future-state prediction auxiliary head" for distillation → compare success rates, proving generalization↑ with 0 extra inference-time cost.

### P9. 3D reconstruction adversarial attacks / lightweighting (on pretrained models)
- **What it is**: adversarial examples against feed-forward 3D reconstruction (VGGT-type)/3D Gaussians, lightweight distillation.
- **Why it's locally sufficient**: use **already-released pretrained reconstruction models** for attack/compression, not training a large model from scratch.
- **Public data**: `ScanNet`/`DTU` subsets, `Tanks&Temples`.

---

## Explicitly excluded (structurally infeasible locally, don't invest)
| Excluded item | Reason |
|---|---|
| **Full training** of video-generation world models, training a unified image-generation model **from scratch** | Requires 8×A100-class multi-GPU |
| Real-hardware VLA grasping / general-scenario robotics / special-terrain navigation **real-hardware evaluation**, autonomous-driving **real-vehicle closed-loop**, connected-vehicle field testing | No robotic arm/robot/vehicle |
| Wind-turbine gearbox oil-film dynamics | Requires CFD cluster / test rig |
| Digital capability and elderly asset allocation | Not a computational topic |

---

## Suggested starting path (ranked by "speed to output × risk")
1. **P1 Knowledge editing** —— runnable today, no environment dependency, one topic fissions into a whole group, CCF-A caliber. **First choice.**
2. **P2 Efficient CoT / RL** —— also pure text, trl runs directly.
3. **P8 World-model distillation (simulation version)** —— has the most "novelty × publishability" tension, avoids real hardware.
4. **P5 Discriminative vision** or **P4 Detection/recommendation/time-series** —— guaranteed SCI Q3–Q4 fallback.

**Discipline: with a single 24 GB GPU, only seriously run one training task at a time. Start with P1, close the loop on one paper, then start the second.**

---

### Pre-work setup (30 minutes)
```
# 1. Fix the vision stack (required for P5–P6/P9; can be skipped for P1–P4)
#    Reinstall torchvision matching torch 2.12.1+cu130, verify:
#    python -c "import torchvision; torchvision.ops.nms"   # no error means OK
# 2. Clean disk: only 579 GB left, clear old HF/Ollama caches before starting a new direction
# 3. Settle on P1, download zsRE/CounterFact/MQuAKE, reproduce the MEMIT baseline
```
