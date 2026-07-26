# Research Directions Overview — PORTFOLIO

> This machine: RTX 5090 24GB single card. Status: 🔵in progress · 🟢ready to start (high reuse) · 🟡ready to start (same family) · ⚪broader pool · ⏸on hold.
> Updated 2026-06-30. Related docs: PLAN / RESEARCH-DEFINITION / BREADTH-ANALYSIS / NEXT-DIRECTIONS.

## 🔵 Main line (in progress)
| ID | Direction | Status | Prerequisites/dependencies |
|---|---|---|---|
| **B6** | key geometry predicts editing damage (mechanism predictor) | GATE experiment queued (engine auto-chains to `run_gate.sh` when done) | harness already built; GATE decides go/no-go |

**GATE criterion**: within-probe partialled Spearman ≥0.15 and permutation p<0.05 → survives; <0.10 → pivot to D1/D2/Qwen-null story.

## 🟢 Level 1 — Direct derivatives of the broad scan (highest reuse, fastest to produce results)
| ID | Direction | One-liner | Reuse | Cost |
|---|---|---|---|---|
| **D1** ⭐ | FT over-collateral-damage paradox | FT damage is 4× that of ROME yet geometrically unpredictable → rebuts the "FT is safer" consensus; KL-regularized FT as the fix | 100% | ~1 day |
| **D2** | Architecture editing safety audit | key orthogonality = pre-deployment safety screening metric; 200× damage gap; report mean_damage rather than only AUROC | 100% | ~1 GPU-day |
| **D3** | Geometry-gated editor routing | compute key-cosine before editing → route high to AlphaEdit, low to vanilla ROME (bridges B4) | ~80% | ~1 GPU-day |

## 🟡 Level 2 — Knowledge-editing same-family branches (originally fan-out, shared harness)
| ID | Direction | Notes |
|---|---|---|
| **B1** | Multi-hop / ripple consistency | originally judged crowded (RippleEdits/MQuAKE saturated); MQuAKE now down, can be unblocked |
| **B2** | Sequential-editing stopping criterion | when to stop |
| **B3** | Lifelong editing and forgetting | originally judged crowded (NAS/RLSEdit/SPHERE already occupy it) |
| **B4** | Edit-site routing | D3 is its concretization |
| **B5** | Multimodal VLM editing | ⏸ high cost (needs VLM harness + visual probes) |

## ⚪ Level 3 — Broader locally-feasible pool (P1–P9)
| ID | Direction | Status |
|---|---|---|
| **P1** | Knowledge editing | = the entire current B-series main line 🔵 |
| **P2** | Efficient CoT / RL post-training | pure text, runs directly with trl |
| **P3** | Agents / multi-agent / safety / long-term memory | reuses local Ollama |
| **P4** | KG reasoning / recommendation / temporal / AIGC detection | SCI as a floor |
| **P5** | Discriminative vision (detection/segmentation/remote sensing) | torchvision already fixed |
| **P6** | Diffusion image-editing LoRA | |
| **P8** | World-model distillation (simulation replacing real robots) | your earliest interest; orthogonal to editing, a second front |
| **P9** | 3D reconstruction adversarial robustness / lightweighting | |

## Recommended progression order
1. **Pass the GATE** (runs automatically) → decide B6's fate.
2. If it passes → **D1 + D2**, the fastest two papers (counterintuitive + mechanistically clear), plus extend to Gemma/Phi architectures + G4 (AlphaEdit causal).
3. If it stalls → **D1 + D2 + Qwen-null** three-in-one still makes a paper.
4. Once editing is thoroughly done → open the second front with **P8 world-model distillation**.

## Code assets (edit-harness/)
- Editors: `rome_native.py` · `ft_editor.py` (includes KL term lambda_kl) · `alphaedit.py` (null-space, pre-write + math verified)
- Experiments: `killgate_keygeom.py` (`--save_matrices` stores raw matrices + resid_norm) · `analyze_matrices.py` (within-probe + permutation-null GATE analysis)
- Orchestration: `engine.py` (broad-scan self-advancing) · `run_gate.sh` (auto-waits for engine → runs GATE) · `download_models.py`
