# Local resource inventory + download bandwidth estimate per direction

> Measured 2026-06-30. Network reachable to HF (status 200, 1.4s). Free disk 579 GB.

---

## 1. Resources already local (these **do not need to be downloaded again**)

### ✅ LLM inference: 51 Ollama models already downloaded, **GPU-verified working**
- Total **317 GB**, covering the full 1B→32B spectrum, all quantized.
- Functionally verified: `qwen3:8b` generates normally, GPU usage 13.5 GB / 95% util → **confirmed running on the GPU, not the CPU**.
- Key usable models: `qwen3:32b`(20G), `qwen2.5:14b-q8`(15G), `qwen3:8b-q8`, `deepseek-r1:14b/8b/7b`, `llama3.1:8b`, `gemma3:12b/4b`, `mistral:7b`, `glm4:9b`, `gemma2:9b-q8`, `yi:9b`, `falcon3:10b`, etc.
- **Implication**: for any stage that is "LLM inference / agent backend / evaluation / data synthesis," **0 bandwidth needed**, ready to run as-is.

### ✅ HF cache 92 GB: 46 models + 34 datasets
- **Datasets ready to use directly** (heavily reused across P2/P3/P4):
  - Reasoning/CoT: `gsm8k`, `bbh`, `mmlu`, `gpqa`, `hellaswag`, `arc`, `winogrande`, `piqa`, `siqa`, `logiqa`, `boolq`, `openbookqa`, `sciq`, `truthful_qa`, `IFEval`, `squad_v2`
  - Agent/tool calling: `xlam-function-calling-60k`, `ToolACE`, `glaive-function-calling`, `hermes-function-calling`, `API-Bank`, `Berkeley-Function-Calling-Leaderboard`, `NexusRaven`, `json-mode-eval`
- **Models ready to use directly**: `bge-m3`, `gte-Qwen2-1.5B`, `bge-reranker-v2-*`, `ms-marco` series cross-encoders, `e5-large`, `stella-400M` (full embedding/reranker suite).

---

## 2. Key distinction: what Ollama models can and cannot do

| Purpose | Ollama (GGUF) model | What's needed |
|---|---|---|
| Inference / agent backend / evaluation / data synthesis | ✅ Use directly, 0 bandwidth | None |
| **LoRA/QLoRA fine-tuning** (P2 RL, P6) | ❌ Not possible | Requires HF **fp16/bf16 safetensors** weights |
| **Weight editing** ROME/MEMIT (P1) | ❌ Not possible | Requires HF fp16 weights |

> Conclusion: **training-type** directions must download a copy of the HF-format base model (one 7–8B ≈ 16 GB); **inference/agent/evaluation-type** directions can directly consume the existing Ollama models, 0 bandwidth.

---

## 3. Download bandwidth estimate per direction (counting only "still need to download")

| Direction | Model to download | Data to download | **Incremental bandwidth** | Already covered |
|---|---|---|---|---|
| **P1 Knowledge editing** | Llama-3-8B HF safetensors **16G** (or GPT-J-6B 12G, either one) | zsRE+CounterFact+MQuAKE ≈ **2G** | **~18 GB** | Post-edit evaluation can use local LLM |
| **P2 CoT/RL** | Reuse P1's 8B base (**0**) | MATH ~0.3G + UltraFeedback ~1G (gsm8k/bbh already available) | **~1.5 GB** | gsm8k/bbh/mmlu all already cached |
| **P3 Agent/safety** | **0** (Ollama as backend) | AdvBench ~50M + AgentBench ~0.5G (tool-calling data already extensively cached) | **~0.6 GB** | xlam/ToolACE/BFCL/APIBank already available |
| **P4 KG/recommendation/time-series/AIGC detection** | embedding already available (**0**) | FB15k/WN18RR ~0.2G + MovieLens ~0.5G + ETT ~0.1G + GenImage **subset** ~5G | **~6 GB** | bge/gte embeddings already available |
| **P5 Discriminative vision** | Pretrained backbone ~2G | COCO **20G** (ImageNet uses a ~5G subset instead of the full 150G) | **~25 GB** | Fix torchvision first |
| **P6 Diffusion image editing** | SDXL-base **7G** + VAE/ControlNet ~2G | InstructPix2Pix/MagicBrush ~3G | **~12 GB** | — |
| **P8 World-model distillation (simulation)** | Base policy weights ~1G | LIBERO ~10G (or CALVIN ~30G, ManiSkill demo ~15G) | **~11–30 GB** | Pure simulation, no real robot |
| **P9 3D reconstruction adversarial/lightweighting** | Pretrained reconstruction model (VGGT/3DGS) ~3G | ScanNet/DTU **subset** ~10G (full 1TB+ not downloaded) | **~13 GB** | — |

---

## 4. Bandwidth totals by starting path

| Starting path | Total download | Notes |
|---|---|---|
| **Pure-text trio** P1+P2+P3 | **~20 GB** | One 8B base (16G) shared by P1/P2, agent uses Ollama; most data already cached. **Best value.** |
| Add P4 as a hedge | +6 GB → **~26 GB** | Embeddings already available |
| Add P8 world model | +11 GB (LIBERO) → **~37 GB** | The most novel direction |
| **Full vision line** +P5+P6+P9 | +50 GB → **~82 GB** | Need to fix torchvision first; ImageNet/ScanNet both use subsets |
| **All 8 directions** | **~102 GB** | Still far less than the 579 GB free disk |

---

## 5. Conclusions and notes
1. **LLMs are ready**: 51 local models GPU-verified working, all "inference/agent/evaluation" stages need 0 bandwidth.
2. **Training base models must be downloaded separately**: GGUF cannot be trained; each training direction needs a copy of HF safetensors (~16 GB each, but a 7–8B model can be reused across directions).
3. **Recommended starting point ~20 GB** (P1+P2+P3 pure text) — shared base model, most data already cached, can finish downloading and start running within half a day.
4. **Use subsets of data**: ImageNet/ScanNet/GenImage full sizes are TB-scale; download subsets as needed, already counted as subsets in the table above.
5. **Disk is sufficient**: even the most aggressive all-9-directions option ~120 GB << 579 GB free disk.
6. Ollama is now started (running as a background `serve`); after the next machine restart, `ollama serve` needs to be run again.
