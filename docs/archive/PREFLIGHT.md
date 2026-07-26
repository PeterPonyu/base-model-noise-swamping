# Fission Engine — Pre-Departure Preflight Report (PREFLIGHT)

> Purpose: **before large-scale download traffic**, use a tiny proxy to fully test everything the local machine needs, and get experiment gates defined for all selected directions.
> Date 2026-06-30 · Machine RTX 5090 Laptop 24GB · env `dl` (torch 2.12.1+cu130).

## TL;DR
**All critical paths have been verified locally (using tiny models <100MB, no large model downloaded).** Found and fixed 2 blocking issues that would have wasted a large download. Now only "run the download scripts wave by wave" remains.

---

## A. Blocking issues found and fixed during preflight (this is the value of testing first)

| # | Issue | Impact scope | Status |
|---|---|---|---|
| **1** | **torchvision 0.24.0 vs torch 2.12.1 ABI mismatch** → `torchvision::nms does not exist` | Fatal: transformers 5.12 loading **any** model (even pure-text Llama/GPT2) crashes via `processing_utils→image_utils→torchvision`. **Not just the vision branch** | ✅ Installed `torchvision 0.27.1+cu130` (torch unchanged); `GPT2LMHeadModel`/`LlamaForCausalLM` now import fine |
| **2** | **`ALL_PROXY=socks://127.0.0.1:7897` breaks the httpx-based huggingface_hub** → `Unknown scheme for proxy URL` | Fatal: **all** model/dataset downloads fail | ✅ `env.sh` now does `unset ALL_PROXY all_proxy`, keeping the http proxy on the same port; downloads verified working |

> Lesson recorded: the earlier claim that "pure text is unaffected by torchvision" was a **misjudgment** — `AutoModelForCausalLM` (the auto class) can lazily import and give a false positive, but actually instantiating `LlamaForCausalLM` pulls in torchvision. **Fixing torchvision is a hard prerequisite for every direction, not just vision.**

---

## B. Local smoke tests passed (tiny proxy, zero large downloads)

| Test | Result | Proxy artifact used |
|---|---|---|
| torch + CUDA on 5090 | ✅ cuda True, dev=RTX 5090 | — |
| Text causal-LM import | ✅ (after fixing torchvision) both GPT2/Llama work | — |
| GPU fp16 load + generate (via proxy download) | ✅ full pipeline in 21.5s | `sshleifer/tiny-gpt2` |
| datasets cached load | ✅ gsm8k test n=1319 | local cache |
| datasets new download (via proxy) | ✅ socratic split | pulled via proxy |
| **Edit primitive ①** layer localization | ✅ `model.layers[i].mlp.down_proj.weight` | `tiny-random-Llama` |
| **Edit primitive ②** forward-hook key extraction | ✅ captured k tensor | same as above |
| **Edit primitive ③** rank-one ΔW logit edit | ✅ `max|Δlogit|=0.0043` | same as above |

→ **The three core MEMIT/ROME primitives run natively on transformers 5.12 + Llama architecture.**

---

## C. Key architecture decision: hand-roll the editing engine natively, not EasyEdit

- `easyeditor` **is not on pip** (`No matching distribution`), requires a clone, and historically pins an old transformers (~4.x), high risk of conflict with our transformers 5.12.
- Since the three editing primitives are already natively verified in the modern env → **write a minimal native ROME/MEMIT** (layer key/value + rank-one/least-squares update), run in the existing `dl` env, **bypassing dependency hell**.
- Baseline comparison: first model choice is **GPT-J-6B** (ungated; CounterFact/zsRE were built directly on it, so results align directly with published numbers).

---

## D. Resources already in place (already downloaded, 0 traffic)
- **51 Ollama LLMs** (317GB, verified working on GPU) — serves as the eval/data-synthesis backend for B3/B4.
- **HF cache 92GB** — eval sets like gsm8k/bbh/mmlu... plus the full bge/gte/ms-marco embedding/reranker suite.
- These **do not count** toward the download budget below.

---

## E. Pending downloads (staged as scripts, **not yet executed**)
`edit-harness/download_manifest.sh` (already validated with `bash -n`, `hf` CLI in place, env.sh in effect):

| wave | contents | traffic | unlocks |
|---|---|---|---|
| `wave0` | GPT-J-6B + CounterFact + zsRE | **~14 GB** | trunk + B1/B2/B3/B6 |
| `wave1` | MQuAKE multi-hop | +0.1 GB | B1 collapse story |
| `wave2` | Qwen2-VL-7B | +18 GB | B5 multimodal (download after the text track matures) |
| `llama` | Llama-3-8B (ungated mirror) | +16 GB | optional modern backbone |

**To start, only `bash download_manifest.sh wave0` (~14GB) is needed.**

---

## F. Experiment gates (all selected directions)
Full definitions in `edit-harness/gates.yaml`: trunk gate + each of B1–B6's entry/gate/kill-gate/GPU cost, status marked PASS/READY/PENDING. Key points:
- **Trunk gate**: MEMIT×GPT-J-6B×CounterFact reproduction efficacy≥0.9 & locality≥0.7, otherwise it's a harness bug.
- Each branch carries a **kill-gate** (kill if no signal validated within 2-3 days), to prevent fission from spreading too thin.
- **B6 mechanism** is marked READY and low-GPU → dedicated as parallel filler for "while the GPU is running other branches."

---

## G. Departure checklist (executable now)
1. `source ~/Desktop/idea-feasibility-analysis/env.sh` (every new shell)
2. `bash edit-harness/download_manifest.sh wave0`  ← **first wave ~14GB**
3. Write a minimal native MEMIT runner (config→JSON) + `queue/` batch-run script
4. Run the trunk gate → once it passes, admit B1 into the queue with `wave1`

> The only remaining actual download/training actions are all gated behind step 2 — **the preflight stage is now complete, no significant traffic consumed.**
