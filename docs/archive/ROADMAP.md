# Knowledge-Editing Fission Roadmap — Single-GPU Serial × Multi-Output Parallel Orchestration Plan

> Core tension: **a single 5090 GPU can only run one training experiment at a time**, but what you want is **a suite of papers (fission)**.
> Solution: treat the GPU as the "sole constrained resource" and schedule it as a queue, split **non-GPU work (design/analysis/writing) into parallel lanes** so the GPU never sits idle, while multiple branches' outputs advance simultaneously.

---

## 0. First Principles of Scheduling

```
GPU experiments = serial bottleneck (one at a time, but a single editing experiment is fast: minute-scale)
Design/analysis/writing = parallelizable (CPU/human/agent, multiple branches advancing at once)

Fission speed = GPU utilization × branch parallelism
              = (overnight batch queue never idle) × (multiple branches designing+analyzing+writing during the day)
```

Key insight: **a single knowledge-editing experiment is minute-scale** (ROME/MEMIT edits a single layer's weights, it's not training from scratch) — so GPU serialization isn't really a severe bottleneck, **a whole night can run dozens of sweep configs**. The real bottleneck is "designing experiments + analyzing results + writing," and these happen to be exactly what can be fissioned into parallel work.

---

## 1. Trunk: Build Once, All Branches Reuse

> The precondition for fission is a unified infrastructure where "swap the data/loss/probe and a new paper comes out." **Week 1 is dedicated entirely to this.**

```
edit-harness/
├── models/       # GPT-J-6B, Llama-3-8B (HF fp16)  ← downloaded once, shared across all branches
├── methods/      # pluggable: ROME, MEMIT → later GRACE/WISE/AlphaEdit/MEND
├── datasets/     # zsRE, CounterFact, MQuAKE → later multimodal/long-text
├── metrics/      # efficacy, generalization, locality, portability, fluency
├── probes/       # mechanism probes (for the B6 theory branch)
├── runner.py     # accepts config(method×model×dataset×metric), outputs structured JSON
└── queue/        # GPU job queue (see §3)
```

**Trunk acceptance gate**: get the MEMIT-on-CounterFact 5-metric baseline running, with results landed in a standard JSON schema. After this, each branch = "add one axis + run the matrix + interpret."

---

## 2. Six Branches (each = one paper) + Dependency DAG

| Branch | Paper's selling point | Dependencies | GPU cost | Order |
|---|---|---|---|---|
| **B1 multi-hop consistency collapse** | Quantifying and diagnosing the collapse of multi-hop reasoning-chain consistency after editing a single fact (MQuAKE) | Trunk only | Low | **First** |
| **B2 sequential-editing stopping criterion** | When to stop continuous editing: a learnable Stopping Criterion | Trunk + sequential loop | Medium | 2 |
| **B3 lifelong editing and forgetting** | Forgetting curves and mitigation under thousands of sequential edits | Reuses B2's sequential loop | High (long sequences) | 3 |
| **B4 RL/routing editing strategy** | Use RL to learn a routing policy for "which layer to edit / whether to attach external memory" | Trunk + policy module | High (RL) | 4 |
| **B5 multimodal editing** | Port editing from LLM to VLM (fix torchvision first) | Trunk ported to VLM | Medium | 5 |
| **B6 editing mechanism/theory** | Mechanistic analysis of why editing generalizes / why it breaks locality | Trunk + probes | **Low / can slot in anytime** | Throughout |

### Dependency relationships (key: B1→B2→B3 share the "sequential editing" infrastructure, build once use three times)
```
        ┌──────────────── Trunk (week 1) ────────────────┐
        │                                                │
       B1 multi-hop collapse ──► B2 stopping criterion ──► B3 lifelong/forgetting   B6 mechanism(throughout, write while GPU busy)
        │                 (shared sequential loop)                  │
        └──► B4 RL routing(after trunk matures) ──► B5 multimodal(after torchvision fixed)
```

**Ordering logic**:
- B1 is cheapest, defines the problem, produces the first finding fastest → use it to shake out the trunk.
- B2/B3 share the "sequential editing loop," write the infrastructure once, get two papers.
- B4 (RL) is the most GPU-hungry, placed after the trunk matures, run overnight for long stretches.
- B5 (multimodal) is a "port," waits until the text-line results are proven + torchvision is fixed.
- B6 (theory) doesn't need much GPU, **specifically used to fill the periods when "the GPU is running something else and your hands are free."**

---

## 3. GPU Scheduling: Job Queue, Overnight Batch Runs, Never Idle

Treat the GPU like a "printer" — all experiments queue into `queue/`, and one runner consumes them sequentially at night:

```bash
# each .sh in queue/ is a self-contained experiment (method×model×dataset×seed)
# runner loop: take one → run → write results to results/ → take the next
nohup bash queue/run_all.sh > logs/gpu_$(date +%F).log 2>&1 &
```

**Queuing discipline**:
1. **Keep each experiment small**: one job per config, minute-to-60-minute scale, so it's easy to interleave and resume from checkpoints.
2. **Overnight batch runs**: before bed, queue up the day's 10–30 designed configs, run them all overnight.
3. **Morning harvest**: waking up = a batch of results is ready → move into the analysis lane.
4. **Leave VRAM headroom**: 24 GB running an 8B fp16 edit uses ~18–20 GB, **only queue one job at a time**, no concurrency (concurrency will OOM).
5. **seed×3**: every key configuration must run 3 seeds before it's publishable — this conveniently fills the queue and saturates GPU utilization.

---

## 4. Four-Lane Orchestration: GPU Serial, Everything Else Fully Parallel (this is the fission engine)

> At any given moment: the GPU is running **branch N**, while you/agents are designing **N+1**, analyzing **N−1**, and writing **N−2**. The GPU never waits on people, people never wait on the GPU.

| Lane | What it does | Who does it | When |
|---|---|---|---|
| **Lane G (GPU, serial)** | Run the experiment sweeps in `queue/` | Machine (overnight) | 24/7, mainly overnight |
| **Lane D (design, parallel)** | Design the next batch of configs, prepare data, write ablation tables | You + subagent | Daytime |
| **Lane A (analysis, parallel)** | Parse completed JSON, produce tables/figures, decide next ablation | subagent (scientist) | After harvest |
| **Lane W (writing, parallel)** | Turn completed analyses into paper sections | subagent (writer) | While GPU runs |

**Daily cadence**:
```
Morning: harvest overnight Lane G results → Lane A analysis (dispatch scientist agent)
Midday:  Lane D designs the next batch of configs (you set direction, agent fills the matrix) + Lane W writes up mature branches
Evening: enqueue new configs → launch overnight Lane G batch run
```

**The key to multi-branch parallelism**: because D/A/W don't consume GPU, **while B1 is running an experiment on the GPU**, you can in parallel: analyze B1's previous batch of results, design B2's sequential loop, and write a draft of B6's theoretical analysis. **One GPU, six branches advancing simultaneously at different stages.**

---

## 5. Phased Timeline (realistic cadence)

| Phase | Week | Lane G (GPU) | Lanes D/A/W (parallel) | Output |
|---|---|---|---|---|
| **0 Trunk** | W1 | Get MEMIT baseline running (5 metrics) | Build runner/queue/results schema | Trunk ready |
| **1 B1** | W2–3 | MQuAKE multi-hop sweep (×3 seeds) | Analyze collapse curves + write B1 | **Paper 1 draft** |
| **2 B2+B3** | W4–6 | Sequential editing + lifelong long-run (overnight) | Design stopping criterion + analyze forgetting curves | **Papers 2, 3 advancing** |
| **3 B4** | W6–8 | RL routing policy training (overnight long-run) | Simultaneously finalize B1 + write B6 theory | **Paper 1 submitted, Paper 4 running** |
| **4 B5** | W8–10 | Fix torchvision → VLM editing | Wrap up B2/B3 writing | **Papers 2/3 submitted** |
| **Throughout B6** | W2–10 | Borrow checkpoints already produced by other branches (low GPU) | Write theoretical analysis whenever GPU is busy | **Paper 6 ready anytime** |

> This is not a promise of "6 papers in 10 weeks," but rather **a pipeline where the 6 branches always have 3–4 running in parallel at different stages**, GPU utilization stays saturated, and writing never gets blocked waiting on experiments.

---

## 6. Kill-Gate for Each Branch (to avoid fission turning into unfocused sprawl)

Before each branch starts, set an **early-death criterion**: if no signal is validated within 2–3 days, cut it and hand the GPU time to the next branch:

- **B1**: Is MQuAKE multi-hop consistency **significantly lower** than single-hop efficacy? No → no story, cut.
- **B2**: Is there a predictable inflection point for "number of edits vs. locality collapse"? No → cut.
- **B3**: Does the forgetting curve monotonically worsen with edit count, with differences between methods? No → cut.
- **B4**: Does RL routing give a stable >2% gain over fixed-layer editing? No → downgrade to an ablation, fold into B2.
- **B5**: Is VLM editing's locality harder than LLM's (a genuinely new phenomenon)? No → cut.
- **B6**: Do the probes yield a falsifiable mechanistic hypothesis? No → fold into another branch's analysis section.

---

## 7. Immediately Actionable "Week One" Checklist
1. **Download** (~18 GB): HF Llama-3-8B safetensors + zsRE/CounterFact/MQuAKE.
2. **Build the trunk**: install `EasyEdit`, wrap a `runner.py` layer (config→JSON) + `queue/` batch-run script.
3. **Run the acceptance gate**: MEMIT × Llama-3-8B × CounterFact, output the five metrics efficacy/generalization/locality/portability/fluency as JSON.
4. **Queue B1**: MQuAKE multi-hop consistency sweep (method×seed×hop-depth), queue it before bed, run it overnight.
5. **Build the analysis template**: a script that aggregates `results/*.json` into tables/figures (reused by Lane A).

---

### Orchestration Summary in One Sentence
> **GPU serial queuing, overnight batch runs, never idle; design/analysis/writing dispatched to agents for parallel fission; B1→B2→B3 share the sequential infrastructure — build once, use three times; B6 theory fills the GPU-busy periods. One 5090, six branches flowing simultaneously at different stages.**
