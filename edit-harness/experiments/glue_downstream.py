"""glue_downstream.py — GLUE downstream-task tracking across a SEQUENTIAL ROME edit
trajectory (2026-07-08). The "table-stakes downstream tracking" damage claims need: does
zero-shot task accuracy on SST-2/MRPC/RTE degrade as unrelated CounterFact edits pile up
on the SAME (never-restored) model?

DESIGN (mirrors the harness's own conventions, reuses its editing + scoring primitives —
does NOT reimplement them):
  * Editing: editors.rome_native.apply_edit, called in a loop WITHOUT restoring between
    edits (the point is CUMULATIVE sequential edits — same "no_restore" concept as
    killgate_keygeom.py's SEQ mode, just without killgate's probe-matrix machinery, since
    here the measurement is GLUE accuracy, not per-probe damage).
  * Edit source: experiments.killgate_keygeom.load_counterfact (dual-path import, this
    module's own established pattern — see cfplus_specificity.py) — plain unrelated
    CounterFact facts, edited in insertion order, exactly like killgate's SEQ mode.
  * Scoring: metrics.next_token_logits + metrics.first_target_token_id — a FORCED two-way
    choice between two single-token label words (same "compare two candidate
    continuations' logits" idiom the harness already uses everywhere: killgate's
    prob_of_token, cfplus_specificity's full_target_scores undamaged rule). This is a
    zero-shot cloze classifier, not a generation-based one — deliberately, to stay
    consistent with the rest of the harness's next-token-logit scoring and avoid a
    second, differently-biased eval methodology.

GLUE TASKS + LABEL WORDS (standard HF glue parquet label ints, verified against the
downloaded data/glue/{sst2,mrpc,rte} validation splits):
  sst2: sentence -> label 0=negative/1=positive
  mrpc: sentence1,sentence2 -> label 0=not_equivalent/1=equivalent
  rte:  sentence1,sentence2 -> label 0=entailment/1=not_entailment

DIVERGENCE FLAG (honest, no network access to verify against a published GLUE prompt
suite): the exact prompt wording below is this module's own construction, not a
reproduction of any specific published template (e.g. not lm-evaluation-harness's exact
strings) — informal accuracy numbers, not a leaderboard-comparable score. Good enough to
answer the driver's actual question (does accuracy trend down as edits accumulate), not
intended as a publishable absolute GLUE number.

Standalone CLI:
  python experiments/glue_downstream.py --model data/models/Qwen2.5-0.5B \
      --n_glue_samples 2 --n_edits 4 --checkpoints 0,2,4 --steps 2 --device cpu \
      --out results/smoke/glue_qwen05b_cpu_smoke.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch

HARNESS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HARNESS not in sys.path:
    sys.path.insert(0, HARNESS)
from metrics import next_token_logits, first_target_token_id  # noqa: E402


# --------------------------------------------------------------------------- #
# 0. task registry — prompt template + two-way label words (see module docstring
#    DIVERGENCE FLAG: this module's own construction, not a reproduced published suite)
# --------------------------------------------------------------------------- #
LABEL_WORDS: Dict[str, Dict[int, str]] = {
    "sst2": {0: "negative", 1: "positive"},
    "mrpc": {0: "no", 1: "yes"},
    "rte": {0: "yes", 1: "no"},   # 0=entailment, 1=not_entailment
}


def build_prompt(task: str, ex: dict) -> str:
    if task == "sst2":
        return f"Review: {ex['sentence'].strip()}\nSentiment (positive or negative)? Answer:"
    if task == "mrpc":
        return (f"Sentence 1: {ex['sentence1'].strip()}\nSentence 2: {ex['sentence2'].strip()}\n"
                f"Question: Do these two sentences mean the same thing? Answer (yes or no):")
    if task == "rte":
        return (f"Premise: {ex['sentence1'].strip()}\nHypothesis: {ex['sentence2'].strip()}\n"
                f"Question: Does the premise entail the hypothesis? Answer (yes or no):")
    raise ValueError(f"unknown GLUE task {task!r}")


# --------------------------------------------------------------------------- #
# 1. data loading — parquet validation splits already on disk (data/glue/{task}/)
# --------------------------------------------------------------------------- #
def load_glue_task(glue_dir: str, task: str, n_samples: int, seed: int) -> List[dict]:
    import pandas as pd
    path = os.path.join(glue_dir, task, "validation-00000-of-00001.parquet")
    df = pd.read_parquet(path)
    n = min(n_samples, len(df))
    df = df.sample(n=n, random_state=seed).reset_index(drop=True)
    return df.to_dict("records")


# --------------------------------------------------------------------------- #
# 2. zero-shot two-way forced-choice accuracy on the (possibly edited) model
# --------------------------------------------------------------------------- #
@torch.no_grad()
def eval_glue_task(model, tok, task: str, examples: List[dict], device: str) -> float:
    if not examples:
        return float("nan")
    words = LABEL_WORDS[task]
    tid0 = first_target_token_id(tok, words[0])
    tid1 = first_target_token_id(tok, words[1])
    correct = 0
    for ex in examples:
        prompt = build_prompt(task, ex)
        logits = next_token_logits(model, tok, prompt, device)
        pred = 0 if float(logits[tid0]) >= float(logits[tid1]) else 1
        if pred == int(ex["label"]):
            correct += 1
    return correct / len(examples)


def eval_all_tasks(model, tok, tasks: Dict[str, List[dict]], device: str) -> Dict[str, float]:
    return {t: eval_glue_task(model, tok, t, ex, device) for t, ex in tasks.items()}


# --------------------------------------------------------------------------- #
# standalone CLI: sequential (no-restore) ROME edit trajectory vs GLUE accuracy
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="GLUE downstream-accuracy tracking across a sequential ROME edit trajectory.")
    ap.add_argument("--model", default=os.path.join(HARNESS, "data", "models", "Qwen2.5-0.5B"))
    ap.add_argument("--data", default=os.path.join(HARNESS, "data", "counterfact.json"),
                    help="edit source (CounterFact-schema); loaded via killgate's load_counterfact")
    ap.add_argument("--glue_dir", default=os.path.join(HARNESS, "data", "glue"))
    ap.add_argument("--tasks", default="sst2,mrpc,rte")
    ap.add_argument("--n_glue_samples", type=int, default=100,
                    help="validation examples PER task PER checkpoint measurement")
    ap.add_argument("--n_edits", type=int, default=100, help="max edits in the trajectory "
                    "(== max(--checkpoints); edits beyond the last checkpoint are never applied)")
    ap.add_argument("--checkpoints", default="0,10,50,100",
                    help="comma ints, edit counts at which GLUE accuracy is (re)measured")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layer", default="auto")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    checkpoints = sorted({int(x) for x in args.checkpoints.split(",") if x.strip() != ""})
    if not checkpoints or checkpoints[0] != 0:
        checkpoints = [0] + checkpoints
    n_edits_needed = max(checkpoints)
    if args.n_edits < n_edits_needed:
        raise SystemExit(f"[glue_seq] --n_edits {args.n_edits} < max(--checkpoints) {n_edits_needed}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from editors.rome_native import apply_edit  # noqa: E402
    HARNESS_ = HARNESS
    if HARNESS_ not in sys.path:
        sys.path.insert(0, HARNESS_)
    from experiments.killgate_keygeom import load_counterfact  # noqa: E402

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(args.device).eval()
    nL = model.config.num_hidden_layers
    layer = nL // 2 if args.layer == "auto" else int(args.layer)
    print(f"[glue_seq] loaded {args.model} ({nL} layers, layer={layer}, device={args.device}) "
          f"{time.time()-t0:.1f}s", flush=True)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    glue_examples = {t: load_glue_task(args.glue_dir, t, args.n_glue_samples, args.seed) for t in tasks}
    for t in tasks:
        print(f"[glue_seq] {t}: {len(glue_examples[t])} validation examples", flush=True)

    # edits: plain unrelated CounterFact facts, no probe bank needed (n_probes=0)
    edits, _probes, *_ = load_counterfact(args.data, n_edits_needed, 0, args.seed)
    if len(edits) < n_edits_needed:
        raise SystemExit(f"[glue_seq] --data only yielded {len(edits)} edits, need {n_edits_needed}")
    print(f"[glue_seq] {len(edits)} edits loaded from {args.data} {time.time()-t0:.1f}s", flush=True)

    cfg = {"layer": layer, "steps": args.steps, "lr": args.lr}
    trajectory = []
    i = 0
    for ckpt in checkpoints:
        while i < ckpt:
            apply_edit(model, tok, edits[i], cfg, args.device)
            i += 1
        accs = eval_all_tasks(model, tok, glue_examples, args.device)
        row = {"n_edits": i, "accuracy": accs}
        trajectory.append(row)
        print(f"[glue_seq] checkpoint n_edits={i} accuracy={accs} {time.time()-t0:.1f}s", flush=True)

    res = {
        "model": args.model, "layer": layer, "seed": args.seed, "steps": args.steps,
        "lr": args.lr, "tasks": tasks, "n_glue_samples": args.n_glue_samples,
        "checkpoints": checkpoints, "trajectory": trajectory,
        "runtime_s": round(time.time() - t0, 1),
    }
    print(json.dumps(res, indent=2), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        tmp = args.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(res, f, indent=2)
        os.replace(tmp, args.out)
        print(f"[glue_seq] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
