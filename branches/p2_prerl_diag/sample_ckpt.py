#!/usr/bin/env python3
"""
sample_ckpt.py — the GPU sampler queued by make_jobs.py (`gen` jobs).

Loads a HF causal-LM checkpoint, draws k independent CoT samples per GSM8K
problem, grades each against the GSM8K gold answer, and writes the
per-problem/per-sample JSON that run_diag.py / diagnostic.py consume.

Output schema (== run_diag.py's documented input contract):
    {"checkpoint": "<basename of --model>",
     "problems": [
        {"problem": "gsm8k/<split>/<idx>",
         "samples": [ {"text": "...", "len": <generated_tokens:int>,
                        "correct": <bool>}, ... x k ]},
        ...],
     "_meta": {...}}     # extra key; diagnostic.load_samples() ignores it

`len` is the generated-token count (len_unit="generated_tokens", matching the
`spec.len_unit` make_jobs.py stamps into the gen job) — NOT a word count.
diagnostic.py._sample_length() prefers this numeric `len` field over a
text.split() fallback, so it is read directly by D_pooled / D_within.

Grading ("boxed_gsm8k", matching make_jobs.py spec.grader): the prompt asks
the model to end with \\boxed{ANSWER}; we extract the last \\boxed{...} in the
generated text and compare it (as a float) to the gold answer parsed from
GSM8K's trailing "#### <num>" line. No \\boxed{} found => graded incorrect.
Garbage accuracy from small/undertrained checkpoints is expected and fine —
this only needs to produce schema-valid output.

GSM8K availability: if `openai/gsm8k` is not resolvable via `datasets` (no
network + not already in the local HF cache), pass --data <path.json> with a
local override: either GSM8K's own {"question","answer"} shape (a JSON list,
"answer" ending in "#### <num>") or {"data": [...]} wrapping the same list.

Runs on the GPU by default (--device cuda, this is the queued dl-rl job); pass
--device cpu --model_dtype fp32 for a CPU-only smoke test on a tiny model
(see the branch's smoke instructions — never run this on a real checkpoint or
the GPU from an AUTHOR pass).

Usage (matches the exact `cmd` make_jobs.py stamps into queue/p2_gen_*.json):
    python sample_ckpt.py --model <path> --dataset openai/gsm8k --config main \\
        --split test --n-problems 200 --k 8 --temperature 0.9 --top-p 1.0 \\
        --max-new-tokens 640 --seed 0 --out samples/<id>.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

PROMPT_TEMPLATE = (
    "Solve the following grade school math problem. Think step by step, then "
    "give the final answer on its own line in the exact form \\boxed{{ANSWER}}, "
    "where ANSWER is a single number with no units, dollar signs, or commas.\n\n"
    "Problem: {question}\n\nSolution:"
)

_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
_GOLD_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ----------------------------------------------------------------------------- #
# Grading — "boxed_gsm8k"
# ----------------------------------------------------------------------------- #

def extract_boxed_answer(text: str) -> Optional[str]:
    """Last \\boxed{...} in the generated text, or None if absent."""
    matches = _BOXED_RE.findall(text)
    return matches[-1].strip() if matches else None


def extract_gold_answer(answer_field: str) -> Optional[str]:
    """GSM8K gold numeric answer from the trailing '#### <num>' line."""
    matches = _GOLD_RE.findall(answer_field)
    return matches[-1].strip() if matches else None


def _normalize_number(raw: Optional[str]) -> Optional[float]:
    if raw is None:
        return None
    s = raw.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def grade_boxed_gsm8k(generated_text: str, gold_answer_field: str) -> bool:
    pred = _normalize_number(extract_boxed_answer(generated_text))
    gold = _normalize_number(extract_gold_answer(gold_answer_field))
    if pred is None or gold is None:
        return False
    return abs(pred - gold) < 1e-4


# ----------------------------------------------------------------------------- #
# Problem loading
# ----------------------------------------------------------------------------- #

def load_problems(args: argparse.Namespace) -> List[Dict[str, str]]:
    """Return up to --n-problems {"question","answer"} dicts.

    --data (if given) overrides the HF dataset with a local JSON file: either
    a bare list of {"question","answer"} items (GSM8K's own shape) or
    {"data": [...]} / {"problems": [...]} wrapping the same list.
    """
    if args.data:
        with open(args.data, "r") as fh:
            raw = json.load(fh)
        items = raw if isinstance(raw, list) else raw.get("data", raw.get("problems"))
        if not isinstance(items, list):
            raise ValueError(f"{args.data}: expected a list (or dict with "
                              "'data'/'problems' list) of {{question,answer}}")
        problems = [{"question": it["question"], "answer": it["answer"]} for it in items]
    else:
        from datasets import load_dataset
        ds = load_dataset(args.dataset, args.config, split=args.split)
        problems = [{"question": ex["question"], "answer": ex["answer"]} for ex in ds]
    return problems[: args.n_problems]


# ----------------------------------------------------------------------------- #
# Sampling
# ----------------------------------------------------------------------------- #

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF causal-LM checkpoint path/id")
    ap.add_argument("--dataset", default="openai/gsm8k")
    ap.add_argument("--config", default="main")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-problems", type=int, default=200)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=640)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--model_dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--data", default=None,
                    help="local JSON override for GSM8K (see module docstring)")
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[sample_ckpt] WARNING: --device cuda requested but CUDA is "
              "unavailable; falling back to cpu")
        device = "cpu"
    dtype = torch.bfloat16 if args.model_dtype == "bf16" else torch.float32

    print(f"[sample_ckpt] loading {args.model} "
          f"(dtype={args.model_dtype}, device={device})")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.to(device)
    model.eval()

    problems = load_problems(args)
    print(f"[sample_ckpt] {len(problems)} problems (split={args.split}), "
          f"k={args.k}, temperature={args.temperature}, top_p={args.top_p}, "
          f"max_new_tokens={args.max_new_tokens}")

    out_problems: List[Dict[str, Any]] = []
    n_correct_total = 0
    n_samples_total = 0
    log_every = max(1, len(problems) // 10) if problems else 1

    for idx, item in enumerate(problems):
        prompt = PROMPT_TEMPLATE.format(question=item["question"])
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = enc["input_ids"].shape[1]
        with torch.no_grad():
            gen = model.generate(
                **enc,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                num_return_sequences=args.k,
                pad_token_id=tokenizer.pad_token_id,
            )
        eos_id = tokenizer.eos_token_id
        samples: List[Dict[str, Any]] = []
        for row in gen:
            gen_ids = row[input_len:].tolist()
            if eos_id is not None and eos_id in gen_ids:
                gen_ids = gen_ids[: gen_ids.index(eos_id)]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            correct = grade_boxed_gsm8k(text, item["answer"])
            samples.append({"text": text, "len": len(gen_ids), "correct": bool(correct)})
            n_samples_total += 1
            n_correct_total += int(correct)
        out_problems.append({"problem": f"gsm8k/{args.split}/{idx}", "samples": samples})
        if (idx + 1) % log_every == 0 or idx == len(problems) - 1:
            print(f"[sample_ckpt] {idx + 1}/{len(problems)} problems done")

    payload = {
        "checkpoint": os.path.basename(args.model.rstrip("/")),
        "problems": out_problems,
        "_meta": {
            "model_path": args.model,
            "dataset": args.data if args.data else args.dataset,
            "config": args.config,
            "split": args.split,
            "n_problems": len(problems),
            "k": args.k,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "len_unit": "generated_tokens",
            "grader": "boxed_gsm8k",
            "seed": args.seed,
            "device": device,
            "model_dtype": args.model_dtype,
            "generated_at": _now(),
            "tool": "sample_ckpt.py",
        },
    }

    acc = (n_correct_total / n_samples_total) if n_samples_total else float("nan")
    print(f"[sample_ckpt] done: {n_samples_total} samples, "
          f"pooled accuracy={acc:.3f}")

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    tmp_path = args.out + ".tmp"
    with open(tmp_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp_path, args.out)
    print(f"[sample_ckpt] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
