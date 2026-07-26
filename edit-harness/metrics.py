"""metrics.py — pure, well-typed evaluation primitives for knowledge editing.

The functions here split into two groups:

1. Model-querying helpers (``next_token_logits``, ``generate_text``) — thin,
   deterministic wrappers around a HuggingFace causal-LM forward / generate.
   They are documented and side-effect free w.r.t. model *weights* (they only
   read), so they are safe to call before and after an edit.

2. Pure scoring functions (``ngram_entropy``, ``locality_score``,
   ``success_from_logits``) — no model, no I/O, fully unit-testable.

Metric vocabulary (matching the ROME / MEMIT / CounterFact literature):

* **efficacy / reliability** — does the *edited* prompt now produce the new
  target object? (argmax next token == target, and P(new) > P(true)).
* **generalization** — does the edit hold on *paraphrase* prompts?
* **locality / specificity** — are *unrelated* facts left unchanged? Measured
  as agreement between the pre-edit and post-edit argmax next token.
* **fluency** — n-gram entropy of free generation (collapse / repetition
  detector). Higher entropy = less degenerate text.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Optional, Sequence

import torch


# --------------------------------------------------------------------------- #
# Tokenisation helpers (shared by metrics AND editors so targets are aligned)  #
# --------------------------------------------------------------------------- #
def target_token_ids(tokenizer, target: str) -> List[int]:
    """Token ids for ``target`` as a *continuation* (leading space prepended).

    Most BPE/SentencePiece tokenizers encode a mid-sentence word with a leading
    space marker, so we prepend one. Falls back to the no-space encoding if the
    space variant is empty. Returns the full id list (used by the FT loss).
    """
    ids = tokenizer.encode(" " + target.strip(), add_special_tokens=False)
    if not ids:
        ids = tokenizer.encode(target.strip(), add_special_tokens=False)
    return ids


def first_target_token_id(tokenizer, target: str) -> int:
    """First continuation token id of ``target`` (used for argmax success)."""
    return target_token_ids(tokenizer, target)[0]


# --------------------------------------------------------------------------- #
# Model-querying helpers                                                       #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def next_token_logits(model, tokenizer, prompt: str, device: str = "cpu") -> torch.Tensor:
    """Return the logits over the vocabulary for the token *following* ``prompt``.

    Shape: ``[vocab_size]`` on CPU (detached). Deterministic; no sampling.
    """
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    out = model(**enc)
    return out.logits[0, -1, :].detach().float().cpu()


@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 30,
    device: str = "cpu",
) -> str:
    """Greedy-decode ``max_new_tokens`` continuation tokens for ``prompt``.

    Greedy (do_sample=False) so the result is deterministic and reproducible.
    Returns only the generated continuation (prompt stripped).
    """
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    gen = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id,
    )
    cont = gen[0, enc["input_ids"].shape[1]:]
    return tokenizer.decode(cont, skip_special_tokens=True)


# --------------------------------------------------------------------------- #
# Pure scoring functions                                                       #
# --------------------------------------------------------------------------- #
def success_from_logits(
    logits: torch.Tensor,
    new_token_id: int,
    true_token_id: Optional[int] = None,
) -> Dict[str, float]:
    """Score a single next-token prediction against the new (and optional true) target.

    Returns a dict with:
      * ``argmax_id``   — argmax token id of ``logits``
      * ``success``     — 1.0 if argmax == ``new_token_id`` else 0.0
      * ``p_new``       — softmax prob mass on ``new_token_id``
      * ``p_true``      — softmax prob mass on ``true_token_id`` (or -1 if none)
      * ``p_new_gt_true`` — 1.0 if P(new) > P(true) (the CounterFact efficacy
        criterion); 1.0 by default when no true target supplied.
    """
    probs = torch.softmax(logits.float(), dim=-1)
    argmax_id = int(torch.argmax(logits).item())
    p_new = float(probs[new_token_id].item())
    res: Dict[str, float] = {
        "argmax_id": float(argmax_id),
        "success": 1.0 if argmax_id == new_token_id else 0.0,
        "p_new": p_new,
    }
    if true_token_id is not None:
        p_true = float(probs[true_token_id].item())
        res["p_true"] = p_true
        res["p_new_gt_true"] = 1.0 if p_new > p_true else 0.0
    else:
        res["p_true"] = -1.0
        res["p_new_gt_true"] = 1.0
    return res


def efficacy(
    model,
    tokenizer,
    prompt: str,
    target_new: str,
    target_true: Optional[str] = None,
    device: str = "cpu",
) -> Dict[str, float]:
    """Efficacy / reliability on the *edited* prompt (see module docstring)."""
    logits = next_token_logits(model, tokenizer, prompt, device)
    new_id = first_target_token_id(tokenizer, target_new)
    true_id = first_target_token_id(tokenizer, target_true) if target_true else None
    out = success_from_logits(logits, new_id, true_id)
    out["new_token_id"] = float(new_id)
    return out


def generalization(
    model,
    tokenizer,
    paraphrase_prompts: Sequence[str],
    target_new: str,
    device: str = "cpu",
) -> Dict[str, float]:
    """Mean efficacy over paraphrase prompts. Returns aggregate + per-prompt list."""
    new_id = first_target_token_id(tokenizer, target_new)
    per: List[float] = []
    p_news: List[float] = []
    for p in paraphrase_prompts:
        logits = next_token_logits(model, tokenizer, p, device)
        r = success_from_logits(logits, new_id)
        per.append(r["success"])
        p_news.append(r["p_new"])
    n = max(len(per), 1)
    return {
        "generalization": sum(per) / n,
        "mean_p_new": sum(p_news) / n,
        "per_prompt_success": per,
        "n_prompts": float(len(per)),
    }


@torch.no_grad()
def argmax_tokens(
    model,
    tokenizer,
    prompts: Sequence[str],
    device: str = "cpu",
) -> List[int]:
    """Argmax next-token id for each prompt. Used to snapshot pre-edit behaviour."""
    return [
        int(torch.argmax(next_token_logits(model, tokenizer, p, device)).item())
        for p in prompts
    ]


def locality_score(pre_tokens: Sequence[int], post_tokens: Sequence[int]) -> Dict[str, float]:
    """Locality / specificity: fraction of unrelated prompts whose argmax token is
    UNCHANGED between the pre-edit and post-edit model. 1.0 == perfectly local.

    Pure: operates on the two id sequences only.
    """
    if not pre_tokens:
        return {"locality": 1.0, "n_prompts": 0.0, "n_changed": 0.0}
    same = sum(1 for a, b in zip(pre_tokens, post_tokens) if a == b)
    n = len(pre_tokens)
    return {
        "locality": same / n,
        "n_prompts": float(n),
        "n_changed": float(n - same),
    }


def ngram_entropy(text: str, ns: Sequence[int] = (2, 3), base: float = 2.0) -> float:
    """Weighted n-gram entropy of ``text`` (a fluency / degeneration proxy).

    For each n in ``ns`` we compute the Shannon entropy of the n-gram frequency
    distribution over whitespace tokens, then average across n. A repetitive /
    collapsed generation yields low entropy; varied text yields high entropy.
    Returns 0.0 for empty / too-short text. Pure function.
    """
    toks = text.split()
    if len(toks) < 2:
        return 0.0
    entropies: List[float] = []
    for n in ns:
        if len(toks) < n:
            continue
        grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
        counts = Counter(grams)
        total = sum(counts.values())
        ent = -sum((c / total) * math.log(c / total, base) for c in counts.values())
        entropies.append(ent)
    if not entropies:
        return 0.0
    return sum(entropies) / len(entropies)


def fluency(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 30,
    device: str = "cpu",
) -> Dict[str, float]:
    """Generate from ``prompt`` and report n-gram entropy of the continuation."""
    text = generate_text(model, tokenizer, prompt, max_new_tokens, device)
    return {
        "fluency_ngram_entropy": ngram_entropy(text),
        "generated_text": text,
    }
