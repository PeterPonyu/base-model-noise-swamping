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


def _is_whitespace_token(tokenizer, tok_id: int) -> bool:
    """True when ``tok_id`` decodes to nothing but whitespace (or to nothing).

    SentencePiece tokenizers (Phi-3.5, Llama-2, Mistral) emit a standalone
    whitespace-marker token (e.g. id 29871 = ``U+2581``) as the FIRST id of any
    string encoded with a leading space. Such a token carries no target identity.
    """
    piece = tokenizer.decode([tok_id])
    return piece.strip() == ""


def first_target_token_id(tokenizer, target: str) -> int:
    """First *content* continuation token id of ``target`` (used for argmax success).

    DEFECT FIXED 2026-07-30 (see docs/findings/findings-PHI35-TOKENIZER-COLLISION-2026-07-30.md):
    this returned ``ids[0]`` verbatim, which on SentencePiece tokenizers is the
    leading whitespace marker for EVERY target. On Phi-3.5, ``" Paris"``,
    ``" Michael"`` and ``"I cannot answer"`` all encoded to a first id of 29871,
    so every target collapsed to the same token: ROME optimised toward emitting
    whitespace and the scorer then read that same token back as "success".
    Leading whitespace-only tokens are now skipped so the id identifies the target.
    """
    ids = target_token_ids(tokenizer, target)
    for tok_id in ids:
        if not _is_whitespace_token(tokenizer, tok_id):
            return tok_id
    # Degenerate target (whitespace only): no content token exists to score.
    raise ValueError(
        f"target {target!r} tokenises to whitespace-only ids {ids}; it cannot "
        "identify an argmax target"
    )


def target_distinguishability(tokenizer, targets: Sequence[str]) -> Dict[str, object]:
    """Measure how well first-content-token argmax separates ``targets``.

    Returns ``n_targets``, ``n_first_tokens``, ``ratio`` (= n_first_tokens /
    n_targets) and up to 10 example ``collisions``.

    Two REGIMES matter and must not be conflated:

    * **Catastrophic** (ratio near 0) — the tokenizer collapses essentially every
      target onto one id. This is the Phi-3.5 whitespace defect: 319 distinct
      CounterFact targets → 1 id (ratio 0.003). Every per-edit efficacy and
      deletion-suppression number in such a cell is meaningless.
    * **Benign prefix sharing** (ratio high but < 1) — occasional pairs share a
      first subword: on fixed Phi-3.5, ``'NBC'``/``'Nissan'`` both start at ``'N'``
      (47 colliding pairs, ratio ~0.85). This is an inherent limitation of
      first-token argmax scoring, not a defect; it is a caveat to DISCLOSE, and
      it is invisible on vocabularies that happen not to collide (Llama, Qwen,
      gemma all score 1.000 on the same 319 targets).
    """
    seen: Dict[int, str] = {}
    collisions: List[tuple] = []
    uniq = {str(t).strip() for t in targets if t is not None and str(t).strip()}
    for target in sorted(uniq):
        tok_id = first_target_token_id(tokenizer, target)
        prior = seen.get(tok_id)
        if prior is not None and prior != target:
            collisions.append((prior, target, tok_id))
        seen[tok_id] = target
    n_t = len(uniq)
    return {
        "n_targets": n_t,
        "n_first_tokens": len(seen),
        "ratio": (len(seen) / n_t) if n_t else 1.0,
        "collisions": collisions[:10],
        "n_collisions": len(collisions),
    }


def assert_targets_distinguishable(
    tokenizer, targets: Sequence[str], min_ratio: float = 0.5
) -> Dict[str, object]:
    """Abort when first-token argmax cannot separate targets at all.

    Guards the Phi-3.5 class of defect (see
    docs/findings/findings-PHI35-TOKENIZER-COLLISION-2026-07-30.md) BEFORE any GPU
    work: had this existed, cell 1 of the phi35 deletion run would have failed
    instead of producing three silently-void cells.

    Fails only in the catastrophic regime (``ratio < min_ratio``); benign prefix
    sharing is returned in the report for the caller to log and the paper to
    disclose, never used to block a legitimate model.
    """
    report = target_distinguishability(tokenizer, targets)
    if report["ratio"] < min_ratio:
        ex = "; ".join(f"{a!r}/{b!r}->{i}" for a, b, i in report["collisions"][:3])
        raise ValueError(
            f"tokenizer collapses targets onto too few first tokens: "
            f"{report['n_first_tokens']}/{report['n_targets']} distinct ids "
            f"(ratio {report['ratio']:.3f} < {min_ratio}); examples: {ex}. "
            "Per-edit efficacy would be unmeasurable on this model."
        )
    return report


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
