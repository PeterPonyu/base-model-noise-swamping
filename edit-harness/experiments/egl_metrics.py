"""egl_metrics.py — canonical CounterFact ES/PS/NS + zsRE E/G/L metrics.

Consumed exclusively by ``experiments/killgate_keygeom.py`` under ``--egl`` (see the
import block there for the exact call sites / ordering). Lets a GPT-2-XL (or any)
cell be compared against the published ROME CounterFact table.

PRIMARY metric = full-target mean log-probability, teacher-forced (the actual
published-table method: Meng et al. 2022, ``eval_utils_counterfact.py:
test_batch_prediction``). For a (prompt, target) pair this means: append the
target's continuation tokens to the prompt, run ONE forward pass, and average
the per-token log-prob over exactly the target's token span (not the prompt's).
ES/PS/NS/E/G then compare the mean log-prob of target_new against target_true
(or the flipped inequality for NS) — NOT a first-token argmax/prob comparison.
This is what makes the numbers comparable to the published ROME CounterFact
table; the earlier first-token-only version was not.

  * **ES** (Efficacy Score)   — mean-logprob(target_new) > mean-logprob(target_true)
    on the rewrite prompt itself, post-edit.
  * **PS** (Paraphrase Score) — same inequality, averaged over the record's
    ``paraphrase_prompts`` (capped at ``--egl_max_paraphrase``).
  * **NS** (Neighborhood Score) — mean-logprob(target_true) > mean-logprob(target_new)
    (note the flipped inequality: neighbors are UNDAMAGED iff the true fact
    still wins) on the record's ``neighborhood_prompts`` (capped at
    ``--egl_max_neighborhood``). Neighborhood prompts ask about *different*
    subjects that share the edited fact's true object, evaluated with the
    SAME (target_new, target_true) token pair as the edit — this is the
    CounterFact convention, not a per-neighbor-prompt target pair.

  zsRE analogues (same E/G/L letters, no formal "Score" name in the zsRE
  literature):
  * **E** — mean-logprob(target_new) > mean-logprob(target_true) on the
    rewrite prompt (target_true here is the loader's ``pred`` field, i.e. the
    dataset's original answer; same formula as ES).
  * **G** — same inequality on the record's single ``rephrase`` prompt.
  * **L** — locality. zsRE's ``loc`` prompt is a topically UNRELATED question
    (no target_new/target_true pair meaningfully attaches to it), so locality
    is measured the standard zsRE way: does the post-edit argmax token on
    ``loc`` still equal the PRE-edit argmax token (behavior unchanged)? This
    is why ``precompute_egl_baselines`` exists for zsRE (one extra forward
    pass per edit, pre-edit, on the base model).

SECONDARY (``_ft`` suffix) = the first-token proxy this module used exclusively
before this fix pass: ``P(target_new) > P(target_true)`` read off a single
next-token distribution, no target continuation appended. ``ES_ft``/``E_ft``
are harness-internal convenience fields that cost nothing (reused verbatim
from killgate's already-computed ``eff['p_new_gt_true']``). PS_ft/NS_ft/G_ft
are NOT computed — doing so under the full-target scheme would require extra
forward passes purely for the proxy, which this module does not add (measure
the real overhead via the smoke harness instead of trusting a stale estimate;
this module no longer claims a fixed "~+5% cell runtime" number since ES/PS/NS
now cost 1 extra forward per (prompt, target) pair — see the module's smoke
notes / killgate's own runtime_s for the honest current number).

All logit reads go through ``metrics.next_token_logits`` / the full-target
scorer below (both fp32/.float()), matching lab convention (no fp16
probability reads, no AUROC).
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch

HARNESS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HARNESS not in sys.path:
    sys.path.insert(0, HARNESS)
from metrics import next_token_logits, first_target_token_id, target_token_ids, efficacy  # noqa: E402


# --------------------------------------------------------------------------- #
# 0. full-target scoring (PRIMARY metric) — mean per-token log-prob of the    #
#    target's FULL continuation, teacher-forced. This is the published ROME   #
#    CounterFact table method (Meng et al. 2022, eval_utils_counterfact.py:   #
#    test_batch_prediction), not a first-token proxy.                        #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def mean_logprob_full_target(model, tok, prompt: str, target: str, device: str) -> float:
    """Mean per-token log-probability of ``target``'s full continuation,
    teacher-forced after ``prompt``. One forward pass over prompt+target
    tokens; logits read fp32 (``.float()``, lab rule) via log_softmax, then
    averaged over exactly the target's token span (prompt tokens excluded).
    """
    target_ids = target_token_ids(tok, target)
    prompt_ids = tok(prompt, return_tensors="pt")["input_ids"][0].tolist()
    full_ids = prompt_ids + target_ids
    input_ids = torch.tensor([full_ids], device=device)
    logits = model(input_ids=input_ids).logits[0].detach().float().cpu()  # [seq, vocab]
    log_probs = torch.log_softmax(logits, dim=-1)
    n_prompt = len(prompt_ids)
    total = 0.0
    for k in range(len(target_ids)):
        pos = n_prompt + k - 1  # logits[pos] predicts full_ids[pos + 1]
        total += log_probs[pos, full_ids[n_prompt + k]].item()
    return total / len(target_ids)


@torch.no_grad()
def full_target_scores(model, tok, prompt: str, target_new: str,
                       target_true: Optional[str], device: str) -> Dict[str, Optional[float]]:
    """Full-target mean-log-prob pair for (target_new, target_true) on
    ``prompt``. One forward pass per target (two total when target_true is
    supplied; batching across targets is not done here — killgate's per-edit
    loop calls this at most a handful of times per edit, see egl smoke notes
    for the measured overhead).

    Returns ``{'lp_new', 'lp_true', 'new_wins'}`` — ``new_wins`` is 1.0 if
    ``lp_new > lp_true`` (1.0 by default when no target_true is supplied,
    mirroring ``metrics.success_from_logits``'s convention).
    """
    lp_new = mean_logprob_full_target(model, tok, prompt, target_new, device)
    if not target_true:
        return {"lp_new": lp_new, "lp_true": None, "new_wins": 1.0}
    lp_true = mean_logprob_full_target(model, tok, prompt, target_true, device)
    return {"lp_new": lp_new, "lp_true": lp_true, "new_wins": 1.0 if lp_new > lp_true else 0.0}


# --------------------------------------------------------------------------- #
# 1. match-back: attach paraphrase/neighborhood (CF) or rephrase/loc (zsRE)    #
#    fields onto the already-loaded `edits` records, by CONTENT (the loader   #
#    shuffles with a seed so file order != edits order).                      #
# --------------------------------------------------------------------------- #
def attach_egl_fields(edits: List[dict], data_path: str, dataset: str) -> int:
    """Mutate each edit dict in place, adding EGL fields matched back to the raw
    data file by (subject, prompt, target_new, target_true) content-equality.
    Never touches the existing edit keys (prompt/subject/target_new/target_true)
    the COS-matrix key capture already depends on. Returns the match count.
    """
    data = json.load(open(data_path))
    lut: Dict[tuple, dict] = {}
    if dataset == "counterfact":
        for d in data:
            rr = d.get("requested_rewrite", d)
            try:
                subj = rr["subject"]
                prompt = rr["prompt"].format(subj) if "{}" in rr["prompt"] else rr["prompt"]
                tnew = rr["target_new"]["str"] if isinstance(rr["target_new"], dict) else rr["target_new"]
                ttrue = rr["target_true"]["str"] if isinstance(rr["target_true"], dict) else rr["target_true"]
            except Exception:
                continue
            key = (subj, prompt, tnew, ttrue)
            lut.setdefault(key, d)  # first occurrence wins — keys are unique across all 21,919
            # CounterFact records (0 duplicate tuples), so first-occurrence is unambiguous
            # regardless of scan order (the loader shuffles, so "scan order" isn't a fixed thing)
        n_match = 0
        for e in edits:
            rec = lut.get((e["subject"], e["prompt"], e["target_new"], e["target_true"]))
            if rec is not None:
                e["paraphrase_prompts"] = list(rec.get("paraphrase_prompts") or [])
                e["neighborhood_prompts"] = list(rec.get("neighborhood_prompts") or [])
                n_match += 1
            else:
                e["paraphrase_prompts"] = []
                e["neighborhood_prompts"] = []
        return n_match
    else:  # zsre
        for d in data:
            s, p, alt, pred = d.get("subject"), d.get("src"), d.get("alt"), d.get("pred")
            if not (s and p and alt and pred):
                continue
            lut.setdefault((s, p, alt, pred), d)
        n_match = 0
        for e in edits:
            rec = lut.get((e["subject"], e["prompt"], e["target_new"], e["target_true"]))
            if rec is not None:
                e["rephrase"] = rec.get("rephrase") or None
                e["loc_prompt"] = rec.get("loc") or None
                e["loc_ans"] = rec.get("loc_ans") or None
                n_match += 1
            else:
                e["rephrase"] = None
                e["loc_prompt"] = None
                e["loc_ans"] = None
        return n_match


# --------------------------------------------------------------------------- #
# 2. base-model baselines (BEFORE the edit loop). ES/PS/NS are pure post-edit #
#    probability comparisons -> no baseline needed for CounterFact. zsRE      #
#    locality needs the pre-edit argmax on the `loc` prompt (see docstring).  #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def precompute_egl_baselines(model, tok, edits: List[dict], device: str, dataset: str,
                             max_neighborhood: int) -> List[dict]:  # max_neighborhood: reserved (unused)
    """One entry per edit, in edit order. CounterFact entries are empty dicts
    (no pre-edit forward pass needed); zsRE entries carry the pre-edit argmax
    (+ P(loc_ans)) on the record's `loc` prompt.

    ``max_neighborhood`` is unused here (no per-neighbor baseline is needed —
    see module docstring) but kept in the signature: killgate_keygeom.py's
    call site passes it positionally (not our call site to change).
    """
    if dataset == "counterfact":
        return [{} for _ in edits]
    baselines = []
    for e in edits:
        loc_prompt, loc_ans = e.get("loc_prompt"), e.get("loc_ans")
        if not loc_prompt or not loc_ans:
            baselines.append({"loc_argmax_pre": None, "loc_ans_token_id": None, "loc_p_ans_pre": None})
            continue
        logits = next_token_logits(model, tok, loc_prompt, device)  # fp32 cpu
        argmax_pre = int(torch.argmax(logits).item())
        try:
            ans_tok = first_target_token_id(tok, loc_ans)
            p_ans_pre = float(torch.softmax(logits, dim=-1)[ans_tok].item())
        except Exception:
            ans_tok, p_ans_pre = None, None
        baselines.append({"loc_argmax_pre": argmax_pre, "loc_ans_token_id": ans_tok,
                          "loc_p_ans_pre": p_ans_pre})
    return baselines


# --------------------------------------------------------------------------- #
# 3. per-edit measurement (ON THE EDITED MODEL, before restore)               #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def measure_egl_one(model, tok, e: dict, base_i: dict, device: str, dataset: str,
                    eff: Optional[dict] = None, max_neighborhood: int = 10,
                    max_paraphrase: int = 2) -> dict:
    """Score one just-edited record. PRIMARY fields (ES/PS/NS, E/G) are the
    full-target mean-log-prob comparison (see module docstring / section 0) —
    comparable to the published ROME CounterFact table. `eff` is killgate's
    ALREADY-COMPUTED efficacy() dict for this same (prompt, target_new,
    target_true) — reused ONLY for the SECONDARY `ES_ft`/`E_ft` first-token
    proxy fields (the primary ES/E can no longer reuse it, since `eff` is a
    first-token quantity). PS_ft/NS_ft/G_ft are not computed: doing so under
    the full-target scheme would require extra forward passes purely for the
    proxy, which this function does not add. Caps (`max_neighborhood`/
    `max_paraphrase`) are enforced HERE, not at attach-time, so they always
    reflect the caller's args.
    """
    if eff is None:
        eff = efficacy(model, tok, e["prompt"], e["target_new"], e.get("target_true"), device)

    rec: Dict[str, object] = {"subject": e.get("subject")}
    if dataset == "counterfact":
        ft = full_target_scores(model, tok, e["prompt"], e["target_new"],
                                e.get("target_true"), device)
        rec["ES"] = float(ft["new_wins"])
        rec["ES_ft"] = float(eff["p_new_gt_true"])  # first-token proxy (harness-internal)

        pps = (e.get("paraphrase_prompts") or [])[:max_paraphrase]
        ps_vals = []
        for pp in pps:
            ft_pp = full_target_scores(model, tok, pp, e["target_new"], e.get("target_true"), device)
            ps_vals.append(ft_pp["new_wins"])
        rec["PS"] = float(np.mean(ps_vals)) if ps_vals else None
        rec["n_paraphrase"] = len(ps_vals)

        nps = (e.get("neighborhood_prompts") or [])[:max_neighborhood]
        ns_vals = []
        for np_prompt in nps:
            ft_np = full_target_scores(model, tok, np_prompt, e["target_new"],
                                       e.get("target_true"), device)
            # neighbor UNDAMAGED iff true still wins (flipped inequality); no target_true
            # means we cannot verify the neighbor is undamaged -> counted as damaged (0.0),
            # matching the old first-token version's p_true=0.0 fallback behaviour.
            undamaged = 1.0 if (ft_np["lp_true"] is not None and ft_np["lp_true"] > ft_np["lp_new"]) else 0.0
            ns_vals.append(undamaged)
        rec["NS"] = float(np.mean(ns_vals)) if ns_vals else None
        rec["n_neighborhood"] = len(ns_vals)
    else:  # zsre
        ft = full_target_scores(model, tok, e["prompt"], e["target_new"],
                                e.get("target_true"), device)
        rec["E"] = float(ft["new_wins"])
        rec["E_ft"] = float(eff["p_new_gt_true"])  # first-token proxy (harness-internal)

        rephrase = e.get("rephrase")
        if rephrase:
            ft_g = full_target_scores(model, tok, rephrase, e["target_new"],
                                      e.get("target_true"), device)
            rec["G"] = float(ft_g["new_wins"])
        else:
            rec["G"] = None

        loc_prompt = e.get("loc_prompt")
        pre_argmax = (base_i or {}).get("loc_argmax_pre")
        if loc_prompt and pre_argmax is not None:
            logits = next_token_logits(model, tok, loc_prompt, device)
            post_argmax = int(torch.argmax(logits).item())
            rec["L"] = 1.0 if post_argmax == pre_argmax else 0.0
        else:
            rec["L"] = None
    return rec


# --------------------------------------------------------------------------- #
# 4. aggregation + npz/sidecar plumbing                                        #
# --------------------------------------------------------------------------- #
def summarize_egl(records: List[dict], dataset: str) -> dict:
    """Aggregate dict embedded into killgate's main result json (res['EGL'])."""
    if not records:
        return {"n_edits": 0}

    def _mean(key):
        vals = [r[key] for r in records if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    if dataset == "counterfact":
        return {
            "n_edits": len(records),
            "ES": _mean("ES"), "PS": _mean("PS"), "NS": _mean("NS"),
            "ES_ft": _mean("ES_ft"),  # first-token proxy, secondary (see module docstring)
            "n_paraphrase_total": int(sum(r.get("n_paraphrase", 0) or 0 for r in records)),
            "n_neighborhood_total": int(sum(r.get("n_neighborhood", 0) or 0 for r in records)),
            "n_paraphrase_missing": int(sum(1 for r in records if not r.get("n_paraphrase"))),
            "n_neighborhood_missing": int(sum(1 for r in records if not r.get("n_neighborhood"))),
        }
    return {
        "n_edits": len(records),
        "E": _mean("E"), "G": _mean("G"), "L": _mean("L"),
        "E_ft": _mean("E_ft"),  # first-token proxy, secondary (see module docstring)
        "n_generality_missing": int(sum(1 for r in records if r.get("G") is None)),
        "n_locality_missing": int(sum(1 for r in records if r.get("L") is None)),
    }


def egl_npz_arrays(records: List[dict]) -> Dict[str, np.ndarray]:
    """Flat [N] float32 arrays (NaN where the field is missing) for the
    save_matrices npz, under UNIFORM names (egl_E/egl_G/egl_L) regardless of
    dataset — CF's ES/PS/NS map onto the same three slots so downstream
    analyzers don't need a dataset switch.
    """
    if not records:
        return {}

    def _arr(key):
        return np.array([float(r[key]) if r.get(key) is not None else np.nan for r in records],
                        dtype=np.float32)

    if "ES" in records[0]:
        return {"egl_E": _arr("ES"), "egl_G": _arr("PS"), "egl_L": _arr("NS")}
    return {"egl_E": _arr("E"), "egl_G": _arr("G"), "egl_L": _arr("L")}


def write_egl_sidecar(path: str, obj: dict) -> None:
    """Atomic (tmp + os.replace) json write — mirrors killgate's own main-json
    commit pattern so a crash mid-write never leaves a truncated sidecar."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)
