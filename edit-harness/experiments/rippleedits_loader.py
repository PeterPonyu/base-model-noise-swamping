"""rippleedits_loader.py — loader for the RippleEdits benchmark (edenbiran/RippleEdits),
converting its native schema into this harness's edit-record format (the same
{subject, prompt, target_new, target_true, ...} dicts produced by killgate_keygeom.py's
load_counterfact/load_zsre/load_mquake) so RippleEdits can feed the SAME key-capture /
damage-metric machinery.

SCHEMA, as actually downloaded to data/rippleedits/{popular,random,recent}.json
(github.com/edenbiran/RippleEdits, data/benchmark/*.json — see
data/DOWNLOADS-20260706.md item 2). Each record:
    {
      "example_type": "popular" | "random" | "recent",
      "edit": {
        "prompt": <full DECLARATIVE sentence with the NEW fact already filled in,
                   e.g. "The name of the country of citizenship of Leonardo DiCaprio
                   is Syria.">,
        "subject_id": <Wikidata QID, e.g. "Q38111">,
        "relation": <relation name, e.g. "COUNTRY_OF_CITIZENSHIP">,
        "target_id": <Wikidata QID of the NEW target>,
        "original_fact": {  # present for popular/random, ABSENT for recent (see below)
          "prompt": <the OLD fact, same template, e.g. "...is United States of America.">,
          "subject_id": ..., "relation": ..., "target_id": ...
        }
      },
      "Logical_Generalization": [...], "Compositionality_I": [...],
      "Compositionality_II": [...], "Subject_Aliasing": [...],
      "Relation_Specificity": [...], "Forgetfulness": [...]  # popular/random only
    }
Each criterion value is a list of "sub-criterion" instances:
    {"test_queries": [{"prompt"|"phrase": <cloze-style prefix, NO answer appended>,
                       "answers": [{"value": str, "aliases": [str, ...]}, ...],
                       "query_type": str, "subject_id": ..., "relation": ...,
                       "target_ids": [...] }, ...],
     "test_condition": "OR"|"AND",
     "condition_queries": [...]}  # base-fact preconditions; NOT applied as a filter here
                                   # (see KNOWN LIMITATIONS below) — read but unused.

TWO KNOWN SCHEMA DEVIATIONS FROM THE TASK BRIEF (found by inspecting the real files, not
assumed — reported rather than silently reconciled):
  1. The brief named 6 criteria including "Preservation"; this download instead has
     "Forgetfulness" in that slot (LG, CI, CII, Subject_Aliasing, Relation_Specificity,
     Forgetfulness — 6 total, same count, different name for the 6th). No "Preservation"
     key exists anywhere in any of the 3 files (verified over all records, all 3 files).
  2. recent.json's "edit" dicts carry NO "original_fact" key at all (verified: 0/1948
     records) — "recent" entities are genuinely new, so there IS no prior fact to
     overwrite. This loader's edit-construction method (a word-level prefix diff against
     original_fact.prompt, see _diff_edit below) has nothing to diff against for recent.json
     and DELIBERATELY DOES NOT invent one (e.g. by guessing a subject span or splitting the
     sentence some other way) — see KNOWN LIMITATIONS.

WHY WORD-LEVEL PREFIX DIFF: RippleEdits' top-level edit.prompt is a full declarative
sentence (answer already filled in), unlike CounterFact's cloze "{}"-templated prompt — not
directly usable by killgate's next-token efficacy/damage machinery, which needs a
prompt STEM plus a target string. But edit.prompt and edit.original_fact.prompt are the
SAME template with only the target substituted, so their common prefix (split on
whitespace, not characters — see the docstring on _word_common_prefix for why character-
level diffing corrupts ~5% of records where the old/new target strings happen to share a
leading substring, e.g. "Carol Chu" vs "Casey DeSantis") recovers the cloze stem, and the
two suffixes (trailing "." stripped) recover target_new/target_true. Verified over the FULL
popular.json + random.json (re-checked 2026-07-06, not a sample): popular.json is 885/885
(100%) clean — zero skips of any kind. random.json is 1913/1922 (99.5%) clean, with 9
records skipped as genuine no-op edits (original_fact.prompt == edit.prompt CHARACTER-FOR-
CHARACTER — 8 of the 9 happen to be gender facts re-asserting the same value, e.g. "The
gender of Peter A Lazzarini is male." on both sides, plus 1 continent fact; this is a
handful of incidental duplicate-value records in the source data, not a systematic
"gender facts are no-ops" pattern — the dataset also contains plenty of GENUINE gender
edits that word-diffing correctly keeps as real edits, e.g. "The gender of Brett Gelman is
intersex organism." vs "...is male." (verified present in popular.json)). Neither file
produced an empty-new-target record in the full scan; the prior-value-unknown case ("...
is .", target_true="") is a separate, real, non-skip path handled explicitly below, not
a crash — see _diff_edit.

KNOWN LIMITATIONS (by design, not oversight — flagged rather than papered over):
  - No subject SURFACE STRING is available anywhere in this schema (only subject_id
    QIDs) — every record built by this loader sets subject=None. killgate's own
    find_subject_last_token_index(tok, prompt, subject=None) already falls back to "last
    token of the whole prompt" in that case (editors/rome_native.py) — a pre-existing,
    documented harness behavior, not a new heuristic invented here. For these cloze-style
    stems (e.g. "...Leonardo DiCaprio is ") the last token usually falls very close to but
    not exactly ON the subject's last token (it's the token right before the blank) — a
    known fidelity approximation for RippleEdits specifically; call sites that need exact
    subject-token keys should not assume RippleEdits keys are captured as precisely as
    CounterFact's (which DOES supply an explicit "subject" string).
  - recent.json is OUT OF SCOPE for load_ripple_edits() (raises SystemExit with a clear
    message) — it has no original_fact to diff against. Its own criterion test_queries
    (which DO have clean cloze prompts) could still be read as pure probes for a future
    "insertion-mode" study, but that is not implemented here (never asked for — this loader
    builds `edits` for a REWRITE-collateral-damage study, and recent.json's records are
    insertions, not rewrites).
  - condition_queries (the official benchmark's precondition gate on which test_queries
    "count") are read into the returned probe dicts for provenance but NOT applied as a
    filter — a v1 simplification. ripple_geometry.py's per-criterion accuracy numbers are
    therefore an upper bound on the officially-gated numbers, not a reproduction of the
    paper's exact metric.
"""
from __future__ import annotations

import json
import os

import numpy as np

CRITERIA = (
    "Logical_Generalization", "Compositionality_I", "Compositionality_II",
    "Subject_Aliasing", "Relation_Specificity", "Forgetfulness",
)


def _word_common_prefix(a: str, b: str) -> str:
    """Longest common prefix of `a`/`b` at WORD boundaries (split on single spaces), not
    characters. Character-level common-prefix corrupts records where the old/new target
    strings share a leading substring by coincidence (e.g. "...is Carol Chu." vs "...is
    Casey DeSantis." would character-diff to "...is Ca", splitting a target string
    mid-word) — validated to affect ~5% of records; word-level diffing avoids this because
    "Carol" != "Casey" as whole tokens, so the prefix correctly stops before either.
    """
    wa, wb = a.split(" "), b.split(" ")
    i, n = 0, min(len(wa), len(wb))
    while i < n and wa[i] == wb[i]:
        i += 1
    pre = " ".join(wa[:i])
    return pre + " " if pre else ""


def _diff_edit(edit: dict):
    """(prompt_stem, target_new, target_true, skip_reason) for one RippleEdits `edit` dict
    that HAS an original_fact. skip_reason is None on success, else a short string
    ("no_original_fact" | "noop_edit" | "empty_new_target") — never raises; callers
    aggregate skip reasons for reporting rather than silently dropping records.
    """
    of = edit.get("original_fact")
    if not of:
        return None, None, None, "no_original_fact"
    p_new, p_old = edit["prompt"], of["prompt"]
    pre = _word_common_prefix(p_new, p_old)
    new_sfx = p_new[len(pre):].rstrip(".").strip()
    old_sfx = p_old[len(pre):].rstrip(".").strip()
    if not pre or not new_sfx:
        return None, None, None, "noop_edit" if p_new == p_old else "empty_new_target"
    return pre, new_sfx, old_sfx, None  # old_sfx may legitimately be "" (unknown prior value)


def _flatten_probes(record: dict, rec_idx: int, criteria):
    """Flatten one record's criterion test_queries into harness-record-like probe dicts.
    Returns dict {criterion_name: [probe, ...]} for criteria PRESENT on this record (a
    criterion key missing from `criteria` or absent/empty on the record is simply omitted,
    never invented as an empty placeholder that could be mistaken for "checked, found
    nothing").
    """
    out = {}
    for crit in criteria:
        entries = record.get(crit)
        if not entries:
            continue
        probes = []
        for sub in entries:
            for q in sub.get("test_queries", []):
                prompt = q.get("prompt") or q.get("phrase")
                answers = q.get("answers") or []
                if not prompt or not answers:
                    continue
                probes.append({
                    "subject": None,  # see module docstring KNOWN LIMITATIONS
                    "prompt": prompt,
                    "target_new": answers[0]["value"],
                    "target_true": None,  # a ripple test query has no "prior" notion
                    "aliases": [a.get("value") for a in answers],
                    "subject_id": q.get("subject_id"),
                    "relation": q.get("relation"),
                    "query_type": q.get("query_type"),
                    "criterion": crit,
                    "test_condition": sub.get("test_condition"),
                    "source_edit_index": rec_idx,
                })
        if probes:
            out[crit] = probes
    return out


def load_ripple_edits(path, n_edits, n_probes, seed=0, criteria=CRITERIA):
    """Load a RippleEdits file (data/rippleedits/{popular,random}.json — NOT recent.json,
    see module docstring) into harness-record format.

    Returns (edits, ripple_probes_by_criterion, unrelated_probes):
      edits: list of up to n_edits harness-format dicts (subject=None, prompt=cloze stem,
        target_new/target_true from the prefix diff), deterministically sampled
        (np.random.default_rng(seed).shuffle over the full file, first n_edits clean
        records taken in shuffled order).
      ripple_probes_by_criterion: dict {criterion: [probe, ...]} pooling test_queries from
        EXACTLY the selected edits' own records (i.e. these ARE the ripple implications of
        the chosen edits — the thing whose accuracy/damage this loader exists to measure).
      unrelated_probes: up to n_probes harness-format dicts built the SAME way as `edits`
        (prefix-diffed), drawn from a DISJOINT slice of the shuffled file (indices
        immediately following the edits slice) — mirrors load_counterfact's edits/probes
        split; used as the "unrelated collateral" comparison bank.

    Raises SystemExit if `path` is recent.json (or any file whose records lack
    original_fact) — see module docstring; never silently degrades to a guessed schema.
    """
    data = json.load(open(path))
    # structural, upfront check (not a post-hoc heuristic): a file where NO record has
    # original_fact (verified: recent.json, 0/1948) is out of scope for this loader's
    # rewrite-diffing method — fail loudly BEFORE spending a full pass skipping every
    # record one by one.
    if data and not any("original_fact" in d["edit"] for d in data[:min(20, len(data))]):
        raise SystemExit(
            f"[rippleedits_loader] {path}: none of the first {min(20, len(data))} records "
            f"carry original_fact (recent.json has NO original_fact for ANY of its 1948 "
            f"records — verified 2026-07-06) — this file is out of scope for "
            f"load_ripple_edits' rewrite-diffing method (see module docstring). Do not "
            f"guess a schema; use popular.json or random.json instead, or read this file's "
            f"own criterion test_queries directly as a pure-probe (no-edit) source if that "
            f"is what is actually wanted."
        )
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(data))

    skip_counts = {"no_original_fact": 0, "noop_edit": 0, "empty_new_target": 0}
    edits, ripple_by_crit, unrelated = [], {}, []
    i = 0
    while i < len(order) and len(edits) < n_edits:
        rec = data[int(order[i])]
        i += 1
        pre, tnew, ttrue, reason = _diff_edit(rec["edit"])
        if reason:
            skip_counts[reason] += 1
            continue
        edits.append({"subject": None, "prompt": pre, "target_new": tnew,
                      "target_true": ttrue, "subject_id": rec["edit"]["subject_id"],
                      "relation": rec["edit"]["relation"],
                      "example_type": rec.get("example_type")})
        for crit, probes in _flatten_probes(rec, len(edits) - 1, criteria).items():
            ripple_by_crit.setdefault(crit, []).extend(probes)

    while i < len(order) and len(unrelated) < n_probes:
        rec = data[int(order[i])]
        i += 1
        pre, tnew, ttrue, reason = _diff_edit(rec["edit"])
        if reason:
            skip_counts[reason] += 1
            continue
        unrelated.append({"subject": None, "prompt": pre, "target_new": tnew,
                          "target_true": ttrue, "subject_id": rec["edit"]["subject_id"],
                          "relation": rec["edit"]["relation"],
                          "example_type": rec.get("example_type")})

    meta = {"n_source_records": len(data), "n_consumed": i, "skip_counts": skip_counts,
            "n_edits": len(edits), "n_unrelated_probes": len(unrelated),
            "criteria_present": sorted(ripple_by_crit.keys()),
            "n_ripple_probes_per_criterion": {k: len(v) for k, v in ripple_by_crit.items()}}
    return edits, ripple_by_crit, unrelated, meta


if __name__ == "__main__":
    # CPU smoke: report record counts per criterion on whichever real files exist under
    # data/rippleedits/ — no model, no torch import needed for this path.
    import sys
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ddir = os.path.join(here, "data", "rippleedits")
    for fn in ("popular.json", "random.json", "recent.json"):
        p = os.path.join(ddir, fn)
        if not os.path.isfile(p):
            print(f"{fn}: NOT FOUND at {p}")
            continue
        if fn == "recent.json":
            try:
                load_ripple_edits(p, 50, 50, seed=0)
                print(f"{fn}: UNEXPECTED — should have raised (no original_fact)")
            except SystemExit as e:
                print(f"{fn}: correctly out-of-scope — {e}")
            continue
        edits, ripple, unrelated, meta = load_ripple_edits(p, 50, 50, seed=0)
        print(f"{fn}: {json.dumps(meta, indent=1)}")
