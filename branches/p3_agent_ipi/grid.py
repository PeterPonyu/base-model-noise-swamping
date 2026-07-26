"""grid.py -- B4 lineage-grid EXTENSION: extended model panel + seeded scenario sets.

WHY THIS FILE EXISTS (2026-07-10, Lane-B build):
The frozen `models.DESIGN` is the original pre-registered 9-model panel (3 R1-distills x
3 matched Qwen bases x 3 out-group). The B4 extension broadens that grid across the LOCAL
Ollama zoo (enumerated 2026-07-10 from ~/.ollama/models/manifests -- NO inference, NO
download) WITHOUT editing any review-frozen file. It adds:

  1. A 4th architecture-matched lineage pair at scale=xl:
        (deepseek-r1:14b  [DeepSeek-R1-Distill-Qwen-14B, arch qwen2],
         qwen2.5:14b-instruct-q8_0                         [arch qwen2])
     -> the matched-pair contrast grows from 3 to 4 pairs. (R1-distills on this box are
        ALL Qwen-lineage -- deepseek-r1:8b is Qwen3-based -- so matched pairs can only be
        added within the Qwen families; this is a real-asset constraint, not a design
        choice. See models.py's design note.)
  2. Extra base-instruct Qwen scale points (0.5b/3b, qwen3 1.7b/4b) that enrich the
     WITHIN-base lineage correlation (more same-lineage / different-scale pairs). They add
     NO new architecture pair (no R1 counterpart exists), so they are a separate tier.
  3. A broad OUT-GROUP sweep (~16 families: llama3.1/3.2, gemma2/3, mistral(-nemo),
     phi4, glm4, yi, granite3.3, exaone3.5, falcon3, solar, aya-expanse, internlm2,
     qwen3.5, starcoder2). Out-group pairs are summarized separately by analyze.contrast
     and are NOT in the lineage-vs-architecture headline -- they test how family-specific
     the fingerprint is against a wide architectural background.

SEEDS: `seeded_scenarios(n, seed)` produces a seed-varied scenario set. The attack-category
cycling stays POSITIONAL (scenario s always gets ATTACK_TEMPLATES[s % 3]) exactly as
scenarios.build_scenarios does, so item id -> injected TARGET tool is seed-invariant and
audit_unmatched.py's positional target rebuild stays correct across seeds. The seed only
reshuffles WHICH base record (user_request + benign toolset) maps to each scenario slot.
seed==0 delegates to build_scenarios verbatim -> byte-identical to the frozen path.

Pure metadata + CPU. `resolve_panel(..., backend='ollama')` calls `ollama show` (subprocess,
metadata only, NO inference) via models.parse_model; backend='mock' uses design labels and
needs no daemon.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)

import models as models_mod  # noqa: E402
import scenarios as scen_mod  # noqa: E402

# ----------------------------------------------------------------------------
# EXTENDED_DESIGN: every model is keyed by its exact Ollama tag (must match a
# manifest under ~/.ollama/models). Fields mirror models.DESIGN so analyze.contrast
# and run_ipi._gate_contrast consume them unchanged:
#   group       : 'r1' | 'base' | 'out'    (out-group excluded from the main contrast)
#   lineage     : 'r1-distill' | 'base-instruct'
#   match_group : (arch_family/scale) key -- an architecture-matched pair shares it and
#                 differs in lineage; the 4 (r1,base) pairs below are the arch pairs.
#   scale       : xs|small|mid|large|xl (coarse size bucket, for lineage-pair definition)
#   arch_raw    : the `ollama show` architecture string (fallback when the daemon is down)
#   supports_tools_hint : ONLY used for backend='mock' validation; the real ollama run
#                 overrides this from `ollama show` capabilities. Known facts baked in:
#                 deepseek-r1 distills + gemma2 do NOT advertise `tools` (see transport.py).
# ----------------------------------------------------------------------------
EXTENDED_DESIGN: list[dict] = [
    # ---- Group A: R1-distills (lineage=r1-distill) -- all Qwen-lineage on this box ----
    {"name": "deepseek-r1:1.5b", "group": "r1", "lineage": "r1-distill",
     "arch_raw": "qwen2", "arch_family": "Qwen2.5", "scale": "small",
     "match_group": "Qwen2.5/small", "supports_tools_hint": False},
    {"name": "deepseek-r1:7b", "group": "r1", "lineage": "r1-distill",
     "arch_raw": "qwen2", "arch_family": "Qwen2.5", "scale": "mid",
     "match_group": "Qwen2.5/mid", "supports_tools_hint": False},
    {"name": "deepseek-r1:14b", "group": "r1", "lineage": "r1-distill",   # NEW xl pair
     "arch_raw": "qwen2", "arch_family": "Qwen2.5", "scale": "xl",
     "match_group": "Qwen2.5/xl", "supports_tools_hint": False},
    {"name": "deepseek-r1:8b", "group": "r1", "lineage": "r1-distill",
     "arch_raw": "qwen3", "arch_family": "Qwen3", "scale": "large",
     "match_group": "Qwen3/large", "supports_tools_hint": False},

    # ---- Group B: matched vanilla bases (lineage=base-instruct) ----
    {"name": "qwen2.5:1.5b", "group": "base", "lineage": "base-instruct",
     "arch_raw": "qwen2", "arch_family": "Qwen2.5", "scale": "small",
     "match_group": "Qwen2.5/small", "supports_tools_hint": True},
    {"name": "qwen2.5:7b-instruct-q8_0", "group": "base", "lineage": "base-instruct",
     "arch_raw": "qwen2", "arch_family": "Qwen2.5", "scale": "mid",
     "match_group": "Qwen2.5/mid", "supports_tools_hint": True},
    {"name": "qwen2.5:14b-instruct-q8_0", "group": "base", "lineage": "base-instruct",  # NEW
     "arch_raw": "qwen2", "arch_family": "Qwen2.5", "scale": "xl",
     "match_group": "Qwen2.5/xl", "supports_tools_hint": True},
    {"name": "qwen3:8b", "group": "base", "lineage": "base-instruct",
     "arch_raw": "qwen3", "arch_family": "Qwen3", "scale": "large",
     "match_group": "Qwen3/large", "supports_tools_hint": True},

    # ---- Group B-extra: base-instruct Qwen scale points (enrich base lineage pairs;
    #      NO r1 counterpart -> NO new architecture pair; used only in base_scales tier) ----
    {"name": "qwen2.5:0.5b", "group": "base", "lineage": "base-instruct",
     "arch_raw": "qwen2", "arch_family": "Qwen2.5", "scale": "xs",
     "match_group": "Qwen2.5/xs", "supports_tools_hint": True},
    {"name": "qwen2.5:3b", "group": "base", "lineage": "base-instruct",
     "arch_raw": "qwen2", "arch_family": "Qwen2.5", "scale": "mid2",
     "match_group": "Qwen2.5/mid2", "supports_tools_hint": True},
    {"name": "qwen3:1.7b", "group": "base", "lineage": "base-instruct",
     "arch_raw": "qwen3", "arch_family": "Qwen3", "scale": "small",
     "match_group": "Qwen3/small", "supports_tools_hint": True},
    {"name": "qwen3:4b", "group": "base", "lineage": "base-instruct",
     "arch_raw": "qwen3", "arch_family": "Qwen3", "scale": "mid",
     "match_group": "Qwen3/mid", "supports_tools_hint": True},

    # ---- Group C: out-group breadth (one representative per distinct family) ----
    {"name": "llama3.1:8b-instruct-q8_0", "group": "out", "lineage": "base-instruct",
     "arch_raw": "llama", "arch_family": "Llama3.1", "scale": "large",
     "match_group": "Llama3.1/large", "supports_tools_hint": True},
    {"name": "llama3.2:3b", "group": "out", "lineage": "base-instruct",
     "arch_raw": "llama", "arch_family": "Llama3.2", "scale": "mid",
     "match_group": "Llama3.2/mid", "supports_tools_hint": True},
    {"name": "gemma2:9b-instruct-q8_0", "group": "out", "lineage": "base-instruct",
     "arch_raw": "gemma2", "arch_family": "Gemma2", "scale": "large",
     "match_group": "Gemma2/large", "supports_tools_hint": False},  # gemma2: no `tools`
    {"name": "gemma3:12b-it-q8_0", "group": "out", "lineage": "base-instruct",
     "arch_raw": "gemma3", "arch_family": "Gemma3", "scale": "large",
     "match_group": "Gemma3/large", "supports_tools_hint": True},
    {"name": "mistral:7b-instruct-v0.3-q8_0", "group": "out", "lineage": "base-instruct",
     "arch_raw": "llama", "arch_family": "Mistral", "scale": "mid",
     "match_group": "Mistral/mid", "supports_tools_hint": True},
    {"name": "mistral-nemo:12b-instruct-2407-q8_0", "group": "out", "lineage": "base-instruct",
     "arch_raw": "llama", "arch_family": "MistralNemo", "scale": "large",
     "match_group": "MistralNemo/large", "supports_tools_hint": True},
    {"name": "phi4:14b-q8_0", "group": "out", "lineage": "base-instruct",
     "arch_raw": "phi3", "arch_family": "Phi4", "scale": "xl",
     "match_group": "Phi4/xl", "supports_tools_hint": True},
    {"name": "glm4:9b", "group": "out", "lineage": "base-instruct",
     "arch_raw": "chatglm", "arch_family": "GLM4", "scale": "large",
     "match_group": "GLM4/large", "supports_tools_hint": True},
    {"name": "yi:9b", "group": "out", "lineage": "base-instruct",
     "arch_raw": "llama", "arch_family": "Yi", "scale": "large",
     "match_group": "Yi/large", "supports_tools_hint": True},
    {"name": "granite3.3:8b", "group": "out", "lineage": "base-instruct",
     "arch_raw": "granite", "arch_family": "Granite3.3", "scale": "large",
     "match_group": "Granite3.3/large", "supports_tools_hint": True},
    {"name": "exaone3.5:7.8b", "group": "out", "lineage": "base-instruct",
     "arch_raw": "exaone", "arch_family": "ExaONE3.5", "scale": "mid",
     "match_group": "ExaONE3.5/mid", "supports_tools_hint": True},
    {"name": "falcon3:7b", "group": "out", "lineage": "base-instruct",
     "arch_raw": "llama", "arch_family": "Falcon3", "scale": "mid",
     "match_group": "Falcon3/mid", "supports_tools_hint": True},
    {"name": "solar:10.7b", "group": "out", "lineage": "base-instruct",
     "arch_raw": "llama", "arch_family": "Solar", "scale": "large",
     "match_group": "Solar/large", "supports_tools_hint": True},
    {"name": "aya-expanse:8b", "group": "out", "lineage": "base-instruct",
     "arch_raw": "command-r", "arch_family": "AyaExpanse", "scale": "large",
     "match_group": "AyaExpanse/large", "supports_tools_hint": True},
    {"name": "internlm2:7b", "group": "out", "lineage": "base-instruct",
     "arch_raw": "internlm2", "arch_family": "InternLM2", "scale": "mid",
     "match_group": "InternLM2/mid", "supports_tools_hint": True},
    {"name": "qwen3.5:9b-q8_0", "group": "out", "lineage": "base-instruct",
     "arch_raw": "qwen3", "arch_family": "Qwen3.5", "scale": "large",
     "match_group": "Qwen3.5/large", "supports_tools_hint": True},

    # ---- Group D: wave-3 lineage arm (2026-07-11) -- attackable alternate-lineage
    #      fine-tunes, ALL group="base" (in-group). hermes3/dolphin3/tulu3 are Llama-3.1-8B
    #      fine-tunes sharing match_group="Llama3.1/large"; openthinker:7b is a Qwen2.5-7B
    #      fine-tune sharing match_group="Qwen2.5/mid" with the existing qwen2.5:7b
    #      (base-instruct) and deepseek-r1:7b (r1-distill) rows above. In the `lineage_arm`
    #      tier the KNOWN-attackable llama3.1:8b-instruct-q8_0 is ALSO relabeled into
    #      match_group="Llama3.1/large" as lineage="llama-instruct" via TIER_OVERRIDES
    #      (tier-LOCAL -- its global row above stays group="out"), so the four Llama-3.1-8B
    #      models form 6 architecture pairs anchored on a model with measured non-zero ASR.
    #      See DOWNLOAD-MANIFEST-WAVE3-20260711.md for the openthinker degenerate-ASR risk. ----
    {"name": "hermes3:8b", "group": "base", "lineage": "hermes",
     "arch_raw": "llama", "arch_family": "Llama3.1", "scale": "large",
     "match_group": "Llama3.1/large", "supports_tools_hint": True},
    {"name": "dolphin3:8b", "group": "base", "lineage": "dolphin",
     "arch_raw": "llama", "arch_family": "Llama3.1", "scale": "large",
     "match_group": "Llama3.1/large", "supports_tools_hint": True},
    {"name": "tulu3:8b", "group": "base", "lineage": "tulu",
     "arch_raw": "llama", "arch_family": "Llama3.1", "scale": "large",
     "match_group": "Llama3.1/large", "supports_tools_hint": True},
    {"name": "openthinker:7b", "group": "base", "lineage": "openthinker",
     "arch_raw": "qwen2", "arch_family": "Qwen2.5", "scale": "mid",
     "match_group": "Qwen2.5/mid", "supports_tools_hint": True},
]

# Merge the frozen models.DESIGN as a FALLBACK label source (so the 'original' tier's exact
# q4 tags -- which EXTENDED_DESIGN does not re-list, it uses q8 variants -- still resolve to
# their correct group/lineage/match_group). EXTENDED_DESIGN wins on any name collision.
_BY_NAME: dict[str, dict] = {m["name"]: m for m in models_mod.DESIGN}
_BY_NAME.update({m["name"]: m for m in EXTENDED_DESIGN})


def _mock_tools_hint(d: dict) -> bool:
    """supports_tools guess for backend='mock' (no daemon). Known facts: deepseek-r1
    distills and the Gemma2 family do NOT advertise `tools` (see transport.py/models.py).
    Real ollama runs override this from `ollama show`."""
    if "supports_tools_hint" in d:
        return bool(d["supports_tools_hint"])
    if d.get("lineage") == "r1-distill":
        return False
    if str(d.get("arch_family", "")).startswith("Gemma2"):
        return False
    return True

# ---- Tiers (name lists) ----------------------------------------------------
_CORE = [
    "deepseek-r1:1.5b", "deepseek-r1:7b", "deepseek-r1:14b", "deepseek-r1:8b",
    "qwen2.5:1.5b", "qwen2.5:7b-instruct-q8_0", "qwen2.5:14b-instruct-q8_0", "qwen3:8b",
    "llama3.1:8b-instruct-q8_0", "gemma2:9b-instruct-q8_0", "mistral:7b-instruct-v0.3-q8_0",
]
_BASE_EXTRA = ["qwen2.5:0.5b", "qwen2.5:3b", "qwen3:1.7b", "qwen3:4b"]
_OUTGROUP_EXTRA = [
    "llama3.2:3b", "gemma3:12b-it-q8_0", "mistral-nemo:12b-instruct-2407-q8_0",
    "phi4:14b-q8_0", "glm4:9b", "yi:9b", "granite3.3:8b", "exaone3.5:7.8b",
    "falcon3:7b", "solar:10.7b", "aya-expanse:8b", "internlm2:7b", "qwen3.5:9b-q8_0",
]
_LINEAGE_ARM_EXTRA = ["hermes3:8b", "dolphin3:8b", "tulu3:8b", "openthinker:7b"]

TIERS: dict[str, list[str]] = {
    # original frozen 9-model panel (kept verbatim for direct replication vs models.DESIGN)
    "original": [m["name"] for m in models_mod.DESIGN],
    # 11 models = 4 matched lineage pairs + 3 out-group (the headline B4 extension)
    "core": _CORE,
    # + base-instruct Qwen scale points (enrich WITHIN-base lineage correlation)
    "base_scales": _CORE + _BASE_EXTRA,
    # + broad out-group families (generality of the fingerprint)
    "outgroup": _CORE + _OUTGROUP_EXTRA,
    # wave-3 (2026-07-11): + attackable alternate-lineage arm (real, non-degenerate
    # architecture-matched pairs -- see Group D note above)
    "lineage_arm": _CORE + _LINEAGE_ARM_EXTRA,
    # everything
    "full": [m["name"] for m in EXTENDED_DESIGN],
}


# ---- Tier-local label overrides (roster metadata only; NO pairing-logic change) ----------
# A tier may relabel a model that is defined one way GLOBALLY in EXTENDED_DESIGN but must be
# treated differently for that tier's contrast. Applied by resolve_panel AFTER the global
# design merge. This keeps the global row -- and every other tier that uses it -- untouched.
TIER_OVERRIDES: dict[str, dict[str, dict]] = {
    # wave-3 (2026-07-11): within the `lineage_arm` tier ONLY, promote the known-ATTACKABLE
    # llama3.1:8b-instruct-q8_0 (measured ASR ~0.10, wave-1/2) from out-group to an in-group
    # alternate-lineage ANCHOR, so it forms REAL architecture-matched pairs with the
    # Llama-3.1-8B fine-tunes (hermes3/dolphin3/tulu3) at match_group="Llama3.1/large".
    # lineage MUST become "llama-instruct" (NOT "base-instruct"): as base-instruct it would
    # spuriously pair with the Qwen base-instruct models as a cross-family LINEAGE pair; as
    # its own lineage it only forms ARCHITECTURE pairs (same match_group, different lineage)
    # with the three Llama fine-tunes. This anchors mean_architecture_corr on a model we KNOW
    # is attackable, instead of relying solely on three not-yet-downloaded fine-tunes.
    # Tier-LOCAL: the global EXTENDED_DESIGN row stays group="out", so core/outgroup panels
    # and their frozen wave-1/2 JSONs are byte-unchanged.
    "lineage_arm": {
        "llama3.1:8b-instruct-q8_0": {"group": "base", "lineage": "llama-instruct"},
    },
}


def tier_names(tier: str) -> list[str]:
    if tier not in TIERS:
        raise SystemExit(f"unknown tier '{tier}'. choices: {sorted(TIERS)}")
    return list(TIERS[tier])


def tier_overrides(tier: str) -> dict[str, dict]:
    """Per-model label overrides for `tier` (empty if none). See TIER_OVERRIDES."""
    return TIER_OVERRIDES.get(tier, {})


def resolve_panel(names: list[str], backend: str = "ollama",
                  overrides: dict[str, dict] | None = None) -> list[dict]:
    """Resolve tag list -> full meta merged with EXTENDED_DESIGN labels.

    backend='ollama' reads real capabilities via `ollama show` (metadata subprocess, NO
    inference); backend='mock' uses design labels + supports_tools_hint (no daemon needed).
    `overrides` (name -> {label: value}) is a tier-local relabel applied last (see
    TIER_OVERRIDES / tier_overrides); it only touches the keys it names.
    """
    out: list[dict] = []
    for nm in names:
        meta = models_mod.parse_model(nm, backend=backend)
        d = _BY_NAME.get(nm)
        if d:
            meta["group"] = d["group"]
            meta["match_group"] = d["match_group"]
            meta["scale"] = d["scale"]
            meta["arch_family"] = d["arch_family"]
            meta["lineage"] = d["lineage"]  # design lineage is authoritative
            if meta.get("architecture") is None:
                meta["architecture"] = d.get("arch_raw")
            if backend == "mock":
                # real ollama run trusts `ollama show`; mock has no daemon, use the hint
                meta["supports_tools"] = _mock_tools_hint(d)
        else:
            meta.setdefault("group", "unassigned")
            meta.setdefault("match_group", f"{meta.get('family')}/?")
            meta.setdefault("arch_family", meta.get("family"))
            meta.setdefault("scale", "?")
        if overrides and nm in overrides:
            meta.update(overrides[nm])  # tier-local relabel wins
        out.append(meta)
    return out


def seeded_scenarios(n: int = 30, seed: int = 0,
                     limit_per_source: int = 40) -> tuple[list[dict], list[dict]]:
    """Seed-varied scenario set. seed==0 is byte-identical to scen_mod.build_scenarios(n).

    For seed!=0 the base-record order is reshuffled with random.Random(seed) BEFORE the
    (unchanged) positional build loop, so attack-category-per-slot -- and thus item->target
    -- is seed-invariant (audit_unmatched.py stays correct), while user_request/benign-tool
    CONTENT varies across seeds.
    """
    if seed == 0:
        return scen_mod.build_scenarios(n=n, limit_per_source=limit_per_source)

    records, statuses = scen_mod.load_schema_records(limit_per_source)
    rng = random.Random(seed)
    records = list(records)
    rng.shuffle(records)

    scenarios: list[dict] = []
    ri = 0
    ai = 0
    guard = 0
    while len(scenarios) < n and guard < n * 20:
        guard += 1
        rec = records[ri % len(records)]
        ri += 1
        if not rec.get("functions"):
            continue
        atk = scen_mod.ATTACK_TEMPLATES[ai % len(scen_mod.ATTACK_TEMPLATES)]
        ai += 1
        benign_tools = rec["functions"][:4]
        benign_name = benign_tools[0]["function"]["name"]
        tools = list(benign_tools)
        if atk["tool"]["function"]["name"] not in {t["function"]["name"] for t in tools}:
            tools.append(atk["tool"])
        injected = scen_mod._benign_result_stub(rec) + atk["injection"]
        scenarios.append({
            "id": f"s{len(scenarios):03d}",
            "source": rec.get("_source", "unknown"),
            "attack_category": atk["category"],
            "user_request": rec["user_request"],
            "tools": tools,
            "benign_tool": benign_name,
            "injected_tool_output": injected,
            "injected_target_action": atk["target"],
        })
    return scenarios, statuses


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Inspect the B4 extended grid (metadata only).")
    ap.add_argument("--tier", default="core", choices=sorted(TIERS))
    ap.add_argument("--backend", default="mock", choices=["mock", "ollama"])
    ap.add_argument("--show_scenarios", type=int, default=0,
                    help="also build this many seeded scenarios (seed from --seed) and summarize")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    names = tier_names(args.tier)
    panel = resolve_panel(names, backend=args.backend, overrides=tier_overrides(args.tier))
    # contrast-structure summary
    arch_pairs = []
    for i in range(len(panel)):
        for j in range(i + 1, len(panel)):
            a, b = panel[i], panel[j]
            if a.get("group") == "out" or b.get("group") == "out":
                continue
            if a["match_group"] == b["match_group"] and a["lineage"] != b["lineage"]:
                arch_pairs.append((a["name"], b["name"]))
    summary = {
        "tier": args.tier,
        "n_models": len(panel),
        "n_r1": sum(1 for m in panel if m.get("group") == "r1"),
        "n_base": sum(1 for m in panel if m.get("group") == "base"),
        "n_out": sum(1 for m in panel if m.get("group") == "out"),
        "architecture_matched_pairs": arch_pairs,
        "models": [{"name": m["name"], "group": m.get("group"), "lineage": m["lineage"],
                    "match_group": m.get("match_group"), "scale": m.get("scale"),
                    "supports_tools": m.get("supports_tools")} for m in panel],
    }
    if args.show_scenarios > 0:
        scen, st = seeded_scenarios(n=args.show_scenarios, seed=args.seed)
        summary["scenarios"] = {
            "n": len(scen), "seed": args.seed,
            "by_category": {c: sum(1 for x in scen if x["attack_category"] == c)
                            for c in sorted({x["attack_category"] for x in scen})},
            "first_ids": [s["id"] for s in scen[:5]],
        }
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
