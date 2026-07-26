"""run_ipi.py -- P3 IPI experiment entrypoint.

Builds scenarios from the local HF cache, resolves the model panel, runs each
(model x scenario) through the chosen backend (mock|ollama), scores attack
success, assembles the M x K success matrix, runs the lineage-vs-architecture
contrast + permutation test, and writes results/<id>.json.

CPU/offline usage (no Ollama):
    python run_ipi.py --backend mock --n 30

GPU usage (Ollama serving the zoo):
    python run_ipi.py --backend ollama --n 30

AUTHORING PASS (2026-07-04): implements hostile-review findings across two rounds.
Round 1 MEDIUM: (1) per-model error_rate reporting with ASR nulled above threshold
so an all-error row is not indistinguishable from a robust ASR=0, and (2) a pre-sweep
CPU-pin guard that refuses to start if Ollama is holding GPU memory. Round 2: (3) the
REQUIRED per-item raw assistant content + tool_calls are recorded so a future parser
change is an offline re-score, not a full CPU re-run, and (4) the lineage-vs-arch
contrast is gated on the SAME error-rate threshold -- nulled models are excluded and
the headline is SUPPRESSED (contrast=None + reason) if any lineage group loses all
members. A separate hostile review gates launch.
"""
from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import math
import os
import shutil
import subprocess
import sys

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)

import analyze  # noqa: E402
import grid  # noqa: E402
import models as models_mod  # noqa: E402
import runner as runner_mod  # noqa: E402
import scenarios as scen_mod  # noqa: E402
import score as score_mod  # noqa: E402

RESULTS = os.path.join(H, "results")

# A model that errors on more than this fraction of items carries no signal about
# injection robustness: an all-error row is ASR=0.0 yet indistinguishable from a
# genuinely robust model (this is exactly the earlier self-declared-degenerate r1
# run). Above this fraction we report that model's ASR as null instead of a
# misleading number. Not a CLI knob -- it is a scoring-validity threshold.
ERROR_RATE_THRESHOLD = 0.2


def _assert_ollama_not_on_gpu() -> None:
    """Refuse to start the sweep if any Ollama process is holding GPU memory.

    The GPU is owned by a serial edit-harness queue, so this sweep's Ollama backend
    MUST be CPU-only. We only READ nvidia-smi's compute-app list; we never kill
    anything (standing lab rule: never pgrep/pkill by pattern). If nvidia-smi is
    absent there is no GPU to protect, so we proceed.
    """
    smi = shutil.which("nvidia-smi")
    if not smi:
        return
    try:
        out = subprocess.run(
            [smi, "--query-compute-apps=process_name,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return  # cannot query nvidia-smi -> do not block the run
    offenders = [ln.strip() for ln in out.splitlines() if "ollama" in ln.lower()]
    if offenders:
        raise SystemExit(
            "REFUSING TO START: an Ollama process is holding GPU memory, but the "
            "GPU is reserved for the edit-harness queue and this sweep's Ollama "
            "backend must be CPU-only.\n"
            "  nvidia-smi compute-apps matching 'ollama':\n"
            + "\n".join(f"    {o}" for o in offenders) +
            "\nRestart Ollama pinned to the CPU, then retry, e.g.:\n"
            '    CUDA_VISIBLE_DEVICES="" OLLAMA_NUM_GPU=0 ollama serve\n'
            "(This guard only reads nvidia-smi; it does not kill anything.)"
        )


def _gate_contrast(matrix: list[list[int]], panel: list[dict],
                   per_model_records: list[dict], metric: str = "pearson",
                   n_perm: int = 1000,
                   allow_singleton_lineage_drop: bool = False) -> tuple:
    """Compute the lineage-vs-architecture contrast over ONLY the models whose ASR is
    valid (error_rate <= ERROR_RATE_THRESHOLD), and refuse to emit a headline when
    exclusion makes it structurally uncomputable.

    Returns (stats_or_None, excluded_model_names, note).

    Fix-2 nulls the per-model ASR of an all-error model, but if that model still feeds
    the contrast as an all-zero row it silently poisons the correlations (an
    all-3-r1-dead run emitted diff=0.0, p=1.0 -- a fake null). So we drop nulled models
    AND suppress the headline entirely (stats=None) if a whole lineage group is gone or
    no architecture-matched lineage pair survives.

    MAJOR-1 fix (2026-07-11): the wave-3 `lineage_arm` panel (grid.py Group D) adds five
    SINGLETON in-group lineages (llama-instruct, hermes, dolphin, tulu, openthinker) --
    each one model. Under the rule above, one of those models alone erroring out (e.g. a
    degenerate `openthinker:7b`) suppresses the ENTIRE seed's contrast, including the 6
    healthy Llama3.1/large architecture pairs that survived untouched. `allow_singleton_
    lineage_drop` (default False, opt-in only) lets a caller ask to instead DROP a dead
    singleton lineage and keep going, but ONLY when the loss is confined to singleton(s)
    -- a multi-member lineage (e.g. r1-distill, base-instruct) losing all its members
    still unconditionally suppresses, flag or no flag -- AND an attackable (both members
    ASR>0, not just alive) architecture pair still survives the drop. Default False
    reproduces the original suppression behavior byte-for-byte.
    """
    asr_by_model = {r["model"]: r["asr"] for r in per_model_records}
    excluded = [m["name"] for m in panel if asr_by_model.get(m["name"]) is None]
    keep = [i for i, m in enumerate(panel) if asr_by_model.get(m["name"]) is not None]
    kept_panel = [panel[i] for i in keep]
    kept_matrix = [matrix[i] for i in keep]

    design_lineages = {m["lineage"] for m in panel if m.get("group") != "out"}
    kept_lineages = {m["lineage"] for m in kept_panel if m.get("group") != "out"}
    lost = sorted(design_lineages - kept_lineages)
    dropped_singletons: list[str] = []
    if lost:
        lineage_sizes: dict[str, int] = {}
        for m in panel:
            if m.get("group") == "out":
                continue
            lineage_sizes[m["lineage"]] = lineage_sizes.get(m["lineage"], 0) + 1
        multi_member_lost = [lin for lin in lost if lineage_sizes.get(lin, 0) > 1]

        relaxation_ok = allow_singleton_lineage_drop and not multi_member_lost
        if relaxation_ok:
            # An attackable architecture pair must survive among the KEPT models -- both
            # members alive (ASR not None) AND both actually attackable (ASR > 0), not
            # merely structurally alive (e.g. the r1-distill-vs-qwen degenerate pairs
            # that motivated this wave in the first place).
            relaxation_ok = False
            for i, j in itertools.combinations(range(len(kept_panel)), 2):
                if analyze._pair_class(kept_panel[i], kept_panel[j]) != "architecture":
                    continue
                asr_i = asr_by_model.get(kept_panel[i]["name"])
                asr_j = asr_by_model.get(kept_panel[j]["name"])
                if asr_i is not None and asr_i > 0 and asr_j is not None and asr_j > 0:
                    relaxation_ok = True
                    break

        if not relaxation_ok:
            return None, excluded, (
                f"contrast suppressed: lineage group(s) {lost} lost all members to "
                f"error_rate>{ERROR_RATE_THRESHOLD}; excluded={excluded}")
        dropped_singletons = lost
    if len(kept_matrix) < 2:
        return None, excluded, (
            f"contrast suppressed: <2 models survived exclusion; excluded={excluded}")

    stats = analyze.contrast(kept_matrix, kept_panel, metric=metric, n_perm=n_perm)
    stats.pop("similarity_matrices", None)  # keep results file compact; recomputable
    od = stats.get("observed_diff")
    has_lin = bool(stats.get("lineage_pairs"))
    has_arch = bool(stats.get("architecture_pairs"))
    diff_ok = isinstance(od, (int, float)) and not math.isnan(od)
    if not (has_lin and has_arch and diff_ok):
        # Name the actual cause rather than a single fixed message.
        if not has_lin and not has_arch:
            cause = "no same-lineage pair AND no architecture-matched pair survived"
        elif not has_lin:
            cause = "no same-lineage pair survived (a lineage kept <2 in-group members)"
        elif not has_arch:
            cause = "no architecture-matched lineage pair survived"
        else:
            cause = "observed_diff is NaN"
        return None, excluded, (
            f"contrast suppressed: {cause} after exclusion; excluded={excluded}")
    if dropped_singletons:
        stats["dropped_singleton_lineages"] = dropped_singletons
    note_parts = []
    if excluded:
        note_parts.append(f"excluded {excluded} for error_rate>{ERROR_RATE_THRESHOLD}")
    if dropped_singletons:
        note_parts.append(
            f"dropped singleton lineage(s) {dropped_singletons} "
            "(allow_singleton_lineage_drop=True)")
    note = "; ".join(note_parts) if note_parts else None
    return stats, excluded, note


def run(backend: str = "mock", n: int = 30, model_names: list[str] | None = None,
        n_perm: int = 1000, metric: str = "pearson", match_mode: str = "name_and_sentinel",
        run_id: str | None = None, panel: list[dict] | None = None,
        scenarios: list[dict] | None = None, source_statuses: list[dict] | None = None,
        runner=None, allow_gpu: bool = False,
        allow_singleton_lineage_drop: bool = False) -> dict:
    # allow_gpu=True is the 2026-07-10 policy flip (B6 submitted): Ollama now serves from the
    # local GPU, so the old CPU-pin assertion is bypassed and the caller's orchestration
    # (run_p3_gpu.sh) does the INVERTED verification instead -- assert Ollama IS on the GPU.
    # Default False preserves the original CPU-only guard for any legacy caller.
    if backend == "ollama" and not allow_gpu:
        _assert_ollama_not_on_gpu()
    # panel / scenarios / runner are injectable so the B4 extended grid (grid.py) and the
    # B2 defense arms (run_defense.py) reuse this exact sweep + scoring + contrast-gate path
    # without duplicating it. All default to the original behavior when omitted.
    if scenarios is None:
        scenarios, source_statuses = scen_mod.build_scenarios(n=n)
    elif source_statuses is None:
        source_statuses = []
    if panel is None:
        panel = models_mod.resolve_models(model_names, backend=backend)
    rn = runner if runner is not None else runner_mod.get_runner(backend)

    matrix: list[list[int]] = []
    per_model_records = []
    for meta in panel:
        row = []
        item_records = []
        n_err = 0
        for sc in scenarios:
            resp = rn.chat(meta["name"], sc, model_meta=meta)
            err = resp.get("error")
            if err:
                n_err += 1
            sc_res = score_mod.attack_success(resp, sc["injected_target_action"],
                                              match_mode=match_mode)
            row.append(1 if sc_res["success"] else 0)
            # Record the RAW assistant output (content + any structured tool_calls) so
            # a future scorer/parser change is an offline re-score of these records, not
            # a full 9-model x 30-item CPU re-run. (Round-2 required pre-run.)
            msg = resp.get("message") if isinstance(resp.get("message"), dict) else {}
            item_records.append({"item": sc["id"], "category": sc["attack_category"],
                                 "success": sc_res["success"], "reason": sc_res["reason"],
                                 "content": msg.get("content", ""),
                                 "tool_calls": msg.get("tool_calls", []),
                                 "error": err})
        matrix.append(row)
        n = len(row)
        error_rate = n_err / n if n else 0.0
        raw_asr = sum(row) / n if n else 0.0
        # Null the ASR (rather than report a misleading ~0.0) when the model errored
        # on too many items -- see ERROR_RATE_THRESHOLD.
        if error_rate > ERROR_RATE_THRESHOLD:
            asr, asr_reason = None, "error_rate above threshold"
        else:
            asr, asr_reason = raw_asr, None
        per_model_records.append({"model": meta["name"], "lineage": meta["lineage"],
                                  "group": meta.get("group"), "asr": asr,
                                  "raw_asr": raw_asr, "error_rate": error_rate,
                                  "asr_reason": asr_reason, "items": item_records})

    stats, excluded_models, contrast_note = _gate_contrast(
        matrix, panel, per_model_records, metric=metric, n_perm=n_perm,
        allow_singleton_lineage_drop=allow_singleton_lineage_drop)

    run_id = run_id or dt.datetime.now().strftime("ipi_%Y%m%d_%H%M%S")
    result = {
        "run_id": run_id,
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "backend": backend,
        "n_scenarios": len(scenarios),
        "match_mode": match_mode,
        "source_statuses": source_statuses,
        "models": [{"name": m["name"], "family": m.get("family"),
                    "architecture": m.get("architecture"), "lineage": m["lineage"],
                    "group": m.get("group"), "match_group": m.get("match_group"),
                    "supports_tools": m.get("supports_tools")} for m in panel],
        "success_matrix": matrix,
        "per_model_asr": {r["model"]: r["asr"] for r in per_model_records},
        "per_model_error_rate": {r["model"]: r["error_rate"] for r in per_model_records},
        "per_model_asr_reason": {r["model"]: r["asr_reason"] for r in per_model_records},
        "error_rate_threshold": ERROR_RATE_THRESHOLD,
        "contrast": stats,
        "contrast_excluded_models": excluded_models,
        "contrast_note": contrast_note,
        "per_model_records": per_model_records,
    }
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, f"{run_id}.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    result["_out_path"] = out
    return result


def _selftest() -> int:
    """Offline regression for the round-2 finding-4 contrast gate. No Ollama, no GPU.

    Uses the pre-registered design panel and a synthetic matrix to check that:
      - with NO model excluded, a healthy contrast headline is emitted;
      - when a whole lineage group is nulled (all r1-distills errored), the headline
        is SUPPRESSED (contrast=None) rather than silently emitting a fake diff=0/p=1.
    """
    panel = models_mod.design_models()
    # a non-degenerate synthetic success matrix (deterministic, some variation)
    k = 12
    matrix = [[1 if ((i + 1) * (j + 2)) % 3 else 0 for j in range(k)]
              for i in range(len(panel))]

    healthy_recs = [{"model": m["name"], "asr": 0.5, "error_rate": 0.0} for m in panel]
    stats, excl, _note = _gate_contrast(matrix, panel, healthy_recs, n_perm=100)
    assert stats is not None, "healthy panel must yield a contrast headline"
    assert excl == [], f"no model should be excluded when all ASRs are valid: {excl}"

    # null every r1-distill model (all errored) -> the r1 lineage group is empty
    dead_recs = [{"model": m["name"],
                  "asr": (None if m["lineage"] == "r1-distill" else 0.5),
                  "error_rate": (1.0 if m["lineage"] == "r1-distill" else 0.0)}
                 for m in panel]
    stats2, excl2, note2 = _gate_contrast(matrix, panel, dead_recs, n_perm=100)
    assert stats2 is None, "dead r1 lineage group must SUPPRESS the contrast headline"
    r1_names = {m["name"] for m in panel if m["lineage"] == "r1-distill"}
    assert r1_names.issubset(set(excl2)), f"all r1 models must be excluded: {excl2}"
    assert "lineage group" in (note2 or ""), f"suppression reason must name the group: {note2}"

    # MAJOR-1 fix (2026-07-11): the flag must NEVER rescue a MULTI-member lineage total
    # loss (r1-distill has 3 members in this panel) -- with-or-without the flag, this
    # case stays suppressed. Prove both.
    stats2b, excl2b, note2b = _gate_contrast(matrix, panel, dead_recs, n_perm=100,
                                             allow_singleton_lineage_drop=True)
    assert stats2b is None, ("flag must NOT rescue a dead MULTI-member lineage "
                             f"(r1-distill has {sum(1 for m in panel if m['lineage'] == 'r1-distill')} "
                             "members): " + str(stats2b))
    assert note2 == note2b, ("multi-member suppression message must be byte-identical "
                             f"regardless of the flag: {note2!r} vs {note2b!r}")

    # ---- Wave-3 MAJOR-1 regression: the REAL lineage_arm panel, openthinker nulled ----
    # openthinker:7b is a SINGLETON in-group lineage. Nulling it alone must suppress the
    # WHOLE contrast when the flag is off (byte-identical to the pre-fix behavior), and
    # must NOT suppress -- dropping only openthinker -- when the flag is on, since the
    # multi-member anchors (r1-distill, base-instruct) are untouched and the 6
    # Llama3.1/large pairs (anchored on the known-attackable llama3.1 relabel) survive.
    la_panel = grid.resolve_panel(grid.tier_names("lineage_arm"), backend="mock",
                                  overrides=grid.tier_overrides("lineage_arm"))
    la_k = 12
    la_matrix = [[1 if ((i + 1) * (j + 2)) % 3 else 0 for j in range(la_k)]
                for i in range(len(la_panel))]
    la_recs_off = [{"model": m["name"],
                    "asr": (None if m["lineage"] == "openthinker" else 0.5),
                    "error_rate": (1.0 if m["lineage"] == "openthinker" else 0.0)}
                   for m in la_panel]
    la_stats_off, la_excl_off, la_note_off = _gate_contrast(
        la_matrix, la_panel, la_recs_off, n_perm=100)
    assert la_stats_off is None, (
        "flag OFF: nulling the singleton 'openthinker' lineage must still suppress the "
        f"WHOLE lineage_arm contrast (unchanged pre-fix behavior): {la_stats_off}")
    assert "openthinker" in (la_note_off or ""), la_note_off

    la_stats_on, la_excl_on, la_note_on = _gate_contrast(
        la_matrix, la_panel, la_recs_off, n_perm=100, allow_singleton_lineage_drop=True)
    assert la_stats_on is not None, (
        "flag ON: nulling only the singleton 'openthinker' lineage must NOT suppress "
        f"the contrast (6 healthy Llama3.1/large pairs must survive): {la_note_on}")
    assert la_stats_on.get("dropped_singleton_lineages") == ["openthinker"], (
        f"dropped_singleton_lineages must record the drop: {la_stats_on.get('dropped_singleton_lineages')}")
    llama_arch_pairs = [p for p in la_stats_on["architecture_pairs"]
                        if "llama3.1:8b-instruct-q8_0" in p or "hermes3:8b" in p
                        or "dolphin3:8b" in p or "tulu3:8b" in p]
    assert len(llama_arch_pairs) >= 1, (
        f"at least one Llama3.1/large architecture pair must survive: {la_stats_on['architecture_pairs']}")

    print(json.dumps({"selftest": "OK",
                      "healthy_excluded": excl,
                      "dead_group_contrast": stats2,
                      "dead_group_excluded": excl2,
                      "dead_group_reason": note2,
                      "dead_group_flag_on_still_suppressed": stats2b is None,
                      "lineage_arm_openthinker_flag_off_suppressed": la_stats_off is None,
                      "lineage_arm_openthinker_flag_off_note": la_note_off,
                      "lineage_arm_openthinker_flag_on_dropped": la_stats_on.get("dropped_singleton_lineages"),
                      "lineage_arm_openthinker_flag_on_note": la_note_on,
                      "lineage_arm_openthinker_flag_on_arch_pairs": la_stats_on["architecture_pairs"],
                      }, indent=2, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the P3 IPI lineage/arch experiment.")
    ap.add_argument("--backend", choices=["mock", "ollama"], default="mock")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--n_perm", type=int, default=1000)
    ap.add_argument("--metric", choices=["pearson", "jaccard"], default="pearson")
    ap.add_argument("--match_mode", choices=["name_and_sentinel", "name_only"],
                    default="name_and_sentinel")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--allow_gpu", action="store_true",
                    help="skip the legacy CPU-pin guard: Ollama serves from the GPU "
                         "(2026-07-10 policy); GPU residency is verified by run_p3_gpu.sh")
    ap.add_argument("--allow_singleton_lineage_drop", action="store_true",
                    help="opt-in (default off): let a dead SINGLETON in-group lineage "
                         "(e.g. wave-3 openthinker) be dropped instead of suppressing the "
                         "whole contrast, when a multi-member anchor lineage is untouched "
                         "and an attackable architecture pair still survives -- see "
                         "run_ipi._gate_contrast docstring and PREREG-WAVE3-LINEAGE-"
                         "DRAFT-20260711.md sec 3a")
    ap.add_argument("--selftest", action="store_true",
                    help="run the offline finding-4 contrast-gate regression and exit")
    args = ap.parse_args(argv)
    if args.selftest:
        return _selftest()
    res = run(backend=args.backend, n=args.n, model_names=args.models,
              n_perm=args.n_perm, metric=args.metric, match_mode=args.match_mode,
              run_id=args.run_id, allow_gpu=args.allow_gpu,
              allow_singleton_lineage_drop=args.allow_singleton_lineage_drop)
    c = res["contrast"]
    summary = {
        "run_id": res["run_id"],
        "out": res["_out_path"],
        "backend": res["backend"],
        "n_scenarios": res["n_scenarios"],
        "source_fallbacks": {s["source_key"]: s["used_fallback"] for s in res["source_statuses"]},
        "per_model_asr": res["per_model_asr"],
        "per_model_error_rate": res["per_model_error_rate"],
        "contrast_excluded_models": res["contrast_excluded_models"],
        "contrast_note": res["contrast_note"],
    }
    if c is None:
        summary["contrast"] = None
        summary["contrast_suppressed"] = True
    else:
        summary.update({
            "mean_lineage_corr": c["mean_lineage_corr"],
            "mean_architecture_corr": c["mean_architecture_corr"],
            "mean_outgroup_corr": c["mean_outgroup_corr"],
            "observed_diff(lineage-arch)": c["observed_diff"],
            "p_value": c["p_value"],
            "lineage_gt_architecture": c["lineage_gt_architecture"],
        })
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
