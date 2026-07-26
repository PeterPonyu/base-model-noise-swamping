"""run_stream.py — replay each {mix}×{seed}×{policy} stream through router+arms, score, emit cells.

For every policy (the `both` router + `cost_only`/`damage_only` ablations + `oracle` +
fixed-strategy + `random` + the FT-merge baseline) this replays the ordered update stream,
absorbs each update with the chosen arm (dryrun-synthetic knowledge model), serves the
downstream queries, and writes one `cell_{model}_{provenance}_{MIX}_{policy}_s{seed}.json`
(quality + cost + discovery; body carries `model`+`provenance` — MAJOR-2 namespacing) that
`scorer/analyze_frame_a.py` aggregates into the frozen P1–P4 verdict. It also emits the MIX-C
structural P2 quantities (`p2_{model}_{provenance}_MIX_C.json`) — from routing, NEVER ErrorCost.

Bindings honored: λ_cost is calibrated on the DISJOINT dev/calibration slice only (grid-min
dev-slice ErrorCost_eval); the router reads `router_view(u)` only; `gt_damage` reaches the scorer
and the oracle but never the router; discovery is scoped to `damaging_gt`.

Idempotent + resumable: an existing cell file is skipped unless `--force`. Real-mode missing
assets HARD-FAIL (no silent short gate). DRYRUN/synthetic needs no torch/GPU/network.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

from . import config as C
from .arms.base import ModelState, make_arm, FtArm
from .cost_harness import SyntheticClock
from .damage_predictor import DamagePredictor
from .router import Router, FixedRouter, RandomRouter, OracleRouter, RouterState
from .stream_builder import StreamBuilder, router_view
from .scorer import scoring
from .scorer.scoring import OutcomeRow, quality, error_cost_eval, discovery, cost_vector, p2_structural

PRIVACY_HINTS = ("privacy_sensitive", "footprint", "offline")
POLICIES = ("both", "cost_only", "damage_only", "oracle",
            "always_edit", "always_grace", "always_rag", "always_ft", "always_reject",
            "random", "ft_merge")


def _make_router(policy: str, predictor: DamagePredictor, seed: int):
    if policy in ("both", "cost_only", "damage_only"):
        return Router(predictor=predictor, mode=policy)
    if policy == "oracle":
        return OracleRouter()
    if policy == "random":
        return RandomRouter(seed=seed)
    if policy == "ft_merge":
        return FixedRouter(arm="ft")            # strongest baseline = continual FT + task-merge.
    if policy.startswith("always_"):
        return FixedRouter(arm=policy.split("_", 1)[1])
    raise ValueError(policy)


def _qvol(update: Dict) -> int:
    return max(1, len(update.get("downstream_query_set", {}).get("efficacy", [1])))


def _replay(updates: List[Dict], router, editor: str = C.DEFAULT_EDITOR) -> Tuple[List[OutcomeRow], Dict]:
    """Replay a stream through one policy; return per-update OutcomeRows + a routing summary."""
    clk = SyntheticClock()
    state = RouterState()
    mstate = ModelState()
    arms = {a: make_arm(a, dryrun=True, editor=editor) for a in C.ARMS}
    ftarm: FtArm = arms["ft"]  # type: ignore
    rows: List[OutcomeRow] = []
    routing = {"edit_on_privacy": 0, "privacy_total": 0, "arm_counts": {}}
    for u in updates:
        dec = router.route(u, state)
        arm = dec["arm"]
        routing["arm_counts"][arm] = routing["arm_counts"].get(arm, 0) + 1
        hint = u.get("serving_hint", "none")
        if hint in PRIVACY_HINTS:
            routing["privacy_total"] += 1
            if arm == "edit":
                routing["edit_on_privacy"] += 1
        outcome = arms[arm].install(u, mstate)
        # periodic FT merge flush.
        if arm == "ft" and len(mstate.ft_pending) >= C.FT_MERGE_INTERVAL_K:
            flush_cost = ftarm.flush(mstate)
        else:
            flush_cost = None
        qvol = _qvol(u)
        serve_over = arms[arm].serve_overhead(store_n=mstate.store_n, k=C.RAG_TOP_K)
        install_gpu = outcome.cost.gpu_s + (flush_cost.gpu_s if flush_cost else 0.0)
        serve_gpu = qvol * (1.0 + serve_over)
        applied = outcome.applied_fact
        rows.append(OutcomeRow(
            t=u["t"], arm=arm, fact_type=u["fact_type"],
            conflict_flag=u["conflict_flag"], damaging_kind=u.get("damaging_kind"),
            applied=applied, stale=(arm == "reject" or outcome.deferred),
            collateral=outcome.collateral, gt_damage=float(u["gt_damage"]),
            efficacy_correct=applied,
            ripple_correct=(applied if u["fact_type"] in ("mquake_mh", "ripple") else None),
            install_gpu_s=install_gpu, serve_gpu_s=serve_gpu, serve_overhead=qvol * serve_over,
            exposure_surface=C.EXPOSURE_SURFACE.get(arm, 0.0),
            store_bytes=clk._store_bytes(arm, mstate.store_n),
            routed_away_from_edit=(arm != "edit")))
    # end-of-stream: flush any queued FT facts (so their efficacy counts). CHARGE the final flush
    # (MINOR-3): its GPU-seconds are attributed to the last FT row so FT cost-parity (P3) is honest.
    ft_rows = [r for r in rows if r.arm == "ft"]
    if mstate.ft_pending:
        final_flush = ftarm.flush(mstate)
        if ft_rows:
            ft_rows[-1].install_gpu_s += final_flush.gpu_s
    for r in ft_rows:
        r.applied = True            # flushed by end-of-stream above.
        r.stale = False
        r.efficacy_correct = True
    # forgetting: FtArm.flush marks a deterministic 1/7 of prior FT facts on ModelState.forgotten;
    # attribute that to the same 1/7 of FT rows so A_cum reflects the merge overwrite.
    for i, r in enumerate(ft_rows):
        if i % 7 == 6:
            r.forgotten_at_end = True
    return rows, routing


# The real-GPU replay now lives in experiments/frame_a/real_replay.py (replay_real) — CPU-mockable,
# reusing killgate _capture_key / metrics primitives, carrying asserts (c)+(d). run_smoke and
# run_real_wave below drive it. The earlier fenced stub here has been removed.


def _replay_cost(router, dev_updates: List[Dict]) -> float:
    """Dev-slice ErrorCost_eval for λ calibration (attaches minimal query sets to raw dev recs)."""
    ups = []
    for i, d in enumerate(dev_updates):
        u = dict(d)
        u.setdefault("t", i)
        u.setdefault("conflict_flag", "none")
        u.setdefault("serving_hint", "none")
        u.setdefault("fact_type", d.get("fact_type", "cf"))
        u.setdefault("subject_key", d.get("edit", {}).get("subject", f"dev{i}"))
        u.setdefault("gt_damage", d.get("gt_damage", 0.0))
        u.setdefault("downstream_query_set", {"efficacy": [d.get("edit", {}).get("prompt", "q")]})
        ups.append(u)
    rows, _ = _replay(ups, router)
    return error_cost_eval(rows)


def run_cell(builder: StreamBuilder, mix: str, seed: int, policy: str,
             predictor: DamagePredictor, calibrated_lambda: Optional[float]) -> Dict:
    updates, manifest = builder.build_stream(mix, seed)
    router = _make_router(policy, predictor, seed)
    if policy in ("both", "cost_only", "damage_only") and calibrated_lambda is not None:
        router.lambda_cost = calibrated_lambda
    rows, routing = _replay(updates, router)
    cell = {
        "mix": mix, "policy": policy, "seed": seed,
        "quality": quality(rows), "cost": cost_vector(rows),
        "error_cost_eval": error_cost_eval(rows), "discovery": discovery(rows),
        "routing": routing, "lambda_cost": getattr(router, "lambda_cost", None),
        "stream_hash": manifest["stream_hash"],
    }
    return cell


def _p2_for_mixC(builder: StreamBuilder, predictor: DamagePredictor, lam: Optional[float]) -> Dict:
    """Structural P2 quantities from the `both` router on MIX-C (never from ErrorCost)."""
    clk = SyntheticClock()
    arm_exposure = {"edit": C.EXPOSURE_SURFACE["edit"], "rag": C.EXPOSURE_SURFACE["rag"]}
    # footprint at a representative store size; overhead is per-query constant-in-k.
    arm_footprint = {"edit": clk._store_bytes("edit", 500), "rag": clk._store_bytes("rag", 500)}
    arm_overhead = {"edit": clk.serve("edit", 1, 500, C.RAG_TOP_K).gpu_s - 1.0,
                    "rag": clk.serve("rag", 1, 500, C.RAG_TOP_K).gpu_s - 1.0}
    p2 = p2_structural(arm_exposure, arm_footprint, arm_overhead)
    # 4th term: router-selects-edit majority on privacy/footprint MIX-C updates (mean over seeds).
    fracs = []
    for s in C.SEEDS:
        r = Router(predictor=predictor, mode="both", lambda_cost=(lam or 1e-2))
        _, routing = _replay(builder.build_stream("MIX_C", s)[0], r)
        tot = routing["privacy_total"]
        fracs.append((routing["edit_on_privacy"] / tot) if tot else 0.0)
    import numpy as np
    p2["router_edit_majority_on_privacy"] = float(np.mean(fracs)) if fracs else 0.0
    return p2


def run_wave(out_cells: str, synthetic: bool = True, cf_cell_seed: int = 0,
             force: bool = False, mixes: Optional[List[str]] = None,
             model_tag: str = "llama-3.2-1b") -> Dict:
    os.makedirs(out_cells, exist_ok=True)
    builder = StreamBuilder(synthetic=synthetic, cf_cell_seed=cf_cell_seed)
    predictor = DamagePredictor()
    mixes = mixes or list(C.MIXES.keys())
    # MAJOR-2 namespacing: every cell file + p2 file is tagged by model AND provenance so a 3B run
    # cannot skip-or-clobber 1B cells and a real analyze can never score stale synthetic cells.
    provenance = "synth" if synthetic else "real"
    # λ calibration on the DISJOINT dev/calibration slice (grid-min dev ErrorCost_eval).
    dev = builder.calibration_slice(sorted({ft for m in mixes for ft in C.MIXES[m]["fact_type_weights"]}))
    lam = Router(predictor=predictor, mode="both").calibrate_lambda(dev, _replay_cost)
    written = []
    for mix in mixes:
        for seed in C.SEEDS:
            for policy in POLICIES:
                path = os.path.join(out_cells, f"cell_{model_tag}_{provenance}_{mix}_{policy}_s{seed}.json")
                if os.path.exists(path) and not force:
                    written.append(path); continue
                cell = run_cell(builder, mix, seed, policy, predictor, lam)
                cell["calibrated_lambda"] = lam
                cell["model"] = model_tag
                cell["provenance"] = provenance
                json.dump(cell, open(path, "w"))
                written.append(path)
    # MIX-C structural P2 (namespaced to match its cells).
    if "MIX_C" in mixes:
        json.dump(_p2_for_mixC(builder, predictor, lam),
                  open(os.path.join(out_cells, f"p2_{model_tag}_{provenance}_MIX_C.json"), "w"))
    return {"n_cells": len(written), "calibrated_lambda": lam, "out_cells": out_cells,
            "model": model_tag, "provenance": provenance}


# ---------------------------------------------------------------- REAL-GPU wave (behind --real)
def _load_model(model_dir: str, dtype: str = "float32"):
    """Load a real HF model + tokenizer (fp32 for ROME). Lazy; only the real wave calls it."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    td = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=td)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev).eval()
    return model, tok, dev


def _score_real_rows(mix: str, policy: str, seed: int, rows, routing, lam, model_tag: str) -> Dict:
    """Score REAL replay rows into a namespaced cell dict (mirrors run_cell for the synthetic path)."""
    return {
        "mix": mix, "policy": policy, "seed": seed,
        "model": model_tag, "provenance": "real",
        "quality": quality(rows), "cost": cost_vector(rows),
        "error_cost_eval": error_cost_eval(rows), "discovery": discovery(rows),
        "routing": routing, "lambda_cost": lam, "n_d_assert": routing.get("n_d_assert", 0),
    }


# ---------------------------------------------------------------- M4: SMOKE launch-gate marker
SMOKE_MARKER = os.path.join(C.HARNESS_ROOT, "engine", "SMOKE_PASS.ok")


def _frame_a_code_checksum() -> str:
    """Content hash over the frame_a package .py files. Ties a SMOKE_PASS marker to the EXACT code
    that produced it — any edit to the package after a passing smoke invalidates the launch gate."""
    import hashlib, glob
    pkg = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha1()
    for f in sorted(glob.glob(os.path.join(pkg, "**", "*.py"), recursive=True)):
        h.update(os.path.relpath(f, pkg).encode())
        with open(f, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()[:16]


def write_smoke_marker(model_dir: str) -> str:
    """Write engine/SMOKE_PASS.ok after a PASSING smoke (model_dir + frame_a code checksum + ts)."""
    import time
    os.makedirs(os.path.dirname(SMOKE_MARKER), exist_ok=True)
    json.dump({"model_dir": os.path.abspath(model_dir), "code_checksum": _frame_a_code_checksum(),
               "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}, open(SMOKE_MARKER, "w"))
    return SMOKE_MARKER


def check_smoke_marker(model_dir: str) -> Tuple[bool, str]:
    """M4 launch gate: the real wave may run ONLY if a FRESH SMOKE_PASS.ok exists — same model_dir
    AND same frame_a code checksum. Missing / stale / code-changed → (False, reason: run SMOKE=1)."""
    if not os.path.exists(SMOKE_MARKER):
        return False, "no SMOKE_PASS.ok marker present — run SMOKE=1 first"
    try:
        m = json.load(open(SMOKE_MARKER))
    except Exception as e:  # noqa: BLE001
        return False, f"unreadable SMOKE_PASS.ok ({e}) — re-run SMOKE=1"
    if m.get("model_dir") != os.path.abspath(model_dir):
        return False, f"marker model_dir {m.get('model_dir')} != {os.path.abspath(model_dir)} — re-run SMOKE=1"
    cur = _frame_a_code_checksum()
    if m.get("code_checksum") != cur:
        return False, f"frame_a code changed since smoke ({m.get('code_checksum')} != {cur}) — re-run SMOKE=1"
    return True, f"SMOKE_PASS.ok fresh (model_dir + code match; ts={m.get('ts')})"


def run_smoke(model_dir: str, cf_cell_seed: int = 0, n: int = 5) -> Dict:
    """LAUNCH GATE: a REAL 5-update micro-stream on the loaded model. All four wired asserts must
    fire+pass — (a),(b) at build; (c) inside the edit arm; (d) live key_cos == stored. FixedRouter
    (edit) forces every update through the edit arm so (c)+(d) fire on all 5. On PASS it writes the
    SMOKE_PASS.ok marker (M4) that gates the full real wave."""
    import torch
    from .real_replay import RealHarness, replay_real
    from .arms.real_backends import RealGraceArm, RealFtLoraMergeArm
    from .arms.base import ModelState
    from .router import FixedRouter
    b = StreamBuilder(synthetic=False, cf_cell_seed=cf_cell_seed)      # asserts (a),(b) at build.
    updates, _man = b.build_stream("MIX_B", 0)
    micro = [u for u in updates if u["fact_type"] == "cf"
             and u.get("gt_damage_provenance", {}).get("cell_seed") is not None][:n]
    if len(micro) < n:
        raise RuntimeError(f"micro-stream needs {n} measured-CF updates, got {len(micro)}")
    for i, u in enumerate(micro):
        u["t"] = i; u["fact_id"] = f"smoke_cf_{i}"
    probe_bank = b.probe_bank(["cf"])[:16]
    model, tok, dev = _load_model(model_dir)
    harness = RealHarness(model=model, tok=tok, device=dev)
    # M1: snapshot base weights to CPU (a GPU-resident deepcopy would double weight VRAM; the
    # per-cell restore load_state_dict copies them back onto the model's device).
    base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    base_checksum = harness.touched_checksum()
    model_dtype = str(next(model.parameters()).dtype)

    # (1) edit-only micro — fires (c) each edit + (d) live==stored each edit.
    rows, routing = replay_real(micro, FixedRouter(arm="edit"), harness, probe_bank)
    assert routing["n_d_assert"] == n, f"(d) must fire {n}×, got {routing['n_d_assert']}"
    assert routing["n_c_assert"] == n, f"(c) must fire {n}×, got {routing['n_c_assert']}"

    # (2) RESTORE PROBE (Item E): edit → restore ; GRACE cell → restore ; FT-merge cell → restore.
    ev_edit = harness.restore_and_verify(base_state, base_checksum, model)
    mstate = ModelState()
    gr = RealGraceArm(model, tok, dev)
    for u in micro[:2]:
        gr.install(u, mstate)                                 # attaches grace hooks/codebooks.
    ev_grace = harness.restore_and_verify(base_state, base_checksum, model)   # A1 clear_grace fires.
    r_after_grace, _ = replay_real(micro[:1], FixedRouter(arm="edit"), harness, probe_bank)
    mstate2 = ModelState()
    ft = RealFtLoraMergeArm(model, tok, dev)
    for u in micro[:3]:
        ft.install(u, mstate2)
    ft.flush(mstate2)                                         # LoRA train + task-merge (VRAM-guarded).
    harness.model = getattr(ft, "model", model)
    ev_ft = harness.restore_and_verify(base_state, base_checksum, model)
    r_after_ft, _ = replay_real(micro[:1], FixedRouter(arm="edit"), harness, probe_bank)
    assert ev_edit["harness_is_base"] and ev_grace["harness_is_base"] and ev_ft["harness_is_base"]

    # (3) NaN tripwire + serving axis.
    any_nan = any(c.get("nan") for c in routing["c_records"])
    assert not any_nan, "(c) NaN tripwire fired — edits produced NaN ΔW (fp16 hazard)"
    assert all(r.serve_gpu_s > 0 for r in rows), "serving cost axis must be measured (non-zero)"

    # Item F — the five reviewer evidence lines.
    c0 = routing["c_records"][0]; d0 = routing["d_records"][0]
    lines = [
        f"[E1 (c) rank-one/norm] s2/s1={c0['s2_s1']:.2e}  ΔW-norm rel-err={c0['norm_relerr']:.2e} "
        f"(all {routing['n_c_assert']} edits rank-one)",
        f"[E2 (d) key_cos]       live={d0['live']:.5f} stored={d0['stored']:.5f} |Δ|={d0['abs_delta']:.2e} "
        f"n_c={routing['n_c_assert']} n_d={routing['n_d_assert']}",
        f"[E3 dtype/NaN]         model_dtype={model_dtype}  NaN-tripwire={'CLEAN' if not any_nan else 'FIRED'}",
        f"[E4 restore probe]     edit:chk_rel={ev_edit['checksum_rel_err']:.1e} "
        f"grace:clear={ev_grace['clear_grace_layers']} ft:chk_rel={ev_ft['checksum_rel_err']:.1e} "
        f"harness_is_base={ev_edit['harness_is_base'] and ev_grace['harness_is_base'] and ev_ft['harness_is_base']}",
        f"[E5 install/serve GPU-s] install={routing['install_gpu_s_by_arm']}  serve={routing['serve_gpu_s_by_arm']}",
    ]
    for ln in lines:
        print(ln)
    del base_state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    marker = write_smoke_marker(model_dir)                    # M4: gate the full real wave on this.
    return {"smoke": "PASS", "n_updates": n, "n_d_assert": routing["n_d_assert"],
            "n_c_assert": routing["n_c_assert"], "evidence": lines, "smoke_marker": marker,
            "asserts": "(a)build (b)build (c)ΔW-parity (d)live-key_cos + restore-probe A1/A2/A3 ALL FIRED+PASSED"}


def run_real_wave(out_cells: str, model_dir: str, cf_cell_seed: int = 0, force: bool = False,
                  mixes: Optional[List[str]] = None, model_tag: str = "llama-3.2-1b",
                  policies_filter: Optional[List[str]] = None) -> Dict:
    """The full real wave. Loads the model ONCE and RESTORES it to base between cells (real edits
    are destructive — every {mix,seed,policy} cell must start from the unedited weights)."""
    import torch
    os.makedirs(out_cells, exist_ok=True)
    from .real_replay import RealHarness, replay_real
    builder = StreamBuilder(synthetic=False, cf_cell_seed=cf_cell_seed)
    predictor = DamagePredictor()
    mixes = mixes or list(C.MIXES.keys())
    dev = builder.calibration_slice(sorted({ft for m in mixes for ft in C.MIXES[m]["fact_type_weights"]}))
    lam = Router(predictor=predictor, mode="both").calibrate_lambda(dev, _replay_cost)
    model, tok, device = _load_model(model_dir)
    # M1: CPU snapshot for per-cell restore (a GPU-resident deepcopy would double weight VRAM;
    # load_state_dict copies back onto the model's device on restore).
    base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    harness = RealHarness(model=model, tok=tok, device=device)
    harness._bind_real()
    base_checksum = harness.touched_checksum()          # A2 reference over the touched weight set.
    written = []
    # M2: accumulate MEASURED per-arm serving on MIX-C so the real P2 overhead_delta is measured,
    # not synthetic. [sum_serve_gpu_s, n_updates] per arm across all MIX-C cells (the always_edit /
    # always_rag policies guarantee both arms are exercised even if the `both` router avoids one).
    real_serve_mixC = {"edit": [0.0, 0], "rag": [0.0, 0]}
    # Policy filter (added 2026-07-20): when the FT-fix rerun driver passes --policies, only the
    # matching policies run; everything else is skipped per-cell (resume-safe). The skip-on-
    # exists check at the cell path (line 386) means a re-invocation with the same filter
    # is idempotent — partial runs are resumed automatically.
    policies = POLICIES
    if policies_filter is not None:
        unknown = [p for p in policies_filter if p not in POLICIES]
        if unknown:
            raise SystemExit(f"[run_real_wave] unknown policies in filter: {unknown}; allowed: {POLICIES}")
        policies = tuple(policies_filter)
        print(f"[run_real_wave] policy filter active: {policies} (skipping {set(POLICIES) - set(policies)})")
    for mix in mixes:
        for seed in C.SEEDS:
            probe_bank = builder.probe_bank(sorted(C.MIXES[mix]["fact_type_weights"]))
            for policy in policies:
                path = os.path.join(out_cells, f"cell_{model_tag}_real_{mix}_{policy}_s{seed}.json")
                if os.path.exists(path) and not force:
                    written.append(path); continue
                # A1+A2+A3: restore base weights + clear_grace + checksum-verify + re-point harness.
                harness.restore_and_verify(base_state, base_checksum, model)
                updates, manifest = builder.build_stream(mix, seed)
                router = _make_router(policy, predictor, seed)
                if policy in ("both", "cost_only", "damage_only"):
                    router.lambda_cost = lam
                rows, routing = replay_real(updates, router, harness, probe_bank)
                if mix == "MIX_C":
                    for a in ("edit", "rag"):
                        real_serve_mixC[a][0] += routing["serve_gpu_s_by_arm"].get(a, 0.0)
                        real_serve_mixC[a][1] += routing["arm_counts"].get(a, 0)
                cell = _score_real_rows(mix, policy, seed, rows, routing, lam, model_tag)
                cell["stream_hash"] = manifest["stream_hash"]
                json.dump(cell, open(path, "w"))
                written.append(path)
    # MIX-C structural P2, NAMESPACED to the real provenance (contract condition 3 — without this
    # a real analyze reports "MIX_C _p2 missing" and P2 cannot be adjudicated). The P2 structural
    # deltas (exposure/footprint/overhead) are model-independent config/cost-model constants and the
    # router-selects-edit-majority term is a routing decision (identical in real and dryrun), so the
    # synthetic-path helper is the correct source here too.
    if "MIX_C" in mixes:
        # M2: start from the synthetic structural P2 (exposure / footprint / router-majority) and
        # OVERRIDE overhead_delta with the MEASURED per-update serving delta (rag − edit) when both
        # arms were exercised on MIX-C; fall back to synthetic ONLY with an explicit provenance tag.
        p2 = _p2_for_mixC(builder, predictor, lam)
        cost_prov = {"overhead_delta": "synthetic", "footprint_delta": "synthetic"}
        if real_serve_mixC["edit"][1] > 0 and real_serve_mixC["rag"][1] > 0:
            mean_edit = real_serve_mixC["edit"][0] / real_serve_mixC["edit"][1]
            mean_rag = real_serve_mixC["rag"][0] / real_serve_mixC["rag"][1]
            p2["overhead_delta"] = mean_rag - mean_edit
            p2["overhead_delta_measured"] = {"mean_serve_gpu_s_edit": mean_edit,
                                             "mean_serve_gpu_s_rag": mean_rag,
                                             "n_edit": real_serve_mixC["edit"][1],
                                             "n_rag": real_serve_mixC["rag"][1]}
            cost_prov["overhead_delta"] = "measured"
        p2["p2_cost_provenance"] = cost_prov
        json.dump(p2, open(os.path.join(out_cells, f"p2_{model_tag}_real_MIX_C.json"), "w"))
    del base_state; torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return {"n_cells": len(written), "calibrated_lambda": lam, "out_cells": out_cells,
            "model": model_tag, "provenance": "real"}


# ---------------------------------------------------------------- selftest
def _selftest() -> None:
    b = StreamBuilder(synthetic=True)
    pred = DamagePredictor()
    # one cell each policy on a small mix runs and scores without error.
    for policy in ("both", "oracle", "always_edit", "random", "ft_merge"):
        cell = run_cell(b, "MIX_B", 0, policy, pred, calibrated_lambda=1e-2)
        assert 0.0 <= cell["quality"]["Q"] <= 1.0, cell["quality"]
        assert cell["cost"]["total_gpu_s"] > 0.0
    # oracle should not be worse than always_edit on locality (it avoids damage).
    q_oracle = run_cell(b, "MIX_B", 0, "oracle", pred, 1e-2)["quality"]["A_loc"]
    q_edit = run_cell(b, "MIX_B", 0, "always_edit", pred, 1e-2)["quality"]["A_loc"]
    assert q_oracle >= q_edit, "oracle must retain locality at least as well as always-edit"
    # MIX-C structural P2: exposure/footprint/overhead deltas positive.
    p2 = _p2_for_mixC(b, pred, 1e-2)
    assert p2["exposure_edit"] == 0.0 and p2["exposure_rag"] > 0.5
    assert p2["footprint_delta"] > 0 and p2["overhead_delta"] > 0
    assert 0.0 <= p2["router_edit_majority_on_privacy"] <= 1.0
    # λ calibration returns a grid value.
    dev = b.calibration_slice(["cf", "zsre"])
    lam = Router(predictor=pred, mode="both").calibrate_lambda(dev, _replay_cost)
    assert lam in C.LAMBDA_COST_GRID
    print("run_stream selftest: PASS")


def main() -> None:
    ap = argparse.ArgumentParser(description="Frame-A stream replay driver")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--synthetic", action="store_true", help="fixture pool (no torch/loaders/GPU)")
    ap.add_argument("--real", action="store_true",
                    help="real-GPU wave: load model + real arms + MEASURED collateral")
    ap.add_argument("--smoke", action="store_true",
                    help="LAUNCH GATE: a real 5-update micro-stream firing all four asserts (a)-(d)")
    ap.add_argument("--check_smoke_marker", action="store_true",
                    help="M4: exit 0 iff a FRESH SMOKE_PASS.ok exists for --model_dir (else exit 1)")
    ap.add_argument("--model", default="llama-3.2-1b", help="model tag for the cell filenames")
    ap.add_argument("--model_dir", default=os.path.join(C.DATA_DIR, "models", "Llama-3.2-1B"),
                    help="on-disk model directory for the real wave / smoke")
    ap.add_argument("--cf_cell_seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out_cells", default=os.path.join(C.RESULTS_DIR, "cells"))
    ap.add_argument("--mixes", default=None,
                    help="comma-separated subset of mixes to run (e.g. MIX_B,MIX_C) — "
                         "for partitioning one wave across machines; default = all")
    ap.add_argument("--policies", default=None,
                    help="comma-separated subset of policies to run (e.g. always_ft,ft_merge) — "
                         "added 2026-07-20 so the FT-fix rerun can touch only the contaminated "
                         "cells without overwriting the 21 un-contaminated ones. Default = all 11.")
    args = ap.parse_args()
    if args.selftest:
        _selftest(); return
    if args.check_smoke_marker:
        ok, why = check_smoke_marker(args.model_dir)
        print(f"SMOKE_MARKER {'FRESH' if ok else 'STALE/MISSING'}: {why}")
        raise SystemExit(0 if ok else 1)
    if args.run:
        if args.real:
            if args.smoke:
                info = run_smoke(args.model_dir, cf_cell_seed=args.cf_cell_seed)
                print(f"SMOKE {info['smoke']}: {info['asserts']} (n_d_assert={info['n_d_assert']})")
                return
            mixes = [m.strip() for m in args.mixes.split(",")] if args.mixes else None
            if mixes:
                unknown = [m for m in mixes if m not in C.MIXES]
                if unknown:
                    raise SystemExit(f"--mixes: unknown mix(es) {unknown}; valid: {list(C.MIXES)}")
            info = run_real_wave(args.out_cells, args.model_dir, cf_cell_seed=args.cf_cell_seed,
                                 force=args.force, model_tag=args.model, mixes=mixes,
                                 policies_filter=(args.policies.split(",") if args.policies else None))
            print(f"REAL wave: {info['n_cells']} cells (λ={info['calibrated_lambda']}) "
                  f"model={info['model']} -> {info['out_cells']}")
            return
        info = run_wave(args.out_cells, synthetic=args.synthetic, cf_cell_seed=args.cf_cell_seed,
                        force=args.force, model_tag=args.model)
        print(f"ran {info['n_cells']} cells (λ={info['calibrated_lambda']}) "
              f"model={info['model']} provenance={info['provenance']} -> {info['out_cells']}")
        return
    ap.print_help()


if __name__ == "__main__":
    main()
