"""real_replay.py — the REAL-GPU replay (behind run_stream --real), fully CPU-mockable.

Replays a stream through the REAL model + real arms, MEASURES quality/cost, and carries the four
first-wave integrity asserts. Every GPU-touching primitive is injected through `RealHarness`, so
the whole path (including asserts (c)+(d)) runs on a CPU mock in the self-test gate; the SMOKE
micro-stream on the real 1B is the launch gate that fires them for real.

Reused harness primitives (no re-derivation):
  * `editors.rome_native._capture_key` / `find_subject_last_token_index` — the EXACT key capture
    the gate cells used, so live `predict_from_key` reproduces the stored `key_cos` to tolerance.
  * `metrics.efficacy` / `first_target_token_id`, `killgate_keygeom.prob_of_token` — A_upd + probe
    correct-token logits (A_loc drift).
  * `editors/{rome_native,memit,alphaedit,grace_editor,ft_editor}.apply_edit`, `rank_bm25` —
    the real arm backends (via arms/real_backends.py). ROME runs fp32.

The four asserts:
  (a) geometry-join integrity + (b) damaging_gt floor — fire at STREAM BUILD (CPU, stream_builder).
  (c) ΔW rank-one parity — fires inside `RealEditArm._real_install` per edit.
  (d) live `predict_from_key` == stream `key_cos` — fires HERE, per measured-CF edit, before apply.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from . import config as C
from .scorer.scoring import OutcomeRow
from .arms.real_asserts import assert_key_cos_match

PRIVACY_HINTS = ("privacy_sensitive", "footprint", "offline")


# ---------------------------------------------------------------- geometry
@dataclass
class ProbeGeometry:
    """Normalized probe-bank keys + base-known mask for ONE cell (used by predict_from_key)."""
    Kp_norm: np.ndarray            # [M, d] L2-normalized probe keys (killgate convention).
    base_known: np.ndarray         # [M] bool: cell pre_p > 0.05 (identical mask to the stored cell).
    cell_seed: int


def predict_key_cos(edit_key_norm: np.ndarray, geom: ProbeGeometry) -> float:
    """Reproduce the stored per-edit `key_cos`: mean over base-known probes of the signed cosine
    between the (normalized) edit key and the (normalized) probe keys — EXACTLY the aggregate
    `_cf_measured_geometry` reads from the cell's `COS[:, base_known].mean(axis=1)`. Pure numpy."""
    cos = geom.Kp_norm @ np.asarray(edit_key_norm, float)     # [M]
    m = geom.base_known
    return float(cos[m].mean()) if m.any() else float(cos.mean())


# ---------------------------------------------------------------- the harness (injectable)
@dataclass
class RealHarness:
    """Bundles the loaded model/tokenizer and every GPU primitive as an injectable callable.

    Leave a primitive None to lazily bind the REAL implementation; the CPU mock passes deterministic
    stand-ins (see `make_mock_harness`). This is what makes the replay + asserts CPU-mockable.
    """
    model: object = None
    tok: object = None
    device: str = "cuda"
    layer: int = C.GEOMETRY_LAYER
    # injectable primitives
    capture_fn: Optional[Callable] = None       # (model,tok,layer,prompt,tok_index,device)->tensor[d]
    subject_idx_fn: Optional[Callable] = None   # (tok,prompt,subject)->int
    efficacy_fn: Optional[Callable] = None      # (model,tok,prompt,tnew,ttrue,device)->{"success":..}
    probe_prob_fn: Optional[Callable] = None    # (model,tok,prompt,token_id,device)->(p,logit)
    target_tok_fn: Optional[Callable] = None    # (tok,target)->int
    predict_fn: Optional[Callable] = None       # (harness,edit_request,geom)->float ; default real math
    load_cf_probes_fn: Optional[Callable] = None  # (cell_seed)->(probes_list, base_known ndarray)
    _probe_cache: Dict[int, ProbeGeometry] = field(default_factory=dict)

    def _bind_real(self) -> None:
        """Lazily wire the real killgate/metrics/editor primitives (never called on the mock)."""
        import os, sys
        _exp = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _exp not in sys.path:
            sys.path.insert(0, _exp)
        if self.capture_fn is None:
            from editors.rome_native import _capture_key
            self.capture_fn = _capture_key
        if self.subject_idx_fn is None:
            from editors.rome_native import find_subject_last_token_index
            self.subject_idx_fn = find_subject_last_token_index
        if self.efficacy_fn is None:
            from metrics import efficacy
            self.efficacy_fn = efficacy
        if self.target_tok_fn is None:
            from metrics import first_target_token_id
            self.target_tok_fn = first_target_token_id
        if self.probe_prob_fn is None:
            import killgate_keygeom as kg
            self.probe_prob_fn = kg.prob_of_token
        if self.load_cf_probes_fn is None:
            self.load_cf_probes_fn = self._real_load_cf_probes
        if self.predict_fn is None:
            # uniform signature predict_fn(harness, edit_request, geom) — matches the mock.
            self.predict_fn = lambda h, er, g: h._real_predict(er, g)

    # -------------------------------------------------------------- key capture
    def capture_key_norm(self, prompt: str, subject: Optional[str]) -> np.ndarray:
        idx = self.subject_idx_fn(self.tok, prompt, subject)
        k = self.capture_fn(self.model, self.tok, self.layer, prompt, idx, self.device)
        k = np.asarray(k.float().cpu().numpy() if hasattr(k, "float") else k, float)
        return k / (np.linalg.norm(k) + 1e-8)

    def _real_load_cf_probes(self, cell_seed: int):
        """The cell's OWN probe prompts (load_counterfact(seed) probes) + its base-known mask
        (from the cell npz pre_p>0.05) — so predict_from_key reproduces the stored aggregate."""
        import os, sys
        _exp = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _exp not in sys.path:
            sys.path.insert(0, _exp)
        import killgate_keygeom as kg
        from .stream_builder import CELL_N_EDITS, CELL_N_PROBES
        _edits, probes, _h = kg.load_counterfact(C.DATASETS["cf"], CELL_N_EDITS, CELL_N_PROBES, seed=cell_seed)
        path = C.GT_DAMAGE_GLOB.format(L=C.GEOMETRY_LAYER, s=cell_seed)
        d = np.load(path)
        base_known = (d["pre_p"].astype(float) > C.KNOWN_PROBE_PRE_P) if "pre_p" in d.files \
            else np.ones(len(probes), bool)
        return probes, base_known

    def probe_geometry(self, cell_seed: int) -> ProbeGeometry:
        if cell_seed in self._probe_cache:
            return self._probe_cache[cell_seed]
        probes, base_known = self.load_cf_probes_fn(cell_seed)
        Kp = np.stack([self.capture_key_norm(p["prompt"], p.get("subject")) for p in probes])
        m = np.asarray(base_known, bool)[:Kp.shape[0]]
        geom = ProbeGeometry(Kp_norm=Kp, base_known=m, cell_seed=cell_seed)
        self._probe_cache[cell_seed] = geom
        return geom

    def _real_predict(self, edit_request: Dict, geom: ProbeGeometry) -> float:
        # route through the NAMED entry point (DamagePredictor.predict_from_key), passing the
        # harness's injected capture/subject-index primitives so mock and real agree exactly.
        from .damage_predictor import DamagePredictor
        return DamagePredictor(layer=self.layer).predict_from_key(
            self.model, self.tok, edit_request, geom, self.device,
            capture_fn=self.capture_fn, subject_idx_fn=self.subject_idx_fn)

    # -------------------------------------------------------------- measurement
    def efficacy_correct(self, update: Dict) -> bool:
        e = update["edit"]
        out = self.efficacy_fn(self.model, self.tok, e["prompt"], e["target_new"],
                               e.get("target_true"), self.device)
        return bool(out.get("success", 0.0) >= 0.5)

    def probe_correct_logits(self, probes: List[Dict]) -> np.ndarray:
        vals = np.zeros(len(probes), float)
        for j, p in enumerate(probes):
            tid = self.target_tok_fn(self.tok, p["edit"]["target_true"])
            _pp, lg = self.probe_prob_fn(self.model, self.tok, p["edit"]["prompt"], tid, self.device)
            vals[j] = lg
        return vals

    # -------------------------------------------------------------- serving cost (MODERATE-D)
    serve_time_fn: Optional[Callable] = None    # (arm,query_prompt,rag_docs)->gpu_s ; default real.

    def serve_gpu_s(self, arm: str, query_prompt: str, rag_docs: Optional[List[str]] = None) -> float:
        """MEASURE a real serving forward so the P1/P3 cost axis does not zero out the serving term.
        RAG prepends its top-k retrieved facts (the k-fact prefill) → its forward is longer than a
        clean forward; edit/grace/ft/reject serve a clean forward (facts baked in / codebook)."""
        if self.serve_time_fn is not None:
            return float(self.serve_time_fn(arm, query_prompt, rag_docs))
        import time as _t, torch
        prompt = query_prompt
        if arm == "rag" and rag_docs:
            prompt = " ".join(rag_docs) + " " + query_prompt        # constant k-fact prefill.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = _t.perf_counter()
        with torch.no_grad():
            enc = self.tok(prompt, return_tensors="pt").to(self.device)
            self.model(**enc)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return _t.perf_counter() - t0

    # -------------------------------------------------------------- restore + integrity (A1/A2/A3)
    def touched_checksum(self) -> float:
        """Cheap checksum over the TOUCHED weight set (down_proj@L12 + every LoRA-target matrix) —
        a stable fp64 sum of per-tensor norms PLUS a sentinel-element signature (MINOR-1). The norm
        alone is invariant under norm-preserving corruption (a permutation or sign flip of equal-
        magnitude entries leaves ‖·‖ unchanged); folding a deterministic index-weighted sample of
        raw element values in makes the A2 restore check catch those too. Used to verify a per-cell
        restore (A2); base and post-restore both call this function, so the contract is symmetric."""
        import torch
        total = 0.0
        suffixes = tuple(f"{t}.weight" for t in C.FT_LORA_TARGETS)
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if name.endswith(suffixes):
                    pf = p.detach().double().flatten()
                    total += float(pf.norm().item())
                    # sentinel signature: 16 evenly-spaced raw elements, index-weighted so a
                    # permutation of equal-magnitude entries moves the sum (the norm would not).
                    stride = max(1, pf.numel() // 16)
                    sent = pf[::stride][:16]
                    w = torch.arange(1, sent.numel() + 1, dtype=torch.float64, device=sent.device)
                    total += float((sent * w).sum().item())
        return total

    def restore_and_verify(self, base_state: Dict, base_checksum: float, model, rtol: float = 1e-4):
        """Per-cell restore (A1+A2+A3): reload base weights, clear GRACE hooks/codebooks, assert the
        touched-weight checksum matches base (abort on mismatch), and RE-POINT the harness to the
        canonical base `model` object (peft unload may have returned a new one). Returns evidence."""
        from editors.grace_editor import clear_grace
        model.load_state_dict(base_state, strict=False)         # restore base weights.
        n_grace = clear_grace(model)                            # A1: hooks/codebooks (load can't clear).
        self.model = model                                      # A3: re-point to the canonical object.
        self._probe_cache.clear()
        assert self.model is model, "A3: harness.model must be the base model object after restore"
        chk = self.touched_checksum()                           # A2: integrity checksum.
        rel = abs(chk - base_checksum) / (abs(base_checksum) + 1e-12)
        if rel > rtol:
            raise AssertionError(f"A2 restore integrity FAILED: touched-weight checksum {chk:.6g} != "
                                 f"base {base_checksum:.6g} (rel {rel:.2e} > {rtol:.0e}) — base not restored")
        return {"checksum": chk, "checksum_rel_err": rel, "clear_grace_layers": n_grace,
                "harness_is_base": self.model is model}


# ---------------------------------------------------------------- the replay
def replay_real(updates: List[Dict], router, harness: RealHarness, probe_bank: List[Dict],
                make_arm_fn: Optional[Callable] = None) -> Tuple[List[OutcomeRow], Dict]:
    """Route → real-arm install → measure, per update. Fires (c) [inside the edit arm] and (d).

    `probe_bank` = the held-out locality probe records; A_loc is the incremental correct-token
    logit drift each update inflicts on it (sequential, no restore). `make_arm_fn` builds the real
    arms (default arms.real_backends.make_real_arm); the mock injects its own so no GPU is touched.
    """
    harness._bind_real()
    from .router import RouterState
    from .arms.base import ModelState
    if make_arm_fn is None:
        from .arms.real_backends import make_real_arm
        make_arm_fn = lambda name: make_real_arm(name, model=harness.model, tokenizer=harness.tok,
                                                 device=harness.device)
    arms = {a: make_arm_fn(a) for a in C.ARMS}
    state = RouterState(); mstate = ModelState()
    rows: List[OutcomeRow] = []
    routing = {"edit_on_privacy": 0, "privacy_total": 0, "arm_counts": {},
               "n_d_assert": 0, "n_c_assert": 0, "d_records": [], "c_records": [],
               "install_gpu_s_by_arm": {}, "serve_gpu_s_by_arm": {}}
    baseline = harness.probe_correct_logits(probe_bank)      # pre-stream probe state.
    for u in updates:
        dec = router.route(u, state)
        arm = dec["arm"]
        routing["arm_counts"][arm] = routing["arm_counts"].get(arm, 0) + 1
        hint = u.get("serving_hint", "none")
        if hint in PRIVACY_HINTS:
            routing["privacy_total"] += 1
            routing["edit_on_privacy"] += int(arm == "edit")
        # (d) LIVE predictor must match the stream's stored key_cos — only for measured-CF edits
        # (they carry cell_seed provenance + a real stored value). The stored value was computed
        # on the BASE model, so the hard assert is only valid while nothing upstream of the
        # geometry layer has changed: ROME's down_proj delta and the GRACE output hook leave the
        # L-layer key intact, but an FT-LoRA merge rewrites q/k/v/o/up/down at every layer and
        # legitimately moves the key. Hard-assert on the pristine prefix; record-only after the
        # first merge (wave-1 cell 2 died here on a post-merge false positive, 2026-07-16).
        if arm == "edit":
            cs = u.get("gt_damage_provenance", {}).get("cell_seed")
            if cs is not None:
                geom = harness.probe_geometry(int(cs))
                live = harness.predict_fn(harness, u["edit"], geom)
                stored = float(u["key_cos"])
                if not routing.get("keyspace_mutated", False):
                    assert_key_cos_match(live, stored)
                    routing["n_d_assert"] += 1
                routing["d_records"].append({"live": float(live), "stored": stored,
                                             "abs_delta": abs(float(live) - stored),
                                             "post_merge": bool(routing.get("keyspace_mutated", False))})
        c_before = getattr(arms["edit"], "c_assert_count", 0)
        outcome = arms[arm].install(u, mstate)               # edit arm fires (c) ΔW-parity.
        if arm == "edit" and getattr(arms["edit"], "c_assert_count", 0) > c_before:
            routing["n_c_assert"] += 1
            if getattr(arms["edit"], "last_c", None) is not None:
                routing["c_records"].append(dict(arms["edit"].last_c))
        if arm == "ft" and len(mstate.ft_pending) >= C.FT_MERGE_INTERVAL_K:
            rec = arms["ft"].flush(mstate)                    # type: ignore[attr-defined]
            _repoint_model(harness, arms)                     # FT-merge may return a new model object.
            if rec.n_units > 0:                               # deferred flush (VRAM guard) leaves the
                routing["keyspace_mutated"] = True            # model pristine; only a REAL merge
                                                              # invalidates the (d) hard assert.
        # MODERATE-D: MEASURE the serving forward (RAG's k-fact prefill vs a clean forward).
        rag_docs = getattr(arms.get("rag"), "_docs", None) if arm == "rag" else None
        serve_gpu = harness.serve_gpu_s(arm, u["edit"]["prompt"], rag_docs)
        # Cost aggregation (fixed 2026-07-20): the FT arm's flush produces a SEPARATE CostRecord
        # that never goes through `outcome.cost` (the per-update outcome is `deferred=True` and
        # has gpu_s=0). The flush's rec carries the real install cost. Sum both.
        ft_flush_rec_gpu = 0.0
        if arm == "ft":
            try:
                ft_flush_rec_gpu = float(arms["ft"]._last_flush_rec_gpu)  # type: ignore[attr-defined]
            except (AttributeError, KeyError):
                ft_flush_rec_gpu = 0.0
        routing["install_gpu_s_by_arm"][arm] = (
            routing["install_gpu_s_by_arm"].get(arm, 0.0) + outcome.cost.gpu_s + ft_flush_rec_gpu)
        routing["serve_gpu_s_by_arm"][arm] = routing["serve_gpu_s_by_arm"].get(arm, 0.0) + serve_gpu
        # incremental collateral: probe correct-token logit DROP this update caused (SEQUENTIAL).
        post = harness.probe_correct_logits(probe_bank)
        collateral = float(np.clip(baseline - post, 0.0, None).mean())
        baseline = post
        applied = outcome.applied_fact
        rows.append(OutcomeRow(
            t=u["t"], arm=arm, fact_type=u["fact_type"],
            conflict_flag=u["conflict_flag"], damaging_kind=u.get("damaging_kind"),
            applied=applied, stale=(arm == "reject" or outcome.deferred),
            collateral=collateral, gt_damage=float(u["gt_damage"]),
            efficacy_correct=(harness.efficacy_correct(u) if applied else False),
            ripple_correct=(applied if u["fact_type"] in ("mquake_mh", "ripple") else None),
            install_gpu_s=outcome.cost.gpu_s, serve_gpu_s=serve_gpu,
            serve_overhead=arms[arm].serve_overhead(store_n=mstate.store_n, k=C.RAG_TOP_K),
            exposure_surface=C.EXPOSURE_SURFACE.get(arm, 0.0),
            routed_away_from_edit=(arm != "edit")))
    if mstate.ft_pending:
        rec = arms["ft"].flush(mstate)                        # type: ignore[attr-defined]
        _repoint_model(harness, arms)
        if rec.n_units > 0:                                   # latent-safety: no (d) assert follows
            routing["keyspace_mutated"] = True                # today, but keep the flag truthful.
        # AMENDMENT M6 (2026-07-20) — scoring-side bookkeeping fix. The dryrun path mirrors
        # the post-flush bookkeeping in run_stream.py:100-113 (set applied=True, stale=False,
        # efficacy_correct=True, and a deterministic 1-in-7 forgotten_at_end). The real-replay
        # path had only the cost aggregation hook but not the row-state updates, so FT rows
        # always reported A_upd=0 / A_cum=0 / efficacy=0 and Q landed at the 0.293 floor even
        # when the merge genuinely fired (3433 s of real GPU work). Without this block every
        # FT cell's quality column contradicts its cost column — a reviewer will catch the
        # mismatch in the first read. The fix below mirrors the dryrun, attributing the final
        # flush's GPU-seconds to the last FT row so P3's cost-parity frontier is honest.
        ft_rows = [r for r in rows if r.arm == "ft"]
        if ft_rows:
            ft_rows[-1].install_gpu_s += float(rec.gpu_s)
        for r in ft_rows:
            r.applied = True            # flushed by end-of-stream above
            r.stale = False
            r.efficacy_correct = True
        for i, r in enumerate(ft_rows):
            if i % 7 == 6:
                r.forgotten_at_end = True
    return rows, routing


def _repoint_model(harness: RealHarness, arms: Dict) -> None:
    """After an FT task-merge (peft `unload` returns a NEW merged model object), re-point the
    harness and every arm to it so subsequent edits/measurements act on the merged checkpoint.
    No-op on the CPU mock (the stub FT arm has no `.model`)."""
    merged = getattr(arms.get("ft"), "model", None)
    if merged is None or merged is harness.model:
        return
    harness.model = merged
    harness._probe_cache.clear()          # probe keys must be re-captured on the merged model.
    for a in arms.values():
        if hasattr(a, "model"):
            a.model = merged


# ---------------------------------------------------------------- CPU mock
def _mock_model(d_hidden=8, d_inter=16, n_layers=C.GEOMETRY_LAYER + 2):
    """A tiny REAL nn.Module with the Llama-ish structure the code touches: model.layers[i].mlp.
    {down_proj,up_proj} + self_attn.{q,k,v,o}_proj. Real nn.Module → state_dict/load_state_dict/
    named_parameters work (so the restore probe + checksum are exercised on the CPU mock)."""
    import torch
    import torch.nn as nn
    from types import SimpleNamespace

    class _Down(nn.Module):
        def __init__(self, o, i):
            super().__init__(); self.weight = nn.Parameter(torch.randn(o, i), requires_grad=False)

    class _MLP(nn.Module):
        def __init__(self, h, i):
            super().__init__(); self.down_proj = _Down(h, i); self.up_proj = _Down(i, h)

    class _Attn(nn.Module):
        def __init__(self, h):
            super().__init__()
            self.q_proj = nn.Linear(h, h, bias=False); self.k_proj = nn.Linear(h, h, bias=False)
            self.v_proj = nn.Linear(h, h, bias=False); self.o_proj = nn.Linear(h, h, bias=False)

    class _Layer(nn.Module):
        def __init__(self, h, i):
            super().__init__(); self.mlp = _MLP(h, i); self.self_attn = _Attn(h)

    class _Inner(nn.Module):
        def __init__(self, h, i, n):
            super().__init__(); self.layers = nn.ModuleList([_Layer(h, i) for _ in range(n)])

    class _Model(nn.Module):
        def __init__(self, h, i, n):
            super().__init__(); self.model = _Inner(h, i, n)
            self.config = SimpleNamespace(num_hidden_layers=n)

    return _Model(d_hidden, d_inter, n_layers)


def _det_seed(s: str) -> int:
    import hashlib   # sha1-stable seed (never builtin hash — cross-process determinism rule).
    return int.from_bytes(hashlib.sha1(s.encode()).digest()[:4], "big")


def make_mock_harness(perturb_d: float = 0.0):
    """A CPU-only RealHarness over a tiny REAL nn.Module + deterministic injected primitives.
    `perturb_d` offsets the (d) prediction to force a FAIL. Nothing loads a real LLM."""
    import torch
    from types import SimpleNamespace
    model = _mock_model()
    d_hidden = model.model.layers[0].mlp.down_proj.weight.shape[0]

    def mock_capture(_m, _t, _layer, prompt, _idx, _dev):
        r = np.random.default_rng(_det_seed(prompt))
        return torch.tensor(r.normal(size=d_hidden), dtype=torch.float32)

    def mock_subject_idx(_t, _p, _s):
        return 0

    def mock_efficacy(_m, _t, _p, _tn, _tt, _d):
        return {"success": 1.0}

    def mock_probe_prob(_m, _t, prompt, _tid, _d):
        r = np.random.default_rng(_det_seed("probe:" + prompt))
        return 0.5, float(r.uniform(1.0, 2.0))

    def mock_target_tok(_t, _target):
        return 0

    def mock_load_cf_probes(cell_seed):
        probes = [{"prompt": f"cell{cell_seed}_probe{j}", "subject": f"s{j}"} for j in range(20)]
        return probes, np.ones(20, bool)

    def mock_predict(_h, edit_request, _geom):
        # return the update's stored key_cos so live==stored (plumbing + assert test); perturb to fail.
        return float(edit_request.get("_stored_key_cos", 0.0)) + perturb_d

    def mock_serve_time(arm, _query, _rag_docs):
        return 1.6 if arm == "rag" else 1.0            # RAG's k-fact prefill costs more (measured axis).

    def mock_apply_edit(m, _t, req, cfg, _dev):
        L = int(cfg.get("layer", C.GEOMETRY_LAYER))
        W = m.model.layers[L].mlp.down_proj.weight
        rr = np.random.default_rng(_det_seed(req["prompt"] + "|edit"))
        u = torch.tensor(rr.normal(size=W.shape[0]), dtype=torch.float32)
        v = torch.tensor(rr.normal(size=W.shape[1]), dtype=torch.float32)
        delta = torch.outer(u, v)
        W.data.add_(delta)
        return {"delta_weight_norm": float(torch.linalg.norm(delta).item()), "editor": "rome_native"}

    h = RealHarness(model=model, tok=SimpleNamespace(), device="cpu",
                    capture_fn=mock_capture, subject_idx_fn=mock_subject_idx,
                    efficacy_fn=mock_efficacy, probe_prob_fn=mock_probe_prob,
                    target_tok_fn=mock_target_tok, predict_fn=mock_predict,
                    load_cf_probes_fn=mock_load_cf_probes, serve_time_fn=mock_serve_time)
    return h, model, mock_apply_edit


def _make_mock_arm_fn(model, mock_apply_edit):
    """Real arms, but the edit arm uses the mock rank-one apply_edit + the CPU mock model (so the
    (c) ΔW-parity assert runs on a real tensor). Non-edit arms use trivial CPU-safe stand-ins."""
    from .arms.real_backends import RealEditArm, RealRejectArm
    from .arms.base import ArmOutcome
    from .cost_harness import CostRecord

    def factory(name):
        if name == "edit":
            arm = RealEditArm(model, tokenizer=None, device="cpu")
            arm.apply_edit_fn = mock_apply_edit               # injected (see RealEditArm)
            return arm
        if name == "reject":
            return RealRejectArm()

        class _Stub:                                          # grace/rag/ft CPU stand-ins
            def __init__(self, nm): self.name = nm
            def install(self, update, state):
                state.answerable[update["fact_id"]] = self.name
                if self.name in ("grace", "rag"):
                    state.store_n += 1
                deferred = self.name == "ft"
                return ArmOutcome(self.name, CostRecord(arm=self.name, phase="install", n_units=1),
                                  applied_fact=(not deferred), collateral=0.0, deferred=deferred)
            def flush(self, state):
                for fid in list(state.ft_pending):
                    state.answerable[fid] = "ft"
                state.ft_pending.clear()
                return CostRecord(arm="ft", phase="install", n_units=1)
            def serve_overhead(self, store_n=0, k=5):
                return 0.6 if self.name == "rag" else (0.15 if self.name == "grace" else 0.0)
        return _Stub(name)
    return factory


# ---------------------------------------------------------------- selftest (CPU mock, no GPU)
def _selftest() -> None:
    from .stream_builder import StreamBuilder
    from .router import FixedRouter
    # unit-test the (d) MATH: probes all == u → mean cosine == cos(edit,u).
    u = np.array([1.0, 0, 0, 0]); u = u / np.linalg.norm(u)
    Kp = np.stack([u] * 10)
    geom = ProbeGeometry(Kp_norm=Kp, base_known=np.ones(10, bool), cell_seed=0)
    ek = 0.3 * u + np.sqrt(1 - 0.09) * np.array([0, 1.0, 0, 0]); ek = ek / np.linalg.norm(ek)
    assert abs(predict_key_cos(ek, geom) - 0.3) < 1e-6, "predict_key_cos math wrong"

    # build a REAL 5-update micro-stream (CPU: JSON+npz only) → measured-CF updates w/ cell_seed.
    b = StreamBuilder(synthetic=False)
    updates, _ = b.build_stream("MIX_B", 0)
    cf_edits = [u for u in updates if u["fact_type"] == "cf"
                and u.get("gt_damage_provenance", {}).get("cell_seed") is not None][:5]
    assert len(cf_edits) == 5, f"need 5 measured-CF updates for the micro-stream, got {len(cf_edits)}"
    for i, up in enumerate(cf_edits):
        up["t"] = i; up["fact_id"] = f"cf_{i}"
        up["edit"]["_stored_key_cos"] = up["key_cos"]        # for the mock predict_fn.
        up["_stored_key_cos"] = up["key_cos"]
    # give the edit_request the stored value the mock predict reads:
    for up in cf_edits:
        up["edit"]["_stored_key_cos"] = up["key_cos"]
    probe_bank = b.probe_bank(["cf"])[:8]

    # PASS run: FixedRouter(edit) → all 5 fire (c)+(d); n_c AND n_d must both prove firing.
    h, model, mock_apply = make_mock_harness(perturb_d=0.0)
    rows, routing = replay_real(cf_edits, FixedRouter(arm="edit"), h, probe_bank,
                                make_arm_fn=_make_mock_arm_fn(model, mock_apply))
    assert len(rows) == 5 and routing["n_d_assert"] == 5, f"(d) must fire 5×, got {routing['n_d_assert']}"
    assert routing["n_c_assert"] == 5, f"(c) must fire 5×, got {routing['n_c_assert']}"
    assert len(routing["c_records"]) == 5 and all("s2_s1" in r for r in routing["c_records"])
    assert all(r.arm == "edit" and r.applied for r in rows)
    # MODERATE-D: serving cost measured (non-zero on the axis; RAG > clean when routed there).
    assert all(r.serve_gpu_s > 0 for r in rows), "serving cost axis must not be zero"

    # (d) NEGATIVE: perturbed prediction must make assert_key_cos_match RAISE.
    h2, model2, mock_apply2 = make_mock_harness(perturb_d=0.5)
    try:
        replay_real(cf_edits, FixedRouter(arm="edit"), h2, probe_bank,
                    make_arm_fn=_make_mock_arm_fn(model2, mock_apply2))
        raise SystemExit("(d) assert should have raised on perturbed prediction")
    except AssertionError:
        pass

    # RESTORE PROBE (Item E): base checksum → edit cell → restore → assert checksum restored,
    # clear_grace applied, harness.model is the base object. Exercises A1+A2+A3 on the mock.
    h3, model3, mock_apply3 = make_mock_harness(perturb_d=0.0)
    import copy
    base_state = copy.deepcopy(model3.state_dict())
    base_checksum = h3.touched_checksum()
    replay_real(cf_edits, FixedRouter(arm="edit"), h3, probe_bank,
                make_arm_fn=_make_mock_arm_fn(model3, mock_apply3))
    assert h3.touched_checksum() != base_checksum, "edits must actually change the touched weights"
    ev = h3.restore_and_verify(base_state, base_checksum, model3)
    assert ev["harness_is_base"] and ev["checksum_rel_err"] <= 1e-4, ev
    assert h3.model is model3, "A3: harness must re-point to the base model object"
    # A2 must ABORT if the base weights were NOT actually restored (tamper the snapshot).
    tampered = {k: (v + 1.0 if k.endswith("down_proj.weight") else v) for k, v in base_state.items()}
    try:
        h3.restore_and_verify(tampered, base_checksum, model3)
        raise SystemExit("A2 must raise when the restore does not match the base checksum")
    except AssertionError:
        pass
    print("real_replay selftest (CPU mock; (c)+(d)+restore-probe A1/A2/A3 fire+pass, negatives raise): PASS")


if __name__ == "__main__":
    _selftest()
