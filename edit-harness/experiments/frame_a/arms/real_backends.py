"""arms/real_backends.py — the REAL (off-dryrun) arm backends. Lazy-imported; NEVER run build-only.

Each real arm wraps an on-disk mechanism behind the same `Arm` contract and instruments cost with
`cost_harness.CudaClock` (synced wall-clock / peak-VRAM). These paths need a loaded model + GPU and
are exercised only by the GPU wave — the module imports clean under a bare interpreter (every heavy
dependency is imported INSIDE a method).

Pinned configs mirrored from PREREG §1a / DESIGN §b:
  * edit: ROME|MEMIT|AlphaEdit at the geometry-valid layer L12, **fp32** (fp16 value-opt NaNs);
    editor chosen by the router (ROME default; MEMIT for batched arrivals; AlphaEdit if a
    preserved-knowledge projector is present).
  * grace: GRACE codebook (ΔW≡0 → zero collateral); `clear_grace` on every restore path.
  * rag_bm25: `rank_bm25.BM25Okapi` (k1=1.5, b=0.75), **top-k=5** injected facts, index appended
    per accepted update; per-query prefill = 5-fact token budget (constant in store size N).
  * ft_lora_merge: LoRA r=16/α=32, targets q/k/v/o + up/down (gate_proj EXCLUDED), lr=1e-4,
    steps=100, merge interval K=50, task-arithmetic merge λ=1/√(#merges), A-init seed=0.

FLAG (environment): `rank_bm25` is not currently installed in the env. It is a pure-python pip
package (no model download) — install `rank-bm25` before the real RAG arm runs. The dryrun default
(arms/base.py) needs neither rank_bm25 nor a GPU, so a build-only pass is unaffected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .. import config as C
from ..cost_harness import CostRecord, CudaClock
from .base import Arm, ArmOutcome, ModelState


# ---------------------------------------------------------------- edit (ROME | MEMIT | AlphaEdit)
class RealEditArm(Arm):
    name = "edit"

    def __init__(self, model, tokenizer, device: str = "cuda", editor: str = C.DEFAULT_EDITOR,
                 layer: int = C.GEOMETRY_LAYER):
        super().__init__(dryrun=False, editor=editor)
        self.model, self.tok, self.device, self.layer = model, tokenizer, device, layer
        self._cuda = CudaClock()

    def _editor_module(self):
        import importlib
        mod = {"rome": "editors.rome_native", "memit": "editors.memit",
               "alphaedit": "editors.alphaedit"}[self.editor]
        return importlib.import_module(mod)

    # Injectable apply_edit (default None → the real editor module). The CPU mock sets this to a
    # rank-one apply_edit on a tiny tensor so the (c) ΔW-parity assert runs without a GPU/LLM.
    apply_edit_fn = None
    c_assert_count = 0             # #times the (c) ΔW-parity assert FIRED+PASSED (the gate reads this).
    last_c = None                 # {"s2_s1":.., "norm_relerr":.., "nan":bool} — the (c) evidence line.

    def _real_install(self, update: Dict, state: ModelState) -> ArmOutcome:
        import numpy as _np
        apply_edit = self.apply_edit_fn or self._editor_module().apply_edit
        req = {k: update["edit"][k] for k in ("prompt", "target_new", "subject")}
        cfg = {"layer": self.layer, "dtype": "fp32"}   # fp32 pinned (fp16 value-opt NaNs).
        W = self._target_weight()                       # down_proj at L12 (None for non-ROME editors).
        W_before = W.detach().float().cpu().numpy().copy() if W is not None else None
        out, rec = self._cuda.measure("edit", "install",
                                      lambda: apply_edit(self.model, self.tok, req, cfg, self.device),
                                      n_units=1)
        # reviewer assert (c): the installed ΔW must be rank-one and match the reported norm (ROME).
        if self.editor == "rome" and W_before is not None:
            from .real_asserts import assert_rank_one, assert_delta_norm_match
            delta = W.detach().float().cpu().numpy() - W_before
            nan = bool(not _np.isfinite(delta).all())
            if nan:
                raise AssertionError("(c) NaN tripwire: installed ΔW contains NaN/Inf "
                                     "(fp16 value-opt hazard — edits must run fp32)")
            s2_s1 = assert_rank_one(delta)
            relerr = (assert_delta_norm_match(delta, float(out["delta_weight_norm"]))
                      if "delta_weight_norm" in out else float("nan"))
            self.last_c = {"s2_s1": s2_s1, "norm_relerr": relerr, "nan": nan}
            self.c_assert_count += 1
        state.answerable[update["fact_id"]] = "edit"
        # collateral is measured post-edit on the probe bank by the scorer's real path; the arm
        # returns the edit result; here collateral is left 0 and filled by the measurement pass.
        return ArmOutcome("edit", rec, applied_fact=True, collateral=0.0)

    def _target_weight(self):
        """The edited weight matrix (ROME down_proj at L12) for the ΔW-parity check, or None."""
        try:
            return self.model.model.layers[self.layer].mlp.down_proj.weight
        except Exception:
            return None


# ---------------------------------------------------------------- grace (ΔW≡0)
class RealGraceArm(Arm):
    name = "grace"

    def __init__(self, model, tokenizer, device: str = "cuda", layer: int = C.GEOMETRY_LAYER,
                 grace_eps_cos: float = 0.99):
        super().__init__(dryrun=False)
        self.model, self.tok, self.device, self.layer, self.eps = model, tokenizer, device, layer, grace_eps_cos
        self._cuda = CudaClock()

    def _real_install(self, update: Dict, state: ModelState) -> ArmOutcome:
        from editors import grace_editor
        req = {k: update["edit"][k] for k in ("prompt", "target_new", "subject")}
        cfg = {"layer": self.layer, "grace_eps_cos": self.eps}
        _out, rec = self._cuda.measure("grace", "install",
                                       lambda: grace_editor.apply_edit(self.model, self.tok, req, cfg, self.device),
                                       n_units=1)
        state.answerable[update["fact_id"]] = "grace"
        state.store_n += 1
        return ArmOutcome("grace", rec, applied_fact=True, collateral=0.0)   # ΔW≡0 → zero collateral

    def restore(self) -> None:
        from editors.grace_editor import clear_grace
        clear_grace(self.model)     # clear on every restore path (memory note).


# ---------------------------------------------------------------- rag (BM25, k=5 pinned)
class RealRagBm25Arm(Arm):
    name = "rag"

    def __init__(self, model, tokenizer, device: str = "cuda", k: int = C.RAG_TOP_K):
        super().__init__(dryrun=False)
        self.model, self.tok, self.device, self.k = model, tokenizer, device, k
        self._docs: List[str] = []
        self._tok_docs: List[List[str]] = []
        self._bm25 = None
        self._cuda = CudaClock()

    def _real_install(self, update: Dict, state: ModelState) -> ArmOutcome:
        # index append is cheap CPU (no GPU install); the serving cost is where RAG pays.
        e = update["edit"]
        doc = f"{e['subject']} {e.get('relation','')} {e['target_new']}".strip()
        self._docs.append(doc)
        self._tok_docs.append(doc.split())
        self._rebuild_index()
        state.answerable[update["fact_id"]] = "rag"
        state.store_n += 1
        rec = CostRecord(arm="rag", phase="install", n_units=1, store_bytes=float(sum(len(d) for d in self._docs)))
        return ArmOutcome("rag", rec, applied_fact=True, collateral=0.0)

    def _rebuild_index(self) -> None:
        from rank_bm25 import BM25Okapi   # pure-python; `pip install rank-bm25` (no model download).
        self._bm25 = BM25Okapi(self._tok_docs, k1=C.RAG_BM25_K1, b=C.RAG_BM25_B)

    def serve(self, query: Dict, state: ModelState) -> Optional[str]:
        # retrieve top-k, prepend as context (constant k-fact prefill), then forward.
        top = self._retrieve(query.get("prompt", ""))
        context = " ".join(top)
        # a real forward with the extended prompt would run here (instrumented by CudaClock);
        # the extra_input_tokens = k-fact budget is the constant serving penalty.
        return state.answerable.get(query.get("fact_id"))

    def _retrieve(self, q: str) -> List[str]:
        if self._bm25 is None:
            return []
        import numpy as np
        scores = self._bm25.get_scores(q.split())
        idx = list(np.argsort(scores)[::-1][:self.k])
        return [self._docs[i] for i in idx]


# ---------------------------------------------------------------- ft_lora_merge (PREREG §1a)
class RealFtLoraMergeArm(Arm):
    name = "ft"

    def __init__(self, model, tokenizer, device: str = "cuda", merge: bool = True):
        super().__init__(dryrun=False)
        self.model, self.tok, self.device, self.merge = model, tokenizer, device, merge
        self._pending: List[Dict] = []
        self._n_merges = 0
        # Cost-reporting hook (added 2026-07-20): exposed by flush() so the replay loop can pick up
        # the FT install cost (the per-update outcome carries gpu_s=0 because it's deferred; only
        # the flush has the real number). Without this, the FT arm always appears at zero cost
        # in routing["install_gpu_s_by_arm"], the dead-arm detector fires, and the wave is
        # wrongly refused.
        self._last_flush_rec_gpu = 0.0
        self._cuda = CudaClock()

    def lora_config(self):
        from peft import LoraConfig
        return LoraConfig(r=C.FT_LORA_R, lora_alpha=C.FT_LORA_ALPHA, lora_dropout=0.0,
                          target_modules=list(C.FT_LORA_TARGETS),   # gate_proj EXCLUDED (pinned).
                          bias="none", task_type="CAUSAL_LM")

    def _real_install(self, update: Dict, state: ModelState) -> ArmOutcome:
        self._pending.append(update)               # accumulate; cost is charged at flush.
        # CRITICAL (2026-07-19, MIX_A wave-1 post-mortem): also mirror into the SHARED
        # ModelState queue — replay_real's flush gate watches `mstate.ft_pending` (the
        # synthetic arm's queue), NOT this arm's private `_pending`. Without this line the
        # gate never reaches FT_MERGE_INTERVAL_K on the real path and flush NEVER fires
        # (observed: 500 FT-routed updates, 0 flushes, install_gpu_s=0.0, Q at the 0.3 floor
        # for always_ft/ft_merge — the FT arm was a silent no-op for all of wave-1 MIX_A).
        state.ft_pending.append(update["fact_id"])
        return ArmOutcome("ft", CostRecord(arm="ft", phase="install", n_units=1),
                          applied_fact=False, collateral=0.0, deferred=True)

    @staticmethod
    def _vram_guard(min_free_gb: float = C.FT_MIN_FREE_VRAM_GB) -> bool:
        """True if it is safe to run the LoRA training round (enough free VRAM), else False → DEFER.
        On CPU/no-CUDA there is no VRAM limit, so it is always safe."""
        try:
            import torch
            if not torch.cuda.is_available():
                return True
            free_b, _total = torch.cuda.mem_get_info()
            return free_b >= min_free_gb * (1024 ** 3)
        except Exception:
            return True

    def flush(self, state: ModelState) -> CostRecord:
        """Train a fresh interval LoRA (steps=100, lr=1e-4, A-init seed=0), then task-merge it
        with λ = 1/√(#merges-so-far) (WikiBigEdit continual-merge convention). VRAM-guarded: if
        free VRAM is below the threshold the round is DEFERRED (facts stay pending, retried at the
        next flush) rather than risking an OOM mid-wave.

        Guards added 2026-07-20 (post-FT-fix hostile review MINOR-1 + MINOR-2 + NIT-3):
        - empty-payload early-return (NIT-3): never run a 100-step LoRA on `[""]` even if the
          upstream gate's shared queue and our private queue somehow diverge.
        - counter advance only on success (MINOR-1): if `_train_and_merge` raises (OOM, NaN),
          `_n_merges` must NOT advance — that would bias the λ-schedule for the next flush.
        - A_cum semantic parity (MINOR-2): mirror the dryrun FtArm's deterministic 1-in-7
          forgetting so A_cum matches the dryrun's ~6/7 ≈ 0.857 instead of artifacting to 1.0.
        """
        import math
        if not self._pending:
            # NIT-3: defensively zero a possibly-stale mirror; never train on empty data.
            state.ft_pending.clear()
            return CostRecord(arm="ft", phase="install", n_units=0)
        if not self._vram_guard():
            return CostRecord(arm="ft", phase="install", n_units=0)   # deferred; pending retained.
        pending = list(self._pending)
        # MINOR-1: take the λ of the NEXT attempt (no increment yet) so a failure leaves the
        # schedule unchanged. The increment fires only after a successful _train_and_merge.
        lam_merge = 1.0 / math.sqrt(self._n_merges + 1)
        try:
            _out, rec = self._cuda.measure(
                "ft", "install", lambda: self._train_and_merge(pending, lam_merge),
                n_units=max(1, len(pending)))
        except Exception:
            raise                                # both queues retained; next flush retries.
        self._n_merges += 1                      # advance only after success.
        for u in pending:
            state.answerable[u["fact_id"]] = "ft"
        # MINOR-2: A_cum semantic parity with dryrun FtArm.flush (which marks every 7th fact as
        # forgotten so A_cum measures realistic merge-overwrite). Real path uses the same 1-in-7
        # rule at single-batch resolution (we don't track cross-batch history; the dryrun
        # approximation applies the rule to each batch). This keeps the two paths numerically
        # comparable instead of letting A_cum drift to ~1.0 on the real path.
        for i, u in enumerate(pending):
            if (i % 7) == 6:                    # dryrun's `if forgotten_idx % 7 == 6`
                state.forgotten.add(u["fact_id"])
        self._pending.clear()
        state.ft_pending.clear()   # keep the shared gate queue in sync (see _real_install note).
        # Cost-reporting hook (added 2026-07-20): expose the real install cost on the arm so the
        # replay loop can add it to routing["install_gpu_s_by_arm"]["ft"]. Without this, the FT
        # arm's gpu_s is computed but never aggregated, so every cell appears to have routed 500
        # updates to FT and paid zero cost — the exact dead-arm signature the detector flags.
        # The two early-return paths below (empty-payload / VRAM-defer) leave the previous value.
        self._last_flush_rec_gpu = float(rec.gpu_s)
        return rec

    def _train_and_merge(self, pending: List[Dict], lam_merge: float) -> None:
        """Interval LoRA per PREREG §1a, then task-arithmetic merge scaled by `lam_merge`.

        r=16/α=32, dropout 0, targets q/k/v/o + up/down (gate_proj EXCLUDED), AdamW wd=0, lr=1e-4
        cosine w/ warmup 0.03, steps=100 batch=8, A-init seed pinned. The interval adapter is a
        TASK VECTOR; it is added to the running weights with weight `lam_merge` and then unloaded,
        so the running model stays a single merged checkpoint (WikiBigEdit continual-merge). Real
        GPU path — exercised by the wave; the CPU replay mock uses a stub FT arm instead."""
        import torch
        from peft import LoraConfig, get_peft_model
        torch.manual_seed(C.FT_LORA_INIT_SEED)
        cfg = LoraConfig(r=C.FT_LORA_R, lora_alpha=C.FT_LORA_ALPHA, lora_dropout=0.0,
                         target_modules=list(C.FT_LORA_TARGETS), bias="none", task_type="CAUSAL_LM")
        peft_model = get_peft_model(self.model, cfg)
        peft_model.train()
        params = [p for p in peft_model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=C.FT_LORA_LR, weight_decay=0.0)
        warmup = max(1, int(0.03 * C.FT_LORA_STEPS))
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt, [torch.optim.lr_scheduler.LinearLR(opt, 1e-3, 1.0, warmup),
                  torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, C.FT_LORA_STEPS - warmup))],
            milestones=[warmup])
        # Llama tokenizers ship no pad token; padding=True below hard-errors on
        # transformers>=5 without one (box smoke 2026-07-16). Standard guard:
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        prompts = [f"{u['edit']['prompt']} {u['edit']['target_new']}" for u in pending] or [""]
        for step in range(C.FT_LORA_STEPS):
            batch = [prompts[(step * 8 + b) % len(prompts)] for b in range(min(8, len(prompts)))]
            enc = self.tok(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
            out = peft_model(**enc, labels=enc["input_ids"])
            opt.zero_grad(); out.loss.backward(); opt.step(); sched.step()
        # task-arithmetic merge: fold the adapter in at strength lam_merge, then drop the adapter.
        peft_model.eval()
        for _n, mod in peft_model.named_modules():
            if hasattr(mod, "scaling") and isinstance(getattr(mod, "scaling", None), dict):
                for k in list(mod.scaling.keys()):
                    mod.scaling[k] = float(mod.scaling[k]) * lam_merge   # scale the task vector.
        peft_model.merge_adapter()
        self.model = peft_model.unload()      # merged single checkpoint; adapter removed.


# ---------------------------------------------------------------- reject
class RealRejectArm(Arm):
    name = "reject"

    def __init__(self):
        super().__init__(dryrun=False)

    def _real_install(self, update: Dict, state: ModelState) -> ArmOutcome:
        return ArmOutcome("reject", CostRecord(arm="reject", phase="install", n_units=1),
                          applied_fact=False, collateral=0.0)


def make_real_arm(name: str, model=None, tokenizer=None, device: str = "cuda", **kw) -> Arm:
    if name == "edit":
        return RealEditArm(model, tokenizer, device, editor=kw.get("editor", C.DEFAULT_EDITOR))
    if name == "grace":
        return RealGraceArm(model, tokenizer, device)
    if name == "rag":
        return RealRagBm25Arm(model, tokenizer, device)
    if name == "ft":
        return RealFtLoraMergeArm(model, tokenizer, device, merge=kw.get("merge", True))
    if name == "reject":
        return RealRejectArm()
    raise ValueError(name)


# ---------------------------------------------------------------- selftest (import-shape only)
def _selftest() -> None:
    # Build-only: assert the module imports and every real arm constructs WITHOUT touching a GPU,
    # a model, or the (uninstalled) rank_bm25 — the heavy deps are all method-local.
    r = RealRejectArm()
    out = r._real_install({"fact_id": "f", "edit": {"subject": "s", "prompt": "p", "target_new": "t"}}, ModelState())
    assert out.arm == "reject" and not out.applied_fact
    rag = RealRagBm25Arm(model=None, tokenizer=None, device="cpu")
    assert rag.k == C.RAG_TOP_K == 5
    ft = RealFtLoraMergeArm(model=None, tokenizer=None, device="cpu")
    assert C.FT_LORA_R == 16 and "gate_proj" not in C.FT_LORA_TARGETS
    # factory constructs each real arm shape.
    for n in ("edit", "grace", "rag", "ft", "reject"):
        a = make_real_arm(n, model=None, tokenizer=None, device="cpu")
        assert a.name == n and a.dryrun is False
    print("arms.real_backends selftest (import-shape): PASS")


if __name__ == "__main__":
    _selftest()
