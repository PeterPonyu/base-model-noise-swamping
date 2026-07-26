"""arms/base.py — the common Arm interface + shared synthetic knowledge model (DRYRUN).

An Arm absorbs one update and serves queries. The contract:
    install(update)             -> ArmOutcome  (cost + knowledge effect)
    serve(query, model_state)   -> answer str
    exposure_surface            -> float in [0,1]  (governance/plaintext liability)
    footprint_bytes(store_n)    -> float           (serving-time fact-store size)

DRYRUN outcome model (deterministic, numpy-only): each update carries `gt_damage` (from the
stream, the pre-existing B6 collateral) and `key_cos` (the geometry signal). Arms differ ONLY
in how they absorb the fact and what collateral they inflict — this is what lets the scorer,
router, oracle and the P1/P2 predicates be exercised build-only.

  edit  : applies the fact to weights; collateral on the probe bank ∝ gt_damage; serves offline
          fine; exposure 0; footprint 0.
  grace : applies the fact via codebook; collateral ≡ 0 (ΔW≡0); exposure ~0.3 (codebook, not raw
          text); footprint grows slowly.
  rag   : applies the fact via retrieval; collateral 0; a LOCAL-INDEX RAG SERVES OFFLINE FINE;
          exposure 1 (raw readable fact store); footprint grows with retained updates.
  ft    : applies at merge intervals (deferred until a flush); collateral moderate; some
          forgetting of earlier facts; exposure 0; footprint 0.
  reject: does not apply the fact (stale); collateral 0; exposure 0.

Real backends (rome_native/grace_editor/memit/alphaedit/ft_editor apply_edit, rank_bm25) are
lazy-imported inside the `_real_*` methods and are never called in a build-only / dryrun run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .. import config as C
from ..cost_harness import CostRecord, SyntheticClock


@dataclass
class ArmOutcome:
    """What absorbing one update did, plus its install cost."""
    arm: str
    cost: CostRecord
    applied_fact: bool          # did the fact become answerable?
    collateral: float           # signed damage inflicted on the probe bank (>=0 typical)
    deferred: bool = False      # ft: fact queued for the next merge flush (not yet answerable)


@dataclass
class ModelState:
    """The synthetic served-model state a stream instance accumulates (DRYRUN)."""
    answerable: Dict[str, str] = field(default_factory=dict)   # fact_id -> arm that carries it
    collateral_total: float = 0.0                              # accumulated probe-bank damage
    forgotten: set = field(default_factory=set)                # fact_ids lost to FT overwrite
    ft_pending: List[str] = field(default_factory=list)        # facts queued for next FT flush
    store_n: int = 0                                           # #facts in the RAG/GRACE store


class Arm:
    """Base arm. Subclasses set `name` and override `_absorb`/`exposure_surface`."""
    name: str = "base"

    def __init__(self, dryrun: bool = True, editor: str = C.DEFAULT_EDITOR):
        self.dryrun = dryrun
        self.editor = editor
        self._clock = SyntheticClock()

    # ------------------------------------------------------------------ interface
    def install(self, update: Dict, state: ModelState) -> ArmOutcome:
        if not self.dryrun:
            return self._real_install(update, state)   # never reached build-only
        return self._absorb(update, state)

    def serve(self, query: Dict, state: ModelState) -> Optional[str]:
        """Return the answer string for a query, or None if unanswerable (stale)."""
        fid = query["fact_id"]
        if fid in state.forgotten:
            return None
        return state.answerable.get(fid)

    @property
    def exposure_surface(self) -> float:
        return C.EXPOSURE_SURFACE[self.name]

    def footprint_bytes(self, store_n: int) -> float:
        return self._clock._store_bytes(self.name, store_n)

    def serve_overhead(self, store_n: int = 0, k: int = C.RAG_TOP_K) -> float:
        """Per-query serving overhead ABOVE a base forward (reference-forward multiples)."""
        base = self._clock.SERVE.get(self.name, 1.0)
        rec = self._clock.serve(self.name, n_queries=1, store_n=store_n, k=k)
        return rec.gpu_s - base + (base - 1.0)   # everything above the base forward=1.0

    # ------------------------------------------------------------------ dryrun absorb
    def _absorb(self, update: Dict, state: ModelState) -> ArmOutcome:
        raise NotImplementedError

    # ------------------------------------------------------------------ real (off-dryrun)
    def _real_install(self, update: Dict, state: ModelState) -> ArmOutcome:
        raise NotImplementedError(
            f"{self.name} real backend is not exercised in a build-only pass")


class EditArm(Arm):
    name = "edit"

    def _absorb(self, update: Dict, state: ModelState) -> ArmOutcome:
        fid = update["fact_id"]
        cost = self._clock.install("edit", n_units=1, store_n=state.store_n)
        # collateral ∝ ground-truth damage of this edit (ROME is the damage-prone primary).
        collateral = float(update["gt_damage"]) if self.editor == "rome" else \
            float(update["gt_damage"]) * 0.02       # AlphaEdit removes ~98%.
        state.answerable[fid] = "edit"
        state.collateral_total += collateral
        return ArmOutcome("edit", cost, applied_fact=True, collateral=collateral)

    def _real_install(self, update: Dict, state: ModelState) -> ArmOutcome:
        # Lazy real path (never reached build-only): editors/{rome_native,memit,alphaedit}.apply_edit
        raise NotImplementedError("edit real backend not exercised build-only")


class GraceArm(Arm):
    name = "grace"

    def _absorb(self, update: Dict, state: ModelState) -> ArmOutcome:
        fid = update["fact_id"]
        cost = self._clock.install("grace", n_units=1, store_n=state.store_n)
        state.answerable[fid] = "grace"
        state.store_n += 1                          # codebook grows (capacity-limited).
        return ArmOutcome("grace", cost, applied_fact=True, collateral=0.0)  # ΔW≡0


class RagArm(Arm):
    name = "rag"

    def _absorb(self, update: Dict, state: ModelState) -> ArmOutcome:
        fid = update["fact_id"]
        cost = self._clock.install("rag", n_units=1, store_n=state.store_n)   # ~0 GPU install
        state.answerable[fid] = "rag"
        state.store_n += 1                          # raw readable fact store grows with N.
        return ArmOutcome("rag", cost, applied_fact=True, collateral=0.0)


class FtArm(Arm):
    name = "ft"

    def _absorb(self, update: Dict, state: ModelState) -> ArmOutcome:
        fid = update["fact_id"]
        state.ft_pending.append(fid)
        # Cost is charged at flush; queuing itself is ~0. Fact is DEFERRED until the flush.
        cost = CostRecord(arm="ft", phase="install", n_units=1)
        return ArmOutcome("ft", cost, applied_fact=False, collateral=0.0, deferred=True)

    def flush(self, state: ModelState) -> CostRecord:
        """Run the periodic LoRA-merge over queued facts (DRYRUN synthetic).

        Cost is ONE training run (steps=100), amortised over the K accumulated facts — a CONSTANT
        per-flush cost, NOT proportional to the pending count (the earlier ×n scaling made FT
        dominate the realised GPU-second budget, which is not how a K-batched LoRA merge costs)."""
        cost = self._clock.install("ft", n_units=1, store_n=state.store_n)   # one LoRA run / flush
        for fid in state.ft_pending:
            state.answerable[fid] = "ft"
        # task-arithmetic merge over many facts overwrites a small fraction of prior FT facts
        # (synthetic forgetting → lowers A_cum). Deterministic: forget every 7th prior ft fact.
        prior = [f for f, a in state.answerable.items() if a == "ft"]
        for i, fid in enumerate(prior):
            if i % 7 == 6:
                state.forgotten.add(fid)
        state.ft_pending.clear()
        return cost


class RejectArm(Arm):
    name = "reject"

    def _absorb(self, update: Dict, state: ModelState) -> ArmOutcome:
        cost = CostRecord(arm="reject", phase="install", n_units=1)   # ~0
        return ArmOutcome("reject", cost, applied_fact=False, collateral=0.0)   # stale


ALL_ARMS = {"edit": EditArm, "grace": GraceArm, "rag": RagArm, "ft": FtArm, "reject": RejectArm}


def make_arm(name: str, dryrun: bool = True, editor: str = C.DEFAULT_EDITOR) -> Arm:
    return ALL_ARMS[name](dryrun=dryrun, editor=editor)


# ---------------------------------------------------------------- selftest
def _selftest() -> None:
    st = ModelState()
    up = {"fact_id": "f0", "gt_damage": 2.0, "key_cos": 0.4}
    # edit inflicts collateral ∝ gt_damage; grace/rag/reject inflict none.
    eo = make_arm("edit").install(up, ModelState())
    assert eo.collateral == 2.0 and eo.applied_fact
    go = make_arm("grace").install(up, ModelState())
    assert go.collateral == 0.0 and go.applied_fact
    ro = make_arm("rag").install(up, ModelState())
    assert ro.collateral == 0.0 and ro.applied_fact
    jo = make_arm("reject").install(up, ModelState())
    assert (not jo.applied_fact) and jo.collateral == 0.0
    # alphaedit editor ~ removes 98% collateral:
    ao = EditArm(editor="alphaedit").install(up, ModelState())
    assert ao.collateral < eo.collateral
    # ft defers then flushes:
    ftarm = make_arm("ft")
    fo = ftarm.install(up, st)
    assert fo.deferred and not fo.applied_fact
    ftarm.flush(st)
    assert st.answerable.get("f0") == "ft"
    # exposure ordering: rag(1) > grace(0.3) > edit(0):
    assert make_arm("rag").exposure_surface > make_arm("grace").exposure_surface > \
        make_arm("edit").exposure_surface == 0.0
    # per-query overhead: rag > edit:
    assert make_arm("rag").serve_overhead(store_n=1000) > make_arm("edit").serve_overhead(store_n=1000)
    # footprint: rag grows with N, edit flat 0:
    assert make_arm("rag").footprint_bytes(1000) > make_arm("rag").footprint_bytes(10)
    assert make_arm("edit").footprint_bytes(10**6) == 0.0
    print("arms.base selftest: PASS")


if __name__ == "__main__":
    _selftest()
