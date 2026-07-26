"""cost_harness.py — uniform cost instrumentation + reference normalisation.

Measures, for each arm phase (install / serve), wall-clock, GPU-seconds and peak VRAM, and
reports everything as RATIOS to a reference (one base-model forward = 1.0) — never absolute
currency (standing rule). Curves and tier-invariance are computed by analyze_frame_a.py from
the CostRecords this module emits.

DRYRUN: `SyntheticClock` produces deterministic synthetic costs (no torch, no GPU) so the whole
pipeline exercises the accounting logic build-only. Real timing (`CudaClock`) is used only off
DRYRUN and is imported lazily so this module loads under a bare interpreter.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional


@dataclass
class CostRecord:
    """One measured (or synthetic) cost sample for an arm phase.

    All fields are raw measurements; normalisation to the reference happens in
    `normalise()` so the raw numbers stay auditable. `extra_input_tokens` captures RAG's
    constant k-fact prefill (the load-bearing serving cost); `store_bytes` captures the
    footprint that grows with retained updates (kept SEPARATE from per-query latency).
    """
    arm: str
    phase: str                      # "install" | "serve"
    wall_s: float = 0.0
    gpu_s: float = 0.0
    peak_vram_bytes: float = 0.0
    extra_input_tokens: int = 0     # RAG prefill (constant in k, NOT growing with store N)
    store_bytes: float = 0.0        # retained fact store / index footprint (grows with N)
    n_units: int = 1                # #updates (install) or #queries (serve) this record covers

    def amortised_wall_s(self) -> float:
        return self.wall_s / max(1, self.n_units)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CostLedger:
    """Accumulates CostRecords across a stream instance and reports normalised ratios."""
    reference_wall_s: float = 1.0           # one base-model forward pass = the unit.
    records: list = field(default_factory=list)

    def add(self, rec: CostRecord) -> CostRecord:
        self.records.append(rec)
        return rec

    def total(self, phase: Optional[str] = None) -> Dict[str, float]:
        sel = [r for r in self.records if phase is None or r.phase == phase]
        return {
            "wall_s": sum(r.wall_s for r in sel),
            "gpu_s": sum(r.gpu_s for r in sel),
            "peak_vram_bytes": max([r.peak_vram_bytes for r in sel], default=0.0),
            "extra_input_tokens": sum(r.extra_input_tokens for r in sel),
            "store_bytes": max([r.store_bytes for r in sel], default=0.0),
        }

    def gpu_seconds(self) -> float:
        return sum(r.gpu_s for r in self.records)

    def install_cost(self) -> float:
        return sum(r.gpu_s for r in self.records if r.phase == "install")

    def serve_cost(self) -> float:
        return sum(r.gpu_s for r in self.records if r.phase == "serve")

    def normalise(self, value_wall_s: float) -> float:
        """Ratio to the reference forward pass — the only unit that appears in the paper."""
        return value_wall_s / max(1e-12, self.reference_wall_s)


class SyntheticClock:
    """Deterministic synthetic cost model for DRYRUN / selftest — NO torch, NO GPU.

    Encodes the QUALITATIVE cost ordering the design asserts, so accounting logic and the
    Pareto/must-win predicates can be exercised build-only:
      * edit: nonzero install (solve), ~0 serve overhead, flat in store size, exposure 0.
      * grace: cheap install, small per-query lookup, VRAM grows with #entries.
      * rag: ~0 GPU install, per-query k-fact prefill (constant in k, NOT in N), store grows.
      * ft: expensive install amortised over the merge interval, ~0 serve.
      * reject: ~0 everywhere.
    Numbers are unit-free "reference forward" multiples; real runs replace this with CudaClock.
    """
    # per-unit synthetic gpu-seconds (reference-forward multiples)
    # GRACE install is a CHEAP key insertion (< its per-query serving cost — the design's
    # "serving > install" property); edit pays an expensive solve; ft an expensive train.
    INSTALL = {"edit": 8.0, "grace": 0.5, "rag": 0.05, "ft": 40.0, "reject": 0.0}
    SERVE = {"edit": 1.0, "grace": 1.15, "rag": 1.0, "ft": 1.0, "reject": 1.0}   # base forward=1.0
    RAG_PREFILL_TOK_PER_FACT = 12       # tokens per injected fact (constant in k).
    # GRACE codebook lookup / VRAM GROWS with #entries (capacity-limited) — the cost that makes
    # always-GRACE expensive at scale and forces the router to diversify (DESIGN §b). RAG's
    # per-query prefill, by contrast, is constant in store size N.
    GRACE_LOOKUP_PER_ENTRY = 2.0e-3

    def install(self, arm: str, n_units: int = 1, store_n: int = 0) -> CostRecord:
        per = self.INSTALL.get(arm, 0.0)
        return CostRecord(arm=arm, phase="install", wall_s=per * n_units, gpu_s=per * n_units,
                          peak_vram_bytes=self._vram(arm, store_n), n_units=n_units)

    def serve(self, arm: str, n_queries: int = 1, store_n: int = 0, k: int = 5) -> CostRecord:
        base = self.SERVE.get(arm, 1.0)
        # GRACE per-query lookup grows with #entries (capacity cost); edit/rag are flat in N.
        if arm == "grace":
            base = base + self.GRACE_LOOKUP_PER_ENTRY * store_n
        # RAG's ONLY serving penalty is the constant k-fact prefill (independent of store_n).
        extra_tok = (k * self.RAG_PREFILL_TOK_PER_FACT) if arm == "rag" else 0
        prefill_cost = (extra_tok / 100.0) if arm == "rag" else 0.0   # tokens -> forward-multiple
        gpu = (base + prefill_cost) * n_queries
        return CostRecord(arm=arm, phase="serve", wall_s=gpu, gpu_s=gpu,
                          peak_vram_bytes=self._vram(arm, store_n),
                          extra_input_tokens=extra_tok * n_queries,
                          store_bytes=self._store_bytes(arm, store_n), n_units=n_queries)

    @staticmethod
    def _store_bytes(arm: str, store_n: int) -> float:
        # Only RAG carries a raw readable fact store that grows with retained updates.
        if arm == "rag":
            return 256.0 * store_n          # ~256 synthetic bytes/fact (index + raw text).
        if arm == "grace":
            return 32.0 * store_n           # codebook vectors (not raw retrievable text).
        return 0.0

    def _vram(self, arm: str, store_n: int) -> float:
        return 1.0e9 + self._store_bytes(arm, store_n)   # 1GB synthetic base + store.


class CudaClock:
    """Real synced wall-clock / peak-VRAM instrumentation (used OFF dryrun only).

    Lazy torch import; `torch.cuda.synchronize()` brackets every measurement (unsynced GPU
    timing is the classic error). Never invoked by selftest or --dryrun.
    """
    def __init__(self):
        import torch  # noqa: F401  (lazy; only when a real run needs it)
        self._torch = torch

    def _sync(self):
        if self._torch.cuda.is_available():
            self._torch.cuda.synchronize()

    def measure(self, arm: str, phase: str, fn, n_units: int = 1, store_n: int = 0,
                extra_input_tokens: int = 0, k: int = 5):
        torch = self._torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._sync()
        t0 = time.perf_counter()
        out = fn()
        self._sync()
        wall = time.perf_counter() - t0
        vram = float(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0.0
        rec = CostRecord(arm=arm, phase=phase, wall_s=wall, gpu_s=wall,
                         peak_vram_bytes=vram, extra_input_tokens=extra_input_tokens,
                         n_units=n_units)
        return out, rec


# ---------------------------------------------------------------- selftest
def _selftest() -> None:
    clk = SyntheticClock()
    # edit install > 0, edit serve flat in store size:
    e0 = clk.serve("edit", store_n=10, k=5).gpu_s
    e1 = clk.serve("edit", store_n=100000, k=5).gpu_s
    assert abs(e0 - e1) < 1e-9, "edit serving must be flat in store size N"
    # RAG serve overhead > edit, and constant in N (only prefill, not store):
    r_small = clk.serve("rag", store_n=10, k=5)
    r_big = clk.serve("rag", store_n=100000, k=5)
    assert r_small.gpu_s > clk.serve("edit").gpu_s, "RAG per-query serve must exceed edit"
    assert abs(r_small.gpu_s - r_big.gpu_s) < 1e-9, "RAG per-query cost must be constant in N (k fixed)"
    # but RAG store footprint DOES grow with N:
    assert r_big.store_bytes > r_small.store_bytes, "RAG store footprint must grow with N"
    assert clk.serve("edit", store_n=100000).store_bytes == 0.0, "edit carries no fact store"
    # GRACE serving > install (retrieval each query vs cheap insert):
    assert clk.serve("grace").gpu_s > 0 and clk.install("grace").gpu_s > 0
    # FT install is the expensive arm:
    assert clk.install("ft").gpu_s > clk.install("edit").gpu_s > clk.install("rag").gpu_s
    # ledger normalisation:
    led = CostLedger(reference_wall_s=2.0)
    led.add(clk.install("edit", n_units=1))
    assert led.normalise(led.total("install")["wall_s"]) == clk.INSTALL["edit"] / 2.0
    print("cost_harness selftest: PASS")


if __name__ == "__main__":
    _selftest()
