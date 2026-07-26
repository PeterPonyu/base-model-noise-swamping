"""CPU wiring test for the real FT arm pending-queue fix (2026-07-19).
Verifies: install mirrors into state.ft_pending; replay-style K-gate would fire;
flush clears BOTH queues, marks answerable, charges >0 units; VRAM-defer retains both;
replay-style gate fires at the K-th install (MINOR-3 regression coverage added 2026-07-20);
multi-defer cycle retains the schedule correctly (NIT-1 added 2026-07-20);
empty-payload flush is a no-op (NIT-3 added 2026-07-20);
counter advances only on success (MINOR-1 added 2026-07-20);
A_cum forgets ~1/7 per batch (MINOR-2 added 2026-07-20).

Monkeypatches _train_and_merge (no peft/model needed) — tests WIRING, not training.
"""
import sys
sys.path.insert(0, ".")
from experiments.frame_a.arms.real_backends import RealFtLoraMergeArm
from experiments.frame_a.arms.base import ModelState
import experiments.frame_a.config as C

arm = RealFtLoraMergeArm(model=None, tokenizer=None, device="cpu")
trained = []
arm._train_and_merge = lambda pending, lam: trained.append((len(pending), lam))

state = ModelState()
K = C.FT_MERGE_INTERVAL_K
# 1) install mirrors into the shared queue
for i in range(K):
    u = {"fact_id": f"f{i}", "edit": {"prompt": f"p{i}", "target_new": "x"}}
    out = arm.install(u, state)
    assert out.deferred and not out.applied_fact
assert len(state.ft_pending) == K, f"shared queue not mirrored: {len(state.ft_pending)}"
assert len(arm._pending) == K
print(f"OK  install mirrors: state.ft_pending={len(state.ft_pending)} arm._pending={len(arm._pending)}")

# 2) replay-style gate now fires; flush trains once, clears BOTH queues, marks answerable
assert len(state.ft_pending) >= K, "replay K-gate would not fire"
rec = arm.flush(state)
assert trained and trained[0][0] == K, f"flush trained on {trained}"
assert abs(trained[0][1] - 1.0) < 1e-9, f"first merge lam should be 1.0, got {trained[0][1]}"
assert len(state.ft_pending) == 0 and len(arm._pending) == 0, "queues not cleared"
assert all(state.answerable[f"f{i}"] == "ft" for i in range(K)), "answerable not marked"
assert rec.n_units == K, f"flush cost units {rec.n_units} != {K}"
print(f"OK  flush: trained on {trained[0][0]} facts (lam={trained[0][1]:.3f}), both queues cleared, answerable marked, n_units={rec.n_units}")

# 2b) MINOR-2: A_cum semantic parity — every 7th fact in the just-flushed batch must be in
#     state.forgotten so A_cum measures realistic merge-overwrite (not artifacting to ~1.0).
forgotten_in_batch = sorted([u for i, u in enumerate([f"f{i}" for i in range(K)]) if (i % 7) == 6])
assert set(forgotten_in_batch) <= state.forgotten, (
    f"missing forgotten: expected {forgotten_in_batch}, got {sorted(state.forgotten)}")
print(f"OK  A_cum forgetting (1-in-7): {len(forgotten_in_batch)} facts forgotten in batch")

# 3) VRAM-defer retains BOTH queues (replay retries next update)
for i in range(3):
    arm.install({"fact_id": f"g{i}", "edit": {"prompt": "p", "target_new": "x"}}, state)
arm._vram_guard = staticmethod(lambda *a, **k: False)
rec2 = arm.flush(state)
assert rec2.n_units == 0, "defer should charge 0 units"
assert len(state.ft_pending) == 3 and len(arm._pending) == 3, "defer must retain both queues"
print("OK  VRAM-defer: both queues retained, 0 units charged")

# 3b) NIT-1: multi-defer cycle — install 3, defer, install 3 more (now 6), defer again,
#     flip _vram_guard back to True, flush, assert all 6 are answerable and λ = 1/√(1) = 1.0.
arm._vram_guard = staticmethod(lambda *a, **k: False)        # ensure defer
for i in range(3):
    arm.install({"fact_id": f"h{i}", "edit": {"prompt": "p", "target_new": "x"}}, state)
rec3a = arm.flush(state)
assert rec3a.n_units == 0 and len(state.ft_pending) == 6
for i in range(3, 6):
    arm.install({"fact_id": f"h{i}", "edit": {"prompt": "p", "target_new": "x"}}, state)
rec3b = arm.flush(state)
assert rec3b.n_units == 0 and len(state.ft_pending) == 9
# Now allow; one merge should consume all 9.
arm._vram_guard = staticmethod(lambda *a, **k: True)
n_before = len(trained)
rec3c = arm.flush(state)
assert rec3c.n_units == 9, f"final flush should consume 9, got {rec3c.n_units}"
assert len(trained) == n_before + 1
assert trained[-1][0] == 9
# After a single merge, _n_merges advances from 1 (post first flush in step 2) to 2 here.
# λ used by this flush was 1/√(_n_merges+1) AT ENTRY = 1/√2 ≈ 0.7071 — NOT bumped on the defer
# rounds, which is the whole point of MINOR-1.
assert abs(trained[-1][1] - 1.0 / (2 ** 0.5)) < 1e-9, (
    f"second-flush lam should be 1/√2 (defer rounds must not advance the counter), got {trained[-1][1]}")
assert all(state.answerable[f"h{i}"] == "ft" for i in range(6)), "all 6 deferred facts should be answerable after final flush"
print(f"OK  multi-defer cycle: 2 defers + 1 success, λ correctly stayed at 1/√2 (counter NOT bumped on defer)")

# 4) NIT-3: empty-payload flush must NOT run training
trained_len_before = len(trained)
arm._vram_guard = staticmethod(lambda *a, **k: True)
rec4 = arm.flush(state)
assert rec4.n_units == 0
assert len(trained) == trained_len_before, "empty-payload flush must not train"
# And the previously-stale shared mirror (from earlier rounds) must be cleared too.
assert len(state.ft_pending) == 0, "empty-payload flush must clear stale ft_pending"
print("OK  empty-payload flush: no training, shared queue cleared (defense against mirror divergence)")

# 5) MINOR-1: counter advance only on success — simulate a training exception and verify
#     _n_merges does NOT advance.
arm._vram_guard = staticmethod(lambda *a, **k: True)
for i in range(K):
    arm.install({"fact_id": f"e{i}", "edit": {"prompt": "p", "target_new": "x"}}, state)
n_merges_before = arm._n_merges
arm._train_and_merge = lambda pending, lam: (_ for _ in ()).throw(RuntimeError("simulated OOM"))
try:
    arm.flush(state)
except RuntimeError:
    pass
assert arm._n_merges == n_merges_before, (
    f"_n_merges must NOT advance on failure (was {n_merges_before}, now {arm._n_merges})")
assert len(state.ft_pending) == K and len(arm._pending) == K, "pending must be retained on failure"
print(f"OK  counter-stable on failure: _n_merges stayed at {arm._n_merges} after simulated exception")

# 6) Cost-reporting hook (the 2026-07-20 root-cause fix): after a successful flush, the arm must
#    expose rec.gpu_s on self._last_flush_rec_gpu so the replay loop can sum it into the
#    routing dict. Without this hook, every FT cell's install_gpu_s stayed at 0.0 even when
#    the merge actually ran — the exact dead-arm signature.
assert hasattr(arm, "_last_flush_rec_gpu"), "arm missing cost-reporting hook attribute"
assert arm._last_flush_rec_gpu > 0.0, (
    f"flush must publish a positive rec.gpu_s; got {arm._last_flush_rec_gpu}")
print(f"OK  cost-reporting hook: _last_flush_rec_gpu={arm._last_flush_rec_gpu:.3f}s (replay can pick it up)")

print("ALL WIRING CHECKS PASSED")
