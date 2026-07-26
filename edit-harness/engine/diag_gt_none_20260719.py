"""One-off diagnostic (systematic-debugging Phase 1): which MIX_A stream updates carry
gt_damage=None, per seed. CPU-only, no GPU touched. Delete after the fix lands."""
import sys
sys.path.insert(0, ".")
from experiments.frame_a.stream_builder import StreamBuilder

b = StreamBuilder(synthetic=False, cf_cell_seed=0)
for seed in (0, 1, 2):
    updates, man = b.build_stream("MIX_A", seed)
    bad = [(i, u.get("fact_type"), u.get("damaging_kind"), u.get("serving_hint"),
            u.get("gt_measured"), u.get("_covered_geom"))
           for i, u in enumerate(updates) if u.get("gt_damage") is None]
    print(f"seed {seed}: n_updates={len(updates)} gt_damage_None={len(bad)}")
    for row in bad[:10]:
        print("   idx", row)
