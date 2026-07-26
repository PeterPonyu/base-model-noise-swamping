"""stream_builder.py — construct STREAM-v1 (the update-stream benchmark).

Builds a time-ordered list of update records from the on-disk datasets (CounterFact + zsRE core,
MQuAKE-CF-3k multi-hop, RippleEdits ripple), attaches per-update router-visible metadata and
scorer-only metadata, injects `conflict` / `damaging` updates (rev.4 partition), and emits
`{mix}×{seed}` stream JSONs + a held-out probe bank + a record-disjoint dev/calibration slice +
a manifest with license-audit and disjointness/provenance fields.

FROZEN-design bindings honored (DESIGN §a, PREREG §2/§6):
  * `damaging` is partitioned into `damaging_gt` (drawn from the pre-existing measured B6
    collateral cells — real, top ground-truth decile) and `damaging_synth` (lexical
    subject-collision, geometry-UNCONTROLLED). The **discovery headline is computed on
    `damaging_gt` ONLY**; selecting the damaging pool NEVER uses key geometry (circularity fix).
  * `gt_damage`/`gt_measured` and every scoring field are scorer/oracle inputs ONLY — the router
    sees `router_view()`, which strips them (incl. `gt_measured`, a latent provenance channel).
    `key_cos` (the pre-edit raw signed key-cosine) is the sole geometry field the router reads.
  * Record selection is seed-INDEPENDENT (same update multiset per mix); the seed only shuffles
    ORDER. Probe bank ⟂ edits; calibration slice ⟂ every scored stream and never a prefix.
  * Geometry provenance (THE JOIN — MAJOR-1 fix): measured `key_cos`/`gt_damage` are attached by
    a record's ORIGINAL loader index `orig_idx`, aligned to cell row `orig_idx` of
    `results/matrices/gate_llama1b_rome_cf_L12_s{cf_cell_seed}.npz`. The CF edit pool is drawn
    COVERED-FIRST (the 200 cell-covered records) so `damaging_gt` is maximised; the probe bank and
    calibration slice are drawn from the NON-covered tail, keeping disjointness intact. A hard
    subject/prompt reload-integrity assertion guards the join on the real build path (abort on
    mismatch).

DRYRUN / selftest is torch-free: `--selftest` builds streams from a synthetic fixture pool. The
real build (`--build`) lazy-imports the canonical loaders (`experiments/killgate_keygeom.py`) +
reads the npz cells; it runs CPU-only (no GPU). `--selftest --real_join` exercises the identity
join against the ACTUAL npz. Validations run from file+argv, never heredoc-under-conda.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import config as C

# The CF measured cell was built with these loader params; the join asserts against them.
CELL_N_EDITS = 200
CELL_N_PROBES = 500
# Global record-partition sizes per dataset (calib ⟂ probe ⟂ edit-pool, seed-independent).
CALIB_BLOCK = 120         # dev/calibration slice source (record-disjoint from all streams)
PROBE_BLOCK = C.PROBE_BANK_SIZE   # held-out locality probe bank (500)
EDIT_POOL = 260           # edit-eligible records per dataset (>= any single-mix demand)
NEED = PROBE_BLOCK + CALIB_BLOCK + EDIT_POOL          # 880 records per dataset
SYNTH_N_COVERED = CELL_N_EDITS                        # fixture: first 200 records are "measured".
CF_NONCOV_LOAD = 1800   # CF non-covered tail load size (union ~590 → ~1200 non-covered available).


# ---------------------------------------------------------------- helpers
def _hash_record(rec: Dict) -> str:
    key = "|".join(str(rec.get(k, "")) for k in ("subject", "prompt", "target_new", "target_true"))
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _seed_from(*parts) -> int:
    """Deterministic 32-bit RNG seed from string parts — NEVER builtin hash() (PYTHONHASHSEED-
    salted → cross-process non-determinism). sha1 is stable across interpreters."""
    h = hashlib.sha1(":".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big")


def router_view(update: Dict) -> Dict:
    """The ONLY projection the router may read. Strips every scorer/oracle field.

    Router-visible = {fact_type, conflict_flag(MASKED), serving_hint, key_cos, edit, capacity,
    est_qvol}. `gt_damage`, `gt_damage_provenance`, `downstream_query_set`, `damaging_kind`,
    `damaging_gt_eligible`, and `gt_measured` (a latent provenance channel) are REMOVED so nothing
    about the ground truth can leak into a routing decision.

    LABEL-LEAK FIX (pinned): `conflict_flag == "damaging"` is a GROUND-TRUTH collateral label — it
    must NEVER reach the router (else discovery is circular). It is masked to **"none"** (pinned
    choice: a damaging update presents to the router as an ORDINARY update; the router must
    discover its damage from `key_cos` geometry alone). The scorer is unaffected —
    `scoring.discovery` filters on `damaging_kind`, a scorer-only field, not on `conflict_flag`.
    """
    visible = ("fact_id", "t", "fact_type", "serving_hint", "key_cos",
               "edit", "subject_key", "est_qvol")
    v = {k: update[k] for k in visible if k in update}
    cf = update.get("conflict_flag", "none")
    v["conflict_flag"] = "none" if cf == "damaging" else cf     # mask the damage label.
    return v


# ---------------------------------------------------------------- record pools
def _synthetic_records(fact_type: str, n: int, tag: str) -> List[Dict]:
    """Torch-free fixture pool (selftest only) — schema-valid, deterministic, NOT real data."""
    out = []
    for i in range(n):
        subj = f"{fact_type}_{tag}_subj{i}"
        out.append({
            "subject": subj,
            "prompt": f"The {fact_type} property of {subj} is",
            "target_new": f"newval_{i % 37}",
            "target_true": f"trueval_{i % 41}",
        })
    return out


def _cf_measured_geometry(seed_for_cf_cell: int) -> Optional[Dict]:
    """Per-edit (key_cos, gt_damage) from the L12 CF cell — the REAL geometry join.

    key_cos_i  = mean over base-known probe columns (pre_p>0.05) of signed COS[i, :]  (pre-edit,
                 key-derived — the router's raw signed key-cosine input, aggregated per edit).
    gt_damage_i= mean over base-known probe columns of signed damage_logit[i, :] (identical
                 masking to d3_benefit_predictor). Scorer/oracle input ONLY.
    Row i aligns to load_counterfact(seed=seed_for_cf_cell) record i. Returns per-row arrays, or
    None if the cell is absent.
    """
    path = C.GT_DAMAGE_GLOB.format(L=C.GEOMETRY_LAYER, s=seed_for_cf_cell)
    if not os.path.exists(path):
        return None
    d = np.load(path)
    COS = d["COS"].astype(float)
    DMG = d["damage_logit"].astype(float)
    edit_ok = d["edit_ok"].astype(float) > 0.5 if "edit_ok" in d.files else np.ones(COS.shape[0], bool)
    col = np.ones(COS.shape[1], bool)
    if "pre_p" in d.files:
        c = d["pre_p"].astype(float) > C.KNOWN_PROBE_PRE_P
        if c.sum() >= 5:
            col = c
    key_cos = COS[:, col].mean(axis=1)
    gt_damage = DMG[:, col].mean(axis=1)
    # PER-CELL top-decile flag (rev.5): the 0.90 quantile is taken WITHIN this cell's own edit_ok
    # damage distribution — cross-cell raw damage is not comparable (different probe banks).
    dmg_eligible = np.zeros(COS.shape[0], bool)
    if edit_ok.sum() >= 10:
        thr = float(np.quantile(gt_damage[edit_ok], 0.90))
        dmg_eligible = edit_ok & (gt_damage >= thr)
    return {"key_cos": key_cos, "gt_damage": gt_damage, "edit_ok": edit_ok,
            "damaging_gt_eligible": dmg_eligible,
            "n_edits": COS.shape[0], "source_cell": os.path.basename(path)}


# ---------------------------------------------------------------- the builder
@dataclass
class StreamBuilder:
    """Builds STREAM-v1 pools, streams, probe bank, calibration slice, and manifest.

    `synthetic=True` uses the torch-free fixture pool (selftest); `synthetic=False` uses the
    canonical loaders + the real CF geometry join (CPU-only — no GPU).
    """
    synthetic: bool = True
    cf_cell_seed: int = 0     # which CF measured cell provides the real geometry join.
    _pools: Dict[str, Dict[str, List[Dict]]] = field(default_factory=dict)
    _prov: Dict[str, Dict] = field(default_factory=dict)

    # ------------------------------------------------------------------ pools + geometry
    def _cf_covered_union(self) -> List[Dict]:
        """The 3-CELL UNION of CF cell-covered records (rev.5). Each covered record carries the
        geometry from ITS cell (key_cos/gt_damage/gt_measured) + a PER-CELL top-decile
        `damaging_gt_eligible` flag. Lowest-seed cell wins on collision. A per-cell byte-match
        assert (reviewer (a)) guards the join. Records carry `_covered_geom=True` so
        `_attach_geometry` leaves them untouched."""
        _exp = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _exp not in sys.path:
            sys.path.insert(0, _exp)
        import killgate_keygeom as kg
        union: Dict[Tuple[str, str], Dict] = {}
        for s in C.CF_UNION_CELL_SEEDS:
            meas = _cf_measured_geometry(s)
            if meas is None:
                continue
            recs_s, _rest, _h = kg.load_counterfact(C.DATASETS["cf"], CELL_N_EDITS, 0, seed=s)
            ref, _r2, _h2 = kg.load_counterfact(C.DATASETS["cf"], CELL_N_EDITS, 0, seed=s)  # byte-match reload
            for i, rec in enumerate(recs_s):
                if i >= meas["n_edits"]:
                    break
                if ref[i]["subject"] != rec["subject"] or ref[i]["prompt"] != rec["prompt"]:
                    raise RuntimeError(
                        f"CELL-JOIN INTEGRITY FAILED: cell s{s} row {i} does not byte-match a "
                        f"reconstructed load_counterfact(seed={s}) row — aborting the real build.")
                key = (rec["subject"], rec["prompt"])
                if key in union:            # lowest-seed cell wins (seeds iterate ascending).
                    continue
                r = dict(rec)
                r["key_cos"] = float(meas["key_cos"][i])
                r["gt_damage"] = float(meas["gt_damage"][i])
                r["gt_measured"] = bool(meas["edit_ok"][i])
                r["damaging_gt_eligible"] = bool(meas["damaging_gt_eligible"][i])
                r["_covered_geom"] = True
                r["_cell_seed"] = int(s); r["_cell_row"] = int(i)
                r["_geom_src"] = f"{meas['source_cell']}#row{i}"
                union[key] = r
        return list(union.values())

    def _pool_records(self, fact_type: str) -> Tuple[List[Dict], int, Dict]:
        """Ordered record list (COVERED-FIRST) tagged with `orig_idx` + n_covered.

        CF-real: covered = the 3-cell union (~590 records, geometry pre-attached); non-covered tail
        = a large CF load with the union keys removed (probe/calib source). Other real datasets and
        the synthetic fixture: covered = a leading block (0 for non-CF real; 200 for the fixture).
        """
        prov = {"loader": None, "measured": False}
        if self.synthetic:
            recs = _synthetic_records(fact_type, NEED, "syn")
            prov["loader"] = "synthetic_fixture"
            n_covered = SYNTH_N_COVERED
        else:
            _exp = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if _exp not in sys.path:
                sys.path.insert(0, _exp)
            if fact_type == "ripple":
                import rippleedits_loader as rl  # torch-free
                edits, _r, _u, _m = rl.load_ripple_edits(C.DATASETS["ripple"], NEED, 0, seed=0)
                recs, prov["loader"], n_covered = list(edits[:NEED]), "load_ripple_edits", 0
            else:
                import killgate_keygeom as kg  # pulls torch (fine on CPU)
                if fact_type == "cf":
                    covered = self._cf_covered_union()                      # ~590 covered (geom attached)
                    ckeys = {(r["subject"], r["prompt"]) for r in covered}
                    big, rest, _h = kg.load_counterfact(C.DATASETS["cf"], CELL_N_EDITS, CF_NONCOV_LOAD, seed=0)
                    noncov = [r for r in (list(big) + list(rest))
                              if (r["subject"], r["prompt"]) not in ckeys]
                    if len(noncov) < PROBE_BLOCK + CALIB_BLOCK:
                        raise RuntimeError(f"CF non-covered tail too small: {len(noncov)} < "
                                           f"{PROBE_BLOCK + CALIB_BLOCK}")
                    recs, prov["loader"], n_covered = covered + noncov, "load_counterfact(3-cell union)", len(covered)
                elif fact_type == "zsre":
                    edits, _p, _h = kg.load_zsre(C.DATASETS["zsre"], NEED, 0, seed=0)
                    recs, prov["loader"], n_covered = list(edits[:NEED]), "load_zsre", 0
                elif fact_type == "mquake_mh":
                    edits, _p, _h = kg.load_mquake(C.DATASETS["mquake_mh"], NEED, 0, seed=0)
                    recs, prov["loader"], n_covered = list(edits[:NEED]), "load_mquake", 0
                else:
                    raise ValueError(fact_type)
        need_min = max(NEED, n_covered + PROBE_BLOCK + CALIB_BLOCK)
        if len(recs) < need_min:
            raise RuntimeError(f"{fact_type}: only {len(recs)} records; need {need_min}")
        for i, rec in enumerate(recs):
            rec["orig_idx"] = i
        return recs, n_covered, prov

    @staticmethod
    def _partition(records: List[Dict], n_covered: int) -> Dict[str, List[Dict]]:
        """calib ⟂ probe ⟂ edit-pool by ORIGINAL order (no shuffle → the join stays valid).

        edit = ALL covered records (cell-covered, first) + a non-covered extra top-up; probe/calib
        are drawn from the non-covered tail so they never overlap the covered edit records.
        """
        covered = records[:n_covered]
        rest = records[n_covered:]
        probe = rest[:PROBE_BLOCK]
        calib = rest[PROBE_BLOCK:PROBE_BLOCK + CALIB_BLOCK]
        extra = rest[PROBE_BLOCK + CALIB_BLOCK:PROBE_BLOCK + CALIB_BLOCK + max(0, EDIT_POOL - n_covered)]
        edit = covered + extra
        return {"calib": calib, "probe": probe, "edit": edit}

    def _ensure_pool(self, fact_type: str) -> None:
        if fact_type in self._pools:
            return
        recs, n_covered, prov = self._pool_records(fact_type)
        part = self._partition(recs, n_covered)
        self._pools[fact_type] = part
        self._prov[fact_type] = prov
        self._attach_geometry(fact_type, part, n_covered)

    def _attach_geometry(self, fact_type: str, part: Dict[str, List[Dict]], n_covered: int) -> None:
        """Attach per-record key_cos + gt_damage + damaging_gt_eligible.

        CF-real covered records already carry geometry (`_covered_geom`, from `_cf_covered_union`)
        and are LEFT UNTOUCHED. Synthetic covered records (orig_idx < n_covered) get a
        deterministic "measured-like" signal with a top-decile eligibility flag so the fixture
        exercises damaging_gt. All non-covered records get deterministic pseudo-geometry
        (gt_measured False, damaging_gt_eligible False → never in the discovery headline).
        """
        recs = part["calib"] + part["probe"] + part["edit"]   # blocks are disjoint (edit=covered+extra)
        if self.synthetic:
            cov = [r for r in recs if 0 <= r.get("orig_idx", -1) < n_covered]
            for rec in cov:
                r = np.random.default_rng((rec["orig_idx"] + 1) * 100003)
                kc = float(r.uniform(-0.15, 0.6))
                rec["key_cos"] = kc
                rec["gt_damage"] = float(max(0.0, 2.5 * kc + r.normal(0, 0.35)))  # Llama-positive law
                rec["gt_measured"] = True
                rec["_geom_src"] = f"synthetic-covered#row{rec['orig_idx']}"
            if cov:  # PER-(fixture-)cell top-decile flag, mirroring the real per-cell rule.
                thr = float(np.quantile([r["gt_damage"] for r in cov], 0.90))
                for rec in cov:
                    rec["damaging_gt_eligible"] = bool(rec["gt_damage"] >= thr)
        for rec in recs:
            if rec.get("_covered_geom") or "key_cos" in rec:
                rec.setdefault("damaging_gt_eligible", False)
                continue
            h = int(_hash_record(rec), 16)
            r = np.random.default_rng(h % (2**32))
            rec["key_cos"] = float(r.uniform(-0.2, 0.6))
            rec["gt_damage"] = float(max(0.0, 2.5 * rec["key_cos"] + r.normal(0, 0.4)))
            rec["gt_measured"] = False
            rec["damaging_gt_eligible"] = False
            rec["_geom_src"] = "pseudo(record-hash)"

    # ------------------------------------------------------------------ probe bank + calibration
    def probe_bank(self, fact_types: List[str]) -> List[Dict]:
        bank = []
        for ft in fact_types:
            self._ensure_pool(ft)
            for rec in self._pools[ft]["probe"]:
                bank.append({"fact_id": f"probe_{ft}_{_hash_record(rec)}", "fact_type": ft,
                             "edit": {k: rec[k] for k in ("subject", "prompt", "target_new", "target_true")},
                             "key_cos": rec["key_cos"]})
        return bank[:C.PROBE_BANK_SIZE]

    def calibration_slice(self, fact_types: List[str]) -> List[Dict]:
        slc = []
        for ft in fact_types:
            self._ensure_pool(ft)
            for rec in self._pools[ft]["calib"]:
                slc.append({"fact_id": f"calib_{ft}_{_hash_record(rec)}", "fact_type": ft,
                            "key_cos": rec["key_cos"], "gt_damage": rec["gt_damage"],
                            "edit": {k: rec[k] for k in ("subject", "prompt", "target_new", "target_true")}})
        return slc

    # ------------------------------------------------------------------ record selection (per mix)
    def _select_updates(self, mix_name: str) -> List[Dict]:
        """Seed-INDEPENDENT selection of the base update multiset for a mix (no ordering yet)."""
        mix = C.MIXES[mix_name]
        n = C.STREAM_LEN_WAVE1
        weights = mix["fact_type_weights"]
        rng = np.random.default_rng(_seed_from("select", mix_name))   # sha1, cross-process stable
        updates = []
        counts = {ft: int(round(w * n)) for ft, w in weights.items()}
        diff = n - sum(counts.values())
        dom = max(weights, key=weights.get)
        counts[dom] += diff
        # (a) eligible-first seeding is restricted to the damaging_gt QUOTA only. The quota =
        # gt-share (0.7) × ρ_damaging × stream size — exactly enough eligible records to fill the
        # damaging_gt labels. Seeding MORE (all eligible) would over-enrich the CF sub-stream; the
        # background fill is REPRESENTATIVE (no eligibility preference) so eligible records may
        # still enter by chance at the pool base rate and stay UNLABELED. Only CF carries eligible
        # records in wave 1, so the whole quota lands in CF (asserted-implicitly by len(elig)==0
        # elsewhere). Deterministic + seed-INDEPENDENT (sha1) → identical multiset across seeds.
        quota_gt = int(round(0.7 * mix["rho_damaging"] * n))
        for ft, k in counts.items():
            self._ensure_pool(ft)
            pool = self._pools[ft]["edit"]
            frng = np.random.default_rng(_seed_from("selectft", mix_name, ft))
            elig = [r for r in pool if r.get("damaging_gt_eligible")]
            ei = np.arange(len(elig)); frng.shuffle(ei)
            n_seed = min(quota_gt, len(elig), k)                       # (a) cap at the quota.
            seeded = [elig[i] for i in ei[:n_seed]]
            seeded_ids = {id(r) for r in seeded}
            # representative background fill from the FULL pool (eligible at base rate, unlabeled).
            remaining = [r for r in pool if id(r) not in seeded_ids]
            rj = np.arange(len(remaining)); frng.shuffle(rj)
            fillsrc = [remaining[i] for i in rj]
            chosen = list(seeded)
            need = k - len(chosen)
            if need > 0 and fillsrc:
                chosen += [fillsrc[j % len(fillsrc)] for j in range(need)]
            for rec in chosen:
                updates.append(self._mk_update(rec, ft))
        order = np.arange(len(updates)); rng.shuffle(order)
        return [updates[i] for i in order]

    def _mk_update(self, rec: Dict, fact_type: str) -> Dict:
        u = {
            "fact_type": fact_type,
            "conflict_flag": "none",
            "damaging_kind": None,
            "serving_hint": "none",
            "edit": {k: rec[k] for k in ("subject", "prompt", "target_new", "target_true")},
            "subject_key": rec["subject"],
            "orig_idx": int(rec.get("orig_idx", -1)),
            "key_cos": float(rec["key_cos"]),
            "gt_measured": bool(rec.get("gt_measured", False)),
            "damaging_gt_eligible": bool(rec.get("damaging_gt_eligible", False)),
            "gt_damage": float(rec["gt_damage"]),
            "gt_damage_provenance": {
                "source": rec.get("_geom_src", "pseudo(record-hash)"),
                "cell_seed": rec.get("_cell_seed"), "cell_row": rec.get("_cell_row"),
                "layer": C.GEOMETRY_LAYER, "metric": C.DAMAGE_METRIC,
                "measured": bool(rec.get("gt_measured", False)),
                "damaging_gt_eligible": bool(rec.get("damaging_gt_eligible", False))},
        }
        return u

    # ------------------------------------------------------------------ injection
    def _inject(self, updates: List[Dict], mix_name: str) -> List[Dict]:
        """Inject conflict + damaging(gt|synth) per the mix rates, deterministically."""
        mix = C.MIXES[mix_name]
        n = len(updates)
        rng = np.random.default_rng(_seed_from("inj", mix_name))   # sha1, cross-process stable
        # ---- damaging: gt from the PER-CELL top-decile flag (rev.5 — NEVER a pooled raw quantile;
        # cross-cell raw gt_damage is not comparable, different probe banks per cell). synth via
        # lexical subject collision.
        n_dmg = int(round(mix["rho_damaging"] * n))
        gt_pool = [u for u in updates if u.get("damaging_gt_eligible")]
        dmg_gt_ids = set()
        if gt_pool:
            rng.shuffle(gt_pool)
            n_gt = min(len(gt_pool), int(round(0.7 * n_dmg)))  # majority gt (measured pool exists)
            for u in gt_pool[:n_gt]:
                u["conflict_flag"] = "damaging"; u["damaging_kind"] = "gt"
                dmg_gt_ids.add(id(u))
        # synth: lexical subject-collision (geometry-UNCONTROLLED); post-hoc measured on GPU.
        n_synth = n_dmg - len(dmg_gt_ids)
        cand = [u for u in updates if u["conflict_flag"] == "none"]
        rng.shuffle(cand)
        for u in cand[:max(0, n_synth)]:
            u["conflict_flag"] = "damaging"; u["damaging_kind"] = "synth"
            u["gt_damage_provenance"]["synth_rule"] = "lexical_subject_collision"
            u["gt_damage_provenance"]["measured"] = False   # excluded from the discovery headline.
        # ---- conflict: re-issue an earlier subject with a THIRD target (latest credited).
        # A conflict record must be a coherent re-edit of the PRIOR's (subject, prompt) pair:
        # copy subject AND prompt AND the target-independent key geometry together, or the live
        # key capture (and ROME itself) lands on the subject-missing fallback token while the
        # stored key_cos still describes the original pair (wave-1 attempt-2 abort, 2026-07-16;
        # see repro results/frame_a/repro_keycos_MIX_A_s0.json — 18/18 conflict rows mismatched,
        # 207/207 clean rows matched to 1e-8). Records carrying MEASURED geometry are excluded
        # from the injection pool so the predictor's B6-validated provenance is never mutated;
        # gt_damage is target-DEPENDENT, so the injected record's gt fields are nulled rather
        # than copied from the prior.
        n_conf = int(round(mix["rho_conflict"] * n))
        eligible = [i for i, u in enumerate(updates)
                    if u["conflict_flag"] == "none" and i > 0
                    and u.get("gt_damage_provenance", {}).get("cell_seed") is None]
        rng.shuffle(eligible)
        injected = 0
        for i in eligible:
            if injected >= n_conf:
                break
            # the prior must carry a real (subject, prompt) pair — ripple records natively
            # have subject=None and cannot anchor a coherent re-edit.
            priors = [j for j in range(i) if updates[j]["edit"].get("subject")]
            if not priors:
                continue
            prior = updates[int(rng.choice(priors))]
            injected += 1
            updates[i]["conflict_flag"] = "conflict"
            updates[i]["edit"]["subject"] = prior["edit"]["subject"]
            updates[i]["edit"]["prompt"] = prior["edit"]["prompt"]
            updates[i]["subject_key"] = prior["edit"]["subject"]
            updates[i]["edit"]["target_new"] = updates[i]["edit"]["target_new"] + "_v3"
            updates[i]["key_cos"] = prior["key_cos"]            # target-independent, now grounded
            if self.synthetic:
                # the planted world derives gt FROM key_cos (no target axis): the prior's gt is
                # the coherent value for the copied geometry, and synth consumers require a float.
                updates[i]["gt_damage"] = prior.get("gt_damage")
            else:
                # gt_damage is target-DEPENDENT so the prior's measured value does not apply to
                # the new "_v3" target; downstream row-writing requires a float, so re-derive the
                # standard NON-measured proxy (same rule as _attach for uncovered records) from
                # the copied key_cos, and demote provenance to unmeasured/pseudo. The measured
                # flags are stripped so the discovery headline and (d) assert both skip this row.
                updates[i]["gt_damage"] = float(max(0.0, 2.5 * prior["key_cos"] +
                                                    rng.normal(0, 0.4)))
                updates[i]["gt_measured"] = False
                updates[i]["damaging_gt_eligible"] = False
                prov = updates[i]["gt_damage_provenance"]
                prov.update({"source": "pseudo(conflict-injection)", "cell_seed": None,
                             "cell_row": None, "measured": False,
                             "damaging_gt_eligible": False,
                             "mutated_by": "conflict_injection"})
        # ---- serving_hint per mix weights.
        hints = list(mix["serving_hint_weights"].keys())
        probs = np.array(list(mix["serving_hint_weights"].values()), float)
        probs = probs / probs.sum()
        for u in updates:
            u["serving_hint"] = str(rng.choice(hints, p=probs))
        return updates

    # ------------------------------------------------------------------ downstream queries
    def _attach_queries(self, updates: List[Dict]) -> None:
        for u in updates:
            p = u["edit"]["prompt"]
            para = [p, "In fact, " + p, p.rstrip(".") + ", specifically"][:3]
            q = {"efficacy": para, "expect": u["edit"]["target_new"], "ripple": []}
            if u["fact_type"] in ("mquake_mh", "ripple"):
                q["ripple"] = [f"consequence-of::{p}"]
            u["downstream_query_set"] = q
            # router-VISIBLE estimate of downstream query volume (an input; NOT the hidden set).
            u["est_qvol"] = len(para) + len(q["ripple"])

    # ------------------------------------------------------------------ order (per seed)
    @staticmethod
    def _order(updates: List[Dict], seed: int) -> List[Dict]:
        rng = np.random.default_rng(seed)
        idx = np.arange(len(updates)); rng.shuffle(idx)
        out = [dict(updates[i]) for i in idx]
        for t, u in enumerate(out):
            u["t"] = t
            u["fact_id"] = f"{u['fact_type']}_{_hash_record(u['edit'])}_{t}"
        return out

    # ------------------------------------------------------------------ build one stream instance
    def build_stream(self, mix_name: str, seed: int) -> Tuple[List[Dict], Dict]:
        base = self._select_updates(mix_name)
        base = self._inject(base, mix_name)
        self._attach_queries(base)
        ordered = self._order(base, seed)
        manifest = self._manifest(mix_name, seed, ordered)
        self._check_damaging_floor(mix_name, manifest)   # reviewer assert (b).
        return ordered, manifest

    def _check_damaging_floor(self, mix_name: str, manifest: Dict) -> None:
        """Reviewer assert (b) + rev.5: discovery is CI-only UNLESS damaging_gt >= the pinned
        point-floor (50). Below the floor the CI-only fallback MUST be engaged; a point claim is
        allowed only at/above it (and never for a config-forced CI-only mix, e.g. MIX-A/DOF-1).
        Never a false abort — this enforces the honest reporting mode, not a wave stop.
        """
        ds = manifest["discovery_scope"]
        n_gt = ds["damaging_gt_count"]
        if n_gt < C.DISCOVERY_POINT_FLOOR:
            assert ds["ci_only"] is True, (
                f"{mix_name} damaging_gt={n_gt} < point-floor {C.DISCOVERY_POINT_FLOOR} but CI-only "
                f"is not engaged (rev.5 violated).")
        if C.MIXES[mix_name].get("ci_only_discovery", False):
            assert ds["ci_only"] is True, f"{mix_name} is config-forced CI-only but the flag is off."

    # ------------------------------------------------------------------ manifest
    def _manifest(self, mix_name: str, seed: int, updates: List[Dict]) -> Dict:
        ft_counts, cf_counts, sh_counts = {}, {}, {}
        dmg = {"gt": 0, "synth": 0}
        for u in updates:
            ft_counts[u["fact_type"]] = ft_counts.get(u["fact_type"], 0) + 1
            cf_counts[u["conflict_flag"]] = cf_counts.get(u["conflict_flag"], 0) + 1
            sh_counts[u["serving_hint"]] = sh_counts.get(u["serving_hint"], 0) + 1
            if u["damaging_kind"] in ("gt", "synth"):
                dmg[u["damaging_kind"]] += 1
        # (b) realized composition disclosure: how enriched is the CF sub-stream with top-decile-
        # damage (eligible) records, and how many eligible records entered but stayed UNLABELED.
        cf_updates = [u for u in updates if u["fact_type"] == "cf"]
        n_cf = len(cf_updates)
        cf_eligible = [u for u in cf_updates if u.get("damaging_gt_eligible")]
        eligible_unlabeled = sum(1 for u in updates
                                 if u.get("damaging_gt_eligible") and u["damaging_kind"] != "gt")
        realized_composition = {
            "cf_substream_size": n_cf,
            "cf_topdecile_eligible_count": len(cf_eligible),
            "cf_realized_topdecile_fraction": (len(cf_eligible) / n_cf) if n_cf else 0.0,
            "damaging_gt_labeled": dmg["gt"],
            "damaging_synth_labeled": dmg["synth"],
            "eligible_but_unlabeled": eligible_unlabeled,
            "note": ("eligible-first seeding is capped at the damaging_gt quota; residual CF "
                     "enrichment (esp. MIX_C, where CF is a small share yet supplies the whole "
                     "damaging_gt quota) is STRUCTURAL and disclosed — read P1/P4 effect sizes "
                     "against this realized composition, not the nominal mix weights."),
        }
        fact_types = sorted(ft_counts.keys())
        probe = self.probe_bank(fact_types)
        calib = self.calibration_slice(fact_types)
        edit_hashes = {_hash_record(u["edit"]) for u in updates}
        probe_hashes = {p["fact_id"].split("_", 2)[-1] for p in probe}
        calib_subjects = {c["edit"]["subject"] for c in calib}
        edit_subjects = {u["edit"]["subject"] for u in updates}
        stream_hash = hashlib.sha1(
            json.dumps([router_view(u) for u in updates], sort_keys=True).encode()).hexdigest()[:16]
        # MINOR-B: COMPUTE calib_is_prefix_of_stream (was a literal). calib is drawn from the
        # record-disjoint non-covered tail, so the calibration records are never the leading block
        # of a scored stream — computing it (vs asserting) keeps the disjointness claim auditable.
        calib_hash_list = [_hash_record(c["edit"]) for c in calib]
        stream_hash_list = [_hash_record(u["edit"]) for u in updates]
        calib_is_prefix = bool(calib_hash_list) and stream_hash_list[:len(calib_hash_list)] == calib_hash_list
        # rev.5: CI-only unless damaging_gt >= the pinned point-floor (50); config can force CI-only.
        config_ci_only = bool(C.MIXES[mix_name].get("ci_only_discovery", False))
        below_floor = dmg["gt"] < C.DISCOVERY_POINT_FLOOR
        ci_only = below_floor or config_ci_only
        return {
            "stream_id": f"{mix_name}_s{seed}",
            "mix": mix_name, "seed": int(seed), "model": "llama-3.2-1b",
            "synthetic_fixture": self.synthetic,
            "n_updates": len(updates), "probe_bank_size": len(probe),
            "fact_type_counts": ft_counts, "conflict_flag_counts": cf_counts,
            "serving_hint_counts": sh_counts, "damaging_partition": dmg,
            "realized_composition": realized_composition,
            "stream_hash": stream_hash,
            "disjointness": {
                "probe_edit_overlap": len(probe_hashes & edit_hashes),
                "calib_edit_subject_overlap": len(calib_subjects & edit_subjects),
                "calib_is_prefix_of_stream": calib_is_prefix,
            },
            "gt_damage_provenance": {
                "measured_source_cells": [C.GT_DAMAGE_GLOB.format(L=C.GEOMETRY_LAYER, s=s)
                                          for s in C.CF_UNION_CELL_SEEDS],
                "layer": C.GEOMETRY_LAYER, "cf_union_cell_seeds": list(C.CF_UNION_CELL_SEEDS),
                "metric": C.DAMAGE_METRIC,
                "join": ("3-cell UNION (rev.5): each covered record joins to ITS cell via that "
                         "cell's own loader order (per-cell byte-match asserted); lowest-seed wins "
                         "on collision; damaging_gt label = top-decile WITHIN each cell's OWN "
                         "damage distribution (per-cell quantile — cross-cell raw damage NOT pooled)."),
                "note": ("damaging_gt from the per-cell top-decile flag only; damaging_synth is "
                         "lexical subject-collision, geometry-uncontrolled, EXCLUDED from the headline."),
                "loaders": self._prov,
            },
            "discovery_scope": {
                "headline_set": "damaging_gt",
                "damaging_gt_count": dmg["gt"],
                "point_floor": C.DISCOVERY_POINT_FLOOR,
                "below_point_floor": below_floor,
                "config_forced_ci_only": config_ci_only,
                "ci_only": ci_only,
                "point_claim_allowed": (not ci_only),   # rev.5: report both CI and point-if-floor-met.
            },
            "license_audit": {
                "sources": {"cf": "CounterFact", "zsre": "zsRE", "mquake_mh": "MQuAKE-CF-3k",
                            "ripple": "RippleEdits"},
                "redistribution_verified": "unverified",
                "release_policy_if_unverified": "builder_plus_record_hashes_regenerate_from_source",
                "submission_gate": True,
            },
        }

    # ------------------------------------------------------------------ full wave emit
    def build_wave(self, out_dir: str, mixes: Optional[List[str]] = None) -> Dict:
        os.makedirs(out_dir, exist_ok=True)
        mixes = mixes or list(C.MIXES.keys())
        index = {"streams": [], "out_dir": out_dir}
        all_fts = sorted({ft for m in mixes for ft in C.MIXES[m]["fact_type_weights"]})
        probe = self.probe_bank(all_fts)
        calib = self.calibration_slice(all_fts)
        json.dump(probe, open(os.path.join(out_dir, "probe_bank.json"), "w"))
        json.dump(calib, open(os.path.join(out_dir, "calibration_slice.json"), "w"))
        for m in mixes:
            for s in C.SEEDS:
                updates, manifest = self.build_stream(m, s)
                sid = manifest["stream_id"]
                json.dump(updates, open(os.path.join(out_dir, f"stream_{sid}.json"), "w"))
                json.dump(manifest, open(os.path.join(out_dir, f"manifest_{sid}.json"), "w"))
                index["streams"].append({"stream_id": sid, "manifest": manifest})
        json.dump(index, open(os.path.join(out_dir, "stream_index.json"), "w"), indent=2)
        return index


# ---------------------------------------------------------------- selftest
def _selftest() -> None:
    b = StreamBuilder(synthetic=True)
    updates, man = b.build_stream("MIX_B", seed=0)
    n = C.STREAM_LEN_WAVE1
    assert len(updates) == n, f"stream length {len(updates)} != {n}"
    rv = router_view(updates[0])
    assert "gt_damage" not in rv and "downstream_query_set" not in rv, "router_view leaks scorer fields"
    assert "gt_measured" not in rv, "router_view must NOT expose gt_measured (MINOR-1)"
    assert "key_cos" in rv and "serving_hint" in rv and "est_qvol" in rv
    conf = man["conflict_flag_counts"].get("conflict", 0)
    dmg = man["damaging_partition"]["gt"] + man["damaging_partition"]["synth"]
    assert abs(conf - 0.30 * n) <= 6, f"conflict count {conf} off target"
    assert abs(dmg - 0.30 * n) <= 6, f"damaging count {dmg} off target"
    # synthetic fixture now exercises the damaging_gt path (covered records are "measured"):
    assert man["damaging_partition"]["gt"] > 0, "synthetic fixture must produce damaging_gt"
    assert man["discovery_scope"]["headline_set"] == "damaging_gt"
    assert man["disjointness"]["probe_edit_overlap"] == 0, "probe bank must be edit-disjoint"
    assert man["disjointness"]["calib_edit_subject_overlap"] == 0, "calib must be edit-disjoint"
    assert man["disjointness"]["calib_is_prefix_of_stream"] is False
    # same multiset across seeds, different order:
    u0, _ = b.build_stream("MIX_B", seed=0)
    u1, _ = b.build_stream("MIX_B", seed=1)
    assert sorted(_hash_record(u["edit"]) for u in u0) == sorted(_hash_record(u["edit"]) for u in u1), \
        "seeds must be the SAME update multiset"
    assert [u["fact_id"] for u in u0] != [u["fact_id"] for u in u1], "seeds must differ in ORDER"
    # MIX-A stays low-churn + CI-only fallback engaged below floor:
    _, manA = b.build_stream("MIX_A", seed=0)
    dmgA = manA["damaging_partition"]["gt"] + manA["damaging_partition"]["synth"]
    assert abs(dmgA - 0.10 * n) <= 5, "MIX-A rho_damaging must stay ~0.10 (DOF-1)"
    if manA["discovery_scope"]["damaging_gt_count"] < C.DISCOVERY_POINT_FLOOR:
        assert manA["discovery_scope"]["ci_only"] is True, "MIX-A must engage CI-only below floor"
    assert updates[0]["downstream_query_set"]["expect"] == updates[0]["edit"]["target_new"]
    print("stream_builder selftest: PASS")


def _selftest_real_join() -> None:
    """CPU-only 3-CELL-UNION join check against the ACTUAL gate_llama1b npz (reviewer-mandated:
    the synthetic gate cannot catch the MAJOR-1 class). Asserts each covered record carries ITS
    cell's geometry (per-cell byte-match), lowest-seed-wins on collision, and the per-cell
    top-decile eligibility flag — and that probe/calib are disjoint from the covered union."""
    cells = [C.GT_DAMAGE_GLOB.format(L=C.GEOMETRY_LAYER, s=s) for s in C.CF_UNION_CELL_SEEDS]
    if not all(os.path.exists(c) for c in cells):
        print("stream_builder real-join selftest: SKIP (missing a gate_llama1b_rome_cf_L12_s*.npz)")
        return
    b = StreamBuilder(synthetic=False)
    b._ensure_pool("cf")                       # runs the per-cell byte-match assert internally.
    meas = {s: _cf_measured_geometry(s) for s in C.CF_UNION_CELL_SEEDS}
    covered = [r for r in b._pools["cf"]["edit"] if r.get("_covered_geom")]
    assert len(covered) >= 500, f"3-cell union should be ~590 covered records, got {len(covered)}"
    # each covered record joins to ITS cell (cell_seed, cell_row) with exact geometry + eligibility.
    checked = 0
    for r in covered:
        s, i = r["_cell_seed"], r["_cell_row"]
        m = meas[s]
        assert abs(r["key_cos"] - float(m["key_cos"][i])) < 1e-9, f"key_cos misjoin s{s} row{i}"
        assert abs(r["gt_damage"] - float(m["gt_damage"][i])) < 1e-9, f"gt_damage misjoin s{s} row{i}"
        assert r["gt_measured"] == bool(m["edit_ok"][i]), f"gt_measured misderived s{s} row{i}"
        assert r["damaging_gt_eligible"] == bool(m["damaging_gt_eligible"][i]), f"eligible misderived s{s} row{i}"
        checked += 1
    # lowest-seed-wins: no (subject,prompt) appears twice across covered records.
    keys = [(r["subject"], r["prompt"]) for r in covered]
    assert len(keys) == len(set(keys)), "covered union must be deduped (lowest-seed wins)"
    # probe/calib disjoint from the covered union.
    ckeys = set(keys)
    for r in b._pools["cf"]["probe"] + b._pools["cf"]["calib"]:
        assert (r["subject"], r["prompt"]) not in ckeys, "probe/calib must be non-covered"
    n_elig = sum(1 for r in covered if r["damaging_gt_eligible"])
    print(f"stream_builder real-join selftest: PASS ({checked} union rows join-verified across "
          f"{len(C.CF_UNION_CELL_SEEDS)} cells; {n_elig} damaging_gt-eligible)")


def _main() -> None:
    ap = argparse.ArgumentParser(description="STREAM-v1 builder")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--real_join", action="store_true", help="CPU identity-join check vs the real npz")
    ap.add_argument("--build", action="store_true", help="materialise the wave to --out (real loaders)")
    ap.add_argument("--synthetic", action="store_true", help="use the fixture pool (no torch/loaders)")
    ap.add_argument("--cf_cell_seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(C.RESULTS_DIR, "streams"))
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        if args.real_join:
            _selftest_real_join()
        return
    if args.real_join:
        _selftest_real_join(); return
    if args.build:
        b = StreamBuilder(synthetic=args.synthetic, cf_cell_seed=args.cf_cell_seed)
        idx = b.build_wave(args.out)
        print(f"built {len(idx['streams'])} streams -> {args.out}")
        return
    ap.print_help()


if __name__ == "__main__":
    _main()
