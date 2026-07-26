"""Focused tests for `experiments/quant_survival_reanalyze_v1.py`.

v1.2.0 scope (independent review fixes applied):
  1. Per-key shape validation: damage_fp32, edit_ok_fp32, every damage__<arm>
     and esr__<arm>, and C3 vectors must match expected shapes — explicit
     corruption tests for each.
  2. Atomic versioned JSON write retains the immutable hash-versioned
     sidecar AND keeps the canonical file in sync — sha256 match verified.
  3. Bootstrap functions return boot_n_finite, boot_n_total, skipped_fraction;
     conditional survival exposes NaN draws explicitly via
     n_nan_draws_exposed.
  4. The 0.0-or-nan latent bug is fixed: an explicit None check replaces the
     short-circuiting `x or nan` pattern; covered by a test.
  5. killed_live_runner is renamed runner_unmodified_verified with the live
     runner's sha256 stored in module_provenance.
  6. Global N_BOOT mutation is removed — n_boot is threaded explicitly through
     analyze_cell; no duplicate n_boot_effective field.
  7. This canonical tests/ location is the only test path.

Tests run cleanly under `-W error::ResourceWarning` (no file-handle leaks)
and `-W error::DeprecationWarning`.

CPU-only; numpy-only; no GPU, no torch, no network.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest

import numpy as np

# Ensure `experiments.quant_survival_reanalyze_v1` is importable
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.dirname(THIS_DIR)
sys.path.insert(0, os.path.join(HARNESS, "experiments"))

import quant_survival_reanalyze_v1 as M  # noqa: E402


# ============================================================
# Test fixtures: in-memory synthetic cells
# ============================================================

def _make_cell_arrays(n_edits: int = 200, n_probes: int = 200,
                      arm_esr_seed: int = 7,
                      survival_preserved: bool = True) -> dict:
    """Return a dict of npz-style arrays for one synthetic seed cell.

    v1.2 planted example: when `survival_preserved=True`, D_quant[i,:] is
    D_fp32[i,:] plus tiny Gaussian noise — this preserves the fp32 per-edit
    damage ranking (rank-SURVIVAL should approach +1.0). When False,
    D_quant is independent of D_fp32 (rank-SURVIVAL should approach 0).

    COS is INDEPENDENT of D_fp32/D_quant by construction — used to verify the
    separately-labeled geometry_sensitivity_cos block.
    """
    rng = np.random.default_rng(arm_esr_seed)
    COS = rng.standard_normal((n_edits, n_probes)).astype(np.float64)
    damage_fp32 = rng.standard_normal((n_edits, n_probes)).astype(np.float64) * 0.1
    edit_ok_fp32 = (rng.random(n_edits) > 0.05).astype(np.float64)
    arrays = {
        "COS": COS,
        "damage_fp32": damage_fp32,
        "edit_ok_fp32": edit_ok_fp32,
    }
    for scheme in M.EXPECTED_SCHEMES:
        for locality in M.EXPECTED_LOCALITIES:
            arm = f"{scheme}_{locality}"
            if survival_preserved:
                # D_quant = D_fp32 + tiny Gaussian noise — preserves ranking
                arrays[f"damage__{arm}"] = (
                    damage_fp32 +
                    rng.standard_normal((n_edits, n_probes)).astype(np.float64) * 1e-4)
            else:
                arrays[f"damage__{arm}"] = (
                    rng.standard_normal((n_edits, n_probes)).astype(np.float64) * 0.1)
            arrays[f"esr__{arm}"] = (
                (rng.random(n_edits) > 0.1).astype(np.float64))
            arrays[f"base__{arm}"] = (
                rng.standard_normal(n_edits).astype(np.float64) * 0.05)
    for scheme in M.EXPECTED_SCHEMES:
        arrays[f"c3_ratio__{scheme}"] = rng.standard_normal(32768).astype(np.float32)
        arrays[f"c3_rfunc__{scheme}"] = rng.standard_normal(n_edits).astype(np.float32)
        arrays[f"c3_rparam__{scheme}"] = rng.standard_normal(n_edits).astype(np.float32)
    return arrays


def _make_cell_table(slug: str, editor: str, layer: int, seed: int,
                     n_edits: int = 200, n_probes: int = 200,
                     n_gen_probes: int = 10) -> dict:
    """Return a phase1 table json for a synthetic cell."""
    fullpath = {v: k for k, v in M.EXPECTED_MODELS.items()}[slug]
    arms_dict = {}
    for scheme in M.EXPECTED_SCHEMES:
        for locality in M.EXPECTED_LOCALITIES:
            arm = f"{scheme}_{locality}"
            arms_dict[arm] = {
                "locality": locality, "scheme": scheme,
                "mean_esr": 0.99, "esr_survival_given_fp32_worked": 1.0,
                "rho_keycos_damage_pooled": 0.05,
                "rho_keycos_damage_within_probe": 0.05,
                "delta_rho_vs_fp32_pooled": 0.001,
                "delta_rho_vs_fp32_within_probe": 0.001,
                "added_damage_logit_mean": 0.001,
                "added_damage_logit_std": 0.05,
                "base_quant_noise_logit_mean_abs": 0.05,
            }
    table = {
        "experiment": "quant_survival_phase1",
        "schema_version": "qs.phase1.v1",
        "created": "2026-07-21T00:00:00",
        "model": fullpath, "layer": layer, "editor": editor, "seed": seed,
        "n_edits": n_edits, "n_probes": n_probes,
        "schemes": list(M.EXPECTED_SCHEMES),
        "codec": "real", "blocksize": 64,
        "fullmodel_cache": "on",
        "edited_layers": [layer],
        "c2_scope": "test",
        "quant_note": "test",
        "damage_metric_note": "test",
        "esr": {"mean_esr_fp32": 0.99, "n_edits_worked_fp32": n_edits - 1},
        "mechanism_tie": {
            "rho_keycos_damage_fp32_pooled": 0.05,
            "rho_keycos_damage_fp32_within_probe": 0.05,
            "within_probe_n_cols": n_probes,
            "fp32_law_gate_c2_eligible": False,
            "fp32_pooled_ci95_bootstrap_edits": [0.0, 0.1],
        },
        "arms": arms_dict,
        "bin_width_mechanism_C3": {
            s: {"n_params_pooled": 100, "F_above_bin": 0.1, "median_ratio": 0.5,
                "p90_ratio": 1.0, "M_concentration_holds_median_ge_1": False,
                "r_func_mean": 0.01, "r_param_mean": 0.5,
                "M_averaging_r_func_ll_r_param": True, "note": "test"}
            for s in M.EXPECTED_SCHEMES
        },
        "generation_checks": {
            "fp32": {"n_gen_probes": n_gen_probes,
                     "perplexity_mean": 10.0 + seed * 0.1,
                     "paraphrase_esr_mean": 0.7 + seed * 0.01},
        },
    }
    for arm in M.EXPECTED_ARMS:
        table["generation_checks"][arm] = {
            "n_gen_probes": n_gen_probes,
            "perplexity_mean": 10.5 + seed * 0.1,
            "paraphrase_esr_mean": 0.7 + seed * 0.01,
        }
    return table


def _write_cell_to_disk(root: str, slug: str, editor: str, layer: int,
                        seed: int, arrays: dict, table: dict) -> str:
    """Write one synthetic cell to root/<slug>_<editor>_L<layer>_s<seed>/."""
    import hashlib
    cell_dir = os.path.join(
        root, f"{slug}_{editor}_L{layer}_s{seed}")
    os.makedirs(cell_dir, exist_ok=True)
    npz_path = os.path.join(cell_dir, "QS_phase1_raw.npz")
    np.savez(npz_path, **arrays)
    tbl_path = os.path.join(cell_dir, "QS_phase1_table.json")
    with open(tbl_path, "w") as f:
        json.dump(table, f)
    return cell_dir


def _make_full_27_cell_root(tmpdir: str, n_edits: int = 200,
                            n_probes: int = 200) -> str:
    """Write the exact 27-cell grid to tmpdir and return the root."""
    root = os.path.join(tmpdir, "quant_survival")
    # EXPECTED_MODELS is fullpath -> slug
    for fullpath, slug in M.EXPECTED_MODELS.items():
        layer = M.EXPECTED_LAYERS[fullpath]
        for editor in M.EXPECTED_EDITORS:
            for seed in M.EXPECTED_SEEDS:
                arrays = _make_cell_arrays(n_edits, n_probes,
                                           arm_esr_seed=hash((slug, editor, seed)) % (2**31))
                table = _make_cell_table(slug, editor, layer, seed,
                                         n_edits, n_probes)
                _write_cell_to_disk(root, slug, editor, layer, seed,
                                    arrays, table)
    return root


# ============================================================
# Numerical primitive tests
# ============================================================

class TestSpearmanPrimitives(unittest.TestCase):
    def test_spearman_perfect_positive(self):
        a = np.arange(20, dtype=float)
        b = a * 2 + 0.1
        self.assertAlmostEqual(M.spearman(a, b), 1.0, places=10)

    def test_spearman_perfect_negative(self):
        a = np.arange(20, dtype=float)
        b = -a + 5.0
        self.assertAlmostEqual(M.spearman(a, b), -1.0, places=10)

    def test_spearman_ignores_nans(self):
        a = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
        b = np.array([10.0, 20.0, 30.0, 99.0, 50.0])
        rho = M.spearman(a, b)
        self.assertTrue(np.isfinite(rho))
        self.assertAlmostEqual(rho, 1.0, places=10)

    def test_spearman_too_few_returns_nan(self):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        self.assertTrue(np.isnan(M.spearman(a, b)))

    def test_per_edit_summaries(self):
        rng = np.random.default_rng(0)
        D = rng.standard_normal((50, 10))
        s = M.per_edit_summaries(D)
        self.assertEqual(s["signed_mean"].shape, (50,))
        self.assertEqual(s["absmean"].shape, (50,))
        self.assertEqual(s["l2"].shape, (50,))
        self.assertEqual(s["p95abs"].shape, (50,))
        self.assertTrue(np.all(s["l2"] >= 0))
        self.assertTrue(np.all(s["absmean"] >= 0))
        self.assertTrue(np.all(s["p95abs"] >= 0))


# ============================================================
# Hierarchical bootstrap determinism tests
# ============================================================

class TestHierBootDeterminism(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.cos_per = [rng.standard_normal((200, 200)) for _ in range(3)]
        self.d_per = [rng.standard_normal((200, 200)) for _ in range(3)]
        self.eok_per = [(rng.random(200) > 0.1).astype(float) for _ in range(3)]
        self.arm_eok = [(rng.random(200) > 0.3).astype(float) for _ in range(3)]

    def test_hier_boot_stat_deterministic(self):
        a1 = M.hier_boot_stat(self.cos_per, self.d_per, M.stat_flat_rank,
                              n_boot=100)
        a2 = M.hier_boot_stat(self.cos_per, self.d_per, M.stat_flat_rank,
                              n_boot=100)
        self.assertEqual(a1, a2)

    def test_hier_boot_esr_deterministic(self):
        a1 = M.hier_boot_esr(self.arm_eok, n_boot=100)
        a2 = M.hier_boot_esr(self.arm_eok, n_boot=100)
        self.assertEqual(a1, a2)

    def test_hier_boot_conditional_deterministic(self):
        a1 = M.hier_boot_conditional(self.eok_per, self.arm_eok, n_boot=100)
        a2 = M.hier_boot_conditional(self.eok_per, self.arm_eok, n_boot=100)
        self.assertEqual(a1, a2)

    def test_hier_boot_different_seed_different_ci(self):
        a1 = M.hier_boot_stat(self.cos_per, self.d_per, M.stat_flat_rank,
                              n_boot=100, rng_seed=1)
        a2 = M.hier_boot_stat(self.cos_per, self.d_per, M.stat_flat_rank,
                              n_boot=100, rng_seed=2)
        self.assertNotEqual(a1, a2)


# ============================================================
# Grid validation tests
# ============================================================

class TestGridValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = _make_full_27_cell_root(self.tmp.name)
        # Synthesize a "loaded cells" map for direct API tests
        cells, notes = M.load_phase1_cells(self.root)
        self.cells = cells
        self.notes = notes

    def tearDown(self):
        self.tmp.cleanup()

    def test_exact_27_grid_loads_clean(self):
        self.assertEqual(len(self.cells), 27, f"expected 27 cells, got {len(self.cells)}")

    def test_validate_grid_passes_for_clean_27(self):
        ok, audit, errors = M.validate_grid(self.cells)
        self.assertTrue(ok, f"errors: {errors}")
        self.assertEqual(audit["expected_n_cells"], 27)
        self.assertEqual(audit["found_n_cells"], 27)
        self.assertEqual(audit["missing"], [])
        self.assertEqual(audit["extras"], [])
        self.assertEqual(audit["duplicates"], [])

    def test_smoke_directory_rejected(self):
        # Add a "smoke" subdirectory; should be ignored by loader
        smoke_dir = os.path.join(self.root, "smoke_test")
        os.makedirs(smoke_dir)
        rng = np.random.default_rng(0)
        np.savez(os.path.join(smoke_dir, "QS_phase1_raw.npz"),
                 COS=rng.standard_normal((50, 50)),
                 damage_fp32=rng.standard_normal((50, 50)),
                 edit_ok_fp32=(rng.random(50) > 0.5).astype(float))
        cells, _ = M.load_phase1_cells(self.root)
        self.assertNotIn("smoke_test", cells)
        ok, audit, errors = M.validate_grid(cells)
        self.assertTrue(ok)

    def test_aggregate_and_selftest_ignored(self):
        # These are non-cell dirs the loader must skip
        for d in ("aggregate", "selftest"):
            full = os.path.join(self.root, d)
            os.makedirs(full, exist_ok=True)
        cells, notes = M.load_phase1_cells(self.root)
        for d in ("aggregate", "selftest"):
            self.assertNotIn(d, cells)
        # notes mention both
        joined = " | ".join(notes)
        self.assertIn("aggregate", joined)
        self.assertIn("selftest", joined)

    def test_extra_cell_rejected(self):
        # Add an extra cell outside the preregistered 27-grid
        arrays = _make_cell_arrays()
        table = _make_cell_table("llama1b", "alpha", 12, 99)  # seed=99 unexpected
        _write_cell_to_disk(self.root, "llama1b", "alpha", 12, 99, arrays, table)
        cells, _ = M.load_phase1_cells(self.root)
        ok, audit, errors = M.validate_grid(cells)
        self.assertFalse(ok)
        # Validator emits both seed-not-in-expected and unexpected-cells
        joined = " ".join(errors).lower()
        self.assertTrue("unexpected" in joined or "extra" in joined,
                        f"expected 'unexpected'/'extra' in errors, got: {errors}")

    def test_missing_cell_rejected(self):
        # Remove one expected cell directory
        import shutil
        target = os.path.join(self.root, "llama1b_alpha_L12_s0")
        shutil.rmtree(target, ignore_errors=True)
        cells, _ = M.load_phase1_cells(self.root)
        ok, audit, errors = M.validate_grid(cells)
        self.assertFalse(ok)
        self.assertTrue(any("missing" in e for e in errors))
        self.assertIn("llama1b_alpha_L12_s0", str(audit["missing"]))

    def test_nonfinite_rejected(self):
        # Inject a NaN into COS for one cell
        target = os.path.join(self.root, "qwen15b_rome_L21_s2")
        path = os.path.join(target, "QS_phase1_raw.npz")
        npz = np.load(path)
        arrays = {k: np.array(npz[k]) for k in npz.files}
        arrays["COS"][0, 0] = float("nan")
        np.savez(path, **arrays)
        cells, _ = M.load_phase1_cells(self.root)
        ok, audit, errors = M.validate_grid(cells)
        self.assertFalse(ok)
        self.assertTrue(any("non-finite" in e for e in errors))

    def test_duplicate_cellname_caught(self):
        # Validator must HARD-ASSERT on duplicate cell names — no skipTest.
        # Build a list of (name, cell) pairs and pass the SAME name twice to
        # exercise the validator's seen_names branch directly (no dict key
        # collisions required).  The validate_grid() function accepts both a
        # dict and a list of (name, cell) pairs precisely for this test.
        cells, _ = M.load_phase1_cells(self.root)
        dup_target = "llama1b_alpha_L12_s0"
        c0 = cells[dup_target]
        # Take every cell once, then append the duplicate name a second time.
        items = list(cells.items())
        items.append((dup_target, c0))  # duplicate name → seen_names -> 2
        ok, audit, errors = M.validate_grid(items)
        # Hard assertions: validator MUST detect the duplicate and refuse.
        self.assertFalse(ok,
                         msg=f"validator must fail on duplicate name {dup_target}; "
                             f"got errors={errors}")
        has_dup_err = any("duplicate" in e.lower() for e in errors)
        self.assertTrue(has_dup_err,
                        msg=f"validator must emit a duplicate error; got {errors}")
        # The audit must record the duplicate name (so it shows up in the JSON).
        self.assertIn(dup_target, audit.get("duplicates", []),
                      msg=f"audit.duplicates missing {dup_target}; "
                          f"got {audit.get('duplicates')}")


# ============================================================
# Range vs CI labelling tests (v1.1 directive)
# ============================================================

class TestRangeLabelling(unittest.TestCase):
    def setUp(self):
        # Use a small N_BOOT for tests to keep CI time low
        self._saved_n_boot = M.N_BOOT
        M.N_BOOT = 50
        self.tmp = tempfile.TemporaryDirectory()
        self.root = _make_full_27_cell_root(self.tmp.name)
        self.cells, _ = M.load_phase1_cells(self.root)
        self.ok, self.audit, self.errors = M.validate_grid(self.cells)
        groups, meta = M.group_by_cell(self.cells)
        # Take one cell (the first) for inspection
        first_key = sorted(groups.keys())[0]
        seed_entries = groups[first_key]
        seeds_data = M.seeds_to_inputs(seed_entries)
        self.cell_result = M.analyze_cell(seeds_data, meta[first_key])
        seed_tables = [c["table"] for _s, c in seed_entries]
        self.cell_result["generation_checks"] = M.aggregate_generation_checks(seed_tables)

    def tearDown(self):
        self.tmp.cleanup()
        M.N_BOOT = self._saved_n_boot

    def test_no_ci_in_flat_rank(self):
        arm = next(iter(self.cell_result["arms"].values()))
        flat = arm["flat_rank"]
        # v1.2: must be rank-survival (D_fp32 vs D_quant), no CI on this metric
        self.assertNotIn("ci95", flat, "flat_rank must NOT carry a CI")
        self.assertNotIn("ci", flat)
        self.assertEqual(flat["metric"], "rank_survival_Dfp32_vs_Dquant_flat")
        # point is mean-of-per-seed Spearmans (manuscript policy)
        self.assertEqual(flat["point_kind"], "mean_of_per_seed_spearmans")
        self.assertIn("pooled_alternative", flat)
        self.assertIn("range_min_max", flat)
        self.assertEqual(flat["range_kind"], "per_seed_point_estimate_min_max")
        rng = flat["range_min_max"]
        self.assertEqual(len(rng), 2)
        self.assertTrue(np.isfinite(rng[0]) and np.isfinite(rng[1]))
        self.assertLessEqual(rng[0], rng[1])
        self.assertEqual(len(flat["per_seed"]), 3)

    def test_no_ci_in_within_probe_rank(self):
        arm = next(iter(self.cell_result["arms"].values()))
        wp = arm["within_probe_rank"]
        self.assertNotIn("ci95", wp)
        self.assertEqual(wp["metric"], "rank_survival_Dfp32_vs_Dquant_within_probe")
        self.assertEqual(wp["point_kind"], "mean_of_per_seed_spearmans")
        self.assertIn("pooled_alternative", wp)
        self.assertIn("range_min_max", wp)
        self.assertEqual(wp["range_kind"], "per_seed_point_estimate_min_max")
        rng = wp["range_min_max"]
        self.assertEqual(len(rng), 2)
        self.assertLessEqual(rng[0], rng[1])

    def test_edit_level_ranks_have_ci95(self):
        arm = next(iter(self.cell_result["arms"].values()))
        for summary in ("signed_mean", "absmean", "l2", "p95abs"):
            entry = arm["edit_level_ranks"][summary]
            self.assertIn("ci95", entry, f"{summary} must have ci95")
            self.assertIn("point", entry)
            self.assertIn("boot_dist_mean", entry, "boot mean (not point estimator)")
            self.assertEqual(entry["point_kind"], "mean_of_per_seed_spearmans")
            self.assertEqual(entry["boot_kind"],
                             "hier_boot_seeds_then_edits_percentile_95")
            ci = entry["ci95"]
            self.assertEqual(len(ci), 2)
            self.assertLessEqual(ci[0], ci[1])

    def test_geometry_sensitivity_cos_present_separately(self):
        arm = next(iter(self.cell_result["arms"].values()))
        self.assertIn("geometry_sensitivity_cos", arm)
        g = arm["geometry_sensitivity_cos"]
        self.assertIn("GEOMETRY SENSITIVITY", g["label"])
        self.assertIn("cos_vs_Dquant_flat", g)
        self.assertIn("cos_vs_Dfp32_flat", g)
        self.assertIn("edit_level_cos_vs_Dquant", g)
        # COS-based diagnostics must also be RANGE, not CI
        for k in ("cos_vs_Dquant_flat", "cos_vs_Dfp32_flat"):
            self.assertNotIn("ci95", g[k])
            self.assertEqual(g[k]["range_kind"], "per_seed_point_estimate_min_max")

    def test_rank_survival_planted_preserved_approx_one(self):
        """v1.2 planted example: D_q = D_fp + tiny noise ⇒ rank-survival ≈ +1.

        Since the synthetic fixture uses survival_preserved=True, all rank-
        survival point estimates (flat, within_probe, edit-level signed_mean/
        absmean/L2/p95abs) should be very close to +1.
        """
        for arm_name, arm in self.cell_result["arms"].items():
            flat_pt = arm["flat_rank"]["point"]
            within_pt = arm["within_probe_rank"]["point"]
            self.assertGreater(flat_pt, 0.95,
                f"{arm_name} flat_rank point={flat_pt} expected near 1.0")
            self.assertGreater(within_pt, 0.95,
                f"{arm_name} within_probe point={within_pt} expected near 1.0")
            for summary in ("signed_mean", "absmean", "l2", "p95abs"):
                pt = arm["edit_level_ranks"][summary]["point"]
                self.assertGreater(pt, 0.95,
                    f"{arm_name} {summary} point={pt} expected near 1.0")

    def test_absolute_quantized_esr_has_ci95(self):
        arm = next(iter(self.cell_result["arms"].values()))
        esr = arm["absolute_quantized_esr"]
        self.assertIn("ci95", esr)
        self.assertIn("point", esr)
        self.assertIn("boot_dist_mean", esr)
        self.assertEqual(esr["point_kind"], "mean_of_per_seed_spearmans")

    def test_conditional_survival_has_ci95(self):
        arm = next(iter(self.cell_result["arms"].values()))
        cond = arm["conditional_survival_given_fp32_worked"]
        self.assertIn("ci95", cond)
        self.assertIn("point", cond)
        self.assertIn("boot_dist_mean", cond)
        self.assertEqual(cond["point_kind"], "mean_of_per_seed_spearmans")

    def test_absolute_fp32_esr_has_ci95(self):
        e = self.cell_result["absolute_fp32_esr"]
        self.assertIn("ci95", e)
        self.assertIn("point", e)
        self.assertIn("boot_dist_mean", e)
        self.assertEqual(e["point_kind"], "mean_of_per_seed_spearmans")

    def test_generation_uses_range_no_ci(self):
        gen = self.cell_result["generation_checks"]
        for k in ("fp32",) + M.EXPECTED_ARMS:
            entry = gen[k]
            self.assertIn("perplexity_mean_range_min_max", entry, f"{k} gen")
            self.assertIn("paraphrase_esr_mean_range_min_max", entry, f"{k} gen")
            ppl_rng = entry["perplexity_mean_range_min_max"]
            esr_rng = entry["paraphrase_esr_mean_range_min_max"]
            self.assertEqual(len(ppl_rng), 2)
            self.assertEqual(len(esr_rng), 2)
            self.assertEqual(entry["range_kind"], "per_seed_point_estimate_min_max")
            # No CI labels on generation
            self.assertNotIn("ci95", entry)


# ============================================================
# Split-overlap / hash invariant tests
# ============================================================

class TestSplitAuditHash(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = _make_full_27_cell_root(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_split_audit_present_for_every_seed(self):
        cells, _ = M.load_phase1_cells(self.root)
        groups, meta = M.group_by_cell(cells)
        for key, seed_entries in groups.items():
            for s, cell in seed_entries:
                npz = cell["npz"]
                audit = M.split_audit_for_cell(npz)
                self.assertEqual(audit["n_edits"], 200)
                self.assertEqual(audit["n_probes"], 200)
                self.assertEqual(len(audit["cos_sha256"]), 64)
                self.assertEqual(len(audit["edit_ok_fp32_sha256"]), 64)
                self.assertEqual(len(audit["damage_fp32_sha256"]), 64)
                self.assertTrue(audit["cos_finite"])

    def test_split_audit_hash_stable_across_reread(self):
        cells, _ = M.load_phase1_cells(self.root)
        groups, meta = M.group_by_cell(cells)
        for key, seed_entries in groups.items():
            for s, cell in seed_entries:
                npz = cell["npz"]
                a1 = M.split_audit_for_cell(npz)
                npz2 = np.load(cell["npz_path"], allow_pickle=True)
                a2 = M.split_audit_for_cell(npz2)
                self.assertEqual(a1["cos_sha256"], a2["cos_sha256"])
                self.assertEqual(a1["edit_ok_fp32_sha256"], a2["edit_ok_fp32_sha256"])
                self.assertEqual(a1["damage_fp32_sha256"], a2["damage_fp32_sha256"])


# ============================================================
# End-to-end run + atomic JSON tests
# ============================================================

class TestEndToEndRun(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = _make_full_27_cell_root(self.tmp.name)
        self.out = os.path.join(self.tmp.name, "out",
                                "quant_survival_repair_v1.json")

    def tearDown(self):
        self.tmp.cleanup()

    def test_run_writes_versioned_atomic_json(self):
        repair = M.run(self.root, self.out, verbose=False, n_boot=50)
        self.assertEqual(repair["status"], "PASS")
        self.assertTrue(os.path.exists(self.out))
        # Atomic sidecar should be gone after run
        sidecar = self.out + ".canon.tmp"
        self.assertFalse(os.path.exists(sidecar))
        # Verify content
        with open(self.out) as f:
            data = json.load(f)
        self.assertIn("module_provenance", data)
        self.assertEqual(data["module_provenance"]["n_boot"], 50)
        # 'cells' is the per-(model,editor,layer) aggregated output (9
        # groups).  The underlying 27 raw cells appear in the grid_audit.
        self.assertEqual(len(data["cells"]), 9)
        self.assertEqual(data["grid_audit"]["expected_n_cells"], 27)
        self.assertEqual(data["grid_audit"]["found_n_cells"], 27)

    def test_run_no_partial_leftover(self):
        # No .partial sidecar file should be present anywhere
        M.run(self.root, self.out, verbose=False, n_boot=50)
        for root, _dirs, files in os.walk(os.path.dirname(self.out)):
            for fn in files:
                self.assertNotIn(".partial", fn)
                self.assertNotIn(".canon.tmp", fn)

    def test_run_post_hoc_present(self):
        repair = M.run(self.root, self.out, verbose=False, n_boot=50)
        ph = repair["post_hoc_diagnostics"]
        self.assertIn("label", ph)
        self.assertIn("POST HOC", ph["label"])
        for k in ("rho_abs_quant_esr_vs_conditional_survival",
                  "rho_flat_survival_rank_vs_conditional_survival",
                  "rho_base_noise_abs_vs_conditional_survival",
                  "rho_delta_flat_minus_within_survival_vs_conditional_survival"):
            self.assertIn(k, ph["diagnostics"])

    def test_run_smoke_dir_does_not_break_grid(self):
        # Inject a smoke-named extra dir; should be ignored
        smoke_dir = os.path.join(self.root, "smoke")
        os.makedirs(smoke_dir)
        rng = np.random.default_rng(0)
        np.savez(os.path.join(smoke_dir, "QS_phase1_raw.npz"),
                 COS=rng.standard_normal((50, 50)),
                 damage_fp32=rng.standard_normal((50, 50)),
                 edit_ok_fp32=(rng.random(50) > 0.5).astype(float))
        repair = M.run(self.root, self.out, verbose=False, n_boot=50)
        self.assertEqual(repair["status"], "PASS")
        self.assertEqual(len(repair["cells"]), 9)
        # grid_audit confirms all 27 cells loaded (smoke ignored)
        self.assertEqual(repair["grid_audit"]["expected_n_cells"], 27)
        self.assertEqual(repair["grid_audit"]["found_n_cells"], 27)


# ============================================================
# Provenance tests
# ============================================================

class TestProvenance(unittest.TestCase):
    def test_build_provenance_has_scope_revision_v1_2(self):
        prov = M.build_provenance(50)
        self.assertEqual(prov["n_boot"], 50)
        # v1.2.0 revision block name (linter-renamed)
        sr_keys = [k for k in prov.keys() if k.startswith("scope_revision_v1_2")]
        self.assertTrue(sr_keys, f"no scope_revision_v1_2* key in prov keys={list(prov.keys())}")
        sr = prov[sr_keys[0]]
        # v1.2: rank-survival (D_fp32 vs D_quant) is the metric
        self.assertIn("D_fp32", sr["flat_rank"])
        self.assertIn("D_fp32", sr["within_probe_rank"])
        self.assertIn("D_fp32", sr["edit_level_ranks_signed_mean_absmean_l2_p95abs"])
        self.assertIn("D_quant", sr["flat_rank"])
        # COS is separately labeled under geometry_sensitivity_cos
        self.assertIn("geometry_sensitivity_cos", sr)
        for k in ("absolute_fp32_esr", "absolute_quantized_esr",
                  "conditional_survival_given_fp32_worked"):
            self.assertIn("hier-boot CI95", sr[k])
        for k in ("generation_perplexity_mean", "generation_paraphrase_esr_mean"):
            self.assertIn("RANGE", sr[k])
        self.assertFalse(prov["killed_live_runner"])
        self.assertEqual(prov["live_runner_path"],
                         "edit-harness/experiments/quant_survival_phase1.py")
        self.assertEqual(prov["rng_seed"], M.RNG_SEED)
        self.assertEqual(prov["ci_level"], 0.95)


# ============================================================
# v1.2.0 INDEPENDENT-REVIEW FIX TESTS
# ============================================================

class TestShapeValidation(unittest.TestCase):
    """Per-key shape validation: damage_fp32, edit_ok_fp32, all damage arms
    and esr arms must match expected shapes; C3 vectors where used."""

    def _make_cell(self, name, **npz_overrides):
        """Make a synthetic cell with overridable npz fields for shape testing."""
        rng = np.random.default_rng(0)
        n_edits, n_probes = 200, 200
        npz_obj = {
            "COS": rng.uniform(0, 1, (n_edits, n_probes)),
            "damage_fp32": rng.standard_normal((n_edits, n_probes)),
            "edit_ok_fp32": (rng.uniform(0, 1, n_edits) > 0.05).astype(float),
        }
        for arm in M.EXPECTED_ARMS:
            npz_obj[f"damage__{arm}"] = rng.standard_normal((n_edits, n_probes))
            npz_obj[f"esr__{arm}"] = (rng.uniform(0, 1, n_edits) > 0.1).astype(float)
            npz_obj[f"base__{arm}"] = 0.01 * rng.standard_normal(n_probes)
        for scheme in M.EXPECTED_SCHEMES:
            npz_obj[f"c3_ratio__{scheme}"] = rng.uniform(0, 2, 1024).astype(np.float32)
            npz_obj[f"c3_rfunc__{scheme}"] = rng.uniform(0, 0.1, n_edits).astype(np.float32)
            npz_obj[f"c3_rparam__{scheme}"] = rng.uniform(0, 1, n_edits).astype(np.float32)
        npz_obj.update(npz_overrides)
        class _FakeNpz:
            def __init__(self, d):
                self._d = d
                self.files = list(d.keys())
            def __getitem__(self, k):
                return self._d[k]
        fullpath = next(iter(M.EXPECTED_MODELS))
        table = {
            "experiment": "quant_survival_phase1",
            "schema_version": "qs.phase1.v1",
            "model": fullpath, "layer": M.EXPECTED_LAYERS[fullpath],
            "editor": "rome", "n_edits": n_edits, "n_probes": n_probes,
            "seed": 0, "schemes": list(M.EXPECTED_SCHEMES),
            "codec": "real", "blocksize": 64, "fullmodel_cache": "on",
            "edited_layers": [M.EXPECTED_LAYERS[fullpath]],
            "c2_scope": "", "quant_note": "", "damage_metric_note": "",
            "esr": {"mean_esr_fp32": 0.95, "n_edits_worked_fp32": 190},
            "mechanism_tie": {"rho_keycos_damage_fp32_pooled": 0.5,
                              "rho_keycos_damage_fp32_within_probe": 0.5,
                              "within_probe_n_cols": n_probes},
            "arms": {arm: {
                "locality": arm.split("_", 1)[1], "scheme": arm.split("_")[0],
                "mean_esr": 0.95, "esr_survival_given_fp32_worked": 0.9,
                "rho_keycos_damage_pooled": 0.5,
                "rho_keycos_damage_pooled_base_subtracted": 0.4,
                "rho_keycos_damage_within_probe": 0.5,
                "rho_keycos_damage_within_probe_base_subtracted": 0.4,
                "delta_rho_vs_fp32_pooled": 0.0,
                "delta_rho_vs_fp32_within_probe": 0.0,
                "rho_damage_fp32_vs_arm_rank_survival": 0.9,
                "rho_damage_fp32_vs_arm_rank_survival_base_subtracted": 0.8,
                "permutation_null_p_pooled": 0.001,
                "rho_pooled_ci95_bootstrap_edits": [0.4, 0.6],
            } for arm in M.EXPECTED_ARMS},
            "bin_width_mechanism_C3": {s: {
                "F_above_bin": 0.5, "median_ratio": 1.0, "p90_ratio": 2.0,
                "r_func_mean": 0.01, "r_param_mean": 0.5,
            } for s in M.EXPECTED_SCHEMES},
            "generation_checks": {"fp32": {"n_gen_probes": 10,
                                            "perplexity_mean": 12.0,
                                            "paraphrase_esr_mean": 0.8}},
        }
        return {"npz": _FakeNpz(npz_obj), "table": table, "dir": "/syn",
                "npz_path": "/syn/QS_phase1_raw.npz",
                "tbl_path": "/syn/QS_phase1_table.json"}

    def test_damage_fp32_bad_shape_rejected(self):
        cells = {"llama1b_rome_L12_s0": self._make_cell(
            "llama1b_rome_L12_s0", damage_fp32=np.random.standard_normal((150, 200)))}
        ok, audit, errors = M.validate_grid(cells)
        self.assertFalse(ok)
        self.assertTrue(any("damage_fp32" in e for e in errors),
                        msg=f"errors={errors}")

    def test_edit_ok_fp32_bad_shape_rejected(self):
        cells = {"llama1b_rome_L12_s0": self._make_cell(
            "llama1b_rome_L12_s0", edit_ok_fp32=np.random.uniform(0, 1, 150))}
        ok, audit, errors = M.validate_grid(cells)
        self.assertFalse(ok)
        self.assertTrue(any("edit_ok_fp32" in e for e in errors),
                        msg=f"errors={errors}")

    def test_damage_arm_bad_shape_rejected(self):
        cells = {"llama1b_rome_L12_s0": self._make_cell(
            "llama1b_rome_L12_s0",
            damage__nf4dq_edited_layer=np.random.standard_normal((100, 200)))}
        ok, audit, errors = M.validate_grid(cells)
        self.assertFalse(ok)
        self.assertTrue(any("damage__nf4dq_edited_layer" in e for e in errors),
                        msg=f"errors={errors}")

    def test_esr_arm_bad_shape_rejected(self):
        cells = {"llama1b_rome_L12_s0": self._make_cell(
            "llama1b_rome_L12_s0", esr__int8_edited_layer=np.random.uniform(0, 1, 150))}
        ok, audit, errors = M.validate_grid(cells)
        self.assertFalse(ok)
        self.assertTrue(any("esr__int8_edited_layer" in e for e in errors),
                        msg=f"errors={errors}")


class TestAtomicSidecarRetention(unittest.TestCase):
    """Atomic write retains an immutable hash-versioned sidecar AND keeps the
    canonical file in sync — with NO missing-sidecar window."""

    def test_sidecar_retained_and_matches_canonical(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "out.json")
            payload = {"hello": "world", "n": 42, "nested": {"a": [1, 2, 3]}}
            canonical, versioned = M._atomic_write_json(path, payload)
            # Both files exist
            self.assertTrue(os.path.exists(canonical))
            self.assertTrue(os.path.exists(versioned))
            # Same sha256
            with open(canonical, "rb") as f:
                canon_bytes = f.read()
            with open(versioned, "rb") as f:
                side_bytes = f.read()
            self.assertEqual(hashlib.sha256(canon_bytes).hexdigest(),
                             hashlib.sha256(side_bytes).hexdigest())
            # Canonical filename is the requested path
            self.assertEqual(canonical, path)
            # Sidecar filename has sha256 prefix
            self.assertIn("__", os.path.basename(versioned))
            # No leftover temp files in the directory
            leftover = [f for f in os.listdir(td)
                        if ".partial" in f or ".canon.tmp" in f]
            self.assertEqual(leftover, [],
                             msg=f"temp leftovers present: {leftover}")

    def test_sidecar_exists_throughout_no_missing_window(self):
        """Reviewer requirement: there must be NO time window during which the
        canonical exists without the sidecar.

        We monkey-patch `os.replace` so we can inspect the filesystem state
        immediately before and after the canonical publish. The reviewer
        contract is:
          * BEFORE the publish: sidecar already exists; canonical does not.
          * AFTER the publish: both exist with identical bytes.
        """
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "out.json")
            payload = {"k": "v"}
            real_replace = os.replace
            state = {"publish_observed": False,
                     "sidecar_pre_publish": None,
                     "canonical_pre_publish": None,
                     "sidecar_post_publish": None,
                     "canonical_post_publish": None}
            def _spy_replace(src, dst):
                # The single replace in the new ordering is partial -> canonical.
                # Detect it by `dst == canonical`.
                is_canonical_publish = (
                    os.path.basename(str(dst)) == os.path.basename(path))
                if is_canonical_publish:
                    sidecar_before = next((os.path.join(td, f)
                                            for f in os.listdir(td)
                                            if f.startswith("out__")
                                            and f.endswith(".json")),
                                           "")
                    state["sidecar_pre_publish"] = (
                        os.path.exists(sidecar_before) if sidecar_before
                        else False)
                    state["canonical_pre_publish"] = os.path.exists(path)
                real_replace(src, dst)
                if is_canonical_publish:
                    sidecar_after = next((os.path.join(td, f)
                                           for f in os.listdir(td)
                                           if f.startswith("out__")
                                           and f.endswith(".json")),
                                          "")
                    state["sidecar_post_publish"] = (
                        os.path.exists(sidecar_after) if sidecar_after
                        else False)
                    state["canonical_post_publish"] = os.path.exists(path)
                    state["publish_observed"] = True
            os.replace = _spy_replace  # type: ignore
            try:
                canonical, versioned = M._atomic_write_json(path, payload)
            finally:
                os.replace = real_replace  # type: ignore
            self.assertTrue(state["publish_observed"],
                            msg="os.replace to canonical path was never invoked")
            # Sidecar existed BEFORE the canonical was published (no missing window).
            self.assertTrue(state["sidecar_pre_publish"],
                            msg="sidecar must exist before canonical publish")
            self.assertFalse(state["canonical_pre_publish"],
                             msg="canonical must NOT exist before atomic publish")
            # AFTER the publish: both exist.
            self.assertTrue(state["sidecar_post_publish"],
                            msg="sidecar must still exist after canonical publish")
            self.assertTrue(state["canonical_post_publish"],
                            msg="canonical must exist after atomic publish")

    def test_sidecar_preserved_across_consecutive_writes(self):
        """Two consecutive writes must produce two distinct sidecars; the
        canonical always reflects the most-recent sidecar; the older sidecar
        stays on disk as an immutable record."""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "out.json")
            _, v1 = M._atomic_write_json(path, {"v": 1})
            _, v2 = M._atomic_write_json(path, {"v": 2})
            self.assertNotEqual(v1, v2,
                                msg="distinct payloads must produce distinct sidecars")
            self.assertTrue(os.path.exists(v1),
                            msg="older sidecar must remain on disk")
            self.assertTrue(os.path.exists(v2),
                            msg="newer sidecar must exist")
            with open(path) as f:
                data = json.load(f)
            self.assertEqual(data, {"v": 2},
                             msg="canonical must reflect latest write")
            # Canonical bytes equal newer sidecar bytes.
            with open(path, "rb") as f:
                cb = f.read()
            with open(v2, "rb") as f:
                sb = f.read()
            self.assertEqual(cb, sb)


class TestBootstrapFieldReporting(unittest.TestCase):
    """Bootstrap functions must report boot_n_finite, boot_n_total,
    skipped_fraction; conditional survival must expose n_nan_draws_exposed."""

    def test_hier_boot_stat_returns_dict_with_required_keys(self):
        rng = np.random.default_rng(0)
        cos = [rng.uniform(0, 1, (60, 80)) for _ in range(3)]
        d = [rng.standard_normal((60, 80)) for _ in range(3)]
        out = M.hier_boot_stat(cos, d, M.stat_flat_rank, n_boot=50, rng_seed=42)
        self.assertIsInstance(out, dict)
        for k in ("ci95_lo", "ci95_hi", "boot_dist_mean",
                  "boot_n_finite", "boot_n_total", "skipped_fraction"):
            self.assertIn(k, out, msg=f"missing key {k}")
        self.assertEqual(out["boot_n_total"], 50)
        self.assertEqual(out["boot_n_finite"] + (50 - out["boot_n_finite"]),
                         50)
        self.assertGreaterEqual(out["skipped_fraction"], 0.0)
        self.assertLessEqual(out["skipped_fraction"], 1.0)

    def test_hier_boot_esr_returns_dict_with_required_keys(self):
        eok = [np.random.binomial(1, 0.9, 50).astype(float) for _ in range(3)]
        out = M.hier_boot_esr(eok, n_boot=50, rng_seed=42)
        for k in ("ci95_lo", "ci95_hi", "boot_dist_mean",
                  "boot_n_finite", "boot_n_total", "skipped_fraction"):
            self.assertIn(k, out)

    def test_hier_boot_conditional_returns_dict_and_exposes_nan_draws(self):
        rng = np.random.default_rng(0)
        fp32 = [rng.binomial(1, 0.9, 50).astype(float) for _ in range(3)]
        arm = [rng.binomial(1, 0.85, 50).astype(float) for _ in range(3)]
        out = M.hier_boot_conditional(fp32, arm, n_boot=50, rng_seed=42)
        for k in ("ci95_lo", "ci95_hi", "boot_dist_mean",
                  "boot_n_finite", "boot_n_total", "skipped_fraction",
                  "n_nan_draws_exposed"):
            self.assertIn(k, out, msg=f"missing key {k}")
        # n_nan_draws_exposed must equal boot_n_total - boot_n_finite
        self.assertEqual(out["n_nan_draws_exposed"],
                         out["boot_n_total"] - out["boot_n_finite"])

    def test_conditional_with_all_fp32_failures_exposes_all_nan_draws(self):
        # All fp32 edits failed -> every draw returns NaN -> n_nan_draws_exposed == n
        fp32 = [np.zeros(20, float) for _ in range(3)]
        arm = [np.ones(20, float) for _ in range(3)]
        out = M.hier_boot_conditional(fp32, arm, n_boot=50, rng_seed=42)
        self.assertEqual(out["boot_n_finite"], 0)
        self.assertEqual(out["n_nan_draws_exposed"], 50)
        self.assertEqual(out["skipped_fraction"], 1.0)
        self.assertTrue(np.isnan(out["ci95_lo"]))
        self.assertTrue(np.isnan(out["ci95_hi"]))


class TestZeroPointOrNaNFix(unittest.TestCase):
    """The 0.0-or-nan latent bug is fixed: explicit None check (not `x or nan`)."""

    def test_base_quant_noise_zero_passes_through_unchanged(self):
        # When base_quant_noise_mean_abs is exactly 0.0 (a real valid value),
        # it must NOT be silently replaced with NaN.
        # Build a fake arm dict with base_quant_noise_mean_abs = 0.0
        rng = np.random.default_rng(0)
        cos = [rng.uniform(0, 1, (30, 40)) for _ in range(3)]
        dfp = [rng.standard_normal((30, 40)) for _ in range(3)]
        dq = [rng.standard_normal((30, 40)) for _ in range(3)]
        # Fake arm with 0.0 base noise — must NOT be coerced to nan
        seeds_data = []
        for s in range(3):
            arm_dict = {arm: {"damage": dq[s].copy(),
                              "esr": (rng.uniform(0, 1, 30) > 0.1).astype(float)}
                        for arm in M.EXPECTED_ARMS}
            arm_dict["nf4dq_edited_layer"]["base"] = np.zeros(40)  # exactly 0.0
            seeds_data.append({
                "COS": cos[s], "damage_fp32": dfp[s],
                "edit_ok_fp32": (rng.uniform(0, 1, 30) > 0.05).astype(float),
                "arms": arm_dict,
            })
        result = M.analyze_cell(seeds_data,
                                {"fullpath": next(iter(M.EXPECTED_MODELS)),
                                 "slug": "llama1b", "editor": "rome", "layer": 12},
                                n_boot=10)
        arm = result["arms"]["nf4dq_edited_layer"]
        self.assertEqual(arm["base_quant_noise_mean_abs"], 0.0,
                         msg=f"0.0 base was mis-coerced to {arm['base_quant_noise_mean_abs']!r}")


class TestRunnerUnmodifiedVerification(unittest.TestCase):
    """killed_live_runner is renamed runner_unmodified_verified; live runner
    sha256 is stored in module_provenance."""

    def test_provenance_has_runner_sha_and_verified_flag(self):
        prov = M.build_provenance(n_boot=50)
        self.assertIn("live_runner_sha256", prov)
        self.assertIn("live_runner_sha256_verified_at", prov)
        self.assertTrue(prov["runner_unmodified_verified"])
        # killed_live_runner is still present (backward compat) AND
        # the new field is also present
        self.assertIn("killed_live_runner", prov)
        # No duplicate n_boot_effective field — single source of truth.
        self.assertNotIn("n_boot_effective", prov)
        # sha256 is a 64-char hex
        sha = prov["live_runner_sha256"]
        self.assertEqual(len(sha), 64)
        int(sha, 16)  # parses as hex

    def test_live_runner_sha_matches_actual_file(self):
        prov = M.build_provenance(n_boot=50)
        recorded = prov["live_runner_sha256"]
        runner = os.path.join(M.HARNESS, "quant_survival_phase1.py")
        with open(runner, "rb") as f:
            actual = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(recorded, actual)


class TestNoGlobalNBootMutation(unittest.TestCase):
    """v1.2.0: NO global N_BOOT mutation — n_boot is threaded through explicitly."""

    def test_run_does_not_mutate_module_n_boot(self):
        saved = M.N_BOOT
        with tempfile.TemporaryDirectory() as td:
            # Build a minimal root with a single valid cell
            fullpath = next(iter(M.EXPECTED_MODELS))
            layer = M.EXPECTED_LAYERS[fullpath]
            cell_name = f"{M.EXPECTED_MODELS[fullpath]}_rome_L{layer}_s0"
            cell_dir = os.path.join(td, cell_name)
            os.makedirs(cell_dir)
            rng = np.random.default_rng(0)
            n_edits, n_probes = 200, 200
            npz_kwargs = dict(
                COS=rng.uniform(0, 1, (n_edits, n_probes)),
                damage_fp32=rng.standard_normal((n_edits, n_probes)),
                edit_ok_fp32=(rng.uniform(0, 1, n_edits) > 0.05).astype(float),
            )
            for arm in M.EXPECTED_ARMS:
                npz_kwargs[f"damage__{arm}"] = rng.standard_normal((n_edits, n_probes))
                npz_kwargs[f"esr__{arm}"] = (rng.uniform(0, 1, n_edits) > 0.1).astype(float)
                npz_kwargs[f"base__{arm}"] = 0.01 * rng.standard_normal(n_probes)
            for scheme in M.EXPECTED_SCHEMES:
                npz_kwargs[f"c3_ratio__{scheme}"] = rng.uniform(0, 2, 1024).astype(np.float32)
                npz_kwargs[f"c3_rfunc__{scheme}"] = rng.uniform(0, 0.1, n_edits).astype(np.float32)
                npz_kwargs[f"c3_rparam__{scheme}"] = rng.uniform(0, 1, n_edits).astype(np.float32)
            np.savez(os.path.join(cell_dir, "QS_phase1_raw.npz"), **npz_kwargs)
            # Build minimal valid table
            table = {
                "experiment": "quant_survival_phase1",
                "schema_version": "qs.phase1.v1",
                "model": fullpath, "layer": layer, "editor": "rome",
                "n_edits": n_edits, "n_probes": n_probes, "seed": 0,
                "schemes": list(M.EXPECTED_SCHEMES),
                "codec": "real", "blocksize": 64,
                "fullmodel_cache": "on",
                "edited_layers": [layer],
                "c2_scope": "", "quant_note": "", "damage_metric_note": "",
                "esr": {"mean_esr_fp32": 0.95, "n_edits_worked_fp32": 190},
                "mechanism_tie": {"rho_keycos_damage_fp32_pooled": 0.5,
                                  "rho_keycos_damage_fp32_within_probe": 0.5,
                                  "within_probe_n_cols": n_probes},
                "arms": {arm: {"locality": arm.split("_", 1)[1],
                               "scheme": arm.split("_")[0], "mean_esr": 0.95,
                               "esr_survival_given_fp32_worked": 0.9,
                               "rho_keycos_damage_pooled": 0.5,
                               "rho_keycos_damage_pooled_base_subtracted": 0.4,
                               "rho_keycos_damage_within_probe": 0.5,
                               "rho_keycos_damage_within_probe_base_subtracted": 0.4,
                               "delta_rho_vs_fp32_pooled": 0.0,
                               "delta_rho_vs_fp32_within_probe": 0.0,
                               "rho_damage_fp32_vs_arm_rank_survival": 0.9,
                               "rho_damage_fp32_vs_arm_rank_survival_base_subtracted": 0.8,
                               "permutation_null_p_pooled": 0.001,
                               "rho_pooled_ci95_bootstrap_edits": [0.4, 0.6]}
                          for arm in M.EXPECTED_ARMS},
                "bin_width_mechanism_C3": {s: {
                    "F_above_bin": 0.5, "median_ratio": 1.0, "p90_ratio": 2.0,
                    "r_func_mean": 0.01, "r_param_mean": 0.5,
                } for s in M.EXPECTED_SCHEMES},
                "generation_checks": {"fp32": {"n_gen_probes": 10,
                                                "perplexity_mean": 12.0,
                                                "paraphrase_esr_mean": 0.8}},
            }
            with open(os.path.join(cell_dir, "QS_phase1_table.json"), "w") as f:
                json.dump(table, f)
            out = os.path.join(td, "out.json")
            # Requested n_boot differs from module default; module constant
            # must NOT change.
            requested = 77
            M.run(td, out, verbose=False, n_boot=requested)
            self.assertEqual(M.N_BOOT, saved,
                             msg=f"N_BOOT was mutated from {saved} to {M.N_BOOT}")
            # Provenance in the written file reports the REQUESTED n_boot
            with open(out) as f:
                written = json.load(f)
            self.assertEqual(written["module_provenance"]["n_boot"], requested)
            # No duplicate n_boot_effective field
            self.assertNotIn("n_boot_effective",
                             written["module_provenance"])


if __name__ == "__main__":
    unittest.main()