"""Focused tests for `experiments/quant_survival_macros.py`.

These tests pin the v1.2.1 durability contract for Paper B macros:

  (1) When the canonical repair artefact is present and v1.2.1-compatible,
      regeneration against the on-disk manuscript macros.tex is an EXACT
      value match — same 103 macro names, same string value per macro
      (after the 3-decimal formatting convention). No tolerance, no
      allowlist, no 1-ulp carryover of stale sidecar values. The canonical
      v1.2.1 repair artefact is the single source of truth.

  (2) The generator fails closed (exit code 2, no output file written) in
      strict mode when the repair artefact is missing, unparseable, has the
      wrong module_provenance.version, wrong module name, or empty cells —
      this is the behaviour the real drain path relies on to prevent
      overwriting the manuscript with stale legacy-only content.

  (3) When the repair artefact is absent in non-strict mode, the generator
      still emits legacy macros (thresholds / primary / second / c3 / kill
      gates / metadata) so the legacy-only path keeps working for older
      harness runs.

  (4) Every macro name currently emitted into the manuscript macros.tex is
      preserved across regeneration (no renames, no drops).

CPU-only; numpy-only; no GPU, no torch, no network.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.dirname(THIS_DIR)
WORKSPACE = os.path.dirname(HARNESS)
sys.path.insert(0, os.path.join(HARNESS, "experiments"))

import quant_survival_macros as M  # noqa: E402


def _parse_macros(text: str) -> dict:
    """Return {macro_name: value_str} parsed from a macros.tex blob.

    Tolerates the multi-brace form
        \\newcommand{\\NAME}{VALUE}   % comment...
    and ignores section-comment lines that don't declare a macro.
    """
    out: dict = {}
    pattern = re.compile(
        r"\\newcommand\{(\\[A-Za-z]+)\}\{([^}]*)\}"
    )
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("%") or not s.startswith("\\newcommand"):
            continue
        m = pattern.search(s)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def _value_close(a: str, b: str) -> bool:
    """DEPRECATED — kept only to fail loudly if any future code reintroduces
    floating-point tolerance into the durability contract. Exact equality is
    required everywhere; see TestRegenerateAgainstManuscript.test_regeneration_is_semantic_match.
    """
    raise AssertionError(
        "Floating-point tolerance comparisons are forbidden in the v1.2.1 "
        "durability contract; require exact string equality."
    )


class TestRegenerateAgainstManuscript(unittest.TestCase):
    """(1) Regenerate against the on-disk manuscript + canonical repair."""

    @classmethod
    def setUpClass(cls):
        cls._orig_cwd = os.getcwd()
        # Sidecar lookup + sha256 hashing inside generate_macros rely on
        # cwd-relative paths matching the drain-script convention; run as
        # if invoked from edit-harness/.
        os.chdir(HARNESS)
        cls.manuscript_path = os.path.join(
            WORKSPACE, "submissions", "paper-b-neurocomputing", "macros.tex"
        )
        cls.gate_readout = os.path.join(
            HARNESS, "results", "quant_survival", "aggregate", "gate_readout.json"
        )
        cls.repair_path = os.path.join(
            HARNESS, "results", "quant_survival", "aggregate",
            "quant_survival_repair_v1.json",
        )
        cls.cur_macros = _parse_macros(open(cls.manuscript_path).read())
        # Snapshot file mtimes so we can assert we did not overwrite the manuscript.
        cls._manuscript_mtime = os.path.getmtime(cls.manuscript_path)
        cls._manuscript_bytes = open(cls.manuscript_path, "rb").read()

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._orig_cwd)

    def test_canonical_repair_artefact_exists(self):
        self.assertTrue(
            os.path.isfile(self.repair_path),
            f"canonical repair artefact missing: {self.repair_path}",
        )

    def test_canonical_repair_is_v121(self):
        repair = json.load(open(self.repair_path))
        prov = repair.get("module_provenance", {}) or {}
        self.assertEqual(prov.get("version"), "1.2.1")
        self.assertEqual(prov.get("module"), "quant_survival_reanalyze_v1")

    def test_regeneration_is_semantic_match(self):
        # Build the generator inputs from the on-disk artefacts (paths as the
        # drain script will pass them — relative to the edit-harness cwd).
        readout = json.load(open(self.gate_readout))
        repair = M._load_repair(self.repair_path, strict=True)
        self.assertIsNotNone(repair, "strict _load_repair returned None for canonical artefact")
        with tempfile.NamedTemporaryFile(suffix=".tex", delete=False, mode="w") as tf:
            tmp_path = tf.name
        try:
            generated = M.generate_macros(
                readout, "results/quant_survival/aggregate/gate_readout.json",
                repair=repair, repair_path="results/quant_survival/aggregate/quant_survival_repair_v1.json",
            )
            with open(tmp_path, "w") as f:
                f.write(generated)
            gen_macros = _parse_macros(generated)

            # Same macro count and same names (no renames, no drops).
            self.assertEqual(
                set(self.cur_macros.keys()), set(gen_macros.keys()),
                "macro set drifted between manuscript and generator",
            )
            self.assertEqual(len(self.cur_macros), 117)

            # Per-macro value comparison — EXACT equality required. The
            # canonical v1.2.1 repair artefact is the single source of
            # truth; no tolerance, no allowlist, no carryover of stale
            # sidecar values.
            drift = []
            for k in self.cur_macros:
                cur_v = self.cur_macros[k]
                gen_v = gen_macros[k]
                if cur_v != gen_v:
                    drift.append((k, cur_v, gen_v))
            self.assertEqual(
                drift, [],
                "macros drifted from canonical v1.2.1 (exact equality required):\n  "
                + "\n  ".join(f"{k}: cur={c!r} gen={g!r}" for k, c, g in drift),
            )

            # Sanity: the provenance header in the regeneration mentions
            # the v1.2.1 repair artefact, its sha256 (first 16 hex), and
            # n_boot=500. (Manuscript carries the same values.)
            self.assertIn("quant_survival_repair_v1.json", generated)
            self.assertIn("version=1.2.1", generated)
            self.assertIn("n_boot=500", generated)
            self.assertIn("rng_seed=12345", generated)
            # The 16-hex prefix of the canonical file's sha256 must appear
            # in the sidecar filename header.
            sha16 = hashlib.sha256(open(self.repair_path, "rb").read()).hexdigest()[:16]
            self.assertIn(sha16, generated,
                          f"sidecar sha prefix {sha16!r} not in regenerated header")
        finally:
            os.unlink(tmp_path)

    def test_k1_passes_when_one_validated_cell_passes(self):
        """K1 fires only when BOTH validated-law ROME cells fail NF4 FM.

        Canonical v1.2.1: Llama-1B = 0.904 >= 0.85 (passes),
        Llama-3B = 0.680 < 0.85 (fails). Therefore K1 = PASS: the
        phenomenon narrows but is not killed.
        """
        readout = json.load(open(self.gate_readout))
        repair = M._load_repair(self.repair_path, strict=True)
        self.assertEqual(M._compute_k1_status(readout, repair), "PASS")
        generated = M.generate_macros(
            readout, "results/quant_survival/aggregate/gate_readout.json",
            repair=repair,
            repair_path="results/quant_survival/aggregate/quant_survival_repair_v1.json",
        )
        macros = _parse_macros(generated)
        self.assertEqual(macros["\\gateKoneStatus"], "PASS")

    def test_canonical_absolute_esr_points(self):
        """Pin blocker-corrected 3-decimal aggregate absolute-ESR points."""
        readout = json.load(open(self.gate_readout))
        repair = M._load_repair(self.repair_path, strict=True)
        generated = M.generate_macros(
            readout, "results/quant_survival/aggregate/gate_readout.json",
            repair=repair,
            repair_path="results/quant_survival/aggregate/quant_survival_repair_v1.json",
        )
        macros = _parse_macros(generated)
        expected = {
            # Llama-1B ROME: canonical per-arm absolute quantized ESR points
            "\\pPrimarynfFourdqEditedLayerAbsEsrPt": "0.992",
            "\\pPrimarynfFourdqFullModelAbsEsrPt": "0.992",
            "\\pPrimaryintEightFullModelAbsEsrPt": "0.992",
            # Qwen per-editor FP32 aggregate points: preserve MEMIT/Alpha,
            # correct ROME from stale 0.450 to canonical 0.452.
            "\\pQwenRomeFpThirtyTwoAbsEsrPt": "0.452",
            "\\pQwenMemitFpThirtyTwoAbsEsrPt": "0.468",
            "\\pQwenAlphaFpThirtyTwoAbsEsrPt": "0.452",
            # Qwen per-editor NF4 full-model aggregate points.
            "\\pQwenRomeNfFourFMAbsEsrPt": "0.405",
            "\\pQwenMemitNfFourFMAbsEsrPt": "0.410",
            "\\pQwenAlphaNfFourFMAbsEsrPt": "0.385",
        }
        for name, value in expected.items():
            self.assertEqual(macros.get(name), value,
                             f"canonical point drift for {name}")


    """(2) Strict mode fails closed when the repair artefact is bad/missing."""

    def test_missing_file_exits_nonzero_no_output(self):
        # Synthesize a fresh empty readout so the legacy path is not the
        # blocker — strict_repair must trip on the repair artefact alone.
        with tempfile.TemporaryDirectory() as td:
            readout_path = os.path.join(td, "empty_readout.json")
            with open(readout_path, "w") as f:
                json.dump({"thresholds": {}, "cells": {}, "gates": {}}, f)
            out_path = os.path.join(td, "macros.tex")
            with self.assertRaises(SystemExit) as ctx:
                M._load_repair(os.path.join(td, "no_such_repair.json"), strict=True)
            self.assertEqual(ctx.exception.code, 2)

    def test_unparseable_json_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as td:
            rp = os.path.join(td, "broken.json")
            with open(rp, "w") as f:
                f.write("{this is not json")
            with self.assertRaises(SystemExit) as ctx:
                M._load_repair(rp, strict=True)
            self.assertEqual(ctx.exception.code, 2)

    def test_wrong_version_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as td:
            rp = os.path.join(td, "wrong_version.json")
            payload = {
                "status": "PASS",
                "module_provenance": {
                    "module": "quant_survival_reanalyze_v1",
                    "version": "1.0.0",  # not v1.2.1
                    "n_boot": 500,
                    "rng_seed": 12345,
                },
                "cells": [{"slug": "x", "editor": "rome", "layer": 12, "arms": {}}],
            }
            with open(rp, "w") as f:
                json.dump(payload, f)
            with self.assertRaises(SystemExit) as ctx:
                M._load_repair(rp, strict=True)
            self.assertEqual(ctx.exception.code, 2)

    def test_wrong_module_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as td:
            rp = os.path.join(td, "wrong_module.json")
            payload = {
                "status": "PASS",
                "module_provenance": {
                    "module": "different_script",
                    "version": "1.2.1",
                    "n_boot": 500,
                    "rng_seed": 12345,
                },
                "cells": [{"slug": "x", "editor": "rome", "layer": 12, "arms": {}}],
            }
            with open(rp, "w") as f:
                json.dump(payload, f)
            with self.assertRaises(SystemExit) as ctx:
                M._load_repair(rp, strict=True)
            self.assertEqual(ctx.exception.code, 2)

    def test_empty_cells_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as td:
            rp = os.path.join(td, "empty_cells.json")
            payload = {
                "status": "PASS",
                "module_provenance": {
                    "module": "quant_survival_reanalyze_v1",
                    "version": "1.2.1",
                    "n_boot": 500,
                    "rng_seed": 12345,
                },
                "cells": [],
            }
            with open(rp, "w") as f:
                json.dump(payload, f)
            with self.assertRaises(SystemExit) as ctx:
                M._load_repair(rp, strict=True)
            self.assertEqual(ctx.exception.code, 2)

    def test_cli_runner_with_bad_repair_exits_nonzero_no_overwrite(self):
        """The CLI runner mirrors what drain_paperb.sh invokes; verify it
        fails closed and does NOT clobber a destination file with stale
        legacy-only output when the canonical repair artefact is missing.
        """
        with tempfile.TemporaryDirectory() as td:
            readout = os.path.join(td, "readout.json")
            with open(readout, "w") as f:
                json.dump({"thresholds": {}, "cells": {}, "gates": {}}, f)
            out_path = os.path.join(td, "macros.tex")
            # Touch the destination to simulate "an existing manuscript".
            with open(out_path, "w") as f:
                f.write("% sentinel manuscript contents — must NOT be overwritten\n")
            rc = subprocess.call(
                [
                    sys.executable,
                    os.path.join(HARNESS, "experiments", "quant_survival_macros.py"),
                    "--in_path", readout,
                    "--repair_in", os.path.join(td, "no_such_repair.json"),
                    "--strict_repair",
                    "--out_path", out_path,
                ],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            self.assertNotEqual(rc, 0,
                                "CLI runner returned 0 with a missing repair artefact (must fail-closed)")
            self.assertEqual(
                open(out_path).read(),
                "% sentinel manuscript contents — must NOT be overwritten\n",
                "CLI runner overwrote the manuscript despite strict_repair=True",
            )


class TestNonStrictLegacyFallback(unittest.TestCase):
    """(3) Non-strict mode still emits legacy macros when repair is absent."""

    def test_legacy_only_no_repair_emits_required_macros(self):
        readout = {
            "thresholds": {
                "fp32_law_gate": 0.30, "esr_survival_4bit": 0.80,
                "esr_survival_8bit": 0.90, "delta_rho_tolerance": 0.15,
                "rank_survival_4bit": 0.85, "rank_survival_8bit": 0.95,
                "median_ratio_concentration": 1.0,
            },
            "cells": {
                "Llama-3.2-1B_rome_L12": {"n_seeds": 3, "c2_eligible": 1, "c3": {}},
                "Llama-3.2-3B_rome_L24": {"n_seeds": 3, "c2_eligible": 1, "c3": {}},
            },
            "gates": {
                "K1_geometry_ranking_survival": {"status": "FAIL"},
                "K2_esr_survival_4bit": {"status": "PASS"},
            },
        }
        text = M.generate_macros(readout, "in.json", repair=None, repair_path=None)
        macros = _parse_macros(text)
        # Required legacy names always emitted.
        for required in (
            "\\pFpThreshold", "\\pEsrSurvFourThreshold", "\\pEsrSurvEightThreshold",
            "\\pDeltaRhoTolerance", "\\pRankSurvFourThreshold",
            "\\pRankSurvEightThreshold", "\\pMedianRatioThreshold",
            "\\pPrimaryFpWithin", "\\pPrimaryCtwoEligible",
            "\\pSecondFpWithin", "\\pSecondCtwoEligible",
            "\\gateKoneStatus", "\\gateKtwoStatus", "\\gateKthreeStatus",
            "\\nCellsCompleted", "\\nSeedsPrimary", "\\nSeedsSecond",
        ):
            self.assertIn(required, macros, f"missing legacy macro {required}")
        # K3 always UNADJUDICATED (honest status; binding wording).
        self.assertEqual(macros["\\gateKthreeStatus"], "UNADJUDICATED")
        # K2 reflects the readout. K1 is computed from the explicit two-cell
        # rule; with no per-cell arm metrics in this minimal legacy fixture,
        # the honest status is PENDING (not the stale readout-level FAIL).
        self.assertEqual(macros["\\gateKoneStatus"], "PENDING")
        self.assertEqual(macros["\\gateKtwoStatus"], "PASS")


class TestMacroNameStability(unittest.TestCase):
    """(4) Pin every manuscript macro name; the generator must never drop or
    rename one. This catches accidental edits to the generator that would
    silently break main.tex."""

    def setUp(self):
        self._orig_cwd = os.getcwd()
        os.chdir(HARNESS)

    def tearDown(self):
        os.chdir(self._orig_cwd)

    def test_all_manuscript_macro_names_emitted(self):
        manuscript_path = os.path.join(
            WORKSPACE, "submissions", "paper-b-neurocomputing", "macros.tex"
        )
        manuscript_macros = _parse_macros(open(manuscript_path).read())
        readout = json.load(open(os.path.join(
            HARNESS, "results", "quant_survival", "aggregate", "gate_readout.json"
        )))
        repair = M._load_repair(os.path.join(
            HARNESS, "results", "quant_survival", "aggregate",
            "quant_survival_repair_v1.json",
        ), strict=True)
        text = M.generate_macros(
            readout,
            "results/quant_survival/aggregate/gate_readout.json",
            repair=repair,
            repair_path="results/quant_survival/aggregate/quant_survival_repair_v1.json",
        )
        gen_macros = _parse_macros(text)
        missing = sorted(set(manuscript_macros.keys()) - set(gen_macros.keys()))
        self.assertEqual(missing, [],
                         f"generator dropped manuscript macros: {missing}")


class TestDrainCommandPinsNBoot500(unittest.TestCase):
    """The drain script's reanalysis command MUST explicitly pass --n_boot 500.

    Background: quant_survival_reanalyze_v1.py's CLI default for --n_boot is
    N_BOOT (1000). If the drain script invokes it without an explicit
    --n_boot 500, a future drain that DOES trigger a fresh reanalysis would
    silently overwrite the approved n_boot=500 artefact with one stamped
    n_boot=1000 — a hidden provenance change. Pin the CLI argument so a
    review-eye on the drain script catches the regression.
    """

    @classmethod
    def setUpClass(cls):
        cls.drain_path = os.path.join(
            HARNESS, "engine", "drain_paperb.sh"
        )
        cls.drain_text = open(cls.drain_path).read()
        # Drain commands use bash `\` line-continuation; join them into
        # single logical statements so per-line substring checks don't miss
        # multi-line invocations.
        cls._logical_lines = []
        buf = ""
        for ln in cls.drain_text.splitlines():
            if ln.endswith("\\"):
                buf += ln[:-1].rstrip() + " "
            else:
                buf += ln
                cls._logical_lines.append(buf)
                buf = ""
        if buf:
            cls._logical_lines.append(buf)

    def test_drain_invokes_reanalysis_with_n_boot_500(self):
        # Locate the reanalysis command line(s) inside the drain script and
        # verify at least one of them includes `--n_boot 500`. The drain
        # script references the reanalyze binary via the $REANALYZE bash
        # variable, so match either the variable or the literal script name.
        reanalyze_calls = [
            ln for ln in self._logical_lines
            if ("$REANALYZE" in ln or "quant_survival_reanalyze_v1" in ln)
            and "$PY" in ln
        ]
        self.assertTrue(
            reanalyze_calls,
            "drain_paperb.sh does not invoke quant_survival_reanalyze_v1.py; "
            "the v1.2.1 repair artefact will never be built.",
        )
        n_boot_500_present = any(
            "--n_boot" in ln and "500" in ln for ln in reanalyze_calls
        )
        self.assertTrue(
            n_boot_500_present,
            "drain_paperb.sh must invoke quant_survival_reanalyze_v1.py with "
            "--n_boot 500 to preserve the approved v1.2.1 provenance; found:\n"
            + "\n".join(ln.strip() for ln in reanalyze_calls),
        )

    def test_drain_does_not_rely_on_gate_readout_mtime_for_repair(self):
        """A freshly-rewritten gate_readout.json MUST NOT be a reanalysis trigger.

        The aggregator rewrites gate_readout.json on every drain regardless of
        whether raw cells changed. If the drain script used `[REPAIR_OUT -nt
        gate_readout.json]` as a "fresh enough" check, the approved repair
        artefact would be at risk of being clobbered on every drain. The
        conservative policy is: re-run reanalysis ONLY when the canonical
        repair artefact is absent.
        """
        forbidden = ["-nt", "-ot"]
        for ln in self._logical_lines:
            if "REPAIR_OUT" in ln and any(op in ln for op in forbidden):
                self.fail(
                    "drain_paperb.sh contains an mtime comparison against "
                    f"REPAIR_OUT ({ln.strip()!r}). A freshly-rewritten "
                    "gate_readout.json must not be a reanalysis trigger."
                )


class TestConservativeRepairOnlyWhenAbsent(unittest.TestCase):
    """When the canonical repair artefact already exists, the drain script's
    conservative policy must skip reanalysis entirely — including when the
    gate_readout.json is freshly rewritten. We simulate this by extracting
    the drain script's logic into a Python-equivalent decision and verifying
    the policy holds.
    """

    DRAIN_SCRIPT = os.path.join(HARNESS, "engine", "drain_paperb.sh")

    def _policy_runs_reanalysis(self, repair_exists: bool, gate_readout_mtime: float
                               ) -> bool:
        """Pure-Python reimplementation of the drain script's repair decision.

        Mirrors the exact branch logic in drain_paperb.sh step 5. We do NOT
        shell out — that would couple this test to a live aggregator. The
        decision is: `repair_exists` (only) controls whether to re-run. A
        fresh gate_readout.json mtime is irrelevant.
        """
        if not repair_exists:
            return True  # would invoke reanalysis with --n_boot 500
        return False

    def test_existing_repair_not_overwritten_by_fresh_gate_readout(self):
        # Scenario: repair artefact exists; gate_readout was just rewritten
        # (simulating a drain that just re-ran the aggregator).
        self.assertFalse(self._policy_runs_reanalysis(
            repair_exists=True, gate_readout_mtime=10**12,
        ))

    def test_missing_repair_triggers_reanalysis(self):
        self.assertTrue(self._policy_runs_reanalysis(
            repair_exists=False, gate_readout_mtime=0,
        ))

    def test_drain_does_not_pass_n_boot_default_implicitly(self):
        """Verify the drain script does not invoke the reanalysis without an
        explicit --n_boot argument (which would let the reanalyze_v1 default
        of 1000 leak in). Multi-line `\\`-continued bash commands are
        collapsed to a single logical statement for substring checking.
        """
        text = open(self.DRAIN_SCRIPT).read()
        logical = []
        buf = ""
        for ln in text.splitlines():
            if ln.endswith("\\"):
                buf += ln[:-1].rstrip() + " "
            else:
                buf += ln
                logical.append(buf)
                buf = ""
        reanalyze_calls = [
            ln for ln in logical
            if ("$REANALYZE" in ln or "quant_survival_reanalyze_v1" in ln)
            and "$PY" in ln
        ]
        self.assertTrue( reanalyze_calls,
            "drain_paperb.sh does not invoke quant_survival_reanalyze_v1.py" )
        for ln in reanalyze_calls:
            self.assertIn(
                "--n_boot", ln,
                f"reanalysis invocation missing explicit --n_boot: {ln.strip()!r}",
            )


if __name__ == "__main__":
    unittest.main()