"""quant_survival_bridge.py — Paper B, Track-2 BRIDGE cell (C4 reconciliation with Hase 2024 /
2407.06483): install ALL n_edits, apply full-model REAL-bit quantization (GPTQ/AWQ), measure
aggregate esr + signed damage drift + Strict-Edit-Locality F1 (the bridge measurement from
Composable Interventions, arXiv 2407.06483).

STATUS: STUB ONLY. Track-2 driver is gated by `engine/PAPERB_DRAIN.ok` AND user ratification of
the Track-2 protocol. Real GPTQ calls are ask-first per project rules.

What this file is:
  * The Track-1 phase-1 runner's SIBLING (experiments/quant_survival_phase1.py), not an
    extension of it. Same two-mode CLI (--selftest CPU-only structural gate / --run GPU cell),
    same output-dir layout under results/quant_survival/<cell>/, same JSON table filename
    (QS_phase1_table.json) and schema SHAPE, so quant_survival_analyze.py can later be extended
    to parse Track-2 tables WITHOUT this stub having changed the aggregator.
  * The dataclass + schema contracts the future Track-2 driver must satisfy (PAPERB-PROSE-SHELL
    §4.3 protocol: install all n=200 edits -> quantize the whole model with the real kernel ->
    aggregate esr per batch, signed damage drift, Strict-Edit-Locality F1; §6.3 C4 two-column
    reconciliation row: replicate Hase's F1 degradation while our signed damage rank survives).

What this file is NOT:
  * Not a runnable driver. --run raises NotImplementedError. No model is instantiated, no GPU is
    touched, no GPTQ/AWQ/AutoGPTQ/AutoAWQ import happens at module level (lazy, ask-first, same
    pattern as quant_survival_track15.py). No network calls anywhere.

METRIC DISCIPLINE (non-negotiable, inherited from Track-1): damage = signed within-probe
damage_logit; signed Spearman, NEVER AUROC; ROME value-opt stays fp32; results quarantined to
results/quant_survival/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

HARNESS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HARNESS)

SCHEMA_VERSION = "qs.bridge.v1"
EXPERIMENT = "quant_survival_bridge"
TABLE_NAME = "QS_phase1_table.json"  # SAME filename/shape as Track-1 (aggregator future-extension)
GATE_FILE = os.path.join(HARNESS, "engine", "PAPERB_DRAIN.ok")

# Track-1 table shape is mirrored from the AUTHORITATIVE writer: analyze() in
# quant_survival_phase1.py + the key sets in quant_survival_schema.py (PHASE1_REQUIRED_*).


# ============================================================ lazy, ask-first quant backends
def _require_gptq():
    """AutoGPTQ is an ask-first install (standing project rule). NEVER imported at module level."""
    import importlib.util
    if importlib.util.find_spec("auto_gptq") is None:
        raise RuntimeError(
            "Track-2 GPTQ requires AutoGPTQ, which is not installed. "
            "Install is ask-first per project rules; run the driver only after user approval."
        )


def _require_awq():
    """AutoAWQ is an ask-first install (standing project rule). NEVER imported at module level."""
    import importlib.util
    if importlib.util.find_spec("awq") is None:
        raise RuntimeError(
            "Track-2 AWQ requires AutoAWQ, which is not installed. "
            "Install is ask-first per project rules; run the driver only after user approval."
        )


# ============================================================ the bridge measurement (STUB)
def StrictEditLocalityF1():
    """Strict-Edit-Locality F1 — the C4 bridge measurement from Composable Interventions
    (Hase 2024, arXiv 2407.06483).

    Contract (per PAPERB-PROSE-SHELL §4.3/§6.3): on the fully-installed (all n_edits), full-model
    real-bit-quantized model, measure whether each edit's behavior is localized to its intended
    target under the STRICT criterion of 2407.06483, aggregate to a single F1, and place it in a
    two-column row next to our signed damage-rank survival: F1 degrades (replication) while the
    relative geometric ranking survives (reconciliation — measured, not rhetorical).

    NOT IMPLEMENTED. The exact strict-locality prompt suite, gold-label construction, and
    batch-decay protocol are part of the Track-2 protocol that awaits user ratification; the
    driver is additionally gated on `engine/PAPERB_DRAIN.ok` (Track-1 drain).
    """
    raise NotImplementedError("Track-2 not approved; needs Track-1 drain first")


# ============================================================ result schema (mirrors Track-1)
@dataclass
class BridgeCellResult:
    """Track-2 bridge-cell table. Top-level fields MIRROR the Track-1 phase-1 table (the shape
    quant_survival_analyze.py parses: quant_survival_phase1.analyze() output +
    quant_survival_schema.PHASE1_REQUIRED_TABLE_KEYS / _ARM_KEYS / _C3_KEYS), so the aggregator
    can be extended later to read Track-2 tables with zero schema drift. `experiment` differs
    ("quant_survival_bridge") so Track-2 tables are skipped until that extension lands — the
    current aggregator is intentionally NOT touched by this stub.

    Bridge-specific C4 payload lives in `bridge` and is TBD pending the ratified Track-2
    protocol; the Track-1-mirrored blocks (esr / mechanism_tie / arms / bin_width_mechanism_C3 /
    generation_checks) keep the Track-1 key names verbatim inside."""

    # ---- identity/meta (Track-1 key names verbatim) ----
    experiment: str = EXPERIMENT
    schema_version: str = SCHEMA_VERSION
    created: str = ""
    model: str = ""
    layer: int = 0
    editor: str = "memit"          # bridge cell per prose shell §6.3 is MEMIT + GPTQ-4bit
    n_edits: int = 200
    n_probes: int = 200
    seed: int = 0
    schemes: List[str] = field(default_factory=lambda: ["gptq4"])
    codec: str = "real"            # real kernels only for a headline number; ask-first installs
    blocksize: int = 64
    fullmodel_cache: Optional[str] = None
    edited_layers: Optional[List[int]] = None
    c2_scope: str = ("Track-2 bridge cell is C4-ONLY (reconciliation with 2407.06483); it never "
                     "carries C2 geometry-ranking claims — those are Track-1 phase-1 cells.")
    quant_note: str = ("full-model REAL-bit quantization AFTER installing all n_edits "
                       "(deployment order); GPTQ/AWQ backends are ask-first lazy imports")
    damage_metric_note: str = "signed damage_logit = pre_l(fp32 unedited) - post_l; never AUROC"

    # ---- Track-1-mirrored measurement blocks (same key names inside) ----
    esr: Dict[str, Any] = field(default_factory=dict)              # mean_esr_fp32, n_edits_worked_fp32, ...
    mechanism_tie: Dict[str, Any] = field(default_factory=dict)    # rho_keycos_damage_fp32_*, ...
    arms: Dict[str, Any] = field(default_factory=dict)             # per-arm dicts w/ Track-1 arm keys
    bin_width_mechanism_C3: Dict[str, Any] = field(default_factory=dict)  # F_above_bin, median_ratio, ...
    generation_checks: Dict[str, Any] = field(default_factory=dict)

    # ---- Track-2-specific C4 payload (TBD — filled by the ratified Track-2 driver) ----
    bridge: Dict[str, Any] = field(default_factory=lambda: {
        "protocol": "install_all_then_quantize_full_model",
        "strict_edit_locality_f1": None,       # TBD: StrictEditLocalityF1 aggregate (C4 bridge)
        "aggregate_esr": None,                 # TBD: post-quant esr over the installed edit batch
        "signed_damage_drift": None,           # TBD: drift of the signed within-probe damage field
        "esr_batch_decay": None,               # TBD: per-batch decay over the install sequence
        "reconciliation_row": None,            # TBD: two-column F1-vs-rank-survival row (§6.3)
    })

    def as_table(self) -> Dict[str, Any]:
        return asdict(self)

    def write(self, out_dir: str, table_out: Optional[str] = None) -> str:
        os.makedirs(out_dir, exist_ok=True)
        out = table_out or os.path.join(out_dir, TABLE_NAME)
        tmp = out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.as_table(), f, indent=2)
        os.replace(tmp, out)
        return out


def build_placeholder_table() -> Dict[str, Any]:
    """A schema-SHAPE-complete placeholder table (values are None/empty), used ONLY by the
    selftest to prove the stub's dataclass still matches the Track-1-required key sets. Never
    written to results/ by any code path in this stub."""
    from experiments.quant_survival_schema import (
        PHASE1_REQUIRED_TABLE_KEYS, PHASE1_REQUIRED_ARM_KEYS, PHASE1_REQUIRED_C3_KEYS,
    )
    t = BridgeCellResult(
        created=time.strftime("%Y-%m-%dT%H:%M:%S"),
        model="placeholder", layer=12, editor="memit",
        schemes=["gptq4"], codec="real",
        esr={"mean_esr_fp32": None, "n_edits_worked_fp32": None},
        mechanism_tie={"rho_keycos_damage_fp32_pooled": None,
                       "rho_keycos_damage_fp32_within_probe": None,
                       "within_probe_n_cols": None,
                       "fp32_law_gate_c2_eligible": None,
                       "fp32_pooled_ci95_bootstrap_edits": [None, None]},
        arms={"gptq4_full_model": {k: None for k in PHASE1_REQUIRED_ARM_KEYS}},
        bin_width_mechanism_C3={"gptq4": {k: None for k in PHASE1_REQUIRED_C3_KEYS}},
        generation_checks={},
    )
    table = t.as_table()
    missing = PHASE1_REQUIRED_TABLE_KEYS - set(table.keys())
    assert not missing, f"placeholder table missing Track-1 table keys {missing}"
    return table


# ============================================================ CPU self-test (NO CUDA, structural)
def selftest() -> bool:
    """CPU-only STRUCTURAL gate — schema-shape check + stub-behavior check. NO model, NO GPU,
    NO GPTQ/AWQ import, NO network. Same fingerprint as quant_survival_phase1.py --selftest."""
    print("[selftest] quant-survival TRACK-2 BRIDGE (stub) — CPU (NO CUDA)", flush=True)

    print("[selftest] (a) schema shape: dataclass -> table covers Track-1 required keys ...", flush=True)
    from experiments.quant_survival_schema import (
        PHASE1_REQUIRED_TABLE_KEYS, PHASE1_REQUIRED_ARM_KEYS, PHASE1_REQUIRED_C3_KEYS,
    )
    table = build_placeholder_table()
    assert table["experiment"] == EXPERIMENT and table["schema_version"] == SCHEMA_VERSION
    assert not (PHASE1_REQUIRED_TABLE_KEYS - set(table.keys())), "table keys drifted from Track-1"
    for arm_name, arm in table["arms"].items():
        assert not (PHASE1_REQUIRED_ARM_KEYS - set(arm.keys())), f"arm {arm_name} key drift"
    for scheme, c in table["bin_width_mechanism_C3"].items():
        assert not (PHASE1_REQUIRED_C3_KEYS - set(c.keys())), f"C3 {scheme} key drift"
    # JSON round-trip (the on-disk table must stay plain-JSON serializable)
    json.loads(json.dumps(table))
    print("[selftest]   schema OK — table/arm/C3 keys match Track-1 (aggregator future-ready)", flush=True)

    print("[selftest] (b) stub behavior: StrictEditLocalityF1 + --run are hard-gated ...", flush=True)
    try:
        StrictEditLocalityF1()
        raise AssertionError("StrictEditLocalityF1 must raise, not return")
    except NotImplementedError as ex:
        assert str(ex) == "Track-2 not approved; needs Track-1 drain first", f"wrong message: {ex}"
    try:
        run(argparse.Namespace())
        raise AssertionError("run() must raise, not return")
    except NotImplementedError as ex:
        assert "pending Track-1 drain + user ratification" in str(ex), f"wrong message: {ex}"
    print("[selftest]   stub gates OK — both raise NotImplementedError with the frozen messages", flush=True)

    print("[selftest] (c) ask-first lazy backends: no top-level auto_gptq/awq import ...", flush=True)
    assert "auto_gptq" not in sys.modules and "awq" not in sys.modules, \
        "a quant backend was imported at module level — must stay lazy/ask-first"
    gptq_err = awq_err = None
    try:
        _require_gptq()
    except RuntimeError as ex:
        gptq_err = str(ex)
    try:
        _require_awq()
    except RuntimeError as ex:
        awq_err = str(ex)
    # Either the backend is genuinely absent (RuntimeError with the ask-first text) or the user
    # already approved+installed it (no error) — both are legal; a SILENT import is not.
    if gptq_err is not None:
        assert "ask-first" in gptq_err, f"_require_gptq message drifted: {gptq_err}"
    if awq_err is not None:
        assert "ask-first" in awq_err, f"_require_awq message drifted: {awq_err}"
    print(f"[selftest]   backends OK — gptq: {'absent (ask-first msg OK)' if gptq_err else 'INSTALLED (user-approved)'}; "
          f"awq: {'absent (ask-first msg OK)' if awq_err else 'INSTALLED (user-approved)'}", flush=True)

    print("\n[selftest] ALL CHECKS PASSED (schema shape + stub gates + lazy backends) "
          "[bridge driver itself NOT runnable — Track-2 pending]", flush=True)
    return True


# ============================================================ GPU run (STUB — never runs)
def run(_args):
    """Track-2 bridge-cell driver body. NOT BUILT. Gated by engine/PAPERB_DRAIN.ok AND user
    ratification of the Track-2 protocol; GPTQ/AWQ installs are ask-first. When ratified, this
    body will: install ALL n_edits (batched MEMIT per §6.3), full-model GPTQ-4bit via
    _require_gptq() (lazy), measure aggregate esr + signed damage drift + StrictEditLocalityF1,
    and write TABLE_NAME via BridgeCellResult.write()."""
    del _args  # stub: arg accepted for caller parity with Track-1, discarded until ratified
    raise NotImplementedError("Track-2 driver pending Track-1 drain + user ratification")


# ============================================================ CLI
def main():
    ap = argparse.ArgumentParser(description="Quant-survival Paper-B Track-2 bridge cell (STUB).")
    ap.add_argument("--selftest", action="store_true",
                    help="CPU structural self-test (no GPU, no quant backends).")
    ap.add_argument("--run", action="store_true",
                    help="GPU bridge-cell run — STUB: raises NotImplementedError until Track-2 "
                         "is ratified and engine/PAPERB_DRAIN.ok exists.")
    ap.add_argument("--model", default=os.path.join(HARNESS, "data", "models", "Llama-3.2-1B"))
    ap.add_argument("--data", default=os.path.join(HARNESS, "data", "counterfact.json"))
    ap.add_argument("--editor", choices=["rome", "memit", "alpha"], default="memit",
                    help="bridge cell per prose shell §6.3 is MEMIT + GPTQ-4bit")
    ap.add_argument("--n_edits", type=int, default=200)
    ap.add_argument("--n_probes", type=int, default=200)
    ap.add_argument("--layer", default="12")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--schemes", default="gptq4", help="comma list; real-bit schemes only (gptq4/awq4)")
    ap.add_argument("--codec", choices=["real"], default="real",
                    help="real kernels ONLY — a simulated bridge number is not quotable")
    ap.add_argument("--memit_layers", default="auto", help="MEMIT layer span ('auto'=4 ending at --layer)")
    ap.add_argument("--n_perm", type=int, default=1000, help="permutation-null draws (prereg N>=1000)")
    ap.add_argument("--n_boot", type=int, default=1000, help="bootstrap resamples for the rho CI")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--out_dir", default=os.path.join(HARNESS, "results", "quant_survival"))
    ap.add_argument("--table_out", default=None)
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if args.run:
        run(args)
        return
    ap.error("nothing to do: pass --selftest or --run")


if __name__ == "__main__":
    main()
