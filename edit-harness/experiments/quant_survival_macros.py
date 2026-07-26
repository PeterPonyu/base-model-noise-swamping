"""quant_survival_macros.py — generate LaTeX macros for Paper B.

Reads:
  - results/quant_survival/aggregate/gate_readout.json (legacy aggregator output)
  - results/quant_survival/aggregate/quant_survival_repair_v1.json
    (immutable reanalysis sidecar; required for v1.2.1 multilevel fields)

Writes:
  - submissions/paper-b-neurocomputing/macros.tex

CPU-only; no GPU; no edits to source results. The reanalysis artefact IS the
canonical source for the v1.2.1 multilevel fields (multilevel rank survival,
Qwen CI widths, base-quant noise, repair version / n_boot / sha256 / sidecar).
The legacy aggregator output remains the source for thresholds, primary / second
aggregate cells, C3 median-ratio, and the kill-gate status readouts.

The generator fails closed: if --strict_repair is passed (the default in the
real drain path) and the canonical repair artefact is missing or incompatible,
the generator exits non-zero rather than emitting stale macros.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import statistics
import sys
from typing import Any, Dict, List, Optional, Tuple

HARNESS = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IN = os.path.join(
    HARNESS, "..", "results", "quant_survival", "aggregate", "gate_readout.json"
)
DEFAULT_REPAIR = os.path.join(
    HARNESS, "..", "results", "quant_survival", "aggregate",
    "quant_survival_repair_v1.json",
)
DEFAULT_OUT = os.path.join(
    HARNESS, "..", "..", "submissions", "paper-b-neurocomputing", "macros.tex"
)

# Repair artefact schema constants — used to detect incompatibility.
REPAIR_SCHEMA_VERSION = "1.2.1"
EXPECTED_REPAIR_MODULE = "quant_survival_reanalyze_v1"


def _fmt(x, ndigits: int = 3) -> str:
    if x is None or (isinstance(x, float) and (x != x)):
        return "NAN"
    if isinstance(x, bool):
        return "1" if x else "0"
    if isinstance(x, int):
        return str(x)
    return f"{x:.{ndigits}f}"


_SCHEME_LATEX = {"nf4": "nfFour", "int8": "intEight"}


def _latex_safe(name: str) -> str:
    """Replace digits in scheme names so resulting LaTeX command names are valid."""
    out = name
    for digit, word in (("4", "Four"), ("8", "Eight")):
        out = out.replace(digit, word)
    return out


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_sidecar(canonical_path: str) -> Optional[str]:
    """Locate the immutable sidecar whose stem begins with the canonical name.

    Sidecars follow the convention `<stem>__<sha256[:16]>.<ext>` written by
    `quant_survival_reanalyze_v1._atomic_write_canonical`. If multiple sidecars
    exist, prefer the one whose embedded prefix matches the canonical file's
    own sha256[:16] (they should always match by construction).
    """
    base = os.path.basename(canonical_path)
    stem, _ = os.path.splitext(base)
    d = os.path.dirname(os.path.abspath(canonical_path))
    sidecars = sorted(glob.glob(os.path.join(d, f"{stem}__*.json")))
    if not sidecars:
        return None
    if not os.path.isfile(canonical_path):
        return sidecars[-1]
    try:
        canonical_sha = _sha256_file(canonical_path)[:16]
    except OSError:
        return sidecars[-1]
    for sc in sidecars:
        if f"__{canonical_sha}" in os.path.basename(sc):
            return sc
    return sidecars[-1]


def _load_repair(repair_path: str, strict: bool) -> Optional[Dict[str, Any]]:
    """Load the repair artefact, validate v1.2.1 schema, return payload or None.

    In strict mode (the drain-path default), incompatibility is fatal.
    In non-strict mode (smoke / re-write-the-legacy-macros path), we emit a
    warning to stderr and return None so the legacy block can be regenerated.
    """
    if not repair_path or not os.path.isfile(repair_path):
        if strict:
            sys.stderr.write(
                f"[quant_survival_macros] FAIL-CLOSED: repair artefact not found: "
                f"{repair_path}\n"
            )
            sys.exit(2)
        sys.stderr.write(
            f"[quant_survival_macros] WARN: repair artefact not found: {repair_path}; "
            "emitting legacy-only macros\n"
        )
        return None
    try:
        repair = json.load(open(repair_path))
    except Exception as e:
        if strict:
            sys.stderr.write(
                f"[quant_survival_macros] FAIL-CLOSED: repair artefact unparseable "
                f"({e!r}): {repair_path}\n"
            )
            sys.exit(2)
        return None
    prov = repair.get("module_provenance", {}) or {}
    if prov.get("version") != REPAIR_SCHEMA_VERSION:
        msg = (
            f"repair version {prov.get('version')!r} != expected "
            f"{REPAIR_SCHEMA_VERSION!r}"
        )
        if strict:
            sys.stderr.write(f"[quant_survival_macros] FAIL-CLOSED: {msg}\n")
            sys.exit(2)
        sys.stderr.write(f"[quant_survival_macros] WARN: {msg}\n")
        return None
    if prov.get("module") != EXPECTED_REPAIR_MODULE:
        msg = (
            f"repair module {prov.get('module')!r} != expected "
            f"{EXPECTED_REPAIR_MODULE!r}"
        )
        if strict:
            sys.stderr.write(f"[quant_survival_macros] FAIL-CLOSED: {msg}\n")
            sys.exit(2)
        return None
    cells = repair.get("cells")
    if not isinstance(cells, list) or not cells:
        if strict:
            sys.stderr.write(
                "[quant_survival_macros] FAIL-CLOSED: repair.cells missing or "
                "empty (expected list of >=1 cells)\n"
            )
            sys.exit(2)
        return None
    return repair


def _find_cell(repair: Dict[str, Any], slug: str, editor: str, layer: int
               ) -> Optional[Dict[str, Any]]:
    for c in repair.get("cells", []) or []:
        if c.get("slug") == slug and c.get("editor") == editor and c.get("layer") == layer:
            return c
    return None


def _get_arm(cell: Dict[str, Any], arm: str) -> Optional[Dict[str, Any]]:
    return (cell.get("arms") or {}).get(arm)


def _nf4fm_rank_point(cell_or_arm: Dict[str, Any], source: str) -> Optional[float]:
    """Return the NF4 full-model rank-survival point for a cell or arm.

    `source` is 'repair' (uses repair['flat_rank']['point']) or 'readout'
    (uses legacy gate readout's
    rho_damage_fp32_vs_arm_rank_survival_mean). Returns None if not derivable.
    """
    if source == "repair":
        arm = _get_arm(cell_or_arm, "nf4dq_full_model") or {}
        flat = arm.get("flat_rank") or {}
        pt = flat.get("point")
        if pt is None or not isinstance(pt, (int, float)):
            return None
        return float(pt)
    # legacy gate readout cell with arms dict
    arm = (cell_or_arm.get("arms") or {}).get("nf4dq_full_model") or {}
    pt = arm.get("rho_damage_fp32_vs_arm_rank_survival_mean")
    if pt is None or not isinstance(pt, (int, float)):
        return None
    return float(pt)


def _compute_k1_status(readout: Dict[str, Any],
                       repair: Optional[Dict[str, Any]]) -> str:
    """K1 (geometry-ranking survival) rule.

    K1 fires (FAIL) ONLY if BOTH validated-law ROME cells fail their NF4
    full-model rank-survival threshold (default 0.85). If one passes and
    the other fails, the rule does NOT fire — K1 reports PASS (the
    phenomenon narrows rather than dies). With the current canonical data:
    Llama-3.2-1B ROME L12 = 0.904 PASS, Llama-3.2-3B ROME L24 = 0.680
    FAIL → K1 = PASS.

    Source priority: repair artefact (canonical) > legacy gate readout.
    Returns 'PENDING' if either validated cell is missing.
    """
    th = float(readout.get("thresholds", {}).get("rank_survival_4bit", 0.85))
    if repair is not None:
        pri = _find_cell(repair, "llama1b", "rome", 12)
        sec = _find_cell(repair, "llama3b", "rome", 24)
        p = _nf4fm_rank_point(pri, "repair") if pri else None
        s = _nf4fm_rank_point(sec, "repair") if sec else None
    else:
        pri = (readout.get("cells") or {}).get("Llama-3.2-1B_rome_L12")
        sec = (readout.get("cells") or {}).get("Llama-3.2-3B_rome_L24")
        p = _nf4fm_rank_point(pri, "readout") if pri else None
        s = _nf4fm_rank_point(sec, "readout") if sec else None
    if p is None or s is None:
        return "PENDING"
    p_pass = p >= th
    s_pass = s >= th
    if (not p_pass) and (not s_pass):
        return "FAIL"  # both fail → K1 fires
    return "PASS"  # at least one passes → rule does not fire


# Mapping from legacy readout cell key to repair (slug, editor, layer).
_LEGACY_KEY_TO_REPAIR = {
    "Llama-3.2-1B_rome_L12": ("llama1b", "rome", 12),
    "Llama-3.2-1B_memit_L12": ("llama1b", "memit", 12),
    "Llama-3.2-1B_alpha_L12": ("llama1b", "alpha", 12),
    "Llama-3.2-3B_rome_L24": ("llama3b", "rome", 24),
    "Llama-3.2-3B_memit_L24": ("llama3b", "memit", 24),
    "Llama-3.2-3B_alpha_L24": ("llama3b", "alpha", 24),
    "Qwen2.5-1.5B_rome_L21": ("qwen15b", "rome", 21),
    "Qwen2.5-1.5B_memit_L21": ("qwen15b", "memit", 21),
    "Qwen2.5-1.5B_alpha_L21": ("qwen15b", "alpha", 21),
}

_TABLE_CELL_SPECS = (
    ("llama1b", "rome", 12, "LOneRome"),
    ("llama1b", "memit", 12, "LOneMemit"),
    ("llama1b", "alpha", 12, "LOneAlpha"),
    ("llama3b", "rome", 24, "LThreeRome"),
    ("llama3b", "memit", 24, "LThreeMemit"),
    ("llama3b", "alpha", 24, "LThreeAlpha"),
    ("qwen15b", "rome", 21, "QOneRome"),
    ("qwen15b", "memit", 21, "QOneMemit"),
    ("qwen15b", "alpha", 21, "QOneAlpha"),
)

_TABLE_ARM_SPECS = (
    ("nf4dq_edited_layer", "NfFourEL"),
    ("nf4dq_full_model", "NfFourFM"),
    ("int8_edited_layer", "IntEightEL"),
    ("int8_full_model", "IntEightFM"),
)


def _repair_cell_for_legacy_key(repair: Optional[Dict[str, Any]],
                                legacy_key: str) -> Optional[Dict[str, Any]]:
    """Look up the repair cell that corresponds to a legacy readout cell key.

    Returns None if the repair artefact is absent or the key is unmapped.
    """
    if repair is None:
        return None
    coord = _LEGACY_KEY_TO_REPAIR.get(legacy_key)
    if coord is None:
        return None
    return _find_cell(repair, *coord)


def _emit_multilevel_block(lines: List[str], cell: Dict[str, Any], arm: str,
                           prefix: str, header: str, range_caveat: str = ""
                           ) -> None:
    """Emit one per-arm multilevel rank-survival block (Flat/Within/Sign/...).

    prefix is the macro prefix (e.g. 'pPri', 'pSec', 'pQwen'). `arm` is the
    arm name (e.g. 'nf4dq_full_model'). Values are formatted with the 3-decimal
    `_fmt` so the printed values match the manuscript convention exactly.
    """
    a = _get_arm(cell, arm) or {}
    flat = a.get("flat_rank") or {}
    within = a.get("within_probe_rank") or {}
    elr = a.get("edit_level_ranks") or {}
    sign = elr.get("signed_mean") or {}
    absmean = elr.get("absmean") or {}
    l2 = elr.get("l2") or {}
    abs_esr = a.get("absolute_quantized_esr") or {}
    cond = a.get("conditional_survival_given_fp32_worked") or {}
    base = a.get("base_quant_noise_mean_abs")

    rng_flat = flat.get("range_min_max") or [None, None]
    rng_within = within.get("range_min_max") or [None, None]
    flat_rng = f"{rng_flat[0]:.4f}-{rng_flat[1]:.4f}" if (
        isinstance(rng_flat[0], (int, float))
        and isinstance(rng_flat[1], (int, float))) else ""
    within_rng = f"{rng_within[0]:.4f}-{rng_within[1]:.4f}" if (
        isinstance(rng_within[0], (int, float))
        and isinstance(rng_within[1], (int, float))) else ""
    rng_suffix = range_caveat

    lines.append(f"% {header}")
    flat_label = f"% flat, {rng_suffix} (range {flat_rng})" if rng_suffix else f"% flat (range {flat_rng})"
    within_label = f"% within-probe, {rng_suffix} (range {within_rng})" if rng_suffix else f"% within-probe (range {within_rng})"
    lines.append(f"\\newcommand{{\\{prefix}FNFMFlat}}{{{_fmt(flat.get('point'))}}}"
                 f"                  {flat_label}")
    lines.append(f"\\newcommand{{\\{prefix}FNFMWithin}}{{{_fmt(within.get('point'))}}}"
                 f"                {within_label}")
    sign_ci = sign.get("ci95") or [None, None]
    lines.append(f"\\newcommand{{\\{prefix}FNFMSignPt}}{{{_fmt(sign.get('point'))}}}"
                 f"               % edit-level signed-mean point")
    lines.append(f"\\newcommand{{\\{prefix}FNFMSignLo}}{{{_fmt(sign_ci[0])}}}"
                 f"               % edit-level signed-mean CI95 low")
    lines.append(f"\\newcommand{{\\{prefix}FNFMSignHi}}{{{_fmt(sign_ci[1])}}}"
                 f"               % edit-level signed-mean CI95 high")
    abs_ci = absmean.get("ci95") or [None, None]
    lines.append(f"\\newcommand{{\\{prefix}FNFMAbsPt}}{{{_fmt(absmean.get('point'))}}}"
                 f"                % edit-level absmean point")
    lines.append(f"\\newcommand{{\\{prefix}FNFMAbsLo}}{{{_fmt(abs_ci[0])}}}"
                 f"                % edit-level absmean CI95 low")
    lines.append(f"\\newcommand{{\\{prefix}FNFMAbsHi}}{{{_fmt(abs_ci[1])}}}"
                 f"                % edit-level absmean CI95 high")
    lines.append(f"\\newcommand{{\\{prefix}FNFMLtwoPt}}{{{_fmt(l2.get('point'))}}}"
                 f"               % edit-level L2 point")
    lines.append(f"\\newcommand{{\\{prefix}FNFMBase}}{{{_fmt(base)}}}"
                 f"                 % base-quant-noise mean |D|")
    abs_esr_ci = abs_esr.get("ci95") or [None, None]
    lines.append(f"\\newcommand{{\\{prefix}FNFMAbsEsrLo}}{{{_fmt(abs_esr_ci[0])}}}"
                 f"             % absolute quantized ESR CI95 low")
    lines.append(f"\\newcommand{{\\{prefix}FNFMAbsEsrHi}}{{{_fmt(abs_esr_ci[1])}}}"
                 f"             % absolute quantized ESR CI95 high")
    cond_ci = cond.get("ci95") or [None, None]
    lines.append(f"\\newcommand{{\\{prefix}FNFMCondLo}}{{{_fmt(cond_ci[0])}}}"
                 f"               % conditional survival CI95 low")
    lines.append(f"\\newcommand{{\\{prefix}FNFMCondHi}}{{{_fmt(cond_ci[1])}}}"
                 f"               % conditional survival CI95 high")


def _mean_numeric(values: Any) -> Optional[float]:
    if not isinstance(values, list):
        return None
    clean = [float(v) for v in values if isinstance(v, (int, float))]
    return statistics.fmean(clean) if clean else None


def _emit_dense_table_macros(lines: List[str], repair: Dict[str, Any]) -> None:
    """Emit all load-bearing numbers used by the dense Paper-B tables."""
    lines.extend(["", "% ---- Dense Paper-B tables (canonical v1.2.1) ----"])
    rise_pcts: List[float] = []
    for slug, editor, layer, cell_prefix in _TABLE_CELL_SPECS:
        cell = _find_cell(repair, slug, editor, layer)
        if cell is None:
            raise ValueError(
                f"repair artefact missing dense-table cell: {slug}/{editor}/L{layer}"
            )
        for arm_name, arm_prefix in _TABLE_ARM_SPECS:
            arm = _get_arm(cell, arm_name)
            if arm is None:
                raise ValueError(
                    f"repair artefact missing dense-table arm: {slug}/{editor}/{arm_name}"
                )
            absolute = (arm.get("absolute_quantized_esr") or {}).get("point")
            conditional = (
                arm.get("conditional_survival_given_fp32_worked") or {}
            ).get("point")
            lines.append(
                f"\\newcommand{{\\pGrid{cell_prefix}{arm_prefix}Abs}}"
                f"{{{_fmt(absolute)}}}   % absolute ESR"
            )
            lines.append(
                f"\\newcommand{{\\pGrid{cell_prefix}{arm_prefix}Cond}}"
                f"{{{_fmt(conditional)}}}   % conditional ESR survival"
            )

        checks = cell.get("generation_checks") or {}
        fp32 = _mean_numeric((checks.get("fp32") or {}).get("per_seed_perplexity_mean"))
        nf4_el = _mean_numeric(
            (checks.get("nf4dq_edited_layer") or {}).get("per_seed_perplexity_mean")
        )
        nf4_fm = _mean_numeric(
            (checks.get("nf4dq_full_model") or {}).get("per_seed_perplexity_mean")
        )
        rise_pct = (100.0 * (nf4_fm / fp32 - 1.0)) if (
            isinstance(fp32, (int, float)) and fp32 != 0
            and isinstance(nf4_fm, (int, float))
        ) else None
        if isinstance(rise_pct, (int, float)):
            rise_pcts.append(float(rise_pct))
        lines.append(
            f"\\newcommand{{\\pPpl{cell_prefix}Fp}}{{{_fmt(fp32, 2)}}}"
            "   % mean FP32-edited generation PPL over seeds"
        )
        lines.append(
            f"\\newcommand{{\\pPpl{cell_prefix}NfFourEL}}{{{_fmt(nf4_el, 2)}}}"
            "   % mean NF4 edited-layer generation PPL over seeds"
        )
        lines.append(
            f"\\newcommand{{\\pPpl{cell_prefix}NfFourFM}}{{{_fmt(nf4_fm, 2)}}}"
            "   % mean NF4 full-model generation PPL over seeds"
        )
        lines.append(
            f"\\newcommand{{\\pPpl{cell_prefix}RisePct}}{{{_fmt(rise_pct, 1)}}}"
            "   % NF4 full-model PPL rise vs FP32-edited, percent"
        )

    if len(rise_pcts) != len(_TABLE_CELL_SPECS):
        raise ValueError(
            "repair artefact missing one or more generation-check PPL rise values"
        )
    lines.append(
        f"\\newcommand{{\\pPplRisePctMin}}{{{_fmt(min(rise_pcts), 2)}}}"
        "   % minimum NF4 full-model PPL rise across dense-table cells, percent"
    )
    lines.append(
        f"\\newcommand{{\\pPplRisePctMax}}{{{_fmt(max(rise_pcts), 2)}}}"
        "   % maximum NF4 full-model PPL rise across dense-table cells, percent"
    )

    for slug, layer, cell_prefix in (
        ("llama1b", 12, "LOneRome"),
        ("llama3b", 24, "LThreeRome"),
    ):
        cell = _find_cell(repair, slug, "rome", layer)
        if cell is None:
            raise ValueError(f"repair artefact missing validated ROME cell: {slug}/L{layer}")
        for arm_name, arm_prefix in _TABLE_ARM_SPECS:
            arm = _get_arm(cell, arm_name) or {}
            flat = (arm.get("flat_rank") or {}).get("point")
            within = (arm.get("within_probe_rank") or {}).get("point")
            edit = ((arm.get("edit_level_ranks") or {}).get("signed_mean") or {}).get("point")
            lines.append(
                f"\\newcommand{{\\pRank{cell_prefix}{arm_prefix}Flat}}"
                f"{{{_fmt(flat)}}}   % legacy flat rank survival"
            )
            lines.append(
                f"\\newcommand{{\\pRank{cell_prefix}{arm_prefix}Within}}"
                f"{{{_fmt(within)}}}   % within-probe rank survival"
            )
            lines.append(
                f"\\newcommand{{\\pRank{cell_prefix}{arm_prefix}Edit}}"
                f"{{{_fmt(edit)}}}   % edit-level signed-mean rank survival"
            )

    lines.append(
        "\\newcommand{\\gateKoneDisplayStatus}{NARROW-FAIL}"
        "   % one of two validated ROME NF4 full-model cells fails"
    )


def generate_macros(readout: Dict[str, Any], source_path: str,
                    repair: Optional[Dict[str, Any]] = None,
                    repair_path: Optional[str] = None) -> str:
    th = readout.get("thresholds", {})
    cells = readout.get("cells", {})
    gates = readout.get("gates", {})

    # Primary cell: Llama-1B ROME L12 (must exist after Phase-1)
    primary = cells.get("Llama-3.2-1B_rome_L12", {})
    # Mandatory second cell: Llama-3B ROME L24
    second = cells.get("Llama-3.2-3B_rome_L24", {})

    # Provenance header — assemble from repair artefact when available.
    if repair is not None:
        prov = repair.get("module_provenance", {}) or {}
        repair_sha = _sha256_file(repair_path) if (repair_path and os.path.isfile(repair_path)) else "UNKNOWN"
        sidecar = _find_sidecar(repair_path) if repair_path else None
        sidecar_name = os.path.basename(sidecar) if sidecar else "MISSING"
        # Emit the source paths exactly as provided (the drain script passes
        # paths relative to the edit-harness cwd, matching the legacy convention).
        legacy_label = source_path if source_path else "MISSING"
        repair_label = repair_path if repair_path else "MISSING"
        header_lines = [
            "% Paper B macros — AUTO-GENERATED from gate readout + repair artifact",
            "% SOURCES:",
            f"%   {legacy_label} (legacy)",
            f"%   {repair_label}",
            f"%     (immutable sidecar: {sidecar_name},",
            f"%      sha256 {repair_sha})",
            f"%     module={prov.get('module', 'UNKNOWN')}, "
            f"version={prov.get('version', 'UNKNOWN')}, "
            f"n_boot={prov.get('n_boot', 'UNKNOWN')}, "
            f"rng_seed={prov.get('rng_seed', 'UNKNOWN')}",
            "% DO NOT hand-edit; regenerate with: python3 experiments/quant_survival_macros.py",
            "",
        ]
    else:
        header_lines = [
            "% Paper B macros — AUTO-GENERATED from gate readout",
            "% SOURCE: " + (source_path or "MISSING"),
            "% DO NOT hand-edit; regenerate with: python3 experiments/quant_survival_macros.py",
            "",
        ]

    lines = list(header_lines) + [
        "% ---- Frozen thresholds ----",
        f"\\newcommand{{\\pFpThreshold}}{{{_fmt(th.get('fp32_law_gate', 0.30))}}}"
        f"   % fp32 law gate (C2 eligibility)",
        f"\\newcommand{{\\pEsrSurvFourThreshold}}{{{_fmt(th.get('esr_survival_4bit', 0.80))}}}"
        f"   % C1 4-bit survival threshold",
        f"\\newcommand{{\\pEsrSurvEightThreshold}}{{{_fmt(th.get('esr_survival_8bit', 0.90))}}}"
        f"   % C1 8-bit survival threshold",
        f"\\newcommand{{\\pDeltaRhoTolerance}}{{{_fmt(th.get('delta_rho_tolerance', 0.15))}}}"
        f"   % C2 Δρ tolerance (justified vs fp32 seed spread before ratification)",
        f"\\newcommand{{\\pRankSurvFourThreshold}}{{{_fmt(th.get('rank_survival_4bit', 0.85))}}}"
        f"   % rank survival 4-bit threshold",
        f"\\newcommand{{\\pRankSurvEightThreshold}}{{{_fmt(th.get('rank_survival_8bit', 0.95))}}}"
        f"   % rank survival 8-bit threshold",
        f"\\newcommand{{\\pMedianRatioThreshold}}{{{_fmt(th.get('median_ratio_concentration', 1.0))}}}"
        f"   % C3 M-concentration median ratio threshold",
        "",
        "% ---- Primary cell: Llama-3.2-1B ROME L12 ----",
        f"\\newcommand{{\\pPrimaryFpWithin}}{{{_fmt(primary.get('fp32_rho_within_probe_mean'))}}}"
        f"   % fp32 within-probe rho (mean over seeds)",
        f"\\newcommand{{\\pPrimaryCtwoEligible}}{{{_fmt(primary.get('c2_eligible'))}}}"
        f"   % 1 if fp32 law >= threshold, 0 otherwise",
        "",
        "% ---- Second cell: Llama-3.2-3B ROME L24 ----",
        f"\\newcommand{{\\pSecondFpWithin}}{{{_fmt(second.get('fp32_rho_within_probe_mean'))}}}"
        f"   % fp32 within-probe rho (mean over seeds)",
        f"\\newcommand{{\\pSecondCtwoEligible}}{{{_fmt(second.get('c2_eligible'))}}}"
        f"   % 1 if fp32 law >= threshold, 0 otherwise",
    ]

    def arm_macros(cell: Dict[str, Any], prefix: str, legacy_key: str):
        rc = _repair_cell_for_legacy_key(repair, legacy_key)
        for arm_name, arm in cell.get("arms", {}).items():
            scheme = _latex_safe(arm_name.split("_")[0])
            loc = "_".join(arm_name.split("_")[1:])
            macro_base = f"{prefix}{scheme}{loc.title().replace('_', '')}"
            lines.append(f"\\newcommand{{\\{macro_base}EsrSurv}}"
                         f"{{{_fmt(arm.get('esr_survival_given_fp32_worked_mean'))}}}"
                         f"   % {arm_name} esr survival | fp32 worked")
            lines.append(f"\\newcommand{{\\{macro_base}DeltaRho}}"
                         f"{{{_fmt(arm.get('delta_rho_vs_fp32_within_mean'))}}}"
                         f"   % {arm_name} Δρ within-probe")
            lines.append(f"\\newcommand{{\\{macro_base}RankSurv}}"
                         f"{{{_fmt(arm.get('rho_damage_fp32_vs_arm_rank_survival_mean'))}}}"
                         f"   % {arm_name} damage rank survival")
            # Per-arm ABSOLUTE ESR (mean over ALL edits, not just those that
            # worked in FP32). Drives Table~\ref{tab:absolutes} 'absolute mean
            # esr' column. Sourced from the repair artefact when present;
            # legacy-only renders emit NAN as a sentinel.
            abs_esr_pt = None
            if rc is not None:
                abs_esr_pt = (_get_arm(rc, arm_name) or {}).get(
                    "absolute_quantized_esr", {}).get("point")
            lines.append(f"\\newcommand{{\\{macro_base}AbsEsrPt}}"
                         f"{{{_fmt(abs_esr_pt)}}}"
                         f"   % {arm_name} absolute ESR point (mean edit_ok_arm)")

    arm_macros(primary, "pPrimary", "Llama-3.2-1B_rome_L12")
    arm_macros(second, "pSecond", "Llama-3.2-3B_rome_L24")

    # Fallback: second cell might be absent (gate-recompute path, or 3B cells
    # not yet drained). Emit NAN placeholders for the four expected arms so
    # \pSecond* macros never dangle — they resolve cleanly on post-drain regen.
    EXPECTED_SECOND_ARMS = [
        "nf4dq_edited_layer", "nf4dq_full_model",
        "int8_edited_layer", "int8_full_model",
    ]
    emitted_second_bases = set()
    for line in lines:
        for arm_name in EXPECTED_SECOND_ARMS:
            scheme = _latex_safe(arm_name.split("_")[0])
            loc = "_".join(arm_name.split("_")[1:])
            marker = f"pSecond{scheme}{loc.title().replace('_', '')}"
            if marker in line:
                emitted_second_bases.add(marker)
    for arm_name in EXPECTED_SECOND_ARMS:
        scheme = _latex_safe(arm_name.split("_")[0])
        loc = "_".join(arm_name.split("_")[1:])
        marker = f"pSecond{scheme}{loc.title().replace('_', '')}"
        if marker not in emitted_second_bases:
            for suffix, desc in (
                ("EsrSurv", "esr survival | fp32 worked"),
                ("DeltaRho", "Δρ within-probe"),
                ("RankSurv", "damage rank survival"),
            ):
                lines.append(
                    f"\\newcommand{{\\{marker}{suffix}}}"
                    f"{{NAN}}   % {arm_name} {desc} (placeholder; second cell absent)"
                )

    def c3_macros(cell: Dict[str, Any], prefix: str):
        for scheme, c in cell.get("c3", {}).items():
            macro_base = f"{prefix}{_latex_safe(scheme.title())}"
            lines.append(f"\\newcommand{{\\{macro_base}Fabove}}"
                         f"{{{_fmt(c.get('F_above_mean'))}}}"
                         f"   % {scheme} F_above (fraction |ΔW|/b >= 1)")
            lines.append(f"\\newcommand{{\\{macro_base}MedianRatio}}"
                         f"{{{_fmt(c.get('median_ratio_mean'))}}}"
                         f"   % {scheme} median |ΔW|/b ratio")
            lines.append(f"\\newcommand{{\\{macro_base}Rfunc}}"
                         f"{{{_fmt(c.get('r_func_mean'))}}}"
                         f"   % {scheme} M-averaging r_func")
            lines.append(f"\\newcommand{{\\{macro_base}Rparam}}"
                         f"{{{_fmt(c.get('r_param_mean'))}}}"
                         f"   % {scheme} M-averaging r_param")

    c3_macros(primary, "pPrimary")
    c3_macros(second, "pSecond")

    # Repair-aware multilevel rank survival block (v1.2.1).
    if repair is not None:
        lines.append("")
        lines.append("% ---- Multilevel rank survival (v1.2.1 repair artefact) ----")
        lines.append("% Definition chain:")
        lines.append("%   flat_rank              = legacy Spearman(D_fp32.ravel(), D_quant.ravel()) over the held-out probe grid")
        lines.append("%   within_probe           = within-probe Spearman, averaged across non-degenerate probe columns")
        lines.append("%   edit_level_signed_mean = per-edit Spearman of mean(D_fp32) vs mean(D_quant) across edits (signed-mean)")
        lines.append("%   edit_level_absmean     = per-edit Spearman of mean(|D_fp32|) vs mean(|D_quant|) (magnitude)")
        lines.append("%   edit_level_l2          = per-edit Spearman of L2(D_fp32) vs L2(D_quant) (L2 magnitude)")
        lines.append("%   base_quant_noise_mean_abs = mean(|D|) under the UNEDITED-quantized baseline arm (magnitude sensitivity)")

        pri_cell = _find_cell(repair, "llama1b", "rome", 12)
        # Qwen per-editor aggregate absolute ESR points — drive the Qwen
        # FP32 / NF4 full-model numbers cited in the abstract, results, and
        # portal metadata. Sourced from the canonical v1.2.1 repair.
        qwen_rome = _find_cell(repair, "qwen15b", "rome", 21)
        qwen_memit = _find_cell(repair, "qwen15b", "memit", 21)
        qwen_alpha = _find_cell(repair, "qwen15b", "alpha", 21)

        def _fp32_pt(cell_obj):
            if cell_obj is None:
                return None
            v = (cell_obj.get("absolute_fp32_esr") or {}).get("point")
            return v if isinstance(v, (int, float)) else None

        def _arm_pt(cell_obj, arm_name):
            if cell_obj is None:
                return None
            v = (_get_arm(cell_obj, arm_name) or {}).get(
                "absolute_quantized_esr", {}).get("point")
            return v if isinstance(v, (int, float)) else None

        lines.append("")
        lines.append("% Qwen-1.5B L21 per-editor aggregate absolute ESR (canonical v1.2.1)")
        lines.append(f"\\newcommand{{\\pQwenRomeFpThirtyTwoAbsEsrPt}}{{{_fmt(_fp32_pt(qwen_rome))}}}"
                     f"   % Qwen ROME L21 FP32 absolute ESR point")
        lines.append(f"\\newcommand{{\\pQwenMemitFpThirtyTwoAbsEsrPt}}{{{_fmt(_fp32_pt(qwen_memit))}}}"
                     f"   % Qwen MEMIT L21 FP32 absolute ESR point")
        lines.append(f"\\newcommand{{\\pQwenAlphaFpThirtyTwoAbsEsrPt}}{{{_fmt(_fp32_pt(qwen_alpha))}}}"
                     f"   % Qwen Alpha L21 FP32 absolute ESR point")
        lines.append(f"\\newcommand{{\\pQwenRomeNfFourFMAbsEsrPt}}{{{_fmt(_arm_pt(qwen_rome, 'nf4dq_full_model'))}}}"
                     f"   % Qwen ROME L21 NF4 full-model absolute ESR point")
        lines.append(f"\\newcommand{{\\pQwenMemitNfFourFMAbsEsrPt}}{{{_fmt(_arm_pt(qwen_memit, 'nf4dq_full_model'))}}}"
                     f"   % Qwen MEMIT L21 NF4 full-model absolute ESR point")
        lines.append(f"\\newcommand{{\\pQwenAlphaNfFourFMAbsEsrPt}}{{{_fmt(_arm_pt(qwen_alpha, 'nf4dq_full_model'))}}}"
                     f"   % Qwen Alpha L21 NF4 full-model absolute ESR point")
        if pri_cell is not None:
            _emit_multilevel_block(
                lines, pri_cell, "nf4dq_full_model", "pPri",
                "Llama-3.2-1B / ROME / L12 / NF4dq full-model",
                range_caveat="legacy",
            )
            lines.append("")
        sec_cell = _find_cell(repair, "llama3b", "rome", 24)
        if sec_cell is not None:
            _emit_multilevel_block(
                lines, sec_cell, "nf4dq_full_model", "pSec",
                "Llama-3.2-3B / ROME / L24 / NF4dq full-model",
                range_caveat="legacy",
            )
            lines.append("")
        qwen_cell = _find_cell(repair, "qwen15b", "rome", 21)
        if qwen_cell is not None:
            _emit_multilevel_block(
                lines, qwen_cell, "nf4dq_full_model", "pQwen",
                "Qwen-2.5-1.5B / ROME / L21 / NF4dq full-model",
                range_caveat="",
            )
            # Qwen-wide CI widths (NF4dq full-model)
            a = _get_arm(qwen_cell, "nf4dq_full_model") or {}
            abs_ci = (a.get("absolute_quantized_esr") or {}).get("ci95") or [None, None]
            cond_ci = (a.get("conditional_survival_given_fp32_worked") or {}).get("ci95") or [None, None]
            abs_width = (abs_ci[1] - abs_ci[0]) if (
                isinstance(abs_ci[0], (int, float))
                and isinstance(abs_ci[1], (int, float))) else None
            cond_width = (cond_ci[1] - cond_ci[0]) if (
                isinstance(cond_ci[0], (int, float))
                and isinstance(cond_ci[1], (int, float))) else None
            lines.append("% Qwen-wide CI widths (NF4dq full-model)")
            lines.append(f"\\newcommand{{\\pQwenAbsEsrWidth}}{{{_fmt(abs_width)}}}"
                         f"              % absolute ESR CI width")
            lines.append(f"\\newcommand{{\\pQwenCondWidth}}{{{_fmt(cond_width)}}}"
                         f"                % conditional survival CI width")
            lines.append("")

        _emit_dense_table_macros(lines, repair)

    lines.extend([
        "",
        "% ---- Kill-gate readouts ----",
        # K1 is computed dynamically from the two validated-law ROME cells
        # (Llama-1B ROME L12, Llama-3B ROME L24) per the rule:
        #   K1 fires (FAIL) ONLY if BOTH validated cells fail their NF4
        #   full-model rank-survival threshold (default 0.85). If at least
        #   one passes the phenomenon narrows and K1 reports PASS. The
        #   legacy gate_readout.json's K1 status field is NOT used.
        f"\\newcommand{{\\gateKoneStatus}}{{{_compute_k1_status(readout, repair)}}}"
        f"   % K1 geometry-ranking survival (two-cell rule: BOTH fail → FAIL)",
        f"\\newcommand{{\\gateKtwoStatus}}{{{gates.get('K2_esr_survival_4bit', {}).get('status', 'PENDING')}}}"
        f"   % K2 esr survival at 4-bit",
        "% Authorial provenance correction: gate_readout.json stores amended K3 status;\n"
        "% manuscript reports amended channel-scale x NF4 min-gap K3 as UNADJUDICATED.\n"
        "\\newcommand{\\gateKthreeStatus}{UNADJUDICATED}"
        "   % amended K3 pending targeted rerun",
        "",
        "% ---- Experiment metadata ----",
    ])
    if repair is not None:
        cells_list = repair.get("cells") or []
        n_cells = len(cells_list)
        pri_cell = _find_cell(repair, "llama1b", "rome", 12)
        sec_cell = _find_cell(repair, "llama3b", "rome", 24)
        n_seeds_pri = (pri_cell or {}).get("n_seeds", 0)
        n_seeds_sec = (sec_cell or {}).get("n_seeds", 0)
        prov = repair.get("module_provenance", {}) or {}
        repair_version = prov.get("version", "UNKNOWN")
        repair_nboot = prov.get("n_boot", "UNKNOWN")
        lines.append(f"\\newcommand{{\\nCellsCompleted}}{{{n_cells}}}"
                     f"   % number of (model/editor/layer) aggregates in repair artifact")
        lines.append(f"\\newcommand{{\\nSeedsPrimary}}{{{n_seeds_pri}}}"
                     f"   % seeds on primary cell")
        lines.append(f"\\newcommand{{\\nSeedsSecond}}{{{n_seeds_sec}}}"
                     f"   % seeds on mandatory second cell")
        lines.append(f"\\newcommand{{\\pRepairVersion}}{{{repair_version}}}"
                     f"   % repair artefact schema version")
        lines.append(f"\\newcommand{{\\pRepairNBoot}}{{{repair_nboot}}}"
                     f"   % repair artefact bootstrap iterations")
    else:
        lines.append(f"\\newcommand{{\\nCellsCompleted}}{{{len(cells)}}}"
                     f"   % number of (model/editor/layer) aggregates in readout")
        lines.append(f"\\newcommand{{\\nSeedsPrimary}}{{{primary.get('n_seeds', 0)}}}"
                     f"   % seeds on primary cell")
        lines.append(f"\\newcommand{{\\nSeedsSecond}}{{{second.get('n_seeds', 0)}}}"
                     f"   % seeds on mandatory second cell")

    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Generate Paper B LaTeX macros from gate readout + repair artifact.")
    ap.add_argument("--in_path", default=DEFAULT_IN, help="legacy gate readout JSON")
    ap.add_argument("--repair_in", default=DEFAULT_REPAIR,
                    help="v1.2.1 repair artefact JSON (canonical)")
    ap.add_argument("--out_path", default=DEFAULT_OUT, help="output macros.tex")
    ap.add_argument("--strict_repair", action="store_true", default=True,
                    help="fail-closed if repair artefact missing/incompatible (default for drain path)")
    ap.add_argument("--no_strict_repair", dest="strict_repair", action="store_false",
                    help="non-strict: emit legacy-only macros when repair artefact missing")
    args = ap.parse_args()

    if not os.path.isfile(args.in_path):
        print(f"[warn] readout not found: {args.in_path}; writing placeholder legacy macros", file=sys.stderr)
        readout = {"thresholds": {}, "cells": {}, "gates": {}}
    else:
        readout = json.load(open(args.in_path))

    repair = _load_repair(args.repair_in, strict=args.strict_repair)

    macros = generate_macros(readout, args.in_path, repair=repair, repair_path=args.repair_in)
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "w") as f:
        f.write(macros)
    print(f"wrote {args.out_path}")


if __name__ == "__main__":
    main()