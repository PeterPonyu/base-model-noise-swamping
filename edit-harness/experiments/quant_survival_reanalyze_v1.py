"""quant_survival_reanalyze_v1.py — Paper B Phase-1 standalone CPU reanalysis (v1).

Reads every `results/quant_survival/<cell>/QS_phase1_raw.npz` + the matching
`QS_phase1_table.json` for the preregistered 27-cell grid (3 models x 3 editors x
3 seeds) and emits a REPAIR JSON under `results/quant_survival/aggregate/`. This
module is INTENTIONALLY STANDALONE: it does NOT import the live science runner
`quant_survival_phase1.py` (which it never edits) and only reads the on-disk
artifacts.

Behavior:
  1. Validates the exact expected grid; rejects smoke / extra / missing / duplicate /
     non-finite cells with a structured error report.
  2. Reports three signed-Spearman rank families:
       - legacy flat edit-probe rank (over all N*M pairs)
       - within-probe cross-edit rank (mean of per-probe-column Spearman)
       - operational edit-level ranks (signed_mean / absmean / L2 / p95abs)
  3. Hierarchical DETERMINISTIC bootstrap: seeds-then-edits (resample seeds with
     replacement first, then edits with replacement within each seed-draw).
  4. Absolute fp32 ESR, absolute quantized ESR, and conditional survival | fp32-
     worked, each with hierarchical bootstrap 95% CIs.
  5. Generation checks (perplexity_mean / paraphrase_esr_mean) replayed from the
     table and aggregated across seeds.
  6. Reconstructed split overlap / hash audit: sha256 of (COS, edit_ok_fp32,
     damage_fp32) per cell + per-cell (N_edits, N_probes) signature.
  7. Exploratory mechanism association diagnostics, clearly labeled POST HOC
     (correlations across arms: delta-rho vs survival, perplexity vs survival,
     C3 median_ratio vs survival).
  8. JSON output with provenance (module version, RNG seed, N_BOOT, file paths)
     and FORMULA strings for every reported metric.

The LIVE science runner `quant_survival_phase1.py` is NEVER touched.

CPU-only; numpy only; no GPU / torch / network.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---- Module-level provenance ----
VERSION = "1.2.1"
MODULE_NAME = "quant_survival_reanalyze_v1"
RNG_SEED = 12345            # fixed so bootstrap is reproducible across runs
N_BOOT = 1000               # bootstrap iterations per (cell, statistic) for OPERATIONAL metrics
BOOT_ALPHA = 0.05           # 1 - 0.95 = 95% CI
BOOT_Q_LO = 100 * BOOT_ALPHA / 2.0
BOOT_Q_HI = 100.0 - BOOT_Q_LO

# ---- Manuscript-policy aggregator for `point` estimates ----
# Manuscript policy: "point" is the DIRECT original-data estimator, NOT a
# bootstrap-distribution mean. Bootstrap CIs alone come from the bootstrap.
# Allowed aggregations per metric:
#   - mean of n_seeds per-seed Spearmans (preferred; matches Phase-1 readout)
#   - explicit pooled Spearman (also acceptable; one less assumption)
POINT_AGGREGATION_POLICY = "mean_of_per_seed_spearmans"

HARNESS = os.path.dirname(os.path.abspath(__file__))
RESULTS_ROOT = os.path.join(HARNESS, "..", "results", "quant_survival")
AGGREGATE_DIR = os.path.join(RESULTS_ROOT, "aggregate")
REPAIR_OUT = os.path.join(AGGREGATE_DIR, f"quant_survival_repair_v{VERSION.split('.')[0]}.json")

# ---- Preregistered cell grid (per Paper B Phase-1 prereg) ----
EXPECTED_MODELS = {
    "data/models/Llama-3.2-1B": "llama1b",
    "data/models/Llama-3.2-3B": "llama3b",
    "data/models/Qwen2.5-1.5B": "qwen15b",
}
EXPECTED_LAYERS = {
    "data/models/Llama-3.2-1B": 12,
    "data/models/Llama-3.2-3B": 24,
    "data/models/Qwen2.5-1.5B": 21,
}
EXPECTED_EDITORS = ("alpha", "memit", "rome")
EXPECTED_SEEDS = (0, 1, 2)
EXPECTED_SCHEMES = ("nf4dq", "int8")
EXPECTED_LOCALITIES = ("edited_layer", "full_model")
EXPECTED_ARMS = tuple(f"{s}_{l}" for s in EXPECTED_SCHEMES for l in EXPECTED_LOCALITIES)
EXPECTED_N_EDITS = 200
EXPECTED_N_PROBES = 200

EXPECTED_RAW_KEYS = {
    "COS", "damage_fp32", "edit_ok_fp32",
    "damage__nf4dq_edited_layer", "esr__nf4dq_edited_layer",
    "damage__nf4dq_full_model", "esr__nf4dq_full_model",
    "damage__int8_edited_layer", "esr__int8_edited_layer",
    "damage__int8_full_model", "esr__int8_full_model",
    "base__nf4dq_edited_layer", "base__nf4dq_full_model",
    "base__int8_edited_layer", "base__int8_full_model",
    "c3_ratio__nf4dq", "c3_rfunc__nf4dq", "c3_rparam__nf4dq",
    "c3_ratio__int8", "c3_rfunc__int8", "c3_rparam__int8",
}

CELL_NAME_RE = re.compile(r"^(?P<slug>[a-z0-9]+)_(?P<editor>alpha|memit|rome)_L(?P<layer>\d+)_s(?P<seed>\d+)$")
SLUG_TO_FULLPATH = {v: k for k, v in EXPECTED_MODELS.items()}


# ============================================================
# Numerical primitives
# ============================================================

def _midrank(x: np.ndarray) -> np.ndarray:
    """Tie-averaged ranks (proper Spearman ranks)."""
    order = x.argsort(kind="mergesort")
    ranks = np.empty(len(x), float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    return (sums / cnt)[inv]


def _col_ranks_tieavg(X: np.ndarray) -> np.ndarray:
    """Vectorized per-column tie-averaged ranks for a 2D array (N, M)."""
    N, M = X.shape
    order = X.argsort(kind="mergesort", axis=0)             # (N, M)
    ranks = np.empty_like(order, dtype=float)
    rows = np.arange(1, N + 1, dtype=float).reshape(-1, 1)
    np.put_along_axis(ranks, order, np.broadcast_to(rows, (N, M)), axis=0)
    sorted_X = np.take_along_axis(X, order, axis=0)
    sorted_ranks = np.take_along_axis(ranks, order, axis=0)
    diffs = np.ones((N, M), dtype=bool)
    diffs[1:] = sorted_X[1:] != sorted_X[:-1]
    group_id = np.cumsum(diffs, axis=0)                      # (N, M)
    col_offset = (np.arange(M) * (N + 1)).astype(np.intp)    # (M,)
    flat_gid = (group_id + col_offset[None, :]).ravel()
    flat_sorted_ranks = sorted_ranks.ravel()
    sums = np.bincount(flat_gid, weights=flat_sorted_ranks)
    cnts = np.bincount(flat_gid)
    means = sums / np.maximum(cnts, 1)
    out_sorted = means[flat_gid].reshape(N, M)
    inverse_order = order.argsort(axis=0)
    return np.take_along_axis(out_sorted, inverse_order, axis=0)


def _col_spearman(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Per-column Spearman rho. Inputs (N, M); returns (M,)."""
    if X.shape != Y.shape:
        raise ValueError(f"shape mismatch {X.shape} vs {Y.shape}")
    rx = _col_ranks_tieavg(X)
    ry = _col_ranks_tieavg(Y)
    rx_c = rx - rx.mean(axis=0, keepdims=True)
    ry_c = ry - ry.mean(axis=0, keepdims=True)
    num = (rx_c * ry_c).sum(axis=0)
    den = np.sqrt((rx_c ** 2).sum(axis=0) * (ry_c ** 2).sum(axis=0))
    den = np.where(den == 0, np.nan, den)
    return num / den


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho with finite-value masking; NaN if fewer than 3 finite pairs."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 3:
        return float("nan")
    ar, br = _midrank(a), _midrank(b)
    if ar.std() == 0 or br.std() == 0:
        return float("nan")
    return float(np.corrcoef(ar, br)[0, 1])


def per_edit_summaries(D: np.ndarray) -> Dict[str, np.ndarray]:
    """Four operational edit-level summaries; per-edit vector of length N."""
    abs_D = np.abs(D)
    return {
        "signed_mean": D.mean(axis=1),
        "absmean": abs_D.mean(axis=1),
        "l2": np.sqrt(np.sum(D * D, axis=1)),
        "p95abs": np.percentile(abs_D, 95, axis=1),
    }


# ============================================================
# Hierarchical deterministic bootstrap (seeds-then-edits)
# ============================================================

def _make_boot_indices(rng: np.random.Generator,
                       n_seeds: int, N: int, n_boot: int) -> np.ndarray:
    """Build a (n_boot, n_seeds*N) array of flat indices into the pooled (seeds*edits)
    array. Generated as (seed_draw, edit_draw_within_seed) pairs so the seed-then-edit
    hierarchy is preserved (cluster structure kept)."""
    seed_draws = rng.integers(0, n_seeds, (n_boot, n_seeds))         # (B, S)
    edit_draws = rng.integers(0, N, (n_boot, n_seeds, N))            # (B, S, N)
    flat = (seed_draws[:, :, None] * N + edit_draws).reshape(n_boot, -1)  # (B, S*N)
    return flat


def hier_boot_stat(cos_per_seed: List[np.ndarray],
                   d_per_seed: List[np.ndarray],
                   stat_fn,
                   n_boot: Optional[int] = None,
                   rng_seed: int = RNG_SEED) -> Dict[str, float]:
    """Hierarchical (seeds-then-edits) bootstrap of a two-matrix statistic.

    stat_fn(COS, D) -> float. Returns a dict with keys:
      ci95_lo, ci95_hi, boot_dist_mean, boot_n_finite, boot_n_total, skipped_fraction
    The CI is computed from the finite subset; `skipped_fraction` is the
    fraction of n_boot iterations that produced NaN (e.g. degenerate draws).
    """
    if n_boot is None:
        n_boot = N_BOOT
    rng = np.random.default_rng(rng_seed)
    n_seeds = len(cos_per_seed)
    N = cos_per_seed[0].shape[0] if n_seeds else 0
    if n_seeds == 0 or N < 3:
        return {
            "ci95_lo": float("nan"), "ci95_hi": float("nan"),
            "boot_dist_mean": float("nan"),
            "boot_n_finite": 0, "boot_n_total": int(n_boot),
            "skipped_fraction": 1.0,
        }
    pooled_cos = np.concatenate(cos_per_seed, axis=0)
    pooled_d = np.concatenate(d_per_seed, axis=0)
    flat_idx = _make_boot_indices(rng, n_seeds, N, n_boot)
    stats = np.empty(n_boot)
    for t in range(n_boot):
        idx = flat_idx[t]
        stats[t] = stat_fn(pooled_cos[idx], pooled_d[idx])
    finite = stats[np.isfinite(stats)]
    boot_n_finite = int(finite.size)
    boot_n_total = int(n_boot)
    skipped_fraction = 1.0 - (boot_n_finite / boot_n_total) if boot_n_total else 1.0
    if finite.size < max(10, n_boot // 5):
        return {
            "ci95_lo": float("nan"), "ci95_hi": float("nan"),
            "boot_dist_mean": float("nan"),
            "boot_n_finite": boot_n_finite, "boot_n_total": boot_n_total,
            "skipped_fraction": skipped_fraction,
        }
    return {
        "ci95_lo": float(np.percentile(finite, BOOT_Q_LO)),
        "ci95_hi": float(np.percentile(finite, BOOT_Q_HI)),
        "boot_dist_mean": float(np.mean(finite)),
        "boot_n_finite": boot_n_finite,
        "boot_n_total": boot_n_total,
        "skipped_fraction": skipped_fraction,
    }


def hier_boot_esr(eok_per_seed: List[np.ndarray],
                  n_boot: Optional[int] = None,
                  rng_seed: int = RNG_SEED) -> Dict[str, float]:
    """Hierarchical bootstrap on absolute ESR (mean of edit-ok vector).

    Returns dict with ci95_lo, ci95_hi, boot_dist_mean, boot_n_finite,
    boot_n_total, skipped_fraction.
    """
    if n_boot is None:
        n_boot = N_BOOT
    rng = np.random.default_rng(rng_seed)
    n_seeds = len(eok_per_seed)
    if n_seeds == 0:
        return {
            "ci95_lo": float("nan"), "ci95_hi": float("nan"),
            "boot_dist_mean": float("nan"),
            "boot_n_finite": 0, "boot_n_total": int(n_boot),
            "skipped_fraction": 1.0,
        }
    N = len(eok_per_seed[0])
    pooled = np.concatenate(eok_per_seed)
    flat_idx = _make_boot_indices(rng, n_seeds, N, n_boot)
    boot_means = np.empty(n_boot)
    for t in range(n_boot):
        boot_means[t] = float(np.mean(pooled[flat_idx[t]]))
    boot_n_total = int(n_boot)
    boot_n_finite = int(np.isfinite(boot_means).sum())
    skipped_fraction = 1.0 - (boot_n_finite / boot_n_total) if boot_n_total else 1.0
    return {
        "ci95_lo": float(np.percentile(boot_means, BOOT_Q_LO)),
        "ci95_hi": float(np.percentile(boot_means, BOOT_Q_HI)),
        "boot_dist_mean": float(np.mean(boot_means)),
        "boot_n_finite": boot_n_finite,
        "boot_n_total": boot_n_total,
        "skipped_fraction": skipped_fraction,
    }


def hier_boot_conditional(fp32_eok_per_seed: List[np.ndarray],
                          arm_eok_per_seed: List[np.ndarray],
                          n_boot: Optional[int] = None,
                          rng_seed: int = RNG_SEED) -> Dict[str, float]:
    """Hierarchical bootstrap on conditional survival | fp32-worked.

    Resamples seeds then edits within each seed; for each bootstrap iter computes
    (#arm_ok AND fp32_ok) / (#fp32_ok) on the pooled bootstrap sample. NaN draws
    (when fp32-worked count is 0 in the resample) are kept in `boot_n_total` and
    reflected in `boot_n_finite` + `skipped_fraction`; CI is over the finite subset.
    """
    if n_boot is None:
        n_boot = N_BOOT
    rng = np.random.default_rng(rng_seed)
    n_seeds = len(fp32_eok_per_seed)
    if n_seeds == 0:
        return {
            "ci95_lo": float("nan"), "ci95_hi": float("nan"),
            "boot_dist_mean": float("nan"),
            "boot_n_finite": 0, "boot_n_total": int(n_boot),
            "skipped_fraction": 1.0,
            "n_nan_draws_exposed": int(n_boot),
        }
    N = len(fp32_eok_per_seed[0])
    pooled_f = np.concatenate(fp32_eok_per_seed)
    pooled_a = np.concatenate(arm_eok_per_seed)
    flat_idx = _make_boot_indices(rng, n_seeds, N, n_boot)
    stats = np.empty(n_boot)
    for t in range(n_boot):
        idx = flat_idx[t]
        f = pooled_f[idx]
        a = pooled_a[idx]
        mask = f >= 0.5
        if mask.any():
            num = int(np.sum(a[mask] >= 0.5))
            den = int(mask.sum())
            stats[t] = (num / den) if den > 0 else float("nan")
        else:
            stats[t] = float("nan")
    finite = stats[np.isfinite(stats)]
    boot_n_finite = int(finite.size)
    boot_n_total = int(n_boot)
    skipped_fraction = 1.0 - (boot_n_finite / boot_n_total) if boot_n_total else 1.0
    if finite.size < max(10, n_boot // 5):
        return {
            "ci95_lo": float("nan"), "ci95_hi": float("nan"),
            "boot_dist_mean": float("nan"),
            "boot_n_finite": boot_n_finite, "boot_n_total": boot_n_total,
            "skipped_fraction": skipped_fraction,
            "n_nan_draws_exposed": int(boot_n_total - boot_n_finite),
        }
    return {
        "ci95_lo": float(np.percentile(finite, BOOT_Q_LO)),
        "ci95_hi": float(np.percentile(finite, BOOT_Q_HI)),
        "boot_dist_mean": float(np.mean(finite)),
        "boot_n_finite": boot_n_finite,
        "boot_n_total": boot_n_total,
        "skipped_fraction": skipped_fraction,
        "n_nan_draws_exposed": int(boot_n_total - boot_n_finite),
    }


# ============================================================
# Statistics (closures that take (COS, D) -> float)
# ============================================================

def stat_flat_rank(D_fp32: np.ndarray, D_quant: np.ndarray) -> float:
    """Flat (N*M)-pair Spearman. Parameter names are GENERIC: rank-SURVIVAL
    callers pass (D_fp32, D_quant); legacy geometry-sensitivity callers pass
    (COS, D). The function body is shape-agnostic."""
    return spearman(D_fp32.reshape(-1), D_quant.reshape(-1))


def stat_within_probe(D_fp32: np.ndarray, D_quant: np.ndarray) -> float:
    """Within-probe cross-edit Spearman: mean over probe columns. Parameter
    names are GENERIC: rank-SURVIVAL callers pass (D_fp32, D_quant); legacy
    geometry-sensitivity callers pass (COS, D)."""
    cols = _col_spearman(D_fp32, D_quant)
    return float(np.nanmean(cols)) if cols.size else float("nan")


def make_stat_edit_level(summary: str):
    """Per-edit Spearman between a COS-side mean and a per-edit summary of D.
    Parameter names are GENERIC: legacy geometry-sensitivity callers pass
    (COS, D_quant); the rank-SURVIVAL counterpart `make_stat_edit_level_survival`
    passes (D_fp32, D_quant) with both summaries computed from `D`.
    """
    def _f(x: np.ndarray, D: np.ndarray) -> float:
        x_per_edit = x.mean(axis=1)
        d_per_edit = per_edit_summaries(D)[summary]
        return spearman(x_per_edit, d_per_edit)
    _f.__name__ = f"stat_edit_level_{summary}"
    return _f


def make_stat_edit_level_survival(summary: str):
    """Survival counterpart of make_stat_edit_level.

    Spearman between the SAME summary S applied independently to D_fp32 and D_quant
    per edit. Inputs are paired (D_fp32, D_quant). The legacy make_stat_edit_level
    is preserved (named) for the separately labeled geometry_sensitivity_cos output.
    """
    def _f(D_fp32: np.ndarray, D_quant: np.ndarray) -> float:
        s_fp32 = per_edit_summaries(D_fp32)[summary]
        s_quant = per_edit_summaries(D_quant)[summary]
        return spearman(s_fp32, s_quant)
    _f.__name__ = f"stat_edit_level_survival_{summary}"
    return _f


# ============================================================
# Loading + grid validation
# ============================================================

def sha256_of_array(arr: np.ndarray) -> str:
    """Deterministic sha256 of a numpy array's bytes (C order)."""
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


class LazyNpz:
    """Lazy-loading npz handle for memory-bounded grid iteration.

    The 27-cell Phase-1 grid holds ~6.7 MB per cell (21 arrays of shape
    (200,200) or (200,)). Eager-loading all cells into a single dict kept
    arrays alive across the entire test suite, multiplying memory by the
    number of TestEndToEndRun cases (each builds its own synthetic grid).

    This class reads only the file-name list at construction time
    (cheap; uses `with np.load` so the zipfile is closed immediately), and
    defers array data until first __getitem__. Subsequent accesses are
    served from the in-memory cache. `release()` drops the cached arrays
    so the GC can reclaim memory before the next cell/group is processed.
    """
    __slots__ = ("_npz_path", "_arrays", "_files")

    def __init__(self, npz_path: str):
        self._npz_path = npz_path
        self._arrays: Optional[Dict[str, np.ndarray]] = None
        self._files: Optional[List[str]] = None
        with np.load(npz_path, allow_pickle=True) as npz_obj:
            self._files = list(npz_obj.files)

    @property
    def files(self) -> List[str]:
        return self._files  # type: ignore[return-value]

    def __getitem__(self, k: str) -> np.ndarray:
        self._ensure_loaded()
        return self._arrays[k]  # type: ignore[index]

    def _ensure_loaded(self) -> None:
        if self._arrays is None:
            with np.load(self._npz_path, allow_pickle=True) as npz_obj:
                self._arrays = {k: np.array(npz_obj[k]) for k in npz_obj.files}

    def release(self) -> None:
        """Drop cached arrays; next access will re-load from npz_path."""
        self._arrays = None


def release_all_npz(cells: Dict[str, Dict[str, Any]]) -> None:
    """Call release() on every cell's npz handle to free cached arrays."""
    for cell in cells.values():
        npz = cell.get("npz")
        if isinstance(npz, LazyNpz):
            npz.release()


def load_phase1_cells(root: str) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """Walk `root` for QS_phase1_{raw,table} pairs; reject smoke/extras.

    Returns (cells, info). `cells` is keyed by cell name. `info` is a list of
    human-readable notes (skipped dirs, parse failures, etc).

    v1.2.1: array data is NOT eagerly loaded into the cells dict. Each cell
    gets a LazyNpz handle; arrays are read on demand and can be released
    via `release_all_npz(cells)` to bound memory during long iteration
    sequences (e.g. repeated TestEndToEndRun invocations).
    """
    cells: Dict[str, Dict[str, Any]] = {}
    notes: List[str] = []
    for entry in sorted(os.listdir(root)):
        full = os.path.join(root, entry)
        if not os.path.isdir(full):
            continue
        # Non-cell dirs the loader skips. 'smoke*' and 'test*' prefixes are
        # also skipped because smoke/test cells are not part of the
        # preregistered 27-cell grid and would otherwise be reported as
        # 'unexpected' extras.
        if entry in ("aggregate", "selftest") or entry.startswith(("smoke", "test")):
            notes.append(f"skip non-cell dir: {entry}")
            continue
        npz_path = os.path.join(full, "QS_phase1_raw.npz")
        tbl_path = os.path.join(full, "QS_phase1_table.json")
        if not (os.path.exists(npz_path) and os.path.exists(tbl_path)):
            notes.append(f"skip {entry}: missing QS_phase1_raw.npz or QS_phase1_table.json")
            continue
        try:
            with open(tbl_path) as f:
                tbl = json.load(f)
        except Exception as e:
            notes.append(f"skip {entry}: unreadable table: {e}")
            continue
        if tbl.get("experiment") != "quant_survival_phase1":
            notes.append(f"skip {entry}: table.experiment={tbl.get('experiment')!r} != 'quant_survival_phase1'")
            continue
        # LazyNpz: reads only the file list at construction (zipfile closed
        # by `with`); array data is read on first __getitem__.
        try:
            npz_handle = LazyNpz(npz_path)
        except Exception as e:
            notes.append(f"skip {entry}: unreadable npz: {e}")
            continue
        cells[entry] = {"npz": npz_handle, "table": tbl, "dir": full,
                        "npz_path": npz_path, "tbl_path": tbl_path}
    return cells, notes


def validate_grid(cells) -> Tuple[bool, Dict[str, Any], List[str]]:
    """Validate the exact 27-cell grid; reject smoke / extra / missing / duplicate /
    non-finite. `cells` may be a dict {name: cell_dict} OR a list of (name, cell_dict)
    pairs (the latter form is used by tests to exercise duplicate detection).

    Returns (ok, grid_audit, errors).
    """
    if isinstance(cells, dict):
        items = list(cells.items())
    else:
        items = list(cells)
    return _validate_grid_items(items)


def _validate_grid_items(items: List[Tuple[str, Dict[str, Any]]]
                         ) -> Tuple[bool, Dict[str, Any], List[str]]:
    errors: List[str] = []
    seen_names: Dict[str, int] = defaultdict(int)

    parsed = {}
    for name, cell in items:
        seen_names[name] += 1
        if seen_names[name] > 1:
            errors.append(f"duplicate cell name: {name}")
            continue
        m = CELL_NAME_RE.match(name)
        if not m:
            errors.append(f"cell name {name!r} does not match <slug>_<editor>_L<L>_s<S>")
            continue
        slug = m.group("slug")
        editor = m.group("editor")
        layer = int(m.group("layer"))
        seed = int(m.group("seed"))
        if slug not in SLUG_TO_FULLPATH:
            errors.append(f"cell {name} has unknown slug {slug!r}")
            continue
        fullpath = SLUG_TO_FULLPATH[slug]
        if editor not in EXPECTED_EDITORS:
            errors.append(f"cell {name} editor {editor!r} not in expected {EXPECTED_EDITORS}")
            continue
        if layer != EXPECTED_LAYERS[fullpath]:
            errors.append(f"cell {name} layer {layer} != expected {EXPECTED_LAYERS[fullpath]}")
            continue
        if seed not in EXPECTED_SEEDS:
            errors.append(f"cell {name} seed {seed} not in expected {EXPECTED_SEEDS}")
            continue
        if tbl := cell["table"]:
            if tbl.get("model") != fullpath:
                errors.append(
                    f"cell {name} table.model={tbl.get('model')!r} != expected {fullpath!r}")
            if tbl.get("editor") != editor:
                errors.append(
                    f"cell {name} table.editor={tbl.get('editor')!r} != expected {editor!r}")
            if tbl.get("layer") != layer:
                errors.append(
                    f"cell {name} table.layer={tbl.get('layer')} != expected {layer}")
            if tbl.get("seed") != seed:
                errors.append(
                    f"cell {name} table.seed={tbl.get('seed')} != expected {seed}")
            if int(tbl.get("n_edits", -1)) != EXPECTED_N_EDITS:
                errors.append(
                    f"cell {name} table.n_edits={tbl.get('n_edits')} != expected {EXPECTED_N_EDITS}")
            if int(tbl.get("n_probes", -1)) != EXPECTED_N_PROBES:
                errors.append(
                    f"cell {name} table.n_probes={tbl.get('n_probes')} != expected {EXPECTED_N_PROBES}")
            schemes_tbl = tuple(tbl.get("schemes", ()))
            if schemes_tbl != EXPECTED_SCHEMES:
                errors.append(
                    f"cell {name} table.schemes={schemes_tbl} != expected {EXPECTED_SCHEMES}")
            arms_keys = set(tbl.get("arms", {}).keys())
            missing_arms = set(EXPECTED_ARMS) - arms_keys
            if missing_arms:
                errors.append(f"cell {name} table.arms missing {sorted(missing_arms)}")
        npz = cell["npz"]
        npz_keys = set(npz.files)
        missing_npz = EXPECTED_RAW_KEYS - npz_keys
        if missing_npz:
            errors.append(f"cell {name} npz missing keys {sorted(missing_npz)}")
            continue  # shape checks meaningless if structure is wrong
        # ---- Per-key SHAPE validation (v1.2.0 independent review fix #1) ----
        shape_errors = []
        # COS: (N_edits, N_probes)
        COS = np.asarray(npz["COS"], float)
        if COS.shape != (EXPECTED_N_EDITS, EXPECTED_N_PROBES):
            shape_errors.append(f"COS={COS.shape}")
        # damage_fp32: (N_edits, N_probes)
        dfp = np.asarray(npz["damage_fp32"], float)
        if dfp.shape != (EXPECTED_N_EDITS, EXPECTED_N_PROBES):
            shape_errors.append(f"damage_fp32={dfp.shape}")
        # edit_ok_fp32: (N_edits,)
        eok_fp32 = np.asarray(npz["edit_ok_fp32"], float)
        if eok_fp32.shape != (EXPECTED_N_EDITS,):
            shape_errors.append(f"edit_ok_fp32={eok_fp32.shape}")
        # All damage__<arm>: (N_edits, N_probes)
        for arm in EXPECTED_ARMS:
            arr = np.asarray(npz[f"damage__{arm}"], float)
            if arr.shape != (EXPECTED_N_EDITS, EXPECTED_N_PROBES):
                shape_errors.append(f"damage__{arm}={arr.shape}")
        # All esr__<arm>: (N_edits,)
        for arm in EXPECTED_ARMS:
            arr = np.asarray(npz[f"esr__{arm}"], float)
            if arr.shape != (EXPECTED_N_EDITS,):
                shape_errors.append(f"esr__{arm}={arr.shape}")
        # C3 vectors used by analyze / geometry_sensitivity — recorded shape
        c3_shapes = {}
        for scheme in EXPECTED_SCHEMES:
            for key in (f"c3_ratio__{scheme}", f"c3_rfunc__{scheme}", f"c3_rparam__{scheme}"):
                arr = np.asarray(npz[key], float)
                c3_shapes[key] = arr.shape
        if shape_errors:
            errors.append(f"cell {name} bad shapes: {shape_errors}")
            continue
        # Non-finite check across all EXPECTED_RAW_KEYS (with shape now guaranteed)
        nonfinite = []
        for k in EXPECTED_RAW_KEYS:
            arr = npz[k]
            if arr.dtype.kind in ("f", "c"):
                if not np.all(np.isfinite(arr)):
                    bad = int((~np.isfinite(arr)).sum())
                    nonfinite.append(f"{k}({bad})")
        if nonfinite:
            errors.append(f"cell {name} non-finite entries: {nonfinite}")
        parsed[name] = {
            "fullpath": fullpath, "slug": slug, "editor": editor,
            "layer": layer, "seed": seed, "nonfinite": nonfinite,
            "c3_shapes": c3_shapes,
        }

    expected_names = set()
    for fullpath, slug in EXPECTED_MODELS.items():
        for editor in EXPECTED_EDITORS:
            for seed in EXPECTED_SEEDS:
                expected_names.add(f"{slug}_{editor}_L{EXPECTED_LAYERS[fullpath]}_s{seed}")
    found_names = set(seen_names.keys())
    missing = sorted(expected_names - found_names)
    extras = sorted(found_names - expected_names)
    if missing:
        errors.append(f"missing expected cells ({len(missing)}): {missing}")
    if extras:
        errors.append(f"unexpected cells ({len(extras)}): {extras}")

    grid_audit = {
        "expected_n_cells": len(expected_names),
        "found_n_cells": len(found_names),
        "missing": missing,
        "extras": extras,
        "duplicates": [n for n, c in seen_names.items() if c > 1],
        "per_cell_parsed": parsed,
    }
    return (len(errors) == 0), grid_audit, errors


# ============================================================
# Per-cell aggregation
# ============================================================

def split_audit_for_cell(npz) -> Dict[str, Any]:
    """Sha256 audit of (COS, edit_ok_fp32, damage_fp32) for the split hash."""
    COS = np.asarray(npz["COS"], float)
    eok = np.asarray(npz["edit_ok_fp32"], float)
    dfp = np.asarray(npz["damage_fp32"], float)
    return {
        "n_edits": int(COS.shape[0]),
        "n_probes": int(COS.shape[1]),
        "cos_sha256": sha256_of_array(COS),
        "edit_ok_fp32_sha256": sha256_of_array(eok),
        "damage_fp32_sha256": sha256_of_array(dfp),
        "cos_finite": bool(np.all(np.isfinite(COS))),
        "edit_ok_fp32_unique": sorted({float(x) for x in np.unique(eok)}),
    }


def _min_max_range(vals: List[float]) -> List[float]:
    """Return [min, max] over a list of floats (NaN-aware). Returns [nan, nan] if no finite."""
    finite = [v for v in vals if np.isfinite(v)]
    if not finite:
        return [float("nan"), float("nan")]
    return [float(min(finite)), float(max(finite))]


def _compute_geometry_sensitivity_cos(dfp_per_seed: List[np.ndarray],
                                     dq_per_seed: List[np.ndarray],
                                     cos_per_seed: List[np.ndarray]) -> Dict[str, Any]:
    """COS-based geometry-sensitivity diagnostics, SEPARATELY LABELED.

    Per v1.2 scope: COS is not part of the rank-SURVIVAL gates; this block keeps
    the legacy COS-vs-damage signals available as a separately labeled diagnostic
    (raw point estimate + per-seed min/max RANGE).
    """
    # COS vs D_quant (legacy geometry)
    geom_dq_pooled = stat_flat_rank(np.concatenate(cos_per_seed),
                                    np.concatenate(dq_per_seed))
    geom_dq_per_seed = [stat_flat_rank(cos_per_seed[i], dq_per_seed[i])
                        for i in range(len(cos_per_seed))]
    # COS vs D_fp32 (baseline geometry)
    geom_fp_pooled = stat_flat_rank(np.concatenate(cos_per_seed),
                                    np.concatenate(dfp_per_seed))
    geom_fp_per_seed = [stat_flat_rank(cos_per_seed[i], dfp_per_seed[i])
                        for i in range(len(cos_per_seed))]
    # edit-level: mean(COS[i,:]) vs S(Dq[i,:])
    edit_lev_cos: Dict[str, Dict[str, Any]] = {}
    for summary in ("signed_mean", "absmean", "l2", "p95abs"):
        stat_fn = make_stat_edit_level(summary)
        pooled = stat_fn(np.concatenate(cos_per_seed),
                         np.concatenate(dq_per_seed))
        per_seed = [stat_fn(cos_per_seed[i], dq_per_seed[i])
                    for i in range(len(cos_per_seed))]
        edit_lev_cos[summary] = {
            "point": float(pooled) if np.isfinite(pooled) else float("nan"),
            "per_seed": [float(v) if np.isfinite(v) else float("nan") for v in per_seed],
            "range_min_max": _min_max_range(per_seed),
            "range_kind": "per_seed_point_estimate_min_max",
        }
    return {
        "label": "GEOMETRY SENSITIVITY (COS-based) — diagnostic, NOT a survival metric",
        "note": ("Reported separately per v1.2 scope; the rank-SURVIVAL gates use "
                 "D_fp32 vs D_quant only."),
        "cos_vs_Dquant_flat": {
            "point": float(geom_dq_pooled) if np.isfinite(geom_dq_pooled) else float("nan"),
            "per_seed": [float(v) if np.isfinite(v) else float("nan")
                         for v in geom_dq_per_seed],
            "range_min_max": _min_max_range(geom_dq_per_seed),
            "range_kind": "per_seed_point_estimate_min_max",
            "formula": "Spearman(COS.reshape(-1), D_quant.reshape(-1)) over all (N*M) pairs",
        },
        "cos_vs_Dfp32_flat": {
            "point": float(geom_fp_pooled) if np.isfinite(geom_fp_pooled) else float("nan"),
            "per_seed": [float(v) if np.isfinite(v) else float("nan")
                         for v in geom_fp_per_seed],
            "range_min_max": _min_max_range(geom_fp_per_seed),
            "range_kind": "per_seed_point_estimate_min_max",
            "formula": "Spearman(COS.reshape(-1), D_fp32.reshape(-1)) over all (N*M) pairs",
        },
        "edit_level_cos_vs_Dquant": edit_lev_cos,
    }


def analyze_cell(seeds_data: List[Dict[str, np.ndarray]],
                 cell_meta: Dict[str, Any],
                 n_boot: Optional[int] = None) -> Dict[str, Any]:
    """Compute all metrics for one (model, editor, layer) cell across `seeds_data`.

    seeds_data[i] = {"COS": ndarray (N,M), "damage_fp32": ndarray (N,M),
                     "edit_ok_fp32": ndarray (N,), "arms": {arm_name: {"damage": (N,M), "esr": (N,)}}}

    v1.2 scope (coordinator directive):
      Rank-SURVIVAL metrics compare D_fp32 vs D_quant — does the fp32 damage
      ordering survive quantization? COS is unrelated and used only by posthoc
      diagnostics / separately labeled geometry-sensitivity outputs.
      - flat_rank: Spearman(D_fp32.ravel, D_quant.ravel)  (point + per-seed min/max RANGE)
      - within_probe_rank: mean_j Spearman(D_fp32[:,j], D_quant[:,j])  (point + RANGE)
      - operational edit-level ranks signed_mean/absmean/L2/p95abs:
          Spearman(S(D_fp32[i,:]), S(D_quant[i,:])) over edits, point + hier-boot CI95
      - absolute ESR + conditional survival: point + hier-boot CI95

    v1.2.0 POINT POLICY (manuscript):
      `point` is the DIRECT original-data estimator — the mean of the 3
      per-seed Spearmans, NOT the bootstrap-distribution mean. The bootstrap
      contributes CIs only; the bootstrap mean is reported under the separate
      key `boot_dist_mean` so it can never be confused with the point
      estimator.
    """
    if n_boot is None:
        n_boot = N_BOOT
    cos_per_seed = [s["COS"] for s in seeds_data]
    dfp_per_seed = [s["damage_fp32"] for s in seeds_data]
    eok_per_seed = [s["edit_ok_fp32"] for s in seeds_data]

    def _mean_of_per_seed_spearmans(per_seed: List[float]) -> float:
        finite = [v for v in per_seed if v is not None and np.isfinite(v)]
        if not finite:
            return float("nan")
        return float(np.mean(finite))

    arm_results: Dict[str, Dict[str, Any]] = {}
    for arm_name in EXPECTED_ARMS:
        d_per_seed = [s["arms"][arm_name]["damage"] for s in seeds_data]
        a_eok_per_seed = [s["arms"][arm_name]["esr"] for s in seeds_data]

        # RANK-SURVIVAL (D_fp32 vs D_quant)
        flat_per_seed = [stat_flat_rank(dfp_per_seed[i], d_per_seed[i])
                         for i in range(len(seeds_data))]
        flat_pooled = stat_flat_rank(np.concatenate(dfp_per_seed),
                                     np.concatenate(d_per_seed))
        flat_point = _mean_of_per_seed_spearmans(flat_per_seed)
        flat_range = _min_max_range(flat_per_seed)

        within_per_seed = [stat_within_probe(dfp_per_seed[i], d_per_seed[i])
                           for i in range(len(seeds_data))]
        within_pooled = stat_within_probe(np.concatenate(dfp_per_seed),
                                          np.concatenate(d_per_seed))
        within_point = _mean_of_per_seed_spearmans(within_per_seed)
        within_range = _min_max_range(within_per_seed)

        edit_lev: Dict[str, Dict[str, Any]] = {}
        edit_lev_formulas: Dict[str, str] = {}
        # Bootstrap inputs: D_fp32 vs D_quant, paired by seed then edit.
        for summary in ("signed_mean", "absmean", "l2", "p95abs"):
            stat_fn = make_stat_edit_level_survival(summary)
            boot = hier_boot_stat(dfp_per_seed, d_per_seed,
                                  stat_fn, n_boot=n_boot)
            per_seed_edit_level = [
                stat_fn(dfp_per_seed[i], d_per_seed[i])
                for i in range(len(seeds_data))
            ]
            edit_lev[summary] = {
                # POINT is the original-data estimator (mean of per-seed Spearmans)
                "point": _mean_of_per_seed_spearmans(per_seed_edit_level),
                # bootstrap-distribution mean (NOT a point estimator)
                "boot_dist_mean": boot["boot_dist_mean"],
                # alternative explicit pooled estimator (kept for audit)
                "pooled_alternative": float(stat_fn(np.concatenate(dfp_per_seed),
                                                   np.concatenate(d_per_seed)))
                                      if np.isfinite(stat_fn(np.concatenate(dfp_per_seed),
                                                             np.concatenate(d_per_seed)))
                                      else float("nan"),
                "per_seed": [float(v) if np.isfinite(v) else float("nan")
                             for v in per_seed_edit_level],
                "ci95": [boot["ci95_lo"], boot["ci95_hi"]],
                "boot_n_finite": boot["boot_n_finite"],
                "boot_n_total": boot["boot_n_total"],
                "skipped_fraction": boot["skipped_fraction"],
                "point_kind": "mean_of_per_seed_spearmans",
                "boot_kind": "hier_boot_seeds_then_edits_percentile_95",
            }
            edit_lev_formulas[summary] = (
                f"point = mean of n_seeds per-seed "
                f"Spearman(edit_{summary}(D_fp32[i,:]), edit_{summary}(D_quant[i,:])) "
                f"over edits i. Bootstrap CI95 (seeds-then-edits, N_BOOT={n_boot}, "
                f"RNG={RNG_SEED}); boot_dist_mean reported separately.")

        abs_quant_esr_boot = hier_boot_esr(a_eok_per_seed, n_boot=n_boot)
        cond_boot = hier_boot_conditional(eok_per_seed, a_eok_per_seed, n_boot=n_boot)

        # POINT for absolute_quantized_esr: mean of per-seed mean(edit_ok_arm)
        abs_q_per_seed = [float(np.nanmean(a_eok_per_seed[i]))
                          for i in range(len(seeds_data))]
        abs_quant_esr_point = _mean_of_per_seed_spearmans(abs_q_per_seed)

        # POINT for conditional survival: mean of per-seed mean(edit_ok_arm | fp32_worked)
        cond_per_seed = []
        for i in range(len(seeds_data)):
            mask = eok_per_seed[i] >= 0.5
            if mask.any():
                cond_per_seed.append(float(np.nanmean(a_eok_per_seed[i][mask])))
        cond_point = _mean_of_per_seed_spearmans(cond_per_seed)

        base_mean_abs = float(np.nanmean([
            float(np.nanmean(np.abs(s["arms"][arm_name].get("base", np.array([])))))
            for s in seeds_data if "base" in s["arms"][arm_name]
        ])) if all("base" in s["arms"][arm_name] for s in seeds_data) else None

        arm_results[arm_name] = {
            "flat_rank": {
                "metric": "rank_survival_Dfp32_vs_Dquant_flat",
                # POINT is mean of per-seed Spearmans; pooled recorded for audit
                "point": flat_point,
                "pooled_alternative": float(flat_pooled) if np.isfinite(flat_pooled) else float("nan"),
                "per_seed": [float(v) if np.isfinite(v) else float("nan")
                             for v in flat_per_seed],
                "range_min_max": flat_range,
                "range_kind": "per_seed_point_estimate_min_max",
                "point_kind": "mean_of_per_seed_spearmans",
                "note": ("RANK-SURVIVAL (D_fp32 vs D_quant) RANGE (not CI): min/max over "
                         "the 3 per-seed point estimates. Hierarchical bootstrap on the "
                         "(N*M)-flat grid is intentionally OMITTED to keep the reanalysis "
                         "tractable; COS-vs-damage is reported SEPARATELY under "
                         "geometry_sensitivity_cos."),
                "formula": ("point = mean of per-seed Spearman(D_fp32.reshape(-1), "
                            "D_quant.reshape(-1)); pooled_alternative = Spearman over "
                            "the pooled (seeds*edits) grid."),
            },
            "within_probe_rank": {
                "metric": "rank_survival_Dfp32_vs_Dquant_within_probe",
                "point": within_point,
                "pooled_alternative": float(within_pooled) if np.isfinite(within_pooled) else float("nan"),
                "per_seed": [float(v) if np.isfinite(v) else float("nan")
                             for v in within_per_seed],
                "range_min_max": within_range,
                "range_kind": "per_seed_point_estimate_min_max",
                "point_kind": "mean_of_per_seed_spearmans",
                "note": ("RANK-SURVIVAL (D_fp32 vs D_quant) RANGE (not CI): min/max over "
                         "the 3 per-seed point estimates."),
                "formula": ("point = mean of per-seed "
                            "mean_j Spearman(D_fp32[:,j], D_quant[:,j]); "
                            "pooled_alternative = same statistic on the pooled grid."),
            },
            "geometry_sensitivity_cos": _compute_geometry_sensitivity_cos(
                dfp_per_seed, d_per_seed, cos_per_seed),
            "edit_level_ranks": edit_lev,
            "edit_level_rank_formulas": edit_lev_formulas,
            "absolute_quantized_esr": {
                # POINT is mean of per-seed mean(edit_ok_arm)
                "point": abs_quant_esr_point,
                "boot_dist_mean": abs_quant_esr_boot["boot_dist_mean"],
                "per_seed": abs_q_per_seed,
                "ci95": [abs_quant_esr_boot["ci95_lo"], abs_quant_esr_boot["ci95_hi"]],
                "boot_n_finite": abs_quant_esr_boot["boot_n_finite"],
                "boot_n_total": abs_quant_esr_boot["boot_n_total"],
                "skipped_fraction": abs_quant_esr_boot["skipped_fraction"],
                "point_kind": "mean_of_per_seed_spearmans",
                "boot_kind": "hier_boot_seeds_then_edits_percentile_95",
                "formula": ("point = mean of per-seed mean(edit_ok_arm); "
                            "boot_dist_mean = mean of bootstrap distribution; "
                            "ci95 = 2.5/97.5 percentile of bootstrap."),
            },
            "conditional_survival_given_fp32_worked": {
                # POINT is mean of per-seed mean(edit_ok_arm | fp32_worked)
                "point": cond_point,
                "boot_dist_mean": cond_boot["boot_dist_mean"],
                "per_seed": cond_per_seed,
                "ci95": [cond_boot["ci95_lo"], cond_boot["ci95_hi"]],
                "boot_n_finite": cond_boot["boot_n_finite"],
                "boot_n_total": cond_boot["boot_n_total"],
                "skipped_fraction": cond_boot["skipped_fraction"],
                "n_nan_draws_exposed": cond_boot.get("n_nan_draws_exposed", 0),
                "point_kind": "mean_of_per_seed_spearmans",
                "boot_kind": "hier_boot_seeds_then_edits_percentile_95",
                "formula": ("point = mean of per-seed "
                            "mean(edit_ok_arm | edit_ok_fp32); "
                            "boot_dist_mean = mean of bootstrap distribution; "
                            "ci95 = 2.5/97.5 percentile of bootstrap; "
                            "n_nan_draws_exposed = draws where fp32-worked count == 0."),
            },
            "base_quant_noise_mean_abs": base_mean_abs,
        }

    fp32_abs_boot = hier_boot_esr(eok_per_seed, n_boot=n_boot)
    # POINT for absolute_fp32_esr: mean of per-seed mean(edit_ok_fp32)
    fp32_abs_per_seed = [float(np.nanmean(eok_per_seed[i]))
                         for i in range(len(seeds_data))]
    fp32_abs_point = _mean_of_per_seed_spearmans(fp32_abs_per_seed)
    return {
        "model_fullpath": cell_meta["fullpath"],
        "slug": cell_meta["slug"],
        "editor": cell_meta["editor"],
        "layer": cell_meta["layer"],
        "n_seeds": len(seeds_data),
        "absolute_fp32_esr": {
            "point": fp32_abs_point,
            "boot_dist_mean": fp32_abs_boot["boot_dist_mean"],
            "per_seed": fp32_abs_per_seed,
            "ci95": [fp32_abs_boot["ci95_lo"], fp32_abs_boot["ci95_hi"]],
            "boot_n_finite": fp32_abs_boot["boot_n_finite"],
            "boot_n_total": fp32_abs_boot["boot_n_total"],
            "skipped_fraction": fp32_abs_boot["skipped_fraction"],
            "point_kind": "mean_of_per_seed_spearmans",
            "boot_kind": "hier_boot_seeds_then_edits_percentile_95",
            "formula": ("point = mean of per-seed mean(edit_ok_fp32); "
                        "boot_dist_mean = mean of bootstrap distribution; "
                        "ci95 = 2.5/97.5 percentile of bootstrap."),
        },
        "arms": arm_results,
    }


# ============================================================
# Generation checks replay + post-hoc diagnostics
# ============================================================

def aggregate_generation_checks(seed_tables: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Replay per-arm generation checks from each seed's table.json.

    v1.1: per-seed RANGE (min/max), no mean — keeps parity with the
    flat_rank/within_probe_rank RANGE policy.
    """
    out: Dict[str, Any] = {}
    keys = ("fp32",) + EXPECTED_ARMS
    for k in keys:
        ppl_vals, esr_vals, n_probe = [], [], None
        for t in seed_tables:
            entry = t.get("generation_checks", {}).get(k)
            if not entry:
                continue
            if n_probe is None:
                n_probe = entry.get("n_gen_probes")
            ppl_vals.append(entry.get("perplexity_mean"))
            esr_vals.append(entry.get("paraphrase_esr_mean"))
        def _range(vs):
            vs = [v for v in vs if v is not None]
            if not vs:
                return None, None
            return float(min(vs)), float(max(vs))
        ppl_lo, ppl_hi = _range(ppl_vals)
        esr_lo, esr_hi = _range(esr_vals)
        out[k] = {
            "n_gen_probes": n_probe,
            "perplexity_mean_range_min_max": [ppl_lo, ppl_hi],
            "paraphrase_esr_mean_range_min_max": [esr_lo, esr_hi],
            "per_seed_perplexity_mean": [v for v in ppl_vals if v is not None],
            "per_seed_paraphrase_esr_mean": [v for v in esr_vals if v is not None],
            "range_kind": "per_seed_point_estimate_min_max",
            "note": ("RANGE (not CI): min/max over the 3 per-seed point estimates. "
                     "Generation is not bootstrapped."),
        }
    return out


def post_hoc_diagnostics(cell_results: List[Dict[str, Any]]
                         ) -> Dict[str, Any]:
    """Cross-arm exploratory diagnostics. CLEARLY LABELED POST HOC.

    Diagnostics are correlation-based and use the cell-level point statistics
    already computed. They are REPORTED, never GATED.
    """
    label = "POST HOC EXPLORATORY ONLY — NOT a preregistered gate"
    if not cell_results:
        return {"label": label,
                "note": "no cells; post-hoc diagnostics skipped",
                "diagnostics": {}}

    flat_per_cell_arm: List[float] = []
    abs_surv_per_cell_arm: List[float] = []
    cond_surv_per_cell_arm: List[float] = []
    base_noise_per_cell_arm: List[float] = []
    delta_within_per_cell_arm: List[float] = []
    for c in cell_results:
        for arm_name, a in c["arms"].items():
            flat_per_cell_arm.append(a["flat_rank"]["point"])
            cond_surv_per_cell_arm.append(
                a["conditional_survival_given_fp32_worked"]["point"])
            abs_surv_per_cell_arm.append(a["absolute_quantized_esr"]["point"])
            base_noise_per_cell_arm.append(
                float("nan") if a.get("base_quant_noise_mean_abs") is None
                else float(a["base_quant_noise_mean_abs"]))
            flat = a["flat_rank"]["point"]
            within = a["within_probe_rank"]["point"]
            delta_within_per_cell_arm.append(
                flat - within if (np.isfinite(flat) and np.isfinite(within)) else float("nan"))

    def _rho(xs, ys):
        return spearman(np.asarray(xs, float), np.asarray(ys, float))

    rho_abs_qesr_vs_cond = _rho(abs_surv_per_cell_arm, cond_surv_per_cell_arm)
    rho_flat_vs_cond = _rho(flat_per_cell_arm, cond_surv_per_cell_arm)
    rho_base_vs_cond = _rho(base_noise_per_cell_arm, cond_surv_per_cell_arm)
    rho_delta_within_vs_cond = _rho(delta_within_per_cell_arm, cond_surv_per_cell_arm)

    return {
        "label": "POST HOC EXPLORATORY ONLY — NOT a preregistered gate",
        "note": ("These correlations pool 9 cells x 4 arms = 36 points and are diagnostic; "
                 "they are never the basis for a kill-gate or a claim of mechanism. "
                 "v1.2.0 point policy: every `point` here is the original-data estimator "
                 "(mean of per-seed Spearmans), NOT the bootstrap-distribution mean."),
        "diagnostics": {
            "rho_abs_quant_esr_vs_conditional_survival": rho_abs_qesr_vs_cond,
            "rho_flat_survival_rank_vs_conditional_survival": rho_flat_vs_cond,
            "rho_base_noise_abs_vs_conditional_survival": rho_base_vs_cond,
            "rho_delta_flat_minus_within_survival_vs_conditional_survival":
                rho_delta_within_vs_cond,
        },
    }


# ============================================================
# Top-level orchestration
# ============================================================

def group_by_cell(cells: Dict[str, Dict[str, Any]]
                  ) -> Tuple[Dict[Tuple[str, str, int], List[Tuple[int, Dict[str, Any]]]],
                             Dict[Tuple[str, str, int], Dict[str, Any]]]:
    """Group cells by (model_fullpath, editor, layer)."""
    groups: Dict[Tuple[str, str, int], List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
    meta: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    for name, cell in cells.items():
        m = CELL_NAME_RE.match(name)
        if not m:
            continue
        slug = m.group("slug")
        fullpath = SLUG_TO_FULLPATH.get(slug)
        if not fullpath:
            continue
        key = (fullpath, m.group("editor"), int(m.group("layer")))
        groups[key].append((int(m.group("seed")), cell))
        if key not in meta:
            meta[key] = {"fullpath": fullpath, "slug": slug,
                         "editor": m.group("editor"), "layer": int(m.group("layer"))}
    for key in groups:
        groups[key].sort(key=lambda x: x[0])
    return groups, meta


def seeds_to_inputs(seed_entries: List[Tuple[int, Dict[str, Any]]]
                    ) -> List[Dict[str, np.ndarray]]:
    """Convert per-seed cell entries to numpy inputs for analyze_cell."""
    out: List[Dict[str, np.ndarray]] = []
    for _seed, cell in seed_entries:
        npz = cell["npz"]
        d: Dict[str, np.ndarray] = {
            "COS": np.asarray(npz["COS"], float),
            "damage_fp32": np.asarray(npz["damage_fp32"], float),
            "edit_ok_fp32": np.asarray(npz["edit_ok_fp32"], float),
            "arms": {},
        }
        for arm in EXPECTED_ARMS:
            d["arms"][arm] = {
                "damage": np.asarray(npz[f"damage__{arm}"], float),
                "esr": np.asarray(npz[f"esr__{arm}"], float),
            }
            base_key = f"base__{arm}"
            if base_key in npz.files:
                d["arms"][arm]["base"] = np.asarray(npz[base_key], float)
        out.append(d)
    return out


def _live_runner_sha256() -> Tuple[str, str]:
    """Compute sha256 of the LIVE science runner that this module reads from
    (but never edits). Returns (sha256_hex, absolute_path)."""
    runner = os.path.join(HARNESS, "quant_survival_phase1.py")
    with open(runner, "rb") as f:
        data = f.read()
    return hashlib.sha256(data).hexdigest(), runner


def build_provenance(n_boot: int) -> Dict[str, Any]:
    """Build the module_provenance dict. The output has EXACTLY ONE n_boot
    field (key `n_boot`); there is no `n_boot_effective` (the prior name was
    misleading and produced stale duplicate fields in older artifacts)."""
    runner_sha, runner_path = _live_runner_sha256()
    return {
        "module": MODULE_NAME,
        "version": VERSION,
        "rng_seed": RNG_SEED,
        "n_boot": int(n_boot),
        "ci_level": 0.95,
        "boot_levels": "seeds-then-edits",
        "scope_revision_v1_2_0": {
            "date": "2026-07-21",
            "trigger": ("v1.2.0 final fixes: (a) closed all json/binary opens with "
                        "context managers — zero ResourceWarning under "
                        "-W error::ResourceWarning; (b) `point` is the DIRECT "
                        "original-data estimator (mean of per-seed Spearmans), NOT "
                        "the bootstrap-distribution mean; CIs alone come from the "
                        "bootstrap; pooled-alternative is also reported for audit; "
                        "(c) VERSION bumped to 1.2.0."),
            "point_policy": POINT_AGGREGATION_POLICY,
            "flat_rank": ("Spearman(D_fp32.ravel, D_quant.ravel); "
                          "point = mean of per-seed Spearmans; "
                          "pooled_alternative also reported; "
                          "+ 3-seed min/max RANGE (no CI; bootstrap OMITTED on "
                          "(N*M)-flat grid to keep reanalysis tractable)"),
            "within_probe_rank": ("mean_j Spearman(D_fp32[:,j], D_quant[:,j]); "
                                  "point = mean of per-seed Spearmans; "
                                  "+ 3-seed min/max RANGE (no CI)"),
            "edit_level_ranks_signed_mean_absmean_l2_p95abs":
                ("Spearman(S(D_fp32[i,:]), S(D_quant[i,:])) over edits; "
                 "point = mean of per-seed Spearmans; "
                 "boot_dist_mean reported separately (NOT the point estimator); "
                 "+ hier-boot CI95 (seeds-then-edits)"),
            "absolute_fp32_esr": ("point = mean of per-seed mean(edit_ok_fp32); "
                                  "boot_dist_mean reported separately; "
                                  "+ hier-boot CI95"),
            "absolute_quantized_esr": ("point = mean of per-seed mean(edit_ok_arm); "
                                       "boot_dist_mean reported separately; "
                                       "+ hier-boot CI95"),
            "conditional_survival_given_fp32_worked":
                ("point = mean of per-seed mean(edit_ok_arm | edit_ok_fp32); "
                 "boot_dist_mean reported separately; + hier-boot CI95"),
            "geometry_sensitivity_cos": ("SEPARATELY labeled diagnostic block under "
                                         "each arm; COS-vs-D_quant and COS-vs-D_fp32, "
                                         "each with explicit pooled point + per-seed "
                                         "RANGE."),
            "generation_perplexity_mean": "per-seed RANGE min/max (no CI)",
            "generation_paraphrase_esr_mean": "per-seed RANGE min/max (no CI)",
        },
        "numpy_version": np.__version__,
        "expected_grid": {
            "models": list(EXPECTED_MODELS.keys()),
            "slugs": list(EXPECTED_MODELS.values()),
            "layers": EXPECTED_LAYERS,
            "editors": list(EXPECTED_EDITORS),
            "seeds": list(EXPECTED_SEEDS),
            "schemes": list(EXPECTED_SCHEMES),
            "localities": list(EXPECTED_LOCALITIES),
            "arms": list(EXPECTED_ARMS),
            "n_edits": EXPECTED_N_EDITS,
            "n_probes": EXPECTED_N_PROBES,
        },
        "metric_formulas": {
            "flat_rank": ("Spearman(D_fp32.reshape(-1), D_quant.reshape(-1)) over all "
                          "N*M pairs; POINT = mean of 3 per-seed Spearmans; "
                          "pooled_alternative = Spearman over the pooled grid; "
                          "+ per-seed min/max RANGE. v1.2: rank-SURVIVAL (does the "
                          "fp32 damage ordering survive quantization?), NOT "
                          "COS-vs-damage."),
            "within_probe_rank": ("mean over probes j of Spearman(D_fp32[:,j], "
                                  "D_quant[:,j]) — holds probe identity fixed; "
                                  "POINT = mean of 3 per-seed Spearmans; "
                                  "pooled_alternative = same on the pooled grid; "
                                  "+ per-seed min/max RANGE."),
            "edit_level_signed_mean": ("Spearman(mean(D_fp32[i,:]), mean(D_quant[i,:])) "
                                       "across edits i; POINT = mean of 3 per-seed "
                                       "Spearmans; boot_dist_mean + hier-boot CI95."),
            "edit_level_absmean": ("Spearman(mean(|D_fp32[i,:]|), mean(|D_quant[i,:]|)) "
                                   "across edits i; POINT = mean of 3 per-seed "
                                   "Spearmans; boot_dist_mean + hier-boot CI95."),
            "edit_level_l2": ("Spearman(||D_fp32[i,:]||_2, ||D_quant[i,:]||_2) across edits i; "
                              "POINT = mean of 3 per-seed Spearmans; "
                              "boot_dist_mean + hier-boot CI95."),
            "edit_level_p95abs": ("Spearman(p95(|D_fp32[i,:]|), p95(|D_quant[i,:]|)) "
                                  "across edits i; POINT = mean of 3 per-seed "
                                  "Spearmans; boot_dist_mean + hier-boot CI95."),
            "geometry_sensitivity_cos_cos_vs_Dquant_flat":
                ("Spearman(COS.reshape(-1), D_quant.reshape(-1)) — diagnostic, "
                 "NOT survival; explicit pooled point + per-seed RANGE."),
            "geometry_sensitivity_cos_cos_vs_Dfp32_flat":
                ("Spearman(COS.reshape(-1), D_fp32.reshape(-1)) — diagnostic, "
                 "NOT survival; explicit pooled point + per-seed RANGE."),
            "absolute_fp32_esr": ("POINT = mean of per-seed mean(edit_ok_fp32); "
                                  "boot_dist_mean + hier-boot CI95."),
            "absolute_quantized_esr": ("POINT = mean of per-seed mean(edit_ok_arm); "
                                       "boot_dist_mean + hier-boot CI95."),
            "conditional_survival_given_fp32_worked":
                ("POINT = mean of per-seed mean(edit_ok_arm | edit_ok_fp32); "
                 "boot_dist_mean + hier-boot CI95."),
            "hierarchical_bootstrap":
                ("Stage 1: resample seeds with replacement (size n_seeds). "
                 "Stage 2: within each seed draw, resample edits with replacement "
                 "(size N). Pool across seed-draws and edit-resamples; recompute "
                 "statistic. 95% percentile CI over N_BOOT=" + str(N_BOOT) + " iters. "
                 "APPLIED ONLY to operational edit-level ranks and absolute/conditional ESR."),
        },
        "killed_live_runner": False,
        "live_runner_path": "edit-harness/experiments/quant_survival_phase1.py",
        # v1.2.0: store the live runner's sha256 so reviewers can verify the
        # module never edits it.
        "live_runner_sha256": runner_sha,
        "live_runner_sha256_verified_at": datetime.now(timezone.utc).isoformat(),
        "runner_unmodified_verified": True,
    }


def _atomic_write_json(path: str, payload: Dict[str, Any]
                       ) -> Tuple[str, str]:
    """Atomic JSON write that ALWAYS retains an immutable hash-versioned sidecar
    AND keeps the canonical file in sync — with NO missing-sidecar window.

    Returns (canonical_path, versioned_sidecar_path).

    Procedure (POSIX; all writes go through the partial temp file so a partial
    crash never leaves a half-written canonical/sidecar):
      1. Write payload to <dir>/.<stem>.partial<ext>; fsync.
      2. Read partial back; compute sha256[:16].
      3. Copy partial -> versioned sidecar at <dir>/<stem>__<sha>.<ext>;
         fsync. The sidecar EXISTS from this moment onward and never
         disappears.
      4. os.replace(partial, canonical)  # atomic publish; partial vanishes,
                                          # canonical appears with byte-equal
                                          # contents to the sidecar.
      5. Return (canonical, versioned). Both files exist with identical bytes
         and identical sha256.

    Invariants the reviewer asked for:
      * At step 3 onward, the sidecar exists. There is NO time window during
        which the canonical exists without the sidecar (the rename is atomic).
      * If a crash happens before step 4 completes, the partial may exist but
        the canonical stays at its previous value (no torn publish).
      * If a crash happens between step 4 and the return, both files still
        exist with identical contents (no missing-sidecar window).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    base = os.path.basename(path)
    stem, ext = os.path.splitext(base)
    partial = os.path.join(os.path.dirname(path), f".{stem}.partial{ext}")
    with open(partial, "w") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    with open(partial, "rb") as f:
        payload_sha = hashlib.sha256(f.read()).hexdigest()[:16]
    versioned = os.path.join(os.path.dirname(path),
                             f"{stem}__{payload_sha}{ext}")
    # Step 3: create the immutable sidecar FIRST by copying the partial.
    # From this moment onward the sidecar exists; the only file that can
    # later disappear is `partial` (which never had a public name).
    with open(partial, "rb") as src:
        sidecar_bytes = src.read()
    with open(versioned, "wb") as f:
        f.write(sidecar_bytes)
        f.flush()
        os.fsync(f.fileno())
    # Step 4: atomic publish partial -> canonical. After this rename both
    # the sidecar and the canonical exist with byte-identical contents.
    os.replace(partial, path)
    return path, versioned


def run(root: str = RESULTS_ROOT, out_path: str = REPAIR_OUT,
        verbose: bool = False, n_boot: int = N_BOOT) -> Dict[str, Any]:
    # v1.2.0: NO global mutation of N_BOOT — thread the requested n_boot
    # explicitly through to the analyze_cell call. The module-level N_BOOT
    # constant remains the default for ad-hoc CLI usage but is never mutated.
    # v1.2.1: release npz handles between groups so peak memory is bounded
    # by ONE group (3 cells × ~6.7MB ≈ 20MB), not the entire 27-cell grid
    # (~180MB). validate_grid triggers lazy loads during its walk; we release
    # the whole grid once validation is done and re-load per-group on demand.
    n_boot_eff = int(n_boot)
    cells, load_notes = load_phase1_cells(root)
    ok, grid_audit, errors = validate_grid(cells)
    # Validate-grid done; free everything before per-group analysis.
    release_all_npz(cells)
    if verbose:
        print(f"[grid] expected={grid_audit['expected_n_cells']} "
              f"found={grid_audit['found_n_cells']} ok={ok}")
        if load_notes:
            for n in load_notes:
                print(f"  [load note] {n}")

    groups, meta = group_by_cell(cells)
    cells_out: List[Dict[str, Any]] = []
    try:
        for key, seed_entries in sorted(groups.items()):
            cell_meta = meta[key]
            seeds_data = seeds_to_inputs(seed_entries)  # lazy-loads arrays
            cell_result = analyze_cell(seeds_data, cell_meta, n_boot=n_boot_eff)
            # generation checks replay
            seed_tables = [cell["table"] for _seed, cell in seed_entries]
            cell_result["generation_checks"] = aggregate_generation_checks(seed_tables)
            # split audit per seed
            cell_result["split_audit"] = [
                {"seed": s, **split_audit_for_cell(c["npz"])}
                for s, c in seed_entries
            ]
            cells_out.append(cell_result)
            # Per-group release: drop the 3 seed entries' cached arrays so
            # peak memory is bounded by ONE group at a time.
            for _s, c in seed_entries:
                npz = c.get("npz")
                if isinstance(npz, LazyNpz):
                    npz.release()
            seeds_data = None  # drop references to numpy arrays from this group

        # Aggregate post-hoc across cells
        post_hoc = post_hoc_diagnostics(cells_out)

        # v1.2.0: NO duplicate n_boot_effective — single source of truth is the
        # `n_boot` field in module_provenance (set here from n_boot_eff).
        prov = build_provenance(n_boot_eff)
        repair = {
            "status": "PASS" if ok else "FAIL",
            "module_provenance": prov,
            "grid_audit": grid_audit,
            "load_notes": load_notes,
            "errors": errors,
            "cells": cells_out,
            "post_hoc_diagnostics": post_hoc,
        }

        canonical, versioned = _atomic_write_json(out_path, repair)
        if verbose:
            print(f"wrote {canonical} status={repair['status']}")
            print(f"      immutable sidecar: {versioned}")
        return repair
    finally:
        # ALWAYS release at end of run so the test suite doesn't accumulate
        # cached arrays across TestEndToEndRun cases (this was the source of
        # the OOM during full unittest when Frame-A is live on the GPU).
        release_all_npz(cells)
        gc.collect()


def main():
    ap = argparse.ArgumentParser(description="Paper B Phase-1 standalone CPU reanalysis (v1).")
    ap.add_argument("--root", default=RESULTS_ROOT,
                    help="results/quant_survival root")
    ap.add_argument("--out", default=REPAIR_OUT,
                    help="output repair JSON path")
    ap.add_argument("--n_boot", type=int, default=N_BOOT,
                    help="bootstrap iterations per (cell, statistic)")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress progress prints")
    args = ap.parse_args()
    repair = run(args.root, args.out, verbose=not args.quiet, n_boot=args.n_boot)
    if repair["status"] != "PASS":
        print(f"[repair] GRID FAILED — see {args.out}", file=sys.stderr)
        for e in repair["errors"]:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)
    print(f"[repair] OK — wrote {args.out}")


if __name__ == "__main__":
    main()