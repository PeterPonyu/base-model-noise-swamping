#!/bin/bash
# run_revwave_rd.sh — R-D revision-wave cell: MEMIT TRUE-COVARIANCE federation
# (Llama-3.2-1B L12, --memit_cov wiki, cf, 3 seeds, g=2..20, n=200).
#
# WHY: the existing MEMIT arm in experiments/merging_editors.py defaults to --memit_cov identity
# ("MEMIT-style", ROME-covariance-free — see the prereg PREREG-FED-EDITORS-2026-07-16.md, which
# explicitly earns the "MEMIT" name only if a REAL (non-identity) covariance is used). This wave
# runs the REAL covariance path: --memit_cov wiki estimates C_l = E[k k^T] per edited layer from
# an external text corpus (added to experiments/merging_editors.py in this same revision —
# _get_or_build_memit_cov / _load_wiki_or_fallback_prompts / the cov_cache/ on-disk cache), rather
# than this cell's own holdout bank (--memit_cov generic).
#
# DATA SOURCE NOTE (binding for the writeup): NO wikitext corpus is present under data/ as of
# 2026-07-16 (verified: `ls data/` has no wiki* entry; ask-first download policy — CLAUDE.md — was
# not exercised for this build). --memit_cov wiki therefore falls through to the documented
# CF-FALLBACK (a broad, cell-independent CounterFact prompt sample, seed=999, independent of
# --seed) — the run will log "source=cf_fallback", NOT "source=wiki". The eventual paper/prereg
# text must say "MEMIT-style multi-layer spread (CF-derived covariance)", never "true wikitext
# covariance", UNLESS a real wikitext file is later dropped under data/ (see
# _wiki_corpus_candidates in experiments/merging_editors.py) and this cell is re-run — the cache
# key includes the source string, so a wiki-corpus re-run after a manual download would NOT reuse
# a stale cf_fallback cache by mistake.
#
# ΔW-FIDELITY GATE HONESTY: this arm reuses the SAME real editors/memit.py apply_edit install as
# the existing --memit_cov generic arm (only the covariance PROMPT SOURCE changes, not the
# editor's math) — the fidelity gate compares the re-derived ΔW against that real install, exactly
# like every other cell this driver's template (run_merging_editors.sh) runs. No downgrade to a
# decomposition-level anchor was needed.
#
# NAMESPACE COLLISION AVOIDED: the bundle directory this cell writes to
# (results/merging_editors/Llama-3.2-1B_wiki_memit_cf_L12_RG/) is DISTINCT from the already-landed
# identity-cov bundle (Llama-3.2-1B_memit_cf_L12_RG/) at the same (model, layer, dataset) — see
# experiments/merging_editors.py's _cov_variant_suffix + the matching COV_SUFFIX line this
# revision added to run_merging_editors.sh. Verified via DRYRUN (see
# docs/plans/REVWAVE-BUILD-NOTES-2026-07-16.md).
#
# THIS DRIVER ADDS NO NEW BASH LOGIC beyond env pass-through — it is a thin wrapper around the
# ALREADY-REVIEWED run_merging_editors.sh (preflight, GPU-idle gate, CPU --selftest smoke gate,
# real-model ΔW-fidelity gate, refuse-clobber, PID-by-file / kill -0 only).
#
# BUILD-ONLY as authored 2026-07-16: CPU-validated only (bash -n, DRYRUN=1, plus a manual CPU
# smoke of the new covariance-cache code path on Qwen2.5-0.5B — see build notes); NOT launched.
set -u
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$H" || exit 1

MODEL_DIR=${MODEL_DIR:-/root/autodl-tmp/models/Llama-3.2-1B}
MODEL_TAG=${MODEL_TAG:-llama1b_truecov}   # distinct from the identity-cov "llama1b" tag used
                                          # elsewhere — table filenames also stay non-colliding
export MODEL_DIR MODEL_TAG
export EDITOR=memit
export DATASET=cf
export LAYER=${LAYER:-12}
export MEMIT_COV=wiki
export RG_SEEDS=${RG_SEEDS:-0,1,2}
export RG_GROUP_SIZES=${RG_GROUP_SIZES:-2,3,5,10,20}
export N_EDITS=${N_EDITS:-200}
export N_HOLDOUT=${N_HOLDOUT:-50}
export KEEP_RATIO=${KEEP_RATIO:-0.99}
# MEMIT's per-layer covariance build (estimate_layer_covariances, ~50k-token target) is a
# ONE-TIME cost per (model, layer-span, source) — cached under results/merging_editors/cov_cache/
# and reused across all 3 seeds, so only the FIRST seed pays it. Padded generously over the
# existing identity-cov MEMIT estimate (run_merging_editors.sh's own EST_MIN convention).
export BUDGET_MIN=${BUDGET_MIN:-180}
export EST_MIN=${EST_MIN:-75}
export DRYRUN=${DRYRUN:-0}

exec ./run_merging_editors.sh
