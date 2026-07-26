"""e1_width_table.py — CP-Edit E1 headline certified-width table (ROME + AlphaEdit).

Reruns the KG-0 machinery on BOTH editors. Produces the 8-row (layer x editor)
headline table, the AlphaEdit/ROME width-ratio vs C4 damage-ratio consistency
block, and the mandatory marginal-scope paragraph.

CPU only. 0 GPU, 0 downloads.
Writes results/cpedit/CP_E1_width_table.{json,md} and CP_E1_marginal_scope.md.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cp_edit import io, conformal

RES = os.path.abspath(os.path.join(io.HERE, "..", "..", "results"))
OUT_JSON = os.path.join(RES, "cpedit", "CP_E1_width_table.json")
OUT_MD = os.path.join(RES, "cpedit", "CP_E1_width_table.md")
OUT_SCOPE = os.path.join(RES, "cpedit", "CP_E1_marginal_scope.md")
C4_PATH = os.path.join(RES, "C4_causal_table.json")

SCOPE_PARAGRAPH = """\
## CP-Edit certificate scope (marginal split-conformal)

The certified upper bounds reported here are **marginal split-conformal
guarantees**: for edits drawn exchangeably from the same distribution as the
calibration set, the bound U_i covers the per-edit signed collateral damage
y_i = mean_j damage_logit[i,j] with probability >= 0.90 **on average over the
exchangeable population**. They are **not** conditional per-individual-edit
guarantees (a specific edit may under- or over-cover; this is exactly what the
E2 Mondrian audit quantifies and repairs), they are **not** valid under
**sequential no-restore editing** (each edit here is applied to the restored
base model; the exchangeability that split-CP requires is broken by cumulative
editing — deferred to E4, betting-martingale / ACI), and they are **not** valid
under **distribution shift** away from the CounterFact edit/probe distribution
(deferred to E3). Coverage is measured on the fixed masked 500-probe set per
seed; the per-edit target is a signed mean over that set (never AUROC), so the
certificate is probe-set-specific by construction. This paragraph ships verbatim
in the paper draft.
"""


def run_editor(editor, B):
    layers = {}
    for L in io.LAYERS:
        d = io.load_layer(editor, L)
        bs = conformal.bootstrap_cp(d["y"], d["scores"], d["seed_labels"],
                                    io.SCORE_ORDER, io.NORMALIZED, B=B)
        layers[str(L)] = {
            "layer": L, "n_cal": bs["n_cal"], "n_test": bs["n_test"],
            "ordering_fraction": bs["ordering_fraction"],
            "per_score": bs["per_score"],
            "mean_y": float(np.mean(d["y"])),
        }
    return layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1000)
    args = ap.parse_args()
    t0 = time.time()

    rome = run_editor("rome", args.B)
    alpha = run_editor("alpha", args.B)
    c4 = json.load(open(C4_PATH))

    rows = []
    c4_tie = {}
    for L in io.LAYERS:
        Ls = str(L)
        for ed, tab in (("rome", rome), ("alpha", alpha)):
            ps = tab[Ls]["per_score"]
            rows.append({
                "layer": L, "editor": ed,
                "n_cal": tab[Ls]["n_cal"], "n_test": tab[Ls]["n_test"],
                "mean_y_damage": round(tab[Ls]["mean_y"], 4),
                "W_SxC": round(ps["SxC"]["mean_width"], 4),
                "W_keycos": round(ps["keycos"]["mean_width"], 4),
                "W_NG": round(ps["NG"]["mean_width"], 4),
                "W_marginal": round(ps["marginal"]["mean_width"], 4),
                "cov_SxC": round(ps["SxC"]["mean_coverage"], 4),
                "cov_keycos": round(ps["keycos"]["mean_coverage"], 4),
                "cov_NG": round(ps["NG"]["mean_coverage"], 4),
                "cov_marginal": round(ps["marginal"]["mean_coverage"], 4),
                "sxc_pct_tighter_than_marginal": round(ps["SxC"]["pct_tighter_than_marginal"], 4),
                "ordering_fraction": tab[Ls]["ordering_fraction"],
            })
        # C4 tie-in: AlphaEdit vs ROME certified-width ratio (use marginal width as the
        # editor-level certified upper bound scale) vs measured damage ratio.
        w_rome = rome[Ls]["per_score"]["marginal"]["mean_width"]
        w_alpha = alpha[Ls]["per_score"]["marginal"]["mean_width"]
        width_ratio = float(w_rome / w_alpha) if abs(w_alpha) > 1e-12 else float("inf")
        dmg_rome = c4["layers"][Ls]["mean_damage_rome"]
        dmg_alpha = c4["layers"][Ls]["mean_damage_alpha"]
        dmg_ratio = float(dmg_rome / dmg_alpha) if abs(dmg_alpha) > 1e-12 else float("inf")
        # also SxC-based width ratio
        ws_rome = rome[Ls]["per_score"]["SxC"]["mean_width"]
        ws_alpha = alpha[Ls]["per_score"]["SxC"]["mean_width"]
        width_ratio_sxc = float(ws_rome / ws_alpha) if abs(ws_alpha) > 1e-12 else float("inf")
        freeze = bool(width_ratio < 10.0 or width_ratio > 100.0)
        c4_tie[Ls] = {
            "certified_width_ratio_marginal(rome/alpha)": round(width_ratio, 2),
            "certified_width_ratio_SxC(rome/alpha)": round(width_ratio_sxc, 2),
            "c4_damage_ratio(rome/alpha)": round(dmg_ratio, 2),
            "expected_approx_40x": True,
            "freeze_and_reconcile": freeze,
            "note": ("width ratio within [10,100] => consistent with C4 damage collapse"
                     if not freeze else "OUT OF [10,100] => freeze-and-reconcile vs C4"),
        }

    # E1 downgrade check: SxC tightness advantage <10% (pre-E5 accounting) at L8-L12
    downgrade_layers = [L for L in ("8", "10", "12")
                        if rome[L]["per_score"]["SxC"]["pct_tighter_than_marginal"] < 0.10]
    out = {
        "experiment": "CP-Edit E1 headline certified-width table (ROME + AlphaEdit)",
        "rng_seed": conformal.RNG_SEED, "B": args.B, "target_coverage": 0.90,
        "score_order": list(io.SCORE_ORDER),
        "rows": rows,
        "c4_tie_in": c4_tie,
        "e1_downgrade_note": {
            "downgrade_to_note": bool(downgrade_layers),
            "layers_below_10pct_preE5": downgrade_layers,
            "note": ("Full accounting (double-centering + NG-partialling) is applied in E5; "
                     "final downgrade decision folds E5 in."),
        },
        "runtime_s": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    for path, obj in ((OUT_JSON, out),):
        tmp = path + ".tmp"; json.dump(obj, open(tmp, "w"), indent=2); os.replace(tmp, path)

    # markdown table
    hdr = ("| layer | editor | n_cal | n_test | mean_y | W_SxC | W_keycos | W_NG | "
           "W_marg | cov_SxC | cov_marg | SxC %tighter | order_frac |\n"
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    lines = [hdr]
    for r in rows:
        lines.append(
            f"| {r['layer']} | {r['editor']} | {r['n_cal']} | {r['n_test']} | "
            f"{r['mean_y_damage']} | {r['W_SxC']} | {r['W_keycos']} | {r['W_NG']} | "
            f"{r['W_marginal']} | {r['cov_SxC']} | {r['cov_marginal']} | "
            f"{r['sxc_pct_tighter_than_marginal']} | {r['ordering_fraction']} |\n")
    lines.append("\n### AlphaEdit/ROME certified-width ratio vs C4 damage ratio\n\n")
    lines.append("| layer | width_ratio(marg) | width_ratio(SxC) | C4 damage_ratio | freeze? |\n")
    lines.append("|---|---|---|---|---|\n")
    for Ls, t in c4_tie.items():
        lines.append(f"| {Ls} | {t['certified_width_ratio_marginal(rome/alpha)']} | "
                     f"{t['certified_width_ratio_SxC(rome/alpha)']} | "
                     f"{t['c4_damage_ratio(rome/alpha)']} | {t['freeze_and_reconcile']} |\n")
    open(OUT_MD, "w").write("".join(lines))
    open(OUT_SCOPE, "w").write(SCOPE_PARAGRAPH)
    print("[e1] wrote", OUT_JSON, OUT_MD, OUT_SCOPE, f"({out['runtime_s']}s)")
    print(json.dumps({"c4_tie_in": c4_tie, "e1_downgrade": out["e1_downgrade_note"]}, indent=2))


if __name__ == "__main__":
    main()
