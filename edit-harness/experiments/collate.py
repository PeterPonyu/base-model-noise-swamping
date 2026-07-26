"""tabulate.py — collate all kill-gate sweep results into one breadth table.

Reads results/*.json (sweep + full runs), prints a comparison sorted by config,
and flags where key-geometry beats the norm-growth baseline. Usage:
  python experiments/tabulate.py
"""
import glob
import json
import os

H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = []
for f in sorted(glob.glob(os.path.join(H, "results", "*.json"))):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    kp = d.get("KNOWN_PROBES") or d.get("ALL_PAIRS") or {}
    if "spearman_cos_damage" not in kp:
        continue
    model = os.path.basename(str(d.get("model", "?")))
    rows.append({
        "config": f"{model[:14]:14s} {d.get('editor','?'):4s} {d.get('dataset','?')[:4]:4s} L{d.get('layer','?')}",
        "edit_ok": d.get("edit_success_rate", float("nan")),
        "rho_cos": kp.get("spearman_cos_damage"),
        "rho_ng": kp.get("spearman_normgrowth_damage"),
        "auroc_cos": kp.get("auroc_cos_broken"),
        "auroc_ng": kp.get("auroc_normgrowth_broken"),
        "verdict": d.get("VERDICT", "?")[:4],
    })

if not rows:
    print("(no results yet)")
    raise SystemExit

hdr = f"{'config':36s} {'edit':>5s} {'ρ_cos':>7s} {'ρ_ng':>7s} {'AUC_cos':>8s} {'AUC_ng':>7s} {'beats?':>6s} {'V':>4s}"
print(hdr)
print("-" * len(hdr))
for r in rows:
    beats = "yes" if (r["auroc_cos"] or 0) > (r["auroc_ng"] or 0) and (r["rho_cos"] or 0) > (r["rho_ng"] or 0) else "no"
    print(f"{r['config']:36s} {r['edit_ok']:5.2f} {r['rho_cos']:7.3f} {r['rho_ng']:7.3f} "
          f"{r['auroc_cos']:8.3f} {r['auroc_ng']:7.3f} {beats:>6s} {r['verdict']:>4s}")

# breadth summary
n = len(rows)
n_beats = sum(1 for r in rows if (r["auroc_cos"] or 0) > (r["auroc_ng"] or 0))
n_pass = sum(1 for r in rows if (r["rho_cos"] or 0) >= 0.2 and (r["auroc_cos"] or 0) >= 0.6)
print("-" * len(hdr))
print(f"BREADTH: {n} configs | key-overlap beats norm-growth in {n_beats}/{n} | "
      f"passes gate (ρ≥0.2 & AUROC≥0.6) in {n_pass}/{n}")
