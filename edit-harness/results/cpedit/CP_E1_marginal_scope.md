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
