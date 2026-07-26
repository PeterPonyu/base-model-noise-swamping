# within-probe NG-partialled Spearman (peek method, 2026-07-02) — regenerates the
# Qwen-1.5B L14 inversion NG-robustness check. Run from edit-harness/.
import numpy as np, glob, sys, json
sys.path.insert(0, 'experiments')
from analyze_matrices import _midrank, within_probe_rhos
def partial_within_probe(COS, D, NG):
    rhos=[]
    for j in range(COS.shape[1]):
        c,dm,ng = COS[:,j], D[:,j], NG[:,j]
        m = ~(np.isnan(c)|np.isnan(dm)|np.isnan(ng))
        if m.sum()<5 or np.std(ng[m])==0: continue
        rc,rd,rn = _midrank(c[m]), _midrank(dm[m]), _midrank(ng[m])
        A=np.vstack([rn-rn.mean(), np.ones(m.sum())]).T
        def resid(y):
            b,_,_,_=np.linalg.lstsq(A, y-y.mean(), rcond=None); return (y-y.mean())-A@b
        ec,ed=resid(rc),resid(rd)
        if np.std(ec)==0 or np.std(ed)==0: continue
        rhos.append(np.corrcoef(ec,ed)[0,1])
    return np.array(rhos)
raw,part=[],[]
glob_pat = sys.argv[1] if len(sys.argv)>1 else 'results/matrices/gate_qwen15b_rome_cf_L14_s*.npz'
for f in sorted(glob.glob(glob_pat)):
    d=np.load(f); COS=d['COS'].astype(float); D=d['damage_logit'].astype(float)
    ng=d['norm_growth'].astype(float); NG=np.repeat(ng[:,None],COS.shape[1],axis=1)
    pp=d['pre_p'].astype(float); ok=d['edit_ok'].astype(float)
    rows=ok>0; cols=pp>0.05
    COS,D,NG=COS[rows][:,cols],D[rows][:,cols],NG[rows][:,cols]
    raw.append(float(np.nanmean(within_probe_rhos(COS,D))))
    part.append(float(np.nanmean(partial_within_probe(COS,D,NG))))
print(json.dumps({'glob':glob_pat,'raw_within_probe':[round(x,4) for x in raw],
 'NG_partialled':[round(x,4) for x in part],
 'raw_mean':round(float(np.mean(raw)),4),'NG_partialled_mean':round(float(np.mean(part)),4)}))
