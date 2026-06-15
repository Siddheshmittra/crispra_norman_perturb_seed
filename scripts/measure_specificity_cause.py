#!/usr/bin/env python3
"""What CAUSES the cell-type-specific off-target guides (if not chromatin)?

For every off-target realized in RPE1 but NOT in Hs27's nominated set, measure its
effect directly in Hs27 (pseudobulk log2FC) and classify the cause of the apparent
specificity: coverage (guide too sparse in Hs27), nomination failure (fires in both,
pipeline missed it), near-threshold, genuine chromatin/headroom, or unexplained.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
H5 = ROOT/"data/raw/fibroblast_CRISPRa_final_pop_singlets_normalized_log1p.h5ad"
TSS = pd.read_csv(ROOT/"vendor/perturb_seed/tss_map.csv").drop_duplicates("gene_name").set_index("gene_name")["tss"].to_dict()

def realized(f):
    d=pd.read_csv(ROOT/f"outputs/candidates/{f}",low_memory=False)
    d["offtarget_log2fc"]=pd.to_numeric(d["offtarget_log2fc"],errors="coerce")
    d["seed_match_len"]=pd.to_numeric(d["seed_match_len"],errors="coerce").fillna(0).astype(int)
    d["seed_match_start"]=pd.to_numeric(d["seed_match_start"],errors="coerce")
    d=d[d["gene"].astype(str).str.upper()!=d["offtarget_gene"].astype(str).str.upper()]
    d["td"]=(d["seed_match_start"]-(d["offtarget_gene"].map(TSS).astype(float)-1)).abs()
    return d[(d.seed_match_len>=8)&(d.offtarget_log2fc>0.5)&(d.td<=1000)]

rh=realized("norman_crispra_singlecell_off_target_candidates.csv")
rr=realized("rpe1_crispra_singlecell_off_target_candidates.csv")
hs_pairs=set(zip(rh.guide_id,rh.offtarget_gene))
spec=rr[[(g,o) not in hs_pairs for g,o in zip(rr.guide_id,rr.offtarget_gene)]].copy()
src_guides=set(spec.guide_id)
print(f"RPE1-specific off-target pairs: {len(spec)} ({len(src_guides)} source guides)")

# stream Hs27, pseudobulk the source guides + NTC (paper recipe)
a=ad.read_h5ad(H5,backed="r"); obs=a.obs; gi=obs["guide_identity"].astype(str).values; gt=obs["guide_target"].astype(str).values
varidx={e:i for i,e in enumerate(a.var.index)}
need=np.isin(gi,list(src_guides))|(gt=="non"); idx=np.where(need)[0]
print(f"streaming {len(idx):,} Hs27 cells for {len(src_guides)} guides + NTC...")
gsum={}; gcnt={}; ntc=np.zeros(len(varidx)); ntn=0
CH=8000
for lo in range(0,len(idx),CH):
    ci=np.sort(idx[lo:lo+CH]); X=a.X[ci,:]; X=X.toarray() if sp.issparse(X) else np.asarray(X); X=np.asarray(X,np.float32)
    np.expm1(X,out=X); rs=X.sum(1,keepdims=True); np.maximum(rs,1e-10,out=rs); X*=1e4/rs; np.log1p(X,out=X)
    cg=gi[ci]; ct=gt[ci]
    for g in np.unique(cg):
        if g not in src_guides: continue
        m=cg==g; gsum[g]=gsum.get(g,0)+X[m].sum(0); gcnt[g]=gcnt.get(g,0)+int(m.sum())
    nm=ct=="non"
    if nm.any(): ntc+=X[nm].sum(0); ntn+=int(nm.sum())
a.file.close()
ntm=ntc/max(ntn,1)
def hs_lfc(g,ens):
    if g not in gcnt or gcnt[g]<20 or ens not in varidx: return None
    j=varidx[ens]; return float(np.log2((gsum[g][j]/gcnt[g]+0.01)/(ntm[j]+0.01)))

# annotate
sysd=pd.read_csv(ROOT/"outputs/metrics/systematic_accessibility.csv").set_index("gene")
hsn=pd.read_csv(ROOT/"outputs/metrics/norman_crispra_ntc_gene_expression.csv").groupby("gene_name")["ntc_mean_log1p_cp10k"].max()
rows=[]
for _,r in spec.iterrows():
    g,off,ens=r.guide_id,r.offtarget_gene,r.offtarget_ensembl
    n_hs=gcnt.get(g,0); lfc=hs_lfc(g,ens)
    atac_hs=sysd.atac_hs.get(off,np.nan) if off in sysd.index else np.nan
    ntc_hs=hsn.get(off,np.nan)
    if n_hs<20: cause="coverage (guide <20 cells in Hs27)"
    elif lfc is None: cause="coverage (gene/guide missing)"
    elif lfc>0.5: cause="nomination failure (fires in BOTH)"
    elif lfc>0.1: cause="near-threshold"
    else:
        if (not np.isnan(atac_hs) and atac_hs<4) or (not np.isnan(ntc_hs) and ntc_hs>0.5): cause="genuine: chromatin/ceiling"
        else: cause="silent in Hs27, open+headroom (residual)"
    rows.append(dict(guide=g.split("_")[0],off=off,rpe1_act=float(r.offtarget_log2fc),hs_lfc=lfc,n_hs=n_hs,atac_hs=atac_hs,ntc_hs=ntc_hs,cause=cause))
df=pd.DataFrame(rows); df.to_csv(ROOT/"outputs/metrics/specificity_cause.csv",index=False)
print("\n===== CAUSE DECOMPOSITION of RPE1-specific off-targets =====")
vc=df.cause.value_counts()
for c,n in vc.items(): print(f"  {n:3d} ({100*n/len(df):4.0f}%)  {c}")
print(f"\n  total: {len(df)}")
print("\nexamples of the residual (silent in Hs27 despite open+headroom):")
print(df[df.cause.str.startswith("silent")][["guide","off","rpe1_act","hs_lfc","atac_hs","ntc_hs"]].head(8).to_string(index=False))
