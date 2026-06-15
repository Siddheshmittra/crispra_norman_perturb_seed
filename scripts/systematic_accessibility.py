#!/usr/bin/env python3
"""Is the accessibility effect systematic (not cherry-picked)?

For every realized off-target gene, score its promoter ATAC in BOTH cell types and
ask whether the cell where it fires is systematically more accessible. Saves a table
for the systematic figure.
"""
from __future__ import annotations
import glob
from pathlib import Path
import numpy as np, pandas as pd, pyBigWig

ROOT = Path(__file__).resolve().parents[1]
CHROM = ROOT / "data/chromatin"
TSS = pd.read_csv(ROOT/"vendor/perturb_seed/tss_map.csv").drop_duplicates("gene_name").set_index("gene_name")
HS = sorted(glob.glob(str(CHROM/"Density_bigwigs_hg38_10mNorm"/"ATAC_HS27*.bw")))
RP = sorted(glob.glob(str(CHROM/"RPE1_ATAC"/"**"/"ATAC_RPE1*.bw"), recursive=True))


def realized(f):
    d = pd.read_csv(ROOT/f"outputs/candidates/{f}", low_memory=False)
    d["offtarget_log2fc"] = pd.to_numeric(d["offtarget_log2fc"], errors="coerce")
    d["seed_match_len"] = pd.to_numeric(d["seed_match_len"], errors="coerce").fillna(0).astype(int)
    d["seed_match_start"] = pd.to_numeric(d["seed_match_start"], errors="coerce")
    d = d[d["gene"].astype(str).str.upper() != d["offtarget_gene"].astype(str).str.upper()]
    d["td"] = (d["seed_match_start"] - (d["offtarget_gene"].map(TSS["tss"]).astype(float)-1)).abs()
    return d[(d.seed_match_len>=8)&(d.offtarget_log2fc>0.5)&(d.td<=1000)]


def atac_all(genes, bws):
    out = {g: [] for g in genes}
    for p in bws:
        bw = pyBigWig.open(p); ch = bw.chroms()
        for g in genes:
            if g not in TSS.index: continue
            r = TSS.loc[g]; c = str(r["chrom"]); t = int(r["tss"])
            if c in ch:
                m = bw.stats(c, max(0,t-1-1000), t-1+1000, type="mean")[0]
                if m is not None: out[g].append(m)
        bw.close()
    return {g: (np.mean(v) if v else np.nan) for g, v in out.items()}


def main():
    rh, rr = realized("norman_crispra_singlecell_off_target_candidates.csv"), realized("rpe1_crispra_singlecell_off_target_candidates.csv")
    # best (max-activation) realized event per (gene, cell)
    hbg = rh.groupby("offtarget_gene")["offtarget_log2fc"].max()
    rbg = rr.groupby("offtarget_gene")["offtarget_log2fc"].max()
    genes = sorted(set(hbg.index) | set(rbg.index))
    ah, ar = atac_all(genes, HS), atac_all(genes, RP)

    rows = []
    for g in genes:
        ih, ir = ah.get(g, np.nan), ar.get(g, np.nan)
        if np.isnan(ih) or np.isnan(ir): continue
        in_h, in_r = g in hbg.index, g in rbg.index
        if in_h and not in_r: fire, atf, ato, act = "Hs27", ih, ir, hbg[g]
        elif in_r and not in_h: fire, atf, ato, act = "RPE1", ir, ih, rbg[g]
        else: fire, atf, ato, act = "both", max(ih,ir), min(ih,ir), max(hbg.get(g,0), rbg.get(g,0))
        rows.append(dict(gene=g, fires=fire, atac_hs=ih, atac_rp=ir, atac_fire=atf, atac_other=ato,
                         dATAC=atf-ato, act=act, specific=(fire!="both")))
    df = pd.DataFrame(rows)
    df.to_csv(ROOT/"outputs/metrics/systematic_accessibility.csv", index=False)

    spec = df[df.specific]
    print(f"realized off-target genes with ATAC in both cells: {len(df)} ({len(spec)} cell-type-specific, {len(df)-len(spec)} shared)")
    print(f"\nCell-type-specific off-targets: is the FIRING cell more accessible?")
    print(f"  median dATAC (firing - other): {spec.dATAC.median():.2f}")
    print(f"  mean dATAC: {spec.dATAC.mean():.2f}")
    print(f"  fraction with dATAC > 0 (firing more open): {(spec.dATAC>0).mean()*100:.0f}%")
    print(f"  fraction where OTHER cell is closed (ATAC<4) i.e. clean accessibility flip: {(spec.atac_other<4).mean()*100:.0f}%")
    print(f"  fraction where BOTH cells open (ATAC>=6 both): {((spec.atac_fire>=6)&(spec.atac_other>=6)).mean()*100:.0f}%")
    from scipy import stats
    t = stats.wilcoxon(spec.dATAC)
    print(f"  Wilcoxon dATAC vs 0: p={t.pvalue:.2e}")
    print(f"\nFor reference, shared (both-cell) off-targets median |dATAC|: {df[~df.specific].dATAC.median():.2f}")


if __name__ == "__main__":
    main()
