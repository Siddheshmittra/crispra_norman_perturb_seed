#!/usr/bin/env python3
"""Cell-type-2 (RPE1) chromatin confirmation of the activatability gate.

Same test as the Hs27 chromatin analysis but for RPE1: does off-target activation
rate track promoter ATAC accessibility (and bulk RNA-seq) in the second cell type?
Closes the gap that the RPE1 cross-cell analysis used expression-only as the proxy.
"""
from __future__ import annotations
import glob, json
from pathlib import Path
import numpy as np, pandas as pd, pyBigWig

ROOT = Path(__file__).resolve().parents[1]
CHROM = ROOT / "data/chromatin"
CA = ROOT / "outputs/candidates/rpe1_crispra_singlecell_off_target_candidates.csv"
NTC = ROOT / "outputs/metrics/rpe1_crispra_ntc_gene_expression.csv"
TSS = ROOT / "vendor/perturb_seed/tss_map.csv"
TPM = CHROM / "RPE1_RNAseq_gene_tpm.tsv"
OUT = ROOT / "outputs/metrics/rpe1_crispra_chromatin_gating.json"
PROM = 1000


def main():
    atac = sorted(g for g in glob.glob(str(CHROM / "RPE1_ATAC" / "**" / "*.bw"), recursive=True))
    print("RPE1 ATAC bigwigs:", [Path(p).name for p in atac])
    tss = pd.read_csv(TSS).dropna(subset=["gene_name", "chrom", "tss"]).drop_duplicates("gene_name").set_index("gene_name")
    genes = tss[["chrom", "tss"]].copy()
    print(f"Scoring promoter ATAC for {len(genes):,} genes (TSS ±{PROM})...")
    acc = np.zeros(len(genes)); cnt = np.zeros(len(genes))
    chrom = genes["chrom"].values; tpos = genes["tss"].astype(int).values
    for p in atac:
        bw = pyBigWig.open(p); chroms = bw.chroms(); vals = np.full(len(genes), np.nan)
        for i in range(len(genes)):
            c = chrom[i]
            if c not in chroms: continue
            s = max(0, tpos[i]-1-PROM); e = min(tpos[i]-1+PROM, chroms[c])
            if s < e:
                try: m = bw.stats(c, s, e, type="mean")[0]
                except Exception: m = None
                vals[i] = m if m is not None else np.nan
        bw.close(); ok = ~np.isnan(vals); acc[ok] += vals[ok]; cnt[ok] += 1
    genes["atac"] = np.where(cnt > 0, acc/np.maximum(cnt, 1), np.nan)
    tpm = pd.read_csv(TPM, sep="\t"); cols = [c for c in tpm.columns if c not in ("gene_id", "gene_name")]
    genes["tpm"] = genes.index.map(tpm.assign(t=tpm[cols].mean(axis=1)).groupby("gene_name")["t"].max())
    genes["ntc"] = genes.index.map(pd.read_csv(NTC).groupby("gene_name")["ntc_mean_log1p_cp10k"].max())

    ca = pd.read_csv(CA, low_memory=False)
    ca["offtarget_log2fc"] = pd.to_numeric(ca["offtarget_log2fc"], errors="coerce")
    ca["seed_match_len"] = pd.to_numeric(ca["seed_match_len"], errors="coerce").fillna(0).astype(int)
    ca = ca[ca["gene"].astype(str).str.upper() != ca["offtarget_gene"].astype(str).str.upper()]
    ev = ca[ca["seed_match_len"] >= 8].merge(genes, left_on="offtarget_gene", right_index=True, how="left")
    ev = ev[ev["offtarget_log2fc"].notna() & ev["atac"].notna()]
    rate = lambda s: float((s["offtarget_log2fc"] > 0.5).mean()) if len(s) else float("nan")
    res = {"n_events": int(len(ev)), "atac_bigwigs": [Path(p).name for p in atac]}
    print(f"\n>=8bp off-target events with RPE1 chromatin: {len(ev):,}")

    q = ev["atac"].quantile([0, .25, .5, .75, 1]).values; q[0] = -np.inf; q[-1] = np.inf
    ev["q"] = pd.cut(ev["atac"], q, labels=["Q1_closed", "Q2", "Q3", "Q4_open"])
    print("\nactivation rate by RPE1 promoter ATAC quartile:")
    res["act_by_atac_quartile"] = {}
    for k in ["Q1_closed", "Q2", "Q3", "Q4_open"]:
        s = ev[ev["q"] == k]; res["act_by_atac_quartile"][k] = {"n": int(len(s)), "act": rate(s)}
        print(f"   {k:<10} n={len(s):>5}  act={100*rate(s):.1f}%")

    def strat(x):
        if pd.isna(x): return None
        return "silent(0)" if x == 0 else "poised(0-0.1]" if x <= 0.1 else "mid(0.1-0.5]" if x <= 0.5 else "expressed(>0.5)"
    ev["ns"] = ev["ntc"].map(strat)
    print("\nmean ATAC + activation per NTC stratum (RPE1):")
    res["by_ntc_stratum"] = {}
    for s_ in ["silent(0)", "poised(0-0.1]", "mid(0.1-0.5]", "expressed(>0.5)"]:
        s = ev[ev["ns"] == s_]
        if not len(s): continue
        d = {"n": int(len(s)), "atac": float(s["atac"].mean()), "median_tpm": float(s["tpm"].median()), "act": rate(s)}
        res["by_ntc_stratum"][s_] = d
        print(f"   {s_:<16} n={d['n']:>5}  ATAC={d['atac']:.2f}  TPM_med={d['median_tpm']:.2f}  act={100*d['act']:.1f}%")

    low = ev[ev["tpm"] <= 5]; med = low["atac"].median()
    print(f"\nheadroom-controlled (TPM<=5, n={len(low)}): activation by ATAC open/closed (median {med:.2f}):")
    res["headroom_controlled"] = {}
    for lab, s in [("ATAC-closed", low[low["atac"] <= med]), ("ATAC-open", low[low["atac"] > med])]:
        res["headroom_controlled"][lab] = {"n": int(len(s)), "act": rate(s)}
        print(f"   {lab:<12} n={len(s):>5}  act={100*rate(s):.1f}%")
    OUT.write_text(json.dumps(res, indent=2)); print("\nWrote", OUT)


if __name__ == "__main__":
    main()
