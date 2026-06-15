#!/usr/bin/env python3
"""Direct chromatin test of the CRISPRa activatability finding (b).

Finding (b) used single-cell NTC expression as a PROXY for "poised/activatable".
Here we test it directly with Hs27 ATAC-seq + CUT&RUN (H3K4me3 active promoter,
H3K27ac active enh/prom, H3K27me3 Polycomb) and bulk RNA-seq TPM at off-target
promoters (TSS +/-1 kb, hg38, 10M-normalized bigwigs):

  - Does off-target activation rate track promoter ACCESSIBILITY / active marks,
    independent of the expression proxy?
  - Are "poised" (19.7%-activated) genes ATAC-open + H3K4me3+, and "silent"
    (0%-activated) genes ATAC-closed?
  - Controlling for headroom (low-expression genes only), does open chromatin
    still predict activation?
"""
from __future__ import annotations
import glob, json
from pathlib import Path
import numpy as np
import pandas as pd
import pyBigWig

ROOT = Path(__file__).resolve().parents[1]
CHROM = ROOT / "data/chromatin"
CA = ROOT / "outputs/candidates/norman_crispra_singlecell_off_target_candidates.csv"
NTC = ROOT / "outputs/metrics/norman_crispra_ntc_gene_expression.csv"
TSS = ROOT / "vendor/perturb_seed/tss_map.csv"
TPM = CHROM / "Hs27_RNAseq_gene_tpm.tsv"
OUT = ROOT / "outputs/metrics/norman_crispra_chromatin_gating.json"
PROM = 1000

MARKS = {"atac": "ATAC", "k4me3": "H3K4me3", "k27ac": "H3K27ac", "k27me3": "H3K27me3"}


def find_bw(substr):
    hits = glob.glob(str(CHROM / "**" / "*.bw"), recursive=True)
    return sorted([h for h in hits if substr.lower() in Path(h).name.lower()])


def promoter_signal(bw_paths, genes_df):
    if not bw_paths:
        return pd.Series(np.nan, index=genes_df.index)
    acc = np.zeros(len(genes_df)); cnt = np.zeros(len(genes_df))
    chrom = genes_df["chrom"].values; tss = genes_df["tss"].astype(int).values
    for p in bw_paths:
        bw = pyBigWig.open(p); chroms = bw.chroms()
        vals = np.full(len(genes_df), np.nan)
        for i in range(len(genes_df)):
            c = chrom[i]
            if c not in chroms:
                continue
            s = max(0, tss[i] - 1 - PROM); e = min(tss[i] - 1 + PROM, chroms[c])
            if s >= e:
                continue
            try:
                m = bw.stats(c, s, e, type="mean")[0]
            except Exception:
                m = None
            vals[i] = m if m is not None else np.nan
        bw.close()
        ok = ~np.isnan(vals); acc[ok] += vals[ok]; cnt[ok] += 1
    return pd.Series(np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan), index=genes_df.index)


def rate(s):
    return float((s["offtarget_log2fc"] > 0.5).mean()) if len(s) else float("nan")


def main():
    bws = {k: find_bw(v) for k, v in MARKS.items()}
    for k, v in bws.items():
        print(f"{MARKS[k]:>9}: {len(v)} bigwig(s)")

    tss = pd.read_csv(TSS).dropna(subset=["gene_name", "chrom", "tss"]).drop_duplicates("gene_name").set_index("gene_name")
    genes = tss[["chrom", "tss"]].copy()
    print(f"Scoring promoter signal for {len(genes):,} genes (TSS +/-{PROM} bp)...")
    for k in MARKS:
        genes[k] = promoter_signal(bws[k], genes).values

    tpm = pd.read_csv(TPM, sep="\t")
    cols = [c for c in tpm.columns if c not in ("gene_id", "gene_name")]
    genes["tpm"] = genes.index.map(tpm.assign(t=tpm[cols].mean(axis=1)).groupby("gene_name")["t"].max())
    genes["ntc"] = genes.index.map(pd.read_csv(NTC).groupby("gene_name")["ntc_mean_log1p_cp10k"].max())

    ca = pd.read_csv(CA, low_memory=False)
    ca["offtarget_log2fc"] = pd.to_numeric(ca["offtarget_log2fc"], errors="coerce")
    ca["seed_match_len"] = pd.to_numeric(ca["seed_match_len"], errors="coerce").fillna(0).astype(int)
    ca = ca[ca["gene"].astype(str).str.upper() != ca["offtarget_gene"].astype(str).str.upper()]
    ev = ca[ca["seed_match_len"] >= 8].merge(genes, left_on="offtarget_gene", right_index=True, how="left")
    ev = ev[ev["offtarget_log2fc"].notna() & ev["atac"].notna()]
    print(f"\n>=8bp off-target seed events with chromatin annotation: {len(ev):,}")
    res = {"n_events": int(len(ev)), "bigwigs": {k: [Path(p).name for p in bws[k]] for k in MARKS}}

    # (i) activation rate by ATAC quartile  -- the direct accessibility test
    q = ev["atac"].quantile([0, .25, .5, .75, 1]).values; q[0] = -np.inf; q[-1] = np.inf
    ev["atac_q"] = pd.cut(ev["atac"], bins=q, labels=["Q1_closed", "Q2", "Q3", "Q4_open"])
    print("\n(i) activation rate by promoter ATAC accessibility quartile:")
    res["act_by_atac_quartile"] = {}
    for lab in ["Q1_closed", "Q2", "Q3", "Q4_open"]:
        s = ev[ev["atac_q"] == lab]; res["act_by_atac_quartile"][lab] = {"n": int(len(s)), "act": rate(s)}
        print(f"   {lab:<10} n={len(s):>5}  act={100*rate(s):.1f}%   meanATAC={s['atac'].mean():.2f}")

    # (ii) replicate (b) with independent bulk TPM
    print("\n(ii) activation rate by bulk RNA-seq TPM (independent of single-cell NTC):")
    ev["tpm_b"] = pd.cut(ev["tpm"], [-np.inf, 0.5, 5, 50, np.inf], labels=["silent(<0.5)", "low(0.5-5]", "mid(5-50]", "high(>50)"])
    res["act_by_tpm"] = {}
    for lab in ["silent(<0.5)", "low(0.5-5]", "mid(5-50]", "high(>50)"]:
        s = ev[ev["tpm_b"] == lab]; res["act_by_tpm"][lab] = {"n": int(len(s)), "act": rate(s)}
        print(f"   {lab:<13} n={len(s):>5}  act={100*rate(s):.1f}%")

    # (iii) chromatin marks per single-cell NTC stratum (poised=open+active? silent=closed?)
    def strat(x):
        if pd.isna(x): return None
        return "silent(0)" if x == 0 else "poised(0-0.1]" if x <= 0.1 else "mid(0.1-0.5]" if x <= 0.5 else "expressed(>0.5)"
    ev["nstr"] = ev["ntc"].map(strat)
    print("\n(iii) mean promoter marks + activation rate per single-cell NTC stratum:")
    print(f"   {'stratum':<16}{'n':>6}{'ATAC':>8}{'K4me3':>8}{'K27ac':>8}{'K27me3':>8}{'act':>8}")
    res["by_ntc_stratum"] = {}
    for s_ in ["silent(0)", "poised(0-0.1]", "mid(0.1-0.5]", "expressed(>0.5)"]:
        s = ev[ev["nstr"] == s_]
        if not len(s): continue
        d = {"n": int(len(s)), "atac": float(s["atac"].mean()), "k4me3": float(s["k4me3"].mean()),
             "k27ac": float(s["k27ac"].mean()), "k27me3": float(s["k27me3"].mean()), "act": rate(s)}
        res["by_ntc_stratum"][s_] = d
        print(f"   {s_:<16}{d['n']:>6}{d['atac']:>8.2f}{d['k4me3']:>8.2f}{d['k27ac']:>8.2f}{d['k27me3']:>8.2f}{100*d['act']:>7.1f}%")

    # (iv) control for headroom: among LOW-expression genes (tpm<=5), does open chromatin still gate activation?
    low = ev[ev["tpm"] <= 5].copy()
    med = low["atac"].median()
    print(f"\n(iv) among low-expression off-targets (TPM<=5, n={len(low)}), activation by ATAC open/closed (split at median {med:.2f}):")
    res["headroom_controlled"] = {}
    for lab, s in [("ATAC-closed", low[low["atac"] <= med]), ("ATAC-open", low[low["atac"] > med])]:
        res["headroom_controlled"][lab] = {"n": int(len(s)), "act": rate(s)}
        print(f"   {lab:<12} n={len(s):>5}  act={100*rate(s):.1f}%")

    OUT.write_text(json.dumps(res, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
