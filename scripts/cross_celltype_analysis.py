#!/usr/bin/env python3
"""Cross-cell-type test of CRISPRa off-target context-dependence (Hs27 vs RPE1).

Same TF-CRISPRa guide library, two cell types with different poised/chromatin
landscapes. Predicts (from the activatability gate): the SAME guides off-target-
activate DIFFERENT genes in each cell type -- whichever genes are poised there.

Tests: (1) the activatability gate replicates in RPE1; (2) realized off-target
rates replicate; (3) realized off-targets track each cell type's poised set.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TSS = ROOT / "vendor/perturb_seed/tss_map.csv"
OUT = ROOT / "outputs/metrics/norman_crispra_cross_celltype.json"
CELLS = {
    "Hs27": (ROOT / "outputs/candidates/norman_crispra_singlecell_off_target_candidates.csv",
             ROOT / "outputs/metrics/norman_crispra_ntc_gene_expression.csv"),
    "RPE1": (ROOT / "outputs/candidates/rpe1_crispra_singlecell_off_target_candidates.csv",
             ROOT / "outputs/metrics/rpe1_crispra_ntc_gene_expression.csv"),
}
ANALYZED = {"Hs27": 3326, "RPE1": None}  # RPE1 filled from its table


def load(cell):
    cand_path, ntc_path = CELLS[cell]
    df = pd.read_csv(cand_path, low_memory=False)
    df["offtarget_log2fc"] = pd.to_numeric(df["offtarget_log2fc"], errors="coerce")
    df["seed_match_len"] = pd.to_numeric(df["seed_match_len"], errors="coerce").fillna(0).astype(int)
    df["seed_match_start"] = pd.to_numeric(df["seed_match_start"], errors="coerce")
    df = df[df["gene"].astype(str).str.upper() != df["offtarget_gene"].astype(str).str.upper()]
    tssmap = pd.read_csv(TSS).drop_duplicates("gene_name").set_index("gene_name")["tss"].to_dict()
    df["tss_dist"] = (df["seed_match_start"] - (df["offtarget_gene"].map(tssmap).astype(float) - 1)).abs()
    ntc = pd.read_csv(ntc_path).groupby("gene_name")["ntc_mean_log1p_cp10k"].max()
    df["ntc"] = df["offtarget_gene"].map(ntc)
    return df, ntc


def stratum(x):
    if pd.isna(x): return None
    return "silent(0)" if x == 0 else "poised(0-0.1]" if x <= 0.1 else "mid(0.1-0.5]" if x <= 0.5 else "expressed(>0.5)"


def realized_pairs(df, tier=8):
    m = ((df["seed_match_len"] >= tier) & (df["offtarget_log2fc"] > 0.5) & (df["tss_dist"] <= 1000))
    return df.loc[m]


def main():
    hs, hs_ntc = load("Hs27")
    rp, rp_ntc = load("RPE1")
    ANALYZED["RPE1"] = int(rp["guide_id"].nunique())
    res = {}

    # (1) activatability gate replicates in RPE1?
    print("=" * 70, "\n(1) Activation rate by NTC stratum -- does the gate replicate in RPE1?\n")
    res["gate_replication"] = {}
    print(f"{'stratum':<16}{'Hs27 act':>12}{'RPE1 act':>12}")
    for cell, df in [("Hs27", hs), ("RPE1", rp)]:
        sub = df[(df["seed_match_len"] >= 8) & df["ntc"].notna()].copy()
        sub["str"] = sub["ntc"].map(stratum)
        res["gate_replication"][cell] = {s: float((sub[sub["str"] == s]["offtarget_log2fc"] > 0.5).mean())
                                         for s in ["silent(0)", "poised(0-0.1]", "mid(0.1-0.5]", "expressed(>0.5)"]}
    for s in ["silent(0)", "poised(0-0.1]", "mid(0.1-0.5]", "expressed(>0.5)"]:
        print(f"{s:<16}{100*res['gate_replication']['Hs27'][s]:>11.1f}%{100*res['gate_replication']['RPE1'][s]:>11.1f}%")

    # (2) realized rates side by side
    print("\n" + "=" * 70, "\n(2) Realized off-target rates (% analyzed guides)\n")
    res["realized"] = {}
    print(f"{'tier':<8}{'Hs27':>18}{'RPE1':>18}")
    for tier in [8, 12]:
        hg = realized_pairs(hs, tier)["guide_id"].nunique()
        rg = realized_pairs(rp, tier)["guide_id"].nunique()
        res["realized"][f">={tier}"] = {"Hs27_guides": int(hg), "Hs27_pct": hg / ANALYZED["Hs27"],
                                        "RPE1_guides": int(rg), "RPE1_pct": rg / ANALYZED["RPE1"]}
        print(f">={tier:<6}{hg:>6} ({100*hg/ANALYZED['Hs27']:>5.2f}%)   {rg:>6} ({100*rg/ANALYZED['RPE1']:>5.2f}%)")
    print(f"   analyzed guides: Hs27={ANALYZED['Hs27']}  RPE1={ANALYZED['RPE1']}")

    # (3) context-dependence: do realized off-targets track each cell's poised set?
    print("\n" + "=" * 70, "\n(3) Context-dependence of the poised set and realized off-targets\n")
    common = set(hs_ntc.index) & set(rp_ntc.index)
    j = pd.DataFrame({"hs": hs_ntc.reindex(common), "rp": rp_ntc.reindex(common)}).dropna()
    res["ntc_correlation"] = float(np.corrcoef(j["hs"], j["rp"])[0, 1])
    hs_poised = set(j[(j.hs > 0) & (j.hs <= 0.1)].index)
    rp_poised = set(j[(j.rp > 0) & (j.rp <= 0.1)].index)
    jac = len(hs_poised & rp_poised) / len(hs_poised | rp_poised)
    res["poised_set"] = {"Hs27_n": len(hs_poised), "RPE1_n": len(rp_poised),
                         "shared": len(hs_poised & rp_poised), "jaccard": jac}
    print(f"per-gene NTC correlation Hs27 vs RPE1: r={res['ntc_correlation']:.3f}")
    print(f"poised genes: Hs27={len(hs_poised)} RPE1={len(rp_poised)} shared={len(hs_poised & rp_poised)} (Jaccard={jac:.2f})")

    # realized off-target GENES per cell, and the other cell's state for them
    hs_real = set(realized_pairs(hs, 8)["offtarget_gene"])
    rp_real = set(realized_pairs(rp, 8)["offtarget_gene"])
    res["realized_offtarget_genes"] = {"Hs27": len(hs_real), "RPE1": len(rp_real),
                                       "shared": len(hs_real & rp_real),
                                       "jaccard": len(hs_real & rp_real) / max(1, len(hs_real | rp_real))}
    print(f"\nrealized off-target genes (>=8bp): Hs27={len(hs_real)} RPE1={len(rp_real)} "
          f"shared={len(hs_real & rp_real)} (Jaccard={res['realized_offtarget_genes']['jaccard']:.2f})")

    # for genes realized in Hs27 only, what is their RPE1 expression state? (expect not-poised in RPE1)
    def state_breakdown(genes, ntc):
        strata = pd.Series({g: stratum(ntc.get(g, np.nan)) for g in genes})
        return strata.value_counts().to_dict()
    hs_only = hs_real - rp_real
    rp_only = rp_real - hs_real
    res["hs_only_in_rpe1_state"] = state_breakdown(hs_only, rp_ntc)
    res["rp_only_in_hs27_state"] = state_breakdown(rp_only, hs_ntc)
    print(f"\nHs27-only realized off-target genes (n={len(hs_only)}): their state IN RPE1: {res['hs_only_in_rpe1_state']}")
    print(f"RPE1-only realized off-target genes (n={len(rp_only)}): their state IN Hs27: {res['rp_only_in_hs27_state']}")
    print("  (prediction: cell-type-specific realized off-targets are NOT poised in the other cell type)")

    OUT.write_text(json.dumps(res, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
