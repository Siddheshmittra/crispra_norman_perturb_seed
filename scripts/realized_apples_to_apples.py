#!/usr/bin/env python3
"""Apples-to-apples realized off-target rates for the single-cell CRISPRa run.

Mirrors the Hartman et al. CRISPRi definition exactly (the same one that
reproduced the paper's 4.67% / 0.83% anchors in the CRISPRi Phase 0 tree):

    realized(tier) = guides with >=1 off-target event such that
        seed_match_len >= tier
        AND off-target effect passes the mode threshold (CRISPRa: log2FC > +0.5)
        AND |seed_match_start - TSS0| <= 1000 bp   (+/-1 kb headline window)
        AND off-target gene != designed target     (exclude on-target / self)

Reported over two denominators (the paper reports the library fraction; the
CRISPRi reproduce gate uses the analyzed fraction -- both are emitted):
    library  = all targeting guides in the sgRNA library
    analyzed = unique guides present in the neighbor-event candidate table

Effects here are pseudobulk log2FC vs non-targeting (CP10k -> log1p,
pseudocount 0.01), i.e. the SAME units as the CRISPRi pipeline -- so the +0.5
threshold means the same thing on both sides.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TIERS = [5, 8, 10, 12]
REALIZED_EFFECT = 0.5      # |log2FC| threshold (CRISPRa: > +0.5)
TSS_WINDOW_BP = 1000       # +/-1 kb headline window (paper primary)

# CRISPRi Replogle K562 anchors (from replogle_k562_reproduce_anchors.csv)
CRISPRI = {
    "realized_ge8_library": 0.0361, "realized_ge8_analyzed": 0.0500,
    "realized_ge12_library": 0.0115, "realized_ge12_analyzed": 0.0159,
    "potential_ge12_spacer": 0.5669, "potential_ge12_construct": 0.7490,
    "library_n": 10709, "analyzed_n": 7738,
}


def add_tss_distance(df: pd.DataFrame, tss_map_csv: Path) -> pd.DataFrame:
    tss = pd.read_csv(tss_map_csv).drop_duplicates("gene_name").set_index("gene_name")["tss"].to_dict()
    out = df.copy()
    start = pd.to_numeric(out["seed_match_start"], errors="coerce")
    tss0 = out["offtarget_gene"].map(lambda g: tss.get(g, np.nan)).astype(float) - 1.0
    out["tss_distance_bp"] = (start - tss0).abs()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", type=Path,
                    default=Path("../outputs/candidates/norman_crispra_singlecell_off_target_candidates.csv"))
    ap.add_argument("--guide-lib", type=Path, default=Path("../outputs/metrics/norman_hs27_guide_library.csv"))
    ap.add_argument("--tss-map", type=Path, default=Path("../vendor/perturb_seed/tss_map.csv"))
    ap.add_argument("--potential-csv", type=Path,
                    default=Path("../outputs/metrics/norman_crispra_genome_wide_potential_APPLES2APPLES_1kb_excl_ontarget.csv"))
    ap.add_argument("--out", type=Path, default=Path("../outputs/metrics/norman_crispra_apples_to_apples_comparison.csv"))
    args = ap.parse_args()

    cand = pd.read_csv(args.candidates)
    cand = add_tss_distance(cand, args.tss_map)
    cand["seed_match_len"] = pd.to_numeric(cand["seed_match_len"], errors="coerce").fillna(0).astype(int)
    cand["offtarget_log2fc"] = pd.to_numeric(cand["offtarget_log2fc"], errors="coerce")
    cand["is_on_target"] = cand["gene"].astype(str).str.upper() == cand["offtarget_gene"].astype(str).str.upper()

    lib = pd.read_csv(args.guide_lib)
    library_n = int((lib["designed_target_gene_name"].astype(str) != "off-target").sum())
    analyzed_n = int(cand["guide_id"].nunique())

    pot = pd.read_csv(args.potential_csv)
    crispra_potential = float(pot["potential_fraction"].iloc[0])

    rows = []
    print(f"\nCandidate (neighbor-event) table: {len(cand):,} rows, "
          f"{analyzed_n:,} unique guides analyzed | library targeting guides: {library_n:,}\n")
    print(f"{'tier':>6} {'realized pairs':>15} {'realized guides':>16} "
          f"{'% library':>11} {'% analyzed':>11}")
    for tier in TIERS:
        m = ((cand["seed_match_len"] >= tier)
             & (cand["offtarget_log2fc"] > REALIZED_EFFECT)   # CRISPRa activation
             & (cand["tss_distance_bp"] <= TSS_WINDOW_BP)
             & (~cand["is_on_target"]))
        sub = cand.loc[m]
        ng = int(sub["guide_id"].nunique())
        npairs = int(len(sub.drop_duplicates(["guide_id", "offtarget_gene"])))
        f_lib = ng / library_n
        f_ana = ng / analyzed_n if analyzed_n else float("nan")
        rows.append({"seed_tier": f">={tier}", "realized_pairs": npairs, "realized_guides": ng,
                     "frac_library": f_lib, "frac_analyzed": f_ana,
                     "library_n": library_n, "analyzed_n": analyzed_n})
        print(f"{'>='+str(tier):>6} {npairs:>15} {ng:>16} {100*f_lib:>10.2f}% {100*f_ana:>10.2f}%")

    out_df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    r8 = out_df[out_df.seed_tier == ">=8"].iloc[0]
    r12 = out_df[out_df.seed_tier == ">=12"].iloc[0]
    summary = {
        "crispra_singlecell": {
            "realized_ge8_library": float(r8.frac_library), "realized_ge8_analyzed": float(r8.frac_analyzed),
            "realized_ge12_library": float(r12.frac_library), "realized_ge12_analyzed": float(r12.frac_analyzed),
            "potential_ge12_offtarget_1kb": crispra_potential,
            "library_n": library_n, "analyzed_n": analyzed_n,
        },
        "crispri_replogle": CRISPRI,
    }
    (args.out.with_suffix(".json")).write_text(json.dumps(summary, indent=2))

    print("\n================ APPLES-TO-APPLES (same method, same units) ================")
    print(f"{'metric':<42}{'CRISPRi (Replogle)':>22}{'CRISPRa (Norman)':>20}")
    print(f"{'realized >=8bp + effect>0.5 (% library)':<42}{100*CRISPRI['realized_ge8_library']:>21.2f}%{100*r8.frac_library:>19.2f}%")
    print(f"{'realized >=8bp + effect>0.5 (% analyzed)':<42}{100*CRISPRI['realized_ge8_analyzed']:>21.2f}%{100*r8.frac_analyzed:>19.2f}%")
    print(f"{'realized >=12bp + effect>0.5 (% library)':<42}{100*CRISPRI['realized_ge12_library']:>21.2f}%{100*r12.frac_library:>19.2f}%")
    print(f"{'realized >=12bp + effect>0.5 (% analyzed)':<42}{100*CRISPRI['realized_ge12_analyzed']:>21.2f}%{100*r12.frac_analyzed:>19.2f}%")
    print(f"{'potential >=12bp off-target, +/-1kb (spacer)':<42}{100*CRISPRI['potential_ge12_spacer']:>21.2f}%{100*crispra_potential:>19.2f}%")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
