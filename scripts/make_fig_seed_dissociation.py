#!/usr/bin/env python3
"""Figure: guide-level seed dissociation — direct vs trans CRISPRa off-targets."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "outputs/metrics"
OUT = ROOT / "outputs/figures/panels/seed_dissociation.png"
COL = {"DIRECT": "#2c7fb8", "TRANS": "#d95f0e", "AMBIGUOUS": "#bdbdbd"}


def main():
    pp = pd.read_csv(M / "seed_dissociation_hs27_per_pair_tier8.csv")
    pg = pd.read_csv(M / "seed_dissociation_hs27_per_guide.csv")
    summ = json.load(open(M / "seed_dissociation_hs27_summary.json"))
    t8 = summ["tiers"]["8"]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))

    # --- left: composition of the 143 testable realized off-targets ---
    cats = ["DIRECT", "TRANS"]
    fr = [t8["fractions"][c]["frac"] * 100 for c in cats]
    ci = [t8["fractions"][c]["ci95"] for c in cats]
    err = [[100 * (fr[i] / 100 - ci[i][0]) for i in range(2)],
           [100 * (ci[i][1] - fr[i] / 100) for i in range(2)]]
    disp = {"DIRECT": "DIRECT", "TRANS": "INDIRECT"}
    bars = ax[0].bar([disp[c] for c in cats], fr, color=[COL[c] for c in cats],
                     yerr=err, capsize=6, edgecolor="k", linewidth=0.6)
    for b, c in zip(bars, cats):
        n = t8["fractions"][c]["n"]
        ax[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 3, f"n={n}",
                   ha="center", fontsize=10)
    ax[0].set_ylabel("% of testable realized off-targets")
    ax[0].set_ylim(0, 100)
    ax[0].set_title(f"Nominal split of {t8['n_testable']} testable ≥8bp off-targets\n"
                    f"({fr[1]:.0f}% INDIRECT) — exploratory; fails split-half (see noise-floor fig)",
                    fontsize=9.5)

    # --- right: per-pair seed-match vs no-match B activation ---
    rows = []
    for (A, B), g in pg.groupby(["A", "B"]):
        cls = pp[(pp.A == A) & (pp.B == B)]
        if not len(cls):
            continue
        cls = cls.iloc[0]["cls"]
        sm = g[g.seed >= 8]
        nm = g[(g.seed < 8) & (g.a_eff)]
        if not len(sm) or not len(nm):
            continue
        rows.append((sm.B_lfc.max(), nm.B_lfc.max(), cls))
    r = pd.DataFrame(rows, columns=["sm", "nm", "cls"])
    disp = {"DIRECT": "DIRECT", "TRANS": "INDIRECT"}
    for c in ("TRANS", "DIRECT"):
        s = r[r.cls == c]
        ax[1].scatter(s.sm, s.nm, c=COL[c], label=f"{disp[c]} (n={len(s)})",
                      alpha=0.75, edgecolor="k", linewidth=0.3, s=36)
    lim = [-3, max(4, r[["sm", "nm"]].max().max() + 0.5)]
    ax[1].axhline(0.5, ls=":", c="grey", lw=1)
    ax[1].axvline(0.5, ls=":", c="grey", lw=1)
    ax[1].plot(lim, lim, ls="--", c="k", lw=0.6)
    ax[1].set_xlim(lim); ax[1].set_ylim(lim)
    ax[1].set_xlabel("off-target B activation by SEED-MATCH guide (log2FC)")
    ax[1].set_ylabel("off-target B activation by\nA-effective NO-MATCH sibling (log2FC)")
    ax[1].set_title("DIRECT: B fires only with the seed (low y)\n"
                    "INDIRECT: B fires without the seed too (high y)", fontsize=10.5)
    ax[1].legend(fontsize=9, loc="upper left")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
