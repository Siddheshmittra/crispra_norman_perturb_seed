#!/usr/bin/env python3
"""Systematic test: does chromatin accessibility explain the cell-type-specificity?
Honest answer = mostly no. Most cell-type-specific off-targets are accessible in BOTH cells."""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs/figures/panels"; FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"savefig.dpi":300,"font.size":10,"font.family":"DejaVu Sans",
                     "axes.spines.top":False,"axes.spines.right":False})
C_HS, C_RP = "#e07a3f", "#4c8c6b"

df = pd.read_csv(ROOT/"outputs/metrics/systematic_accessibility.csv")
s = df[df.specific].copy()
hsn = pd.read_csv(ROOT/"outputs/metrics/norman_crispra_ntc_gene_expression.csv").groupby("gene_name")["ntc_mean_log1p_cp10k"].max()
rpn = pd.read_csv(ROOT/"outputs/metrics/rpe1_crispra_ntc_gene_expression.csv").groupby("gene_name")["ntc_mean_log1p_cp10k"].max()
s["ntc_other"] = np.where(s.fires=="Hs27", s.gene.map(rpn), s.gene.map(hsn))

fig, ax = plt.subplots(figsize=(7.4, 6.4))
lim = 45
sc = s.copy(); sc["xo"] = sc.atac_other.clip(0, lim); sc["yf"] = sc.atac_fire.clip(0, lim)
ax.axhspan(0, lim, xmin=0, xmax=4/lim, color="#c0392b", alpha=0.08, lw=0)  # other-cell closed zone
for cell, c in [("Hs27", C_HS), ("RPE1", C_RP)]:
    d = sc[sc.fires == cell]
    ax.scatter(d.xo, d.yf, s=26, color=c, alpha=0.6, edgecolor="white", lw=0.4, label=f"fires in {cell} (n={len(d)})")
ax.plot([0, lim], [0, lim], ls="--", color="#888", lw=1)
ax.axvline(4, color="#c0392b", lw=1, ls=":")
ax.text(4.3, lim*0.96, "gene closed where it\ndoesn't fire\n(accessibility explains it)", fontsize=8, color="#c0392b", va="top")
ax.set_xlabel("promoter ATAC in the cell where the off-target does NOT fire")
ax.set_ylabel("promoter ATAC in the cell where it fires")
ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.legend(loc="lower right", frameon=False)

flip = (s.atac_other < 4).mean()*100
both_open = ((s.atac_fire>=6)&(s.atac_other>=6)).mean()*100
nostate = (~((s.atac_other<4)|(s.ntc_other>0.5)|(s.ntc_other==0))).mean()*100
ax.set_title("Most cell-type-specific off-targets are accessible in BOTH cell types\n"
             "→ chromatin accessibility does not explain the cell-type-specificity", fontsize=11, fontweight="bold")
ax.text(0.03, 0.97, f"n = {len(s)} cell-type-specific off-targets\n"
        f"{both_open:.0f}% ATAC-open in both cells\n"
        f"{flip:.0f}% closed where it doesn't fire (accessibility flip)\n"
        f"{nostate:.0f}% open + headroom in both → detection / near-threshold",
        transform=ax.transAxes, va="top", fontsize=8.5,
        bbox=dict(boxstyle="round", fc="white", ec="#ccc"))
fig.tight_layout()
fig.savefig(FIG/"systematic_accessibility_not_the_driver.png", bbox_inches="tight")
plt.close(fig)
print("wrote", FIG/"systematic_accessibility_not_the_driver.png")
print(f"flip={flip:.0f}% both_open={both_open:.0f}% nostate={nostate:.0f}%")
