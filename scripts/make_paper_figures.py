#!/usr/bin/env python3
"""Paper-style figures for the CRISPRa off-target findings (Hartman-paper aesthetic).

Fig 1  CRISPRa vs CRISPRi seed off-target burden (potential, realized, seed->effect)
Fig 2  CRISPRa off-target activation is gated by chromatin state
Fig 3  Replication and context-dependence across two cell types (Hs27, RPE1)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[1]
STEP0 = ROOT.parent / "Step 0- pervasiveness" / "phase0_perturb_seed"
FIG = ROOT / "outputs/figures"
FIG.mkdir(parents=True, exist_ok=True)
M = ROOT / "outputs/metrics"

# ---- paper-ish style ----
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 10,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "font.family": "DejaVu Sans", "legend.frameon": False, "legend.fontsize": 9,
})
C_I, C_A = "#3b6ea5", "#d1495b"          # CRISPRi, CRISPRa
C_HS, C_RP = "#e07a3f", "#4c8c6b"        # Hs27, RPE1
C_POISED = "#e6b84c"
MARKCOL = {"ATAC": "#4c72b0", "H3K4me3": "#55a868", "H3K27ac": "#c79a3f", "H3K27me3": "#8172b3"}


def panel_label(ax, s):
    ax.text(-0.16, 1.06, s, transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")


def tss_dist(df, gene_col, start_col, tssmap):
    return (pd.to_numeric(df[start_col], errors="coerce") - (df[gene_col].map(tssmap).astype(float) - 1)).abs()


def seed_conversion():
    """Fraction of off-target seed events that clear |log2FC|>0.5, by seed length, per mode."""
    tssmap = pd.read_csv(ROOT / "vendor/perturb_seed/tss_map.csv").drop_duplicates("gene_name").set_index("gene_name")["tss"].to_dict()
    # CRISPRa Hs27
    ca = pd.read_csv(M.parent / "candidates/norman_crispra_singlecell_off_target_candidates.csv", low_memory=False)
    ca["offtarget_log2fc"] = pd.to_numeric(ca["offtarget_log2fc"], errors="coerce")
    ca["seed_match_len"] = pd.to_numeric(ca["seed_match_len"], errors="coerce").fillna(0).astype(int)
    ca = ca[ca["gene"].astype(str).str.upper() != ca["offtarget_gene"].astype(str).str.upper()]
    ca["d"] = tss_dist(ca, "offtarget_gene", "seed_match_start", tssmap)
    ca = ca[(ca["d"] <= 1000) & ca["offtarget_log2fc"].notna()]
    # CRISPRi Replogle
    ci = pd.read_parquet(STEP0 / "outputs/candidates/replogle_k562_candidate_level_table.parquet")
    ci = ci[(~ci["is_on_target"]) & (ci["tss_distance_bp"] <= 1000)]
    def curve(df, sl, eff, passfn):
        out = []
        for L in range(5, 13):
            s = df[df[sl] == L] if L < 12 else df[df[sl] >= 12]
            if len(s) >= 5:
                out.append((L, float(passfn(s[eff]).mean()), len(s)))
        return pd.DataFrame(out, columns=["L", "frac", "n"])
    ca_c = curve(ca, "seed_match_len", "offtarget_log2fc", lambda x: x > 0.5)
    ci_c = curve(ci, "seed_length", "repression_magnitude", lambda x: x >= 0.5)
    return ci_c, ca_c


def fig1():
    ci_c, ca_c = seed_conversion()
    fig, ax = plt.subplots(1, 3, figsize=(13.2, 4.1))

    # (a) potential
    ax[0].bar([0, 1], [56.69, 77.52], color=[C_I, C_A], width=0.62, edgecolor="black", linewidth=0.6)
    for x, v in zip([0, 1], [56.69, 77.52]):
        ax[0].text(x, v + 1.5, f"{v:.1f}%", ha="center", fontweight="bold")
    ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["CRISPRi\n(K562)", "CRISPRa\n(Hs27)"])
    ax[0].set_ylim(0, 90); ax[0].set_ylabel("% guides with ≥12 bp\noff-target promoter seed")
    ax[0].set_title("Sequence-level potential"); panel_label(ax[0], "a")

    # (b) realized by seed tier
    tiers = ["≥8 bp", "≥12 bp"]; x = np.arange(2); w = 0.26
    vals = {"CRISPRi": [5.00, 1.59], "CRISPRa Hs27": [4.72, 0.06], "CRISPRa RPE1": [4.66, 0.06]}
    cols = [C_I, C_A, C_RP]
    for i, (k, v) in enumerate(vals.items()):
        b = ax[1].bar(x + (i - 1) * w, v, w, label=k, color=cols[i], edgecolor="black", linewidth=0.5)
        for rect, val in zip(b, v):
            ax[1].text(rect.get_x() + w / 2, val + 0.08, f"{val:.2f}", ha="center", fontsize=7.5)
    ax[1].set_xticks(x); ax[1].set_xticklabels(tiers)
    ax[1].set_ylabel("% analyzed guides (realized)"); ax[1].set_ylim(0, 6)
    ax[1].set_title("Realized off-target burden"); ax[1].legend(loc="upper right")
    ax[1].annotate("comparable", (0, 5.1), ha="center", fontsize=8, color="gray")
    ax[1].annotate("CRISPRa ≈0", (1, 1.75), ha="center", fontsize=8, color="gray")
    panel_label(ax[1], "b")

    # (c) seed -> effect conversion
    ax[2].plot(ci_c["L"], 100 * ci_c["frac"], "-o", color=C_I, label="CRISPRi (repress >0.5)", lw=2)
    ax[2].plot(ca_c["L"], 100 * ca_c["frac"], "-o", color=C_A, label="CRISPRa (activate >0.5)", lw=2)
    ax[2].set_xlabel("seed match length (bp)"); ax[2].set_ylabel("% of seed events with effect")
    ax[2].set_xticks(range(5, 13)); ax[2].set_xticklabels([str(i) for i in range(5, 12)] + ["12+"])
    ax[2].set_title("Effect conversion vs seed length"); ax[2].legend(loc="upper left")
    panel_label(ax[2], "c")

    fig.suptitle("Figure 1.  CRISPRa vs CRISPRi seed-driven off-target burden (matched method)", y=1.02, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "FIG1_crispra_vs_crispri_burden.png", bbox_inches="tight")
    plt.close(fig)
    print("Fig1 written;  CRISPRi conv:", dict(zip(ci_c.L, (100*ci_c.frac).round(1))),
          "\n             CRISPRa conv:", dict(zip(ca_c.L, (100*ca_c.frac).round(1))))


def fig2():
    ch = json.load(open(M / "norman_crispra_chromatin_gating.json"))
    bc = json.load(open(M / "norman_crispra_biology_conditioned.json"))
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.4))

    # (a) activation by expression stratum (poised peak)
    strata = ["silent(0)", "low(0-0.1]", "mid(0.1-0.5]", "high(>0.5)"]
    lab = ["silent", "poised", "mid", "expressed"]
    rates = [100 * bc["b1_activation_by_ntc"][s]["act_rate"] for s in strata]
    cols = ["#9aa6b2", C_POISED, "#bcae8e", "#9aa6b2"]
    b = ax[0, 0].bar(lab, rates, color=cols, edgecolor="black", linewidth=0.6)
    for r, v in zip(b, rates): ax[0, 0].text(r.get_x() + r.get_width()/2, v + 0.4, f"{v:.1f}%", ha="center", fontweight="bold")
    ax[0, 0].set_ylabel("off-target activation rate"); ax[0, 0].set_ylim(0, 24)
    ax[0, 0].set_xlabel("off-target gene baseline expression")
    ax[0, 0].set_title("Activation fires only at poised genes"); panel_label(ax[0, 0], "a")

    # (b) chromatin marks per stratum
    st = ["silent(0)", "poised(0-0.1]", "mid(0.1-0.5]", "expressed(>0.5)"]
    stlab = ["silent", "poised", "mid", "expressed"]
    marks = ["atac", "k4me3", "k27ac", "k27me3"]; mlab = ["ATAC", "H3K4me3", "H3K27ac", "H3K27me3"]
    xx = np.arange(len(st)); w = 0.2
    for i, mk in enumerate(marks):
        vals = [ch["by_ntc_stratum"][s][mk] for s in st]
        ax[0, 1].bar(xx + (i - 1.5) * w, vals, w, label=mlab[i], color=list(MARKCOL.values())[i], edgecolor="black", linewidth=0.4)
    ax[0, 1].set_xticks(xx); ax[0, 1].set_xticklabels(stlab)
    ax[0, 1].set_ylabel("mean promoter signal (TSS ±1 kb)")
    ax[0, 1].set_title("Poised = open + active marks; silent = Polycomb"); ax[0, 1].legend(ncol=2)
    ax[0, 1].annotate("Polycomb\nH3K27me3", (0, 14.3), (0.55, 17), fontsize=8, color=MARKCOL["H3K27me3"],
                      ha="center", arrowprops=dict(arrowstyle="->", color=MARKCOL["H3K27me3"]))
    panel_label(ax[0, 1], "b")

    # (c) activation by ATAC quartile
    q = ["Q1_closed", "Q2", "Q3", "Q4_open"]; qr = [100 * ch["act_by_atac_quartile"][k]["act"] for k in q]
    ax[1, 0].plot([1, 2, 3, 4], qr, "-o", color=MARKCOL["ATAC"], lw=2)
    for xx2, v in zip([1, 2, 3, 4], qr): ax[1, 0].text(xx2, v + 0.5, f"{v:.1f}%", ha="center", fontsize=8.5)
    ax[1, 0].set_xticks([1, 2, 3, 4]); ax[1, 0].set_xticklabels(["closed\nQ1", "Q2", "Q3", "open\nQ4"])
    ax[1, 0].set_ylabel("off-target activation rate"); ax[1, 0].set_ylim(0, 21)
    ax[1, 0].set_xlabel("promoter ATAC accessibility quartile")
    ax[1, 0].set_title("Activation requires open chromatin"); panel_label(ax[1, 0], "c")

    # (d) headroom-controlled
    hc = ch["headroom_controlled"]
    vals = [100 * hc["ATAC-closed"]["act"], 100 * hc["ATAC-open"]["act"]]
    b = ax[1, 1].bar(["ATAC-closed", "ATAC-open"], vals, color=["#9aa6b2", MARKCOL["ATAC"]], edgecolor="black", linewidth=0.6, width=0.55)
    for r, v in zip(b, vals): ax[1, 1].text(r.get_x()+r.get_width()/2, v+0.4, f"{v:.1f}%", ha="center", fontweight="bold")
    ax[1, 1].set_ylabel("off-target activation rate"); ax[1, 1].set_ylim(0, 21)
    ax[1, 1].set_title("Among low-expression genes only\n(headroom held constant)")
    ax[1, 1].annotate("4.4×", (0.5, 11), ha="center", fontsize=12, fontweight="bold", color="gray")
    panel_label(ax[1, 1], "d")

    fig.suptitle("Figure 2.  CRISPRa off-target activation is gated by chromatin state (Hs27 ATAC + CUT&RUN)", y=1.01, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "FIG2_chromatin_gating.png", bbox_inches="tight")
    plt.close(fig)
    print("Fig2 written")


def fig3():
    cc = json.load(open(M / "norman_crispra_cross_celltype.json"))
    fig, ax = plt.subplots(1, 3, figsize=(13.2, 4.2))

    # (a) gate replication
    strata = ["silent(0)", "poised(0-0.1]", "mid(0.1-0.5]", "expressed(>0.5)"]; lab = ["silent", "poised", "mid", "expr"]
    xx = np.arange(4); w = 0.36
    hs = [100 * cc["gate_replication"]["Hs27"][s] for s in strata]
    rp = [100 * cc["gate_replication"]["RPE1"][s] for s in strata]
    ax[0].bar(xx - w/2, hs, w, label="Hs27", color=C_HS, edgecolor="black", linewidth=0.5)
    ax[0].bar(xx + w/2, rp, w, label="RPE1", color=C_RP, edgecolor="black", linewidth=0.5)
    ax[0].set_xticks(xx); ax[0].set_xticklabels(lab); ax[0].set_ylabel("off-target activation rate")
    ax[0].set_title("Poised-gating replicates\nacross cell types"); ax[0].legend(); ax[0].set_ylim(0, 24)
    panel_label(ax[0], "a")

    # (b) realized rate replication
    tiers = ["≥8 bp", "≥12 bp"]; x = np.arange(2)
    hs = [100 * cc["realized"][">=8"]["Hs27_pct"], 100 * cc["realized"][">=12"]["Hs27_pct"]]
    rp = [100 * cc["realized"][">=8"]["RPE1_pct"], 100 * cc["realized"][">=12"]["RPE1_pct"]]
    ax[1].bar(x - w/2, hs, w, label="Hs27", color=C_HS, edgecolor="black", linewidth=0.5)
    ax[1].bar(x + w/2, rp, w, label="RPE1", color=C_RP, edgecolor="black", linewidth=0.5)
    for xi, (h, r) in enumerate(zip(hs, rp)):
        ax[1].text(xi - w/2, h + 0.08, f"{h:.2f}", ha="center", fontsize=7.5)
        ax[1].text(xi + w/2, r + 0.08, f"{r:.2f}", ha="center", fontsize=7.5)
    ax[1].set_xticks(x); ax[1].set_xticklabels(tiers); ax[1].set_ylabel("% analyzed guides (realized)")
    ax[1].set_title("Realized burden replicates"); ax[1].legend(); ax[1].set_ylim(0, 5.6)
    panel_label(ax[1], "b")

    # (c) realized off-target gene overlap (2-set Venn)
    g = cc["realized_offtarget_genes"]
    hs_only, sh, rp_only = g["Hs27"] - g["shared"], g["shared"], g["RPE1"] - g["shared"]
    ax[2].add_patch(Circle((0.38, 0.5), 0.30, color=C_HS, alpha=0.45))
    ax[2].add_patch(Circle((0.62, 0.5), 0.34, color=C_RP, alpha=0.45))
    ax[2].text(0.24, 0.5, f"Hs27\nonly\n{hs_only}", ha="center", va="center", fontsize=9, fontweight="bold")
    ax[2].text(0.50, 0.5, f"{sh}", ha="center", va="center", fontsize=11, fontweight="bold")
    ax[2].text(0.78, 0.5, f"RPE1\nonly\n{rp_only}", ha="center", va="center", fontsize=9, fontweight="bold")
    ax[2].text(0.5, 0.04, f"Jaccard = {g['jaccard']:.2f} — but ~86% fire-capable in both;\nmuch is detection, not biology", ha="center", fontsize=8)
    ax[2].set_xlim(0, 1); ax[2].set_ylim(0, 1); ax[2].axis("off")
    ax[2].set_title("Nominated off-target genes differ\n(mostly a detection artifact)"); panel_label(ax[2], "c")

    fig.suptitle("Figure 3.  Replication across cell types — apparent off-target specificity is mostly a detection artifact", y=1.02, fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "FIG3_cross_celltype.png", bbox_inches="tight")
    plt.close(fig)
    print("Fig3 written")


if __name__ == "__main__":
    fig1(); fig2(); fig3()
    print("\nAll figures in", FIG)
