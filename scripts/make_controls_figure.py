#!/usr/bin/env python3
"""Single convince-a-skeptic figure for the CRISPRa off-target controls battery.

Panels A-D: Hs27 (local). E: P4 real CRISPRi seed effect collapsing under
downsampling to Norman depth (cloud). F: R1 RPE1 replication of the equivalence
bounds (cloud)."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "outputs/metrics"
spk = pd.read_csv(M / "controls_spikein.csv")
neg = pd.read_csv(M / "controls_negative.csv")
eqv = pd.read_csv(M / "controls_equivalence.csv")
crispri = pd.read_csv(M / "controls_crispri_downsample.csv")        # P4 (cloud)
eqv_rpe1 = pd.read_csv(M / "controls_equivalence_rpe1.csv")         # R1 (cloud)

fig, ax = plt.subplots(2, 3, figsize=(16.5, 8.8))
RED, BLUE, GRAY, GREEN, PURP = "#E24B4A", "#378ADD", "#888780", "#1D9E75", "#8A4FBE"

# --- A: spike-in detection floor (seed-perm p vs injected delta) ---
a = ax[0, 0]
d = [0.0] + list(spk.delta)
p = [0.653] + list(spk.seedperm_p.clip(lower=1e-4))
a.plot(d, p, "o-", color=BLUE, lw=2, ms=7)
a.axhline(0.05, color=RED, ls="--", lw=1.2, label="p = 0.05")
a.annotate("real data\n(p = 0.65)", (0.0, 0.653), textcoords="offset points",
           xytext=(18, -4), fontsize=9, color=GRAY)
a.set_yscale("log"); a.set_xlabel("injected seed effect (log2FC)")
a.set_ylabel("aggregate seed-permutation p")
a.set_title("A. Positive control: we detect injected\nseed effects down to ~0.1 log2FC", fontsize=11)
a.legend(fontsize=9); a.grid(alpha=.25)

# --- B: within-guide 'effect' is reproduced by nulls (artifact) ---
b = ax[0, 1]
labels = ["real\nobserved", "scramble\nnull", "NTC-vs-NTC\nnull"]
meds = [0.098, float(neg[neg.label.str.contains("scramble")].within_med),
        float(neg[neg.label.str.contains("NTC")].within_med)]
b.bar(labels, meds, color=[RED, GRAY, GRAY], width=.6)
b.axhline(0, color="k", lw=.8)
for i, v in enumerate(meds):
    b.text(i, v + .004, f"{v:+.3f}", ha="center", fontsize=9)
b.set_ylabel("within-guide median diff (log2FC)")
b.set_title("B. The within-guide 'seed signal' is an artifact:\nnull controls reproduce it", fontsize=11)
b.grid(alpha=.25, axis="y")

# --- C: per-pair sensitivity vs recovered effect ---
c = ax[0, 2]
c.plot(spk.median_recovered, spk.sensitivity * 100, "s-", color=GREEN, lw=2, ms=7)
c.axhline(spk.base_rate.iloc[0] * 100, color=GRAY, ls=":", label="background (no seed)")
c.set_xlabel("recovered off-target effect (log2FC)")
c.set_ylabel("% of pairs crossing +0.5 (sensitivity)")
c.set_title("C. Per-pair detection: only strong effects\n(>~1 log2FC) are individually callable", fontsize=11)
c.legend(fontsize=9); c.grid(alpha=.25)

# --- D: equivalence forest by seed tier (Hs27) ---
d2 = ax[1, 0]
tiers = eqv.tier.tolist()
ypos = np.arange(len(tiers))[::-1]
d2.axvspan(-0.2, 0.2, color=GREEN, alpha=.12, label="negligible (|δ|<0.2)")
d2.axvline(0, color="k", lw=.8)
for i, r in eqv.iterrows():
    yy = ypos[i]
    d2.plot([r.ci_lo, r.ci_hi], [yy, yy], color=BLUE, lw=2)
    d2.plot(r.diff_vs_bg, yy, "o", color=BLUE, ms=8)
d2.set_yticks(ypos); d2.set_yticklabels(tiers)
d2.set_xlabel("seed effect vs background (log2FC)")
d2.set_title("D. Equivalence (Hs27): 8-11 bp bounded < 0.06;\n>=12 bp (n=14) unresolved", fontsize=11)
d2.legend(fontsize=9, loc="lower right"); d2.grid(alpha=.25, axis="x")

# --- E: P4 real CRISPRi seed effect collapses under downsampling ---
e = ax[1, 1]
cr = crispri.copy()
cr["cells"] = cr.median_cells_per_guide
cr = cr.sort_values("cells")
e.plot(cr.cells, cr.fisher_OR, "o-", color=PURP, lw=2, ms=7, label="seed≥8 / seed<8 odds ratio")
e.axhline(1.0, color="k", lw=.8, ls="--")
e.axvline(22, color=RED, ls=":", lw=1.4)
e.annotate("Norman CRISPRa\ndepth (~22 cells)", (22, e.get_ylim()[1]*0.7),
           fontsize=8.5, color=RED, ha="left")
for _, r in cr.iterrows():
    e.text(r.cells, r.fisher_OR + .08, f"{r.fisher_OR:.2f}", ha="center", fontsize=8)
e.set_xscale("log"); e.set_xlabel("cells per guide (downsampled)")
e.set_ylabel("CRISPRi seed enrichment (odds ratio)")
e.set_title("E. Positive control (real CRISPRi): a true seed effect\n"
            "(OR≈2.9) collapses toward background at 22 cells/guide", fontsize=11)
e.legend(fontsize=8.5, loc="upper left"); e.grid(alpha=.25)

# --- F: R1 RPE1 replicates the equivalence verdict ---
f = ax[1, 2]
f.axvspan(-0.2, 0.2, color=GREEN, alpha=.12, label="negligible (|δ|<0.2)")
f.axvline(0, color="k", lw=.8)
mh = eqv.set_index("tier"); mr = eqv_rpe1.set_index("tier")
common = [t for t in mh.index if t in mr.index]
ypos = np.arange(len(common))[::-1]
for i, t in enumerate(common):
    yy = ypos[i]
    f.plot([mh.loc[t].ci_lo, mh.loc[t].ci_hi], [yy + .12, yy + .12], color=BLUE, lw=2)
    f.plot(mh.loc[t].diff_vs_bg, yy + .12, "o", color=BLUE, ms=7,
           label="Hs27" if i == 0 else None)
    f.plot([mr.loc[t].ci_lo, mr.loc[t].ci_hi], [yy - .12, yy - .12], color=RED, lw=2)
    f.plot(mr.loc[t].diff_vs_bg, yy - .12, "s", color=RED, ms=7,
           label="RPE1 (replication)" if i == 0 else None)
f.set_yticks(ypos); f.set_yticklabels(common)
f.set_xlabel("seed effect vs background (log2FC)")
f.set_title("F. Replication (R1): RPE1 reproduces the bound —\n5-11 bp negligible, >=12 bp (n=17) unresolved", fontsize=11)
f.legend(fontsize=8.5, loc="lower right"); f.grid(alpha=.25, axis="x")

fig.suptitle("CRISPRa off-target controls: the pipeline measures real off-targets (A,C,E), the apparent signal is an artifact (B), "
             "and the true seed effect is negligible & replicated (D,F)",
             fontsize=12.5, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = ROOT / "outputs/figures/controls_battery.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
