#!/usr/bin/env python3
"""Figure: split-half reproducibility of nominated off-target pairs.

The punchline of the depth limitation. On-target activation reproduces strongly
(r=+0.83); random A->B pairs set a noise floor (r=+0.30); both DIRECT and
INDIRECT nominated off-target classes fall BELOW that floor (winner's curse),
so per-pair off-target calls are not resolvable at ~22 cells/guide.

Values are the settled split-half correlations from trans_mechanism_reality.py
(reported in CRISPRa_offtarget_report_2026-06-14_figures.md, Section 2).
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/figures/panels/splithalf_noisefloor.png"

# (label, r, color)  — house scheme: CRISPRa red family, neutral grays, control green
BARS = [
    ("On-target\nA→A\n(positive control)", 0.83, "#2ca25f"),
    ("Random A→B\n(noise floor)",            0.30, "#969696"),
    ("DIRECT\nnominated",                          -0.13, "#d95f0e"),
    ("INDIRECT\nnominated",                        -0.18, "#c51b8a"),
]


def main():
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    labels = [b[0] for b in BARS]
    vals = [b[1] for b in BARS]
    cols = [b[2] for b in BARS]
    x = range(len(BARS))

    bars = ax.bar(x, vals, color=cols, edgecolor="k", linewidth=0.7, width=0.62, zorder=3)

    # noise-floor reference line
    ax.axhline(0.30, ls="--", lw=1.3, color="#525252", zorder=2)
    ax.text(len(BARS) - 0.5, 0.33, "noise floor (r = +0.30)",
            ha="right", va="bottom", fontsize=9, color="#525252", style="italic")
    ax.axhline(0, color="k", lw=0.8, zorder=2)

    for xi, v in zip(x, vals):
        off = 0.03 if v >= 0 else -0.03
        va = "bottom" if v >= 0 else "top"
        ax.text(xi, v + off, f"{v:+.2f}", ha="center", va=va, fontsize=11, fontweight="bold")

    # shade the below-floor zone
    ax.axhspan(-0.35, 0.30, color="#fdd", alpha=0.18, zorder=0)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("split-half reproducibility  (Pearson r)", fontsize=11)
    ax.set_ylim(-0.35, 0.95)
    ax.set_title("Per-pair off-target calls fall below the noise floor\n"
                 "(Norman atlas, ~22 cells/guide)", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
