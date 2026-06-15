#!/usr/bin/env python3
"""Does the AGGREGATE CRISPRa off-target burden survive a seed-permuted null?

The per-pair direct/trans calls don't reproduce (winner's curse, see trans_mechanism_reality.py).
Separate, properly-powered question: across ALL neighbor events, does a strong SEED match (>=8 bp)
actually enrich for a realized activation effect (offtarget_log2fc >= +0.5)? If seed off-targeting
is real at the population level, seed>=8 events should show large effects far more than seed<8 — and
far more than a null where seed lengths are shuffled across events (breaking the seed<->effect link).

Universe: outputs/metrics/norman_crispra_all_neighbor_events_seeded.csv (9436 guide->neighbor events,
each with seed_match_len 0..13 and the guide's measured offtarget_log2fc on that neighbor).

Tests:
  (1) realized-activation rate by seed-length bin
  (2) seed>=8 vs seed<8 enrichment for effect>=0.5  (Fisher OR)
  (3) seed-permutation null: shuffle seed_match_len across events, 5000x -> p for observed seed>=8 rate
  (4) within-guide paired: do seed>=8 neighbors out-activate seed<8 neighbors of the SAME guide?
      (controls for guide potency)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
# CORRECT universe = the candidate table behind the 4.7% realized headline (3326 guides),
# NOT all_neighbor_events_seeded.csv (1102 guides = the discarded 0.18% Codex artifact).
df = pd.read_csv(ROOT / "outputs/candidates/norman_crispra_singlecell_off_target_candidates.csv",
                 low_memory=False)
df["seed_match_len"] = pd.to_numeric(df["seed_match_len"], errors="coerce")
df["offtarget_log2fc"] = pd.to_numeric(df["offtarget_log2fc"], errors="coerce")
df = df.dropna(subset=["offtarget_log2fc", "seed_match_len"]).copy()
df["seed_match_len"] = df["seed_match_len"].astype(int)
df["seed8"] = df.seed_match_len >= 8
ACT = 0.5
df["realized"] = df.offtarget_log2fc >= ACT     # CRISPRa activation criterion
n = len(df)
print(f"neighbor events: {n} | seed>=8: {df.seed8.sum()} ({df.seed8.mean()*100:.0f}%) | "
      f"activation>=+0.5: {df.realized.sum()} ({df.realized.mean()*100:.1f}%)\n")

# (1) realized rate by seed bin
df["bin"] = pd.cut(df.seed_match_len, [-1,0,4,7,11,13], labels=["0","1-4","5-7","8-11","12-13"])
print("=== (1) activation rate & mean effect by seed-match length ===")
g = df.groupby("bin", observed=True).agg(n=("realized","size"), act_rate=("realized","mean"),
                                          mean_lfc=("offtarget_log2fc","mean"))
g["act_rate"] = (g.act_rate*100).round(1)
print(g.to_string())

# (2) seed>=8 vs seed<8 enrichment
a = int(df[df.seed8].realized.sum()); b = int(len(df[df.seed8]) - a)
c = int(df[~df.seed8].realized.sum()); d = int(len(df[~df.seed8]) - c)
OR, fish_p = stats.fisher_exact([[a,b],[c,d]])
r8, r0 = a/(a+b), c/(c+d)
print(f"\n=== (2) seed>=8 vs seed<8 ===")
print(f"  realized-activation rate: seed>=8 {r8*100:.1f}% ({a}/{a+b}) vs seed<8 {r0*100:.1f}% ({c}/{c+d})")
print(f"  Fisher OR={OR:.2f}, p={fish_p:.2e}  (OR>>1 & p<<0.05 => seed predicts activation)")

# (3) seed-permutation null on the seed>=8 realized rate
rng = np.random.RandomState(0)
seed_len = df.seed_match_len.values.copy(); realized = df.realized.values
obs_rate = r8; NPERM = 5000
null = np.empty(NPERM)
for i in range(NPERM):
    perm = rng.permutation(seed_len)
    m = perm >= 8
    null[i] = realized[m].mean() if m.any() else 0.0
p_perm = (np.sum(null >= obs_rate) + 1) / (NPERM + 1)
print(f"\n=== (3) seed-permutation null (shuffle seed lengths, {NPERM}x) ===")
print(f"  observed seed>=8 activation rate: {obs_rate*100:.1f}%")
print(f"  null rate: mean {null.mean()*100:.1f}%, 95% [{np.percentile(null,2.5)*100:.1f}, {np.percentile(null,97.5)*100:.1f}]")
print(f"  one-sided p(observed beats null): {p_perm:.4f}  (small => aggregate seed effect is REAL)")

# (4) within-guide paired test (controls for guide potency)
rows = []
for gid, sub in df.groupby("guide_id"):
    s8 = sub[sub.seed8].offtarget_log2fc; s0 = sub[~sub.seed8].offtarget_log2fc
    if len(s8) and len(s0):
        rows.append((s8.mean(), s0.mean()))
pa = np.array(rows)
print(f"\n=== (4) within-guide paired (guides with both seed>=8 and seed<8 neighbors, n={len(pa)}) ===")
if len(pa) > 5:
    w = stats.wilcoxon(pa[:,0], pa[:,1])
    print(f"  mean offtarget_log2fc: seed>=8 neighbors {pa[:,0].mean():.3f} vs seed<8 {pa[:,1].mean():.3f}")
    print(f"  paired diff median {np.median(pa[:,0]-pa[:,1]):+.3f}, Wilcoxon p={w.pvalue:.2e} "
          f"(seed>=8 higher in same guide => seed effect real, guide-strength-controlled)")
print("\nVERDICT: aggregate seed off-target signal is REAL iff (2) OR>>1, (3) p small, (4) seed>=8 > seed<8.")
