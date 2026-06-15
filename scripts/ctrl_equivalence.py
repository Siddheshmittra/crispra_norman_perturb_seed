#!/usr/bin/env python3
"""E1 equivalence (TOST) + M2 seed-tier stratification, on the real candidate
table. Establishes the tightest bound on the CRISPRa seed off-target effect and
flags the >=12 bp tier as the one regime we cannot exclude. Cluster-bootstrap
over guides; no streaming."""
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DELTA_MARGIN = 0.2


def cluster_boot(vals_by_guide_a, vals_by_guide_b, stat, NB=2000, seed=0):
    """Bootstrap a difference-of-means statistic, resampling guides."""
    guides = sorted(set(vals_by_guide_a) | set(vals_by_guide_b))
    rng = np.random.RandomState(seed)
    out = np.empty(NB)
    for i in range(NB):
        gs = rng.choice(guides, len(guides), replace=True)
        a = np.concatenate([vals_by_guide_a.get(g, np.array([])) for g in gs])
        b = np.concatenate([vals_by_guide_b.get(g, np.array([])) for g in gs])
        out[i] = stat(a, b)
    return out


def main():
    cand = pd.read_csv(ROOT / "outputs/candidates/norman_crispra_singlecell_off_target_candidates.csv",
                       low_memory=False)
    cand["seed_match_len"] = pd.to_numeric(cand.seed_match_len, errors="coerce")
    cand["offtarget_log2fc"] = pd.to_numeric(cand.offtarget_log2fc, errors="coerce")
    cand = cand.dropna(subset=["offtarget_log2fc", "seed_match_len"])

    # background = no/short seed (<5 bp); test tiers against it
    bg = cand[cand.seed_match_len < 5]
    bg_by = {g: s.offtarget_log2fc.values for g, s in bg.groupby("guide_id")}
    bg_mean = bg.offtarget_log2fc.mean()

    tiers = {"5-7 bp": (5, 7), "8-11 bp": (8, 11), ">=12 bp": (12, 99)}
    print(f"background (<5 bp) mean log2FC = {bg_mean:+.3f}  (n={len(bg)})\n")
    print(f"{'tier':<10}{'n':>6}{'mean':>9}{'diff vs bg':>12}{'95% CI':>22}{'TOST<0.2':>10}")
    rows = []
    for name, (lo, hi) in tiers.items():
        t = cand[(cand.seed_match_len >= lo) & (cand.seed_match_len <= hi)]
        t_by = {g: s.offtarget_log2fc.values for g, s in t.groupby("guide_id")}
        diff = t.offtarget_log2fc.mean() - bg_mean
        boot = cluster_boot(t_by, bg_by,
                            lambda a, b: (a.mean() if len(a) else np.nan) - (b.mean() if len(b) else np.nan))
        boot = boot[~np.isnan(boot)]
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
        # TOST: equivalent to within +/-margin if the 90% CI lies inside (-d, d)
        c90lo, c90hi = np.percentile(boot, [5, 95])
        tost_pass = (c90lo > -DELTA_MARGIN) and (c90hi < DELTA_MARGIN)
        print(f"{name:<10}{len(t):>6}{t.offtarget_log2fc.mean():>+9.3f}{diff:>+12.3f}"
              f"{f'[{ci_lo:+.3f},{ci_hi:+.3f}]':>22}{'YES' if tost_pass else 'NO':>10}")
        rows.append(dict(tier=name, n=len(t), mean=t.offtarget_log2fc.mean(),
                         diff_vs_bg=diff, ci_lo=ci_lo, ci_hi=ci_hi,
                         tost_pass_0p2=bool(tost_pass)))

    # tightest symmetric bound that the seed>=8 effect clears (the equivalence headline)
    s8 = cand[cand.seed_match_len >= 8]
    s8_by = {g: s.offtarget_log2fc.values for g, s in s8.groupby("guide_id")}
    boot = cluster_boot(s8_by, bg_by,
                        lambda a, b: (a.mean() if len(a) else np.nan) - (b.mean() if len(b) else np.nan))
    boot = boot[~np.isnan(boot)]
    c90 = np.percentile(boot, [5, 95])
    tightest = max(abs(c90[0]), abs(c90[1]))
    print(f"\nseed>=8 vs bg: diff {s8.offtarget_log2fc.mean()-bg_mean:+.3f}, "
          f"90% CI [{c90[0]:+.3f},{c90[1]:+.3f}]")
    print(f"=> EQUIVALENCE: the CRISPRa seed>=8 mean effect is statistically bounded "
          f"below |{tightest:.3f}| log2FC (clears the 0.2 margin: "
          f"{'YES' if tightest < DELTA_MARGIN else 'NO'}).")
    print(f"\nHONEST CAVEAT: the >=12 bp tier has n={int((cand.seed_match_len>=12).sum())} "
          f"candidates -> its CI is wide; a CRISPRi-like effect there is NOT excluded.")
    pd.DataFrame(rows).to_csv(ROOT / "outputs/metrics/controls_equivalence.csv", index=False)
    print("\nwrote outputs/metrics/controls_equivalence.csv")


if __name__ == "__main__":
    main()
