#!/usr/bin/env python3
"""Negative controls: prove the detection pipeline yields NO seed->effect signal
when there is no real perturbation.

N1 NTC-vs-NTC: split the non-targeting cells into a baseline pool + a set of
   pseudo-guides (~22 cells each). Each pseudo-guide INHERITS a real guide's
   candidate neighbour genes + seed lengths, but its cells are pure NTC. Compute
   the off-target log2FC on those candidates and run the SAME seed-permutation +
   within-guide tests. If the pipeline is clean, seed>=8 candidates must NOT
   out-activate seed<8 (no real biology can exist in NTC cells).

N2 Scramble: same, but pseudo-guides draw from ALL cells with guide labels
   permuted (any apparent seed effect is then label noise).
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import controls_lib as cl

ROOT = Path(__file__).resolve().parents[1]
H5 = ROOT / "data/raw/fibroblast_CRISPRa_final_pop_singlets_normalized_log1p.h5ad"
ACT = 0.5
CELLS_PER_PG = 22
SEED = 0


def seed_tests(df, label):
    """Run the seed-perm + within-guide battery on a candidate table; print + return."""
    df = df.dropna(subset=["offtarget_log2fc", "seed_match_len"]).copy()
    df["seed8"] = df.seed_match_len >= 8
    df["realized"] = df.offtarget_log2fc >= ACT
    a = int(df[df.seed8].realized.sum()); b = int(len(df[df.seed8]) - a)
    c = int(df[~df.seed8].realized.sum()); d = int(len(df[~df.seed8]) - c)
    r8 = a / max(a + b, 1); r0 = c / max(c + d, 1)
    OR, fish_p = stats.fisher_exact([[a, b], [c, d]]) if (a + b) and (c + d) else (np.nan, np.nan)
    # seed-permutation null
    rng = np.random.RandomState(0)
    sl = df.seed_match_len.values.copy(); rz = df.realized.values
    NPERM = 5000; null = np.empty(NPERM)
    for i in range(NPERM):
        m = rng.permutation(sl) >= 8
        null[i] = rz[m].mean() if m.any() else 0.0
    p_perm = (np.sum(null >= r8) + 1) / (NPERM + 1)
    # within-guide paired
    rows = []
    for gid, sub in df.groupby("guide_id"):
        s8 = sub[sub.seed8].offtarget_log2fc; s0 = sub[~sub.seed8].offtarget_log2fc
        if len(s8) and len(s0):
            rows.append((s8.mean(), s0.mean()))
    pa = np.array(rows)
    if len(pa) > 5:
        w_p = stats.wilcoxon(pa[:, 0], pa[:, 1]).pvalue
        med = float(np.median(pa[:, 0] - pa[:, 1]))
    else:
        w_p, med = np.nan, np.nan
    print(f"\n[{label}] events={len(df)} seed>=8={int(df.seed8.sum())}")
    print(f"  activation seed>=8 {r8*100:.1f}% vs seed<8 {r0*100:.1f}% | Fisher OR={OR:.2f} p={fish_p:.2g}")
    print(f"  seed-perm null p={p_perm:.3f}  (LARGE => clean null)")
    print(f"  within-guide median diff {med:+.3f} Wilcoxon p={w_p:.3g} (n={len(pa)})")
    return dict(label=label, n=len(df), r8=r8, r0=r0, fisher_OR=OR, fisher_p=fish_p,
                seedperm_p=p_perm, within_med=med, within_p=w_p, n_pairs=len(pa))


def main():
    gi, gt, var_index = cl.read_obs(H5)
    var_set = set(var_index)
    ntc_csv = pd.read_csv(ROOT / "outputs/metrics/norman_crispra_ntc_gene_expression.csv")
    ntm_lookup = dict(zip(ntc_csv.ensembl, ntc_csv.ntc_mean_log1p_cp10k))

    cand = pd.read_csv(ROOT / "outputs/candidates/norman_crispra_singlecell_off_target_candidates.csv",
                       low_memory=False)
    cand = cand[cand.offtarget_ensembl.isin(var_set)].copy()
    cand["seed_match_len"] = pd.to_numeric(cand.seed_match_len, errors="coerce")
    cand = cand.dropna(subset=["seed_match_len"])
    cand_by_guide = {g: sub for g, sub in cand.groupby("guide_id")}
    real_guides = list(cand_by_guide.keys())

    rng = np.random.RandomState(SEED)

    # ---------- N1: NTC-vs-NTC ----------
    ntc_idx = np.where(gt == "non")[0]
    rng.shuffle(ntc_idx)
    half = len(ntc_idx) // 2
    baseline_idx = ntc_idx[:half]                      # NTC baseline pool
    pool_idx = ntc_idx[half:]                          # cells -> pseudo-guides
    n_pg = len(pool_idx) // CELLS_PER_PG
    chosen = rng.choice(real_guides, size=n_pg, replace=False) \
        if n_pg <= len(real_guides) else rng.choice(real_guides, size=n_pg, replace=True)

    eff = np.array(["__none__"] * len(gi), dtype=object)
    pg_rows = []
    ens_needed = set()
    for k in range(n_pg):
        cells = pool_idx[k * CELLS_PER_PG:(k + 1) * CELLS_PER_PG]
        pg = f"PG{k}"
        eff[cells] = pg
        sub = cand_by_guide[chosen[k]]
        for r in sub.itertuples(index=False):
            pg_rows.append((pg, r.offtarget_ensembl, int(r.seed_match_len)))
            ens_needed.add(r.offtarget_ensembl)
    nt_cells = np.zeros(len(gi), bool); nt_cells[baseline_idx] = True
    guide_set = {f"PG{k}" for k in range(n_pg)}
    print(f"N1 NTC-vs-NTC: {n_pg} pseudo-guides x {CELLS_PER_PG} cells; "
          f"{len(ens_needed)} candidate genes; baseline {len(baseline_idx)} NTC")
    gsum, gcnt, ntm, ei = cl.stream_ctrl(H5, sorted(ens_needed), eff, guide_set,
                                         nt_cells=nt_cells, seed=SEED)
    rec = []
    for pg, ens, sl in pg_rows:
        v = cl.lfc(gsum, gcnt, ntm, ei, pg, ens, min_cells=CELLS_PER_PG)
        if v is not None:
            rec.append((pg, ens, sl, v))
    n1 = pd.DataFrame(rec, columns=["guide_id", "offtarget_ensembl", "seed_match_len", "offtarget_log2fc"])
    res_n1 = seed_tests(n1, "N1 NTC-vs-NTC")

    # ---------- N2: scrambled guide labels ----------
    real_cells = np.where((gt != "non") & np.isin(gi, real_guides))[0]
    perm_labels = gi[real_cells].copy()
    rng.shuffle(perm_labels)                            # break cell<->guide link
    eff2 = np.array(["__none__"] * len(gi), dtype=object)
    eff2[real_cells] = perm_labels
    nt_all = np.zeros(len(gi), bool); nt_all[gt == "non"] = True
    # keep guides that still have >=20 scrambled cells
    vc = pd.Series(perm_labels).value_counts()
    keep2 = set(vc[vc >= 20].index)
    ens2 = set(cand[cand.guide_id.isin(keep2)].offtarget_ensembl)
    print(f"\nN2 scramble: {len(keep2)} guides w/ >=20 scrambled cells; {len(ens2)} genes")
    gs2, gc2, nt2, ei2 = cl.stream_ctrl(H5, sorted(ens2), eff2, keep2,
                                        nt_cells=nt_all, seed=SEED)
    rec2 = []
    for g in keep2:
        for r in cand_by_guide[g].itertuples(index=False):
            v = cl.lfc(gs2, gc2, nt2, ei2, g, r.offtarget_ensembl)
            if v is not None:
                rec2.append((g, r.offtarget_ensembl, int(r.seed_match_len), v))
    n2 = pd.DataFrame(rec2, columns=["guide_id", "offtarget_ensembl", "seed_match_len", "offtarget_log2fc"])
    res_n2 = seed_tests(n2, "N2 scramble")

    out = pd.DataFrame([res_n1, res_n2])
    out.to_csv(ROOT / "outputs/metrics/controls_negative.csv", index=False)
    print("\nwrote outputs/metrics/controls_negative.csv")
    print("EXPECT both: seed-perm p large, within-guide ns -> pipeline invents no signal.")


if __name__ == "__main__":
    main()
