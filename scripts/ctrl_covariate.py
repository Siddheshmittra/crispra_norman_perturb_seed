#!/usr/bin/env python3
"""M1: is the tiny within-guide 'seed effect' real, or a covariate artifact?

The scramble null (ctrl_negative.py N2) already reproduced the +0.10 within-guide
median, implying it is a property of WHICH genes carry long seeds, not of the
perturbation. Here we test that directly: regress the off-target log2FC on the
seed-length indicator WITH and WITHOUT covariates (baseline expression, TSS
distance, GC content, neighbour rank) + guide fixed effects. If the seed
coefficient collapses to ~0 once covariates are in, the 'effect' is confounding.
Cluster-bootstrap over guides for inference. No streaming (uses the real table).
"""
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def gc_frac(s):
    if not isinstance(s, str) or not s:
        return np.nan
    s = s.upper()
    n = sum(c in "ACGT" for c in s)
    return (s.count("G") + s.count("C")) / n if n else np.nan


def main():
    cand = pd.read_csv(ROOT / "outputs/candidates/norman_crispra_singlecell_off_target_candidates.csv",
                       low_memory=False)
    for c in ["seed_match_len", "offtarget_log2fc", "seed_match_start", "neighbor_rank"]:
        cand[c] = pd.to_numeric(cand[c], errors="coerce")
    cand = cand.dropna(subset=["offtarget_log2fc", "seed_match_len"]).copy()

    ntc = pd.read_csv(ROOT / "outputs/metrics/norman_crispra_ntc_gene_expression.csv")
    ntm = dict(zip(ntc.ensembl, ntc.ntc_mean_log1p_cp10k))
    tss = pd.read_csv(ROOT / "vendor/perturb_seed/tss_map.csv").drop_duplicates("gene_name")
    tss_pos = dict(zip(tss.gene_name, tss.tss))

    cand["seed8"] = (cand.seed_match_len >= 8).astype(float)
    cand["base_expr"] = cand.offtarget_ensembl.map(ntm).fillna(0.0)
    cand["gc"] = cand.genomic_sequence.map(gc_frac)
    t = cand.offtarget_gene.map(tss_pos)
    cand["tss_dist"] = (cand.seed_match_start - t).abs()
    cand["tss_dist"] = cand["tss_dist"].fillna(cand["tss_dist"].median())
    cand["gc"] = cand["gc"].fillna(cand["gc"].median())
    cand["nrank"] = cand.neighbor_rank.fillna(cand.neighbor_rank.median())

    y = cand.offtarget_log2fc.values
    g = cand.guide_id.values

    def ols_coef(X, y):
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return beta

    def design(df, cols, fe=False):
        parts = [np.ones(len(df))]
        for c in cols:
            v = df[c].values.astype(float)
            parts.append((v - v.mean()) / (v.std() + 1e-9) if c != "seed8" else v)
        X = np.column_stack(parts)
        return X

    specs = {
        "unadjusted": ["seed8"],
        "+ expr": ["seed8", "base_expr"],
        "+ expr,tss,gc,rank": ["seed8", "base_expr", "tss_dist", "gc", "nrank"],
    }
    # cluster bootstrap over guides
    guides = np.array(sorted(set(g)))
    gidx = {gg: np.where(g == gg)[0] for gg in guides}
    rng = np.random.RandomState(0)
    NB = 1000

    print(f"events={len(cand)} seed>=8={int(cand.seed8.sum())}\n")
    print(f"{'spec':<22}{'seed8 coef':>12}{'95% CI':>22}{'boot p':>9}")
    rows = []
    for name, cols in specs.items():
        X = design(cand, cols)
        si = cols.index("seed8") + 1
        b = ols_coef(X, y)[si]
        boot = np.empty(NB)
        for i in range(NB):
            samp = np.concatenate([gidx[gg] for gg in rng.choice(guides, len(guides), replace=True)])
            Xb = X[samp]; yb = y[samp]
            try:
                boot[i] = ols_coef(Xb, yb)[si]
            except Exception:
                boot[i] = np.nan
        boot = boot[~np.isnan(boot)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
        print(f"{name:<22}{b:>+12.4f}{f'[{lo:+.4f},{hi:+.4f}]':>22}{p:>9.3f}")
        rows.append(dict(spec=name, seed8_coef=b, ci_lo=lo, ci_hi=hi, boot_p=p))

    # guide fixed effects (within-guide), fully adjusted: demean y & covars within guide
    df = cand.copy()
    for c in ["offtarget_log2fc", "base_expr", "tss_dist", "gc", "nrank", "seed8"]:
        df[c + "_d"] = df[c] - df.groupby("guide_id")[c].transform("mean")
    Xfe = np.column_stack([df.seed8_d, df.base_expr_d, df.tss_dist_d, df.gc_d, df.nrank_d])
    bfe = ols_coef(Xfe, df.offtarget_log2fc_d.values)[0]
    boot = np.empty(NB)
    for i in range(NB):
        samp = np.concatenate([gidx[gg] for gg in rng.choice(guides, len(guides), replace=True)])
        try:
            boot[i] = ols_coef(Xfe[samp], df.offtarget_log2fc_d.values[samp])[0]
        except Exception:
            boot[i] = np.nan
    boot = boot[~np.isnan(boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    print(f"{'within-guide FE adj':<22}{bfe:>+12.4f}{f'[{lo:+.4f},{hi:+.4f}]':>22}{p:>9.3f}")
    rows.append(dict(spec="within-guide FE adj", seed8_coef=bfe, ci_lo=lo, ci_hi=hi, boot_p=p))

    pd.DataFrame(rows).to_csv(ROOT / "outputs/metrics/controls_covariate.csv", index=False)
    print("\nwrote outputs/metrics/controls_covariate.csv")
    print("READ: if seed8 coef collapses toward 0 (CI spans 0) once expression is")
    print("added, the within-guide 'seed effect' is a covariate artifact (matches scramble).")


if __name__ == "__main__":
    main()
