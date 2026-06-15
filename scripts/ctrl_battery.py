#!/usr/bin/env python3
"""Consolidated, config-driven controls battery (negative + spike-in + covariate
+ equivalence) for one cell type. CELL env var selects Hs27 (local) or RPE1
(cloud replication). Reuses the validated helpers from the per-control scripts.

  CELL=rpe1 python src/ctrl_battery.py
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import controls_lib as cl
import ctrl_config as C
from ctrl_negative import seed_tests
from ctrl_covariate import gc_frac
from ctrl_equivalence import cluster_boot

ACT, CPG, DELTAS, DELTA_MARGIN, SEED = 0.5, 22, [0.1, 0.25, 0.5, 1.0, 2.0], 0.2, 0


def load_inputs():
    gi, gt, var_index = cl.read_obs(C.H5)
    var_set = set(var_index)
    ntc = pd.read_csv(C.NTC)
    ntm_lookup = dict(zip(ntc.ensembl, ntc.ntc_mean_log1p_cp10k))
    cand = pd.read_csv(C.CAND, low_memory=False)
    cand = cand[cand.offtarget_ensembl.isin(var_set)].copy()
    cand["seed_match_len"] = pd.to_numeric(cand.seed_match_len, errors="coerce")
    cand["offtarget_log2fc"] = pd.to_numeric(cand.offtarget_log2fc, errors="coerce")
    cand = cand.dropna(subset=["seed_match_len"])
    return gi, gt, ntm_lookup, cand


def negative(gi, gt, ntm_lookup, cand):
    rng = np.random.RandomState(SEED)
    cand_by = {g: s for g, s in cand.groupby("guide_id")}
    real_guides = list(cand_by)
    ntc_idx = np.where(gt == "non")[0]; rng.shuffle(ntc_idx)
    half = len(ntc_idx) // 2
    baseline_idx, pool_idx = ntc_idx[:half], ntc_idx[half:]
    n_pg = len(pool_idx) // CPG
    chosen = rng.choice(real_guides, n_pg, replace=n_pg > len(real_guides))
    eff = np.array(["__none__"] * len(gi), dtype=object)
    pg_rows, ens = [], set()
    for k in range(n_pg):
        cells = pool_idx[k * CPG:(k + 1) * CPG]; pg = f"PG{k}"; eff[cells] = pg
        for r in cand_by[chosen[k]].itertuples(index=False):
            if pd.notna(r.offtarget_log2fc) or True:
                pg_rows.append((pg, r.offtarget_ensembl, int(r.seed_match_len))); ens.add(r.offtarget_ensembl)
    nt = np.zeros(len(gi), bool); nt[baseline_idx] = True
    gs, gc, ntm, ei = cl.stream_ctrl(C.H5, sorted(ens), eff, {f"PG{k}" for k in range(n_pg)},
                                     nt_cells=nt, seed=SEED)
    rec = [(pg, e, sl, cl.lfc(gs, gc, ntm, ei, pg, e, min_cells=CPG))
           for pg, e, sl in pg_rows]
    n1 = pd.DataFrame([r for r in rec if r[3] is not None],
                      columns=["guide_id", "offtarget_ensembl", "seed_match_len", "offtarget_log2fc"])
    return seed_tests(n1, f"N1 NTC-vs-NTC [{C.TAG}]")


def spikein(gi, gt, ntm_lookup, cand):
    cand = cand.dropna(subset=["offtarget_log2fc"]).copy()
    cand["seed8"] = cand.seed_match_len >= 8
    real_guides = sorted(cand.guide_id.unique())
    ens = sorted(cand.offtarget_ensembl.unique())
    nt = (gt == "non")
    gsB, gcB, ntB, eiB = cl.stream_ctrl(C.H5, ens, gi, set(real_guides), nt_cells=nt)
    base = {}
    rows = []
    for r in cand.itertuples(index=False):
        v = cl.lfc(gsB, gcB, ntB, eiB, r.guide_id, r.offtarget_ensembl)
        if v is not None:
            base[(r.guide_id, r.offtarget_ensembl)] = v
            rows.append((r.guide_id, r.offtarget_ensembl, int(r.seed_match_len), v))
    base_tab = pd.DataFrame(rows, columns=["guide_id", "offtarget_ensembl", "seed_match_len", "offtarget_log2fc"])
    seed_tests(base_tab, f"REAL observed [{C.TAG}]")
    s8 = cand[cand.seed8][["guide_id", "offtarget_ensembl"]].values
    base_rate = float((base_tab[base_tab.seed_match_len >= 8].offtarget_log2fc >= ACT).mean())
    out = []
    for d in DELTAS:
        inj = {}
        for g, e in s8:
            inj.setdefault(g, {})[e] = d
        gsI, gcI, ntI, eiI = cl.stream_ctrl(C.H5, ens, gi, set(real_guides),
                                            inject=inj, ntm_lookup=ntm_lookup, nt_cells=nt)
        rows, recov = [], []
        for r in cand.itertuples(index=False):
            key = (r.guide_id, r.offtarget_ensembl)
            if r.seed8:
                v = cl.lfc(gsI, gcI, ntI, eiI, r.guide_id, r.offtarget_ensembl)
                if v is not None and key in base:
                    recov.append(v - base[key])
            else:
                v = base.get(key)
            if v is not None:
                rows.append((r.guide_id, r.offtarget_ensembl, int(r.seed_match_len), v))
        tab = pd.DataFrame(rows, columns=["guide_id", "offtarget_ensembl", "seed_match_len", "offtarget_log2fc"])
        res = seed_tests(tab, f"inject d={d} [{C.TAG}]")
        sens = float((tab[tab.seed_match_len >= 8].offtarget_log2fc >= ACT).mean())
        res.update(delta=d, sensitivity=sens, base_rate=base_rate,
                   median_recovered=float(np.median(recov)) if recov else np.nan)
        out.append(res)
    return out


def covariate(cand):
    df = cand.dropna(subset=["offtarget_log2fc"]).copy()
    ntc = pd.read_csv(C.NTC); ntm = dict(zip(ntc.ensembl, ntc.ntc_mean_log1p_cp10k))
    tss = pd.read_csv(C.TSS).drop_duplicates("gene_name"); tpos = dict(zip(tss.gene_name, tss.tss))
    for c in ["seed_match_start", "neighbor_rank"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["seed8"] = (df.seed_match_len >= 8).astype(float)
    df["base_expr"] = df.offtarget_ensembl.map(ntm).fillna(0.0)
    df["gc"] = df.genomic_sequence.map(gc_frac)
    df["tss_dist"] = (df.seed_match_start - df.offtarget_gene.map(tpos)).abs()
    for c in ["gc", "tss_dist", "neighbor_rank"]:
        df[c] = df[c].fillna(df[c].median())
    y = df.offtarget_log2fc.values; g = df.guide_id.values
    guides = np.array(sorted(set(g))); gidx = {gg: np.where(g == gg)[0] for gg in guides}
    rng = np.random.RandomState(0)
    def coef(X, yy):
        b, *_ = np.linalg.lstsq(X, yy, rcond=None); return b
    def design(cols):
        parts = [np.ones(len(df))]
        for c in cols:
            v = df[c].values.astype(float)
            parts.append(v if c == "seed8" else (v - v.mean()) / (v.std() + 1e-9))
        return np.column_stack(parts)
    res = []
    for name, cols in {"unadjusted": ["seed8"],
                       "adj_expr_tss_gc_rank": ["seed8", "base_expr", "tss_dist", "gc", "neighbor_rank"]}.items():
        X = design(cols); si = cols.index("seed8") + 1; b = coef(X, y)[si]
        boot = []
        for _ in range(800):
            samp = np.concatenate([gidx[gg] for gg in rng.choice(guides, len(guides), replace=True)])
            try: boot.append(coef(X[samp], y[samp])[si])
            except Exception: pass
        boot = np.array(boot); lo, hi = np.percentile(boot, [2.5, 97.5])
        p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
        print(f"  [{C.TAG}] covariate {name}: seed8={b:+.4f} CI[{lo:+.4f},{hi:+.4f}] p={p:.3f}")
        res.append(dict(spec=name, seed8_coef=b, ci_lo=lo, ci_hi=hi, boot_p=p))
    return res


def equivalence(cand):
    df = cand.dropna(subset=["offtarget_log2fc"])
    bg = df[df.seed_match_len < 5]; bg_by = {g: s.offtarget_log2fc.values for g, s in bg.groupby("guide_id")}
    bg_mean = bg.offtarget_log2fc.mean()
    res = []
    for name, (lo, hi) in {"5-7 bp": (5, 7), "8-11 bp": (8, 11), ">=12 bp": (12, 99)}.items():
        t = df[(df.seed_match_len >= lo) & (df.seed_match_len <= hi)]
        t_by = {g: s.offtarget_log2fc.values for g, s in t.groupby("guide_id")}
        boot = cluster_boot(t_by, bg_by, lambda a, b: (a.mean() if len(a) else np.nan) - (b.mean() if len(b) else np.nan))
        boot = boot[~np.isnan(boot)]; cl_, ch = np.percentile(boot, [2.5, 97.5])
        c90 = np.percentile(boot, [5, 95]); tost = (c90[0] > -DELTA_MARGIN) and (c90[1] < DELTA_MARGIN)
        print(f"  [{C.TAG}] {name}: diff={t.offtarget_log2fc.mean()-bg_mean:+.3f} CI[{cl_:+.3f},{ch:+.3f}] tost<0.2={tost}")
        res.append(dict(tier=name, n=len(t), diff_vs_bg=t.offtarget_log2fc.mean() - bg_mean,
                        ci_lo=cl_, ci_hi=ch, tost_pass_0p2=bool(tost)))
    return res


def main():
    print(f"=== CONTROLS BATTERY [{C.TAG}] | H5={C.H5} ===")
    gi, gt, ntm_lookup, cand = load_inputs()
    print(f"cells={len(gi)} candidates={len(cand)} guides={cand.guide_id.nunique()}")
    print("\n--- negative ---"); n1 = negative(gi, gt, ntm_lookup, cand)
    print("\n--- covariate ---"); cov = covariate(cand)
    print("\n--- equivalence ---"); eqv = equivalence(cand)
    print("\n--- spike-in ---"); spk = spikein(gi, gt, ntm_lookup, cand)
    pd.DataFrame([n1]).to_csv(C.out("negative"), index=False)
    pd.DataFrame(cov).to_csv(C.out("covariate"), index=False)
    pd.DataFrame(eqv).to_csv(C.out("equivalence"), index=False)
    pd.DataFrame(spk).to_csv(C.out("spikein"), index=False)
    print(f"\nwrote tagged outputs for {C.TAG}")


if __name__ == "__main__":
    main()
