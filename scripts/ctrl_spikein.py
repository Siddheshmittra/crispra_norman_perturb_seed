#!/usr/bin/env python3
"""Positive control: prove the pipeline DETECTS a seed off-target effect when one
is injected. Two readouts:

(A) Per-pair sensitivity: inject magnitude delta into seed>=8 (guide,gene) pairs;
    fraction whose recovered lfc crosses the +0.5 'realized' threshold vs delta.
(B) Aggregate recovery: inject delta into ALL seed>=8 candidates, rebuild the
    candidate table (seed<8 = baseline, seed>=8 = injected), and recompute the
    seed-permutation p. Shows the smallest delta at which the AGGREGATE test we
    used would have fired. Combined with the observed real p=0.65, this converts
    'we saw nothing' into 'an effect >= X would have been detected'.

Depth curve: repeat at downsampled cells/guide via --depths.
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import controls_lib as cl
from ctrl_negative import seed_tests

ROOT = Path(__file__).resolve().parents[1]
H5 = ROOT / "data/raw/fibroblast_CRISPRa_final_pop_singlets_normalized_log1p.h5ad"
DELTAS = [0.1, 0.25, 0.5, 1.0, 2.0]
DEPTHS = [None]  # native depth; depth curve added in ctrl_depth.py


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
    cand["seed8"] = cand.seed_match_len >= 8

    real_guides = sorted(cand.guide_id.unique())
    ens_needed = sorted(cand.offtarget_ensembl.unique())
    nt_cells = (gt == "non")
    eff = gi

    # ---- baseline pass (no injection): the real observed table ----
    print("=== baseline pass (no injection) ===")
    gsB, gcB, ntB, eiB = cl.stream_ctrl(H5, ens_needed, eff, set(real_guides),
                                        nt_cells=nt_cells)

    def table_from(gs, gc, nt, ei, inj_seed8=False):
        rows = []
        for r in cand.itertuples(index=False):
            v = cl.lfc(gs, gc, nt, ei, r.guide_id, r.offtarget_ensembl)
            if v is not None:
                rows.append((r.guide_id, r.offtarget_ensembl, int(r.seed_match_len), v))
        return pd.DataFrame(rows, columns=["guide_id", "offtarget_ensembl",
                                           "seed_match_len", "offtarget_log2fc"])

    base_tab = table_from(gsB, gcB, ntB, eiB)
    base_lfc = {(r.guide_id, r.offtarget_ensembl): r.offtarget_log2fc
                for r in base_tab.itertuples(index=False)}
    print(f"baseline table: {len(base_tab)} events")
    seed_tests(base_tab, "REAL observed (baseline)")

    seed8_pairs = cand[cand.seed8][["guide_id", "offtarget_ensembl"]].values
    results = []
    for delta in DELTAS:
        inject = {}
        for g, e in seed8_pairs:
            inject.setdefault(g, {})[e] = delta
        print(f"\n=== injected pass delta={delta} (into {len(seed8_pairs)} seed>=8 pairs) ===")
        gsI, gcI, ntI, eiI = cl.stream_ctrl(H5, ens_needed, eff, set(real_guides),
                                            inject=inject, ntm_lookup=ntm_lookup,
                                            nt_cells=nt_cells)
        # build mixed table: seed>=8 from injected, seed<8 from baseline
        rows = []
        recov = []
        for r in cand.itertuples(index=False):
            key = (r.guide_id, r.offtarget_ensembl)
            if r.seed8:
                v = cl.lfc(gsI, gcI, ntI, eiI, r.guide_id, r.offtarget_ensembl)
                if v is not None and key in base_lfc:
                    recov.append(v - base_lfc[key])
            else:
                v = base_lfc.get(key)
            if v is not None:
                rows.append((r.guide_id, r.offtarget_ensembl, int(r.seed_match_len), v))
        tab = pd.DataFrame(rows, columns=["guide_id", "offtarget_ensembl",
                                          "seed_match_len", "offtarget_log2fc"])
        res = seed_tests(tab, f"injected delta={delta}")
        # per-pair sensitivity: injected seed>=8 crossing +0.5
        inj_vals = tab[tab.seed_match_len >= 8].offtarget_log2fc
        base_rate = (base_tab[base_tab.seed_match_len >= 8].offtarget_log2fc >= 0.5).mean()
        sens = float((inj_vals >= 0.5).mean())
        med_recov = float(np.median(recov)) if recov else np.nan
        res.update(delta=delta, sensitivity=sens, base_rate=float(base_rate),
                   median_recovered=med_recov)
        print(f"  per-pair: {sens*100:.0f}% of injected seed>=8 cross +0.5 "
              f"(baseline {base_rate*100:.0f}%); median recovered lfc {med_recov:+.2f}")
        results.append(res)

    out = pd.DataFrame(results)
    out.to_csv(ROOT / "outputs/metrics/controls_spikein.csv", index=False)
    print("\nwrote outputs/metrics/controls_spikein.csv")
    print("EXPECT: seed-perm p drops below 0.05 for delta >= some X; that X (and the")
    print("per-pair sensitivity curve) IS the detection floor. Real observed p=0.65.")


if __name__ == "__main__":
    main()
