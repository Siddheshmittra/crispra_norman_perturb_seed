#!/usr/bin/env python3
"""Validate controls_lib injection: spike a known delta into (guide, gene) pairs
and confirm the recovered pipeline-lfc tracks it. Runs on a small guide subset."""
from pathlib import Path
import numpy as np, pandas as pd
import controls_lib as cl

ROOT = Path(__file__).resolve().parents[1]
H5 = ROOT / "data/raw/fibroblast_CRISPRa_final_pop_singlets_normalized_log1p.h5ad"

gi, gt, var_index = cl.read_obs(H5)
var_set = set(var_index)
vc = pd.Series(gi[gt != "non"]).value_counts()
ok_guides = set(vc[vc >= 25].index)

cand = pd.read_csv(ROOT / "outputs/candidates/norman_crispra_singlecell_off_target_candidates.csv",
                   low_memory=False)
cand = cand[cand.guide_id.isin(ok_guides) & cand.offtarget_ensembl.isin(var_set)]
# one test pair per guide, take 24 distinct guides
pairs = cand.drop_duplicates("guide_id").head(24)[["guide_id", "offtarget_ensembl"]].values
deltas = [0.0, 0.5, 1.0, 2.0] * 6
test = [(g, e, deltas[i]) for i, (g, e) in enumerate(pairs)]

guide_set = set(p[0] for p in test)
ens_needed = sorted({p[1] for p in test})
nt_cells = (gt == "non")

ntc_csv = pd.read_csv(ROOT / "outputs/metrics/norman_crispra_ntc_gene_expression.csv")
ntm_lookup = dict(zip(ntc_csv.ensembl, ntc_csv.ntc_mean_log1p_cp10k))

print("=== pass A: baseline (no injection) ===")
gsumA, gcntA, ntmA, eiA = cl.stream_ctrl(H5, ens_needed, gi, guide_set,
                                         inject=None, nt_cells=nt_cells)
inject = {}
for g, e, d in test:
    if d > 0:
        inject.setdefault(g, {})[e] = d
print("=== pass B: injected ===")
gsumB, gcntB, ntmB, eiB = cl.stream_ctrl(H5, ens_needed, gi, guide_set,
                                         inject=inject, ntm_lookup=ntm_lookup,
                                         nt_cells=nt_cells)

print(f"\n{'guide':<22}{'n':>5}{'delta':>7}{'base_lfc':>10}{'inj_lfc':>9}{'recovered':>11}")
rows = []
for g, e, d in test:
    nb = gcntA.get(g, 0)
    la = cl.lfc(gsumA, gcntA, ntmA, eiA, g, e)
    lb = cl.lfc(gsumB, gcntB, ntmB, eiB, g, e)
    if la is None or lb is None:
        continue
    rec = lb - la
    rows.append((d, rec))
    print(f"{g[:22]:<22}{nb:>5}{d:>7.2f}{la:>10.3f}{lb:>9.3f}{rec:>11.3f}")

rows = np.array(rows)
print("\n=== recovered effect by injected delta ===")
for d in sorted(set(rows[:, 0])):
    r = rows[rows[:, 0] == d, 1]
    print(f"  delta={d:.2f}: mean recovered = {r.mean():+.3f} (n={len(r)}, "
          f"range {r.min():+.3f}..{r.max():+.3f})")
print("\nEXPECT: delta=0 -> ~0; recovered increases monotonically with delta.")
