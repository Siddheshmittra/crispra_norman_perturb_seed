#!/usr/bin/env python3
"""P4 — the strongest positive control: a REAL CRISPRi seed off-target effect,
measured at full depth and then downsampled to Norman's ~22 cells/guide.

Replogle K562 GWPS has a known, strong seed->repression dose-response (paper +
our Phase 0 reconfirm: seed-perm p ~ 3e-4). We recompute it from the per-cell
h5ad with the SAME pseudobulk machinery as the CRISPRa battery (controls_lib),
then subsample to 22 cells/guide and show the seed-perm signal COLLAPSES toward
background. That proves two things at once:
  (i)  a real seed effect is genuinely invisible at Norman's depth, so
  (ii) Norman CRISPRa's flatness is a depth limit, not a pipeline artifact.

CRISPRi off-targets are repression (log2fc < 0). We reuse the CRISPRa-oriented
seed_tests() by passing the SIGN-FLIPPED effect (-log2fc), so "realized
repression >= 0.5" maps onto the same >= ACT test and the seed-perm / within-
guide logic carries over unchanged.

Run on the cloud VM:
  H5=/data/replogle_k562/K562_gwps_raw_singlecell_01.h5ad \
  CAND=outputs/candidates/replogle_k562_candidate_level_table.csv \
  python src/ctrl_crispri_downsample.py
"""
from pathlib import Path
import os
import numpy as np
import pandas as pd
import anndata as ad
import controls_lib as cl
from ctrl_negative import seed_tests, ACT

ROOT = Path(__file__).resolve().parents[1]
H5 = Path(os.environ.get("H5", "/home/azureuser/data/replogle_k562/"
                               "K562_gwps_raw_singlecell_01.h5ad"))
CAND = Path(os.environ.get(
    "CAND", ROOT / "Step 0- pervasiveness/phase0_perturb_seed/outputs/"
                   "candidates/replogle_k562_candidate_level_table.csv"))
OUT = ROOT / "outputs/metrics/controls_crispri_downsample.csv"
GUIDE_COL = os.environ.get("GUIDE_COL", "sgID_AB")
# Replogle GWPS: per-cell guide labels live in sgID_AB ("non-targeting_xxxx|..."),
# but the targeted-gene column ('gene') flags NTC cells cleanly as "non-targeting".
NT_TARGET_COL = os.environ.get("NT_TARGET_COL", "gene")
NT_LABEL = os.environ.get("NT_LABEL", "non-targeting")
LOG1P_INPUT = os.environ.get("LOG1P_INPUT", "0") == "1"   # GWPS X = raw counts
DEPTHS = [None, 200, 100, 50, 22]          # None = full depth
SEED = 0
MIN_GUIDE_CELLS = 20


def detect_var_id_col(h5_path, ensg_sample):
    """Find which var field carries the ENSG ids used by the candidate table."""
    a = ad.read_h5ad(h5_path, backed="r")
    ensg = set(ensg_sample)
    cands = {"__index__": set(a.var.index.astype(str))}
    for c in a.var.columns:
        try:
            cands[c] = set(a.var[c].astype(str).values)
        except Exception:
            pass
    a.file.close()
    best, best_ov = None, -1
    for name, vals in cands.items():
        ov = len(ensg & vals)
        if ov > best_ov:
            best, best_ov = name, ov
    print(f"  var id overlap: {best} matches {best_ov}/{len(ensg)} candidate ENSGs")
    return None if best == "__index__" else best


def load_candidates():
    df = pd.read_csv(CAND, low_memory=False)
    # canonicalise to the battery's column names
    df = df.rename(columns={"candidate_gene_id": "offtarget_ensembl",
                            "seed_length": "seed_match_len"})
    df["seed_match_len"] = pd.to_numeric(df.seed_match_len, errors="coerce")
    if "is_on_target" in df.columns:
        df = df[df.is_on_target.fillna(0).astype(float) == 0]
    df = df.dropna(subset=["seed_match_len", "offtarget_ensembl", "guide_id"])
    df = df[df.offtarget_ensembl.astype(str).str.startswith("ENSG")]
    return df[["guide_id", "offtarget_ensembl", "seed_match_len"]].drop_duplicates()


def main():
    print(f"=== P4 CRISPRi downsample | H5={H5} ===")
    cand = load_candidates()
    print(f"candidate pairs={len(cand)} guides={cand.guide_id.nunique()} "
          f"seed>=8 pairs={int((cand.seed_match_len>=8).sum())}")

    ens_sample = cand.offtarget_ensembl.unique()[:2000]
    var_id_col = detect_var_id_col(H5, ens_sample)

    gi, gt, var_index = cl.read_obs(H5, guide_col=GUIDE_COL,
                                    target_col=NT_TARGET_COL, var_id_col=var_id_col)
    var_set = set(var_index)
    cand = cand[cand.offtarget_ensembl.isin(var_set)].copy()
    obs_guides = set(np.unique(gi))
    cand = cand[cand.guide_id.isin(obs_guides)].copy()
    # NTC baseline = cells whose targeted-gene column is the non-targeting label
    nt_cells = (gt == NT_LABEL) if gt is not None else (gi == NT_LABEL)
    print(f"cells={len(gi)} NTC={int(nt_cells.sum())} | candidates in h5ad/obs: "
          f"pairs={len(cand)} guides={cand.guide_id.nunique()} "
          f"seed>=8={int((cand.seed_match_len>=8).sum())}")
    if cand.guide_id.nunique() < 50:
        print("WARNING: few guides overlap the h5ad — check guide_id format vs "
              f"{GUIDE_COL}; sample obs guides:", list(obs_guides)[:3])

    guide_set = set(cand.guide_id.unique())
    ens = sorted(cand.offtarget_ensembl.unique())

    rows = []
    for depth in DEPTHS:
        tag = "full" if depth is None else f"{depth}cells"
        gs, gc, ntm, ei = cl.stream_ctrl(
            H5, ens, gi, guide_set, downsample=depth, nt_cells=nt_cells,
            seed=SEED, var_id_col=var_id_col, log1p_input=LOG1P_INPUT)
        rec = []
        for r in cand.itertuples(index=False):
            v = cl.lfc(gs, gc, ntm, ei, r.guide_id, r.offtarget_ensembl,
                       min_cells=MIN_GUIDE_CELLS)
            if v is not None:
                # CRISPRi: repression is negative log2fc -> flip so the
                # activation-oriented seed_tests measures repression strength.
                rec.append((r.guide_id, r.offtarget_ensembl,
                            int(r.seed_match_len), -v))
        tab = pd.DataFrame(rec, columns=["guide_id", "offtarget_ensembl",
                                         "seed_match_len", "offtarget_log2fc"])
        n_guides = tab.guide_id.nunique()
        med_cells = (pd.Series(gc).reindex(guide_set).dropna().median())
        res = seed_tests(tab, f"CRISPRi {tag}")
        res.update(depth=("full" if depth is None else depth),
                   n_guides_measured=int(n_guides),
                   median_cells_per_guide=float(med_cells))
        rows.append(res)

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")
    print("EXPECT: 'full' seed-perm p tiny (~1e-3 or smaller); collapses toward "
          "1.0 as depth -> 22, mirroring Norman CRISPRa's null at 22 cells/guide.")


if __name__ == "__main__":
    main()
