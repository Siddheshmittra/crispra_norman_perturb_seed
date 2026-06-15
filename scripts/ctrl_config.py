#!/usr/bin/env python3
"""Path resolver for the controls battery, keyed by the CELL env var so the same
scripts run on Hs27 (local) and RPE1 (cloud replication)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CELL = os.environ.get("CELL", "hs27").lower()

_CFG = {
    "hs27": dict(
        h5="data/raw/fibroblast_CRISPRa_final_pop_singlets_normalized_log1p.h5ad",
        cand="outputs/candidates/norman_crispra_singlecell_off_target_candidates.csv",
        ntc="outputs/metrics/norman_crispra_ntc_gene_expression.csv",
        tag="hs27",
    ),
    "rpe1": dict(
        h5="data/raw_rpe1/rpe1_CRISPRa_final_pop_singlets_normalized_log1p.h5ad",
        cand="outputs/candidates/rpe1_crispra_singlecell_off_target_candidates.csv",
        ntc="outputs/metrics/rpe1_crispra_ntc_gene_expression.csv",
        tag="rpe1",
    ),
}

c = _CFG[CELL]
H5 = ROOT / os.environ.get("H5_OVERRIDE", c["h5"])
CAND = ROOT / c["cand"]
NTC = ROOT / c["ntc"]
TAG = c["tag"]
TSS = ROOT / "vendor/perturb_seed/tss_map.csv"
METRICS = ROOT / "outputs/metrics"


def out(name):
    return METRICS / f"controls_{name}_{TAG}.csv"
