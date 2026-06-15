#!/usr/bin/env python3
"""Reproduce the Norman notebook CRISPRa fitness-screen seed regression gate.

The upstream notebook reads Gilbert et al. Table S3
`NIHMS630425-supplement-10.xlsx`, sheet `CRISPRa Library`, constructs gene and
5-mer seed design matrices, and regresses gamma/rho phenotypes on those terms.

This script is intentionally a hard gate. If the XLSX is unavailable or is an
HTML challenge page, it writes a failing anchor artifact and exits non-zero.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr


REQUIRED_COLUMNS = [
    "sgRNA ID",
    "Protospacer sequence",
    "Growth phenotype (gamma)",
    "CTx-DTA phenotype (rho)",
]

EXPECTED_NOTEBOOK_ANCHORS = {
    "n_guides_filtered": 198_095,
    "n_design_cols": 16_806,
    "n_seed_terms": 1_023,
    "gamma_top_negative_seeds": ["AGGAG", "GGGGA", "GGAGG", "AGGGG", "GGGGG", "GAGAG"],
    "gamma_top_positive_seeds": ["TCGGG", "CCGGG", "ACGGG", "TGGGG", "GCGGG", "GTGGG"],
    "top_abs_gamma_seed": "AGGAG",
    "top_abs_gamma_coef_min": -0.033,
    "top_abs_gamma_coef_max": -0.030,
}


def detect_file_type(path: Path) -> str:
    if not path.exists():
        return "missing"
    if zipfile.is_zipfile(path):
        return "xlsx"
    head = path.read_bytes()[:100_000].decode("utf-8", errors="ignore").lower()
    if "recaptcha" in head or "checking your browser" in head:
        return "html_recaptcha"
    if head.lstrip().startswith("<!doctype html") or head.lstrip().startswith("<html"):
        return "html"
    return "unknown"


def write_gate_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["anchor", "expected", "observed", "status", "notes"]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def pass_fail(condition: bool) -> str:
    return "PASS" if condition else "FAIL"


def load_crispra_table(xlsx_path: Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, skiprows=1, sheet_name="CRISPRa Library")
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    df = df.copy()
    df["sgRNA ID"] = df["sgRNA ID"].astype(str)
    df = df[df["Protospacer sequence"].map(lambda x: isinstance(x, str) and len(x) >= 20)]
    df["gene"] = df["sgRNA ID"].map(lambda x: x.split("-")[0])
    df["sequence"] = df["Protospacer sequence"].map(lambda x: x[-8:-3].upper())
    df = df[
        ~df["Protospacer sequence"].str.contains("N", na=False)
        & ~df["gene"].str.contains("2014", na=False)
    ]
    guide_counts = df.groupby("gene")["sgRNA ID"].transform("count")
    df = df[guide_counts >= 10].reset_index(drop=True)
    return df


def fit_sparse_seed_regression(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    y = df[["Growth phenotype (gamma)", "CTx-DTA phenotype (rho)"]].astype(np.float64)
    y.columns = ["gamma", "rho"]

    genes = pd.Index(sorted(df["gene"].unique()))
    if "negative_control" in genes:
        genes = genes.drop("negative_control")
    gene_to_col = {gene: i for i, gene in enumerate(genes)}

    seed_counts = df["sequence"].value_counts()
    seeds = pd.Index(sorted(seed_counts[seed_counts >= 20].index))
    seed_to_col = {seed: len(genes) + i for i, seed in enumerate(seeds)}
    intercept_col = len(genes) + len(seeds)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for row_idx, (gene, seed) in enumerate(zip(df["gene"], df["sequence"], strict=False)):
        if gene in gene_to_col:
            rows.append(row_idx)
            cols.append(gene_to_col[gene])
            data.append(1.0)
        if seed in seed_to_col:
            rows.append(row_idx)
            cols.append(seed_to_col[seed])
            data.append(1.0)
        rows.append(row_idx)
        cols.append(intercept_col)
        data.append(1.0)

    design = sparse.csr_matrix(
        (np.array(data, dtype=np.float64), (rows, cols)),
        shape=(len(df), intercept_col + 1),
    )

    records: list[dict[str, object]] = []
    for phenotype in ["gamma", "rho"]:
        result = lsqr(design, y[phenotype].to_numpy(), atol=1e-10, btol=1e-10, iter_lim=20_000)
        coef = result[0]
        for seed, col in seed_to_col.items():
            records.append(
                {
                    "term": f"{seed}_s",
                    "seed": seed,
                    "phenotype": phenotype,
                    "coef": float(coef[col]),
                }
            )

    stats = {
        "n_guides_filtered": int(len(df)),
        "n_gene_terms": int(len(genes)),
        "n_seed_terms": int(len(seeds)),
        "n_design_cols": int(design.shape[1]),
        "n_design_rows": int(design.shape[0]),
    }
    return pd.DataFrame.from_records(records), stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xlsx", type=Path, default=Path("data/raw/NIHMS630425-supplement-10.xlsx"))
    parser.add_argument(
        "--out-anchors",
        type=Path,
        default=Path("outputs/metrics/norman_crispra_reproduce_anchors.csv"),
    )
    parser.add_argument(
        "--out-seed-coefs",
        type=Path,
        default=Path("outputs/metrics/norman_crispra_fitness_seed_coefficients.csv"),
    )
    parser.add_argument("--expected-seed", default=None)
    parser.add_argument("--expected-gamma-min", type=float, default=None)
    parser.add_argument("--expected-gamma-max", type=float, default=None)
    args = parser.parse_args()

    file_type = detect_file_type(args.xlsx)
    if file_type != "xlsx":
        write_gate_rows(
            args.out_anchors,
            [
                {
                    "anchor": "fitness_screen_seed_regression_input",
                    "expected": "valid NIHMS630425-supplement-10.xlsx workbook",
                    "observed": file_type,
                    "status": "FAIL",
                    "notes": f"{args.xlsx} is required before downstream CRISPRa perturb_seed steps can run.",
                },
                {
                    "anchor": "downstream_gate",
                    "expected": "PASS",
                    "observed": "blocked before reproduction",
                    "status": "FAIL",
                    "notes": "Step 0 hard gate was not satisfied.",
                },
            ],
        )
        return 2

    df = load_crispra_table(args.xlsx)
    seed_coefs, stats = fit_sparse_seed_regression(df)
    args.out_seed_coefs.parent.mkdir(parents=True, exist_ok=True)
    seed_coefs.to_csv(args.out_seed_coefs, index=False)

    gamma = seed_coefs[seed_coefs["phenotype"].eq("gamma")].copy()
    top_abs = gamma.iloc[gamma["coef"].abs().to_numpy().argmax()]

    rows = [
        {
            "anchor": "fitness_screen_seed_regression_input",
            "expected": "valid NIHMS630425-supplement-10.xlsx workbook",
            "observed": "xlsx",
            "status": "PASS",
            "notes": str(args.xlsx),
        }
    ]

    if args.expected_seed is not None:
        observed_coef = gamma.loc[gamma["seed"].eq(args.expected_seed), "coef"]
        if observed_coef.empty:
            status = "FAIL"
            observed = f"seed {args.expected_seed!r} absent"
        else:
            coef = float(observed_coef.iloc[0])
            lo = args.expected_gamma_min if args.expected_gamma_min is not None else -np.inf
            hi = args.expected_gamma_max if args.expected_gamma_max is not None else np.inf
            status = "PASS" if lo <= coef <= hi else "FAIL"
            observed = str(coef)
        expected = f"{args.expected_seed} gamma in [{args.expected_gamma_min}, {args.expected_gamma_max}]"
        rows.append(
            {
                "anchor": "configured_seed_gamma_effect",
                "expected": expected,
                "observed": observed,
                "status": status,
                "notes": f"seed_coefficients={args.out_seed_coefs}",
            }
        )
    else:
        expected = EXPECTED_NOTEBOOK_ANCHORS
        neg_seeds = gamma.sort_values("coef").head(6)["seed"].tolist()
        pos_seeds = gamma.sort_values("coef", ascending=False).head(6)["seed"].tolist()
        dim_status = pass_fail(
            stats["n_guides_filtered"] == expected["n_guides_filtered"]
            and stats["n_design_cols"] == expected["n_design_cols"]
            and stats["n_seed_terms"] == expected["n_seed_terms"]
        )
        neg_status = pass_fail(neg_seeds == expected["gamma_top_negative_seeds"])
        pos_status = pass_fail(pos_seeds == expected["gamma_top_positive_seeds"])
        top_abs_status = pass_fail(
            top_abs["seed"] == expected["top_abs_gamma_seed"]
            and expected["top_abs_gamma_coef_min"] <= float(top_abs["coef"]) <= expected["top_abs_gamma_coef_max"]
        )
        rows.extend(
            [
                {
                    "anchor": "notebook_design_dimensions",
                    "expected": json.dumps(
                        {
                            "n_guides_filtered": expected["n_guides_filtered"],
                            "n_design_cols": expected["n_design_cols"],
                            "n_seed_terms": expected["n_seed_terms"],
                        },
                        sort_keys=True,
                    ),
                    "observed": json.dumps(
                        {
                            "n_guides_filtered": stats["n_guides_filtered"],
                            "n_design_cols": stats["n_design_cols"],
                            "n_seed_terms": stats["n_seed_terms"],
                        },
                        sort_keys=True,
                    ),
                    "status": dim_status,
                    "notes": "Matches the authors' CRISPRa notebook setup.",
                },
                {
                    "anchor": "crispra_gamma_top_negative_seed_outliers",
                    "expected": ",".join(expected["gamma_top_negative_seeds"]),
                    "observed": ",".join(neg_seeds),
                    "status": neg_status,
                    "notes": "Six negative seed labels annotated in the embedded CRISPRa seed phenotype plot.",
                },
                {
                    "anchor": "crispra_gamma_top_positive_seed_outliers",
                    "expected": ",".join(expected["gamma_top_positive_seeds"]),
                    "observed": ",".join(pos_seeds),
                    "status": pos_status,
                    "notes": "Six positive seed labels annotated in the embedded CRISPRa seed phenotype plot.",
                },
                {
                    "anchor": "crispra_gamma_top_abs_seed_effect",
                    "expected": (
                        f"{expected['top_abs_gamma_seed']} gamma in "
                        f"[{expected['top_abs_gamma_coef_min']}, {expected['top_abs_gamma_coef_max']}]"
                    ),
                    "observed": json.dumps(
                        {"seed": top_abs["seed"], "coef": float(top_abs["coef"])},
                        sort_keys=True,
                    ),
                    "status": top_abs_status,
                    "notes": f"seed_coefficients={args.out_seed_coefs}",
                },
            ]
        )
        status = pass_fail(all(row["status"] == "PASS" for row in rows))

    rows.append(
        {
            "anchor": "downstream_gate",
            "expected": "PASS",
            "observed": status,
            "status": "PASS" if status == "PASS" else "FAIL",
            "notes": "Downstream steps may run only when this row is PASS.",
        }
    )
    write_gate_rows(args.out_anchors, rows)
    return 0 if status == "PASS" else 3


if __name__ == "__main__":
    sys.exit(main())
