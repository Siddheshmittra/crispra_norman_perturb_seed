#!/usr/bin/env python3
"""Keystone for the CRISPRa off-target controls battery.

One backed pass over the single-cell h5ad (paper recipe: expm1 -> full-library
CP10k -> log1p) that supports three transforms, so spike-in / depth / NTC-null /
scramble all reuse the SAME validated pseudobulk machinery as the real analysis:

  inject      : multiply a gene's COUNT by 2^delta in the cells of a chosen guide
                (per-cell synthetic off-target, applied before CP10k so noise +
                library structure are preserved).
  override    : replace each cell's effective guide label (NTC-vs-NTC, scramble).
  downsample  : keep at most n cells per (effective) guide.

The effect measure is identical to seed_dissociation_lib.lfc:
    log2((mean_log1pCP10k(guide) + 0.01) / (mean_log1pCP10k(NTC) + 0.01)).
So a "recovered" injected effect is reported in the pipeline's own units, and is
calibrated empirically against the injected delta (see validate_injection()).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]


def read_obs(h5_path: Path, guide_col: str = "guide_identity",
             target_col: str | None = "guide_target",
             var_id_col: str | None = None):
    """Fast backed read of the guide labels + the gene id index.

    Defaults reproduce the validated CRISPRa path (guide_identity / guide_target,
    ensembl in var.index). For other atlases (e.g. Replogle GWPS: guide_col=
    'sgID_AB', no target col, ensembl in a var column) pass overrides.

    Returns (gi, gt, var_index). gt is None if target_col is None/absent.
    var_index is a.var[var_id_col] when var_id_col is given, else a.var.index.
    """
    a = ad.read_h5ad(h5_path, backed="r")
    gi = a.obs[guide_col].astype(str).values
    if target_col is not None and target_col in a.obs.columns:
        gt = a.obs[target_col].astype(str).values
    else:
        gt = None
    if var_id_col is not None and var_id_col in a.var.columns:
        var_index = list(a.var[var_id_col].astype(str).values)
    else:
        var_index = list(a.var.index.astype(str))
    a.file.close()
    return gi, gt, var_index


def build_downsample_mask(eff_guide: np.ndarray, keep: set,
                          downsample, seed: int = 0) -> np.ndarray:
    """Boolean mask over cells: keep <= n cells for each guide in `keep`.

    downsample: None (keep all), int (same n for every guide), or dict guide->n.
    Deterministic given seed.
    """
    if downsample is None:
        return np.isin(eff_guide, list(keep))
    rng = np.random.RandomState(seed)
    mask = np.zeros(len(eff_guide), bool)
    order = rng.permutation(len(eff_guide))          # shuffle so the kept subset is random
    seen: dict = {}
    for i in order:
        g = eff_guide[i]
        if g not in keep:
            continue
        n = downsample.get(g) if isinstance(downsample, dict) else downsample
        if n is None:
            mask[i] = True
            continue
        c = seen.get(g, 0)
        if c < n:
            mask[i] = True
            seen[g] = c + 1
    return mask


def stream_ctrl(h5_path: Path, ens_needed: list[str],
                eff_guide: np.ndarray, guide_set: set,
                inject: dict | None = None, ntm_lookup: dict | None = None,
                downsample=None, nt_cells: np.ndarray | None = None,
                chunk: int = 8000, seed: int = 0, var_id_col: str | None = None,
                log1p_input: bool = True):
    """One backed pass returning per-guide pseudobulk over ens_needed.

    Parameters
    ----------
    eff_guide : per-cell effective guide label (len == n_cells). For the real
                analysis pass obs['guide_identity']; for NTC-null / scramble pass
                a relabelled array.
    guide_set : the effective guide labels to accumulate.
    inject    : {guide -> {ensembl_gene: delta_log2fc}} ADDITIVE activation in
                log1p-CP10k space: each perturbed cell's gene value is bumped by
                c = (ntm_gene + 0.01)*(2^delta - 1), so the recovered pseudobulk
                lfc ~= delta even for a SILENT off-target gene (CRISPRa switches
                genes on; multiplicative injection cannot raise a 0-count gene).
                Requires ntm_lookup {ens: ntm_log1pCP10k} (e.g. from the NTC csv).
    downsample: None | int | {guide: n}.
    nt_cells  : optional boolean mask selecting the NTC baseline cells.

    Returns gsum {guide->vec}, gcnt {guide->int}, ntm vec, ens_idx.
    """
    a = ad.read_h5ad(h5_path, backed="r")
    n_cells = a.shape[0]
    if var_id_col is not None and var_id_col in a.var.columns:
        var_ids = a.var[var_id_col].astype(str).values
    else:
        var_ids = a.var.index.astype(str)
    var_pos = {e: i for i, e in enumerate(var_ids)}

    ens_needed = [e for e in ens_needed if e in var_pos]
    keep_cols = np.array([var_pos[e] for e in ens_needed], dtype=int)
    ens_idx = {e: i for i, e in enumerate(ens_needed)}
    nE = len(ens_needed)
    ntm_lookup = ntm_lookup or {}

    # injected genes -> (LOCAL col in keep_cols, additive bump c) per guide
    inj_local: dict = {}
    if inject:
        for g, genes in inject.items():
            for ens, delta in genes.items():
                if ens in ens_idx:
                    base = float(ntm_lookup.get(ens, 0.0))
                    c = (base + 0.01) * (2.0 ** float(delta) - 1.0)
                    inj_local.setdefault(g, []).append((ens_idx[ens], c))

    if nt_cells is None:
        nt_cells = np.zeros(n_cells, bool)

    keep = set(guide_set)
    cell_mask = build_downsample_mask(eff_guide, keep, downsample, seed)
    idx = np.where(cell_mask | nt_cells)[0]
    print(f"  streaming {len(idx):,} cells ({len(keep)} guides + "
          f"{int(nt_cells.sum())} NTC baseline); {nE} genes"
          + (f"; injecting {len(inj_local)} guides" if inj_local else ""), flush=True)

    gsum: dict = {}
    gcnt: dict = {}
    ntc = np.zeros(nE, np.float64)
    ntn = 0
    for lo in range(0, len(idx), chunk):
        ci = np.sort(idx[lo:lo + chunk])
        X = a.X[ci, :]
        X = X.toarray() if sp.issparse(X) else np.asarray(X)
        X = np.asarray(X, np.float32)
        if log1p_input:
            np.expm1(X, out=X)                     # log1p(CP10k) -> recovered counts
        # else: X is already raw counts (e.g. Replogle GWPS) -> normalize directly
        rs = X.sum(1, keepdims=True)               # full-library CP10k denominator
        np.maximum(rs, 1e-10, out=rs)
        X *= 1e4 / rs
        X = X[:, keep_cols]                         # subset AFTER normalization
        np.log1p(X, out=X)
        cg = eff_guide[ci]
        # ---- additive CRISPRa-style injection in log1p-CP10k space ----
        if inj_local:
            for li, g in enumerate(cg):
                if g in inj_local:
                    for lcol, c in inj_local[g]:
                        X[li, lcol] += c
        kept = cell_mask[ci]
        for g in np.unique(cg[kept]):
            m = (cg == g) & kept
            gsum[g] = gsum.get(g, 0) + X[m].sum(0)
            gcnt[g] = gcnt.get(g, 0) + int(m.sum())
        ntm_chunk = nt_cells[ci]
        if ntm_chunk.any():
            ntc += X[ntm_chunk].sum(0)
            ntn += int(ntm_chunk.sum())
    a.file.close()
    ntm = ntc / max(ntn, 1)
    return gsum, gcnt, ntm, ens_idx


def lfc(gsum, gcnt, ntm, ens_idx, guide: str, ens: str, min_cells: int = 20):
    """Pipeline-identical pseudobulk log2FC; None if <min_cells or gene absent."""
    if guide not in gcnt or gcnt[guide] < min_cells or ens not in ens_idx:
        return None
    j = ens_idx[ens]
    return float(np.log2((gsum[guide][j] / gcnt[guide] + 0.01) / (ntm[j] + 0.01)))
