#!/usr/bin/env python3
"""Shared machinery for the guide-level seed-dissociation control.

The dissociation logic (Hartman et al. 2026's own gold-standard control): a
nominated off-target (gene A's guide activates gene B via a promoter seed match)
is a GENUINE DIRECT seed off-target only if B is activated by A's seed-matching
guide(s) but NOT by A's *other* guides that lack the B seed match -- even though
those sibling guides DO activate A. If every A-effective guide activates B
regardless of seed, B is a TRANS/downstream consequence of activating A, not a
direct off-target.

This module provides: promoter sequence fetch from hg38, seed recompute (reusing
the exact vendor rule that built the candidate table), and the one-pass backed
h5ad streamer that yields per-guide pseudobulk log2FC (paper recipe) PLUS the
per-cell values of every off-target (B) gene needed for the Mann-Whitney test.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# self-contained seed primitives (verbatim copy of the vendor rule; no scanpy dep)
from seed_match_core import (  # noqa: E402
    find_max_seed_match, load_fai, extract_genome_region,
)

GENOME = ROOT / "data/ref/hg38.fa"
FAI = ROOT / "data/ref/hg38.fa.fai"
TSS_CSV = ROOT / "vendor/perturb_seed/tss_map.csv"


def load_tss(path: Path = TSS_CSV) -> dict:
    """gene_name -> (chrom, tss0, strand); tss0 is 0-based."""
    df = pd.read_csv(path).drop_duplicates("gene_name")
    out = {}
    for r in df.itertuples(index=False):
        strand = getattr(r, "strand", "+")
        out[r.gene_name] = (r.chrom, int(r.tss) - 1, strand)
    return out


class PromoterFetcher:
    """Fetch +/-half bp of genomic sequence around a gene's TSS (cached)."""

    def __init__(self, half: int = 2000):
        self.half = half
        self.fai = load_fai(FAI)
        self.tss = load_tss()
        self._cache: dict = {}

    def seq(self, gene: str):
        if gene in self._cache:
            return self._cache[gene]
        s = None
        if gene in self.tss:
            chrom, tss0, _ = self.tss[gene]
            s = extract_genome_region(self.fai, chrom, tss0 - self.half,
                                      tss0 + self.half, GENOME) or None
        self._cache[gene] = s
        return s

    def tss_dist(self, gene: str, seed_match_start: float) -> float:
        """Distance of a genomic seed-match coord to the gene's TSS (bp)."""
        if gene not in self.tss or pd.isna(seed_match_start):
            return np.inf
        _, tss0, _ = self.tss[gene]
        return abs(float(seed_match_start) - tss0)


def recompute_seed_len(spacer: str, prom_seq: str | None) -> int:
    """Longest PAM-proximal seed match of `spacer` anywhere in `prom_seq`.

    Uses the SAME rule (find_max_seed_match) that built the candidate table, so
    SEED-MATCH / NO-MATCH calls are method-consistent with the nominations.
    Returns 0 if no promoter or no match.
    """
    if not prom_seq or not isinstance(spacer, str) or len(spacer) < 5:
        return 0
    best, *_ = find_max_seed_match(spacer, prom_seq, min_len=5)
    return int(best)


def stream_pseudobulk(h5_path: Path, guide_set: set[str], ens_needed: list[str],
                      bcols_ens: list[str], chunk: int = 8000,
                      nt_labels=("non",)):
    """One backed pass over the h5ad (paper recipe: expm1 -> CP10k -> log1p).

    Returns
    -------
    gsum  : {guide -> np.ndarray over ens_needed}   (summed log1p-CP10k)
    gcnt  : {guide -> int cell count}
    ntm   : np.ndarray over ens_needed             (pooled-NTC mean log1p-CP10k)
    percell : {guide -> np.ndarray [n_cells, len(bcols_ens)]}  per-cell B values
    ens_idx : {ensembl -> local col index in the ens_needed-ordered vectors}
    bcol_idx: {ensembl -> local col index within the percell matrices}
    """
    a = ad.read_h5ad(h5_path, backed="r")
    gi = a.obs["guide_identity"].astype(str).values
    gt = a.obs["guide_target"].astype(str).values
    var_pos = {e: i for i, e in enumerate(a.var.index)}

    ens_needed = [e for e in ens_needed if e in var_pos]
    bcols_ens = [e for e in bcols_ens if e in var_pos]
    keep_cols = np.array([var_pos[e] for e in ens_needed], dtype=int)
    ens_idx = {e: i for i, e in enumerate(ens_needed)}
    # column positions (within the ens_needed-sliced matrix) of the B genes
    bcol_local = np.array([ens_idx[e] for e in bcols_ens], dtype=int)
    bcol_idx = {e: i for i, e in enumerate(bcols_ens)}

    ntset = {s.lower() for s in nt_labels}
    is_nt = np.array([s.lower() in ntset for s in gt])
    need = np.isin(gi, list(guide_set)) | is_nt
    idx = np.where(need)[0]
    print(f"  streaming {len(idx):,} cells "
          f"({len(guide_set)} guides + {(gt=='non').sum()} NTC); "
          f"{len(ens_needed)} genes, {len(bcols_ens)} B-cols", flush=True)

    nE = len(ens_needed)
    gsum: dict = {}
    gcnt: dict = {}
    pc_chunks: dict = {}   # guide -> list of per-chunk B arrays
    ntc = np.zeros(nE, np.float64)
    ntn = 0
    for lo in range(0, len(idx), chunk):
        ci = np.sort(idx[lo:lo + chunk])
        X = a.X[ci, :]
        X = X.toarray() if sp.issparse(X) else np.asarray(X)
        X = np.asarray(X, np.float32)
        np.expm1(X, out=X)                       # data is log1p -> recover counts
        rs = X.sum(1, keepdims=True)             # FULL-library size = correct CP10k denominator
        np.maximum(rs, 1e-10, out=rs)            # (was computed on the gene subset -> bug)
        X *= 1e4 / rs
        X = X[:, keep_cols]                       # subset AFTER normalization
        np.log1p(X, out=X)
        cg = gi[ci]
        ct = gt[ci]
        for g in np.unique(cg):
            if g not in guide_set:
                continue
            m = cg == g
            gsum[g] = gsum.get(g, 0) + X[m].sum(0)
            gcnt[g] = gcnt.get(g, 0) + int(m.sum())
            pc_chunks.setdefault(g, []).append(X[np.ix_(m, bcol_local)].copy())
        nm = np.array([s.lower() in ntset for s in ct])
        if nm.any():
            ntc += X[nm].sum(0)
            ntn += int(nm.sum())
    a.file.close()

    ntm = ntc / max(ntn, 1)
    percell = {g: np.vstack(chunks) for g, chunks in pc_chunks.items()}
    return gsum, gcnt, ntm, percell, ens_idx, bcol_idx


def lfc(gsum, gcnt, ntm, ens_idx, guide: str, ens: str, min_cells: int = 20):
    """Pseudobulk log2FC of `ens` under `guide` vs pooled NTC; None if <min_cells."""
    if guide not in gcnt or gcnt[guide] < min_cells or ens not in ens_idx:
        return None
    j = ens_idx[ens]
    return float(np.log2((gsum[guide][j] / gcnt[guide] + 0.01) / (ntm[j] + 0.01)))
