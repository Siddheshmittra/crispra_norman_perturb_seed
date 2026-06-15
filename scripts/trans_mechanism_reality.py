#!/usr/bin/env python3
"""Are the guide-dissociated A->B effects real, or noise? Split-half reproducibility test.

Split each TF's cells into two deterministic halves (even/odd index), compute the pooled A->B
log2FC INDEPENDENTLY in each half, and correlate half1 vs half2 across pair sets:
  - on-target A->A     POSITIVE CONTROL (real, strong) -> must reproduce (high r)
  - random A-B pairs   NOISE FLOOR                      -> r ~ 0
  - DIRECT pairs       the seed-specific class
  - TRANS pairs        the not-guide-specific class

If TRANS (and DIRECT) A->B effects sit at the random-pair noise floor while on-target reproduces,
the per-pair dissociation calls are not real effects. Reuses the trans_mechanism_de recipe.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
H5 = ROOT / "data/raw/fibroblast_CRISPRa_final_pop_singlets_normalized_log1p.h5ad"
NT, CHUNK, PC = "non", 4000, 0.01

# pairs (Hs27 only; this is the Hs27 atlas)
pp = pd.read_csv(ROOT / "outputs/metrics/seed_dissociation_hs27_per_pair_tier8.csv")
trans  = pp[pp.cls == "TRANS"][["A", "B"]].values.tolist()
direct = pp[pp.cls == "DIRECT"][["A", "B"]].values.tolist()

a = ad.read_h5ad(H5, backed="r")
gt = a.obs["guide_target"].astype(str).values
sym = a.var["gene_name"].astype(str).values
sym2j = {}; [sym2j.setdefault(s, j) for j, s in enumerate(sym)]
tfs = sorted(set(gt)); tf_code = {t: i for i, t in enumerate(tfs)}
codes = np.array([tf_code[t] for t in gt]); nT = len(tfs)

# columns we need: all candidate B + all A (for on-target) + a random gene panel
need_sym = set([b for _, b in trans + direct] + [aa for aa, _ in trans + direct])
rng = np.random.RandomState(0)
all_expr_idx = [sym2j[s] for s in sym if s in sym2j]
rand_cols = rng.choice(len(sym), 400, replace=False)
need_j = sorted(set([sym2j[s] for s in need_sym if s in sym2j]) | set(rand_cols.tolist()))
jpos = {j: i for i, j in enumerate(need_j)}
nGn = len(need_j)

# accumulate per-TF sum of log1p-CP10k over the two halves, for needed columns only
acc = [np.zeros((nT, nGn)), np.zeros((nT, nGn))]
cnt = [np.zeros(nT, np.int64), np.zeros(nT, np.int64)]
n = a.n_obs
need_j_arr = np.array(need_j)
for lo in range(0, n, CHUNK):
    hi = min(lo + CHUNK, n)
    X = a.X[lo:hi, :]; X = X.toarray() if sp.issparse(X) else np.asarray(X)
    X = np.asarray(X, np.float32); np.expm1(X, out=X)
    rs = X.sum(1, keepdims=True); np.maximum(rs, 1e-10, out=rs); X *= 1e4 / rs; np.log1p(X, out=X)
    Xn = X[:, need_j_arr]
    c = codes[lo:hi]; idx = np.arange(lo, hi)
    for h in (0, 1):
        m = (idx % 2) == h
        if not m.any(): continue
        cc = c[m]; S = sp.csr_matrix((np.ones(m.sum(), np.float32), (np.arange(m.sum()), cc)), shape=(m.sum(), nT))
        acc[h] += (S.T @ Xn[m]).astype(np.float64)
        cnt[h] += np.asarray(S.sum(0)).ravel().astype(np.int64)
a.file.close()

def lfc_half(h):
    mean = acc[h] / np.maximum(cnt[h][:, None], 1)
    ntm = mean[tf_code[NT]]
    return np.log2((mean + PC) / (ntm + PC))   # nT x nGn
L0, L1 = lfc_half(0), lfc_half(1)

def get(L, A, B):
    if A not in tf_code or B not in sym2j or sym2j[B] not in jpos: return np.nan
    return L[tf_code[A], jpos[sym2j[B]]]

def repro(pairs, label):
    x = np.array([get(L0, A, B) for A, B in pairs]); y = np.array([get(L1, A, B) for A, B in pairs])
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    r = stats.pearsonr(x, y)[0] if len(x) > 3 else np.nan
    print(f"  {label:22s} n={len(x):3d}  split-half r={r:+.2f}  |  mean|effect| h0={np.abs(x).mean():.2f}  "
          f"frac|>0.5| {np.mean(np.abs(x)>0.5)*100:.0f}%")
    return r

print("=== split-half reproducibility of pooled A->B log2FC ===")
ontgt = [(A, A) for A in sorted(set([a for a, _ in trans + direct])) if A in tf_code]
randp = [(tfs[rng.randint(nT)], sym[rand_cols[rng.randint(len(rand_cols))]]) for _ in range(200)]
repro(ontgt, "on-target A->A (POS)")
repro(direct, "DIRECT pairs")
repro(trans,  "TRANS pairs")
repro(randp,  "random A-B (NULL)")
print("\nInterpretation: if on-target r is high but TRANS r ~ random-pair r, the per-pair")
print("dissociation A->B effects are at the noise floor (not individually real).")
