#!/usr/bin/env python3
"""HEADLINE control: guide-level seed dissociation (direct vs trans off-targets).

For every nominated, realized off-target pair (gene A activates off-target gene B
via a >=T bp promoter seed match), test whether B activation tracks the SEED or
the on-target gene:

  DIRECT     : B activated by A's seed-match guide(s) AND by NO A-effective
               sibling guide that lacks the B seed match  -> genuine seed off-target
  TRANS      : >=1 A-effective NO-MATCH sibling also activates B -> B fires
               regardless of the seed -> downstream/trans consequence of A, not direct
  AMBIGUOUS  : structurally undissociable (no A-effective no-match sibling, or the
               seed-match guide is not on-target-effective, or coverage too low)

Controls: no-match siblings count only if on-target-effective (A log2FC>0.5, >=20
cells); seed status RECOMPUTED from hg38 within +/-2kb (a no-match sibling must
have <8bp seed even in the wider window); per-pair one-sided Mann-Whitney U on B's
per-cell expression (seed-match cells vs A-effective no-match cells), BH-FDR across
pairs; bootstrap 95% CIs on the DIRECT/TRANS/AMBIGUOUS fractions.

Usage: .venv/bin/python src/seed_dissociation.py [hs27|rpe1]
"""
from __future__ import annotations
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import anndata as ad
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from seed_dissociation_lib import (  # noqa: E402
    PromoterFetcher, recompute_seed_len, stream_pseudobulk, lfc,
)

CELL = sys.argv[1] if len(sys.argv) > 1 else "hs27"
CFG = {
    "hs27": dict(
        h5=ROOT / "data/raw/fibroblast_CRISPRa_final_pop_singlets_normalized_log1p.h5ad",
        cand=ROOT / "outputs/candidates/norman_crispra_singlecell_off_target_candidates.csv",
    ),
    "rpe1": dict(
        h5=ROOT / "data/raw_rpe1/rpe1_CRISPRa_final_pop_singlets_normalized_log1p.h5ad",
        cand=ROOT / "outputs/candidates/rpe1_crispra_singlecell_off_target_candidates.csv",
    ),
}[CELL]

TIERS = [8, 12]          # seed-length tiers to evaluate
EFF = 0.5                # log2FC activation threshold (paper recipe)
NOMATCH_MAX = 8          # a clean no-match sibling has recomputed seed < this (in +/-2kb)
RNG = np.random.default_rng(0)
NBOOT = 2000


def bh(pvals):
    """Benjamini-Hochberg q-values."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        r = n - rank
        prev = min(prev, p[i] * n / r)
        q[i] = prev
    return q


def main():
    print(f"=== seed dissociation [{CELL}] ===", flush=True)
    pf = PromoterFetcher(half=2000)

    # ---- obs maps: every gene's full guide set + each guide's spacer (this dataset) ----
    a = ad.read_h5ad(CFG["h5"], backed="r")
    cols = ["guide_identity", "guide_target"] + (["protospacer"] if "protospacer" in a.obs else [])
    obs = a.obs[cols].astype(str)
    if "protospacer" not in obs:
        obs["protospacer"] = obs["guide_identity"].str.split("_", n=1).str[-1]
    var_gene2ens = {}
    for ens, sym in zip(a.var.index, a.var["gene_name"].astype(str)):
        var_gene2ens.setdefault(sym, ens)   # symbol -> first ensembl
    a.file.close()
    # detect non-targeting control label(s) (Hs27="non"; others may differ)
    KNOWN_NT = {"non", "non-targeting", "nontargeting", "ntc", "control",
                "negative", "safe-harbor", "safe_harbor", "no-target"}
    present = {g.lower() for g in obs["guide_target"].unique()}
    NT_LABELS = sorted({g for g in obs["guide_target"].unique() if g.lower() in KNOWN_NT}) or ["non"]
    ntset = {s.lower() for s in NT_LABELS}
    print(f"  non-targeting label(s): {NT_LABELS}", flush=True)
    gene2guides: dict = {}
    guide2spacer: dict = {}
    for gid, gt, ps in obs.itertuples(index=False):
        if gt.lower() in ntset:
            continue
        gene2guides.setdefault(gt, set()).add(gid)
        if gid not in guide2spacer:
            sp = ps if ps and ps != "nan" and set(ps) <= set("ACGTN") else gid.split("_", 1)[-1]
            guide2spacer[gid] = sp.upper()
    print(f"  {len(gene2guides)} genes, {len(guide2spacer)} guides in obs", flush=True)

    # ---- nominated-realized pairs per tier ----
    d = pd.read_csv(CFG["cand"], low_memory=False)
    for c in ("offtarget_log2fc", "seed_match_len", "seed_match_start"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[d["gene"].astype(str).str.upper() != d["offtarget_gene"].astype(str).str.upper()].copy()
    d["td"] = [pf.tss_dist(g, s) for g, s in zip(d["offtarget_gene"].astype(str), d["seed_match_start"])]
    # realized off-target = >=8bp seed + activation + within 1kb of off-target TSS (paper definition)
    realized = d[(d.seed_match_len >= 8) & (d.offtarget_log2fc > EFF) & (d.td <= 1000)].copy()

    pairs = {}   # (A,B) -> dict
    for _, r in realized.iterrows():
        A, B = str(r["gene"]), str(r["offtarget_gene"])
        ens = str(r["offtarget_ensembl"])
        key = (A, B)
        p = pairs.setdefault(key, dict(A=A, B=B, ensB=ens, max_seed=0, sm_guides=set()))
        p["max_seed"] = max(p["max_seed"], int(r["seed_match_len"]))
        p["sm_guides"].add(str(r["guide_id"]))
    print(f"  {len(pairs)} unique realized (A,B) pairs", flush=True)

    # ---- enumerate + classify siblings (seed recompute vs B promoter) ----
    ens_needed, bcols = set(), set()
    for (A, B), p in pairs.items():
        ensB = p["ensB"] if p["ensB"] in set(var_gene2ens.values()) else var_gene2ens.get(B)
        p["ensB"] = ensB
        ensA = var_gene2ens.get(A)
        p["ensA"] = ensA
        if ensA:
            ens_needed.add(ensA)
        if ensB:
            ens_needed.add(ensB)
            bcols.add(ensB)
        prom = pf.seq(B)
        sibs = sorted(gene2guides.get(A, set()))
        p["siblings"] = []
        for g in sibs:
            slen = recompute_seed_len(guide2spacer.get(g, ""), prom)
            p["siblings"].append((g, slen))
        ens_needed.add(ensA) if ensA else None
    # guides to stream: every sibling of every pair's A
    guide_set = {g for p in pairs.values() for g, _ in p["siblings"]}
    guide_set &= set(guide2spacer)
    ens_needed = [e for e in ens_needed if e]
    bcols = [e for e in bcols if e]

    # ---- one streaming pass ----
    gsum, gcnt, ntm, percell, ens_idx, bcol_idx = stream_pseudobulk(
        CFG["h5"], guide_set, ens_needed, bcols, nt_labels=NT_LABELS)

    # ---- per-guide A/B log2FC ----
    def Albl(g, ensA):
        return lfc(gsum, gcnt, ntm, ens_idx, g, ensA)

    def Blbl(g, ensB):
        return lfc(gsum, gcnt, ntm, ens_idx, g, ensB)

    per_guide_rows, per_pair = [], {t: [] for t in TIERS}
    for (A, B), p in pairs.items():
        ensA, ensB = p["ensA"], p["ensB"]
        guide_stats = []
        for g, slen in p["siblings"]:
            al, bl = Albl(g, ensA), Blbl(g, ensB)
            n = gcnt.get(g, 0)
            a_eff = (al is not None and al > EFF and n >= 20)
            b_act = (bl is not None and bl > EFF)
            guide_stats.append(dict(guide=g, seed=slen, A_lfc=al, B_lfc=bl,
                                    n=n, a_eff=a_eff, b_act=b_act))
            per_guide_rows.append(dict(A=A, B=B, **guide_stats[-1]))
        # classify per tier. Paper's logic: B is realized by the nominated seed-match
        # guide; a pair is DIRECT iff A's on-target-EFFECTIVE no-match siblings do NOT
        # activate B (B requires the seed), TRANS iff >=1 such sibling activates B.
        for T in TIERS:
            if p["max_seed"] < T:        # pair only evaluated at tiers it is realized at
                continue
            sm = [s for s in guide_stats if s["seed"] >= T]                          # seed-match siblings
            nm_eff = [s for s in guide_stats if s["seed"] < NOMATCH_MAX and s["a_eff"]]  # on-target-effective no-match controls
            sm_act = [s for s in sm if s["b_act"]]
            nm_eff_act = [s for s in nm_eff if s["b_act"]]
            # statistical test: pooled cells of seed-match vs A-effective no-match siblings on B
            mwu_p = np.nan
            if ensB in bcol_idx and sm and nm_eff:
                bj = bcol_idx[ensB]
                xs = np.concatenate([percell[s["guide"]][:, bj] for s in sm if s["guide"] in percell]) if sm else np.array([])
                xn = np.concatenate([percell[s["guide"]][:, bj] for s in nm_eff if s["guide"] in percell]) if nm_eff else np.array([])
                if len(xs) >= 20 and len(xn) >= 20:
                    try:
                        mwu_p = float(mannwhitneyu(xs, xn, alternative="greater").pvalue)
                    except ValueError:
                        mwu_p = np.nan
            # classification
            if not nm_eff:
                cls = "AMBIGUOUS"
            elif len(nm_eff_act) == 0:
                cls = "DIRECT"
            else:
                cls = "TRANS"
            per_pair[T].append(dict(
                A=A, B=B, tier=T, max_seed=p["max_seed"],
                n_sm=len(sm), n_sm_act=len(sm_act),
                n_nm_eff=len(nm_eff), n_nm_eff_act=len(nm_eff_act),
                cls=cls, mwu_p=mwu_p))

    # ---- BH + bootstrap CIs + summary ----
    pd.DataFrame(per_guide_rows).to_csv(
        ROOT / f"outputs/metrics/seed_dissociation_{CELL}_per_guide.csv", index=False)

    summary = {"cell": CELL, "n_pairs": len(pairs), "tiers": {}}
    for T in TIERS:
        pp = pd.DataFrame(per_pair[T])
        ps = pp["mwu_p"].values
        q = np.full(len(pp), np.nan)
        ok = ~np.isnan(ps)
        if ok.sum():
            q[ok] = bh(ps[ok])
        pp["q_bh"] = q
        pp["direct_confirmed"] = (pp["cls"] == "DIRECT") & (pp["q_bh"] < 0.05)
        pp.to_csv(ROOT / f"outputs/metrics/seed_dissociation_{CELL}_per_pair_tier{T}.csv", index=False)

        testable = pp[pp["cls"] != "AMBIGUOUS"]
        n_t = len(testable)
        frac = {}
        for c in ("DIRECT", "TRANS"):
            k = int((testable["cls"] == c).sum())
            # bootstrap CI over testable pairs
            if n_t:
                idx = RNG.integers(0, n_t, size=(NBOOT, n_t))
                cls_arr = (testable["cls"].values == c)
                boot = cls_arr[idx].mean(1)
                ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
            else:
                ci = [None, None]
            frac[c] = dict(n=k, frac=(k / n_t if n_t else None), ci95=ci)
        summary["tiers"][str(T)] = dict(
            n_pairs=len(pp), n_ambiguous=int((pp["cls"] == "AMBIGUOUS").sum()),
            n_testable=n_t,
            n_direct=int((pp["cls"] == "DIRECT").sum()),
            n_direct_confirmed=int(pp["direct_confirmed"].sum()),
            n_trans=int((pp["cls"] == "TRANS").sum()),
            fractions=frac,
            note=("FRAGILE-N: <10 testable pairs" if n_t < 10 else None))

    with open(ROOT / f"outputs/metrics/seed_dissociation_{CELL}_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nwrote outputs/metrics/seed_dissociation_{CELL}_*.{{csv,json}}")


if __name__ == "__main__":
    main()
