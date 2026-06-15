#!/usr/bin/env python3
"""Analysis 3: bootstrap + exact-binomial 95% CIs on every headline number, with
explicit FRAGILE-N flags (any rate resting on <10 events).

Headline numbers audited:
  - potential off-target >=12bp  (reported 77.5% of spacers)
  - realized off-target >=8bp    (reported ~4.7%)   [library & analyzed denominators]
  - realized off-target >=12bp   (reported ~0.06%)  [fragile: ~2 events]
  - poised-stratum activation    (reported 19.7%)   + the DIRECT-only rate (6.1%)
  - guide-dissociation DIRECT fraction (61.5%)

Usage: .venv/bin/python src/bootstrap_headline_cis.py
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from seed_dissociation_lib import PromoterFetcher  # noqa: E402

RNG = np.random.default_rng(0)
NBOOT = 2000
OUT = ROOT / "outputs/metrics/headline_bootstrap_cis.json"


def cp_ci(k, n, alpha=0.05):
    """Clopper-Pearson exact binomial 95% CI."""
    if n == 0:
        return [None, None]
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return [float(lo), float(hi)]


def boot_prop(indicator):
    """Bootstrap 95% CI for a proportion given a 0/1 indicator array."""
    m = np.asarray(indicator, float)
    if len(m) == 0:
        return [None, None]
    idx = RNG.integers(0, len(m), size=(NBOOT, len(m)))
    b = m[idx].mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def entry(k, n, label):
    rate = k / n if n else None
    return dict(label=label, k=int(k), n=int(n), rate=rate,
                exact_ci95=cp_ci(k, n),
                fragile=(k < 10), note=("FRAGILE-N (<10 events)" if k < 10 else None))


def main():
    pf = PromoterFetcher(half=2000)
    res = {"nboot": NBOOT}

    # ---- realized rates from the candidate table ----
    d = pd.read_csv(ROOT / "outputs/candidates/norman_crispra_singlecell_off_target_candidates.csv", low_memory=False)
    for c in ("offtarget_log2fc", "seed_match_len", "seed_match_start"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[d.gene.astype(str).str.upper() != d.offtarget_gene.astype(str).str.upper()].copy()
    d["td"] = [pf.tss_dist(g, s) for g, s in zip(d.offtarget_gene.astype(str), d.seed_match_start)]
    analyzed = sorted(d.guide_id.astype(str).unique())
    analyzed_n = len(analyzed)
    lib = pd.read_csv(ROOT / "outputs/metrics/norman_hs27_guide_library.csv")
    library_n = int((lib["designed_target_gene_name"].astype(str) != "off-target").sum())

    def realized_guides(T):
        r = d[(d.seed_match_len >= T) & (d.offtarget_log2fc > 0.5) & (d.td <= 1000)]
        return set(r.guide_id.astype(str))

    res["realized"] = {}
    for T in (8, 12):
        rg = realized_guides(T)
        k = len(rg)
        # analyzed denominator: bootstrap over the analyzed guide set
        ind = np.array([g in rg for g in analyzed], float)
        res["realized"][f">={T}bp"] = {
            "analyzed": {**entry(k, analyzed_n, f"realized >={T}bp (% analyzed)"),
                         "boot_ci95": boot_prop(ind)},
            "library": entry(k, library_n, f"realized >={T}bp (% library)"),
        }

    # ---- potential >=12bp ----
    pot = pd.read_csv(ROOT / "outputs/metrics/norman_crispra_genome_wide_potential_APPLES2APPLES_1kb_excl_ontarget.csv")
    row = pot[pot.min_seed_len == 12].iloc[0]
    k, n = int(row.n_spacers_with_match), int(row.n_spacers_total)
    res["potential_ge12bp"] = {**entry(k, n, "potential >=12bp (% spacers)"),
                               "boot_ci95": boot_prop(np.array([1] * k + [0] * (n - k), float))}

    # ---- poised-stratum activation (any vs DIRECT-only) ----
    cg = json.load(open(ROOT / "outputs/metrics/chromatin_gating_direct_vs_trans.json"))
    g = cg.get("gate_retest_by_ntc_stratum", {}).get("poised(0-0.1]")
    if g:
        n = g["n"]
        for key, rate in [("any_activation", g["any_act"]), ("direct_activation", g["direct_act"])]:
            k = int(round(rate * n))
            res[f"poised_{key}"] = {**entry(k, n, f"poised-stratum {key}"),
                                    "boot_ci95": boot_prop(np.array([1] * k + [0] * (n - k), float))}

    # ---- dissociation DIRECT fraction ----
    diss = json.load(open(ROOT / "outputs/metrics/seed_dissociation_hs27_summary.json"))
    for T in ("8", "12"):
        t = diss["tiers"].get(T)
        if not t:
            continue
        nt, nd = t["n_testable"], t["n_direct"]
        res[f"dissociation_direct_frac_tier{T}"] = {
            **entry(nd, nt, f"DIRECT fraction of testable realized off-targets (>={T}bp)"),
            "n_trans": t["n_trans"], "n_ambiguous": t["n_ambiguous"]}

    OUT.write_text(json.dumps(res, indent=2))
    # pretty print
    def show(e):
        ci = e.get("exact_ci95", [None, None])
        b = e.get("boot_ci95")
        flag = "  *** FRAGILE-N ***" if e.get("fragile") else ""
        bs = f"  boot[{b[0]*100:.1f},{b[1]*100:.1f}]" if b and b[0] is not None else ""
        print(f"  {e['label']:<52} {100*e['rate']:6.2f}%  (n={e['n']}, k={e['k']})  "
              f"exact[{ci[0]*100:.2f},{ci[1]*100:.2f}]{bs}{flag}")
    print("\n=== HEADLINE NUMBERS WITH 95% CIs ===")
    show(res["potential_ge12bp"])
    for T in (8, 12):
        show(res["realized"][f">={T}bp"]["analyzed"])
        show(res["realized"][f">={T}bp"]["library"])
    for key in ("poised_any_activation", "poised_direct_activation"):
        if key in res:
            show(res[key])
    for T in ("8", "12"):
        k = f"dissociation_direct_frac_tier{T}"
        if k in res:
            show(res[k])
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
