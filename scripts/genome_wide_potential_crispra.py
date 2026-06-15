#!/usr/bin/env python3
"""Genome-wide sequence-only seed+PAM potential for Norman CRISPRa guides."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd


TIERS = [5, 8, 10, 12]
DIST_BANDS = [(0, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 10**12)]
RC_MAP = str.maketrans("ACGTacgt", "TGCAtgca")


def revcomp(seq: str) -> str:
    return seq.translate(RC_MAP)[::-1].upper()


def load_fai(path: Path) -> dict[str, tuple[int, int, int, int]]:
    out = {}
    with path.open() as fh:
        for line in fh:
            chrom, size, offset, linebases, linewidth = line.rstrip().split("\t")[:5]
            out[chrom] = (int(size), int(offset), int(linebases), int(linewidth))
    return out


def extract_region(fh, fai: dict, chrom: str, start: int, end: int) -> str:
    if chrom not in fai:
        return ""
    chrom_len, offset, linebases, linewidth = fai[chrom]
    start = max(0, start)
    end = min(chrom_len, end)
    if start >= end:
        return ""
    byte_start = offset + (start // linebases) * linewidth + (start % linebases)
    byte_last = offset + ((end - 1) // linebases) * linewidth + ((end - 1) % linebases)
    fh.seek(byte_start)
    return fh.read(byte_last - byte_start + 1).decode("ascii", errors="ignore").replace("\n", "").upper()[: end - start]


def band_label(distance: float) -> str:
    for lo, hi in DIST_BANDS:
        if lo <= distance < hi:
            return f"{lo}-{hi if hi < 10**12 else 'inf'}"
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guide-library", type=Path, default=Path("outputs/metrics/norman_hs27_guide_library.csv"))
    parser.add_argument("--tss-map", type=Path, default=Path("vendor/perturb_seed/tss_map.csv"))
    parser.add_argument("--genome-fa", type=Path, default=Path("data/ref/hg38.fa"))
    parser.add_argument("--genome-fai", type=Path, default=Path("data/ref/hg38.fa.fai"))
    parser.add_argument("--upstream", type=int, default=2000)
    parser.add_argument("--downstream", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=Path("outputs/metrics/norman_crispra_genome_wide_potential.csv"))
    args = parser.parse_args()

    guides = pd.read_csv(args.guide_library)
    guides = guides[guides["designed_target_gene_name"].ne("off-target")].copy()
    guides = guides[guides["seq"].astype(str).str.len().eq(20)]
    n_guides = int(len(guides))

    seed_to_count: dict[int, dict[str, int]] = {}
    for tier in TIERS:
        seed_to_count[tier] = guides["seq"].astype(str).str[-tier:].str.upper().value_counts().to_dict()

    tss = pd.read_csv(args.tss_map)
    tss = tss.drop_duplicates("gene_name").copy()
    n_genes = int(len(tss))
    fai = load_fai(args.genome_fai)

    potential_pairs = {tier: 0 for tier in TIERS}
    band_pairs = {(tier, band_label(lo)): 0 for tier in TIERS for lo, _ in DIST_BANDS}
    matched_seeds = {tier: set() for tier in TIERS}

    with args.genome_fa.open("rb") as fh:
        for i, row in enumerate(tss.itertuples(index=False), 1):
            gene = str(row.gene_name)
            chrom = str(row.chrom)
            tss0 = int(row.tss) - 1
            strand = getattr(row, "strand", "+")
            if strand == "+":
                start, end = tss0 - args.upstream, tss0 + args.downstream
            else:
                start, end = tss0 - args.downstream, tss0 + args.upstream
            seq = extract_region(fh, fai, chrom, start, end)
            if not seq:
                continue

            matched: dict[int, dict[str, float]] = {tier: {} for tier in TIERS}
            n = len(seq)

            # Forward PAM: protospacer immediately upstream of NGG.
            for pam_start in range(0, n - 2):
                if seq[pam_start + 1 : pam_start + 3] == "GG":
                    genomic_seed_end = start + pam_start
                    for tier in TIERS:
                        if pam_start >= tier:
                            seed = seq[pam_start - tier : pam_start]
                            if seed in seed_to_count[tier]:
                                dist = abs(genomic_seed_end - tier - tss0)
                                matched[tier][seed] = min(dist, matched[tier].get(seed, dist))

                # Reverse PAM: CCN followed by reverse-complement seed on the plus strand.
                if seq[pam_start : pam_start + 2] == "CC":
                    for tier in TIERS:
                        seed_start = pam_start + 3
                        seed_end = seed_start + tier
                        if seed_end <= n:
                            seed = revcomp(seq[seed_start:seed_end])
                            if seed in seed_to_count[tier]:
                                dist = abs((start + seed_start) - tss0)
                                matched[tier][seed] = min(dist, matched[tier].get(seed, dist))

            for tier, seed_dist in matched.items():
                for seed, dist in seed_dist.items():
                    count = int(seed_to_count[tier][seed])
                    potential_pairs[tier] += count
                    band_pairs[(tier, band_label(dist))] += count
                    matched_seeds[tier].add(seed)

            if i % 5000 == 0:
                print(f"scanned {i:,}/{n_genes:,} promoters", flush=True)

    rows = []
    denom = n_guides * n_genes
    for tier in TIERS:
        n_guides_with_any = int(sum(seed_to_count[tier][seed] for seed in matched_seeds[tier]))
        rows.append(
            {
                "scope": "genome_wide_sequence_only",
                "seed_tier": f">={tier}",
                "distance_band": "all",
                "n_guides": n_guides,
                "n_genes": n_genes,
                "n_guides_with_any_potential": n_guides_with_any,
                "guide_any_potential_rate": n_guides_with_any / n_guides,
                "potential_pairs": potential_pairs[tier],
                "denominator_pairs": denom,
                "potential_rate": potential_pairs[tier] / denom,
            }
        )
        for lo, _ in DIST_BANDS:
            label = band_label(lo)
            rows.append(
                {
                    "scope": "genome_wide_sequence_only",
                    "seed_tier": f">={tier}",
                    "distance_band": label,
                    "n_guides": n_guides,
                    "n_genes": n_genes,
                    "n_guides_with_any_potential": n_guides_with_any,
                    "guide_any_potential_rate": n_guides_with_any / n_guides,
                    "potential_pairs": band_pairs[(tier, label)],
                    "denominator_pairs": denom,
                    "potential_rate": band_pairs[(tier, label)] / denom,
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(args.out)


if __name__ == "__main__":
    main()
