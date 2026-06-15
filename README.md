# Perturb Seed — CRISPRa (Norman/Southard TF atlas)

## Overview

This repository runs the **`perturb_seed`** seed-driven off-target pipeline
([AustinHartman/perturb_seed](https://github.com/AustinHartman/perturb_seed)) on a genome-scale
**CRISPRa** Perturb-seq atlas, matched apples-to-apples against the CRISPRi datasets from the
preprint: **"Systematic identification of seed-driven off-target effects in Perturb-seq
experiments."**

It is the CRISPRa counterpart to
[`perturb_seed_reproduce`](https://github.com/AustinHartman/perturb_seed_reproduce) — same method
and effect definition, with the sign flipped for activation.

## Run the pipeline

This is the whole ask — generate the CRISPRa off-target candidates:

```bash
python vendor/perturb_seed/off_target_pipeline_perturb.py --params configs/norman_hs27_crispra_singlecell.yaml
```

Output is `data/norman_hs27_2kb_filt_no_lfc_cutoff_off_target_candidates.csv.gz`, the **same schema
as upstream** (`gene`, `guide_id`, `target_log2fc`, `offtarget_gene`, `offtarget_log2fc`,
`neighbor_rank`, `direction`, `seed`, `seed_match_len`, `guide_hamming_dist`,
`seed_match_{chrom,start,strand}`), with log2FC positive for activation. The only CRISPRa change is
`mode: crispra`, `expm1_input: true` (the Norman `.X` is already log1p), and
`require_source_guide_effective: false` — see *What changed* below.

Download the Norman AnnData first (see *Data*); the guide library it reads
(`guide_libraries/sgrna_norman.csv`) is committed here. Everything else the analysis emits —
figures, metrics tables, intermediate CSVs — is **generated locally** from these inputs and is not
tracked in the repo.

## Repository Structure

- **`vendor/perturb_seed/`** — the pipeline (adapted for CRISPRa) and its assets (`tss_map.csv`,
  `guide_libraries/`, `input_yamls/`, `environment.yml`)
- **`configs/`** — CRISPRa run configs (`norman_hs27…` canonical, `rpe1…` second cell type)
- **`guide_libraries/`** — Norman/Southard sgRNA library (`sgrna_norman.csv`), the run's `--params` input
- **`data/`** — gzipped off-target candidate tables (the committed analysis output; figures and metrics tables are generated locally)
- **`scripts/`** — analysis organized by result (the off-target characterization below)

## What changed (vs `perturb_seed`)

Backwards-compatible and sign-explicit: every new parameter defaults to the original CRISPRi
behavior, so the four CRISPRi YAMLs still reproduce upstream byte-for-byte. The sign is explicit in
one place rather than negating the data:

```python
def effect_passes(log2fc, threshold):
    # CRISPRi off-targets are repressed below -threshold;
    # CRISPRa off-targets are activated above +threshold.
    if effect_mode() == "crispri":
        return float(log2fc) < -float(threshold)
    return float(log2fc) >  float(threshold)
```

The seed search, neighbor graph, PCA, kNN, and `tss_map.csv` are untouched.

## The off-target characterization (`scripts/`)

Everything past the run is the "is CRISPRa actually worse?" analysis, grouped by result. Each runs
off the candidate table in `data/` (plus the AnnData / ATAC inputs in the data section).

- **Gate (run first)** — `reproduce_seed_fitness_anchor.py` reproduces the Gilbert seed-fitness
  regression; downstream numbers aren't trusted until it passes.
- **Burden, apples-to-apples** — `genome_wide_potential_crispra.py` (sequence potential),
  `realized_apples_to_apples.py` (realized rate), `trans_mechanism_seedperm.py` (seed→effect curve
  + permutation null) → `make_paper_figures.py` (FIG1).
- **On-target positive control** — `trans_mechanism_reality.py` (split-half *r* ≈ 0.8).
- **Chromatin gating** — `chromatin_gating_test.py`, `chromatin_gating_rpe1.py` →
  `make_paper_figures.py` (FIG2).
- **Cross-cell-type / specificity** — `cross_celltype_analysis.py`, `systematic_accessibility.py`,
  `measure_specificity_cause.py` → `make_fig_systematic.py`.
- **Seed dissociation (DIRECT vs INDIRECT)** — `seed_dissociation.py` (+ `seed_dissociation_lib.py`,
  `seed_match_core.py`), `bootstrap_headline_cis.py` → `make_fig_seed_dissociation.py`,
  `make_fig_splithalf_noisefloor.py`.
- **Controls battery (depth)** — `ctrl_battery.py` orchestrates `ctrl_{config,covariate,equivalence,
  negative,spikein,crispri_downsample}.py` + `ctrl_validate.py` (all on `controls_lib.py`) →
  `make_controls_figure.py`.

## Installation

```bash
conda env create -f environment.yml
conda activate vcp
```

## Data

AnnData objects and chromatin tracks download separately. Hs27 CRISPRa — Zenodo [15200179](https://zenodo.org/records/15200179);
RPE1 CRISPRa — Zenodo [15213619](https://zenodo.org/records/15213619); ATAC + CUT&RUN + RNA-seq —
Zenodo [15215216](https://zenodo.org/records/15215216); guide library —
[norman-lab-msk/TFs_CRISPRa](https://github.com/norman-lab-msk/TFs_CRISPRa); genome hg38 (UCSC).

## Resources

- 📄 [Read the preprint](https://www.biorxiv.org/content/10.64898/2026.03.27.714658v2)
- 🔧 [Off-target candidate generation — perturb_seed](https://github.com/AustinHartman/perturb_seed)
- 🔁 [CRISPRi reproduction — perturb_seed_reproduce](https://github.com/AustinHartman/perturb_seed_reproduce)
- 🌐 [Seed-finder web app](https://crispr-seed-finder.vercel.app/)

Built on Austin Hartman's `perturb_seed` (see `vendor/perturb_seed/LICENSE`). Roth Lab, 2026.
