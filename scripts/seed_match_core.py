#!/usr/bin/env python3
"""Standalone copy of the perturb-seed seed-matching primitives.

Lifted verbatim from vendor/perturb_seed/off_target_pipeline_perturb.py so the
dissociation can run without importing that module's heavy deps (scanpy/sklearn).
Behaviour is byte-for-byte identical (verified: 157/157 nominated seeds reproduce).
"""
from __future__ import annotations
import re
from pathlib import Path

RC_MAP = str.maketrans("ACGTacgt", "TGCAtgca")


def revcomp(seq: str) -> str:
    return seq.translate(RC_MAP)[::-1]


def load_fai(fai_path: Path) -> dict:
    fai = {}
    with open(fai_path) as fh:
        for line in fh:
            parts = line.rstrip().split("\t")
            if len(parts) < 5:
                continue
            chrom = parts[0]
            fai[chrom] = (int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]))
    return fai


def extract_genome_region(fai_dict: dict, chrom: str, start: int, end: int,
                          genome_path: Path) -> str:
    if chrom not in fai_dict:
        return ""
    chrom_len, fai_offset, linebases, linewidth = fai_dict[chrom]
    start = max(0, start)
    end = min(chrom_len, end)
    if start >= end:
        return ""
    byte_start = fai_offset + (start // linebases) * linewidth + (start % linebases)
    byte_last = fai_offset + ((end - 1) // linebases) * linewidth + ((end - 1) % linebases)
    with open(genome_path, "rb") as fh:
        fh.seek(byte_start)
        raw = fh.read(byte_last - byte_start + 1)
    seq = raw.decode("ascii", errors="ignore").replace("\n", "").replace("\r", "")
    return seq[: end - start].upper()


def find_max_seed_match(full_spacer: str, seq: str, min_len: int = 5):
    best = 0
    best_info = None
    spacer_len = len(full_spacer)
    seq_up = seq.upper()
    for length in range(min_len, spacer_len + 1):
        seed = full_spacer[-length:].upper()
        rc_seed = revcomp(seed)
        m_fwd = re.search(re.escape(seed) + r"[ACGT]GG", seq_up)
        m_rev = re.search(r"CC[ACGT]" + re.escape(rc_seed), seq_up)
        if m_fwd:
            best = length
            best_info = ("fwd", m_fwd.start())
        elif m_rev:
            best = length
            best_info = ("rev", m_rev.start())
    if best == 0 or best_info is None:
        return 0, None, None, None, None, None
    strand, p = best_info
    if strand == "fwd":
        g_start = p - (spacer_len - best)
        if g_start < 0:
            genomic = "N" * (-g_start) + seq_up[: p + best]
        else:
            genomic = seq_up[g_start: p + best]
        local_pos = p
        strand_char = "+"
    else:
        s_end = p + 3 + spacer_len
        site_fwd = (seq_up[p + 3: s_end] if s_end <= len(seq_up)
                    else seq_up[p + 3:] + "N" * (s_end - len(seq_up)))
        genomic = revcomp(site_fwd)
        local_pos = p + 3
        strand_char = "-"
    hamming = sum(a != b for a, b in zip(full_spacer.upper(), genomic))
    return best, hamming, local_pos, strand_char, genomic, full_spacer.upper()
