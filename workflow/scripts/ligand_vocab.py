"""Adaptive ligand-fragment vocabulary, derived from the run's own outputs.

Nothing about the construct set is hard-coded, so the same code reports any library:

  * the feature list comes from the decode itself (the observed ``fragment_id``s,
    optionally enriched with the expected-sequences CSV named in the run log so
    expected-but-absent features can be flagged);
  * the UMI/anchor (backbone) set is read from the tool's own ``*.umis.parquet``
    ``anchor_name`` column -- fallback: expected sequences that carry ``N`` wildcards;
  * each feature's structural CLASS is tokenised from its name (the variable
    size/strand tokens are stripped, the alphabetic stem is the class);
  * class ORDER follows first appearance in the expected CSV (else alphabetical),
    and class COLOURS are assigned from a fixed qualitative palette by that order.

Run it on a different construct set and it re-derives a different vocabulary with
no code change.
"""
import re
from pathlib import Path

import polars as pl

# A feature's name carries two *variable* tokens we strip to reach its class stem:
# a size token (`10nt`, `1ntbulge`) and a strand token (`Strand1`, `-Strand-2`).
_STRAND = re.compile(r"[-_ ]*strand[-_ ]*(\d+)", re.I)
_SIZE = re.compile(r"(\d+)\s*nt", re.I)

# Colour-blind-friendly qualitative palette; assigned to classes by their order.
PALETTE = [
    "#4f9cff", "#3fb950", "#e3b341", "#db61a2", "#a371f7",
    "#d2691e", "#26c6da", "#ef5350", "#8bc34a", "#ab47bc",
    "#ffa657", "#79c0ff",
]
GREY = "#8b949e"          # backbone / anchor / "other"


def strip_rc(fid: str) -> str:
    """Collapse the reverse-complement orientation: ``X_rc`` and ``X`` are one feature."""
    return fid[:-3] if fid.endswith("_rc") else fid


def parse_feature(name: str):
    """``(class, length, strand)`` tokenised from a feature name -- all data-driven.

    ``length`` is the nt size token if present (bulge size for bulge features), else
    ``None`` (e.g. junctions carry no size). ``strand`` is the StrandN index or ``None``.
    """
    base = strip_rc(name)
    ms = _STRAND.search(base)
    strand = int(ms.group(1)) if ms else None
    stem = _STRAND.sub("", base)
    ml = _SIZE.search(stem)
    length = int(ml.group(1)) if ml else None
    stem = _SIZE.sub("", stem)
    stem = re.sub(r"[-_]+", "", stem)      # drop separators
    stem = re.sub(r"\d+$", "", stem).strip()   # drop trailing index digits
    return (stem or "other"), length, strand


def read_expected(csv_path):
    """``[(name, seq), ...]`` from the expected-sequences CSV, or ``[]`` if unavailable."""
    p = Path(csv_path) if csv_path else None
    if not (p and p.exists()):
        return []
    df = pl.read_csv(p)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    name_c = next((c for c in df.columns if c.lower().startswith("name")), df.columns[0])
    seq_c = next((c for c in df.columns if c.lower().startswith("seq")), df.columns[-1])
    return list(zip(df[name_c].to_list(), df[seq_c].to_list()))


def anchor_set(umis_parquet=None, expected=None):
    """Backbone/anchor feature names.

    Authoritative source: the distinct ``anchor_name`` in the tool's umis parquet.
    Fallback: expected-sequence names that carry an ``N`` wildcard (UMI constructs).
    """
    p = Path(umis_parquet) if umis_parquet else None
    if p and p.exists():
        names = (
            pl.scan_parquet(p).select(pl.col("anchor_name").unique())
            .collect()["anchor_name"].to_list()
        )
        if names:
            return set(names)
    if expected:
        return {n for n, s in expected if "N" in str(s).upper()}
    return set()


def class_order(feature_bases, expected=None, anchors=frozenset()):
    """Payload classes ordered by first appearance in the expected CSV, then by name."""
    order, seen = [], set()
    src = [n for n, _ in expected] if expected else sorted(feature_bases)
    for name in src:
        b = strip_rc(name)
        if b in anchors:
            continue
        c = parse_feature(b)[0]
        if c not in seen:
            seen.add(c); order.append(c)
    for b in sorted(feature_bases):     # observed classes not in the expected CSV
        if b in anchors:
            continue
        c = parse_feature(b)[0]
        if c not in seen:
            seen.add(c); order.append(c)
    return order


def class_colors(order):
    """Map each class to a stable colour by its position in ``order``."""
    return {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(order)}


def parse_log(log_path):
    """Best-effort provenance + UMI-pass counts from the ligateddna ``.log``."""
    out = {}
    p = Path(log_path) if log_path else None
    if not (p and p.exists()):
        return out
    txt = p.read_text()
    for key, pat in (
        ("expected", r"expected\s*:\s*(\S+)"),
        ("branch", r"branch\s*:\s*(.+)"),
        ("binary", r"binary\s*:\s*(\S+)"),
        ("min_identity", r"Min identity:\s*([\d.]+)"),
    ):
        m = re.search(pat, txt)
        if m:
            out[key] = m.group(1).strip()
    m = re.search(r"(\d+)\s+reads fused,\s*(\d+)\s+dropped", txt)
    if m:
        out["fused"], out["dropped"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"missing_anchor\s+(\d+).*?strand_conflict\s+(\d+)", txt)
    if m:
        out["missing_anchor"], out["strand_conflict"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"Anchor trim:\s*(\d+)", txt)
    if m:
        out["trimmed"] = int(m.group(1))
    return out
