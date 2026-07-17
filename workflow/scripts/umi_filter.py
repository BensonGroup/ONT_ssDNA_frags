#!/usr/bin/env python3
"""Per-read UMI classification: the frame every UMI decision is made on.

A LIBRARY, not a stage. It used to drive its own `umifilt/` tree, whose only consumer was
featurecount; featurecount now reads `decode/` (raw) and `umi/` (clean) -- two fragment reports
bracketing the UMI processing -- so that tree had nothing left reading it and the stage was folded
into umi_correct.py, which had been running this same classification internally all along.

`build_per_read` is the single frame from which umi_correct derives: which reads are garbage (and
under which class), the per-read audit, and the survivors it corrects. Keeping it here rather than
inline in umi_correct keeps the classification one importable, testable thing.

The classification RULE itself lives one level down in umi_filter_lib.py -- deliberately
dependency-free (no polars) so it unit-tests under a bare python3. This module is the polars layer
over it.

Nothing here ever COLLAPSES. This data must not be UMI-deduplicated -- the typical molecule is seen
about once, so there is almost nothing to remove and collapsing would destroy real signal
(clarification.md:667).
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))  # import the local helpers
import umi_filter_lib as uf

Q10_PHRED = 10                 # "below Q10" == Phred < 10 (ASCII '+' == Phred 10 == chr(43))
CLASS_ORDER = ["drop_nonacgt", "drop_lowcomplexity", "drop_umilen"]  # for stable summary columns


# --------------------------------------------------------------------------- pure-ish helpers
def _qual_stats(qual: str) -> dict:
    """Per-UMI-quality-string stats from a Phred+33 string: sum of Phred, count below Q10."""
    sp = 0
    nb = 0
    for ch in qual:
        p = ord(ch) - 33
        sp += p
        if p < Q10_PHRED:
            nb += 1
    return {"_sp": sp, "_nb": nb}


def _row_features(fused_seq: str, l0: int, l1: int, h: int, f: float) -> dict:
    """maxrun / max_base_frac / umi_class for one read (delegates to umi_filter_lib)."""
    return {
        "maxrun": uf.maxrun(fused_seq),
        "max_base_frac": uf.max_base_frac(fused_seq),
        "umi_class": uf.classify(
            fused_seq, l0, l1, max_homopolymer=h, max_base_frac=f,
            umi_len_min=uf.UMI_LEN_MIN, umi_len_max=uf.UMI_LEN_MAX,
        ),
    }


def build_per_read(fused_pq: Path, umis_pq: Path, max_homopolymer: int, max_base_frac: float):
    """Per-read frame: read_acc, fused_seq, fused_len, umi stats, maxrun, max_base_frac, umi_class.

    umi_correct.py imports this module for exactly this function and CLASS_ORDER.
    """
    fused = pl.read_parquet(fused_pq, columns=["read_acc", "fused_seq", "fused_len"])

    umi = pl.read_parquet(umis_pq, columns=["read_acc", "umi_index", "umi_len", "umi_qual"])
    # SANITY: zero non-ACGT characters expected across all umi_seq -- but the check must exist,
    # so verify the classification's non-ACGT branch is exercised only via fused_seq (below).
    qstat = umi.with_columns(
        pl.col("umi_qual").map_elements(
            _qual_stats, return_dtype=pl.Struct({"_sp": pl.Int64, "_nb": pl.Int64})
        ).alias("_q")
    ).unnest("_q")

    stats = qstat.group_by("read_acc").agg(
        (pl.col("_sp").sum() / pl.col("umi_len").sum()).alias("mean_umi_phred"),
        pl.col("_nb").sum().cast(pl.Int64).alias("n_umi_bases_below_q10"),
    )
    lens = umi.group_by("read_acc").agg(
        pl.col("umi_len").sort_by("umi_index").alias("_lens")
    ).with_columns(
        pl.col("_lens").list.get(0).alias("umi_len_0"),
        pl.col("_lens").list.get(1).alias("umi_len_1"),
    ).drop("_lens")

    base = fused.join(lens, on="read_acc", how="left").join(stats, on="read_acc", how="left")
    if base.select(pl.col("umi_len_0").is_null().any()).item():
        raise AssertionError("some reads in fused.parquet lack umi rows (umi_len_0 null)")

    feat = base.with_columns(
        pl.struct("fused_seq", "umi_len_0", "umi_len_1").map_elements(
            lambda r: _row_features(
                r["fused_seq"], r["umi_len_0"], r["umi_len_1"], max_homopolymer, max_base_frac),
            return_dtype=pl.Struct(
                {"maxrun": pl.Int64, "max_base_frac": pl.Float64, "umi_class": pl.String}),
        ).alias("_feat")
    ).unnest("_feat").drop("umi_len_0", "umi_len_1")
    return feat
