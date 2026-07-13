#!/usr/bin/env python
"""Assemble one group's corrected-UMI x sample read-count matrix from per-sample species tables.

For a selection-screen group (the R0 initial library as the FIRST column, then the group's rounds
in order), this stacks each sample's `<STEM>.umi_species.tsv` (`fused_seq  n_reads`, corrected UMIs)
into a long table and emits:

  <out-prefix>.long.parquet   fused_umi, sample, n_reads   (canonical; 1 row per (UMI, sample))
  <out-prefix>.mtx            MatrixMarket coordinate integer general (rows = distinct fused_umi
                              ordered by sorted sequence; cols = the given sample order, baseline 1st)
  <out-prefix>.umis.tsv       row labels (fused_umi) in row-index order
  <out-prefix>.samples.tsv    column labels (sample) in column order

The row key is the CORRECTED fused UMI (starcode ed<=1 centroid), so the same molecule lineage
matches across rounds. Deterministic: rows sorted by sequence, columns by the given order.

CLI:
    build_umi_matrix.py --group G --samples s1,s2,... --species-tsvs t1,t2,... --out-prefix PREFIX
(`--samples` and `--species-tsvs` are parallel, comma-separated, baseline first.)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Build a per-group corrected-UMI x sample matrix.")
    ap.add_argument("--group", required=True)
    ap.add_argument("--samples", required=True, help="comma-separated sample names, baseline first")
    ap.add_argument("--species-tsvs", required=True, help="comma-separated species TSV paths (parallel to --samples)")
    ap.add_argument("--out-prefix", required=True, help="output path prefix (<prefix>.long.parquet etc.)")
    args = ap.parse_args(argv[1:])

    samples = args.samples.split(",")
    tsvs = args.species_tsvs.split(",")
    if len(samples) != len(tsvs):
        ap.error(f"--samples ({len(samples)}) and --species-tsvs ({len(tsvs)}) differ in length")
    out_prefix = args.out_prefix

    longs = []
    for s, t in zip(samples, tsvs):
        df = pl.read_csv(t, separator="\t")  # header: fused_seq, n_reads
        cols = df.columns
        df = df.rename({cols[0]: "fused_umi", cols[1]: "n_reads"})
        longs.append(df.select("fused_umi", "n_reads").with_columns(pl.lit(s).alias("sample")))
    long = pl.concat(longs).select("fused_umi", "sample", "n_reads")
    # (fused_umi, sample) is unique by construction; SUM guards any surprise duplicate.
    long = (
        long.group_by("fused_umi", "sample").agg(pl.col("n_reads").sum())
        .select("fused_umi", "sample", "n_reads")
    )
    long.write_parquet(f"{out_prefix}.long.parquet")

    # Row index: distinct fused_umi ordered by sorted sequence (1-based, deterministic).
    umi_list = long.select("fused_umi").unique().sort("fused_umi")["fused_umi"].to_list()
    umi_row = pl.DataFrame({"fused_umi": umi_list, "row": list(range(1, len(umi_list) + 1))})
    # Column index: the given sample order (1-based), baseline first.
    sample_col = pl.DataFrame({"sample": samples, "col": list(range(1, len(samples) + 1))})

    trip = (
        long.join(umi_row, on="fused_umi").join(sample_col, on="sample")
        .select("row", "col", "n_reads").sort(["row", "col"])
    )
    nrows, ncols, nnz = len(umi_list), len(samples), trip.height

    with open(f"{out_prefix}.mtx", "w") as f:
        f.write("%%MatrixMarket matrix coordinate integer general\n")
        f.write(f"%% group={args.group}; rows=distinct corrected fused_umi; cols=samples (baseline first)\n")
        f.write(f"{nrows} {ncols} {nnz}\n")
        for r, c, v in trip.iter_rows():
            f.write(f"{r} {c} {v}\n")

    Path(f"{out_prefix}.umis.tsv").write_text("\n".join(umi_list) + ("\n" if umi_list else ""))
    Path(f"{out_prefix}.samples.tsv").write_text("\n".join(samples) + "\n")

    print(f"[{args.group}] matrix {nrows} UMIs x {ncols} samples, {nnz} nonzeros -> {out_prefix}.*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
