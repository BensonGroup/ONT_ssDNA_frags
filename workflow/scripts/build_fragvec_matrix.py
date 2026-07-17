#!/usr/bin/env python
"""Assemble one group's fragment-vector x sample matrices (BOTH axes) from per-sample counts.

Columns are the group's samples with the R0 initial library FIRST. Unlike the per-clone UMI matrix
(whose R0 column is ~all zeros, because R0's sequencing sampled ~10^-6 of the molecule pool and never
saw the clones that later win), the R0 column here is **densely populated** -- at the vector level
counts aggregate over thousands of clones, so R0 is a genuine baseline.

Emits:
  <out-prefix>.long.parquet     vector, sample, n_reads, n_umis   (canonical, both axes)
  <out-prefix>.reads.mtx        MatrixMarket -- ABUNDANCE (n_reads)
  <out-prefix>.umis.mtx         MatrixMarket -- INCIDENCE (n_umis = independent clones)
  <out-prefix>.vectors.tsv      row labels (fragment vector) in row-index order
  <out-prefix>.samples.tsv      column labels in column order (baseline first)

Rows are ordered by total reads across the group (descending), then by vector, so the top of the
matrix is the signal-carrying end. Deterministic.

CLI:
    build_fragvec_matrix.py --group G --samples s1,s2,... --counts c1,c2,... --out-prefix PREFIX
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl


def _write_mtx(path: str, trips: pl.DataFrame, nrows: int, ncols: int, comment: str, value: str):
    with open(path, "w") as f:
        f.write("%%MatrixMarket matrix coordinate integer general\n")
        f.write(f"%% {comment}\n")
        f.write(f"{nrows} {ncols} {trips.height}\n")
        for r, c, v in trips.select("row", "col", value).iter_rows():
            f.write(f"{r} {c} {v}\n")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Per-group fragment-vector x sample matrices.")
    ap.add_argument("--group", required=True)
    ap.add_argument("--samples", required=True, help="comma-separated, baseline first")
    ap.add_argument("--counts", required=True, help="comma-separated fragvec_counts.parquet paths")
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args(argv[1:])

    samples = args.samples.split(",")
    paths = args.counts.split(",")
    if len(samples) != len(paths):
        ap.error(f"--samples ({len(samples)}) and --counts ({len(paths)}) differ in length")
    out = args.out_prefix

    longs = []
    for s, p in zip(samples, paths):
        df = pl.read_parquet(p, columns=["vector", "n_reads", "n_umis"])
        longs.append(df.with_columns(pl.lit(s).alias("sample")))
    long = pl.concat(longs).select("vector", "sample", "n_reads", "n_umis")
    long.write_parquet(f"{out}.long.parquet")

    # Row order: total abundance across the group, descending (signal first), then vector.
    order = (long.group_by("vector").agg(pl.col("n_reads").sum().alias("tot"))
             .sort(["tot", "vector"], descending=[True, False]))
    vecs = order["vector"].to_list()
    vrow = pl.DataFrame({"vector": vecs, "row": list(range(1, len(vecs) + 1))})
    scol = pl.DataFrame({"sample": samples, "col": list(range(1, len(samples) + 1))})

    trips = (long.join(vrow, on="vector").join(scol, on="sample")
             .select("row", "col", "n_reads", "n_umis").sort(["row", "col"]))
    nrows, ncols = len(vecs), len(samples)
    _write_mtx(f"{out}.reads.mtx", trips, nrows, ncols,
               f"group={args.group}; ABUNDANCE (n_reads); rows=fragment vectors; cols=samples (baseline first)",
               "n_reads")
    _write_mtx(f"{out}.umis.mtx", trips, nrows, ncols,
               f"group={args.group}; INCIDENCE (n_umis = independent clones); rows=fragment vectors; cols=samples",
               "n_umis")

    Path(f"{out}.vectors.tsv").write_text("\n".join(vecs) + ("\n" if vecs else ""))
    Path(f"{out}.samples.tsv").write_text("\n".join(samples) + "\n")

    print(f"[{args.group}] {nrows:,} vectors x {ncols} samples, {trips.height:,} nonzeros -> {out}.*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
