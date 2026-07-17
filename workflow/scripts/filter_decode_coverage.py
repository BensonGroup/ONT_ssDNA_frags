#!/usr/bin/env python3
"""Drop low-coverage reads from a decode sample tree.

The ONLY read-level drop between `decode` and the analysis branches. A read whose decoded coverage
falls below --min-coverage is mostly sequence the decoder could not resolve; left in, it carries a
UMI and a fragment vector and inflates every downstream count exactly as if it were clean.

Placing the drop HERE, upstream of the fork, is the point: `umifilt/`, `umi/` and `fragvec/` all
read the filtered tree, so every branch sees ONE consistent read set. The raw `decode/` tree is
never modified -- it stays the full, unfiltered record.

This script does NOT measure coverage; `rule coverage` already did, on the decode tree, and wrote
`<s>.coverage.parquet` there. Coverage is a property of the DECODE step, so it is measured exactly
once, upstream, and both this filter and the cross report read that one artifact (see coverage_lib
for the metric itself). This script only decides who survives.

The output tree is a SIBLING of the decode tree carrying identical basenames (covfilt/ <- decode/,
covfilt_initial/ <- decode_initial/), so it is a drop-in source for umi_filter / umi_correct /
build_fragvec, exactly as umifilt/ is a drop-in mirror for featurecount.

Usage:
    filter_decode_coverage.py --src-stem results/decode/NB01/NB01 \
                              --dst-dir results/covfilt/NB01 [--min-coverage 0.9] [--force]
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import coverage_lib as cl
import tree_io as tio

TOOL_VERSION = "filter_decode_coverage.py/2.0"   # 2.0: consumes rule coverage's artifact


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Per-sample decode coverage filter.")
    ap.add_argument("--src-stem", default=os.environ.get("L2F_STEM"),
                    help="path prefix; <src-stem>.coverage.parquet etc. must exist "
                         "(default $L2F_STEM)")
    ap.add_argument("--dst-dir", required=True, help="output sample directory")
    ap.add_argument("--min-coverage", type=float, default=cl.DEFAULT_MIN_COVERAGE,
                    help="keep a read whose decoded coverage is >= this (default "
                         f"{cl.DEFAULT_MIN_COVERAGE})")
    ap.add_argument("--force", action="store_true",
                    help="overwrite a non-empty destination sample dir")
    args = ap.parse_args(argv[1:])

    if not args.src_stem:
        ap.error("--src-stem is required (or set $L2F_STEM)")
    if not 0.0 <= args.min_coverage <= 1.0:
        ap.error(f"--min-coverage must be in [0, 1], got {args.min_coverage}")

    src_stem = Path(args.src_stem)
    if not src_stem.is_absolute():
        src_stem = Path.cwd() / src_stem
    stem = src_stem.name
    dst_dir = Path(args.dst_dir).resolve()

    cov_pq = tio.part(src_stem, ".coverage.parquet")
    fused_pq = tio.part(src_stem, ".fused.parquet")
    umis_pq = tio.part(src_stem, ".umis.parquet")
    frags_pq = tio.part(src_stem, ".fragments.parquet")
    trimmed_pq = tio.part(src_stem, ".trimmed.parquet")
    src_fq = tio.part(src_stem, ".fused_reads.fq")
    src_trunc = tio.part(src_stem, ".truncated.fq")
    src_log = tio.part(src_stem, ".ligateddna.log")
    for p in (cov_pq, fused_pq, umis_pq, frags_pq, trimmed_pq, src_fq):
        if not p.exists():
            print(f"[FATAL] missing input: {p}", file=sys.stderr)
            return 1

    tio.prepare_dst(dst_dir, args.force)

    # ---- select survivors from the PRE-filter coverage record -------------------------------
    cov = pl.read_parquet(cov_pq, columns=["read_acc", "cov"])
    reads_in = cov.height
    kept = cov.filter(pl.col("cov") >= args.min_coverage)
    reads_kept = kept.height
    dropped_lowcoverage = reads_in - reads_kept
    # SANITY: mirrors umi_filter's STEP-1 invariant -- drops + keeps == input.
    assert dropped_lowcoverage + reads_kept == reads_in, \
        f"{dropped_lowcoverage} + {reads_kept} != {reads_in}"

    # ---- write the filtered sample tree ----------------------------------------------------
    survivors_df = kept.select("read_acc").sort("read_acc")
    survivors = set(survivors_df["read_acc"].to_list())
    tio.write_filtered_parquet(fused_pq, tio.part(Path(dst_dir / stem), ".fused.parquet"),
                               survivors_df, ["read_acc"])
    tio.write_filtered_parquet(trimmed_pq, tio.part(Path(dst_dir / stem), ".trimmed.parquet"),
                               survivors_df, ["read_acc"])
    tio.write_filtered_parquet(umis_pq, tio.part(Path(dst_dir / stem), ".umis.parquet"),
                               survivors_df, ["read_acc", "umi_index"])
    tio.write_filtered_parquet(frags_pq, tio.part(Path(dst_dir / stem), ".fragments.parquet"),
                               survivors_df,
                               ["read_acc", "start", "stop", "fragment_id", "edit_distance"])

    dst_fq = tio.part(Path(dst_dir / stem), ".fused_reads.fq")
    n_written = tio.stream_filter_fq(src_fq, dst_fq, survivors)
    assert n_written == reads_kept, f"fq wrote {n_written} != {reads_kept} survivors"

    # truncated.fq: symlink to the (disjoint, unfiltered) source -- cf. umi_filter. Left UNDECLARED
    # by the rule for the same reason: a declared symlink invites cleanup to delete decode's real file.
    dst_trunc = tio.part(Path(dst_dir / stem), ".truncated.fq")
    if dst_trunc.exists() or dst_trunc.is_symlink():
        dst_trunc.unlink()
    if src_trunc.exists():
        os.symlink(os.path.abspath(src_trunc), dst_trunc)

    # ligateddna.log: copy + appended provenance block
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    orig_log = src_log.read_text() if src_log.exists() else ""
    median_cov = float(cov["cov"].median()) if reads_in else 0.0
    provenance = (
        "\n\n### filter_decode_coverage provenance ###\n"
        f"tool: {TOOL_VERSION}\n"
        f"timestamp_utc: {ts}\n"
        f"metric: see coverage_lib; measured upstream by `rule coverage` on the decode tree\n"
        f"params: min_coverage={args.min_coverage}\n"
        f"counts: reads_in={reads_in} dropped_lowcoverage={dropped_lowcoverage} "
        f"reads_kept={reads_kept} median_cov={round(median_cov, 4)}\n"
    )
    (tio.part(Path(dst_dir / stem), ".ligateddna.log")).write_text(orig_log + provenance)

    # ---- per-sample summary row (covfilt_summary concatenates these) -----------------------
    pct_dropped = round(100.0 * dropped_lowcoverage / reads_in, 4) if reads_in else 0.0
    row = {
        "sample": stem,
        "reads_in": reads_in,
        "dropped_lowcoverage": dropped_lowcoverage,
        "reads_kept": reads_kept,
        "pct_dropped": pct_dropped,
        "median_cov": round(median_cov, 4),
        "min_coverage": args.min_coverage,
    }
    pl.DataFrame([row]).write_csv(dst_dir / f"{stem}.coverage_summary.tsv", separator="\t")

    print(f"[{stem}] {reads_in:,} reads in -> {reads_kept:,} kept "
          f"({pct_dropped}% dropped below coverage {args.min_coverage}; "
          f"median coverage {median_cov:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
