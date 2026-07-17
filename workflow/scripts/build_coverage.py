#!/usr/bin/env python3
"""Measure per-read decode coverage for one sample; write <STEM>.coverage.parquet.

Coverage is a property of the DECODE step -- how much of each read the decoder actually explained --
so it is measured HERE, on the decode tree, and exactly once. Two consumers read the artifact and
neither recomputes it:

    filter_decode_coverage.py   selects survivors (cov >= decode_filter.min_coverage)
    build_featurecount_report_cross.py   renders the distribution

That split is what keeps the report honest. The report's own tree sits DOWNSTREAM of the coverage
filter, so measuring there would draw a distribution truncated at the threshold and read as a bug;
reading this PRE-filter record shows the real shape, with the cut drawn on it.

The metric itself lives in coverage_lib -- this script is only the driver that writes it down.

Usage:
    build_coverage.py --src-stem results/decode/NB01/NB01
    build_coverage.py --frags F.parquet --fq F.fq --out F.coverage.parquet
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import coverage_lib as cl
import tree_io as tio


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Per-sample decode coverage measurement.")
    ap.add_argument("--src-stem", default=os.environ.get("L2F_STEM"),
                    help="path prefix; reads <stem>.fragments.parquet + <stem>.fused_reads.fq and "
                         "writes <stem>.coverage.parquet (default $L2F_STEM)")
    ap.add_argument("--frags", help="override: fragments parquet")
    ap.add_argument("--fq", help="override: fused_reads.fq (the read-length denominator)")
    ap.add_argument("--out", help="override: output coverage parquet")
    args = ap.parse_args(argv[1:])

    if args.src_stem:
        src_stem = Path(args.src_stem)
        if not src_stem.is_absolute():
            src_stem = Path.cwd() / src_stem
        frags = Path(args.frags) if args.frags else tio.part(src_stem, ".fragments.parquet")
        fq = Path(args.fq) if args.fq else tio.part(src_stem, ".fused_reads.fq")
        out = Path(args.out) if args.out else tio.part(src_stem, ".coverage.parquet")
    elif args.frags and args.fq and args.out:
        frags, fq, out = Path(args.frags), Path(args.fq), Path(args.out)
    else:
        ap.error("--src-stem is required (or set $L2F_STEM), or pass --frags/--fq/--out")

    for p in (frags, fq):
        if not p.exists():
            print(f"[FATAL] missing input: {p}", file=sys.stderr)
            return 1

    cov = cl.read_coverage(frags, fq)
    cov.write_parquet(out)

    n = cov.height
    med = float(cov["cov"].median()) if n else 0.0
    print(f"[{out.name}] {n:,} reads; median coverage {med:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
