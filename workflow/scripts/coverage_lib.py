#!/usr/bin/env python3
"""Per-read decode coverage: the fraction of a read that decoded fragments explain.

The SINGLE SOURCE OF TRUTH for the coverage metric. The decode coverage filter
(``filter_decode_coverage.py``) and the cross-sample fragment report
(``build_featurecount_report_cross.py``) both call ``read_coverage()``, so the number a read is
judged on and the number a figure draws cannot drift apart.

    coverage = sum(stop - start) over ALL fragment calls  /  post-trim read length

Every part of that formula is a decision, so each is pinned here rather than at a call site:

  numerator    ALL fragments, ANCHORS INCLUDED. Anchors are backbone features every read carries,
               but they are genuinely decoded bases; excluding them measures something else (a
               payload-only coverage sits far lower, where a 0.9 cut would reject nearly every
               read).

  no merging   The decoder emits NON-OVERLAPPING intervals by construction: its search recurses
               only into the GAPS between hits already found, and a ligand landing on an anchor
               flank is pruned by an explicit overlap check. A plain sum is therefore exact and
               interval merging is unnecessary. The clip to [0, 1] is a defensive guard against a
               malformed read (two anchors are not mutually overlap-checked), not load-bearing.

  denominator  The read's length in ``<stem>.fused_reads.fq`` -- the POST-TRIM, anchor-bracketed
               span the decoder actually covered. NOT ``fused.parquet``'s ``fused_len``, which is
               the length of the ~24nt concatenated UMI key and is not a read length at all.

  read universe  The FASTQ, not the fragments table. A read that decoded NO fragment at all has
               coverage 0.0 and must be visible as such -- for a filter, silently skipping it
               would let the worst reads through. So the join is fq LEFT fragments, with a null
               covered folded to 0.

Unlike ``umi_filter_lib`` this module cannot be dependency-free -- the metric is a join over two
files, so polars is imported. Run the self-test inside the env:

    pixi run -e default python workflow/scripts/coverage_lib.py
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

# Mirrored by config `decode_filter.min_coverage` and the --min-coverage CLI flag.
DEFAULT_MIN_COVERAGE = 0.9   # keep a read whose coverage is >= this; drop it when <


def fq_lengths(path) -> pl.DataFrame:
    """read_acc -> sequence length from a FASTQ (the post-trim read length denominator)."""
    data = Path(path).read_bytes()
    lines = data.split(b"\n")
    accs, lens = [], []
    for i in range(0, len(lines) - 1, 4):
        head = lines[i]
        if not head.startswith(b"@"):
            continue
        accs.append(head[1:].split(None, 1)[0].decode())
        lens.append(len(lines[i + 1]))
    return pl.DataFrame({"read_acc": accs, "fqlen": lens},
                        schema={"read_acc": pl.String, "fqlen": pl.Int64})


def read_coverage(frags_pq, fq_path) -> pl.DataFrame:
    """Per-read coverage over the FASTQ's reads.

    Returns columns ``read_acc, covered, fqlen, cov`` -- one row per read in ``fq_path``. Reads
    with no fragment call get ``covered = 0`` and ``cov = 0.0``; a zero-length read gets
    ``cov = 0.0`` rather than a division by zero.
    """
    lens = fq_lengths(fq_path)
    covered = (
        pl.scan_parquet(frags_pq)
        .group_by("read_acc")
        .agg(covered=(pl.col("stop") - pl.col("start")).sum().cast(pl.Int64))
        .collect()
    )
    return (
        lens.join(covered, on="read_acc", how="left")
        .with_columns(pl.col("covered").fill_null(0))
        .with_columns(cov=pl.when(pl.col("fqlen") > 0)
                      .then((pl.col("covered") / pl.col("fqlen")).clip(0.0, 1.0))
                      .otherwise(0.0))
        .sort("read_acc")
    )


def _selftest() -> None:
    import tempfile

    def _fq(recs):
        return "".join(f"@{a} extra\n{'A' * n}\n+\n{'I' * n}\n" for a, n in recs)

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        fq = d / "s.fused_reads.fq"
        pq = d / "s.fragments.parquet"

        # r_full  100nt, one 100nt call            -> 1.00  keep
        # r_090   100nt, two calls 50+40           -> 0.90  keep  (boundary: >= keeps)
        # r_089   100nt, two calls 50+39           -> 0.89  drop  (boundary: < drops)
        # r_none  100nt, NO fragment call          -> 0.00  drop  (left join, not skipped)
        # r_zero    0nt, no call                   -> 0.00  drop  (no ZeroDivisionError)
        fq.write_text(_fq([("r_full", 100), ("r_090", 100), ("r_089", 100),
                           ("r_none", 100), ("r_zero", 0)]))
        pl.DataFrame(
            {
                "read_acc": ["r_full", "r_090", "r_090", "r_089", "r_089"],
                "fragment_id": ["a", "a", "b", "a", "b"],
                "start": [0, 0, 50, 0, 50],
                "stop": [100, 50, 90, 50, 89],
                "edit_distance": [0, 0, 0, 0, 0],
            },
            schema={"read_acc": pl.String, "fragment_id": pl.String, "start": pl.UInt64,
                    "stop": pl.UInt64, "edit_distance": pl.Int32},
        ).write_parquet(pq)

        cov = read_coverage(pq, fq)
        got = {r["read_acc"]: r["cov"] for r in cov.iter_rows(named=True)}

        # every fq read present exactly once -- the fragments table does not define the universe
        assert cov.height == 5, cov.height
        assert set(got) == {"r_full", "r_090", "r_089", "r_none", "r_zero"}, got

        assert got["r_full"] == 1.0, got["r_full"]
        assert abs(got["r_090"] - 0.90) < 1e-12, got["r_090"]
        assert abs(got["r_089"] - 0.89) < 1e-12, got["r_089"]
        assert got["r_none"] == 0.0, got["r_none"]     # no fragment call -> 0, NOT absent
        assert got["r_zero"] == 0.0, got["r_zero"]     # 0nt read -> 0, not inf/nan

        # the boundary the config default sits on: >= keeps, < drops
        keep = set(cov.filter(pl.col("cov") >= DEFAULT_MIN_COVERAGE)["read_acc"].to_list())
        assert keep == {"r_full", "r_090"}, keep

        # covered is the plain sum (no interval merging) and the fq supplies the denominator
        row = {r["read_acc"]: r for r in cov.iter_rows(named=True)}["r_090"]
        assert row["covered"] == 90 and row["fqlen"] == 100, row

        # a read carrying MORE covered bases than its length clips to 1.0 rather than exceeding it
        fq2 = d / "t.fused_reads.fq"
        pq2 = d / "t.fragments.parquet"
        fq2.write_text(_fq([("r_over", 10)]))
        pl.DataFrame(
            {"read_acc": ["r_over"], "fragment_id": ["a"], "start": [0], "stop": [999],
             "edit_distance": [0]},
            schema={"read_acc": pl.String, "fragment_id": pl.String, "start": pl.UInt64,
                    "stop": pl.UInt64, "edit_distance": pl.Int32},
        ).write_parquet(pq2)
        assert read_coverage(pq2, fq2)["cov"].to_list() == [1.0]

    print("coverage_lib self-test: ALL PASS")


if __name__ == "__main__":
    _selftest()
