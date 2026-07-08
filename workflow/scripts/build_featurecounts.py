#!/usr/bin/env python
"""Per-read feature counts for the 2026 initial library (LigandFrags2 umi-anchor-window decode).

Adapted from workflow/build_featurecounts.py for the single-sample, UMI-anchor-window Parquet
outputs in ligand2frags_new/. Differences from the NB/clean-set original:

  * ONE sample, no per-barcode split -- the tidy-long key is (read_acc, fragment_id), no `barcode`.
  * The construct vocabulary is NOT hard-coded: which features are backbone/anchor (vs ligand
    payload) is read from the tool's own `*.umis.parquet` anchor_name set (see ligand_vocab.py).
  * Source is Parquet, so `--check` validates the long table against `*.fragments.parquet`
    directly (there is no raw CSV to tally against).

Emits a TIDY LONG feature count -- read_acc | fragment_id | count (UInt32), one row per
(read_acc, fragment_id), counting duplicate detections within a read. Features are counted
EXACTLY as the tool emits them (incl `_rc` orientation and the UMI anchors).

    Reads:  $L2F_STEM.fragments.parquet   (read_acc, fragment_id, start, stop, edit_distance)
            $L2F_STEM.umis.parquet        (anchor_name -> backbone set, for the payload split)
    Writes: $L2F_STEM.featurecount.parquet         read_acc | fragment_id | count
            $L2F_STEM.featurecount.summary.tsv      one-row run summary

    L2F_STEM default = ligand2frags_new/2026initial_library ; override to point at another decode.

Run:    pixi run python workflow/build_featurecounts_2026.py
Verify: pixi run python workflow/build_featurecounts_2026.py --check
"""
import os
import sys
from pathlib import Path

import polars as pl

from ligand_vocab import anchor_set

ROOT = Path(__file__).resolve().parents[1]
STEM = Path(os.environ.get("L2F_STEM", ROOT / "ligand2frags_new" / "2026initial_library"))
if not STEM.is_absolute():
    STEM = ROOT / STEM


def part(ext: str) -> Path:
    return STEM.parent / f"{STEM.name}{ext}"


FRAGS = part(".fragments.parquet")
UMIS = part(".umis.parquet")
FC = part(".featurecount.parquet")
SUMM = part(".featurecount.summary.tsv")


def featurecount_lazy(pq_path: Path) -> pl.LazyFrame:
    """Lazy tidy-long count: one row per (read_acc, fragment_id) with a duplicate-detection count."""
    return (
        pl.scan_parquet(pq_path)
        .group_by("read_acc", "fragment_id")
        .agg(pl.len().alias("count").cast(pl.UInt32))
    )


def anchor_ids() -> list[str]:
    """Backbone/anchor fragment_ids (both orientations), derived from the tool's umis parquet."""
    anchors = anchor_set(UMIS)
    return sorted(set(anchors) | {f"{a}_rc" for a in anchors})


def build() -> dict:
    """Write the feature-count Parquet; return a one-row run summary (payload vs backbone split)."""
    featurecount_lazy(FRAGS).sink_parquet(FC)
    anc = anchor_ids()
    lf = pl.scan_parquet(FC)

    overall = lf.select(
        n_reads=pl.col("read_acc").n_unique(),
        n_features=pl.col("fragment_id").n_unique(),
        n_pairs=pl.len(),
        n_fragment_rows=pl.col("count").sum(),
    ).collect().to_dicts()[0]

    payload = lf.filter(~pl.col("fragment_id").is_in(anc)) if anc else lf
    pl_stats = payload.select(
        n_payload_features=pl.col("fragment_id").n_unique(),
        n_payload_pairs=pl.len(),
        n_payload_rows=pl.col("count").sum(),
        n_reads_with_payload=pl.col("read_acc").n_unique(),
    ).collect().to_dicts()[0]

    n_reads = max(overall["n_reads"], 1)
    row = {
        "sample": STEM.name,
        "n_reads": overall["n_reads"],
        "n_reads_with_payload": pl_stats["n_reads_with_payload"],
        "n_features": overall["n_features"],
        "n_payload_features": pl_stats["n_payload_features"],
        "n_anchor_features": len(anc),
        "n_pairs": overall["n_pairs"],
        "n_fragment_rows": int(overall["n_fragment_rows"]),
        "n_payload_rows": int(pl_stats["n_payload_rows"]),
        "mean_payload_copies_per_read": round(pl_stats["n_payload_rows"] / n_reads, 4),
        "mean_payload_copies_per_read_with": round(
            pl_stats["n_payload_rows"] / max(pl_stats["n_reads_with_payload"], 1), 4),
    }
    pl.DataFrame([row]).write_csv(SUMM, separator="\t")
    return row


def check() -> int:
    """Validate the long table against the raw fragments parquet. Returns 0 if all pass."""
    if not FC.exists():
        print(f"missing {FC} -- run the build first", file=sys.stderr)
        return 1
    fc = pl.scan_parquet(FC)
    fr = pl.scan_parquet(FRAGS)

    got_total = int(fc.select(pl.col("count").sum()).collect().item())
    want_total = int(fr.select(pl.len()).collect().item())
    got_reads = int(fc.select(pl.col("read_acc").n_unique()).collect().item())
    want_reads = int(fr.select(pl.col("read_acc").n_unique()).collect().item())

    # Per-feature totals must reproduce a plain tally of the raw fragments parquet.
    a = fc.group_by("fragment_id").agg(pl.col("count").sum().alias("c_fc")).collect()
    b = fr.group_by("fragment_id").agg(pl.len().alias("c_raw")).collect()
    j = a.join(b, on="fragment_id", how="full", coalesce=True).fill_null(0)
    mism = j.filter(pl.col("c_fc") != pl.col("c_raw"))

    ok = True

    def report(name, good, detail=""):
        nonlocal ok
        ok = ok and good
        print(f"  [{'OK ' if good else 'BAD'}] {name}{(': ' + detail) if detail else ''}")

    print(f"check {STEM.name}:")
    report("sum(count) == fragments rows", got_total == want_total,
           f"{got_total:,} vs {want_total:,}")
    report("distinct reads match", got_reads == want_reads,
           f"{got_reads:,} vs {want_reads:,}")
    report("every per-feature total matches raw tally", mism.height == 0,
           f"{mism.height} feature(s) differ" if mism.height else f"{j.height} features")
    print(f"  => {'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[1] == "--check":
        return check()
    if not FRAGS.exists():
        print(f"fragments parquet not found: {FRAGS}", file=sys.stderr)
        return 1
    row = build()
    print(f"[{row['sample']}] {FRAGS.name} -> {FC.name}")
    print(f"  {row['n_reads']:>10,} reads decoded  "
          f"({row['n_reads_with_payload']:,} with >=1 payload fragment)")
    print(f"  {row['n_payload_rows']:>10,} payload calls across "
          f"{row['n_payload_features']} payload features "
          f"(+{row['n_anchor_features']} anchor features)")
    print(f"  {row['mean_payload_copies_per_read']:>10} payload copies / decoded read")
    print(f"  -> {FC}  ({FC.stat().st_size / 1e6:.1f} MB)")
    print(f"  -> {SUMM}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
