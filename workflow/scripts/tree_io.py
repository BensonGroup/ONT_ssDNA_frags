#!/usr/bin/env python3
"""Sample-tree IO: subset a decode-shaped sample directory to a set of surviving reads.

Every filtering stage in this workflow does the same mechanical thing — pick a set of `read_acc`s,
then mirror the sample tree keeping only those reads — and each does it for its own reason:

    covfilt/    survivors = coverage >= min_coverage        (filter_decode_coverage.py)
    umifilt/    survivors = umi_class == "keep"             (umi_filter.py)
    umi/        survivors = umi_class == "keep" + payload   (umi_correct.py)

Nothing here knows what a UMI, a coverage or a ligand is. The only domain concept is `read_acc`,
which is the DECODE step's read identifier and so belongs to every stage equally — "filtered" below
means "subset to the given read_acc set", nothing more. These helpers used to live in the UMI dedup
driver, which meant the upstream coverage filter had to import the downstream UMI stage just to copy
a parquet. They are neutral, so they live in a neutral module.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import polars as pl


def part(stem: Path, ext: str) -> Path:
    """`<dir>/<name>` + ext -> `<dir>/<name><ext>` (a stem is a path prefix, not a real file)."""
    return stem.parent / f"{stem.name}{ext}"


def read_acc_from_header(header: str) -> str:
    """read_acc = the FASTQ header's first whitespace token, minus the leading '@'."""
    return header[1:].split(None, 1)[0]


def prepare_dst(dst_dir: Path, force: bool) -> None:
    """Create dst_dir; fail loudly if it already holds files unless --force (then wipe them)."""
    if dst_dir.exists():
        entries = list(dst_dir.iterdir())
        if entries and not force:
            names = ", ".join(sorted(p.name for p in entries)[:8])
            raise SystemExit(
                f"[FATAL] destination {dst_dir} is not empty ({len(entries)} entr(y/ies): {names} ...).\n"
                f"        Refusing to overwrite a previous/partial run. Pass --force to overwrite.")
        for p in entries:
            if p.is_dir() and not p.is_symlink():
                shutil.rmtree(p)
            else:
                p.unlink()
    dst_dir.mkdir(parents=True, exist_ok=True)


def write_filtered_parquet(src: Path, dst: Path, survivors: pl.DataFrame, sort_by: list[str]) -> None:
    """Filter src parquet to survivor read_accs (semi-join), sort deterministically, write.

    `survivors` is a one-column read_acc frame. Fragment/UMI parquets are one-to-many in read_acc,
    so the semi-join drops ALL rows of a dropped read — that is what keeps the read-level unit
    uniform across every artifact in a tree.
    """
    df = (
        pl.scan_parquet(src)
        .join(survivors.lazy(), on="read_acc", how="semi")
        .sort(sort_by)
        .collect()
    )
    df.write_parquet(dst)


def stream_filter_fq(src_fq: Path, dst_fq: Path, survivors: set[str]) -> int:
    """Copy only survivor records from src_fq to dst_fq, preserving original record order.

    Streams 4-line records; does NOT load the whole FASTQ into RAM as a list.
    """
    n = 0
    with open(src_fq, "rt") as fin, open(dst_fq, "wt") as fout:
        while True:
            header = fin.readline()
            if not header:
                break
            seq = fin.readline()
            plus = fin.readline()
            qual = fin.readline()
            if read_acc_from_header(header) in survivors:
                fout.write(header)
                fout.write(seq)
                fout.write(plus)
                fout.write(qual)
                n += 1
    return n
