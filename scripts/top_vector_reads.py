#!/usr/bin/env python
"""Extract the last-round read sequences of the most-enriched fitness vectors.

STANDALONE per-study analysis -- NOT part of the Snakemake DAG. It reads the
already-produced fitness tables under ``results/fitness/``, the per-round read maps under
``results/fragvec/<sample>/`` and the decoded FASTQs under ``results/decode/<sample>/``,
and writes FASTQ (+ a ranking sidecar) under ``results/fitness/top_reads/``. Nothing here
mutates the pipeline, its config, or any existing ``results/`` tree.

What it computes
----------------
For each subject, ``fitness_step_ratios.py`` wrote
``<subject>_vs_<control>.log2_step_ratios.tsv`` -- one row per fragment-vector with
empty-corrected ``log2_R{c}_vs_R{p}`` step columns. Here each vector is ranked by::

    log2_sum[v] = sum over all log2_* step columns of that row

descending, as-is (no read-count filter). The top ``--top-n`` vectors per subject are
selected, and every read assigned to those vectors **in the subject's last selection round**
is emitted as FASTQ.

Why the FASTQ join
------------------
``<sample>.fragvec_reads.parquet`` (columns ``read_acc, ont_read, vector, umi``) maps reads
to vectors but holds no sequence. The sequence lives in
``<sample>.fused_reads.fq``, whose header's first whitespace token equals the parquet
``read_acc`` exactly (``<start>:<end>|<uuid>``). So we take the top vectors' ``read_acc`` from
the parquet and stream the FASTQ to pull the sequences -- and their QUALITY, which is copied
through verbatim, so the output can be fed straight to a consensus caller (NGSpeciesID
``--fastq``). Only ``strand=`` is kept from the decode header tail; see ``fastq_io`` for why
pychopper's raw ``umi1=``/``umi2=`` are dropped.

CLI (all defaulted, so it runs with no args from the repo root)
---------------------------------------------------------------
    pixi run -e default python scripts/top_vector_reads.py \
      --fitness-dir results/fitness \
      --fragvec-dir results/fragvec \
      --decode-dir  results/decode \
      --subjects    A549,HEK293,RAW \
      --control     empty \
      --top-n       1 \
      --out-dir     results/fitness/top_reads

Emits (under --out-dir, per subject):
    <subject>_vs_<control>.top<N>.fq             reads of the top-N vectors (last round)
    <subject>_vs_<control>.top<N>.ranking.tsv    rank/vector/log2_sum/read counts
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fastq_io import read_fastq_records, strand_field, write_fastq   # noqa: E402  one FASTQ reader

_ROUND_RE = re.compile(r"_R(\d+)$")


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# ranking
# --------------------------------------------------------------------------- #
def rank_vectors(fitness_tsv: Path, top_n: int) -> pl.DataFrame:
    """Top-``top_n`` vectors of one ``<subject>_vs_<control>`` table, ranked by the sum
    of all ``log2_*`` step columns (descending). Returns columns
    ``rank, vector, log2_sum, subject_max_reads`` (rank is 1-based)."""
    df = pl.read_csv(fitness_tsv, separator="\t")
    log_cols = [c for c in df.columns if c.startswith("log2_")]
    if not log_cols:
        raise SystemExit(f"[FATAL] {fitness_tsv}: no log2_* columns to rank by")
    extra = ["subject_max_reads"] if "subject_max_reads" in df.columns else []
    ranked = (
        df.with_columns(
            pl.sum_horizontal([pl.col(c) for c in log_cols]).round(4).alias("log2_sum")
        )
        .sort("log2_sum", descending=True)
        .head(top_n)
        .with_row_index("rank", offset=1)
        .select(["rank", "vector", "log2_sum", *extra])
    )
    return ranked


# --------------------------------------------------------------------------- #
# last round + read extraction
# --------------------------------------------------------------------------- #
def last_round_sample(subject: str, fragvec_dir: Path) -> str:
    """The subject's highest-numbered ``<subject>_R<n>`` round-sample folder name."""
    rounds: list[tuple[int, str]] = []
    for d in fragvec_dir.glob(f"{subject}_R*"):
        if not d.is_dir():
            continue
        m = _ROUND_RE.search(d.name)
        if m:
            rounds.append((int(m.group(1)), d.name))
    if not rounds:
        raise SystemExit(
            f"[FATAL] no {subject}_R<n> round folders under {fragvec_dir}"
        )
    return max(rounds)[1]


def read_acc_map(
    reads_parquet: Path, ranked: pl.DataFrame
) -> tuple[dict[str, tuple[int, str]], dict[str, int]]:
    """``read_acc -> (rank, vector)`` for the ranked vectors in the last-round parquet,
    plus ``vector -> read_count`` in that round (for reporting / 0-read warnings)."""
    vec_rank = dict(ranked.select(["vector", "rank"]).iter_rows())
    reads = pl.read_parquet(reads_parquet, columns=["read_acc", "vector"]).filter(
        pl.col("vector").is_in(list(vec_rank))
    )
    acc_map = {
        acc: (vec_rank[vec], vec)
        for acc, vec in reads.select(["read_acc", "vector"]).iter_rows()
    }
    counts = dict(
        reads.group_by("vector").agg(pl.len().alias("n")).iter_rows()
    )
    return acc_map, counts


def emit_fastq(
    records: list[tuple[int, str, str, str, str, str]], path: Path
) -> int:
    """Write FASTQ from (rank, read_acc, vector, header_tail, sequence, quality) rows, ordered
    by (rank, read_acc). Header carries provenance; the quality is decode's, verbatim.
    Returns record count."""
    rows = sorted(records, key=lambda r: (r[0], r[1]))
    return write_fastq(path, (
        (f"{acc} rank={rank} vector={vector} {strand_field(tail)}".rstrip(), seq, qual)
        for rank, acc, vector, tail, seq, qual in rows
    ))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fitness-dir", default="results/fitness", type=Path)
    ap.add_argument("--fragvec-dir", default="results/fragvec", type=Path)
    ap.add_argument("--decode-dir", default="results/decode", type=Path)
    ap.add_argument("--subjects", default="A549,HEK293,RAW",
                    help="comma-separated subject groups")
    ap.add_argument("--control", default="empty",
                    help="control suffix in <subject>_vs_<control> table names")
    ap.add_argument("--top-n", type=int, default=1,
                    help="number of top-ranked vectors per subject to extract")
    ap.add_argument("--out-dir", default="results/fitness/top_reads", type=Path)
    args = ap.parse_args(argv[1:])

    subjects = [s for s in args.subjects.split(",") if s]
    control = args.control
    n = args.top_n
    if n < 1:
        raise SystemExit("[FATAL] --top-n must be >= 1")

    for subj in subjects:
        tag = f"{subj}_vs_{control}"
        fitness_tsv = args.fitness_dir / f"{tag}.log2_step_ratios.tsv"
        if not fitness_tsv.exists():
            log(f"[WARN] {subj}: missing {fitness_tsv}; skipping")
            continue

        ranked = rank_vectors(fitness_tsv, n)
        last = last_round_sample(subj, args.fragvec_dir)
        reads_p = args.fragvec_dir / last / f"{last}.fragvec_reads.parquet"
        fq_p = args.decode_dir / last / f"{last}.fused_reads.fq"
        for p in (reads_p, fq_p):
            if not p.exists():
                raise SystemExit(f"[FATAL] {subj}: missing {p}")

        acc_map, vec_counts = read_acc_map(reads_p, ranked)
        for vec, rk in ranked.select(["vector", "rank"]).iter_rows():
            if vec_counts.get(vec, 0) == 0:
                log(f"[WARN] {subj}: rank {rk} vector has 0 reads in {last} "
                    f"({vec}); nothing to emit for it")

        seqs = read_fastq_records(fq_p, set(acc_map))
        missing = len(acc_map) - len(seqs)
        if missing:
            log(f"[WARN] {subj}: {missing}/{len(acc_map)} read_acc from {last} "
                f"fragvec_reads not found in {fq_p.name}")

        records = [
            (rank, acc, vec, *seqs[acc])
            for acc, (rank, vec) in acc_map.items()
            if acc in seqs
        ]
        fq_out = args.out_dir / f"{tag}.top{n}.fq"
        n_written = emit_fastq(records, fq_out)
        log(f"[INFO] wrote {fq_out}  ({n_written} reads from {last}, "
            f"top {ranked.height} vector(s))")

        # ranking sidecar: rank/vector/log2_sum/subject_max_reads/last_round/last_round_reads
        sidecar = ranked.with_columns(
            pl.lit(last).alias("last_round"),
            pl.col("vector").replace_strict(vec_counts, default=0)
              .alias("last_round_reads"),
        )
        sidecar_p = args.out_dir / f"{tag}.top{n}.ranking.tsv"
        sidecar_p.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_csv(sidecar_p, separator="\t")
        log(f"[INFO] wrote {sidecar_p}  ({sidecar.height} rows x {sidecar.width} cols)")

    log("[INFO] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
