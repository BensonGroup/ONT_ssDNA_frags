#!/usr/bin/env python
"""Per-sample fragment-vector quantification: the TWO-AXIS readout (abundance + incidence).

A read's **fragment vector** is its sorted multiset of PAYLOAD `fragment_id`s -- backbone/anchor
features excluded (every read carries them, so they carry no information about which ligand was
decoded), `_rc` kept DISTINCT (a reverse-ligated fragment is a different species). That vector IS the
library member, and it is where the statistical power lives: ~4-10 reads per vector, versus ~1.0 reads
per UMI in this data.

Per sample and vector we count two axes (the CRISPR-UMI convention, Michlits et al. Nat Methods 2017):

  n_reads            ABUNDANCE  -- how many reads carry this vector
  n_umis             INCIDENCE  -- how many INDEPENDENT clones (distinct UMIs) carry it
  max_reads_per_umi  the jackpot detector

The pair is what makes a call trustworthy: 12,000 reads spread over 12,000 clones is reproducible
selection; 12,000 reads from ONE clone is a single amplified lineage (or a PCR jackpot) and must not
be read as enrichment.

`rpm` is the DEPTH-NORMALISED FREQUENCY (reads per million payload reads) -- the only value that is
comparable ACROSS samples, and what the trajectories are built on. It deliberately does NOT scale by the
sample's observed richness: Kuai et al.'s `count x total_diversity / total_reads` uses total_diversity =
the LIBRARY SIZE, a design constant identical in every sample. Substituting the OBSERVED per-sample
richness is wrong twice -- it differs per sample (so it distorts the very cross-sample comparison it is
meant to enable; the factor swings 4.5x across A549 here) and it is itself depth-dependent.

Counts below --min-reads are FLAGGED (`is_noise`), never dropped -- a vector at 3 reads in R1 and 800 in
R5 is exactly the hit you want.

Also emits SUB-VECTOR marginals (single fragments and fragment pairs) as a lossy DENOISING view --
a supplement to the full-vector counts, never a replacement (Faver et al. 2019: the full vector is
the library member; aggregation loses the individual molecule).

CLI:
    build_fragvec.py --src-stem DECODE_STEM --umi-fused PATH --umi-col COL --dst-dir DIR
                     [--min-reads 10] [--max-frags-for-pairs 12] [--force]
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fragvec_lib as fvl              # clonotype construction, shared with the jackpot detector
from ligand_vocab import anchor_set  # authoritative backbone/anchor names (from the umis parquet)
from complexity_lib import c_eff, collision_prob     # depth-robust effective complexity (see complexity_lib docstring)


def part(stem: Path, ext: str) -> Path:
    return stem.parent / f"{stem.name}{ext}"


# Per-read provenance map: which reads produced each vector. `read_vectors` already builds this
# (read_acc, vector, umi) and the aggregates throw it away -- persisting it makes vector -> read a
# forward lookup. `ont_read` is the read_acc after the `|` (the ONT read id: read_acc is
# `<start:stop>|<ont_read>`), so it joins straight to the FASTQ header; read_acc is kept verbatim so
# the segment span survives.
READS_SCHEMA = {"read_acc": pl.String, "ont_read": pl.String,
                "vector": pl.String, "umi": pl.String}


def _write_reads(rv, path: Path) -> None:
    (rv.with_columns(pl.col("read_acc").str.split("|").list.last().alias("ont_read"))
       .select("read_acc", "ont_read", "vector", "umi")
       .sort(["vector", "read_acc"])              # group a vector's reads together; deterministic
       .write_parquet(path))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Per-sample fragment-vector two-axis quantification.")
    ap.add_argument("--src-stem", required=True,
                    help="stem supplying <stem>.fragments.parquet + <stem>.umis.parquet (the umi/ stem "
                         "when using corrected UMIs -- umi_correct carries the fragment calls through "
                         "unchanged -- else the decode stem)")
    ap.add_argument("--umis", default=None,
                    help="umis.parquet for the anchor vocabulary (default: <src-stem>.umis.parquet). "
                         "Needed because the umi/ tree does not re-emit umis.parquet.")
    ap.add_argument("--umi-fused", required=True,
                    help="fused parquet supplying the per-read UMI (umi/ corrected, or decode/ raw)")
    ap.add_argument("--umi-col", default="fused_seq_corrected",
                    help="UMI column in --umi-fused (fused_seq_corrected | fused_seq)")
    ap.add_argument("--dst-dir", required=True)
    ap.add_argument("--min-reads", type=int, default=10,
                    help="counts below this are FLAGGED as noise (not dropped)")
    ap.add_argument("--max-frags-for-pairs", type=int, default=12,
                    help="reads with more distinct payload fragments are skipped in the PAIR marginal")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv[1:])

    src = Path(args.src_stem).resolve()
    stem = src.name
    dst = Path(args.dst_dir).resolve()
    frags_pq = part(src, ".fragments.parquet")
    umis_pq = Path(args.umis).resolve() if args.umis else part(src, ".umis.parquet")
    umi_pq = Path(args.umi_fused).resolve()
    for p in (frags_pq, umis_pq, umi_pq):
        if not p.exists():
            print(f"[FATAL] missing input: {p}", file=sys.stderr)
            return 1
    if dst.exists() and any(dst.iterdir()) and not args.force:
        print(f"[FATAL] {dst} not empty; pass --force", file=sys.stderr)
        return 1
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    # ---- the fragment vector ---------------------------------------------------------------------
    # Built by fragvec_lib -- the same code the jackpot detector cuts on, so the ratio a clonotype is
    # flagged for and the counts reported here cannot drift. See that module for why the vector is in
    # DECODED (positional) order with anchors included, and which reads it keeps.
    #
    # Exclusions belong to ANALYSES, not to the record, and each is DECLARED in the summary:
    #   * WHICH READS are kept: anchor-only reads (no payload) are dropped -> reads_dropped_anchor_only
    #   * sub-vector marginals are payload-only and order-agnostic -> marginals_exclude_anchors
    anchors = anchor_set(umis_pq)
    anc_ids = fvl.anchor_ids(anchors)
    rv, reads_decoded = fvl.read_vectors(frags_pq, umi_pq, args.umi_col, anc_ids)
    reads_payload = rv.height

    # payload-only fragment rows: the sub-vector marginals below are the only thing that needs them
    fr = pl.read_parquet(frags_pq, columns=["read_acc", "fragment_id", "start", "stop"])
    pay = fr.filter(~pl.col("fragment_id").is_in(anc_ids))
    umi = (pl.read_parquet(umi_pq, columns=["read_acc", args.umi_col])
           .rename({args.umi_col: "umi"}))

    if reads_payload == 0:
        print(f"[{stem}] no payload-bearing reads", file=sys.stderr)
        pl.DataFrame(schema={"vector": pl.String, "n_reads": pl.UInt32, "n_umis": pl.UInt32,
                             "max_reads_per_umi": pl.UInt32, "rpm": pl.Float64,
                             "is_noise": pl.Boolean}).write_parquet(dst / f"{stem}.fragvec_counts.parquet")
        pl.DataFrame(schema={"level": pl.String, "feature": pl.String,
                             "n_reads": pl.UInt32, "n_umis": pl.UInt32}
                     ).write_parquet(dst / f"{stem}.subvec_counts.parquet")
        pl.DataFrame(schema={"vector": pl.String, "umi": pl.String, "n_reads": pl.UInt32}
                     ).write_parquet(dst / f"{stem}.fragvec_clones.parquet")
        pl.DataFrame(schema=READS_SCHEMA).write_parquet(dst / f"{stem}.fragvec_reads.parquet")
        pl.DataFrame([{"sample": stem, "reads_decoded": reads_decoded, "reads_payload": 0,
                       "n_vectors": 0}]).write_csv(dst / f"{stem}.fragvec_summary.tsv", separator="\t")
        return 0

    # ---- TWO AXES: abundance (n_reads) and incidence (n_umis, = independent clones) --------------
    per_vu = fvl.vector_umi_counts(rv)
    per_v = fvl.per_vector_counts(per_vu, args.min_reads)
    n_vec = per_v.height
    total = int(per_v["n_reads"].sum())
    per_v.write_parquet(dst / f"{stem}.fragvec_counts.parquet")

    # Clone membership for the SIGNAL-CARRYING vectors only (n_reads >= min_reads) -- and the
    # substrate the jackpot blacklist is built from.
    keep = per_v.filter(~pl.col("is_noise")).select("vector")
    fvl.clone_membership(per_vu, keep).write_parquet(dst / f"{stem}.fragvec_clones.parquet")

    # Per-read provenance for the WHOLE payload-bearing set (not just signal vectors): the exact
    # (read_acc, vector, umi) map `read_vectors` produced. Aggregating it by vector reproduces
    # fragvec_counts (n_reads = row count, n_umis = distinct umi), so it cannot drift from the counts.
    _write_reads(rv, dst / f"{stem}.fragvec_reads.parquet")

    # ---- SUB-VECTOR marginals (lossy denoising view; NOT a replacement) --------------------------
    # Always PAYLOAD-only, regardless of --include-anchors: the anchors are carried by every read, so a
    # marginal over them is a constant (n_reads = all reads) and tells you nothing, and including them
    # would inflate the pair combinatorics for no gain.
    pu = pay.join(umi, on="read_acc", how="inner")
    singles = (
        pu.group_by("fragment_id").agg(
            pl.col("read_acc").n_unique().cast(pl.UInt32).alias("n_reads"),
            pl.col("umi").n_unique().cast(pl.UInt32).alias("n_umis"),
        ).rename({"fragment_id": "feature"}).with_columns(pl.lit("single").alias("level"))
    )
    # pairs: self-join per read over DISTINCT payload fragments (k^2 rows/read, so cap k)
    dist = pu.select("read_acc", "fragment_id", "umi").unique()
    kper = dist.group_by("read_acc").agg(pl.len().alias("k"))
    ok = kper.filter(pl.col("k") <= args.max_frags_for_pairs).select("read_acc")
    n_skipped = kper.filter(pl.col("k") > args.max_frags_for_pairs).height
    d2 = dist.join(ok, on="read_acc", how="semi")
    pairs = (
        d2.join(d2.select("read_acc", pl.col("fragment_id").alias("f2")), on="read_acc")
        .filter(pl.col("fragment_id") < pl.col("f2"))
        .with_columns((pl.col("fragment_id") + "|" + pl.col("f2")).alias("feature"))
    )
    pair_counts = (
        pairs.group_by("feature").agg(
            pl.col("read_acc").n_unique().cast(pl.UInt32).alias("n_reads"),
            pl.col("umi").n_unique().cast(pl.UInt32).alias("n_umis"),
        ).with_columns(pl.lit("pair").alias("level"))
        if pairs.height else
        pl.DataFrame(schema={"feature": pl.String, "n_reads": pl.UInt32,
                             "n_umis": pl.UInt32, "level": pl.String})
    )
    (pl.concat([singles.select("level", "feature", "n_reads", "n_umis"),
                pair_counts.select("level", "feature", "n_reads", "n_umis")])
     .sort(["level", "n_reads"], descending=[False, True])
     .write_parquet(dst / f"{stem}.subvec_counts.parquet"))
    if n_skipped:
        print(f"  [pairs] skipped {n_skipped:,} reads with > {args.max_frags_for_pairs} distinct "
              f"payload fragments", file=sys.stderr)

    # ---- summary ---------------------------------------------------------------------------------
    ge = per_v.filter(~pl.col("is_noise"))
    top = int(per_v["n_reads"].max())
    # DEPTH WARNING: n_vectors (and pct_reads_in_vectors_ge_min) are RICHNESS counts -- they grow with
    # sequencing depth, so they must NOT be compared across samples raw. c_eff is depth-robust; the
    # cross-sample report additionally rarefies every sample to a common depth. See complexity_lib.
    _counts = per_v["n_reads"].to_list()
    _N, _D, _pairs, _C, _lower = c_eff(_counts)
    _pc = collision_prob(_counts)
    row = {
        "sample": stem,
        "reads_decoded": reads_decoded,
        "reads_payload": reads_payload,
        "n_vectors": n_vec,
        "mean_reads_per_vector": round(total / n_vec, 3),
        "max_reads_vector": top,
        "top_vector_read_share_pct": round(100.0 * top / total, 3),
        "n_vectors_ge_min": ge.height,
        "pct_reads_in_vectors_ge_min": round(100.0 * int(ge["n_reads"].sum() or 0) / total, 2),
        "min_reads": args.min_reads,
        "umi_source": args.umi_col,
        # --- depth-robust complexity (compare THESE across samples, not n_vectors) ---
        "dup_pairs": _pairs,
        "collision_prob": _pc,   # sum p_i^2 — the UNBIASED depth-robust quantity (c_eff is its reciprocal)
        "c_eff": round(_C, 1) if _C != float("inf") else None,
        "c_eff_is_lower_bound": _lower,
        "n_over_c_eff": round(_N / _C, 4) if _C not in (0, float("inf")) else None,
        # --- the record, and every declared exclusion (the `vector` column excludes NOTHING) ---
        "vector_order": "decoded_positional_5to3",   # NOT alphabetical: order is identity
        "vector_includes_anchors": True,
        "anchors": ";".join(anc_ids),
        "reads_dropped_anchor_only": reads_decoded - reads_payload,
        "marginals_exclude_anchors": True,
        "marginals_order_agnostic": True,
    }
    pl.DataFrame([row]).write_csv(dst / f"{stem}.fragvec_summary.tsv", separator="\t")

    print(f"[{stem}] {reads_payload:,} payload reads -> {n_vec:,} fragment vectors "
          f"(mean {total/n_vec:.1f} reads/vector, max {top:,}); "
          f"{ge.height:,} vectors >= {args.min_reads} reads carry "
          f"{row['pct_reads_in_vectors_ge_min']}% of reads")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
