#!/usr/bin/env python3
"""Fragment-vector (clonotype) construction: the single source of truth for the two-axis readout.

A read's **fragment vector** is its decoded fragment content in POSITIONAL order; that vector IS the
library member. Per vector this counts two axes (the CRISPR-UMI convention, Michlits et al. Nat
Methods 2017):

    n_reads            ABUNDANCE  -- how many reads carry this vector
    n_umis             INCIDENCE  -- how many INDEPENDENT clones (distinct UMIs) carry it
    max_reads_per_umi  the jackpot detector

Two stages need that table and they must not disagree about it:

    build_fragvec.py        the quantification itself, on the CLEANED umi tree
    detect_umi_jackpots.py  the jackpot cut, on the R0 library's CORRECTED tree

The detector used to read `fragvec_initial/`, which made detection depend on fragvec and forced the
whole jackpot step downstream of the branch it was supposed to clean. Both now build the table from
`fragments.parquet` + a fused parquet -- the pair `umi_correct` already emits -- so the number a
clonotype is cut on and the number fragvec reports are computed by the same code. Same reason
`coverage_lib` and `umi_filter_lib` exist.

Run the self-test:  pixi run -e default python workflow/scripts/fragvec_lib.py
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

# ORDER IS IDENTITY (see read_vectors): a ligated construct A-B is a different product from B-A.
VECTOR_SEP = "|"


def anchor_ids(anchors) -> list[str]:
    """The anchor vocabulary in both orientations -- `_rc` is a distinct species, not the same one."""
    return sorted(set(anchors) | {f"{a}_rc" for a in anchors})


def read_vectors(frags_pq: Path, umi_pq: Path, umi_col: str, anc_ids: list[str]):
    """(rv, reads_decoded) -- one row per payload-bearing read carrying a UMI: read_acc, vector, umi.

    The vector records the read's FULL decoded content in DECODED (POSITIONAL) ORDER -- 5'->3' along
    the oriented read -- with ANCHORS INCLUDED, duplicates preserved and `_rc` distinct. This is the
    ligation architecture, e.g.
        UMI1_anchor|Linear20ntStrand2|Linear3ntbulgeStrand2|Linear20ntStrand2|UMI2_anchor
    The anchors FLANK the payload; they are not adjacent. The same fragment may legitimately appear
    more than once at different positions.

    ORDER IS IDENTITY. The vector must NOT be alphabetically sorted -- doing so merges genuinely
    distinct ligation products (measured: it collapsed 16,705 real vectors down to 9,617 in
    A549_R5). Ties are broken by (start, stop, fragment_id) purely for determinism.

    WHICH READS are kept: anchor-only reads (no payload fragment) are dropped -- a read carrying
    nothing but the backbone decoded no ligand. `reads_decoded` is returned so a caller can declare
    how many it dropped; it counts reads in the fragments parquet, so that parquet MUST already be
    filtered to the same read set as the fused one or the count silently runs high.
    """
    fr = pl.read_parquet(frags_pq, columns=["read_acc", "fragment_id", "start", "stop"])
    reads_decoded = int(fr["read_acc"].n_unique())
    pay = fr.filter(~pl.col("fragment_id").is_in(anc_ids))

    umi = (pl.read_parquet(umi_pq, columns=["read_acc", umi_col])
           .rename({umi_col: "umi"}))

    payload_reads = pay.select("read_acc").unique()
    vec = (
        fr.join(payload_reads, on="read_acc", how="semi")        # full content, payload-bearing reads
        .sort(["read_acc", "start", "stop", "fragment_id"])      # DECODED ORDER along the read
        .group_by("read_acc", maintain_order=True)
        .agg(pl.col("fragment_id").str.join(VECTOR_SEP).alias("vector"))
    )
    rv = vec.join(umi, on="read_acc", how="inner")     # payload-bearing reads carrying a UMI
    return rv, reads_decoded


def vector_umi_counts(rv):
    """(vector, umi, c) -- reads per clone per clonotype. The substrate for both axes."""
    return rv.group_by(["vector", "umi"]).agg(pl.len().alias("c"))


def per_vector_counts(per_vu, min_reads: int):
    """(vector, n_reads, n_umis, max_reads_per_umi, rpm, is_noise), sorted by abundance desc.

    Caller must guard `per_vu` non-empty: `rpm` divides by the payload-read total.

    `rpm` is the DEPTH-NORMALISED FREQUENCY (reads per million payload reads) -- the only value
    comparable ACROSS samples, and what the trajectories are built on. It deliberately does NOT
    scale by this sample's distinct-vector count. Kuai et al.'s `count x total_diversity /
    total_reads` uses total_diversity = the LIBRARY SIZE, a design constant identical in every
    sample (it only sets where "enrichment = 1" falls). Substituting the OBSERVED per-sample
    richness would be wrong twice over: it differs per sample (so it distorts the very trajectory it
    is meant to make comparable -- the factor swings 4.5x across A549 here) and it is itself
    depth-dependent. A constant factor rescales the y-axis and changes nothing; a per-sample one
    corrupts the comparison.

    `is_noise` FLAGS, never drops -- a vector at 3 reads in R1 and 800 in R5 is exactly the hit you
    want.
    """
    per_v = (
        per_vu.group_by("vector").agg(
            pl.col("c").sum().cast(pl.UInt32).alias("n_reads"),        # abundance
            pl.len().cast(pl.UInt32).alias("n_umis"),                  # incidence: distinct clones
            pl.col("c").max().cast(pl.UInt32).alias("max_reads_per_umi"),
        )
        .sort(["n_reads", "vector"], descending=[True, False])
    )
    total = int(per_v["n_reads"].sum())
    return per_v.with_columns(
        (1e6 * pl.col("n_reads") / total).alias("rpm"),
        (pl.col("n_reads") < min_reads).alias("is_noise"),
    )


def clone_membership(per_vu, keep_vectors):
    """(vector, umi, n_reads) for the given vectors only -- which clones carry each clonotype.

    Needed to tell, across the two selection arms, whether agreement is clone-SHARED (weak:
    correlated ancestry from the pre-split amplified pool) or clone-INDEPENDENT (strong). Aggregates
    throw this away. Restricted to the signal-carrying vectors, which carry most of the reads.

    This is also what the jackpot blacklist is built from: the UMIs of a flagged clonotype.
    """
    return (per_vu.join(keep_vectors, on="vector", how="semi")
            .select("vector", "umi", pl.col("c").cast(pl.UInt32).alias("n_reads"))
            .sort(["vector", "umi"]))


# Empty-case schemas, so a sample with no payload-bearing read still writes a well-typed file.
COUNTS_SCHEMA = {"vector": pl.String, "n_reads": pl.UInt32, "n_umis": pl.UInt32,
                 "max_reads_per_umi": pl.UInt32, "rpm": pl.Float64, "is_noise": pl.Boolean}
CLONES_SCHEMA = {"vector": pl.String, "umi": pl.String, "n_reads": pl.UInt32}


def _selftest() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        frags, fused = d / "f.parquet", d / "u.parquet"

        # r1,r2,r3 -> vector "A|lig1|Z"; r4 -> "A|lig2|Z" reversed order is a DIFFERENT vector
        # r5 -> anchor-only (dropped); r6 -> payload but no UMI row (inner join drops it)
        pl.DataFrame(
            {
                "read_acc": ["r1", "r1", "r1", "r2", "r2", "r2", "r3", "r3", "r3",
                             "r4", "r4", "r4", "r5", "r6", "r6"],
                "fragment_id": ["A", "lig1", "Z"] * 3 + ["A", "lig2", "Z"] + ["A"] + ["A", "lig1"],
                "start": [0, 10, 20] * 4 + [0] + [0, 10],
                "stop": [10, 20, 30] * 4 + [10] + [10, 20],
            },
            schema={"read_acc": pl.String, "fragment_id": pl.String,
                    "start": pl.UInt64, "stop": pl.UInt64},
        ).write_parquet(frags)
        pl.DataFrame(
            {"read_acc": ["r1", "r2", "r3", "r4", "r5"],
             "fused_seq_corrected": ["U1", "U1", "U2", "U3", "U4"]},
            schema={"read_acc": pl.String, "fused_seq_corrected": pl.String},
        ).write_parquet(fused)

        anc = anchor_ids(["A", "Z"])
        assert anc == ["A", "A_rc", "Z", "Z_rc"], anc

        rv, reads_decoded = read_vectors(frags, fused, "fused_seq_corrected", anc)
        assert reads_decoded == 6, reads_decoded          # every read in the fragments parquet
        got = {r["read_acc"]: r["vector"] for r in rv.iter_rows(named=True)}
        assert got == {"r1": "A|lig1|Z", "r2": "A|lig1|Z", "r3": "A|lig1|Z",
                       "r4": "A|lig2|Z"}, got
        assert "r5" not in got, "anchor-only read must be dropped"
        assert "r6" not in got, "payload read with no UMI must be dropped by the inner join"
        # anchors are IN the vector -- they flank the payload
        assert all(v.startswith("A|") and v.endswith("|Z") for v in got.values())

        per_vu = vector_umi_counts(rv)
        per_v = per_vector_counts(per_vu, min_reads=3)
        rows = {r["vector"]: r for r in per_v.iter_rows(named=True)}

        # "A|lig1|Z": 3 reads over 2 clones (U1 twice, U2 once) -> max_reads_per_umi 2
        a = rows["A|lig1|Z"]
        assert (a["n_reads"], a["n_umis"], a["max_reads_per_umi"]) == (3, 2, 2), a
        assert a["is_noise"] is False                     # 3 >= min_reads
        # "A|lig2|Z": 1 read, 1 clone -> flagged noise, NOT dropped
        b = rows["A|lig2|Z"]
        assert (b["n_reads"], b["n_umis"], b["max_reads_per_umi"]) == (1, 1, 1), b
        assert b["is_noise"] is True
        assert per_v.height == 2, "noise vectors are flagged, never dropped"

        # rpm is over payload reads (4), not over reads_decoded (6)
        assert abs(a["rpm"] - 1e6 * 3 / 4) < 1e-6, a["rpm"]
        assert abs(sum(r["rpm"] for r in rows.values()) - 1e6) < 1e-6

        # abundance-descending sort
        assert per_v["vector"].to_list() == ["A|lig1|Z", "A|lig2|Z"]

        # clone membership, restricted to the signal-carrying vectors
        keep = per_v.filter(~pl.col("is_noise")).select("vector")
        cl = clone_membership(per_vu, keep)
        assert set(cl["vector"].to_list()) == {"A|lig1|Z"}, "noise vector must not appear"
        assert {(r["umi"], r["n_reads"]) for r in cl.iter_rows(named=True)} == {("U1", 2), ("U2", 1)}

        # the jackpot signal: ratio = n_reads / n_umis, per clonotype
        ratio = {v: r["n_reads"] / r["n_umis"] for v, r in rows.items()}
        assert abs(ratio["A|lig1|Z"] - 1.5) < 1e-9, ratio

    print("fragvec_lib self-test: ALL PASS")


if __name__ == "__main__":
    _selftest()
