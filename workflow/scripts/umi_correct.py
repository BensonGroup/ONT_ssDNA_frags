#!/usr/bin/env python
"""Per-sample UMI QC-filter + ed<=1 correction (NO collapse) for the ligand2frags UMI branch.

Given one decoded sample (``<STEM>.fused.parquet`` + ``.umis.parquet``), this:

  STEP 1  QC-filters every read from its fused UMI sequence and its two UMI lengths, dropping
          garbage UMIs (non-ACGT / low complexity / bad length) via ``umi_filter_lib`` -- the same
          single, unit-tested classifier the umifilt stage uses (via ``umi_filter.build_per_read``).
  STEP 1b DROPS anchor-only reads (valid UMI, but no decoded PAYLOAD fragment). Abundance here
          counts only reads that actually decoded a ligand, so a read carrying nothing but the
          backbone/anchor features does not contribute to a molecule's read count. Anchors come
          from the umis parquet via ``ligand_vocab.anchor_set`` (both orientations, ``_rc``) -- the
          same authoritative source ``build_featurecounts.py`` uses.
  STEP 2  clusters the surviving UMIs with ``starcode -d 1 -s`` (spheres, Levenshtein distance 1)
          and CORRECTS each read's fused UMI to its cluster's most-abundant ed<=1 variant
          (starcode's centroid, col1). NO read is dropped -- every read that passes STEP 1 is
          written, just relabelled. This is the ONLY stage that runs starcode; the pipeline never
          collapses UMIs, so correction is what starcode is here for.
  STEP 3  writes, per sample:
            <STEM>.fused.parquet   one row per KEPT read: read_acc, strand, fused_seq (original),
                                   fused_len, fused_seq_corrected, umi_read_count (reads sharing
                                   this read's corrected UMI in this sample)
            <STEM>.umi_species.tsv corrected `fused_seq  n_reads` (the UMI x sample matrix substrate)
            <STEM>.umi_summary.tsv one-row filter/correction summary
          plus the starcode I/O (<STEM>.umi_counts.tsv / .umi_clusters.tsv) for provenance.

The UMI classification (``build_per_read`` / ``CLASS_ORDER``) is imported from ``umi_filter`` so the
two UMI stages cannot drift about what counts as garbage; the sample-tree IO comes from the neutral
``tree_io``. Deterministic: stable sorts throughout.

CLI:
    umi_correct.py --src-stem PATH --dst-dir DIR --qc-dir DIR
                   [--max-homopolymer 8] [--max-base-frac 0.7] [--starcode starcode] [--force]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tree_io as tio                 # stage-neutral sample-tree IO (part / prepare_dst / ...)
import umi_filter as ufilt            # the shared UMI classification: build_per_read / CLASS_ORDER
import umi_filter_lib as uf
from ligand_vocab import anchor_set  # authoritative backbone/anchor names (from the umis parquet)
from complexity_lib import c_eff, collision_prob     # depth-robust effective complexity (see complexity_lib docstring)

TOOL_VERSION = "umi_correct.py/1.0"

# starcode clustering is pinned to a SINGLE thread on purpose. Multithreaded starcode is
# functionally identical (same clusters, counts, centroids, row order) but emits the
# comma-separated col3 member SEQUENCES in a thread-schedule-dependent order, which makes the
# raw <STEM>.umi_clusters.tsv non-byte-reproducible across re-runs (empirically verified:
# repeated -t 8 runs on the same input DIFFER byte-for-byte, differing only in col3 member order;
# -t 1 runs are byte-identical). -t 1 costs ~0.5s/sample here (~600k unique UMIs). Downstream is
# member-order-invariant (every parquet is re-sorted by read_acc), so results are identical either
# way. This lives here because correction is the only stage that runs starcode.
STARCODE_THREADS = 1


def run_starcode(starcode: str, counts_tsv: Path, clusters_tsv: Path) -> list[str]:
    """Run starcode -d 1 -s --print-clusters -t 1; return the exact argv (for provenance).

    Pinned to one thread so the persisted raw cluster TSV is byte-reproducible (see note above).
    """
    argv = [starcode, "-d", "1", "-s", "--print-clusters",
            "-t", str(STARCODE_THREADS), "-i", str(counts_tsv), "-o", str(clusters_tsv)]
    res = subprocess.run(argv, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"starcode failed ({res.returncode}):\n{res.stderr}")
    return argv


def parse_cluster_centroids(clusters_tsv: Path):
    """Parse `starcode -s --print-clusters` output: list of (centroid_seq, total_count, [member_seqs]).

    col1 = centroid = the sphere's MOST-ABUNDANT sequence (the ed<=1 variant we correct members to),
    col2 = total_count, col3 = comma-separated member sequences.
    """
    rows = []
    with open(clusters_tsv, "rt") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"unexpected starcode line ({len(parts)} cols): {line!r}")
            rows.append((parts[0], int(parts[1]), parts[2].split(",")))
    return rows


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Per-sample UMI QC-filter + ed<=1 correction (keep all reads).")
    ap.add_argument("--src-stem", default=os.environ.get("L2F_STEM"),
                    help="path prefix; <src-stem>.fused.parquet / .umis.parquet must exist (default $L2F_STEM)")
    ap.add_argument("--dst-dir", required=True, help="output sample directory (umi/<sample>/)")
    ap.add_argument("--qc-dir", required=True, help="QC directory (== dst-dir; species/summary land here)")
    ap.add_argument("--max-homopolymer", type=int, default=uf.DEFAULT_MAX_HOMOPOLYMER)
    ap.add_argument("--max-base-frac", type=float, default=uf.DEFAULT_MAX_BASE_FRAC)
    ap.add_argument("--starcode", default=os.environ.get("L2F_STARCODE", "starcode"))
    ap.add_argument("--force", action="store_true", help="overwrite a non-empty destination sample dir")
    args = ap.parse_args(argv[1:])

    if not args.src_stem:
        ap.error("--src-stem is required (or set $L2F_STEM)")
    src_stem = Path(args.src_stem)
    if not src_stem.is_absolute():
        src_stem = Path.cwd() / src_stem
    stem = src_stem.name
    dst_dir = Path(args.dst_dir).resolve()
    qc_dir = Path(args.qc_dir).resolve()

    fused_pq = tio.part(src_stem, ".fused.parquet")
    umis_pq = tio.part(src_stem, ".umis.parquet")
    frags_pq = tio.part(src_stem, ".fragments.parquet")
    src_log = tio.part(src_stem, ".ligateddna.log")
    for p in (fused_pq, umis_pq, frags_pq):
        if not p.exists():
            print(f"[FATAL] missing input: {p}", file=sys.stderr)
            return 1

    tio.prepare_dst(dst_dir, args.force)
    qc_dir.mkdir(parents=True, exist_ok=True)

    # ---- STEP 1: QC filter -----------------------------------------------------------------
    perread = ufilt.build_per_read(fused_pq, umis_pq, args.max_homopolymer, args.max_base_frac)
    reads_in = perread.height
    class_counts = dict(perread.group_by("umi_class").agg(pl.len().alias("n")).iter_rows())
    dropped = {c: int(class_counts.get(c, 0)) for c in ufilt.CLASS_ORDER}
    dropped_total = sum(dropped.values())
    umi_ok = perread.filter(pl.col("umi_class") == "keep").select("read_acc", "fused_seq")
    reads_umi_ok = umi_ok.height
    assert dropped_total + reads_umi_ok == reads_in, f"{dropped_total} + {reads_umi_ok} != {reads_in}"

    # ---- STEP 1b: drop anchor-only reads (no decoded payload fragment) ----------------------
    # Abundance counts only reads that actually decoded a ligand. Anchors are the backbone features
    # every read carries (both orientations), so a read with nothing but anchors carries no ligand
    # and must not inflate a molecule's read count.
    anchors = anchor_set(umis_pq)
    anc_ids = sorted(set(anchors) | {f"{a}_rc" for a in anchors})
    frags = pl.read_parquet(frags_pq)   # ALL columns: fragment_id, start, stop, edit_distance, ...
    payload_reads = frags.filter(~pl.col("fragment_id").is_in(anc_ids)).select("read_acc").unique()
    kept = umi_ok.join(payload_reads, on="read_acc", how="semi")
    reads_kept = kept.height
    dropped_no_payload = reads_umi_ok - reads_kept
    n_species_raw = int(kept["fused_seq"].n_unique())

    # ---- audit: one row per INPUT read ------------------------------------------------------
    # A DROPPED read is still here, carrying the class that explains why -- the only per-read record
    # of that; the summary keeps the counts alone. `kept` distinguishes the two drop reasons: a
    # garbage UMI (umi_class != "keep") from an anchor-only read (umi_class == "keep", kept False).
    kept_accs = kept.select("read_acc").with_columns(pl.lit(True).alias("kept"))
    audit = (
        perread.join(kept_accs, on="read_acc", how="left")
        .with_columns(pl.col("kept").fill_null(False))
        .select(
            "read_acc", "fused_seq", "fused_len", "maxrun", "max_base_frac",
            "mean_umi_phred", "n_umi_bases_below_q10", "umi_class", "kept",
        )
        .sort("read_acc")
    )
    assert audit.height == reads_in
    assert int(audit.select(pl.col("kept").sum()).item()) == reads_kept
    audit.write_parquet(qc_dir / f"{stem}.umi_audit.parquet")

    # ---- STEP 2: correct to the abundant ed<=1 centroid ------------------------------------
    counts_tsv = qc_dir / f"{stem}.umi_counts.tsv"
    clusters_tsv = qc_dir / f"{stem}.umi_clusters.tsv"
    counts = (
        kept.group_by("fused_seq").agg(pl.len().alias("count"))
        .sort(["count", "fused_seq"], descending=[True, False])
    )
    counts.write_csv(counts_tsv, separator="\t", include_header=False)

    if reads_kept == 0:
        clusters_tsv.write_text("")
        starcode_argv = ["<skipped: no kept reads>"]
        corr_map = pl.DataFrame(schema={"fused_seq": pl.String, "fused_seq_corrected": pl.String})
    else:
        starcode_argv = run_starcode(args.starcode, counts_tsv, clusters_tsv)
        cent_rows = parse_cluster_centroids(clusters_tsv)
        seqs, cents = [], []
        for centroid, _tot, members in cent_rows:
            for m in members:
                seqs.append(m)
                cents.append(centroid)
        corr_map = pl.DataFrame({"fused_seq": seqs, "fused_seq_corrected": cents},
                                schema={"fused_seq": pl.String, "fused_seq_corrected": pl.String})
        # every kept fused_seq was in the starcode input, so it must map; guard anyway.
        unmapped = kept.join(corr_map, on="fused_seq", how="anti").height
        if unmapped:
            raise AssertionError(f"{unmapped} kept fused_seq values absent from starcode clusters")

    kept_corr = (
        kept.join(corr_map, on="fused_seq", how="left")
        .with_columns(pl.col("fused_seq_corrected").fill_null(pl.col("fused_seq")))
    )
    species = (
        kept_corr.group_by("fused_seq_corrected").agg(pl.len().alias("n_reads"))
        .sort(["n_reads", "fused_seq_corrected"], descending=[True, False])
    )
    n_species_corrected = species.height
    n_species_merged = n_species_raw - n_species_corrected

    # ---- STEP 3: write the corrected per-sample outputs ------------------------------------
    kept_out = kept_corr.join(
        species.rename({"n_reads": "umi_read_count"}), on="fused_seq_corrected", how="left"
    ).select("read_acc", "fused_seq_corrected", "umi_read_count")

    fused_full = pl.read_parquet(fused_pq)  # read_acc, strand, fused_seq, fused_len
    out = (
        fused_full.join(kept_out, on="read_acc", how="inner")  # inner -> kept reads only
        .select("read_acc", "strand", "fused_seq", "fused_len", "fused_seq_corrected", "umi_read_count")
        .sort("read_acc")
    )
    assert out.height == reads_kept, f"corrected fused wrote {out.height} != {reads_kept} kept reads"
    n_reads_relabeled = int(out.filter(pl.col("fused_seq") != pl.col("fused_seq_corrected")).height)
    out.write_parquet(dst_dir / f"{stem}.fused.parquet")

    species.write_csv(dst_dir / f"{stem}.umi_species.tsv", separator="\t", include_header=True)

    # Carry the decode UMI ROWS through, filtered to the kept reads. featurecount and report_sample
    # derive the anchor VOCABULARY from this file (ligand_vocab.anchor_set), and they find it by
    # stem -- so a decode-shaped tree must carry it or they cannot run here.
    tio.write_filtered_parquet(umis_pq, dst_dir / f"{stem}.umis.parquet",
                               out.select("read_acc"), ["read_acc", "umi_index"])

    # ligateddna.log: copied + a provenance block, so the tree records how it was made. parse_log
    # recovers the expected-sequences CSV from it, which every report needs.
    _ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _orig = src_log.read_text() if src_log.exists() else ""
    (dst_dir / f"{stem}.ligateddna.log").write_text(_orig + (
        "\n\n### umi_correct provenance ###\n"
        f"tool: {TOOL_VERSION}\n"
        f"timestamp_utc: {_ts}\n"
        f"mode: filter + ed<=1 correct (NO collapse -- this data must not be UMI-deduplicated)\n"
        f"params: max_homopolymer={args.max_homopolymer} max_base_frac={args.max_base_frac} "
        f"umi_len_window=[{uf.UMI_LEN_MIN},{uf.UMI_LEN_MAX}] starcode_threads={STARCODE_THREADS}\n"
        f"counts: reads_in={reads_in} dropped_total={dropped_total} "
        f"reads_umi_ok={reads_umi_ok} dropped_no_payload={dropped_no_payload} "
        f"reads_kept={reads_kept} n_reads_relabeled={n_reads_relabeled}\n"))

    # Carry the decode FRAGMENT CALLS through, filtered to the kept reads and otherwise untouched.
    # UMI correction filters reads and rewrites the UMI sequence; it changes NOTHING at the fragment
    # level -- fragment_id / start / stop / edit_distance are exactly as the decoder emitted them. Keeping
    # them here makes the tree self-contained: corrected UMI + the decoded architecture (including
    # the start/stop positions needed to rebuild the POSITIONAL fragment vector) in one place.
    (frags.join(out.select("read_acc"), on="read_acc", how="semi")
     .sort([c for c in ("read_acc", "start", "stop", "fragment_id") if c in frags.columns])
     .write_parquet(dst_dir / f"{stem}.fragments.parquet"))

    # Depth-robust MOLECULAR complexity. n_species_* are richness counts and grow with sequencing
    # depth, so they must NOT be compared across samples raw; c_eff estimates a property of the pool
    # (its read-collision probability) and is depth-robust. A sample with no duplicates at all (an
    # un-bottlenecked control) yields a LOWER BOUND -- see c_eff_is_lower_bound.
    _counts = species["n_reads"].to_list()
    _N, _D, _pairs, _C, _lower = c_eff(_counts)
    _pc = collision_prob(_counts)

    row = {
        "sample": stem,
        "reads_in": reads_in,
        "dropped_nonacgt": dropped["drop_nonacgt"],
        "dropped_lowcomplexity": dropped["drop_lowcomplexity"],
        "dropped_umilen": dropped["drop_umilen"],
        "dropped_total": dropped_total,
        "reads_umi_ok": reads_umi_ok,
        "dropped_no_payload": dropped_no_payload,
        "reads_kept": reads_kept,
        "n_species_raw": n_species_raw,
        "n_species_corrected": n_species_corrected,
        "n_species_merged": n_species_merged,
        "n_reads_relabeled": n_reads_relabeled,
        # --- depth-robust complexity (compare THESE across samples, not n_species_*) ---
        "dup_pairs": _pairs,
        "collision_prob": _pc,   # sum p_i^2 — the UNBIASED depth-robust quantity (c_eff is its reciprocal)
        "c_eff": round(_C, 1) if _C != float("inf") else None,
        "c_eff_is_lower_bound": _lower,
        "n_over_c_eff": round(_N / _C, 4) if _C not in (0, float("inf")) else None,
    }
    pl.DataFrame([row]).write_csv(dst_dir / f"{stem}.umi_summary.tsv", separator="\t")

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{stem}] {reads_in:,} reads in -> {reads_umi_ok:,} UMI-ok "
          f"(dropped {dropped_total:,}) -> {reads_kept:,} payload-bearing "
          f"(dropped {dropped_no_payload:,} anchor-only); "
          f"{n_species_raw:,} -> {n_species_corrected:,} UMI species "
          f"({n_species_merged:,} merged, {n_reads_relabeled:,} reads relabelled) @ {ts}")
    print(f"  starcode: {' '.join(starcode_argv)}")
    print(f"  -> {dst_dir}/{stem}.fused.parquet, .umi_species.tsv, .umi_summary.tsv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
