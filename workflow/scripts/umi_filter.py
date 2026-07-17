#!/usr/bin/env python
"""Per-sample UMI garbage-filter (+ optional starcode dedup) for the ligand2frags pipeline.

Given one decoded sample (the ``<STEM>.fused.parquet`` / ``.umis.parquet`` / ``.fragments.parquet``
/ ``.trimmed.parquet`` / ``.fused_reads.fq`` set), this:

  STEP 1  classifies every read from its fused UMI sequence (``fused_seq``) and its two UMI
          lengths, dropping garbage UMIs (non-ACGT / low complexity / bad length). The class
          rules live in ``umi_filter_lib`` (the single, unit-tested source of truth).
  STEP 2  (only with ``--collapse``) collapses the surviving reads into molecules with
          ``starcode -d 1 -s`` (spheres), picking one representative read per cluster by highest
          whole-read base-quality sum (Picard MarkDuplicates convention; read from the FASTQ),
          tie-broken by smallest read_acc. WITHOUT ``--collapse`` (the default) this step is
          skipped and every read that passes STEP 1 is kept.
  STEP 3  writes a filtered copy of the sample (survivors only) plus QC.

Style/conventions follow workflow/build_featurecounts_2026.py (polars, pl.scan_parquet where
sensible, L2F_STEM-style env config). Deterministic: stable sorts, sequences sorted before
starcode, outputs sorted by read_acc before writing.

CLI:
    umi_dedup.py --src-stem PATH --dst-dir DIR --qc-dir DIR
                 [--collapse] [--threads N] [--max-homopolymer 8] [--max-base-frac 0.7]
                 [--starcode starcode] [--force]

    --src-stem  path prefix such that <src-stem>.fused.parquet etc. exist
                (defaults to $L2F_STEM). STEM = its basename.
    --dst-dir   output sample directory (files written as <dst-dir>/<STEM>.*)
    --qc-dir    shared QC directory (files written as <qc-dir>/<STEM>.*, plus the per-sample
                <qc-dir>/<STEM>.umi_dedup_summary.tsv row the batch driver concatenates)
    --collapse  after filtering, collapse duplicate molecules with starcode (UMI dedup). Omit
                (the default) to keep every read that passes the UMI QC filter.
    --force     overwrite a non-empty destination sample dir (idempotent re-run)

starcode command emitted (verbatim, with -s spheres mandatory, -d 1 explicit, -t 1 pinned):
    <starcode> -d 1 -s --print-clusters -t 1 -i <counts.tsv> -o <clusters.tsv>
    starcode clustering is pinned to a SINGLE thread: multithreaded starcode is functionally
    identical (same clusters, counts, centroids, row order) but emits the comma-separated col3
    member sequences in a thread-schedule-dependent order, which makes the raw
    qc/<STEM>.umi_clusters.tsv non-byte-reproducible across re-runs. -t 1 is fully deterministic
    and costs ~0.5s/sample at this scale (~600k unique UMIs). The representative pick and every
    parquet/audit are member-order-invariant, so derived results are unchanged.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))  # import the local pure helpers
import umi_filter_lib as uf

TOOL_VERSION = "umi_dedup.py/1.0"
Q10_PHRED = 10                 # "below Q10" == Phred < 10 (ASCII '+' == Phred 10 == chr(43))
CLASS_ORDER = ["drop_nonacgt", "drop_lowcomplexity", "drop_umilen"]  # for stable summary columns


# --------------------------------------------------------------------------- pure-ish helpers
def _qual_stats(qual: str) -> dict:
    """Per-UMI-quality-string stats from a Phred+33 string: sum of Phred, count below Q10."""
    sp = 0
    nb = 0
    for ch in qual:
        p = ord(ch) - 33
        sp += p
        if p < Q10_PHRED:
            nb += 1
    return {"_sp": sp, "_nb": nb}


def _row_features(fused_seq: str, l0: int, l1: int, h: int, f: float) -> dict:
    """maxrun / max_base_frac / umi_class for one read (delegates to umi_filter_lib)."""
    return {
        "maxrun": uf.maxrun(fused_seq),
        "max_base_frac": uf.max_base_frac(fused_seq),
        "umi_class": uf.classify(
            fused_seq, l0, l1, max_homopolymer=h, max_base_frac=f,
            umi_len_min=uf.UMI_LEN_MIN, umi_len_max=uf.UMI_LEN_MAX,
        ),
    }


def read_acc_from_header(header: str) -> str:
    """read_acc = the FASTQ header's first whitespace token, minus the leading '@'."""
    return header[1:].split(None, 1)[0]


def stream_qual_sums(fq_path: Path, wanted: set[str]) -> dict[str, int]:
    """Single streaming pass over a FASTQ: sum of Phred+33 base qualities over each WANTED read.

    Streams 4-line records; never materialises the file as a list of records.
    """
    sums: dict[str, int] = {}
    with open(fq_path, "rt") as fh:
        while True:
            header = fh.readline()
            if not header:
                break
            fh.readline()            # sequence
            fh.readline()            # '+'
            qual = fh.readline()     # quality (Phred+33)
            acc = read_acc_from_header(header)
            if acc in wanted:
                sums[acc] = sum(ord(c) - 33 for c in qual.rstrip("\r\n"))
    return sums


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


# --------------------------------------------------------------------------- starcode
def starcode_version(starcode: str) -> str:
    try:
        out = subprocess.run([starcode, "--version"], capture_output=True, text=True)
        v = (out.stderr or out.stdout).strip().splitlines()
        return v[0] if v else "unknown"
    except Exception as exc:  # noqa: BLE001
        return f"unknown ({exc})"


# starcode clustering is pinned to a SINGLE thread on purpose. Multithreaded starcode is
# functionally identical (same clusters, counts, centroids, row order) but emits the
# comma-separated col3 member SEQUENCES in a thread-schedule-dependent order, which makes the
# raw qc/<STEM>.umi_clusters.tsv non-byte-reproducible across re-runs (empirically verified:
# repeated -t 8 runs on the same input DIFFER byte-for-byte, differing only in col3 member order;
# -t 1 runs are byte-identical). -t 1 costs ~0.5s/sample here (~600k unique UMIs). Downstream is
# member-order-invariant (the representative is picked by a total order over reads, and every
# parquet/audit is re-sorted by read_acc), so results are identical either way.
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


def parse_clusters(clusters_tsv: Path, seq2accs: dict[str, list[str]], qual_sum: dict[str, int]):
    """Parse `-s --print-clusters` output (3 cols: centroid, total_count, member seqs CSV).

    starcode's centroid (col1) is IGNORED; the representative is chosen here by whole-read
    quality. Returns (cluster_rows, n_clusters) where each cluster_row is
    (cluster_id, member_read_accs, representative_acc, cluster_size).
    """
    rows = []
    cid = 0
    with open(clusters_tsv, "rt") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"unexpected starcode line ({len(parts)} cols): {line!r}")
            total_count = int(parts[1])
            member_seqs = parts[2].split(",")
            accs: list[str] = []
            for s in member_seqs:
                mapped = seq2accs.get(s)
                if mapped is None:
                    raise KeyError(f"starcode emitted seq {s!r} absent from the kept-read counts")
                accs.extend(mapped)
            if len(accs) != total_count:
                raise AssertionError(
                    f"cluster {cid}: member reads {len(accs)} != starcode count {total_count}")
            # Representative: max whole-read base-quality sum; tie -> lexicographically smallest acc.
            rep = min(accs, key=lambda a: (-qual_sum[a], a))
            rows.append((cid, accs, rep, len(accs)))
            cid += 1
    return rows, cid


# --------------------------------------------------------------------------- main driver
def part(stem: Path, ext: str) -> Path:
    return stem.parent / f"{stem.name}{ext}"


def build_per_read(fused_pq: Path, umis_pq: Path, max_homopolymer: int, max_base_frac: float):
    """Per-read frame: read_acc, fused_seq, fused_len, umi stats, maxrun, max_base_frac, umi_class."""
    fused = pl.read_parquet(fused_pq, columns=["read_acc", "fused_seq", "fused_len"])

    umi = pl.read_parquet(umis_pq, columns=["read_acc", "umi_index", "umi_len", "umi_qual"])
    # SANITY: zero non-ACGT characters expected across all umi_seq -- but the check must exist,
    # so verify the classification's non-ACGT branch is exercised only via fused_seq (below).
    qstat = umi.with_columns(
        pl.col("umi_qual").map_elements(
            _qual_stats, return_dtype=pl.Struct({"_sp": pl.Int64, "_nb": pl.Int64})
        ).alias("_q")
    ).unnest("_q")

    stats = qstat.group_by("read_acc").agg(
        (pl.col("_sp").sum() / pl.col("umi_len").sum()).alias("mean_umi_phred"),
        pl.col("_nb").sum().cast(pl.Int64).alias("n_umi_bases_below_q10"),
    )
    lens = umi.group_by("read_acc").agg(
        pl.col("umi_len").sort_by("umi_index").alias("_lens")
    ).with_columns(
        pl.col("_lens").list.get(0).alias("umi_len_0"),
        pl.col("_lens").list.get(1).alias("umi_len_1"),
    ).drop("_lens")

    base = fused.join(lens, on="read_acc", how="left").join(stats, on="read_acc", how="left")
    if base.select(pl.col("umi_len_0").is_null().any()).item():
        raise AssertionError("some reads in fused.parquet lack umi rows (umi_len_0 null)")

    feat = base.with_columns(
        pl.struct("fused_seq", "umi_len_0", "umi_len_1").map_elements(
            lambda r: _row_features(
                r["fused_seq"], r["umi_len_0"], r["umi_len_1"], max_homopolymer, max_base_frac),
            return_dtype=pl.Struct(
                {"maxrun": pl.Int64, "max_base_frac": pl.Float64, "umi_class": pl.String}),
        ).alias("_feat")
    ).unnest("_feat").drop("umi_len_0", "umi_len_1")
    return feat


def prepare_dst(dst_dir: Path, force: bool) -> None:
    """Create dst_dir; fail loudly if it already holds files unless --force (then wipe them)."""
    if dst_dir.exists():
        entries = list(dst_dir.iterdir())
        if entries and not force:
            names = ", ".join(sorted(p.name for p in entries)[:8])
            raise SystemExit(
                f"[FATAL] destination {dst_dir} is not empty ({len(entries)} entr(y/ies): {names} ...).\n"
                f"        Refusing to overwrite a previous/partial run. Pass --force to overwrite "
                f"(or set FORCE=1 for run_umi_dedup.sh).")
        for p in entries:
            if p.is_dir() and not p.is_symlink():
                shutil.rmtree(p)
            else:
                p.unlink()
    dst_dir.mkdir(parents=True, exist_ok=True)


def write_filtered_parquet(src: Path, dst: Path, survivors: pl.DataFrame, sort_by: list[str]) -> None:
    """Filter src parquet to survivor read_accs (semi-join), sort deterministically, write."""
    df = (
        pl.scan_parquet(src)
        .join(survivors.lazy(), on="read_acc", how="semi")
        .sort(sort_by)
        .collect()
    )
    df.write_parquet(dst)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Per-sample UMI filter + starcode dedup.")
    ap.add_argument("--src-stem", default=os.environ.get("L2F_STEM"),
                    help="path prefix; <src-stem>.fused.parquet etc. must exist (default $L2F_STEM)")
    ap.add_argument("--dst-dir", required=True, help="output sample directory")
    ap.add_argument("--qc-dir", required=True, help="shared QC directory")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("L2F_THREADS", "4")),
                    help="recorded in provenance; starcode clustering itself is pinned to a single "
                         "thread for byte-reproducible raw cluster output (see run_starcode)")
    ap.add_argument("--max-homopolymer", type=int, default=uf.DEFAULT_MAX_HOMOPOLYMER)
    ap.add_argument("--max-base-frac", type=float, default=uf.DEFAULT_MAX_BASE_FRAC)
    ap.add_argument("--starcode", default=os.environ.get("L2F_STARCODE", "starcode"))
    ap.add_argument("--collapse", action="store_true",
                    help="after filtering, collapse duplicate molecules with starcode -d 1 -s "
                         "(UMI dedup); default keeps every read that passes the UMI QC filter")
    ap.add_argument("--force", action="store_true",
                    help="overwrite a non-empty destination sample dir")
    args = ap.parse_args(argv[1:])

    if not args.src_stem:
        ap.error("--src-stem is required (or set $L2F_STEM)")

    src_stem = Path(args.src_stem)
    if not src_stem.is_absolute():
        src_stem = Path.cwd() / src_stem
    stem = src_stem.name
    dst_dir = Path(args.dst_dir).resolve()
    qc_dir = Path(args.qc_dir).resolve()

    fused_pq = part(src_stem, ".fused.parquet")
    umis_pq = part(src_stem, ".umis.parquet")
    frags_pq = part(src_stem, ".fragments.parquet")
    trimmed_pq = part(src_stem, ".trimmed.parquet")
    src_fq = part(src_stem, ".fused_reads.fq")
    src_trunc = part(src_stem, ".truncated.fq")
    src_log = part(src_stem, ".ligateddna.log")
    for p in (fused_pq, umis_pq, frags_pq, trimmed_pq, src_fq):
        if not p.exists():
            print(f"[FATAL] missing input: {p}", file=sys.stderr)
            return 1

    prepare_dst(dst_dir, args.force)
    qc_dir.mkdir(parents=True, exist_ok=True)

    # ---- STEP 1: classify every read -------------------------------------------------------
    perread = build_per_read(fused_pq, umis_pq, args.max_homopolymer, args.max_base_frac)
    reads_in = perread.height

    class_counts = dict(
        perread.group_by("umi_class").agg(pl.len().alias("n"))
        .iter_rows()
    )
    dropped = {c: int(class_counts.get(c, 0)) for c in CLASS_ORDER}
    dropped_total = sum(dropped.values())
    kept = perread.filter(pl.col("umi_class") == "keep")
    reads_clustered = kept.height
    # SANITY: per-class drops + clustered == input.
    assert dropped_total + reads_clustered == reads_in, \
        f"{dropped_total} + {reads_clustered} != {reads_in}"

    # ---- STEP 2: (optional) collapse survivors into molecules with starcode -----------------
    # Both declared QC files (umi_counts.tsv, umi_clusters.tsv) are always written so the Snakemake
    # rule's outputs exist in either mode. Without --collapse we stop after the filter: every kept
    # read survives and the cluster columns are left null.
    counts_tsv = qc_dir / f"{stem}.umi_counts.tsv"
    clusters_tsv = qc_dir / f"{stem}.umi_clusters.tsv"

    counts = (
        kept.group_by("fused_seq").agg(pl.len().alias("count"))
        .sort(["count", "fused_seq"], descending=[True, False])
    )
    counts.write_csv(counts_tsv, separator="\t", include_header=False)

    if not args.collapse:
        # Filter-only (default): no starcode clustering, no correction. Each kept read is its own
        # molecule; the audit's cluster_id / cluster_size stay null and every kept read is a
        # "representative" (i.e. survives to the filtered tree).
        clusters_tsv.write_text("")
        survivors = set(kept["read_acc"].to_list())
        n_clusters = reads_clustered
        dup_reads_removed = 0
        starcode_argv = ["<skipped: filter-only (no --collapse)>"]
        cl_df = kept.select("read_acc").with_columns(
            pl.lit(None, dtype=pl.Int64).alias("cluster_id"),
            pl.lit(None, dtype=pl.Int64).alias("cluster_size"),
            pl.lit(True, dtype=pl.Boolean).alias("is_representative"),
        )
    else:
        # seq -> read_accs (for mapping cluster members back to reads)
        seq2accs = {
            s: accs for s, accs in
            kept.group_by("fused_seq").agg(pl.col("read_acc")).iter_rows()
        }

        if reads_clustered == 0:
            clusters_tsv.write_text("")
            cluster_rows, n_clusters = [], 0
            qual_sum: dict[str, int] = {}
            starcode_argv = ["<skipped: no kept reads>"]
        else:
            kept_accs = set(kept["read_acc"].to_list())
            qual_sum = stream_qual_sums(src_fq, kept_accs)
            missing = kept_accs - qual_sum.keys()
            if missing:
                raise AssertionError(f"{len(missing)} kept reads have no FASTQ record (e.g. {next(iter(missing))})")
            starcode_argv = run_starcode(args.starcode, counts_tsv, clusters_tsv)
            cluster_rows, n_clusters = parse_clusters(clusters_tsv, seq2accs, qual_sum)

        dup_reads_removed = reads_clustered - n_clusters

        # cluster membership table (kept reads only) for the audit
        acc_cid, acc_size, acc_rep, survivors_list = [], [], [], []
        accs_flat = []
        for cid, accs, rep, size in cluster_rows:
            survivors_list.append(rep)
            for a in accs:
                accs_flat.append(a)
                acc_cid.append(cid)
                acc_size.append(size)
                acc_rep.append(a == rep)
        survivors = set(survivors_list)
        # SANITY: surviving read_acc set has no duplicates and matches n_clusters.
        assert len(survivors) == len(survivors_list) == n_clusters, "duplicate or miscounted survivors"

        cl_df = pl.DataFrame(
            {"read_acc": accs_flat, "cluster_id": acc_cid, "cluster_size": acc_size,
             "is_representative": acc_rep},
            schema={"read_acc": pl.String, "cluster_id": pl.Int64, "cluster_size": pl.Int64,
                    "is_representative": pl.Boolean},
        )

    molecules_out = n_clusters
    # SANITY: molecules_out == n_clusters (by construction; assert to lock the invariant).
    assert molecules_out == n_clusters

    # ---- audit: one row per INPUT read -----------------------------------------------------
    audit = (
        perread.join(cl_df, on="read_acc", how="left")
        .with_columns(pl.col("is_representative").fill_null(False).alias("kept"))
        .select(
            "read_acc", "fused_seq", "fused_len", "maxrun", "max_base_frac",
            "mean_umi_phred", "n_umi_bases_below_q10", "umi_class",
            "cluster_id", "cluster_size", "is_representative", "kept",
        )
        .sort("read_acc")
    )
    assert audit.height == reads_in
    assert int(audit.select(pl.col("kept").sum()).item()) == molecules_out

    audit.write_parquet(qc_dir / f"{stem}.umi_audit.parquet")

    # ---- STEP 3: write the filtered sample tree --------------------------------------------
    survivors_df = pl.DataFrame({"read_acc": sorted(survivors)}, schema={"read_acc": pl.String})
    write_filtered_parquet(fused_pq, part(Path(dst_dir / stem), ".fused.parquet"),
                           survivors_df, ["read_acc"])
    write_filtered_parquet(trimmed_pq, part(Path(dst_dir / stem), ".trimmed.parquet"),
                           survivors_df, ["read_acc"])
    write_filtered_parquet(umis_pq, part(Path(dst_dir / stem), ".umis.parquet"),
                           survivors_df, ["read_acc", "umi_index"])
    write_filtered_parquet(frags_pq, part(Path(dst_dir / stem), ".fragments.parquet"),
                           survivors_df, ["read_acc", "start", "stop", "fragment_id", "edit_distance"])

    dst_fq = part(Path(dst_dir / stem), ".fused_reads.fq")
    n_written = stream_filter_fq(src_fq, dst_fq, survivors)
    assert n_written == molecules_out, f"fq wrote {n_written} != {molecules_out} survivors"

    # truncated.fq: symlink to the (disjoint, unfiltered) source
    dst_trunc = part(Path(dst_dir / stem), ".truncated.fq")
    if dst_trunc.exists() or dst_trunc.is_symlink():
        dst_trunc.unlink()
    if src_trunc.exists():
        os.symlink(os.path.abspath(src_trunc), dst_trunc)

    # ligateddna.log: copy + appended provenance block
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    orig_log = src_log.read_text() if src_log.exists() else ""
    mode = "dedup (filter + starcode collapse)" if args.collapse else "filter-only (no collapse)"
    sc_version = starcode_version(args.starcode) if args.collapse else "<not used: filter-only>"
    starcode_params = (f" d=1 spheres=True starcode_threads={STARCODE_THREADS}"
                       if args.collapse else "")
    provenance = (
        "\n\n### umi_dedup provenance ###\n"
        f"tool: {TOOL_VERSION}\n"
        f"timestamp_utc: {ts}\n"
        f"mode: {mode}\n"
        f"starcode_version: {sc_version}\n"
        f"starcode_cmd: {' '.join(starcode_argv)}\n"
        f"params: max_homopolymer={args.max_homopolymer} max_base_frac={args.max_base_frac} "
        f"umi_len_window=[{uf.UMI_LEN_MIN},{uf.UMI_LEN_MAX}]{starcode_params} "
        f"requested_threads={args.threads}\n"
        f"counts: reads_in={reads_in} dropped_nonacgt={dropped['drop_nonacgt']} "
        f"dropped_lowcomplexity={dropped['drop_lowcomplexity']} dropped_umilen={dropped['drop_umilen']} "
        f"dropped_total={dropped_total} reads_clustered={reads_clustered} n_clusters={n_clusters} "
        f"molecules_out={molecules_out} dup_reads_removed={dup_reads_removed}\n"
    )
    (part(Path(dst_dir / stem), ".ligateddna.log")).write_text(orig_log + provenance)

    # ---- per-sample summary row (batch driver concatenates these) --------------------------
    pct_reads_removed = round(100.0 * (reads_in - molecules_out) / reads_in, 4) if reads_in else 0.0
    row = {
        "sample": stem,
        "reads_in": reads_in,
        "dropped_nonacgt": dropped["drop_nonacgt"],
        "dropped_lowcomplexity": dropped["drop_lowcomplexity"],
        "dropped_umilen": dropped["drop_umilen"],
        "dropped_total": dropped_total,
        "reads_clustered": reads_clustered,
        "n_clusters": n_clusters,
        "molecules_out": molecules_out,
        "dup_reads_removed": dup_reads_removed,
        "pct_reads_removed": pct_reads_removed,
    }
    pl.DataFrame([row]).write_csv(qc_dir / f"{stem}.umi_dedup_summary.tsv", separator="\t")

    if args.collapse:
        print(f"[{stem}] {reads_in:,} reads in -> {molecules_out:,} molecules out "
              f"({pct_reads_removed}% removed)")
    else:
        print(f"[{stem}] {reads_in:,} reads in -> {molecules_out:,} reads kept "
              f"(filter-only; {pct_reads_removed}% removed)")
    print(f"  dropped: nonacgt={dropped['drop_nonacgt']:,} "
          f"lowcomplexity={dropped['drop_lowcomplexity']:,} umilen={dropped['drop_umilen']:,} "
          f"(total {dropped_total:,})")
    if args.collapse:
        print(f"  clustered {reads_clustered:,} reads -> {n_clusters:,} clusters "
              f"(dedup removed {dup_reads_removed:,})")
    print(f"  -> {dst_dir}")
    print(f"  -> {qc_dir}/{stem}.umi_audit.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
