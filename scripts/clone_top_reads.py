#!/usr/bin/env python
"""FASTQ of the dominant clones (fused UMIs), one file per group, ranked by p.

STANDALONE per-study analysis -- NOT part of the Snakemake DAG. It reads the dominance
tables written by ``clonotype_clone_ranks.py`` under ``results/clonerank/``, the per-round
read maps under ``results/fragvec/<sample>/`` and the decoded FASTQs under
``results/decode/<sample>/``, and writes FASTQ (+ a ranking sidecar) under
``results/clonerank/top_reads/``. Nothing here mutates the pipeline, its config, or any
existing ``results/`` tree.

The clone-level analogue of ``top_vector_reads.py`` -- that one extracts the reads of the
top fitness VECTORS; this one extracts the reads of the top dominant CLONES, one level down.

What it computes
----------------
``clonotype_clone_ranks.py`` flags, per sample, which clonotypes carry a clone more
abundant than undersampling alone explains (``p_dominant``). Each such clonotype has one
winning clone, its ``top_umi``. This script collects **every dominant clone with
``p_dominant <= alpha``**, pools them per group across all rounds, ranks them by p
(ascending -- rank 1 = most confident), and writes **all reads of each clone** as FASTQ.

"All reads of the clone" = ``fragvec_reads.parquet`` filtered on ``vector == V AND umi ==
top_umi`` -- that one molecule's read family (about ``max_reads_per_umi`` reads). The R0/R1
and ``empty``/``initial`` rounds carry no dominant clones by construction, so a group's
FASTA is its selected rounds only.

The read -> sequence join
-------------------------
``<sample>.fragvec_reads.parquet`` (columns ``read_acc, ont_read, vector, umi``) maps reads
to clones but holds no sequence. The sequence lives in ``<sample>.fused_reads.fq``, whose
header's first whitespace token equals the parquet ``read_acc`` exactly
(``<start>:<stop>|<uuid>``). So we take the dominant clones' ``read_acc`` from the parquet
and stream the FASTQ once per sample to pull the sequences AND their quality, which is copied
through verbatim so the output feeds a consensus caller directly (NGSpeciesID ``--fastq``).
A read in the parquet but absent from the FASTQ is counted (``reads_missing``) and logged,
never silently dropped -- that is the "inside the fastq" filter.

``read_acc`` is the ONLY thing joined on, and that is what makes the header's ``umi=``
trustworthy. The fq is a decode-stage file, i.e. it predates UMI correction
(pychopper -> decode -> umi_correct -> fragvec), and its header tail carries pychopper's raw
``umi1=``/``umi2=`` -- a different, uncorrected call. ``umi=`` instead comes from the parquet
side (``build_fragvec.py --umi-col fused_seq_corrected``, the starcode ed<=1 centroid), so only
``strand=`` is kept from the tail; see ``fastq_io`` for the worked example of the two
disagreeing. ``umi_correct.py`` filters and relabels reads but never renames them, so
``read_acc`` is stable across both trees.

CLI (all defaulted, so it runs with no args from the repo root)
---------------------------------------------------------------
    pixi run -e default python scripts/clone_top_reads.py \
      --clonerank-dir       results/clonerank \
      --fragvec-dir         results/fragvec \
      --initial-dir         results/fragvec_initial \
      --decode-dir          results/decode \
      --decode-initial-dir  results/decode_initial \
      --groups              "" \
      --alpha               0.05 \
      --out-dir             results/clonerank/top_reads \
      --max-reads-per-clone 0

The dominant set is whatever ``clonotype_clone_ranks.py`` wrote: it filters ``is_dominant``
(the report's definition: p < test-alpha AND n_reads >= min_reads), so when that step ran
fitness-restricted, only fitness-top clonotypes' groups appear here and the sidecar/FASTA
carry the ``log2_sum`` / ``fit_rank`` enrichment columns through. ``--alpha`` only tightens
further. Stale per-group outputs (e.g. ``*-SN`` after the restriction) are pruned.

Emits (under --out-dir, per group with >=1 dominant clone):
    <group>.dominant_clone_reads.fq             reads of every dominant clone, ranked by p
    <group>.dominant_clone_reads.ranking.tsv    rank/clone/p/dominance/log2_sum/n_pos_steps/read counts
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fastq_io import read_fastq_records, strand_field, write_fastq   # noqa: E402  one FASTQ reader


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# the dominant set
# --------------------------------------------------------------------------- #
def load_dominant(clonerank_dir: Path, alpha: float) -> pl.DataFrame:
    """Every dominant clone (p_dominant <= alpha) across all samples, one row each.

    Columns: sample, group, round, vector, umi(=top_umi), p_dominant, dominance,
    clonotype_n_reads, n_umis, max_reads_per_umi, log2_sum, fit_rank, n_pos_steps.
    """
    tsvs = sorted(glob.glob(str(clonerank_dir / "*.clonotype_dominance.tsv")))
    if not tsvs:
        raise SystemExit(f"[FATAL] no *.clonotype_dominance.tsv under {clonerank_dir} "
                         f"(run clonotype_clone_ranks.py first)")
    frames = []
    for p in tsvs:
        d = pl.read_csv(p, separator="\t")
        if not d.height:
            continue
        # log2_sum / fit_rank / n_pos_steps are present when clonotype_clone_ranks ran
        # fitness-restricted; carry them through when they exist, fill null otherwise, so the
        # schema stays uniform to concat. n_pos_steps is trajectory SHAPE (how many of the
        # round-to-round log2 steps rose), never a filter -- it says whether a high log2_sum
        # came from sustained enrichment or from one lucky round.
        for c, dt in (("log2_sum", pl.Float64), ("fit_rank", pl.Int64), ("n_pos_steps", pl.Int64)):
            if c not in d.columns:
                d = d.with_columns(pl.lit(None, dtype=dt).alias(c))
        # `is_dominant` is the report's definition (p < test-alpha AND n_reads >= min_reads); the
        # <= alpha is an optional further tightening. Filtering on `is_dominant` keeps the FASTA set
        # identical to the report -- without it, uncensored sub-min_reads clonotypes leak in.
        hit = d.filter(pl.col("is_dominant") & (pl.col("p_dominant") <= alpha))
        if hit.height:
            frames.append(hit.select(
                "sample", "group", "round", "vector",
                pl.col("top_umi").alias("umi"), "p_dominant", "dominance",
                pl.col("n_reads").alias("clonotype_n_reads"), "n_umis", "max_reads_per_umi",
                "log2_sum", "fit_rank", "n_pos_steps"))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames)


# --------------------------------------------------------------------------- #
# read extraction (per sample, so each parquet/FASTQ is read once)
# --------------------------------------------------------------------------- #
def sample_paths(sample: str, args) -> tuple[Path, Path]:
    """(fragvec_reads.parquet, fused_reads.fq) for a sample, picking the initial tree if that
    is where the sample lives. FATAL if neither reads map is present."""
    main = args.fragvec_dir / sample / f"{sample}.fragvec_reads.parquet"
    init = args.initial_dir / sample / f"{sample}.fragvec_reads.parquet"
    if main.exists():
        return main, args.decode_dir / sample / f"{sample}.fused_reads.fq"
    if init.exists():
        return init, args.decode_initial_dir / sample / f"{sample}.fused_reads.fq"
    raise SystemExit(f"[FATAL] {sample}: no fragvec_reads.parquet under "
                     f"{args.fragvec_dir}/ or {args.initial_dir}/")


def clone_read_accs(reads_parquet: Path, clones: pl.DataFrame,
                    max_reads: int) -> tuple[dict[str, tuple[str, str]], dict[tuple[str, str], int]]:
    """(acc_map, present_counts) for the given clones in one sample's reads parquet.

    acc_map:        read_acc -> (vector, umi)          the reads to pull from the FASTQ
    present_counts: (vector, umi) -> reads in the parquet (before any FASTQ filtering)

    ``max_reads`` > 0 caps how many reads are emitted per clone (deterministic: read_acc
    sorted). 0 keeps them all.
    """
    keys = clones.select("vector", "umi").unique()
    reads = (pl.read_parquet(reads_parquet, columns=["read_acc", "vector", "umi"])
             .join(keys, on=["vector", "umi"], how="semi")
             .sort("read_acc"))
    present_counts = {(v, u): n for v, u, n in
                      reads.group_by(["vector", "umi"]).agg(pl.len().alias("n")).iter_rows()}
    if max_reads > 0:
        reads = (reads.with_columns(pl.col("read_acc").cum_count().over(["vector", "umi"]).alias("_i"))
                 .filter(pl.col("_i") <= max_reads).drop("_i"))
    acc_map = {acc: (v, u) for acc, v, u in reads.iter_rows()}
    return acc_map, present_counts


# --------------------------------------------------------------------------- #
# per group
# --------------------------------------------------------------------------- #
def process_group(group: str, dom: pl.DataFrame, args) -> None:
    """Write <group>.dominant_clone_reads.fq + .ranking.tsv for one group."""
    # rank across the whole group by p (asc); tie-break dominance desc, then vector
    ranked = (dom.sort(["p_dominant", "dominance", "vector"], descending=[False, True, False])
              .with_row_index("rank", offset=1))

    # A clone is keyed by (sample, vector, umi), NOT (vector, umi): the SAME fused UMI on the SAME
    # vector can be dominant in more than one round, and those are distinct observations with
    # distinct reads and their own rank/round. Keying on the pair alone collapses them.
    clone_meta = {(r["sample"], r["vector"], r["umi"]): r for r in ranked.iter_rows(named=True)}

    # gather reads sample by sample (each parquet + FASTQ read once). Within one sample
    # (vector, umi) is unique, so the sample-local map is unambiguous.
    records: list[dict] = []                                             # one per emitted read
    present: dict[tuple[str, str, str], int] = {}                         # (sample, v, u) -> reads in parquet
    emitted: dict[tuple[str, str, str], int] = {}                         # (sample, v, u) -> reads with a seq
    total_missing = 0
    for sample in ranked["sample"].unique(maintain_order=True).to_list():
        sub = ranked.filter(pl.col("sample") == sample)
        reads_p, fq_p = sample_paths(sample, args)
        if not fq_p.exists():
            raise SystemExit(f"[FATAL] {sample}: missing FASTQ {fq_p}")
        acc_map, pcounts = clone_read_accs(reads_p, sub, args.max_reads_per_clone)
        present.update({(sample, v, u): n for (v, u), n in pcounts.items()})
        seqs = read_fastq_records(fq_p, set(acc_map))
        missing = len(acc_map) - len(seqs)
        if missing:
            total_missing += missing
            log(f"[WARN] {group}/{sample}: {missing}/{len(acc_map)} reads not found in {fq_p.name}")
        for acc, (v, u) in acc_map.items():
            if acc not in seqs:
                continue
            m = clone_meta[(sample, v, u)]
            tail, seq, qual = seqs[acc]
            records.append({"rank": m["rank"], "acc": acc, "v": v, "u": u, "tail": tail, "seq": seq,
                            "qual": qual, "p": m["p_dominant"], "round": m["round"],
                            "log2_sum": m["log2_sum"], "n_pos_steps": m["n_pos_steps"]})
            emitted[(sample, v, u)] = emitted.get((sample, v, u), 0) + 1

    # FASTQ, ordered by (rank, read_acc). Quality is decode's, verbatim.
    def _header(r: dict) -> str:
        ls = f"{r['log2_sum']:.3f}" if r["log2_sum"] is not None else "NA"
        nps = r["n_pos_steps"] if r["n_pos_steps"] is not None else "NA"
        return (f"{r['acc']} rank={r['rank']} p={r['p']:.2e} log2_sum={ls} n_pos_steps={nps} "
                f"round=R{r['round']} vector={r['v']} umi={r['u']} "
                f"{strand_field(r['tail'])}").rstrip()

    fq_out = args.out_dir / f"{group}.dominant_clone_reads.fq"
    write_fastq(fq_out, ((_header(r), r["seq"], r["qual"])
                         for r in sorted(records, key=lambda r: (r["rank"], r["acc"]))))
    log(f"[INFO] wrote {fq_out}  ({len(records)} reads from {ranked.height} clone(s))")

    # ranking sidecar, one row per dominant clone (sample, vector, umi)
    sidecar = ranked.with_columns(
        pl.struct(["sample", "vector", "umi"]).map_elements(
            lambda s: emitted.get((s["sample"], s["vector"], s["umi"]), 0), return_dtype=pl.Int64
        ).alias("reads_emitted"),
        pl.struct(["sample", "vector", "umi"]).map_elements(
            lambda s: present.get((s["sample"], s["vector"], s["umi"]), 0), return_dtype=pl.Int64
        ).alias("reads_present"),
    ).with_columns((pl.col("reads_present") - pl.col("reads_emitted")).alias("reads_missing")).select(
        "rank", "sample", "group", "round", "vector", "umi", "p_dominant", "dominance", "log2_sum",
        "fit_rank", "n_pos_steps", "clonotype_n_reads", "n_umis", "max_reads_per_umi",
        "reads_emitted", "reads_missing")
    sidecar_p = args.out_dir / f"{group}.dominant_clone_reads.ranking.tsv"
    sidecar.write_csv(sidecar_p, separator="\t")
    log(f"[INFO] wrote {sidecar_p}  ({sidecar.height} rows x {sidecar.width} cols, "
        f"{total_missing} reads missing from FASTQs)")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clonerank-dir", default="results/clonerank", type=Path)
    ap.add_argument("--fragvec-dir", default="results/fragvec", type=Path)
    ap.add_argument("--initial-dir", default="results/fragvec_initial", type=Path)
    ap.add_argument("--decode-dir", default="results/decode", type=Path)
    ap.add_argument("--decode-initial-dir", default="results/decode_initial", type=Path)
    ap.add_argument("--groups", default="",
                    help="comma-separated groups to emit (empty = all discovered)")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="further tighten: keep is_dominant clones with p_dominant <= this")
    ap.add_argument("--out-dir", default="results/clonerank/top_reads", type=Path)
    ap.add_argument("--max-reads-per-clone", type=int, default=0,
                    help="cap reads emitted per clone (0 = all)")
    args = ap.parse_args(argv[1:])

    dom = load_dominant(args.clonerank_dir, args.alpha)
    if not dom.height:
        log(f"[WARN] no is_dominant clones with p_dominant <= {args.alpha}; nothing to write")
        return 0

    wanted = [g for g in args.groups.split(",") if g] or None
    groups = [g for g in dom["group"].unique(maintain_order=True).to_list()
              if wanted is None or g in wanted]
    log(f"[INFO] {dom.height:,} dominant clones across groups: "
        f"{', '.join(sorted(set(dom['group'].to_list())))}")

    # prune stale outputs of groups no longer present (e.g. *-SN after the dominance step became
    # fitness-restricted) so top_reads/ reflects only the current group set
    if wanted is None:
        for f in sorted(args.out_dir.glob("*.dominant_clone_reads.*")):
            g = f.name.split(".dominant_clone_reads.")[0]
            if g not in groups:
                f.unlink()
                log(f"[INFO] pruned stale {f.name} (group no longer dominant/fitness-scoped)")

    for group in groups:
        gd = dom.filter(pl.col("group") == group)
        if not gd.height:
            log(f"[INFO] {group}: no dominant clones; skipping")
            continue
        process_group(group, gd, args)

    log("[INFO] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
