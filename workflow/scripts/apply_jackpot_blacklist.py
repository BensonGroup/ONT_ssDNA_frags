#!/usr/bin/env python3
"""Drop jackpot-artefact reads from one sample's corrected UMI tree: umicorr/<s>/ -> umi/<s>/.

ONE-DIRECTIONAL, FILTER-ONLY. Nothing is re-corrected, re-clustered or collapsed: this removes the
reads whose corrected UMI is on the blacklist that the R0 library produced, and re-derives the
aggregates over the survivors. That distinction is the whole licence for this step --
clarification.md:667 forbids deduplicating (collapsing every high-count lineage destroys the
signal); dropping a NAMED, AUDITED set of artefact barcodes exposed by the naive library does not.

The blacklist is keyed on `fused_seq_corrected`, so this must run on the CORRECTED tree. Detection
happens once, on R0 only (see detect_umi_jackpots.py for why); every sample is filtered against that
one blacklist.

Why this writes the whole tree rather than just the fused parquet: `build_fragvec.py` derives
`reads_decoded` from the FRAGMENTS parquet and inner-joins the UMIs, so passing fragments through
unfiltered would silently over-count `reads_decoded` and `reads_dropped_anchor_only`. Every artifact
in a tree must agree on its read set.

Usage:
    apply_jackpot_blacklist.py --src-stem results/umicorr/NB01/NB01 \
        --dst-dir results/umi/NB01 --blacklist results/jackpot/R0.umi_blacklist.tsv [--match-ed1]
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tree_io as tio
from complexity_lib import c_eff, collision_prob

TOOL_VERSION = "apply_jackpot_blacklist.py/1.0"


def within_ed1(a: str, b: str) -> bool:
    """True iff Levenshtein edit distance(a, b) <= 1 (substitution / single indel / equal).

    Pure, deterministic, allocation-light. Length gap > 1 => False immediately. Equal length =>
    at most one mismatch. Length differs by 1 => the longer must equal the shorter with exactly
    one character inserted (single deletion), checked by a two-pointer walk allowing one skip.
    """
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        mism = 0
        for ca, cb in zip(a, b):
            if ca != cb:
                mism += 1
                if mism > 1:
                    return False
        return True
    # length differs by exactly 1: b is the shorter, a the longer
    if la < lb:
        a, b = b, a
        la, lb = lb, la
    i = j = 0
    skipped = False
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1
            j += 1
        else:
            if skipped:
                return False
            skipped = True
            i += 1  # skip one char in the longer string
    return True


def removed_read_accs(fused: pl.DataFrame, blacklist_umis: list, match_ed1: bool):
    """(removed_accs set, per_umi_hits dict). Matches on `fused_seq_corrected`.

    Exact set membership always; with ``match_ed1`` also removes reads within Levenshtein 1 of any
    blacklisted UMI. A removed read is attributed to the FIRST blacklist UMI (blacklist order) it
    matches, which keeps the attribution deterministic.
    """
    bl_set = set(blacklist_umis)
    hits = {u: 0 for u in blacklist_umis}
    removed, cache = set(), {}
    for acc, seq in zip(fused["read_acc"].to_list(), fused["fused_seq_corrected"].to_list()):
        if seq in bl_set:
            removed.add(acc)
            hits[seq] += 1
            continue
        if not match_ed1:
            continue
        if seq not in cache:                       # per DISTINCT sequence, not per read
            cache[seq] = next((u for u in blacklist_umis if within_ed1(seq, u)), None)
        hit = cache[seq]
        if hit is not None:
            removed.add(acc)
            hits[hit] += 1
    return removed, hits


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Drop jackpot-blacklisted reads from a umicorr tree.")
    ap.add_argument("--src-stem", required=True, help="umicorr stem, e.g. results/umicorr/NB01/NB01")
    ap.add_argument("--dst-dir", required=True, help="output sample dir, e.g. results/umi/NB01")
    ap.add_argument("--blacklist", required=True, help="the R0 library's umi_blacklist.tsv")
    ap.add_argument("--match-ed1", action="store_true",
                    help="also remove reads within Levenshtein 1 of a blacklisted UMI "
                         "(default: exact match only)")
    ap.add_argument("--force", action="store_true", help="overwrite a non-empty destination")
    args = ap.parse_args(argv[1:])

    src = Path(args.src_stem)
    if not src.is_absolute():
        src = Path.cwd() / src
    stem = src.name
    dst = Path(args.dst_dir).resolve()

    fused_pq = tio.part(src, ".fused.parquet")
    frags_pq = tio.part(src, ".fragments.parquet")
    umis_pq = tio.part(src, ".umis.parquet")
    summ_tsv = tio.part(src, ".umi_summary.tsv")
    src_log = tio.part(src, ".ligateddna.log")
    for p in (fused_pq, frags_pq, umis_pq, summ_tsv, Path(args.blacklist)):
        if not p.exists():
            print(f"[FATAL] missing input: {p}", file=sys.stderr)
            return 1

    tio.prepare_dst(dst, args.force)

    bl = pl.read_csv(args.blacklist, separator="\t")
    if "umi" not in bl.columns:
        print(f"[FATAL] blacklist has no `umi` column: {args.blacklist}", file=sys.stderr)
        return 1
    # dedupe preserving first-appearance order, so the ed1 attribution above is deterministic
    seen, bl_umis = set(), []
    for u in bl["umi"].to_list():
        if u not in seen:
            seen.add(u)
            bl_umis.append(u)

    fused = pl.read_parquet(fused_pq)
    reads_in = fused.height
    removed, hits = removed_read_accs(fused, bl_umis, args.match_ed1)
    kept = fused.filter(~pl.col("read_acc").is_in(list(removed)))
    reads_kept = kept.height
    reads_removed = reads_in - reads_kept
    # SANITY: the house invariant -- drops + keeps == input.
    assert reads_removed + reads_kept == reads_in, f"{reads_removed} + {reads_kept} != {reads_in}"

    # ---- re-derive the aggregates over the SURVIVORS ----------------------------------------
    # umi_read_count is "reads sharing this read's corrected UMI in this sample" -- it must be
    # recomputed, not carried, or a partially-removed molecule reports a stale family size.
    species = (
        kept.group_by("fused_seq_corrected").agg(pl.len().alias("n_reads"))
        .sort(["n_reads", "fused_seq_corrected"], descending=[True, False])
    )
    out = (
        kept.drop("umi_read_count")
        .join(species.rename({"n_reads": "umi_read_count"}), on="fused_seq_corrected", how="left")
        .select("read_acc", "strand", "fused_seq", "fused_len", "fused_seq_corrected",
                pl.col("umi_read_count").cast(pl.UInt32))
        .sort("read_acc")
    )
    out.write_parquet(dst / f"{stem}.fused.parquet")
    # include_header=True: this file replaces umi_correct's, and build_umi_matrix.py /
    # effective_complexity.py read it WITH a header. A headerless copy would silently eat a row.
    species.write_csv(dst / f"{stem}.umi_species.tsv", separator="\t", include_header=True)

    # fragments MUST follow the same read set (see the module docstring)
    tio.write_filtered_parquet(
        frags_pq, dst / f"{stem}.fragments.parquet", out.select("read_acc"),
        ["read_acc", "start", "stop", "fragment_id"])
    # umis too: featurecount and report_sample read the anchor VOCABULARY from it, by stem
    tio.write_filtered_parquet(
        umis_pq, dst / f"{stem}.umis.parquet", out.select("read_acc"), ["read_acc", "umi_index"])

    # ---- summary: carry umicorr's funnel, append the jackpot columns, re-derive complexity ---
    base = pl.read_csv(summ_tsv, separator="\t").to_dicts()[0]
    counts = species["n_reads"].to_list()
    _N, _D, _pairs, _C, _lower = c_eff(counts)
    base.update({
        "reads_kept": reads_kept,
        "n_species_corrected": species.height,
        "jackpot_blacklist_umis": len(bl_umis),
        "jackpot_umis_hit": sum(1 for v in hits.values() if v),
        "jackpot_reads_removed": reads_removed,
        "jackpot_pct_removed": round(100.0 * reads_removed / reads_in, 4) if reads_in else 0.0,
        "jackpot_match_ed1": args.match_ed1,
        # C_eff is re-derived HERE, on the cleaned pool -- that is the point of the whole stage
        # (clarification.md:399-403: the weighting is quadratic, so an artifact in the top lineage
        # corrupts S more than any biology can).
        "dup_pairs": _pairs,
        "collision_prob": collision_prob(counts),
        "c_eff": round(_C, 1) if _C != float("inf") else None,
        "c_eff_is_lower_bound": _lower,
        "n_over_c_eff": round(_N / _C, 4) if _C not in (0, float("inf")) else None,
    })
    pl.DataFrame([base]).write_csv(dst / f"{stem}.umi_summary.tsv", separator="\t")

    # ligateddna.log: carried + a provenance block. parse_log recovers the expected-sequences CSV
    # from it, which every report reads.
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _orig = src_log.read_text() if src_log.exists() else ""
    (dst / f"{stem}.ligateddna.log").write_text(_orig + (
        "\n\n### jackpot filter provenance ###\n"
        f"tool: {TOOL_VERSION}\n"
        f"timestamp_utc: {ts}\n"
        f"mode: filter-only (a named, audited artefact set -- NOT deduplication)\n"
        f"blacklist: {args.blacklist}\n"
        f"params: match_ed1={args.match_ed1} blacklist_umis={len(bl_umis)}\n"
        f"counts: reads_in={reads_in} jackpot_reads_removed={reads_removed} "
        f"reads_kept={reads_kept} n_species_corrected={species.height}\n"))

    pct = round(100.0 * reads_removed / reads_in, 4) if reads_in else 0.0
    print(f"[{stem}] {reads_in:,} reads in -> {reads_kept:,} kept "
          f"({reads_removed:,} jackpot reads removed, {pct}%); "
          f"{species.height:,} UMI species @ {ts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
