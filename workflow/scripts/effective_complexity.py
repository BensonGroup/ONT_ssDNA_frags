#!/usr/bin/env python
"""Effective molecular complexity (C_eff) of a sequenced pool, estimated from UMI duplicate structure.

WHAT IT MEASURES
    C_eff = 1 / P(two randomly chosen distinct reads came from the SAME original molecule)

    This is the inverse-Simpson ("effective number of species") complexity. Unlike the raw count of
    distinct UMIs -- which just grows with sequencing depth and never converges -- C_eff estimates a
    property of the POOL, not of the run, so it is comparable across samples of different depth.

THE ESTIMATOR
    Delegated in full to complexity_lib (the single source of truth; see its docstring for the
    derivation, the Cauchy-Schwarz proof that C_eff is a LOWER BOUND on richness, and the
    zero-collision case). In outline: n_i = reads carrying UMI i; N = sum(n_i);
    S = sum(n_i(n_i-1)) = ordered pairs of distinct reads sharing a UMI;
    P_collision = S / (N(N-1));  C_eff = 1 / P_collision.

NEVER REPORT C_eff ALONE (clarification.md 8.5)
    A FALL in C_eff means the pool CONCENTRATED -- but that can happen either because lineages were
    LOST (richness fell) or because the survivors became UNEVEN (richness held, skew grew), and
    C_eff cannot tell you which. So every row here also carries `dup_pairs` (how much evidence the
    estimate rests on), `max_n_i`, and `top10_share_of_S` (the share of the collisions carried by
    the ten biggest lineages). Lost lineages do not create high-count UMIs; PCR skew and clonal
    expansion do. Read them together or not at all.

INTERPRETING N / C_eff
    << 1  you sample a small fraction of the pool          -> nearly all UMIs seen once, no duplicates
    ~= 1  your read count has caught up with the pool
    >> 1  you sequence each surviving molecule many times  -> heavy duplication

TWO WAYS TO RUN IT

  1. In the workflow (`rule umi_complexity`, rules/umi.smk) -- reads the per-UMI counts that
     `umi_correct` already wrote, and emits the two TSVs the UMI cross report renders:

         effective_complexity.py --species NAME=<s>.umi_species.tsv [...] --meta sample_metadata.tsv
                                 --baseline <R0 name> --out-tsv umi_complexity.tsv
                                 --out-collapse umi_collapse.tsv

  2. By hand, off-workflow -- `--fused` re-derives the counts from a `<sample>.fused.parquet`, which
     is the ONLY way to run this on a sample that never went through `umi_correct`. That is exactly
     the empty / PCR-only round's case, so this path stays:

         effective_complexity.py --fused NAME=PATH [...] [--umi-col fused_seq]
                                 [--baseline NAME] [--pair EMPTY=REAL ...]

    --baseline   the R0 library, used to report the collapse of every other sample against it
    --pair       decompose a collapse into PROTOCOL vs SELECTION by matching an empty (no-cell) round
                 to the real round with the SAME number of transfers/PCR cycles, e.g.
                     --pair empty_R5=A549_R5
                 With --meta this pairing is derived automatically from `group: empty` + `round`.

THE THREE COLLAPSE KINDS, AND THE ONE-ROW INVARIANT
    There is no `role` field: the sample's `group` decides everything, and `empty` is a reserved
    group name (report_lib.EMPTY_GROUP) meaning "the PCR-only / no-cell round".

        protocol   C_eff(R0)       / C_eff(empty_Rk)   emitted on the EMPTY samples only
        total      C_eff(R0)       / C_eff(sample)     emitted on every EXPERIMENTAL sample
        selection  C_eff(empty_Rk) / C_eff(sample)     experimental samples WITH a round-matched empty

    EVERY NON-INITIAL SAMPLE HAS EXACTLY ONE COLLAPSE-VS-R0 ROW: `protocol` if it is in the `empty`
    group, `total` otherwise. `selection` rows are EXTRA. The empty round is therefore presented ONCE,
    under ONE label -- it never also carries a `total`, and it is never its own `selection` (which
    would be empty/empty = exactly 1.0, a number divided by itself printed as a measurement).
    `total = protocol x selection` still holds exactly, by cancellation of C_eff(empty), and is now
    derivable from three DISTINCT rows rather than asserted across duplicates.

    A consumer that wants "the collapse vs R0" must therefore key on `kind in (total, protocol)`, not
    on `total` alone -- the two are mutually exclusive per sample, so the lookup is unambiguous.

The human-readable table always goes to stdout, including in --out-tsv mode -- Snakemake captures it
into the rule's log, so the provenance of every emitted number is free.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from complexity_lib import c_eff as _c_eff       # the single source of truth for the estimator
from complexity_lib import summarize as _summarize
from report_lib import BASELINE_GROUP, EMPTY_GROUP, load_meta

TOP_K = 10          # the "top-lineage share" in clarification.md 8.5


# ------------------------------------------------------------------------------- count loaders
def counts_from_species(path):
    """Per-UMI read counts from a `<sample>.umi_species.tsv` (fused_seq <TAB> n_reads).

    Preferred in-workflow: `umi_correct` already collapsed reads to per-molecule counts, so
    re-aggregating the fused parquet would just recompute what is on disk.
    """
    counts = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            f = line.rstrip("\n").split("\t")
            if len(f) < 2:
                continue
            if i == 0 and not f[1].strip().lstrip("-").isdigit():   # header row
                continue
            counts.append(int(f[1]))
    return counts


def counts_from_fused(path, umi_col):
    """Per-UMI read counts re-derived from a `<sample>.fused.parquet` (one UMI per read).

    polars is imported lazily so the --species path -- the one the workflow uses -- has no
    third-party dependency at all and stays testable with plain python3.
    """
    import polars as pl
    vc = pl.read_parquet(path, columns=[umi_col]).group_by(umi_col).agg(pl.len().alias("n"))
    return vc["n"].to_list()


# ------------------------------------------------------------------------------- the extra fields
def skew_fields(counts, top_k=TOP_K):
    """(max_n_i, top-k share of S) -- the two numbers that separate lineage LOSS from clonal SKEW.

    Returns share=None when S == 0 (no collisions at all): there is no collision mass to apportion,
    and 0.0 would read as "the top lineages carry none of it", which is a different claim.
    """
    if not counts:
        return 0, None
    ordered = sorted(counts, reverse=True)
    S = sum(n * (n - 1) for n in ordered)
    if S == 0:
        return ordered[0], None
    return ordered[0], round(sum(n * (n - 1) for n in ordered[:top_k]) / S, 4)


def measure(name, counts, rarefy_to=None):
    """One sample's full complexity record. The field set comes from complexity_lib.summarize()."""
    rec = dict(_summarize(counts, rarefy_to=rarefy_to))
    rec["name"] = name
    rec["max_n_i"], rec["top10_share_of_S"] = skew_fields(counts)
    # the UNROUNDED C_eff, kept out of the TSV but used for every ratio below
    rec["_c"] = _c_eff(counts)[3]
    return rec


# ------------------------------------------------------------------------------- collapse ratios
def _ratio(num, den):
    """A collapse ratio, with its bound direction propagated from the two C_eff estimates.

    A C_eff that is itself only a lower bound (no collisions observed) makes the ratio a bound too,
    and in which DIRECTION depends on which side it sits: a bounded numerator gives a lower bound on
    the collapse, a bounded denominator an upper bound. Both bounded -> the ratio says nothing, and
    we say so rather than printing a number that looks like a measurement.
    """
    if not num["_c"] or not den["_c"] or math.isinf(den["_c"]) or math.isinf(num["_c"]):
        return None, "?"
    r = _round_ratio(num["_c"] / den["_c"])
    nb, db = num["c_eff_is_lower_bound"], den["c_eff_is_lower_bound"]
    if nb and db:
        return r, "?"
    if nb:
        return r, ">"
    if db:
        return r, "<"
    return r, ""


def _round_ratio(r):
    """Round a collapse ratio for reading -- but NEVER down to zero.

    `round(r, 4)` is fine at or above 1x (a collapse), and destroys the measurement below it: a sub-1
    ratio means complexity HELD OR ROSE, and an un-bottlenecked `empty` round (no duplicate reads, a
    huge C_eff) can push C_eff(R0)/C_eff(empty) to 1e-5 or smaller. Flattened to 0.0 that reads as a
    total collapse -- the exact opposite of what happened -- and downstream truthiness tests would
    then blank the cell entirely. Keep 4 SIGNIFICANT figures below 1 instead of 4 decimal places.
    """
    return round(r, 4) if r >= 1 else float(f"{r:.4g}")


def _empty_samples(recs, pairs):
    """The names of the empty (PCR-only / no-cell) rounds.

    Two entry points name the empty round two different ways, so both are honoured and UNIONED:

      * the workflow / `--meta` path -- the sample's config `group` IS the reserved name `empty`.
        There is no `role` field any more; the group decides everything.
      * the off-workflow `--pair EMPTY=REAL` path -- there is no config and usually no `--meta`, so
        the sample's `group` is blank. The empty round is then, by definition, whatever was named on
        the LEFT of a `--pair`. Without this, `--pair` would emit no `protocol` row at all, which is
        the one number that flag exists to produce.

    Union, not either-or: a `group: empty` sample must NEVER be handed a `total` row, whichever flag
    the caller reached for.
    """
    empties = {n for n, r in recs.items() if r.get("group") == EMPTY_GROUP}
    if pairs is not None:
        empties |= {e for e in pairs.values() if e in recs}
    return empties


def collapse_rows(recs, baseline, pairs=None):
    """Collapse of the pool relative to the R0 library, decomposed where an empty round exists.

        protocol   C_eff(R0)       / C_eff(empty_Rk)   on the EMPTY samples -- transfers + PCR alone
        total      C_eff(R0)       / C_eff(sample)     on every EXPERIMENTAL sample
        selection  C_eff(empty_Rk) / C_eff(sample)     <- the biologically real part

    THE INVARIANT: every non-initial sample gets exactly ONE of {total, protocol}. The empty round's
    collapse vs R0 IS the protocol collapse, so it is labelled `protocol` and stated ONCE -- not
    restated as a duplicate `protocol` row hung off every experimental sample (which is what this
    used to do), and never as a `total`.

    THE TRAP this guards: the empty round is itself a round-k sample, so a naive "match each sample
    to the empty of the same round" makes it ITS OWN match -- a `selection` row of empty/empty =
    exactly 1.0, a number divided by itself rendered as a real measurement. The gate is on the GROUP
    (`_empty_samples`), not on the round.

    PAIRING RULE: an empty round is matched to a real sample BY ROUND, and one empty fans out to
    EVERY experimental sample of that round -- an empty R5 has seen the same transfers and PCR cycles
    as both arms' R5, so it is the null for both. Two empties in the same round is an error, not a
    coin-flip: which one is the null for that round is not something this script may guess.

    NOTE this is a DESCRIPTIVE decomposition, not a causal one (clarification.md 7): it attributes
    to "protocol" whatever the empty round also lost, which assumes the two lose independently.
    """
    base = recs.get(baseline)
    if base is None:
        return []

    empties = _empty_samples(recs, pairs)

    by_round = {}
    if pairs is None:                        # derive the pairing from group+round (the --meta path)
        for n, r in recs.items():
            if n not in empties:
                continue
            rnd = r.get("round")
            if rnd in by_round:
                # KEPT even though common.smk now rejects this at PARSE time: the off-workflow
                # `--fused --meta` path never goes near snakemake or the config, so this is still
                # the only guard there.
                raise SystemExit(
                    f"[FATAL] empty-round samples {by_round[rnd]!r} and {n!r} are both round {rnd}. "
                    f"An empty round is the null FOR ITS ROUND; which of the two is that null is not "
                    f"something this script may guess. Give them distinct rounds.")
            by_round[rnd] = n

        # WARN, do not raise. An empty round at a round no arm ran is not a broken config -- it is a
        # legitimate thing to have sequenced. But it then yields ZERO selection rows, and without
        # this line nothing would say why the decomposition is missing.
        exp_rounds = {r.get("round") for n, r in recs.items()
                      if n != baseline and n not in empties and r.get("group") != BASELINE_GROUP}
        for rnd, n in by_round.items():
            if rnd not in exp_rounds:
                sys.stderr.write(
                    f"[WARN] empty round {n!r} is round {rnd}, but no experimental sample is at "
                    f"round {rnd}. It still gets its own `protocol` collapse row; it is simply the "
                    f"null for nothing, so NO `selection` rows are derived from it.\n")

    rows = []
    for n, r in recs.items():
        if n == baseline or r.get("group") == BASELINE_GROUP:
            continue                                     # the R0 library has no collapse vs itself
        # `sample` is the row's SUBJECT (whose collapse is being decomposed); `num`/`den` name the two
        # samples the ratio is actually taken between. Naming them explicitly is the difference
        # between a readable row and one that claims to compare two samples it never touched.
        if n in empties:
            kinds = [("protocol", base, r)]              # its collapse vs R0 IS the protocol collapse
        else:
            kinds = [("total", base, r)]
            empty = pairs.get(n) if pairs is not None else by_round.get(r.get("round"))
            ctl = recs.get(empty) if empty else None
            if ctl is not None:                          # a round-matched empty round exists
                kinds.append(("selection", ctl, r))
        for kind, num, den in kinds:
            ratio, bound = _ratio(num, den)
            rows.append({"group": r.get("group", ""), "round": r.get("round", ""), "sample": n,
                         "kind": kind, "num": num["name"], "den": den["name"],
                         "ratio": ratio, "ratio_bound": bound})
    return rows


# ------------------------------------------------------------------------------------------ output
TSV_COLS = ["name", "group", "round", "n_reads", "n_distinct", "dup_pairs",
            "collision_prob", "c_eff", "c_eff_is_lower_bound", "n_over_c_eff",
            "max_n_i", "top10_share_of_S", "rarefied_to", "n_distinct_rarefied"]
# UNCHANGED by the group-only sample model, deliberately: `kind` + `sample` already say everything a
# consumer needs (which of the three collapses this is, and whose). Add no column, add no kind.
COLLAPSE_COLS = ["group", "round", "sample", "kind", "num", "den", "ratio", "ratio_bound"]


def _cell(v):
    if v is None:
        return ""                     # never let None/inf/nan reach a table cell
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def write_tsv(path, cols, rows):
    with open(path, "w") as out:
        out.write("\t".join(cols) + "\n")
        for r in rows:
            out.write("\t".join(_cell(r.get(c)) for c in cols) + "\n")


def print_table(recs):
    print(f"{'sample':<22}{'N reads':>11}{'distinct':>11}{'dup pairs':>11}"
          f"{'C_eff':>18}{'N/C_eff':>10}{'max n_i':>9}{'top10 %S':>10}")
    for name, r in recs.items():
        C = r["_c"]
        mark = ">" if r["c_eff_is_lower_bound"] else " "
        ce = "inf" if math.isinf(C) else f"{C:,.0f}"
        noc = "-" if not C or math.isinf(C) else f"{r['n_reads'] / C:.3f}"
        top = "-" if r["top10_share_of_S"] is None else f"{100 * r['top10_share_of_S']:.1f}%"
        print(f"{name:<22}{r['n_reads']:>11,}{r['n_distinct']:>11,}{r['dup_pairs']:>11,}"
              f"{mark}{ce:>17}{noc:>10}{r['max_n_i']:>9,}{top:>10}")
    print("\n  '>' = no duplicates observed; C_eff is a 95% LOWER BOUND (see complexity_lib).")
    print("  C_eff is a lower bound on richness, NEVER a headcount -- read it with dup_pairs,")
    print("  max n_i and top10 %S: a fall in C_eff is lineage LOSS or clonal SKEW, and only")
    print("  those columns say which.")


def fold(v):
    """A collapse ratio. Below 1x it is NOT a collapse -- complexity held or rose -- and rounding it
    to '0.0x' would hide that, so small ratios keep their significant figures."""
    if v is None:
        return "-"
    return f"{v:,.1f}x" if v >= 1 else f"{v:.3g}x"


def print_collapse(rows):
    if not rows:
        return
    print("\nCollapse (C_eff ratios; '>' / '<' = the ratio is a bound, see --out-collapse)")
    for r in rows:
        tag = {"total": "total collapse    ",
               "protocol": "PROTOCOL collapse ",
               "selection": "SELECTION collapse"}[r["kind"]]
        # The protocol line is the EMPTY ROUND'S OWN collapse vs R0 -- it is that sample's one and
        # only vs-R0 row, standing exactly where an experimental sample's `total` stands. Saying so
        # here is what stops it being read as a fourth, free-floating number.
        note = {"protocol": "   <- the empty round's OWN collapse vs R0 (transfers + PCR, no cells)",
                "selection": "   <- the biologically real part"}.get(r["kind"], "")
        pair = f"{r['num']} -> {r['den']}"
        print(f"  {tag} {pair:<48} {r['ratio_bound']}{fold(r['ratio']):>12}{note}")
    if any(r["kind"] == "protocol" for r in rows):
        print("\n  Every non-initial sample has exactly ONE row above vs R0: `protocol` for the empty")
        print("  round, `total` for everything else. total = protocol x selection (C_eff(empty)")
        print("  cancels), so the three are one number stated three ways, not three measurements.")


# --------------------------------------------------------------------------------------------- cli
def main(argv):
    ap = argparse.ArgumentParser(description="Effective molecular complexity from UMI duplicates.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--species", nargs="+", metavar="NAME=PATH",
                     help="one or more NAME=<sample>.umi_species.tsv (the counts umi_correct wrote)")
    src.add_argument("--fused", nargs="+", metavar="NAME=PATH",
                     help="one or more NAME=<sample>.fused.parquet (re-derives the counts)")
    ap.add_argument("--umi-col", default="fused_seq",
                    help="--fused only: fused_seq (raw) | fused_seq_corrected (after the umi/ branch)")
    ap.add_argument("--meta", default=None,
                    help="sample_metadata.tsv (name/group/round/idx); supplies group+round and lets "
                         "the empty-round pairing be derived instead of spelled out with --pair")
    ap.add_argument("--baseline", default=None, help="name of the R0 library, for the total collapse")
    ap.add_argument("--pair", nargs="*", default=None, metavar="EMPTY=REAL",
                    help="decompose protocol vs selection explicitly (overrides the --meta pairing)")
    ap.add_argument("--rarefy-to", type=int, default=None,
                    help="common depth for the rarefied richness column (default: the shallowest "
                         "sample, which is the only depth every sample can be rarefied DOWN to)")
    ap.add_argument("--out-tsv", default=None, help="per-sample complexity table")
    ap.add_argument("--out-collapse", default=None, help="per-sample collapse ratios")
    args = ap.parse_args(argv[1:])

    if args.species and args.umi_col != "fused_seq":
        ap.error("--umi-col applies to --fused only (--species reads counts that are already per-UMI)")

    specs = args.species or args.fused
    counts = {}
    for spec in specs:
        name, _, path = spec.partition("=")
        if not path:
            ap.error(f"expected NAME=PATH, got {spec!r}")
        counts[name] = (counts_from_species(path) if args.species
                        else counts_from_fused(path, args.umi_col))

    # The rarefaction depth: the SHALLOWEST sample, over every sample in this run (the R0 library
    # included). complexity_lib.rarefy() returns NaN when asked to rarefy UP, so the global minimum
    # is the only depth at which every row is computable -- and rarefying the baseline to a
    # different depth than the samples it is the baseline FOR would reintroduce exactly the
    # depth artefact the column exists to remove.
    m = args.rarefy_to or min((sum(c) for c in counts.values() if c), default=None)

    meta = load_meta(args.meta) if args.meta else {}
    recs = {}
    for name, c in counts.items():
        r = measure(name, c, rarefy_to=m)
        mm = meta.get(name, {})
        r["group"], r["round"] = mm.get("group", ""), mm.get("round", "")
        recs[name] = r

    print_table(recs)

    pairs = None
    if args.pair is not None:
        pairs = {}
        for spec in args.pair:
            empty, _, real = spec.partition("=")
            if empty not in recs or real not in recs:
                sys.stderr.write(f"[skip --pair {spec}: unknown sample]\n")
                continue
            pairs[real] = empty

    rows = []
    if args.baseline:
        if args.baseline not in recs:
            sys.stderr.write(f"[FATAL] --baseline {args.baseline!r} is not among the samples given\n")
            return 1
        rows = collapse_rows(recs, args.baseline, pairs)
        print_collapse(rows)

    if args.out_tsv:
        write_tsv(args.out_tsv, TSV_COLS, [recs[n] for n in recs])
        print(f"\n[complexity] {len(recs)} samples -> {args.out_tsv}")
    if args.out_collapse:
        write_tsv(args.out_collapse, COLLAPSE_COLS, rows)
        print(f"[complexity] {len(rows)} collapse rows -> {args.out_collapse}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
