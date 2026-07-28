#!/usr/bin/env python
"""Per-vector round-over-round fitness (log2 fold-change) from fragvec matrices.

STANDALONE per-study analysis -- NOT part of the Snakemake DAG. It reads the
already-produced per-group fragment-vector matrices under ``results/fragvec/`` and
writes new tables under ``results/fitness/``. Nothing here mutates the pipeline,
its config, or any existing ``results/`` tree.

What it computes
----------------
For each group (subjects + the control) and each fragment-vector, the abundance
change from every selection round to the round before it, on a depth-normalized
frequency scale::

    rpm[v, s]              = 1e6 * n_reads[v, s] / T_s          (T_s = sample depth)
    log2_R{c}_vs_R{p}[v]   = log2( (rpm[v, c] + EPS) / (rpm[v, p] + EPS) )

The EPS floor (on the rpm scale) keeps zeros finite and depth-independent: a vector
absent from both neighbouring rounds scores exactly 0; a dropout scores strongly
negative, an emergence strongly positive. This mirrors the legacy
``fig_enrichment`` convention ``log2((c + EPS) / (a + EPS))``.

The control ("empty", the no-cell / PCR-only arm) is then subtracted from each
subject in log space -- i.e. the subject's fold-change is *divided* by the empty's
fold-change -- to strip protocol/PCR drift from real selection::

    A549_vs_empty[v, step] = A549_log2[v, step] - empty_log2[v, step]

A vector absent from the empty arm contributes 0 (no correction). Because the R0
baseline is shared across every group, the first step's baseline cancels:
``A549_vs_empty[R1_vs_R0] = log2(A549_R1_rpm / empty_R1_rpm)``.

Data source (most efficient structure)
--------------------------------------
The per-group ``<group>.fragvec_x_sample.long.parquet`` (columns
``vector, sample, n_reads, n_umis``) already bundles the shared R0 baseline plus all
of a group's rounds in one file, so a single read + pivot yields the vector x sample
matrix. The ``.samples.tsv`` sidecar gives the canonical sample order (baseline
first). The ``.mtx`` sidecars hold the same numbers but need their label files
reconstructed -- more IO for no gain.

CLI (all defaulted, so it runs with no args from the repo root)
---------------------------------------------------------------
    pixi run -e default python scripts/fitness_step_ratios.py \
      --fragvec-dir results/fragvec \
      --meta        results/umi/sample_metadata.tsv \
      --subjects    A549,MDA \
      --control     empty \
      --out-dir     results/fitness \
      --min-reads   10 \
      --eps         1.0

Emits (under --out-dir):
    <subject>.log2_step_ratios.tsv       one per subject   (signal vectors)
    <control>.log2_step_ratios.tsv       the control arm   (signal vectors)
    <subject>_vs_<control>.log2_step_ratios.tsv            (subject signal vectors,
                                                            empty-subtracted)
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fitness_lib import RPM, group_long_path, sample_depths   # noqa: E402  one definition of T

_ROUND_RE = re.compile(r"_R(\d+)$")


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# sample -> round labelling
# --------------------------------------------------------------------------- #
def load_round_map(meta_path: Path) -> dict[str, int]:
    """name -> round int, from sample_metadata.tsv (columns name/group/round/idx).

    Returns {} when the file is absent; callers fall back to name parsing.
    """
    if not meta_path.exists():
        log(f"[WARN] metadata not found ({meta_path}); falling back to name parsing")
        return {}
    out: dict[str, int] = {}
    with open(meta_path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        try:
            i_name, i_round = header.index("name"), header.index("round")
        except ValueError as e:
            raise SystemExit(f"[FATAL] {meta_path}: missing name/round column ({e})")
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) <= max(i_name, i_round):
                continue
            out[f[i_name]] = int(f[i_round])
    return out


def sample_round(name: str, round_map: dict[str, int]) -> int:
    """Round index for a sample: metadata first, else a trailing ``_R<n>`` in the
    name, else 0 (the baseline / initial library)."""
    if name in round_map:
        return round_map[name]
    m = _ROUND_RE.search(name)
    return int(m.group(1)) if m else 0


# --------------------------------------------------------------------------- #
# per-group step table
# --------------------------------------------------------------------------- #
def group_step_table(
    group: str, fragvec_dir: Path, round_map: dict[str, int], eps: float
) -> tuple[pl.DataFrame, list[str]]:
    """Full per-vector step table for one group (ALL vectors, unfiltered).

    Returns (df, steps) where df has columns
        vector, <log2 step cols...>, reads_R<r>..., max_reads
    and steps is the ordered list of log2 step column names.
    """
    long_p = group_long_path(fragvec_dir, group)
    samples_p = fragvec_dir / f"{group}.fragvec_x_sample.samples.tsv"
    if not long_p.exists():
        raise SystemExit(f"[FATAL] missing matrix: {long_p}")
    if not samples_p.exists():
        raise SystemExit(f"[FATAL] missing sample order: {samples_p}")

    ordered = [s for s in samples_p.read_text().splitlines() if s.strip()]
    rounds = [sample_round(s, round_map) for s in ordered]

    df = pl.read_parquet(long_p, columns=["vector", "sample", "n_reads"])

    # depth per sample over the FULL table (true frequency denominator). Shared with
    # fitness_lib.auto_min_log2_sum, which prices the threshold off exactly this number.
    tot = sample_depths(df)

    wide = df.pivot(
        on="sample", index="vector", values="n_reads", aggregate_function="sum"
    ).fill_null(0)

    # each round appears once per group, so reads_R<r> names are unique
    reads_rename = {s: f"reads_R{r}" for s, r in zip(ordered, rounds)}
    rpm_expr = {
        s: (pl.col(s).cast(pl.Float64) * RPM / float(tot[s])) for s in ordered
    }

    # log2 fold-change of consecutive rounds in canonical order
    steps: list[str] = []
    step_exprs: list[pl.Expr] = []
    pairs = list(zip(ordered, rounds))
    for (sp, rp), (sc, rc) in zip(pairs, pairs[1:]):
        name = f"log2_R{rc}_vs_R{rp}"
        steps.append(name)
        step_exprs.append(
            ((rpm_expr[sc] + eps) / (rpm_expr[sp] + eps)).log(2).round(4).alias(name)
        )

    out = wide.with_columns(step_exprs)
    out = out.with_columns(
        [pl.col(s).alias(reads_rename[s]) for s in ordered]
    )
    out = out.with_columns(
        pl.max_horizontal([pl.col(s) for s in ordered]).alias("max_reads")
    )
    keep = ["vector", *steps, *[reads_rename[s] for s in ordered], "max_reads"]
    out = out.select(keep).sort("max_reads", descending=True)
    return out, steps


def write_tsv(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path, separator="\t")
    log(f"[INFO] wrote {path}  ({df.height:,} rows x {df.width} cols)")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fragvec-dir", default="results/fragvec", type=Path)
    ap.add_argument("--meta", default="results/umi/sample_metadata.tsv", type=Path)
    ap.add_argument("--subjects", default="A549,MDA",
                    help="comma-separated subject groups (empty-normalized)")
    ap.add_argument("--control", default="empty",
                    help="control group subtracted from each subject in log space")
    ap.add_argument("--out-dir", default="results/fitness", type=Path)
    ap.add_argument("--min-reads", type=int, default=10,
                    help="signal filter: keep a vector with >= this n_reads in >=1 round")
    ap.add_argument("--eps", type=float, default=1.0,
                    help="pseudocount on the rpm scale (keeps zero ratios finite)")
    args = ap.parse_args(argv[1:])

    subjects = [s for s in args.subjects.split(",") if s]
    control = args.control
    round_map = load_round_map(args.meta)

    # control: full table (background for subtraction) + its own signal deliverable
    ctrl_full, ctrl_steps = group_step_table(control, args.fragvec_dir, round_map, args.eps)
    ctrl_sig = ctrl_full.filter(pl.col("max_reads") >= args.min_reads)
    write_tsv(ctrl_sig, args.out_dir / f"{control}.log2_step_ratios.tsv")

    ctrl_bg = ctrl_full.select(["vector", *ctrl_steps])  # vector + empty log2 steps

    for subj in subjects:
        subj_full, subj_steps = group_step_table(subj, args.fragvec_dir, round_map, args.eps)
        subj_sig = subj_full.filter(pl.col("max_reads") >= args.min_reads)
        write_tsv(subj_sig, args.out_dir / f"{subj}.log2_step_ratios.tsv")

        common = [s for s in subj_steps if s in ctrl_steps]
        if common != subj_steps or common != ctrl_steps:
            log(f"[WARN] {subj} vs {control}: subtracting only shared steps "
                f"{common} (subject={subj_steps}, control={ctrl_steps})")

        # subject signal vectors, empty-subtracted on shared steps
        bg = ctrl_bg.rename({s: f"__ctrl__{s}" for s in ctrl_steps})
        joined = subj_sig.join(bg, on="vector", how="left")
        empty_present = pl.col(f"__ctrl__{common[0]}").is_not_null().alias("empty_present")
        corrected = [
            (pl.col(s) - pl.col(f"__ctrl__{s}").fill_null(0.0)).round(4).alias(s)
            for s in common
        ]
        vs = joined.with_columns([*corrected, empty_present]).select(
            ["vector", *common, "max_reads", "empty_present"]
        ).rename({"max_reads": "subject_max_reads"})
        write_tsv(vs, args.out_dir / f"{subj}_vs_{control}.log2_step_ratios.tsv")

    log("[INFO] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
