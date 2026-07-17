#!/usr/bin/env python
"""Regenerate the UMI & fragvec CROSS-REPORTS on jackpot-blacklist-CLEANED data.

ONE-DIRECTIONAL, FILTER-ONLY. Jackpot detection runs AFTER umi correction; this step does
nothing but DROP the reads whose corrected UMI is on the jackpot blacklist. Every surviving
read keeps its data exactly as before -- nothing is re-corrected or re-clustered. Because the
blacklist is a set of *corrected* UMIs and a corrected-UMI species is either wholly kept or
wholly dropped, the survivor aggregates are the existing aggregates with the blacklisted rows
removed (and, for clonotypes that contained a blacklisted UMI, re-summed over their surviving
UMIs). Surviving species' and unaffected clonotypes' numbers are untouched.

It then REUSES the pipeline's own downstream scripts on the survivor tables -- no report or
aggregation logic is reimplemented:
    effective_complexity.py   -> cleaned umi_complexity.tsv / umi_collapse.tsv
    build_fragvec_matrix.py   -> cleaned <group>.fragvec_x_sample.* matrices
    build_umi_report_cross.py -> cleaned umi/umi_report_cross.html      (env-driven)
    build_fragvec_report_cross.py -> cleaned fragvec/fragvec_report_cross.html (env-driven)

The initial (R0) library is resolved from the config's reserved 'initial' group -- never a
hardcoded sample name -- and is written to the sibling *_initial/ trees exactly like the real
pipeline, so the report globs see only non-initial samples and inject the baseline via L2F_R0_STEM.

CLI:
    build_jackpot_clean_reports.py [--results-dir DIR] [--blacklist TSV] [--out-dir DIR]
        [--sample-metadata TSV] [--min-reads 10] [--samples S1,S2,...] [--skip-reports] [--force]
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_lib import BASELINE_GROUP, load_meta
from complexity_lib import c_eff, collision_prob

SCRIPTS = Path(__file__).resolve().parent
PY = sys.executable
DEFAULT_RES = Path("/home/hongj/projects/nb_invivo/results")


# ---- survivor summary helpers (recompute ONLY aggregate counts over survivors) -------------
def _cplx(counts: list[int]) -> dict:
    """dup_pairs / collision_prob / c_eff / n_over_c_eff from a list of per-species read counts."""
    n, d, pairs, ce, lower = c_eff(counts)
    return {
        "dup_pairs": pairs,
        "collision_prob": collision_prob(counts),
        "c_eff": round(ce, 1) if ce != float("inf") else None,
        "c_eff_is_lower_bound": lower,
        "n_over_c_eff": round(n / ce, 4) if ce not in (0, float("inf")) else None,
    }


def _write_tsv_row(path: Path, row: dict) -> None:
    pl.DataFrame([row]).write_csv(path, separator="\t")


def clean_umi_sample(fused_pq: Path, orig_summary: Path, bl: set, out_dir: Path, sample: str):
    """Drop blacklisted corrected UMIs from one sample's per-read fused table; emit the
    survivor fused.parquet, umi_species.tsv and umi_summary.tsv the umi cross-report reads."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fused = pl.read_parquet(fused_pq)
    surv = fused.filter(~pl.col("fused_seq_corrected").is_in(list(bl)))
    surv.write_parquet(out_dir / f"{sample}.fused.parquet")

    species = (surv.group_by("fused_seq_corrected").agg(pl.len().alias("n_reads"))
               .sort(["n_reads", "fused_seq_corrected"], descending=[True, False]))
    species.write_csv(out_dir / f"{sample}.umi_species.tsv", separator="\t")

    # start from the original summary; override ONLY the kept-side (survivor) count fields.
    base = pl.read_csv(orig_summary, separator="\t").row(0, named=True) if orig_summary.exists() else {"sample": sample}
    n_raw = surv["fused_seq"].n_unique()
    n_corr = surv["fused_seq_corrected"].n_unique()
    base.update({
        "reads_kept": surv.height,
        "n_species_raw": n_raw,
        "n_species_corrected": n_corr,
        "n_species_merged": n_raw - n_corr,
        "n_reads_relabeled": surv.filter(pl.col("fused_seq") != pl.col("fused_seq_corrected")).height,
    })
    base.update(_cplx(species["n_reads"].to_list()))
    _write_tsv_row(out_dir / f"{sample}.umi_summary.tsv", base)
    return fused.height - surv.height  # reads removed


def clean_fragvec_sample(counts_pq: Path, clones_pq: Path, orig_summary: Path, bl: set,
                         out_dir: Path, sample: str, min_reads: int):
    """Drop blacklisted UMIs; re-sum only the clonotypes that contained one. Unaffected
    clonotypes (incl. the whole noise tail) keep their exact original numbers."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = pl.read_parquet(counts_pq)
    clones = pl.read_parquet(clones_pq)
    bl_list = list(bl)

    affected = set(clones.filter(pl.col("umi").is_in(bl_list))["vector"].to_list())
    surv_clones = clones.filter(~pl.col("umi").is_in(bl_list))
    surv_clones.write_parquet(out_dir / f"{sample}.fragvec_clones.parquet")

    # recompute the affected clonotypes from their surviving clones (clones hold every UMI of a
    # signal-carrying vector, so sum/count/max over survivors is exact).
    if affected:
        recomp = (surv_clones.filter(pl.col("vector").is_in(list(affected)))
                  .group_by("vector").agg(pl.col("n_reads").sum().alias("n_reads"),
                                          pl.col("umi").n_unique().alias("n_umis"),
                                          pl.col("n_reads").max().alias("max_reads_per_umi")))
        kept = counts.filter(~pl.col("vector").is_in(list(affected))).select(
            "vector", "n_reads", "n_umis", "max_reads_per_umi")
        surv_counts = pl.concat([kept, recomp.select("vector", "n_reads", "n_umis", "max_reads_per_umi")])
    else:
        surv_counts = counts.select("vector", "n_reads", "n_umis", "max_reads_per_umi")

    total = int(surv_counts["n_reads"].sum())
    surv_counts = (surv_counts
                   .with_columns((1e6 * pl.col("n_reads") / total).alias("rpm"),
                                 (pl.col("n_reads") < min_reads).alias("is_noise"))
                   .sort(["n_reads", "vector"], descending=[True, False]))
    surv_counts.write_parquet(out_dir / f"{sample}.fragvec_counts.parquet")

    # subvec passthrough if present (report does not read it, but keep the dir self-consistent)
    sub = counts_pq.parent / f"{sample}.subvec_counts.parquet"
    if sub.exists():
        pl.read_parquet(sub).write_parquet(out_dir / f"{sample}.subvec_counts.parquet")

    # summary: recompute the count fields the fragvec report reads; keep funnel/metadata fields.
    base = pl.read_csv(orig_summary, separator="\t").row(0, named=True) if orig_summary.exists() else {"sample": sample}
    nreads = surv_counts["n_reads"].to_list()
    npay = sum(nreads)
    ge = surv_counts.filter(pl.col("n_reads") >= min_reads)
    orig_payload = int(counts["n_reads"].sum())
    removed = orig_payload - npay
    base.update({
        "reads_payload": npay,
        "n_vectors": surv_counts.height,
        "mean_reads_per_vector": round(npay / surv_counts.height, 3) if surv_counts.height else 0,
        "max_reads_vector": max(nreads) if nreads else 0,
        "top_vector_read_share_pct": round(100 * max(nreads) / npay, 3) if npay else 0,
        "n_vectors_ge_min": ge.height,
        "pct_reads_in_vectors_ge_min": round(100 * int(ge["n_reads"].sum()) / npay, 1) if npay else 0,
    })
    if "reads_decoded" in base and isinstance(base["reads_decoded"], (int, float)):
        base["reads_decoded"] = int(base["reads_decoded"]) - removed
    base.update(_cplx(nreads))
    _write_tsv_row(out_dir / f"{sample}.fragvec_summary.tsv", base)


def run(cmd, env=None):
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, env=env)


def main():
    ap = argparse.ArgumentParser(description="Regenerate umi/fragvec cross-reports on jackpot-cleaned survivor data.")
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RES)
    ap.add_argument("--blacklist", type=Path, default=None,
                    help="default <out-dir>/<INITIAL>.umi_blacklist.tsv")
    ap.add_argument("--out-dir", type=Path, default=None, help="default <results-dir>/jackpot_clean")
    ap.add_argument("--sample-metadata", type=Path, default=None)
    ap.add_argument("--min-reads", type=int, default=10)
    ap.add_argument("--samples", default=None, help="comma list to restrict (debug); default all")
    ap.add_argument("--skip-reports", action="store_true", help="build survivor tables only, no HTML")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    rd = args.results_dir.resolve()
    out = (args.out_dir or (rd / "jackpot_clean")).resolve()
    meta_tsv = args.sample_metadata or (rd / "umifilt" / "sample_metadata.tsv")
    meta = load_meta(meta_tsv)
    initial = next((n for n, m in meta.items() if m.get("group") == BASELINE_GROUP), None)
    if initial is None:
        sys.exit(f"[FATAL] no '{BASELINE_GROUP}'-group sample in {meta_tsv}")
    bl_tsv = args.blacklist or (out / f"{initial}.umi_blacklist.tsv")
    bl = set(pl.read_csv(bl_tsv, separator="\t")["umi"].to_list())
    print(f"[clean] initial={initial}  blacklist={bl_tsv.name} ({len(bl)} UMIs)  out={out}")

    order = sorted(meta.items(), key=lambda kv: kv[1].get("idx", 0))
    names = [n for n, _ in order]
    if args.samples:
        want = set(args.samples.split(","))
        names = [n for n in names if n in want]

    removed_tbl = []
    for s in names:
        is_init = meta[s].get("group") == BASELINE_GROUP
        ustem = "umi_initial" if is_init else "umi"
        fstem = "fragvec_initial" if is_init else "fragvec"
        u_out = out / ustem / s
        f_out = out / fstem / s
        rem = clean_umi_sample(rd / ustem / s / f"{s}.fused.parquet",
                               rd / ustem / s / f"{s}.umi_summary.tsv", bl, u_out, s)
        clean_fragvec_sample(rd / fstem / s / f"{s}.fragvec_counts.parquet",
                             rd / fstem / s / f"{s}.fragvec_clones.parquet",
                             rd / fstem / s / f"{s}.fragvec_summary.tsv", bl, f_out, s, args.min_reads)
        removed_tbl.append((s, rem))
        print(f"[clean] {s:5s} ({meta[s].get('group')}): removed {rem:,} reads")

    if args.skip_reports:
        print("[clean] --skip-reports: survivor tables written; stopping before HTML.")
        return 0

    # ---- reuse the pipeline's own downstream scripts on the survivor tables -----------------
    def umi_species(s):
        st = "umi_initial" if meta[s].get("group") == BASELINE_GROUP else "umi"
        return out / st / s / f"{s}.umi_species.tsv"
    species_args = [f"{s}={umi_species(s)}" for s in names]
    (out / "umi").mkdir(parents=True, exist_ok=True)
    run([PY, SCRIPTS / "effective_complexity.py", "--species", *species_args,
         "--meta", meta_tsv, "--baseline", initial,
         "--out-tsv", out / "umi" / "umi_complexity.tsv",
         "--out-collapse", out / "umi" / "umi_collapse.tsv"])

    groups = {}
    for n, m in order:
        groups.setdefault(m.get("group"), []).append(n)
    for g, members in groups.items():
        if g == BASELINE_GROUP:
            continue
        smpls = [initial] + [x for x in members]
        def fv_counts(s):
            st = "fragvec_initial" if meta[s].get("group") == BASELINE_GROUP else "fragvec"
            return out / st / s / f"{s}.fragvec_counts.parquet"
        run([PY, SCRIPTS / "build_fragvec_matrix.py", "--group", g,
             "--samples", ",".join(smpls), "--counts", ",".join(str(fv_counts(s)) for s in smpls),
             "--out-prefix", out / "fragvec" / f"{g}.fragvec_x_sample"])

    env = os.environ.copy()
    env.update({
        "L2F_BASE": str(out / "umi"),
        "L2F_R0_STEM": str(out / "umi_initial" / initial / initial),
        "L2F_SAMPLES": str(meta_tsv),
        "L2F_DECODE_BASE": str(rd / "decode"),
        "L2F_R0_FRAGS": str(rd / "decode_initial" / initial / f"{initial}.fragments.parquet"),
        "L2F_COMPLEXITY": str(out / "umi" / "umi_complexity.tsv"),
        "L2F_COLLAPSE": str(out / "umi" / "umi_collapse.tsv"),
    })
    run([PY, SCRIPTS / "build_umi_report_cross.py"], env=env)

    env2 = os.environ.copy()
    env2.update({
        "L2F_BASE": str(out / "fragvec"),
        "L2F_R0_STEM": str(out / "fragvec_initial" / initial / initial),
        "L2F_SAMPLES": str(meta_tsv),
        "L2F_MIN_READS": str(args.min_reads),
    })
    run([PY, SCRIPTS / "build_fragvec_report_cross.py"], env=env2)

    print(f"[clean] DONE. reports:\n  {out}/umi/umi_report_cross.html\n  {out}/fragvec/fragvec_report_cross.html")
    print("[clean] reads removed per sample:")
    for s, r in removed_tbl:
        print(f"    {s:5s} {r:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
