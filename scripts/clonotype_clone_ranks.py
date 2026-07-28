#!/usr/bin/env python
"""Rank the clones (fused UMIs) inside each clonotype (fragment vector).

STANDALONE per-study analysis -- NOT part of the Snakemake DAG. It reads the
already-produced per-sample clone tables under ``results/fragvec/`` and writes new
tables + one HTML report under ``results/clonerank/``. Nothing here mutates the
pipeline, its config, or any existing ``results/`` tree.

The two levels, and why they are called this
--------------------------------------------
A **clonotype** is one fragment vector -- one ligation architecture, positional
5'->3' with anchors included, e.g.::

    UMI1_anchor|Linear14ntStrand2|Linear13ntStrand2|UMI2_anchor

A **clone** is one fused UMI carrying that architecture: an independent physical
molecule of it. Family = clonotype, member = clone. Those are already the words
``fragvec_lib.py`` uses ("reads per clone per clonotype", fragvec_lib.py:78), and
"family" is NOT reused here because ``detect_umi_jackpots.py:313`` already spends
it on the other level -- a "UMI family" there is reads on ONE umi, i.e. a clone.

What it computes
----------------
Three nested tiers off one substrate. Tier 1 is the plain ranked table; tiers 2 and
3 add to it and never replace it.

1. RANK -- every clone of every clonotype, ordered by reads, with its share.
2. DOMINANCE -- does that top rank mean anything? Per clonotype with N reads over
   k observed clones and `top` reads on the biggest one::

       null:  the N reads fall multinomially over the k clones, equally likely
       p_dominant = min(1, k * P(Binomial(N, 1/k) >= top))     # union bound over k

   The null is CONSERVATIVE by construction. The true clone pool R is >= the
   observed k (we are undersampling -- clarification.md section 4), and a smaller
   pool concentrates reads MORE, so using k makes the null harder to reject than
   reality warrants. Anything flagged is flagged despite the handicap. It also
   needs no estimate of R, which is exactly the unknown the depth problem denies us.

   This matters because the rank ALONE is a coin flip in an unselected sample.
   Measured on disk over clonotypes with >=100 reads: A549_R1 and empty_R1 tops sit
   at 2-3 reads out of ~150 (noise), while A549_R5 throws 33/159 and 30/130 (real).
   Same table shape, opposite meaning.

3. TRAJECTORY -- the same (vector, umi) pair across rounds, so a lineage taking
   over its clonotype under selection is visible as a slope, not a snapshot.

Censoring -- the trap this file is careful about
------------------------------------------------
``<s>.fragvec_clones.parquet`` holds only clonotypes at or above ``min_reads`` IN
THAT SAMPLE (build_fragvec.py:155). A clonotype below the floor in R1 but above it
in R5 is simply ABSENT from R1, not zero. The trajectory therefore distinguishes:

    clonotype absent from that sample's clone table  ->  CENSORED, written "."
    clonotype present, this clone absent             ->  a genuine 0

Writing 0 for a censored cell would manufacture the exact "clone rose from nothing"
story this analysis exists to test. If ``<s>.fragvec_reads.parquet`` (uncensored,
fragvec.smk:58) is present it is preferred automatically and censoring drops to 0.

Fitness restriction (ON by default)
-----------------------------------
By default the test is scoped to the FITNESS-TOP clonotypes: those enriched under
selection per ``results/fitness/<group>_vs_<control>.log2_step_ratios.tsv`` (summed
empty-corrected log2 fold-change >= ``--min-log2-sum``, via ``fitness_lib``). A clone
can dominate its architecture in one round without that architecture being selected;
this keeps only the enriched architectures. Because ``p_dominant`` is universe-independent
(it depends only on a clonotype's own read counts), this is a pure FILTER -- the survivors
keep their exact p-values. Groups with no ``<group>_vs_<control>`` table (``empty``,
``initial``, ``*-SN``) drop out, so the ``empty`` negative control is not in the report;
pass ``--fitness-dir ''`` for the full-calibration view over every clonotype.

What ``log2_sum`` actually is, and how the threshold is derived from it
----------------------------------------------------------------------
The "sum over the round steps" is NOT a trajectory statistic. ``fitness_step_ratios.py``
adds ``EPS`` on the round-level rpm scale, so every intermediate ``(rpm_r + EPS)`` appears
once as a numerator and once as a denominator and cancels; the shared R0 baseline has the
same depth in every group's long table (T = 515392), so its term cancels too. The sum
therefore telescopes EXACTLY to the endpoint ratio::

    log2_sum  ==  log2( (subject_last_rpm + EPS) / (control_last_rpm + EPS) )

Verified on disk over all 3859 A549 vectors: max |difference| 3.4e-4, which is only the
4dp rounding of the five step columns, and both criteria select the identical 1146 vectors.
The intermediate rounds contribute NOTHING to the ranking -- a vector that climbed
monotonically and one that was flat until a single last-round spike score the same. Do not
read a high ``log2_sum`` as evidence of sustained enrichment.

That makes the threshold a statement about ENDPOINT read support, and one that can be
computed rather than typed. Since ``control_last_rpm >= 0``, the identity above bounds every
vector by its own numerator::

    log2_sum  <=  log2( subject_last_rpm + EPS )

so a vector carrying a single read at the endpoint scores at most ``log2(1e6/T_last/EPS + 1)``
for that sample's depth ``T_last``. Sitting the bar just above that number is therefore an
EXACT sufficient condition for ">= 2 endpoint reads", not a heuristic. That is what
``--min-log2-sum auto`` (the default) does, per group, via ``fitness_lib.auto_min_log2_sum``:

    A549   endpoint A549_R5, T = 62611   -> 1-read floor 4.0851 -> threshold 4.0951 -> 306 of 3859
    MDA    endpoint MDA_R5,  T = 104641  -> 1-read floor 3.4001 -> threshold 3.4101 -> 1477 of 4972

The margin over the floor is ``fitness_lib.AUTO_MARGIN`` (0.01), sized to clear the 4dp step
rounding above while staying far below the ~0.95 gap to the 2-read score. Confirmed on disk:
the survivors' minimum endpoint read count is exactly 2 in both arms. The converse does not
hold and is not meant to -- many >=2-read vectors fall below the bar because the control grew
too; this is a magnitude gate whose FLOOR is pinned to the depth.

Because ``T_last`` is per group, so is the threshold: the two arms are uniform in MEANING
("not explainable by a single endpoint read") but NOT in magnitude, so their clonotype counts
are not comparable with each other. The endpoint round is read off the LAST ``log2_R*`` column
of the fitness table, not assumed to be the group's max round -- when the control stops early,
only the shared steps are subtracted and the sum telescopes to that earlier round.

Passing a float instead pins one bar across every group, which is what the old hard-coded
``4.1`` did: it is A549's floor, i.e. the shallowest arm's, applied to MDA as well.

CLI (all defaulted, so it runs with no args from the repo root)
---------------------------------------------------------------
    pixi run -e default python scripts/clonotype_clone_ranks.py \
      --fragvec-dir         results/fragvec \
      --initial-dir         results/fragvec_initial \
      --meta                results/umi/sample_metadata.tsv \
      --out-dir             results/clonerank \
      --min-reads           10 \
      --alpha               0.05 \
      --top-n               0 \
      --traj-top            3 \
      --fitness-dir         results/fitness \
      --fitness-control     empty \
      --min-log2-sum        auto \
      --eps                 1.0 \
      --auto-min-last-reads 2

``--eps`` must match the ``--eps`` ``fitness_step_ratios.py`` ran with (both default 1.0):
the auto threshold is derived from it, and it is not recorded in the fitness tables.

``--alpha`` is a PER-CLONOTYPE threshold and is deliberately NOT Bonferroni-corrected
across the thousands of clonotypes in a sample. Two reasons it is still readable: the
union bound inside ``p_dominant`` already spends a factor of k per clonotype, and the
flat null uses the OBSERVED clone count (see above), which over-concentrates the null.
Both push the same way, so the realised false-positive rate runs below nominal. Read the
per-sample dominant counts as the calibration: the ``empty`` arm and R1 are the samples
where the true answer is "none", so whatever they report is the working noise floor.

Emits (under --out-dir):
    <sample>.clone_ranks.parquet          tier 1: every clone, ranked
    <sample>.clonotype_dominance.tsv      tier 2: one row per clonotype + the test; in fitness mode
                                          also log2_sum / fit_rank and the trajectory-shape columns
                                          n_steps / n_pos_steps / min_step / is_monotone (reported,
                                          never filtered on -- only log2_sum gates)
    <group>.clone_trajectories.tsv        tier 3: (vector, umi) across rounds, "." = censored
    <group>.clone_trajectories.parquet    the same, typed nulls -- prefer this for re-loading
    clonerank_report.html                 the four panels

Reading the trajectory TSV back requires declaring the censor marker::

    pl.read_csv(p, separator="\\t", null_values=["."], infer_schema_length=20000)

or the reads_R* columns fail to parse as ints. The parquet needs no such declaration.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import polars as pl

# This script is standalone (scripts/), but two of its libs are DAG-owned and live in
# workflow/scripts/ -- so the path spans BOTH dirs: siblings first, pipeline libs second.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))                                  # standalone siblings: fitness_lib
sys.path.insert(1, str(_HERE.parent / "workflow" / "scripts"))  # DAG libs: complexity_lib, report_lib
from complexity_lib import c_eff                      # noqa: E402  depth-robust concentration
from fitness_lib import (                             # noqa: E402  shared "fitness-top" definition
    auto_min_log2_sum, load_fitness_top,
)
from report_lib import (                              # noqa: E402  STDLIB-ONLY by invariant
    BASELINE_GROUP, GREY, color_key_html, esc, fig_html, first_of_group, group_color, load_meta,
)

_ROUND_RE = re.compile(r"_R(\d+)$")

# One hue per ranked clone in the trajectory panel. A different axis from the group palette --
# inside a trajectory the thing being told apart is the clone, not the sample. Same reasoning as
# VECTOR_PALETTE in build_fragvec_report_cross.py; do not "fix" it to group_color().
CLONE_PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
                 "#EECA3B", "#B279A2", "#FF9DA6"]


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# the dominance test
# --------------------------------------------------------------------------- #
def binom_sf_ge(n: int, p: float, m: int) -> float:
    """P(Binomial(n, p) >= m), exact, summed in log space.

    Log space via lgamma rather than math.comb: n runs to ~67k here, and the direct
    ratio C(n,i) p^i (1-p)^(n-i) overflows the binomial coefficient long before the
    product underflows. Summation starts at m and walks UP; above the mode the terms
    decay geometrically, so it converges in a handful of iterations for exactly the
    tail we care about. The `i > mode` guard stops the loop breaking early on the way
    up to the mode, where terms are still rising.
    """
    if m <= 0:
        return 1.0
    if m > n:
        return 0.0
    if p >= 1.0:
        return 1.0
    if p <= 0.0:
        return 0.0
    log_p, log_q = math.log(p), math.log1p(-p)
    lg_n = math.lgamma(n + 1)
    mode = n * p
    total = 0.0
    for i in range(m, n + 1):
        term = math.exp(lg_n - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                        + i * log_p + (n - i) * log_q)
        total += term
        if i > mode and term < 1e-18 * max(total, 1e-300):
            break
    return min(1.0, total)


def p_dominant(n_reads: int, n_clones: int, top: int) -> float:
    """Union-bounded p-value that ANY clone reaches `top` reads under the flat null.

    Returns 1.0 for the two uninformative cases: a single observed clone (the null
    then says all reads land on it, so top == N is certain) and a top at or below the
    null mean (nothing to explain).
    """
    if n_clones <= 1:
        return 1.0
    if top <= n_reads / n_clones:
        return 1.0
    return min(1.0, n_clones * binom_sf_ge(n_reads, 1.0 / n_clones, top))


# --------------------------------------------------------------------------- #
# sample discovery / round labelling
# --------------------------------------------------------------------------- #
def sample_round(name: str, meta: dict) -> int:
    """Round index: metadata first, else a trailing ``_R<n>``, else 0 (the baseline)."""
    if name in meta:
        return int(meta[name].get("round", 0) or 0)
    m = _ROUND_RE.search(name)
    return int(m.group(1)) if m else 0


def discover(fragvec_dir: Path, initial_dir: Path, meta: dict) -> list[dict]:
    """Every sample carrying a clone table, baseline first. Mirrors build_fragvec_report_cross.discover."""
    out: list[dict] = []
    for base, is_baseline in ((fragvec_dir, False), (initial_dir, True)):
        if not base.is_dir():
            continue
        for d in sorted(p for p in base.glob("*/") if p.is_dir()):
            n = d.name
            clones = d / f"{n}.fragvec_clones.parquet"
            if not clones.exists():
                continue
            mm = meta.get(n, {})
            out.append({
                "name": n,
                "clones": clones,
                "reads": d / f"{n}.fragvec_reads.parquet",   # uncensored, if it was ever built
                "counts": d / f"{n}.fragvec_counts.parquet",
                "group": mm.get("group", BASELINE_GROUP if is_baseline else n),
                "round": sample_round(n, meta),
                "idx": -1 if is_baseline else int(mm.get("idx", 9999) or 9999),
                "is_baseline": is_baseline,
            })
    out.sort(key=lambda s: (s["idx"], s["round"], s["name"]))
    return out


# --------------------------------------------------------------------------- #
# tier 1: rank every clone
# --------------------------------------------------------------------------- #
def clone_ranks(s: dict, top_n: int) -> tuple[pl.DataFrame, bool]:
    """(ranked clones, used_uncensored). Columns:

        sample, group, round, vector, rank, umi, n_reads, share,
        clonotype_n_reads, clonotype_n_umis
    """
    used_uncensored = False
    if s["reads"].exists():
        # the per-read map covers the WHOLE payload-bearing set, not just signal clonotypes
        rv = pl.read_parquet(s["reads"], columns=["vector", "umi"])
        clones = rv.group_by(["vector", "umi"]).agg(pl.len().cast(pl.UInt32).alias("n_reads"))
        used_uncensored = True
    else:
        clones = pl.read_parquet(s["clones"], columns=["vector", "umi", "n_reads"])

    ranked = (
        clones
        .sort(["vector", "n_reads", "umi"], descending=[False, True, False])
        .with_columns(
            pl.col("umi").cum_count().over("vector").cast(pl.UInt32).alias("rank"),
            pl.col("n_reads").sum().over("vector").cast(pl.UInt32).alias("clonotype_n_reads"),
            pl.len().over("vector").cast(pl.UInt32).alias("clonotype_n_umis"),
        )
        .with_columns(
            (pl.col("n_reads") / pl.col("clonotype_n_reads")).alias("share"),
            pl.lit(s["name"]).alias("sample"),
            pl.lit(s["group"]).alias("group"),
            pl.lit(s["round"]).cast(pl.Int32).alias("round"),
        )
    )
    if top_n > 0:
        ranked = ranked.filter(pl.col("rank") <= top_n)
    return ranked.select("sample", "group", "round", "vector", "rank", "umi", "n_reads",
                         "share", "clonotype_n_reads", "clonotype_n_umis"), used_uncensored


# --------------------------------------------------------------------------- #
# tier 2: per-clonotype dominance
# --------------------------------------------------------------------------- #
def dominance_table(s: dict, ranked: pl.DataFrame, alpha: float, min_reads: int) -> pl.DataFrame:
    """One row per clonotype: the concentration of reads over its clones, and the test.

    `dominance` is the EXISTING definition (detect_umi_jackpots.py:28) -- max_reads_per_umi
    / n_reads -- reused verbatim so the jackpot report and this one cannot disagree.
    """
    agg = (ranked.group_by("vector")
           .agg(pl.col("n_reads").sort(descending=True).alias("cnts"),
                pl.col("umi").sort_by("n_reads", descending=True).first().alias("top_umi"))
           .sort("vector"))

    rows = []
    for vector, cnts, top_umi in agg.iter_rows():
        # complexity_lib owns the estimator AND its zero-collision lower bound -- the common
        # case here, since most clonotypes are all-singleton (clarification.md section 4).
        N, k, dup_pairs, ce, is_bound = c_eff(cnts)
        top = int(cnts[0])
        rows.append({
            "sample": s["name"], "group": s["group"], "round": s["round"], "vector": vector,
            "n_reads": int(N), "n_umis": int(k), "max_reads_per_umi": top,
            "dominance": top / N if N else 0.0, "top_umi": top_umi,
            "clone_dup_pairs": int(dup_pairs),
            "c_eff_clone": ce, "c_eff_is_lower_bound": is_bound,
            "n_over_c_eff_clone": (N / ce) if ce else float("inf"),
            "p_dominant": p_dominant(int(N), int(k), top),
        })
    if not rows:
        return pl.DataFrame(schema={"sample": pl.String, "group": pl.String, "round": pl.Int32,
                                    "vector": pl.String, "n_reads": pl.UInt32, "n_umis": pl.UInt32,
                                    "max_reads_per_umi": pl.UInt32, "dominance": pl.Float64,
                                    "top_umi": pl.String, "clone_dup_pairs": pl.Int64,
                                    "c_eff_clone": pl.Float64, "c_eff_is_lower_bound": pl.Boolean,
                                    "n_over_c_eff_clone": pl.Float64, "p_dominant": pl.Float64,
                                    "is_dominant": pl.Boolean})
    return (pl.DataFrame(rows)
            .with_columns(((pl.col("p_dominant") < alpha)
                           & (pl.col("n_reads") >= min_reads)).alias("is_dominant"))
            .sort(["p_dominant", "dominance"], descending=[False, True]))


def crosscheck(s: dict, dom: pl.DataFrame) -> None:
    """The clone rows must reproduce fragvec_counts exactly. A mismatch is a join bug, not data."""
    if not s["counts"].exists():
        log(f"[WARN] {s['name']}: no fragvec_counts.parquet; skipping invariant check")
        return
    ref = pl.read_parquet(s["counts"], columns=["vector", "n_reads", "n_umis", "max_reads_per_umi"])
    j = dom.join(ref, on="vector", how="inner", suffix="_ref")
    bad = j.filter((pl.col("n_reads") != pl.col("n_reads_ref"))
                   | (pl.col("n_umis") != pl.col("n_umis_ref"))
                   | (pl.col("max_reads_per_umi") != pl.col("max_reads_per_umi_ref")))
    if bad.height:
        log(f"[WARN] {s['name']}: {bad.height:,}/{j.height:,} clonotypes disagree with "
            f"fragvec_counts (n_reads / n_umis / max_reads_per_umi)")
    else:
        log(f"[INFO] {s['name']}: {j.height:,} clonotypes match fragvec_counts exactly")


# --------------------------------------------------------------------------- #
# tier 3: cross-round clone trajectories
# --------------------------------------------------------------------------- #
def trajectories(group: str, gsamples: list[dict], ranks: dict, doms: dict,
                 traj_top: int) -> tuple[pl.DataFrame, int]:
    """(wide table, censored_cells) for every (vector, umi) that ranks high anywhere in the group.

    A cell is CENSORED (null -> "." on write) when the clonotype is absent from that
    sample's clone table -- below min_reads there -- which is NOT the same as the clone
    being absent from a clonotype that was measured (a genuine 0).
    """
    seeds = (pl.concat([ranks[s["name"]].filter(pl.col("rank") <= traj_top)
                        .select("vector", "umi") for s in gsamples])
             .unique())
    if not seeds.height:
        return pl.DataFrame(), 0

    out = seeds.sort(["vector", "umi"])
    censored_cells = 0
    for s in gsamples:
        r = s["round"]
        seen = doms[s["name"]].select("vector").unique()          # clonotypes measured at all
        part = (ranks[s["name"]].select("vector", "umi", "n_reads", "rank", "share")
                .rename({"n_reads": f"reads_R{r}", "rank": f"rank_R{r}", "share": f"share_R{r}"}))
        out = (out.join(part, on=["vector", "umi"], how="left")
               .join(seen.with_columns(pl.lit(True).alias(f"__seen_R{r}")),
                     on="vector", how="left"))
        # clonotype measured -> a missing clone is a real 0; clonotype unmeasured -> censored
        out = out.with_columns(
            pl.when(pl.col(f"__seen_R{r}").is_null()).then(None)
              .otherwise(pl.col(f"reads_R{r}").fill_null(0)).alias(f"reads_R{r}"),
            pl.when(pl.col(f"__seen_R{r}").is_null()).then(None)
              .otherwise(pl.col(f"share_R{r}").fill_null(0.0)).alias(f"share_R{r}"),
            pl.col(f"__seen_R{r}").is_null().alias(f"censored_R{r}"),
        ).drop(f"__seen_R{r}")
        censored_cells += int(out[f"censored_R{r}"].sum())

    rounds = [s["round"] for s in gsamples]
    cols = ["vector", "umi"]
    for r in rounds:
        cols += [f"reads_R{r}", f"rank_R{r}", f"share_R{r}", f"censored_R{r}"]
    last = f"reads_R{rounds[-1]}"
    return out.select(cols).sort(last, descending=True, nulls_last=True), censored_cells


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def style(fig, height, title=None, top=None):
    fig.update_layout(
        height=height, template="plotly_white",
        title=(dict(text=title, y=1.0, yanchor="top", yref="container", pad=dict(t=6))
               if title else None),
        margin=dict(l=60, r=20, t=(top or (64 if title else 20)), b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), font=dict(size=12))
    return fig


def summary_table(samples, doms, gcol, alpha):
    rows = []
    for s in samples:
        d = doms[s["name"]]
        n_dom = int(d["is_dominant"].sum()) if d.height else 0
        med_dom = float(d["dominance"].median()) if d.height else 0.0
        med_k = float(d["n_umis"].median()) if d.height else 0.0
        swatch = (f"<span style='display:inline-block;width:.7rem;height:.7rem;border-radius:2px;"
                  f"margin-right:.4rem;background:{gcol.get(s['group'], GREY)}'></span>")
        rows.append(
            f"<tr><td>{esc(s['name'])}</td><td>{swatch}{esc(s['group'])}</td>"
            f"<td class='num'>{d.height:,}</td><td class='num'>{n_dom:,}</td>"
            f"<td class='num'>{med_dom:.4f}</td><td class='num'>{med_k:,.0f}</td></tr>")
    return ("<h2>Clonotypes and their clones</h2>"
            f"<p class='small'>A clonotype is <b>dominant</b> when p &lt; {alpha} under the flat null "
            "(union-bounded over its observed clones) — i.e. its top clone carries more reads than "
            "undersampling alone explains. <code>dominance</code> is "
            "<code>max_reads_per_umi / n_reads</code>, the same definition the jackpot report uses.</p>"
            "<table><thead><tr><th>sample</th><th>group</th>"
            "<th title='DEPTH-DEPENDENT — do not compare across samples'>clonotypes ⚠</th>"
            "<th>dominant</th><th>median dominance</th><th>median clones/clonotype</th>"
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def fig_dominance(samples, doms, gcol):
    import plotly.graph_objects as go
    fig = go.Figure()
    show = first_of_group(samples)
    for s, first in zip(samples, show):
        d = doms[s["name"]]
        if not d.height:
            continue
        fig.add_trace(go.Scattergl(
            x=d["n_reads"].to_list(), y=d["dominance"].to_list(), mode="markers",
            name=s["group"], legendgroup=s["group"], showlegend=bool(first),
            marker=dict(size=5, opacity=.55, color=gcol.get(s["group"], GREY)),
            text=[s["name"]] * d.height,
            hovertemplate="%{text}<br>reads %{x}<br>dominance %{y:.3f}<extra></extra>"))
        hit = d.filter(pl.col("is_dominant"))
        if hit.height:
            fig.add_trace(go.Scattergl(
                x=hit["n_reads"].to_list(), y=hit["dominance"].to_list(), mode="markers",
                name=f"{s['group']} dominant", legendgroup=s["group"], showlegend=False,
                marker=dict(size=10, color="rgba(0,0,0,0)", line=dict(width=1.6, color="#d1242f")),
                text=[s["name"]] * hit.height,
                hovertemplate="%{text} DOMINANT<br>reads %{x}<br>dominance %{y:.3f}<extra></extra>"))
    fig.update_xaxes(type="log", title="clonotype n_reads")
    fig.update_yaxes(type="log", title="dominance  (top clone's share of the clonotype)")
    return fig_html(style(
        fig, 470,
        "Dominance vs clonotype depth — ringed points beat the flat null; the low band is undersampling"))


def dominant_table(doms, limit=40):
    allrows = pl.concat([d for d in doms.values() if d.height]) if doms else pl.DataFrame()
    if not allrows.height:
        return "<p class='small'>No clonotypes.</p>"
    # Ranked by p, NOT by dominance. Sorting on dominance puts a 10-of-12-reads clonotype (3 clones)
    # above a 33-of-159 one (118 clones) -- but the second is the far stronger result, and the
    # first is nearly what 3 clones give you by chance. p already carries the depth.
    hit = allrows.filter(pl.col("is_dominant")).sort("p_dominant").head(limit)
    if not hit.height:
        return ("<h2>Dominant clonotypes</h2><p class='small'>None — in every sample the top clone is "
                "within what undersampling alone explains. That is the expected result for an "
                "un-bottlenecked library.</p>")
    rows = "".join(
        f"<tr><td style='font-family:monospace;font-size:.72rem'>{esc(r['vector'][:58])}"
        f"{'…' if len(r['vector']) > 58 else ''}</td><td>{esc(r['sample'])}</td>"
        f"<td class='mono'>{esc(r['top_umi'])}</td>"
        f"<td class='num'>{r['max_reads_per_umi']:,}/{r['n_reads']:,}</td>"
        f"<td class='num'>{r['n_umis']:,}</td><td class='num'>{r['dominance']:.3f}</td>"
        f"<td class='num'>{r['p_dominant']:.2e}</td></tr>"
        for r in hit.iter_rows(named=True))
    return ("<h2>Dominant clonotypes</h2>"
            f"<p class='small'>Top {hit.height:,} by <b>p</b>, not by dominance — p carries the depth, "
            "so a shallow clonotype cannot outrank a deep one on a high share alone. These are the "
            "clonotypes where asking &ldquo;which fused UMI is most abundant?&rdquo; has a defensible "
            "answer.</p>"
            "<div style='overflow-x:auto'><table><thead><tr><th>clonotype (vector)</th><th>sample</th>"
            "<th>top clone (fused UMI)</th><th>top / N</th><th>clones</th><th>dominance</th>"
            "<th>p</th></tr></thead><tbody>" + rows + "</tbody></table></div>")


def fig_trajectories(group, traj, gsamples, top_k=8):
    import plotly.graph_objects as go
    if not traj.height:
        return ""
    rounds = [s["round"] for s in gsamples]
    last = f"share_R{rounds[-1]}"
    top = traj.sort(last, descending=True, nulls_last=True).head(top_k)
    fig = go.Figure()
    for i, r in enumerate(top.iter_rows(named=True)):
        ys = [r[f"share_R{rd}"] for rd in rounds]      # None where censored -> a GAP, never a 0
        fig.add_trace(go.Scatter(
            x=rounds, y=ys, mode="lines+markers", connectgaps=False,
            name=f"{r['umi'][:12]}… ({r['vector'][:22]}…)",
            line=dict(color=CLONE_PALETTE[i % len(CLONE_PALETTE)], width=2),
            marker=dict(size=7)))
    fig.update_xaxes(title="round", tickmode="array", tickvals=rounds)
    fig.update_yaxes(title="clone's share of its clonotype")
    return fig_html(style(fig, 460, f"{group} — clone share within its clonotype across rounds",
                          top=110))


_CAVEATS = """<h2>How to read this — and how not to</h2>
<div class='callout'>
<p><b>A top-ranked clone is not automatically a real one.</b> Almost every UMI in this data is a
singleton, which is the signature of undersampling, not of a broken experiment
(<code>clarification.md</code> §4). So <em>every</em> clonotype has a &ldquo;most abundant&rdquo; clone
even when its reads are spread perfectly evenly. Read the rank together with
<code>p_dominant</code>; on its own the rank is a coin flip.</p>
<p><b>Dominance in a selected round is signal; in R0 it was already removed.</b> These tables are built
from the <code>umi/</code> tree, i.e. <em>after</em> the jackpot blacklist. The R0 library's amplified
lineages are gone. Do not re-filter high-dominance clonotypes in a selected round — there, a clone
taking over its clonotype is the thing you are looking for.</p>
<p><b>Censored is not zero.</b> A <code>.</code> in a trajectory means the clonotype fell below
<code>min_reads</code> in that sample and was never measured. It does not mean the clone was absent.</p>
<p><b>Clonotype and clone counts grow with depth.</b> Columns marked ⚠ are properties of the run, not
of the library, and must not be compared across samples of different depth.</p>
</div>"""


def build_report(out_path, samples, doms, trajs, gcol, alpha, note=None):
    from plotly.offline import get_plotlyjs
    # only the report's samples (fitness mode keeps the baseline in `doms` for trajectory R0, but its
    # rows lack the log2_sum/fit_rank/shape columns the restricted rows carry -- concat and it clashes)
    rep_doms = {s["name"]: doms[s["name"]] for s in samples}
    H = ["<!doctype html><html><head><meta charset='utf-8'>",
         "<meta name='viewport' content='width=device-width, initial-scale=1'>",
         "<title>Clone ranking within clonotypes</title>", _STYLE,
         f"<script>{get_plotlyjs()}</script></head><body>",
         "<h1>Clone ranking within clonotypes</h1>",
         "<p class='small'>A <b>clonotype</b> is one fragment vector — one ligation architecture. Its "
         "<b>clones</b> are the independent molecules (fused UMIs) carrying it. This report ranks the "
         "clones inside each clonotype and asks whether the top rank means anything. Standalone "
         "per-study analysis, not part of the workflow DAG.</p>",
         (f"<div class='callout'><p>{note}</p></div>" if note else ""),
         color_key_html(gcol, samples),
         summary_table(samples, rep_doms, gcol, alpha),
         "<h2>Is the top clone real?</h2>",
         fig_dominance(samples, rep_doms, gcol),
         dominant_table(rep_doms),
         "<h2>Clone trajectories</h2>",
         "<p class='small'>Only clones that rank high somewhere in the group. Gaps are censored "
         "rounds — the clonotype was below <code>min_reads</code> there, which is not a zero.</p>"]
    for group, (traj, gsamples) in trajs.items():
        H.append(fig_trajectories(group, traj, gsamples))
    H.append(_CAVEATS)
    H.append("</body></html>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(x for x in H if x))
    log(f"[INFO] wrote {out_path}")


# --------------------------------------------------------------------------- #
# io helpers
# --------------------------------------------------------------------------- #
def write_tsv(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path, separator="\t", null_value=".")     # "." marks a CENSORED cell
    log(f"[INFO] wrote {path}  ({df.height:,} rows x {df.width} cols)")


def write_parquet(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    log(f"[INFO] wrote {path}  ({df.height:,} rows x {df.width} cols)")


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #
def selftest() -> int:
    # the log-space tail must equal a brute-force exact sum
    for n, p, m in [(10, 0.5, 7), (25, 0.1, 6), (40, 1 / 8, 12), (60, 1 / 20, 5), (12, 0.25, 0)]:
        brute = sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(max(m, 0), n + 1))
        got = binom_sf_ge(n, p, m)
        assert abs(brute - got) < 1e-12, f"tail {n},{p},{m}: {got} != {brute}"

    assert binom_sf_ge(10, 0.5, 11) == 0.0
    assert binom_sf_ge(10, 0.5, 0) == 1.0

    # uninformative cases collapse to p = 1
    assert p_dominant(100, 1, 100) == 1.0
    assert p_dominant(100, 50, 2) == 1.0                     # top == the null mean

    # the measured contrast: A549_R1-like noise must NOT be called; A549_R5-like must be
    noise = p_dominant(150, 148, 3)
    real = p_dominant(159, 118, 33)
    assert noise > 0.05, f"noise case called dominant (p={noise})"
    assert real < 1e-6, f"real case not called dominant (p={real})"

    # The union bound must be >= the TRUE P(max >= top) under the flat multinomial. Verified by
    # EXACT enumeration, not simulation: Monte Carlo error is the same size as the gap being
    # tested, so a sampled check cannot tell a correct bound from a slightly violated one.
    def exact_p_max_ge(n: int, k: int, top: int) -> float:
        """P(max cell >= top) for n balls in k equally-likely cells, by exact enumeration."""
        total = 0.0
        lg_n = math.lgamma(n + 1)

        def walk(cell: int, left: int, counts: list[int]):
            nonlocal total
            if cell == k - 1:
                c = counts + [left]
                if max(c) >= top:
                    logp = lg_n - sum(math.lgamma(x + 1) for x in c) - n * math.log(k)
                    total += math.exp(logp)
                return
            for x in range(left + 1):
                walk(cell + 1, left - x, counts + [x])

        walk(0, n, [])
        return total

    for n, k, top in [(8, 3, 5), (10, 4, 4), (12, 3, 7), (9, 5, 3)]:
        exact = exact_p_max_ge(n, k, top)
        bound = min(1.0, k * binom_sf_ge(n, 1.0 / k, top))
        assert bound >= exact - 1e-12, f"union bound {bound} < exact {exact} for n={n},k={k},top={top}"

    # It must be a real bound, not an equality that happens to pass. Note the bound is EXACT
    # whenever 2*top > n -- two cells cannot both reach top, so the events are disjoint and the
    # union bound is the union. Strict looseness needs an overlapping case, i.e. top <= n/2.
    # n=12,k=6,top=4: two cells CAN both reach 4 (8 <= 12), and pigeonhole does not force it
    # (6*3 = 18 >= 12), so the bound must sit strictly above the exact probability.
    loose_bound = min(1.0, 6 * binom_sf_ge(12, 1 / 6, 4))
    assert loose_bound > exact_p_max_ge(12, 6, 4) + 1e-9, "union bound unexpectedly tight"
    tight_bound = min(1.0, 3 * binom_sf_ge(12, 1 / 3, 7))
    assert abs(tight_bound - exact_p_max_ge(12, 3, 7)) < 1e-12, "disjoint case should be exact"

    log(f"[INFO] selftest OK  (noise p={noise:.3g}, real p={real:.3g}; "
        f"union bound verified >= exact on 4 enumerated cases)")
    return 0


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fragvec-dir", default="results/fragvec", type=Path)
    ap.add_argument("--initial-dir", default="results/fragvec_initial", type=Path)
    ap.add_argument("--meta", default="results/umi/sample_metadata.tsv", type=Path)
    ap.add_argument("--out-dir", default="results/clonerank", type=Path)
    ap.add_argument("--min-reads", type=int, default=10,
                    help="clonotype floor for the dominance call (matches fragvec.min_reads)")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="p_dominant threshold for is_dominant (per clonotype, NOT corrected "
                         "across the thousands of clonotypes in a sample -- see --help note)")
    ap.add_argument("--top-n", type=int, default=0,
                    help="keep only the top N clones per clonotype (0 = keep all)")
    ap.add_argument("--traj-top", type=int, default=3,
                    help="a clone seeds a trajectory if it ranks <= this in any sample of the group")
    ap.add_argument("--fitness-dir", default="results/fitness",
                    help="restrict the test to fitness-top clonotypes from this dir's "
                         "<group>_vs_<control>.log2_step_ratios.tsv; pass '' for full calibration mode "
                         "(kept as a str, not Path, so '' disables cleanly -- Path('') is '.')")
    ap.add_argument("--fitness-control", default="empty",
                    help="the <group>_vs_<control> suffix identifying the fitness tables")
    ap.add_argument("--min-log2-sum", default="auto",
                    help="keep clonotypes with summed empty-corrected log2 fold-change >= this "
                         "(fitness enrichment threshold; only used when --fitness-dir is set). "
                         "'auto' (the default) derives it PER GROUP from that group's own endpoint "
                         "depth, so it self-corrects when sequencing depth changes; pass a float "
                         "to pin one bar across every group instead")
    ap.add_argument("--eps", type=float, default=1.0,
                    help="the pseudocount fitness_step_ratios.py was run with (its --eps). The "
                         "auto threshold is derived from it, so a mismatch silently moves the bar")
    ap.add_argument("--auto-min-last-reads", type=int, default=2,
                    help="what 'auto' buys: the endpoint read support a clonotype must have to "
                         "clear the derived threshold (2 = exclude single-read-at-endpoint vectors)")
    ap.add_argument("--no-report", action="store_true", help="skip the HTML")
    ap.add_argument("--selftest", action="store_true", help="run the estimator checks and exit")
    args = ap.parse_args(argv[1:])

    if args.selftest:
        return selftest()

    meta = load_meta(str(args.meta)) if args.meta.exists() else {}
    if not meta:
        log(f"[WARN] metadata not found or empty ({args.meta}); falling back to name parsing")

    samples = discover(args.fragvec_dir, args.initial_dir, meta)
    if not samples:
        raise SystemExit(f"[FATAL] no clone tables found under {args.fragvec_dir} / {args.initial_dir}")
    log(f"[INFO] {len(samples)} samples: {', '.join(s['name'] for s in samples)}")

    # --- fitness restriction: which clonotypes are even eligible to be tested/reported ---
    # ON by default. The dominance p is universe-independent, so this is a FILTER (same p-values for
    # the survivors), applied per group from its <group>_vs_<control> fitness table. Groups with no
    # such table (empty, initial, *-SN) drop out. `--fitness-dir ''` restores full calibration mode.
    fitness_on = bool(args.fitness_dir.strip()) and Path(args.fitness_dir).is_dir()
    auto_thresh = str(args.min_log2_sum).strip().lower() == "auto"
    fixed_thresh = None
    if not auto_thresh:
        try:
            fixed_thresh = float(args.min_log2_sum)
        except ValueError:
            raise SystemExit(f"[FATAL] --min-log2-sum: expected a float or 'auto', got "
                             f"{args.min_log2_sum!r}")
    all_groups = list(dict.fromkeys(s["group"] for s in samples if not s["is_baseline"]))
    fit_top: dict[str, pl.DataFrame] = {}
    prov: dict[str, dict] = {}              # how 'auto' priced each group (empty when pinned)
    if fitness_on:
        for g in all_groups:
            if auto_thresh:
                # priced off THIS group's endpoint depth -- a group whose fitness table is absent
                # returns None here and drops out exactly as load_fitness_top would have made it
                got = auto_min_log2_sum(
                    args.fitness_dir, args.fragvec_dir, g, args.fitness_control,
                    {s["round"]: s["name"] for s in samples
                     if s["group"] == g and not s["is_baseline"]},
                    eps=args.eps, min_last_reads=args.auto_min_last_reads)
                if got is None:
                    continue
                thr, p = got
            else:
                thr, p = fixed_thresh, None
            ft = load_fitness_top(args.fitness_dir, g, args.fitness_control, thr)
            if ft is None:
                continue
            fit_top[g] = ft
            if p:
                prov[g] = p
                log(f"[INFO] {g}: auto min_log2_sum = {thr:.4f}  (endpoint R{p['endpoint_round']} "
                    f"= {p['endpoint_sample']}, T = {p['T']:,}, eps = {p['eps']}; 1-read floor "
                    f"{p['floor']:.4f} + margin {p['margin']:.4f} -> requires >= "
                    f"{p['min_last_reads']} reads at R{p['endpoint_round']})")
        if not fit_top:
            raise SystemExit(f"[FATAL] --fitness-dir {args.fitness_dir} has no "
                             f"<group>_vs_{args.fitness_control}.log2_step_ratios.tsv; "
                             f"pass --fitness-dir '' for full calibration mode")
        dropped = [g for g in all_groups if g not in fit_top]
        how = ("per-group auto" if auto_thresh else f"log2_sum >= {fixed_thresh}")
        log(f"[INFO] fitness restriction ON ({how}, control={args.fitness_control}): "
            f"kept {sorted(fit_top)} "
            f"({', '.join(f'{g}:{fit_top[g].height}' for g in sorted(fit_top))} clonotypes); "
            f"dropped (no fitness table) {dropped}")
        if auto_thresh:
            # the bar DIFFERS between arms by construction, so the counts above are not comparable
            log("[INFO] auto thresholds are per group and depth-derived — the clonotype counts "
                "above sit at different fold-change bars and must not be compared across groups")
    else:
        log("[INFO] fitness restriction OFF (--fitness-dir ''): full calibration mode")

    ranks: dict[str, pl.DataFrame] = {}
    doms: dict[str, pl.DataFrame] = {}
    written: list[dict] = []            # samples actually written this run (the report's sample set)
    for s in samples:
        g = s["group"]
        if fitness_on and not s["is_baseline"] and g not in fit_top:
            log(f"[WARN] {s['name']}: group '{g}' has no fitness table; skipped")
            continue
        ranked, uncensored = clone_ranks(s, args.top_n)
        if uncensored:
            log(f"[INFO] {s['name']}: using fragvec_reads.parquet (uncensored)")
        # restrict the WRITTEN ranks to the group's fitness-top vectors (baseline stays full in
        # memory so each group's trajectory can still pull its R0 column)
        if fitness_on and not s["is_baseline"]:
            ranked = ranked.join(fit_top[g].select("vector"), on="vector", how="semi")
        ranks[s["name"]] = ranked

        dom = dominance_table(s, ranked, args.alpha, args.min_reads)
        if fitness_on and not s["is_baseline"]:
            # carries log2_sum + fit_rank AND the shape columns (n_steps, n_pos_steps, min_step,
            # is_monotone). Only log2_sum was thresholded; the shape is annotation -- see fitness_lib.
            dom = dom.join(fit_top[g], on="vector", how="left")
        doms[s["name"]] = dom
        crosscheck(s, dom)

        # baseline is kept in memory for trajectory R0 but is not itself a fitness-scoped output
        if fitness_on and s["is_baseline"]:
            continue
        written.append(s)
        write_parquet(ranked, args.out_dir / f"{s['name']}.clone_ranks.parquet")
        write_tsv(dom, args.out_dir / f"{s['name']}.clonotype_dominance.tsv")
        log(f"[INFO] {s['name']}: {dom.height:,} clonotypes, "
            f"{int(dom['is_dominant'].sum()) if dom.height else 0:,} dominant (p < {args.alpha})")

    # In fitness mode, prune stale per-sample outputs of now-excluded samples so results/clonerank/
    # reflects only the fitness-scoped set (else clone_top_reads would re-read stale empty/*-SN tables).
    if fitness_on:
        keep_names = {s["name"] for s in written}
        for s in samples:
            if s["name"] in keep_names:
                continue
            for suffix in (".clone_ranks.parquet", ".clonotype_dominance.tsv"):
                stale = args.out_dir / f"{s['name']}{suffix}"
                if stale.exists():
                    stale.unlink()
                    log(f"[INFO] pruned stale {stale.name} (group not fitness-scoped)")
        # per-GROUP trajectory files of dropped groups (covered groups get overwritten below)
        for f in sorted(args.out_dir.glob("*.clone_trajectories.*")):
            g = f.name.split(".clone_trajectories.")[0]
            if g not in fit_top:
                f.unlink()
                log(f"[INFO] pruned stale {f.name} (group not fitness-scoped)")

    # trajectories, per group (the baseline joins every group as its R0)
    baseline = [s for s in samples if s["is_baseline"]]
    groups = sorted(fit_top) if fitness_on else all_groups
    trajs: dict[str, tuple[pl.DataFrame, list[dict]]] = {}
    for g in groups:
        gs = baseline + [s for s in samples if s["group"] == g and not s["is_baseline"]]
        gs.sort(key=lambda s: s["round"])
        if len({s["round"] for s in gs}) != len(gs):
            log(f"[WARN] {g}: duplicate rounds {[s['round'] for s in gs]}; trajectory columns "
                f"would collide — skipping this group")
            continue
        # in fitness mode restrict every round (incl the full-in-memory baseline) to THIS group's set
        if fitness_on:
            keep = fit_top[g].select("vector")
            ranks_g = {s["name"]: ranks[s["name"]].join(keep, on="vector", how="semi") for s in gs}
            doms_g = {s["name"]: doms[s["name"]].join(keep, on="vector", how="semi") for s in gs}
        else:
            ranks_g, doms_g = ranks, doms
        traj, censored = trajectories(g, gs, ranks_g, doms_g, args.traj_top)
        if not traj.height:
            log(f"[WARN] {g}: no trajectory seeds")
            continue
        trajs[g] = (traj, gs)
        # BOTH formats on purpose: the TSV is for reading (censored cells show as "."), the parquet
        # for re-loading (typed nulls, no marker to declare). Reading the TSV back into polars needs
        # null_values=["."] or the reads_R* columns fail to parse as ints -- the parquet does not.
        write_tsv(traj, args.out_dir / f"{g}.clone_trajectories.tsv")
        write_parquet(traj, args.out_dir / f"{g}.clone_trajectories.parquet")
        log(f"[INFO] {g}: {traj.height:,} clone trajectories, {censored:,} censored cells "
            f"(clonotype below min_reads in that round — written as '.')")

    if not args.no_report:
        rep_samples = written if fitness_on else samples
        gcol = group_color(rep_samples)     # same colour per group as the other cross reports
        note = None
        if fitness_on:
            head = (f"Restricted to <b>fitness-top clonotypes</b> (summed empty-corrected "
                    f"log2 fold-change &ge; the threshold below, in "
                    f"<code>&lt;group&gt;_vs_{esc(args.fitness_control)}</code>). Groups without a "
                    f"fitness table are excluded, so the <code>empty</code>/<code>initial</code> "
                    f"negative control is not shown here — rerun with <code>--fitness-dir ''</code> "
                    f"for the full calibration view.")
            if prov:
                bars = "; ".join(
                    f"<b>{esc(g)}</b> &ge; {prov[g]['threshold']:.4f} "
                    f"(endpoint R{prov[g]['endpoint_round']} = "
                    f"<code>{esc(prov[g]['endpoint_sample'])}</code>, T = {prov[g]['T']:,})"
                    for g in sorted(prov))
                k = next(iter(prov.values()))["min_last_reads"]
                note = (f"{head}</p><p>Each threshold is <b>derived from that group's own endpoint "
                        f"depth</b>, not typed in: {bars}. The bar is set just above the score a "
                        f"single endpoint read can reach, so clearing it means <b>&ge; {k} reads "
                        f"at the last round</b> — uniform in meaning, and therefore <b>different "
                        f"in magnitude between arms</b>. The per-group clonotype counts below sit "
                        f"at different bars and must not be compared with each other.")
            else:
                note = (f"{head}</p><p>Threshold pinned at "
                        f"<b>{float(args.min_log2_sum):.4f}</b> for every group "
                        f"(<code>--min-log2-sum</code>); it is depth-dependent, so re-derive it "
                        f"with <code>--min-log2-sum auto</code> if the depth changed.")
        build_report(args.out_dir / "clonerank_report.html", rep_samples, doms, trajs, gcol,
                     args.alpha, note)

    log("[INFO] done.")
    return 0


_STYLE = """<style>
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:2rem auto;max-width:1150px;padding:0 1rem;color:#1a1a1a}
h1{font-size:1.6rem}h2{font-size:1.2rem;margin-top:2rem;border-bottom:1px solid #eee;padding-bottom:.3rem}
h3{font-size:1.05rem;margin-top:1.4rem}
table{border-collapse:collapse;font-size:.85rem;width:100%}th,td{border:1px solid #e2e2e2;padding:.25rem .5rem;text-align:left}
th{background:#f6f8fa}td.num{text-align:right;font-variant-numeric:tabular-nums}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.75rem;word-break:break-all}
.small{color:#666;font-size:.85rem}code{background:#f0f2f4;padding:.05rem .3rem;border-radius:3px}
.callout{border-left:3px solid #d0d7de;background:#f6f8fa;padding:.6rem 1rem;margin:1rem 0;font-size:.88rem}
.callout p{margin:.5rem 0}
@media(prefers-color-scheme:dark){body{background:#15181c;color:#e6e6e6}th{background:#20242a}
th,td{border-color:#333}h2{border-color:#333}code{background:#20242a}
.callout{background:#1b1f24;border-color:#333}}
</style>"""


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
