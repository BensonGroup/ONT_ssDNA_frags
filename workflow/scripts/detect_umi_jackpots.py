#!/usr/bin/env python
"""Per-study UMI/fragvec "jackpot" outlier detector for the initial (R0) library.

DETECTION IS NAIVE-LIBRARY ONLY. A collapsed reads/UMI ratio is an artefact signal ONLY in the naive
INITIAL library: there every construct starts at comparable abundance, so a fragvec whose reads
pile onto ~one UMI can only be an amplification/synthesis artefact. In a post-selection round a
genuinely enriched winner ALSO has a high reads/UMI ratio, so detecting there would wrongly flag
real hits. That asymmetry is why the stage is fan-in/fan-out -- detect once on R0, apply the
resulting blacklist to every round (apply_jackpot_blacklist.py). `rule jackpot_detect` names INITIAL
directly, so there is no wildcard to point this at the wrong sample.

The converse limit, and it matters: jackpotting is worst when FEW templates enter the PCR, so the R0
library is the LEAST susceptible sample in the study. This catches the synthesis/aliquot artefacts
ALREADY IN the pool. It does NOT catch a jackpot generated inside a bottlenecked selected round, and
clean behaviour here does not license assuming a bottlenecked sample's high-count lineages are
bias-free (clarification.md).

A *jackpot* is a fragvec (fragment `vector`) whose reads collapse onto very few UMIs: thousands of
physical copies of one construct whose UMI slot failed to randomise, so they pile onto ~one barcode.
Ranked by raw abundance it looks like a top hit; behaviourally it is an amplification/synthesis
artefact, not a selection winner or a generic PCR winner.

THE SIGNAL IS THE PIPELINE'S OWN. Per fragvec: `n_reads, n_umis, max_reads_per_umi` -- the two-axis
readout build_fragvec exists to produce, whose docstring calls `max_reads_per_umi` "the jackpot
detector". This does not invent a statistic; it applies a behavioural, design-independent cut:

    ratio      = n_reads / n_umis            (reads per independent UMI)
    dominance  = max_reads_per_umi / n_reads (fraction carried by the single top UMI)
    FLAG  <=>  n_reads >= --min-reads  AND  ratio > --ratio-max

It builds that table itself, via fragvec_lib, from the R0 library's CORRECTED tree -- the same code
build_fragvec runs, so the two cannot drift. It used to READ `fragvec_initial/`, which made detection
depend on fragvec and forced this whole step downstream of the branch it cleans; that is why it was
a post-hoc retro-fit rather than a stage.

`ratio_max = 10` is cited: longread_umi `BIN_CLUSTER_RATIO=10` (Karst et al., Nat Methods 2021). A
data-driven robust cut -- median + k*MAD of log10(ratio) over signal-carrying fragvecs -- is
computed and REPORTED for transparency, but the action stays the cited threshold.

On NB01 (3,624 fragvecs with n_reads>=10) the ratio has p50=1.12, p99=1.57, then a huge gap to the
artefact at 48.7; `ratio>10` flags exactly 3 fragvecs and correctly ignores the abundant-but-diverse
aptamer loop (19,601 reads / 18,214 UMIs, ratio 1.08).

This is a FILTER, not a deduplication: it names an audited set of artefact barcodes. High-count
lineages are the signal here and are never collapsed (clarification.md:667).

DELIVERABLES (into --out-dir):
    <S>.umi_blacklist.tsv           one row per (flagged fragvec, UMI) where the UMI's OWN
                                    n_reads >= --min-reads -- the dominant artefact barcode and its
                                    larger frame-shift variants, not singletons.
    <S>.umi_blacklist_summary.tsv   thresholds, counts, robust cut, ratio percentiles, tool
                                    version, input file names (provenance).
    <S>.fragvec_jackpot_report.html ratio-vs-abundance scatter (clean separation) + flagged table.
    <S>.umi_jackpot_report.html     flagged-UMI table (composition annotations) + UMI family-size
                                    distribution with flagged barcodes marked.

Composition metrics (homopolymer maxrun via `umi_filter_lib.maxrun`, distinct-base count, base
Shannon entropy in bits) are ANNOTATIONS on each flagged UMI -- never filter drivers.

The blacklist is produced per study from that study's own initial library, because which sequences
are anomalous is study-specific (synthesis / aliquot). Downstream excludes by the artefact UMI
SEQUENCE -- precise: it removes the artefact's reads while preserving legitimate reads that share
the same fragment layout.

DETECT ONLY. Filtering a round against the blacklist is apply_jackpot_blacklist.py, which writes a
whole sample tree (fragments must follow the same read set as the fused parquet, or the downstream
`reads_decoded` runs high).

CLI:
    detect_umi_jackpots.py --src-stem results/umicorr_initial/R0/R0 \\
                           --umis results/decode_initial/R0/R0.umis.parquet \\
                           --out-dir results/jackpot --sample R0 \\
                           [--min-reads 10] [--ratio-max 10] [--robust-k 5] [--force]
"""
from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fragvec_lib as fvl              # clonotype construction, shared with build_fragvec
import tree_io as tio
from ligand_vocab import anchor_set    # authoritative backbone/anchor names (from the umis parquet)
from umi_filter_lib import maxrun
from report_lib import GREY, esc, fig_html

TOOL = "detect_umi_jackpots.py"
VERSION = "2.0"   # 2.0: builds its own clonotype table (no fragvec dependency); detect-only

# Highlight colour for the flagged jackpots in every figure (a warning vermillion, kept clear
# of the group palette so it never reads as a cell line).
FLAG_COLOR = "#D55E00"


# --------------------------------------------------------------------------- composition annotations
def _entropy_bits(seq: str) -> float:
    """Shannon entropy (base-2, bits) over the base composition of ``seq`` (0.0 for empty).

    Purely descriptive: the artefact barcode is C/A-heavy, so this quantifies HOW skewed it is
    for the reader -- it is never used to include or exclude a UMI (see module docstring)."""
    n = len(seq)
    if n == 0:
        return 0.0
    counts: dict[str, int] = {}
    for c in seq:
        counts[c] = counts.get(c, 0) + 1
    h = 0.0
    for k in counts.values():
        p = k / n
        h -= p * math.log2(p)
    return h


def _n_distinct_bases(seq: str) -> int:
    return len({c for c in seq if c in "ACGT"})


# --------------------------------------------------------------------------- IO
def load_clonotypes(frags_pq: Path, fused_pq: Path, umis_pq: Path, min_reads: int):
    """(counts, UMIs) for the R0 library, built from its CORRECTED tree.

    This used to read `fragvec_initial/*.fragvec_counts.parquet`, which made detection depend on
    fragvec -- and therefore forced the jackpot step DOWNSTREAM of the branch it is meant to clean.
    The fragvec table is now built here, from the same (fragments, fused) pair umi_correct emits,
    via fragvec_lib -- the same code build_fragvec uses, so the ratio a fragvec is flagged on and
    the counts fragvec later reports cannot drift.
    """
    anc_ids = fvl.anchor_ids(anchor_set(umis_pq))
    rv, _reads_decoded = fvl.read_vectors(frags_pq, fused_pq, "fused_seq_corrected", anc_ids)
    if rv.height == 0:
        return (pl.DataFrame(schema=fvl.COUNTS_SCHEMA), pl.DataFrame(schema=fvl.CLONES_SCHEMA))
    per_vu = fvl.vector_umi_counts(rv)
    counts = fvl.per_vector_counts(per_vu, min_reads)
    # Clone membership over the SIGNAL-CARRYING vectors -- the same restriction build_fragvec
    # applies, which is why min_reads here must be fragvec.min_reads and not an independent floor.
    keep = counts.filter(~pl.col("is_noise")).select("vector")
    clones = fvl.clone_membership(per_vu, keep)
    return counts, clones


# --------------------------------------------------------------------------- detector
def detect(counts: pl.DataFrame, min_reads: int, ratio_max: float, robust_k: float):
    """Apply the behavioural jackpot cut and compute the transparency statistics.

    Returns (flagged_df, stats). `flagged_df` carries ratio/dominance and is sorted
    deterministically (ratio desc, then vector). `stats` holds the robust cut, percentiles and
    signal-fragvec count for the summary + reports.
    """
    d = counts.with_columns(
        (pl.col("n_reads") / pl.col("n_umis")).alias("ratio"),
        (pl.col("max_reads_per_umi") / pl.col("n_reads")).alias("dominance"),
    )
    signal = d.filter(pl.col("n_reads") >= min_reads)
    # data-driven robust cut: median + k*MAD of log10(ratio) over signal clonotypes.
    logr = signal.select(pl.col("ratio").log10().alias("lr"))["lr"]
    med = float(logr.median()) if signal.height else float("nan")
    mad = float((logr - med).abs().median()) if signal.height else float("nan")
    robust_cut = 10.0 ** (med + robust_k * mad) if signal.height else float("nan")

    flagged = (signal.filter(pl.col("ratio") > ratio_max)
               .sort(["ratio", "vector"], descending=[True, False]))
    stats = {
        "n_signal": signal.height,
        "n_flagged": flagged.height,
        "ratio_p50": float(signal["ratio"].quantile(0.50)) if signal.height else float("nan"),
        "ratio_p90": float(signal["ratio"].quantile(0.90)) if signal.height else float("nan"),
        "ratio_p99": float(signal["ratio"].quantile(0.99)) if signal.height else float("nan"),
        "log10_ratio_median": med,
        "log10_ratio_mad": mad,
        "robust_cut_ratio": robust_cut,
    }
    return d, flagged, stats


def build_blacklist(flagged: pl.DataFrame, clones: pl.DataFrame, min_reads: int,
                    ratio_max: float) -> pl.DataFrame:
    """One row per (flagged fragvec, UMI) where the UMI's OWN n_reads >= min_reads.

    That captures the dominant artefact barcode and its larger frame-shift variants while
    excluding singletons. Composition columns are annotations only. Stable deterministic sort:
    ratio desc, then vector, then umi.
    """
    ctx = flagged.select(
        "vector", "n_umis", "ratio", "dominance", "max_reads_per_umi", "rpm")
    umis = (clones.join(flagged.select("vector"), on="vector", how="inner")
            .filter(pl.col("n_reads") >= min_reads)
            .rename({"n_reads": "umi_n_reads"}))
    bl = umis.join(ctx, on="vector", how="left")

    rows = []
    reason = f"jackpot: reads collapse onto few UMIs (ratio>{ratio_max:g})"
    threshold = f"n_reads>={min_reads} & ratio>{ratio_max:g}"
    method = f"{TOOL} v{VERSION}: ratio=n_reads/n_umis (build_fragvec fragvec); cited BIN_CLUSTER_RATIO"
    for r in bl.iter_rows(named=True):
        u = r["umi"]
        rows.append({
            "vector": r["vector"],
            "umi": u,
            "n_reads": int(r["umi_n_reads"]),
            "n_umis": int(r["n_umis"]),
            "ratio": round(float(r["ratio"]), 6),
            "dominance": round(float(r["dominance"]), 6),
            "max_reads_per_umi": int(r["max_reads_per_umi"]),
            "rpm": round(float(r["rpm"]), 6),
            "homopolymer_maxrun": maxrun(u),
            "n_distinct_bases": _n_distinct_bases(u),
            "base_entropy_bits": round(_entropy_bits(u), 6),
            "reason": reason,
            "threshold": threshold,
            "method": method,
        })
    cols = ["vector", "umi", "n_reads", "n_umis", "ratio", "dominance", "max_reads_per_umi",
            "rpm", "homopolymer_maxrun", "n_distinct_bases", "base_entropy_bits",
            "reason", "threshold", "method"]
    if not rows:
        return pl.DataFrame(schema={c: (pl.Utf8 if c in ("vector", "umi", "reason", "threshold",
                                                          "method") else pl.Float64) for c in cols})
    df = pl.DataFrame(rows).select(cols)
    return df.sort(["ratio", "vector", "umi"], descending=[True, False, False])


# --------------------------------------------------------------------------- styling / html
def style(fig, height, title=None):
    fig.update_layout(height=height, title=title, template="plotly_white",
                      margin=dict(l=60, r=20, t=50 if title else 20, b=50),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                      font=dict(size=12))
    return fig


_STYLE = """<style>
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:2rem auto;max-width:1150px;padding:0 1rem;color:#1a1a1a}
h1{font-size:1.6rem}h2{font-size:1.2rem;margin-top:2rem;border-bottom:1px solid #eee;padding-bottom:.3rem}
h3{font-size:1.05rem;margin-top:1.4rem}
table{border-collapse:collapse;font-size:.85rem;width:100%}th,td{border:1px solid #e2e2e2;padding:.25rem .5rem;text-align:left}
th{background:#f6f8fa}td.num{text-align:right;font-variant-numeric:tabular-nums}
.small{color:#666;font-size:.85rem}code{background:#f0f2f4;padding:.05rem .3rem;border-radius:3px}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.75rem;word-break:break-all}
.flag{color:#D55E00;font-weight:600}
@media(prefers-color-scheme:dark){body{background:#15181c;color:#e6e6e6}th{background:#20242a}
th,td{border-color:#333}h2{border-color:#333}code{background:#20242a}.small{color:#9aa4ae}}
</style>"""


def _head(title):
    return [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>{esc(title)}</title>", _STYLE,
        f"<script>{get_plotlyjs()}</script></head><body>",
    ]


def flagged_clonotype_table(flagged: pl.DataFrame):
    body = "".join(
        f"<tr><td class='mono'>{esc(r['vector'])}</td>"
        f"<td class='num'>{r['n_reads']:,}</td><td class='num'>{r['n_umis']:,}</td>"
        f"<td class='num'>{r['max_reads_per_umi']:,}</td>"
        f"<td class='num flag'>{r['ratio']:.2f}</td>"
        f"<td class='num'>{r['dominance']:.3f}</td>"
        f"<td class='num'>{r['rpm']:,.1f}</td></tr>"
        for r in flagged.iter_rows(named=True))
    return ("<div style='overflow-x:auto'><table><thead><tr><th>fragvec (vector)</th>"
            "<th>n_reads</th><th>n_umis</th><th>max_reads_per_umi</th><th>ratio</th>"
            "<th>dominance</th><th>rpm</th></tr></thead>"
            f"<tbody>{body}</tbody></table></div>")


def fig_scatter(d: pl.DataFrame, flagged: pl.DataFrame, ratio_max, robust_cut, min_reads):
    """ratio (y, log) vs n_reads (x, log) over ALL fragvecs; flagged jackpots highlighted."""
    healthy = d.join(flagged.select("vector"), on="vector", how="anti")
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=healthy["n_reads"].to_list(), y=healthy["ratio"].to_list(), mode="markers",
        name="fragvecs", marker=dict(size=4, opacity=0.35, color=GREY),
        hovertemplate="reads %{x}<br>ratio %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=flagged["n_reads"].to_list(), y=flagged["ratio"].to_list(), mode="markers",
        name="FLAGGED jackpot", marker=dict(size=13, color=FLAG_COLOR, symbol="x",
                                            line=dict(width=1, color=FLAG_COLOR)),
        customdata=[esc(v) for v in flagged["vector"].to_list()],      # the FULL vector
        hovertemplate="<b>%{customdata}</b><br>reads %{x}<br>ratio %{y:.2f}<extra></extra>"))
    fig.add_hline(y=ratio_max, line=dict(color=FLAG_COLOR, dash="dash"),
                  annotation_text=f"ratio_max = {ratio_max:g} (cited)", annotation_position="top left")
    if robust_cut == robust_cut and math.isfinite(robust_cut):  # not NaN/inf
        fig.add_hline(y=robust_cut, line=dict(color=GREY, dash="dot"),
                      annotation_text=f"robust cut = {robust_cut:.2f}", annotation_position="bottom left")
    fig.add_vline(x=min_reads, line=dict(color=GREY, dash="dot"), opacity=0.5)
    fig.update_xaxes(type="log", title="n_reads  (abundance)")
    fig.update_yaxes(type="log", title="ratio = n_reads / n_umis  (reads per independent UMI)")
    return fig_html(style(fig, 520,
        "Reads-per-UMI vs abundance — the flagged jackpots sit alone in the top-right corner, "
        "orders of magnitude above the healthy population"))


def fig_family_sizes(clones: pl.DataFrame, flagged: pl.DataFrame, blacklist: pl.DataFrame, min_reads):
    """UMI family-size distribution with the flagged barcodes marked as outliers."""
    # per-UMI read counts across flagged clonotypes vs everywhere else
    fl_v = set(flagged["vector"].to_list())
    per_umi = clones.group_by("umi").agg(pl.col("n_reads").sum().alias("fam"))
    all_fam = per_umi["fam"].to_list()
    flag_fam = blacklist["n_reads"].to_list()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=all_fam, marker_color=GREY, opacity=0.75,
                               name="all UMI families", xbins=dict(start=0)))
    fig.update_xaxes(type="log", title="UMI family size (reads on one UMI)")
    fig.update_yaxes(type="log", title="number of UMIs")
    for fam, umi in zip(blacklist["n_reads"].to_list(), blacklist["umi"].to_list()):
        fig.add_vline(x=fam, line=dict(color=FLAG_COLOR, width=1, dash="dot"), opacity=0.5)
    # mark the dominant artefact explicitly
    if blacklist.height:
        top = blacklist.sort("n_reads", descending=True).row(0, named=True)
        fig.add_annotation(x=math.log10(max(top["n_reads"], 1)), y=0, yref="paper",
                           text=f"<b>{esc(top['umi'])}</b><br>{top['n_reads']:,} reads",
                           showarrow=True, arrowhead=2, ax=-40, ay=-60,
                           font=dict(size=10, color=FLAG_COLOR), arrowcolor=FLAG_COLOR)
    return fig_html(style(fig, 460,
        "UMI family-size distribution — the blacklisted barcodes (dotted lines) are extreme "
        "outliers in the tail"))


def flagged_umi_table(blacklist: pl.DataFrame):
    body = "".join(
        f"<tr><td class='mono'>{esc(r['umi'])}</td>"
        f"<td class='num'>{r['n_reads']:,}</td>"
        f"<td class='num'>{r['homopolymer_maxrun']}</td>"
        f"<td class='num'>{r['n_distinct_bases']}</td>"
        f"<td class='num'>{r['base_entropy_bits']:.2f}</td>"
        f"<td class='num flag'>{r['ratio']:.2f}</td>"
        f"<td class='num'>{r['dominance']:.3f}</td>"
        f"<td class='mono' title='{esc(r['vector'])}'>{esc(r['vector'][:48])}"
        f"{'…' if len(r['vector'])>48 else ''}</td></tr>"
        for r in blacklist.iter_rows(named=True))
    return ("<div style='overflow-x:auto'><table><thead><tr><th>UMI (barcode)</th>"
            "<th>family reads</th><th>homopolymer_maxrun</th><th>n_distinct_bases</th>"
            "<th>base_entropy_bits</th><th>fragvec ratio</th><th>dominance</th>"
            "<th>fragvec</th></tr></thead>"
            f"<tbody>{body}</tbody></table></div>")


# --------------------------------------------------------------------------- report assembly
def write_fragvec_report(out: Path, sample, d, flagged, stats, ratio_max, min_reads):
    H = _head(f"{sample} — fragvec jackpot report")
    H += [
        f"<h1>{esc(sample)} — fragvec jackpot report</h1>",
        "<p class='small'>Behavioural, design-independent detector: a fragvec is flagged as a "
        f"<b>jackpot</b> when it has ≥{min_reads} reads AND <code>ratio = n_reads/n_umis &gt; "
        f"{ratio_max:g}</code> (its reads collapse onto very few independent UMIs). "
        f"<code>ratio_max={ratio_max:g}</code> is cited (longread_umi <code>BIN_CLUSTER_RATIO</code>, "
        "Karst et al. Nat Methods 2021). This is workflow QC, not a biological claim.</p>",
        f"<p class='small'>Over {stats['n_signal']:,} signal-carrying fragvecs "
        f"(n_reads≥{min_reads}) the ratio has p50={stats['ratio_p50']:.2f}, "
        f"p90={stats['ratio_p90']:.2f}, p99={stats['ratio_p99']:.2f}, then a gap to the flagged "
        f"outliers. The data-driven robust cut (median + k·MAD on log10 ratio) is "
        f"{stats['robust_cut_ratio']:.2f}; the default action uses the cited {ratio_max:g}.</p>",
        "<h2>Reads-per-UMI vs abundance</h2>",
        fig_scatter(d, flagged, ratio_max, stats["robust_cut_ratio"], min_reads),
        f"<h2>Flagged fragvecs ({flagged.height})</h2>",
        flagged_clonotype_table(flagged),
        "<h2>What this does and does not license</h2>",
        "<p class='small'>Detection runs on the <b>naive R0 library only</b>, and the blacklist is "
        "applied to every round. That asymmetry is forced: post-selection, a genuinely enriched "
        "winner <i>also</i> collapses its reads onto few UMIs, so the ratio no longer separates "
        "an artefact from a hit — detecting there would blacklist true winners.</p>",
        "<p class='small'>The converse limit is worth stating plainly: jackpotting is worst when "
        "<b>few templates enter the PCR</b>, so the R0 library is the <i>least</i> susceptible "
        "sample in the study. What this catches is the synthesis/aliquot artefacts already present "
        "in the pool. It does <b>not</b> catch a jackpot generated inside a bottlenecked selected "
        "round, and clean behaviour here does not license the assumption that a bottlenecked "
        "sample's high-count lineages are bias-free.</p>",
        "<p class='small'>This is a filter, not a deduplication: it drops a named, audited set of "
        "artefact barcodes. High-count lineages are the signal here and are never collapsed.</p>",
        "</body></html>",
    ]
    out.write_text("\n".join(x for x in H if x))


def write_umi_report(out: Path, sample, blacklist, clones, flagged, stats, min_reads, ratio_max):
    H = _head(f"{sample} — umi jackpot report")
    H += [
        f"<h1>{esc(sample)} — UMI jackpot report</h1>",
        "<p class='small'>The UMIs blacklisted from the flagged fragvecs: for each flagged "
        f"fragvec, the barcodes whose own read count is ≥{min_reads} (the dominant artefact "
        "barcode plus its larger frame-shift variants, not singletons). Composition columns "
        "(homopolymer run, distinct bases, base Shannon entropy in bits) are <b>annotations only</b>, "
        "never filter drivers.</p>",
        f"<p class='small'>{blacklist.height} UMI rows across {flagged.height} flagged fragvecs. "
        "Downstream analyses exclude these barcode sequences from later rounds (which cannot "
        "UMI-dedup but can still read the UMI), removing the artefact's reads while preserving "
        "legitimate reads that share the same fragment layout.</p>",
        "<h2>Flagged UMIs</h2>",
        flagged_umi_table(blacklist),
        "<h2>UMI family-size distribution</h2>",
        fig_family_sizes(clones, flagged, blacklist, min_reads),
        "</body></html>",
    ]
    out.write_text("\n".join(x for x in H if x))


def write_summary(out: Path, sample, stats, min_reads, ratio_max, robust_k, blacklist,
                  frags_pq, fused_pq):
    ts = datetime.now(timezone.utc).isoformat()
    rows = [
        ("sample", sample),
        ("tool", f"{TOOL} v{VERSION}"),
        ("generated_utc", ts),
        # the clonotype table is built HERE from these two, via fragvec_lib -- it is not read from
        # the fragvec tree, which now sits downstream of this stage
        ("input_fragments", str(frags_pq)),
        ("input_fused_corrected", str(fused_pq)),
        ("min_reads", min_reads),
        ("ratio_max", ratio_max),
        ("robust_k", robust_k),
        ("n_signal_clonotypes", stats["n_signal"]),
        ("n_flagged_clonotypes", stats["n_flagged"]),
        ("n_blacklist_rows", blacklist.height),
        ("ratio_p50", f"{stats['ratio_p50']:.6g}"),
        ("ratio_p90", f"{stats['ratio_p90']:.6g}"),
        ("ratio_p99", f"{stats['ratio_p99']:.6g}"),
        ("log10_ratio_median", f"{stats['log10_ratio_median']:.6g}"),
        ("log10_ratio_mad", f"{stats['log10_ratio_mad']:.6g}"),
        ("robust_cut_ratio", f"{stats['robust_cut_ratio']:.6g}"),
    ]
    lines = ["key\tvalue"] + [f"{k}\t{v}" for k, v in rows]
    out.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- main
def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Jackpot detector for the naive (R0) library. Detect only; see "
                    "apply_jackpot_blacklist.py to filter a round against the blacklist.")
    # No default: the rule names INITIAL explicitly. There is no wildcard to get wrong, which is
    # how "initial-only" is enforced now -- the old --allow-noninitial escape hatch is gone with it.
    ap.add_argument("--sample", required=True, help="the R0 library's sample name")
    ap.add_argument("--src-stem", required=True,
                    help="the R0 library's CORRECTED stem, e.g. results/umicorr_initial/R0/R0 "
                         "(reads <stem>.fragments.parquet + <stem>.fused.parquet)")
    ap.add_argument("--umis", required=True,
                    help="decode umis.parquet, for the anchor VOCABULARY only")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="output dir for the blacklist, its summary and the two audit reports")
    ap.add_argument("--min-reads", type=int, default=10,
                    help="signal floor. MUST equal fragvec.min_reads: the UMI table is "
                         "restricted by that same floor, so a lower one here would flag "
                         "fragvecs whose UMIs are absent from it and drop them silently")
    ap.add_argument("--ratio-max", type=float, default=10.0,
                    help="flag if n_reads/n_umis exceeds this (default 10, cited)")
    ap.add_argument("--robust-k", type=float, default=5.0,
                    help="k for the REPORTED median+k*MAD cut (never acted on)")
    ap.add_argument("--force", action="store_true", help="overwrite existing outputs")
    args = ap.parse_args(argv[1:])

    src = Path(args.src_stem)
    if not src.is_absolute():
        src = Path.cwd() / src
    frags_pq = tio.part(src, ".fragments.parquet")
    fused_pq = tio.part(src, ".fused.parquet")
    umis_pq = Path(args.umis).resolve()
    for q in (frags_pq, fused_pq, umis_pq):
        if not q.exists():
            sys.stderr.write(f"[FATAL] missing input: {q}\n")
            return 1

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    bl_out = out_dir / f"{args.sample}.umi_blacklist.tsv"
    sum_out = out_dir / f"{args.sample}.umi_blacklist_summary.tsv"
    fv_out = out_dir / f"{args.sample}.fragvec_jackpot_report.html"
    umi_out = out_dir / f"{args.sample}.umi_jackpot_report.html"
    if not args.force:
        for q in (bl_out, sum_out, fv_out, umi_out):
            if q.exists():
                sys.stderr.write(f"[FATAL] exists (use --force): {q}\n")
                return 1

    counts, clones = load_clonotypes(frags_pq, fused_pq, umis_pq, args.min_reads)
    d, flagged, stats = detect(counts, args.min_reads, args.ratio_max, args.robust_k)
    blacklist = build_blacklist(flagged, clones, args.min_reads, args.ratio_max)

    blacklist.write_csv(bl_out, separator="\t")
    write_summary(sum_out, args.sample, stats, args.min_reads, args.ratio_max, args.robust_k,
                  blacklist, frags_pq, fused_pq)
    write_fragvec_report(fv_out, args.sample, d, flagged, stats, args.ratio_max, args.min_reads)
    write_umi_report(umi_out, args.sample, blacklist, clones, flagged, stats, args.min_reads,
                     args.ratio_max)

    rc = stats["robust_cut_ratio"]
    rc_s = f"{rc:.2f}" if rc == rc and math.isfinite(rc) else "n/a"
    print(f"[jackpot] {args.sample}: {stats['n_signal']:,} signal fragvecs, "
          f"{stats['n_flagged']} flagged, {blacklist.height} blacklist rows "
          f"(reported robust cut {rc_s}; action = cited ratio_max {args.ratio_max:g})")
    for label, q in (("blacklist", bl_out), ("summary", sum_out),
                     ("fragvec report", fv_out), ("umi report", umi_out)):
        print(f"[jackpot]   {label:15s} -> {q}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
