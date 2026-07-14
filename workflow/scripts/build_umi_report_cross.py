#!/usr/bin/env python
"""Cross-sample UMI characterisation report (branch `umi`, mode `correct`).

Descriptive workflow QC over the per-sample corrected-UMI outputs (`umi/<s>/<s>.umi_species.tsv`,
`.umi_summary.tsv`, `.fused.parquet`) and the complexity tables (`umi_complexity.tsv`,
`umi_collapse.tsv`). One self-contained HTML file, plotly figures inlined.

EVERY FIGURE IS ORGANISED BY `group` -- the config's cell line. One colour per group (report_lib's
CVD-validated palette, the same colour in all three cross reports), the R0 initial library always
grey and dashed because it is a shared baseline, not a selection arm.

`group` is the ONLY sample axis -- there is no `role` field (see report_lib's reserved names). The
`empty` arm (the PCR-only / no-cell rounds) is a PEER of every selection arm HERE: its own colour,
its own C_eff trajectory line, its own UpSet panel, its own purity trace, and NO special-casing in
any figure function. The one place this report knows the word `empty` is the "Protocol collapse"
subsection, which renders the `kind: protocol` collapse rows -- and even that reads the `kind`, not
the group name. If you find yourself writing `if group == EMPTY_GROUP` inside a figure, stop.

Panels (no biological interpretation -- this is a workflow):
  * Summary        per-sample reads in / filtered (by class) / corrected (species merged, relabelled).
  * Complexity     C_eff vs selection round per group, the collapse relative to R0, and the
                   depth-corrected richness. THE depth-robust view; read it before anything below.
  * Abundance      rank-abundance curves (log-log) and D50/D90 = the proportion of distinct UMI
                   species whose cumulative reads reach 50% / 90% (+ a Lorenz-style curve).
  * Overlap        per non-baseline group, an UpSet-like view of distinct corrected-UMI species
                   shared across the group's samples (top-N intersections).
  * Validity QC    UMI-group ligand-signature purity vs group size (join corrected UMI -> decode
                   fragments), per group, and a collision null model (expected chance ed<=1 merges
                   vs observed).

THIS SCRIPT COMPUTES NO COMPLEXITY. It renders `umi_complexity.tsv` / `umi_collapse.tsv`, which
`rule umi_complexity` produced via effective_complexity.py -> complexity_lib. That is deliberate:
the estimator has exactly two callers in this branch (umi_correct.py and effective_complexity.py),
so the number in a table and the number in a figure cannot drift apart. It used to recompute c_eff
and the rarefaction depth here, over the DISCOVERED sample set rather than the DECLARED one -- which
silently disagreed with the table whenever report.r0_baseline was off.

Parameterised entirely by environment variables (set by rule report_umi_cross):
    L2F_BASE          umi/ dir to glob `*/` for experimental sample subdirs
    L2F_R0_STEM       umi_initial/.../<INITIAL> stem for the R0 baseline ("" disables it)
    L2F_SAMPLES       config-derived metadata TSV (name<TAB>group<TAB>round<TAB>idx)
    L2F_COMPLEXITY    per-sample complexity table (umi_complexity.tsv)
    L2F_COLLAPSE      per-sample collapse ratios (umi_collapse.tsv)
    L2F_DECODE_BASE   decode/ dir (experimental fragments for the purity QC)
    L2F_R0_FRAGS      the initial library's decode fragments.parquet ("" if no baseline)
"""
from __future__ import annotations

import csv
import math
import os
import sys
from pathlib import Path

import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import get_plotlyjs

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_lib import (
    BASELINE_GREY, BASELINE_GROUP, GREY, color_key_html, esc, fig_html, first_of_group, group_band,
    group_color, group_dividers, load_meta,
)

BASE = Path(os.environ.get("L2F_BASE", "")).resolve()
R0_STEM = os.environ.get("L2F_R0_STEM", "").strip()
SAMPLES_TSV = os.environ.get("L2F_SAMPLES", "").strip()
COMPLEXITY_TSV = os.environ.get("L2F_COMPLEXITY", "").strip()
COLLAPSE_TSV = os.environ.get("L2F_COLLAPSE", "").strip()
DECODE_BASE = os.environ.get("L2F_DECODE_BASE", "").strip()
R0_FRAGS = os.environ.get("L2F_R0_FRAGS", "").strip()
OUT = BASE / "umi_report_cross.html"


def style(fig, height, title=None, top=None):
    # `top` reserves extra top margin for a rotated group band (see report_lib.group_band): a
    # rotated label is ~0.7*(width + height) tall and would otherwise collide with the title.
    fig.update_layout(
        height=height, title=title, template="plotly_white",
        margin=dict(l=60, r=20, t=(top if top else (50 if title else 20)), b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        font=dict(size=12),
    )
    return fig


def sci(v):
    return "" if v is None else f"{v:.3g}"


# --------------------------------------------------------------------------- discovery + metadata
def discover(meta):
    samples = []
    for d in sorted(p for p in BASE.glob("*/") if p.is_dir()):
        name = d.name
        sp = d / f"{name}.umi_species.tsv"
        if not sp.exists():
            continue
        m = meta.get(name, {})
        samples.append({
            "name": name, "dir": d, "species": sp,
            "summary": d / f"{name}.umi_summary.tsv",
            "fused": d / f"{name}.fused.parquet",
            "frags": (Path(DECODE_BASE) / name / f"{name}.fragments.parquet") if DECODE_BASE else None,
            "group": m.get("group", _parse_group(name)),
            "round": m.get("round", _parse_round(name)),
            "idx": m.get("idx", 10_000),
            "is_baseline": False,
        })
    samples.sort(key=lambda s: (s["idx"], s["round"], s["name"]))
    if R0_STEM:
        r0 = Path(R0_STEM)
        if (r0.parent / f"{r0.name}.umi_species.tsv").exists():
            m = meta.get(r0.name, {})
            samples.insert(0, {
                "name": r0.name, "dir": r0.parent, "species": r0.parent / f"{r0.name}.umi_species.tsv",
                "summary": r0.parent / f"{r0.name}.umi_summary.tsv",
                "fused": r0.parent / f"{r0.name}.fused.parquet",
                "frags": Path(R0_FRAGS) if R0_FRAGS else None,
                # the baseline's group/round now come from the config like everyone else's; the
                # literals are only the fallback for a standalone run with no metadata TSV.
                # `is_baseline` is what every consumer keys on -- there is no `role` field.
                "group": m.get("group", BASELINE_GROUP), "round": m.get("round", 0),
                "idx": -1, "is_baseline": True,
            })
    return samples


def _parse_group(name):
    import re
    m = re.match(r"(?P<g>.+?)[_-]?R\d+$", name)
    return m.group("g") if m else name


def _parse_round(name):
    import re
    m = re.search(r"R(\d+)$", name)
    return int(m.group(1)) if m else 0


def load_species(s):
    df = pl.read_csv(s["species"], separator="\t")
    cols = df.columns
    return df.rename({cols[0]: "umi", cols[1]: "n_reads"}).select("umi", "n_reads")


def load_summary(s):
    if not s["summary"].exists():
        return {}
    df = pl.read_csv(s["summary"], separator="\t")
    return df.row(0, named=True) if df.height else {}


def load_table(path):
    """A TSV emitted by effective_complexity.py -> list of dicts, numbers coerced, blanks -> None."""
    if not path or not Path(path).exists():
        return []
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            rows.append({k: _num(v) for k, v in r.items()})
    return rows


def _num(v):
    if v is None or v == "":
        return None
    if v in ("true", "false"):
        return v == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def fmt(v, nd=0):
    """Table cell: thousands-separated, and never `None`/`inf`/`nan` (an empty cell is honest)."""
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return ""
    if isinstance(v, bool):
        return "yes" if v else ""
    if isinstance(v, float):
        return f"{v:,.{nd}f}" if abs(v) >= 0.001 or v == 0 else f"{v:.3g}"
    return f"{v:,}"


def collapse_vs_r0(collapse):
    """{sample: its ONE collapse-vs-R0 row} -- keyed on the row, NOT on the label it carries.

    THE INVARIANT (effective_complexity.py): every non-initial sample has exactly ONE collapse row
    measured against R0. Its `kind` is `protocol` when the sample is in the `empty` group -- with no
    cells, C_eff(R0)/C_eff(empty) IS what the transfers and PCR cost -- and `total` otherwise. The
    two are MUTUALLY EXCLUSIVE per sample (see the if/else that emits them), so this lookup is
    unambiguous, and `kind` still says which one it is, so a caller can keep the label honest.

    Filtering on `kind == "total"` instead -- as this file used to -- silently drops the empty arm:
    no fold-drop annotation on its trajectory, a blank "collapse vs R0" cell in the table. That is
    the exact gap the empty arm exists to close, so it must not be reintroduced. `selection` rows
    are extra, never a substitute; they are rendered by protocol_collapse_section() alone.
    """
    return {r["sample"]: r for r in collapse if r["kind"] in ("total", "protocol")}


def ratio_num(v, nd=1):
    """A collapse ratio, formatted for reading WITHOUT rounding it into a different story.

    Below 1x it is not a collapse at all -- complexity held or ROSE -- and a fixed `,.1f` would print
    0.0239 as `0.0` and 0.294 as `0.3`: a fabricated zero on the one arm most likely to produce a
    sub-1 ratio (an un-bottlenecked `empty` round has a HUGE C_eff, so C_eff(R0)/C_eff(empty) < 1).
    Small ratios therefore keep their significant figures -- the same rule as effective_complexity's
    fold(), which the workflow log already prints, so the report and the log cannot disagree.
    """
    return f"{v:,.{nd}f}" if v >= 1 else f"{v:.3g}"


def ratio_cell(row, nd=1):
    """A collapse ratio cell, carrying its bound marker (`<` / `>`) when it is not a point estimate.

    `is None`, not falsiness: a ratio that is genuinely 0-ish is a MEASUREMENT (complexity rose by
    more than the 4 significant figures the TSV carries), not a missing value, and blanking it would
    hide the very number the empty arm exists to state.
    """
    if not row or row.get("ratio") is None:
        return ""
    return f"{esc(row.get('ratio_bound') or '')}{ratio_num(row['ratio'], nd)}×"


def c_eff_cell(row):
    """C_eff, prefixed `>` when it is only a 95% lower bound (no collisions observed).

    Never render a bound as a point estimate: an un-bottlenecked pool has NO duplicate reads, so
    1/S is unbounded and N(N-1)/6 is all that can honestly be said. See complexity_lib.c_eff().
    """
    if row.get("c_eff") is None:
        return ""
    mark = "&gt; " if row.get("c_eff_is_lower_bound") else ""
    return f"{mark}{row['c_eff']:,.0f}"


# --------------------------------------------------------------------------- panels
def summary_table(samples, summaries, cplx, gcol):
    # DEPTH-DEPENDENT columns are flagged: they grow with sequencing depth, so they describe the RUN,
    # not the LIBRARY, and must not be compared across samples of unequal depth. Same idiom as the
    # fragvec cross report. c_eff / collision_prob are the depth-ROBUST alternatives.
    depth_dep = {"n_species_raw", "n_species_corrected", "n_species_merged"}
    cols = ["reads_in", "dropped_nonacgt", "dropped_lowcomplexity", "dropped_umilen",
            "dropped_total", "reads_umi_ok", "dropped_no_payload", "reads_kept",
            "n_species_raw", "n_species_corrected", "n_species_merged", "n_reads_relabeled",
            "dup_pairs", "collision_prob", "c_eff", "n_over_c_eff"]
    head = "".join(
        f"<th title='DEPTH-DEPENDENT — do not compare across samples'>{esc(c)} ⚠</th>"
        if c in depth_dep else f"<th>{esc(c)}</th>" for c in cols)
    by_name = {r["name"]: r for r in cplx}
    rows = []
    for s in samples:
        d = dict(summaries.get(s["name"], {}))
        d.update({k: v for k, v in by_name.get(s["name"], {}).items() if v is not None})
        cells = []
        for c in cols:
            if c == "c_eff":
                cells.append(f"<td class='num'>{c_eff_cell(by_name.get(s['name'], {}))}</td>")
            elif c == "collision_prob":
                cells.append(f"<td class='num'>{sci(d.get(c))}</td>")
            elif c == "n_over_c_eff":
                cells.append(f"<td class='num'>{fmt(d.get(c), 3)}</td>")
            else:
                cells.append(f"<td class='num'>{fmt(d.get(c))}</td>")
        swatch = (f"<span style='display:inline-block;width:.7rem;height:.7rem;border-radius:2px;"
                  f"margin-right:.4rem;background:{gcol.get(s['group'], GREY)}'></span>")
        rows.append(f"<tr><td>{esc(s['name'])}</td><td>{swatch}{esc(s['group'])}</td>"
                    f"<td class='num'>{s['round']}</td>{''.join(cells)}</tr>")
    return (f"<h2>Filter &amp; correction summary</h2><div style='overflow-x:auto'><table>"
            f"<thead><tr><th>sample</th><th>group</th><th>round</th>{head}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def fig_ceff_trajectory(samples, cplx, collapse, gcol):
    """C_eff vs selection round, one line per group, all starting from the shared R0 library.

    The headline figure of the UMI branch. Log y because the collapse spans orders of magnitude.

    Error bars are C_eff/sqrt(dup_pairs) -- the delta-method standard error of a reciprocal, with S
    treated as quasi-Poisson in its collision count. They are not decoration: two samples can carry
    near-identical C_eff point estimates while one rests on 237 collisions and the other on 3,076,
    a ~3.6x difference in relative precision. A bare point estimate hides exactly that.
    """
    by_name = {r["name"]: r for r in cplx}
    base = next((s for s in samples if s["is_baseline"]), None)
    brow = by_name.get(base["name"]) if base else None
    vs_r0 = collapse_vs_r0(collapse)

    groups = [g for g in dict.fromkeys(s["group"] for s in samples if not s["is_baseline"])]
    fig = go.Figure()
    for g in groups:
        members = sorted((s for s in samples if s["group"] == g and not s["is_baseline"]),
                         key=lambda s: s["round"])
        pts = ([(brow["round"] if brow.get("round") is not None else 0, brow)] if brow else [])
        pts += [(s["round"], by_name[s["name"]]) for s in members if s["name"] in by_name]
        pts = [(r, row) for r, row in pts if row and row.get("c_eff") is not None]
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1]["c_eff"] for p in pts]
        # a lower bound is drawn hollow: it is not a measurement of where the point IS
        lb = [bool(p[1].get("c_eff_is_lower_bound")) for p in pts]
        err = [(p[1]["c_eff"] / math.sqrt(p[1]["dup_pairs"])) if p[1].get("dup_pairs") else 0
               for p in pts]
        names = [p[1]["name"] for p in pts]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers", name=esc(g), legendgroup=g,
            line=dict(color=gcol.get(g, GREY), width=2.2),
            marker=dict(size=10, color=gcol.get(g, GREY),
                        symbol=["circle-open" if b else "circle" for b in lb],
                        line=dict(width=2, color=gcol.get(g, GREY))),
            error_y=dict(type="data", array=err, visible=True, thickness=1.2, width=4,
                         color=gcol.get(g, GREY)),
            customdata=list(zip(names, [p[1].get("dup_pairs") or 0 for p in pts], lb)),
            hovertemplate=("%{customdata[0]}<br>C_eff %{y:,.0f}<br>"
                           "%{customdata[1]:,} duplicate pairs<extra>" + esc(g) + "</extra>"),
        ))
        # The fold-drop, stated at the end of each arm. `vs_r0` is keyed on the collapse-vs-R0 row
        # WHATEVER IT IS CALLED, so the empty arm -- whose only row is `kind: protocol` -- gets its
        # annotation like every other group. This is not a special case: it is the absence of one.
        #
        # A BOUNDED ratio IS annotated, carrying its `<` / `>` marker. It has to be. An un-bottlenecked
        # empty round has NO duplicate reads at all (no cells => nothing is selected away => every
        # molecule is seen once), so its c_eff is only a 95% LOWER bound, so its collapse ratio is a
        # BOUND. Skipping bounded ratios here -- which this line used to do -- blanked the empty arm's
        # annotation in EXACTLY the case the annotation exists for, and the fixture never caught it
        # because it built the empty round as a copy of a real round (which has duplicates, hence no
        # bound). `ratio_cell()` already carries the marker in every table on this page; this line must
        # not disagree with it.
        last = members[-1]["name"] if members else None
        row = vs_r0.get(last)
        if row and row.get("ratio") is not None:
            v = row["ratio"]
            b = esc(row.get("ratio_bound") or "")
            # Below 1x there was no drop: complexity held or ROSE (the shape of an un-bottlenecked
            # empty round, whose line visibly climbs above R0 right here). Print the number with its
            # significant figures and say what it is -- `,.0f` would label that line "0× drop", a
            # fabricated zero with the wrong word attached to it.
            txt = (f"{b}{ratio_num(v, 0)}× drop" if v >= 1
                   else f"{b}{ratio_num(v)}× — complexity ROSE")
            fig.add_annotation(x=xs[-1], y=math.log10(ys[-1]), xref="x", yref="y",
                               text=f"<b>{esc(g)}</b>  {txt}",
                               xanchor="left", xshift=12, showarrow=False,
                               font=dict(size=11, color=gcol.get(g, GREY)))
    if brow and brow.get("c_eff"):
        fig.add_annotation(x=brow.get("round") or 0, y=math.log10(brow["c_eff"]),
                           text=f"initial library {brow['c_eff'] / 1e6:,.0f}M",
                           yshift=16, showarrow=False, font=dict(size=10, color=GREY))
    rounds = sorted({s["round"] for s in samples})
    fig.update_xaxes(title="selection round", tickmode="array", tickvals=rounds,
                     ticktext=[("R0<br>(initial)" if r == 0 else f"R{r}") for r in rounds],
                     range=[min(rounds) - 0.3, max(rounds) + 1.1])
    fig.update_yaxes(type="log", title="C_eff (effective lineage complexity)")
    note = ("<p class='small'>Hollow markers are <b>lower bounds</b> (no duplicate reads observed — "
            "the reciprocal is unbounded and only <code>N(N−1)/6</code> can be claimed). Error bars "
            "are <code>C_eff/√dup_pairs</code>: an estimate resting on 237 collisions is far softer "
            "than one resting on 20,754, and the point estimates alone do not show that.</p>")
    return note + fig_html(style(fig, 520, "Effective complexity by selection round"))


def ceff_table(samples, cplx, collapse, gcol):
    """C_eff with the three companions clarification.md §8.5 requires it never be read without."""
    by_name = {r["name"]: r for r in cplx}
    vs_r0 = collapse_vs_r0(collapse)      # one row per sample, `total` OR `protocol` -- see helper
    order = [s for s in samples if s["name"] in by_name]
    rows = []
    for s in order:
        r = by_name[s["name"]]
        # ONE formatter for every collapse ratio in this report (the table, the protocol section, the
        # trajectory annotation), so a sub-1 ratio cannot render as `0.0×` in one place and 0.294 in
        # another. Do not re-implement it inline.
        ratio = ratio_cell(vs_r0.get(s["name"]))
        top = ("" if r.get("top10_share_of_S") is None
               else f"{100 * r['top10_share_of_S']:.1f}%")
        swatch = (f"<span style='display:inline-block;width:.7rem;height:.7rem;border-radius:2px;"
                  f"margin-right:.4rem;background:{gcol.get(s['group'], GREY)}'></span>")
        rows.append(
            f"<tr><td>{swatch}{esc(s['name'])}</td>"
            f"<td class='num'>{fmt(r.get('n_reads'))}</td>"
            f"<td class='num'>{fmt(r.get('n_distinct'))}</td>"
            f"<td class='num'>{fmt(r.get('n_distinct_rarefied'), 0)}</td>"
            f"<td class='num'>{fmt(r.get('dup_pairs'))}</td>"
            f"<td class='num'>{sci(r.get('collision_prob'))}</td>"
            f"<td class='num'><b>{c_eff_cell(r)}</b></td>"
            f"<td class='num'>{fmt(r.get('n_over_c_eff'), 3)}</td>"
            f"<td class='num'>{fmt(r.get('max_n_i'))}</td>"
            f"<td class='num'>{top}</td>"
            f"<td class='num'>{ratio}</td></tr>")
    m = next((r.get("rarefied_to") for r in cplx if r.get("rarefied_to")), None)
    head = (
        "<tr>"
        "<th>sample</th>"
        "<th title='total payload reads'>N reads</th>"
        "<th title='DEPTH-DEPENDENT — do not compare across samples'>distinct ⚠</th>"
        f"<th title='distinct UMIs rarefied to a common depth of {fmt(m)} reads'>rarefied</th>"
        "<th title='observed duplicate read-pairs — how much evidence C_eff rests on'>dup pairs</th>"
        "<th title='S = Σpᵢ² — the primitive, unbiased at any depth'>S</th>"
        "<th title='1/S — depth-robust; a LOWER BOUND on richness, never a headcount'>C_eff</th>"
        "<th title='&lt;&lt;1 = you sampled a small fraction of the pool'>N/C_eff</th>"
        "<th title='reads on the single most abundant lineage'>max nᵢ</th>"
        "<th title='share of all collisions carried by the top 10 lineages'>top10 %S</th>"
        "<th title='C_eff(R0) / C_eff(sample) — the sample&#39;s ONE collapse-vs-R0 row: kind "
        "`total` for a selection arm, kind `protocol` for the empty arm (same ratio, and with no "
        "cells it measures the transfers + PCR alone)'>collapse vs R0</th>"
        "</tr>")
    return (f"<div style='overflow-x:auto'><table><thead>{head}</thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def fig_depth_complexity(samples, cplx, gcol):
    """Richness at a COMMON depth, beside the raw count it corrects.

    Renders umi_complexity.tsv; computes nothing. Colour carries the GROUP; the raw-vs-rarefied
    contrast is carried by a hatch pattern, so hue is never doing two jobs at once.
    """
    by_name = {r["name"]: r for r in cplx}
    order = [s for s in samples if s["name"] in by_name]
    if not order:
        return "<p class='small'>No complexity table.</p>"
    names = [s["name"] for s in order]
    rar = [by_name[n].get("n_distinct_rarefied") for n in names]
    raw = [by_name[n].get("n_distinct") for n in names]
    colors = [gcol.get(s["group"], GREY) for s in order]
    m = next((r.get("rarefied_to") for r in cplx if r.get("rarefied_to")), None)
    depths = [by_name[n].get("n_reads") or 0 for n in names]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=rar, name=f"distinct UMIs, RAREFIED to {fmt(m)} reads",
                         marker=dict(color=colors)))
    fig.add_trace(go.Bar(x=names, y=raw, name="raw distinct UMIs (DEPTH-DEPENDENT)",
                         marker=dict(color=colors, opacity=0.35,
                                     pattern=dict(shape="/", solidity=0.35))))
    for x in group_dividers(order):
        fig.add_vline(x=x, line_color=GREY, line_width=1, opacity=0.5)
    group_band(fig, order, yshift=6)
    fig.update_layout(barmode="group", yaxis=dict(title="distinct corrected UMIs"))
    lo, hi = min(depths), max(depths)
    note = (
        f"<p class='small'><b>Read depth spans {lo:,}–{hi:,} ({hi / lo:.1f}×) across these samples.</b> "
        f"A raw species count grows with depth, so the hatched bars are not comparable — compare the "
        f"solid ones (rarefied to a common {fmt(m)} reads). <em>If a number gets bigger when you "
        f"sequence more, it is not a property of your library.</em> Every panel below this section is "
        f"a species count and inherits this caveat.</p>"
        f"<p class='small'>⚠ <b>But read the rarefied bars with care too: at this depth they are "
        f"measuring the instrument, not the pool.</b> Rarefaction saturates when the pool is far "
        f"larger than the depth — nearly every read lands on an unseen lineage, so E[distinct] → m "
        f"for <em>any</em> sufficiently large pool. That is why R0 and the deep-collapsed rounds "
        f"look almost equally rich here while their <code>C_eff</code> differs by orders of "
        f"magnitude. Where the two disagree, <code>C_eff</code> is the one still saying something "
        f"about the library.</p>")
    return note + fig_html(style(fig, 480, "Richness at a common depth (depth-corrected)", top=88))


def protocol_collapse_section(cplx, collapse, gcol):
    """The `kind: protocol` rows -- what the plumbing cost, with no cells in the tube.

    THE ONLY PLACE THIS REPORT KNOWS THE WORD `empty`, and even here it reads the collapse `kind`,
    not the group name: no figure is special-cased, because the empty arm is a peer group and needs
    no help to be drawn. ABSENT ENTIRELY when the config declares no empty arm -- there are then no
    `protocol` rows, no `selection` rows, and the report degrades to `total` alone, rendering
    exactly as it did before this section existed. That is the acceptance criterion, not a fallback.
    """
    prot = [r for r in collapse if r["kind"] == "protocol"]
    if not prot:
        return ""
    by_name = {r["name"]: r for r in cplx}
    sel = [r for r in collapse if r["kind"] == "selection"]
    total = {r["sample"]: r for r in collapse if r["kind"] == "total"}
    by_empty = {r["sample"]: r for r in prot}

    def swatch(g):
        return (f"<span style='display:inline-block;width:.7rem;height:.7rem;border-radius:2px;"
                f"margin-right:.4rem;background:{gcol.get(g, GREY)}'></span>")

    prot_rows = "".join(
        f"<tr><td>{swatch(r['group'])}{esc(r['sample'])}</td>"
        f"<td class='num'>{fmt(r.get('round'))}</td>"
        f"<td class='num'>{c_eff_cell(by_name.get(r['sample'], {}))}</td>"
        f"<td class='num'><b>{ratio_cell(r)}</b></td></tr>"
        for r in sorted(prot, key=lambda r: (r.get("round") or 0, r["sample"])))
    out = [
        "<h3>Protocol collapse — the empty round</h3>",
        "<p class='small'>The <code>empty</code> arm went through the same transfers and the same PCR "
        "as a selection arm, with <b>no cells in the tube</b>. Nothing could be selected for, so "
        "whatever complexity it lost relative to R0 is what the <em>protocol alone</em> cost — the "
        "handling and the re-amplification, with no biology in it. That is the number below. It is "
        "not a correction that has been applied to anything: it is measured on its own arm, and the "
        "arms below are the ones it lets you divide it out of.</p>",
        f"<div style='overflow-x:auto'><table><thead><tr><th>empty round</th><th>round</th>"
        f"<th title='the empty round&#39;s own effective complexity'>C_eff</th>"
        f"<th title='C_eff(R0) / C_eff(empty) — the sample&#39;s ONE collapse-vs-R0 row, here "
        f"labelled `protocol` because with no cells it measures the protocol alone'>"
        f"protocol collapse (vs R0)</th></tr></thead><tbody>{prot_rows}</tbody></table></div>",
        "<p class='small'>This is not a separate measurement bolted on beside the figures: <b>the "
        "empty arm's line on the C<sub>eff</sub> trajectory above IS this number, drawn</b>, and its "
        "cell in the <code>collapse vs R0</code> column above IS this number, tabulated. One "
        "collapse row, one label, three renderings of it.</p>",
    ]
    if sel:
        srows = "".join(
            f"<tr><td>{swatch(r['group'])}{esc(r['sample'])}</td>"
            f"<td class='num'>{fmt(r.get('round'))}</td>"
            f"<td class='num'>{ratio_cell(total.get(r['sample']), nd=2)}</td>"
            f"<td class='num'>{ratio_cell(by_empty.get(r['num']), nd=2)}</td>"
            f"<td class='num'><b>{ratio_cell(r, nd=2)}</b></td>"
            f"<td>{esc(r['num'])}</td></tr>"
            for r in sorted(sel, key=lambda r: (r.get("round") or 0, r["sample"])))
        out += [
            f"<div style='overflow-x:auto'><table><thead><tr><th>sample</th><th>round</th>"
            f"<th title='C_eff(R0) / C_eff(sample) — everything that happened'>total</th>"
            f"<th title='C_eff(R0) / C_eff(empty) — the protocol&#39;s share, from the row above'>"
            f"protocol</th>"
            f"<th title='C_eff(empty) / C_eff(sample) — what is left once the protocol is divided "
            f"out'>selection</th><th>empty round used</th></tr></thead>"
            f"<tbody>{srows}</tbody></table></div>",
            "<p class='small'><b>total = protocol × selection</b>, exactly: "
            "<code>C_eff(R0)/C_eff(sample) = [C_eff(R0)/C_eff(empty)] × "
            "[C_eff(empty)/C_eff(sample)]</code>, the empty round's C<sub>eff</sub> cancelling. The "
            "identity is exact by that cancellation, not a fit — though the cells above are rounded "
            "for reading, so multiplying them by hand closes only to the digits shown. The three "
            "columns are three <em>distinct</em> collapse rows, not one number restated three "
            "times. <code>selection</code> is the column with any biology in it: it pairs a sample "
            "to the empty round at the <b>same</b> round — the one that went through the same "
            "number of transfers and PCR cycles — so a sample at a round with no matching empty "
            "round has a <code>total</code> and nothing else. The decomposition is then simply not "
            "available for it, and is not guessed at.</p>",
        ]
    return "\n".join(out)


def _plot_idx(n, cap=2000):
    """Log-spaced indices over [0, n-1] so a huge rank-abundance/Lorenz curve stays light to draw
    without changing its shape (metrics are computed on the full data, only the drawn line thins)."""
    if n <= cap:
        return list(range(n))
    step = math.log(n) / (cap - 1)
    return sorted({0, n - 1} | {min(n - 1, int(math.exp(step * i)) - 1) for i in range(cap)})


def _round_dash(s):
    """Rounds WITHIN a group are told apart by dash, not by hue or by a legend entry — the legend
    names groups (3 entries), not samples (11)."""
    if s["is_baseline"]:
        return "dot"
    return ["solid", "dash", "longdash", "dashdot", "longdashdot"][(s["round"] - 1) % 5]


def fig_rank_abundance(samples, species, gcol):
    fig = go.Figure()
    show = first_of_group(samples)
    for s, first in zip(samples, show):
        df = species[s["name"]].sort("n_reads", descending=True)
        y = df["n_reads"].to_list()
        if not y:
            continue
        idx = _plot_idx(len(y))
        fig.add_trace(go.Scatter(
            x=[i + 1 for i in idx], y=[y[i] for i in idx], mode="lines", name=esc(s["group"]),
            legendgroup=s["group"], showlegend=first,
            line=dict(color=gcol.get(s["group"], GREY), width=1.6, dash=_round_dash(s)),
            hovertemplate="rank %{x}<br>%{y} reads<extra>" + esc(s["name"]) + "</extra>",
        ))
    fig.update_xaxes(type="log", title="UMI species rank")
    fig.update_yaxes(type="log", title="reads")
    return fig_html(style(fig, 460, "Rank–abundance (corrected UMI species)"))


def _d_proportion(counts_desc, frac):
    total = sum(counts_desc)
    if total == 0 or not counts_desc:
        return float("nan")
    cum = 0
    for k, c in enumerate(counts_desc, start=1):
        cum += c
        if cum >= frac * total:
            return k / len(counts_desc)
    return 1.0


def fig_complexity(samples, species, gcol):
    order, d50, d90 = [], [], []
    lorenz = go.Figure()
    show = first_of_group(samples)
    for s, first in zip(samples, show):
        cd = species[s["name"]].sort("n_reads", descending=True)["n_reads"].to_list()
        if not cd:
            continue
        order.append(s)
        d50.append(100 * _d_proportion(cd, 0.5))
        d90.append(100 * _d_proportion(cd, 0.9))
        n = len(cd)
        total = sum(cd)
        cum, xs_full, ys_full = 0, [0.0], [0.0]
        for i, c in enumerate(cd, start=1):
            cum += c
            xs_full.append(i / n)
            ys_full.append(cum / total)
        idx = [0] + [i + 1 for i in _plot_idx(n)]  # keep the (0,0) origin + thinned curve
        lorenz.add_trace(go.Scatter(
            x=[xs_full[i] for i in idx], y=[ys_full[i] for i in idx], mode="lines",
            name=esc(s["group"]), legendgroup=s["group"], showlegend=first,
            line=dict(color=gcol.get(s["group"], GREY), width=1.4, dash=_round_dash(s)),
            hovertemplate="%{x:.1%} of species<br>%{y:.1%} of reads<extra>"
                          + esc(s["name"]) + "</extra>",
        ))
    lorenz.add_shape(type="line", x0=0, y0=0.5, x1=1, y1=0.5, line=dict(color=GREY, dash="dash", width=1))
    lorenz.add_shape(type="line", x0=0, y0=0.9, x1=1, y1=0.9, line=dict(color=GREY, dash="dash", width=1))
    lorenz.update_xaxes(title="cumulative fraction of species (ranked by abundance)")
    lorenz.update_yaxes(title="cumulative fraction of reads")

    names = [s["name"] for s in order]
    colors = [gcol.get(s["group"], GREY) for s in order]
    bars = go.Figure()
    bars.add_trace(go.Bar(x=names, y=d50, name="D50", marker=dict(color=colors)))
    bars.add_trace(go.Bar(x=names, y=d90, name="D90",
                          marker=dict(color=colors, opacity=0.4,
                                      pattern=dict(shape="/", solidity=0.35))))
    for x in group_dividers(order):
        bars.add_vline(x=x, line_color=GREY, line_width=1, opacity=0.5)
    group_band(bars, order, yshift=6)
    bars.update_yaxes(title="% of distinct species reaching 50% / 90% of reads")
    bars.update_layout(barmode="group")
    return (fig_html(style(bars, 440, "Complexity: D50 / D90 (lower = more concentrated)", top=88))
            + fig_html(style(lorenz, 460, "Cumulative abundance (Lorenz)")))


def fig_upset(group, gsamples, species, color, top_n=24):
    long = pl.concat([
        species[nm].select(pl.col("umi").alias("fused_umi")).with_columns(pl.lit(nm).alias("sample"))
        for nm in gsamples if species[nm].height
    ]) if any(species[nm].height for nm in gsamples) else None
    if long is None or long.height == 0:
        return f"<p class='small'>No UMI species for group {esc(group)}.</p>"
    bitmap = pl.DataFrame({"sample": gsamples, "bitval": [1 << i for i in range(len(gsamples))]})
    patt = long.join(bitmap, on="sample").group_by("fused_umi").agg(pl.col("bitval").sum().alias("mask"))
    counts = patt.group_by("mask").agg(pl.len().alias("size")).sort("size", descending=True)
    masks = []
    for mask, size in counts.head(top_n).iter_rows():
        members = [j for j in range(len(gsamples)) if mask & (1 << j)]
        masks.append((members, size))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.04)
    xs = list(range(len(masks)))
    fig.add_trace(go.Bar(x=xs, y=[m[1] for m in masks], marker_color=color,
                         hovertemplate="%{y} species<extra></extra>"), row=1, col=1)
    for xi, (members, _sz) in enumerate(masks):
        # faint dots for all rows, solid for members, connected
        fig.add_trace(go.Scatter(x=[xi] * len(gsamples), y=list(range(len(gsamples))), mode="markers",
                                 marker=dict(color="#e6e6e6", size=9), showlegend=False,
                                 hoverinfo="skip"), row=2, col=1)
        if members:
            fig.add_trace(go.Scatter(x=[xi] * len(members), y=members, mode="markers+lines",
                                     marker=dict(color=color, size=10),
                                     line=dict(color=color, width=2),
                                     showlegend=False, hoverinfo="skip"), row=2, col=1)
    fig.update_yaxes(title="species", row=1, col=1)
    fig.update_yaxes(tickmode="array", tickvals=list(range(len(gsamples))),
                     ticktext=[esc(x) for x in gsamples], row=2, col=1, autorange="reversed")
    fig.update_xaxes(showticklabels=False, row=2, col=1)
    fig.update_layout(showlegend=False)
    return fig_html(style(fig, 460,
                          f"Shared corrected-UMI species across {esc(group)} "
                          f"(top {len(masks)} intersections)"))


def _read_signatures(frags_path):
    """read_acc -> the read's decoded fragment vector, EXACTLY as the tool emits it.

    Anchors and `_rc` are KEPT, and both carry information. Reads are already oriented (pychopper),
    so `F1` vs `F1_rc` means the fragment was LIGATED in reverse — a genuinely different species, not
    a read-direction artifact — and the backbone anchors are structural features like any other
    fragment. Same convention as build_featurecounts.py ("counted EXACTLY as the tool emits them").
    Reads with no payload are already dropped upstream (umi_correct STEP 1b), so nothing can score as
    trivially pure on an anchors-only signature. Sorted, so the multiset (incl. duplicate detections)
    is order-invariant.
    """
    fr = pl.read_parquet(frags_path, columns=["read_acc", "fragment_id"])
    return fr.group_by("read_acc").agg(pl.col("fragment_id").sort().str.join("|").alias("sig"))


BINS = [(2, 2), (3, 4), (5, 9), (10, 29), (30, 99), (100, 10 ** 9)]


def _bin_label(lo, hi):
    return f"{lo}" if lo == hi else (f"{lo}+" if hi > 10 ** 8 else f"{lo}-{hi}")


def fig_purity(samples, gcol):
    """Purity per GROUP, not pooled.

    This used to concat every sample into one aggregate, which is precisely the comparison the
    report exists to make impossible: a pooled purity curve cannot show that one arm's UMI groups
    are dirtier than another's, and it silently lets the deepest sample dominate the mixture.
    """
    per_group = {}
    for s in samples:
        if not s["fused"].exists() or not s["frags"] or not Path(s["frags"]).exists():
            continue
        try:
            fused = pl.read_parquet(s["fused"], columns=["read_acc", "fused_seq_corrected"])
            sig = _read_signatures(s["frags"])
            joined = fused.join(sig, on="read_acc", how="inner")
            if joined.height == 0:
                continue
            grp = (joined.group_by("fused_seq_corrected")
                   .agg(pl.len().alias("size"), pl.col("sig").value_counts().alias("_vc")))
            grp = grp.with_columns(
                pl.col("_vc").list.eval(pl.element().struct.field("count").max()).list.first().alias("dom")
            ).with_columns((pl.col("dom") / pl.col("size")).alias("purity")).drop("_vc")
            per_group.setdefault(s["group"], []).append(grp.select("size", "purity"))
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f"[purity] skip {s['name']}: {exc}\n")
    if not per_group:
        return "<p class='small'>Purity QC skipped (decode fragments not available).</p>"

    fig = go.Figure()
    drawn = False
    for g, parts in per_group.items():
        allg = pl.concat(parts).filter(pl.col("size") >= 2)
        if allg.height == 0:
            continue
        labels, frac_pure, ngroups = [], [], []
        for lo, hi in BINS:
            sub = allg.filter((pl.col("size") >= lo) & (pl.col("size") <= hi))
            if sub.height == 0:
                continue
            labels.append(_bin_label(lo, hi))
            frac_pure.append(100 * sub.filter(pl.col("purity") >= 0.999).height / sub.height)
            ngroups.append(sub.height)
        if not labels:
            continue
        drawn = True
        fig.add_trace(go.Bar(
            x=labels, y=frac_pure, name=esc(g), legendgroup=g,
            marker_color=gcol.get(g, GREY), customdata=ngroups,
            hovertemplate="%{y:.1f}% pure<br>%{customdata:,} groups<extra>" + esc(g) + "</extra>"))
    if not drawn:
        return "<p class='small'>No multi-read UMI groups for the purity QC.</p>"
    fig.update_xaxes(title="UMI group size (reads)", categoryorder="array",
                     categoryarray=[_bin_label(lo, hi) for lo, hi in BINS])
    fig.update_yaxes(title="% of UMI groups with a single ligand signature", range=[0, 105])
    fig.update_layout(barmode="group")
    return fig_html(style(fig, 440, "Validity: ligand-signature purity vs UMI group size, by group"))


def fig_collision(samples, species, summaries, gcol):
    order, observed, expected = [], [], []
    for s in samples:
        d = summaries.get(s["name"], {})
        merged = d.get("n_species_merged")
        n_raw = d.get("n_species_raw")
        if merged is None or not n_raw:
            continue
        sp = species[s["name"]]
        if sp.height == 0:
            continue
        L = float(sp["umi"].str.len_chars().mean() or 0)
        nbhd = 7 * L  # ~ Levenshtein-1 neighbourhood size (subs 3L + indels ~4L)
        exp = (n_raw * (n_raw - 1) / 2.0) * nbhd / (4.0 ** L) if L else 0.0
        order.append(s)
        observed.append(int(merged))
        expected.append(exp)
    if not order:
        return "<p class='small'>Collision null model unavailable.</p>"
    names = [s["name"] for s in order]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=observed, name="observed ed≤1 merges",
                         marker=dict(color=[gcol.get(s["group"], GREY) for s in order])))
    fig.add_trace(go.Bar(x=names, y=expected, name="expected by chance",
                         marker=dict(color=GREY, opacity=0.45,
                                     pattern=dict(shape="/", solidity=0.35))))
    for x in group_dividers(order):
        fig.add_vline(x=x, line_color=GREY, line_width=1, opacity=0.5)
    group_band(fig, order, yshift=6)
    fig.update_yaxes(title="UMI merges", type="log")
    fig.update_layout(barmode="group")
    return fig_html(style(fig, 420, "Collision null model: observed corrections vs chance", top=88))


# --------------------------------------------------------------------------- assemble
def main():
    if not BASE or not BASE.exists():
        sys.stderr.write(f"[FATAL] L2F_BASE not found: {BASE}\n")
        return 1
    meta = load_meta(SAMPLES_TSV)
    samples = discover(meta)
    if not samples:
        OUT.write_text("<!doctype html><meta charset='utf-8'><p>No UMI samples discovered.</p>")
        print(f"[umi report] no samples -> {OUT}")
        return 0
    species = {s["name"]: load_species(s) for s in samples}
    summaries = {s["name"]: load_summary(s) for s in samples}
    cplx = load_table(COMPLEXITY_TSV)
    collapse = load_table(COLLAPSE_TSV)
    gcol = group_color(samples)
    if not cplx:
        sys.stderr.write("[warn] no complexity table (L2F_COMPLEXITY); C_eff panels will be empty\n")

    H = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>UMI characterisation — cross-sample</title>",
        _STYLE,
        f"<script>{get_plotlyjs()}</script>",
        "</head><body>",
        "<h1>UMI characterisation — cross-sample</h1>",
        "<p class='small'>Descriptive QC over corrected UMI species (mode <code>correct</code>). "
        "This is a workflow output — no biological conclusions are drawn. Every figure is coloured "
        "by the config <code>group</code>; the initial library is the shared R0 baseline both arms "
        "start from, and is drawn grey and dashed throughout.</p>",
        color_key_html(gcol, samples),
        summary_table(samples, summaries, cplx, gcol),

        "<h2>Effective complexity (C<sub>eff</sub>)</h2>",
        "<p class='small'><code>C_eff = 1 / P(two random distinct reads came from the same original "
        "molecule)</code> — the inverse-Simpson complexity of the molecule pool. Unlike a distinct-UMI "
        "count, it does not inflate with sequencing depth, so it is the one complexity number "
        "comparable across these samples (whose depth spans ~6×).</p>",
        "<p class='small'><b>C<sub>eff</sub> is a lower bound on richness, never a headcount.</b> "
        "By Cauchy–Schwarz it equals the true number of lineages only for a perfectly even pool, and "
        "sits far below it for a skewed one. Two consequences worth stating plainly: (1) "
        "<code>C_eff</code> here <em>exceeds</em> the observed distinct-UMI count in every sample — "
        "no contradiction, since the bound is against the <em>true</em> richness, not the "
        "<em>observed</em> one; it simply says the pool is nowhere near exhausted even at R5. (2) A "
        "<em>fall</em> in <code>C_eff</code> means the pool concentrated, but that can be lineages "
        "being <b>lost</b> or the survivors becoming <b>uneven</b>, and <code>C_eff</code> alone "
        "cannot tell you which — so read it next to <code>dup pairs</code>, <code>max nᵢ</code> and "
        "<code>top10 %S</code> below. Lost lineages do not create high-count UMIs; PCR skew and "
        "clonal expansion do.</p>",
        fig_ceff_trajectory(samples, cplx, collapse, gcol),
        ceff_table(samples, cplx, collapse, gcol),
        fig_depth_complexity(samples, cplx, gcol),
        # "" when the config declares no `empty` arm -- the join below drops it, and the report
        # renders exactly as it did before this subsection existed.
        protocol_collapse_section(cplx, collapse, gcol),

        "<h2>Abundance</h2>",
        "<p class='small'>⚠ Both panels below are species counts and are therefore <b>depth-sensitive</b>: "
        "the rank-abundance curve's length is the species count, and D50/D90 are proportions over that "
        "same depth-dependent denominator. Read them for shape, not for cross-sample magnitude. Within "
        "a group, rounds are told apart by line dash.</p>",
        fig_rank_abundance(samples, species, gcol),
        fig_complexity(samples, species, gcol),
        "<h2>Species overlap</h2>",
        "<p class='small'>⚠ Shared-species counts grow with the depth of <b>both</b> samples, so an UpSet "
        "built on unequal depths overstates the overlap of the deeper pair. Rarefaction is not meaningful "
        "here (it would change set membership), so treat these intersections as qualitative.</p>",
    ]
    # every non-baseline group gets an UpSet panel -- `empty` included, with no branch for it
    groups = list(dict.fromkeys(s["group"] for s in samples if not s["is_baseline"]))
    base_name = next((s["name"] for s in samples if s["is_baseline"]), None)
    for g in groups:
        gs = ([base_name] if base_name else []) + [s["name"] for s in samples
                                                   if s["group"] == g and not s["is_baseline"]]
        H.append(fig_upset(g, gs, species, gcol.get(g, GREY)))
    H += [
        "<h2>Validity QC</h2>",
        fig_purity(samples, gcol),
        fig_collision(samples, species, summaries, gcol),
        "</body></html>",
    ]
    OUT.write_text("\n".join(x for x in H if x))
    print(f"[umi report] {len(samples)} samples, {len(cplx)} complexity rows -> {OUT}")
    return 0


_STYLE = """<style>
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:2rem auto;max-width:1100px;padding:0 1rem;color:#1a1a1a}
h1{font-size:1.6rem}h2{font-size:1.2rem;margin-top:2rem;border-bottom:1px solid #eee;padding-bottom:.3rem}
table{border-collapse:collapse;font-size:.85rem}th,td{border:1px solid #e2e2e2;padding:.25rem .5rem;text-align:left}
th{background:#f6f8fa}td.num{text-align:right;font-variant-numeric:tabular-nums}
th[title]{cursor:help;text-decoration:underline dotted 1px}
.small{color:#666;font-size:.85rem}code{background:#f0f2f4;padding:.05rem .3rem;border-radius:3px}
@media(prefers-color-scheme:dark){body{background:#15181c;color:#e6e6e6}th{background:#20242a}
th,td{border-color:#333}h2{border-color:#333}code{background:#20242a}.small{color:#9aa4ae}}
</style>"""


if __name__ == "__main__":
    raise SystemExit(main())
