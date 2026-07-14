#!/usr/bin/env python
"""Cross-sample (comparison) feature-count report over many decoded samples.

The single-sample report profiles one library; this one compares ALL samples decoded under a base
directory -- here the A549 and MDA selection series (R1..R5 each) under ligand2frags_new2/. It reads
each sample's per-read feature counts and lines them up feature-by-feature so you can read, per
ligand fragment, how abundance shifts ACROSS SELECTION ROUNDS (within a cell line) and BETWEEN CELL
LINES (at a matched round).

Everything is DATA-DRIVEN (see ligand_vocab.py): the sample list is discovered from the base dir,
the cell-line / round of each sample is parsed from its name, and the construct vocabulary + classes
+ colours are derived from the decode itself. Nothing is hard-coded, so it re-derives for any panel.

    Reads (per sample subdir <base>/<S>/):
        <S>.featurecount.parquet, <S>.featurecount.summary.tsv, <S>.umis.parquet, <S>.ligateddna.log
    Writes:
        <base>/cross_sample_featurecount_report.html   (interactive, plotly.js inlined)

    L2F_BASE default = ligand2frags_new2

Run: pixi run python workflow/build_featurecount_report_cross.py
"""
import math
import os
import re
from pathlib import Path

import plotly.graph_objects as go
import polars as pl
from plotly.offline import get_plotlyjs

from ligand_vocab import (
    anchor_set, class_colors, class_order, parse_feature, parse_log, read_expected, GREY, PALETTE,
)
# The GROUP layer is shared with the umi/fragvec cross reports so a cell line is the same colour in
# all three. ligand_vocab's GREY/PALETTE stay: they colour fragment CLASSES, a different axis.
from report_lib import (
    BASELINE_GREY, BASELINE_GROUP, color_key_html, esc, fig_html, group_color, load_meta,
)
from report_lib import group_band as _group_band
from report_lib import group_dividers as _group_dividers

ROOT = Path(__file__).resolve().parents[1]
BASE = Path(os.environ.get("L2F_BASE", ROOT / "ligand2frags_new2"))
if not BASE.is_absolute():
    BASE = ROOT / BASE
# Optional pre-selection library, added as a shared R0 baseline column and used as the
# fold-change denominator for enrichment. Set L2F_R0_STEM="" to disable.
_r0 = os.environ.get("L2F_R0_STEM", str(ROOT / "ligand2frags_new2" / "2026initial" / "2026initial_library"))
R0_STEM = (Path(_r0) if _r0 else None)
if R0_STEM is not None and not R0_STEM.is_absolute():
    R0_STEM = ROOT / R0_STEM
OUT = BASE / "cross_sample_featurecount_report.html"

# Optional per-sample metadata written by Snakemake from the config `samples` table
# (TSV: name<TAB>group<TAB>round<TAB>idx<TAB>role). When present it is authoritative: `group` becomes
# the grouping/ordering axis and `round`/`idx` order samples within and across groups. Absent
# (e.g. a standalone run), the script falls back to parsing group+round from the sample name.
#
# The metadata now carries the initial library's row too, so the baseline's group comes from the
# config like every other sample's. It used to be hardcoded here as "library" while the umi and
# fragvec reports hardcoded "initial" -- three reports, three names for one sample, which only
# became visible once they started sharing a colour key.
_meta_path = os.environ.get("L2F_SAMPLES", "")
SAMPLE_META = load_meta(_meta_path)

RC = r"_rc$"
EPS = 1e-4
MAXBUCKET = 8                # payload-fragments-per-read tail bucket ("8+"); tail beyond is negligible
COVBINS = 25                 # coverage-fraction histogram bins over [0, 1]


def fq_lengths(path):
    """read_acc -> sequence length from a FASTQ (used as the post-trim read length denominator)."""
    data = Path(path).read_bytes()
    lines = data.split(b"\n")
    accs, lens = [], []
    for i in range(0, len(lines) - 1, 4):
        head = lines[i]
        if not head.startswith(b"@"):
            continue
        accs.append(head[1:].split(None, 1)[0].decode())
        lens.append(len(lines[i + 1]))
    return pl.DataFrame({"read_acc": accs, "fqlen": lens})


# ------------------------------------------------------------------------ discover + load
def parse_sample_name(name):
    """(cell_line, round) parsed from a sample name like 'A549_R3'; round None if absent."""
    m = re.match(r"(?P<line>.+?)[_-]?R(?P<round>\d+)$", name)
    if m:
        return m.group("line").rstrip("_-"), int(m.group("round"))
    return name, None


def sample_group_round(name):
    """(group, round, idx) for a sample — config metadata when provided, else name-parsed."""
    if name in SAMPLE_META:
        m = SAMPLE_META[name]
        return m["group"], m["round"], m["idx"]
    line, rnd = parse_sample_name(name)
    return line, rnd, 10**9


def baseline_group():
    """The initial library's config `group` — from the metadata when it is there, else the shared
    BASELINE_GROUP contract that report_lib.group_color() keys the grey off."""
    if R0_STEM is not None and R0_STEM.name in SAMPLE_META:
        return SAMPLE_META[R0_STEM.name]["group"]
    return BASELINE_GROUP


def discover():
    """Every <base>/<S>/ subdir that has a feature-count summary, as ordered sample dicts.
    Samples are grouped by `group` (config, else name-parsed); within a group they are ordered
    by round then config order, and the groups themselves appear in config order (first seen).
    The grouping is carried on the `line` key that every figure already orders/separates by."""
    samples = []
    for d in sorted(BASE.glob("*/")):
        S = d.name
        summ = d / f"{S}.featurecount.summary.tsv"
        fc = d / f"{S}.featurecount.parquet"
        if summ.exists() and fc.exists():
            grp, rnd, idx = sample_group_round(S)
            samples.append(dict(name=S, dir=d, fc=fc, summ=summ,
                                frags=d / f"{S}.fragments.parquet",
                                fused_fq=d / f"{S}.fused_reads.fq",
                                umis=d / f"{S}.umis.parquet", log=d / f"{S}.ligateddna.log",
                                line=grp, round=rnd, idx=idx, is_baseline=False))
    grank = {}                                    # group -> its first config index (order key)
    for s in samples:
        grank[s["line"]] = min(grank.get(s["line"], 10**9), s["idx"])
    samples.sort(key=lambda s: (grank[s["line"]],
                                s["round"] if s["round"] is not None else 1e9,
                                s["idx"], s["name"]))
    return samples


def load_baseline():
    """The optional R0 baseline as a sample dict (round 0, shared across cell lines), or None."""
    if R0_STEM is None:
        return None
    fc = R0_STEM.parent / f"{R0_STEM.name}.featurecount.parquet"
    summ = R0_STEM.parent / f"{R0_STEM.name}.featurecount.summary.tsv"
    if not (fc.exists() and summ.exists()):
        return None
    return dict(name=R0_STEM.name, dir=R0_STEM.parent, fc=fc, summ=summ,
                frags=R0_STEM.parent / f"{R0_STEM.name}.fragments.parquet",
                fused_fq=R0_STEM.parent / f"{R0_STEM.name}.fused_reads.fq",
                umis=R0_STEM.parent / f"{R0_STEM.name}.umis.parquet",
                log=R0_STEM.parent / f"{R0_STEM.name}.ligateddna.log",
                line=baseline_group(), round=0, idx=-1, is_baseline=True)


def load(samples):
    # Shared vocabulary: expected CSV from the first sample's log; anchors from it / a umis parquet.
    prov0 = parse_log(samples[0]["log"]) if samples else {}
    expected = read_expected(prov0.get("expected"))
    anchors = anchor_set(samples[0]["umis"] if samples else None, expected)

    anc_ids = list(set(anchors) | {f"{a}_rc" for a in anchors})
    per = {}          # name -> {base -> mean_copies}   (strand-collapsed)
    per_fwd = {}      # name -> {base -> mean_copies}   (forward orientation only)
    per_rc = {}       # name -> {base -> mean_copies}   (_rc orientation only)
    totals = {}       # name -> {base -> total calls}
    meta = {}         # name -> summary dict
    ppr = {}          # name -> {payload-frags-in-read bucket -> fraction of reads}
    edh = {}          # name -> {edit_distance -> fraction of payload calls}
    cov = {}          # name -> {coverage bin -> fraction of reads}  (None if no fused_reads.fq)
    all_bases = set()
    for s in samples:
        summ = pl.read_csv(s["summ"], separator="\t").to_dicts()[0]
        meta[s["name"]] = summ
        n_reads = max(int(summ["n_reads"]), 1)
        fc_lf = pl.scan_parquet(s["fc"])
        # per-orientation totals (keep _rc separate); collapse to base for `per`/`totals`.
        agg = fc_lf.group_by("fragment_id").agg(total=pl.col("count").sum()).collect()
        m, t, mf, mr = {}, {}, {}, {}
        for r in agg.iter_rows(named=True):
            fid = r["fragment_id"]
            base = fid[:-3] if fid.endswith("_rc") else fid
            if base in anchors:
                continue
            all_bases.add(base)
            m[base] = m.get(base, 0.0) + r["total"] / n_reads
            t[base] = t.get(base, 0) + int(r["total"])
            (mr if fid.endswith("_rc") else mf)[base] = r["total"] / n_reads
        per[s["name"]] = m
        per_fwd[s["name"]] = mf
        per_rc[s["name"]] = mr
        totals[s["name"]] = t

        # payload fragment copies per read (0 = amplification-loop-only reads), as a fraction
        pr = (fc_lf.filter(~pl.col("fragment_id").is_in(anc_ids))
              .group_by("read_acc").agg(nfrag=pl.col("count").sum())
              .with_columns(bucket=pl.min_horizontal("nfrag", pl.lit(MAXBUCKET)))
              .group_by("bucket").agg(reads=pl.len()).collect())
        cnt = {int(r["bucket"]): int(r["reads"]) for r in pr.iter_rows(named=True)}
        cnt[0] = n_reads - int(summ["n_reads_with_payload"])
        ppr[s["name"]] = {b: cnt.get(b, 0) / n_reads for b in range(MAXBUCKET + 1)}

        # edit-distance distribution of payload calls, as a fraction
        if s["frags"].exists():
            ed = (pl.scan_parquet(s["frags"]).filter(~pl.col("fragment_id").is_in(anc_ids))
                  .group_by("edit_distance").agg(n=pl.len()).collect())
            ec = {int(r["edit_distance"]): int(r["n"]) for r in ed.iter_rows(named=True)}
            tot = sum(ec.values()) or 1
            edh[s["name"]] = {e: c / tot for e, c in ec.items()}
        else:
            edh[s["name"]] = {}

        # coverage: bases covered by ANY expected fragment ÷ post-trim read length
        # (fused_reads.fq). Distribution over reads, per sample; skipped if the fq is absent.
        fq = s.get("fused_fq")
        if s["frags"].exists() and fq is not None and Path(fq).exists():
            covered = (pl.scan_parquet(s["frags"])
                       .group_by("read_acc")
                       .agg(covered=(pl.col("stop") - pl.col("start")).sum())
                       .collect())
            cj = (covered.join(fq_lengths(fq), on="read_acc", how="inner")
                  .with_columns(cov=(pl.col("covered") / pl.col("fqlen")).clip(0.0, 1.0))
                  .with_columns(bin=(pl.col("cov") * COVBINS).floor()
                                .clip(0, COVBINS - 1).cast(pl.Int64)))
            hist = cj.group_by("bin").agg(reads=pl.len())
            n_cov = cj.height or 1
            cov[s["name"]] = {int(r["bin"]): r["reads"] / n_cov for r in hist.iter_rows(named=True)}
            cov[s["name"]]["_median"] = float(cj["cov"].median())
        else:
            cov[s["name"]] = None

    order = class_order(all_bases, expected, anchors)
    colors = class_colors(order)

    def sortkey(b):
        cls, length, _ = parse_feature(b)
        ci = order.index(cls) if cls in order else 99
        return (ci, length if length is not None else -1, b)

    bases = sorted(all_bases, key=sortkey)
    baseline = next((s["name"] for s in samples if s.get("is_baseline")), None)
    return dict(samples=samples, per=per, per_fwd=per_fwd, per_rc=per_rc, totals=totals, meta=meta,
                bases=bases, ppr=ppr, edh=edh, cov=cov,
                order=order, colors=colors, expected=expected, anchors=anchors, baseline=baseline)


def feat_class(b):
    return parse_feature(b)[0]


# ------------------------------------------------------------------------ plotly helpers
def style(fig, height, title=None):
    fig.update_layout(
        template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=GREY, size=12), height=height,
        margin=dict(l=10, r=10, t=40 if title else 12, b=12),
        title=(dict(text=title, x=0, xref="paper", font=dict(size=15)) if title else None),
        legend=dict(font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(font_size=12),
    )
    grid = dict(gridcolor="rgba(139,148,158,0.18)", zerolinecolor="rgba(139,148,158,0.35)",
                linecolor="rgba(139,148,158,0.4)")
    fig.update_xaxes(**grid)
    fig.update_yaxes(**grid)
    return fig


# This report carries the group on `line`, not `group`; the shared helpers take a `key` for exactly
# that. Thin adapters keep the ~8 existing call sites untouched.
def line_boundaries(samples):
    """x indices where the group changes (for a vertical separator)."""
    return _group_dividers(samples, key="line")


def group_band(fig, samples, yshift):
    """One group name centred above each contiguous block of that group's columns."""
    return _group_band(fig, samples, yshift, key="line")


def col_label(s):
    """Full label (group + round) — used in hovers and the line-plot legends."""
    if s.get("is_baseline"):
        return "R0 library"
    return f'{s["line"]} R{s["round"]}' if s["round"] is not None else s["name"]


def col_short(s):
    """Compact per-column tick label. Columns are ordered group-first and the group is named
    once by the group band, so a column needs only its round (or the sample name when it has
    no round) — no per-column group prefix to repeat."""
    if s.get("is_baseline"):
        return "R0"
    return f'R{s["round"]}' if s["round"] is not None else s["name"]


# ------------------------------------------------------------------------ figures
def fig_heatmap(d):
    """Two heatmaps side by side sharing the feature rows: forward orientation | _rc orientation."""
    from plotly.subplots import make_subplots
    samples = d["samples"]
    bases = d["bases"]
    xs = [col_short(s) for s in samples]      # per-column tick = round only; group named by band
    xpos = list(range(len(samples)))          # numeric x so duplicate labels never merge columns
    ny = list(range(len(bases)))

    def panel(per_key):
        z, hov = [], []
        for b in bases:
            rz, rh = [], []
            for s in samples:
                mc = d[per_key][s["name"]].get(b, 0.0)
                rz.append(math.log10(mc) if mc > 0 else None)
                rh.append(f"{col_label(s)}<br><b>{b}</b><br>mean copies/read {mc:.4f}")
            z.append(rz); hov.append(rh)
        return z, hov

    zf, hf = panel("per_fwd")
    zr, hr = panel("per_rc")
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.03,
                        column_widths=[0.62, 0.38],
                        subplot_titles=["forward orientation", "reverse-complement (_rc)"])
    fig.add_trace(go.Heatmap(z=zf, x=xpos, y=ny, text=hf, hoverinfo="text", coloraxis="coloraxis",
                             hoverongaps=False), row=1, col=1)
    fig.add_trace(go.Heatmap(z=zr, x=xpos, y=ny, text=hr, hoverinfo="text", coloraxis="coloraxis",
                             hoverongaps=False), row=1, col=2)
    yt = [f'<span style="color:{d["colors"].get(feat_class(b), GREY)}">{esc(b)}</span>' for b in bases]
    fig.update_yaxes(tickmode="array", tickvals=ny, ticktext=yt, tickfont=dict(size=8),
                     autorange="reversed", row=1, col=1)
    classes = [feat_class(b) for b in bases]
    for col in (1, 2):
        for i in range(1, len(classes)):
            if classes[i] != classes[i - 1]:
                fig.add_hline(y=i - 0.5, line_color="rgba(139,148,158,0.5)", line_width=1,
                              row=1, col=col)
        for x in line_boundaries(samples):
            fig.add_vline(x=x, line_color=GREY, line_width=2, row=1, col=col)
        fig.update_xaxes(tickmode="array", tickvals=xpos, ticktext=xs,
                         tickangle=-40, tickfont=dict(size=10), side="top", row=1, col=col)
    # group-name band above the forward panel's per-column round labels (the grouping layer)
    group_band(fig, samples, yshift=34)
    # lift the orientation subplot titles clear above the group band (topmost layer)
    fig.update_annotations(yshift=96, selector={"text": "forward orientation"})
    fig.update_annotations(yshift=96, selector={"text": "reverse-complement (_rc)"})
    style(fig, height=max(520, 15 * len(bases) + 130))
    # Colour stays log-transformed, but the colorbar ticks show REAL copies/read at each decade
    # (0.00001, 0.001, 0.1, 1 ...) so no "log10" mental step is needed; hence no colorbar title.
    vals = [v for pk in ("per_fwd", "per_rc") for m in d[pk].values() for v in m.values() if v > 0]
    lo = math.floor(math.log10(min(vals))) if vals else -5
    hi = math.ceil(math.log10(max(vals))) if vals else 0
    tickvals = list(range(lo, hi + 1))

    def fmt(v):
        val = 10.0 ** v
        return f"{val:.0f}" if v >= 0 else f"{val:.{-v}f}"

    fig.update_layout(margin=dict(t=152),
                      coloraxis=dict(colorscale="Viridis", cmin=lo, cmax=hi,
                                     colorbar=dict(tickvals=tickvals,
                                                   ticktext=[fmt(v) for v in tickvals],
                                                   thickness=12, len=0.5)))
    return fig_html(fig)


def fig_class_composition(d):
    samples = d["samples"]
    xfull = [col_label(s) for s in samples]   # full label for hover
    xshort = [col_short(s) for s in samples]  # per-column tick = round only; group named by band
    xpos = list(range(len(samples)))          # numeric x so duplicate labels never merge bars
    fig = go.Figure()
    for c in d["order"]:
        ys = []
        for s in samples:
            tot = sum(v for b, v in d["totals"][s["name"]].items() if feat_class(b) == c)
            all_tot = sum(d["totals"][s["name"]].values()) or 1
            ys.append(100 * tot / all_tot)
        fig.add_trace(go.Bar(x=xpos, y=ys, name=c, marker_color=d["colors"][c], customdata=xfull,
                             hovertemplate="%{customdata}<br>" + c
                             + " %{y:.1f}% of payload<extra></extra>"))
    fig.update_layout(barmode="stack")
    for x in line_boundaries(samples):
        fig.add_vline(x=x, line_color=GREY, line_width=2)
    group_band(fig, samples, yshift=16)
    fig.update_xaxes(tickmode="array", tickvals=xpos, ticktext=xshort, tickangle=-40)
    fig.update_yaxes(title="% of payload calls", range=[0, 100])
    style(fig, height=420)
    fig.update_layout(margin=dict(t=88))      # room for the rotated group band above the bars
    return fig_html(fig)


def by_line(samples):
    """Group the real (non-baseline) samples by group, each ordered by round."""
    lines = {}
    for s in samples:
        if s.get("is_baseline"):
            continue
        lines.setdefault(s["line"], []).append(s)
    for v in lines.values():
        v.sort(key=lambda s: (s["round"] if s["round"] is not None else 1e9))
    return lines


def fig_enrichment(d):
    """Per-feature log2 fold-change of mean copies/read, ONE COLUMN PER SAMPLE (samples are NOT
    aggregated), columns grouped by `group`. Denominator is the shared R0 library when present,
    else each group's own first round (without a baseline, a group's first-round sample has no
    column). Positive (enriched) = red, negative (depleted) = blue."""
    base_name = d.get("baseline")
    lines = by_line(d["samples"])
    first_of = {ln: ss[0] for ln, ss in lines.items()}     # each group's first-round sample

    cols = []                                              # (sample, denom_name, denom_label)
    for s in d["samples"]:
        if s.get("is_baseline"):
            continue
        if base_name is not None:
            cols.append((s, base_name, "R0"))
        else:
            first = first_of[s["line"]]
            if first["name"] != s["name"]:
                cols.append((s, first["name"], col_label(first)))
    if not cols:
        return ""
    bases = d["bases"]
    samples_c = [s for s, _, _ in cols]
    xs = [col_short(s) for s in samples_c]    # per-column tick = round only; group named by band
    xpos = list(range(len(cols)))
    z, hov = [], []
    for b in bases:
        row_z, row_h = [], []
        for s, dn, dl in cols:
            a = d["per"][dn].get(b, 0.0)
            c = d["per"][s["name"]].get(b, 0.0)
            lfc = math.log2((c + EPS) / (a + EPS))
            row_z.append(lfc)
            row_h.append(f"{esc(s['line'])} · {col_label(s)}<br><b>{b}</b><br>"
                         f"{dl} {a:.4f} → {c:.4f}<br>log2 FC {lfc:+.2f}")
        z.append(row_z); hov.append(row_h)
    lim = max((abs(v) for row in z for v in row), default=1.0) or 1.0
    yt = [f'<span style="color:{d["colors"].get(feat_class(b), GREY)}">{esc(b)}</span>' for b in bases]
    # reversescale so positive log2FC (enriched) = red and negative (depleted) = blue
    fig = go.Figure(go.Heatmap(
        z=z, x=xpos, y=list(range(len(bases))), text=hov, hoverinfo="text",
        colorscale="RdBu", reversescale=True, zmid=0, zmin=-lim, zmax=lim,
        colorbar=dict(title="log2 FC", thickness=12, len=0.5)))
    fig.update_yaxes(tickmode="array", tickvals=list(range(len(bases))), ticktext=yt,
                     tickfont=dict(size=8), autorange="reversed")
    classes = [feat_class(b) for b in bases]
    for i in range(1, len(classes)):
        if classes[i] != classes[i - 1]:
            fig.add_hline(y=i - 0.5, line_color="rgba(139,148,158,0.5)", line_width=1)
    for x in line_boundaries(samples_c):
        fig.add_vline(x=x, line_color=GREY, line_width=2)
    group_band(fig, samples_c, yshift=36)
    fig.update_xaxes(tickmode="array", tickvals=xpos, ticktext=xs, side="top",
                     tickangle=-40, tickfont=dict(size=10))
    style(fig, height=max(520, 15 * len(bases) + 120))
    fig.update_layout(margin=dict(t=127))
    return fig_html(fig)


def fig_trajectories(d, topn=10):
    """Per-group round trajectories for the most abundant features (log y); groups with ≥2 rounds."""
    lines = by_line(d["samples"])
    lines = {ln: ss for ln, ss in lines.items() if len(ss) >= 2}
    if not lines:
        return ""
    peak = {b: max(d["per"][s["name"]].get(b, 0.0) for s in d["samples"]) for b in d["bases"]}
    top = sorted(d["bases"], key=lambda b: peak[b], reverse=True)[:topn]
    from plotly.subplots import make_subplots
    lnames = list(lines)
    base_name = d.get("baseline")   # shared R0 point prepended to every line, if present
    fig = make_subplots(rows=1, cols=len(lnames), shared_yaxes=True,
                        subplot_titles=[esc(x) for x in lnames], horizontal_spacing=0.04)
    for ci, ln in enumerate(lnames, start=1):
        ss = lines[ln]
        xs = ([0] if base_name else []) + [s["round"] for s in ss]
        for b in top:
            ys = ([d["per"][base_name].get(b, 0.0)] if base_name else []) + \
                 [d["per"][s["name"]].get(b, 0.0) for s in ss]
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines+markers", name=b, legendgroup=b,
                showlegend=(ci == 1), marker=dict(size=6, color=d["colors"].get(feat_class(b), GREY)),
                line=dict(width=1.6, color=d["colors"].get(feat_class(b), GREY)),
                hovertemplate=f"{ln} R%{{x}}<br><b>{b}</b><br>%{{y:.4f}} copies/read<extra></extra>"),
                row=1, col=ci)
        fig.update_xaxes(title=("round (0 = library)" if base_name else "round"),
                         dtick=1, row=1, col=ci)
    fig.update_yaxes(type="log", title="mean copies/read", row=1, col=1)
    style(fig, height=480)
    # extra top margin so the per-cell-line subplot titles are not clipped
    fig.update_layout(legend=dict(font=dict(size=9)), margin=dict(t=50))
    return fig_html(fig)


def fig_contrast(d):
    """At the shared max round, one group vs the other, mean copies/read per feature (log-log).
    Only drawn when there are exactly two groups that share a (numeric) round."""
    lines = by_line(d["samples"])
    if len(lines) != 2:
        return ""
    lnames = list(lines)
    rounds_a = {s["round"] for s in lines[lnames[0]]}
    rounds_b = {s["round"] for s in lines[lnames[1]]}
    shared = {r for r in (rounds_a & rounds_b) if r is not None}
    if not shared:
        return ""
    rnd = max(shared)
    sa = next(s for s in lines[lnames[0]] if s["round"] == rnd)
    sb = next(s for s in lines[lnames[1]] if s["round"] == rnd)
    fig = go.Figure()
    for c in d["order"]:
        bs = [b for b in d["bases"] if feat_class(b) == c]
        xs = [d["per"][sa["name"]].get(b, 0.0) + EPS for b in bs]
        ys = [d["per"][sb["name"]].get(b, 0.0) + EPS for b in bs]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers", name=c, text=bs,
            marker=dict(size=9, color=d["colors"][c], line=dict(width=0.5, color="rgba(0,0,0,0.25)")),
            hovertemplate=("<b>%{text}</b> · " + c + f"<br>{lnames[0]} %{{x:.4f}}"
                           f"<br>{lnames[1]} %{{y:.4f}}<extra></extra>")))
    allv = [d["per"][sa["name"]].get(b, 0.0) for b in d["bases"]] + \
           [d["per"][sb["name"]].get(b, 0.0) for b in d["bases"]]
    hi = max([v for v in allv if v > 0], default=1.0)
    fig.add_trace(go.Scatter(x=[EPS, hi], y=[EPS, hi], mode="lines", showlegend=False,
                             line=dict(dash="dash", color="rgba(139,148,158,0.6)"), hoverinfo="skip"))
    fig.update_xaxes(type="log", title=f"{lnames[0]} R{rnd} — mean copies/read")
    fig.update_yaxes(type="log", title=f"{lnames[1]} R{rnd} — mean copies/read")
    style(fig, height=520, title=f"Group contrast at R{rnd}: off-diagonal = group-selective ligands")
    return fig_html(fig)


# ------------------------------------------------------------------------ multiplicity & quality
def sample_palette(d):
    """Sample -> its GROUP's colour (baseline grey), shared by the multiplicity/quality panels.

    It used to hand every sample its own hue off ligand_vocab's PALETTE -- the same list that colours
    fragment CLASSES elsewhere on this page, so one hue meant "class X" in the heatmap and "sample Y"
    here. Now colour means group, everywhere, in all three cross reports. Rounds inside a group are
    told apart by dash (sample_dash) on the line panels, and by position + the group band on the bars.
    """
    gcol = group_color(d["samples"], key="line")
    return {s["name"]: (BASELINE_GREY if s.get("is_baseline")
                        else gcol.get(s["line"], GREY)) for s in d["samples"]}


DASHES = ["solid", "dash", "longdash", "dashdot", "longdashdot"]


def sample_dash(s):
    """Rounds within a group are separated by dash, since hue is now spent on the group."""
    if s.get("is_baseline") or s["round"] is None:
        return "dot"
    return DASHES[(s["round"] - 1) % len(DASHES)]


def fig_multiplicity(d):
    cols = sample_palette(d)
    xs = list(range(MAXBUCKET + 1))
    labels = [str(x) for x in xs[:-1]] + [f"{MAXBUCKET}+"]
    fig = go.Figure()
    for s in d["samples"]:
        nm = s["name"]
        ys = [100 * d["ppr"][nm].get(x, 0.0) for x in xs]
        fig.add_trace(go.Scatter(
            x=labels, y=ys, mode="lines+markers", name=col_label(s), legendgroup=s["line"],
            marker=dict(size=5, color=cols[nm]),
            line=dict(width=1.6, color=cols[nm], dash=sample_dash(s)),
            hovertemplate=f"{col_label(s)}<br>%{{x}} payload frags<br>%{{y:.1f}}% of reads<extra></extra>"))
    fig.update_xaxes(title="payload fragment copies in a read")
    fig.update_yaxes(title="% of reads")
    style(fig, height=380)
    return fig_html(fig)


def fig_quality(d):
    cols = sample_palette(d)
    maxed = max((max(v) for v in d["edh"].values() if v), default=13)
    xs = list(range(maxed + 1))
    fig = go.Figure()
    for s in d["samples"]:
        nm = s["name"]
        ys = [100 * d["edh"][nm].get(x, 0.0) for x in xs]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers", name=col_label(s), legendgroup=s["line"],
            marker=dict(size=5, color=cols[nm]),
            line=dict(width=1.6, color=cols[nm], dash=sample_dash(s)),
            hovertemplate=f"{col_label(s)}<br>edit distance %{{x}}<br>%{{y:.1f}}% of calls<extra></extra>"))
    fig.update_xaxes(title="edit distance (payload calls)", dtick=1)
    fig.update_yaxes(title="% of calls")
    style(fig, height=380)
    return fig_html(fig)


def fig_depth(d):
    """Absolute sequencing depth per sample: reads decoded, and the payload-bearing subset.

    The other quality panels are all normalised per sample, so this is the only place the raw
    depth shows. Both series carry the sample's own colour (shared with the coverage plot beside
    it) and are told apart by opacity, not hue."""
    cols = sample_palette(d)
    samples = d["samples"]
    xpos = list(range(len(samples)))          # numeric x so duplicate round labels never merge bars
    xshort = [col_short(s) for s in samples]
    xfull = [col_label(s) for s in samples]
    bar = [cols[s["name"]] for s in samples]
    n_reads = [int(d["meta"][s["name"]]["n_reads"]) for s in samples]
    n_pay = [int(d["meta"][s["name"]]["n_reads_with_payload"]) for s in samples]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=xpos, y=n_reads, name="reads decoded", customdata=xfull,
                         marker=dict(color=bar, opacity=0.42),
                         hovertemplate="%{customdata}<br>reads decoded %{y:,}<extra></extra>"))
    fig.add_trace(go.Bar(x=xpos, y=n_pay, name="with payload", customdata=xfull,
                         marker=dict(color=bar),
                         hovertemplate="%{customdata}<br>with &ge;1 payload frag "
                                       "%{y:,}<extra></extra>"))
    # a legend swatch can't stand for a per-bar colour list, and the coverage plot beside this one
    # already legends the sample colours -- the two series are named in the hover and the caption.
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.05, showlegend=False)
    for x in line_boundaries(samples):
        fig.add_vline(x=x, line_color=GREY, line_width=2)
    group_band(fig, samples, yshift=16)
    fig.update_xaxes(tickmode="array", tickvals=xpos, ticktext=xshort, tickangle=-40)
    fig.update_yaxes(title="reads", tickformat="~s")
    style(fig, height=380)
    fig.update_layout(margin=dict(t=88))      # room for the rotated group band above the bars
    return fig_html(fig)


def fig_coverage(d):
    """Per-read coverage = expected-fragment bases ÷ post-trim read length; distribution per sample."""
    cols = sample_palette(d)
    have = [s for s in d["samples"] if d["cov"].get(s["name"])]
    if not have:
        return "", []
    centers = [100 * (i + 0.5) / COVBINS for i in range(COVBINS)]
    fig = go.Figure()
    for s in have:
        nm = s["name"]
        ys = [100 * d["cov"][nm].get(i, 0.0) for i in range(COVBINS)]
        fig.add_trace(go.Scatter(
            x=centers, y=ys, mode="lines", name=col_label(s), legendgroup=s["line"],
            line=dict(width=1.8, color=cols[nm], dash=sample_dash(s)),
            hovertemplate=f"{col_label(s)}<br>~%{{x:.0f}}% covered<br>%{{y:.1f}}% of reads<extra></extra>"))
    # x starts at 60: under 0.2% of any sample's reads fall below that, and the clip gives the
    # informative 60-100 region ~2.5x the width.
    fig.update_xaxes(title="% of read length covered by expected fragments", range=[60, 100])
    fig.update_yaxes(title="% of reads")
    style(fig, height=380)
    fig.update_layout(margin=dict(t=88))      # match fig_depth's top margin so the plots align
    missing = [s["name"] for s in d["samples"] if not d["cov"].get(s["name"])]
    return fig_html(fig), missing


# ------------------------------------------------------------------------ table
def overview_table(d):
    rows = ["<table>", "<thead><tr><th>sample</th><th>group</th><th class=num>round</th>"
            "<th class=num>reads decoded</th><th class=num>% with payload</th>"
            "<th class=num>copies/read</th></tr></thead>", "<tbody>"]
    for s in d["samples"]:
        m = d["meta"][s["name"]]
        n = int(m["n_reads"]); wp = int(m["n_reads_with_payload"])
        rows.append(
            f"<tr><td>{esc(s['name'])}</td><td>{esc(s['line'])}</td>"
            f"<td class=num>{'' if s['round'] is None else s['round']}</td>"
            f"<td class=num>{n:,}</td><td class=num>{100*wp/max(n,1):.1f}</td>"
            f"<td class=num>{float(m['mean_payload_copies_per_read']):.2f}</td></tr>")
    rows += ["</tbody>", "</table>"]
    return f'<div style="overflow-x:auto">{"".join(rows)}</div>'


# ------------------------------------------------------------------------ assemble
_kit = Path(os.path.expanduser("~/.claude/assets/html-kit/style.css"))
STYLE = _kit.read_text() if _kit.exists() else ""
PLOTLYJS = get_plotlyjs()          # the plotly.js bundle, inlined once per page
THEME_INIT = ("<script>(function(){try{var t=localStorage.getItem('theme');"
              "if(!t)t=matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light';"
              "document.documentElement.setAttribute('data-theme',t);}catch(e){}})();</script>")
THEME_BTN = ('<button class=themebtn onclick="(function(){var d=document.documentElement,'
             "t=d.getAttribute('data-theme')=='dark'?'light':'dark';"
             "d.setAttribute('data-theme',t);try{localStorage.setItem('theme',t);}catch(e){}})()\">"
             "&#9680; theme</button>")


def main():
    samples = discover()
    if not samples:
        raise SystemExit(f"no decoded samples found under {BASE} "
                         f"(need <S>/<S>.featurecount.parquet + .summary.tsv)")
    baseline = load_baseline()
    if baseline:
        samples = [baseline] + samples          # R0 library first, shared across cell lines
    d = load(samples)
    lines = sorted({s["line"] for s in samples if not s.get("is_baseline")})
    n_real = sum(1 for s in samples if not s.get("is_baseline"))
    r0_note = (" plus the pre-selection library as a shared <b>R0</b> baseline"
               if baseline else "")
    legend = " &nbsp; ".join(
        f'<b style="color:{d["colors"][c]}">&#9632; {esc(c)}</b>' for c in d["order"])

    cov_html, cov_missing = fig_coverage(d)
    cov_note = ""
    if cov_html and cov_missing:
        cov_note = ('<p class="small">Coverage omits '
                    + ", ".join(f"<code>{esc(m)}</code>" for m in cov_missing)
                    + " (no <code>fused_reads.fq</code> retained for the post-trim read length).</p>")
    elif not cov_html:
        cov_html = '<p class="small">Coverage unavailable — no <code>fused_reads.fq</code> found.</p>'

    H = [
        "<!doctype html>", "<html lang=en>", "<head>", "<meta charset=utf-8>",
        '<meta name=viewport content="width=device-width,initial-scale=1">', THEME_INIT,
        "<title>Cross-sample ligand fragment feature counts</title>",
        f"<style>{STYLE}</style>", "</head>", "<body>",
        f"<script>{PLOTLYJS}</script>",          # inline plotly.js once; all figures reference it
        THEME_BTN,
        "<h1>Cross-sample ligand fragment feature counts</h1>",
        f'<p class="subtitle">Per-read ligand-fragment profiles of {n_real} decoded samples '
        f'({esc(", ".join(lines))}){r0_note}, lined up feature-by-feature. Charts are interactive — '
        "hover for exact numbers, click legend entries to toggle classes/features, drag to zoom.</p>",
        '<p class="small">Sample colour = the config <b>group</b> (the same colour for a cell line in '
        "every cross report); rounds within a group are told apart by line dash. The fragment-class "
        "colours in the legend below are a different axis — they name features, not samples.</p>",
        color_key_html(group_color(samples, key="line"), samples, key="line"),

        '<div class="callout aim"><p><b>Aim.</b> For every ligand fragment, how does <b>mean copies '
        "per decoded read</b> shift <b>across selection rounds</b> (within a group) and "
        "<b>between groups</b>? Each sample keeps its own value; samples are ordered and labelled by "
        "<b>group</b> (config <code>group</code>). All values strand-collapsed and normalised "
        "per read; UMI anchors excluded (backbone). Vocabulary and classes are derived from the "
        "decodes themselves.</p></div>",

        "<h2>Quality</h2>",
        "<p><b>Depth</b> (left): reads decoded per sample (pale) and the subset carrying at least "
        "one payload fragment (solid) — the absolute scale that every other panel here normalises "
        "away. <b>Coverage</b> (right): fraction of each read's length (post-trim, from "
        "<code>fused_reads.fq</code>) that is covered by expected-sequence fragments — how "
        "completely the panel explains a read, as the distribution of reads per sample. The "
        "coverage axis starts at 60% because fewer than 0.2% of reads fall below it.</p>",
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;align-items:start">',
        f'<div style="min-width:0">{fig_depth(d)}</div>',
        f'<div style="min-width:0">{cov_html}</div>',
        "</div>",
        cov_note,
        "<p>Left: how many payload fragment copies a read carries (0 = amplification loop-only "
        "reads), as % of each sample's reads. Right: edit-distance of payload calls against the "
        "expected sequences (lower = cleaner match), as % of each sample's calls. Both normalised "
        "per sample so sequencing depth doesn't dominate; toggle samples in the legend.</p>",
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;align-items:start">',
        f'<div style="min-width:0">{fig_multiplicity(d)}</div>',
        f'<div style="min-width:0">{fig_quality(d)}</div>',
        "</div>",

        "<h2>Samples</h2>", overview_table(d),

        "<h2>Every feature × every sample</h2>",
        "<p>Master grid of mean copies/read (log colour). Rows are features coloured by structural "
        "class (with class separators); columns are samples, ordered by group (labelled band + heavy "
        "vertical rule) then round. The two panels split orientation — <b>forward</b> (left) and its "
        "<b>reverse-complement</b> <code>_rc</code> (right), same feature rows — so you can see the "
        "<code>_rc</code> calls are a small minority. Scan a <b>row</b> for one ligand across "
        "groups/rounds, a <b>column</b> for one sample's profile.</p>",
        f'<p class="small">Classes: {legend}</p>',
        fig_heatmap(d),

        "<h2>Class composition per sample</h2>",
        "<p>Share of payload calls contributed by each structural class — how the ligand class mix "
        "moves across rounds and differs between groups.</p>",
        fig_class_composition(d),

        "<h2>Enrichment vs baseline</h2>",
        "<p>Per-feature log2 fold-change of mean copies/read, <b>one column per sample</b> (grouped "
        "by group), each vs "
        + ("the <b>R0 library</b>" if baseline else "its group's first round")
        + ": red = enriched, blue = depleted.</p>",
        fig_enrichment(d),
        "<p>The most abundant features as explicit per-round trajectories (log y), for groups with "
        "more than one round" + (" (round 0 = the shared library)" if baseline else "") + ":</p>",
        fig_trajectories(d),

        "<h2>Group contrast</h2>",
        "<p>For a two-group experiment: at the shared final round, each feature's mean copies/read in "
        "one group vs the other. Points on the dashed diagonal behave identically; off-diagonal points "
        "are group-selective. (Shown only when there are exactly two groups.)</p>",
        fig_contrast(d),

        "<h2>Method &amp; caveats</h2>",
        '<dl class="gloss">',
        "<dt>mean copies / read</dt><dd>Σ calls of a feature ÷ reads decoded, per sample; "
        "strand-collapsed (fwd + <code>_rc</code>), UMI anchors excluded.</dd>",
        "<dt>normalisation</dt><dd>Samples differ in depth, so every panel uses the per-read metric, "
        "never raw counts.</dd>",
        "<dt>enrichment</dt><dd>log2((final-round + 1e-4) ÷ (baseline + 1e-4)) of mean copies/read "
        "within a cell line; baseline = "
        + ("the shared R0 pre-selection library." if baseline else "each line's first round.")
        + "</dd>",
        ("<dt>R0 baseline</dt><dd>The pre-selection library "
         f"(<code>{esc(R0_STEM.name)}</code>) is shared by both cell lines: it is the round-0 point "
         "of every trajectory and the denominator of every enrichment fold-change.</dd>"
         if baseline else ""),
        "<dt>orientation</dt><dd>The master grid splits <b>forward</b> vs <b>reverse-complement</b> "
        "(<code>_rc</code>); every other panel sums the two.</dd>",
        "<dt>coverage</dt><dd>Σ(stop − start) over a read's expected fragments (anchors included) ÷ "
        "the read's post-trim length from <code>fused_reads.fq</code>, clipped to [0, 1].</dd>",
        "<dt>vocabulary</dt><dd>Feature set, backbone/anchor set and structural classes are derived "
        "from the decodes (<code>workflow/ligand_vocab.py</code>); nothing hard-coded.</dd>",
        "</dl>",
        '<p class="small">Source: <code>' + esc(str(BASE.name)) +
        "/*/*.featurecount.parquet</code> → <code>workflow/build_featurecount_report_cross.py</code>.</p>",
        "</body>", "</html>",
    ]
    OUT.write_text("\n".join(x for x in H if x))
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1e6:.2f} MB)  [{len(samples)} samples]")


if __name__ == "__main__":
    main()
