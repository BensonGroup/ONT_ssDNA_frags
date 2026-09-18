#!/usr/bin/env python
"""Self-contained INTERACTIVE (Plotly) report of the 2026 initial-library feature counts.

Adapted from workflow/build_featurecount_report.py. That report compared many samples across
selection rounds and tissues; this decode is a SINGLE library, so the report is a one-sample
ligand-fragment profile instead: what does an average decoded read carry, how do the structural
classes compare, and how clean/multiplexed are the calls.

Everything about the construct set is DATA-DRIVEN (see ligand_vocab.py): the feature list, the
backbone/anchor set, and the structural classes/colours are all derived from the run's own
outputs, so the same script reports any library with no code change.

    Reads:  $L2F_STEM.featurecount.parquet          (from build_featurecounts_2026.py)
            $L2F_STEM.featurecount.summary.tsv
            $L2F_STEM.fragments.parquet              (edit distances)
            $L2F_STEM.umis.parquet                   (anchor set)
            $L2F_STEM.ligateddna.log                 (provenance + expected-seqs CSV path)
    Writes: $L2F_STEM.featurecount_report.html       (interactive, plotly.js inlined)

    L2F_STEM default = ligand2frags_new/2026initial_library

Run: pixi run python workflow/build_featurecount_report_2026.py
"""
import html
import os
from pathlib import Path

import plotly.graph_objects as go
import polars as pl

from ligand_vocab import (
    anchor_set, class_colors, class_order, parse_feature, parse_log, read_expected, GREY,
)

ROOT = Path(__file__).resolve().parents[1]
STEM = Path(os.environ.get("L2F_STEM", ROOT / "ligand2frags_new" / "2026initial_library"))
if not STEM.is_absolute():
    STEM = ROOT / STEM


def part(ext: str) -> Path:
    return STEM.parent / f"{STEM.name}{ext}"


FC = part(".featurecount.parquet")
SUMM = part(".featurecount.summary.tsv")
FRAGS = part(".fragments.parquet")
UMIS = part(".umis.parquet")
LOG = part(".ligateddna.log")
OUT = part(".featurecount_report.html")

RC = r"_rc$"          # strand-collapse: X_rc and X are one feature
MAXBUCKET = 15        # fragments-per-read histogram tail bucket


def esc(s):
    return html.escape(str(s))


# ------------------------------------------------------------------------ load & aggregate
def load():
    prov = parse_log(LOG)
    expected = read_expected(prov.get("expected") or ROOT / "exp" / "nonexistent.csv")
    anchors = anchor_set(UMIS, expected)
    anc_ids = list(set(anchors) | {f"{a}_rc" for a in anchors})

    summ = pl.read_csv(SUMM, separator="\t").to_dicts()[0]
    n_reads = int(summ["n_reads"])
    # A sample can decode to zero reads (e.g. every read dropped as missing_anchor). The
    # per-read rates are then undefined; `denom` keeps them finite and the aggregates below
    # are empty anyway, so every rate comes out 0.
    denom = n_reads or 1

    fc = pl.scan_parquet(FC)
    # Per-feature, strand-collapsed (fwd + _rc merged; reads counted once via n_unique).
    base_agg = (
        fc.with_columns(base=pl.col("fragment_id").str.replace(RC, ""))
        .group_by("base")
        .agg(total=pl.col("count").sum(), n_reads_with=pl.col("read_acc").n_unique())
        .collect()
    )
    # Median match quality (edit distance) per feature, straight off the fragments store.
    ed = (
        pl.scan_parquet(FRAGS)
        .with_columns(base=pl.col("fragment_id").str.replace(RC, ""))
        .group_by("base")
        .agg(ed_median=pl.col("edit_distance").median())
        .collect()
    )
    agg = base_agg.join(ed, on="base", how="left")

    feats = {}
    for r in agg.iter_rows(named=True):
        b = r["base"]
        cls, length, _ = parse_feature(b)
        feats[b] = {
            "base": b, "is_anchor": b in anchors,
            "class": "anchor" if b in anchors else cls, "length": length,
            "total": int(r["total"]), "n_reads_with": int(r["n_reads_with"]),
            "mean_copies": r["total"] / denom,
            "pct_reads": 100.0 * r["n_reads_with"] / denom,
            "ed_median": r["ed_median"],
        }

    order = class_order([b for b in feats if not feats[b]["is_anchor"]], expected, anchors)
    colors = class_colors(order)

    # Payload fragments (copies) per read; reads with zero payload added back explicitly.
    ppr = (
        fc.filter(~pl.col("fragment_id").is_in(anc_ids))
        .group_by("read_acc").agg(nfrag=pl.col("count").sum())
        .with_columns(bucket=pl.min_horizontal("nfrag", pl.lit(MAXBUCKET)))
        .group_by("bucket").agg(reads=pl.len()).sort("bucket").collect()
    )
    ppr = {int(r["bucket"]): int(r["reads"]) for r in ppr.iter_rows(named=True)}
    ppr[0] = n_reads - int(summ["n_reads_with_payload"])

    # Overall payload edit-distance distribution.
    edh = (
        pl.scan_parquet(FRAGS).filter(~pl.col("fragment_id").is_in(anc_ids))
        .group_by("edit_distance").agg(n=pl.len()).sort("edit_distance").collect()
    )
    edh = {int(r["edit_distance"]): int(r["n"]) for r in edh.iter_rows(named=True)}

    expected_bases = {n for n, s in expected if n not in anchors and "N" not in str(s).upper()}
    absent = sorted(expected_bases - set(feats)) if expected else []

    return dict(prov=prov, n_reads=n_reads, summ=summ, feats=feats, order=order,
                colors=colors, ppr=ppr, edh=edh, absent=absent)


# ------------------------------------------------------------------------ plotly helpers
def style(fig, height, title=None):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
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


def fig_html(fig, first=False):
    return fig.to_html(
        full_html=False, include_plotlyjs=("inline" if first else False),
        default_width="100%", config={"displaylogo": False, "responsive": True},
    )


def sortkey(f, order):
    ci = order.index(f["class"]) if f["class"] in order else 99
    L = f["length"] if f["length"] is not None else -1
    return (ci, L, f["base"])


# ------------------------------------------------------------------------ figures
def fig_master(d, first):
    payload = [f for f in d["feats"].values() if not f["is_anchor"]]
    payload.sort(key=lambda f: sortkey(f, d["order"]))
    cats = [f["base"] for f in payload]
    fig = go.Figure()
    for c in d["order"]:
        fs = [f for f in payload if f["class"] == c]
        if not fs:
            continue
        fig.add_trace(go.Bar(
            orientation="h", name=c, marker_color=d["colors"][c],
            y=[f["base"] for f in fs], x=[f["mean_copies"] for f in fs],
            customdata=[[f["total"], f["n_reads_with"], f["pct_reads"],
                         ("—" if f["length"] is None else f["length"]),
                         ("—" if f["ed_median"] is None else round(f["ed_median"], 1))] for f in fs],
            hovertemplate=("<b>%{y}</b> · " + c +
                           "<br>mean copies/read <b>%{x:.4f}</b>"
                           "<br>total calls %{customdata[0]:,}"
                           "<br>reads with %{customdata[1]:,} (%{customdata[2]:.1f}%)"
                           "<br>nt size %{customdata[3]} · median edit dist %{customdata[4]}"
                           "<extra></extra>"),
        ))
    fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(cats)),
                     tickfont=dict(size=9))
    fig.update_xaxes(type="log", title="mean copies per decoded read (log)")
    style(fig, height=max(420, 20 * len(cats) + 90))
    fig.update_layout(barmode="stack", legend=dict(traceorder="normal"))
    return fig_html(fig, first)


def fig_backbone(d):
    anc = sorted((f for f in d["feats"].values() if f["is_anchor"]), key=lambda f: f["base"])
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f["base"] for f in anc], y=[f["mean_copies"] for f in anc],
        marker_color=GREY, text=[f"{f['mean_copies']:.3f}" for f in anc], textposition="outside",
        customdata=[[f["pct_reads"], f["total"]] for f in anc],
        hovertemplate=("<b>%{x}</b><br>mean copies/read %{y:.4f}"
                       "<br>in %{customdata[0]:.1f}% of reads · %{customdata[1]:,} calls<extra></extra>"),
    ))
    fig.add_hline(y=1.0, line_dash="dash", line_color="rgba(139,148,158,0.6)",
                  annotation_text="1 / read", annotation_position="top left")
    fig.update_yaxes(title="mean copies per read", rangemode="tozero")
    style(fig, height=300)
    return fig_html(fig)


def fig_classes(d):
    rows = []
    for c in d["order"]:
        fs = [f for f in d["feats"].values() if f["class"] == c]
        tot = sum(f["total"] for f in fs)
        rows.append((c, tot, sum(f["mean_copies"] for f in fs), len(fs)))
    payload_total = sum(r[1] for r in rows) or 1
    fig = go.Figure(go.Bar(
        x=[r[0] for r in rows], y=[r[1] for r in rows],
        marker_color=[d["colors"][r[0]] for r in rows],
        customdata=[[100 * r[1] / payload_total, r[2], r[3]] for r in rows],
        text=[f"{100*r[1]/payload_total:.0f}%" for r in rows], textposition="outside",
        hovertemplate=("<b>%{x}</b><br>%{y:,} calls (%{customdata[0]:.1f}% of fragment calls)"
                       "<br>Σ mean copies/read %{customdata[1]:.3f}"
                       "<br>%{customdata[2]} features<extra></extra>"),
    ))
    fig.update_yaxes(type="log", title="total fragment calls (log)")
    style(fig, height=360)
    return fig_html(fig)


def fig_perread(d):
    xs = list(range(0, MAXBUCKET + 1))
    ys = [d["ppr"].get(x, 0) for x in xs]
    labels = [str(x) for x in xs[:-1]] + [f"{MAXBUCKET}+"]
    fig = go.Figure(go.Bar(
        x=labels, y=ys, marker_color="#4f9cff",
        hovertemplate="%{x} fragments<br><b>%{y:,}</b> reads<extra></extra>"))
    fig.update_xaxes(title="fragment copies in a read")
    fig.update_yaxes(title="reads")
    style(fig, height=340)
    return fig_html(fig)


def fig_editdist(d):
    xs = sorted(d["edh"])
    fig = go.Figure(go.Bar(
        x=xs, y=[d["edh"][x] for x in xs], marker_color="#3fb950",
        hovertemplate="edit distance %{x}<br><b>%{y:,}</b> calls<extra></extra>"))
    fig.update_xaxes(title="edit distance (fragment calls)", dtick=1)
    fig.update_yaxes(title="calls")
    style(fig, height=340)
    return fig_html(fig)


def fig_length(d):
    fig = go.Figure()
    any_pt = False
    for c in d["order"]:
        fs = [f for f in d["feats"].values()
              if f["class"] == c and f["length"] is not None]
        if not fs:
            continue
        any_pt = True
        fig.add_trace(go.Scatter(
            mode="markers", name=c, marker=dict(color=d["colors"][c], size=9,
                                                line=dict(width=0.5, color="rgba(0,0,0,0.25)")),
            x=[f["length"] for f in fs], y=[f["mean_copies"] for f in fs],
            text=[f["base"] for f in fs],
            hovertemplate="<b>%{text}</b><br>nt size %{x}<br>mean copies/read %{y:.4f}<extra></extra>",
        ))
    if not any_pt:
        return ""
    fig.update_xaxes(title="nominal fragment size (nt / bulge-nt)")
    fig.update_yaxes(type="log", title="mean copies per read (log)")
    style(fig, height=420)
    return fig_html(fig)


# ------------------------------------------------------------------------ HTML tables
def stat_tiles(d):
    s = d["summ"]
    n = d["n_reads"]
    wp = int(s["n_reads_with_payload"])
    carry = f"{100*wp/n:.1f}% carry ≥1 ligand fragment" if n else "no reads decoded"
    tiles = [
        ("reads decoded", f"{n:,}", "≥1 UMI-anchored, fused"),
        ("with a fragment", f"{wp:,}", carry),
        ("fragment calls", f"{int(s['n_payload_rows']):,}", "ligand-fragment detections"),
        ("fragment features", f"{int(s['n_payload_features'])}", "distinct incl. _rc orientation"),
        ("copies / read", f"{s['mean_payload_copies_per_read']:.2f}", "fragment calls ÷ reads"),
        ("copies / read+", f"{s['mean_payload_copies_per_read_with']:.2f}", "÷ reads with ≥1 fragment"),
    ]
    cells = "".join(
        f'<div class="tile"><div class="tk">{esc(k)}</div>'
        f'<div class="tv">{esc(v)}</div><div class="tn small">{esc(note)}</div></div>'
        for k, v, note in tiles)
    return f'<div class="tiles">{cells}</div>'


def provenance(d):
    p = d["prov"]
    if not p:
        return ""
    bits = []
    if p.get("branch"):
        bits.append(f"tool branch <code>{esc(p['branch'])}</code>")
    if p.get("min_identity"):
        bits.append(f"min identity <code>{esc(p['min_identity'])}</code>")
    if "fused" in p:
        drop_detail = ""
        if "missing_anchor" in p and "strand_conflict" in p:
            drop_detail = (f" (missing_anchor {p['missing_anchor']:,}, "
                           f"strand_conflict {p['strand_conflict']:,})")
        bits.append(f"<code>{p['fused']:,}</code> reads fused, "
                    f"<code>{p['dropped']:,}</code> dropped{drop_detail}")
    if p.get("expected"):
        bits.append(f"expected set <code>{esc(Path(p['expected']).name)}</code>")
    return '<p class="small">' + " · ".join(bits) + "</p>" if bits else ""


def feature_table(d):
    payload = [f for f in d["feats"].values() if not f["is_anchor"]]
    payload.sort(key=lambda f: sortkey(f, d["order"]))
    rows = ["<table>",
            "<thead><tr><th>feature</th><th>class</th><th class=num>nt</th>"
            "<th class=num>total calls</th><th class=num>reads with</th>"
            "<th class=num>% reads</th><th class=num>copies/read</th>"
            "<th class=num>med. edit</th></tr></thead>", "<tbody>"]
    prev = None
    for f in payload:
        sep = ' style="border-top:2px solid rgba(139,148,158,.4)"' if f["class"] != prev else ""
        prev = f["class"]
        col = d["colors"].get(f["class"], GREY)
        rows.append(
            f"<tr{sep}><td style='color:{col};font-weight:600'>{esc(f['base'])}</td>"
            f"<td class=small>{esc(f['class'])}</td>"
            f"<td class=num>{'' if f['length'] is None else f['length']}</td>"
            f"<td class=num>{f['total']:,}</td><td class=num>{f['n_reads_with']:,}</td>"
            f"<td class=num>{f['pct_reads']:.1f}</td><td class=num>{f['mean_copies']:.4f}</td>"
            f"<td class=num>{'' if f['ed_median'] is None else round(f['ed_median'],1)}</td></tr>")
    rows += ["</tbody>", "</table>"]
    return f'<div style="overflow-x:auto;max-height:560px;overflow-y:auto">{"".join(rows)}</div>'


# ------------------------------------------------------------------------ assemble
_kit = Path(os.path.expanduser("~/.claude/assets/html-kit/style.css"))
STYLE = _kit.read_text() if _kit.exists() else ""
EXTRA = """
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:.7rem;margin:1rem 0}
.tile{border:1px solid rgba(139,148,158,.25);border-radius:10px;padding:.7rem .85rem}
.tile .tk{font-size:.72rem;text-transform:uppercase;letter-spacing:.04em;opacity:.75}
.tile .tv{font-size:1.5rem;font-weight:700;line-height:1.1;margin:.15rem 0}
.tile .tn{opacity:.7}
"""
THEME_INIT = ("<script>(function(){try{var t=localStorage.getItem('theme');"
              "if(!t)t=matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light';"
              "document.documentElement.setAttribute('data-theme',t);}catch(e){}})();</script>")
THEME_BTN = ('<button class=themebtn onclick="(function(){var d=document.documentElement,'
             "t=d.getAttribute('data-theme')=='dark'?'light':'dark';"
             "d.setAttribute('data-theme',t);try{localStorage.setItem('theme',t);}catch(e){}})()\">"
             "&#9680; theme</button>")


def zero_note(d):
    """Banner for a sample the decoder fused no reads for: say so, and say why."""
    p = d["prov"]
    bits = []
    if p.get("dropped"):
        bits.append(f"all <code>{p['dropped']:,}</code> reads were dropped")
    if p.get("missing_anchor"):
        bits.append(f"<code>{p['missing_anchor']:,}</code> for a missing UMI anchor")
    if p.get("strand_conflict"):
        bits.append(f"<code>{p['strand_conflict']:,}</code> for a strand conflict")
    why = (" — " + ", ".join(bits)) if bits else ""
    return ('<div class="callout note"><p><b>No reads decoded.</b> The decoder fused 0 reads for '
            f"this sample{why}. There are no feature counts to plot, so the figures and the "
            "per-feature table are omitted; every count below is zero.</p></div>")


def main():
    d = load()
    n_payload_feats = int(d["summ"]["n_payload_features"])
    n_classes = len(d["order"])
    legend = " &nbsp; ".join(
        f'<b style="color:{d["colors"][c]}">&#9632; {esc(c)}</b>' for c in d["order"])

    absent_note = ""
    if d["absent"]:
        absent_note = ('<div class="callout note"><p><b>Expected but unseen:</b> '
                       + ", ".join(f"<code>{esc(b)}</code>" for b in d["absent"])
                       + " — present in the expected-sequences set but with zero calls here.</p></div>")

    head = [
        "<!doctype html>", "<html lang=en>", "<head>", "<meta charset=utf-8>",
        '<meta name=viewport content="width=device-width,initial-scale=1">', THEME_INIT,
        f"<title>Ligand fragment feature counts — {esc(STEM.name)}</title>",
        f"<style>{STYLE}{EXTRA}</style>", "</head>", "<body>", THEME_BTN,
        f"<h1>Ligand fragment feature counts &mdash; <code>{esc(STEM.name)}</code></h1>",
        '<p class="subtitle">Per-read decode of the ligated-DNA initial library '
        "(LigandFrags2, UMI-anchor-window branch). Single sample: what ligand fragments does an "
        "average decoded read carry, how do the structural classes compare, and how clean are "
        "the calls. Charts are interactive — hover for exact numbers, click legend entries to "
        "toggle classes, drag to zoom.</p>",
        provenance(d),
    ]

    # Zero decoded reads is a legitimate outcome, not an error: emit a small, honest report
    # (the figures would all be empty and the whole expected set would list as "unseen").
    if d["n_reads"] == 0:
        H = head + [zero_note(d), stat_tiles(d), "</body>", "</html>"]
        OUT.write_text("\n".join(x for x in H if x))
        print(f"wrote {OUT}  (no reads decoded)")
        return

    H = head + [
        '<div class="callout aim"><p><b>Aim.</b> For every ligand fragment, how many copies does '
        "an average decoded read carry (<b>mean copies / read</b> = total calls ÷ reads decoded)? "
        "Everything is <b>strand-collapsed</b> (a feature and its <code>_rc</code> are one row) and "
        "split into <b>ligand fragments</b> vs the <b>UMI anchors</b> (backbone, one per "
        "read by construction). The construct vocabulary and its classes are derived from the "
        f"decode itself — {n_payload_feats} fragment features across {n_classes} structural "
        "classes.</p></div>",

        stat_tiles(d),
        absent_note,

        "<h2>Every fragment feature</h2>",
        "<p>Mean copies per decoded read for each ligand fragment (log axis), grouped and coloured "
        "by structural class. Longer/unique fragments are the trustworthy signal; very short "
        "Linear calls are partly mis-segmentation, so read the profile by class.</p>",
        f'<p class="small">Classes: {legend}</p>',
        fig_master(d, first=True),

        "<h2>Backbone — UMI anchors</h2>",
        "<p>The two UMI anchors bracket every read, so ~1 copy/read each is expected; a value above "
        "1 flags concatemeric (multi-construct) reads.</p>",
        fig_backbone(d),

        "<h2>By structural class</h2>",
        "<p>Total fragment calls contributed by each class (log axis; label = share of all fragment "
        "calls). Hover for the summed mean-copies/read and feature count.</p>",
        fig_classes(d),

        "<h2>Multiplicity &amp; quality</h2>",
        "<p>Left: how many fragment copies a read carries (0 = amplification loop-only "
        "reads). Right: edit-distance of fragment calls against the expected sequences "
        "(lower = cleaner match).</p>",
        '<div style="display:flex;gap:1rem;flex-wrap:wrap;align-items:flex-start">',
        f'<div style="flex:1;min-width:320px">{fig_perread(d)}</div>',
        f'<div style="flex:1;min-width:320px">{fig_editdist(d)}</div>',
        "</div>",

        "<h2>Size vs abundance</h2>",
        "<p>Mean copies/read against nominal fragment size, per class. A downward trend toward short "
        "fragments is the tell-tale of over-calling / mis-segmentation of small motifs.</p>",
        fig_length(d),

        "<h2>Per-feature table</h2>",
        feature_table(d),

        "<h2>Method &amp; caveats</h2>",
        '<dl class="gloss">',
        "<dt>mean copies / read</dt><dd>Σ calls of a feature ÷ reads decoded "
        "(<code>summary.tsv n_reads</code>). Strand-collapsed (fwd + <code>_rc</code>).</dd>",
        "<dt>reads decoded</dt><dd>UMI-anchored reads that fused and passed the anchor-window trim; "
        "not total sequenced reads.</dd>",
        "<dt>fragment vs anchor</dt><dd>UMI anchors are the backbone, read from the "
        "tool's <code>umis.parquet</code> and reported separately from the ligand fragments.</dd>",
        "<dt>class</dt><dd>Tokenised from each feature name (size/strand tokens stripped); nothing "
        "is hard-coded, so the class set is whatever the construct library contains.</dd>",
        "<dt>edit distance</dt><dd>Per-call edit distance to the expected sequence emitted by the "
        "decoder; the per-feature column is the median.</dd>",
        "</dl>",
        '<p class="small">Source: <code>' + esc(FC.name) + "</code> + <code>" + esc(FRAGS.name)
        + "</code> → <code>workflow/build_featurecount_report_2026.py</code> "
        "(vocabulary via <code>workflow/ligand_vocab.py</code>).</p>",
        "</body>", "</html>",
    ]
    OUT.write_text("\n".join(x for x in H if x))
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
