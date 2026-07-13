#!/usr/bin/env python
"""Cross-sample UMI characterisation report (branch `umi`, mode `correct`).

Descriptive workflow QC over the per-sample corrected-UMI outputs (`umi/<s>/<s>.umi_species.tsv`,
`.umi_summary.tsv`, `.fused.parquet`). One self-contained HTML file, plotly figures inlined.

Panels (no biological interpretation — this is a workflow):
  * Summary        per-sample reads in / filtered (by class) / corrected (species merged, reads relabelled).
  * Abundance      rank-abundance curves (log-log) and D50/D90 = the proportion of distinct UMI
                   species whose cumulative reads reach 50% / 90% (+ a Lorenz-style curve).
  * Overlap        per experimental group, an UpSet-like view of distinct corrected-UMI species shared
                   across the group's samples (top-N intersections).
  * Validity QC    UMI-group ligand-signature purity vs group size (join corrected UMI -> decode
                   fragments), and a collision null model (expected chance ed<=1 merges vs observed).

Parameterised entirely by environment variables (set by rule report_umi_cross):
    L2F_BASE          umi/ dir to glob `*/` for experimental sample subdirs
    L2F_R0_STEM       umi_initial/.../<INITIAL> stem for the R0 baseline ("" disables it)
    L2F_SAMPLES       config-derived metadata TSV (name<TAB>group<TAB>round<TAB>idx)
    L2F_DECODE_BASE   decode/ dir (experimental fragments for the purity QC)
    L2F_R0_FRAGS      the initial library's decode fragments.parquet ("" if no baseline)
"""
from __future__ import annotations

import html
import math
import os
import sys
from pathlib import Path

import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import get_plotlyjs

BASE = Path(os.environ.get("L2F_BASE", "")).resolve()
R0_STEM = os.environ.get("L2F_R0_STEM", "").strip()
SAMPLES_TSV = os.environ.get("L2F_SAMPLES", "").strip()
DECODE_BASE = os.environ.get("L2F_DECODE_BASE", "").strip()
R0_FRAGS = os.environ.get("L2F_R0_FRAGS", "").strip()
OUT = BASE / "umi_report_cross.html"

PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
           "#EECA3B", "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC"]
GREY = "#9aa0a6"


def esc(s) -> str:
    return html.escape(str(s))


def fig_html(fig) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       default_width="100%", config={"displaylogo": False, "responsive": True})


def style(fig, height, title=None):
    fig.update_layout(
        height=height, title=title, template="plotly_white",
        margin=dict(l=60, r=20, t=50 if title else 20, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        font=dict(size=12),
    )
    return fig


# --------------------------------------------------------------------------- discovery + metadata
def load_meta():
    meta = {}
    if SAMPLES_TSV and Path(SAMPLES_TSV).exists():
        for i, line in enumerate(Path(SAMPLES_TSV).read_text().splitlines()):
            if i == 0 or not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                meta[parts[0]] = {"group": parts[1], "round": _int(parts[2]), "idx": _int(parts[3]) if len(parts) > 3 else i}
    return meta


def _int(x, default=0):
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


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
            samples.insert(0, {
                "name": r0.name, "dir": r0.parent, "species": r0.parent / f"{r0.name}.umi_species.tsv",
                "summary": r0.parent / f"{r0.name}.umi_summary.tsv",
                "fused": r0.parent / f"{r0.name}.fused.parquet",
                "frags": Path(R0_FRAGS) if R0_FRAGS else None,
                "group": "initial", "round": 0, "idx": -1, "is_baseline": True,
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


def group_color(samples):
    groups = list(dict.fromkeys(s["group"] for s in samples))
    return {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(groups)}


# --------------------------------------------------------------------------- panels
def summary_table(samples, summaries):
    cols = ["reads_in", "dropped_nonacgt", "dropped_lowcomplexity", "dropped_umilen",
            "dropped_total", "reads_umi_ok", "dropped_no_payload", "reads_kept",
            "n_species_raw", "n_species_corrected", "n_species_merged", "n_reads_relabeled"]
    head = "".join(f"<th>{esc(c)}</th>" for c in cols)
    rows = []
    for s in samples:
        d = summaries.get(s["name"], {})
        cells = "".join(f"<td class='num'>{d.get(c, ''):,}</td>" if isinstance(d.get(c), int)
                        else f"<td class='num'>{esc(d.get(c, ''))}</td>" for c in cols)
        rows.append(f"<tr><td>{esc(s['name'])}</td><td>{esc(s['group'])}</td>"
                    f"<td class='num'>{s['round']}</td>{cells}</tr>")
    return (f"<h2>Filter &amp; correction summary</h2><div style='overflow-x:auto'><table>"
            f"<thead><tr><th>sample</th><th>group</th><th>round</th>{head}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def _plot_idx(n, cap=2000):
    """Log-spaced indices over [0, n-1] so a huge rank-abundance/Lorenz curve stays light to draw
    without changing its shape (metrics are computed on the full data, only the drawn line thins)."""
    if n <= cap:
        return list(range(n))
    step = math.log(n) / (cap - 1)
    return sorted({0, n - 1} | {min(n - 1, int(math.exp(step * i)) - 1) for i in range(cap)})


def fig_rank_abundance(samples, species, gcol):
    fig = go.Figure()
    for s in samples:
        df = species[s["name"]].sort("n_reads", descending=True)
        y = df["n_reads"].to_list()
        if not y:
            continue
        idx = _plot_idx(len(y))
        fig.add_trace(go.Scatter(
            x=[i + 1 for i in idx], y=[y[i] for i in idx], mode="lines", name=esc(s["name"]),
            line=dict(color=gcol.get(s["group"], GREY), width=1.6,
                      dash="dot" if s["is_baseline"] else "solid"),
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
    names, d50, d90 = [], [], []
    lorenz = go.Figure()
    for s in samples:
        cd = species[s["name"]].sort("n_reads", descending=True)["n_reads"].to_list()
        if not cd:
            continue
        names.append(s["name"])
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
        xs = [xs_full[i] for i in idx]
        ys = [ys_full[i] for i in idx]
        lorenz.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", name=esc(s["name"]),
            line=dict(color=gcol.get(s["group"], GREY), width=1.4,
                      dash="dot" if s["is_baseline"] else "solid"),
        ))
    lorenz.add_shape(type="line", x0=0, y0=0.5, x1=1, y1=0.5, line=dict(color=GREY, dash="dash", width=1))
    lorenz.add_shape(type="line", x0=0, y0=0.9, x1=1, y1=0.9, line=dict(color=GREY, dash="dash", width=1))
    lorenz.update_xaxes(title="cumulative fraction of species (ranked by abundance)")
    lorenz.update_yaxes(title="cumulative fraction of reads")
    bars = go.Figure()
    bars.add_trace(go.Bar(x=names, y=d50, name="D50", marker_color="#4C78A8"))
    bars.add_trace(go.Bar(x=names, y=d90, name="D90", marker_color="#F58518"))
    bars.update_yaxes(title="% of distinct species reaching 50% / 90% of reads")
    bars.update_layout(barmode="group")
    return (fig_html(style(bars, 420, "Complexity: D50 / D90 (lower = more concentrated)"))
            + fig_html(style(lorenz, 460, "Cumulative abundance (Lorenz)")))


def fig_upset(group, gsamples, species, top_n=24):
    long = pl.concat([
        species[nm].select(pl.col("umi").alias("fused_umi")).with_columns(pl.lit(nm).alias("sample"))
        for nm in gsamples if species[nm].height
    ]) if any(species[nm].height for nm in gsamples) else None
    if long is None or long.height == 0:
        return f"<p class='small'>No UMI species for group {esc(group)}.</p>"
    bitmap = pl.DataFrame({"sample": gsamples, "bitval": [1 << i for i in range(len(gsamples))]})
    patt = long.join(bitmap, on="sample").group_by("fused_umi").agg(pl.col("bitval").sum().alias("mask"))
    counts = patt.group_by("mask").agg(pl.len().alias("size")).sort("size", descending=True)
    rows = counts.head(top_n).iter_rows()  # (mask, size)
    masks = []
    for mask, size in rows:
        members = [j for j in range(len(gsamples)) if mask & (1 << j)]
        masks.append((members, size))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.04)
    xs = list(range(len(masks)))
    fig.add_trace(go.Bar(x=xs, y=[m[1] for m in masks], marker_color="#4C78A8",
                         hovertemplate="%{y} species<extra></extra>"), row=1, col=1)
    for xi, (members, _sz) in enumerate(masks):
        # faint dots for all rows, solid for members, connected
        fig.add_trace(go.Scatter(x=[xi] * len(gsamples), y=list(range(len(gsamples))), mode="markers",
                                 marker=dict(color="#e6e6e6", size=9), showlegend=False,
                                 hoverinfo="skip"), row=2, col=1)
        if members:
            fig.add_trace(go.Scatter(x=[xi] * len(members), y=members, mode="markers+lines",
                                     marker=dict(color="#333", size=10), line=dict(color="#333", width=2),
                                     showlegend=False, hoverinfo="skip"), row=2, col=1)
    fig.update_yaxes(title="species", row=1, col=1)
    fig.update_yaxes(tickmode="array", tickvals=list(range(len(gsamples))),
                     ticktext=[esc(x) for x in gsamples], row=2, col=1, autorange="reversed")
    fig.update_xaxes(showticklabels=False, row=2, col=1)
    return fig_html(style(fig, 460, f"Shared corrected-UMI species across {esc(group)} (top {len(masks)} intersections)"))


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


def fig_purity(samples):
    parts = []
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
                   .agg(pl.len().alias("size"),
                        pl.col("sig").value_counts().alias("_vc")))
            # dominant signature share per UMI group
            grp = grp.with_columns(
                pl.col("_vc").list.eval(pl.element().struct.field("count").max()).list.first().alias("dom")
            ).with_columns((pl.col("dom") / pl.col("size")).alias("purity")).drop("_vc")
            parts.append(grp.select("size", "purity"))
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f"[purity] skip {s['name']}: {exc}\n")
    if not parts:
        return "<p class='small'>Purity QC skipped (decode fragments not available).</p>"
    allg = pl.concat(parts).filter(pl.col("size") >= 2)
    if allg.height == 0:
        return "<p class='small'>No multi-read UMI groups for the purity QC.</p>"
    bins = [(2, 2), (3, 4), (5, 9), (10, 29), (30, 99), (100, 10 ** 9)]
    labels, frac_pure, mean_pur, ngroups = [], [], [], []
    for lo, hi in bins:
        sub = allg.filter((pl.col("size") >= lo) & (pl.col("size") <= hi))
        if sub.height == 0:
            continue
        labels.append(f"{lo}" if lo == hi else (f"{lo}+" if hi > 10 ** 8 else f"{lo}-{hi}"))
        frac_pure.append(100 * sub.filter(pl.col("purity") >= 0.999).height / sub.height)
        mean_pur.append(100 * float(sub["purity"].mean()))
        ngroups.append(sub.height)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=frac_pure, name="% pure groups", marker_color="#54A24B",
                         customdata=ngroups, hovertemplate="%{y:.1f}% pure<br>%{customdata} groups<extra></extra>"))
    fig.add_trace(go.Scatter(x=labels, y=mean_pur, name="mean purity", mode="lines+markers",
                             line=dict(color="#E45756")))
    fig.update_xaxes(title="UMI group size (reads)")
    fig.update_yaxes(title="% (pure groups / mean purity)", range=[0, 105])
    return fig_html(style(fig, 420, "Validity: ligand-signature purity vs UMI group size"))


def fig_collision(samples, species, summaries):
    names, observed, expected = [], [], []
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
        names.append(s["name"])
        observed.append(int(merged))
        expected.append(exp)
    if not names:
        return "<p class='small'>Collision null model unavailable.</p>"
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=observed, name="observed ed≤1 merges", marker_color="#4C78A8"))
    fig.add_trace(go.Bar(x=names, y=expected, name="expected by chance", marker_color=GREY))
    fig.update_yaxes(title="UMI merges", type="log")
    fig.update_layout(barmode="group")
    return fig_html(style(fig, 400, "Collision null model: observed corrections vs chance"))


# --------------------------------------------------------------------------- assemble
def main():
    if not BASE or not BASE.exists():
        sys.stderr.write(f"[FATAL] L2F_BASE not found: {BASE}\n")
        return 1
    meta = load_meta()
    samples = discover(meta)
    if not samples:
        OUT.write_text("<!doctype html><meta charset='utf-8'><p>No UMI samples discovered.</p>")
        print(f"[umi report] no samples -> {OUT}")
        return 0
    species = {s["name"]: load_species(s) for s in samples}
    summaries = {s["name"]: load_summary(s) for s in samples}
    gcol = group_color(samples)

    H = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>UMI characterisation — cross-sample</title>",
        _STYLE,
        f"<script>{get_plotlyjs()}</script>",
        "</head><body>",
        "<h1>UMI characterisation — cross-sample</h1>",
        "<p class='small'>Descriptive QC over corrected UMI species (mode <code>correct</code>). "
        "This is a workflow output — no biological conclusions are drawn.</p>",
        summary_table(samples, summaries),
        "<h2>Abundance</h2>",
        fig_rank_abundance(samples, species, gcol),
        fig_complexity(samples, species, gcol),
        "<h2>Species overlap</h2>",
    ]
    exp_groups = list(dict.fromkeys(s["group"] for s in samples if not s["is_baseline"]))
    base_name = next((s["name"] for s in samples if s["is_baseline"]), None)
    for g in exp_groups:
        gs = ([base_name] if base_name else []) + [s["name"] for s in samples
                                                   if s["group"] == g and not s["is_baseline"]]
        H.append(fig_upset(g, gs, species))
    H += [
        "<h2>Validity QC</h2>",
        fig_purity(samples),
        fig_collision(samples, species, summaries),
        "</body></html>",
    ]
    OUT.write_text("\n".join(x for x in H if x))
    print(f"[umi report] {len(samples)} samples -> {OUT}")
    return 0


_STYLE = """<style>
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:2rem auto;max-width:1100px;padding:0 1rem;color:#1a1a1a}
h1{font-size:1.6rem}h2{font-size:1.2rem;margin-top:2rem;border-bottom:1px solid #eee;padding-bottom:.3rem}
table{border-collapse:collapse;font-size:.85rem}th,td{border:1px solid #e2e2e2;padding:.25rem .5rem;text-align:left}
th{background:#f6f8fa}td.num{text-align:right;font-variant-numeric:tabular-nums}
.small{color:#666;font-size:.85rem}code{background:#f0f2f4;padding:.05rem .3rem;border-radius:3px}
@media(prefers-color-scheme:dark){body{background:#15181c;color:#e6e6e6}th{background:#20242a}
th,td{border-color:#333}h2{border-color:#333}code{background:#20242a}}
</style>"""


if __name__ == "__main__":
    raise SystemExit(main())
