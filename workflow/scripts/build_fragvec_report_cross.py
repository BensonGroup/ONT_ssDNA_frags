#!/usr/bin/env python
"""Cross-sample fragment-vector report: the two-axis readout (abundance + incidence).

Descriptive workflow QC. One self-contained HTML file, plotly inlined.

Panels:
  * Summary / complexity collapse -- distinct fragment vectors per sample. Selection COLLAPSES the
    library (A549: ~103k vectors at R1 -> ~10k at R5) and concentrates reads onto fewer vectors.
  * Incidence vs abundance -- n_umis (independent clones) against n_reads, per vector. Points near the
    diagonal are carried by MANY clones (reproducible selection); points far BELOW are many reads from
    FEW clones (a single amplified lineage / PCR jackpot) and must not be read as enrichment.
  * Jackpot table -- the vectors that look enriched but rest on one or two clones.
  * Enrichment trajectories -- depth-normalised counts across rounds (Kuai et al. 2018:
    n_reads * n_vectors / total_reads), R0 as baseline. Sub-threshold counts are greyed: single-digit
    counts do not reproduce between replicates.
  * Cross-arm stratification -- for vectors enriched in BOTH cell lines, is the agreement CLONE-SHARED
    (weak: both arms inherited the same clones from the pre-split amplified pool) or CLONE-INDEPENDENT
    (strong: different clones reached the same vector in each arm)?

Env (set by rule report_fragvec_cross):
    L2F_BASE       fragvec/ dir (globbed for sample subdirs + <group>.fragvec_x_sample.long.parquet)
    L2F_R0_STEM    fragvec_initial/.../<INITIAL> stem ("" disables the baseline)
    L2F_SAMPLES    metadata TSV (name<TAB>group<TAB>round<TAB>idx)
    L2F_MIN_READS  noise threshold (counts below this are flagged, not dropped)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import polars as pl
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

sys.path.insert(0, str(Path(__file__).resolve().parent))
# DEFERRED: unlike the umi branch, this report still RECOMPUTES c_eff (see fig_collapse). Giving it
# the same treatment -- a `fragvec_complexity` rule emitting a TSV that this file merely renders --
# is the symmetric follow-up; the UMI branch is where the C_eff work landed first.
from complexity_lib import c_eff, rarefy   # depth-robust complexity; see complexity_lib docstring
from report_lib import (
    BASELINE_GROUP, GREY, color_key_html, esc, fig_html, first_of_group, group_color, load_meta,
)

BASE = Path(os.environ.get("L2F_BASE", "")).resolve()
R0_STEM = os.environ.get("L2F_R0_STEM", "").strip()
SAMPLES_TSV = os.environ.get("L2F_SAMPLES", "").strip()
MIN_READS = int(os.environ.get("L2F_MIN_READS", "10"))
OUT = BASE / "fragvec_report_cross.html"

# Colours the top-N VECTORS in fig_trajectories -- one hue per vector, indexed by rank. That is a
# different axis from the group palette and must stay a separate list. Do not "fix" it to
# group_color(): the trajectory panel is already per-group, and inside it the thing being told apart
# is the vector, not the sample.
VECTOR_PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
                  "#EECA3B", "#B279A2", "#FF9DA6"]


def style(fig, height, title=None, top=None):
    # Title pinned to the CONTAINER top, not floated in the middle of the top margin: the legend is
    # horizontal at y=1.02 and grows UPWARD, so plotly's default (baseline at the centre of the top
    # margin) puts the title exactly where a multi-row legend lands. `top` reserves the band that
    # legend needs -- see fig_trajectories.
    fig.update_layout(
        height=height, template="plotly_white",
        title=(dict(text=title, y=1.0, yanchor="top", yref="container", pad=dict(t=6))
               if title else None),
        # 64 = title band (~28) + one legend row (~19) + slack. The old 50 left the title's
        # descender sitting on the legend's top edge; `top` buys more for a multi-row legend.
        margin=dict(l=60, r=20, t=(top or (64 if title else 20)), b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), font=dict(size=12))
    return fig


def discover(meta):
    out = []
    for d in sorted(p for p in BASE.glob("*/") if p.is_dir()):
        n = d.name
        c = d / f"{n}.fragvec_counts.parquet"
        if not c.exists():
            continue
        mm = meta.get(n, {})
        out.append({"name": n, "counts": c, "summary": d / f"{n}.fragvec_summary.tsv",
                    "clones": d / f"{n}.fragvec_clones.parquet",
                    "group": mm.get("group", n), "round": mm.get("round", 0),
                    "idx": mm.get("idx", 9999), "is_baseline": False})
    out.sort(key=lambda s: (s["idx"], s["round"], s["name"]))
    if R0_STEM:
        r0 = Path(R0_STEM)
        if (r0.parent / f"{r0.name}.fragvec_counts.parquet").exists():
            mm = meta.get(r0.name, {})
            out.insert(0, {"name": r0.name, "counts": r0.parent / f"{r0.name}.fragvec_counts.parquet",
                           "summary": r0.parent / f"{r0.name}.fragvec_summary.tsv",
                           "clones": r0.parent / f"{r0.name}.fragvec_clones.parquet",
                           # from the config like everyone else's; the literals are only the fallback
                           # for a standalone run with no metadata TSV. `is_baseline` is what every
                           # consumer keys on -- there is no `role` field.
                           "group": mm.get("group", BASELINE_GROUP), "round": mm.get("round", 0),
                           "idx": -1, "is_baseline": True})
    return out


def summary_table(samples, summ, gcol):
    cols = ["reads_decoded", "reads_payload", "n_vectors", "c_eff", "n_over_c_eff",
            "mean_reads_per_vector", "max_reads_vector", "n_vectors_ge_min",
            "pct_reads_in_vectors_ge_min"]
    # mark the columns that are properties of the RUN, not of the library
    depth_dep = {"n_vectors", "n_vectors_ge_min", "pct_reads_in_vectors_ge_min"}
    head = "".join(
        f"<th title='DEPTH-DEPENDENT — do not compare across samples'>{esc(c)} ⚠</th>"
        if c in depth_dep else f"<th>{esc(c)}</th>" for c in cols)
    rows = []
    for s in samples:
        d = summ.get(s["name"], {})
        cells = "".join(f"<td class='num'>{d.get(c, ''):,}</td>" if isinstance(d.get(c), int)
                        else f"<td class='num'>{esc(d.get(c, ''))}</td>" for c in cols)
        swatch = (f"<span style='display:inline-block;width:.7rem;height:.7rem;border-radius:2px;"
                  f"margin-right:.4rem;background:{gcol.get(s['group'], GREY)}'></span>")
        rows.append(f"<tr><td>{esc(s['name'])}</td><td>{swatch}{esc(s['group'])}</td>{cells}</tr>")
    return (f"<h2>Complexity &amp; depth</h2><p class='small'>Only vectors with &ge;{MIN_READS} reads are "
            f"analysable — single-digit counts do not reproduce between replicates. Columns marked ⚠ are "
            f"RICHNESS counts: they grow with sequencing depth and must <b>not</b> be compared across "
            f"samples. Use <code>c_eff</code> (depth-robust) or the rarefied counts in the next panel."
            f"</p><div style='overflow-x:auto'><table><thead><tr><th>sample</th>"
            f"<th>group</th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>")


def fig_collapse(samples, counts, gcol):
    """Complexity, measured DEPTH-ROBUSTLY.

    The raw count of distinct vectors is a property of the RUN, not the library: it grows with
    sequencing depth. Depth here spans ~6x, so the raw count cannot be compared across samples. We show
    instead (a) richness RAREFIED to the shallowest sample's depth, and (b) c_eff, which estimates the
    pool's collision probability and does not inflate with depth. The raw count is drawn only faintly,
    to make the size of the artefact visible.
    """
    names = [s["name"] for s in samples]
    vecs = {n: counts[n]["n_reads"].to_list() for n in names}
    depths = {n: sum(vecs[n]) for n in names}
    m = min(depths.values())                       # common (shallowest) depth
    raw = [len(vecs[n]) for n in names]
    rar = [rarefy(vecs[n], m) for n in names]
    ce = [c_eff(vecs[n])[3] for n in names]

    colors = [gcol.get(s["group"], GREY) for s in samples]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=rar, name=f"distinct vectors, RAREFIED to {m:,} reads",
                         marker=dict(color=colors)))
    fig.add_trace(go.Bar(x=names, y=raw, name="raw distinct vectors (DEPTH-DEPENDENT — do not compare)",
                         marker=dict(color=colors, opacity=0.35,
                                     pattern=dict(shape="/", solidity=0.35))))
    fig.add_trace(go.Scatter(x=names, y=ce, name="c_eff (depth-robust)", yaxis="y2",
                             mode="lines+markers", line=dict(color=GREY, width=2, dash="dot"),
                             marker=dict(size=7, color=colors)))
    fig.update_layout(barmode="group",
                      yaxis=dict(title="distinct fragment vectors"),
                      yaxis2=dict(title="c_eff", overlaying="y", side="right", type="log"))
    note = (f"<p class='small'><b>Depth varies across these samples "
            f"({min(depths.values()):,}–{max(depths.values()):,} payload reads, "
            f"{max(depths.values())/min(depths.values()):.1f}×).</b> A raw species count grows with "
            f"depth, so the grey bars are <em>not</em> comparable across samples — they are shown only "
            f"to expose the size of the artefact. Compare the blue bars (rarefied to a common "
            f"{m:,} reads) or the red line (c_eff). Rule of thumb: <em>if a number gets bigger when you "
            f"sequence more, it is not a property of your library.</em></p>")
    return note + fig_html(style(fig, 460, "Library complexity — depth-corrected"))


def fig_anchor_only(samples, summ, gcol):
    """Reads with a valid UMI and both anchors but NO payload fragment. They are dropped (they decode no
    ligand), but the fraction VARIES by round — a systematic drift would bias enrichment."""
    names, pct, colors = [], [], []
    for s in samples:
        d = summ.get(s["name"], {})
        dec, drop = d.get("reads_decoded"), d.get("reads_dropped_anchor_only")
        if not dec or drop is None:
            continue
        names.append(s["name"])
        pct.append(100.0 * drop / dec)
        colors.append(gcol.get(s["group"], GREY))
    if not names:
        return ""
    fig = go.Figure(go.Bar(x=names, y=pct, marker=dict(color=colors),
                           hovertemplate="%{y:.1f}% anchor-only<extra></extra>"))
    fig.update_yaxes(title="% of decoded reads with NO payload fragment")
    return ("<h3>Anchor-only reads (dropped)</h3><p class='small'>Reads carrying a valid UMI and both "
            "backbone anchors but <b>no payload fragment</b> — they decode no ligand, so they are "
            "excluded from every count (declared as <code>reads_dropped_anchor_only</code>). Watch for "
            "systematic drift across rounds: if this fraction moves with selection, it biases "
            "enrichment.</p>" + fig_html(style(fig, 340)))


def fig_incidence(samples, counts, gcol):
    """n_umis (independent clones) vs n_reads. Diagonal = trustworthy; far below = jackpot."""
    fig = go.Figure()
    for s, first in zip(samples, first_of_group(samples)):
        df = counts[s["name"]].filter(pl.col("n_reads") >= MIN_READS)
        if df.height == 0:
            continue
        fig.add_trace(go.Scattergl(
            x=df["n_reads"].to_list(), y=df["n_umis"].to_list(), mode="markers",
            name=esc(s["group"]), legendgroup=s["group"], showlegend=first,
            marker=dict(size=4, opacity=0.45, color=gcol.get(s["group"], GREY)),
            hovertemplate="reads %{x}<br>clones %{y}<extra>" + esc(s["name"]) + "</extra>"))
    lo, hi = MIN_READS, max((int(counts[s["name"]]["n_reads"].max() or 1)) for s in samples)
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="1 read / clone",
                             line=dict(color=GREY, dash="dash")))
    fig.update_xaxes(type="log", title="n_reads  (abundance)")
    fig.update_yaxes(type="log", title="n_umis  (incidence — independent clones)")
    return fig_html(style(fig, 520,
        "Incidence vs abundance — on the dashed line = many independent clones (trustworthy); "
        "far below = few clones carrying many reads (jackpot)"))


def jackpot_table(samples, counts, top=25):
    rows = []
    for s in samples:
        df = (counts[s["name"]].filter((pl.col("n_reads") >= MIN_READS) & (pl.col("n_umis") <= 2))
              .sort("n_reads", descending=True).head(top))
        for r in df.iter_rows(named=True):
            rows.append((s["name"], r["vector"], r["n_reads"], r["n_umis"], r["max_reads_per_umi"]))
    rows.sort(key=lambda x: -x[2])
    if not rows:
        return "<p class='small'>No jackpot vectors (every vector above threshold rests on ≥3 clones).</p>"
    body = "".join(
        f"<tr><td>{esc(a)}</td><td style='font-family:monospace;font-size:.75rem'>{esc(b[:70])}"
        f"{'…' if len(b) > 70 else ''}</td><td class='num'>{c:,}</td><td class='num'>{d}</td>"
        f"<td class='num'>{e:,}</td></tr>" for a, b, c, d, e in rows[:top])
    return (f"<h3>Jackpot flags</h3><p class='small'>Vectors with ≥{MIN_READS} reads but ≤2 clones — "
            f"the reads come from one amplified lineage, so the apparent enrichment is NOT independent "
            f"evidence.</p><div style='overflow-x:auto'><table><thead><tr><th>sample</th><th>vector</th>"
            f"<th>n_reads</th><th>n_umis</th><th>max_reads_per_umi</th></tr></thead>"
            f"<tbody>{body}</tbody></table></div>")


def fig_trajectories(group, longp, top=15):
    """Trajectories on DEPTH-NORMALISED FREQUENCY (reads per million).

    Normalise by depth ONLY. Do not scale by a sample's observed richness: that factor differs per
    sample (it swings ~4.5x here) and is itself depth-dependent, so it would distort the very
    cross-sample comparison this panel exists to make. A vector held at constant frequency must draw a
    flat line -- that is the correctness test for this panel.

    COLOUR MEANS VECTOR HERE, NOT GROUP. This panel is already per-group (one figure per arm), and the
    thing being told apart inside it is the top-N vectors -- so VECTOR_PALETTE is indexed by rank, and
    that is correct. Do not "unify" it with group_color().
    """
    df = pl.read_parquet(longp)
    order = [s for s in df["sample"].unique(maintain_order=True).to_list()]
    tot = df.group_by("sample").agg(pl.col("n_reads").sum().alias("tot"))
    d = df.join(tot, on="sample").with_columns(
        (1e6 * pl.col("n_reads") / pl.col("tot")).alias("rpm"))
    last = order[-1]
    tops = (d.filter(pl.col("sample") == last).sort("rpm", descending=True)
            .head(top)["vector"].to_list())
    fig = go.Figure()
    for i, v in enumerate(tops):
        sub = d.filter(pl.col("vector") == v)
        m = dict(zip(sub["sample"].to_list(), zip(sub["rpm"].to_list(), sub["n_reads"].to_list())))
        ys = [m.get(s, (0, 0))[0] for s in order]
        nr = [m.get(s, (0, 0))[1] for s in order]
        fig.add_trace(go.Scatter(
            x=order, y=ys, mode="lines+markers", name=v[:28] + ("…" if len(v) > 28 else ""),
            line=dict(color=VECTOR_PALETTE[i % len(VECTOR_PALETTE)], width=1.6),
            marker=dict(size=[9 if n >= MIN_READS else 5 for n in nr],
                        symbol=["circle" if n >= MIN_READS else "circle-open" for n in nr]),
            customdata=nr,
            hovertemplate="%{x}<br>%{y:.1f} RPM<br>%{customdata} reads<extra></extra>"))
    fig.update_yaxes(type="log", title="reads per million payload reads (depth-normalised)")
    # One legend entry per vector, ~4 per row at this width; the legend grows UPWARD out of the plot
    # area, so reserve its band explicitly or it swallows the title. Height stays 480 -- the plot
    # area pays for the band.
    rows = -(-len(tops) // 4)
    return fig_html(style(fig, 480,
        f"{esc(group)} — top {len(tops)} vectors at {esc(order[-1])} (open markers = below "
        f"{MIN_READS} reads: noise)", top=34 + 19 * rows + 10))


def cross_arm(samples, groups):
    """For vectors above threshold in BOTH arms: is the agreement clone-SHARED or clone-INDEPENDENT?"""
    if len(groups) < 2:
        return ""
    ga, gb = groups[0], groups[1]

    def arm_clones(g):
        acc = {}
        for s in samples:
            if s["group"] != g or not s["clones"].exists():
                continue
            df = pl.read_parquet(s["clones"], columns=["vector", "umi"])
            for v, u in df.iter_rows():
                acc.setdefault(v, set()).add(u)
        return acc

    A, B = arm_clones(ga), arm_clones(gb)
    shared_vecs = set(A) & set(B)
    if not shared_vecs:
        return f"<p class='small'>No vectors above threshold in both {esc(ga)} and {esc(gb)}.</p>"
    rows = []
    for v in shared_vecs:
        ca, cb = A[v], B[v]
        inter = len(ca & cb)
        jac = inter / len(ca | cb)
        rows.append((v, len(ca), len(cb), inter, jac))
    rows.sort(key=lambda x: -(x[1] + x[2]))
    jacs = [r[4] for r in rows]
    indep = sum(1 for j in jacs if j < 0.01)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=jacs, nbinsx=40, marker_color="#54A24B"))
    fig.update_xaxes(title="clone-overlap (Jaccard) between the two arms, per shared vector")
    fig.update_yaxes(title="vectors")
    h = fig_html(style(fig, 380, f"Cross-arm agreement: {esc(ga)} vs {esc(gb)}"))
    body = "".join(
        f"<tr><td style='font-family:monospace;font-size:.75rem'>{esc(v[:60])}{'…' if len(v)>60 else ''}</td>"
        f"<td class='num'>{a:,}</td><td class='num'>{b:,}</td><td class='num'>{i:,}</td>"
        f"<td class='num'>{j:.3f}</td><td>{'clone-SHARED (weak)' if j >= 0.01 else 'clone-INDEPENDENT (strong)'}</td></tr>"
        for v, a, b, i, j in rows[:20])
    return (f"<h2>Cross-arm stratification</h2><p class='small'>The library was amplified once and then "
            f"split, so both arms inherited the SAME clones — a vector enriched in both arms via the "
            f"<em>same</em> clones is correlated ancestry (weak evidence). Via <em>different</em> clones "
            f"it is independent evidence of a real, cell-line-independent effect. "
            f"<b>{indep:,} / {len(rows):,}</b> shared vectors are clone-independent.</p>{h}"
            f"<div style='overflow-x:auto'><table><thead><tr><th>vector</th><th>clones {esc(ga)}</th>"
            f"<th>clones {esc(gb)}</th><th>shared</th><th>Jaccard</th><th>verdict</th></tr></thead>"
            f"<tbody>{body}</tbody></table></div>")


def main():
    if not BASE.exists():
        sys.stderr.write(f"[FATAL] L2F_BASE not found: {BASE}\n")
        return 1
    meta = load_meta(SAMPLES_TSV)
    samples = discover(meta)
    if not samples:
        OUT.write_text("<!doctype html><meta charset='utf-8'><p>No fragvec samples discovered.</p>")
        return 0
    counts = {s["name"]: pl.read_parquet(s["counts"]) for s in samples}
    summ = {}
    for s in samples:
        if s["summary"].exists():
            d = pl.read_csv(s["summary"], separator="\t")
            summ[s["name"]] = d.row(0, named=True) if d.height else {}
    groups = [g for g in dict.fromkeys(s["group"] for s in samples if not s["is_baseline"])]
    gcol = group_color(samples)     # same colour per cell line as the other two cross reports

    H = ["<!doctype html><html><head><meta charset='utf-8'>",
         "<meta name='viewport' content='width=device-width, initial-scale=1'>",
         "<title>Fragment-vector quantification — cross-sample</title>", _STYLE,
         f"<script>{get_plotlyjs()}</script></head><body>",
         "<h1>Fragment-vector quantification — cross-sample</h1>",
         "<p class='small'>Two axes per vector: <b>n_reads</b> (abundance) and <b>n_umis</b> "
         "(incidence — independent clones). A vector is only trustworthy when its reads come from many "
         "clones. Descriptive workflow QC — no biological conclusions are drawn. Figures are coloured "
         "by the config <code>group</code>; the initial library is the shared R0 baseline, in grey.</p>",
         color_key_html(gcol, samples),
         summary_table(samples, summ, gcol),
         fig_collapse(samples, counts, gcol),
         fig_anchor_only(samples, summ, gcol),
         "<h2>Incidence vs abundance (jackpot control)</h2>",
         fig_incidence(samples, counts, gcol),
         jackpot_table(samples, counts),
         "<h2>Enrichment trajectories</h2>"]
    for g in groups:
        lp = BASE / f"{g}.fragvec_x_sample.long.parquet"
        if lp.exists():
            H.append(fig_trajectories(g, lp))
    H.append(cross_arm(samples, groups))
    H.append("</body></html>")
    OUT.write_text("\n".join(x for x in H if x))
    print(f"[fragvec report] {len(samples)} samples, {len(groups)} groups -> {OUT}")
    return 0


_STYLE = """<style>
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:2rem auto;max-width:1150px;padding:0 1rem;color:#1a1a1a}
h1{font-size:1.6rem}h2{font-size:1.2rem;margin-top:2rem;border-bottom:1px solid #eee;padding-bottom:.3rem}
h3{font-size:1.05rem;margin-top:1.4rem}
table{border-collapse:collapse;font-size:.85rem;width:100%}th,td{border:1px solid #e2e2e2;padding:.25rem .5rem;text-align:left}
th{background:#f6f8fa}td.num{text-align:right;font-variant-numeric:tabular-nums}
.small{color:#666;font-size:.85rem}code{background:#f0f2f4;padding:.05rem .3rem;border-radius:3px}
@media(prefers-color-scheme:dark){body{background:#15181c;color:#e6e6e6}th{background:#20242a}
th,td{border-color:#333}h2{border-color:#333}code{background:#20242a}}
</style>"""


if __name__ == "__main__":
    raise SystemExit(main())
