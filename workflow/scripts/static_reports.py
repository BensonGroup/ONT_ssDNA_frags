#!/usr/bin/env python
"""Print-ready (Science-style) PDF/PNG redraws of every Plotly figure in the workflow's HTML reports.

Ported from the cells project's `batch3_html_static_figs.py` + `html_report_plotting.py`. The rule
passes the exact HTML reports `rule all` declares; nothing is globbed. Each plot's data are decoded
verbatim from its literal `Plotly.newPlot(...)` call (no JavaScript is run) and redrawn with
matplotlib, so the export has real ink and no hover-only identity. Exact duplicate plots share one
export. Display text uses the fragvec / UMI terminology; source arrays are never modified.

Writes, under --outdir: <report>/figures/F<plot>_<view>_*.pdf|svg (vector only; the SVG feeds index.html), figure_legends.md and
report_static.pdf per report, plus all_report_figures.pdf, index.html and manifest.json.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import re
import textwrap
import traceback
from bisect import bisect_right
from html import escape, unescape
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages                     # noqa: E402
from matplotlib.colors import to_rgba                                    # noqa: E402
from matplotlib.text import Text                                         # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator   # noqa: E402

import ligand_vocab                                                      # noqa: E402
import report_lib as RL                                                  # noqa: E402
import static_style as S                                                 # noqa: E402

#: Report colours that do not survive print are swapped by POSITION: grey stays the recessive ink,
#: and fragment-class colours (ligand_vocab.PALETTE, assigned by class order) map to Okabe-Ito.
SOURCE_COLORS = {RL.GREY.lower(): S.GREY, ligand_vocab.GREY.lower(): S.GREY,
                 **{c.lower(): o for c, o in zip(ligand_vocab.PALETTE, S.OKABE_ITO_LIST[:7])}}
#: Set by main(): sample name -> (group, round), from the config-derived sample table.
META: dict[str, tuple[str, int]] = {}
G: S.Groups


# ----------------------------------------------------------------------------- text and arrays

def label(value) -> str:
    """Display terminology (fragvec, UMI) applied to text; source data stay verbatim."""
    value = unescape(re.sub(r"<[^>]+>", "", ("" if value is None else str(value)).replace("<br>", "\n")))
    for pattern, repl in ((r"\bclonotypes\b", "fragvecs"), (r"\bclonotype\b", "fragvec"),
                          (r"\bfragment[- ]vectors\b", "fragvecs"), (r"\bfragment[- ]vector\b", "fragvec"),
                          (r"\bclones\b", "UMIs"), (r"\bclone\b", "UMI")):
        value = re.sub(pattern, repl, value, flags=re.I)
    return value


def compact(value: float, _pos=None) -> str:
    for divisor, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(value) >= divisor:
            return f"{value / divisor:g}{suffix}"
    return f"{value:g}"


def array(value, dtype=None):
    """Plotly arrays may be literal lists or base64 `bdata` typed arrays."""
    if isinstance(value, dict) and "bdata" in value:
        result = np.frombuffer(base64.b64decode(value["bdata"]), dtype=np.dtype(value["dtype"]))
        if value.get("shape"):
            shape = value["shape"]
            result = result.reshape(tuple(int(v) for v in shape.split(",")) if isinstance(shape, str) else shape)
    else:
        result = np.asarray(value)
    return result.astype(dtype) if dtype is not None else result


def numbers(value):
    return array(value, float)


def title_of(plot):
    title = plot["layout"].get("title", {})
    title = title.get("text", "") if isinstance(title, dict) else title
    if title == "NanoComp (by sample)":
        title = plot.get("heading")
    return label(title or plot.get("heading") or plot.get("report_title") or "Report figure")


def wrapped(value, width=85):
    return "\n".join(textwrap.fill(line, width) for line in label(value).splitlines())


def sample_info(name):
    """(group, round) for a trace or tick name: a config sample, the R0 library, or `<group> R<k>`."""
    name = label(name)
    if name in META:
        return META[name]
    if "initial" in name.lower() or name.startswith("R0 library"):
        return RL.BASELINE_GROUP, 0
    for group in sorted(G.order, key=len, reverse=True):
        match = re.match(re.escape(group) + r"(?:[_ ]R(\d+))?(?:\b|$)", name)
        if match:
            return group, int(match[1]) if match[1] else None
    return None, None


def trace_sample(trace):
    for candidate in (trace.get("name"), trace.get("legendgroup")):
        group, rnd = sample_info(candidate)
        if group:
            if rnd is None and isinstance(trace.get("text"), list) and trace["text"]:
                _, rnd = sample_info(trace["text"][0])
            if rnd is None:
                extra = re.search(r"<extra>(.*?)</extra>", trace.get("hovertemplate", ""))
                if extra:
                    _, rnd = sample_info(extra[1])
            return group, rnd
    return None, None


def short_sample(value):
    group, rnd = sample_info(value)
    return f"{group} R{rnd}" if group and rnd is not None else label(value)


def color_for(trace, index=0):
    group, _ = trace_sample(trace)
    if group:
        return G.color(group)
    original = trace.get("line", {}).get("color", trace.get("marker", {}).get("color"))
    if isinstance(original, str) and original.startswith("#"):
        return SOURCE_COLORS.get(original.lower(), original)
    if "flagged" in label(trace.get("name")).lower() or "dominant" in label(trace.get("name")).lower():
        return S.OKABE_ITO["vermillion"]
    return S.OKABE_ITO_LIST[index % len(S.OKABE_ITO_LIST)]


# ------------------------------------------------------------------------------------------- axes

def axis_layout(layout, ref):
    key = ref[0] + "axis" + ref[1:]
    result = dict(layout.get(key, {}))
    if result.get("matches"):
        result = {**axis_layout(layout, result.pop("matches")), **result}
    for key in ("tickvals", "ticktext", "range", "categoryarray"):
        if isinstance(result.get(key), dict) and "bdata" in result[key]:
            result[key] = array(result[key]).tolist()
    return result


def axis_title(spec):
    value = spec.get("title", "")
    return label(value.get("text", "") if isinstance(value, dict) else value)


def categories(traces, dim, spec):
    values = [v for t in traces for v in array(t.get(dim, [])).tolist() if v is not None]
    if spec.get("categoryarray"):
        return list(spec["categoryarray"])
    if values and all(isinstance(v, str) for v in values):
        return list(dict.fromkeys(values))
    return None


def sample_tick_labels(traces, spec, layout, xref="x"):
    """Replace repeated R1/R2 ticks with full sample labels when the data allow it."""
    for trace in traces:
        custom = trace.get("customdata")
        if isinstance(custom, list) and custom and len(custom) == len(trace.get("x", [])):
            if all(isinstance(v, str) and sample_info(v)[0] for v in custom):
                return list(range(len(custom))), [short_sample(v) for v in custom]
    ticks, texts = spec.get("tickvals"), spec.get("ticktext")
    if ticks is None or texts is None:
        return None
    annotations = [a for a in layout.get("annotations", [])
                   if a.get("xref", "x") == xref and label(a.get("text")) in G.order]
    if annotations and all(re.fullmatch(r"R\d+", label(t)) for t in texts):
        centers = [(float(a["x"]), label(a["text"])) for a in annotations]
        texts = [f"{min(centers, key=lambda p: abs(float(x) - p[0]))[1]} {label(t)}" for x, t in zip(ticks, texts)]
    return ticks, [label(t) for t in texts]


def scale_info(traces, dim, spec):
    cat = categories(traces, dim, spec)
    vals = np.concatenate([numbers(t[dim]) for t in traces if dim in t]) if not cat and traces else np.array([])
    pos = vals[np.isfinite(vals) & (vals > 0)] if vals.size else vals
    log = spec.get("type") == "log"
    return {"categories": cat, "log": log, "floor": float(pos.min()) / 3 if pos.size else 0.1,
            "zero": bool(log and vals.size and np.any(vals == 0))}


def coords(values, info):
    if info["categories"] is not None:
        lookup = {v: i for i, v in enumerate(info["categories"])}
        return np.array([lookup[v] for v in array(values).tolist()], dtype=float)
    result = numbers(values).copy()
    if info["log"]:
        result[result == 0] = info["floor"]       # zeros sit on a labelled floor, not dropped
    return result


def configure_axis(ax, dim, spec, info, traces, layout, ref):
    setlim = ax.set_xlim if dim == "x" else ax.set_ylim
    setticks = ax.set_xticks if dim == "x" else ax.set_yticks
    axis = ax.xaxis if dim == "x" else ax.yaxis
    (ax.set_xlabel if dim == "x" else ax.set_ylabel)(wrapped(axis_title(spec), 58), fontsize=6.5)
    if info["log"]:
        (ax.set_xscale if dim == "x" else ax.set_yscale)("log")
        axis.set_minor_locator(NullLocator())
    if info["categories"] is not None:
        cats = info["categories"]
        setticks(range(len(cats)), [short_sample(v) for v in cats], fontsize=5.4)
        if dim == "x" and len(cats) > 5:
            ax.tick_params(axis="x", labelrotation=55)
            plt.setp(ax.get_xticklabels(), ha="right")
        setlim(-0.6, len(cats) - 0.4)
    else:
        ticks = sample_tick_labels(traces, spec, layout, ref) if dim == "x" else None
        if ticks is None and "tickvals" in spec:
            ticks = (spec["tickvals"], [label(t) for t in spec.get("ticktext", spec["tickvals"])])
        if ticks:
            setticks(*ticks, fontsize=5.4)
            if dim == "x" and len(ticks[0]) > 6:
                plt.setp(ax.get_xticklabels(), rotation=55, ha="right")
        else:
            axis.set_major_formatter(FuncFormatter(compact))
            if not info["log"]:
                axis.set_major_locator(MaxNLocator(nbins=5, integer=spec.get("dtick") == 1))
    if "range" in spec:
        setlim([10 ** v for v in spec["range"]] if info["log"] else spec["range"])
    if spec.get("autorange") == "reversed":
        (ax.invert_xaxis if dim == "x" else ax.invert_yaxis)()
    if info["zero"]:
        ticks = [v for v in axis.get_majorticklocs() if v > info["floor"] * 1.001]
        axis.set_major_locator(FixedLocator([info["floor"], *ticks]))
        axis.set_major_formatter(FuncFormatter(
            lambda v, _p: "0" if np.isclose(v, info["floor"], rtol=1e-8, atol=0) else compact(v)))
    for tick in ax.get_xticklabels() if dim == "x" else ax.get_yticklabels():
        group = sample_info(tick.get_text())[0]
        if group:
            tick.set_color(G.color(group))


def reference_lines(ax, plot, xref="x", yref="y"):
    """Keep threshold lines and their labels; drop decorative group headings."""
    for shape in plot["layout"].get("shapes", []):
        if shape.get("type") != "line":
            continue
        if shape.get("xref", "x").replace(" domain", "") != xref or shape.get("yref", "y").replace(" domain", "") != yref:
            continue
        style = dict(color=SOURCE_COLORS.get(shape.get("line", {}).get("color", "").lower(), S.GREY),
                     lw=0.55, ls=(0, (2, 2)), alpha=shape.get("opacity", 0.7))
        if shape["x0"] == shape["x1"]:
            ax.axvline(shape["x0"], **style)
        elif shape["y0"] == shape["y1"]:
            ax.axhline(shape["y0"], **style)
    for note in plot["layout"].get("annotations", []):
        text = label(note.get("text"))
        xr, yr = note.get("xref", "x"), note.get("yref", "y")
        if text in G.order or xr == "paper" or yr == "paper":
            continue
        if xr.replace(" domain", "") != xref or yr.replace(" domain", "") != yref:
            continue
        transform = ax.transAxes if "domain" in xr and "domain" in yr else (
            ax.get_yaxis_transform() if "domain" in xr else ax.get_xaxis_transform() if "domain" in yr else ax.transData)
        x, y = note.get("x", 0), note.get("y", 0)
        # plotly places a data-referenced annotation on a LOG axis at log10(value); matplotlib wants the value
        if "domain" not in xr and axis_layout(plot["layout"], xr).get("type") == "log":
            x = 10 ** float(x)
        if "domain" not in yr and axis_layout(plot["layout"], yr).get("type") == "log":
            y = 10 ** float(y)
        ax.text(x, y, wrapped(text, 45), transform=transform,
                fontsize=5.2, color=S.GREY, va="bottom", ha="left", clip_on=False)


# --------------------------------------------------------------------------------------- figures

def bar_figure(plot, indices, suffix=""):
    traces = [plot["data"][i] for i in indices]
    horizontal = all(t.get("orientation") == "h" for t in traces)
    xref, yref = traces[0].get("xaxis", "x"), traces[0].get("yaxis", "y")
    xs, ys = axis_layout(plot["layout"], xref), axis_layout(plot["layout"], yref)
    xi, yi = scale_info(traces, "x", xs), scale_info(traces, "y", ys)
    catdim = "y" if horizontal else "x"
    ci, ni = (yi, xi) if horizontal else (xi, yi)
    ncat = len(ci["categories"] or list(dict.fromkeys(v for t in traces for v in array(t[catdim]).tolist())))
    fig, ax = plt.subplots(figsize=S.figsize("3col", max(0.48, ncat * 0.019 + 0.18) if horizontal else 0.52))
    fig.subplots_adjust(left=0.31 if horizontal else 0.12, right=0.975, bottom=0.14 if horizontal else 0.29, top=0.90)
    stacked = plot["layout"].get("barmode") in ("stack", "relative")
    disjoint = sum(array(t[catdim]).size for t in traces) == ncat   # traces on different categories
    groups = 1 if stacked or disjoint else len(traces)
    offsets = {}
    for j, trace in enumerate(traces):
        x, y = coords(trace["x"], xi), coords(trace["y"], yi)
        pos, val = (y, x) if horizontal else (x, y)
        raw_val = numbers(trace["x"] if horizontal else trace["y"])
        width = 0.76 / groups
        positions = pos + (0 if groups == 1 else (j - (groups - 1) / 2) * width)
        base = np.array([offsets.get(float(p), 0.0) for p in pos]) if stacked else np.zeros(len(pos))
        color = color_for(trace, j)
        source = trace.get("marker", {}).get("color")
        colors = [SOURCE_COLORS.get(str(c).lower(), color) for c in source] if isinstance(source, list) else [color] * len(pos)
        names = trace.get("customdata", trace.get("x", []))
        if not horizontal and isinstance(names, list) and len(names) == len(pos):
            colors = [G.color(sample_info(n)[0]) if isinstance(n, str) and sample_info(n)[0] else c
                      for n, c in zip(names, colors)]
        translucent = trace.get("marker", {}).get("opacity", 1) < 0.7
        hatch = ("///" if translucent or trace.get("marker", {}).get("pattern") else
                 ("", "///", "...", "xx", "--", "++", "oo")[j % 7] if stacked else "")
        kwargs = dict(color=[to_rgba(c, 0.18 if translucent else 1) for c in colors], edgecolor=colors,
                      linewidth=0.35, hatch=hatch, label=label(trace.get("name", "")))
        bars = (ax.barh(positions, val, height=width, left=base, **kwargs) if horizontal
                else ax.bar(positions, val, width=width, bottom=base, **kwargs))
        if stacked:
            for p, v, b in zip(pos, raw_val, base):
                offsets[float(p)] = b + v
        if (len(traces) == 1 or disjoint) and ncat <= 25:
            ax.bar_label(bars, labels=[f"{v:,.0f}" if float(v).is_integer() else f"{v:.3g}" for v in raw_val],
                         fontsize=5.1, padding=2)
        if ni["zero"]:
            for p, v in zip(positions, raw_val):
                if v == 0:
                    ax.text(*((ni["floor"], p) if horizontal else (p, ni["floor"])), "0", fontsize=5,
                            va="center" if horizontal else "baseline", ha="left" if horizontal else "center")
    configure_axis(ax, "x", xs, xi, traces, plot["layout"], xref)
    configure_axis(ax, "y", ys, yi, traces, plot["layout"], yref)
    if not horizontal and not yi["log"] and "range" not in ys:
        ax.set_ylim(bottom=0, top=max(ax.get_ylim()[1], 1) * 1.10)
    if horizontal and not xi["log"] and "range" not in xs:
        ax.set_xlim(left=0)
    handles, names = ax.get_legend_handles_labels()
    if names and not (disjoint and not horizontal and all(trace_sample(t)[0] for t in traces)):
        ax.legend(handles, names, loc="upper left", bbox_to_anchor=(0, -0.17 if horizontal else 1.03),
                  ncol=min(4, len(names)), fontsize=5.4, handlelength=1.2)
    reference_lines(ax, plot, xref, yref)
    fig.suptitle(wrapped(title_of(plot) + suffix), fontsize=7, y=0.99)
    return fig, []


#: plotly's named dashes as on/off px, the same unit as report_lib.DASHES
NAMED_DASHES = {"solid": (), "dot": (2, 2.5), "dash": (6, 3), "longdash": (10, 3),
                "dashdot": (6, 2.5, 1.5, 2.5), "longdashdot": (10, 2.5, 1.5, 2.5)}


def trace_dash(trace):
    """The trace's OWN plotly dash as a matplotlib linestyle, or None when it set none. Every
    report trace now carries its PlotStyle dash, so print reproduces the page exactly."""
    dash = (trace.get("line") or {}).get("dash")
    if not dash:
        return None
    pat = (NAMED_DASHES[dash] if dash in NAMED_DASHES
           else tuple(float(v.strip().removesuffix("px")) for v in dash.split(",")))
    return "-" if not pat else (0, tuple(round(RL.MPL_DASH_SCALE * v, 2) for v in pat))


def draw_trace(ax, trace, index, xi, yi, name=None, histogram_steps=False):
    x, y = coords(trace["x"], xi), coords(trace["y"], yi)
    group, rnd = trace_sample(trace)
    color = color_for(trace, index)
    marker = (G.marker(group) if group else S.MARKERS[index % len(S.MARKERS)])
    dash = trace_dash(trace) or S.DASHES[(rnd if rnd is not None else index) % len(S.DASHES)]
    mode = trace.get("mode", "markers" if trace.get("type") == "scattergl" else "lines")
    text = name if name is not None else label(trace.get("name", f"series {index + 1}"))
    if histogram_steps:
        ax.step(x, y, where="mid", color=color, ls=dash, lw=0.9, label=text)
        return
    if "lines" in mode:
        ax.plot(x, y, color=color, ls=dash, lw=0.9, label=text if "markers" not in mode else "_nolegend_")
    if "markers" in mode:
        symbols = trace.get("marker", {}).get("symbol", "circle")
        hollow = (np.array(["open" in str(s) for s in symbols]) if isinstance(symbols, list)
                  else np.repeat("open" in str(symbols), len(x)))
        ring = trace.get("marker", {}).get("color") == "rgba(0,0,0,0)"
        if ring:
            hollow[:] = True
            color = S.OKABE_ITO["vermillion"]
        if trace.get("type") == "scattergl":
            ax.scatter(x, y, s=8 if ring else 3, marker=marker, facecolors="none" if ring else color,
                       edgecolors=color, linewidths=0.6 if ring else 0, alpha=trace.get("marker", {}).get("opacity", 0.65),
                       label=text, rasterized=True, zorder=4 if ring else 2)
        else:
            for state in (False, True):
                mask = hollow == state
                if mask.any():
                    ax.plot(x[mask], y[mask], ls="none", marker=marker, ms=3.0, mfc="white" if state else color,
                            mec=color, mew=0.65, color=color, label=text if state == bool(hollow[0]) else "_nolegend_")
    if "error_y" in trace and trace["error_y"].get("array") is not None:
        ax.errorbar(x, y, yerr=numbers(trace["error_y"]["array"]), fmt="none", color=color, lw=0.6, capsize=1.5)


def curve_figure(plot, indices, suffix="", facet_samples=False, individual=False, histogram_steps=False):
    traces = [plot["data"][i] for i in indices]
    xref, yref = traces[0].get("xaxis", "x"), traces[0].get("yaxis", "y")
    xs, ys = axis_layout(plot["layout"], xref), axis_layout(plot["layout"], yref)
    xi, yi = scale_info(traces, "x", xs), scale_info(traces, "y", ys)
    if histogram_steps and not axis_title(xs):
        xs = {**xs, "title": {"text": "Read length (nt; log scale)" if "log transformed" in title_of(plot) else "Read length (nt)"}}
    if individual:
        panels = [(f"fragvec {j + 1:02d}", [j]) for j in range(len(traces))]
    elif facet_samples:
        panels = [("empty (control)" if g == RL.EMPTY_GROUP else g,
                   [j for j, t in enumerate(traces) if trace_sample(t)[0] == g])
                  for g in G.order if any(trace_sample(t)[0] == g for t in traces)]
    else:
        panels = [("", list(range(len(traces))))]
    ncols = (4 if len(panels) == 8 else 3) if len(panels) > 1 else 1
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=S.figsize("3col", max(0.52, nrows * 0.27 + 0.15) if len(panels) > 1 else 0.60),
                             squeeze=False, sharex=len(panels) > 1, sharey=len(panels) > 1)
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.20 if len(panels) > 1 else 0.31, top=0.87,
                        hspace=0.82 if individual else 0.57, wspace=0.31)
    references = [j for j, t in enumerate(traces) if trace_sample(t)[0] is None] if facet_samples else []
    notes = []
    for pi, (panel, local) in enumerate(panels):
        ax = axes.flat[pi]
        for j in [*local, *references]:
            _, rnd = trace_sample(traces[j])
            name = (f"R{rnd}" + (" dominant" if "dominant" in traces[j].get("name", "") else "")
                    if facet_samples and rnd is not None else label(traces[j].get("name", f"series {j + 1}")))
            draw_trace(ax, traces[j], j, xi, yi, "_nolegend_" if individual else name, histogram_steps)
        configure_axis(ax, "x", xs, xi, traces, plot["layout"], xref)
        configure_axis(ax, "y", ys, yi, traces, plot["layout"], yref)
        reference_lines(ax, plot, xref, yref)
        if individual:
            ident = label(traces[local[0]].get("name", ""))
            ax.set_title(panel + "\n" + textwrap.shorten(ident, 34, placeholder="…"), fontsize=5.4)
            notes.append(f"{panel}: {ident}")
            if "share" in axis_title(ys) and not yi["log"]:
                ax.set_ylim(0, 1.07)
            ax.set_xlabel("round" if "round" in axis_title(xs) else "sample", fontsize=5.8)
        elif panel:
            ax.set_title(panel, fontsize=6.5)
        if len(panels) > 1:
            if pi % ncols:
                ax.set_ylabel("")
            if pi < (nrows - 1) * ncols:
                ax.set_xlabel("")
        handles, names = ax.get_legend_handles_labels()
        if handles:
            unique = dict(zip(names, handles))        # dedupe repeated reference labels only
            ax.legend(unique.values(), unique.keys(), fontsize=5.1, ncol=2 if facet_samples else 3, handlelength=2.0,
                      loc="upper right" if facet_samples else "upper left",
                      bbox_to_anchor=None if facet_samples else (0, -0.23))
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    fig.suptitle(wrapped(title_of(plot) + suffix, 105), fontsize=7, y=0.99)
    if histogram_steps:
        notes.append("Stored histogram bin centers and counts are drawn as step curves, faceted by sample group; no rebinning.")
    if xi["zero"] or yi["zero"]:
        notes.append("Zeros on logarithmic axes sit on a labelled plotting floor below the smallest positive value.")
    return fig, notes


def heatmap_figure(plot, index):
    trace = plot["data"][index]
    z = numbers(trace["z"])
    xref, yref = trace.get("xaxis", "x"), trace.get("yaxis", "y")
    xs, ys = axis_layout(plot["layout"], xref), axis_layout(plot["layout"], yref)
    if not ys.get("ticktext"):
        ys = {**ys, **{k: v for k, v in plot["layout"].get("yaxis", {}).items() if k in ("ticktext", "tickvals")}}
    shared = plot["layout"].get(trace.get("coloraxis", ""), {})
    vmin, vmax = trace.get("zmin", shared.get("cmin")), trace.get("zmax", shared.get("cmax"))
    diverging = trace.get("zmid") == 0 or (vmin is not None and vmax is not None and vmin < 0 < vmax)
    fig, ax = plt.subplots(figsize=S.figsize("3col", max(0.75, z.shape[0] * 0.019 + 0.22)))
    fig.subplots_adjust(left=0.29, right=0.85, bottom=0.16, top=0.87)
    cmap = matplotlib.colormaps["RdBu_r" if diverging else "viridis"].with_extremes(bad="#e7e9ec")
    im = ax.imshow(np.ma.masked_invalid(z), interpolation="nearest", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    xticks = sample_tick_labels([trace], xs, plot["layout"], xref) or (
        np.arange(z.shape[1]), [label(v) for v in array(trace.get("x", np.arange(z.shape[1])))])
    ax.set_xticks(*xticks, rotation=90, fontsize=5.3)
    ax.set_yticks(np.arange(z.shape[0]), [label(v) for v in ys.get("ticktext", array(trace.get("y", np.arange(z.shape[0]))))], fontsize=5.0)
    ax.tick_params(length=0)
    ax.set_xlabel(axis_title(xs))
    ax.set_ylabel(axis_title(ys))
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = {**shared.get("colorbar", {}), **trace.get("colorbar", {})}
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.025)
    cb.ax.tick_params(labelsize=5.2, length=2, width=0.5)
    if "tickvals" in colorbar:
        cb.set_ticks(colorbar["tickvals"], labels=[label(v) for v in colorbar.get("ticktext", colorbar["tickvals"])])
    cbtitle = colorbar.get("title", "")
    cb.set_label(label(cbtitle.get("text", "") if isinstance(cbtitle, dict) else cbtitle) or
                 ("log2 fold change" if diverging else "mean copies per read"), fontsize=6)
    heading = title_of(plot)
    paper = [a for a in plot["layout"].get("annotations", []) if a.get("xref") == "paper" and
             ("orientation" in a.get("text", "") or "reverse-complement" in a.get("text", ""))]
    if index < len(paper):
        heading += " · " + label(paper[index]["text"])
    fig.suptitle(wrapped(heading), fontsize=7, y=0.98)
    return fig, ["Heatmap values, missing cells, colour limits and colorbar ticks are preserved from the report."]


def histogram_figure(plot, indices):
    values = numbers(plot["data"][indices[0]]["x"])
    values = values[np.isfinite(values)]
    trace = plot["data"][indices[0]]
    fig, ax = plt.subplots(figsize=S.figsize("3col", 0.52))
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.19, top=0.85)
    xs, ys = axis_layout(plot["layout"], "x"), axis_layout(plot["layout"], "y")
    notes, edges = [], None
    binspec = trace.get("xbins", {})
    if binspec.get("size") is not None:
        size = float(binspec["size"])
        edges = np.arange(float(binspec.get("start", values.min())), float(binspec.get("end", values.max())) + size, size)
    elif np.all(values == np.floor(values)) and len(np.unique(values)) < 250:
        support, counts = np.unique(values, return_counts=True)
        ax.bar(support, counts, width=np.maximum(support * 0.15, 0.6) if xs.get("type") == "log" else 0.8, color=S.GREY)
        notes.append("Unbinned integer observations are drawn as exact frequencies, not browser bins.")
    else:
        edges = np.histogram_bin_edges(values, bins=int(trace.get("nbinsx", 40)))
        notes.append(f"{len(edges) - 1} equal-width bins over the full observed range, zeros included.")
    if edges is not None:
        counts, edges = np.histogram(values, bins=edges)
        ax.stairs(counts, edges, fill=True, color=S.OKABE_ITO["blue"])
    if xs.get("type") == "log":
        ax.set_xscale("log")
    if ys.get("type") == "log":
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)
    ax.set_xlabel(wrapped(axis_title(xs), 60))
    ax.set_ylabel(wrapped(axis_title(ys), 60))
    ax.xaxis.set_major_formatter(FuncFormatter(compact))
    ax.yaxis.set_major_formatter(FuncFormatter(compact))
    reference_lines(ax, plot)
    fig.suptitle(wrapped(title_of(plot)), fontsize=7, y=0.99)
    notes.append(f"All {len(values):,} embedded observations enter the histogram.")
    return fig, notes


def violin_figure(plot, indices):
    traces = [plot["data"][i] for i in indices]
    values = [numbers(t["y"]) for t in traces]
    fig, ax = plt.subplots(figsize=S.figsize("3col", 0.53))
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.29, top=0.87)
    for j, (trace, val) in enumerate(zip(traces, values)):
        val, color = val[np.isfinite(val)], color_for(trace, j)
        if np.ptp(val) == 0:
            ax.plot([j - 0.3, j + 0.3], [val[0], val[0]], color=color, lw=1)
        else:
            body = ax.violinplot([val], positions=[j], widths=0.8, showextrema=False, points=200, bw_method="scott")["bodies"][0]
            body.set_facecolor(to_rgba(color, 0.45))
            body.set_edgecolor(color)
            body.set_linewidth(0.65)
    ax.set_xticks(range(len(traces)), [short_sample(t.get("name", "")) for t in traces], fontsize=5.3, rotation=55, ha="right")
    ys = axis_layout(plot["layout"], "y")
    ax.set_ylabel(axis_title(ys))
    if ys.get("type") == "log":
        ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(compact))
    fig.suptitle(wrapped(axis_title(ys) + " by sample"), fontsize=7, y=0.99)
    return fig, ["Every embedded observation is retained; violins are matplotlib Gaussian KDEs (Scott bandwidth) "
                 "and can differ in outline from Plotly's browser KDE."]


def render(plot):
    """Yield (figure, view name, covered source-trace indices, notes) for one source plot."""
    traces, ids = plot["data"], list(range(len(plot["data"])))
    types = {t.get("type", "scatter") for t in traces}
    if not types <= {"bar", "scatter", "scattergl", "heatmap", "histogram", "violin"}:
        raise ValueError(f"Unsupported trace types: {types}")
    if types == {"heatmap"}:
        for i in ids:
            fig, notes = heatmap_figure(plot, i)
            yield fig, f"heatmap_{i + 1}", [i], notes
        return
    if types in ({"histogram"}, {"violin"}):
        fig, notes = (histogram_figure if types == {"histogram"} else violin_figure)(plot, ids)
        yield fig, types.pop(), ids, notes
        return
    partitions = {}
    for i, trace in enumerate(traces):
        partitions.setdefault((trace.get("xaxis", "x"), trace.get("yaxis", "y")), []).append(i)
    paper = [a for a in plot["layout"].get("annotations", []) if a.get("xref") == "paper"]
    for number, ((_xref, yref), indices) in enumerate(partitions.items(), 1):
        sub = [traces[i] for i in indices]
        kinds = {t.get("type", "scatter") for t in sub}
        suffix = ""
        if len(partitions) > 1:
            suffix = " · " + (label(paper[number - 1]["text"]) if len(paper) == len(partitions)
                              else axis_title(axis_layout(plot["layout"], yref)))
        if kinds == {"bar"} and plot["layout"].get("barmode") == "overlay" and len(sub) >= 10:
            fig, notes = curve_figure(plot, indices, suffix, facet_samples=True, histogram_steps=True)
        elif kinds == {"bar"}:
            fig, notes = bar_figure(plot, indices, suffix)
        elif kinds <= {"scatter", "scattergl"}:
            fig, notes = curve_figure(plot, indices, suffix,
                                      facet_samples=sum(trace_sample(t)[0] is not None for t in sub) >= 10,
                                      individual=plot["source"].startswith("fragvec/") and len(sub) == 15)
        else:
            raise ValueError(f"Unsupported mixed panel types: {kinds}")
        yield fig, f"panel_{number}" if len(partitions) > 1 else "plot", indices, notes


def finish(fig):
    """Apply terminology after the renderers wrote their text; refuse text below 5 pt."""
    for text in fig.findobj(Text):
        text.set_text(label(text.get_text()))
    fig.canvas.draw()
    if not S.check_min_font(fig):
        raise ValueError("Figure contains text below 5 pt")


# ---------------------------------------------------------------------------------------- driver

def extract(path: Path, relative: str) -> tuple[str, list[dict]]:
    """Decode every literal Plotly.newPlot(div, data, layout) call, with its nearest heading."""
    source = path.read_text(encoding="utf-8")
    title = re.search(r"<title[^>]*>(.*?)</title>", source, re.S | re.I)
    report_title = unescape(title[1]).strip() if title else path.stem
    headings = list(re.finditer(r"<h[1-6]\b[^>]*>(.*?)</h[1-6]>", source, re.S | re.I))
    positions = [h.start() for h in headings]
    decoder, plots = json.JSONDecoder(), []
    for match in re.finditer(r'Plotly\.newPlot\(\s*"', source):
        pos, args = match.end() - 1, []
        for i in range(3):
            value, pos = decoder.raw_decode(source, pos)
            args.append(value)
            if i < 2:
                while source[pos].isspace():
                    pos += 1
                if source[pos] != ",":
                    raise ValueError(f"Unexpected newPlot argument at {path}:{pos}")
                pos += 1
                while source[pos].isspace():
                    pos += 1
        div_id, data, layout = args
        if layout.get("updatemenus") or layout.get("sliders"):
            raise ValueError(f"Interactive states require explicit expansion: {path}")
        hi = bisect_right(positions, match.start()) - 1
        heading = unescape(headings[hi][1]) if hi >= 0 else report_title
        if not layout.get("title", {}).get("text"):
            xlabel = axis_title(layout.get("xaxis", {}))
            if "payload fragment copies" in xlabel:
                heading = "Payload multiplicity"
            elif "edit distance" in xlabel:
                heading = "Payload edit distance"
            elif "% of read length" in xlabel:
                heading = "Fragment coverage per read"
            elif data and data[0].get("name") == "reads decoded":
                heading = "Reads decoded and reads with payload"
            if relative.endswith(".featurecount_report.html"):
                heading = path.name.split(".featurecount_report.html")[0] + " · " + label(heading)
        elif relative.startswith("qc/") and layout["title"]["text"] == "NanoComp (by sample)":
            heading = axis_title(layout.get("yaxis", {})) + " by sample"
        plots.append({"id": div_id, "data": data, "layout": {k: v for k, v in layout.items() if k != "template"},
                      "heading": label(heading), "report_title": label(report_title),
                      "source": relative, "source_plot": len(plots) + 1})
    return label(report_title), plots


def fingerprint(plot) -> str:
    """Same data + layout = same plot, whatever div or report wraps it (NanoComp repeats its own)."""
    payload = {k: plot[k] for k in ("data", "layout")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def slug(text, limit=75) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", label(text)).strip("_")[:limit].rstrip("_") or "figure"


def main(argv=None) -> int:
    global META, G
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, required=True, help="the workflow outdir the reports live under")
    ap.add_argument("--meta", type=Path, required=True, help="the config-derived sample_metadata.tsv")
    ap.add_argument("--reports", nargs="+", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    a = ap.parse_args(argv)
    meta = RL.load_meta(a.meta)
    META = {name: (m["group"], m["round"]) for name, m in meta.items()}
    G = S.Groups([{"name": n, **m} for n, m in meta.items()])
    out = a.outdir
    out.mkdir(parents=True, exist_ok=True)
    plt.style.use(str(S.STYLE))
    reports, unique, failures = [], {}, []
    with PdfPages(out / "all_report_figures.pdf") as combined:
        for path in a.reports:
            relative = str(path.resolve().relative_to(a.root.resolve()))
            try:
                title, plots = extract(path, relative)
            except Exception as error:
                failures.append({"source": relative, "error": f"{type(error).__name__}: {error}"})
                continue
            figdir = out / Path(relative).with_suffix("") / "figures"
            figdir.mkdir(parents=True, exist_ok=True)
            report = {"source": relative, "title": title, "plots": [], "views": []}
            reports.append(report)
            captions = [f"# {title}", "", f"Source: `{path.resolve()}`", ""]
            with PdfPages(figdir / "report_static.pdf") as per_report:
                for plot in plots:
                    digest = fingerprint(plot)
                    report["plots"].append(digest)
                    if digest in unique:
                        continue
                    unique[digest] = {"source": relative, "source_plot": plot["source_plot"], "title": title_of(plot), "views": []}
                    covered = set()
                    try:
                        for vi, (fig, view, trace_ids, notes) in enumerate(render(plot), 1):
                            finish(fig)
                            stem = f"F{plot['source_plot']:02d}_{vi:02d}_{slug(view)}"
                            for ext in ("pdf", "svg"):
                                fig.savefig(figdir / f"{stem}.{ext}")
                            combined.savefig(fig)
                            per_report.savefig(fig)
                            plt.close(fig)
                            covered.update(trace_ids)
                            name = title_of(plot) + (f" · {label(view)}" if view not in ("plot", "histogram", "violin") else "")
                            unique[digest]["views"].append({"title": name, "svg": str((figdir / f"{stem}.svg").relative_to(out)),
                                                            "pdf": str((figdir / f"{stem}.pdf").relative_to(out))})
                            report["views"].append(unique[digest]["views"][-1])
                            captions += [f"## {stem}", "", name, "", *[label(n) for n in notes], "", f"![{name}]({stem}.svg)", ""]
                        missed = set(range(len(plot["data"]))) - covered
                        if missed:
                            raise ValueError(f"Unrendered source traces: {sorted(missed)}")
                    except Exception as error:
                        plt.close("all")
                        failures.append({"source": relative, "source_plot": plot["source_plot"],
                                         "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()})
                        print(f"[WARN] {relative} plot {plot['source_plot']}: {error}", flush=True)
            (figdir / "figure_legends.md").write_text("\n".join(captions) + "\n")
            print(f"{relative}: {len(plots)} source plots", flush=True)
    body = ["<!doctype html><meta charset='utf-8'><title>Static report figures</title>",
            "<style>body{font:15px system-ui;max-width:1300px;margin:30px auto;padding:0 16px;color:#202428;background:#fff}"
            "details{border-top:1px solid #ddd;padding:14px 0}summary{cursor:pointer;font-weight:600}"
            ".figs{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:20px}"
            "img{width:100%;height:auto}figcaption{font-size:13px}a{color:#0072b2}</style>",
            "<h1>Static report figures</h1>",
            f"<p>{len(reports)} reports · {len(unique)} unique plots · "
            f"{sum(len(u['views']) for u in unique.values())} figures · <a href='all_report_figures.pdf'>all (PDF)</a></p>"]
    for report in reports:
        body.append(f"<details><summary>{escape(report['title'])} <small>({escape(report['source'])})</small></summary><div class='figs'>")
        body += [f"<figure><a href='{escape(v['svg'])}'><img loading='lazy' src='{escape(v['svg'])}' alt='{escape(v['title'])}'></a>"
                 f"<figcaption>{escape(v['title'])} · <a href='{escape(v['pdf'])}'>PDF</a></figcaption></figure>"
                 for v in report["views"]]
        body.append("</div></details>")
    if failures:
        body.append("<h2>Export failures</h2><pre>" + escape(json.dumps(failures, indent=2)) + "</pre>")
    (out / "index.html").write_text("\n".join(body) + "\n")
    (out / "manifest.json").write_text(json.dumps({"reports": reports, "unique_plots": list(unique.values()),
                                                   "failures": failures}, indent=2) + "\n")
    print(f"[INFO] reports {len(reports)} · unique plots {len(unique)} · failures {len(failures)}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
