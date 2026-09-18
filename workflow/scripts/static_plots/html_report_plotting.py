"""Matplotlib renderers for the literal Plotly data in workflow HTML reports.

Source arrays remain untouched. Display-only aggregation (histograms and violin
density estimates) is explicitly recorded in each figure's notes.
"""
from __future__ import annotations

import base64
from collections import Counter
import math
import re
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator
import numpy as np

import palettes as P
import umi_report_cross_figs as U

FAMILY_ORDER = ("3WJ", "3WJunpairedT", "4WJ", "KL", "Hairpin", "Linear", "Linearbulge")
FAMILY_COLOR = dict(zip(FAMILY_ORDER, [P.OKABE_ITO[k] for k in
                        ("blue", "bluish_green", "orange", "reddish_purple",
                         "sky_blue", "black", "vermillion")]))
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "h")
DASHES = ("-", (0, (3, 1.3)), (0, (1, 1.2)), (0, (4, 1, 1, 1)))
SOURCE_COLORS = {
    "#7d8590": P.GREY, "#8b949e": P.GREY, "#0072b2": U.GROUP_COLOR["A549"],
    "#009e73": U.GROUP_COLOR["MDA"], "#d55e00": U.GROUP_COLOR["empty"],
    "#8250df": U.GROUP_COLOR["A549-SN"], "#a86c3f": U.GROUP_COLOR["MDA-SN"],
    **dict(zip(("#4f9cff", "#3fb950", "#e3b341", "#db61a2", "#a371f7", "#d2691e", "#26c6da"),
               FAMILY_COLOR.values())),
}


def label(value) -> str:
    """Apply the user's terminology to display text, retaining source data verbatim."""
    value = U.plain("" if value is None else str(value))
    value = re.sub(r"\bpayload[- ]fragments?\b", lambda m: "fragments" if m[0].lower().endswith("s") else "fragment", value, flags=re.I)
    value = re.sub(r"\bpayloads\b", "fragments", value, flags=re.I)
    value = re.sub(r"\bpayload\b", "fragment", value, flags=re.I)
    value = re.sub(r"\b(?:unique|distinct) sequences\b", "constructs", value, flags=re.I)
    value = re.sub(r"\b(?:unique|distinct) sequence\b", "construct", value, flags=re.I)
    value = re.sub(r"\bclonotypes\b", "fragvecs", value, flags=re.I)
    value = re.sub(r"\bclonotype\b", "fragvec", value, flags=re.I)
    value = re.sub(r"\bfragment[- ]vectors\b", "fragvecs", value, flags=re.I)
    value = re.sub(r"\bfragment[- ]vector\b", "fragvec", value, flags=re.I)
    value = re.sub(r"\bclones\b", "UMIs", value, flags=re.I)
    return re.sub(r"\bclone\b", "UMI", value, flags=re.I)


def array(value, dtype=None):
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
    name = label(name)
    if name in U.SAMPLE_META:
        return U.SAMPLE_META[name]
    if name == "initial" or name.startswith("R0 library"):
        return "initial", 0
    for group in sorted(U.GROUP_COLOR, key=len, reverse=True):
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
        return U.GROUP_COLOR[group]
    if trace.get("name") in FAMILY_COLOR:
        return FAMILY_COLOR[trace["name"]]
    original = trace.get("line", {}).get("color", trace.get("marker", {}).get("color"))
    if isinstance(original, str) and original.lower() in SOURCE_COLORS:
        return SOURCE_COLORS[original.lower()]
    if "flagged" in label(trace.get("name")).lower() or "dominant" in label(trace.get("name")).lower():
        return P.OKABE_ITO["vermillion"]
    return P.OKABE_ITO_LIST[index % len(P.OKABE_ITO_LIST)]


def axis_layout(layout, ref):
    key = ref[0] + "axis" + ref[1:]
    result = dict(layout.get(key, {}))
    if result.get("matches"):
        parent = axis_layout(layout, result.pop("matches"))
        result = {**parent, **result}
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
    """Replace repeated R1/R2 ticks with full sample labels when metadata allow it."""
    for trace in traces:
        custom = trace.get("customdata")
        if isinstance(custom, list) and len(custom) == len(trace.get("x", [])) and custom:
            if all(isinstance(v, str) and sample_info(v)[0] for v in custom):
                return list(range(len(custom))), [short_sample(v) for v in custom]
    ticks, texts = spec.get("tickvals"), spec.get("ticktext")
    if ticks is None or texts is None:
        return None
    annotations = [a for a in layout.get("annotations", [])
                   if a.get("xref", "x") == xref and label(a.get("text")) in U.GROUP_COLOR]
    if annotations and all(re.fullmatch(r"R\d+", label(t)) for t in texts):
        centers = [(float(a["x"]), label(a["text"])) for a in annotations]
        texts = [f"{min(centers, key=lambda p: abs(float(x) - p[0]))[1]} {label(t)}"
                 for x, t in zip(ticks, texts)]
    return ticks, [label(t) for t in texts]


def scale_info(traces, dim, spec):
    cat = categories(traces, dim, spec)
    log = spec.get("type") == "log"
    vals = np.concatenate([numbers(t[dim]) for t in traces if dim in t]) if not cat and traces else np.array([])
    pos = vals[np.isfinite(vals) & (vals > 0)] if vals.size else vals
    floor = float(pos.min()) / 3 if pos.size else 0.1
    zero = bool(log and vals.size and np.any(vals == 0))
    return {"categories": cat, "log": log, "floor": floor, "zero": zero}


def coords(values, info):
    if info["categories"] is not None:
        lookup = {v: i for i, v in enumerate(info["categories"])}
        return np.array([lookup[v] for v in array(values).tolist()], dtype=float)
    result = numbers(values).copy()
    if info["log"]:
        result[result == 0] = info["floor"]
    return result


def configure_axis(ax, dim, spec, info, traces, layout, ref):
    setscale = ax.set_xscale if dim == "x" else ax.set_yscale
    setlim = ax.set_xlim if dim == "x" else ax.set_ylim
    setticks = ax.set_xticks if dim == "x" else ax.set_yticks
    axis = ax.xaxis if dim == "x" else ax.yaxis
    (ax.set_xlabel if dim == "x" else ax.set_ylabel)(wrapped(axis_title(spec), 58), fontsize=6.5)
    if info["log"]:
        setscale("log")
        axis.set_minor_locator(NullLocator())
    if info["categories"] is not None:
        cats = info["categories"]
        setticks(range(len(cats)), [short_sample(v) for v in cats], fontsize=5.4)
        if dim == "x":
            ax.tick_params(axis="x", labelrotation=55 if len(cats) > 5 else 0)
            if len(cats) > 5:
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
        elif info["log"]:
            axis.set_major_formatter(FuncFormatter(U.compact))
        else:
            axis.set_major_formatter(FuncFormatter(U.compact))
            axis.set_major_locator(MaxNLocator(nbins=5, integer=spec.get("dtick") == 1))
    if "range" in spec:
        limits = spec["range"]
        setlim([10 ** v for v in limits] if info["log"] else limits)
    if spec.get("autorange") == "reversed":
        (ax.invert_xaxis if dim == "x" else ax.invert_yaxis)()
    if info["zero"]:
        ticks = [v for v in axis.get_majorticklocs() if v > info["floor"] * 1.001]
        axis.set_major_locator(FixedLocator([info["floor"], *ticks]))
        axis.set_major_formatter(FuncFormatter(lambda v, _p: "0" if np.isclose(v, info["floor"], rtol=1e-8, atol=0)
                                               else U.compact(v)))
    for tick in ax.get_xticklabels() if dim == "x" else ax.get_yticklabels():
        if sample_info(tick.get_text())[0] == "empty":
            tick.set_color(U.GROUP_COLOR["empty"])


def reference_lines(ax, plot, xref="x", yref="y"):
    """Retain threshold lines and their labels, but not decorative group headings."""
    for shape in plot["layout"].get("shapes", []):
        if shape.get("type") != "line":
            continue
        xr, yr = shape.get("xref", "x"), shape.get("yref", "y")
        if xr.replace(" domain", "") != xref or yr.replace(" domain", "") != yref:
            continue
        style = dict(color=SOURCE_COLORS.get(shape.get("line", {}).get("color", "").lower(), P.GREY),
                     lw=0.55, ls=(0, (2, 2)), alpha=shape.get("opacity", 0.7))
        if shape["x0"] == shape["x1"]:
            ax.axvline(shape["x0"], **style)
        elif shape["y0"] == shape["y1"]:
            ax.axhline(shape["y0"], **style)
    for note in plot["layout"].get("annotations", []):
        text = label(note.get("text"))
        if text in U.GROUP_COLOR or note.get("xref") == "paper" or note.get("yref") == "paper":
            continue
        xr, yr = note.get("xref", "x"), note.get("yref", "y")
        if xr.replace(" domain", "") != xref or yr.replace(" domain", "") != yref:
            continue
        transform = ax.transAxes if "domain" in xr and "domain" in yr else (
            ax.get_yaxis_transform() if "domain" in xr else ax.get_xaxis_transform() if "domain" in yr else ax.transData)
        ax.text(note.get("x", 0), note.get("y", 0), wrapped(text, 45), transform=transform,
                fontsize=5.2, color=P.GREY, va="bottom", ha="left", clip_on=False)


def bar_figure(plot, indices, suffix=""):
    traces = [plot["data"][i] for i in indices]
    horizontal = all(t.get("orientation") == "h" for t in traces)
    xref, yref = traces[0].get("xaxis", "x"), traces[0].get("yaxis", "y")
    xs, ys = axis_layout(plot["layout"], xref), axis_layout(plot["layout"], yref)
    xi, yi = scale_info(traces, "x", xs), scale_info(traces, "y", ys)
    catdim = "y" if horizontal else "x"
    ci, ni = (yi, xi) if horizontal else (xi, yi)
    ncat = len(ci["categories"] or list(dict.fromkeys(v for t in traces for v in array(t[catdim]).tolist())))
    ratio = max(0.48, ncat * 0.019 + 0.18) if horizontal else 0.52
    fig, ax = plt.subplots(figsize=P.figsize("3col", ratio))
    fig.subplots_adjust(left=0.31 if horizontal else 0.12, right=0.975,
                        bottom=0.14 if horizontal else 0.29, top=0.90)
    stacked = plot["layout"].get("barmode") in ("stack", "relative")
    # Traces occupying different categories share the full bar width.
    disjoint = sum(array(t[catdim]).size for t in traces) == ncat
    groups = 1 if stacked or disjoint else len(traces)
    offsets = {}  # cumulative bar bottoms keyed by the actual category position
    for j, trace in enumerate(traces):
        x, y = coords(trace["x"], xi), coords(trace["y"], yi)
        pos, val = (y, x) if horizontal else (x, y)
        raw_val = numbers(trace["x"] if horizontal else trace["y"])
        width = 0.76 / groups
        positions = pos + (0 if groups == 1 else (j - (groups - 1) / 2) * width)
        base = np.array([offsets.get(float(p), 0.0) for p in pos]) if stacked else np.zeros(len(pos))
        color = color_for(trace, j)
        source_color = trace.get("marker", {}).get("color")
        if isinstance(source_color, list):
            colors = [SOURCE_COLORS.get(str(c).lower(), color) for c in source_color]
        else:
            colors = [color] * len(pos)
        labels = trace.get("customdata", trace.get("x", []))
        if not horizontal and isinstance(labels, list) and len(labels) == len(pos):
            colors = [U.GROUP_COLOR.get(sample_info(n)[0], c) if isinstance(n, str) else c
                      for n, c in zip(labels, colors)]
        translucent = trace.get("marker", {}).get("opacity", 1) < 0.7
        hatch = ("///" if translucent or trace.get("marker", {}).get("pattern") else
                 ("", "///", "...", "xx", "--", "++", "oo")[j % 7] if stacked else "")
        faces = [to_rgba(c, 0.18 if translucent else 1) for c in colors]
        kwargs = dict(color=faces, edgecolor=colors, linewidth=0.35, hatch=hatch,
                      label=label(trace.get("name", "")))
        bars = (ax.barh(positions, val, height=width, left=base, **kwargs) if horizontal
                else ax.bar(positions, val, width=width, bottom=base, **kwargs))
        if stacked:
            for p, v, b in zip(pos, raw_val, base):
                offsets[float(p)] = b + v
        if (len(traces) == 1 or disjoint) and ncat <= 25:
            texts = [f"{v:,.0f}" if float(v).is_integer() else f"{v:.3g}" for v in raw_val]
            ax.bar_label(bars, labels=texts, fontsize=5.1, padding=2, rotation=0)
        if ni["zero"]:
            for p, v in zip(positions, raw_val):
                if v == 0:
                    ax.text(ni["floor"], p, "0", fontsize=5, va="center") if horizontal else ax.text(p, ni["floor"], "0", fontsize=5, ha="center")
    configure_axis(ax, "x", xs, xi, traces, plot["layout"], xref)
    configure_axis(ax, "y", ys, yi, traces, plot["layout"], yref)
    if not horizontal and not yi["log"] and "range" not in ys:
        ax.set_ylim(bottom=0, top=max(ax.get_ylim()[1], 1) * 1.10)
    if horizontal and not xi["log"] and "range" not in xs:
        ax.set_xlim(left=0)
    handles, labels_ = ax.get_legend_handles_labels()
    sample_bars = disjoint and not horizontal and all(trace_sample(t)[0] for t in traces)
    if labels_ and not sample_bars:
        ax.legend(handles, labels_, loc="upper left", bbox_to_anchor=(0, -0.17 if horizontal else 1.03),
                  ncol=min(4, len(labels_)), fontsize=5.4, handlelength=1.2)
    reference_lines(ax, plot, xref, yref)
    fig.suptitle(wrapped(title_of(plot) + suffix), fontsize=7, y=0.99)
    return fig, []


def draw_trace(ax, trace, index, xi, yi, name=None, histogram_steps=False):
    x, y = coords(trace["x"], xi), coords(trace["y"], yi)
    group, rnd = trace_sample(trace)
    color = color_for(trace, index)
    marker = U.ROUND_MARKER.get(rnd, MARKERS[index % len(MARKERS)])
    dash = U.ROUND_LS.get(rnd, DASHES[index % len(DASHES)])
    mode = trace.get("mode", "markers" if trace.get("type") == "scattergl" else "lines")
    text = name if name is not None else label(trace.get("name", f"series {index + 1}"))
    if histogram_steps:
        ax.step(x, y, where="mid", color=color, ls=dash, lw=0.9, label=text)
        return
    if "lines" in mode:
        ax.plot(x, y, color=color, ls=dash, lw=0.9, label=text if "markers" not in mode else "_nolegend_")
    if "markers" in mode:
        symbols = trace.get("marker", {}).get("symbol", "circle")
        open_ = np.array(["open" in str(s) for s in symbols]) if isinstance(symbols, list) else np.repeat("open" in str(symbols), len(x))
        ring = trace.get("marker", {}).get("color") == "rgba(0,0,0,0)"
        if ring:
            open_[:] = True
            color = P.OKABE_ITO["vermillion"]
        if trace.get("type") == "scattergl":
            size = 8 if ring else 3
            ax.scatter(x, y, s=size, marker=marker, facecolors="none" if ring else color,
                       edgecolors=color, linewidths=0.6 if ring else 0,
                       alpha=trace.get("marker", {}).get("opacity", 0.65),
                       label=text, rasterized=True, zorder=4 if ring else 2)
        else:
            for hollow in (False, True):
                mask = open_ == hollow
                if mask.any():
                    ax.plot(x[mask], y[mask], ls="none", marker=marker, ms=3.0,
                            mfc="white" if hollow else color, mec=color, mew=0.65,
                            color=color, label=text if hollow == bool(open_[0]) else "_nolegend_")
    if "error_y" in trace and trace["error_y"].get("array") is not None:
        ax.errorbar(x, y, yerr=numbers(trace["error_y"]["array"]), fmt="none", color=color, lw=0.6, capsize=1.5)


def curve_figure(plot, indices, suffix="", facet_samples=False, individual=False, histogram_steps=False):
    traces = [plot["data"][i] for i in indices]
    xref, yref = traces[0].get("xaxis", "x"), traces[0].get("yaxis", "y")
    xs, ys = axis_layout(plot["layout"], xref), axis_layout(plot["layout"], yref)
    xi, yi = scale_info(traces, "x", xs), scale_info(traces, "y", ys)
    if histogram_steps and not axis_title(xs):
        # Log-length reports already store log10 coordinates and label ticks in nt.
        xs = {**xs, "title": {"text": "Read length (nt; log scale)" if "log transformed" in title_of(plot)
                              else "Read length (nt)"}}
    if individual:
        panels = [(f"{('UMI' if 'clonerank' in plot.get('source', '') else 'fragvec')} {j + 1:02d}", [j]) for j in range(len(traces))]
    elif facet_samples:
        panels = [(U.group_label(g), [j for j, t in enumerate(traces) if trace_sample(t)[0] == g])
                  for g in U.GROUP_ORDER if any(trace_sample(t)[0] == g for t in traces)]
    else:
        panels = [("", list(range(len(traces))))]
    ncols = (4 if len(panels) == 8 else 3) if len(panels) > 1 else 1
    nrows = math.ceil(len(panels) / ncols)
    ratio = max(0.52, nrows * 0.27 + 0.15) if len(panels) > 1 else 0.60
    fig, axes = plt.subplots(nrows, ncols, figsize=P.figsize("3col", ratio), squeeze=False,
                             sharex=len(panels) > 1, sharey=len(panels) > 1)
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.20 if len(panels) > 1 else 0.31,
                        top=0.87, hspace=0.82 if individual else 0.57, wspace=0.31)
    references = [j for j, t in enumerate(traces) if trace_sample(t)[0] is None] if facet_samples else []
    notes = []
    for pi, (panel, local) in enumerate(panels):
        ax = axes.flat[pi]
        for j in [*local, *references]:
            trace = traces[j]
            _, rnd = trace_sample(trace)
            name = (f"R{rnd}" + (" dominant" if "dominant" in trace.get("name", "") else "")) if facet_samples and rnd is not None else label(trace.get("name", f"series {j + 1}"))
            if individual:
                name = "_nolegend_"
            draw_trace(ax, trace, j, xi, yi, name, histogram_steps)
        configure_axis(ax, "x", xs, xi, traces, plot["layout"], xref)
        configure_axis(ax, "y", ys, yi, traces, plot["layout"], yref)
        reference_lines(ax, plot, xref, yref)
        if individual:
            ident = label(traces[local[0]].get("name", ""))
            ax.set_title(panel + "\n" + textwrap.shorten(ident, 34, placeholder="…"), fontsize=5.4)
            notes.append(f"{panel}: {ident}")
            if "per million fragment reads" in axis_title(ys):
                ax.set_ylabel("Reads per million\nfragment reads", fontsize=6)
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
        handles, labels_ = ax.get_legend_handles_labels()
        if handles:
            # Deduplicate only repeated reference labels, not biological series.
            unique = dict(zip(labels_, handles))
            ax.legend(unique.values(), unique.keys(), fontsize=5.1,
                      loc="upper right" if facet_samples else "upper left",
                      bbox_to_anchor=None if facet_samples else (0, -0.23),
                      ncol=2 if facet_samples else 3, handlelength=2.0)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    fig.suptitle(wrapped(title_of(plot) + suffix, 105), fontsize=7, y=0.99)
    if histogram_steps:
        notes.append("Stored histogram bin centers and counts are drawn as step curves, faceted by sample group; no rebinning.")
    if xi["zero"] or yi["zero"]:
        notes.append("Zeros on logarithmic axes are retained on a labelled plotting floor below the smallest positive value.")
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
    fig, ax = plt.subplots(figsize=P.figsize("3col", max(0.75, z.shape[0] * 0.019 + 0.22)))
    fig.subplots_adjust(left=0.29, right=0.85, bottom=0.16, top=0.87)
    cmap = matplotlib.colormaps["RdBu_r" if diverging else "viridis"].copy()
    cmap.set_bad("#e7e9ec")
    im = ax.imshow(np.ma.masked_invalid(z), interpolation="nearest", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    xticks = sample_tick_labels([trace], xs, plot["layout"], xref)
    if xticks is None:
        xticks = (np.arange(z.shape[1]), [label(v) for v in array(trace.get("x", np.arange(z.shape[1])))])
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
    if paper and index < len(paper):
        heading += " · " + label(paper[index]["text"])
    fig.suptitle(wrapped(heading), fontsize=7, y=0.98)
    return fig, ["Heatmap values, missing cells, color limits, and colorbar tick values are preserved from the report."]


def histogram_figure(plot, indices):
    trace = plot["data"][indices[0]]
    values = numbers(trace["x"])
    values = values[np.isfinite(values)]
    fig, ax = plt.subplots(figsize=P.figsize("3col", 0.52))
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.19, top=0.85)
    xs, ys = axis_layout(plot["layout"], "x"), axis_layout(plot["layout"], "y")
    notes = []
    binspec = trace.get("xbins", {})
    if binspec.get("size") is not None:
        size = float(binspec["size"])
        edges = np.arange(float(binspec.get("start", values.min())), float(binspec.get("end", values.max())) + size, size)
    elif np.all(values == np.floor(values)) and len(np.unique(values)) < 250:
        support, counts = np.unique(values, return_counts=True)
        ax.bar(support, counts, width=np.maximum(support * 0.15, 0.6) if xs.get("type") == "log" else 0.8, color=P.GREY)
        edges = None
        notes.append("Unbinned integer observations are displayed as exact frequencies at each observed family size; automatic browser bins are replaced by discrete counts.")
    else:
        edges = np.histogram_bin_edges(values, bins=int(trace.get("nbinsx", 40)))
        notes.append(f"The report supplies unbinned observations. The static display uses {len(edges) - 1} equal-width bins over the complete observed range, including zeros.")
    if edges is not None:
        counts, edges = np.histogram(values, bins=edges)
        ax.stairs(counts, edges, fill=True, color=P.OKABE_ITO["blue"])
        notes.append("Histogram bin edges: " + json_numbers(edges))
        notes.append("Histogram counts: " + json_numbers(counts))
    if xs.get("type") == "log":
        ax.set_xscale("log")
    if ys.get("type") == "log":
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)
    ax.set_xlabel(wrapped(axis_title(xs), 60))
    ax.set_ylabel(wrapped(axis_title(ys), 60))
    ax.xaxis.set_major_formatter(FuncFormatter(U.compact))
    ax.yaxis.set_major_formatter(FuncFormatter(U.compact))
    reference_lines(ax, plot)
    fig.suptitle(wrapped(title_of(plot)), fontsize=7, y=0.99)
    notes.append(f"All {len(values):,} embedded observations enter the histogram.")
    return fig, notes


def json_numbers(values):
    return "[" + ", ".join(f"{v:.12g}" for v in values) + "]"


def violin_figure(plot, indices):
    traces = [plot["data"][i] for i in indices]
    values = [numbers(t["y"]) for t in traces]
    fig, ax = plt.subplots(figsize=P.figsize("3col", 0.53))
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.29, top=0.87)
    for j, (trace, val) in enumerate(zip(traces, values)):
        val = val[np.isfinite(val)]
        color = color_for(trace, j)
        if np.ptp(val) == 0:
            ax.plot([j - 0.3, j + 0.3], [val[0], val[0]], color=color, lw=1)
        else:
            body = ax.violinplot([val], positions=[j], widths=0.8, showextrema=False,
                                 points=200, bw_method="scott")["bodies"][0]
            body.set_facecolor(to_rgba(color, 0.45))
            body.set_edgecolor(color)
            body.set_linewidth(0.65)
    ax.set_xticks(range(len(traces)), [short_sample(t.get("name", "")) for t in traces],
                  fontsize=5.3, rotation=55, ha="right")
    ys = axis_layout(plot["layout"], "y")
    ax.set_ylabel(axis_title(ys))
    if ys.get("type") == "log":
        ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(U.compact))
    heading = axis_title(ys) + " by sample"
    fig.suptitle(wrapped(heading), fontsize=7, y=0.99)
    return fig, ["Every embedded observation is retained. Violin outlines are redrawn with Matplotlib's Gaussian KDE (Scott bandwidth, 200 evaluation points), bounded by each observed range; their outlines can differ from Plotly's browser KDE.",
                 "Embedded observations per sample: " + "; ".join(f"{short_sample(t['name'])}: {len(v):,}" for t, v in zip(traces, values))]


def dominance_grid(plot, indices):
    """A549/MDA rows and R1–R5 columns, with paired x axes and one y range."""
    traces = [plot["data"][i] for i in indices]
    xs = axis_layout(plot["layout"], "x")
    ys = axis_layout(plot["layout"], "y")
    xi, yi = scale_info(traces, "x", xs), scale_info(traces, "y", ys)
    fig, axes = plt.subplots(2, 5, figsize=P.figsize("3col", 0.57),
                             sharex="col", sharey=True)
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.20, top=0.86,
                        wspace=0.18, hspace=0.22)
    positive_y = np.concatenate([numbers(t["y"]) for t in traces])
    positive_y = positive_y[np.isfinite(positive_y) & (positive_y > 0)]
    ymin = float(positive_y.min()) / 1.4
    for col, rnd in enumerate(range(1, 6)):
        column = [t for t in traces if trace_sample(t)[1] == rnd]
        xvalues = np.concatenate([numbers(t["x"]) for t in column])
        xvalues = xvalues[np.isfinite(xvalues) & (xvalues > 0)]
        xlimits = (float(xvalues.min()) / 1.4, float(xvalues.max()) * 1.4)
        for row, group in enumerate(("A549", "MDA")):
            ax = axes[row, col]
            n_rings = 0
            for j, trace in enumerate(traces):
                if trace_sample(trace) == (group, rnd):
                    draw_trace(ax, trace, j, xi, yi, name="_nolegend_")
                    if trace.get("marker", {}).get("color") == "rgba(0,0,0,0)":
                        n_rings += array(trace["x"]).size
            ax.text(0.97, 0.97, f"n={n_rings}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=6,
                    color=P.OKABE_ITO["vermillion"])
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(xlimits)
            ax.set_ylim(ymin, 1.25)
            ax.xaxis.set_major_formatter(FuncFormatter(U.compact))
            ax.yaxis.set_major_formatter(FuncFormatter(U.compact))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_minor_locator(NullLocator())
            ax.tick_params(labelsize=5.3)
            if row == 0:
                ax.set_title(f"R{rnd}", fontsize=6.5)
            else:
                ax.set_xlabel("Fragvec reads", fontsize=6)
            if col == 0:
                ax.set_ylabel("Dominance\n(top UMI share)", fontsize=6)
    for row, group in enumerate(("A549", "MDA")):
        bounds = axes[row, 0].get_position()
        fig.text(0.018, bounds.y0 + bounds.height / 2, group, rotation=90,
                 color=U.GROUP_COLOR[group], fontsize=7, fontweight="bold", va="center")
    fig.suptitle("UMI dominance within fragvecs", fontsize=7, y=0.98)
    fig.legend(handles=[
        Line2D([], [], color=U.GROUP_COLOR["A549"], marker="o", ls="none", ms=3, label="A549"),
        Line2D([], [], color=U.GROUP_COLOR["MDA"], marker="o", ls="none", ms=3, label="MDA"),
        Line2D([], [], color=P.OKABE_ITO["vermillion"], marker="o", mfc="none", ls="none", ms=4,
               label="dominant (ringed in source report)"),
    ], loc="lower center", ncol=3, bbox_to_anchor=(0.55, 0.005), fontsize=5.5)
    return fig, ["Requested layout: A549 and MDA rows, R1–R5 columns. Each round shares its x range between rows; both rows use the identical logarithmic y range. Every embedded point and dominance ring for these ten samples is retained. Orange n labels give the number of dominance rings in each panel. Other report samples are exported in a companion figure."]


def render(plot):
    """Yield (figure, view name, covered source-trace indices, display notes)."""
    traces = plot["data"]
    ids = list(range(len(traces)))
    if plot.get("source", "").startswith("clonerank_initial/") and title_of(plot).startswith("Dominance"):
        selected = [i for i in ids if trace_sample(traces[i])[0] in ("A549", "MDA")]
        other = [i for i in ids if i not in selected]
        fig, notes = dominance_grid(plot, selected)
        yield fig, "dominance_A549_MDA_by_round", selected, notes
        if other:
            fig, notes = curve_figure(plot, other, " · other report samples", facet_samples=True)
            yield fig, "dominance_other_samples", other, notes
        return
    if plot.get("source") == "umi/umi_report_cross.html":
        def literal(value):
            if isinstance(value, dict):
                if "bdata" in value:
                    return array(value).tolist()
                return {key: literal(item) for key, item in value.items()}
            if isinstance(value, list):
                return [literal(item) for item in value]
            return value
        plot = literal(plot)
        name, builder, caption = U.specification(plot)
        yield builder(plot), name, ids, [label(caption)]
        return
    types = {t.get("type", "scatter") for t in traces}
    if not types <= {"bar", "scatter", "scattergl", "heatmap", "histogram", "violin"}:
        raise ValueError(f"Unsupported trace types: {types}")
    if types == {"heatmap"}:
        for i in ids:
            fig, notes = heatmap_figure(plot, i)
            yield fig, f"heatmap_{i + 1}", [i], notes
        return
    if types == {"histogram"}:
        fig, notes = histogram_figure(plot, ids)
        yield fig, "histogram", ids, notes
        return
    if types == {"violin"}:
        fig, notes = violin_figure(plot, ids)
        yield fig, "violin", ids, notes
        return
    partitions = {}
    for i, trace in enumerate(traces):
        key = (trace.get("xaxis", "x"), trace.get("yaxis", "y"))
        partitions.setdefault(key, []).append(i)
    for part_number, ((xref, yref), indices) in enumerate(partitions.items(), 1):
        sub = [traces[i] for i in indices]
        kinds = {t.get("type", "scatter") for t in sub}
        suffix = ""
        if len(partitions) > 1:
            paper = [a for a in plot["layout"].get("annotations", []) if a.get("xref") == "paper"]
            if len(paper) == len(partitions):
                suffix = " · " + label(paper[part_number - 1]["text"])
            else:
                suffix = " · " + axis_title(axis_layout(plot["layout"], yref))
        if kinds == {"bar"} and plot["layout"].get("barmode") == "overlay" and len(sub) >= 10:
            fig, notes = curve_figure(plot, indices, suffix, facet_samples=True, histogram_steps=True)
        elif kinds == {"bar"}:
            fig, notes = bar_figure(plot, indices, suffix)
        elif kinds <= {"scatter", "scattergl"}:
            by_sample = sum(trace_sample(t)[0] is not None for t in sub) >= 10
            single_profiles = ("clonerank" in plot.get("source", "") and kinds == {"scatter"}) or (
                plot.get("source", "").startswith("fragvec/") and len(sub) == 15)
            fig, notes = curve_figure(plot, indices, suffix, facet_samples=by_sample, individual=single_profiles)
        else:
            raise ValueError(f"Unsupported mixed panel types: {kinds}")
        yield fig, f"panel_{part_number}" if len(partitions) > 1 else "plot", indices, notes


def finish(fig):
    """Normalize terminology after specialized renderers have created their text."""
    for text in fig.findobj(Text):
        text.set_text(label(text.get_text()))
    fig.canvas.draw()
    if not P.check_min_font(fig):
        raise ValueError("Figure contains text below 5 pt")
