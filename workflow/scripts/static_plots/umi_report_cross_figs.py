"""UMI static renderers reused from scripts/umi_report_cross_figs.py.

Workflow metadata supplies sample identities; panel counts and axes follow the report.
"""
from __future__ import annotations

import csv
import math
from html import unescape
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, NullLocator, PercentFormatter
import numpy as np

import palettes as P

SAMPLE_META = {}
GROUP_ORDER = ("initial", "A549", "MDA", "empty", "A549-SN", "MDA-SN")
# The reference script reserves vermillion for empty, and uses green for MDA
# in V01. Keep these identities local: the shared palette serves other figures.
GROUP_COLOR = {
    "initial": P.GREY,
    "A549": P.OKABE_ITO["blue"],
    "MDA": P.OKABE_ITO["bluish_green"],
    "empty": P.OKABE_ITO["vermillion"],
    "A549-SN": P.OKABE_ITO["reddish_purple"],
    "MDA-SN": P.OKABE_ITO["orange"],
}
GROUP_LS = {
    "A549": "-", "MDA": (0, (4, 1.5)), "empty": (0, (1, 1.3)),
    "A549-SN": (0, (4, 1.2, 1, 1.2)), "MDA-SN": (0, (6, 1.5)),
}
ROUND_LS = {
    0: (0, (1, 1.2)), 1: "-", 2: (0, (3, 1.2)), 3: (0, (6, 1.5)),
    4: (0, (4, 1.2, 1, 1.2)), 5: (0, (6, 1.2, 1, 1.2, 1, 1.2)),
}
ROUND_MARKER = {0: "o", 1: "s", 2: "^", 3: "D", 4: "v", 5: "P"}


def plain(text: str) -> str:
    return unescape(re.sub(r"<[^>]+>", "", text.replace("<br>", "\n")))


def configure(path):
    """Use config identities and group order, never infer identities from filenames."""
    global SAMPLE_META, GROUP_ORDER, GROUP_COLOR, GROUP_LS, ROUND_LS, ROUND_MARKER
    with path.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"No samples in metadata: {path}")
    rows.sort(key=lambda row: int(row["idx"]))
    SAMPLE_META = {row["name"]: (row["group"], int(row["round"])) for row in rows}
    if len(SAMPLE_META) != len(rows):
        raise ValueError(f"Duplicate sample names in metadata: {path}")
    groups = list(dict.fromkeys(row["group"] for row in rows if row["group"] != "initial"))
    GROUP_ORDER = tuple(["initial", *groups])
    GROUP_COLOR = P.workflow_group_colors(groups)
    dashes = ["-", (0, (3, 1.2)), (0, (6, 1.5)), (0, (4, 1.2, 1, 1.2))]
    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h"]
    GROUP_LS = {group: dashes[i % len(dashes)] for i, group in enumerate(groups)}
    rounds = sorted({rnd for _, rnd in SAMPLE_META.values()})
    ROUND_LS = {rnd: dashes[i % len(dashes)] for i, rnd in enumerate(rounds)}
    ROUND_MARKER = {rnd: markers[i % len(markers)] for i, rnd in enumerate(rounds)}


def present_groups(traces, key):
    present = {trace[key] for trace in traces}
    unknown = present - set(GROUP_ORDER)
    if unknown:
        raise ValueError(f"Plot groups absent from workflow metadata: {sorted(unknown)}")
    return [group for group in GROUP_ORDER if group in present]


def facet_axes(groups, **kwargs):
    ncols = min(3, max(1, len(groups)))
    nrows = max(1, math.ceil(len(groups) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=P.figsize("3col", 0.32 * nrows + 0.15),
                             squeeze=False, **kwargs)
    for ax in list(axes.flat)[len(groups):]:
        ax.set_visible(False)
    return fig, axes


def sample_parts(sample: str) -> tuple[str, int]:
    if sample in SAMPLE_META:
        return SAMPLE_META[sample]
    raise ValueError(f"Sample is absent from workflow metadata: {sample}")


def short_sample(sample: str) -> str:
    group, rnd = sample_parts(sample)
    return "initial R0" if group == "initial" else f"{group} R{rnd}"


def group_label(group: str) -> str:
    return "empty (control)" if group == "empty" else group


def compact(value: float, _pos=None) -> str:
    for divisor, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(value) >= divisor:
            return f"{value / divisor:g}{suffix}"
    return f"{value:g}"


def heading(fig, text: str) -> None:
    fig.suptitle(text, fontsize=7, y=0.99)


def panel_label(ax, letter: str) -> None:
    ax.text(-0.13, 1.04, letter, transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="bottom", ha="right")


def sample_axis(ax, names: list[str]) -> None:
    """Retain report order and label every sample; control also has a text key."""
    ax.set_xticks(np.arange(len(names)), [short_sample(n) for n in names],
                  rotation=55, ha="right", fontsize=5.5)
    ax.set_xlim(-0.65, len(names) - 0.35)
    previous = None
    for i, (label, name) in enumerate(zip(ax.get_xticklabels(), names)):
        group, _ = sample_parts(name)
        if group == "empty":
            label.set_color(GROUP_COLOR[group])
        if previous is not None and group != previous:
            ax.axvline(i - 0.5, color=P.GREY, lw=0.4, ls=(0, (2, 2)), alpha=0.5)
        previous = group


def fig_trajectory(plot: dict):
    fig, ax = plt.subplots(figsize=P.figsize("3col", 0.52))
    fig.subplots_adjust(left=0.10, right=0.75, bottom=0.16, top=0.86)
    annotations = [plain(a["text"]) for a in plot["layout"].get("annotations", [])]
    ends = []
    baseline = None
    for trace in plot["data"]:
        group = trace["name"]
        color, marker = GROUP_COLOR[group], P.group_marker(group)
        x, y = np.asarray(trace["x"]), np.asarray(trace["y"])
        error = np.asarray(trace["error_y"]["array"])
        lower_bounds = np.asarray([row[2] for row in trace["customdata"]], dtype=bool)
        start = 0
        if sample_parts(trace["customdata"][0][0])[0] == "initial":
            candidate = (x[0], y[0], error[0], bool(lower_bounds[0]))
            if baseline is not None and candidate != baseline:
                raise ValueError("Trajectory arms disagree on the shared initial library")
            baseline = candidate
            start = 1
        for j in range(len(x) - 1):
            ax.plot(x[j:j + 2], y[j:j + 2], color=color, ls=GROUP_LS[group],
                    lw=1.0, alpha=0.45 if x[j + 1] - x[j] > 1 else 1.0)
        for j in range(start, len(x)):
            ax.errorbar(x[j], y[j], yerr=None if lower_bounds[j] else error[j],
                        marker=marker, ms=3.5, mfc="white" if lower_bounds[j] else color,
                        mec=color, color=color, elinewidth=0.65, capsize=1.8, mew=0.7,
                        zorder=4)
            if lower_bounds[j]:
                ax.annotate("", (x[j], y[j] * 1.5), (x[j], y[j]),
                            arrowprops={"arrowstyle": "->", "color": color, "lw": 0.7})
        label = next((a for a in annotations if a.startswith(group + " ")), group)
        if group == "empty":
            label = label.replace("empty", "empty (control)", 1)
        ends.append((x[-1], y[-1], label, color))
    if baseline is not None:
        x0, y0, se0, lb0 = baseline
        ax.errorbar(x0, y0, yerr=None if lb0 else se0, marker="*", ms=6,
                    color=P.GREY, mfc="white" if lb0 else P.GREY,
                    capsize=1.8, elinewidth=0.65, zorder=6)
        initial_label = next((a for a in annotations if a.startswith("initial library")), "initial library")
        ax.annotate(initial_label, (x0, y0), xytext=(8, 10), textcoords="offset points",
                    fontsize=5.8, color=P.GREY)
    ax.set_yscale("log")
    rounds = sorted({value for trace in plot["data"] for value in trace["x"]})
    ax.set_xlim(min(rounds) - 0.25, max(rounds) + 0.35)
    values = [v for t in plot["data"] for v in t["y"]]
    ax.set_ylim(min(values) * 0.55, max(values) * 1.8)
    ax.yaxis.set_major_formatter(FuncFormatter(compact))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xticks(rounds, ["R0\n(initial)" if r == 0 else f"R{r}" for r in rounds])
    ax.set_xlabel("selection round")
    ax.set_ylabel(r"Effective lineage complexity, $C_{\mathrm{eff}}$")
    # End labels retain the HTML fold-change annotation. Spread only their text
    # positions in log space; leader lines terminate at the actual observations.
    occupied = -np.inf
    for x, y, label, color in sorted(ends, key=lambda end: end[1]):
        log_y = max(np.log10(y), occupied + 0.17)
        occupied = log_y
        ax.annotate(label, (x, y), xytext=(max(rounds) + 0.65, 10 ** log_y),
                    fontsize=5.8, color=color, va="center", annotation_clip=False,
                    arrowprops={"arrowstyle": "-", "color": color, "lw": 0.5})
    heading(fig, "Effective complexity by selection round")
    return fig


def fig_sample_bars(plot: dict, kind: str):
    fig, ax = plt.subplots(figsize=P.figsize("3col", 0.48))
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.27, top=0.81)
    traces = plot["data"]
    names = traces[0]["x"]
    colors = [GROUP_COLOR[sample_parts(name)[0]] for name in names]
    x = np.arange(len(names))
    labels = {
        "richness": [traces[0]["name"].replace("distinct UMIs, RAREFIED", "Rarefied"),
                     "Raw distinct UMIs (depth-dependent)"],
        "concentration": ["D50", "D90"],
        "collisions": ["Observed merges (edit distance ≤ 1)", "Expected by chance"],
    }[kind]
    for i, trace in enumerate(traces):
        if trace["x"] != names:
            raise ValueError("Sample bar traces use different sample orders")
        edge = colors if kind != "collisions" or i == 0 else [P.GREY] * len(names)
        face = edge if i == 0 else [to_rgba(c, 0.18) for c in edge]
        ax.bar(x + (i - 0.5) * 0.36, trace["y"], width=0.34,
               color=face, edgecolor=edge, linewidth=0.45, hatch="" if i == 0 else "////")
    sample_axis(ax, names)
    ax.yaxis.set_major_formatter(FuncFormatter(compact))
    if kind == "richness":
        ax.set_ylabel("Distinct corrected UMIs")
        ax.set_ylim(bottom=0)
    elif kind == "concentration":
        ax.set_ylabel("Species reaching 50% / 90% of reads (%)", fontsize=6.5)
        ax.set_ylim(0, 100)
    else:
        ax.set_ylabel("UMI merges")
        ax.yaxis.set_minor_locator(NullLocator())
        # The current report has no zeros; retain any future ones on a named floor.
        positive = [v for t in traces for v in t["y"] if v > 0]
        floor = min(positive) / 3 if positive else 0.1
        ax.set_ylim(floor, max(positive) * 2 if positive else 1)
        ax.set_yscale("log")
        for i, trace in enumerate(traces):
            for j, value in enumerate(trace["y"]):
                if value == 0:
                    ax.plot(j + (i - 0.5) * 0.36, floor, marker="v", ms=3, color=colors[j])
                    ax.annotate("0", (j + (i - 0.5) * 0.36, floor),
                                xytext=(0, 4), textcoords="offset points", ha="center", fontsize=5)
        ax.set_ylim(floor, max(positive) * 2 if positive else 1)
    handles = [Patch(facecolor=P.GREY, edgecolor=P.GREY, label=labels[0]),
               Patch(facecolor=to_rgba(P.GREY, 0.18), edgecolor=P.GREY,
                     hatch="////", label=labels[1])]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.015),
              ncol=2, fontsize=5.8, borderaxespad=0)
    heading(fig, plain(plot["layout"]["title"]["text"]))
    if "empty" in GROUP_COLOR:
        fig.text(0.985, 0.005, "empty = no-cell control", ha="right", fontsize=5.3, color=GROUP_COLOR["empty"])
    return fig


def raw_distinct_plot(plots: list[dict]) -> dict:
    """Select the full-depth distinct-corrected-UMI series from the richness plot."""
    matches = [(plot, trace) for plot in plots for trace in plot["data"]
               if trace.get("name", "").lower().startswith("raw distinct umis")]
    if len(matches) != 1:
        raise ValueError(f"Expected one raw distinct UMI series; found {len(matches)}")
    plot, trace = matches[0]
    return {"id": plot["id"], "data": [trace],
            "layout": {"title": {"text": "Raw distinct corrected UMIs"}}}


def fig_raw_distinct(plot: dict):
    """One annotated bar per sample, using only the selected raw-count series."""
    if len(plot["data"]) != 1:
        raise ValueError("Raw distinct UMI figure requires exactly one series")
    trace = plot["data"][0]
    names = trace["x"]
    values = np.asarray(trace["y"], dtype=float)
    if not values.size or not np.all(np.isfinite(values) & (values >= 0) & (values == np.floor(values))):
        raise ValueError("Raw distinct UMI counts must be nonnegative integers")
    colors = [GROUP_COLOR[sample_parts(name)[0]] for name in names]
    fig, ax = plt.subplots(figsize=P.figsize("3col", 0.48))
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.27, top=0.89)
    bars = ax.bar(np.arange(len(names)), values, color=colors, width=0.72)
    ax.bar_label(bars, labels=[f"{value:,.0f}" for value in values], padding=3, fontsize=5.4)
    sample_axis(ax, names)
    ax.set_ylim(0, max(float(values.max()), 1.0) * 1.15)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:,.0f}"))
    ax.set_ylabel("Distinct corrected UMIs (count)")
    heading(fig, "Raw distinct corrected UMIs")
    if "empty" in GROUP_COLOR:
        fig.text(0.985, 0.005, "empty = no-cell control", ha="right", fontsize=5.3,
                 color=GROUP_COLOR["empty"])
    return fig


def trace_round(trace: dict) -> int:
    match = re.search(r"<extra>(.*?)</extra>", trace["hovertemplate"])
    if not match:
        raise ValueError("Curve trace does not identify its sample")
    return sample_parts(plain(match[1]))[1]


def fig_curves(plot: dict, kind: str):
    groups = present_groups(plot["data"], "legendgroup")
    fig, axes = facet_axes(groups, sharex=True, sharey=True)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.20, top=0.86,
                        wspace=0.22, hspace=0.39)
    for index, (ax, group) in enumerate(zip(axes.flat, groups)):
        for trace in plot["data"]:
            if trace["legendgroup"] != group:
                continue
            rnd = trace_round(trace)
            n = len(trace["x"])
            positions = np.unique(np.linspace(0, n - 1, 6, dtype=int))
            ax.plot(trace["x"], trace["y"], color=GROUP_COLOR[group], lw=0.9,
                    ls=ROUND_LS[rnd], marker=ROUND_MARKER[rnd], ms=2.7,
                    markevery=positions, mfc="white", mew=0.6)
        ax.set_title(group_label(group), color=GROUP_COLOR[group], fontsize=6.5)
        panel_label(ax, chr(65 + index))
        if kind == "rank":
            ax.set_xscale("log")
            ax.set_yscale("log")
            positive = [v for t in plot["data"] for v in t["y"] if v is not None and v > 0]
            ax.set_ylim(min(positive) * 0.8, max(positive) * 1.25)
            ax.set_xticks([1, 100, 10000, 1000000])
            ax.xaxis.set_major_formatter(FuncFormatter(compact))
            ax.yaxis.set_major_formatter(FuncFormatter(compact))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_minor_locator(NullLocator())
        else:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.03)
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])
            ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        if index % axes.shape[1] == 0:
            ax.set_ylabel("Reads" if kind == "rank" else "Cumulative reads")
    handles = [Line2D([], [], color=P.GREY, ls=ROUND_LS[r], marker=ROUND_MARKER[r],
                      mfc="white", ms=3, lw=0.9, label="R0 (initial)" if r == 0 else f"R{r}")
               for r in ROUND_LS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.0),
               ncol=6, fontsize=5.8, handlelength=2.8)
    fig.text(0.54, 0.115, "UMI species rank" if kind == "rank"
             else "Cumulative species (ranked from most abundant)", ha="center", fontsize=7)
    heading(fig, plain(plot["layout"]["title"]["text"]))
    return fig


def fig_intersections(plot: dict):
    title = plain(plot["layout"]["title"]["text"])
    match = re.search(r"across (.+?) \(top (\d+) intersections\)", title)
    if not match:
        raise ValueError(f"Unsupported intersection title: {title}")
    group = match[1]
    counts = np.asarray(plot["data"][0]["y"])
    n = len(counts)
    members = [trace for trace in plot["data"] if trace.get("mode") == "markers+lines"]
    names = plot["layout"]["yaxis2"]["ticktext"]
    if len(members) != n or int(match[2]) != n:
        raise ValueError("Intersection counts and memberships disagree")
    fig = plt.figure(figsize=P.figsize("3col" if n > 10 else "2col", 0.51 if n > 10 else 0.65))
    grid = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.3], hspace=0.035,
                            left=0.17, right=0.985, bottom=0.10, top=0.86)
    ax = fig.add_subplot(grid[0])
    dots = fig.add_subplot(grid[1], sharex=ax)
    color = GROUP_COLOR[group]
    ax.bar(np.arange(n), counts, width=0.68, color=color, linewidth=0)
    # Keep the report's linear scale. Labels make even one-species intersections
    # legible without fabricating a visible minimum bar height.
    for i, value in enumerate(counts):
        ax.annotate(f"{value:,}", (i, value), xytext=(0, 3), textcoords="offset points",
                    rotation=90 if n > 10 else 0, ha="center", va="bottom", fontsize=5.2)
    ax.set_ylim(0, counts.max() * (1.31 if n > 10 else 1.16))
    ax.set_xlim(-0.65, n - 0.35)
    ax.set_ylabel("Corrected UMI species", fontsize=6.5)
    ax.yaxis.set_major_formatter(FuncFormatter(compact))
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    for row in range(len(names)):
        if row % 2 == 0:
            dots.axhspan(row - 0.48, row + 0.48, facecolor="#f4f5f7", edgecolor="none", zorder=0)
        dots.plot(np.arange(n), [row] * n, "o", color="#dfe3e8", ms=2.7, zorder=1)
    for i, trace in enumerate(members):
        if trace["x"] != [i] * len(trace["y"]):
            raise ValueError("Unexpected intersection column order")
        dots.plot(trace["x"], trace["y"], color=color, marker="o", ms=3.2, lw=0.9, zorder=3)
    dots.set_yticks(range(len(names)), [short_sample(name) for name in names], fontsize=5.8)
    dots.set_ylim(len(names) - 0.45, -0.55)
    dots.set_xticks([])
    dots.tick_params(length=0)
    for spine in dots.spines.values():
        spine.set_visible(False)
    if group == "empty":
        for label in dots.get_yticklabels()[1:]:
            label.set_color(color)
    heading(fig, f"{group_label(group)} · top {n} corrected-UMI intersections")
    fig.text(0.58, 0.015, "Connected dots identify the samples in each exclusive intersection",
             ha="center", fontsize=5.5)
    return fig


def fig_purity(plot: dict):
    groups = present_groups(plot["data"], "name")
    fig, axes = facet_axes(groups, sharex=True, sharey=True)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.18, top=0.85,
                        wspace=0.22, hspace=0.40)
    bins = plot["layout"]["xaxis"]["categoryarray"]
    by_group = {trace["name"]: trace for trace in plot["data"]}
    for index, (ax, group) in enumerate(zip(axes.flat, groups)):
        trace = by_group[group]
        ax.set_ylim(0, 115)
        ax.set_yticks([0, 50, 100])
        for category, value, total in zip(trace["x"], trace["y"], trace["customdata"]):
            pos = bins.index(category)
            ax.bar(pos, value, color=GROUP_COLOR[group], width=0.66)
            ax.annotate(f"{total:,}", (pos, value), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=5.0)
        for i, category in enumerate(bins):
            if category not in trace["x"]:
                ax.text(i, 4, "–", ha="center", fontsize=6, color=P.GREY)
        ax.set_xticks(range(len(bins)), bins, rotation=45, ha="right", fontsize=5.5)
        ax.set_title(group_label(group), fontsize=6.5, color=GROUP_COLOR[group])
        if index % axes.shape[1] == 0:
            ax.set_ylabel("Single ligand signature (%)", fontsize=6.5)
        panel_label(ax, chr(65 + index))
    heading(fig, "Ligand-signature purity by UMI group size")
    fig.text(0.54, 0.06, "UMI group size (reads)", ha="center", fontsize=7)
    fig.text(0.54, 0.005, "Numbers above bars = UMI groups assessed; – = no reported groups",
             ha="center", fontsize=5.5)
    return fig


def specification(plot: dict):
    title = plot["layout"]["title"]["text"]
    if title.startswith("Effective complexity by"):
        return "effective_complexity", fig_trajectory, (
            "Report C_eff values on a logarithmic axis; error bars are the embedded "
            "C_eff / sqrt(duplicate-pairs) standard errors. The shared initial library "
            "is drawn once. Original rounded fold-drop labels are retained. Faint "
            "segments bridge unobserved rounds and do not add measurements. Hollow "
            "symbols and upward arrows denote lower bounds if present.")
    if title == "Raw distinct corrected UMIs":
        return "raw_distinct_umis", fig_raw_distinct, (
            "One bar per sample shows the raw distinct corrected UMI count at its "
            "full observed sequencing depth. Exact counts are annotated above the "
            "bars on a linear axis starting at zero. Values and sample order are "
            "copied from the report's raw distinct UMI series; raw here means "
            "before rarefaction, after UMI correction. This count depends on read depth.")
    if title.startswith("Richness at a common"):
        return "rarefied_richness", lambda p: fig_sample_bars(p, "richness"), (
            "Solid bars reproduce richness rarefied to the common read depth printed "
            "in the legend; hatched bars show raw distinct corrected UMIs. Both series "
            "retain the report's sample order and linear axis. Raw richness is "
            "depth-dependent; near-saturation at the rarefaction depth can obscure "
            "differences between larger pools.")
    if title.startswith("Rank–abundance"):
        return "rank_abundance", lambda p: fig_curves(p, "rank"), (
            "Report curves, faceted by group, with common logarithmic axes. "
            "Dashes and marker shapes identify selection rounds. Every stored curve "
            "point is retained; markers highlight a subset of those same points. "
            "These are the report's sampled curves, not a reconstruction from raw reads. "
            "Curve length and shape depend on sequencing depth.")
    if title.startswith("Complexity: D50"):
        return "d50_d90", lambda p: fig_sample_bars(p, "concentration"), (
            "The percentage of distinct corrected-UMI species accounting for 50% "
            "(solid) or 90% (hatched) of reads. Lower values describe greater "
            "concentration. Values and sample order are copied from the report; "
            "the denominator depends on sequencing depth.")
    if title.startswith("Cumulative abundance"):
        return "cumulative_abundance", lambda p: fig_curves(p, "cumulative"), (
            "Cumulative-abundance curves, faceted by group on common linear "
            "percentage axes. Species run from most to least abundant, as in the "
            "report (rather than the least-first convention of a classical Lorenz "
            "curve). All stored points, including zeros, are retained.")
    if title.startswith("Shared corrected-UMI"):
        match = re.search(r"across (.+?) \(top (\d+) intersections\)", title)
        if match:
            return "species_overlap_" + match[1], fig_intersections, (
                f"The report's top {match[2]} exclusive intersections for {match[1]}, "
                "including the initial library. Bar heights retain the original "
                "linear scale and order; exact counts label every bar, including "
                "very small intersections. Connected dots reproduce the embedded "
                "sample memberships. Omitted intersections are not reconstructed "
                "or treated as zero. Counts are sensitive to unequal read depths.")
    if title.startswith("Validity:"):
        return "ligand_signature_purity", fig_purity, (
            "Percentage of UMI groups with a single ligand signature, faceted by "
            "sample group. Percentages and denominators are copied from the report; "
            "the tooltip denominators are printed above bars. A dash marks an "
            "unreported size bin, not a zero purity measurement. No confidence "
            "intervals are inferred.")
    if title.startswith("Collision null model:"):
        return "corrections_vs_chance", lambda p: fig_sample_bars(p, "collisions"), (
            "Observed UMI merges at edit distance ≤ 1 (solid group colours) versus "
            "the embedded chance expectation (grey hatching). The original "
            "logarithmic axis, fractional expected counts, and sample order are "
            "retained. Any zero counts are explicitly labelled at a plotting floor.")
    raise ValueError(f"No static renderer for report figure: {title}")
