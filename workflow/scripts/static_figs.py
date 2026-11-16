#!/usr/bin/env python
"""Print-ready figures 13-22: cross-group AGREEMENT (overlap, concordance, ordination), from the
`static_tables.py` tables -- the last level of the analysis series, after static_analysis.py's 01-12.

The run-QC figures this set used to open with (raw summary, read-length/quality ECDFs, stage
waterfall, filter losses, decode QC, error decomposition, UMI and fragvec layer output) are drawn
ONCE, by static_analysis.py, in run order; the error decomposition moved into its figure 04. Numbers
are not written anywhere but FIRST + position in FIGURES.

Ported from the cells project's `batch_effect_static_figs.py`, within-batch mode only: one config
is one run, so there is no batch channel. Colour, marker and dash identify the GROUP (same colours as
the HTML reports); round is a printed numeric marker; same-round group pairs are faceted by family.
The script only draws: every number comes from a table, and every figure is redrawn rather than
screenshotted so nothing depends on a hover tooltip or a page theme.

Writes 13_*.pdf .. 22_*.pdf (vector only), figure_legends.md and figures_manifest.json (status per
figure; an unavailable one still gets a labelled placeholder so the set is always complete).
"""
from __future__ import annotations

import argparse
import json
import textwrap
from collections import namedtuple
from pathlib import Path

import numpy as np
import polars as pl

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
from matplotlib.lines import Line2D                                      # noqa: E402
from matplotlib.patches import Patch, Rectangle                          # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator   # noqa: E402

import static_style as S                                                 # noqa: E402

BLACK, GREY, PURP = S.BLACK, S.GREY, S.OKABE_ITO["reddish_purple"]
PAIR_LS = S.DASHES
Block = namedtuple("Block", "sid pair key_a key_b group group_b round")

#: Set by main() from run_manifest.json.
SAMPLE_META: dict[str, dict] = {}
BLOCKS: list[Block] = []
PARAMS: dict = {}
G: S.Groups
FIGURE_ISSUES: list[str] = []

WITHIN_KIND = {
    "same_round_groups": ("same round, different groups", "o", BLACK, BLACK),
    "same_group_cross_round": ("same group, different rounds", "s", PURP, "none"),
    "initial_reference": ("initial versus selected", "^", GREY, "none"),
    "different_group_and_round": ("different groups and rounds", "D", GREY, "none"),
}


class FigureUnavailable(ValueError):
    """An analysis explicitly has no scorable measurements; this is not a builder failure."""


# ----------------------------------------------------------------------------------- scaffolding

def panel_label(ax, letter: str, dx: float = -0.09, dy: float = 1.03) -> None:
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="right")


def _number(value) -> float:
    """Missing or non-finite measurements stay missing, never zero."""
    try:
        number = float(value)
        return number if np.isfinite(number) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _issue(reason: str) -> None:
    if reason not in FIGURE_ISSUES:
        FIGURE_ISSUES.append(reason)


def _tsv(out: Path, name: str) -> pl.DataFrame:
    data = pl.read_csv(out / name, separator="\t", comment_prefix="#", infer_schema_length=200000)
    if "key" in data.columns:
        for row in data.iter_rows(named=True):
            meta = SAMPLE_META.get(row["key"])
            if meta is None or any(f in row and row[f] != meta[f] for f in ("group", "round")):
                raise ValueError(f"{name}: sample row disagrees with the run manifest: {row['key']}")
    return data


def _check_keys(keys, source: str) -> None:
    missing = set(SAMPLE_META) - set(keys)
    if missing:
        _issue(f"{source}: missing sample rows: {', '.join(sorted(missing))}")


def _groups() -> list[str]:
    return sorted({r["group"] for r in SAMPLE_META.values()}, key=G.rank)


def _ordered_samples() -> list[dict]:
    return sorted(SAMPLE_META.values(), key=lambda r: (G.rank(r["group"]), r["round"]))


def _round_style(rnd: int, group: str, group_b: str | None = None) -> dict:
    """Round = dash + printed numeric marker; line colour is group A, marker colour group B."""
    index = sorted({r["round"] for r in SAMPLE_META.values()}).index(rnd)
    return dict(color=G.color(group), lw=.8, ls=PAIR_LS[index % len(PAIR_LS)], marker=f"${rnd}$",
                mfc=G.color(group_b or group), mec=G.color(group_b or group), ms=5.5, label=f"R{rnd}")


def _missing(ax, xs, label: str) -> None:
    """An unavailable measurement is marked in axes space, not at numeric zero."""
    xs = list(xs)
    if xs:
        ax.plot(xs, [.035] * len(xs), transform=ax.get_xaxis_transform(), marker="x", ms=3.2,
                color=BLACK, ls="none", clip_on=False, zorder=8)
        if not any(t.get_gid() == "missing-key" for t in ax.figure.texts):
            ax.figure.text(.99, .003, "× unavailable; not zero", ha="right", va="bottom", fontsize=5,
                           color=BLACK, gid="missing-key")
        _issue(label)


def _zero_floor_ticks(ax, floor: float, ticks: list[float]) -> None:
    """Label the plotting floor '0' on a log axis: zero-sharing pairs are a result, not missing."""
    ax.yaxis.set_major_locator(FixedLocator([floor] + ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: "0" if np.isclose(v, floor) else f"{v:,.0f}"))


def _family_axes():
    families = list(dict.fromkeys(b.pair for b in BLOCKS))
    if not families:
        raise FigureUnavailable("no same-round group pairs in this run")
    nrow = (len(families) + 1) // 2
    fig, axes = plt.subplots(nrow, 2, figsize=S.figsize("3col", max(.4, .235 * nrow)), squeeze=False)
    for i, (family, ax) in enumerate(zip(families, axes.flat)):
        blk = next(b for b in BLOCKS if b.pair == family)
        ax.set_title("→", fontsize=6.5)
        ax.text(.03, 1.03, f"A: {blk.group}", transform=ax.transAxes, va="bottom", fontsize=6.5,
                color=G.color(blk.group))
        ax.text(.97, 1.03, f"B: {blk.group_b}", transform=ax.transAxes, va="bottom", ha="right",
                fontsize=6.5, color=G.color(blk.group_b))
        panel_label(ax, chr(65 + i), dx=-.10)
    for ax in list(axes.flat)[len(families):]:
        ax.set_visible(False)
    fig.subplots_adjust(hspace=.72, wspace=.30, bottom=.075, top=.96)
    return fig, list(zip(families, axes.flat))


def _pair_rows(df: pl.DataFrame, blk: Block) -> pl.DataFrame:
    if "comparison_id" in df.columns:
        return df.filter(pl.col("comparison_id") == blk.sid)
    return df.filter(((pl.col("key_a") == blk.key_a) & (pl.col("key_b") == blk.key_b))
                     | ((pl.col("key_a") == blk.key_b) & (pl.col("key_b") == blk.key_a)))


def _within_pair_curves(out: Path, view: str):
    filename, xcol, ycol, xlabel, ylabel, stem = {
        "threshold": ("umi_overlap.tsv", "threshold_min_reads", "shared", "Minimum reads in BOTH samples",
                      "Shared UMIs", "F11_umi_shared_vs_threshold"),
        "order": ("vector_concordance.tsv", "stratum_n_frag", "lin_ccc", "Payload fragments / vector",
                  "Lin's CCC", "F14_clonotype_agreement_by_fragment_order"),
        "detection": ("detection_vs_abundance.tsv", "a_count_lo", "detection_rate",
                      "Clone count in A (bin lower edge)", "Detected in B (%)", "F15_clonotype_detection_vs_abundance"),
    }[view]
    data = _tsv(out, filename)
    if view == "order":
        data = data.filter((pl.col("stratum_n_frag") != "all") & (pl.col("filter") == "abundant_ge10_either"))
    if not any(np.isfinite(_number(v)) for v in data[ycol]):
        raise FigureUnavailable(f"{filename}: no scorable {view} measurements")
    expected_x = {"threshold": [1, 2, 3, 5, 10], "order": list(range(1, 7)),
                  "detection": [1, 2, 3, 5, 10, 30, 100, 300, 1000]}[view]
    fig, facets = _family_axes()
    for family, ax in facets:
        for blk in (b for b in BLOCKS if b.pair == family):
            rows = list(_pair_rows(data, blk).iter_rows(named=True))
            if not rows:
                _issue(f"{filename}: no rows for {blk.sid}")
                ax.plot([], [], **{**_round_style(blk.round, blk.group, blk.group_b), "label": f"R{blk.round} unavailable"})
                continue
            values = {_number(r[xcol]): _number(r[ycol]) for r in rows if np.isfinite(_number(r[xcol]))}
            x = np.array(sorted(set(expected_x) | set(values)), dtype=float)
            y = np.array([values.get(v, np.nan) for v in x])      # missing bins are gaps, not lines
            if view == "threshold":
                y = np.where(y == 0, .3, y)
            elif view == "detection":
                y *= 100
            ax.plot(x, y, **_round_style(blk.round, blk.group, blk.group_b))
            _missing(ax, x[~np.isfinite(y)], f"{filename}: unavailable bins for {blk.sid}")
        if view == "threshold":
            ax.set_yscale("log")
            ax.set_xticks([1, 2, 3, 5, 10])
            ax.set_ylim(bottom=.18)
            _zero_floor_ticks(ax, .3, [1, 10, 100, 1000, 10000, 100000])
        elif view == "detection":
            ax.set_xscale("log")
            ax.set_ylim(0, 104)
        else:
            ax.axhline(0, color=GREY, lw=.5)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_ylim(-1.05, 1.05)
        ax.legend(loc="best", fontsize=5.5, ncol=3, frameon=False)
    fig.supxlabel(xlabel, fontsize=7)
    fig.supylabel(ylabel, fontsize=7)
    return fig, stem


def _within_ccc(out: Path):
    feature = _tsv(out, "feature_concordance.tsv")
    vector = _tsv(out, "vector_concordance.tsv").filter(
        (pl.col("filter") == "present_in_either") & (pl.col("stratum_n_frag") == "all"))
    fig, facets = _family_axes()
    bounds = []
    for family, ax in facets:
        for df, label, color, marker, fill in ((feature, "library features", BLACK, "s", "full"),
                                               (vector, "fragment vectors", GREY, "o", "none")):
            xx, yy = [], []
            for blk in (b for b in BLOCKS if b.pair == family):
                sel = _pair_rows(df, blk)
                row = sel.row(0, named=True) if sel.height else {}
                y, lo, hi = (_number(row.get(c)) for c in ("lin_ccc", "ccc_lo", "ccc_hi"))
                xx.append(blk.round)
                yy.append(y)
                if not np.isfinite(y):
                    _missing(ax, [blk.round], f"{label}: CCC unavailable for {blk.sid}")
                elif np.isfinite(lo) and np.isfinite(hi):
                    # percentile intervals need not enclose the estimate: draw the actual endpoints
                    ax.vlines(blk.round, lo, hi, color=color, lw=.8)
                    ax.hlines([lo, hi], blk.round - .065, blk.round + .065, color=color, lw=.8)
                    bounds.extend((lo, hi, y))
                else:
                    bounds.append(y)
                    _issue(f"{label}: bootstrap interval unavailable for {blk.sid}")
            ax.plot(xx, yy, marker=marker, ms=3.8, color=color, lw=.8, fillstyle=fill,
                    ls="-" if fill == "full" else (0, (3, 1.5)), label=label)
        ax.set_xticks(sorted({b.round for b in BLOCKS if b.pair == family}))
        ax.axhline(0, color=GREY, lw=.5)
    low, high = (min(-.1, min(bounds) - .06), max(1.04, max(bounds) + .04)) if bounds else (-1.05, 1.05)
    for _, ax in facets:
        ax.set_ylim(low, high)
    fig.legend(handles=facets[0][1].get_legend_handles_labels()[0], loc="lower center", ncol=2,
               fontsize=6, bbox_to_anchor=(.5, .017))
    fig.subplots_adjust(bottom=.115)
    fig.supxlabel("Selection round", fontsize=7)
    fig.supylabel("Lin's concordance correlation coefficient", fontsize=7)
    return fig, "F13_clonotype_agreement_ccc"


def _ordination_info(out: Path) -> dict:
    return json.loads((out / "ordination_info.json").read_text())


def _within_pca(out: Path):
    by = {r["key"]: r for r in _tsv(out, "ordination_coords.tsv").iter_rows(named=True)}
    _check_keys(by, "ordination_coords.tsv")
    info = _ordination_info(out)
    if info.get("status") != "computed":
        raise FigureUnavailable(f"PCA: {info.get('status')}")
    variance = [_number(v) for v in (info.get("explained_variance") or [None, None])[:2]]
    fig, ax = plt.subplots(figsize=S.figsize("3col", .63))
    for gi, group in enumerate(_groups()):
        rows = [r for r in _ordered_samples() if r["group"] == group]
        x = np.array([_number(by.get(r["key"], {}).get("PC1")) for r in rows])
        y = np.array([_number(by.get(r["key"], {}).get("PC2")) for r in rows])
        ax.plot(x, y, ls=PAIR_LS[gi % len(PAIR_LS)], lw=.65, marker=G.marker(group), ms=5.0,
                mfc="white", mec=G.color(group), color=G.color(group), label=group)
        for r, xx, yy in zip(rows, x, y):
            if np.isfinite(xx) and np.isfinite(yy):
                ax.annotate(f"R{r['round']}", (xx, yy), xytext=(4, 4), textcoords="offset points",
                            fontsize=5.5, color=BLACK)
            else:
                _issue(f"ordination coordinates unavailable: {r['key']}")
    for axis, v, label in ((ax.xaxis, variance[0], "PC1"), (ax.yaxis, variance[1], "PC2")):
        axis.set_label_text(f"{label} ({100 * v:.1f}% of variance)" if np.isfinite(v) else f"{label} (variance unavailable)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=6, frameon=False,
              title="group; labels = round", title_fontsize=6)
    fig.subplots_adjust(right=.76, bottom=.14, top=.95)
    return fig, "F16_clonotype_ordination_pca"


def _complexity_fit(out: Path, data: pl.DataFrame | None = None) -> dict:
    """The one recorded mask/fit is authoritative for both the plot and its legend; never refit."""
    fit = json.loads((out / "complexity_fit.json").read_text())
    data = _tsv(out, "complexity_dependence.tsv") if data is None else data
    included = [r for r in data.iter_rows(named=True) if r["fit_included"] is True]
    if len(included) != fit["n"] or set(fit["comparison_ids"]) != {r["comparison_id"] for r in included}:
        raise ValueError("complexity_fit.json disagrees with the fit_included mask")
    return fit


def _within_complexity(out: Path):
    data = _tsv(out, "complexity_dependence.tsv")
    fit = _complexity_fit(out, data)
    xv = np.array([_number(v) for v in data["clones_a"]])
    yv = np.array([_number(v) for v in data["mean_abs_log2fc"]])
    good = (xv > 0) & np.isfinite(yv)
    if not good.any():
        raise FigureUnavailable("No finite complexity dependence measurements at positive clone depth")
    fig, facets = _family_axes()
    xlim = (float(xv[good].min()) * .8, float(xv[good].max()) * 1.25)
    ylim = (0, max(.1, float(yv[good].max()) * 1.20))
    slope, intercept = _number(fit.get("slope")), _number(fit.get("intercept"))
    has_fit = fit.get("status") == "computed" and np.isfinite(slope) and np.isfinite(intercept) and fit["n"] > 2
    for family, ax in facets:
        if has_fit:
            xx = np.geomspace(*xlim, 60)
            ax.plot(xx, slope * np.log10(xx) + intercept, color=GREY, ls=(0, (4, 2)), lw=.7)
        for blk in (b for b in BLOCKS if b.pair == family):
            sel = _pair_rows(data, blk)
            row = sel.row(0, named=True) if sel.height else {}
            x, y = _number(row.get("clones_a")), _number(row.get("mean_abs_log2fc"))
            if not (x > 0 and np.isfinite(y)):
                _issue(f"complexity statistic unavailable: {blk.sid}")
                ax.plot([], [], **{**_round_style(blk.round, blk.group, blk.group_b), "label": f"R{blk.round} unavailable"})
                continue
            ax.plot([x], [y], **{**_round_style(blk.round, blk.group, blk.group_b), "ls": "none"})
            if row.get("fit_included") is not True:
                ax.plot([x], [y], marker="o", ms=9, mfc="none", mec=BLACK, ls="none", lw=.6)
        ax.set_xscale("log")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.legend(loc="best", ncol=3, fontsize=5.5, frameon=False)
    handles = [Line2D([], [], color=GREY, ls=(0, (4, 2)), lw=.7,
                      label=f"pooled descriptive OLS (n={fit['n']})" if has_fit else
                      f"pooled descriptive fit: {fit.get('status', 'unavailable')}")]
    if not has_fit:
        _issue(f"descriptive complexity fit: {fit.get('status', 'unavailable')}")
    if any(r["fit_included"] is not True for r in data.iter_rows(named=True)):
        handles.append(Line2D([], [], marker="o", mfc="none", mec=BLACK, ls="none", label="ring: excluded from fit"))
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=6, bbox_to_anchor=(.5, .017))
    fig.subplots_adjust(bottom=.115)
    fig.supxlabel("Clone depth of group A (clones)", fontsize=7)
    fig.supylabel("Mean |log$_2$ fold change| (B minus A)", fontsize=7)
    return fig, "F17_clonotype_complexity_dependence"


def _within_overlap(out: Path, clonotype: bool = False) -> pl.DataFrame:
    name = "clonotype_overlap.tsv" if clonotype else "umi_overlap.tsv"
    threshold = "threshold_min_count" if clonotype else "threshold_min_reads"
    data = _tsv(out, name).filter(pl.col(threshold) == 1)
    kinds, seen = [], set()
    for row in data.iter_rows(named=True):
        a, b = SAMPLE_META[row["key_a"]], SAMPLE_META[row["key_b"]]
        pair = frozenset((row["key_a"], row["key_b"]))
        if len(pair) != 2 or pair in seen:
            raise ValueError(f"{name}: self or duplicate pair {row['key_a']}, {row['key_b']}")
        seen.add(pair)
        kinds.append("initial_reference" if a["round"] == 0 or b["round"] == 0 else
                     "same_round_groups" if a["round"] == b["round"] and a["group"] != b["group"] else
                     "same_group_cross_round" if a["group"] == b["group"] else "different_group_and_round")
    expected = len(SAMPLE_META) * (len(SAMPLE_META) - 1) // 2
    if len(seen) != expected:
        _issue(f"{name}: {len(seen)} of {expected} overlap pairs available")
    return data.with_columns(pl.Series("kind", kinds, dtype=pl.String))


def _kind_handles() -> list:
    return [Line2D([], [], ls="none", marker=marker, ms=4, mec=BLACK, mfc="white", label=label)
            for label, marker, _, _ in WITHIN_KIND.values()] + [
        Line2D([], [], ls="none", marker="o", ms=4, color=G.color(g), label=g) for g in _groups()]


def _overlap_mark(row: dict, marker: str, ms: float = 4) -> dict:
    """Split markers show both endpoint groups; shape is the pair kind."""
    return dict(ls="none", marker=marker, ms=ms, fillstyle="left", mec=BLACK, mew=.35,
                mfc=G.color(SAMPLE_META[row["key_a"]]["group"]), mfcalt=G.color(SAMPLE_META[row["key_b"]]["group"]))


def _within_heatmap(out: Path):
    data = _within_overlap(out)
    samples = _ordered_samples()
    n, index = len(samples), {r["key"]: i for i, r in enumerate(samples)}
    positives = [_number(r["obs_over_exp"]) for r in data.iter_rows(named=True) if _number(r["obs_over_exp"]) > 0]
    floor = float(np.log10(min(positives)) - .5) if positives else -1.0
    z, zero = np.full((n, n), np.nan), np.zeros((n, n), dtype=bool)
    for r in data.iter_rows(named=True):
        i, j, v = index[r["key_a"]], index[r["key_b"]], _number(r["obs_over_exp"])
        if np.isfinite(v) and v >= 0:
            z[i, j] = z[j, i] = np.log10(v) if v > 0 else floor
            zero[i, j] = zero[j, i] = v == 0
        else:
            _issue("UMI observed/expected ratio unavailable for one or more pairs")
    vmax = max(float(np.nanmax(z)) if np.isfinite(z).any() else floor + 1, floor + .5)
    fig, ax = plt.subplots(figsize=S.figsize("3col", .94))
    image = ax.imshow(z, vmin=floor, vmax=vmax, interpolation="nearest",
                      cmap=matplotlib.colormaps[S.CMAP_SEQUENTIAL].with_extremes(bad="#eeeeee"))
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor="#eeeeee", edgecolor="white", hatch="////", lw=.3))
            elif n <= 20:
                value = "NA" if not np.isfinite(z[i, j]) else "0" if zero[i, j] else f"{z[i, j]:.1f}"
                dark = np.isfinite(z[i, j]) and (z[i, j] - floor) / (vmax - floor) < .55
                ax.text(j, i, value, fontsize=5, ha="center", va="center", color="white" if dark else BLACK)
    for blk in BLOCKS:
        for i, j in ((index[blk.key_a], index[blk.key_b]), (index[blk.key_b], index[blk.key_a])):
            ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor="none", edgecolor=BLACK, lw=.9))
    labels = [f"{r['group']} R{r['round']}" for r in samples]
    ax.set_xticks(range(n), labels, rotation=90, fontsize=5.5)
    ax.set_yticks(range(n), labels, fontsize=5.5)
    for ticks in (ax.get_xticklabels(), ax.get_yticklabels()):
        for tick, sample in zip(ticks, samples):
            tick.set_color(G.color(sample["group"]))
    for i in range(1, n):
        if samples[i]["group"] != samples[i - 1]["group"]:
            ax.axhline(i - .5, color=BLACK, lw=.55)
            ax.axvline(i - .5, color=BLACK, lw=.55)
    ax.tick_params(length=0)
    fig.colorbar(image, ax=ax, fraction=.035, pad=.025, label="log$_{10}$ observed / expected shared UMIs")
    ax.legend(handles=[Patch(facecolor="none", edgecolor=BLACK, label="same-round group comparison"),
                       Patch(facecolor="#eeeeee", edgecolor="white", hatch="////", label="self-comparison"),
                       Patch(facecolor="#eeeeee", label="NA: unavailable")],
              loc="upper left", bbox_to_anchor=(0, -.22), ncol=2, fontsize=5.5, frameon=False)
    fig.subplots_adjust(left=.15, bottom=.23, right=.88, top=.98)
    return fig, "F09_umi_overlap_heatmap"


def _within_expected(out: Path):
    data = _within_overlap(out)
    fig, ax = plt.subplots(figsize=S.figsize("3col", .62))
    positives = []
    for kind, (_, marker, _c, _f) in WITHIN_KIND.items():
        rows = data.filter(pl.col("kind") == kind)
        x = np.array([_number(v) for v in rows["expected_shared"]])
        y = np.array([_number(v) for v in rows["shared"]])
        good = (x > 0) & np.isfinite(y) & (y >= 0)
        if not good.all():
            _issue("observed/expected scatter: unavailable or nonpositive expectation cannot sit on a log axis")
        yy = np.where(y == 0, .3, y)
        for i in np.flatnonzero(good):
            ax.plot([x[i]], [yy[i]], **_overlap_mark(rows.row(int(i), named=True), marker))
        positives.extend(x[good])
        positives.extend(yy[good])
    lo, hi = (min(positives) * .6, max(positives) * 2) if positives else (.1, 10)
    ax.plot([lo, hi], [lo, hi], color=GREY, lw=.6, ls=(0, (3, 2)))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(.18, hi)
    _zero_floor_ticks(ax, .3, [1, 10, 100, 1000, 10000, 100000])
    ax.set_xlabel("Expected shared UMIs, $n_a n_b/c_{eff}$ (approximate yardstick)")
    ax.set_ylabel("Observed shared UMIs")
    ax.legend(handles=_kind_handles() + [Line2D([], [], color=GREY, lw=.6, ls=(0, (3, 2)), label="observed = expected")],
              loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=5.5, frameon=False)
    fig.subplots_adjust(right=.65, left=.10, bottom=.16, top=.96)
    return fig, "F10_umi_observed_vs_expected"


def _within_overlap_summary(out: Path, clonotype: bool = False):
    data = _within_overlap(out, clonotype)
    unit, noun = ("clonotype", "clonotypes") if clonotype else ("umis", "UMIs")
    fig, axes = plt.subplots(1, 2, figsize=S.figsize("3col", .65))
    depth = np.sqrt(np.array([_number(v) for v in data[f"n_{unit}_a"]]) * np.array([_number(v) for v in data[f"n_{unit}_b"]]))
    shared = np.array([_number(v) for v in data["shared"]])
    jaccard = np.array([_number(v) for v in data["jaccard"]])
    kinds = np.asarray(data["kind"])
    zero_shared = .3
    pos_j = jaccard[jaccard > 0]
    zero_j = min(float(pos_j.min()) * .4, 1e-3) if pos_j.size else 1e-7
    yy = [np.where(shared == 0, zero_shared, shared), np.where(jaccard == 0, zero_j, jaccard)]
    extremes = []
    for kind, (label, marker, _c, _f) in WITHIN_KIND.items():
        for ax, values in zip(axes, yy):
            for i in np.flatnonzero((kinds == kind) & (depth > 0) & np.isfinite(values) & (values > 0)):
                ax.plot([depth[i]], [values[i]], **_overlap_mark(data.row(int(i), named=True), marker, ms=3.5))
        available = (kinds == kind) & (depth > 0) & np.isfinite(jaccard)
        if available.any():
            i = int(np.nanargmax(np.where(available, jaccard, np.nan)))
            row = data.row(i, named=True)
            axes[1].plot([depth[i]], [yy[1][i]], marker="o", ms=7, mec=BLACK, mfc="none", ls="none", mew=.7)
            a, b = SAMPLE_META[row["key_a"]], SAMPLE_META[row["key_b"]]
            extremes.append(f"{label}: {a['group']} R{a['round']} / {b['group']} R{b['round']}; J={jaccard[i]:.3g}")
    if not ((depth > 0) & np.isfinite(shared) & np.isfinite(jaccard)).all():
        _issue(f"{noun} overlap: unavailable values/zero-depth pairs cannot be placed on logarithmic axes")
    good_depth = depth[(depth > 0) & np.isfinite(depth)]
    xlim = (float(good_depth.min()) * .8, float(good_depth.max()) * 1.25) if good_depth.size else (1, 10)
    finite_shared = shared[np.isfinite(shared)]
    ymax = max(3, float(finite_shared.max()) * 2.5) if finite_shared.size else 3
    # slope-2 reference lines n_a*n_b/c_eff = x^2/c_eff, decades spanning measured and implied c_eff
    exponents = []
    ce = np.array([_number(v) for v in data["c_eff_pooled_geom"]])
    if (ce > 0).any():
        exponents.extend(np.log10([ce[ce > 0].min(), ce[ce > 0].max()]))
    implied = np.divide(depth ** 2, shared, out=np.full_like(depth, np.nan), where=shared > 0)
    implied = implied[(implied > 0) & np.isfinite(implied)]
    if implied.size:
        exponents.extend(np.log10(np.percentile(implied, [2, 98])))
    if exponents:
        low, high = int(np.floor(min(exponents))), int(np.ceil(max(exponents)))
        for exponent in range(low, high + 1, max(1, (high - low) // 6 + 1)):
            xx = np.asarray(xlim)
            axes[0].plot(xx, xx ** 2 / 10.0 ** exponent, color=GREY, lw=.5, ls=(0, (4, 2)), zorder=0)
            left = max(xlim[0], np.sqrt(.18 * 10.0 ** exponent))
            right = min(xlim[1], np.sqrt(ymax * 10.0 ** exponent))
            if left < right:
                xlab = np.exp(.85 * np.log(left) + .15 * np.log(right))
                axes[0].annotate(f"$c_{{eff}}=10^{{{exponent}}}$", (xlab, xlab ** 2 / 10.0 ** exponent),
                                 xytext=(2, 2), textcoords="offset points", fontsize=5, color=BLACK)
    for i, ax in enumerate(axes):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*xlim)
        ax.set_xlabel(rf"Pair depth, $\sqrt{{n_a n_b}}$ ({noun})")
        panel_label(ax, "AB"[i])
    axes[0].set_ylim(.18, ymax)
    _zero_floor_ticks(axes[0], zero_shared, [1, 10, 100, 1000, 10000])
    axes[0].set_ylabel(f"Shared {noun}")
    axes[1].set_ylabel("Jaccard index")
    axes[1].axhline(zero_j, color=GREY, lw=.4, ls=(0, (1, 2)))
    axes[1].set_ylim(bottom=zero_j * .6)
    axes[1].yaxis.set_major_locator(FixedLocator([zero_j] + [10.0 ** e for e in range(-7, 1) if 10.0 ** e > zero_j * 1.1]))
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if v == zero_j else f"{v:g}"))
    fig.legend(handles=_kind_handles(), loc="lower center", bbox_to_anchor=(.5, .19), ncol=3, fontsize=5.5)
    for i, text in enumerate(extremes):
        fig.text(.10, .175 - i * .030, text, fontsize=5.5, color=BLACK, va="top")
    fig.subplots_adjust(bottom=.43, top=.95, wspace=.35, left=.10, right=.97)
    return fig, "F19_clonotype_overlap_summary" if clonotype else "F18_umi_overlap_summary"


#: The number of the first figure here: static_analysis.py's series runs 01-12 and this continues it.
FIRST = 13
#: (name, title, builder) in the order they are read: UMI-level agreement, then fragvec-level. A
#: figure's number and file stem come from its position here -- see stem().
FIGURES = [
    ("umi_overlap_heatmap", "Shared-UMI observed/expected matrix", lambda out: _within_heatmap(out)),
    ("umi_observed_vs_expected", "Observed versus expected UMI sharing", lambda out: _within_expected(out)),
    ("umi_shared_vs_threshold", "UMI threshold response", lambda out: _within_pair_curves(out, "threshold")),
    ("umi_overlap_summary", "UMI overlap distribution", lambda out: _within_overlap_summary(out)),
    ("fragvec_agreement_ccc", "Agreement at library-feature and fragvec scales", lambda out: _within_ccc(out)),
    ("fragvec_agreement_by_fragment_order", "Agreement by payload order",
     lambda out: _within_pair_curves(out, "order")),
    ("fragvec_detection_vs_abundance", "Directional fragvec detection at matched depth",
     lambda out: _within_pair_curves(out, "detection")),
    ("fragvec_overlap_summary", "Fragvec overlap distribution",
     lambda out: _within_overlap_summary(out, clonotype=True)),
    ("fragvec_ordination_pca", "All-sample CLR PCA", lambda out: _within_pca(out)),
    ("fragvec_complexity_dependence", "Descriptive complexity dependence", lambda out: _within_complexity(out)),
]


def stem(name: str) -> str:
    """'umi_overlap_heatmap' -> '13_umi_overlap_heatmap': the number is the figure's place in the series."""
    return f"{FIRST + [n for n, _, _ in FIGURES].index(name):02d}_{name}"


# ------------------------------------------------------------------------------------- legends

def write_legends(out: Path, figdir: Path, records: list[dict]) -> None:
    """figure_legends.md: the prose the images carry no room for, with values read from the tables."""
    n, nb = len(SAMPLE_META), len(BLOCKS)
    pairs = n * (n - 1) // 2
    families = list(dict.fromkeys(b.pair for b in BLOCKS))
    facet = ("One panel per group-pair family. Titles give A → B; numeric markers and dash identify selection "
             "round. Lines use group A colour; numeric markers use group B colour. Groups are ordered as they "
             "first appear in the config's `samples:`, which fixes A and B. Initial R0 has no same-round partner "
             "and is absent only from these paired views.")
    overlap = ("One point per unordered pair. Split marker halves show the two endpoint group colours. Shape is "
               "the pair kind: circles same round/different groups, squares same group/different rounds, "
               "triangles initial versus selected, diamonds different groups and rounds.")
    bodies = {
        "umi_overlap_heatmap": f"All {n} samples and {pairs} pairs at ≥1 read. Colour is log10(observed shared corrected UMIs / n_a n_b / c_eff), a uniform-pool yardstick, not a calibrated null. 0 marks zero sharing, drawn half a decade below the smallest positive ratio; grey NA is unavailable; the hatched diagonal is self-comparison. Black outlines mark the {nb} same-round group comparisons.",
        "umi_observed_vs_expected": f"All {pairs} pairs at ≥1 read on log–log axes; the dashed diagonal is equality. Zero sharing sits on a floor labelled 0. Read the vertical offset as an order of magnitude: c_eff is unstable when duplicates are scarce.",
        "umi_shared_vs_threshold": "Shared corrected UMIs for each same-round pair as the minimum read count required in BOTH samples rises through 1, 2, 3, 5 and 10. Zero sharing sits on a floor labelled 0; unavailable values are gaps.",
        "umi_overlap_summary": f"All {pairs} pairs at ≥1 read per UMI. A: shared UMIs; B: Jaccard. x is √(n_a n_b). Dashed lines in A are x²/c_eff per decade of effective richness, a yardstick, not a fit. The most-overlapped pair of each kind is ringed and named.",
        "fragvec_agreement_ccc": "Lin's CCC on log2 CPM + 1 for the library-feature marginals (black filled squares) and fragment vectors (grey open circles), both present-in-either. Bars are the 95% percentile bootstrap endpoints (1,000 resamples over features), not biological replication.",
        "fragvec_agreement_by_fragment_order": "Fragment-vector CCC within each payload-order stratum (vectors with ≥10 clones in either sample; strata need ≥10 vectors). A trend alone does not isolate a read-length effect.",
        "fragvec_detection_vs_abundance": "Fraction of A's fragvecs re-detected in B after multivariate-hypergeometric thinning of both to the shallower clone depth, by A's abundance bin (lower edge). Directional A → B.",
        "fragvec_overlap_summary": f"Figure {stem('umi_overlap_summary')[:2]}'s construction for all {pairs} fragvec pairs at ≥1 clone. The yardsticks use fragvec c_eff, not UMI c_eff. Matching vectors do not establish molecular identity.",
        "fragvec_ordination_pca": f"CLR PCA over all {n} samples (vectors with ≥10 clones in some sample); Euclidean distance on CLR is Aitchison distance. Labels give round.",
        "fragvec_complexity_dependence": "Mean |log2 CPM(B) − log2 CPM(A)| over vectors with ≥10 clones in either sample, against A's clone depth, with the one pooled descriptive OLS recorded in complexity_fit.json; a ring marks a point outside its fit mask. No independence-based p-value.",
    }
    faceted = {"umi_shared_vs_threshold", "fragvec_agreement_ccc", "fragvec_agreement_by_fragment_order",
               "fragvec_detection_vs_abundance", "fragvec_complexity_dependence"}
    pairwise = {"umi_observed_vs_expected", "umi_overlap_summary", "fragvec_overlap_summary"}
    lines = [f"# Figure legends — cross-group agreement ({FIRST}–{FIRST + len(FIGURES) - 1})", "",
             f"Figures 01–{FIRST - 1:02d} (raw sequencing → read funnel → decode → fragments → UMI → fragvec) are in "
             "`../analysis/`; these continue the series with how samples and groups agree.", "",
             f"**{n} samples**, **{nb} same-round, different-group comparisons** in **{len(families)} group-pair "
             f"families**, **{pairs} all-sample overlap pairs**. One config is one run: there is no batch factor, so "
             "the paired views and the complexity fit are descriptive, not tests.", "",
             "A black × near an axis baseline is an UNAVAILABLE measurement, not zero; reasons are listed per "
             "figure below and in figures_manifest.json.", "",
             "## Group colours", "", *[f"- {g}: `{G.color(g)}`" for g in _groups()], ""]
    titles = {name: title for name, title, _ in FIGURES}
    for record in records:
        name = record["name"]
        lines += [f"## Figure {record['stem'][:2]} — `{record['stem']}`", "", f"**{titles[name]}.** {bodies[name]}", ""]
        if name in faceted:
            lines += [facet, ""]
        if name in pairwise:
            lines += [overlap, ""]
        if name == "fragvec_ordination_pca":
            ev = [_number(v) for v in (_ordination_info(out).get("explained_variance") or [None, None])[:2]]
            lines += [f"Explained variance: PC1 {100 * ev[0]:.1f}%; PC2 {100 * ev[1]:.1f}%." if all(np.isfinite(ev))
                      else "Explained variance unavailable.", ""]
        if name == "fragvec_complexity_dependence":
            fit = json.loads((out / "complexity_fit.json").read_text())
            lines += [f"Recorded fit: n={fit['n']}; slope={fit.get('slope')}; intercept={fit.get('intercept')}; "
                      f"r={fit.get('r')}; status={fit.get('status')}.", ""]
        lines += [f"Status: `{record['status']}`.", *[f"- {w}" for w in record.get("warnings", [])]]
        if record.get("reason"):
            lines.append(f"Reason: {record['reason']}")
        lines.append("")
    (figdir / "figure_legends.md").write_text("\n".join(lines))


def _placeholder(fid: str, stem: str, status: str, reason: str):
    fig, ax = plt.subplots(figsize=S.figsize("3col", .35))
    ax.set_axis_off()
    ax.text(.03, .82, f"{fid} · {status}", transform=ax.transAxes, fontsize=7, fontweight="bold")
    ax.text(.03, .65, textwrap.fill(reason, 110), transform=ax.transAxes, fontsize=6, va="top", linespacing=1.5)
    return fig, stem


def main(argv=None) -> int:
    global G, SAMPLE_META, BLOCKS, PARAMS
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tables", type=Path, required=True, help="the static_tables.py output directory")
    ap.add_argument("--outdir", type=Path, required=True)
    a = ap.parse_args(argv)
    manifest = json.loads((a.tables / "run_manifest.json").read_text())
    G = S.Groups(manifest["samples"])
    SAMPLE_META = {s["name"]: {"key": s["name"], "group": s["group"], "round": s["round"]} for s in manifest["samples"]}
    BLOCKS = sorted((Block(c["comparison_id"], c["comparison_family"], c["key_a"], c["key_b"], c["group_a"],
                           c["group_b"], c["round"]) for c in manifest["comparisons"]),
                    key=lambda b: (G.rank(b.group), G.rank(b.group_b), b.round))
    PARAMS = manifest["parameters"]
    a.outdir.mkdir(parents=True, exist_ok=True)
    plt.style.use(str(S.STYLE))
    records, failed = [], []
    for name, _title, build in FIGURES:
        fstem = stem(name)
        fid = fstem[:2]
        FIGURE_ISSUES.clear()
        record = {"family": fid, "name": name, "stem": fstem}
        try:
            fig, _ = build(a.tables)       # the registry, not the builder, names the file
            record["status"] = "computed"
        except Exception as error:   # one family failing must not cost the others
            plt.close("all")
            record["status"] = "unavailable" if isinstance(error, (FileNotFoundError, FigureUnavailable)) else "failed"
            record["reason"] = f"{type(error).__name__}: {error}"
            failed += [fid] if record["status"] == "failed" else []
            print(f"  {fid}: {record['status']} — {record['reason']}", flush=True)
            fig, _ = _placeholder(fid, fstem, record["status"], record["reason"])
        record["warnings"] = list(FIGURE_ISSUES)
        fonts_ok = S.check_min_font(fig)
        fig.savefig(a.outdir / f"{fstem}.pdf")
        plt.close(fig)
        record["artifacts"] = [f"{fstem}.pdf"]
        print(f"  {fstem:<44s} {record['status']}{'' if fonts_ok else '  FONT < 5 pt'}", flush=True)
        records.append(record)
    write_legends(a.tables, a.outdir, records)
    (a.outdir / "figures_manifest.json").write_text(json.dumps(
        {"tables": str(a.tables.resolve()), "figures": records}, indent=2) + "\n")
    if failed:
        raise SystemExit(f"[FATAL] figure families failed: {', '.join(failed)}; see figures_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
