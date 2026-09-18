"""Within-batch W01–W10/V01 renderers reused from the supplied reference.

Figures read assessment TSVs; sample labels and available comparisons follow
those tables. Captions describe measurements without asserting study outcomes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.lines import Line2D                              # noqa: E402
from matplotlib.patches import Patch, Rectangle                  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter        # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import palettes as P                                             # noqa: E402
from venn_geometry import draw_venn, row_scale          # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
STYLE = HERE / "science.mplstyle"

GREY = P.GREY
BLACK = P.OKABE_ITO["black"]

#: Pair class -> (colour, marker, printed label). The two classes involving `empty` take the warm
#: hues because `empty` is the no-template control: it is the thing every other series is being
#: read against, and it should never look like just another category.
CLASS_STYLE = {
    "bio x bio":     (P.OKABE_ITO["blue"],           "o", "selected x selected"),
    "bio x empty":   (P.OKABE_ITO["vermillion"],     "s", "selected x empty (control)"),
    "empty x empty": (P.OKABE_ITO["orange"],         "D", "empty x empty (control)"),
    "init x bio":    (P.OKABE_ITO["bluish_green"],   "^", "initial x selected"),
    "init x empty":  (P.OKABE_ITO["reddish_purple"], "v", "initial x empty"),
}
CLASS_ORDER = ("bio x bio", "init x bio", "empty x empty", "bio x empty", "init x empty")

#: Payload-order strata as drawn. `pooled` is separated by a rule on every panel that shows it,
#: because it is not a fourth stratum -- it is the three of them added up, and the whole finding is
#: that the sum hides them.
STRATA_DRAW = ("<=3", "4", ">=5")
STRAT_LABEL = {"<=3": "≤ 3", "4": "4", ">=5": "≥ 5", "pooled": "pooled"}

GROUP_ORDER = ("initial", "A549", "A549-SN", "MDA", "MDA-SN", "empty")


# ------------------------------------------------------------------------------------ scaffolding

def use_style() -> None:
    plt.style.use(str(STYLE))


def _tsv(out: Path, name: str) -> pl.DataFrame:
    return pl.read_csv(out / name, separator="\t", comment_prefix="#", infer_schema_length=300000,
                       schema_overrides={"n_umis_tier_in_a": pl.String})


def save(fig, stem: str, figdir: Path, formats) -> None:
    ok = P.check_min_font(fig)
    for ext in formats:
        fig.savefig(figdir / f"{stem}.{ext}")
    w_cm = fig.get_size_inches()[0] * 2.54
    print(f"  {stem:<40s} {w_cm:5.1f} cm  {'fonts ok' if ok else 'FONT < 5 pt'}"
          f"  -> {', '.join(formats)}")
    plt.close(fig)


def panel_label(ax, letter: str, dx: float = -0.13, dy: float = 1.04) -> None:
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=8, fontweight="bold",
            va="bottom", ha="right")


def zero_floor(ax, floor: float, ticks) -> None:
    """Label the plotting floor '0' on a log axis. Zero has no place on a log scale, but dropping a
    zero-sharing pair would delete the measurement, so it is drawn at `floor` and the tick says so."""
    ax.yaxis.set_major_locator(FixedLocator([floor] + list(ticks)))
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _p: "0" if np.isclose(v, floor) else (f"{v:,.0f}" if v >= 1 else f"{v:g}")))


def class_handles(classes=None, **kw) -> list:
    classes = CLASS_ORDER if classes is None else classes
    return [Line2D([], [], color=CLASS_STYLE[c][0], marker=CLASS_STYLE[c][1], ms=3.4, lw=1.0,
                   label=CLASS_STYLE[c][2], **kw) for c in classes if c in CLASS_STYLE]


def _q(v: np.ndarray) -> tuple[float, float, float]:
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (np.nan, np.nan, np.nan)
    return (float(np.percentile(v, 25)), float(np.median(v)), float(np.percentile(v, 75)))


def sample_order(out: Path) -> list[str]:
    s = _tsv(out, "sample_sizes.tsv")
    rank = {g: i for i, g in enumerate(GROUP_ORDER)}
    rows = sorted(s.iter_rows(named=True), key=lambda r: (rank.get(r["group"], 99), r["round"]))
    return [r["sample"] for r in rows]


def short_label(name: str, group: str, rnd: int) -> str:
    return "initial R0" if group == "initial" else f"{group} R{rnd}"


def best_depth(d: pl.DataFrame) -> int:
    """The deepest matching depth available. Deeper is better here: shallower depths push the
    order>=5 counts to zero for every class and the panel stops discriminating."""
    return int(d["depth_m"].max())


# =============================================================================== W01: the headline

def fig_w01(out: Path):
    """Render W01 from the supplied within-batch tables."""
    d = _tsv(out, "depth_matched_overlap.tsv").filter(pl.col("tier") == "singleton")
    m = best_depth(d)
    d = d.filter(pl.col("depth_m") == m)
    n_s = len(set(d["key_a"].to_list()) | set(d["key_b"].to_list()))

    fig, axes = plt.subplots(1, 2, figsize=P.figsize("2col", 0.50))
    xs = {"pooled": 0.0, "<=3": 1.4, "4": 2.4, ">=5": 3.4}
    floor_a, floor_b = 0.25, 3e-5

    for k, (col, ax, floor, ylab) in enumerate([
            ("shared_mean", axes[0], floor_a, f"Shared singleton fragvecs (mean of {int(d['n_reps'][0])} draws)"),
            ("overlap_coef_mean", axes[1], floor_b, "Overlap coefficient")]):
        for c in CLASS_ORDER:
            col_, mk, _lab = CLASS_STYLE[c]
            xv, lo, mid, hi = [], [], [], []
            for st in ("pooled",) + STRATA_DRAW:
                v = d.filter((pl.col("pair_class") == c) & (pl.col("order_stratum") == st))[col].to_numpy()
                q1, md, q3 = _q(np.asarray(v, dtype=float))
                xv.append(xs[st]); lo.append(max(q1, floor)); mid.append(max(md, floor)); hi.append(max(q3, floor))
            xv = np.array(xv)
            # the pooled point is deliberately NOT joined to the stratified ones: it is their sum,
            # not a fourth stratum, and a connecting line would read as a trend.
            ax.plot(xv[1:], mid[1:], color=col_, marker=mk, ms=3.4, lw=1.0, zorder=4)
            ax.plot(xv[:1], mid[:1], color=col_, marker=mk, ms=3.4, ls="none", zorder=4)
            ax.vlines(xv, lo, hi, color=col_, lw=0.8, alpha=0.55, zorder=3)
        ax.axvline(0.7, color=GREY, lw=0.6, ls=(0, (2, 2)))
        ax.set_yscale("log")
        ax.set_xticks(list(xs.values()))
        ax.set_xticklabels([STRAT_LABEL[s] for s in ("pooled",) + STRATA_DRAW], fontsize=6)
        ax.set_xlabel("fragment modules per vector (order)")
        ax.set_ylabel(ylab, fontsize=6.5)
        ax.set_xlim(-0.55, 3.95)
        panel_label(ax, "AB"[k])
    zero_floor(axes[0], floor_a, [1, 10, 100, 1000])
    axes[0].set_title("shared count", fontsize=6.5)
    axes[1].set_title("shared, normalised by the smaller set", fontsize=6.5)
    axes[0].text(0.0, 0.02, "not stratified", transform=axes[0].get_xaxis_transform(),
                 ha="center", va="bottom", fontsize=5.0, color=GREY, rotation=90)

    lk = axes[1].legend(handles=class_handles(), loc="upper left", bbox_to_anchor=(1.02, 1.0),
                        fontsize=5.2, frameon=False, title="pair class", title_fontsize=5.6,
                        handletextpad=0.4, borderaxespad=0.0)
    lk._legend_box.align = "left"
    fig.suptitle(f"median over pairs, bars = IQR · depth matched to {m:,} UMIs per sample "
                 f"({n_s} samples) · singleton tier", fontsize=6, y=0.995)
    fig.subplots_adjust(wspace=0.34, bottom=0.20, top=0.85, right=0.76, left=0.09)
    return fig, "W01_overlap_by_order_stratum"


# ============================================================================ W02: depth response

def fig_w02(out: Path):
    """Render W02 from the supplied within-batch tables."""
    d = _tsv(out, "depth_matched_overlap.tsv").filter(pl.col("tier") == "singleton")
    depths = sorted(d["depth_m"].unique().to_list())

    fig, axes = plt.subplots(1, 2, figsize=P.figsize("2col", 0.50))
    floor = 0.03
    for c in CLASS_ORDER:
        col, mk, _ = CLASS_STYLE[c]
        y = []
        for m in depths:
            v = d.filter((pl.col("depth_m") == m) & (pl.col("pair_class") == c)
                         & (pl.col("order_stratum") == ">=5"))["shared_mean"].to_numpy()
            y.append(max(float(np.median(v)) if v.size else np.nan, floor))
        axes[0].plot(depths, y, color=col, marker=mk, ms=3.4, lw=1.0)
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    zero_floor(axes[0], floor, [0.1, 1, 10])
    axes[0].set_xlabel("UMIs per sample after depth matching")
    axes[0].set_ylabel("median shared singleton fragvecs, order $\\geq$ 5", fontsize=6.5)
    axes[0].set_title("order $\\geq$ 5 sharing against depth", fontsize=6.5)
    panel_label(axes[0], "A")

    for st in STRATA_DRAW:
        r = []
        for m in depths:
            s = d.filter((pl.col("depth_m") == m) & (pl.col("order_stratum") == st))
            bb = s.filter(pl.col("pair_class") == "bio x bio")
            be = s.filter(pl.col("pair_class") == "bio x empty")
            bb_size = (bb["n_a_mean"] * bb["n_b_mean"]).sum()
            be_size = (be["n_a_mean"] * be["n_b_mean"]).sum()
            num = bb["shared_mean"].sum() / bb_size if bb_size else np.nan
            den = be["shared_mean"].sum() / be_size if be_size else np.nan
            r.append(num / den if den > 0 else np.nan)
        ls = {"<=3": (0, (1, 1.2)), "4": (0, (4, 1.5)), ">=5": "-"}[st]
        axes[1].plot(depths, r, color=BLACK if st == ">=5" else GREY, ls=ls, lw=1.1,
                     marker="o" if st == ">=5" else None, ms=3.0)
        axes[1].annotate(f"order {STRAT_LABEL[st]}", xy=(depths[-1], r[-1]), xytext=(3, 0),
                         textcoords="offset points", fontsize=5.4, va="center",
                         color=BLACK if st == ">=5" else GREY)
    axes[1].axhline(1.0, color=P.OKABE_ITO["vermillion"], lw=0.8, ls=(0, (3, 1.5)))
    axes[1].annotate("no excess over the empty control", xy=(depths[0], 1.0), xytext=(0, 3),
                     textcoords="offset points", fontsize=5.2, color=P.OKABE_ITO["vermillion"])
    axes[1].set_xscale("log")
    axes[1].set_xlabel("UMIs per sample after depth matching")
    axes[1].set_ylabel("size-adjusted sharing,\nselected$\\times$selected / selected$\\times$empty",
                       fontsize=6.5)
    axes[1].set_title("size-adjusted sharing ratio", fontsize=6.5)
    axes[1].set_xlim(depths[0] * 0.85, depths[-1] * 1.9)
    panel_label(axes[1], "B")

    # Only the depths actually run are labelled. A log axis's default decade ticks land between the
    # three measured points and crowd into each other at this width, implying a continuum that was
    # never sampled.
    for ax in axes:
        ax.xaxis.set_major_locator(FixedLocator(depths))
        ax.xaxis.set_minor_locator(FixedLocator([]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{int(v):,}"))

    lk = axes[0].legend(handles=class_handles(), loc="upper left", fontsize=5.0, frameon=False,
                        handletextpad=0.4, title="pair class", title_fontsize=5.4)
    lk._legend_box.align = "left"
    fig.subplots_adjust(wspace=0.42, bottom=0.20, top=0.90, left=0.10, right=0.90)
    return fig, "W02_depth_response"


# ====================================================================== W03: the cost of the filter

def fig_w03(out: Path):
    """Render W03 from the supplied within-batch tables."""
    t = _tsv(out, "abundance_tiers.tsv")
    tiers = ["1", "2", "3-4", "5-9", "10+"]
    x = np.arange(len(tiers))

    fig, axes = plt.subplots(1, 2, figsize=P.figsize("2col", 0.48))
    for k, (col, ax, ylab) in enumerate([
            ("p_present_in_b", axes[0], "P( present in B )"),
            ("p_singleton_in_b_given_present", axes[1], "P( singleton in B | present in B )")]):
        for c in CLASS_ORDER:
            colr, mk, _ = CLASS_STYLE[c]
            mid, lo, hi = [], [], []
            for tl in tiers:
                v = t.filter((pl.col("pair_class") == c) & (pl.col("n_umis_tier_in_a") == tl))[col].to_numpy()
                q1, md, q3 = _q(np.asarray(v, dtype=float))
                mid.append(md); lo.append(q1); hi.append(q3)
            ax.plot(x, mid, color=colr, marker=mk, ms=3.2, lw=1.0, zorder=4)
            ax.fill_between(x, lo, hi, color=colr, alpha=0.13, lw=0, zorder=2)
        ax.set_xticks(x); ax.set_xticklabels(tiers, fontsize=6)
        ax.set_ylabel(ylab, fontsize=6.5)
        ax.set_ylim(-0.03, 1.06)
        panel_label(ax, "AB"[k])
    axes[0].axvline(0, color=GREY, lw=0.6, ls=(0, (2, 2)))
    axes[1].axvline(0, color=GREY, lw=0.6, ls=(0, (2, 2)))
    axes[0].set_title("re-detection by abundance", fontsize=6.5)
    axes[1].set_title("singleton status by abundance", fontsize=6.5)
    # The marker for the analysed tier sits at the TOP of the axis: at the bottom-left it lands on
    # the very series it is pointing at, in the one panel where those series start near zero.
    for ax in axes:
        ax.annotate("the analysed tier", xy=(0, 1.045), xytext=(3, 0), textcoords="offset points",
                    fontsize=5.2, color=GREY, va="center", ha="left")
    lk = axes[1].legend(handles=class_handles(), loc="upper left", bbox_to_anchor=(1.02, 1.0),
                        fontsize=5.2, frameon=False, title="pair class", title_fontsize=5.6,
                        handletextpad=0.4, borderaxespad=0.0)
    lk._legend_box.align = "left"
    fig.supxlabel("UMIs carrying the vector in sample A  ($n_{umis}$)", fontsize=7, y=0.045)
    fig.suptitle("median over ordered pairs, band = IQR", fontsize=6, y=0.985)
    fig.subplots_adjust(wspace=0.30, bottom=0.20, top=0.84, right=0.75, left=0.09)
    return fig, "W03_singleton_filter_selection"


# ==================================================================== W04: separation, and tiers

def fig_w04(out: Path):
    """Render W04 from the supplied within-batch tables."""
    f = _tsv(out, "fragvec_overlap.tsv")
    d = _tsv(out, "depth_matched_overlap.tsv").filter(pl.col("tier") == "singleton")
    m = best_depth(d)
    d = d.filter(pl.col("depth_m") == m)
    fig, axes = plt.subplots(1, 2, figsize=P.figsize("2col", 0.50))

    sgl = f.filter(pl.col("tier") == "singleton")
    ys = np.arange(len(CLASS_ORDER))[::-1]
    values = d["overlap_coef_mean"].to_numpy()
    positive_coefficients = values[np.isfinite(values) & (values > 0)]
    coefficient_floor = float(positive_coefficients.min()) / 3 if positive_coefficients.size else 0.001
    for y, c in zip(ys, CLASS_ORDER):
        col, mk, _ = CLASS_STYLE[c]
        a = float(np.median(d.filter((pl.col("pair_class") == c)
                                     & (pl.col("order_stratum") == "pooled"))["overlap_coef_mean"].to_numpy()))
        b = float(np.median(d.filter((pl.col("pair_class") == c)
                                     & (pl.col("order_stratum") == ">=5"))["overlap_coef_mean"].to_numpy()))
        a, b = (coefficient_floor if value == 0 else value for value in (a, b))
        axes[0].plot([a, b], [y, y], color=col, lw=1.0, zorder=3)
        axes[0].plot([a], [y], color=col, marker="o", ms=4.2, mfc="white", mew=1.0, zorder=4)
        axes[0].plot([b], [y], color=col, marker=mk, ms=4.0, zorder=5)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([CLASS_STYLE[c][2] for c in CLASS_ORDER], fontsize=5.8)
    axes[0].set_xscale("log")
    if np.any(values == 0):
        axes[0].xaxis.set_major_formatter(FuncFormatter(lambda v, _p: "0" if np.isclose(v, coefficient_floor) else f"{v:g}"))
        axes[0].xaxis.set_major_locator(FixedLocator([coefficient_floor, *[t for t in axes[0].get_xticks() if t > coefficient_floor]]))
    axes[0].set_xlabel("median overlap coefficient")
    axes[0].set_title(f"pooled (hollow) $\\rightarrow$ order $\\geq$ 5 (filled)\n"
                      f"depth matched to {m:,} UMIs", fontsize=6.5)
    axes[0].legend(handles=[
        Line2D([], [], ls="none", marker="o", ms=4.2, mfc="white", mec=BLACK, mew=1.0,
               label="pooled over all orders"),
        Line2D([], [], ls="none", marker="o", ms=4.0, color=BLACK, label="order $\\geq$ 5 only"),
    ], loc="lower right", fontsize=5.2, frameon=False)
    panel_label(axes[0], "A", dx=-0.62)

    piv = (sgl.filter(pl.col("order_stratum") == "pooled")
           .select("key_a", "key_b", "pair_class", pl.col("jaccard").alias("j_sgl"))
           .join(f.filter((pl.col("tier") == "all") & (pl.col("order_stratum") == "pooled"))
                 .select("key_a", "key_b", pl.col("jaccard").alias("j_all")),
                 on=["key_a", "key_b"], how="inner"))
    ja = piv["j_all"].to_numpy(); js = piv["j_sgl"].to_numpy(); pc = piv["pair_class"].to_numpy()
    positive = np.concatenate([ja[ja > 0], js[js > 0]])
    floor = float(positive.min()) / 3 if positive.size else 0.001
    for c in CLASS_ORDER:
        m = pc == c
        if not m.any():
            continue
        col, mk, _ = CLASS_STYLE[c]
        axes[1].plot(np.where(ja[m] == 0, floor, ja[m]), np.where(js[m] == 0, floor, js[m]), ls="none", marker=mk, ms=2.6, mfc="none", mec=col, mew=0.7)
    lim = [floor, max(float(positive.max()) * 1.4, floor * 10) if positive.size else 1.0]
    axes[1].plot(lim, lim, color=GREY, lw=0.6, ls=(0, (3, 1.5)))
    slope = float(np.nansum(ja * js) / np.nansum(ja * ja)) if np.nansum(ja * ja) else float("nan")
    if np.isfinite(slope):
        axes[1].plot(lim, [slope * v for v in lim], color=BLACK, lw=0.8)
    # Both reference lines are labelled INSIDE the panel, at a point chosen part-way along each
    # line rather than at its end: the identity line exits through the top-right corner, where an
    # end-anchored label is clipped by the axes rather than drawn.
    xa = 10 ** (np.log10(lim[0]) + 0.62 * (np.log10(lim[1]) - np.log10(lim[0])))
    axes[1].annotate("identity", xy=(xa, xa), xytext=(-1, 3), textcoords="offset points",
                     fontsize=5.2, color=GREY, ha="right", va="bottom", rotation=38)
    if np.isfinite(slope):
        axes[1].annotate(f"fit, slope {slope:.2f}", xy=(lim[1], slope * lim[1]), xytext=(-2, -7),
                         textcoords="offset points", fontsize=5.2, color=BLACK, ha="right")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    if np.any(ja == 0) or np.any(js == 0):
        zero_floor(axes[1], floor, [t for t in axes[1].get_yticks() if t > floor])
        axes[1].xaxis.set_major_formatter(FuncFormatter(lambda v, _p: "0" if np.isclose(v, floor) else f"{v:g}"))
        axes[1].xaxis.set_major_locator(FixedLocator([floor, *[t for t in axes[1].get_xticks() if t > floor]]))
    axes[1].set_xlim(*lim); axes[1].set_ylim(*lim)
    axes[1].set_xlabel("Jaccard, all fragvecs")
    axes[1].set_ylabel("Jaccard, singleton fragvecs", fontsize=6.5)
    axes[1].set_title("singleton versus unrestricted\nJaccard overlap (native depth)",
                      fontsize=6.5)
    panel_label(axes[1], "B")
    fig.subplots_adjust(wspace=0.40, bottom=0.20, top=0.84, left=0.26, right=0.97)
    return fig, "W04_stratum_separation_and_tier"


# ======================================================================== W05: the pairwise matrix

def fig_w05(out: Path):
    """Render W05 from the supplied within-batch tables."""
    d = _tsv(out, "depth_matched_overlap.tsv")
    m = best_depth(d)
    d = d.filter((pl.col("tier") == "singleton") & (pl.col("order_stratum") == ">=5")
                 & (pl.col("depth_m") == m))
    s = _tsv(out, "sample_sizes.tsv")
    grp = {r["sample"]: r["group"] for r in s.iter_rows(named=True)}
    rnd = {r["sample"]: r["round"] for r in s.iter_rows(named=True)}
    names = [n for n in sample_order(out)
             if n in set(d["key_a"].to_list()) | set(d["key_b"].to_list())]
    pos = {n: i for i, n in enumerate(names)}
    n = len(names)

    z = np.full((n, n), np.nan)
    for r in d.iter_rows(named=True):
        i, j = pos[r["key_a"]], pos[r["key_b"]]
        z[i, j] = z[j, i] = r["shared_mean"]

    fig, ax = plt.subplots(figsize=P.figsize("3col", 0.86))
    vmax = float(np.nanmax(z))
    im = ax.imshow(z, cmap=P.CMAP_SEQUENTIAL, vmin=0.0, vmax=vmax, interpolation="nearest")
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor="#e9edf1",
                                       edgecolor="white", lw=.4, hatch="////", zorder=2))
                continue
            ax.text(j, i, f"{z[i, j]:.0f}" if z[i, j] >= 1 else "0", ha="center", va="center",
                    fontsize=5.0, color="white" if z[i, j] < 0.55 * vmax else "black", zorder=3)
    labs = [short_label(x, grp[x], rnd[x]) for x in names]
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labs, rotation=90, fontsize=5.2)
    ax.set_yticklabels(labs, fontsize=5.2)
    for axis in (ax.get_xticklabels(), ax.get_yticklabels()):
        for lbl, x in zip(axis, names):
            if grp[x] == "empty":
                lbl.set_color(P.OKABE_ITO["vermillion"])
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", lw=.7)
    ax.tick_params(which="minor", length=0); ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    # the control block, boxed: it is the reference the rest of the map is read against
    idx = [i for i, x in enumerate(names) if grp[x] == "empty"]
    if idx:
        lo, hi = min(idx) - .5, max(idx) + .5
        ax.add_patch(Rectangle((lo, lo), hi - lo, hi - lo, facecolor="none",
                               edgecolor=P.OKABE_ITO["vermillion"], lw=1.2, zorder=6))
    cb = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.02)
    cb.set_label("shared singleton fragvecs, order $\\geq$ 5", fontsize=6.5)
    cb.ax.tick_params(labelsize=5.5, width=.5, length=2)
    cb.outline.set_linewidth(.5)
    ax.legend(handles=[
        Patch(facecolor="none", edgecolor=P.OKABE_ITO["vermillion"], lw=1.2,
              label="empty x empty (no-template control)"),
        Patch(facecolor="#e9edf1", hatch="////", edgecolor="white", label="self-comparison"),
    ], loc="upper left", bbox_to_anchor=(0.0, -0.16), ncol=1, fontsize=5.6)
    ax.set_title(f"depth matched to {m:,} UMIs per sample, singleton tier", fontsize=6.5)
    return fig, "W05_shared_matrix_order_ge5"


# ============================================================== W06: architecture against molecule

def fig_w06(out: Path):
    """Render W06 from the supplied within-batch tables."""
    f = (_tsv(out, "fragvec_overlap.tsv")
         .filter((pl.col("tier") == "singleton") & (pl.col("order_stratum") == "pooled"))
         .select("key_a", "key_b", "pair_class", pl.col("shared").alias("sh_frag")))
    u = (_tsv(out, "umi_overlap.tsv")
         .filter((pl.col("population") == "all_umis") & (pl.col("threshold_min_reads") == 1))
         .select("key_a", "key_b", pl.col("shared").alias("sh_umi")))
    d = f.join(u, on=["key_a", "key_b"], how="inner")
    xf = d["sh_frag"].to_numpy().astype(float)
    yu = d["sh_umi"].to_numpy().astype(float)
    pc = d["pair_class"].to_numpy()
    floor = 0.35
    xp = np.where(xf > 0, xf, floor)
    yp = np.where(yu > 0, yu, floor)

    fig, ax = plt.subplots(figsize=P.figsize("1col", 0.86))
    for c in CLASS_ORDER:
        m = pc == c
        if not m.any():
            continue
        col, mk, _ = CLASS_STYLE[c]
        ax.plot(xp[m], yp[m], ls="none", marker=mk, ms=2.8, mfc="none", mec=col, mew=0.7)
    ax.set_xscale("log"); ax.set_yscale("log")
    zero_floor(ax, floor, [1, 10, 100])
    if np.any(xf == 0):
        ax.xaxis.set_major_locator(FixedLocator([floor, 1, 10, 100]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: "0" if np.isclose(v, floor) else f"{v:g}"))
    ax.set_xlabel("shared singleton fragvecs (architectures)")
    ax.set_ylabel("shared UMIs (physical molecules)", fontsize=6.5)
    ax.axhline(floor, color=GREY, lw=0.4, ls=(0, (1, 2)))
    ax.set_title("shared architectures and shared UMIs", fontsize=6.5)
    lk = ax.legend(handles=class_handles(), loc="upper left", fontsize=5.0, frameon=False,
                   handletextpad=0.4, title="pair class", title_fontsize=5.2)
    lk._legend_box.align = "left"
    fig.subplots_adjust(bottom=0.16, left=0.19, top=0.90, right=0.97)
    return fig, "W06_fragvec_vs_umi_sharing"


# ============================================================================ W07: what is shared

def fig_w07(out: Path):
    """Render W07 from the supplied within-batch tables."""
    s = _tsv(out, "sharing_spectrum.tsv").filter(pl.col("tier") == "singleton")
    fig, ax = plt.subplots(figsize=P.figsize("1col", 0.82))
    styles = {"<=3": (P.OKABE_ITO["orange"], "o", "-"),
              "4": (P.OKABE_ITO["bluish_green"], "s", (0, (4, 1.5))),
              ">=5": (P.OKABE_ITO["blue"], "^", (0, (1, 1.2)))}
    floor = 0.5
    for st in STRATA_DRAW:
        sub = s.filter(pl.col("order_stratum") == st).sort("n_samples_carrying")
        x = sub["n_samples_carrying"].to_numpy()
        y = sub["n_vectors"].to_numpy().astype(float)
        col, mk, ls = styles[st]
        ax.plot(x, np.where(y > 0, y, floor), color=col, marker=mk, ms=2.8, lw=1.0, ls=ls,
                label=f"order {STRAT_LABEL[st]}")
    ax.set_yscale("log")
    zero_floor(ax, floor, [1, 100, 10000, 1000000])
    ax.set_xlabel("number of samples carrying the vector")
    ax.set_ylabel("singleton fragvecs", fontsize=6.5)
    ax.set_xticks(range(1, int(s["n_samples_carrying"].max()) + 1, max(1, int(s["n_samples_carrying"].max()) // 10)))
    ax.legend(loc="upper right", fontsize=5.4, frameon=False, title="fragment order",
              title_fontsize=5.6)
    ax.set_title("prevalence spectrum, singleton tier", fontsize=6.5)
    fig.subplots_adjust(bottom=0.17, left=0.20, top=0.90, right=0.97)
    return fig, "W07_prevalence_spectrum"


# ================================================================================ W08: the depths

def fig_w08(out: Path):
    """Render W08 from the supplied within-batch tables."""
    s = _tsv(out, "sample_sizes.tsv")
    names = sample_order(out)
    by = {r["sample"]: r for r in s.iter_rows(named=True)}
    x = np.arange(len(names))
    cols = [P.OKABE_ITO["vermillion"] if by[n]["group"] == "empty" else
            (GREY if by[n]["group"] == "initial" else P.OKABE_ITO["blue"]) for n in names]

    fig, axes = plt.subplots(2, 1, figsize=P.figsize("2col", 0.62), sharex=True)
    axes[0].bar(x, [by[n]["sum_n_umis"] for n in names], color=cols, width=0.72,
                label="UMIs sequenced")
    axes[0].plot(x, [by[n]["n_singleton"] for n in names], color=BLACK, marker="o", ms=2.8,
                 lw=0.9, ls="none", label="singleton fragvecs")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("count (log)", fontsize=6.5)
    axes[0].legend(loc="upper right", fontsize=5.4, frameon=False, ncol=2)
    axes[0].set_title("UMI depth and singleton richness by sample",
                      fontsize=6.5)
    panel_label(axes[0], "A", dx=-0.06)

    axes[1].bar(x, [by[n]["pct_singleton"] for n in names], color=cols, width=0.72)
    axes[1].set_ylabel("% of fragvecs that are\nsingletons", fontsize=6.5)
    axes[1].set_ylim(0, 100)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([short_label(n, by[n]["group"], by[n]["round"]) for n in names],
                            rotation=90, fontsize=5.4)
    for lbl, n in zip(axes[1].get_xticklabels(), names):
        if by[n]["group"] == "empty":
            lbl.set_color(P.OKABE_ITO["vermillion"])
    panel_label(axes[1], "B", dx=-0.06)
    axes[1].legend(handles=[
        Patch(facecolor=P.OKABE_ITO["blue"], label="selection arm"),
        Patch(facecolor=P.OKABE_ITO["vermillion"], label="empty (control)"),
        Patch(facecolor=GREY, label="initial library"),
    ], loc="lower right", fontsize=5.2, frameon=False, ncol=3)
    fig.subplots_adjust(hspace=0.12, bottom=0.26, left=0.11, right=0.97, top=0.92)
    return fig, "W08_sample_depth_and_richness"


# =========================================================================== W09: the class contrast

def fig_w09(out: Path):
    """Render W09 from the supplied within-batch tables."""
    c = _tsv(out, "class_contrasts.tsv").filter(pl.col("contrast_bio_over_bioempty").is_finite())
    order = [s for s in ("pooled",) + STRATA_DRAW if s in c["order_stratum"].to_list()]
    y = np.arange(len(order))[::-1]
    fig, ax = plt.subplots(figsize=P.figsize("2col", 0.38))
    obs = [float(c.filter(pl.col("order_stratum") == s)["contrast_bio_over_bioempty"][0])
           for s in order]
    hi = max(obs) * 1.16
    for yy, st in zip(y, order):
        r = c.filter(pl.col("order_stratum") == st).to_dicts()[0]
        ax.plot([r["null_median"], r["null_p95"]], [yy, yy], color=GREY, lw=4.0, alpha=0.45,
                solid_capstyle="butt", zorder=2)
        v = r["contrast_bio_over_bioempty"]
        col = BLACK if st != "pooled" else GREY
        ax.plot([v], [yy], marker="o", ms=4.6, color=col, zorder=4)
        # the label goes on whichever side of the point has room, so the rightmost stratum's
        # p-value does not run off the axis and the leftmost does not sit on the reference line
        right = v < 0.75 * hi
        ax.annotate(f"p = {r['p_one_sided']:.3f}", xy=(v, yy),
                    xytext=(7 if right else -7, 0), textcoords="offset points", fontsize=5.6,
                    color=col, ha="left" if right else "right", va="center")
    ax.axvline(1.0, color=P.OKABE_ITO["vermillion"], lw=0.9, ls=(0, (3, 1.5)), zorder=1)
    ax.annotate("no excess over the empty control", xy=(1.0, -0.52), xytext=(4, 0),
                textcoords="offset points", fontsize=5.4, color=P.OKABE_ITO["vermillion"],
                va="center")
    ax.set_yticks(y)
    ax.set_yticklabels([("pooled\n(not stratified)" if s == "pooled" else f"order {STRAT_LABEL[s]}")
                        for s in order], fontsize=6)
    ax.set_xlabel("selected$\\times$selected sharing / selected$\\times$empty sharing, "
                  "at equal pair size")
    ax.set_xlim(min(0.92, min(obs) * 0.8), max(1.2, hi))
    ax.set_ylim(-0.85, y.max() + 0.55)
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="none", ms=4.6, color=BLACK, label="observed"),
        Line2D([], [], lw=4.0, color=GREY, alpha=0.45,
               label="label-permutation null, median to 95th percentile"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, fontsize=5.4, frameon=False)
    fig.subplots_adjust(left=0.17, bottom=0.34, top=0.96, right=0.98)
    return fig, "W09_class_contrast"



# ===================================================================== W10: final retained reads

def fig_w10(out: Path, accounting: Path | None = None):
    """Render W10 from the supplied within-batch tables."""
    accounting = accounting or (out / "sample_read_accounting.tsv")
    counts = _tsv(accounting.parent, accounting.name)
    if counts["sample"].n_unique() != counts.height:
        raise ValueError(f"Duplicate sample names in {accounting}; select one cohort")
    # Select every sample in this accounting table; no overlap-table membership
    # or metadata enters W10. GROUP_ORDER only controls the visual arrangement.
    metadata = {r["sample"]: r for r in counts.iter_rows(named=True)}
    rank = {group: i for i, group in enumerate(GROUP_ORDER)}
    names = sorted(metadata, key=lambda name: (rank.get(metadata[name]["group"], 99),
                                               metadata[name]["round"], name))
    values = []
    for name in names:
        value = metadata[name]["umi_reads_kept"]
        value = float(value) if value is not None else np.nan
        if np.isinf(value) or (np.isfinite(value) and (value < 0 or not value.is_integer())):
            raise ValueError(f"Invalid final read count for {name}: {value}")
        values.append(value)
    values = np.asarray(values)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError(f"No final retained read counts available in {accounting}")

    x = np.arange(len(names))
    colors = [P.OKABE_ITO["vermillion"] if metadata[n]["group"] == "empty" else
              (GREY if metadata[n]["group"] == "initial" else P.OKABE_ITO["blue"])
              for n in names]
    fig, ax = plt.subplots(figsize=P.figsize("3col", 0.48))
    bars = ax.bar(x[finite], values[finite], color=np.asarray(colors)[finite], width=0.72)
    ax.bar_label(bars, labels=[f"{v:,.0f}" for v in values[finite]], padding=3, fontsize=5.4)
    for i in x[~finite]:
        ax.text(i, 0.015, "NA", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", color=GREY, fontsize=5.4)
    for i in range(1, len(names)):
        if metadata[names[i]]["group"] != metadata[names[i - 1]]["group"]:
            ax.axvline(i - 0.5, color=GREY, lw=0.4, ls=(0, (2, 2)), alpha=0.45)
    ax.set_xticks(x, [short_label(n, metadata[n]["group"], metadata[n]["round"]) for n in names],
                  rotation=90, fontsize=5.5)
    for label, name in zip(ax.get_xticklabels(), names):
        if metadata[name]["group"] == "empty":
            label.set_color(P.OKABE_ITO["vermillion"])
    ax.set_xlim(-0.65, len(names) - 0.35)
    ax.set_ylim(0, max(float(values[finite].max()), 1.0) * 1.15)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:,.0f}"))
    ax.set_ylabel("Retained reads (UMI-kept)")
    ax.set_title("Reads retained in the clean UMI layer", fontsize=6.5)
    ax.legend(handles=[
        Patch(facecolor=P.OKABE_ITO["blue"], label="selection arm"),
        Patch(facecolor=P.OKABE_ITO["vermillion"], label="empty (control)"),
        Patch(facecolor=GREY, label="initial library"),
    ], loc="upper right", fontsize=5.5, frameon=False)
    fig.subplots_adjust(bottom=0.25, left=0.12, right=0.99, top=0.91)
    return fig, "W10_final_retained_reads"



# ==================================================================================== V01: the Venns

def fig_v01(out: Path):
    """Render V01 from the supplied within-batch tables."""
    v = _tsv(out, "venn_counts.tsv")
    #: (table key, printed row heading). The heading names the row once, on the left, instead of
    #: repeating the space in all three panel titles -- at three columns those titles collide.
    space_row = [("UMI", "UMI space\n(physical molecules)"),
                 ("singleton fragvec, all orders", "singleton fragvec\nall orders"),
                 ("singleton fragvec, order>=5", "singleton fragvec\norder $\\geq$ 5")]
    spaces = [(k, lab) for k, lab in space_row if k in v["space"].to_list()]
    metadata = {r["sample"]: r for r in _tsv(out, "sample_sizes.tsv").iter_rows(named=True)}
    arms = [name for name in sample_order(out) if name in v["arm"].to_list()]
    hatches = ("///", "\\", "...", "xx", "++")
    colors = P.workflow_group_colors(GROUP_ORDER)
    arm_style = {name: (colors[metadata[name]["group"]], hatches[i % len(hatches)],
                        short_label(name, metadata[name]["group"], metadata[name]["round"]))
                 for i, name in enumerate(arms)}

    panels = []
    for key, _lab in spaces:
        row = []
        for arm in arms:
            r = v.filter((pl.col("space") == key) & (pl.col("arm") == arm)).to_dicts()[0]
            col, hat, lab = arm_style[arm]
            row.append({"n_left": int(r["n_initial"]), "n_right": int(r["n_arm"]),
                        "shared": int(r["shared"]), "arm": lab, "color": col, "hatch": hat,
                        "space": key})
        panels.append(row)

    nr, nc = len(panels), len(arms)
    w = P.WIDTH_3COL_IN
    fig, axes = plt.subplots(nr, nc, figsize=(w, w * 0.30 * nr), squeeze=False)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.93, bottom=0.05, hspace=0.80, wspace=0.10)
    for i, row in enumerate(panels):
        s = row_scale(row, width=1.0)
        unit = "UMI" if row[0]["space"] == "UMI" else "fragvec"
        for j, rec in enumerate(row):
            ax = axes[i][j]
            draw_venn(ax, rec["n_left"], rec["n_right"], rec["shared"],
                      label_left="initial library", label_right=rec["arm"],
                      color_right=rec["color"], hatch_right=rec["hatch"], scale=s, unit=unit)
            # `draw_venn` pluralises its unit unconditionally; at these counts a panel can legitimately
            # share exactly one, and "1 fragvecs" in a figure caption is simply wrong.
            for t in ax.texts:
                if t.get_text() == f"shared 1 {unit}s":
                    t.set_text(f"shared 1 {unit}")
            ax.set_title(f"{chr(65 + i * nc + j)}   {rec['arm']}"
                         + ("  (control)" if rec["arm"].startswith("empty") else ""),
                         fontsize=6.2, loc="left")
        # Equal data limits preserve the shared area scale after Matplotlib's
        # per-panel autoscaling, including when one arm has many more members.
        visible = [ax for ax, rec in zip(axes[i], row) if rec["n_left"] and rec["n_right"]]
        if visible:
            xlim = (min(ax.get_xlim()[0] for ax in visible), max(ax.get_xlim()[1] for ax in visible))
            ylim = (min(ax.get_ylim()[0] for ax in visible), max(ax.get_ylim()[1] for ax in visible))
            for ax in visible:
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
        axes[i][0].text(-0.16, 0.45, spaces[i][1], transform=axes[i][0].transAxes,
                        rotation=90, ha="center", va="center", fontsize=6.2)
    fig.suptitle("initial library against final sampled rounds — one area scale per row",
                 fontsize=6.5, y=0.998)
    return fig, "V01_venn_initial_vs_final_round"


# =========================================================================================== legends

BUILDERS = {
    "W01": fig_w01, "W02": fig_w02, "W03": fig_w03, "W04": fig_w04, "W05": fig_w05,
    "W06": fig_w06, "W07": fig_w07, "W08": fig_w08, "W09": fig_w09,
    "W10": fig_w10, "V01": fig_v01,
}
