#!/usr/bin/env python
"""The static analysis series: one figure per step of the workflow, in the order the analysis runs.

Each figure states ONE message (its title) and every panel is a facet of that message:

    01  raw sequencing      how deep and how good each sample is, before anything is filtered
    02  read funnel         how many reads every step keeps, and why it removes the rest
    03  decode QC           how well the decoder explains the reads, and what the coverage cut drops
    04  raw fragment calls  what the decoder called and how accurately, before any UMI processing
    05  UMI QC              what UMI filtering, ed<=1 correction and the jackpot blacklist remove
    06  fragment quant      the ligand fragment composition of the CLEAN reads   (clean tree)
    07  UMI complexity      effective complexity per round and its collapse relative to R0
    08  UMI abundance       how reads distribute over UMIs, and how dominance grows
    09  UMI sharing         corrected UMIs shared across each selection's rounds (UpSet)
    10  fragvec quant       whether vectors are deep enough to quantify, and their complexity
    11  fragvec trajectory  the last round's top vectors, traced back through the rounds
    12  fragvec sharing     vectors shared across each selection's rounds (UpSet)

Figures 13-22 (static_figs.py) continue the series with cross-group agreement.

Every number is read from the step that produced it (clean_summary, pychopper's batch_summary, the
decoder's own log line, coverage_summary, umi_summary, featurecount/fragvec summaries, the complexity
tables) -- nothing is re-estimated here. Each figure's data are written as a TSV beside it, so a
number on a page can be traced to a file. Sample identity is report_lib.PlotStyle: group = colour +
marker shape, each sample its own line dash -- the same as in the HTML reports. PDF only.

Every input path arrives from the Snakemake rule (`--sheet`); nothing is globbed.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import textwrap
from pathlib import Path

import numpy as np
import polars as pl

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages                     # noqa: E402
from matplotlib.lines import Line2D                                      # noqa: E402
from matplotlib.patches import Patch                                     # noqa: E402

import ligand_vocab as LV                                                # noqa: E402
import report_lib as RL                                                  # noqa: E402
import static_style as S                                                 # noqa: E402

ANCHORS = {"UMI1_anchor", "UMI1_anchor_rc", "UMI2_anchor", "UMI2_anchor_rc"}
UMI_BINS = [(1, 1, "1"), (2, 2, "2"), (3, 5, "3–5"), (6, 10, "6–10"), (11, 10 ** 12, ">10")]
TOP_VECTORS = 10
HEATMAP_ROWS = 60
#: The decoder's own accounting line, e.g.
#: "UMI pass: 421964 reads fused, 83568 dropped (missing_anchor 17799, strand_conflict 65769)"
LOG_UMI = re.compile(r"UMI pass:\s*(\d+) reads fused,\s*(\d+) dropped\s*\(([^)]*)\)")


class Ctx:
    """Everything the figures share: samples (config order), their style, parameters, outputs."""

    def __init__(self, a):
        self.samples = sorted(json.loads(a.sheet), key=lambda s: s["idx"])
        self.G = S.Groups(self.samples)
        self.a = a
        self.out = a.outdir
        self.tables = a.outdir / "tables"
        self.tables.mkdir(parents=True, exist_ok=True)
        self.names = [s["name"] for s in self.samples]
        self.by = {s["name"]: s for s in self.samples}

    #: Set by main() for the figure being built: its number and file stem, both from FIGURES order.
    number: int = 0
    stem: str = ""

    def write(self, df: pl.DataFrame) -> pl.DataFrame:
        """The current figure's data, as tables/<stem>.tsv."""
        df.write_csv(self.tables / f"{self.stem}.tsv", separator="\t")
        return df


# ------------------------------------------------------------------------------------ io helpers

def tsv_rows(path) -> list[dict]:
    if not path or not Path(path).exists():
        return []
    lines = [ln for ln in Path(path).read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    return [dict(zip(lines[0].split("\t"), ln.split("\t"))) for ln in lines[1:]] if lines else []


def tsv_row(path) -> dict:
    rows = tsv_rows(path)
    return rows[0] if rows else {}


def num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def pct(part, whole) -> float:
    """100 * part / whole; NaN (never a fabricated 0) when either is missing or whole is 0."""
    part, whole = num(part), num(whole)
    return 100.0 * part / whole if np.isfinite(part) and np.isfinite(whole) and whole > 0 else float("nan")


def decoder_outcome(log_path) -> dict:
    """{'fused': n, <drop reason>: n, ...} from the decoder's own `UMI pass:` line."""
    text = Path(log_path).read_text() if log_path and Path(log_path).exists() else ""
    m = LOG_UMI.search(text)
    if not m:
        return {}
    out = {"fused": int(m[1])}
    for part in m[3].split(","):
        bits = part.strip().rsplit(" ", 1)
        if len(bits) == 2 and bits[1].isdigit():
            out[bits[0].replace("_", " ")] = int(bits[1])
    return out


# ------------------------------------------------------------------------------- drawing helpers

def label_panel(ax, letter: str, dx: float = -0.08, dy: float = 1.04) -> None:
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="right")


def sample_axis(ax, ctx, names, vertical=True) -> None:
    """Samples on the category axis in config order: full names, group dividers between blocks."""
    pos = np.arange(len(names))
    if vertical:
        ax.set_xticks(pos, names, rotation=90, fontsize=5.2)
        ax.set_xlim(-0.6, len(names) - 0.4)
    else:
        ax.set_yticks(pos, names, fontsize=5.2)
        ax.set_ylim(len(names) - 0.4, -0.6)
    groups = [ctx.by[n]["group"] for n in names]
    for i in range(1, len(names)):
        if groups[i] != groups[i - 1]:
            (ax.axvline if vertical else ax.axhline)(i - 0.5, color=S.GREY, lw=0.5, ls=(0, (1, 1.5)))


def group_bars(ax, ctx, names, values, fmt=None, log=False, **kw) -> None:
    """One bar per sample in its group's colour, the value printed above it when `fmt` is given."""
    pos = np.arange(len(names))
    vals = np.array(values, dtype=float)
    ax.bar(pos, vals, width=0.72, color=[ctx.G[n].color for n in names], edgecolor="white", lw=0.4, **kw)
    if log:
        ax.set_yscale("log")
    if fmt:
        for x, v in zip(pos, vals):
            if np.isfinite(v):
                ax.text(x, v, fmt(v), ha="center", va="bottom", fontsize=4.6, rotation=90)
    sample_axis(ax, ctx, names)


def curve_axes(fig, cell, ctx, names):
    """[(ax, members)]: one axis while ctx.G.panels() says one panel is readable, else one small axis
    per group (each with the R0 baseline) laid out in `cell`."""
    panels = ctx.G.panels([ctx.by[n] for n in names])
    if len(panels) == 1:
        return [(fig.add_subplot(cell), [s["name"] for s in panels[0][1]])]
    cols = min(4, len(panels))
    sub = cell.subgridspec(math.ceil(len(panels) / cols), cols, hspace=0.6, wspace=0.3)
    out = []
    for i, (title, members) in enumerate(panels):
        ax = fig.add_subplot(sub[i // cols, i % cols])
        ax.set_title(title, fontsize=6)
        out.append((ax, [s["name"] for s in members]))
    return out


def sample_legend(fig, ctx, names, markers=False, **kw):
    """One entry per sample under its group's name -- the print twin of the reports' legend."""
    handles, labels = [], []
    for g in ctx.G.order:
        members = [n for n in names if ctx.by[n]["group"] == g]
        if not members:
            continue
        if members != [g]:                       # a lone sample named like its group needs no header
            handles.append(Line2D([], [], ls="none"))
            labels.append(g)
        for n in members:
            st = ctx.G[n].mpl(lw=1.0, markers=markers)
            st.pop("label")
            handles.append(Line2D([], [], **st))
            labels.append(n)
    kw = {"loc": "center left", "bbox_to_anchor": (1.0, 0.5), "fontsize": 5, "frameon": False,
          "handlelength": 3.2, **kw}
    leg = fig.legend(handles, labels, **kw)
    for text, label in zip(leg.get_texts(), labels):
        if label in ctx.G.order and label not in names:
            text.set_fontweight("bold")
    return leg


def r0_point(ax, ctx, x, y, hollow=False) -> None:
    """The initial library: ONE grey point every group's line starts from (not one per group)."""
    if np.isfinite(y):
        ax.plot(x, y, marker="o", ms=4.2, color=RL.BASELINE_GREY, mfc="white" if hollow else RL.BASELINE_GREY,
                zorder=4, ls="none")


def message(fig, ctx, text: str) -> None:
    fig.suptitle(f"{ctx.number:02d}  {text}", fontsize=7.5, fontweight="bold", x=0.01, ha="left", y=0.995)


def no_data(ax, why: str) -> None:
    why = "\n".join(textwrap.fill(part, 38) for part in why.split("\n"))
    ax.text(0.5, 0.5, why, ha="center", va="center", fontsize=6, color=S.GREY, transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


# =============================================================================== 01 raw sequencing

def fig_raw(ctx):
    rows = []
    for s in ctx.samples:
        prof = pl.read_csv(s["profile"], separator="\t", has_header=False,
                           new_columns=["length", "quality"]) if Path(s["profile"]).exists() else None
        raw = num(tsv_row_for(ctx.a.clean, "sample", s["name"]).get("total_reads"))
        L = prof["length"].cast(pl.Float64).to_numpy() if prof is not None else np.array([])
        Q = prof["quality"].cast(pl.Float64).to_numpy() if prof is not None else np.array([])
        rows.append({"sample": s["name"], "group": s["group"], "round": s["round"], "raw_reads": raw,
                     "bases": float(L.sum()) if L.size else float("nan"),
                     "median_length": float(np.median(L)) if L.size else float("nan"),
                     "n50": n50(L), "median_quality": float(np.median(Q)) if Q.size else float("nan"),
                     "pct_q_ge_10": 100 * float(np.mean(Q >= 10)) if Q.size else float("nan"),
                     "_L": L, "_Q": Q})
    ctx.write(pl.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]))

    fig = plt.figure(figsize=S.figsize("3col", 0.62))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.95, wspace=0.28,
                          left=0.07, right=0.84, top=0.90, bottom=0.14)
    ax = fig.add_subplot(gs[0, 0])
    group_bars(ax, ctx, ctx.names, [r["raw_reads"] for r in rows], fmt=lambda v: f"{v / 1e6:.2f}M")
    ax.set_ylabel("Raw reads")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1e6:g}M"))
    label_panel(ax, "A")
    ax = fig.add_subplot(gs[0, 1])
    group_bars(ax, ctx, ctx.names, [r["median_quality"] for r in rows], fmt=lambda v: f"{v:.1f}")
    ax.set_ylabel("Median read quality (Phred)")
    label_panel(ax, "B")
    by = {r["sample"]: r for r in rows}
    for key, cell, xlabel, letter, cut in (("_L", gs[1, 0], "Read length (nt, log)", "C", ctx.a.minlen),
                                           ("_Q", gs[1, 1], "Per-read mean quality (Phred)", "D", None)):
        for i, (ax, members) in enumerate(curve_axes(fig, cell, ctx, ctx.names)):
            for n in members:
                v = np.sort(by[n][key])
                if not v.size:
                    continue
                idx = np.unique(np.r_[np.linspace(0, v.size - 1, 600).astype(int)])
                ax.plot(v[idx], 1 - (idx + 1) / v.size, **ctx.G[n].mpl(lw=0.8))
            if key == "_L":
                ax.set_xscale("log")
            if cut:
                ax.axvline(cut, color=S.GREY, lw=0.6, ls=(0, (1, 1.5)))
                ax.text(cut, 1.0, f" clean: > {cut} nt", fontsize=4.8, color=S.GREY, va="top")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Fraction of reads ≥ x")
            if i == 0:
                label_panel(ax, letter)
    sample_legend(fig, ctx, ctx.names)
    message(fig, ctx, "Raw sequencing: depth and read quality per sample, before any filtering")
    return fig, {
        "A": "raw reads per sample (clean_summary total_reads, seqkit)",
        "B": "median per-read mean quality (seqkit fx2tab of the raw FASTQ)",
        "C": "read-length survival curve, fraction of reads at least x nt; dotted = the clean length cut",
        "D": "per-read mean-quality survival curve"}


def tsv_row_for(path, key, value) -> dict:
    return next((r for r in tsv_rows(path) if r.get(key) == value), {})


def n50(lengths: np.ndarray) -> float:
    if not lengths.size:
        return float("nan")
    v = np.sort(lengths)[::-1]
    c = np.cumsum(v)
    return float(v[np.searchsorted(c, c[-1] / 2)])


# ================================================================================= 02 read funnel

def funnel_rows(ctx, s) -> list[dict]:
    """One row per (step, reason) with that step's OWN input and output counts."""
    n = s["name"]
    c = tsv_row_for(ctx.a.clean, "sample", n)
    p = tsv_row_for(ctx.a.preprocess, "Sample", n)
    v = tsv_row_for(ctx.a.covfilt, "sample", n) if ctx.a.covfilt else {}
    u = tsv_row(s["umi_summary"])
    dec = decoder_outcome(s["decode_log"])
    rows = []

    def step(name, n_in, n_out, losses):
        rows.append({"step": name, "n_in": n_in, "n_out": n_out, "losses": losses})

    raw = num(c.get("total_reads"))
    hosts = {k[:-len("_mapped")]: num(c[k]) for k in c if k.endswith("_mapped")}
    unmapped = num(c.get("unmapped_all")) if "unmapped_all" in c else raw - sum(hosts.values())
    step("host depletion", raw, unmapped, {f"{g} reads": x for g, x in hosts.items()})
    kept = next((num(c[k]) for k in c if k.startswith("kept_gt")), float("nan"))
    short = next((num(c[k]) for k in c if k.startswith("dropped_le")), float("nan"))
    step("length filter", unmapped, kept, {f"≤ {ctx.a.minlen} nt": short})
    fl_in, fl = num(p.get("input_pass")), num(p.get("full_length"))
    step("pychopper", fl_in, fl, {"not full-length": fl_in - fl})
    if dec:
        drops = {k: x for k, x in dec.items() if k != "fused"}
        step("decode (UMI anchors)", dec["fused"] + sum(drops.values()), dec["fused"], drops)
    if v:
        step("coverage filter", num(v.get("reads_in")), num(v.get("reads_kept")),
             {f"decoded coverage < {num(v.get('min_coverage')):g}": num(v.get("dropped_lowcoverage"))})
    step("UMI QC", num(u.get("reads_in")), num(u.get("reads_umi_ok")),
         {"UMI non-ACGT": num(u.get("dropped_nonacgt")),
          "UMI low complexity": num(u.get("dropped_lowcomplexity")),
          "UMI wrong length": num(u.get("dropped_umilen"))})
    step("anchor-only", num(u.get("reads_umi_ok")), num(u.get("reads_kept")),
         {"no ligand fragment": num(u.get("dropped_no_payload"))})
    if "jackpot_reads_removed" in u:
        removed = num(u.get("jackpot_reads_removed"))
        step("jackpot blacklist", num(u.get("reads_kept")), num(u.get("reads_kept")) - removed,
             {"jackpot UMI (R0)": removed})
    final = num(tsv_row(s["clean_fc_summary"]).get("n_reads"))

    out, prev_out = [], raw
    for i, r in enumerate(rows, start=1):
        gap = prev_out - r["n_in"]                 # reads that left between two steps' own counts
        base = {"sample": n, "group": s["group"], "round": s["round"], "step_order": i,
                "step": r["step"], "n_in": r["n_in"], "n_out": r["n_out"],
                "pct_of_raw_remaining": 100 * r["n_out"] / raw if raw else float("nan")}
        if np.isfinite(gap) and gap != 0:
            out.append({**base, "reason": f"unaccounted before {r['step']}", "n_removed": gap})
        for reason, x in r["losses"].items():
            out.append({**base, "reason": reason, "n_removed": x})
        prev_out = r["n_out"]
    for row in out:
        row["pct_of_raw"] = 100 * row["n_removed"] / raw if raw else float("nan")
        row["pct_of_step_input"] = 100 * row["n_removed"] / row["n_in"] if row["n_in"] else float("nan")
        row["final_clean_reads"] = final
        row["raw_reads"] = raw
    return out


def fig_funnel(ctx):
    df = ctx.write(pl.DataFrame([r for s in ctx.samples for r in funnel_rows(ctx, s)]))
    steps = df.select("step_order", "step").unique().sort("step_order")["step"].to_list()
    reasons = (df.filter(pl.col("n_removed") != 0).select("step_order", "reason").unique()
               .sort("step_order", "reason")["reason"].to_list())
    reasons = list(dict.fromkeys(reasons))
    step_of = dict(df.select("reason", "step").unique().iter_rows())
    step_color = dict(zip(steps, S.OKABE_ITO_LIST[:7] + RL.GROUP_PALETTE[3:] + RL.extra_colors(4)))
    hatches = ["", "////", "....", "xxxx", "\\\\\\\\"]
    seen_in_step: dict = {}
    reason_style = {}
    for rsn in reasons:
        st = step_of[rsn]
        k = seen_in_step.get(st, 0)
        seen_in_step[st] = k + 1
        reason_style[rsn] = dict(color=step_color[st] if "unaccounted" not in rsn else "#d0d0d0",
                                 hatch=hatches[k % len(hatches)])

    fig = plt.figure(figsize=S.figsize("3col", 0.80))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], wspace=0.55, left=0.07, right=0.80, top=0.90,
                          bottom=0.22)
    # A: remaining reads after each step
    for i, (ax, members) in enumerate(curve_axes(fig, gs[0, 0], ctx, ctx.names)):
        for n in members:
            sub = df.filter(pl.col("sample") == n).unique("step_order").sort("step_order")
            y = [100.0] + sub["pct_of_raw_remaining"].to_list()
            ax.plot(range(len(y)), y, **ctx.G[n].mpl(lw=0.8, ms=2.6, markers=True))
        ax.set_yscale("log")
        ax.set_xticks(range(len(steps) + 1), ["raw"] + steps, rotation=60, ha="right", fontsize=5)
        ax.set_ylabel("Reads remaining (% of raw, log)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
        if i == 0:
            label_panel(ax, "A")
    # B: what each step removed, as % of the sample's raw reads, stacked in step order
    ax = fig.add_subplot(gs[0, 1])
    left = np.zeros(len(ctx.names))
    for rsn in reasons:
        vals = np.array([df.filter((pl.col("sample") == n) & (pl.col("reason") == rsn))["pct_of_raw"].sum()
                         for n in ctx.names])
        ax.barh(np.arange(len(ctx.names)), vals, left=left, height=0.72, edgecolor="white", lw=0.3,
                **reason_style[rsn])
        left += vals
    finals = [df.filter(pl.col("sample") == n)["final_clean_reads"][0] / df.filter(pl.col("sample") == n)["raw_reads"][0] * 100
              for n in ctx.names]
    for y, (x, f) in enumerate(zip(left, finals)):
        ax.text(min(x, 100) + 1, y, f"{f:.1f}% kept", va="center", fontsize=4.6)
    sample_axis(ax, ctx, ctx.names, vertical=False)
    ax.set_xlim(0, 118)
    ax.set_xlabel("Reads removed (% of raw)")
    label_panel(ax, "B", dx=-0.25)
    handles = [Patch(facecolor=reason_style[r]["color"], hatch=reason_style[r]["hatch"], edgecolor="white",
                     label=f"{step_of[r]}: {r}") for r in reasons]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.80, 0.55), fontsize=5, frameon=False)
    sample_legend(fig, ctx, ctx.names, markers=True, loc="upper left", bbox_to_anchor=(0.0, 0.10), ncol=6,
                  fontsize=4.6)
    message(fig, ctx, "Read funnel: reads kept by every processing step, and why the rest were removed")
    return fig, {
        "A": "reads remaining after each step as % of the raw reads; each step's own output count",
        "B": "reads each step removed, by its own stated reason, as % of the sample's raw reads, stacked "
             "in step order; 'unaccounted before X' = the previous step's output minus step X's own input count",
        "sources": "clean_summary.tsv, preprocess/batch_summary.tsv, the decoder's ligateddna.log "
                   "'UMI pass' line, covfilt/coverage_summary.tsv, umi_summary.tsv, clean featurecount summary"}


# ================================================================================== 03 decode QC

def fig_decode(ctx):
    rows, hist = [], {}
    grid = np.linspace(0.5, 1.0, 101)
    cut = ctx.a.min_coverage
    for s in ctx.samples:
        dec = decoder_outcome(s["decode_log"])
        total = sum(dec.values()) if dec else float("nan")
        cov = np.sort(pl.read_parquet(s["coverage"], columns=["cov"])["cov"].to_numpy())
        # survival: % of reads whose coverage is at least x, on a fixed grid (smooth, no bin artefacts)
        hist[s["name"]] = 100 * (1 - np.searchsorted(cov, grid, side="left") / max(1, cov.size))
        v = tsv_row_for(ctx.a.covfilt, "sample", s["name"]) if ctx.a.covfilt else {}
        rows.append({"sample": s["name"], "group": s["group"], "round": s["round"],
                     "full_length_reads": total, **{f"decoder_{k}": v for k, v in dec.items()},
                     "median_coverage": float(np.median(cov)) if cov.size else float("nan"),
                     "pct_dropped_by_coverage_filter": pct(v.get("dropped_lowcoverage"), v.get("reads_in"))})
    df = ctx.write(pl.DataFrame(rows))
    outcomes = [c for c in df.columns if c.startswith("decoder_")]

    fig = plt.figure(figsize=S.figsize("3col", 0.62))
    gs = fig.add_gridspec(2, 2, hspace=1.0, wspace=0.28, left=0.07, right=0.84, top=0.90, bottom=0.14)
    ax = fig.add_subplot(gs[0, 0])
    bottom = np.zeros(len(ctx.names))
    colors = [S.OKABE_ITO["blue"], S.OKABE_ITO["vermillion"], S.OKABE_ITO["orange"], S.OKABE_ITO["reddish_purple"], S.GREY]
    for c, color in zip(outcomes, colors):
        vals = 100 * df[c].fill_null(0).to_numpy() / df["full_length_reads"].to_numpy()
        ax.bar(np.arange(len(ctx.names)), vals, bottom=bottom, width=0.72, color=color, edgecolor="white", lw=0.3,
               label=c.removeprefix("decoder_"))
        bottom += vals
    sample_axis(ax, ctx, ctx.names)
    ax.set_ylabel("% of full-length reads")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(outcomes), frameon=False)
    label_panel(ax, "A", dx=-0.12)
    ax = fig.add_subplot(gs[0, 1])
    if cut is None:
        no_data(ax, "coverage filter off")
    else:
        group_bars(ax, ctx, ctx.names, df["pct_dropped_by_coverage_filter"].to_list(), fmt=lambda v: f"{v:.0f}%")
        ax.set_ylabel(f"% dropped (coverage < {cut:g})")
    label_panel(ax, "B", dx=-0.12)
    for i, (ax, members) in enumerate(curve_axes(fig, gs[1, :], ctx, ctx.names)):
        for n in members:
            ax.plot(100 * grid, hist[n], **ctx.G[n].mpl(lw=0.8))
        if cut is not None:
            ax.axvline(100 * cut, color=S.GREY, lw=0.6, ls=(0, (1, 1.5)))
            ax.text(100 * cut, 2, f" kept: ≥ {100 * cut:.0f}%", fontsize=4.8, color=S.GREY, va="bottom")
        ax.set_xlim(50, 100)
        ax.set_ylim(0, 102)
        ax.set_xlabel("Decoded coverage x (% of read length explained by fragment calls)")
        ax.set_ylabel("% reads with coverage ≥ x")
        if i == 0:
            label_panel(ax, "C", dx=-0.06)
    sample_legend(fig, ctx, ctx.names)
    message(fig, ctx, "Decode QC: how many full-length reads decode, and how much of each read the decoder explains")
    return fig, {
        "A": "decoder outcome of every full-length read, from its own 'UMI pass' log line",
        "B": "share of decoded reads the coverage filter dropped (covfilt/coverage_summary.tsv)",
        "C": "per-read decoded coverage as a survival curve (share of reads at or above x), measured on "
             "the raw decode tree; dotted = the cut, reads left of it are dropped"}


# ============================================================================ 04 raw fragment calls

def fragment_profile(fc_path, frags_path, n_reads):
    """Payload copies per read (0 = anchor-only) and payload-call edit distances."""
    fc = pl.read_parquet(fc_path)
    per = (fc.filter(~pl.col("fragment_id").is_in(list(ANCHORS)))
           .group_by("read_acc").agg(pl.col("count").sum().alias("n")))
    counts = per["n"].to_numpy()
    mult = np.bincount(np.minimum(counts, 10), minlength=11).astype(float)
    mult[0] += max(0, n_reads - per.height)
    ed = (pl.scan_parquet(frags_path).filter(~pl.col("fragment_id").is_in(list(ANCHORS)))
          .select("edit_distance").collect()["edit_distance"].to_numpy())
    edh = np.bincount(np.minimum(np.maximum(ed, 0), 8), minlength=9).astype(float)
    return 100 * mult / max(1, mult.sum()), 100 * edh / max(1, edh.sum())


def fig_raw_fragments(ctx):
    rows, mult, edh = [], {}, {}
    for s in ctx.samples:
        sm = tsv_row(s["decode_fc_summary"])
        n_reads = int(num(sm.get("n_reads")))
        mult[s["name"]], edh[s["name"]] = fragment_profile(s["decode_fc"], s["decode_frags"], n_reads)
        rows.append({"sample": s["name"], "group": s["group"], "round": s["round"], "reads_decoded": n_reads,
                     "pct_with_fragment": pct(sm.get("n_reads_with_payload"), n_reads),
                     "mean_payload_copies_per_read": num(sm.get("mean_payload_copies_per_read")),
                     **{f"pct_reads_{k}{'+' if k == 10 else ''}_fragments": v for k, v in enumerate(mult[s["name"]])},
                     **{f"pct_calls_ed{k}{'+' if k == 8 else ''}": v for k, v in enumerate(edh[s["name"]])}})
    qc = {r["key"]: r for r in tsv_rows(ctx.a.qc_panel)}
    kinds = [("mismatch_rate", "mismatch"), ("insertion_rate", "insertion"), ("deletion_rate", "deletion")]
    for r in rows:
        q = qc.get(r["sample"], {})
        r.update({k: num(q.get(k)) for k, _ in kinds},
                 identity=num(q.get("identity_from_edit_distance")),
                 library_members_detected=num(q.get("library_members_detected")),
                 library_members_designed=num(q.get("library_members_designed")))
    df = ctx.write(pl.DataFrame(rows))

    fig = plt.figure(figsize=S.figsize("3col", 0.62))
    gs = fig.add_gridspec(2, 6, hspace=0.95, wspace=1.6, left=0.07, right=0.84, top=0.90, bottom=0.14)
    ax = fig.add_subplot(gs[0, 0:2])
    group_bars(ax, ctx, ctx.names, df["pct_with_fragment"].to_list(), fmt=lambda v: f"{v:.0f}%")
    ax.set_ylabel("% reads with ≥1 fragment")
    ax.set_ylim(0, 105)
    label_panel(ax, "A", dx=-0.2)
    ax = fig.add_subplot(gs[0, 2:4])
    group_bars(ax, ctx, ctx.names, df["mean_payload_copies_per_read"].to_list(), fmt=lambda v: f"{v:.2f}")
    ax.set_ylabel("Fragment copies / read")
    label_panel(ax, "B", dx=-0.2)
    ax = fig.add_subplot(gs[0, 4:6])
    if df["mismatch_rate"].is_nan().all():
        no_data(ax, "error decomposition unavailable (static_tables' ont_qc_panel.tsv)")
    else:
        bottom = np.zeros(len(ctx.names))
        for (col, lab), color in zip(kinds, (S.OKABE_ITO["blue"], S.OKABE_ITO["orange"], S.OKABE_ITO["vermillion"])):
            vals = 100 * df[col].fill_nan(0).to_numpy()
            ax.bar(np.arange(len(ctx.names)), vals, bottom=bottom, width=0.72, color=color, edgecolor="white",
                   lw=0.3, label=lab)
            bottom += vals
        sample_axis(ax, ctx, ctx.names)
        ax.set_ylabel("Errors per 100 reference bases")
        ax.legend(fontsize=5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    label_panel(ax, "C", dx=-0.2)
    for cell, data, xt, xlabel, ylabel, letter in (
            (gs[1, 0:3], mult, [str(k) for k in range(10)] + ["10+"], "Fragment copies in a read", "% of decoded reads", "D"),
            (gs[1, 3:6], edh, [str(k) for k in range(8)] + ["8+"], "Edit distance of a fragment call", "% of fragment calls", "E")):
        for i, (ax, members) in enumerate(curve_axes(fig, cell, ctx, ctx.names)):
            for n in members:
                ax.plot(range(len(xt)), data[n], **ctx.G[n].mpl(lw=0.8, ms=2.2, markers=True))
            ax.set_xticks(range(len(xt)), xt, fontsize=5)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            if i == 0:
                label_panel(ax, letter)
    sample_legend(fig, ctx, ctx.names, markers=True)
    message(fig, ctx, "Raw fragment calls: what the decoder called in every read, before any UMI processing")
    return fig, {"A": "decoded reads carrying at least one ligand fragment (decode-tree featurecount summary)",
                 "B": "mean ligand-fragment copies per decoded read",
                 "C": "call accuracy: mismatches, insertions and deletions per 100 reference bases, from a fitting "
                      "re-alignment of up to 20,000 sampled fragment calls per sample (static_tables' "
                      "ont_qc_panel.tsv, which also carries identity and library members detected)",
                 "D": "fragment copies per read; 0 = anchor-only reads, removed later by the UMI step",
                 "E": "edit distance of every ligand-fragment call against its expected sequence"}


# ======================================================================================= 05 UMI QC

def fig_umi_qc(ctx):
    rows = []
    for s in ctx.samples:
        u = tsv_row(s["umi_summary"])
        reads_in = num(u.get("reads_in"))
        r = {"sample": s["name"], "group": s["group"], "round": s["round"], "reads_in": reads_in}
        for col, label in (("dropped_nonacgt", "non-ACGT"), ("dropped_lowcomplexity", "low complexity"),
                           ("dropped_umilen", "wrong length"), ("dropped_no_payload", "anchor-only"),
                           ("jackpot_reads_removed", "jackpot UMI")):
            if col in u:
                r[f"pct_{label}"] = pct(u[col], reads_in)
        r["pct_species_merged"] = pct(u.get("n_species_merged"), u.get("n_species_raw"))
        r["pct_reads_relabelled"] = pct(u.get("n_reads_relabeled"), u.get("reads_kept"))
        r["jackpot_umis_hit"] = num(u.get("jackpot_umis_hit"))
        r["jackpot_reads_removed"] = num(u.get("jackpot_reads_removed"))
        rows.append(r)
    df = ctx.write(pl.DataFrame(rows))
    reasons = [c for c in df.columns if c.startswith("pct_") and c not in ("pct_species_merged", "pct_reads_relabelled")]
    bl = {r["key"]: r["value"] for r in tsv_rows(ctx.a.jackpot_summary)} if ctx.a.jackpot_summary else {}

    fig = plt.figure(figsize=S.figsize("3col", 0.42))
    gs = fig.add_gridspec(1, 3, wspace=0.50, left=0.07, right=0.99, top=0.78, bottom=0.30)
    ax = fig.add_subplot(gs[0, 0])
    bottom = np.zeros(len(ctx.names))
    for c, color in zip(reasons, [S.OKABE_ITO[k] for k in ("black", "orange", "sky_blue", "reddish_purple", "vermillion")]):
        vals = df[c].fill_nan(0).fill_null(0).to_numpy()
        ax.bar(np.arange(len(ctx.names)), vals, bottom=bottom, width=0.72, color=color, edgecolor="white", lw=0.3,
               label=c.removeprefix("pct_"))
        bottom += vals
    sample_axis(ax, ctx, ctx.names)
    ax.set_ylabel("% of reads entering the UMI step")
    ax.legend(fontsize=5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    label_panel(ax, "A", dx=-0.14)
    ax = fig.add_subplot(gs[0, 1])
    pos = np.arange(len(ctx.names))
    colors = [ctx.G[n].color for n in ctx.names]
    ax.bar(pos - 0.18, df["pct_species_merged"].to_numpy(), width=0.36, color=colors, label="UMI species merged")
    ax.bar(pos + 0.18, df["pct_reads_relabelled"].to_numpy(), width=0.36, color=colors, alpha=0.45,
           hatch="////", edgecolor="white", lw=0.3, label="reads relabelled")
    sample_axis(ax, ctx, ctx.names)
    ax.set_ylabel("ed ≤ 1 correction (%)")
    ax.legend(handles=[Patch(color=S.GREY, label="UMI species merged into a neighbour"),
                       Patch(facecolor=S.GREY, alpha=0.45, hatch="////", edgecolor="white", label="reads relabelled")],
              fontsize=5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    label_panel(ax, "B", dx=-0.14)
    ax = fig.add_subplot(gs[0, 2])
    if df["jackpot_reads_removed"].is_nan().all():
        no_data(ax, "jackpot filter off")
    elif df["jackpot_reads_removed"].fill_nan(0).sum() == 0:
        no_data(ax, "the R0 jackpot blacklist removed no reads\n"
                    f"({bl.get('n_blacklist_rows', '0')} blacklisted UMIs; "
                    f"{bl.get('n_flagged_clonotypes', '0')} of {bl.get('n_signal_clonotypes', '?')} R0 vectors "
                    f"above ratio {bl.get('ratio_max', '?')})")
    else:
        group_bars(ax, ctx, ctx.names, df["jackpot_reads_removed"].to_list(), fmt=lambda v: f"{v:,.0f}")
        ax.set_ylabel("Reads removed by the R0 jackpot blacklist")
        if bl:
            ax.set_title(f"blacklist: {bl.get('n_blacklist_rows', '?')} UMIs from "
                         f"{bl.get('n_flagged_clonotypes', '?')} flagged vectors in R0 "
                         f"(ratio > {bl.get('ratio_max', '?')})", fontsize=5.5)
    label_panel(ax, "C", dx=-0.14)
    message(fig, ctx, "UMI QC: what UMI filtering, ed ≤ 1 correction and the R0 jackpot blacklist remove")
    return fig, {"A": "reads removed by each UMI-step rule, % of the reads entering the UMI step (umi_summary)",
                 "B": "ed ≤ 1 correction: raw UMI species merged into a more abundant neighbour, and reads "
                      "whose UMI was relabelled; nothing is collapsed",
                 "C": "reads removed per sample by the jackpot blacklist detected on the R0 library"}


# ================================================================================ 06 fragment quant

def fig_fragment_quant(ctx):
    expected = LV.read_expected(ctx.a.expected)
    per, rows = {}, []
    for s in ctx.samples:
        fc = pl.read_parquet(s["clean_fc"]).filter(~pl.col("fragment_id").is_in(list(ANCHORS)))
        n_reads = num(tsv_row(s["clean_fc_summary"]).get("n_reads"))
        agg = (fc.with_columns(pl.col("fragment_id").str.replace(r"_rc$", "").alias("feature"))
               .group_by("feature").agg(pl.col("count").sum().alias("copies")))
        per[s["name"]] = {f: c / n_reads for f, c in agg.iter_rows()}
        for f, c in agg.iter_rows():
            rows.append({"sample": s["name"], "group": s["group"], "round": s["round"], "feature": f,
                         "class": LV.parse_feature(f)[0], "copies": c, "copies_per_read": c / n_reads})
    df = ctx.write(pl.DataFrame(rows))
    order = LV.class_order(set(df["feature"].to_list()), expected, ANCHORS)
    ccol = LV.class_colors(order)

    fig = plt.figure(figsize=S.figsize("3col", 0.92))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.8, 1.6], hspace=0.55, wspace=0.30,
                          left=0.10, right=0.92, top=0.92, bottom=0.08)
    ax = fig.add_subplot(gs[0, 0])
    group_bars(ax, ctx, ctx.names, [sum(per[n].values()) for n in ctx.names], fmt=lambda v: f"{v:.2f}")
    ax.set_ylabel("Fragment copies / clean read")
    label_panel(ax, "A")
    ax = fig.add_subplot(gs[0, 1])
    bottom = np.zeros(len(ctx.names))
    for c in order:
        vals = np.array([100 * sum(v for f, v in per[n].items() if LV.parse_feature(f)[0] == c) / max(1e-12, sum(per[n].values()))
                         for n in ctx.names])
        ax.bar(np.arange(len(ctx.names)), vals, bottom=bottom, width=0.72, color=ccol[c], edgecolor="white", lw=0.3, label=c)
        bottom += vals
    sample_axis(ax, ctx, ctx.names)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of fragment copies, by class")
    ax.legend(fontsize=5, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    label_panel(ax, "B")
    ax = fig.add_subplot(gs[1, :])
    base = next((s["name"] for s in ctx.samples if s["group"] == RL.BASELINE_GROUP), None)
    cols = [n for n in ctx.names if n != base]
    allf = {f for n in ctx.names for f in per[n]}
    top = sorted(allf, key=lambda f: -sum(per[n].get(f, 0) for n in ctx.names))[:HEATMAP_ROWS]
    feats = sorted(top, key=lambda f: (order.index(LV.parse_feature(f)[0])
                                       if LV.parse_feature(f)[0] in order else 99, f))
    eps = 1e-5
    if base:
        z = np.array([[math.log2((per[n].get(f, 0) + eps) / (per[base].get(f, 0) + eps)) for n in cols] for f in feats])
        lim = max(1.0, float(np.nanpercentile(np.abs(z), 99)))
        im = ax.imshow(z, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
        cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        cb.set_label("log2 (copies/read ÷ R0)", fontsize=5.5)
        sample_axis(ax, ctx, cols)
        ax.set_yticks(range(len(feats)), feats, fontsize=4.2)
        for t, f in zip(ax.get_yticklabels(), feats):
            t.set_color(ccol.get(LV.parse_feature(f)[0], S.GREY))
        which = (f"All {len(feats)} ligand fragments" if len(feats) == len(allf)
                 else f"The {len(feats)} most abundant ligand fragments (of {len(allf)})")
        ax.set_title(f"{which} in every sample, relative to the initial library ({base})", fontsize=6)
    else:
        no_data(ax, "no initial library: enrichment relative to R0 unavailable")
    label_panel(ax, "C", dx=-0.06, dy=1.01)
    message(fig, ctx, "Fragment quantification on the clean reads: ligand composition per sample and its change from R0")
    return fig, {"A": "ligand-fragment copies per clean read (anchors excluded, strands collapsed)",
                 "B": "share of fragment copies by structural class",
                 "C": f"log2 fold-change of copies/read versus the initial library for the {HEATMAP_ROWS} most "
                      "abundant features (rows, coloured by class, grouped by class) in every sample "
                      "(columns, config order); every feature is in the table"}


# =============================================================================== 07 UMI complexity

def fig_umi_complexity(ctx):
    cx = pl.read_csv(ctx.a.complexity, separator="\t", infer_schema_length=0)
    co = pl.read_csv(ctx.a.collapse, separator="\t", infer_schema_length=0) if Path(ctx.a.collapse).exists() else None
    ctx.write(cx)
    by = {r["name"]: r for r in cx.iter_rows(named=True)}

    fig = plt.figure(figsize=S.figsize("3col", 0.46))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 1], wspace=0.50, left=0.06, right=0.99, top=0.80, bottom=0.28)
    ax = fig.add_subplot(gs[0, 0])
    base = next((s["name"] for s in ctx.samples if s["group"] == RL.BASELINE_GROUP), None)
    for g in ctx.G.groups:
        members = ([base] if base else []) + [s["name"] for s in ctx.samples if s["group"] == g]
        members = [n for n in members if n in by]
        x = [int(by[n]["round"]) for n in members]
        y = np.array([num(by[n]["c_eff"]) for n in members])
        err = np.array([num(by[n]["c_eff"]) / math.sqrt(num(by[n]["dup_pairs"])) if num(by[n]["dup_pairs"]) > 0 else 0
                        for n in members])
        lb = [by[n]["c_eff_is_lower_bound"] == "true" for n in members]
        ax.errorbar(x, y, yerr=err, color=ctx.G.color(g), lw=0.9, capsize=1.4, elinewidth=0.5, zorder=2)
        for n, xi, yi, b in zip(members, x, y, lb):
            if n != base:
                ax.plot(xi, yi, marker=ctx.G.marker(g), ms=3.6, color=ctx.G.color(g),
                        mfc="white" if b else ctx.G.color(g), zorder=3)
    if base in by:
        r0_point(ax, ctx, int(by[base]["round"]), num(by[base]["c_eff"]))
    ax.set_yscale("log")
    ax.set_xlabel("Selection round (0 = initial library)")
    ax.set_ylabel("C_eff (effective lineage complexity)")
    ax.legend(handles=[Line2D([], [], color=ctx.G.color(g), marker=ctx.G.marker(g), ms=3.4, lw=0.9, label=g)
                       for g in ctx.G.groups] + [Line2D([], [], color=S.GREY, marker="o", mfc="white", ls="none",
                                                        ms=3.4, label="lower bound (no duplicates)")],
              fontsize=5, frameon=False, loc="lower left")
    label_panel(ax, "A")
    ax = fig.add_subplot(gs[0, 1])
    if co is not None and co.height:
        main = co.filter(pl.col("kind").is_in(["total", "protocol"]))
        sel = {r["sample"]: num(r["ratio"]) for r in co.filter(pl.col("kind") == "selection").iter_rows(named=True)}
        names = [n for n in ctx.names if n in set(main["sample"].to_list())]
        vals = [num(main.filter(pl.col("sample") == n)["ratio"][0]) for n in names]
        group_bars(ax, ctx, names, vals, log=True)
        sx = [i for i, n in enumerate(names) if n in sel]
        ax.plot(sx, [sel[names[i]] for i in sx], ls="none", marker="_", ms=6, mew=1.2, color=S.BLACK,
                label="selection part (vs round-matched empty round)")
        ax.axhline(1, color=S.GREY, lw=0.5)
        ax.set_ylabel("Collapse vs R0 (fold, log)")
        if sx:
            ax.legend(fontsize=5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    else:
        no_data(ax, "no collapse table")
    label_panel(ax, "B")
    ax = fig.add_subplot(gs[0, 2])
    names = [n for n in ctx.names if n in by]
    group_bars(ax, ctx, names, [max(num(by[n]["dup_pairs"]), 0.5) for n in names], log=True)
    ax.set_ylabel("Duplicate read pairs (log)")
    ax.set_title("the evidence each C_eff rests on", fontsize=5.5)
    label_panel(ax, "C", dx=-0.14)
    message(fig, ctx, "UMI complexity: effective complexity falls with selection rounds, relative to the initial library")
    return fig, {"A": "C_eff per round, one line per group from the shared R0 point; bars = C_eff/√dup_pairs; "
                      "hollow = lower bound (no duplicate reads observed)",
                 "B": "collapse relative to R0 (kind total; protocol for the empty arm), log; black ticks = "
                      "the selection part where a round-matched empty round exists (umi_collapse.tsv)",
                 "C": "duplicate read pairs behind each C_eff (its precision: C_eff/√dup_pairs); a sample "
                      "with none has only a lower bound, drawn at 0.5 (umi_complexity.tsv)"}


# ================================================================================ 08 UMI abundance

def fig_umi_abundance(ctx):
    rows, curves = [], {}
    for s in ctx.samples:
        sp = pl.read_csv(s["umi_species"], separator="\t")
        c = np.sort(sp[sp.columns[1]].to_numpy())[::-1]
        tot = c.sum()
        r = {"sample": s["name"], "group": s["group"], "round": s["round"], "n_umis": c.size, "n_reads": int(tot),
             "top10_read_share_pct": 100 * c[:10].sum() / tot if tot else float("nan"),
             "top100_read_share_pct": 100 * c[:100].sum() / tot if tot else float("nan")}
        for lo, hi, lab in UMI_BINS:
            r[f"pct_umis_{lab}_reads"] = 100 * float(np.mean((c >= lo) & (c <= hi))) if c.size else float("nan")
        rows.append(r)
        idx = np.unique(np.r_[0, np.geomspace(1, max(1, c.size), 400).astype(int) - 1, c.size - 1]) if c.size else []
        curves[s["name"]] = (np.asarray(idx) + 1, c[idx] if c.size else [])
    df = ctx.write(pl.DataFrame(rows))

    fig = plt.figure(figsize=S.figsize("3col", 0.50))
    gs = fig.add_gridspec(1, 3, wspace=0.45, left=0.06, right=0.84, top=0.78, bottom=0.27)
    ax = fig.add_subplot(gs[0, 0])
    bottom = np.zeros(len(ctx.names))
    ramp = ["#9ecae1", "#4292c6", "#2171b5", "#08306b"]          # one hue, light -> dark = more reads
    for (lo, hi, lab), color in zip(UMI_BINS[1:], ramp):
        vals = df[f"pct_umis_{lab}_reads"].to_numpy()
        ax.bar(np.arange(len(ctx.names)), vals, bottom=bottom, width=0.72, color=color, edgecolor="white", lw=0.3,
               label=f"{lab} reads")
        bottom += vals
    for x, (b, single) in enumerate(zip(bottom, df["pct_umis_1_reads"].to_list())):
        ax.text(x, b, f"{single:.1f}", ha="center", va="bottom", fontsize=4.4, rotation=90)
    sample_axis(ax, ctx, ctx.names)
    ax.set_ylabel("% of UMIs seen more than once")
    ax.legend(fontsize=5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4,
              columnspacing=0.8, handlelength=1.0,
              title="reads per UMI   (number above bar = % seen once)", title_fontsize=5)
    label_panel(ax, "A", dx=-0.14)
    ax = fig.add_subplot(gs[0, 1])
    group_bars(ax, ctx, ctx.names, df["top10_read_share_pct"].to_list(), fmt=lambda v: f"{v:.1f}")
    ax.set_ylabel("% reads on the top 10 UMIs")
    label_panel(ax, "B", dx=-0.14)
    for i, (ax, members) in enumerate(curve_axes(fig, gs[0, 2], ctx, ctx.names)):
        for n in members:
            x, y = curves[n]
            ax.plot(x, y, **ctx.G[n].mpl(lw=0.8))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("UMI rank")
        ax.set_ylabel("Reads")
        if i == 0:
            label_panel(ax, "C")
    sample_legend(fig, ctx, ctx.names)
    message(fig, ctx, "UMI abundance: most UMIs are seen once; read share on the top UMIs shows dominance")
    return fig, {"A": "UMIs seen more than once, binned by how many reads carry them; the number above each "
                      "bar is the % seen exactly once (corrected UMI species table)",
                 "B": "share of reads on the 10 most abundant corrected UMIs",
                 "C": "rank-abundance of corrected UMIs (depth-dependent length: read for shape)"}


# ================================================================================ 09 fragvec quant

def fig_fragvec(ctx):
    rows, clouds = [], {}
    mr = ctx.a.min_reads
    for s in ctx.samples:
        f = tsv_row(s["fragvec_summary"])
        rows.append({"sample": s["name"], "group": s["group"], "round": s["round"],
                     "reads_payload": num(f.get("reads_payload")), "n_vectors": num(f.get("n_vectors")),
                     "n_vectors_ge_min": num(f.get("n_vectors_ge_min")),
                     "pct_reads_in_vectors_ge_min": num(f.get("pct_reads_in_vectors_ge_min")),
                     "top_vector_read_share_pct": num(f.get("top_vector_read_share_pct")),
                     "c_eff": num(f.get("c_eff")), "c_eff_is_lower_bound": f.get("c_eff_is_lower_bound") == "true"})
        clouds[s["name"]] = (pl.read_parquet(s["fragvec_counts"], columns=["n_reads", "n_umis"])
                             .filter(pl.col("n_reads") >= mr)
                             .with_columns((pl.col("n_reads") / pl.col("n_umis")).alias("reads_per_umi")))
    df = ctx.write(pl.DataFrame(rows))
    by = {r["sample"]: r for r in rows}
    jp = {r["key"]: r["value"] for r in tsv_rows(ctx.a.jackpot_summary)} if ctx.a.jackpot_summary else {}

    fig = plt.figure(figsize=S.figsize("3col", 0.62))
    gs = fig.add_gridspec(2, 3, hspace=0.95, wspace=0.42, left=0.06, right=0.84, top=0.90, bottom=0.14)
    ax = fig.add_subplot(gs[0, 0])
    group_bars(ax, ctx, ctx.names, df["pct_reads_in_vectors_ge_min"].to_list(), fmt=lambda v: f"{v:.0f}")
    ax.set_ylim(0, 105)
    ax.set_ylabel(f"% reads in vectors ≥ {mr} reads")
    label_panel(ax, "A", dx=-0.16)
    ax = fig.add_subplot(gs[0, 1])
    group_bars(ax, ctx, ctx.names, df["n_vectors_ge_min"].to_list(), fmt=lambda v: f"{v:,.0f}")
    ax.set_ylabel(f"Vectors with ≥ {mr} reads")
    label_panel(ax, "B", dx=-0.16)
    ax = fig.add_subplot(gs[0, 2])
    base = next((s["name"] for s in ctx.samples if s["group"] == RL.BASELINE_GROUP), None)
    for g in ctx.G.groups:
        members = ([base] if base else []) + [s["name"] for s in ctx.samples if s["group"] == g]
        ax.plot([ctx.by[n]["round"] for n in members], [by[n]["c_eff"] for n in members], color=ctx.G.color(g),
                marker=ctx.G.marker(g), ms=3.2, lw=0.9, label=g, markevery=list(range(1 if base else 0, len(members))))
    if base:
        r0_point(ax, ctx, ctx.by[base]["round"], by[base]["c_eff"])
    ax.set_yscale("log")
    ax.set_xlabel("Selection round")
    ax.set_ylabel("Vector C_eff")
    ax.legend(fontsize=5, frameon=False)
    label_panel(ax, "C")
    for i, (ax, members) in enumerate(curve_axes(fig, gs[1, :], ctx, ctx.names)):
        hi = mr
        for n in members:
            d = clouds[n]
            if d.height:
                st = ctx.G[n].mpl(markers=True)
                ax.scatter(d["n_reads"], d["reads_per_umi"], s=3, marker=st["marker"], color=st["color"],
                           alpha=0.4, linewidths=0, rasterized=True)
        ax.axhline(1, color=S.GREY, lw=0.6, ls=(0, (3, 2)))
        ratio_max = num(jp.get("ratio_max"))
        ax.set_xscale("log")
        ax.set_yscale("log")
        if np.isfinite(ratio_max):
            ax.axhline(ratio_max, color=S.OKABE_ITO["vermillion"], lw=0.6, ls=(0, (1, 1.5)))
            ax.text(ax.get_xlim()[0], ratio_max, f" jackpot cut ({ratio_max:g} reads/UMI)", fontsize=4.8,
                    color=S.OKABE_ITO["vermillion"], va="bottom")
        ax.set_xlabel("Reads per vector (abundance)")
        ax.set_ylabel("Reads per UMI in the vector")
        if i == 0:
            label_panel(ax, "D", dx=-0.04)
    fig.legend(handles=[Line2D([], [], ls="none", marker=ctx.G.marker(g), color=ctx.G.color(g), label=g)
                        for g in ctx.G.order], loc="center left", bbox_to_anchor=(0.85, 0.28), fontsize=5, frameon=False)
    message(fig, ctx, "Fragvec quantification: vectors are deep enough to quantify, and their complexity collapses with selection")
    return fig, {"A": f"share of payload reads in vectors with at least {mr} reads (fragvec_summary)",
                 "B": f"vectors with at least {mr} reads (depth-dependent)",
                 "C": "vector-level C_eff per round, from fragvec_summary (rendered, not recomputed)",
                 "D": f"reads per UMI against reads per vector, for vectors with ≥ {mr} reads; dashed = every "
                      "read from its own UMI; vermillion = the jackpot ratio cut applied on R0"}


# =========================================================================== 10 fragvec trajectories

def fig_trajectories(ctx):
    panels, rows = [], []
    for g in ctx.G.groups:
        members = [s for s in ctx.samples if s["group"] == g]
        long = pl.read_parquet(members[0]["fragvec_long"])
        order = long["sample"].unique(maintain_order=True).to_list()
        tot = long.group_by("sample").agg(pl.col("n_reads").sum().alias("tot"))
        d = long.join(tot, on="sample").with_columns((1e6 * pl.col("n_reads") / pl.col("tot")).alias("rpm"))
        tops = d.filter(pl.col("sample") == order[-1]).sort("rpm", descending=True).head(TOP_VECTORS)["vector"].to_list()
        panels.append((g, order, d, tops))
        rows += [{"group": g, "rank_at_last_round": i + 1, "vector": v, **{
            f"rpm_{n}": float(d.filter((pl.col("vector") == v) & (pl.col("sample") == n))["rpm"].sum()) for n in order}}
            for i, v in enumerate(tops)]
    ctx.write(pl.DataFrame(rows) if rows else pl.DataFrame({"group": []}))
    if not panels:
        return None, {}
    nrow = len(panels)
    fig, axes = plt.subplots(nrow, 1, figsize=S.figsize("3col", 0.22 * nrow + 0.10), squeeze=False)
    fig.subplots_adjust(left=0.07, right=0.66, top=1 - 0.05 / (0.22 * nrow + 0.10), bottom=0.06, hspace=0.85)
    palette = S.OKABE_ITO_LIST[:7] + [RL.GROUP_PALETTE[3]]
    for k, (g, order, d, tops) in enumerate(panels):
        ax = axes.flat[k]
        handles = []
        for i, v in enumerate(tops):
            lk = RL.Look(name=v, group=g, round=None, color=palette[i % len(palette)], shape=i % len(RL.SHAPES),
                         dash=i // len(palette))
            y = [float(d.filter((pl.col("vector") == v) & (pl.col("sample") == n))["rpm"].sum()) for n in order]
            st = lk.mpl(lw=0.8, ms=2.6, markers=True)
            st["label"] = textwrap.shorten(v.replace("UMI1_anchor|", "").replace("|UMI2_anchor", ""), 46, placeholder="…")
            handles.append(ax.plot(range(len(order)), np.maximum(y, 0.5), **st)[0])
        ax.set_yscale("log")
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_xticks(range(len(order)), order, fontsize=5)
        ax.set_ylabel("Reads per million")
        ax.set_title(f"{g}: top {len(tops)} vectors at {order[-1]}", fontsize=6)
        ax.legend(handles=handles, fontsize=5, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
        label_panel(ax, chr(65 + k))
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    message(fig, ctx, "Fragvec trajectories: the vectors that dominate each group's last round, traced back to R0")
    return fig, {p: f"{g}: depth-normalised frequency (RPM) of the {TOP_VECTORS} most frequent vectors at the "
                    f"group's last round, in every round of that group; anchors dropped from the labels; "
                    f"zero drawn at 0.5 RPM" for p, (g, *_rest) in zip("ABCDEFGHIJKL", panels)}


# ================================================================================ sharing (UpSet)

def upset_panel(fig, cell, ctx, group, sets, top_n=15):
    """One selection's UpSet: exclusive intersections (report_lib.intersections -- the same counts
    as the HTML report) as log bars over a dot matrix; each row labelled with its set size."""
    names, color = list(sets), ctx.G.color(group)
    inter = RL.intersections(sets)[:top_n]
    sub = cell.subgridspec(2, 1, height_ratios=[1.25, 1], hspace=0.06)
    ax, mx = fig.add_subplot(sub[0]), fig.add_subplot(sub[1], sharex=None)
    x = np.arange(len(inter))
    ax.bar(x, [c for _, c in inter], width=0.66, color=color)
    for xi, (_, c) in zip(x, inter):
        ax.text(xi, c, f"{c:,}", ha="center", va="bottom", fontsize=4.2, rotation=90)
    ax.set_yscale("log")
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlim(-0.6, max(len(inter), 1) - 0.4)
    ax.set_xticks([])
    ax.set_ylabel("Shared (log)")
    ax.set_title(group, fontsize=6.5, color=color, fontweight="bold")
    for xi, (mem, _) in zip(x, inter):
        ys = [names.index(m) for m in mem]
        mx.scatter([xi] * len(names), range(len(names)), s=7, color="#e3e3e3", linewidths=0)
        mx.plot([xi] * len(ys), ys, color=color, lw=0.9, marker="o", ms=2.6)
    mx.set_xlim(ax.get_xlim())
    mx.set_yticks(range(len(names)), [f"{n} ({len(set(sets[n])):,})" for n in names], fontsize=4.8)
    mx.set_ylim(len(names) - 0.5, -0.5)
    mx.set_xticks([])
    for side in ("top", "right", "bottom"):
        mx.spines[side].set_visible(False)
    return [(group, mem, c) for mem, c in inter]


def sharing_figure(ctx, sets_of, message_text, panels):
    """One UpSet panel per selection (R0 + its rounds), small multiples, and the counts table."""
    base = [s["name"] for s in ctx.samples if s["group"] == RL.BASELINE_GROUP]
    groups = ctx.G.groups
    cols = 2 if len(groups) > 1 else 1
    nrow = math.ceil(len(groups) / cols)
    fig = plt.figure(figsize=S.figsize("3col", 0.36 * nrow + 0.06))
    gs = fig.add_gridspec(nrow, cols, hspace=0.45, wspace=0.42, left=0.11, right=0.98,
                          top=1 - 0.06 / (0.36 * nrow + 0.06), bottom=0.03)
    rows = []
    for k, g in enumerate(groups):
        names = base + [s["name"] for s in ctx.samples if s["group"] == g]
        sets = {n: sets_of(n) for n in names}
        for grp, mem, c in upset_panel(fig, gs[k // cols, k % cols], ctx, g, sets):
            rows.append({"group": grp, "shared_by": "|".join(mem), "n_samples": len(mem), "n": c})
        rows += [{"group": g, "shared_by": f"set size: {n}", "n_samples": 0, "n": len(set(sets[n]))} for n in names]
    ctx.write(pl.DataFrame(rows))
    message(fig, ctx, message_text)
    return fig, panels


def fig_umi_sharing(ctx):
    species = {s["name"]: s["umi_species"] for s in ctx.samples}

    def sets_of(n):
        sp = pl.read_csv(species[n], separator="\t")
        return sp[sp.columns[0]].to_list()

    return sharing_figure(ctx, sets_of, "UMI sharing: corrected UMIs shared across the rounds of each selection", {
        "panels": "one per selection: rows = the R0 library, then the selection's rounds; bars = the 15 largest "
                  "exclusive intersections (each UMI counted once, under exactly the samples it occurs in), log "
                  "scale; row labels = each sample's distinct corrected UMIs. Shared counts grow with the depth of "
                  "every sample involved -- read them against the set sizes"})


def fig_fragvec_sharing(ctx):
    counts = {s["name"]: s["fragvec_counts"] for s in ctx.samples}
    mr = ctx.a.min_reads

    def sets_of(n):
        return (pl.read_parquet(counts[n], columns=["vector", "n_reads"])
                .filter(pl.col("n_reads") >= mr)["vector"].to_list())

    return sharing_figure(ctx, sets_of, "Fragvec sharing: vectors shared across the rounds of each selection", {
        "panels": f"one per selection: rows = the R0 library, then the selection's rounds; a vector counts as "
                  f"present with ≥ {mr} reads (the noise cut); bars = the 15 largest exclusive intersections, "
                  f"log scale; row labels = each sample's vectors with ≥ {mr} reads"})


# ===================================================================================== driver

#: The series, in the order the analysis runs. A figure's number and file stem come from HERE.
FIGURES = [("raw_sequencing", fig_raw), ("read_funnel", fig_funnel), ("decode_qc", fig_decode),
           ("raw_fragment_calls", fig_raw_fragments), ("umi_qc", fig_umi_qc),
           ("fragment_quant", fig_fragment_quant), ("umi_complexity", fig_umi_complexity),
           ("umi_abundance", fig_umi_abundance), ("umi_sharing", fig_umi_sharing),
           ("fragvec_quant", fig_fragvec), ("fragvec_trajectories", fig_trajectories),
           ("fragvec_sharing", fig_fragvec_sharing)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sheet", required=True, help="JSON list of samples, config order, with their files")
    ap.add_argument("--clean", required=True)
    ap.add_argument("--preprocess", required=True)
    ap.add_argument("--covfilt", default="", help="covfilt/coverage_summary.tsv; empty if the filter is off")
    ap.add_argument("--complexity", required=True)
    ap.add_argument("--collapse", required=True)
    ap.add_argument("--jackpot-summary", default="", help="jackpot/<R0>.umi_blacklist_summary.tsv; empty if off")
    ap.add_argument("--qc-panel", default="", help="static_tables' ont_qc_panel.tsv (sampled error decomposition)")
    ap.add_argument("--expected", required=True)
    ap.add_argument("--minlen", type=int, required=True)
    ap.add_argument("--min-coverage", type=float, default=None)
    ap.add_argument("--min-reads", type=int, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    a = ap.parse_args(argv)
    a.outdir.mkdir(parents=True, exist_ok=True)
    plt.style.use(str(S.STYLE))
    ctx = Ctx(a)
    legends, failures = [f"# Analysis figures — {len(ctx.samples)} samples", ""], []
    with PdfPages(a.outdir / "analysis_figures.pdf") as book:
        for number, (name, build) in enumerate(FIGURES, start=1):
            stem = f"{number:02d}_{name}"
            ctx.number, ctx.stem = number, stem
            try:
                fig, panels = build(ctx)
            except Exception as error:                       # one broken step must not hide the rest
                failures.append(f"{stem}: {type(error).__name__}: {error}")
                print(f"[WARN] {stem}: {error}", flush=True)
                plt.close("all")
                continue
            if fig is None:
                continue
            fig.savefig(a.outdir / f"{stem}.pdf")
            book.savefig(fig)
            legends += [f"## {fig.get_suptitle()}", "", f"`{stem}.pdf` · data: `tables/{stem}.tsv`", "",
                        *[f"- **{k}** — {v}" for k, v in panels.items()], ""]
            plt.close(fig)
            print(f"  {stem}.pdf", flush=True)
    if failures:
        legends += ["## Not drawn", "", *[f"- {f}" for f in failures], ""]
    (a.outdir / "figure_legends.md").write_text("\n".join(legends) + "\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
