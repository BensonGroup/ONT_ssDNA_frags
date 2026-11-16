#!/usr/bin/env python
"""The shared GROUP VOCABULARY, and the presentation layer for the three cross-sample reports.

This module is TWO things, and the second one is newer and easy to miss:

1. THE VOCABULARY. `BASELINE_GROUP` / `EMPTY_GROUP` are the two reserved group names, and they are
   the ONLY definition of them anywhere. The `group` field of a config sample decides everything --
   which tree a sample is decoded into, whether it forms a group matrix, what its collapse row is
   called (there is no `role` field; it was removed). `common.smk` imports these two names FROM
   HERE, so they are workflow vocabulary, not just report vocabulary.

2. THE PRESENTATION LAYER. The things that must AGREE across `build_umi_report_cross.py`,
   `build_fragvec_report_cross.py` and `build_featurecount_report_cross.py` -- html escaping, figure
   embedding, and the GROUP layer: a group is the organising dimension of every diagram, and a group
   must be the same colour in every report.

HARD INVARIANT -- THIS MODULE MUST STAY STDLIB-ONLY AT IMPORT. Because of (1) it is now imported at
WORKFLOW-PARSE TIME by `common.smk`, i.e. under the `snakemake` pixi env, which has NO plotly and NO
polars. Add `import plotly` (or polars, numpy, ...) at its top and THE ENTIRE WORKFLOW STOPS PARSING
-- not one rule, all of it, before any job is scheduled. `fig_html` only calls a method on the figure
it is handed, and `PlotStyle.curves` -- the one method that builds a figure -- imports plotly inside
its body, where only a report ever reaches it. Keep both that way. The corollary benefit: the
palette self-test at the bottom runs under plain `python3 report_lib.py`.

WHAT DOES NOT LIVE HERE: `style()` and the CSS. They only look duplicated. The featurecount report
uses transparent backgrounds, different margins, its own axis colours, and an external CSS kit with
a theme toggle; the other two use `plotly_white` and an inline `_STYLE`. Unifying them would
silently restyle two reports for no benefit. Leave each report's own look alone.
"""
from __future__ import annotations

import html
from pathlib import Path

# --------------------------------------------------------------------------------------- palette
# Colour-blind-safe qualitative palette for sample GROUPS (cell lines), from Okabe-Ito. Every
# pairwise separation is CHECKED, not asserted -- run `python3 report_lib.py`, which reports the
# minimum pairwise dE under normal / deuteranopic / protanopic vision (all >= 12) and the minimum
# contrast against BOTH report backgrounds (all >= 3, so no colour depends on the relief rule).
#
# Constraints that shaped it, in case you are tempted to swap a colour in: the baseline grey is in
# the pairwise set too (a group must not read as the baseline), and a bright green like the scratch
# figure's #1baf7a drops to 2.8:1 on white -- it needs direct labels to survive, which a shared
# palette cannot guarantee. #009E73 is the darkest green that clears 3:1 on white while staying
# clear of the blue under deuteranopia.
#
# NOT drawn from ligand_vocab.PALETTE on purpose: that list already does double duty in the
# featurecount report for fragment CLASSES, so reusing it for groups would make the same hue mean
# two different things on one page.
#
# The palette is NEVER cycled: a colour reused for a second group would make two arms read as one.
# The last three slots were chosen by searching hue families (magenta / sky / maroon) for the set
# whose WORST pair, over all 9 colours including the baseline grey, stays furthest apart under all
# three vision models: min pairwise dE 16.0, every slot >= 3:1 on both report backgrounds.
# A project with MORE groups is valid and is not rejected: group_color() extends the palette with
# extra_colors(), each new colour the candidate furthest from every colour already in use. Those
# extras are distinct but no longer guaranteed CVD-safe, so the legend names every group.
GROUP_PALETTE = [
    "#0072B2",   # blue        <- A549 in the current config
    "#009E73",   # green       <- MDA
    "#D55E00",   # vermillion
    "#8250df",   # purple
    "#a86c3f",   # brown
    "#df0cb5",   # magenta
    "#4c94e6",   # sky blue
    "#aa4148",   # maroon
]
# The R0 / initial library: a shared baseline, NOT a selection arm. Deliberately recessive, but it
# still has to be READABLE -- #7d8590 is the grey that clears 3:1 against both the light and the dark
# report background (the old #9aa0a6 was 2.6:1 on white, i.e. below the floor on the default theme).
BASELINE_GREY = "#7d8590"
GREY = "#7d8590"            # neutral marks (expected-by-chance bars, reference lines)

# ------------------------------------------------------------------------- the reserved group names
# The two RESERVED values of a sample's config `group` (config/config.yaml). `common.smk` imports
# both from here and validates them at parse time -- exactly one sample may be `initial`, and a
# group name that is not in the declared `groups:` list RAISES. They are the whole sample model:
# there is no `role` field, the group decides everything.
#
# `initial` -- the R0 library. A shared baseline, not a selection arm: it is decoded into its own
#     `*_initial/` tree and it never consumes a palette slot (see group_color()).
# `empty`   -- the PCR-only / no-cell rounds. A PEER of every experimental arm in every respect: its
#     own group matrix, its own palette slot, its own C_eff trajectory line, its own UpSet panel. It
#     is special in exactly ONE place -- the collapse table, where its collapse vs R0 is labelled
#     `protocol` rather than `total`, because with no cells it measures what the plumbing cost. If
#     you find yourself writing `if group == EMPTY_GROUP` inside a figure function, the design has
#     failed.
BASELINE_GROUP = "initial"
EMPTY_GROUP = "empty"


# ----------------------------------------------------------------------------------- html helpers
def esc(s) -> str:
    return html.escape(str(s))


def fig_html(fig) -> str:
    """Embed a plotly figure as a div. plotly.js is inlined ONCE per page by the caller, so every
    figure references the already-loaded global -- order-independent, safe to reorder sections."""
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       default_width="100%", config={"displaylogo": False, "responsive": True})


# ---------------------------------------------------------------------------------------- metadata
def _int(x, default=0):
    try:
        return int(str(x).strip())
    except (TypeError, ValueError):
        return default


def load_meta(path):
    """The config-derived sample table -> {name: {group, round, idx}}.

    Schema (written by `rule cross_sample_metadata`, reports.smk):

        name <TAB> group <TAB> round <TAB> idx

    There is NO `role` column: the group decides everything (see the reserved names above).

    APPEND-ONLY. Never insert a column before `idx`: `idx` is the config-order sort key, and a
    shifted parse degrades it to a constant, which collapses the sort so the cell lines interleave
    by round instead of forming the contiguous blocks group_spans() needs. New columns go on the end,
    where a short-field parse leaves them empty and everything else still lines up.

    A STALE 5-column file (the old `... <TAB> idx <TAB> role`) still parses, and its trailing `role`
    is DROPPED by the [:4] slice -- narrowing is safe precisely because `role` was the last column,
    so nothing before it shifts. No migration shim, no inference: role is gone, not renamed.
    Returns {} when the file is absent -- every caller has a name-parsing fallback.
    """
    meta = {}
    if not path or not Path(path).exists():
        return meta
    for i, line in enumerate(Path(path).read_text().splitlines()):
        if i == 0 or not line.strip():           # header / blanks
            continue
        f = [x.strip() for x in (line.split("\t") + [""] * 4)[:4]]
        name, group, rnd, idx = f
        meta[name] = {
            "group": group or name,
            "round": _int(rnd, 0),
            "idx": _int(idx, i),
        }
    return meta


# ------------------------------------------------------------------------------------- group layer
def group_order(samples, key="group"):
    """The distinct non-baseline groups, in CONFIG order.

    Accepts either a discovered sample list (list of dicts, already sorted by config `idx`) or a
    load_meta() dict. `key` is the field the group is carried on -- "group" in the umi/fragvec
    reports, "line" in the featurecount one.
    """
    if isinstance(samples, dict):
        items = sorted(samples.values(), key=lambda m: (m.get("idx", 0), m.get("round", 0)))
        key = "group"
    else:
        items = samples
    # The baseline is dropped by GROUP NAME ALONE, and that is now the only filter needed.
    #
    # WHY THIS IS SOUND -- read before "simplifying" the reservation away. `initial` is a RESERVED
    # group name, validated at parse time by common.smk: exactly one sample may carry it, and an
    # undeclared group raises. So `group == BASELINE_GROUP` <=> "this is the R0 baseline", for every
    # caller: a load_meta() dict (group comes from the validated config) and a discovered sample list
    # alike (both cross reports inject their baseline row with the literal group/line "initial").
    #
    # It was NOT sound before this refactor, and that is the trap: `group` was then a free-form
    # string, so the ROLE filter was the real guard and this line was only a backstop. Un-reserve
    # `initial` -- let it become just another group name -- and the baseline silently reappears as a
    # selection arm here, stealing palette slot 0 and shifting every cell line's colour by one.
    return [g for g in dict.fromkeys(s.get(key) for s in items) if g and g != BASELINE_GROUP]


def group_color(groups, key="group"):
    """group -> hex colour.

    The baseline group ALWAYS maps to BASELINE_GREY and NEVER consumes a palette slot. That is the
    whole point: the R0 library is injected into some reports and not others (config
    report.r0_baseline), and if it took slot 0 then enabling the baseline would shift every cell
    line's colour by one. Excluding it keeps a cell line's colour stable across all three reports
    and across that toggle.

    `groups` may be an ordered list of group names, a discovered sample list, or a load_meta() dict.
    """
    if not (isinstance(groups, list) and groups and isinstance(groups[0], str)):
        groups = group_order(groups, key=key)
    out = {BASELINE_GROUP: BASELINE_GREY}
    # filter here too, not only in group_order(): callers may hand us a plain list of names that
    # still contains the baseline, and it must not shift the others by one. Deduplicated, so a list
    # that repeats a group cannot hand one group two slots.
    arms = [g for g in dict.fromkeys(groups) if g and g != BASELINE_GROUP]
    extra = len(arms) - len(GROUP_PALETTE)
    if extra > 0:
        _warn_once(f"[report_lib] {len(arms)} groups exceed the {len(GROUP_PALETTE)} colour-blind-"
                   f"checked group colours; {', '.join(arms[len(GROUP_PALETTE):])} get generated "
                   f"colours (distinct, not CVD-checked). Every group also has its own marker shape, "
                   f"and line figures split into one panel per group.")
    palette = GROUP_PALETTE + extra_colors(extra)
    for i, g in enumerate(arms):
        out[g] = palette[i]
    return out


_WARNED: set = set()


def _warn_once(msg):
    if msg not in _WARNED:
        _WARNED.add(msg)
        import sys
        print(msg, file=sys.stderr)


_EXTRA_CACHE: list = []


def extra_colors(n):
    """`n` more group colours, none equal or close to any colour already in use.

    Deterministic greedy farthest-point choice over an HLS grid: each new colour is the candidate
    (>= 3:1 on both report backgrounds) whose smallest dE to everything already chosen -- the
    palette, the baseline grey, and the extras before it -- is largest, taking the worst of normal,
    deuteranopic and protanopic vision. Only reached by projects with more groups than
    GROUP_PALETTE has slots."""
    if n <= 0:
        return []
    import colorsys
    while len(_EXTRA_CACHE) < n:
        used = GROUP_PALETTE + [BASELINE_GREY] + _EXTRA_CACHE
        best, best_d = None, -1.0
        for h in range(0, 360, 5):
            for light in (0.35, 0.42, 0.50, 0.58):
                for sat in (0.55, 0.75, 0.95):
                    r, g, b = colorsys.hls_to_rgb(h / 360, light, sat)
                    c = "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))
                    if c in used or min(contrast(c, LIGHT_SURFACE), contrast(c, DARK_SURFACE)) < 3:
                        continue
                    d = min(delta_e(c, u, m) for u in used for m in (_NORMAL, _DEUTAN, _PROTAN))
                    if d > best_d:
                        best, best_d = c, d
        _EXTRA_CACHE.append(best)
    return _EXTRA_CACHE[:n]


def intersections(sets):
    """UpSet counts: every element counted ONCE, under the exact combination of sets it occurs in.

    `sets` = {name: iterable of ids}. Returns [(member_names, n)] largest first (ties: by which sets,
    in the order of `sets`), members in the order of `sets`. The single definition behind every UpSet
    view, interactive and static."""
    names, mask = list(sets), {}
    for i, n in enumerate(names):
        bit = 1 << i
        for x in sets[n]:
            mask[x] = mask.get(x, 0) | bit
    counts = {}
    for m in mask.values():
        counts[m] = counts.get(m, 0) + 1
    order = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(tuple(names[j] for j in range(len(names)) if m >> j & 1), c) for m, c in order]


# ------------------------------------------------------------------------------------ plot style
# ONE object decides how every sample is drawn, on the page (plotly) and in print (matplotlib):
#
#   group  -> COLOUR (GROUP_PALETTE, extended past its checked slots) + marker SHAPE
#   sample -> its group's colour and shape + its own LINE DASH
#
# A sample's dash is its POSITION in its group in config order, not its round number. Keying on the
# round (the old `(round - 1) % 5`) gave two replicates of one round, or rounds 1 and 6, the same
# (colour, dash). Position-in-group is unique by construction, so (colour, dash) names one sample.
#
# Deliberately NOT spent on identity: marker size and line width (they read as magnitude), the
# open/filled variant (hollow already means "lower bound" / "below min_reads" in several panels), and
# bar hatching (it already means "raw, depth-dependent" next to a solid rarefied bar).
#
# Shapes: every plotly symbol with an exact matplotlib twin, index for index, filled shapes first.
SHAPES = [("circle", "o"), ("square", "s"), ("diamond", "D"), ("triangle-up", "^"),
          ("triangle-down", "v"), ("cross", "P"), ("x", "X"), ("triangle-left", "<"),
          ("triangle-right", ">"), ("pentagon", "p"), ("hexagon", "h"), ("star", "*"),
          ("octagon", "8"), ("hexagon2", "H"), ("diamond-tall", "d"),
          ("y-down", "1"), ("y-up", "2"), ("y-left", "3"), ("y-right", "4"),
          ("cross-thin", "+"), ("x-thin", "x"), ("line-ns", "|"), ("line-ew", "_")]
# Dashes: on/off lengths in px. Past this list they are generated and never repeat (see Look.dashes).
DASHES = [(), (6, 3), (10, 3), (6, 2.5, 1.5, 2.5), (2, 2.5), (10, 2.5, 1.5, 2.5), (3, 3),
          (6, 2.5, 1.5, 2.5, 1.5, 2.5), (14, 3), (10, 2.5, 3, 2.5)]
BASELINE_DASH = 4                    # the R0 library is dotted -- and the only grey line
MPL_DASH_SCALE = 0.55                # px on screen -> points at print line widths


class Look:
    """How ONE sample is drawn. `trace()` / `hover()` feed plotly, `mpl()` feeds matplotlib.

    A plain class, NOT a @dataclass: common.smk loads this module with importlib WITHOUT registering
    it in sys.modules, and dataclasses look the module up there -- the workflow would stop parsing."""
    __slots__ = ("name", "group", "round", "color", "shape", "dash")

    def __init__(self, name: str, group: str, round: int | None, color: str, shape: int, dash: int):
        self.name, self.group, self.round = name, group, round
        self.color, self.shape, self.dash = color, shape, dash

    @property
    def dashes(self) -> tuple:
        if self.dash < len(DASHES):
            return DASHES[self.dash]
        k = self.dash - len(DASHES)        # a run of dots after a stepped dash, 3.5px gaps
        return (4 + 2 * (k % 6), 3.5) + (1.5, 3.5) * (1 + k // 6)

    @property
    def label(self) -> str:
        """The FULL identity, for hovers: the sample name, then its group and round."""
        tail = " · ".join(x for x in (self.group, "" if self.round is None else f"R{self.round}") if x)
        return f"<b>{esc(self.name)}</b>" + (f"<br>{esc(tail)}" if tail else "")

    # ------------------------------------------------------------------ plotly
    def trace(self, mode="lines", width=1.6, size=7, **marker) -> dict:
        """Everything a per-sample trace needs: its own legend entry under its group's title, its
        line, and (for a marker mode) its group's shape. `legendgroup` links a group's entries, so a
        click on the group title toggles the whole arm."""
        kw = dict(name=esc(self.name), legendgroup=self.group,
                  legendgrouptitle_text=esc(self.group), mode=mode)
        if "lines" in mode:
            kw["line"] = dict(color=self.color, width=width,
                              dash=",".join(f"{v:g}px" for v in self.dashes) or "solid")
        if "markers" in mode:
            kw["marker"] = dict(color=self.color, size=size, symbol=SHAPES[self.shape][0], **marker)
        return kw

    def hover(self, body: str) -> str:
        """A hovertemplate that always leads with the full sample label (never cut short)."""
        return f"{self.label}<br>{body}<extra></extra>"

    # ------------------------------------------------------------------ matplotlib
    def mpl(self, lw=0.9, ms=3.4, markers=False) -> dict:
        ls = "-" if not self.dashes else (0, tuple(round(MPL_DASH_SCALE * v, 2) for v in self.dashes))
        kw = dict(color=self.color, ls=ls, lw=lw, label=self.name)
        if markers:
            kw.update(marker=SHAPES[self.shape][1], ms=ms, mec=self.color)
        return kw


class PlotStyle:
    """The project's visual encoding, built once from the CONFIG sample table.

    `samples` is a load_meta() dict or a list of sample dicts (`name`, `round`, the group under
    `key`). Build it from load_meta() when you have it: positions are then taken over the DECLARED
    samples, so a sample missing from one report cannot shift its group-mates' dashes there.

        style = PlotStyle(meta)
        fig.add_trace(go.Scatter(x=..., y=..., **style[s].trace(), hovertemplate=style[s].hover("...")))
        ax.plot(x, y, **style[s].mpl())
    """

    MAX_LINES = 24       # more curves than this in one panel -> one panel per group

    def __init__(self, samples, key="group"):
        if isinstance(samples, dict):
            rows = sorted(({"name": n, **m} for n, m in samples.items()),
                          key=lambda m: (m.get("idx", 0), m.get("round") or 0))
            key = "group"
        else:
            rows = list(samples)
        self.key = key
        self.groups = group_order(rows, key=key)                 # arms, config order
        self.colors = group_color(self.groups)
        self.shapes = {BASELINE_GROUP: 0, **{g: i % len(SHAPES) for i, g in enumerate(self.groups)}}
        seen, self.looks = {}, {}
        for s in rows:
            g = str(s.get(key) or "")
            pos = seen[g] = seen.get(g, -1) + 1
            self.looks[s["name"]] = Look(
                name=s["name"], group=g, round=s.get("round"), color=self.colors.get(g, GREY),
                shape=self.shapes.get(g, 0), dash=BASELINE_DASH if g == BASELINE_GROUP else pos)

    def __getitem__(self, sample) -> Look:
        """A Look by sample name or sample dict."""
        return self.looks[sample if isinstance(sample, str) else sample["name"]]

    def __contains__(self, sample) -> bool:
        return (sample if isinstance(sample, str) else sample["name"]) in self.looks

    def color(self, group) -> str:
        return self.colors.get(group, GREY)

    def marker(self, group) -> str:
        """The group's matplotlib marker -- the twin of its plotly symbol."""
        return SHAPES[self.shapes.get(group, 0)][1]

    def symbol(self, group) -> str:
        return SHAPES[self.shapes.get(group, 0)][0]

    @classmethod
    def of(cls, meta, samples, key="group"):
        """From the config table when it covers every discovered sample, else from the discovered
        list itself (a standalone run with no metadata TSV)."""
        if meta and all(s["name"] in meta for s in samples):
            return cls(meta)
        return cls(samples, key=key)

    def panels(self, samples) -> list:
        """[(title, samples)]: ONE panel while it stays readable -- every group on a checked colour
        and <= MAX_LINES curves -- else one panel per group, each also carrying the R0 baseline."""
        samples = list(samples)
        groups = [self[s].group for s in samples]
        arms = [g for g in dict.fromkeys(groups) if g != BASELINE_GROUP]
        if len(arms) <= len(GROUP_PALETTE) and len(samples) <= self.MAX_LINES:
            return [("", samples)]
        base = [s for s, g in zip(samples, groups) if g == BASELINE_GROUP]
        return [(a, base + [s for s, g in zip(samples, groups) if g == a]) for a in arms]

    def curves(self, samples, curve, mode="lines", width=1.6, size=7, height=460, gl=False, **marker):
        """A plotly figure with one trace per sample, each drawn with its Look and its full label on
        hover, split into one panel per group (2 columns) when one panel would be unreadable.

        `curve(sample)` returns (x, y, hover_body) -- hover_body may use %{x}/%{y} -- or None to
        leave the sample out. `gl=True` draws WebGL scatter, for point clouds; extra keywords go to
        every marker (e.g. opacity). The ONLY method here that builds a figure, so plotly is imported
        lazily: this module must stay importable at workflow-parse time (see the docstring)."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        panels = self.panels(samples)
        cols = 1 if len(panels) == 1 else 2
        rows = -(-len(panels) // cols)
        fig = make_subplots(rows=rows, cols=cols, shared_xaxes=True, shared_yaxes=True,
                            horizontal_spacing=0.06, vertical_spacing=min(0.12, 0.5 / rows),
                            subplot_titles=[esc(t) for t, _ in panels] if cols > 1 else None)
        shown = set()
        for i, (_, members) in enumerate(panels):
            for s in members:
                got = curve(s)
                if not got:
                    continue
                lk = self[s]
                trace = go.Scattergl if gl else go.Scatter
                fig.add_trace(trace(x=got[0], y=got[1], **lk.trace(mode, width, size, **marker),
                                    hovertemplate=lk.hover(got[2]),
                                    showlegend=lk.name not in shown),
                              row=i // cols + 1, col=i % cols + 1)
                shown.add(lk.name)
        fig.update_layout(height=height if rows == 1 else max(height, 300 * rows))
        return self.layout(fig)

    def upset(self, sets, group, top_n=24, height=460):
        """UpSet view of ONE selection: `sets` = {sample: ids}, rows in the given order (R0 first, then
        the rounds). Bars = the top_n EXCLUSIVE intersections (intersections() -- the same counts the
        static figure draws); the dot matrix names their members; each row label carries its set size,
        because a shared count only means something next to the sizes it was drawn from."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        names, color = list(sets), self.color(group)
        sizes = {n: len(set(v)) for n, v in sets.items()}
        inter = intersections(sets)[:top_n]
        xs, rows = list(range(len(inter))), list(range(len(names)))
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.04)
        fig.add_trace(go.Bar(x=xs, y=[c for _, c in inter], marker_color=color,
                             customdata=["<br>".join(esc(m) for m in mem) for mem, _ in inter],
                             hovertemplate="%{y:,} shared by exactly:<br>%{customdata}<extra></extra>"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=[x for x in xs for _ in rows], y=rows * len(xs), mode="markers",
                                 marker=dict(color="#e6e6e6", size=9), hoverinfo="skip"), row=2, col=1)
        for x, (mem, _) in zip(xs, inter):
            fig.add_trace(go.Scatter(x=[x] * len(mem), y=[names.index(m) for m in mem], mode="markers+lines",
                                     marker=dict(color=color, size=10), line=dict(color=color, width=2),
                                     hoverinfo="skip"), row=2, col=1)
        fig.update_yaxes(title_text="shared", row=1, col=1)
        fig.update_yaxes(tickmode="array", tickvals=rows, autorange="reversed", row=2, col=1,
                         ticktext=[f"{esc(n)} ({sizes[n]:,})" for n in names])
        fig.update_xaxes(showticklabels=False, row=2, col=1)
        fig.update_layout(showlegend=False, height=height)
        return self.layout(fig, legend_right=False)

    @staticmethod
    def layout(fig, legend_right=True):
        """Report-wide plotly settings: hover boxes never truncate a name (plotly cuts at 15
        characters by default), and the per-sample legend sits right of the plot, grouped."""
        fig.update_layout(hoverlabel=dict(namelength=-1))
        if legend_right:
            fig.update_layout(legend=dict(orientation="v", x=1.02, xanchor="left", y=1,
                                          yanchor="top", groupclick="toggleitem",
                                          tracegroupgap=6, font=dict(size=11)))
        return fig


def group_spans(samples, key="group"):
    """Contiguous (group, first_index, last_index) runs over the ordered samples -- used to place
    one group label centred above each block of that group's columns."""
    spans = []
    for i, s in enumerate(samples):
        g = s.get(key)
        if spans and spans[-1][0] == g:
            spans[-1][2] = i
        else:
            spans.append([g, i, i])
    return [(g, a, b) for g, a, b in spans]


def group_dividers(samples, key="group"):
    """x positions where the group changes (for a vertical separator between blocks)."""
    gs = [s.get(key) for s in samples]
    return [i - 0.5 for i in range(1, len(gs)) if gs[i] != gs[i - 1]]


def group_band(fig, samples, yshift, key="group"):
    """One group name centred above each contiguous block of that group's columns.

    Rotated -45°, the same reading direction as the column tick labels below it. A block can be a
    single column wide, and at that pitch horizontal names overlap: clearing them needs pitch >
    label width (~40px), whereas rotated ones only need pitch·sin45 > label height (~14px).
    Rotation makes a label ~0.7·(width + height) tall, so every caller reserves extra top margin.
    """
    for g, a, b in group_spans(samples, key=key):
        fig.add_annotation(x=(a + b) / 2, y=1.0, xref="x", yref="paper", yshift=yshift,
                           xanchor="center", yanchor="bottom", textangle=-45,
                           showarrow=False, text=f"<b>{esc(g)}</b>", font=dict(size=11))


def first_of_group(samples, key="group"):
    """[bool] -- True at each sample that is the first of its group in the ordered list.

    Use it as `showlegend=`. `legendgroup=` alone does NOT collapse 11 per-sample legend entries
    into 3 per-group ones: it links their toggling, but every trace still gets its own entry. One
    visible entry per group, with the rounds inside it encoded by dash/opacity, is the goal.
    """
    seen, out = set(), []
    for s in samples:
        g = s.get(key)
        out.append(g not in seen)
        seen.add(g)
    return out


def color_key_html(gcol, samples, key="group"):
    """A one-line colour key naming each group's colour, with its member samples on hover.

    Inline styles, not a CSS class: the three reports have three different stylesheets (one of them
    external), and this has to look the same in all of them.
    """
    chips = []
    for g in dict.fromkeys(s.get(key) for s in samples if s.get(key)):
        members = ", ".join(s["name"] for s in samples if s.get(key) == g)
        base = " (baseline)" if g == BASELINE_GROUP else ""
        chips.append(
            f"<span title='{esc(members)}' style='display:inline-flex;align-items:center;"
            f"gap:.35rem;margin-right:1.1rem'>"
            f"<span style='width:.8rem;height:.8rem;border-radius:2px;flex:0 0 auto;"
            f"background:{gcol.get(g, GREY)}'></span>{esc(g)}{base}</span>")
    return ("<p class='small' style='display:flex;flex-wrap:wrap;align-items:center;"
            "margin:.4rem 0 1.2rem'>" + "".join(chips) + "</p>")


# ------------------------------------------------------------- palette self-test (colorimetry)
# Ported from the scratch C_eff figure, which validated its two colours rather than trusting them.
# Kept because a palette claim that is never checked is a palette claim that quietly rots.
def _rgb(h):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]


def _lin(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _luminance(h):
    r, g, b = (_lin(c) for c in _rgb(h))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b):
    la, lb = _luminance(a), _luminance(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def _mul(m, v):
    return [sum(m[i][j] * v[j] for j in range(3)) for i in range(3)]


# linear sRGB <-> LMS (Hunt-Pointer-Estevez, as used by Viénot 1999)
_RGB2LMS = [[17.8824, 43.5161, 4.11935],
            [3.45565, 27.1554, 3.86714],
            [0.0299566, 0.184309, 1.46709]]
_LMS2RGB = [[0.0809444479, -0.1305044090, 0.1167721820],
            [-0.0102485335, 0.0540193266, -0.1136147180],
            [-0.0003652968, -0.0041216147, 0.6935112680]]
_DEUTAN = [[1.0, 0.0, 0.0], [0.494207, 0.0, 1.24827], [0.0, 0.0, 1.0]]
_PROTAN = [[0.0, 2.02344, -2.52581], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
_NORMAL = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def simulate(h, m):
    lms = _mul(_RGB2LMS, [_lin(c) for c in _rgb(h)])
    rgb = _mul(_LMS2RGB, _mul(m, lms))
    return [min(1.0, max(0.0, c)) for c in rgb]


def _lab(lin_rgb):
    """linear sRGB -> CIE Lab (D65)."""
    M = [[0.4124564, 0.3575761, 0.1804375],
         [0.2126729, 0.7151522, 0.0721750],
         [0.0193339, 0.1191920, 0.9503041]]
    x, y, z = _mul(M, lin_rgb)
    xn, yn, zn = 0.95047, 1.0, 1.08883

    def f(t):
        return t ** (1 / 3) if t > 216 / 24389 else (841 / 108) * t + 4 / 29

    fx, fy, fz = f(x / xn), f(y / yn), f(z / zn)
    return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)]


def delta_e(h1, h2, m):
    l1, l2 = _lab(simulate(h1, m)), _lab(simulate(h2, m))
    return sum((a - b) ** 2 for a, b in zip(l1, l2)) ** 0.5


LIGHT_SURFACE, DARK_SURFACE = "#ffffff", "#15181c"   # the two report backgrounds
DE_TARGET, DE_FLOOR = 12.0, 8.0


def validate(palette=None, verbose=True):
    """Every pair of group colours must stay separable under deuteranopia and protanopia, and each
    must stay visible on both report backgrounds. Returns True iff nothing is below the floor."""
    pal = list(palette or GROUP_PALETTE) + [BASELINE_GREY]
    ok = True
    for vname, m in (("normal", _NORMAL), ("deuteranopia", _DEUTAN), ("protanopia", _PROTAN)):
        worst, wpair = 1e9, None
        for i in range(len(pal)):
            for j in range(i + 1, len(pal)):
                de = delta_e(pal[i], pal[j], m)
                if de < worst:
                    worst, wpair = de, (pal[i], pal[j])
        ok &= worst >= DE_FLOOR
        verdict = "PASS" if worst >= DE_TARGET else ("FLOOR" if worst >= DE_FLOOR else "FAIL")
        if verbose:
            print(f"  min pairwise dE  {vname:<13} = {worst:6.1f}  {verdict:5}  "
                  f"(worst pair {wpair[0]} / {wpair[1]})")
    for surface, label in ((LIGHT_SURFACE, "light"), (DARK_SURFACE, "dark")):
        worst = min(contrast(h, surface) for h in pal)
        if verbose:
            v = "PASS" if worst >= 3.0 else "WARN -> direct labels required (relief rule)"
            print(f"  min contrast vs {label:<5} surface = {worst:6.2f}   {v}")
    return ok


if __name__ == "__main__":   # python3 report_lib.py
    print(f"GROUP_PALETTE validation ({len(GROUP_PALETTE)} groups + baseline)")
    good = validate()
    assert group_color(["A549", "MDA"])["A549"] == GROUP_PALETTE[0], "baseline stole a palette slot"
    assert group_color(["initial", "A549"])["A549"] == GROUP_PALETTE[0], "baseline stole a slot"
    assert group_color(["A549", "MDA"])["initial"] == BASELINE_GREY
    s = [{"group": "initial"}, {"group": "A"}, {"group": "A"}, {"group": "B"}]
    assert group_spans(s) == [("initial", 0, 0), ("A", 1, 2), ("B", 3, 3)]
    assert group_dividers(s) == [0.5, 2.5]
    assert first_of_group(s) == [True, True, False, True]
    assert group_order(s) == ["A", "B"], "group_order must drop the baseline"

    # ---- the group-only sample model (no `role` field anywhere in these dicts) ----------------
    # `empty` is a PEER, not a baseline: it DOES consume a palette slot, in declared order.
    assert group_color(["A549", "MDA", EMPTY_GROUP])[EMPTY_GROUP] == GROUP_PALETTE[2], \
        "the empty arm is a peer group -- it must take a palette slot"
    assert group_color(["A549", "MDA", EMPTY_GROUP])["A549"] == GROUP_PALETTE[0], \
        "adding the empty arm must not shift the cell lines' colours"
    # THE REGRESSION THE REFACTOR RESTS ON: the baseline is dropped and the empty arm is KEPT with
    # NO `role` key present on either dict. Put `role` back and this still passes -- which is the
    # point: nothing here may depend on it.
    assert group_order([{"group": BASELINE_GROUP}, {"group": EMPTY_GROUP}]) == [EMPTY_GROUP], \
        "group_order must drop the baseline BY GROUP NAME and keep the empty arm"
    assert load_meta(None) == {}, "a missing metadata TSV is not an error"

    # ---- loadable the way common.smk loads it: importlib, NOT registered in sys.modules -------
    import importlib.util
    _spec = importlib.util.spec_from_file_location("report_lib_parse_check", __file__)
    _spec.loader.exec_module(importlib.util.module_from_spec(_spec))

    # ---- the plot style: (colour, dash) names exactly one sample ----------------------------
    reps = [{"name": "initial", "group": BASELINE_GROUP, "round": 0}]
    reps += [{"name": f"A_R{r}_{k}", "group": "A", "round": r} for r in (1, 6) for k in (1, 2)]
    reps += [{"name": f"B_R{r}", "group": "B", "round": r} for r in range(1, 25)]
    style = PlotStyle(reps)
    pairs = [(lk.color, lk.dashes) for lk in style.looks.values()]
    assert len(set(pairs)) == len(pairs), "two samples share a (colour, dash) pair"
    assert style["initial"].color == BASELINE_GREY and style["A_R1_1"].color == GROUP_PALETTE[0]
    assert style["A_R1_1"].trace()["line"]["dash"] == "solid" != style["A_R1_2"].trace()["line"]["dash"]
    assert style.marker("A") == style["A_R6_2"].mpl(markers=True)["marker"] != style.marker("B"), \
        "marker shape is the group's second channel: shared within a group, distinct across groups"
    assert len({Look("x", "g", 1, "#000000", 0, i).dashes for i in range(200)}) == 200, "dashes repeat"
    assert len({p for p, _ in SHAPES}) == len({m for _, m in SHAPES}) == len(SHAPES), "shape twins drifted"
    assert len(style.panels(reps)) == 2, "29 curves > MAX_LINES -> one panel per group (A, B)"
    assert len(style.panels(reps[:5])) == 1
    assert intersections({"a": [1, 2, 3], "b": [2, 3, 4], "c": [3]}) == \
        [(("a",), 1), (("b",), 1), (("a", "b"), 1), (("a", "b", "c"), 1)], "exclusive intersections"
    assert sum(n for _, n in intersections({"a": "xyz", "b": "yzw"})) == 4, "each element counted once"
    wide = [{"name": f"s{i}", "group": f"g{i}", "round": 1} for i in range(len(GROUP_PALETTE) + 2)]
    assert len(PlotStyle(wide).panels(wide)) == len(wide), "past the checked palette -> one panel each"
    for n in (len(GROUP_PALETTE), len(GROUP_PALETTE) + 6):   # within, and past, the palette
        many = group_color([f"g{i}" for i in range(n)])
        assert len(set(many.values())) == n + 1, "every group (and the baseline) needs its own colour"
    assert group_color(["g0"])["g0"] == GROUP_PALETTE[0], "extras must not disturb the palette"
    print("report_lib self-test OK" if good else "report_lib self-test: PALETTE BELOW FLOOR")
    raise SystemExit(0 if good else 1)
