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

HARD INVARIANT -- THIS MODULE MUST STAY STDLIB-ONLY. Because of (1) it is now imported at
WORKFLOW-PARSE TIME by `common.smk`, i.e. under the `snakemake` pixi env, which has NO plotly and NO
polars. Add `import plotly` (or polars, numpy, ...) at its top and THE ENTIRE WORKFLOW STOPS PARSING
-- not one rule, all of it, before any job is scheduled. `fig_html` only calls a method on the figure
it is handed; keep it that way. The corollary benefit: the palette self-test at the bottom runs under
plain `python3 report_lib.py`.

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
GROUP_PALETTE = [
    "#0072B2",   # blue        <- A549 in the current config
    "#009E73",   # green       <- MDA
    "#D55E00",   # vermillion
    "#8250df",   # purple
    "#a86c3f",   # brown
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
    # still contains the baseline, and it must not shift the others by one.
    for i, g in enumerate([g for g in groups if g != BASELINE_GROUP]):
        out[g] = GROUP_PALETTE[i % len(GROUP_PALETTE)]
    return out


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
    print("report_lib self-test OK" if good else "report_lib self-test: PALETTE BELOW FLOOR")
    raise SystemExit(0 if good else 1)
