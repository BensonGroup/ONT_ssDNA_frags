"""
Colour palettes and Science/AAAS size helpers for the neuron figure rebuild.

Rules (see publication_output_plan.md):
  - Categorical (genotype, cluster, lineage) -> Okabe-Ito 8-colour CVD-safe set
    (Wong 2011, Nat Methods "Points of view: Color blindness"), the palette Nature's
    figure guide points authors to.
  - Sequential (expression, density, pseudotime) -> viridis / magma (perceptually
    uniform, greyscale-safe).
  - Diverging (z-score, log2FC) -> vlag / RdBu_r used symmetrically about 0. Never red-green.

Widths are the Science / AAAS columns: 5.7 / 12.1 / 18.4 cm.
"""
from __future__ import annotations

# --- Science / AAAS figure widths (cm and inches) ---
CM = 1.0 / 2.54
WIDTH_1COL_CM = 5.7
WIDTH_2COL_CM = 12.1
WIDTH_3COL_CM = 18.4
WIDTH_1COL_IN = WIDTH_1COL_CM * CM
WIDTH_2COL_IN = WIDTH_2COL_CM * CM
WIDTH_3COL_IN = WIDTH_3COL_CM * CM
MAX_TEXT_PT = 7   # never exceed
MIN_TEXT_PT = 5   # never go below
MIN_SYMBOL_PT = 6

# --- Okabe-Ito categorical (CVD-safe). Order chosen so the first two read as a clean pair. ---
OKABE_ITO = {
    "orange":       "#E69F00",
    "sky_blue":     "#56B4E9",
    "bluish_green": "#009E73",
    "yellow":       "#F0E442",
    "blue":         "#0072B2",
    "vermillion":   "#D55E00",
    "reddish_purple": "#CC79A7",
    "black":        "#000000",
}
OKABE_ITO_LIST = list(OKABE_ITO.values())

# --- Study-specific role -> colour maps (edit here, not in each figure script) ---
GENOTYPE = {          # CT = neutral blue, Cre (perturbed) = vermillion
    "CT":  OKABE_ITO["blue"],
    "Cre": OKABE_ITO["vermillion"],
}
# The 4-state annotation from notebooks/20260413.ipynb
CELLSTATE = {
    "Mature":                      OKABE_ITO["blue"],
    "Neuroblasts":                 OKABE_ITO["bluish_green"],
    "Dediff":                      OKABE_ITO["orange"],
    "Terminal Degenerative Neurons": OKABE_ITO["vermillion"],
}
# The 5-group consolidation from run_all.py Part 1
CONSOLIDATION = {
    "Mature":        OKABE_ITO["blue"],
    "Transitioning": OKABE_ITO["sky_blue"],
    "Stress":        OKABE_ITO["orange"],
    "Progenitor":    OKABE_ITO["bluish_green"],
    "Deep dediff":   OKABE_ITO["vermillion"],
}

# --- The three sequencing batches -------------------------------------------------------------
# Colour is NEVER the only channel: every figure that separates batches also separates them by
# marker, line style or hatch, so the distinction survives greyscale printing and every form of
# colour-vision deficiency. Run the static figure script with --validate to see the set simulated.
#
# b3 is reddish-purple rather than the more obvious bluish-green because green is already spoken
# for: F2 uses it for the WITHIN-batch pair kinds, which are a different axis of the same figure.
# Reusing it would make two unrelated categories share a hue in the one figure showing both.
BATCH_COLOR = {
    "b1": OKABE_ITO["blue"],            # #0072B2
    "b2": OKABE_ITO["vermillion"],      # #D55E00
    "b3": OKABE_ITO["reddish_purple"],  # #CC79A7
}
BATCH_LS = {"b1": "-", "b2": (0, (3.5, 1.5)), "b3": (0, (5, 1.2, 1, 1.2))}
BATCH_HATCH = {"b1": "", "b2": "///", "b3": "..."}

#: Marker FILL carries the batch on point figures, because marker SHAPE is spoken for by the sample
#: group (below). Matplotlib's `fillstyle` gives three states that survive greyscale, so batch keeps
#: a redundant non-colour channel even though shape no longer encodes it.
BATCH_FILL = {"b1": "full", "b2": "none", "b3": "top"}

# --- Sample group -> marker shape -------------------------------------------------------------
# SHAPE IS THE SAMPLE, NOT THE BATCH. Rounds of one group are the same material sampled at
# different selection rounds, so they share a shape and a reader can follow one sample across a
# figure without reading labels. Batch is then carried by colour, marker fill and line style --
# three channels, so the distinction still survives greyscale and colour-vision deficiency.
# Single-tree plots use group colour; legacy cross-batch plots retain BATCH_COLOR.
# Fixed identities keep colours stable when a presentation selects fewer groups.
GROUP_ORDER = ("initial", "empty", "A549", "MDA", "A549-SN", "MDA-SN")

GROUP_COLOR = {
    "A549": OKABE_ITO["blue"],
    "MDA": OKABE_ITO["vermillion"],
    "empty": OKABE_ITO["bluish_green"],
    "A549-SN": OKABE_ITO["sky_blue"],
    "MDA-SN": OKABE_ITO["reddish_purple"],
    "initial": OKABE_ITO["black"],
}


def group_color(group: str) -> str:
    """Stable group colour, independent of sample order or selected groups."""
    return GROUP_COLOR.get(group, OKABE_ITO_LIST[sum(map(ord, group)) % len(OKABE_ITO_LIST)])


GROUP_MARKER = {
    "initial":  "o",
    "A549":     "s",
    "A549-SN":  "P",
    "MDA":      "^",
    "MDA-SN":   "v",
    "empty":    "D",
    "empty1":   "h",          # batch 1's empty arm -- different material from batch 3's

    "HEK293":   "X",
    "RAW":      "*",
}
#: Fallback pool for a group not in the table above, so a new group gets its own shape rather than
#: silently colliding with `initial`.
_MARKER_POOL = ["<", ">", "p", "h", "8", "d"]


def group_marker(group: str) -> str:
    """Marker shape for a sample group, stable across runs and figures."""
    if group in GROUP_MARKER:
        return GROUP_MARKER[group]
    return _MARKER_POOL[sum(map(ord, group)) % len(_MARKER_POOL)]

#: Recessive ink: reference lines, background series. Not a batch colour.
GREY = "#7d8590"

def workflow_group_colors(groups):
    """Shared report/Venn colours in configured group order, with a grey baseline."""
    groups = list(dict.fromkeys(group for group in groups if group != "initial"))
    colors = [OKABE_ITO[key] for key in
              ("blue", "bluish_green", "vermillion", "reddish_purple", "orange")]
    return {"initial": GREY, **{group: colors[i % len(colors)] for i, group in enumerate(groups)}}


# --- Continuous colormaps (names; resolve with matplotlib.colormaps[...]) ---
CMAP_SEQUENTIAL = "viridis"    # expression / density
CMAP_PSEUDOTIME = "magma"      # pseudotime
CMAP_DIVERGING  = "vlag"       # z-scores / log2FC, symmetric about 0 (seaborn); fallback "RdBu_r"


def figsize(width="3col", ratio=0.62):
    """Return (w, h) in inches for a Science column width. ratio = h/w."""
    w = {"1col": WIDTH_1COL_IN, "2col": WIDTH_2COL_IN, "3col": WIDTH_3COL_IN}[width]
    return (w, w * ratio)


def diverging_norm(vmax, vmin=None):
    """A matplotlib TwoSlopeNorm centred at 0 for log2FC / z-score maps."""
    from matplotlib.colors import TwoSlopeNorm
    vmin = -vmax if vmin is None else vmin
    return TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)


def simulate_cvd(hex_colors, kind="deuteranomaly", severity=100):
    """
    Simulate colour-vision deficiency for a list of hex colours so a palette can be
    checked for distinguishability. Uses colorspacious if available; otherwise returns
    the input unchanged with a printed warning (so scripts never hard-fail).
    kind in {"deuteranomaly","protanomaly","tritanomaly"}.
    """
    try:
        import numpy as np
        from colorspacious import cspace_convert
        from matplotlib.colors import to_rgb, to_hex
        cvd = {"name": "sRGB1+CVD", "cvd_type": kind, "severity": severity}
        out = []
        for c in hex_colors:
            rgb = np.array(to_rgb(c))
            sim = cspace_convert(rgb, cvd, "sRGB1")
            out.append(to_hex(np.clip(sim, 0, 1)))
        return out
    except Exception as e:  # colorspacious not installed
        print(f"[simulate_cvd] skipped ({type(e).__name__}); install colorspacious to enable")
        return list(hex_colors)


def check_min_font(fig):
    """Assert no text artist in `fig` renders below MIN_TEXT_PT at the figure's size."""
    bad = [t.get_text() for t in fig.findobj(match=lambda o: hasattr(o, "get_fontsize"))
           if getattr(t, "get_text", lambda: "")() and t.get_fontsize() < MIN_TEXT_PT]
    if bad:
        print(f"[check_min_font] WARNING: {len(bad)} text items < {MIN_TEXT_PT} pt")
    return not bad
