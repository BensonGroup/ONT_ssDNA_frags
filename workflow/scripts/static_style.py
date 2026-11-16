"""Shared print style for the static figures: Science column widths, Okabe-Ito inks, group identity.

Ported from the cells project's `palettes.py`, minus its batch channels and study-named groups.
Sample and group identity come from `report_lib.PlotStyle`, so a group has the same colour and
marker shape, and a sample the same line dash, in the static figures as in the HTML reports.
"""
from __future__ import annotations

from pathlib import Path

import report_lib as RL

STYLE = Path(__file__).with_name("science.mplstyle")
WIDTH_CM = {"1col": 5.7, "2col": 12.1, "3col": 18.4}
MIN_TEXT_PT = 5
OKABE_ITO = {"orange": "#E69F00", "sky_blue": "#56B4E9", "bluish_green": "#009E73",
             "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00",
             "reddish_purple": "#CC79A7", "black": "#000000"}
OKABE_ITO_LIST = list(OKABE_ITO.values())
GREY = RL.GREY
BLACK = OKABE_ITO["black"]
CMAP_SEQUENTIAL = "viridis"
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "p", "8")
DASHES = ("-", (0, (4, 1.4)), (0, (1, 1.1)), (0, (5, 1.2, 1, 1.2)),
          (0, (2.5, 1, 1, 1, 1, 1)), (0, (7, 1.6, 1, 1.6)), (0, (3, 1, 1, 1)), (0, (6, 1.2)))


def figsize(width: str = "3col", ratio: float = 0.62) -> tuple[float, float]:
    w = WIDTH_CM[width] / 2.54
    return (w, w * ratio)


def check_min_font(fig) -> bool:
    """False (with a warning) if any text in `fig` renders below MIN_TEXT_PT."""
    bad = [t for t in fig.findobj(match=lambda o: hasattr(o, "get_fontsize"))
           if getattr(t, "get_text", lambda: "")() and t.get_fontsize() < MIN_TEXT_PT]
    if bad:
        print(f"[WARN] {len(bad)} text items < {MIN_TEXT_PT} pt")
    return not bad


class Groups(RL.PlotStyle):
    """The reports' PlotStyle (group colour + shape, a dash per sample) plus the print order:
    `initial` first, then the groups by first appearance in `samples:` -- the order the HTML reports
    colour by. `G[sample].mpl()` draws a sample exactly as its report trace looks."""

    def __init__(self, samples: list[dict]):
        super().__init__(sorted(samples, key=lambda s: s.get("idx", 0)))
        has_base = any(s.get("group") == RL.BASELINE_GROUP for s in samples)
        self.order = ([RL.BASELINE_GROUP] if has_base else []) + self.groups

    def rank(self, group: str) -> int:
        return self.order.index(group) if group in self.order else len(self.order)
