"""Area-proportional Venn geometry reused from scripts/venn_initial_vs_r5_fig.py."""
import math
from matplotlib.patches import Circle
import palettes as P
INITIAL_COLOR = P.OKABE_ITO["sky_blue"]
GREY = P.GREY

def lens_area(d: float, ra: float, rb: float) -> float:
    """Area of the lens where two circles of radii ra, rb whose centres are d apart intersect."""
    if d >= ra + rb:
        return 0.0
    if d <= abs(ra - rb):
        return math.pi * min(ra, rb) ** 2
    a = ra * ra * math.acos((d * d + ra * ra - rb * rb) / (2 * d * ra))
    b = rb * rb * math.acos((d * d + rb * rb - ra * ra) / (2 * d * rb))
    c = 0.5 * math.sqrt((-d + ra + rb) * (d + ra - rb) * (d - ra + rb) * (d + ra + rb))
    return a + b - c


def solve_centre_distance(ra: float, rb: float, inter: float) -> float:
    """The d that makes the lens area exactly `inter`.

    `lens_area` is continuous and strictly decreasing in d on [|ra-rb|, ra+rb], so plain bisection
    converges and needs no derivative. Returns the tangent distance when inter == 0 and the
    fully-contained distance when inter == min(area).
    """
    if inter <= 0:
        return ra + rb
    lo, hi = abs(ra - rb), ra + rb
    if inter >= math.pi * min(ra, rb) ** 2:
        return lo
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if lens_area(mid, ra, rb) > inter:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _fmt(n: int) -> str:
    """Thousands separated by a thin space -- a comma reads as a decimal point to half the world."""
    return f"{n:,}".replace(",", " ")


# ----------------------------------------------------------------------------------- the drawing
def draw_venn(ax, n_left: int, n_right: int, shared: int, *,
              label_left: str, label_right: str, color_right: str, hatch_right: str,
              scale: float, unit: str) -> None:
    """One area-proportional two-set Venn on `ax`.

    `scale` is the row's shared area->figure-units factor, passed in rather than derived per panel
    so both panels of a row draw the identical `initial` circle at the identical size.
    """
    if not 0 <= shared <= min(n_left, n_right):
        raise ValueError("Invalid Venn set/intersection sizes")
    if not n_left or not n_right:
        ax.set_axis_off()
        ax.text(0.5, 0.5, f"{label_left}: {n_left:,}\n{label_right}: {n_right:,}\nShared: {shared:,}",
                transform=ax.transAxes, ha="center", va="center", fontsize=6)
        return
    ra = math.sqrt(n_left / math.pi) * scale
    rb = math.sqrt(n_right / math.pi) * scale
    d = solve_centre_distance(ra, rb, shared * scale * scale)

    cx_a, cx_b = 0.0, d
    ax.add_patch(Circle((cx_a, 0), ra, facecolor=INITIAL_COLOR, alpha=0.45,
                        edgecolor=INITIAL_COLOR, linewidth=0.6, zorder=2))
    ax.add_patch(Circle((cx_b, 0), rb, facecolor=color_right, alpha=0.45, hatch=hatch_right,
                        edgecolor=color_right, linewidth=0.6, zorder=3))

    only_left, only_right = n_left - shared, n_right - shared
    overlap_w = max(0.0, ra + rb - d)          # the lens's extent along x

    # exclusive counts, each parked in the fat part of its own circle
    ax.text(cx_a - ra * 0.42, 0, _fmt(only_left), ha="center", va="center", fontsize=6, zorder=5)
    ax.text(cx_b + rb * 0.42, 0, _fmt(only_right), ha="center", va="center", fontsize=6, zorder=5)

    # direct set labels -- so hue and hatch are both redundant, never the only carrier of identity
    ax.text(0.02, 0.98, label_left, transform=ax.transAxes, ha="left", va="top",
            fontsize=6.5, zorder=5)
    ax.text(0.98, 0.98, label_right, transform=ax.transAxes, ha="right", va="top",
            fontsize=6.5, zorder=5)

    pad = max(ra, rb) * 0.30
    ax.set_xlim(cx_a - ra - pad, cx_b + rb + pad)
    ax.set_ylim(-max(ra, rb) - pad * 0.65, max(ra, rb) + pad * 2.2)
    ax.set_aspect("equal")
    ax.axis("off")

    # The shared count goes INSIDE the lens when the lens can actually hold it, and on a leader
    # line when it cannot. Which of the two applies is not guessed from a ratio: the label is
    # drawn, its rendered box is measured against the lens width, and it is moved out only if it
    # does not fit. Leader lines keep small intersections legible.
    lens_cx = 0.5 * ((cx_a + ra) + (cx_b - rb))
    txt = _fmt(shared)
    t = ax.text(lens_cx, 0, txt, ha="center", va="center", fontsize=6, fontweight="bold", zorder=6)
    fits = False
    if shared > 0 and overlap_w > 0:
        fig = ax.get_figure()
        fig.canvas.draw()
        bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
        inv = ax.transData.inverted()
        w_data = abs(inv.transform((bb.x1, 0))[0] - inv.transform((bb.x0, 0))[0])
        fits = w_data < overlap_w * 0.80
    if not fits:
        t.remove()
        y_tip = min(ra, rb) * 0.30
        ax.annotate(txt, xy=(lens_cx, y_tip), xytext=(lens_cx, max(ra, rb) * 1.22),
                    ha="center", va="bottom", fontsize=6, fontweight="bold", zorder=6,
                    arrowprops=dict(arrowstyle="-", color=GREY, linewidth=0.5,
                                    shrinkA=1.0, shrinkB=0.5))

    # The factual sub-caption, on two lines so it stays inside a half-width panel: the one
    # intersection expressed against each set in turn, because neither direction alone is the
    # complete description: both fractions are needed when set sizes differ.
    ax.text(0.5, -0.03, f"shared {_fmt(shared)} {unit}s", transform=ax.transAxes,
            ha="center", va="top", fontsize=5.5, color="black")
    ax.text(0.5, -0.115,
            f"{100 * shared / n_right:.2f}% of {label_right}  ·  "
            f"{100 * shared / n_left:.2f}% of {label_left}",
            transform=ax.transAxes, ha="center", va="top", fontsize=5.5, color=GREY)


def row_scale(rows: list[dict], width: float) -> float:
    """One area scale for a whole row, set by whichever of its panels lays out widest.

    Solved by iteration because the layout width depends on d, which depends on the scale: start
    from the scale that would fit two tangent circles and shrink until every panel fits.
    """
    best = None
    for r in rows:
        ra_u = math.sqrt(r["n_left"] / math.pi)
        rb_u = math.sqrt(r["n_right"] / math.pi)
        s = width / (2 * (ra_u + rb_u)) if ra_u + rb_u else 1.0            # tangent case: the widest a layout can ever be
        for _ in range(40):
            ra, rb = ra_u * s, rb_u * s
            d = solve_centre_distance(ra, rb, r["shared"] * s * s)
            span = ra + d + rb
            if span <= width:
                break
            s *= width / span
        best = s if best is None else min(best, s)
    return best
