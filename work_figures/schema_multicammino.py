"""
Figure: multipath propagation (chapter 2).

Output: images/schema_multicammino.pdf
Usage (from the repository root): python work_figures/schema_multicammino.py
"""
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

import figure_style as fs

def router(ax, cx, cy, antennas):
    """Router seen from the front, centred in (cx, cy); `antennas` are the x offsets of the antennas."""
    for dx in antennas:
        ax.plot([cx + dx, cx + dx], [cy + 0.17, cy + 0.62], color=fs.INK, lw=1.4,
                solid_capstyle="round", zorder=5)
    ax.add_patch(Polygon([(cx - 0.5, cy + 0.12), (cx + 0.5, cy + 0.12), (cx + 0.43, cy + 0.22),
                          (cx - 0.43, cy + 0.22)], closed=True, fc=fs.tint(fs.INK, 0.7), ec="none", zorder=5))
    ax.add_patch(FancyBboxPatch((cx - 0.55, cy - 0.16), 1.1, 0.30, boxstyle="round,pad=0,rounding_size=0.05",
                                fc=fs.INK, ec="none", zorder=6))
    for dx in (-0.37, -0.26, -0.15, -0.04, 0.07):         
        ax.add_patch(Circle((cx + dx, cy - 0.01), 0.025, fc=fs.tint(fs.INK, 0.3), ec="none", zorder=7))

def person(ax, x, y, color):
    """Person seen from above (shoulders and head) centred in (x, y)."""
    ax.add_patch(FancyBboxPatch((x - 0.42, y - 0.15), 0.84, 0.30, boxstyle="round,pad=0,rounding_size=0.1",
                                fc="white", ec=color, lw=0.8, zorder=8))
    ax.add_patch(Circle((x, y), 0.13, fc=fs.tint(color, 0.35), ec=color, lw=0.8, zorder=9))

def panel_room(ax):
    fs.text(ax, -0.1, 5.3, fs.bold("(a)"), ha="left")
    # walls
    for y in (4.4, -0.2):
        ax.add_patch(Rectangle((0, y), 12, 0.2, fc="none", ec=fs.tint(fs.GREY, 0.7), lw=0, hatch="//////", zorder=1))
    fs.rect(ax, 0, 0, 12, 4.4, fc="none", ec=fs.GREY, lw=0.7, z=1)
    fs.text(ax, 0.1, 4.33, "parete", size=fs.FOOT, color=fs.GREY, ha="left", va="top")
    fs.text(ax, 0.1, 0.07, "parete", size=fs.FOOT, color=fs.GREY, ha="left", va="bottom")

    # static paths: direct and reflected by the two walls
    fs.arrow(ax, (1.95, 2.4), (9.85, 2.4), color=fs.GREY, lw=0.9)
    fs.text(ax, 5.9, 2.47, "cammino diretto", size=fs.FOOT, va="bottom")
    for wall_y, start_y, end_y in ((4.4, 2.58, 2.58), (0.0, 2.2, 2.2)):
        fs.line(ax, [(1.6, start_y), (6, wall_y)], color=fs.GREY, lw=0.9)
        fs.arrow(ax, (6, wall_y), (9.9, end_y), color=fs.GREY, lw=0.9)
        ax.add_patch(Circle((6, wall_y), 0.056, fc=fs.GREY, ec="none", zorder=3))

    # dynamic path an instant later (person in light grey)
    fs.line(ax, [(1.95, 2.2), (6.75, 0.95), (9.85, 2.24)], color=fs.tint(fs.RED, 0.5), lw=0.9, dashes=(3, 2))
    person(ax, 6.75, 0.95, fs.tint(fs.GREY, 0.55))
    # dynamic path
    fs.line(ax, [(1.95, 2.3), (6.3, 1.65)], color=fs.RED, lw=1.1)
    fs.arrow(ax, (6.3, 1.65), (9.85, 2.3), color=fs.RED, lw=1.1)
    person(ax, 6.3, 1.65, fs.INK)
    fs.text(ax, 6.3, 2.03, "persona", size=fs.FOOT)
    fs.arrow(ax, (6.43, 1.47), (6.64, 1.15), lw=0.7)        # displacement of the person
    fs.text(ax, 6.75, 1.36, "$v$", size=fs.FOOT)

    # transmitter and receiver
    router(ax, 1.35, 2.4, [0])
    fs.text(ax, 1.35, 1.95, "TX", bold=True)
    fs.text(ax, 1.35, 1.6, "1 antenna", size=fs.FOOT)
    router(ax, 10.45, 2.4, [-0.3, 0, 0.3])
    fs.text(ax, 10.45, 1.95, "RX", bold=True)
    fs.text(ax, 10.45, 1.6, "3 antenne", size=fs.FOOT)

    # legend
    legend = [("cammini statici", fs.GREY, 0.9, None),
              ("cammino dinamico", fs.RED, 1.1, None),
              ("lo stesso cammino un istante dopo, con la persona in grigio chiaro", fs.tint(fs.RED, 0.5), 0.9, (3, 2))]
    for i, (label, color, lw, dashes) in enumerate(legend):
        y = -0.75 - 0.45 * i
        fs.line(ax, [(0.3, y), (1.2, y)], color=color, lw=lw, dashes=dashes)
        fs.text(ax, 1.4, y, label, size=fs.FOOT, ha="left")

def panel_amplitude(ax, dy=-5.6):
    fs.text(ax, -0.1, 3.35 + dy, fs.bold("(b)"), ha="left")
    start = 4.6                                                # the person starts moving here
    fs.rect(ax, start, dy, 7.1, 2.65, fc=fs.tint(fs.LIGHT, 0.7), ec="none", z=0)
    fs.line(ax, [(start, dy), (start, 2.65 + dy)], color=fs.GREY, lw=0.5, dashes=(2, 2), z=1)
    fs.text(ax, 2.7, 2.42 + dy, "persona ferma", size=fs.FOOT)
    fs.text(ax, 8.15, 2.42 + dy, "persona in movimento", size=fs.FOOT)

    period = 0.85
    x = np.linspace(0.85, start, 160)
    ax.plot(x, dy + 1.2 + 0.035 * np.sin(np.radians(x * 1300)), color=fs.INK, lw=0.9, zorder=3)
    x = np.linspace(start, 11.6, 400)
    y = 1.2 + 0.62 * (1 - np.exp(-(x - start) * 3)) * np.sin(2 * np.pi * (x - start) / period)
    ax.plot(x, dy + y, color=fs.RED, lw=1.0, zorder=3)

    pa = start + period * 6.25
    pb = pa + period
    for p in (pa, pb):
        fs.line(ax, [(p, 1.82 + dy), (p, 0.32 + dy)], color=fs.GREY, lw=0.5, dashes=(2, 1.5))
    fs.arrow(ax, (pa, 0.4 + dy), (pb, 0.4 + dy), lw=0.5, both=True)
    fs.text(ax, (pa + pb) / 2, 0.18 + dy, "periodo", size=fs.FOOT,
            bbox=dict(fc=fs.tint(fs.LIGHT, 0.7), ec="none", pad=0.5))

    fs.arrow(ax, (0.8, dy), (12.1, dy), lw=0.6)
    fs.text(ax, 12.1, dy - 0.08, "tempo", size=fs.FOOT, ha="right", va="top")
    fs.arrow(ax, (0.8, dy), (0.8, 2.95 + dy), lw=0.6)
    fs.text(ax, 0.98, 2.95 + dy, "ampiezza", size=fs.FOOT, ha="left")

def main():
    fig, ax = fs.canvas(-0.4, 12.4, -6.0, 5.6)
    panel_room(ax)
    panel_amplitude(ax)
    fs.save(fig, "schema_multicammino")

if __name__ == "__main__":
    main()
