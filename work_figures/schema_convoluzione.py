"""
Figure: example of a 2D convolution. A 3x3 filter slides over a 5x5 input.

Output: images/schema_convoluzione.pdf
Usage (from the repository root): python work_figures/schema_convoluzione.py
"""
import numpy as np
from matplotlib.path import Path

import figure_style as fs

INPUT = np.array([[1, 2, 0, 1, 0],
                  [0, 1, 3, 1, 1],
                  [2, 1, 0, 0, 2],
                  [1, 0, 1, 2, 1],
                  [0, 2, 1, 0, 1]])
FILTER = np.array([[1, 0, 1],
                   [0, 1, 0],
                   [1, 0, 1]])
C = 0.62                                                       # side of a cell (cm)

def convolve(x, k):
    n = x.shape[0] - k.shape[0] + 1
    return np.array([[(x[i:i + k.shape[0], j:j + k.shape[1]] * k).sum() for j in range(n)] for i in range(n)])

def grid(ax, m, x0, y0, fill=None, bold=()):
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            x, y = x0 + j * C, y0 - (i + 1) * C
            colour = fill(i, j) if fill else None
            if colour is not None:
                fs.rect(ax, x, y, C, C, fc=colour, ec="none", z=1)
            fs.rect(ax, x, y, C, C, fc="none", ec=fs.tint(fs.GREY, 0.7), lw=0.4, z=2)
            fs.text(ax, x + C / 2, y + C / 2, str(m[i, j]), bold=(i, j) in bold)

def main():
    output = convolve(INPUT, FILTER)
    fig, ax = fs.canvas(-0.3, 9.3, -4.6, 1.9)
    blue_fill = fs.tint(fs.BLUE, 0.12)

    # input: the 3x3 window covered by the filter is highlighted
    grid(ax, INPUT, 0, 0, fill=lambda i, j: blue_fill if i < 3 and j < 3 else None)
    fs.rect(ax, 0, -3 * C, 3 * C, 3 * C, fc="none", ec=fs.BLUE, lw=1.1, z=3)
    fs.text(ax, 2.5 * C, 0.45, "Ingresso", bold=True)
    fs.text(ax, 2.5 * C, -5 * C - 0.35, r"$5 \times 5$", size=fs.FOOT, color=fs.GREY)

    fs.text(ax, 5 * C + 0.55, -2.5 * C, "$*$", size=fs.LARGE)

    # filter
    fx, fy = 5 * C + 1.1, -C
    grid(ax, FILTER, fx, fy, fill=lambda i, j: blue_fill)
    fs.rect(ax, fx, fy - 3 * C, 3 * C, 3 * C, fc="none", ec=fs.BLUE, lw=1.1, z=3)
    fs.text(ax, fx + 1.5 * C, fy + C + 0.45, "Filtro", bold=True)
    fs.text(ax, fx + 1.5 * C, fy - 4 * C - 0.35, r"$3 \times 3$", size=fs.FOOT, color=fs.GREY)

    fs.text(ax, 8 * C + 1.65, -2.5 * C, "$=$", size=fs.LARGE)

    # output: the first value is the result of the highlighted window
    ox, oy = 8 * C + 2.2, -C
    grid(ax, output, ox, oy, fill=lambda i, j: fs.tint(fs.RED, 0.15) if (i, j) == (0, 0) else None, bold=[(0, 0)])
    fs.rect(ax, ox, oy - C, C, C, fc="none", ec=fs.RED, lw=1.1, z=3)
    fs.text(ax, ox + 1.5 * C, oy + C + 0.45, "Uscita", bold=True)
    fs.text(ax, ox + 1.5 * C, oy - 4 * C - 0.35, r"$3 \times 3$", size=fs.FOOT, color=fs.GREY)

    # curved arrow from the window to its output value (cubic Bezier)
    p0, p3 = np.array([3 * C, 0.05]), np.array([ox + 0.5 * C, oy + 0.08])
    d = 0.39 * np.linalg.norm(p3 - p0)
    p1 = p0 + d * np.array([np.cos(np.radians(25)), np.sin(np.radians(25))])
    p2 = p3 + d * np.array([np.cos(np.radians(110)), np.sin(np.radians(110))])
    fs.arrow(ax, None, None, color=fs.RED, lw=0.7,
             path=Path([p0, p1, p2, p3], [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]))

    # computation of the first value
    window = INPUT[:3, :3]
    terms = [" + ".join(rf"{a}{{\cdot}}{b}" for a, b in zip(window[i], FILTER[i])) for i in range(3)]
    formula = r" \;+\; ".join(terms) + rf" \;=\; \mathbf{{{output[0, 0]}}}"
    fs.text(ax, 6.2 * C + 1.1, -5 * C - 0.75, f"${formula}$", va="top")

    fs.save(fig, "schema_convoluzione")

if __name__ == "__main__":
    main()
