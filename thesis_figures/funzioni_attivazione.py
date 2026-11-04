"""
Figure: the three activation functions described in chapter 2 (ReLU, sigmoid, hyperbolic tangent).

Output: images/funzioni_attivazione.pdf
Usage (from the repository root): python thesis_figures/funzioni_attivazione.py
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

import figure_style as fs

FUNCTIONS = [("ReLU", lambda x: np.maximum(0, x), (-0.4, 4.2), [0, 1, 2, 3, 4]),
             ("sigmoide", lambda x: 1 / (1 + np.exp(-x)), (-0.08, 1.08), [0, 0.25, 0.5, 0.75, 1]),
             ("tangente iperbolica", np.tanh, (-1.15, 1.15), [-1, -0.5, 0, 0.5, 1])]

# numbers with the decimal comma, as in the text of the thesis
comma = FuncFormatter(lambda v, _: fs.number(v))


def main():
    fig, axes = plt.subplots(1, 3, figsize=(13.92 / 2.54, 4.6 / 2.54))     # width of the text of the thesis
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.22, top=0.88, wspace=0.42)
    x = np.linspace(-4, 4, 400)
    for ax, (title, f, ylim, yticks) in zip(axes, FUNCTIONS):
        ax.axhline(0, color=fs.tint(fs.GREY, 0.6), lw=0.4, zorder=0)
        ax.axvline(0, color=fs.tint(fs.GREY, 0.6), lw=0.4, zorder=0)
        ax.plot(x, f(x), color=fs.BLUE, lw=1.2)
        ax.set(xlim=(-4, 4), ylim=ylim, xticks=[-2.5, 0, 2.5], yticks=yticks)
        ax.set_title(title, fontsize=fs.SMALL, pad=4)
        ax.set_xlabel("ingresso", fontsize=fs.SMALL)
        ax.xaxis.set_major_formatter(comma)
        ax.yaxis.set_major_formatter(comma)
        ax.tick_params(labelsize=fs.FOOT, width=0.4, length=3, color=fs.INK)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(0.5)
            ax.spines[side].set_color(fs.INK)
    axes[0].set_ylabel("uscita", fontsize=fs.SMALL)
    fs.save(fig, "funzioni_attivazione")


if __name__ == "__main__":
    main()
