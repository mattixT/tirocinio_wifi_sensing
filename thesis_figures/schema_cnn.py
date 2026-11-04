"""
Figure: general structure of a CNN applied to the CSI. The three antenna pairs are the input
channels; a filter slides over them and produces the feature maps of the convolutional layers,
then the fully connected layers and the softmax give the probability of each of the 6 activities.
The probabilities on the right are only an example.

Output: images/schema_cnn.pdf
Usage (from the repository root): python thesis_figures/schema_cnn.py
"""
from matplotlib.patches import Circle

import figure_style as fs

PROBABILITIES = {"box": 0.01, "circle": 0.01, "clean": 0.02, "fall": 0.01, "run": 0.07, "walk": 0.88}


def stack(ax, x, y, side, n, step):
    """n feature maps of size side x side, drawn from the back to the front."""
    for i in reversed(range(n)):
        fs.rect(ax, x + i * step, y + i * step, side, side, fc=fs.LIGHT, lw=0.4, z=2)


def main():
    fig, ax = fs.canvas(-0.6, 16.6, -1.6, 2.6)
    csi = fs.load_sample()

    # input: the three antenna pairs as channels (pair 1 in front)
    w, h = 2.7, 1.75
    for layer, (k, d) in enumerate(((2, 0.36), (1, 0.18), (0, 0.0))):       # from the back to the front
        fs.heatmap(ax, csi[k], d, d, w, h, z=1 + layer)
        fs.rect(ax, d, d, w, h, fc="none", lw=0.5, z=1.5 + layer)
    fs.rect(ax, 0.95, 1.05, 0.45, 0.45, fc="none", ec="white", lw=1.0, z=5)     # filter
    fs.text(ax, 1.175, 1.0, "filtro", size=fs.FOOT, color="white", bold=True, va="top")
    fs.text(ax, 1.5, -0.45, "ingresso: coppie di antenne\n(canali) $\\times$ sottoportanti $\\times$ tempo",
            size=fs.FOOT, va="top")

    # convolutional layers: more and smaller feature maps from left to right
    fs.arrow(ax, (3.25, 1.05), (3.75, 1.05))
    stack(ax, 4.0, 0.25, 1.55, 5, 0.09)
    target = (4.85, 1.45)                    # value produced by the filter in the first feature map
    for corner in ((1.4, 1.5), (1.4, 1.05)):
        fs.line(ax, [corner, target], color=fs.RED, lw=0.6, dashes=(3, 2), z=6)
    ax.add_patch(Circle(target, 0.053, fc=fs.RED, ec="none", zorder=7))
    fs.arrow(ax, (6.05, 1.05), (6.5, 1.05))
    stack(ax, 6.7, 0.55, 0.95, 8, 0.07)
    fs.arrow(ax, (8.25, 1.05), (8.7, 1.05))
    stack(ax, 8.9, 0.8, 0.55, 10, 0.05)
    fs.text(ax, 6.95, -0.45, "livelli convoluzionali + ReLU:\nmappe di caratteristiche", size=fs.FOOT, va="top")

    # fully connected layers (8 -> 4 -> 6 neurons in the drawing)
    fs.arrow(ax, (10.05, 1.05), (10.5, 1.05))
    a = [(10.85, 1.05 + (i - 4.5) * 0.36) for i in range(1, 9)]
    b = [(11.75, 1.05 + (i - 2.5) * 0.36) for i in range(1, 5)]
    c = [(12.65, 1.05 + (3.5 - i) * 0.36) for i in range(1, 7)]
    for left, right in ((a, b), (b, c)):
        for p in left:
            for q in right:
                fs.line(ax, [p, q], color=fs.tint(fs.GREY, 0.45), lw=0.3, z=1)
    r = 0.135
    for p in a + b:
        fs.node(ax, *p, r, lw=0.5)
    fs.text(ax, 11.3, -0.45, "livelli\ncompletamente\nconnessi", size=fs.FOOT, va="top")

    # output: one neuron per activity, with the probability given by the softmax
    for p, (activity, prob) in zip(c, PROBABILITIES.items()):
        fs.node(ax, *p, r, fc=fs.tint(fs.BLUE, 0.12), ec=fs.BLUE, lw=0.8)
        fs.text(ax, p[0] + 0.25, p[1], activity, size=fs.FOOT, ha="left")
        fs.text(ax, p[0] + 1.3, p[1], fs.number(prob, 2), size=fs.SCRIPT, color=fs.GREY, ha="left")
        colour = fs.RED if prob == max(PROBABILITIES.values()) else fs.tint(fs.GREY, 0.7)
        fs.rect(ax, p[0] + 2.1, p[1] - 0.07, prob * 1.6, 0.14, fc=colour, ec="none", z=3)
    fs.text(ax, 14.0, -0.45, "softmax: probabilità\ndelle 6 attività", size=fs.FOOT, va="top")

    fs.save(fig, "schema_cnn")


if __name__ == "__main__":
    main()
