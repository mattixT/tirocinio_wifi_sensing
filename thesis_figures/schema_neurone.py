"""
Figure: (a) artificial neuron (weighted sum of the inputs plus bias, then the activation function g)
and (b) a small feed-forward network with an input layer, two hidden layers and an output layer.

Output: images/schema_neurone.pdf
Usage (from the repository root): python thesis_figures/schema_neurone.py
"""
import figure_style as fs


def panel_neuron(ax):
    fs.text(ax, -0.4, 3.0, fs.bold("(a)") + " Neurone artificiale", ha="left")
    r_in, r_sum = 0.31, 0.5
    s = (2.6, 0.85)                                            # centre of the sum
    inputs = [((0, 2.1), "$x_1$", "$w_1$", "above"), ((0, 1.2), "$x_2$", "$w_2$", "above"),
              ((0, -0.4), "$x_n$", "$w_n$", "below")]
    for p, x_label, w_label, side in inputs:
        fs.node(ax, *p, r_in, label=x_label)
        a, b = fs.between(p, s, r_in, r_sum)
        fs.arrow(ax, a, b)
        mx, my = a[0] + 0.45 * (b[0] - a[0]), a[1] + 0.45 * (b[1] - a[1])
        if side == "above":
            fs.text(ax, mx, my + 0.08, w_label, size=fs.FOOT, va="bottom")
        else:
            fs.text(ax, mx, my - 0.08, w_label, size=fs.FOOT, va="top")
    fs.dots(ax, 0, 0.45, vertical=True)

    fs.node(ax, *s, r_sum, fc=fs.tint(fs.BLUE, 0.12), ec=fs.BLUE, lw=0.9, label=r"$\Sigma$")
    fs.text(ax, 2.6, 2.35, "bias $b$")
    fs.arrow(ax, (2.6, 2.2), (2.6, s[1] + r_sum))

    gx, gy, gw, gh = 4.6, 0.85, 0.85, 0.6                      # activation function
    fs.rect(ax, gx - gw / 2, gy - gh / 2, gw, gh, lw=0.8)
    fs.text(ax, gx, gy, "$g$")
    fs.arrow(ax, (s[0] + r_sum, s[1]), (gx - gw / 2, gy))
    fs.arrow(ax, (gx + gw / 2, gy), (gx + gw / 2 + 1.1, gy))
    fs.text(ax, gx + gw / 2 + 1.18, gy, "$u$", ha="left")

    fs.text(ax, 0, -1.05, "ingressi", size=fs.FOOT, color=fs.GREY)
    fs.text(ax, 2.6, -0.05, "somma pesata", size=fs.FOOT, color=fs.GREY)
    fs.text(ax, 4.6, 0.05, "funzione di\nattivazione", size=fs.FOOT, color=fs.GREY, va="top")


def panel_network(ax, dx=7.6):
    fs.text(ax, dx - 0.4, 3.0, fs.bold("(b)") + " Rete feed-forward", ha="left")
    xs = [0, 1.35, 2.7, 4.05]
    sizes = [4, 5, 5, 3]
    layers = [[(dx + x, 0.85 + (i - (n + 1) / 2) * 0.62) for i in range(1, n + 1)] for x, n in zip(xs, sizes)]
    for left, right in zip(layers[:-1], layers[1:]):          # every neuron is connected to the next layer
        for p in left:
            for q in right:
                fs.line(ax, [p, q], color=fs.tint(fs.GREY, 0.45), lw=0.35, z=1)
    r = 0.21
    for p in layers[0]:
        fs.node(ax, *p, r, fc=fs.LIGHT)
    for layer in layers[1:-1]:
        for p in layer:
            fs.node(ax, *p, r)
    for p in layers[-1]:
        fs.node(ax, *p, r, fc=fs.tint(fs.BLUE, 0.12), ec=fs.BLUE, lw=0.9)
    fs.text(ax, dx, -1.05, "livello di\ningresso", size=fs.FOOT, color=fs.GREY)
    fs.text(ax, dx + 2.025, -1.05, "livelli nascosti", size=fs.FOOT, color=fs.GREY)
    fs.text(ax, dx + 4.05, -1.05, "livello di\nuscita", size=fs.FOOT, color=fs.GREY)


def main():
    fig, ax = fs.canvas(-0.6, 12.0, -1.6, 3.3)
    panel_neuron(ax)
    panel_network(ax)
    fs.save(fig, "schema_neurone")


if __name__ == "__main__":
    main()
