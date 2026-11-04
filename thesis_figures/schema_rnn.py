"""
Figure: how a recurrent network (RNN, GRU, LSTM) reads a CSI sample. At every instant t the cell
receives the amplitudes of that instant (x_t, all antenna pairs and subcarriers) and the memory of
the previous instants (h_{t-1}); the last memory h_T goes to a fully connected layer that gives
the 6 activities.

Output: images/schema_rnn.pdf
Usage (from the repository root): python thesis_figures/schema_rnn.py
"""
import figure_style as fs

COLUMNS = [(1.35, "1"), (3.15, "2"), (4.95, "3"), (10.35, "T")]     # instants shown: x position, index
CELL_Y, CELL_W, CELL_H = -1.35, 1.15, 0.6


def main():
    fig, ax = fs.canvas(-0.4, 14.4, -3.0, 3.6)
    csi = fs.load_sample()

    # sample: the three antenna pairs over time
    left, width, height = 0.9, 10.0, 0.82
    for k, y in enumerate((2.0, 1.1, 0.2)):
        fs.heatmap(ax, csi[k], left, y, width, height)
        fs.rect(ax, left, y, width, height, fc="none", lw=0.4, z=2)
        fs.text(ax, left - 0.1, y + 0.41, f"coppia {k + 1}", size=fs.FOOT, ha="right")
    fs.arrow(ax, (left, 3.15), (left + width, 3.15))
    fs.text(ax, left + width / 2, 3.22, "tempo", size=fs.FOOT, va="bottom")

    # one column of the sample (one instant) for every cell
    cells = []
    for x, t in COLUMNS:
        fs.rect(ax, x - 0.1, 0.2, 0.2, 2.62, fc="none", ec="white", lw=1.1, z=3)
        fs.rect(ax, x - 0.13, 0.17, 0.26, 2.68, fc="none", lw=0.4, z=3)
        fs.rect(ax, x - CELL_W / 2, CELL_Y - CELL_H / 2, CELL_W, CELL_H, lw=0.7, z=3)
        fs.text(ax, x, CELL_Y, "cella")
        fs.arrow(ax, (x, 0.12), (x, CELL_Y + CELL_H / 2))
        fs.text(ax, x + 0.08, 0.12 + 0.45 * (CELL_Y + CELL_H / 2 - 0.12), f"$x_{{{t}}}$", size=fs.FOOT, ha="left")
        cells.append(x)

    # memory passed from one cell to the next
    def memory(x0, x1, label=None):
        fs.arrow(ax, (x0, CELL_Y), (x1, CELL_Y), color=fs.BLUE, lw=0.8)
        if label:
            fs.text(ax, (x0 + x1) / 2, CELL_Y + 0.07, label, size=fs.FOOT, va="bottom")

    half = CELL_W / 2
    fs.text(ax, 0.05, CELL_Y, "$h_0$")
    memory(0.3, cells[0] - half)
    memory(cells[0] + half, cells[1] - half, "$h_1$")
    memory(cells[1] + half, cells[2] - half, "$h_2$")
    memory(cells[2] + half, cells[2] + half + 0.95, "$h_3$")
    fs.dots(ax, 7.15, CELL_Y)
    memory(7.75, cells[3] - half, "$h_{T-1}$")

    # last memory -> fully connected layer -> activities
    fc_x, fc_w, fc_h = 12.85, 2.2, 0.75
    fs.rect(ax, fc_x - fc_w / 2, CELL_Y - fc_h / 2, fc_w, fc_h, fc=fs.LIGHT, lw=0.7, z=3)
    fs.text(ax, fc_x, CELL_Y, "livello compl.\nconnesso", size=fs.FOOT)
    memory(cells[3] + half, fc_x - fc_w / 2, "$h_T$")
    fs.arrow(ax, (fc_x, CELL_Y + fc_h / 2), (fc_x, CELL_Y + fc_h / 2 + 0.85))
    fs.text(ax, fc_x, CELL_Y + fc_h / 2 + 0.92, "6 attività\n(softmax)", size=fs.FOOT, va="bottom")

    # legend
    fs.text(ax, 0.9, -2.3, "$x_t$: le ampiezze dell’istante $t$ (coppie $\\times$ sottoportanti)",
            size=fs.FOOT, color=fs.GREY, ha="left")
    fs.text(ax, 0.9, -2.72, "$h_t$: la memoria (stato nascosto) dopo aver letto i primi $t$ istanti",
            size=fs.FOOT, color=fs.GREY, ha="left")

    fs.save(fig, "schema_rnn")


if __name__ == "__main__":
    main()
