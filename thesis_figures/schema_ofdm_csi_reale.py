"""
Figure (chapter 2): (a) OFDM subcarriers: each one has its maximum where all the others are zero;
(b) CSI amplitude of a real recording of NTU-Fi HAR (walk0.mat, antenna pair 1): one row is a
subcarrier over time, one column is a packet.

Output: images/schema_ofdm_csi_reale.pdf
Usage (from the repository root): python thesis_figures/schema_ofdm_csi_reale.py
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

import thesis_font as tf
from thesis_font import BLUE, GREY, tint

SAMPLE = tf.DATASET / "train_amp" / "walk" / "walk0.mat"
N_SUB = 7                                                # subcarriers drawn in (a)
RATE = 500                                               # packets per second


def panel_ofdm(ax):
    f = np.linspace(0, N_SUB + 1, 2000)
    for k in range(1, N_SUB + 1):
        ax.axvline(k, color="#dddddd", lw=0.5, zorder=0)
        y = np.where(np.abs(f - k) <= 1, np.sinc(f - k), 0)        # main lobe of the subcarrier k
        ax.plot(f, y, color=tint(BLUE, 0.4 + 0.6 * (k - 1) / (N_SUB - 1)), lw=1.1)
    ax.annotate("al centro di una sottoportante\nle altre valgono zero", xy=(4, 1.0), xytext=(4.25, 1.32),
                fontsize=6.5, color=GREY, va="bottom",
                arrowprops=dict(arrowstyle="->", color=GREY, lw=0.6))
    ax.set_xticks(range(1, N_SUB + 1), [f"$f_{k}$" for k in range(1, N_SUB + 1)])
    ax.set_yticks([])
    ax.set_xlim(0, N_SUB + 1)
    ax.set_ylim(-0.05, 1.75)
    ax.set_xlabel("frequenza")
    ax.set_title(tf.bold("(a)") + " Sottoportanti OFDM", loc="left")
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)


def panel_csi(fig, ax):
    csi = loadmat(SAMPLE)["CSIamp"].reshape(3, 114, 2000)[0]       # antenna pair 1, all the packets
    seconds = csi.shape[1] / RATE
    im = ax.imshow(csi, aspect="auto", cmap="viridis", vmin=0, vmax=55, extent=(0, seconds, 114, 0))
    ax.axhline(30, color="white", ls="--", lw=0.6)
    ax.text(0.05, 28, "una riga: una sottoportante nel tempo", color="white", fontsize=6.5, va="bottom")
    ax.axvline(1, color="white", ls="--", lw=0.6)
    ax.text(1.05, 110, "una colonna: un pacchetto", color="white", fontsize=6.5, va="bottom")
    ax.set_xticks(np.arange(0, seconds + 0.01, 0.5))
    ax.set_xlabel(f"tempo (s) — {csi.shape[1]} pacchetti, {RATE} al secondo")
    ax.set_ylabel("sottoportante")
    ax.set_title(tf.bold("(b)") + " Ampiezza del CSI di una registrazione reale", loc="left")
    fig.colorbar(im, ax=ax, label="ampiezza", fraction=0.05, pad=0.03)


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=tf.size(1.0, 5.6), layout="constrained",
                                   gridspec_kw={"width_ratios": [1, 1.45]})
    panel_ofdm(ax1)
    panel_csi(fig, ax2)
    tf.save(fig, "schema_ofdm_csi_reale")


if __name__ == "__main__":
    main()
