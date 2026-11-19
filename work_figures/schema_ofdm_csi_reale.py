"""
Figure (chapter 2): OFDM subcarriers and CSI amplitude. 

Output: images/schema_ofdm_csi_reale.pdf
Usage (from the repository root): python work_figures/schema_ofdm_csi_reale.py
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

import thesis_font as tf
from thesis_font import BLUE, GREY, tint

SAMPLE = tf.DATASET / "train_amp" / "walk" / "walk0.mat"
N_SUB = 7                                               
RATE = 500                                              

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
    ax.set_xlabel("Frequenza")
    ax.set_title(tf.bold("(a)"), loc="left")
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)

def panel_csi(fig, ax):
    csi = loadmat(SAMPLE)["CSIamp"].reshape(3, 114, 2000)[0]       # antenna pair 1, all the packets
    n_packets = csi.shape[1]
    im = ax.imshow(csi, aspect="auto", cmap="viridis", vmin=0, vmax=55, extent=(0, n_packets, 114, 0))
    ax.axhline(30, color="white", ls="--", lw=0.6)
    ax.text(25, 28, "una riga: una sottoportante nel tempo", color="white", fontsize=6.5, va="bottom")
    ax.axvline(500, color="white", ls="--", lw=0.6)
    ax.text(525, 110, "una colonna: un pacchetto", color="white", fontsize=6.5, va="bottom")
    ticks = range(0, n_packets + 1, 500)
    ax.set_xticks(ticks, [str(t) for t in ticks])                 
    ax.set_xlabel(f"Pacchetto")
    ax.set_ylabel("Sottoportante")
    ax.set_title(tf.bold("(b)"), loc="left")
    fig.colorbar(im, ax=ax, label="Ampiezza CSI", fraction=0.05, pad=0.03)

def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=tf.size(1.0, 5.6), layout="constrained",
                                   gridspec_kw={"width_ratios": [1, 1.45]})
    panel_ofdm(ax1)
    panel_csi(fig, ax2)
    tf.save(fig, "schema_ofdm_csi_reale")

if __name__ == "__main__":
    main()
