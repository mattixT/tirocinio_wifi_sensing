"""
Shared style and drawing helpers for the schematic figures.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from scipy.io import loadmat

# font of the thesis and colours of the schematic figures, shared with all the other figures
from thesis_font import BASE, BLUE, GREY, IMAGES, INK, LIGHT, RED, USE_LATEX, bold, number, pdf_fonts, tint 

SAMPLE = BASE / "03_DATASET" / "NTU-Fi_HAR" / "train_amp" / "walk" / "walk0.mat"  # CSI used in the figures

SMALL = 9      
FOOT = 8           
SCRIPT = 7         
LARGE = 14         

plt.rcParams.update({
    "font.size": SMALL,
    "lines.scale_dashes": False,    
    "hatch.linewidth": 0.5,
})

def canvas(x0, x1, y0, y1):
    """Empty figure where the axes go from (x0, y0) to (x1, y1) in centimetres."""
    fig = plt.figure(figsize=((x1 - x0) / 2.54, (y1 - y0) / 2.54))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return fig, ax

def save(fig, name):
    """Save the figure as images/<name>.pdf, cropped to its content."""
    for ax in fig.axes:
        if not ax.axison:                
            for artist in ax.get_children():
                if artist is not ax.patch:
                    artist.set_clip_on(False)
    IMAGES.mkdir(parents=True, exist_ok=True)
    out = IMAGES / f"{name}.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"saved {out}  [text: {'LaTeX' if USE_LATEX else 'Latin Modern from fonts/'}; fonts: {', '.join(pdf_fonts(out))}]")

def text(ax, x, y, s, size=SMALL, color=INK, ha="center", va="center", bold=False, linespacing=1.25, **kw):
    """Text in (x, y). With LaTeX the bold is written as \\textbf (line by line)."""
    weight = "normal"
    if bold and USE_LATEX:
        s = "\n".join(rf"\textbf{{{line}}}" for line in s.split("\n"))
    elif bold:
        weight = "bold"
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va,
                   fontweight=weight, linespacing=linespacing, zorder=10, **kw)

def line(ax, points, color=INK, lw=0.6, dashes=None, z=2):
    xs, ys = zip(*points)
    kw = {"dashes": dashes} if dashes else {}
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="butt", solid_joinstyle="miter", zorder=z, **kw)

def arrow(ax, p0, p1, color=INK, lw=0.6, both=False, z=3, path=None):
    """Straight arrow from p0 to p1 (or along a matplotlib Path, if given)."""
    style = "<|-|>" if both else "-|>"
    kw = {"path": path} if path is not None else {"posA": p0, "posB": p1}
    ax.add_patch(FancyArrowPatch(**kw, arrowstyle=f"{style},head_length=5,head_width=1.8",
                                 mutation_scale=1, shrinkA=0, shrinkB=0, color=color, lw=lw,
                                 capstyle="butt", joinstyle="miter", zorder=z))

def rect(ax, x, y, w, h, fc="white", ec=INK, lw=0.6, z=2, **kw):
    """Rectangle with the lower-left corner in (x, y)."""
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, zorder=z, **kw))

def node(ax, x, y, r, fc="white", ec=INK, lw=0.6, label=None, size=SMALL, z=4):
    """Circle (neuron) centred in (x, y), optionally with a label inside."""
    ax.add_patch(Circle((x, y), r, fc=fc, ec=ec, lw=lw, zorder=z))
    if label:
        text(ax, x, y, label, size=size)

def dots(ax, x, y, vertical=False, gap=0.14, r=0.022, color=INK):
    """Three dots centred in (x, y), horizontal (like \\cdots) or vertical (like \\vdots)."""
    for k in (-1, 0, 1):
        cx, cy = (x, y + k * gap) if vertical else (x + k * gap, y)
        ax.add_patch(Circle((cx, cy), r, fc=color, ec="none", zorder=10))

def between(p, q, rp=0.0, rq=0.0):
    """End points of a segment from p to q shortened by rp at the start and rq at the end
    (used to make arrows start and stop on the border of a circle)."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    u = (q - p) / np.linalg.norm(q - p)
    return tuple(p + u * rp), tuple(q - u * rq)

def load_sample():
    """CSI amplitude of SAMPLE as (antenna pairs, subcarriers, time) = (3, 114, 500):
    one packet every 4, as in the preprocessing of the experiments."""
    csi = loadmat(SAMPLE)["CSIamp"].reshape(3, 114, 2000)
    return csi[:, :, ::4]

def heatmap(ax, matrix, x, y, w, h, z=1):
    """Draw a CSI matrix (subcarriers x time) in the rectangle with lower-left corner (x, y),
    with the same colour scale used in the rest of the thesis (viridis, 0-55)."""
    ax.imshow(matrix, cmap="viridis", vmin=0, vmax=55, extent=(x, x + w, y, y + h),
              interpolation="none", zorder=z)
    ax.set_aspect("equal")
