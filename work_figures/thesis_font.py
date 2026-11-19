"""
Common look of all the figures of the thesis.
"""
import glob
import locale
import os
import re
import shutil
import zlib
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backend_bases import register_backend
from matplotlib.colors import to_rgb

BASE = Path("/Users/mattiatagliola/Desktop/TIROCINIO_TESI")
DATASET = BASE / "03_DATASET" / "NTU-Fi_HAR"
EXPERIMENTS = BASE / "04_ESPERIMENTI"
IMAGES = BASE / "07_RELAZIONE" / "images"          # figures used by the thesis

TEXT_WIDTH_CM = 13.92                              # \textwidth of the thesis (sapthesis, oneside): 396 pt

def size(fraction=1.0, height_cm=6.0):
    """figsize (inches) of a figure that takes `fraction` of the text width and is height_cm high.
    It must match the width of \\includegraphics, e.g. size(0.72, 6) for width=0.72\\textwidth."""
    return fraction * TEXT_WIDTH_CM / 2.54, height_cm / 2.54

FONTS = Path(__file__).resolve().parent / "fonts"
FONT = "Latin Modern Roman"
for font_file in FONTS.glob("*.otf"):
    font_manager.fontManager.addfont(str(font_file))
font_manager.fontManager.ttflist = [f for f in font_manager.fontManager.ttflist
                                    if f.name != FONT or Path(f.fname).parent == FONTS]
try:   
    font_manager.findfont(font_manager.FontProperties(family=FONT), fallback_to_default=False)
except ValueError:
    raise SystemExit(f"Latin Modern not found: the folder {FONTS} must contain the .otf files")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [FONT, "DejaVu Serif"],  
    "mathtext.fontset": "custom",            
    "mathtext.rm": FONT,
    "mathtext.it": f"{FONT}:italic",
    "mathtext.bf": f"{FONT}:bold",
    "mathtext.cal": FONT,
    "mathtext.sf": FONT,
    "mathtext.fallback": "cm",               
    "pdf.fonttype": 3,                       
    "axes.unicode_minus": True,
})

TEX_DIRS = ["/Library/TeX/texbin", *sorted(glob.glob("/usr/local/texlive/*/bin/*"), reverse=True)]
LATEX_PROGRAMS = ("pdflatex", "kpsewhich")

def find_latex():
    """True if LaTeX can be used. Programs started from the Dock (e.g. an editor) often do not have
    the TeX folder in the PATH: in that case it is added here."""
    if all(shutil.which(p) for p in LATEX_PROGRAMS):
        return True
    for folder in TEX_DIRS:
        if all((Path(folder) / p).exists() for p in LATEX_PROGRAMS):
            os.environ["PATH"] = folder + os.pathsep + os.environ.get("PATH", "")
            return True
    return False

USE_LATEX = find_latex()
if USE_LATEX:
    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "pgf.rcfonts": False,
        "pgf.preamble": (r"\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{amsmath}"
                         r"\usepackage{icomma}"                   
                         r"\DeclareUnicodeCharacter{2192}{\ensuremath{\rightarrow}}"
                         r"\DeclareUnicodeCharacter{2248}{\ensuremath{\approx}}"
                         r"\DeclareUnicodeCharacter{2264}{\ensuremath{\leq}}"
                         r"\DeclareUnicodeCharacter{2265}{\ensuremath{\geq}}"
                         r"\DeclareUnicodeCharacter{03C3}{\ensuremath{\sigma}}"),
    })
    register_backend("pdf", "matplotlib.backends.backend_pgf")      

try:
    locale.setlocale(locale.LC_NUMERIC, "it_IT.UTF-8")
    plt.rcParams["axes.formatter.use_locale"] = True
except locale.Error:
    print("thesis_font: Italian locale not available, the axes keep the decimal point")

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.titlesize": 8.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "grid.linewidth": 0.5,
    "patch.linewidth": 0.6,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})

# colours of the schematic figures
INK = "#1A1A1A"     # text and main lines
GREY = "#7A7A7A"    # secondary elements
LIGHT = "#EDEDED"   # fills
BLUE = "#2F6496"    # main elements of the model
RED = "#A8443A"     # result / dynamic element

def tint(color, amount):
    r, g, b = to_rgb(color)
    return (1 - amount * (1 - r), 1 - amount * (1 - g), 1 - amount * (1 - b))

# fixed colour per model, the same in every figure
MODELS = {"cnn5": "CNN-5", "gru": "GRU", "lstm": "LSTM"}
COLORS = {"cnn5": "#2a78d6", "gru": "#eb6834", "lstm": "#1baf7a"}

def number(value, decimals=None):
    s = f"{value:.{decimals}f}" if decimals is not None else f"{value:g}"
    return "$" + s.replace(".", "{,}") + "$"

def bold(s):
    return rf"\textbf{{{s}}}" if USE_LATEX else rf"$\mathbf{{{s}}}$"

def comma(value, decimals=2, sign=False):
    s = f"{value:+.{decimals}f}" if sign else f"{value:.{decimals}f}"
    return s.replace(".", ",").replace("-", "\u2212")

def percent():
    return r"\%" if USE_LATEX else "%"

def pdf_fonts(path):
    data = Path(path).read_bytes()
    chunks = [data]
    for stream in re.findall(rb"stream\r?\n(.*?)endstream", data, re.S):
        try:
            chunks.append(zlib.decompress(stream))
        except zlib.error:
            pass
    pattern = rb"/(?:BaseFont|FontName)\s*/(?:[A-Z]{6}\+)?([A-Za-z0-9-]+)"
    return sorted({n.decode() for chunk in chunks for n in re.findall(pattern, chunk)})

def save(fig, name, folder=IMAGES):
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / f"{name}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out}  [text: {'LaTeX' if USE_LATEX else 'Latin Modern from fonts/'}; "
          f"fonts: {', '.join(pdf_fonts(out))}]")
