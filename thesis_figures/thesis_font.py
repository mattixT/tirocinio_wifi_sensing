"""
Common look of all the figures of the thesis.

Importing this module (import thesis_font) sets matplotlib so that every figure uses:
    - the font of the thesis, Latin Modern (package lmodern). The PDFs are written by pdflatex
      itself, so words, numbers and formulas look exactly as in the text; if LaTeX is not found,
      the Latin Modern files of the folder fonts/ are used, which look almost the same;
    - the decimal comma on the axes (0,5 instead of 0.5), as in the text of the thesis;
    - text sizes for a figure drawn at its final size: with size(...) the figure has the width it
      has in the thesis, so 8 pt in matplotlib are 8 pt on the page.

Usage in a plotting script:
    import thesis_font
    fig, ax = plt.subplots(figsize=thesis_font.size(1.0, 6))   # whole text width, 6 cm high
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


# ------------------------------------------------------------------ font
# 1. On screen and when LaTeX is not available: the Latin Modern files of the folder fonts/.
FONTS = Path(__file__).resolve().parent / "fonts"
FONT = "Latin Modern Roman"
for font_file in FONTS.glob("*.otf"):
    font_manager.fontManager.addfont(str(font_file))
# if other sizes of Latin Modern are installed, keep only the 10 pt design of the folder fonts/
font_manager.fontManager.ttflist = [f for f in font_manager.fontManager.ttflist
                                    if f.name != FONT or Path(f.fname).parent == FONTS]
try:    # stop with an error instead of silently using another font
    font_manager.findfont(font_manager.FontProperties(family=FONT), fallback_to_default=False)
except ValueError:
    raise SystemExit(f"Latin Modern not found: the folder {FONTS} must contain the .otf files")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [FONT, "DejaVu Serif"],    # DejaVu Serif only for symbols missing in Latin Modern
    "mathtext.fontset": "custom",            # formulas with the same font as the text ...
    "mathtext.rm": FONT,
    "mathtext.it": f"{FONT}:italic",
    "mathtext.bf": f"{FONT}:bold",
    "mathtext.cal": FONT,
    "mathtext.sf": FONT,
    "mathtext.fallback": "cm",               # ... and Computer Modern for the symbols it does not have
    "pdf.fonttype": 3,                       # Latin Modern is an OpenType (CFF) font: Type 3 embeds it correctly
    "axes.unicode_minus": True,
})

# 2. The PDFs, when LaTeX is installed: they are produced by pdflatex itself (PGF backend), with
#    the package lmodern as in the thesis. Words, numbers and formulas look exactly as in the text,
#    and only the letters that are used are embedded, so the files stay small.
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
                         r"\usepackage{icomma}"                    # 0,5 without a space after the comma
                         r"\DeclareUnicodeCharacter{2192}{\ensuremath{\rightarrow}}"
                         r"\DeclareUnicodeCharacter{2248}{\ensuremath{\approx}}"
                         r"\DeclareUnicodeCharacter{2264}{\ensuremath{\leq}}"
                         r"\DeclareUnicodeCharacter{2265}{\ensuremath{\geq}}"
                         r"\DeclareUnicodeCharacter{03C3}{\ensuremath{\sigma}}"),
    })
    register_backend("pdf", "matplotlib.backends.backend_pgf")      # savefig(...pdf) goes through pdflatex

# ------------------------------------------------------------------ decimal comma on the axes
try:
    locale.setlocale(locale.LC_NUMERIC, "it_IT.UTF-8")
    plt.rcParams["axes.formatter.use_locale"] = True
except locale.Error:
    print("thesis_font: Italian locale not available, the axes keep the decimal point")

# ------------------------------------------------------------------ sizes (figure at its final size)
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
    """Mix a colour with white: tint(BLUE, 0.12) keeps 12% of the colour (light fill)."""
    r, g, b = to_rgb(color)
    return (1 - amount * (1 - r), 1 - amount * (1 - g), 1 - amount * (1 - b))


# fixed colour per model, the same in every figure
MODELS = {"cnn5": "CNN-5", "gru": "GRU", "lstm": "LSTM"}
COLORS = {"cnn5": "#2a78d6", "gru": "#eb6834", "lstm": "#1baf7a"}


def number(value, decimals=None):
    """Number with the decimal comma, written as a formula (as in the text of the thesis):
    number(0.5) -> '$0{,}5$', number(-2.5) -> '$-2{,}5$', number(0.88, 2) -> '$0{,}88$'."""
    s = f"{value:.{decimals}f}" if decimals is not None else f"{value:g}"
    return "$" + s.replace(".", "{,}") + "$"


def bold(s):
    """Bold piece of a longer text, e.g. bold("(a)") + " Vista dall’alto della stanza"."""
    return rf"\textbf{{{s}}}" if USE_LATEX else rf"$\mathbf{{{s}}}$"


def comma(value, decimals=2, sign=False):
    """Number as text with the decimal comma and the typographic minus: comma(-6.15, 1, True) -> '−6,2'."""
    s = f"{value:+.{decimals}f}" if sign else f"{value:.{decimals}f}"
    return s.replace(".", ",").replace("-", "\u2212")


def percent():
    """The % sign for a text of the figure (with LaTeX it must be written \\%)."""
    return r"\%" if USE_LATEX else "%"


def pdf_fonts(path):
    """Names of the fonts embedded in a PDF (to check that only Latin Modern is used)."""
    data = Path(path).read_bytes()
    chunks = [data]
    for stream in re.findall(rb"stream\r?\n(.*?)endstream", data, re.S):    # compressed parts
        try:
            chunks.append(zlib.decompress(stream))
        except zlib.error:
            pass
    pattern = rb"/(?:BaseFont|FontName)\s*/(?:[A-Z]{6}\+)?([A-Za-z0-9-]+)"
    return sorted({n.decode() for chunk in chunks for n in re.findall(pattern, chunk)})


def save(fig, name, folder=IMAGES):
    """Save the figure as <folder>/<name>.pdf and print the fonts it contains."""
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / f"{name}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out}  [text: {'LaTeX' if USE_LATEX else 'Latin Modern from fonts/'}; "
          f"fonts: {', '.join(pdf_fonts(out))}]")
