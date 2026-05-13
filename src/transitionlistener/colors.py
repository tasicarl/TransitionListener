"""Color palettes and Matplotlib helpers used across TransitionListener plots.

The constants exposed here encode the corporate identity color palettes that the
project uses for plots in the documentation as well as publication-quality
figures. A couple of helper utilities are provided to make it easy to size
figures consistently and to configure Matplotlib's global font settings.

Part of TransitionListener v2.0
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from cycler import cycler

RWTHblau = (0 / 255, 84 / 255, 159 / 255)
RWTHblau75 = (0 / 255, 84 / 255, 159 / 255, 0.75)
RWTHblau50 = (0 / 255, 84 / 255, 159 / 255, 0.5)
RWTHblau25 = (0 / 255, 84 / 255, 159 / 255, 0.25)
RWTHblau10 = (0 / 255, 84 / 255, 159 / 255, 0.1)
RWTHblau20 = (0 / 255, 84 / 255, 159 / 255, 0.2)
RWTHblau40 = (0 / 255, 84 / 255, 159 / 255, 0.4)
RWTHblau30 = (0 / 255, 84 / 255, 159 / 255, 0.3)
RWTHblau60 = (0 / 255, 84 / 255, 159 / 255, 0.6)
RWTHschwarz = (0 / 255, 0 / 255, 0 / 255)
RWTHmagenta = (227 / 255, 0 / 255, 102 / 255)
RWTHgelb = (255 / 255, 237 / 255, 0 / 255)
RWTHpetrol = (0 / 255, 97 / 255, 101 / 255)
RWTHpetrol25 = (0 / 255, 97 / 255, 101 / 255, 0.25)
RWTHtuerkis = (0 / 255, 152 / 255, 161 / 255)
RWTHgruen = (87 / 255, 171 / 255, 39 / 255)
RWTHgruen10 = (87 / 255, 171 / 255, 39 / 255, 0.1)
RWTHgruen25 = (87 / 255, 171 / 255, 39 / 255, 0.25)
RWTHmaigruen = (189 / 255, 205 / 255, 0 / 255)
RWTHmaigruen25 = (189 / 255, 205 / 255, 0 / 255, 0.25)
RWTHmaigruen50 = (189 / 255, 205 / 255, 0 / 255, 0.5)
RWTHorange = (246 / 255, 168 / 255, 0 / 255)
RWTHorange25 = (246 / 255, 168 / 255, 0 / 255, 0.25)
RWTHorange50 = (246 / 255, 168 / 255, 0 / 255, 0.5)
RWTHrot = (204 / 255, 7 / 255, 30 / 255)
RWTHrot25 = (204 / 255, 7 / 255, 30 / 255, 0.25)
RWTHrot50 = (204 / 255, 7 / 255, 30 / 255, 0.5)
RWTHbordeaux = (161 / 255, 16 / 255, 53 / 255)
RWTHbordeaux50 = (161 / 255, 16 / 255, 53 / 255, 0.5)
RWTHbordeaux25 = (161 / 255, 16 / 255, 53 / 255, 0.25)
RWTHbordeaux10 = (161 / 255, 16 / 255, 53 / 255, 0.1)
RWTHviolett = (97 / 255, 33 / 255, 88 / 255)
RWTHviolett50 = (97 / 255, 33 / 255, 88 / 255, 0.5)
RWTHlila = (122 / 255, 111 / 255, 172 / 255)

DESYcyan = (0 / 255, 159 / 255, 223 / 255)
DESYcyan50 = (0 / 255, 159 / 255, 223 / 255, 0.5)
DESYcyan25 = (0 / 255, 159 / 255, 223 / 255, 0.25)
DESYcyan75 = (0 / 255, 159 / 255, 223 / 255, 0.75)
DESYcyan10 = (0 / 255, 159 / 255, 223 / 255, 0.10)
DESYrot = (230 / 255, 65 / 255, 35 / 255)
DESYrot50 = (230 / 255, 65 / 255, 35 / 255, 0.5)
DESYdunkelrot = (153 / 255, 0 / 255, 0 / 255)
DESYorange = (241 / 255, 143 / 255, 31 / 255)
DESYorange25 = (241 / 255, 143 / 255, 31 / 255, 0.25)
DESYdunkelblau = (0 / 255, 74 / 255, 110 / 255)
DESYaubergine = (139 / 255, 60 / 255, 126 / 255)
DESYaubergine50 = (139 / 255, 60 / 255, 126 / 255, 0.5)
DESYviolett = (146 / 255, 124 / 255, 184 / 255)
DESYviolett50 = (146 / 255, 124 / 255, 184 / 255, 0.5)
DESYlila = (82 / 255, 78 / 255, 156 / 255)
DESYlila50 = (82 / 255, 78 / 255, 156 / 255, 0.5)
DESYhellgruen = (122 / 255, 158 / 255, 31 / 255)
DESYhellgruen50 = (122 / 255, 158 / 255, 31 / 255, 0.5)
DESYtuerkis = (45 / 255, 164 / 255, 164 / 255)
DESYmagenta = (208 / 255, 0 / 255, 111 / 255)
DESYolive = (158 / 255, 150 / 255, 0 / 255)
DESYhellbraun = (170 / 255, 132 / 255, 106 / 255)
DESYdunkelbraun = (133 / 255, 88 / 255, 62 / 255)
DESYdunkelolive = (89 / 255, 98 / 255, 29 / 255)
DESYpetrol = (0 / 255, 106 / 255, 133 / 255)
DESYgruen = (0 / 255, 168 / 255, 62 / 255)
DESYgelb = (230 / 255, 169 / 255, 15 / 255)

HumboldtGreen = (129 / 255, 238 / 255, 169 / 255)

invisible = (255 / 255, 255 / 255, 255 / 255, 0)

colors = [DESYcyan, DESYcyan25]
cmapDESYcyan = LinearSegmentedColormap.from_list("mycmap", colors)

colors = [DESYorange, DESYorange25]
cmapDESYorange = LinearSegmentedColormap.from_list("mycmap", colors)

colors = [DESYcyan25, DESYcyan]
cmapNG = LinearSegmentedColormap.from_list("mycmap", colors)

colors = [DESYorange25, DESYorange]
cmapIPTA = LinearSegmentedColormap.from_list("mycmap", colors)

NG = DESYcyan
IPTA = DESYorange


colors = [DESYorange, DESYcyan]
cmapDESYcyanorange = LinearSegmentedColormap.from_list("mycmap", colors)

colors = [
    DESYpetrol,
    DESYtuerkis,
    DESYgruen,
    DESYhellgruen,
    DESYorange,
    DESYrot,
    DESYdunkelrot,
    DESYviolett,
    DESYlila,
]
cmapDESYrainbow = LinearSegmentedColormap.from_list("DESYrainbow", colors)

colors = [
    DESYorange,
    HumboldtGreen
]
cmapTL = LinearSegmentedColormap.from_list("TL", colors)

colors = [
    DESYorange,
    HumboldtGreen,
    DESYcyan
]
cmapTL2 = LinearSegmentedColormap.from_list("TL2", colors)

colors = [RWTHblau10, RWTHblau]
cmapRWTHblau = LinearSegmentedColormap.from_list("mycmap", colors)

colors = [RWTHblau25, RWTHblau]
cmapRWTHblau2 = LinearSegmentedColormap.from_list("mycmap", colors)

colors = [RWTHblau50, RWTHblau]
cmapRWTHblau3 = LinearSegmentedColormap.from_list("mycmap", colors)

colors = [RWTHgruen10, RWTHgruen]
cmapRWTHgruen = LinearSegmentedColormap.from_list("mycmap", colors)

colors = [DESYhellgruen, invisible]
cmapDESYhellgruen = LinearSegmentedColormap.from_list("mycmap", colors)

colors = [RWTHbordeaux10, RWTHbordeaux]
cmapRWTHbordeaux = LinearSegmentedColormap.from_list("mycmap", colors)

colors = [
    RWTHpetrol,
    RWTHtuerkis,
    RWTHgruen,
    RWTHmaigruen,
    RWTHorange,
    RWTHrot,
    RWTHbordeaux,
    RWTHviolett,
    RWTHlila,
]
cmapRWTHrainbow = LinearSegmentedColormap.from_list("mycmap", colors)

colors = [
    RWTHtuerkis,
    RWTHgruen,
    RWTHmaigruen,
    RWTHorange,
    RWTHrot,
    RWTHbordeaux,
    RWTHviolett,
]
cmapRWTHrainbow2 = LinearSegmentedColormap.from_list("mycmap", colors)

colors = [RWTHgruen, RWTHmaigruen, RWTHorange, RWTHrot, RWTHbordeaux, RWTHviolett]
cmapRWTHrainbow3 = LinearSegmentedColormap.from_list("mycmap", colors)

colors = [RWTHblau, RWTHrot]
cmapRWTHtemp = LinearSegmentedColormap.from_list("mycmap", colors)

colors = ["white", "white"]
cmapWhite = LinearSegmentedColormap.from_list("mycmap", colors)

colors = [
    (255 / 255, 255 / 255, 255 / 255, 0),
    (200 / 255, 200 / 255 / 255, 200 / 255, 0.1),
    (150 / 255, 150 / 255, 150 / 255, 0.4),
    (100 / 255, 100 / 255, 100 / 255, 0.5),
    (50 / 255, 50 / 255, 50 / 255, 0.6),
    (0, 0, 0, 0.75),
]
cmapGreyshade = LinearSegmentedColormap.from_list("mycmap", colors)


def set_size(width: float | str, fraction : float = 1, subplots: tuple[int, int] = (1, 1)) -> tuple[float, float]:
    """Return figure dimensions that avoid downstream scaling artefacts.

    Parameters
    ----------
    width: float | str
        Either the Matplotlib figure width in points or a preset identifier
        such as ``\"JCAP\"`` or ``\"beamer\"`` that maps to a known text width.
    fraction: float, optional
        Fraction of the available width the figure should occupy. Defaults to
        ``1`` which uses the full width.
    subplots: tuple[int, int], optional
        Tuple describing the subplot layout (rows, columns). The information is
        used to slightly enlarge multi-panel figures so that axis labels do not
        overlap.

    Returns
    -------
    fig_dim: tuple[float, float]
        Figure width and height in inches.
    """
    if width == "JCAP":
        width_pt = 440.0
    elif width == "beamer":
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt

    # Figure height in inches
    if subplots == (1, 3):
        fig_height_in = fig_width_in * golden_ratio / 2 * 1.2
    elif subplots != (1, 1):
        fig_height_in = (
            fig_width_in * golden_ratio * 1.2
        )  # to include title, some extra space
    else:  # (1,1) case
        fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def set_font():
    """Update Matplotlib's global ``rcParams`` with the project's stylistic defaults.

    The configuration mirrors the publication layout used in the documentation:
    LaTeX-rendered text, a sans-serif font family, harmonised font sizes for titles
    and axis labels, and a colour cycle based on the DESY palette defined in this
    module. Calling this function is idempotent and affects figures created
    afterwards.
    """
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
            "mathtext.fontset": "dejavusans",
            "axes.labelsize": 10,
            "font.size": 10,
            "axes.titlesize": 10,
            "figure.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "lines.markersize": 5,
            "axes.prop_cycle": cycler(
                color=[
                    DESYcyan,
                    DESYorange,
                    DESYmagenta,
                    DESYgruen,
                    DESYrot,
                    DESYdunkelblau,
                    DESYviolett,
                    DESYhellgruen,
                    DESYgelb,
                ]
            ),
        }
    )
