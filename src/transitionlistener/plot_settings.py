"""Matplotlib defaults and helper utilities for rendering TransitionListener plots.

Part of TransitionListener v2.0
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

from .colors import *
from cycler import cycler
import matplotlib.pyplot as plt
import numpy as np
import importlib.resources
import PIL

plot_settings = {
    "text.usetex": False,
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
    "hatch.linewidth" : 4.5,
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

TLcmap = cmapTL2 #cmapDESYrainbow or cmapTL or cmapTL2

logo_settings = {
    "logo_size": 0.27,
    "logo_xy_ratio": 0.5
}

def _load_logo_array(filename: str):
    """Load a logo image from the package resources as a NumPy array."""
    resource = importlib.resources.files("transitionlistener.logo") / filename
    with importlib.resources.as_file(resource) as path:
        with PIL.Image.open(path) as image:
            return np.array(image)


def load_small_logo():
    """Return the small TransitionListener logo as an RGBA array."""
    return _load_logo_array('TL-logo_small.png')


def load_large_logo():
    """Return the large TransitionListener logo as an RGBA array."""
    return _load_logo_array('TL-logo_large.png')

def add_TL_logo(loc : str, magnification : float = 1, ax : plt.Axes = None,
                zorder : int = 3, size : float = logo_settings["logo_size"],
                small_logo : bool = True):
    """Inset the TransitionListener logo into a Matplotlib axes.

    Parameters
    ----------
    loc : str
        Where to place the inset (``\"upper right\"``, ``\"upper left\"``, etc.).
    magnification : float, optional
        Scale factor applied to the default logo size.
    ax : matplotlib.axes.Axes, optional
        Target axes. Defaults to ``plt.gca()`` if omitted.
    zorder : int, optional
        Drawing order for the inset axes.
    size : float, optional
        Baseline inset size expressed as a fraction of the parent axes.
    small_logo : bool, optional
        Choose between the small horizontal logo and the large square variant.
    """
    size = size * magnification
    size_x = size
    if small_logo:
        size_y = size * logo_settings["logo_xy_ratio"]
    else:
        size_y = size
        
    if ax is None:
        ax = plt.gca()
    if loc == "upper right":
        inset = ax.inset_axes([1-size_x, 1-size_y, size_x, size_y], zorder=zorder)
        inset.set_aspect("equal", anchor="NE")
    elif loc == "upper left":
        inset = ax.inset_axes([0, 1-size_y, size_x, size_y], zorder=zorder)
        inset.set_aspect("equal", anchor="NW")
    elif loc == "lower right":
        inset = ax.inset_axes([1-size_x, 0, size_x, size_y], zorder=zorder)
        inset.set_aspect("equal", anchor="SE")
    elif loc == "lower left":
        inset = ax.inset_axes([0, 0, size_x, size_y], zorder=zorder)
        inset.set_aspect("equal", anchor="SW")
    if loc == "outside":
        inset = ax.inset_axes([1.05, 1-size_y, size_x, size_y], zorder=zorder)
        inset.set_aspect("equal", anchor="SW")
        
    
    if small_logo:
        inset.imshow(load_small_logo(), interpolation='none')
    else:
        inset.imshow(load_large_logo(), interpolation='none')
    inset.axis('off')
