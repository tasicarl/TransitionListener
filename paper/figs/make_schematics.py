"""Pedagogical schematics of the paper (no data):
  fig_pipeline.pdf  the computation from the rate to the pulsar-timing likelihood as a
                    flowchart with pictograms, and every modelling choice of figures 7 and 8 as a
                    labelled chip at the stage where it acts (R: rate, C: cosmology, D: data)
  fig_levers.pdf    (not in the paper) what the data fix: amplitude, position of the band on the
                    rising side and the shape of the rising side

fig_pipeline is drawn on a canvas of the full text width, in points (one unit of the axes is one
point on the page), and passes figure_checks.gate (text over text and over busy backgrounds,
text of at least 5 pt) plus the checks of its own: every chip text inside its chip, every
pictogram inside its stage and clear of the title, the arrows and the chips.
Run: python make_schematics.py
"""
import os
import sys
import time

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch, Ellipse, Wedge, Rectangle, Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from figure_checks import TEXTWIDTH_IN, gate, size_table  # noqa: E402

DESYdunkelblau = (0 / 255, 74 / 255, 110 / 255)
DESYcyan = (0 / 255, 159 / 255, 223 / 255)
DESYorange = (241 / 255, 143 / 255, 31 / 255)
DESYgruen = (0 / 255, 168 / 255, 62 / 255)
DESYviolett = (146 / 255, 124 / 255, 184 / 255)
DESYrot = (230 / 255, 65 / 255, 35 / 255)
DESYgelb = (230 / 255, 169 / 255, 15 / 255)
DESYmagenta = (208 / 255, 0 / 255, 111 / 255)
DESYlila = (82 / 255, 78 / 255, 156 / 255)
DESYdunkelbraun = (133 / 255, 88 / 255, 62 / 255)
GREY = (0.45, 0.45, 0.45)
# the three groups of modelling choices, in the colours of figures 7 and 8 (make_figures.py)
C_R, C_C, C_S, C_D = DESYcyan, DESYorange, DESYmagenta, DESYgruen
UPS = "ϒ"        # the upsilon with curled arms (as TeX's \Upsilon); DejaVu's mathtext \Upsilon reads as "Y"
mpl.rcParams.update({"font.size": 8, "axes.linewidth": 0.8, "font.family": "sans-serif",
                     "mathtext.fontset": "dejavusans", "figure.dpi": 150})

CHIP_FS = 5.5         # chip text, points as printed (>= 5)
TAB_W = 15.0          # width of the coloured tab with the symbol
LINE1, LINE2 = 12.5, 19.0   # height of a chip with one and with two lines of text
CHIP_GAP = 2.5


# ------------------------------------------------------------------ building blocks
def _wrap(ax, text, fs, width):
    """The text on one line if it fits `width` points, else broken at the word that gives the
    two most even lines (a break inside $...$ is not allowed)."""
    rend = ax.figure.canvas.get_renderer()
    w = lambda s: ax.text(0, 0, s, fontsize=fs)
    t = w(text); tw = t.get_window_extent(rend).width * 72 / ax.figure.dpi; t.remove()
    if tw <= width:
        return text, 1
    words = text.split(" ")
    best = None
    for i in range(1, len(words)):
        a, b = " ".join(words[:i]), " ".join(words[i:])
        if a.count("$") % 2:
            continue
        ta, tb = w(a), w(b)
        wa = ta.get_window_extent(rend).width * 72 / ax.figure.dpi
        wb = tb.get_window_extent(rend).width * 72 / ax.figure.dpi
        ta.remove(); tb.remove()
        if max(wa, wb) <= width and (best is None or max(wa, wb) < best[0]):
            best = (max(wa, wb), a + "\n" + b)
    assert best is not None, f"chip text {text!r} does not fit on two lines of {width:.0f} pt"
    return best[1], 2


def chip(ax, x, y_top, w, label, text, colour, chips):
    """A rounded chip: the symbol in white on a coloured tab, its meaning next to it, both
    vertically centred. Returns the bottom of the chip."""
    avail = w - TAB_W - 6.0
    text, nl = _wrap(ax, text, CHIP_FS, avail)
    assert nl == 1, f"chip {label} needs two lines: {text!r}"
    h = LINE1 if nl == 1 else LINE2
    y = y_top - h / 2
    ax.add_patch(FancyBboxPatch((x, y_top - h), w, h, boxstyle="round,pad=0,rounding_size=2.2",
                                fc="white", ec=colour, lw=0.7, zorder=5))
    ax.add_patch(FancyBboxPatch((x, y_top - h), TAB_W, h, boxstyle="round,pad=0,rounding_size=2.2",
                                fc=colour, ec=colour, lw=0.7, zorder=6))
    ax.add_patch(Rectangle((x + TAB_W - 2.5, y_top - h), 2.5, h, fc=colour, ec=colour, lw=0.7, zorder=6))
    # one line is centred on its baseline, so that a line with brackets or subscripts sits at the
    # same height as one without; two lines are centred on their box
    t1 = ax.text(x + TAB_W / 2, y, label, ha="center", va="center_baseline", fontsize=CHIP_FS, color="white", zorder=7)
    t2 = ax.text(x + TAB_W + 3.0, y, text, ha="left", va="center_baseline" if nl == 1 else "center", fontsize=CHIP_FS,
                 color="0.15", zorder=7, linespacing=1.15)
    if nl == 1:
        # a line with a superscript is taller than the chip allows on its baseline: centre its box
        bb = t2.get_window_extent(ax.figure.canvas.get_renderer())
        y0_, y1_ = (v * 72 / ax.figure.dpi for v in (bb.y0, bb.y1))
        y0_ += ax.get_ylim()[0]; y1_ += ax.get_ylim()[0]
        if y1_ > y_top - 0.8 or y0_ < y_top - h + 0.8:
            t2.set_va("center")
    chips.append(((x, y_top - h, x + w, y_top), (x, y_top - h, x + TAB_W, y_top), t1, t2))
    return y_top - h


TITLE_PT, TITLE_DY = 8.0, 14.0          # title font size, and the centre of the two-line title block below the box top


def stage_box(ax, x, y0, w, h, title, colour):
    """The rounded box of a stage with its bold title, centred in a block of two lines at the top (a
    title too long for one line carries an explicit line break)."""
    ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0,rounding_size=5",
                                fc=(*colour, 0.07), ec=(*colour, 0.9), lw=1.0, zorder=1))
    return ax.text(x + w / 2, y0 + h - TITLE_DY, title, ha="center", va="center", multialignment="center",
                   fontsize=TITLE_PT, color=colour, fontweight="bold", linespacing=1.15, zorder=3)


def arrow(ax, x0, y0, x1, y1, colour="0.3"):
    return ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>,head_length=4.5,head_width=3",
                                        lw=1.6, color=colour, zorder=8, shrinkA=0, shrinkB=0))


# ------------------------------------------------------------------ pictograms (u: points per unit)
def picto_potential(ax, cx, cy, u):
    """V(phi) with a barrier, a false vacuum at zero and a deeper true vacuum, and the path
    of the thermal nucleation over the barrier."""
    phi = np.linspace(-0.12, 1.18, 300)
    V = phi**2 * (1 - phi) ** 2 - 0.12 * phi**3 * (4 - 3 * phi)       # false vacuum at 0, deeper true vacuum at 1
    V = V / (V.max() - V.min()) * 0.75
    X, Y = cx - 0.7 * u + 1.15 * u * phi, cy - 0.32 * u + V * u
    out = [ax.plot(X, Y, color=DESYdunkelblau, lw=1.4, zorder=4)[0]]
    i0 = np.argmin(abs(phi)); i1 = np.argmin(abs(phi - 1.0))
    out.append(ax.add_patch(Circle((X[i0], Y[i0] + 0.06 * u), 0.05 * u, fc=DESYcyan, ec=DESYdunkelblau, lw=0.6, zorder=6)))
    # a dashed shaft, and a solid head: the same arrow drawn solid and clipped to a disc around its tip
    ends = dict(xy=(X[i1] - 0.05 * u, Y[i1] + 0.1 * u), xytext=(X[i0] + 0.08 * u, Y[i0] + 0.12 * u))
    out.append(ax.annotate("", **ends, arrowprops=dict(arrowstyle="-", color=DESYrot, lw=0.9, connectionstyle="arc3,rad=-0.45",
                                                       linestyle=(0, (2, 1.5)), shrinkB=5), zorder=5))
    head = ax.annotate("", **ends, arrowprops=dict(arrowstyle="-|>,head_length=0.45,head_width=0.24", color=DESYrot, lw=0.9,
                                                   connectionstyle="arc3,rad=-0.45"), zorder=5)
    head.arrow_patch.set_clip_path(Circle(ends["xy"], 0.15 * u, transform=ax.transData))
    out.append(head)
    return out


def picto_bubbles(ax, cx, cy, u, seed=3):
    """Percolating bubbles, drawn as the outline of their union: every circle with a
    thick edge first, then every circle filled without edge on top."""
    rng = np.random.default_rng(seed)
    # scattered over a nearly square patch, so that no direction is preferred
    bubbles = [(rng.uniform(-0.72, 0.72), rng.uniform(-0.5, 0.5), rng.uniform(0.08, 0.22)) for _ in range(22)]
    mx = np.mean([b[0] for b in bubbles]); my = np.mean([b[1] for b in bubbles])
    out = []
    for x, y, r in bubbles:
        out.append(ax.add_patch(Circle((cx + (x - mx) * u, cy + (y - my) * u), r * u, fc=DESYdunkelblau, lw=0, zorder=4)))
    for x, y, r in bubbles:
        out.append(ax.add_patch(Circle((cx + (x - mx) * u, cy + (y - my) * u), r * u - 0.8, fc=(0.80, 0.91, 0.97), lw=0, zorder=5)))
    return out


def picto_plasma(ax, cx, cy, u):
    """A thermometer rising in a hot plasma: reheating."""
    out = [ax.add_patch(FancyBboxPatch((cx - 0.07 * u, cy - 0.25 * u), 0.14 * u, 0.7 * u,
                                       boxstyle=f"round,pad=0,rounding_size={0.07 * u}", fc="white", ec="0.3", lw=0.8, zorder=4)),
           ax.add_patch(Circle((cx, cy - 0.3 * u), 0.13 * u, fc=DESYrot, ec="0.3", lw=0.8, zorder=5)),
           ax.add_patch(Rectangle((cx - 0.035 * u, cy - 0.3 * u), 0.07 * u, 0.62 * u, fc=DESYrot, lw=0, zorder=5))]
    rng = np.random.default_rng(7)
    for _ in range(26):
        x, y = cx + rng.uniform(-0.55, 0.55) * u, cy + rng.uniform(-0.4, 0.45) * u
        if abs(x - cx) < 0.2 * u:
            continue
        out.append(ax.add_patch(Circle((x, y), 0.028 * u, fc=rng.choice([DESYorange, DESYgelb, DESYrot]), lw=0,
                                       zorder=4, alpha=0.8)))
    return out


def picto_collision(ax, cx, cy, u):
    """Two colliding bubbles with the shells around them (in the colour of the source block) and the
    emitted gravitational waves."""
    out = []
    for dx in (-0.28, 0.28):                       # both bubbles first, then both shells on top
        out.append(ax.add_patch(Circle((cx + dx * u, cy), 0.34 * u, fc=(*DESYcyan, 0.18), ec=DESYdunkelblau, lw=0.9, zorder=4)))
    for dx in (-0.28, 0.28):
        out.append(ax.add_patch(Circle((cx + dx * u, cy), 0.39 * u, fc="none", ec=DESYmagenta, lw=1.4, zorder=5, alpha=0.9)))
    for r in (0.57, 0.64, 0.71):                   # clear of the shells, which reach 0.6 u from the centre
        out.append(ax.add_patch(Wedge((cx, cy), r * u, 60, 120, width=0.015 * u, fc=DESYviolett, lw=0, zorder=4)))
        out.append(ax.add_patch(Wedge((cx, cy), r * u, 240, 300, width=0.015 * u, fc=DESYviolett, lw=0, zorder=4)))
    return out


def picto_redshift(ax, cx, cy, u):
    """A wave whose wavelength grows from left to right: the redshift to today."""
    x = np.linspace(0.0, 1.0, 400)
    phase = 2 * np.pi * (14 * x - 9.5 * x**2)            # wavelength grows along the train
    return [ax.plot(cx - 0.75 * u + 1.5 * u * x, cy + 0.09 * u * np.sin(phase), color=DESYviolett, lw=1.0, zorder=4)[0]]


def picto_telescope(ax, cx, cy, u, tilt=30.0):
    """A radio telescope: a parabolic dish, tilted by `tilt` degrees towards the upper left, with
    the feed on struts at its focus, on a pedestal and a base. (cx, cy) is the vertex of the dish."""
    t = np.radians(tilt)
    rot = lambda x, y: (cx + x * np.cos(t) - y * np.sin(t), cy + x * np.sin(t) + y * np.cos(t))
    xd = np.linspace(-0.26, 0.26, 60) * u
    yin = xd**2 / (0.9 * u)                                     # focal length 0.225 u
    yout = yin - 0.045 * u
    X, Y = rot(np.concatenate([xd, xd[::-1]]), np.concatenate([yin, yout[::-1]]))
    out = [ax.add_patch(Polygon(np.column_stack([X, Y]), closed=True, fc="0.88", ec="0.25", lw=0.7, zorder=6))]
    fx, fy = rot(0.0, 0.225 * u)
    for sx in (-1, 1):
        rx, ry = rot(sx * 0.24 * u, 0.24**2 * u / 0.9)
        out.append(ax.plot([rx, fx], [ry, fy], color="0.3", lw=0.5, zorder=6)[0])
    out.append(ax.add_patch(Rectangle((fx - 0.025 * u, fy - 0.025 * u), 0.05 * u, 0.05 * u, angle=0.0, fc="0.3", lw=0, zorder=7)))
    # pedestal from the back of the dish down to a base
    bx, by = rot(0.0, -0.045 * u)
    out.append(ax.plot([bx, cx], [by, cy - 0.3 * u], color="0.3", lw=1.3, zorder=5)[0])
    out.append(ax.add_patch(Polygon([(cx - 0.16 * u, cy - 0.36 * u), (cx + 0.16 * u, cy - 0.36 * u),
                                     (cx + 0.09 * u, cy - 0.29 * u), (cx - 0.09 * u, cy - 0.29 * u)],
                                    closed=True, fc="0.45", lw=0, zorder=5)))
    return out


def picto_pta(ax, cx, cy, u):
    """A radio telescope, pulsars with beams around it, a black-hole binary with its orbit at the
    upper right, and a noisy timing residual below."""
    out = picto_telescope(ax, cx, cy + 0.02 * u, 1.15 * u)
    for ang in (35, 145, 215, 325):
        a = np.radians(ang)
        px, py = cx + 0.78 * u * np.cos(a), cy + 0.5 * u * np.sin(a)
        # the line of sight, from the pulsar towards the telescope, stopping short of the dish
        out.append(ax.plot([px, cx + 0.45 * (px - cx)], [py, cy + 0.45 * (py - cy)], color="0.6", lw=0.5,
                           ls=(0, (1, 1.5)), zorder=3)[0])
        out.append(ax.plot([px], [py], marker="*", color=DESYgelb, ms=6.5, mec="0.3", mew=0.4, zorder=5)[0])
    # the binary: two black holes on a tilted orbit, with two arcs of the emitted waves
    bx, by = cx + 1.08 * u, cy + 0.42 * u
    out.append(ax.add_patch(Ellipse((bx, by), 0.36 * u, 0.13 * u, angle=-15, fc="none", ec="0.45", lw=0.5, ls=(0, (2, 1.2)), zorder=4)))
    for sx in (-1, 1):
        out.append(ax.add_patch(Circle((bx + sx * 0.17 * u * np.cos(np.radians(15)), by - sx * 0.17 * u * np.sin(np.radians(15))),
                                       0.045 * u, fc="0.1", lw=0, zorder=5)))
    for r in (0.27, 0.34):
        out.append(ax.add_patch(Wedge((bx, by), r * u, 195, 240, width=0.015 * u, fc="0.45", lw=0, zorder=4)))
    tt = np.linspace(0, 1, 120)
    rng = np.random.default_rng(1)
    res = 0.09 * np.sin(2 * np.pi * 1.3 * tt) + 0.03 * rng.standard_normal(120)
    out.append(ax.plot(cx - 0.6 * u + 1.2 * u * tt, cy - 0.72 * u + res * u, color=DESYgruen, lw=0.6, zorder=5)[0])
    return out


def _extent(fig, arts):
    """The bounding box of a pictogram in data coordinates, the arrows of annotations included."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    bbs = []
    for a in arts:
        if getattr(a, "arrow_patch", None) is not None:
            bbs.append(a.arrow_patch.get_window_extent(rend))
        else:
            bbs.append(a.get_window_extent(rend))
    inv = arts[0].axes.transData.inverted()
    (x0, y0), (x1, y1) = inv.transform([(min(b.x0 for b in bbs), min(b.y0 for b in bbs)),
                                        (max(b.x1 for b in bbs), max(b.y1 for b in bbs))])
    return x0, y0, x1, y1


def centred(fig, draw, ax, cx, cy, u):
    """Draw a pictogram so that the centre of its bounding box lies at (cx, cy): draw it once,
    measure it, remove it and draw it again shifted by the offset."""
    arts = draw(ax, cx, cy, u)
    x0, y0, x1, y1 = _extent(fig, arts)
    for a in arts:
        a.remove()
    return draw(ax, cx + cx - 0.5 * (x0 + x1), cy + cy - 0.5 * (y0 + y1), u)


# ------------------------------------------------------------------ the figure
def fig_pipeline():
    Wc = TEXTWIDTH_IN * 72.0                       # the canvas in points
    margin, gap = 1.5, 4.0
    W = (Wc - 2 * margin - 3 * gap) / 4
    xs = [margin + i * (W + gap) for i in range(4)]
    # the labels of figure 6 (make_figures.FIG_LABELS), shortened so that every chip is one line
    rate = [("N1", r"$\mu_4$, $\mu_3$ halved or doubled"),
            ("N2", r"$A_{\rm dyn}\times10$ or $/10$"), ("N3", "LO instead of NLO"),
            ("N4", r"$V_{\rm 4D}$ + Daisy, $\mu = \pi T$"), ("N5", r"$V_{\rm 4D}$ + Daisy, $\mu = v$")]
    perc = [("P1", "Expansion history"),
            ("P2", r"$f_{\rm perc} = 0.63$ instead of $0.29$"), ("P3", r"Decreasing $a^3P_{\rm f}$ at $T_{\rm p}$"),
            ("P4", r"$(\beta/H)_S$ for $(\beta/H)_{RH}$, $(\beta/H)_\Gamma$"), ("P5", "Instantaneous reheating")]
    src = [("G1", "Spectra without expansion"), ("G2", "Free-streaming shells"),
           ("G3", "Higgsless template"), ("G4", "Hybrid template"), ("G5", "Envelope approximation")]
    data = [("B1", "Binary background model"), ("B2", "Hellings–Downs spectrum"),
            ("B3", "DMGP instead of DMX"), ("B4", "Number of frequency bins"),
            ("B5", "PTA data set")]
    # vertical layout from the top, in points: the boxes (title, pictograms,
    # chips); the height of the canvas follows from the tallest list of chips
    top_gap = 2.0                                  # no legend: each block has its own colour
    pic_top, pic_h, chip_gap_top = 32.0, 80.0, 8.0
    # a first pass on a scratch axes measures the chips
    probe = plt.figure(figsize=(TEXTWIDTH_IN, 4.0)); pax = probe.add_axes([0, 0, 1, 1])
    pax.set_xlim(0, Wc); pax.set_ylim(0, 4.0 * 72)
    heights = []
    for items in (rate, perc, src, data):
        y, scratch = 0.0, []
        for lab, txt in items:
            y = chip(pax, 0, y, W - 6, lab, txt, C_R, scratch) - CHIP_GAP
        heights.append(-y)
    plt.close(probe)
    box_h = pic_top + pic_h + chip_gap_top + max(heights) + 4.0
    Hc = top_gap + box_h + margin
    fig = plt.figure(figsize=(TEXTWIDTH_IN, Hc / 72.0))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, Wc); ax.set_ylim(0, Hc); ax.axis("off")
    box_top = Hc - top_gap
    y0 = box_top - box_h
    boxes = [("Nucleation rate", C_R), ("Percolation and\nreheating", C_C),
             ("Gravitational\nwave source", C_S), ("Binaries and\npulsar data", C_D)]
    titles = [stage_box(ax, x, y0, W, box_h, title, col) for x, (title, col) in zip(xs, boxes)]
    # pictograms, centred in the area between the title and the chips
    pc = box_top - pic_top - pic_h / 2
    u = 44.0
    pictos = [centred(fig, picto_potential, ax, xs[0] + W / 2, pc, u),
              centred(fig, picto_bubbles, ax, xs[1] + W / 2, pc, 1.15 * u),
              centred(fig, picto_collision, ax, xs[2] + W / 2, pc, 0.95 * u),
              centred(fig, picto_pta, ax, xs[3] + W / 2, pc, 0.9 * u)]
    chips = []
    for x, items, col in zip(xs, (rate, perc, src, data), (C_R, C_C, C_S, C_D)):
        y = box_top - pic_top - pic_h - chip_gap_top
        for lab, txt in items:
            y = chip(ax, x + 3, y, W - 6, lab, txt, col, chips) - CHIP_GAP
    # the checks of the schematic itself
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    to_pt = lambda bb: (bb.x0 * 72 / fig.dpi, bb.y0 * 72 / fig.dpi, bb.x1 * 72 / fig.dpi, bb.y1 * 72 / fig.dpi)
    extra = []
    for (cx0, cy0, cx1, cy1), (tx0, ty0, tx1, ty1), t1, t2 in chips:
        for t, (a0, b0, a1, b1), pad in ((t1, (tx0, cy0, tx1, cy1), 1.0), (t2, (tx1, cy0, cx1, cy1), 1.0)):
            x0, y0_, x1, y1 = to_pt(t.get_window_extent(rend))
            if not (x0 >= a0 + pad and x1 <= a1 - pad and y0_ >= b0 + 0.5 and y1 <= b1 - 0.5):
                extra.append(f"chip text {t.get_text()!r} not inside its chip")
            mid = 0.5 * (y0_ + y1) - 0.5 * (b0 + b1)
            if abs(mid) > 1.5:
                extra.append(f"chip text {t.get_text()!r} off centre by {mid:+.1f} pt")
    for i, t in enumerate(titles):                 # bold titles keep 3 pt from the edges of their box
        x0, _, x1, _ = to_pt(t.get_window_extent(rend))
        if x0 < xs[i] + 3 or x1 > xs[i] + W - 3:
            extra.append(f"title {t.get_text()!r} closer than 3 pt to its box")
    for i, arts in enumerate(pictos):
        bbs = [to_pt(a.get_window_extent(rend)) for a in arts]
        px0, py0 = min(b[0] for b in bbs), min(b[1] for b in bbs)
        px1, py1 = max(b[2] for b in bbs), max(b[3] for b in bbs)
        tb = to_pt(titles[i].get_window_extent(rend))
        top_chip = max(c[0][3] for c in chips if xs[i] <= c[0][0] <= xs[i] + W)
        if not (px0 >= xs[i] + 3 and px1 <= xs[i] + W - 3 and py1 <= tb[1] - 3 and py0 >= top_chip + 3):
            extra.append(f"pictogram of stage {i + 1} leaves its area: x {px0:.0f}..{px1:.0f} of "
                         f"{xs[i]:.0f}..{xs[i] + W:.0f}, y {py0:.0f}..{py1:.0f} between {top_chip:.0f} and {tb[1]:.0f}")
    gate(fig, "fig_pipeline.pdf", min_pt=5.0, extra=extra)
    fig.savefig(os.path.join(HERE, "fig_pipeline.pdf")); plt.close(fig); print("wrote fig_pipeline.pdf")


# ------------------------------------------------------------------ the levers (not in the paper)
def bpl(f, fp, a=2.0, b=2.0, c=2.0, A=1.0):
    x = f / fp
    return A * (a + b) ** c / (b * x ** (-a / c) + a * x ** (b / c)) ** c


def fig_levers():
    f = np.logspace(-9.3, -6.8, 400)
    band = (2e-9, 2.8e-8)
    data_f = np.geomspace(2.5e-9, 2.4e-8, 7)
    data_y = bpl(data_f, 2.5e-8, A=1.0e-8) * np.array([1.2, 0.8, 1.1, 0.9, 1.15, 0.85, 1.05])
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35), sharey=True, gridspec_kw={"wspace": 0.06})
    specs = [
        ("amplitude $\\propto (H/\\beta)^2$",
         [(bpl(f, 2.5e-8, A=1.0e-8), DESYdunkelblau, "-", r"slower, $\beta/H = 10$"),
          (bpl(f, 3.6e-8, A=0.35e-8), DESYcyan, "--", r"faster, $\beta/H = 15$")]),
        ("peak frequency $\\propto (\\beta/H)\\,T_{\\rm reh}$",
         [(bpl(f, 2.5e-8, A=1.0e-8), DESYdunkelblau, "-", r"$v$"),
          (bpl(f, 5.0e-8, A=1.0e-8), DESYorange, "--", r"$2v$")]),
        ("slope of the rising side",
         [(bpl(f, 2.5e-8, A=1.0e-8), DESYdunkelblau, "-", r"$a = 2$ (central)"),
          (bpl(f, 2.5e-8, a=1.0, b=2.0, c=1.0, A=1.0e-8), DESYviolett, "--", r"$a = 1$"),
          (bpl(f, 2.5e-8, a=3.0, b=1.0, c=1.9, A=1.0e-8), DESYrot, ":", r"$a = 3$")]),
    ]
    for ax, (title, curves) in zip(axes, specs):
        ax.axvspan(*band, color=DESYgruen, alpha=0.12, lw=0)
        ax.errorbar(data_f, data_y, yerr=0.45 * data_y, fmt="o", ms=2.2, color=DESYgruen, lw=0.7, capsize=0)
        for y, col, ls, lab in curves:
            ax.plot(f, y, color=col, ls=ls, lw=1.5, label=lab)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(3e-11, 5e-8); ax.set_xlim(f[0], f[-1])
        ax.set_title(title, fontsize=7.2, pad=3)
        ax.set_xlabel(r"$f$", fontsize=8); ax.set_xticks([]); ax.set_yticks([])
        ax.minorticks_off()
        ax.legend(loc="lower right", frameon=False, fontsize=6)
        ax.text(2.3e-9, 2.6e-8, "PTA band", fontsize=6, color=DESYgruen)
    axes[0].set_ylabel(r"$h^2\Omega_{\rm GW}$", fontsize=8)
    for ax, txt in zip(axes, (r"$\beta/H$ depends on $g$: moves the region in $g$",
                              r"$T_{\rm reh} \propto v$: moves the region in $v$",
                              r"set by the source template: moves it in $v$")):
        ax.text(0.5, -0.12, txt, transform=ax.transAxes, fontsize=6.5, ha="center", va="top", color="0.3", style="italic")
    fig.savefig(os.path.join(HERE, "fig_levers.pdf"), bbox_inches="tight"); plt.close(fig); print("wrote fig_levers.pdf")


if __name__ == "__main__":
    t_start = time.time()
    fig_pipeline()
    out = os.path.join(HERE, "fig_pipeline.pdf")
    assert os.path.exists(out) and os.path.getmtime(out) >= np.floor(t_start), "fig_pipeline.pdf not written in this run"
    size_table([(out, 1.0)])
    print("schematics build complete, all checks passed, written in this run: fig_pipeline.pdf")
