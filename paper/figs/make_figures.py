"""Stage A: every figure and every \\input table of main.tex from the shipped results in paper/data.

  fig_pipeline.pdf      the schematic of the pipeline (make_schematics.py)
  fig_rate.pdf          the exponent of the nucleation rate against T at g = 0.5 (LO, NLO and the 4D daisy rates with
                        their scale bands), and T_p and (beta/H)_RH against g on the benchmark line (fig_transition)
  fig_spectra.pdf       spectra at the NANOGrav best fit (default, G1 to G5) against the free-spectrum violins of the data
  fig_plane.pdf         the (g, v) plane: preferred regions, contours of (beta/H)_RH, vacuum trapping,
                        the local criterion, the template range and the edge of the 4D daisy rate
  fig_hardening.pdf     the preference for the phase transition against the binary lifetime
  fig_budget.pdf        movement of the NANOGrav 68% region under each choice
  fig_foreground.pdf    the preferred regions with the binary foreground (appendix E)
  ../switch_numbers.tex, ../switch_numbers_ipta.tex   the numbers behind the budget (tables of appendix E)
  ../budget_table.tex   the theory budget along the line (budget_table.py)

A missing input stops the build with FileNotFoundError. The figures draw what the analysis wrote; the few fields they derive from it by a computation (the
transition table on fine grids, the spectra of fig_spectra) are kept in figure_data/ (see _cached), so that
redrawing computes nothing once they are there.
Every figure is drawn at the width at which main.tex includes it (FRAC) and saved on that canvas,
not with a tight bounding box, so that a point in the code is a point on the page. Every figure
passes figure_checks.gate (text over text and over busy backgrounds, legends over data, clearance
from the frames, sizes of at least 6 pt) before it is saved; the build ends by asserting that every
output was written in this run and that every PDF has its included width.
Run: python make_figures.py [names...]
"""
import glob
import hashlib
import inspect
import json
import os
import sys

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from figure_checks import TEXTWIDTH_IN, check_clearance, check_text, freeze, gate, size_table  # noqa: E402

# TransitionListener / DESY colours (src/transitionlistener/colors.py)
DESYdunkelblau = (0 / 255, 74 / 255, 110 / 255)
DESYcyan = (0 / 255, 159 / 255, 223 / 255)
DESYpetrol = (0 / 255, 106 / 255, 133 / 255)
DESYorange = (241 / 255, 143 / 255, 31 / 255)
DESYrot = (230 / 255, 65 / 255, 35 / 255)
DESYgruen = (0 / 255, 168 / 255, 62 / 255)
DESYviolett = (146 / 255, 124 / 255, 184 / 255)
DESYgelb = (230 / 255, 169 / 255, 15 / 255)
DESYhellgruen = (122 / 255, 158 / 255, 31 / 255)
DESYmagenta = (208 / 255, 0 / 255, 111 / 255)
DESYlila = (82 / 255, 78 / 255, 156 / 255)
DESYdunkelbraun = (133 / 255, 88 / 255, 62 / 255)
DESYaubergine = (139 / 255, 60 / 255, 126 / 255)
GREY = (0.45, 0.45, 0.45)

mpl.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.dpi": 150,
    "axes.grid": False, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "xtick.minor.visible": True, "ytick.minor.visible": True,
    "axes.linewidth": 0.8, "font.family": "sans-serif", "mathtext.fontset": "dejavusans",
    "axes.unicode_minus": True, "legend.frameon": False,
})
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))          # the repository (src/ for the plasma tables)
PAPER = os.path.dirname(HERE)
DATA = os.path.join(PAPER, "data")                               # every input of the figures and tables
REPRODUCE = os.path.join(PAPER, "reproduce")                     # the code of stage B, for its tables and models
CAMPAIGN = os.path.join(DATA, "linescan_points")


def _need(path):
    """`path`, which must exist: a missing input stops the build instead of changing a figure."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"input of the figures missing: {path} (see paper/README.md, stage B)")
    return path


def _reproduce_on_path():
    """paper/reproduce/paths.py on sys.path, which puts every folder of stage B on it as well."""
    if REPRODUCE not in sys.path:
        sys.path.insert(0, REPRODUCE)
    import paths
    return paths
# The data sets in purple and yellow (NG = DESYlila, IPTA = DESYgelb; cyan and orange are the colours of the
# nucleation and percolation steps in figures 1, 6 and 7), in every figure that shows them (4, 5, 8); the violet
# lines of figures 4 and 5 are brown next to them. The rate variants keep one colour in every figure: NLO dark blue,
# LO orange, V_4D + Daisy at pi T dark red-orange, at the vacuum scale magenta (also their edges in figure 5); LO and
# V_4D + Daisy overlap, so they get hues far apart, and yellow, green and violet, which read poorly as thin lines or
# bands, are avoided; the three groups of modelling choices (figures 3, 7, 8) keep their colours.
DESYdunkelrot = (153 / 255, 0 / 255, 0 / 255)
C_LO, C_DET, C_4D = DESYorange, DESYdunkelblau, (190 / 255, 60 / 255, 20 / 255)   # N4 dark red-orange
C_4D6 = DESYmagenta           # N5: the 4D daisy rate with the couplings fixed at the vacuum (N4 runs them to pi T)
C_NG, C_IPTA = DESYlila, DESYgelb
C_PP, C_COSMO, C_DATA = DESYcyan, DESYorange, DESYgruen
MPL_GEV = 2.435e18
H100_HZ = 100e3 / 3.0857e22
UPS = "ϒ"        # the upsilon with curled arms (as TeX's \Upsilon); DejaVu's mathtext \Upsilon reads as "Y"

# the fraction of \textwidth at which main.tex includes each figure
FRAC = {"fig_rate": 1.0, "fig_transition": 0.95, "fig_spectra": 1.0, "fig_plane": 1.0,
        "fig_budget": 1.0, "fig_foreground": 0.85, "fig_hardening": 0.6}


# figures drawn smaller than they are printed, so that LaTeX scales their text up to that of the page (8 pt
# labels at 4/3 print at 10.7 pt): the printed width over the drawn width
SCALE = {"fig_spectra": 4 / 3, "fig_plane": 1.15, "fig_budget": 1.12}


def width(name):
    """The drawn width in inches: the printed width over SCALE."""
    return FRAC[name] * TEXTWIDTH_IN / SCALE.get(name, 1.0)


def save(fig, name, extra=()):
    """Freeze the layout, run the checks (raising on any finding), and save on the fixed canvas."""
    gate(fig, name, extra=extra)
    fig.savefig(os.path.join(HERE, name))
    plt.close(fig)
    print("wrote", name)


def mixing_total(row):
    """The mixing sum over J >= 0: a row without the flag includes_l0 excludes the J = 0 term, which is
    then added from mixing_l0 (reproduce/linescan/nlo_table_one.py, same rule)."""
    m = row["mixing"]
    return m["value"] + (0.0 if m.get("includes_l0") else row["mixing_l0"])


LINESCAN = os.environ.get("LINESCAN", "linescan_u1_mu0_1GeV.json")   # the fermion-free campaign by default
SCAN_NPZ = os.environ.get("SCAN_NPZ", "plane_scan_u1.npz")


def linescan():
    return json.load(open(_need(os.path.join(DATA, LINESCAN))))


def _tuple_legend(**kw):
    return dict(handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)}, **kw)


def _first_clean(fig, text, candidates):
    """Place `text` (a Text artist) at the first of `candidates` (dicts of Text properties) at
    which the golden-rule check reports nothing about it, and return the chosen candidate."""
    for cand in candidates:
        cand = dict(cand)
        if "xy" in cand:                               # an annotation may also move the point it labels
            text.xy = cand.pop("xy")
        text.update(cand)
        text.set_visible(True)
        if not [b for b in check_text(fig) + check_clearance(fig) if repr(text.get_text().replace("$", "")[:46]) in b]:
            return cand
    raise SystemExit(f"no clean position for {text.get_text()!r}")


# ---------------------------------------------------------------- figure 1
def _point_dir(g):
    """The line-scan point of the coupling g (summary and Delta S tables at mu4 x 0.5, 1, 2)."""
    for d in glob.glob(os.path.join(CAMPAIGN, f"g{g:.3f}_u1_v*")):
        for mu in ("0.5", "1", "2"):
            _need(os.path.join(d, f"table_mu{mu}.json"))
        return d, "fermion-free campaign, $\\mu_0 = 1$ GeV"
    raise FileNotFoundError(f"input of the figures missing: the line-scan point g = {g:.3f} in {CAMPAIGN}")


def _exact_table_model():
    """reproduce/rate/build_exact_table.py (the exponents of the percolation tables, from the action caches in
    data/action_cache)."""
    _reproduce_on_path()
    import build_exact_table
    return build_exact_table


def _equal_gaps(fig, axs, gap_pt=14.0):
    """Equal axes widths and equal white space between each panel and the decorations (tick and axis labels) of the
    next one, the outer edges of the decorations kept where they are."""
    rend = fig.canvas.get_renderer()
    W = fig.bbox.width
    pos = [a.get_position() for a in axs]
    tb = [a.get_tightbbox(rend) for a in axs]
    left = [p.x0 * W - t.x0 for p, t in zip(pos, tb)]          # decorations left of each axes, px
    right = [t.x1 - p.x1 * W for p, t in zip(pos, tb)]
    x_lo, x_hi = tb[0].x0, tb[-1].x1
    gap = gap_pt * fig.dpi / 72
    w = (x_hi - x_lo - sum(left) - sum(right) - gap * (len(axs) - 1)) / len(axs)
    x = x_lo
    for a, p, l_, r_ in zip(axs, pos, left, right):
        a.set_position([(x + l_) / W, p.y0, w / W, p.height])
        x += l_ + w + r_ + gap


P4_LABEL_G = (0.43, 0.44)   # the P4 label on its line within this range of g
BESIDE_TAGS = {"N5": (-0.45, 0.25)}    # edges labelled beside themselves, in this log10 v range of the centre
EDGE_ARROW_FRACS = {"N5": (0.12, 0.85)}   # clear of the N5 label beside the edge
G6_ARROW_LV = {1: np.log10(18.0)}   # the upper arrow of the GW-template line at v = 18 GeV, between the T_reh = 600 MeV guide and the 2 sigma line
P4_ARROW_LV = {1: 1.0}        # the upper P4 arrow at v = 10 GeV, above the regions the data prefer
P4_LABEL_C = -0.62            # log10 v of the centre of the P4 label: low on the line, over the T_reh line
PLANE_EDGES = False    # draw the N4 and N5 edges of vacuum trapping in fig_plane
TAG_PAD = 0.8           # white margin around the N4 and N5 tags, in points, small so that the box cuts little
C_TREH = "0.55"         # the T_reh bound, lighter than the frame
TREH_G = 0.49          # the T_reh label sits on its line at this coupling
TREH_GUIDES = [0.06, 0.6]   # further T_reh contours in GeV, named on the right axis
LS_TREH_GUIDE = (0, (5, 1.5, 1, 1.5))   # dash-dotted: dotted is (beta/H)_RH, dashed the binaries
PERC_DX = 1.04         # the Percolation label this factor left of the LO percolation line
FS_ALONG = 6.2         # the curve labels of the exponent panel, in points
LS_LO, LS_4D = "--", ":"   # the curves overlap: LO dashed, N4 dotted, NLO solid and on top; N5 solid
Z_NLO = 2.6
LABEL_X_4D6 = 0.17     # the N5 label centred at this multiple of the NLO percolation temperature
LABEL_DY_4D6 = 3.3     # the baseline of the N5 label, in points above its curve
LABEL_X_LO = 1.2       # the LO label at this multiple of the NLO percolation temperature
LABEL_X_4D = 0.60      # the label of N4 in the exponent panel, at this multiple of the LO percolation temperature,
LABEL_DY_4D = 2.5      # and its baseline this many points above the upper edge of its band and of the LO band


def fig_rate():
    """Figure 2. Panel a: the exponent of the nucleation rate against T in keV at g = 0.5, at leading order
    (LO matching and the LO bounce) and at next-to-leading order (NLO matching, the bounce and the determinant
    correction), each with its own matching-scale band, and the 4D daisy rate with the couplings run to pi T
    (with its band) and fixed at the vacuum scale (without a band); vertical lines at the percolation
    temperature of each. Panels b and c: T_p and beta/H along the line (fig_transition)."""
    # three panels: the exponents, then T_p and beta/H along the line
    fig, ax_all = plt.subplots(1, 3, figsize=(width("fig_rate"), 2.55), layout="constrained",
                               gridspec_kw=dict(width_ratios=[1, 1, 1]))
    ax0 = ax_all[0]
    axes = [ax0]
    ymin, ymax = np.inf, -np.inf
    curves = {}
    for col, g in enumerate((0.500,)):          # one coupling
        d, tag = _point_dir(g)
        # the exponents exactly as the percolation tables use them (reproduce/rate/build_exact_table.Model): S_3 on the full temperature grid of the action caches plus the
        # correction Delta S interpolated between its five tabulated temperatures. NLO: S_3 of the
        # NLO-matched theory plus Delta S, at mu_4 and with mu_4 halved and doubled; LO: the LO
        # bounce on the LO-matched theory, with its own matching-scale band (kinds lom05, lom2)
        BET = _exact_table_model()
        m = BET.Model(g)
        T = m.T
        S_det = m.exponent("nlo")
        edges = [m.exponent("mu05"), m.exponent("mu2")]
        S_lo = m.exponent("fulllo")
        edges_lo = []
        for kind in ("lom05", "lom2"):
            rows_k = json.load(open(_need(os.path.join(_reproduce_on_path().ACTION_CACHE, f"act_{kind}_g{g:.3f}.json"))))["rows"]
            Tk = np.array([r["T"] for r in rows_k]); Sk = np.array([r["S3"] for r in rows_k])
            edges_lo.append(np.interp(np.log(T), np.log(Tk), Sk, left=np.nan, right=np.nan))
        keep = np.isfinite(S_det) & np.isfinite(S_lo) & np.all([np.isfinite(e) for e in edges + edges_lo], axis=0)
        T, S_det, S_lo = T[keep], S_det[keep], S_lo[keep]
        edges = [e[keep] for e in edges]; edges_lo = [e[keep] for e in edges_lo]
        # percolation temperatures of the three levels from the direct integration along the
        # line (data/line_budget.json); the axis is T in keV, nominal values rather than T / T_p^NLO
        lb = {o["g"]: o for o in json.load(open(_need(os.path.join(DATA, "line_budget.json"))))}[g]
        tp_nlo = lb["nlo"]["T_p"]
        K = tp_nlo * 1e6                           # T_p^NLO in keV: the window and the inset are set in units of it
        x = T * 1e6
        win = (T / tp_nlo > 0.0) & (T / tp_nlo < 1.75)
        T, x, S_det, S_lo = T[win], x[win], S_det[win], S_lo[win]
        edges = [e[win] for e in edges]; edges_lo = [e[win] for e in edges_lo]
        x_perc = {k: lb[k]["T_p"] * 1e6 for k in ("fulllo", "lorate", "nlo")}
        summ_all = json.load(open(os.path.join(d, "summary.json")))
        lo_obs = (summ_all.get("lo3d", {}).get("1", {}) or {}).get("observables") or {}
        S_crit = None                              # no criterion line: the vertical lines are named in the panel
        # V_4D + Daisy with the LO prefactor. N5 (fixed couplings): S_3/T of the default 4D potential (kind n5 of
        # reproduce/daisy/daisy_scale_band.py), drawn without a band (the main text explains why);
        # N4: the potential with the couplings run to mu = pi T, its band mu = pi T/2 and 2 pi T (kind bounce_rg of
        # reproduce/daisy/compute_daisy_actions.py)
        B4 = json.load(open(_need(os.path.join(DATA, f"scale_band_g{g:.3f}.json"))))
        x6 = np.array(B4["T_GeV"]) * 1e6
        S6 = np.array(B4["n5"]["S3_over_T"])
        e6 = (S6, S6)                              # no band: the limits of the axes and the label follow the curve
        rg = lambda tag: (lambda r: (np.array([q["T_GeV"] for q in r]), np.array([q["S3_over_T"] for q in r], float)))(
            sorted(json.load(open(_need(os.path.join(DATA, "daisy_cache",
                                                     f"daisy_bounce_rg{tag}_g{g:.3f}.json"))))["rows"], key=lambda q: q["T_GeV"]))
        T5, S5 = rg("")
        win5 = (T5 >= x6.min() / 1e6) & (T5 <= x6.max() / 1e6) & np.isfinite(S5)
        x4, S4 = T5[win5] * 1e6, S5[win5]
        K4 = [np.interp(np.log(x4), np.log(Tk * 1e6), Sk) for Tk, Sk in (rg("_mu0.5"), rg("_mu2"))]
        e4 = (np.min(K4, axis=0), np.max(K4, axis=0))
        x_perc["4d"] = _daisy_line([g], rg=True)[g][0] * 1e6
        x_perc["4d6"] = _daisy_line([g])[g][0] * 1e6
        curves[col] = (x, S_lo, S_det, edges_lo, edges, S_crit, g, x_perc, (x4, S4, e4), (x6, S6, e6))
        in4 = x4 <= 1.6 * K
        in6 = x6 <= 1.6 * K
        ymin = min(ymin, S_lo.min(), S_det.min(), np.minimum(*edges_lo).min(), e4[0][in4].min(), e6[0][in6].min())
        ymax = max(ymax, S_det.max(), edges[1].max(), np.maximum(*edges_lo).max(), e4[1][in4].max(), e6[1][in6].max(),
                   (S_crit.max() if S_crit is not None else ymax))
    insets, vlines = [], []
    for col in range(len(curves)):
        x, S_lo, S_det, edges_lo, edges, S_crit, g, x_perc, (x4, S4, e4), (x6, S6, e6) = curves[col]
        ax = axes[col]
        # no band for N5 (the text explains why); only its central curve
        ln_4d6, = ax.plot(x6, S6, "-", color=C_4D6, lw=1.4, label="4D fixed")
        # stacking: percolation lines at the back, then the N4 band, the LO band and line above it,
        # NLO, and the dotted N4 line on top of everything
        ax.fill_between(x4, e4[0], e4[1], color=C_4D, alpha=0.25, lw=0, zorder=1.0)
        ln_4d, = ax.plot(x4, S4, LS_4D, color=C_4D, lw=1.4, label="4D", zorder=Z_NLO + 0.2)
        ax.fill_between(x, np.minimum(*edges_lo), np.maximum(*edges_lo), color=C_LO, alpha=0.3, lw=0, zorder=1.1)
        ln_lo, = ax.plot(x, S_lo, LS_LO, color=C_LO, lw=1.4, label="LO", zorder=2.0)
        ax.fill_between(x, np.minimum(*edges), np.maximum(*edges), color=C_DET, alpha=0.3, lw=0)
        ln_det, = ax.plot(x, S_det, "-", color=C_DET, lw=1.6, label="NLO", zorder=Z_NLO)
        if col == 0:
            own_lines = {"NLO": ln_det, "LO": ln_lo, "4D": ln_4d, "4D6": ln_4d6}
        for key, cc, ls in (("fulllo", C_LO, "-"), ("nlo", C_DET, "-"), ("4d", C_4D, "-"), ("4d6", C_4D6, "-")):
            vlines.append((ax, ax.axvline(x_perc[key], color=cc, lw=0.8, ls=ls, alpha=0.8, zorder=0.5), key))   # behind the curves
        # a logarithmic temperature axis with the default major and minor tick positions
        ax.set_xscale("log")                   # log-spaced T: the 4D rate percolates an order of magnitude lower
        ax.set_xlim(max(x.min(), x4.min(), x6.min()), 1.6 * K); ax.set_ylim(ymin - 3, ymax + 8)   # room for the label above the fixed-coupling curve
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(20))
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(5))
        ax.set_xlabel(r"$T$  [keV]")
    axes[0].set_title(r"Exponent $S_{\rm tot}$")      # titles, not y labels
    fig_transition(fig, ax_all[1:])            # panels b and c drawn before the layout is frozen
    freeze(fig)
    _equal_gaps(fig, ax_all)
    # each percolation line rises from the axis to its own curve, where it marks T_p, and is cut there instead of
    # crossing the panel and the labels
    x_, S_lo_, S_det_, _, _, _, _, x_perc_, (x4_, S4_, _), (x6_, S6_, _) = curves[0]
    on_curve = {"fulllo": (x_, S_lo_), "nlo": (x_, S_det_), "4d": (x4_, S4_), "4d6": (x6_, S6_)}
    for a, ln, key in vlines:
        xc, yc = on_curve[key]
        y_end = float(np.interp(x_perc_[key], xc, yc))
        ln.set_ydata([0.0, a.transAxes.inverted().transform(a.transData.transform((x_perc_[key], y_end)))[1]])
    # the curves labelled along themselves, each on one line, rotated with the curve and on its free side:
    # NLO below its curve and LO above its band, both at large T; the percolation label above its line, between it and N5
    # percolation marker
    x, S_lo, S_det, edges_lo, edges, S_crit, g, x_perc, (x4, S4, e4), (x6, S6, e6) = curves[0]
    ax = axes[0]
    fig.canvas.draw()
    at = lambda y, x0: float(np.interp(x0, x, y))

    def along(txt, col_, y, x0, offset_pt, own, xg=None):
        """The text bent along the curve y(x): each glyph (a $...$ group counts as one) placed in turn by arc length in
        display coordinates, rotated with the local tangent, its baseline offset_pt from the curve (positive: above);
        the label is centred on x0."""
        import re as _re
        tr = ax.transData
        rend = fig.canvas.get_renderer()
        xg = x if xg is None else xg
        xs_ = np.linspace(xg[0], xg[-1], 4000)
        P = tr.transform(np.column_stack([xs_, np.interp(xs_, xg, y)]))
        seg = np.hypot(*np.diff(P, axis=0).T)
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        tang = np.arctan2(np.gradient(P[:, 1]), np.gradient(P[:, 0]))
        tokens = _re.findall(r"\$[^$]*\$|.", txt)

        def width(tok):
            if tok == " ":
                return width("a a") - width("aa")
            t_ = ax.text(0, 0, tok, fontsize=FS_ALONG); w_ = t_.get_window_extent(renderer=rend).width; t_.remove()
            return w_
        ws = [width(tk) for tk in tokens]
        s_mid = float(np.interp(tr.transform((x0, np.interp(x0, xg, y)))[0], P[:, 0], arc))
        s_ = s_mid - 0.5 * sum(ws)
        off = offset_pt * fig.dpi / 72
        group = object()
        placed = []
        for tk, w_ in zip(tokens, ws):
            sc = s_ + 0.5 * w_
            s_ += w_
            if tk == " ":
                continue
            px, py = np.interp(sc, arc, P[:, 0]), np.interp(sc, arc, P[:, 1])
            a_ = float(np.interp(sc, arc, tang))
            xd, yd = tr.inverted().transform((px - off * np.sin(a_), py + off * np.cos(a_)))
            t_ = ax.text(xd, yd, tk, color=col_, fontsize=FS_ALONG, ha="center", va="baseline", rotation=np.degrees(a_),
                         rotation_mode="anchor")
            t_._labels, t_._group = own, group
            placed.append(t_)
        return placed

    xl, xr = ax.get_xlim()
    # on the log axis, between the fixed-coupling 4D and the LO percolation lines (geometric means)
    gm = lambda a_, b_: float(np.sqrt(a_ * b_))
    x_mid = gm(x_perc["4d6"], x_perc["fulllo"])
    along("NLO", C_DET, np.minimum(*edges), x_mid, -9.0, own_lines["NLO"])
    # the N4 and LO labels sit at the same height above their curves, and the three gaps (the N5 percolation line to the
    # N4 label, N4 label to LO label, LO label to the right axis) are equal; N4 runs inside the LO band: its label sits
    # above the pair without a leader, and the colour assigns it, so it is exempt from the nearest-curve check
    y4_lab = np.maximum(e4[1], np.interp(x4, x, np.maximum(*edges_lo)))
    to_disp = lambda xv: ax.transData.transform((xv, 0.0))[0]
    to_data = lambda px: ax.transData.inverted().transform((px, 0.0))[0]
    x_line, x_axis = to_disp(x_perc["4d6"]), ax.transAxes.transform((1.0, 0.0))[0]
    c4, cl = LABEL_X_4D * x_perc["fulllo"], LABEL_X_LO * x_perc["nlo"]
    for _ in range(4):
        t4 = along(r"$V_{\rm 4D}$ + Daisy, running to $\pi T$", C_4D, y4_lab, c4, LABEL_DY_4D, None, xg=x4)
        tl = along("LO", C_LO, np.maximum(*edges_lo), cl, LABEL_DY_4D, own_lines["LO"])
        rend = fig.canvas.get_renderer()
        ext = lambda ts: (min(t.get_window_extent(rend).x0 for t in ts), max(t.get_window_extent(rend).x1 for t in ts))
        (a4, b4), (al, bl) = ext(t4), ext(tl)
        gap = ((a4 - x_line) + (al - b4) + (x_axis - bl)) / 3
        c4 = to_data(to_disp(c4) + (x_line + gap - a4))
        cl = to_data(to_disp(cl) + (x_line + 2 * gap + (b4 - a4) - al))
        for t in t4 + tl:
            t.remove()
    along(r"$V_{\rm 4D}$ + Daisy, running to $\pi T$", C_4D, y4_lab, c4, LABEL_DY_4D, None, xg=x4)
    along("LO", C_LO, np.maximum(*edges_lo), cl, LABEL_DY_4D, own_lines["LO"])
    # just above the fixed-coupling curve, right of where the percolation line crosses it;
    # exempt from the nearest-curve check like the N4 label, the colour assigns it
    along(r"$V_{\rm 4D}$ + Daisy, $\mu = v$", C_4D6, e6[1], LABEL_X_4D6 * x_perc["nlo"], LABEL_DY_4D6, None, xg=x6)
    # the vertical lines named once, upright beside the LO line, in its colour
    y0 = ax.get_ylim()[0]
    ax.text(x_perc["fulllo"] / PERC_DX, y0 + 4.0, "Percolation", rotation=90, color=C_LO, ha="right", va="bottom",
            fontsize=FS_ALONG)
    ax.text(0.05, 0.94, rf"$g = {g:g}$, $\mu_0 = 1\,$GeV", transform=ax.transAxes, ha="left", va="top", fontsize=7)
    save(fig, "fig_rate.pdf")


# ---------------------------------------------------------------- figure 2
def _daisy_line(couplings, rg=False):
    """T_p, alpha and (beta/H)_RH of the 4D daisy rate on the line v = e^{1/6} GeV/g, at the given
    couplings (those of the line budget, so that the differences to NLO need no interpolation of
    the NLO values), from data/daisy_line.json and daisy_line_rg.json (reproduce/daisy/daisy_line.py). Couplings
    at which the transition does not percolate are left out."""
    direct = _need(os.path.join(DATA, "daisy_line_rg.json" if rg else "daisy_line.json"))
    # exact percolation of the daisy rate on the line itself (reproduce/daisy/daisy_line.py), resolved to the end of
    # percolation
    return {float(r["g"]): (r["T_p"], r["alpha"], r["betaH_R"]) for r in json.load(open(direct)) if r["ok"]}


def _symlog_percent(ax, lim=150.0):
    """A difference in per cent on a symmetric-log axis, linear within +-5%."""
    ax.set_yscale("symlog", linthresh=5.0, linscale=1.0)
    ax.set_ylim(-lim, lim)
    ticks = [-100, -10, 0, 10, 100]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"${t:d}$" for t in ticks])
    minor = [s * m for s in (-1, 1) for m in (1, 2, 3, 4, 5, 20, 30, 40, 50, 60, 70, 80, 90)]
    ax.set_yticks([m for m in minor if abs(m) < lim], minor=True)


def fig_transition(fig=None, axs=None):
    """T_p (in GeV) and (beta/H)_RH along the line v = e^{1/6} mu0/g (mu0 = 1 GeV) by exact
    percolation (data/line_budget.json, with the figure-only couplings that resolve the e+- and
    muon annihilation): NLO, LO (LO matching and the LO bounce), and the 4D daisy rate with the couplings
    run to pi T and fixed at the vacuum scale (data/daisy_line_rg.json and daisy_line.json; it percolates on
    the line down to g = 0.445). In the (beta/H) panel also the two other speeds of the same NLO rate,
    (beta/H)_S (dashed) and (beta/H)_Gamma (dash-dotted), in the colour of NLO. Curves through the computed
    couplings, no markers; hatched where the NLO transition does not complete. Drawn into panels b and c of
    fig_rate when axs is given, with a key of the three speeds; alone, with labels next to the curves. alpha is
    not shown: it is 10^8 or more wherever the data point and drops out of the signal."""
    # the NLO curves where the phase transition completes (a final temperature T_f exists); below the smallest such
    # coupling the vacuum stays trapped (hatched), as in the grey region of fig_plane
    res_all = json.load(open(_need(os.path.join(DATA, "line_budget.json"))))
    # the NLO curves run down to where the rate stops percolating, like the LO and 4D curves; the
    # hatched region marks where it percolates no more or does not complete
    done = lambda o: bool(o["nlo"]) and np.isfinite(o["nlo"]["T_f"])
    perc = lambda o: bool(o["nlo"]) and np.isfinite(o["nlo"].get("T_p") or np.nan)
    res = [o for o in res_all if perc(o)]
    g = np.array([o["g"] for o in res])
    g_done = min(o["g"] for o in res_all if done(o))
    g_trapped = max(o["g"] for o in res_all if not done(o) and o["g"] < g_done)
    g_edge = 0.5 * (g_trapped + g_done)                 # the edge of vacuum trapping, between the two couplings
    g_left = min(o["g"] for o in res_all)               # the axes start at the smallest computed coupling
    Q = lambda var, key: np.array([(o[var] or {}).get(key, np.nan) if o.get(var) else np.nan for o in res])
    band = {q: np.array([o.get("entries_pct", {}).get(q, {}).get("total", np.nan) for o in res]) / 100
            for q in ("T_p", "betaH_R")}
    d4 = _daisy_line(g, rg=True)               # N4: the couplings run to mu = pi T
    g4 = np.array(sorted(d4))
    d6 = _daisy_line(g)                        # N5: the couplings fixed at the vacuum
    g6 = np.array(sorted(d6))
    nlo_at = lambda key, x: np.exp(np.interp(x, g, np.log(Q("nlo", key))))   # NLO at the daisy couplings
    own_fig = fig is None
    if own_fig:
        fig, axes = plt.subplots(1, 2, figsize=(width("fig_transition"), 2.6), layout="constrained", squeeze=False)
    else:
        axes = np.array([list(axs)])
    specs = (("T_p", r"Percolation temperature $T_{\rm p}$  [GeV]", 0), ("betaH_R", r"Transition speed $\beta/H$", 2))
    C_S3 = C_GAM = C_DET                       # the same NLO rate: its colour, other line styles
    hl = {}                                    # the line of each curve, for the labels
    for col, (key, lab, i4) in enumerate(specs):
        ax = axes[0, col]
        c, full = Q("nlo", key), Q("fulllo", key)
        y4 = np.array([d4[x][i4] for x in g4])
        c4 = nlo_at(key, g4)
        # no theory band here: it is practically invisible at this size
        # LO dashed and N4 dotted where the curves overlap; in the beta/H panel dashed and dash-dotted name the
        # other definitions of beta/H, so there LO is solid and N4 stays dotted, which no definition uses
        ls_lo, ls_4d = (LS_LO, LS_4D) if key == "T_p" else ("-", LS_4D)
        # the NLO curves end where the transition stops completing, at the hatched region (declared as the end of
        # their domain); LO ends where it stops percolating and leaves the panel there
        hl[(col, "nlo")], = ax.plot(g, c, "-", color=C_DET, lw=1.6, zorder=Z_NLO, gid="domain_end")
        hl[(col, "lo")], = ax.plot(g, full, ls_lo, color=C_LO, lw=1.2, gid="physical_end")
        # it stops where the daisy rate stops percolating
        hl[(col, "4d")], = ax.plot(g4, y4, ls_4d, color=C_4D, lw=1.2, gid="physical_end")
        hl[(col, "4d6")], = ax.plot(g6, np.array([d6[x][i4] for x in g6]), "-", color=C_4D6, lw=1.2,
                                    gid="physical_end")
        if key == "betaH_R":
            hl[(col, "s3")], = ax.plot(g, Q("nlo", "betaH_S"), "--", color=C_S3, lw=1.1, gid="domain_end")
            hl[(col, "gam")], = ax.plot(g, Q("nlo", "beta_Gamma_H"), "-.", color=C_GAM, lw=1.1, gid="domain_end")
        (ax.set_ylabel if own_fig else ax.set_title)(lab)
        ax.set_yscale("log")
        if key == "T_p":
            # a fixed range: the 4D daisy T_p falls to zero at the end of percolation (g = 0.4452) and leaves the
            # panel through its lower edge; autoscaling would fit the axis to its last point
            ax.set_ylim(3e-13, 0.3)
        ax.set_xlabel(r"$g$")
        ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8])
        ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))
        ax.xaxis.set_minor_locator(mpl.ticker.FixedLocator(np.round(np.arange(0.38, 0.8001, 0.01), 3)))
        # the axes span exactly the computed couplings, so that no curve stops inside the axes for want of points
        # and no room is left beyond the data, down to the trapped ones
        ax.set_xlim(float(g_left), float(g.max()))
        ax.axvspan(g_left, g_edge, facecolor="none", edgecolor="0.65", hatch="////", lw=0.0, zorder=0.2)
    if not own_fig:
        # the benchmark line of both panels, as in the exponent panel
        for a_ in axes[0]:
            a_.text(0.09, 0.94, r"$\mu_0 = 1\,$GeV", transform=a_.transAxes, ha="left", va="top", fontsize=7)
        # figure 2: the curves are named in the exponent panel; here only the key of the three speeds,
        # grey lines in their line styles
        a = axes[0, 1]
        a.set_ylim(1.0, 100.0)
        hs = [mpl.lines.Line2D([], [], color=GREY, lw=1.2, ls=ls_) for ls_ in ("-", "--", "-.")]
        a.legend(hs, [r"$(\beta/H)_R$", r"$(\beta/H)_S$", r"$(\beta/H)_\Gamma$"], loc="lower right", fontsize=7,
                 frameon=False, handlelength=2.4)
        return
    # labels next to the curves. T_p: where each curve crosses a chosen temperature, in the white to its
    # side; the speeds: fanned out past the right end of the curves, with short leaders
    lab = dict(fontsize=7)
    a = axes[0, 0]
    lg = lambda x: np.log10(np.asarray(x, dtype=float))
    cross = lambda gg, yy, y: float(np.interp(np.log10(y), lg(yy), gg))      # T_p rises with g
    c_t, l_t = Q("nlo", "T_p"), Q("fulllo", "T_p")
    d_t = np.array([d4[x][0] for x in g4])
    # each next to its own curve, with every other plotted line clearly further away (searched): candidates on both
    # sides of each curve along g
    fig.canvas.draw()
    y_lo, y_hi = a.get_ylim()
    grid_ = [(x, y) for x in np.linspace(float(g.min()), float(g.max()), 40)
             for y in np.geomspace(y_lo, y_hi, 40)]
    for txt, col_, own, g_lead in (("NLO", C_DET, "nlo", 0.52), ("LO", C_LO, "lo", 0.62),
                                   (r"$V_{\rm 4D}$ + Daisy", C_4D, "4d", 0.55)):
        best = _place_text(fig, a, txt, grid_, fontsize=7, mask_grey=False, own=hl[(0, own)], color=col_)
        if best is not None and best[1] < 0.004:
            a.text(*best[0], txt, color=col_, ha="left", va="center", **lab)._labels = hl[(0, own)]
            continue
        # a curve between two others without room beside it (LO between NLO and the daisy rate): a short
        # leader from the nearest free spot to the curve
        ln_ = hl[(0, own)]
        target = (g_lead, float(np.exp(np.interp(g_lead, ln_.get_xdata(), np.log(ln_.get_ydata())))))
        best = _place_text(fig, a, txt, grid_, fontsize=7, mask_grey=False, target=target, leader_of=ln_, color=col_)
        assert best is not None and best[1] < 0.004, f"no clean room for {txt!r}: {best}"
        an = a.annotate(txt, xy=target, xytext=best[0], color=col_, ha="left", va="center",
                        arrowprops=dict(arrowstyle="-", color=col_, lw=0.6, shrinkA=2, shrinkB=1), **lab)
        an._labels = ln_
    a = axes[0, 1]
    a.set_ylim(1.0, 250.0)
    fig.canvas.draw()
    from figure_checks import freeze
    freeze(fig)
    on = lambda arr, gg, x: float(np.exp(np.interp(x, gg, np.log(arr))))
    s3, rh, gm, lo = Q("nlo", "betaH_S"), Q("nlo", "betaH_R"), Q("nlo", "beta_Gamma_H"), Q("fulllo", "betaH_R")
    d4b = np.array([d4[x][2] for x in g4])
    gl = np.linspace(float(g.min()) + 0.005, float(g.max()) - 0.005, 18)
    below = lambda c, gg: [(x, on(c, gg, x) / f_) for x in gl if gg.min() <= x <= gg.max() for f_ in np.geomspace(1.25, 6, 10)]
    # the three speeds of the NLO rate: right-aligned one above the other in the empty upper left, in the order of
    # their curves, each with a leader to its own curve at one coupling, so that the leaders run side by side and
    # none crosses another. The curves are nested, so the leaders of the lower two pass the curves
    # above them; they cross no other line and no label.
    from figure_checks import plot_lines, line_distances
    trio = ((r"$(\beta/H)_S$", s3, "s3"), (r"$(\beta/H)_R$", rh, "nlo"), (r"$(\beta/H)_\Gamma$", gm, "gam"))
    pt = fig.dpi / 72
    rend = fig.canvas.get_renderer()
    sizes = []
    for txt, _, _ in trio:
        tmp = a.text(0, 0, txt, fontsize=7)
        sizes.append((tmp.get_window_extent(rend).width, tmp.get_window_extent(rend).height))
        tmp.remove()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    frame, pad = a.get_window_extent(), 2.5 * pt
    trio_lines = [hl[(1, k)] for _, _, k in trio]
    foreign = [ln for ln in plot_lines(a) if not any(ln is t_ for t_ in trio_lines)]
    hstep = max(h for _, h in sizes) + 2.5 * pt
    every = plot_lines(a)

    def crosses(p, q, r, s_):
        o = lambda a_, b_, c_: np.sign((b_[0] - a_[0]) * (c_[1] - a_[1]) - (b_[1] - a_[1]) * (c_[0] - a_[0]))
        return o(p, q, r) * o(p, q, s_) < 0 and o(r, s_, p) * o(r, s_, q) < 0
    # where the three curves are visually furthest apart: the smaller of the two perpendicular gaps between
    # neighbouring curves, in points: the leaders touch the curves where one can tell them apart without zooming in
    def perp_gap(c1, c2, x):
        p1 = a.transData.transform((x, on(c1, g, x)))
        p2 = a.transData.transform((x, on(c2, g, x)))
        q = a.transData.transform((x + 0.002, on(c1, g, x + 0.002)))
        slope = (q[1] - p1[1]) / (q[0] - p1[0])
        return abs(p1[1] - p2[1]) / np.hypot(1.0, slope) / pt
    xs_gap = np.arange(0.43, 0.775, 0.005)
    gap = np.array([min(perp_gap(s3, rh, x), perp_gap(rh, gm, x)) for x in xs_gap])
    # the leaders end where both gaps are at least 1.5 pt (at small g, and around g = 0.69); among the clean
    # stacks the search prefers the largest gap
    g_targets = xs_gap[gap >= 1.5]
    print(f"  fig_transition: the speeds are furthest apart ({gap.max():.1f} pt) at g = {xs_gap[np.argmax(gap)]:.3f}; "
          f"end points searched at g = {g_targets.min():.3f} to {g_targets.max():.3f}")
    best = None
    why = dict(sep=0, box=0, cross=0, clear=0, foreign=0)
    for side in ("left", "right"):
        sgn = 1.0 if side == "left" else -1.0          # the direction from the labels to the curves
        for g_t in g_targets:
            for dg in (-0.03, -0.015, 0.0, 0.015, 0.03):
                gs_ = [g_t + (2 - i) * dg for i in range(3)]
                if min(gs_) < float(g.min()) or max(gs_) > float(g.max()):
                    continue
                ends = [np.array(a.transData.transform((x_, on(c, g, x_)))) for x_, (_, c, _) in zip(gs_, trio)]
                if min(np.hypot(*(ends[i] - ends[j])) for i in range(3) for j in range(i + 1, 3)) < 7 * pt:
                    why["sep"] += 1
                    continue                           # the leaders end at clearly separate points
                gap_end = min(min(perp_gap(s3, rh, x_), perp_gap(rh, gm, x_)) for x_ in gs_)
                xs_lab = (np.linspace(float(g.min()) + 0.03, min(gs_) - 0.01, 16) if side == "left"
                          else np.linspace(max(gs_) + 0.01, float(g.max()) - 0.03, 16))
                for xe in xs_lab:
                    for ym in np.geomspace(15.0, 220.0, 36):
                        xe_px, ym_px = a.transData.transform((xe, ym))
                        segs, boxes, b_tot = [], [], 0.0
                        for i, (w, h) in enumerate(sizes):
                            cy = ym_px + (1 - i) * hstep
                            cx = xe_px - sgn * w / 2           # the labels end (left) or start (right) at xe
                            W, Hh = w + 2 * pad, h + 2 * pad
                            if (cx - W / 2 < frame.x0 + 4 * pt or cx + W / 2 > frame.x1 - 4 * pt
                                    or cy + Hh / 2 > frame.y1 - 4 * pt):
                                break
                            b = _busy(fig, img, cx, cy, W, Hh, 0.0)
                            start = np.array([xe_px + sgn * 2 * pt, cy])
                            if b > 0.004 or sgn * (ends[i][0] - start[0]) < 8 * pt or ends[i][1] > cy - 4 * pt:
                                break                  # a clean box, and a leader down towards the curves
                            b_tot += b
                            segs.append((start, ends[i]))
                            boxes.append((cx, cy, w, h))
                        if len(segs) < 3:
                            why["box"] += 1
                            continue
                        length = sum(float(np.hypot(*(e - s_))) for s_, e in segs)
                        key = (round(b_tot / 0.002), -round(gap_end), length)
                        if best is not None and key >= best[0]:
                            continue
                        # the leaders do not cross each other, pass no line other than the three curves, and
                        # every label keeps 6 pt from every line
                        if any(crosses(*segs[i], *segs[j]) for i in range(3) for j in range(i + 1, 3)):
                            why["cross"] += 1
                            continue
                        if min(min(line_distances(fig, a, *bx, every)) for bx in boxes) < 6.0:
                            why["clear"] += 1
                            continue
                        pts = [s_ + u * (e - s_) for s_, e in segs for u in np.linspace(0.05, 0.95, 19)]
                        if foreign and min(min(line_distances(fig, a, px, py, 0.0, 0.0, foreign))
                                           for px, py in pts) < 1.5:
                            why["foreign"] += 1
                            continue
                        best = (key, gs_, xe, ym, side)
    assert best is not None, f"fig_transition: no clean stack for the labels of the three speeds: {why}"
    _, gs_, xe, ym, side = best
    xe_px, ym_px = a.transData.transform((xe, ym))
    for i, (txt, c, k) in enumerate(trio):
        xy_t = a.transData.inverted().transform((xe_px, ym_px + (1 - i) * hstep))
        an = a.annotate(txt, xy=(gs_[i], on(c, g, gs_[i])), xytext=tuple(xy_t), color=C_DET, fontsize=7,
                        ha="right" if side == "left" else "left", va="center",
                        arrowprops=dict(arrowstyle="-", color=C_DET, lw=0.6, shrinkA=2, shrinkB=1,
                                        relpos=(1.0, 0.5) if side == "left" else (0.0, 0.5)))
        an._labels = hl[(1, k)]
    print(f"  fig_transition: the three speeds stacked {side} of the curves at g = {xe:.3f}, leaders to g = "
          + ", ".join(f"{x_:.3f}" for x_ in gs_) + f", {best[0][1]:+d} pt gap")
    # in figure 2 the LO curve keeps the colour it is labelled with in the T_p panel (no room between NLO and 4D)
    for txt, col_, target, cand, own in ((r"$V_{\rm 4D}$ + Daisy", C_4D, (0.62, on(d4b, g4, 0.62)), below(d4b, g4), "4d"),
                                         ("LO", C_LO, (0.70, on(lo, g, 0.70)), below(d4b, g4), "lo"))[:1 if not own_fig else 2]:
        best = _place_text(fig, a, txt, cand, fontsize=7, mask_grey=False, target=target, leader_of=hl[(1, own)],
                           color=col_)
        assert best is not None and best[1] < 0.004, f"no clean room for {txt!r}: {best}"
        an = a.annotate(txt, xy=target, xytext=best[0], color=col_, fontsize=7, ha="left", va="center",
                        arrowprops=dict(arrowstyle="-", color=col_, lw=0.6, shrinkA=2, shrinkB=1))
        an._labels = hl[(1, own)]
        print(f"  fig_transition label {txt!r} ({col_ == C_DET and 'NLO' or col_ == C_LO and 'LO' or 'daisy'}): "
              f"{100 * best[1]:.1f}% of its box covered")
    if own_fig:
        save(fig, "fig_transition.pdf")


# ---------------------------------------------------------------- spectra
def _load_scan():
    return _grid_npz(_need(os.path.join(DATA, SCAN_NPZ)))


def _dof(T, dark=0.0):
    """g_rho and g_s of the Standard Model plasma plus the dark degrees of freedom."""
    tab = np.genfromtxt(os.path.join(ROOT, "src", "transitionlistener", "tab_data", "geff_SM.dat"))
    return (np.interp(np.log(T), np.log(tab[:, 0]), tab[:, 1]) + dark,
            np.interp(np.log(T), np.log(tab[:, 0]), tab[:, 3]) + dark)


SPECTRA_POINT = {}


def _plane_scan():
    """plane_scan.py as a module (its spectra, reheating and transition parameters)."""
    os.environ.setdefault("PLANE_SCAN_ARGS", f"--linescan {os.path.join(DATA, 'linescan_u1_mu0_1GeV.json')}")
    _reproduce_on_path()
    import plane_scan
    return plane_scan


def _spectra():
    """The spectra of fig_spectra at the NANOGrav best fit, through the cache (_cached): every input of
    plane_scan's central point and reheating, and TransitionListener's sources, enter the digest."""
    bf = json.load(open(_need(os.path.join(DATA, "best_fits.json"))))
    g0, lv0 = bf["nlo"]["NG15"]["at"]
    args = os.environ.get("PLANE_SCAN_ARGS", f"--linescan {os.path.join(DATA, 'linescan_u1_mu0_1GeV.json')}")
    files = [os.path.join(DATA, n) for n in ("best_fits.json", "tl_nodes_table.npz", "kappa_nodes_table.npz",
                                             "linescan_u1_mu0_1GeV.json", "daisy_table_plane.npz",
                                             "daisy_table_plane_direct.npz", "daisy_table_plane_tl.npz",
                                             "daisy_rg_table_plane.npz")]
    files += [os.path.join(REPRODUCE, "paths.py"), os.path.join(REPRODUCE, "plane", "plane_scan.py"),
              os.path.join(REPRODUCE, "plane", "transition_table.py"), TABLE_NPZ]
    files = [_need(p) for p in files]
    files += [os.path.join(dp, n) for dp, dn, fn in os.walk(os.path.join(ROOT, "src", "transitionlistener"))
              if "__pycache__" not in dp for n in fn if not n.endswith(".pyc")]

    def compute():
        ps = _plane_scan()
        _, bh, al, bv = ps.central(g0, lv0)
        reh = ps.reheating(g0, lv0)
        f = np.logspace(-9.3, -7.0, 400)
        V = dict(ps.COSMO_VARIANTS)
        spec = lambda n: ps.make_spectrum(V[n], reh, bh, al, bv)(f)
        return dict(f=f, default=spec("expansion, dissipative"), free=spec("expansion, bulk flow"),
                    flat=spec("no expansion, bubbles R^-3"),
                    envelope=spec("no expansion, envelope"),
                    sw=spec("sound waves"), hybrid=spec("LISA update 2019"), g=np.array(g0), logv=np.array(lv0), betaH=np.array(bh), T_reh=np.array(reh[0]))
    # the paths in args relative to the repository, so that a checkout elsewhere finds the cache
    extra = (g0, lv0, args.replace(ROOT + os.sep, ""), os.environ.get("DAISY_TABLE", "bounce"), inspect.getsource(_spectra))
    return _cached("spectra_best_fit_NG15", _digest(files, (), extra), compute)


def fig_spectra():
    """Spectra at the NANOGrav best fit (data/best_fits.json, reproduce/plane/best_fits.py) against the free-spectrum posteriors of the two data sets
    (ceffyl densities in pta_free_spectra.npz), every spectrum TransitionListener's, built as in the likelihood scan
    (plane_scan.make_spectrum): the default, G1 (no expansion), G2 (free shells), G3 (the Higgsless template of LISA
    2024), G4 (the hybrid template of LISA 2019) and G5 (the envelope approximation); violins: the central 95% of each
    bin's posterior. Few style elements, the data under the lines: the data as filled violins of one style, drawn first;
    every template a solid line of one width, the default black and thicker; the binaries as a band behind the data;
    one legend below the axes instead of labels with leaders."""
    S = _spectra()
    f, default, free, flat, env, sw, hyb = (S[k] for k in ("f", "default", "free", "flat", "envelope", "sw", "hybrid"))
    SPECTRA_POINT.update({k: float(S[k]) for k in ("g", "logv", "betaH", "T_reh")})
    from scipy.ndimage import gaussian_filter1d
    fs = np.load(_need(os.path.join(DATA, "pta_free_spectra.npz")))
    fig, ax = plt.subplots(figsize=(width("fig_spectra"), 3.0), layout="constrained")
    # the binaries predicted under the survey-averaged priors, from holodeck over the whole axis
    # (reproduce/binaries/binaries_prior_band_holodeck.py): median and the 1 sigma band over the prior and the
    # realisations, behind the data
    BB = np.load(_need(os.path.join(DATA, "binaries_prior_band_holodeck_recent.npz")))
    fb, qb = BB["f"], 10.0 ** BB["log10_h2omega"]
    C_BIN = DESYgruen
    ax.fill_between(fb, qb[1], qb[3], color=C_BIN, alpha=0.25, lw=0, zorder=-0.5)
    ax.plot(fb, qb[2], "--", color=C_BIN, lw=1.1, zorder=2)
    for name, colour in (("IPTA2", C_IPTA), ("NG15v2", C_NG)):
        freqs, grid, dens, T = fs[name + "_freqs"], fs[name + "_rho_grid"], fs[name + "_density"], float(fs[name + "_Tspan"])
        for fi, dd in zip(freqs, dens):
            y = np.log10(8 * np.pi**4 * fi**5 * T / H100_HZ**2) + 2 * grid
            pdf = gaussian_filter1d(np.exp(dd - dd.max()), 0.02 / float(grid[1] - grid[0]), mode="nearest")
            pdf /= pdf.max()
            cdf = np.cumsum(pdf) / pdf.sum()
            keep = (cdf > 0.025) & (cdf < 0.975)
            w = pdf * 0.018                           # half width in log10 f
            xl, xr, yy = fi * 10 ** (-w[keep]), fi * 10 ** (w[keep]), 10 ** y[keep]
            # an opaque tint (the colour at 35 % on white), so that the binary band behind does not show through
            tint = 0.35 * np.asarray(mpl.colors.to_rgb(colour)) + 0.65
            ax.fill_betweenx(yy, xl, xr, facecolor=tint, edgecolor=colour, lw=0.5, zorder=1)
    # purple for the hybrid template, brown for the envelope approximation; green: binaries; lila, gelb: data
    C_FLAT, C_FREE, C_SW, C_HYB, C_ENV = DESYcyan, DESYorange, DESYmagenta, DESYaubergine, DESYdunkelbraun
    lw = 1.0                                     # the alternatives thinner than the default
    ax.plot(f, env, "-", color=C_ENV, lw=lw, zorder=3, label="Envelope approximation (G5)")
    ax.plot(f, hyb, "-", color=C_HYB, lw=lw, zorder=3, label="Hybrid template, LISA 2019 (G4)")
    ax.plot(f, sw, "-", color=C_SW, lw=lw, zorder=3, label="Higgsless template, LISA 2024 (G3)")
    ax.plot(f, free, "-", color=C_FREE, lw=lw, zorder=3, label="Free-streaming shells (G2)")
    ax.plot(f, flat, "-", color=C_FLAT, lw=lw, zorder=3, label="No expansion during the transition (G1)")
    ax.plot(f, default, "-", color="k", lw=2.0, zorder=4, label="This work's default")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-13, 1e-6); ax.set_xlim(8e-10, 1e-7)
    ax.set_xlabel(r"$f$  [Hz]"); ax.set_ylabel(r"$h^2\Omega_{\rm GW}$")
    h, l = ax.get_legend_handles_labels()
    h, l = h[::-1], l[::-1]
    h.append((Patch(facecolor=C_BIN, alpha=0.25, lw=0), plt.Line2D([], [], color=C_BIN, ls="--", lw=1.1)))
    l.append(r"Binaries, survey-averaged priors, $1\sigma$")
    tint = lambda c: 0.35 * np.asarray(mpl.colors.to_rgb(c)) + 0.65
    h.append(Patch(facecolor=tint(C_NG), edgecolor=C_NG, lw=0.5)); l.append(r"NANOGrav 15$\,$yr, $2\sigma$")
    h.append(Patch(facecolor=tint(C_IPTA), edgecolor=C_IPTA, lw=0.5)); l.append(r"IPTA DR2, $2\sigma$")
    # the legend below the axes, as in figure 4: the upper left holds the IPTA violins of the lowest bins
    order = list(range(9))                       # filled column by column: default, G1, G2, G3, G4 | G5, binaries, data
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="outside lower center", ncol=2, fontsize=6,
               frameon=False, handlelength=2.0, columnspacing=1.6, labelspacing=0.35, **_tuple_legend())
    save(fig, "fig_spectra.pdf")


# ---------------------------------------------------------------- the hardening of the binaries
def fig_hardening():
    """The local significance of the preference for the phase transition against the binary lifetime tau_f
    (reproduce/binaries/preference_tau_profile.py): nu_inner and the other four astrophysical parameters
    profiled under their prior, (g, v) maximised, for the three prior sets; the survey-averaged one, the default, solid
    black. Dots: the best fit of the binaries alone, where each curve has its minimum. Labels next to the
    curves, the axis cut at 5 sigma, the binaryfog logo in the lower right below the curves."""
    d = np.load(_need(os.path.join(DATA, "preference_tau_profile.npz")))
    free = json.load(open(_need(os.path.join(DATA, "preference_tau0.1.json"))))
    tau = d["tau"]
    # (prior set, label, colour, line style, label at tau_f [Gyr], above (+1) or below (-1) the curve)
    # the two alternatives in two blues of the colour map Blues
    curves = [("nanograv", "NANOGrav prior", plt.get_cmap("Blues")(0.6), "-.", 4.0, -1),
              ("literature", "Tomczak\u2013Kormendy prior", plt.get_cmap("Blues")(0.9), "--", 3.2, -1),
              ("recent", "Survey-averaged prior", "k", "-", 4.0, +1)]
    w = width("fig_hardening")
    fig, ax = plt.subplots(figsize=(w, 0.72 * w), layout="constrained")
    ax.set_xlim(tau[0], tau[-1]); ax.set_ylim(1.5, 5.0)
    fig.canvas.draw()
    for which, label, col, ls, t_lab, side in curves:
        sig = d[f"sigma_cp_v2_{which}"]
        ax.plot(tau, sig, color=col, ls=ls, lw=1.4, zorder=3)
        # the label along the curve, 3.5 pt off it on the given side
        p0, p1 = (ax.transData.transform((t_, np.interp(t_, tau, sig))) for t_ in (t_lab - 0.3, t_lab + 0.3))
        ang = np.arctan2(p1[1] - p0[1], p1[0] - p0[0])
        off = 3.5 * side * np.array([-np.sin(ang), np.cos(ang)])
        ax.annotate(label, (t_lab, np.interp(t_lab, tau, sig)), xytext=tuple(off), textcoords="offset points",
                    ha="center", va="bottom" if side > 0 else "top", rotation=np.degrees(ang),
                    rotation_mode="anchor", color=col, fontsize=7)
        t0 = free["cp_v2"][which]["theta_bin"][4]
        ax.plot(t0, np.interp(t0, tau, sig), "o", ms=3.5, color=col, zorder=4, clip_on=False)
    ax.set_xlim(tau[0], tau[-1])                      # the prior range of tau_f, linear
    ax.set_ylim(1.5, 5.0)
    ax.set_yticks([2, 3, 4, 5])
    ax.set_xticks([2, 4, 6, 8, 10])
    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
    ax.set_xlabel(r"Binary hardening timescale $\tau_{\rm f}$ [Gyr]")
    ax.set_ylabel(r"Local significance [$\sigma$]")
    fig.canvas.draw()
    fig.set_layout_engine("none")
    import matplotlib.image as mpimg
    bb = ax.get_position()
    W, H = fig.get_size_inches()
    h = 0.26 * bb.height
    lw_ = h * (1200 / 889) * H / W                 # width / height of the binaryfog logo (black holes above the name)
    # lower right, below the lowest curve, which stays above 2.6 sigma for tau_f > 6.5 Gyr; the image at its own
    # resolution (interpolation="none"), as in the binaryfog paper
    lax = fig.add_axes((bb.x1 - 0.03 * bb.width - lw_, bb.y0 + 0.03 * bb.height, lw_, h), zorder=10)
    lax.imshow(mpimg.imread(os.path.join(HERE, "binaryfog_logo_300.png")), interpolation="none")
    lax.set_axis_off(); lax.set_anchor("C"); lax.set_label("logo")
    save(fig, "fig_hardening.pdf")


# ---------------------------------------------------------------- the plane
def _levels(lnl):
    """68% and 95% levels of a 2D log-likelihood map (flat prior in g, log10 v)."""
    m = np.nanmax(lnl)
    p = np.exp(np.nan_to_num(lnl - m, nan=-np.inf))
    p /= p.sum()
    s = np.sort(p.ravel())[::-1]
    c = np.cumsum(s)
    return [s[np.searchsorted(c, 0.95)], s[np.searchsorted(c, 0.68)], s[0] * 1.0001], p


def _interval(p, axis_vals, axis, level=0.68):
    marg = p.sum(axis=1) if axis == 0 else p.sum(axis=0)
    c = np.cumsum(marg) / marg.sum()
    return np.interp((1 - level) / 2, c, axis_vals), np.interp((1 + level) / 2, c, axis_vals)


def _fine(G, LOGV, p, factor=4):
    """The posterior on a grid `factor` times finer, by cubic interpolation between the nodes, so
    that the contours follow the posterior rather than the cells of the likelihood grid (the
    levels are those of the original grid)."""
    from scipy.interpolate import RegularGridInterpolator
    Gf = np.linspace(G[0], G[-1], factor * (len(G) - 1) + 1)
    Lf = np.linspace(LOGV[0], LOGV[-1], factor * (len(LOGV) - 1) + 1)
    f = RegularGridInterpolator((G, LOGV), np.nan_to_num(p, nan=0.0), method="cubic")
    GG, LL = np.meshgrid(Gf, Lf, indexing="ij")
    return Gf, Lf, f(np.stack([GG.ravel(), LL.ravel()], axis=-1)).reshape(GG.shape)


def _log_v_axis(ax, label=True):
    """The v axis (data in log10 v) labelled with powers of ten."""
    lo, hi = ax.get_ylim()
    major = [k for k in range(int(np.floor(lo)), int(np.ceil(hi)) + 1) if lo - 1e-9 <= k <= hi + 1e-9]
    ax.set_yticks(major)
    ax.set_yticklabels([rf"$10^{{{k}}}$" if k not in (0, 1) else ("$1$" if k == 0 else "$10$") for k in major])
    minor = [k + np.log10(m) for k in range(int(np.floor(lo)) - 1, int(np.ceil(hi)) + 1) for m in range(2, 10)]
    ax.set_yticks([x for x in minor if lo <= x <= hi], minor=True)
    if label:
        ax.set_ylabel(r"$v$  [GeV]")


def _g_axis(ax, right=None):
    """The g axis of the plane and of the chains: 0.40 to 0.60 (or to `right`), labelled from the second tick so that
    no label sits in the corner next to the lowest v label."""
    right = G_RIGHT if right is None else right
    ax.set_xlim(G_LEFT, right)
    step = 0.05 if right <= 0.6 + 1e-9 else 0.1
    ticks = list(np.round(np.arange(G_LEFT, right + 1e-9, step), 2))
    ax.set_xticks(ticks)
    ax.set_xticklabels([""] + [f"{t:.2f}" for t in ticks[1:]])
    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.01))
    ax.set_xlabel(r"$g$")


def _display_lnl(lnl, ds, G=None, LOGV=None):
    """The map from which a region is drawn. The IPTA DR2 free spectrum is a kernel density estimate
    on coarse histogram cells, so that its likelihood (through binaryfog) fluctuates between neighbouring cells of the plane and draws jagged edges; it is
    smoothed with a Gaussian of 0.005 in g and 0.05 in log10 v (two spacings of the grid; cells
    without a transition left out). NANOGrav is smooth and left as it is. Only the drawing uses
    this; the numbers of the text come from the grid itself."""
    if not ds.startswith("IPTA2"):
        return lnl
    from scipy.ndimage import gaussian_filter
    ok = np.isfinite(lnl)
    sig = (2.0, 2.0) if G is None else (0.005 / float(G[1] - G[0]), 0.05 / float(LOGV[1] - LOGV[0]))
    num = gaussian_filter(np.where(ok, lnl, 0.0), sig, mode="nearest")
    den = gaussian_filter(ok.astype(float), sig, mode="nearest")
    return np.where(ok, num / np.maximum(den, 1e-12), np.nan)


A_ZERO = 1.52    # the speed at which the amplitude bracket of eq. (lv_amp) vanishes
PLATEAU_TOL = 0.1  # in ln L: the fluctuation of the IPTA likelihood between neighbouring cells


def _display_map(G, LOGV, lnl, ds, bracket=False):
    """(Gf, Lf, density, levels) from which a region is drawn: the display map of _display_lnl on
    the fine grid of _fine, levels from the coarse grid. Cells without a completed transition
    would pull the cubic interpolation to zero across a whole cell, so that the edge of a region
    at the boundary of vacuum trapping would follow the cells (a staircase). These cells (the
    negative completion margin of the transition table; the map must be empty there) take the
    value of the nearest cell with a transition before the interpolation, and the edge is drawn
    from the smooth completion margin, as the grey region of figure 5. Cells that are empty for
    another reason (the IPTA likelihood is not defined for spectra far outside the data, at small
    g and v) keep a zero density.
    With bracket=True (the default template, whose amplitude bracket takes (beta/H)_Gamma) the
    same is done at the zero of the bracket: below (beta/H)_Gamma = 1.52 the predicted spectrum
    vanishes and the likelihood is one constant, that of the noise (and foreground) alone; the
    cells in which (beta/H)_Gamma is not positive, a sliver next to vacuum trapping that the scan
    leaves out (0.2% of the IPTA posterior with the foreground if counted), are drawn as part of
    this plateau. With
    the foreground this constant lies within 0.01 of the IPTA 95% level, so that a contour drawn
    through the coarse cells wanders through the plateau. The plateau is then drawn as the
    constant up to the smooth zero line of the table, and the cells beside it are interpolated
    from the side with a signal.
    Where the transition contributes little, the likelihood approaches the same constant. With the
    IPTA foreground this plateau carries 10% of the posterior and the 95% level falls on it (0.006
    below), so that which part of the plateau a 95% contour encloses is decided by the fluctuations
    of the likelihood (0.1 to 0.2 between cells). If the 95% level lies within PLATEAU_TOL of the
    plateau, the whole plateau is drawn as inside: the region is then the set in which the fit is
    not worse than with the noise and foreground alone, and holds more than 95% of the posterior
    (99.9% for IPTA with the foreground; NANOGrav, 0.37 above its plateau, is not affected)."""
    from scipy.ndimage import distance_transform_edt
    disp = _display_lnl(lnl, ds, G, LOGV)
    lv, p = _levels(disp)
    ok = np.isfinite(disp)
    trap = ~(_completion_margin(G, LOGV) >= 0)
    # a map with likelihood in cells the table calls trapped is not of this kind; the smooth edge itself can pass a
    # few cells with likelihood between the coupling rows of the table, which the final mask below removes
    if not trap.any() or (ok & ~(_table_field("margin", G, LOGV, "T_f") >= 0)).any():
        Gf, Lf, pf = _fine(G, LOGV, p)
        return Gf, Lf, pf, lv
    empty, plateau = trap.copy(), None
    if bracket:
        # no signal: the bracket is zero, or (beta/H)_Gamma is not positive (the scan leaves these
        # cells out; the table masks them cell by cell, which would draw a staircase)
        bg = _table_field("get", G, LOGV, "beta_Gamma_H")
        zero = ~trap & ((bg <= A_ZERO) | ~np.isfinite(bg))
        if (zero & np.isfinite(lnl)).any() and np.ptp(lnl[zero & np.isfinite(lnl)]) < 1e-6:
            empty, plateau = trap | zero, float(np.median(lnl[zero & np.isfinite(lnl)]))
    idx = distance_transform_edt(empty, return_distances=False, return_indices=True)
    fdisp = _display_lnl(np.where(empty, np.nan, lnl)[tuple(idx)], ds, G, LOGV)
    m = np.nanmax(disp)
    S = np.sum(np.exp(np.nan_to_num(disp - m, nan=-np.inf)))         # the normalisation of _levels
    Gf, Lf, pf = _fine(G, LOGV, np.exp(np.nan_to_num(fdisp - m, nan=-np.inf)) / S)
    if plateau is not None:
        p_plat = np.exp(plateau - m) / S
        bgf = _table_field("get", Gf, Lf, "beta_Gamma_H")
        pf = np.where((bgf <= A_ZERO) | ~np.isfinite(bgf), p_plat, pf)
        if abs(np.log(lv[0] / p_plat)) < PLATEAU_TOL:
            lv = [p_plat * np.exp(-PLATEAU_TOL)] + list(lv[1:])
    return Gf, Lf, np.where(_completion_margin(Gf, Lf) >= 0, pf, 0.0), lv


# The slow tail of the NANOGrav region ends in a strip along (beta/H)_Gamma ~ 1.6, next to the zero of
# the amplitude bracket of the expansion fit, only 0.0005 to 0.001 wide in g: the scan (0.0025 in g)
# samples it at isolated cells. data/tail_check/tail_patch.npz (reproduce/plane/tail_fine.py) has the default
# NANOGrav likelihood there on a grid 10 x finer in g and 5 x in log10 v.
TAIL_PATCH = os.path.join(DATA, "tail_check", "tail_patch.npz")
# the foreground maps on the same grid (reproduce/binaries/tail_fog.py): lnL_holodeck, lnL_pl (marginalised), lnL_fog_prof
TAIL_FOG = os.path.join(DATA, "tail_check", "tail_fog.npz")
# the population marginalised over the survey-averaged priors on the same grid (reproduce/binaries/tail_fog_prior.py), figure 4
TAIL_FOG_GSA = os.path.join(DATA, "tail_check", "tail_fog_recent.npz")
# the population profiled under the survey-averaged priors on the same grid (reproduce/binaries/tail_fog_prof_prior.py), the foreground figure
TAIL_FOG_PROF_GSA = os.path.join(DATA, "tail_check", "tail_fog_prof_recent.npz")


#: the prior range of v of every fit, binaries or not: 10 MeV to 100 GeV (round numbers, one range for all
#: variants). The scan reaches 3.2 MeV and 10 TeV; the rows outside are dropped on loading. Without binaries
#: no variant has more than 1e-7 of its posterior above 100 GeV; below 10 MeV the largest mass is 3 % (IPTA, hybrid
#: template: its 2 sigma lower end of v 9.0 -> 10.7 MeV). With the binaries the plateau makes the intervals and D
#: depend on the range (survey-averaged priors, NANOGrav: D 1.95, 2.4, 2.9 up to 10, 32, 100 GeV).
PRIOR_LOGV = (-2.0, 2.0)
BINARIES_LOGV = PRIOR_LOGV


class _GridNpz(dict):
    """The arrays of an npz file on the scan grid with every axis of log10 v cut to PRIOR_LOGV."""
    @property
    def files(self):
        return list(self)


def _grid_npz(path):
    with np.load(path, allow_pickle=True) as z:
        d = {k: z[k] for k in z.files}
    L = d["LOGV"]
    keep = (L > PRIOR_LOGV[0] - 1e-9) & (L < PRIOR_LOGV[1] + 1e-9)
    n = len(L)
    for k, a in d.items():
        if isinstance(a, np.ndarray) and a.ndim >= 1 and a.shape[-1] == n:
            d[k] = a[..., keep]
    return _GridNpz(d)


def _binaries_box(lnl, LOGV):
    """A map with the binaries marginalised, NaN (posterior zero) outside BINARIES_LOGV."""
    out = np.array(lnl, dtype=float)
    out[:, (LOGV < BINARIES_LOGV[0] - 1e-9) | (LOGV > BINARIES_LOGV[1] + 1e-9)] = np.nan
    return out


def _patched(G, LOGV, lnl, key, patch=None, ratio=False, box=None):
    """(Gf, Lf, density, levels): the posterior density of a coarse map (flat prior in g, log10 v)
    on the grid of the tail patch extended over the whole plane, cubic between the coarse nodes
    and the patch's own values inside it; normalised on the coarse grid, the 68% and 95% levels
    from the combined grid. With ratio=True the profile-likelihood ratio exp(ln L - max) instead of a density,
    with the levels exp(-3.09), exp(-1.15) of the 2D 2 and 1 sigma regions.
    box=(lo, hi): the posterior is zero outside lo < log10 v < hi (BINARIES_LOGV). The cut is applied after the
    interpolation: a map cut before it steps to zero at the edge, and the cubic interpolation overshoots there and
    draws a hook into the contour that reaches the edge."""
    from scipy.interpolate import RegularGridInterpolator
    P = np.load(patch or TAIL_PATCH)
    PG, PL = P["G"], P["LOGV"]
    dg, dl = float(PG[1] - PG[0]), float(PL[1] - PL[0])
    Gf = np.round(G[0] + dg * np.arange(int(round((G[-1] - G[0]) / dg)) + 1), 6)
    Lf = np.round(LOGV[0] + dl * np.arange(int(round((LOGV[-1] - LOGV[0]) / dl)) + 1), 5)
    inbox = np.ones(len(LOGV), bool) if box is None else (LOGV > box[0] - 1e-9) & (LOGV < box[1] + 1e-9)
    m = np.nanmax(lnl[:, inbox])
    w = np.exp(np.nan_to_num(lnl - m, nan=-np.inf))
    z = w[:, inbox].sum() * float(G[1] - G[0]) * float(LOGV[1] - LOGV[0])
    f = RegularGridInterpolator((G, LOGV), w / z, method="cubic")
    GG, LL = np.meshgrid(Gf, Lf, indexing="ij")
    dens = np.maximum(f(np.stack([GG.ravel(), LL.ravel()], axis=-1)).reshape(GG.shape), 0.0)
    if box is not None:
        dens[:, (Lf < box[0] - 1e-9) | (Lf > box[1] + 1e-9)] = 0.0
    i0, j0 = int(np.argmin(np.abs(Gf - PG[0]))), int(np.argmin(np.abs(Lf - PL[0])))
    assert np.allclose(Gf[i0:i0 + len(PG)], PG) and np.allclose(Lf[j0:j0 + len(PL)], PL), "patch off the grid"
    sub = np.array(P[key], dtype=float)
    shift = np.zeros(sub.shape)
    if key in ("lnL_holodeck", "lnL_pl"):
        # a marginal over the binaries is a Monte Carlo sum, and the patch and the scan have their own draws (up to 0.07 in
        # ln L apart): the patch takes the scan's value at the shared nodes, the difference linear between them, so that
        # no step is drawn at the edge of the patch
        ia, ja = np.where((G >= PG[0] - 1e-9) & (G <= PG[-1] + 1e-9))[0], np.where((LOGV >= PL[0] - 1e-9) & (LOGV <= PL[-1] + 1e-9))[0]
        ip, jp = [int(np.argmin(np.abs(PG - x))) for x in G[ia]], [int(np.argmin(np.abs(PL - x))) for x in LOGV[ja]]
        diff = lnl[np.ix_(ia, ja)] - sub[np.ix_(ip, jp)]
        bad = ~np.isfinite(diff)
        if not bad.all():
            from scipy.ndimage import distance_transform_edt
            diff = diff[tuple(distance_transform_edt(bad, return_distances=False, return_indices=True))]
            GP, LP = np.meshgrid(PG, PL, indexing="ij")
            shift = RegularGridInterpolator((G[ia], LOGV[ja]), diff, bounds_error=False, fill_value=None)(
                np.stack([GP.ravel(), LP.ravel()], axis=-1)).reshape(sub.shape)
        sub = sub + shift
    if f"plateau_{key}" in P.files:
        # cells without a spectrum (the amplitude bracket at zero) take the no-signal value, inside the completion
        # margin, which also sets the edge at vacuum trapping, as _display_map does on the coarse grid; left empty,
        # they would make the contour follow the cells of the transition table (0.005 in g) along the edge
        marg = _completion_margin(PG, PL)
        sub = np.where(~np.isfinite(sub) & (marg >= 0), float(P[f"plateau_{key}"]) + shift, sub)
        sub = np.where(marg >= 0, sub, np.nan)
    dens[i0:i0 + len(PG), j0:j0 + len(PL)] = np.exp(np.nan_to_num(sub - m, nan=-np.inf)) / z
    if ratio:
        # the maximum over both grids, so that a patch cell above the coarse maximum sets the reference
        r = dens * z
        r /= r.max()
        lv = [np.exp(-3.09), np.exp(-1.15), 1.0001]
        return Gf, Lf, _drop_specks(r, lv), lv
    s = np.sort(dens.ravel())[::-1]
    c = np.cumsum(s) / s.sum()
    lv = [s[np.searchsorted(c, 0.95)], s[np.searchsorted(c, 0.68)], s[0] * 1.0001]
    return Gf, Lf, _drop_specks(dens, lv), lv


def _drop_specks(dens, lv, frac=1e-4):
    """The density with every component above a level that holds less than frac of the posterior
    lowered just below that level. At its lower tip the strip of the tail is narrower than the
    patch's cells (0.0001 in g), so that it breaks into single cells there, which a contour would
    draw as a chain of specks; they hold no measurable mass."""
    from scipy import ndimage
    out, tot = dens.copy(), dens.sum()
    for lev in lv[:2]:
        lab, n = ndimage.label(out >= lev)
        for i in range(1, n + 1):
            sel = lab == i
            if out[sel].sum() < frac * tot:
                out[sel] = np.minimum(out[sel], 0.999 * lev)
    return out


def _regions(ax, d, k, filled=True, foreground=False):
    G, LOGV = d["G"], d["LOGV"]
    default = k == list(d["names"]).index("nlo")
    patch = default and bool(_need(TAIL_PATCH))
    for ds, colour in (("NG15", C_NG), ("IPTA2", C_IPTA)):
        if ds == "NG15" and patch:
            Gf, Lf, pf, lv = _patched(G, LOGV, d[ds][k], "lnL")
        else:
            Gf, Lf, pf, lv = _display_map(G, LOGV, d[ds][k], ds, bracket=default)
        if filled:
            # an opaque white copy of the 2 sigma region underneath both data sets: it hides the bound lines, and the
            # colours on top stay light
            ax.contourf(Gf, Lf, pf.T, levels=[lv[0], np.inf], colors=["white"], alpha=1.0, zorder=0.9)
            ax.contourf(Gf, Lf, pf.T, levels=[lv[0], lv[1]], colors=[colour], alpha=A_2SIG)
            ax.contourf(Gf, Lf, pf.T, levels=[lv[1], np.inf], colors=[colour], alpha=A_1SIG)
        else:
            ax.contour(Gf, Lf, pf.T, levels=[lv[0]], colors=[colour], linewidths=0.8)
            ax.contour(Gf, Lf, pf.T, levels=[lv[1]], colors=[colour], linewidths=1.5)
        if foreground:
            # the 68% (thick) and 95% (thin) regions with the binaryfog population marginalised over the
            # survey-averaged priors, those of this work's default: NANOGrav from
            # data/ng15_fog_priors.npz (reproduce/binaries/prior_study.py), IPTA DR2 with the population on
            # its own frequencies from data/ipta_fog_recent.npz (ipta_fog_priors.py)
            fm, key = {"NG15": ("ng15_fog_priors.npz", "lnL_marg_recent"),
                       "IPTA2": ("ipta_fog_recent.npz", "lnL_holodeck")}[ds]
            M = _grid_npz(_need(os.path.join(DATA, fm)))
            assert np.allclose(M["G"], G, rtol=0, atol=1e-12) and np.allclose(M["LOGV"], LOGV, rtol=0, atol=1e-12)
            if ds == "NG15" and patch and _need(TAIL_FOG_GSA):
                # the strip at g ~ 0.41 from the foreground map on the tail-patch grid, as the filled region
                Gf, Lf, pbf, lvb = _patched(G, LOGV, M[key], "lnL_holodeck", patch=TAIL_FOG_GSA, box=BINARIES_LOGV)
            else:
                Gf, Lf, pbf, lvb = _display_map(G, LOGV, _binaries_box(M[key], LOGV), ds, bracket=default)
                pbf = _drop_specks(pbf, lvb, frac=2e-3)
                # where the plateau reaches vacuum trapping the region ends at the grey area: no contour along that
                # edge, which would follow the cells of the display grid (a staircase)
                pbf = np.where(_completion_margin(Gf, Lf) >= 0, pbf, np.nan)
            ax.contour(Gf, Lf, pbf.T, levels=[lvb[1]], colors=[colour], linewidths=0.9, linestyles="--")
            ax.contour(Gf, Lf, pbf.T, levels=[lvb[0]], colors=[colour], linewidths=0.5, linestyles="--")


def _table():
    """reproduce/plane/transition_table.py (exact percolation on the (g, log10 v) plane, data/transition_table.npz)."""
    _reproduce_on_path()
    import transition_table as TT
    return TT


# ---------------------------------------------------------------- figure data
# Two steps of the figures compute instead of read: the transition table evaluated on the grids of fig_plane
# and of the region maps (PCHIP in g and log10 v, about two minutes per field on the 577 x 401 grid of
# _fine; fig_plane and fig_foreground need eleven fields), and the spectra of fig_spectra (TransitionListener
# through plane_scan, whose import also sets up the likelihoods). Their results are kept in figure_data/, one file
# per field, with the SHA-256 digest of everything the field depends on: the files it is computed from, the
# code that computes it and its grid. A build reads a file whose digest matches and otherwise recomputes and
# rewrites it, and says so; a changed input is never read stale, and redrawing computes nothing.
FIGDATA = os.path.join(HERE, "figure_data")
TABLE_NPZ = os.environ.get("TRANSITION_TABLE", os.path.join(DATA, "transition_table.npz"))
_FILE_SHA = {}


def _file_sha(path):
    path = os.path.abspath(path)
    st = os.stat(path)
    key = (path, st.st_mtime_ns, st.st_size)
    if key not in _FILE_SHA:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        _FILE_SHA[key] = h.hexdigest()
    return _FILE_SHA[key]


def _digest(files, arrays=(), extra=()):
    """SHA-256 of the contents of `files` (with their paths relative to the repository), of `arrays` and of
    repr(extra)."""
    h = hashlib.sha256()
    for p in sorted({os.path.abspath(p) for p in files}):
        h.update(os.path.relpath(p, ROOT).encode())
        h.update(_file_sha(p).encode())
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(repr((a.dtype.str, a.shape)).encode())
        h.update(a.tobytes())
    h.update(repr(extra).encode())
    return h.hexdigest()


def _cached(slot, digest, compute):
    """The dict of arrays that compute() returns, read from figure_data/<slot>.npz when that file carries
    `digest`, else computed and saved there."""
    path = os.path.join(FIGDATA, slot + ".npz")
    if os.path.exists(path):
        with np.load(path) as z:
            if str(z["digest"]) == digest:
                return {k: z[k] for k in z.files if k != "digest"}
    out = compute()
    os.makedirs(FIGDATA, exist_ok=True)
    tmp = os.path.join(FIGDATA, f".{slot}.{os.getpid()}.npz")
    np.savez_compressed(tmp, digest=np.array(digest), **out)
    os.replace(tmp, path)
    print(f"  figure data: computed figure_data/{slot}.npz")
    return out


def _table_field(what, g, logv, key=None, variant="nlo"):
    """On the grid of the axes g and log10 v (indexing ij), through the cache: TT.get(key) (what="get") or
    TT.margin(key) (what="margin") of `variant`, or TT.reheating_tl (what="reheating"; the tuple T_reh,
    g_eff, h_eff)."""
    TT = _table()
    g, logv = np.asarray(g, float), np.asarray(logv, float)
    files = [TT.__file__, _need(TABLE_NPZ)] + ([_need(os.path.join(DATA, "tl_nodes_table.npz"))] if what == "reheating" else [])
    names = (what, key, variant) if what != "reheating" else (what,)
    slot = "table_" + "_".join(names) + f"_{len(g)}x{len(logv)}_{_digest([], [g, logv])[:8]}"

    def compute():
        Gm, Lm = np.meshgrid(g, logv, indexing="ij")
        if what == "reheating":
            return dict(zip(("T_reh", "g_eff", "h_eff"), TT.reheating_tl(Gm, Lm)))
        return {"y": (TT.get if what == "get" else TT.margin)(key, Gm, Lm, variant)}
    out = _cached(slot, _digest(files, [g, logv], (names, inspect.getsource(_table_field))), compute)
    return (out["T_reh"], out["g_eff"], out["h_eff"]) if what == "reheating" else out["y"]


def _completion_edge(variant="nlo"):
    """log10(v/GeV) of the edge of vacuum trapping (true-vacuum fraction 0.99 not reached) on each coupling row of
    the transition table, where the row's ln I_end crosses its threshold (linear between the two nodes that bracket
    it), as (g, edge) of the rows that cross inside the table."""
    TT = _table()
    z = TT._ln_i_end(variant)
    thr = float(np.log(-np.log1p(-TT.P_FINAL)))
    gs, xs = [], []
    for i, g in enumerate(TT.G_TAB):
        j = np.flatnonzero((z[i, :-1] >= thr) & (z[i, 1:] < thr))
        if len(j):
            j = j[0]
            gs.append(g)
            xs.append(TT.LOGV_TAB[j] + (z[i, j] - thr) / (z[i, j] - z[i, j + 1]) * (TT.LOGV_TAB[j + 1] - TT.LOGV_TAB[j]))
    return np.array(gs), np.array(xs)


def _completion_margin(g, logv, variant="nlo"):
    """On the grid of the axes g and log10 v (indexing ij): positive where the transition completes, negative in the
    vacuum-trapped region, zero on its edge, in units of ln I_end (the edge distance in log10 v times 4 ln 10, the
    slope of ln I_end along log10 v at fixed g). The edge is the PCHIP in g of the row crossings of
    _completion_edge. TT.margin interpolates ln I_end itself in both directions; between the coupling rows of
    the table (0.005 apart) that field bends differently at each v, and its zero line wiggles between the
    rows, while the row crossings lie on a smooth curve. Both edges coincide on the rows."""
    from scipy.interpolate import PchipInterpolator
    gs, xs = _completion_edge(variant)
    assert np.all(np.diff(xs) > 0), "the edge of vacuum trapping is not monotonic in g"
    g, logv = np.asarray(g, float), np.asarray(logv, float)
    x = np.where(g <= gs[-1], PchipInterpolator(gs, xs, extrapolate=True)(np.minimum(g, gs[-1])), np.inf)
    return 4.0 * np.log(10.0) * (x[:, None] - logv[None, :])


# 3 to 25: the 30 contour leaves the plotted range through its right edge, below the axis that
# carries the labels
BH_LEVELS = [2, 3, 4, 5, 6, 8, 10, 15, 20, 25]
BH_UNTAGGED = [30, 35, 40, 45]   # drawn but not named on the upper axis


def _plane_overlays(ax, gg, lv, labels=True, main=True):
    """What the transition itself says about the plane, from the exact-percolation table:
    contours of (beta/H)_RH; the region without a completed transition (grey: no percolation,
    or no final temperature T_f with a true-vacuum fraction of 0.99); the local criterion of
    Ellis et al., to the left of which a^3 P_f still grows at T_p (informative only: the
    transition completes on both sides); and the edge of the simulated range of the default
    template, beta_Gamma = 4 H_V (ref. 2511.15687)."""
    bh = _table_field("get", gg, lv, "betaH_R")
    tf = _table_field("get", gg, lv, "T_f")
    dl = _table_field("get", gg, lv, "dlnV_dt_over_H")
    # drawn from the smooth completion margin, whose zero is the edge (T_f needs more than percolation)
    mg = _completion_margin(gg, lv)
    nop = mg < 0
    # the table's own NaN pattern of T_f differs from it only next to the edge, between the coupling rows
    near = np.abs(mg) < 4.0 * np.log(10.0) * 0.1
    assert np.array_equal(nop[~near], (~np.isfinite(bh) | ~np.isfinite(tf))[~near]), \
        "the completion margin and the NaN pattern of the table disagree away from the edge"
    ax.contourf(gg, lv, mg.T, levels=[-np.inf, 0.0], colors=["0.65"], zorder=0)
    cs = ax.contour(gg, lv, bh.T, levels=BH_LEVELS + BH_UNTAGGED, colors=["0.55"], linewidths=0.5,
                    linestyles=":", zorder=2.5)   # over the regions and their contours; the stars above them
    if labels:
        # the values on a secondary axis above the frame, at the points where the contours meet the
        # upper edge: a label along a contour this steep would cross the frame or other curves
        row = bh[:, -1]
        ok = np.isfinite(row)
        ticks, names = [], []
        for L in BH_LEVELS:
            if ok.sum() > 1 and row[ok].min() < L < row[ok].max():
                ticks.append(float(np.interp(L, row[ok], gg[ok])))
                names.append(f"{L:d}")
        top = ax.secondary_xaxis("top")
        top.set_xticks(ticks)
        top.set_xticklabels(names, fontsize=7)
        top.tick_params(which="minor", top=False)
        top.tick_params(direction="in", length=3, color="0.45", pad=5)
        top.set_xlabel(r"$(\beta/H)_R$ along the dotted lines", fontsize=8, labelpad=3)
        ax.tick_params(which="both", top=False)
    # the local criterion is informative, dotted; grey in the appendix panels, where red marks the bounds
    # three lines in bright colours that differ from each other and from the data's purple and yellow
    # P4 dark orange, the GW-template line magenta, the N4 edge dark blue, the N5 edge cyan
    ax._p4 = ax.contour(gg, lv, dl.T, levels=[0.0], colors=[C_P4 if main else "0.25"], linewidths=1.1,
                        linestyles="-", zorder=Z_BOUND)
    if main:
        # the simulated range of the default template in its own normalisation, beta/H_V >= 4
        # (beta/H_V = (beta/H)_Gamma H_p/H_V; a variant of the scan restricts the template to it). On this line H_p/H_V = 0.927 to
        # 0.936 (measured on the table), so it is (beta/H)_Gamma = 4.3 in the standard definition
        bhV = _table_field("get", gg, lv, "beta_Gamma_H") * _table_field("get", gg, lv, "H_p_over_HV")
        ax._g6 = ax.contour(gg, lv, bhV.T, levels=[4.0], colors=[C_G6], linewidths=1.1, linestyles="-", zorder=Z_BOUND)
    return nop


G_LEFT, G_RIGHT = 0.40, 0.60
A_1SIG, A_2SIG = 0.6, 0.25   # fill of the 1 and 2 sigma regions, over an opaque white copy of the 2 sigma region
Z_BOUND = 0.5    # the bound lines and their arrows behind the preferred regions
G6_LABEL_LV = (-0.25, -0.05)     # log10 v range for the centre of the beta label
G6_ARROW_FRACS = (0.3, 0.7)    # along the beta line, below and above its label
P4_ARROW_FRACS = (0.45, 0.8)   # along the P4 line, below its label
SIDE_ARROW_FRACS = (0.3, 0.7)   # where the arrows leave the vacuum-trapping and beta edges, along their visible length
DAISY_EDGE_STYLE = dict(color=DESYdunkelblau, lw=1.2, ls="-")        # the N4 edge in fig_plane: solid, dashed is the binaries
DAISY6_EDGE_STYLE = dict(color=DESYcyan, lw=1.2, ls="-")      # the N5 edge
# the edges of vacuum trapping drawn in fig_plane: (row of the scan, tag, style, legend text)
DAISY_EDGES = (("4D daisy RG", "N4", DAISY_EDGE_STYLE, r"N4: vacuum trapping for $V_{\rm 4D}$ + Daisy at $\mu = \pi T$"),
               ("4D daisy", "N5", DAISY6_EDGE_STYLE, r"N5: vacuum trapping for $V_{\rm 4D}$ + Daisy at $\mu = v$"))
C_P4, C_G6 = DESYorange, DESYmagenta                        # the Ellis-Lewicki-No line and the slowest simulation




def _place_beside(fig, ax, cs, text, colour, logvs, hint, fontsize=6.5, gap_pt=1.5):
    """Text parallel to the longest segment of the contour set cs, gap_pt beside it on the side given by hint (a
    direction in display coordinates), centred at the candidate log10 v whose rotated box covers the fewest non-white
    pixels. For a line that runs next to another one, where a label on the line would cover both. Returns the Text."""
    seg = max(cs.allsegs[0], key=len)
    o = np.argsort(seg[:, 1]); sy, sx = seg[o, 1], seg[o, 0]
    tmp = ax.text(0, 0, text, fontsize=fontsize); fig.canvas.draw()
    bb = tmp.get_window_extent(); tmp.remove()
    w, h = bb.width, bb.height
    fig.canvas.draw(); img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    best = None
    for lvv in logvs:
        p0 = ax.transData.transform((np.interp(lvv - 0.05, sy, sx), lvv - 0.05))
        p1 = ax.transData.transform((np.interp(lvv + 0.05, sy, sx), lvv + 0.05))
        t = (p1 - p0) / np.hypot(*(p1 - p0))
        n = np.array([-t[1], t[0]])
        if n @ np.asarray(hint, float) < 0:
            n = -n
        ang = np.degrees(np.arctan2(t[1], t[0]))
        ang = ang - 180 if ang > 90 else ang + 180 if ang < -90 else ang
        c = ax.transData.transform((np.interp(lvv, sy, sx), lvv)) + (gap_pt * fig.dpi / 72 + h / 2) * n
        b = _busy(fig, img, c[0], c[1], w, h, ang, mask_grey=False)
        if best is None or b < best[1]:
            best = (c, b, ang, lvv)
    c, b, ang, lvv = best
    xy = ax.transData.inverted().transform(c)
    txt = ax.text(*xy, text, color=colour, fontsize=fontsize, rotation=ang, rotation_mode="anchor", ha="center",
                  va="center", zorder=3.3)
    print(f"  label {text!r} beside its line at log10 v = {lvv:.2f}, {100 * b:.1f}% of its box covered")
    return txt


def _side_band(fig, ax, cs, hint, width_pt=3.0, colour="0.65", alpha=0.7):
    """A strip width_pt wide along the longest segment of the contour set cs, on the side given by hint (a direction
    in display coordinates), in the colour of the vacuum-trapping region: it marks the line as the edge of such a
    region."""
    from matplotlib.patches import Polygon
    seg = max(cs.allsegs[0], key=len)
    P = ax.transData.transform(seg)
    t = np.gradient(P, axis=0)
    n = np.column_stack([-t[:, 1], t[:, 0]]) / np.hypot(t[:, 0], t[:, 1])[:, None]
    n[n @ np.asarray(hint, float) < 0] *= -1
    Q = P + width_pt * fig.dpi / 72 * n
    poly = ax.transData.inverted().transform(np.vstack([P, Q[::-1]]))
    return ax.add_patch(Polygon(poly, closed=True, facecolor=colour, edgecolor="none", alpha=alpha, zorder=2.4))


def _arrows_around(fig, ax, cs, label, colour, hint, gap_pt=8.0, length_pt=7.0, lw=0.8, avoid=(), clear_pt=4.0,
                   sides=(-1, 1), at_lv=None, **_):
    """Two arrows from the line of the contour set cs, perpendicular to it towards the side given by hint (a direction
    in display coordinates), one gap_pt before and one gap_pt after its inline label: the points gap_pt beyond the ends
    of the text along its own direction, moved to the nearest point of the line; an arrow within clear_pt of a line in
    avoid moves on, away from the label, until it is clear."""
    from matplotlib.patches import FancyArrowPatch
    P = ax.transData.transform(np.concatenate(cs.allsegs[0]))
    bb = label.get_window_extent()
    c = np.array([(bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2])
    a_ = np.radians(label.get_rotation())
    u = np.array([np.cos(a_), np.sin(a_)])
    tmp = ax.text(0, 0, label.get_text(), fontsize=label.get_fontsize()); fig.canvas.draw()
    half = tmp.get_window_extent().width / 2; tmp.remove()
    L, g = length_pt * fig.dpi / 72, gap_pt * fig.dpi / 72
    box = ax.get_window_extent()
    others = [ax.transData.transform(np.concatenate(o.allsegs[0])) for o in avoid if o.allsegs[0]]
    for sgn in sides:
        for step in range(40):
            q = c + sgn * (half + g + step * 2 * fig.dpi / 72) * u
            if at_lv is not None and sgn in at_lv:
                # a fixed height for this arrow: the point of the line at log10 v = at_lv[sgn]
                q = ax.transData.transform((0.0, at_lv[sgn]))
                q = P[np.argmin(np.abs(P[:, 1] - q[1]))]
            p0 = P[int(np.argmin(np.hypot(*(P - q).T)))]
            near = P[np.hypot(*(P - p0).T) < 6 * fig.dpi / 72]
            t = near[-1] - near[0] if len(near) > 1 else u
            n = np.array([-t[1], t[0]]) / np.hypot(*t)
            if n @ np.asarray(hint, float) < 0:
                n = -n
            seg_ = p0 + np.linspace(0, L, 8)[:, None] * n
            if all(np.min(np.hypot(*(O[:, None, :] - seg_[None]).T)) > clear_pt * fig.dpi / 72 for O in others):
                break
        if not (box.x0 < p0[0] < box.x1 and box.y0 < p0[1] < box.y1):
            continue
        a, b = ax.transData.inverted().transform([p0, p0 + L * n])
        ax.add_patch(FancyArrowPatch(tuple(a), tuple(b), arrowstyle="-|>", color=colour, lw=lw, mutation_scale=6,
                                     shrinkA=0, shrinkB=0, zorder=Z_BOUND + 0.1))

def _side_arrows(fig, ax, cs, colour, hint, fracs=(0.25, 0.75), length_pt=7.0, lw=0.8, joined=False):
    """Short arrows from the longest segment of the contour set cs, perpendicular to it, towards the side given by
    hint (a direction in display coordinates): the line is the edge of a region on that side. fracs
    are positions along the visible arc length of the segment."""
    from matplotlib.patches import FancyArrowPatch
    # joined: all pieces of a line that runs along g, in order of g (an inline label splits it in two)
    seg = (np.concatenate(cs.allsegs[0])[np.argsort(np.concatenate(cs.allsegs[0])[:, 0])] if joined
           else max(cs.allsegs[0], key=len))
    P = ax.transData.transform(seg)
    box = ax.get_window_extent()
    inside = (P[:, 0] > box.x0) & (P[:, 0] < box.x1) & (P[:, 1] > box.y0) & (P[:, 1] < box.y1)
    P = P[inside]
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
    L = length_pt * fig.dpi / 72
    out = []
    for f in fracs:
        i = int(np.clip(np.searchsorted(arc, f * arc[-1]), 1, len(P) - 2))
        t = P[i + 1] - P[i - 1]
        n = np.array([-t[1], t[0]]) / np.hypot(*t)
        if n @ np.asarray(hint, float) < 0:
            n = -n
        a, b = ax.transData.inverted().transform([P[i], P[i] + L * n])
        # a patch, not an annotation: it marks a side and is no leader of a label
        out.append(ax.add_patch(FancyArrowPatch(tuple(a), tuple(b), arrowstyle="-|>", color=colour, lw=lw,
                                                mutation_scale=6, shrinkA=0, shrinkB=0, zorder=3)))
    return out


def _busy(fig, img, cx, cy, w, h, ang, mask_grey=False):
    """Fraction of pixels that differ from white inside a rotated box (display pixels, centre cx, cy,
    width w, height h, angle ang in degrees), in an image of the canvas drawn without the label."""
    H = img.shape[0]
    c, s = np.cos(np.radians(ang)), np.sin(np.radians(ang))
    u = np.linspace(-w / 2, w / 2, max(int(w), 2)); v = np.linspace(-h / 2, h / 2, max(int(h), 2))
    U, Vv = np.meshgrid(u, v)
    X = np.round(cx + U * c - Vv * s).astype(int); Y = np.round(cy + U * s + Vv * c).astype(int)
    ok = (X >= 0) & (X < img.shape[1]) & (Y >= 0) & (Y < H)
    if not ok.all():
        return 1.0
    px = img[H - 1 - Y, X].astype(int)
    if not mask_grey:
        return float(np.mean(np.max(np.abs(px - 255), axis=-1) > 12))
    # coloured or dark pixels count; light neutral grey (the dotted guides, which a white patch may
    # interrupt) does not, unless it covers a quarter of the box (a grey fill)
    chroma = px.max(axis=-1) - px.min(axis=-1)
    coloured = (chroma > 25) | (px.max(axis=-1) < 90)
    grey = ~coloured & (px.min(axis=-1) < 243)
    return float(np.mean(coloured)) + (1.0 if np.mean(grey) > 0.25 else 0.0)


def _place_inline(fig, ax, cs, text, colour, logvs, fontsize=6.5, pad=2.5, frame_pt=4.0, hide=(), rotated=True,
                  tint=None):
    """Label the contour line cs along itself (clabel, inline) at the candidate log10 v where the box of
    the label, with the line itself hidden, covers the fewest non-white pixels and keeps frame_pt from
    the frame. Returns the chosen log10 v and its busy fraction."""
    seg = max(cs.allsegs[0], key=len)
    order = np.argsort(seg[:, 1]); sy, sx = seg[order, 1], seg[order, 0]
    tmp = ax.text(0, 0, text, fontsize=fontsize); fig.canvas.draw()
    bb = tmp.get_window_extent(); tmp.remove()
    w, h = bb.width + 2 * pad * fig.dpi / 72, bb.height + 2 * pad * fig.dpi / 72
    # the labelled line, and lines that run under it (hide), are left out of the measured image: the white
    # box behind the label interrupts them
    for a in (cs,) + tuple(hide):
        a.set_visible(False)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    for a in (cs,) + tuple(hide):
        a.set_visible(True)
    frame = ax.get_window_extent(); margin = frame_pt * fig.dpi / 72
    best = None
    for lvv in logvs:
        x = float(np.interp(lvv, sy, sx))
        p0 = ax.transData.transform((float(np.interp(lvv - 0.05, sy, sx)), lvv - 0.05))
        p1 = ax.transData.transform((float(np.interp(lvv + 0.05, sy, sx)), lvv + 0.05))
        ang = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
        ang = ang - 180 if ang > 90 else ang + 180 if ang < -90 else ang
        cx, cy = ax.transData.transform((x, lvv))
        half = 0.5 * (abs(w * np.cos(np.radians(ang))) + abs(h * np.sin(np.radians(ang)))), \
            0.5 * (abs(w * np.sin(np.radians(ang))) + abs(h * np.cos(np.radians(ang))))
        if cx - half[0] < frame.x0 + margin or cx + half[0] > frame.x1 - margin or \
                cy - half[1] < frame.y0 + margin or cy + half[1] > frame.y1 - margin:
            continue
        # the axis-aligned box of the rotated label, which is what the figure check reads
        # rotated: the box of the text itself, turned with it; the axis-aligned box of a label on a steep line is mostly
        # empty and its white patch would cut the lines next to the label
        b = (_busy(fig, img, cx, cy, w, h, ang, mask_grey=True) if rotated
             else _busy(fig, img, cx, cy, 2 * half[0], 2 * half[1], 0.0, mask_grey=True))
        if os.environ.get("FIG_LABEL_DEBUG"):
            Hh = img.shape[0]; c_, s_ = np.cos(np.radians(ang)), np.sin(np.radians(ang))
            U, Vv = np.meshgrid(np.linspace(-w / 2, w / 2, int(w)), np.linspace(-h / 2, h / 2, int(h)))
            X = np.round(cx + U * c_ - Vv * s_).astype(int); Y = np.round(cy + U * s_ + Vv * c_).astype(int)
            px = img[Hh - 1 - Y, X].astype(int); chroma = px.max(-1) - px.min(-1)
            col = px[(chroma > 25) | (px.max(-1) < 90)]
            top = []
            if len(col):
                u, n = np.unique(col // 32 * 32, axis=0, return_counts=True)
                top = [(tuple(int(v) for v in u[i]), int(n[i])) for i in np.argsort(-n)[:3]]
            print(f"    {text}: log10 v {lvv:+.2f} g {x:.4f} angle {ang:5.1f} covered {100 * b:5.1f}% {top}")
        if best is None or b < best[1]:
            best = (lvv, b, x)
    if best is None:
        return None
    ax.clabel(cs, [cs.levels[0]], inline=True, inline_spacing=min(5, 2 * pad), fontsize=fontsize, colors=[colour],
              fmt={cs.levels[0]: text}, manual=[(best[2], best[0])])
    # above every line (the P4 line runs over the edges), the label above its box
    _white_patch(fig, ax, cs.labelTexts[-1], pad, rotated=rotated, zorder=3.2, tint=tint)
    cs.labelTexts[-1].set_zorder(3.3)
    print(f"  label {text!r} along its line at log10 v = {best[0]:.2f}, {100 * best[1]:.1f}% of its box covered")
    return best[:2]


def _place_text(fig, ax, text, candidates, fontsize=6.5, pad=2.5, frame_pt=4.0, mask_grey=True, target=None,
                tol=0.002, own=None, ratio=2.0, leader_of=None, own_min_pt=3.0, **kw):
    """Put text at the candidate (x, y) (data coordinates, ha=left, va=center) whose box covers the fewest
    non-white pixels of the canvas as drawn, clear of the frame. Returns the Text."""
    tmp = ax.text(0, 0, text, fontsize=fontsize, ha="left", va="center", **kw); fig.canvas.draw()
    bb = tmp.get_window_extent(); tmp.remove()
    w, h = bb.width + 2 * pad * fig.dpi / 72, bb.height + 2 * pad * fig.dpi / 72
    fig.canvas.draw(); img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    frame = ax.get_window_extent(); margin = frame_pt * fig.dpi / 72
    gap = 2.0 * fig.dpi / 72
    others_bb = [mpl.transforms.Bbox.from_extents(o.x0 - gap, o.y0 - gap, o.x1 + gap, o.y1 + gap)
                 for o in (mpl.text.Text.get_window_extent(x_) for x_ in ax.texts
                           if x_.get_visible() and x_.get_text().strip())]
    best = None
    why = {"frame": 0, "leader": 0, "nearest": 0, "kept": 0}
    for xy in candidates:
        cx, cy = ax.transData.transform(xy)
        cx += w / 2 - pad * fig.dpi / 72
        if cx - w / 2 < frame.x0 + margin or cx + w / 2 > frame.x1 - margin or \
                cy - h / 2 < frame.y0 + margin or cy + h / 2 > frame.y1 - margin:
            why["frame"] += 1
            continue
        b = _busy(fig, img, cx, cy, w, h, 0.0, mask_grey=mask_grey)
        if target is not None and others_bb:
            # the leader, from the label to its target, may not cross an existing label
            tx, ty = ax.transData.transform(target)
            s = np.linspace(0.0, 1.0, 60)[:, None]
            seg = np.array([cx, cy]) + s * (np.array([tx, ty]) - np.array([cx, cy]))
            outside = (np.abs(seg[:, 0] - cx) > w / 2) | (np.abs(seg[:, 1] - cy) > h / 2)
            seg = seg[outside]
            if any(((seg[:, 0] > o.x0) & (seg[:, 0] < o.x1) & (seg[:, 1] > o.y0) & (seg[:, 1] < o.y1)).any()
                   for o in others_bb):
                why["leader"] += 1
                continue
            if leader_of is not None and len(seg):
                # nor another plotted line: the leader reaches its curve without passing a different one
                from figure_checks import plot_lines, line_distances
                hit = False
                for ln in plot_lines(ax):
                    if ln is leader_of:
                        continue
                    if min(line_distances(fig, ax, px, py, 0.0, 0.0, [ln])[0] for px, py in seg[::3]) < 1.0:
                        hit = True
                        break
                if hit:
                    why["leader"] += 1
                    continue
        if own is not None:
            # the nearest plotted line is the labelled one; every other line
            # is `ratio` times as far (2 by default, relaxed in steps to 1 where nothing passes), 3 pt further
            # and 5 pt away, so that the reading is unambiguous
            from figure_checks import plot_lines, line_distances
            lines = plot_lines(ax)
            ds = line_distances(fig, ax, cx, cy, w - 2 * pad * fig.dpi / 72, h - 2 * pad * fig.dpi / 72, lines)
            d_own = ds[lines.index(own)]
            # and the label keeps a padding of own_min_pt from its own curve
            if d_own < own_min_pt:
                why["nearest"] += 1
                continue
            if any(d <= max(ratio * d_own, d_own + (3.0 if ratio > 1.0 else 2.0), 5.0)
                   for ln, d in zip(lines, ds) if ln is not own):
                why["nearest"] += 1
                continue
        why["kept"] += 1
        # with a target (the point a leader runs to), the clean candidate nearest to it; with only an own
        # curve, the clean candidate nearest to that curve
        dist = 0.0 if target is None else float(np.hypot(*(np.array([cx, cy]) - ax.transData.transform(target))))
        if target is None and own is not None:
            target, dist = None, d_own
        key = (round(b / tol), dist) if (target is not None or own is not None) else (b, 0.0)
        if best is None or key < best[2]:
            best = (xy, b, key)
    if best is None and own is not None and ratio > 1.0:
        # the reading must stay unambiguous: relax the factor only down to "nearest by 3 pt"
        return _place_text(fig, ax, text, candidates, fontsize, pad, frame_pt, mask_grey, target, tol, own,
                           ratio=max(1.0, ratio - 0.25), **kw)
    if best is None:
        print(f"  _place_text: no place for {text!r}: {why}")

    return None if best is None else best[:2]


def _white_patch(fig, ax, txt, pad, rotated=False, zorder=2.9, tint=None):
    """A white box behind a label, its axis-aligned bounding box sized in points: it interrupts the dotted
    guides under the label as an inline contour label does. tint: a rounded box filled with that colour at 20% over
    white (opaque, so that the fill is uniform), for the labels of the bounds in fig_plane."""
    import matplotlib.transforms as mtr
    fig.canvas.draw()
    bb = txt.get_window_extent()
    # the axis-aligned box of the text (rotated or not), which is what the figure check reads; rotated: the box of
    # the unrotated text, turned with it
    ang = txt.get_rotation() if rotated else 0.0
    if rotated:
        tmp = ax.text(0, 0, txt.get_text(), fontsize=txt.get_fontsize(), linespacing=txt._linespacing)
        fig.canvas.draw()
        b0 = tmp.get_window_extent(); tmp.remove()
        w, h = b0.width * 72 / fig.dpi, b0.height * 72 / fig.dpi
    else:
        w, h = bb.width * 72 / fig.dpi, bb.height * 72 / fig.dpi
    # centred on the text's own box (its anchor is the left edge for the labels with leaders)
    x, y = ax.transData.inverted().transform(((bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2))
    tr = (mtr.Affine2D().scale(1.0 / 72).rotate_deg(ang) + fig.dpi_scale_trans
          + mtr.ScaledTranslation(x, y, ax.transData))
    # just below the text: it covers the dotted guides and the ends of a broken labelled line, and
    # nothing else, since the placement keeps every label clear of coloured lines and regions
    if tint is None:
        ax.add_patch(Rectangle((-w / 2 - pad, -h / 2 - pad), w + 2 * pad, h + 2 * pad, transform=tr,
                               facecolor="white", edgecolor="none", zorder=zorder))
    else:
        from matplotlib.patches import FancyBboxPatch
        fc = 0.2 * np.asarray(mpl.colors.to_rgb(tint)) + 0.8
        ax.add_patch(FancyBboxPatch((-w / 2, -h / 2), w, h, boxstyle=f"round,pad={pad},rounding_size={pad + 1.5}",
                                    transform=tr, facecolor=fc, edgecolor="none", zorder=zorder))


def _close_label_gaps(fig, ax, gap_pt=12.0):
    """A white box over the gap between two labels side by side and closer than gap_pt: each label's white patch
    interrupts the dotted guides under it, and a guide crossing both would keep a fragment of a few points between
    them."""
    fig.canvas.draw()
    boxes = [t.get_window_extent() for t in ax.texts if t.get_visible() and t.get_text().strip()]
    inv = ax.transData.inverted()
    tol = gap_pt * fig.dpi / 72
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            l, r = (a, b) if a.x1 <= b.x0 else (b, a)
            if r.x0 - l.x1 <= 0 or r.x0 - l.x1 > tol:
                continue
            y0, y1 = max(a.y0, b.y0), min(a.y1, b.y1)
            if y1 <= y0:
                continue
            (gx0, gy0), (gx1, gy1) = inv.transform([(l.x1, y0), (r.x0, y1)])
            ax.add_patch(Rectangle((gx0, gy0), gx1 - gx0, gy1 - gy0, facecolor="white", edgecolor="none", zorder=2.9))


def _daisy_edge(ax, d, row="4D daisy", style=None):
    """The edge of the region in which the scan's 4D-daisy likelihood is finite (the daisy rate
    completes the transition there), drawn from the mask smoothed with a Gaussian of 0.004 in g
    and 0.15 in log10 v and interpolated to a four times finer grid. The mask steps at the cells
    of the scan (0.0025 in g; the daisy table has nodes 0.0025 apart along the edge), and the
    smoothing turns the steps into the line through their middles. Returns the contour set, or None when
    the edge does not cross the plotted range."""
    from scipy.ndimage import gaussian_filter
    from scipy.interpolate import RegularGridInterpolator
    G, LOGV, names = d["G"], d["LOGV"], list(d["names"])
    style = DAISY_EDGE_STYLE if style is None else style
    if row not in names:
        return None
    ok = np.isfinite(d["NG15"][names.index(row)]).astype(float)
    if ok.min() == ok.max():
        return None
    sm = gaussian_filter(ok, (0.004 / float(G[1] - G[0]), 0.15 / float(LOGV[1] - LOGV[0])), mode="nearest")
    Gf = np.linspace(G[0], G[-1], 4 * (len(G) - 1) + 1)
    Lf = np.linspace(LOGV[0], LOGV[-1], 4 * (len(LOGV) - 1) + 1)
    GG, LL = np.meshgrid(Gf, Lf, indexing="ij")
    smf = RegularGridInterpolator((G, LOGV), sm)(np.stack([GG.ravel(), LL.ravel()], -1)).reshape(GG.shape)
    inside = (GG >= G_LEFT) & (GG <= G_RIGHT)
    if smf[inside].min() > 0.5 or smf[inside].max() < 0.5:
        return None
    return ax.contour(Gf, Lf, smf.T, levels=[0.5], colors=[style["color"]],
                      linewidths=style["lw"], linestyles=style["ls"], zorder=3)


def _fg_handle(lw):
    """The legend entry of the dashed regions with the population: NANOGrav and IPTA."""
    ng = plt.Line2D([], [], color=C_NG, ls="--", lw=lw)
    return (ng, plt.Line2D([], [], color=C_IPTA, ls="--", lw=lw))


def fig_plane(which="main"):
    """which="main": the PTA-preferred regions with what the transition itself implies
    (vacuum trapping, the local criterion, the template's simulated range, the edge of the 4D
    daisy rate) and contours of (beta/H)_RH.
"""
    d = dict(_grid_npz(_need(os.path.join(DATA, SCAN_NPZ))))
    G, LOGV, names = d["G"], d["LOGV"], list(d["names"])
    k = names.index("nlo")
    ylo, yhi = LOGV[0], LOGV[-1]
    if which == "main":
        # the prior range reaches down to v = 0.01 GeV, but no region of the defaults comes below 0.05 GeV: drawn from 0.03
        ylo = max(ylo, -1.5)
        # and up to 100 GeV; the regions drawn here end below 10 GeV
        yhi = min(yhi, 1.5)
    gg = np.linspace(G_LEFT, G_RIGHT, 161)
    lv = np.linspace(ylo, yhi, 201)
    if which == "main":
        fig, ax = plt.subplots(figsize=(width("fig_plane"), 3.75), layout="constrained")
        axes = [ax]
        nop_ = _plane_overlays(ax, gg, lv)
        _regions(ax, d, k, filled=True, foreground=True)
        tr = _table_field("reheating", gg, lv)[0]
        # no shading for late reheating, whose bound depends on the analysis, but a line at the 95% lower bound of
        # nucleosynthesis, the CMB and galaxy surveys, T_reh > 5.96 MeV (Barbieri:2025moq)
        ax._treh = ax.contour(gg, lv, np.log10(tr).T, levels=[np.log10(5.96e-3)], colors=[C_TREH], linewidths=0.9,
                              linestyles="-", zorder=Z_BOUND - 0.1)   # behind the other bounds
        # T_reh = 60 and 600 MeV as guides over the regions, as thin and light as the (beta/H)_R contours; the values
        # 6, 60 and 600 MeV on a right-hand axis where the lines meet the right edge, as (beta/H)_RH on the upper axis
        ltr = np.where(nop_, np.nan, np.log10(tr))     # no reheating where the transition does not complete
        ax._treh_guides = ax.contour(gg, lv, ltr.T, levels=np.log10(TREH_GUIDES), colors=["0.7"], linewidths=0.3,
                                     linestyles=[LS_TREH_GUIDE], zorder=2.5)
        col_ = ltr[-1, :]
        ok_ = np.isfinite(col_)
        assert np.all(np.diff(col_[ok_]) > 0), "T_reh not monotonic in v at the right edge"
        rticks = [(float(np.interp(np.log10(T_), col_[ok_], lv[ok_])), f"{1e3 * T_:.0f}") for T_ in [5.96e-3] + TREH_GUIDES
                  if col_[ok_].min() < np.log10(T_) < col_[ok_].max()]
        right_ = ax.secondary_yaxis("right")
        right_.set_yticks([y_ for y_, _ in rticks])
        right_.set_yticklabels([t_ for _, t_ in rticks], fontsize=7)
        right_.tick_params(which="minor", right=False)
        right_.tick_params(direction="in", length=3, color="0.45", pad=5)
        right_.spines["right"].set_visible(False)      # the frame is the main axes' own
        right_.set_ylabel(r"$T_{\rm reh}$ along the dashed lines  [MeV]", fontsize=8, labelpad=3)
        print("  fig_plane: T_reh guides meet the right edge at v = "
              + ", ".join(f"{10 ** y_:.2f} GeV ({t_} MeV)" for y_, t_ in rticks))
        # the N4 and N5 edges are drawn only with PLANE_EDGES; the variants are discussed in section 4.1
        edges = {tag_: None for _, tag_, _, _ in DAISY_EDGES}
        for row_, tag_, style_, _ in DAISY_EDGES if PLANE_EDGES else ():
            edges[tag_] = _daisy_edge(ax, d, row_, style_)
            if edges[tag_] is None:
                print(f"  fig_plane: the {row_!r} row of the scan has no edge inside the plotted range; none drawn")
        # the maximised best fits at the defaults (data/best_fits.json, reproduce/plane/best_fits.py)
        BF = json.load(open(_need(os.path.join(DATA, "best_fits.json"))))["nlo"]
        for ds, colour in (("NG15", C_NG), ("IPTA2", C_IPTA)):
            gb, lb = BF[ds]["at"]
            ax.plot([gb], [lb], marker="*", ms=9, mfc=colour, mec="0.1", mew=0.6, ls="none", zorder=5)
        # the remaining entries in three columns of two
        h = [((Patch(facecolor=C_NG, alpha=A_1SIG), Patch(facecolor=C_NG, alpha=A_2SIG)), r"NANOGrav 15$\,$yr, $1\sigma$, $2\sigma$"),
             ((Patch(facecolor=C_IPTA, alpha=A_1SIG), Patch(facecolor=C_IPTA, alpha=A_2SIG)), r"IPTA DR2, $1\sigma$, $2\sigma$"),
             # filled column by column: the best fits under the data sets on the left
             ((plt.Line2D([], [], marker="*", ms=8, mfc=C_NG, mec="0.1", mew=0.6, ls="none"),
               plt.Line2D([], [], marker="*", ms=8, mfc=C_IPTA, mec="0.1", mew=0.6, ls="none")), "Best fits"),
             (_fg_handle(0.9), r"$1\sigma$ with binary population"),
             (_fg_handle(0.5), r"$2\sigma$ with binary population"),
             (Patch(facecolor="0.65"), "Vacuum trapping"),
             *[(plt.Line2D([], [], **style_), text_)
               for _, tag_, style_, text_ in DAISY_EDGES if edges[tag_] is not None]]
        # 7 pt drawn at SCALE 1.15 prints at 8 pt, as the legend of figure 3 (6 pt at 4/3)
        fig.legend([a for a, _ in h], [b for _, b in h], loc="outside lower center", ncol=2, fontsize=7.0,
                   handlelength=2.8, columnspacing=1.8, **_tuple_legend())
        # the axes and the layout are fixed before the labels are placed: the search measures the final axes
        ax.set_ylim(ylo, yhi)
        _g_axis(ax)
        _log_v_axis(ax)
        ax.tick_params(which="both", right=False)      # the right-hand side carries the T_reh axis
        from figure_checks import freeze
        freeze(fig)
        # the three lines labelled along themselves, each where the label covers nothing else (searched
        # along the line); the Ellis-Lewicki-No line, too short for that, by a text beside it with a leader
        lvs = np.linspace(ylo + 0.1, yhi - 0.1, 61)
        # short tags on the lines, the meaning in the legend: the drawing is narrower than it is printed (SCALE), and
        # two-line labels would cover the regions with their white boxes
        low = lvs[lvs < -0.6]                 # below v = 0.25 GeV, where no region reaches
        # its white box turns with the text, else the box of this long label on the steep line would cut the T_reh
        # line; above the T_reh line and below the regions
        lv_g6 = _place_inline(fig, ax, ax._g6, "GW template extrapolated", C_G6, lvs[(lvs > G6_LABEL_LV[0]) & (lvs < G6_LABEL_LV[1])], rotated=True, fontsize=6.0, pad=1.5, tint=C_G6)[0]
        if ax._treh.allsegs[0]:
            # the line runs almost along g: its label on the line, like the other bounds, at g = TREH_G
            seg = max(ax._treh.allsegs[0], key=len)
            ytr = float(np.interp(TREH_G, *seg[np.argsort(seg[:, 0])].T))
            ax.clabel(ax._treh, [ax._treh.levels[0]], inline=True, inline_spacing=5, fontsize=6.5, colors=[C_TREH],
                      fmt={ax._treh.levels[0]: "BBN reheating bound"}, manual=[(TREH_G, ytr)])
            _white_patch(fig, ax, ax._treh.labelTexts[-1], 1.5, rotated=True, zorder=3.2, tint=C_TREH)
            ax._treh.labelTexts[-1].set_zorder(3.3)

        for _, tag_, style_, _ in DAISY_EDGES:
            if edges[tag_] is not None:
                # the edge named by what it bounds; N5 runs against the P4 line, so its label sits beside it on the
                # trapped side, low down. Another edge is labelled on itself, first below v ~ 0.25 GeV, where no region
                # reaches, else higher up, in both cases away from the height of the G6 label, which the narrower
                # drawing (SCALE) brings close; below v ~ 20 GeV the Ellis-Lewicki-No line runs under the edge, and
                # the label interrupts both
                if tag_ in BESIDE_TAGS:
                    t_ = _place_beside(fig, ax, edges[tag_], f"Trapping @ {tag_}", style_["color"],
                                       lvs[(lvs > BESIDE_TAGS[tag_][0]) & (lvs < BESIDE_TAGS[tag_][1])], (-1.0, 1.0))
                    _white_patch(fig, ax, t_, TAG_PAD, rotated=True, zorder=3.2)
                    continue
                for cand in (low, lvs):
                    if _place_inline(fig, ax, edges[tag_], f"Trapping @ {tag_}", style_["color"],
                                     cand[np.abs(cand - lv_g6) > 0.3], hide=(ax._p4,), pad=TAG_PAD) is not None:
                        break
                else:
                    raise RuntimeError(f"no place for the tag {tag_} on its edge")
        _close_label_gaps(fig, ax)
        # the excluded side of each bound: vacuum trapping and the slow transitions at small g (upper left), late
        # reheating below the T_reh line
        for t_, cs_, col_ in [(t_, edges[t_], st_["color"]) for _, t_, st_, _ in DAISY_EDGES if edges[t_] is not None]:
            _side_arrows(fig, ax, cs_, col_, (-1.0, 1.0), fracs=EDGE_ARROW_FRACS.get(t_, SIDE_ARROW_FRACS))
        # P4 labelled on its line at log10 v = P4_LABEL_C, in two lines centred on the line, in a tinted box like the
        # other bounds
        seg = max(ax._p4.allsegs[0], key=len); o = np.argsort(seg[:, 1])
        xp = float(np.interp(P4_LABEL_C, seg[o, 1], seg[o, 0]))
        q0, q1 = ax.transData.transform([(np.interp(P4_LABEL_C - 0.05, seg[o, 1], seg[o, 0]), P4_LABEL_C - 0.05),
                                         (np.interp(P4_LABEL_C + 0.05, seg[o, 1], seg[o, 0]), P4_LABEL_C + 0.05)])
        ang = float(np.degrees(np.arctan2(*(q1 - q0)[::-1])))
        p4 = ax.text(xp, P4_LABEL_C, "False-vacuum volume\nstill grows at $T_{\\rm p}$", color=C_P4, fontsize=6.0, rotation=ang,
                     rotation_mode="anchor", ha="center", va="center", linespacing=1.1, zorder=3.3)
        _white_patch(fig, ax, p4, 1.5, rotated=True, zorder=3.2, tint=C_P4)
        # one arrow just before and one just after each label, along its line, towards the side on
        # which the criterion fails: the slow transitions and P4 to the upper left, late reheating downwards
        _arrows_around(fig, ax, ax._g6, ax._g6.labelTexts[-1], C_G6, (-1.0, 1.0), avoid=(ax._treh,), at_lv=G6_ARROW_LV)
        _arrows_around(fig, ax, ax._p4, p4, C_P4, (-1.0, 1.0), avoid=(ax._treh,), at_lv=P4_ARROW_LV)
        if ax._treh.allsegs[0]:
            _arrows_around(fig, ax, ax._treh, ax._treh.labelTexts[-1], C_TREH, (0.0, -1.0), along=0)
        out = "fig_plane.pdf"
    else:
        raise ValueError(f"fig_plane: unknown variant {which!r}")
    for a in axes:
        a.set_ylim(ylo, yhi)
        _g_axis(a)
    _log_v_axis(axes[0])
    save(fig, out)


# ---------------------------------------------------------------- appendix E
def fig_foreground():
    """The two well-posed answers with the binary foreground (appendix E): the NANOGrav
    1 sigma (thick) and 2 sigma (thin) regions of the profile-likelihood ratio, the foreground parameters maximised
    with their prior as a penalty and Delta ln L = 1.15 and 3.09 from the maximum (dashed, frequentist), and of the
    posterior with the foreground marginalised over its prior and flat priors in g and log10 v (solid,
    reproduce/binaries/marginal_foreground.py, Monte Carlo over the prior), for the (A, gamma) power law (left) and
    the binaryfog population under the survey-averaged priors (right), over the posterior region without a
    foreground (grey). The profiled likelihood with flat priors in (g, log10 v) answers neither question."""
    d = dict(_grid_npz(_need(os.path.join(DATA, SCAN_NPZ))))
    M = _grid_npz(_need(os.path.join(DATA, "marginal_foreground.npz")))
    G, LOGV, names = d["G"], d["LOGV"], list(d["names"])
    k = names.index("nlo")
    # the population under the survey-averaged priors, as the dashed contours of figure 4: the profile over the six
    # astrophysical parameters (data/ng15_fog_prof_recent.npz, the output of reproduce/binaries/prior_study.py for the
    # survey-averaged priors) and the marginal of data/ng15_fog_priors.npz
    FP = _grid_npz(_need(os.path.join(DATA, "ng15_fog_prof_recent.npz")))
    FM = _grid_npz(_need(os.path.join(DATA, "ng15_fog_priors.npz")))
    for z in (FP, FM):
        assert np.allclose(z["G"], G, atol=0) and np.allclose(z["LOGV"], LOGV, atol=0)
    panels = ((r"$(A_{\rm bhb}, \gamma_{\rm bhb})$ power law", d["NG15_bhb"][k], M["lnL_pl"], "lnL_bhb", "lnL_pl",
               TAIL_PATCH, TAIL_FOG),
              (r"$\mathtt{binaryfog}$ population", FP["lnL_prof"], FM["lnL_marg_recent"], "lnL_fog_prof", "lnL_holodeck",
               TAIL_FOG_PROF_GSA, TAIL_FOG_GSA))
    fig, axes = plt.subplots(1, 2, figsize=(width("fig_foreground"), 2.8), sharey=True, layout="constrained")
    for ax, (title, prof, marg, pkey, mkey, ppatch, mpatch) in zip(axes, panels):
        # the NANOGrav maps are smooth and drawn without smoothing (the IPTA smoothing would lower the peak by 0.7
        # and move the 95% edge); the strip at g ~ 0.41 from the maps on the tail-patch grid
        Gf, Lf, p0, lv0 = _patched(G, LOGV, d["NG15"][k], "lnL")
        ax.contourf(Gf, Lf, p0.T, levels=[lv0[0], lv0[1], np.inf], colors=["0.88", "0.72"])
        # frequentist: the profile-likelihood ratio, contours of exp(ln L - max) at exp(-3.09) and exp(-1.15)
        Gf, Lf, rf, lvr = _patched(G, LOGV, prof, pkey, ratio=True, patch=ppatch)
        ax.contour(Gf, Lf, rf.T, levels=lvr[:2], colors=[C_NG], linewidths=[0.8, 1.6], linestyles="--")
        Gf, Lf, pf, lv = _patched(G, LOGV, _binaries_box(marg, LOGV), mkey, patch=mpatch)
        ax.contour(Gf, Lf, pf.T, levels=[lv[0], lv[1]], colors=[C_NG], linewidths=[0.8, 1.6], linestyles="-")
        ax.set_title(title, fontsize=8)
        # the whole prior in g: the marginalised 2 sigma region with the population reaches g = 0.75; the profiled one
        # closes at g ~ 0.70 (the binaries alone lie 3.37 below the maximum under the survey-averaged priors, outside
        # the threshold 3.09); in v the whole prior range
        ax.set_ylim(*PRIOR_LOGV)   # the whole prior range in v, 10 MeV to 100 GeV
        _g_axis(ax, right=float(G[-1]))
        h = [(plt.Line2D([], [], color=C_NG, lw=1.6, ls="--"), "Profiled likelihood"),
             (plt.Line2D([], [], color=C_NG, lw=1.6), "Marginalised posterior"),
             ((Patch(facecolor="0.72"), Patch(facecolor="0.88")), "No binaries")]
        ax.legend([a for a, _ in h], [b for _, b in h], loc="upper right", fontsize=6, handlelength=2.2,
                  labelspacing=0.3, borderaxespad=0.6, **_tuple_legend())
    _log_v_axis(axes[0])
    save(fig, "fig_foreground.pdf")


# ---------------------------------------------------------------- the budget
LABELS = {
    # the row labels of figures 7 and 8, short enough that the rows fit the text width
    "mu4 x0.5": r"Matching scale $\mu_4/2$", "mu4 x2": r"Matching scale $2\mu_4$",
    "mu3 x0.5": r"EFT scale $\mu_3/2$", "mu3 x2": r"EFT scale $2\mu_3$",
    "dimension six": "Sextic operator",
    "prefactor x10": r"Dynamical prefactor $\times 10$", "prefactor /10": r"Dynamical prefactor $/10$",
    "higher order +": r"Next order, $-x_3|\delta S|$", "higher order -": r"Next order, $+x_3|\delta S|$",
    "leading order": "LO bounce, NLO matching",
    "full leading order": "LO instead of NLO",
    "4D daisy RG": r"$V_{\rm 4D}$ + Daisy at the thermal scale",
    "4D daisy": r"$V_{\rm 4D}$ + Daisy at the vacuum scale",
    "percolation at 1/e": r"Percolation at $P_{\rm t} = 1 - 1/\mathrm{e}$",
    "false-vacuum background": "False-vacuum expansion rate",
    "local completion criterion": r"$a^3(1-P_{\rm t})$ decreasing at $T_{\rm p}$",
    "T_nuc instead of T_p": r"$(\beta/H)_S$ at $T_{\rm nuc}$",
    "alpha from the trace anomaly": r"$\alpha$ from the trace anomaly",
    "beta/H from the action slope": r"$(\beta/H)_S$ instead of $(\beta/H)_R$ and $(\beta/H)_\Gamma$",
    "T_reh = T_p (1+alpha)^1/4": r"$T_{\rm reh} = T_{\rm p}(1+\alpha)^{1/4}$",
    "dark sector in the plasma at T_reh": r"Dark sector in the plasma at $T_{\rm reh}$",
    "dark sector with its own temperature": r"Dark sector with its own $T$",
    "no expansion, bubbles R^-3": "No Hubble expansion during transition",
    "expansion, bulk flow": "Free-streaming shells",
    "sharp causality tail": "Sharp causal tail",
    "no expansion, envelope": "Envelope approximation",
    "sound waves": "Higgsless template (LISA 2024)",
    "LISA update 2019": "Hybrid template (LISA 2019)",
    "sound waves, guo": "Source lifetime of Guo et al.",
    "binary foreground": r"$(A, \gamma)$ power-law binaries",
    "binaryfog": r"$\mathtt{binaryfog}$, NANOGrav priors",
    "binaryfog, Tomczak-Kormendy": r"$\mathtt{binaryfog}$, Tomczak–Kormendy",
    "binaryfog, survey-averaged": r"$\mathtt{binaryfog}$, galaxy survey-averaged",
    "2024 release": "Pre-erratum likelihood",
    "expansion factors at the separation speed": r"Expansion factors at $(\beta/H)_R$",
    "expansion factors at beta/H_V": r"Expansion factors at $(H/H_{\rm V})(\beta/H)_\Gamma$",
    "template within its simulated range": r"Template valid for $(H/H_{\rm V})(\beta/H)_\Gamma > 4$",
    "expansion factors at the matched exponential rate": r"Expansion factors at matched ${\rm e}^{\beta t}$",
    "instantaneous reheating": r"Instantaneous reheating",
    "HD instead of CURN": "Hellings–Downs spectrum",
    "DMGP instead of DMX": "DMGP noise model",
    "bins: 5": "5 frequency bins", "bins: 30": "30 frequency bins",
    "bins: 30 (IPTA)": "30 frequency bins",
}
SYMBOL = {
    # rows of the budget, numbered in the order in which sections 2 and 3 introduce them; variants
    # computed but not shown (the LO bounce on the NLO-matched theory, the lattice-size shifts, both
    # template conventions together) are quoted in the text only. Within a symbol the rows follow
    # the order of this dict, which is that of the table of switches.
    "mu4 x0.5": "N1", "mu4 x2": "N1", "mu3 x0.5": "N1", "mu3 x2": "N1",
    "prefactor x10": "N2", "prefactor /10": "N2",
    "full leading order": "N3", "4D daisy RG": "N4", "4D daisy": "N5",
    # P1, the expansion history during the transition; the secluded dark sector (a propto 1/T) is computed but not
    # shown
    "false-vacuum background": "P1",
    "percolation at 1/e": "P2", "local completion criterion": "P3",
    "beta/H from the action slope": "P4", "expansion factors at the separation speed": "P4",
    "expansion factors at beta/H_V": "P4",
    "T_reh = T_p (1+alpha)^1/4": "P5",
    "no expansion, bubbles R^-3": "G1", "expansion, bulk flow": "G2",
    "sound waves": "G3", "LISA update 2019": "G4", "no expansion, envelope": "G5",
    "sound waves, guo": "G6", "template within its simulated range": "G7",
    # B1 the binary foreground in both descriptions, B2 to B4 the pulsar data, B5 the other data set
    # within B1 the dict order sets the row order: power law, survey-averaged, NANOGrav, Tomczak-Kormendy priors
    "binary foreground": "B1", "binaryfog, survey-averaged": "B1", "binaryfog": "B1",
    "binaryfog, Tomczak-Kormendy": "B1", "2024 release": "B6", "HD instead of CURN": "B2",
    "DMGP instead of DMX": "B3", "bins: 5": "B4", "bins: 30": "B4", "bins: 30 (IPTA)": "B4",
    "IPTA DR2": "B5",
}
# rows measured against another row instead of the defaults, so that they isolate one choice
REFERENCE_ROW = {"sound waves, guo": "sound waves"}
# every row measured against something other than the defaults: the one above. Every data switch, the DMGP noise
# model included, and the largest other fit (G5) are measured against the default
NOT_AGAINST_DEFAULT = set(REFERENCE_ROW)


def _symkey(name):
    s = SYMBOL.get(name, "Z99")
    order = list(SYMBOL).index(name) if name in SYMBOL else 0
    return ("NPGBZ".index(s[0]), int(s[1:]), order)


def _moments(p, G, LOGV):
    """Posterior mean and covariance on the grid (flat prior in g and log10 v)."""
    Gm, Lm = np.meshgrid(G, LOGV, indexing="ij")
    w = p / p.sum()
    m = np.array([np.sum(w * Gm), np.sum(w * Lm)])
    dx = np.stack([Gm - m[0], Lm - m[1]])
    C = np.einsum("iab,jab,ab->ij", dx, dx, w)
    return m, C


OTHER_FITS = {}      # per data set: D and the intervals of each of the six other fits without expansion


#: a posterior is cut by the scan on a side where its marginal density in the outermost cell exceeds this fraction of
#: its peak: the ridge of high likelihood then continues beyond the grid. Within the prior range of v (PRIOR_LOGV)
#: only the IPTA posterior of the hybrid template is cut, at its lower end of v: the budget asserts it (fig_budget,
#: table_switch_numbers). The plateau of the binary rows
#: (B1, B2), a few per cent of the peak at every edge, is the limit in which the binaries alone explain the signal
EDGE_DENSITY = 0.1


def _cut_sides(p):
    """The sides of the scan (g_lo, g_hi, v_lo, v_hi) at which the posterior p is cut (EDGE_DENSITY)."""
    mg, mv = p.sum(axis=1), p.sum(axis=0)
    return dict(g_lo=mg[0] > EDGE_DENSITY * mg.max(), g_hi=mg[-1] > EDGE_DENSITY * mg.max(),
                v_lo=mv[0] > EDGE_DENSITY * mv.max(), v_hi=mv[-1] > EDGE_DENSITY * mv.max())


def _sgn(x):
    """A signed integer per cent without a negative zero."""
    return f"{int(round(x)):+d}"


def _budget_scan():
    """The scan as the budget uses it: for NANOGrav the two binary foregrounds (B1, B2) marginalised over
    their priors (reproduce/binaries/marginal_foreground.py) instead of profiled. A profile of the population over its
    six parameters lets every point take the best-fitting population whatever its prior probability, and with
    flat priors on (g, v) the plateau of such points at large g takes the posterior mass: D = 12.5 profiled
    against 1.5 marginalised, which an MCMC of the full eight-parameter posterior confirms.
    The plane (figure 5) keeps the profiled contours."""
    d = dict(_grid_npz(_need(os.path.join(DATA, SCAN_NPZ))))
    mf = _grid_npz(_need(os.path.join(DATA, "marginal_foreground.npz")))
    assert np.allclose(mf["G"], d["G"]) and np.allclose(mf["LOGV"], d["LOGV"])
    # every map with the binaries marginalised on the range BINARIES_LOGV
    d["NG15_bhb"] = _binaries_box(mf["lnL_pl"], d["LOGV"])
    d["NG15_binaryfog"] = _binaries_box(mf["lnL_holodeck"], d["LOGV"])
    # the population under the other two prior sets of table 2 (reproduce/binaries/prior_study.py)
    # (data/ng15_fog_priors.npz: the lnL_marg maps of prior_study.py on the grid of the scan)
    P = _grid_npz(_need(os.path.join(DATA, "ng15_fog_priors.npz")))
    assert np.allclose(P["G"], d["G"], atol=0) and np.allclose(P["LOGV"], d["LOGV"], atol=0)
    for key, prior in (("tk", "literature"), ("gsa", "recent")):
        d["NG15_binaryfog_" + key] = _binaries_box(P["lnL_marg_" + prior], d["LOGV"])
    # IPTA DR2 with the population on its own frequencies (reproduce/binaries/ipta_fog.py), measured against IPTA through
    # binaryfog without foreground, so that both maps share one evaluation of the data (near the best fit it differs
    # from the PTArcade IPTA2 map by a constant)
    fi = _need(os.path.join(DATA, "ipta_fog.npz"))
    I = _grid_npz(fi)
    assert np.allclose(I["G"], d["G"]) and np.allclose(I["LOGV"], d["LOGV"])
    d["IPTA2_binaryfog"], d["IPTA2_binaryfog_ref"] = _binaries_box(I["lnL_holodeck"], d["LOGV"]), I["lnL"]
    for key, prior in (("tk", "literature"), ("gsa", "recent")):      # reproduce/binaries/ipta_fog_priors.py
        fp = _need(os.path.join(DATA, f"ipta_fog_{prior}.npz"))
        P = _grid_npz(fp)
        assert np.allclose(P["G"], d["G"], atol=0) and np.allclose(P["LOGV"], d["LOGV"], atol=0)
        d["IPTA2_binaryfog_" + key], d["IPTA2_binaryfog_" + key + "_ref"] = _binaries_box(P["lnL_holodeck"], d["LOGV"]), I["lnL"]
    if "lnL_pl" in I.files:              # the power law marginalised over the IPTA prior, as B1 for NANOGrav
        d["IPTA2_bhb"], d["IPTA2_bhb_ref"] = _binaries_box(I["lnL_pl"], d["LOGV"]), I["lnL"]
    return d


def budget_rows(dataset="NG15"):
    """Every switch as [group, name, (shift_g, widen_g), (shift_v, widen_v), note, dist, area, edge, sign]:
    along each axis, the movement of the centre of the 68% interval and the change of its
    full width, in per cent of the default interval's full width; dist, the distance of the
    posterior means in units of the default posterior's standard deviation along its
    principal axes (Mahalanobis); area, the ratio of the 68% areas; edge, whether the 68%
    region reaches the boundary of the grid (the bar is then a lower limit); sign, the direction of
    the shift along g and log10 v (+1 towards larger values)."""
    d = _budget_scan()
    G, LOGV, names = d["G"], d["LOGV"], list(d["names"])
    k0 = names.index("nlo")
    lnl = d[dataset]
    lv0, p0 = _levels(lnl[k0])
    m0, C0 = _moments(p0, G, LOGV)
    Ci = np.linalg.inv(C0)

    def intervals(p):
        return {axis: _interval(p, vals, axis) for axis, vals in ((0, G), (1, LOGV))}
    ref0 = intervals(p0)
    width = {axis: ref0[axis][1] - ref0[axis][0] for axis in (0, 1)}

    def measure(arr, ref_arr=None):
        if not np.isfinite(arr).any():
            return None
        lv, p = _levels(arr)
        iv = intervals(p)
        pr, ivr = (p0, ref0)
        if ref_arr is not None:
            _, pr = _levels(ref_arr); ivr = intervals(pr)
        out, sign = [], []
        for axis in (0, 1):
            c, h = 0.5 * (iv[axis][0] + iv[axis][1]), 0.5 * (iv[axis][1] - iv[axis][0])
            cr, hr = 0.5 * (ivr[axis][0] + ivr[axis][1]), 0.5 * (ivr[axis][1] - ivr[axis][0])
            # the shift of the centre (unsigned; its direction in `sign`) and the change of the full
            # width with its sign: positive for a wider, negative for a narrower region, both in per
            # cent of the default width
            out.append((100 * abs(c - cr) / width[axis], 100 * 2 * (h - hr) / width[axis]))
            sign.append(1 if c >= cr else -1)
        m, _ = _moments(p, G, LOGV)
        mr, _ = _moments(pr, G, LOGV)
        dm = m - mr
        dist = float(np.sqrt(dm @ Ci @ dm))
        lvr = lv0 if ref_arr is None else _levels(ref_arr)[0]
        in68 = p >= lv[1]
        area = float(np.sum(in68)) / float(np.sum(pr >= lvr[1]))
        # a region is cut by the scan when one of its marginal intervals (68% or 95%, of g or of log10 v)
        # ends at the edge of the grid; the shifts and D are then lower bounds. A 68% region that merely touches
        # the edge, with a few per mille of the posterior in the outermost cells, is not cut.
        def _at_edge(lo, hi, vals):
            step = vals[1] - vals[0]
            return lo - vals[0] < 0.6 * step or vals[-1] - hi < 0.6 * step
        # also cut where the posterior density at an edge is high although no interval ends there (EDGE_DENSITY)
        cut = _cut_sides(p)
        edge = any(_at_edge(*_interval(p, G, 0, lev), G) or _at_edge(*_interval(p, LOGV, 1, lev), LOGV)
                   for lev in (0.68, 0.95)) or any(cut.values())
        # the marginal 68% and 95% intervals themselves (table 5), each end flagged where it lies at the edge; the
        # 95% interval also on a side at which the posterior is cut
        ivs = {}
        for key, vals, axis in (("g", G, 0), ("v", LOGV, 1)):
            step = vals[1] - vals[0]
            for lab, lev in (("1", 0.68), ("2", 0.95)):
                lo, hi = _interval(p, vals, axis, lev)
                c_lo, c_hi = (cut[key + "_lo"], cut[key + "_hi"]) if lab == "2" else (False, False)
                ivs[key + lab] = (lo, hi, lo - vals[0] < 0.6 * step or c_lo, vals[-1] - hi < 0.6 * step or c_hi)
        return out, dist, area, edge, tuple(sign), ivs

    rows = []
    for k, name in enumerate(names):
        if name == "nlo" or name not in SYMBOL:
            continue
        ref = lnl[names.index(REFERENCE_ROW[name])] if name in REFERENCE_ROW else None
        r = measure(lnl[k], ref)
        if r is None:
            continue
        note = None
        if name in ("4D daisy", "4D daisy RG"):
            _, p4 = _levels(lnl[k]); lo4, hi4 = _interval(p4, G, 0)
            note = rf"(at $g = {lo4:.2f}$ to ${hi4:.2f}$)"
        rows.append([SYMBOL[name][0], name, r[0][0], r[0][1], note, r[1], r[2], r[3], r[4], r[5]])
    # B1 in the order power law, survey-averaged, NANOGrav, Tomczak-Kormendy priors
    for key, name in ((dataset + "_bhb", "binary foreground"),
                      (dataset + "_binaryfog_gsa", "binaryfog, survey-averaged"),
                      (dataset + "_binaryfog", "binaryfog"),
                      (dataset + "_binaryfog_tk", "binaryfog, Tomczak-Kormendy"),
                      (dataset + "_hd", "HD instead of CURN"), (dataset + "_dmgp", "DMGP instead of DMX"),
                      (dataset + "_v1", "2024 release"), (dataset + "_bins5", "bins: 5"),
                      (dataset + "_bins30", "bins: 30" if dataset == "NG15" else "bins: 30 (IPTA)")):
        if key not in d:
            continue
        arr = d[key] if d[key].ndim == 2 else d[key][k0]
        ref = None
        if name.startswith("binaryfog") and key + "_ref" in d:
            ref = d[key + "_ref"]
        if name == "binary foreground" and dataset + "_bhb_ref" in d:
            ref = d[dataset + "_bhb_ref"]
        r = measure(arr, ref)
        if r is None:
            continue
        note = None
        rows.append(["B", name, r[0][0], r[0][1], note, r[1], r[2], r[3], r[4], r[5]])
    # B5: the default analysis on the IPTA DR2 data, measured against the NANOGrav default
    if dataset == "NG15" and "IPTA2" in d:
        r = measure(d["IPTA2"][k0])
        if r is not None:
            rows.append(["B", "IPTA DR2", r[0][0], r[0][1], None, r[1], r[2], r[3], r[4], r[5]])
    # the six other fits without expansion of the reference of G1, each against the default, for the text of G5;
    # the row G5 itself is the envelope approximation
    others = [n for n in names if n.startswith("no expansion") and n != "no expansion, bubbles R^-3"]
    OTHER_FITS[dataset] = {}
    for n in others:
        m_ = measure(lnl[names.index(n)])
        if m_ is not None:
            OTHER_FITS[dataset][n] = dict(D=m_[1], g1=m_[5]["g1"][:2], g2=m_[5]["g2"][:2],
                                          v1=tuple(10 ** x for x in m_[5]["v1"][:2]),
                                          v2=tuple(10 ** x for x in m_[5]["v2"][:2]),
                                          dlnLmax=float(np.nanmax(lnl[names.index(n)]) - np.nanmax(lnl[k0])))
    rows.sort(key=lambda r: _symkey(r[1]))
    return rows, dict(width_g=width[0], width_logv=width[1], dg=float(G[1] - G[0]), dlogv=float(LOGV[1] - LOGV[0]),
                      default=measure(lnl[k0])[5])


_BUDGET = {}


def _budget(dataset):
    if dataset not in _BUDGET:
        _BUDGET[dataset] = budget_rows(dataset)
    return _BUDGET[dataset]


def _tint(c, f=0.42):
    """An opaque light shade of a colour: f of the colour on white."""
    return tuple(f * np.asarray(c[:3]) + (1 - f))


# Pairs of rows that the budget figure and the table show as one row, the direction with the larger
# distance D, per data set.
FIGURE_PAIRS = {
    ("mu4 x0.5", "mu4 x2"): r"$\mu_4/2$ or $2\mu_4$",
    ("mu3 x0.5", "mu3 x2"): r"$\mu_3/2$ or $2\mu_3$",
    ("prefactor x10", "prefactor /10"): r"$A_{\rm dyn}\times10$ or $/10$",
}
# the row labels of figure 5, short so that the panels keep their width; the full names are in
# table 4 and the text
FIG_LABELS = {
    "full leading order": "LO instead of NLO", "4D daisy RG": r"$V_{\rm 4D}$ + Daisy at the thermal scale",
    "4D daisy": r"$V_{\rm 4D}$ + Daisy at the vacuum scale",
    "false-vacuum background": r"$H^2 \approx \rho_{\rm f}/3M_{\rm Pl}^2$",
    "dark sector with its own temperature": "Secluded dark sector",
    "percolation at 1/e": r"$f_{\rm perc} = 0.63$ instead of $0.29$",
    "local completion criterion": r"$a^3(1-P_{\rm t})$ decreases at percolation",
    "beta/H from the action slope": r"$(\beta/H)_S$ instead of $(\beta/H)_R$ and $(\beta/H)_\Gamma$",
    "T_reh = T_p (1+alpha)^1/4": "Instantaneous reheating",
    "no expansion, bubbles R^-3": "No expansion between bubbles", "expansion, bulk flow": "Free-streaming shells",
    "no expansion, envelope": "Envelope approximation",
    "sound waves": "Higgsless template (LISA 2024)", "LISA update 2019": "Hybrid template (LISA 2019)",
    "binary foreground": r"$(A, \gamma)$ power-law binaries", "binaryfog": "NANOGrav population prior",
    "binaryfog, Tomczak-Kormendy": "Tomczak\u2013Kormendy population prior", "binaryfog, survey-averaged": "Survey-avg. population prior",
    "HD instead of CURN": "Hellings–Downs instead of CURN", "DMGP instead of DMX": "DMGP instead of DMX",
    "bins: 5": "5 bins", "bins: 30": "30 bins", "bins: 30 (IPTA)": "30 bins",
    "IPTA DR2": r"IPTA DR2 instead of NANOGrav 15$\,$yr",
}


PAIR_SHORT = {
    ("mu4 x0.5", "mu4 x2"): r"$\mu_4/2$ or $2\mu_4$", ("mu3 x0.5", "mu3 x2"): r"$\mu_3/2$ or $2\mu_3$",
    ("prefactor x10", "prefactor /10"): r"$\Adyn\times10$ or $/10$",
}


#: switches computed but not shown; the NNLO estimate is quoted in a footnote
DROPPED_ROWS = {"sound waves, guo", "template within its simulated range", "2024 release",
                "expansion factors at the separation speed", "expansion factors at beta/H_V",
                "higher order +", "higher order -"}


def _figure_rows(rows):
    """The rows of the budget figure and table: each pair of FIGURE_PAIRS as one row. Returns the
    rows, the figure label of each kept member and the pair (its first name) each kept member stands for."""
    by_name = {r[1]: r for r in rows}
    labels, pair_of = {}, {}
    move = lambda r: r[5]          # the direction with the larger distance D of the posterior means
    for (a, b), lab in FIGURE_PAIRS.items():
        if a in by_name and b in by_name:
            k = max((by_name[a], by_name[b]), key=move)[1]
            labels[k], pair_of[k] = lab, (a, b)
    drop = {n for pair in FIGURE_PAIRS for n in pair} - set(labels)
    drop |= DROPPED_ROWS
    return [r for r in rows if r[1] not in drop], labels, pair_of


C_SRC = DESYmagenta
# the four steps of figure 1, each with its own colour: the blocks of figures 6 and 7
BLOCKS = [("N", "Nucleation rate", C_PP), ("P", "Percolation and reheating", C_COSMO),
          ("G", "Gravitational wave source", C_SRC), ("B", "Binaries and pulsar data", C_DATA)]
_SPECIAL = {"binary foreground": "_bhb", "binaryfog": "_binaryfog", "binaryfog, Tomczak-Kormendy": "_binaryfog_tk",
            "binaryfog, survey-averaged": "_binaryfog_gsa", "HD instead of CURN": "_hd",
            "DMGP instead of DMX": "_dmgp", "2024 release": "_v1", "bins: 5": "_bins5", "bins: 30": "_bins30",
            "bins: 30 (IPTA)": "_bins30"}


def _block(sym):
    """The step of a switch: N, P, G or B."""
    return sym[0]


def _row_array(d, dataset, name):
    """The likelihood map of one row of the budget."""
    names, k0 = list(d["names"]), list(d["names"]).index("nlo")
    if name == "IPTA DR2":
        return d["IPTA2"][k0]
    if name in _SPECIAL:
        a = d[dataset + _SPECIAL[name]]
        return a if a.ndim == 2 else a[k0]
    return d[dataset][names.index(name)]


def _row_reference(d, dataset, name):
    """The map a row is compared with when that is not the default (table 3), else None."""
    names = list(d["names"])
    if name in REFERENCE_ROW:
        return _row_array(d, dataset, REFERENCE_ROW[name])
    return None


def _intervals(arr, G, LOGV):
    """The 68% and 95% intervals of the marginal posteriors of g and log10 v, and whether the 68% and
    95% regions reach each edge of the scan (the interval then continues beyond it)."""
    lv, p = _levels(arr)
    out = {}
    for key, lev, L in (("1", 0.68, lv[1]), ("2", 0.95, lv[0])):
        out["g" + key] = _interval(p, G, 0, lev)
        out["v" + key] = _interval(p, LOGV, 1, lev)
        inr = p >= L
        out["edge" + key] = dict(g_lo=bool(inr[0, :].any()), g_hi=bool(inr[-1, :].any()),
                                 v_lo=bool(inr[:, 0].any()), v_hi=bool(inr[:, -1].any()))
    return out


def fig_budget(dataset="NG15"):
    """The preferred region under each alternative, shown directly: for every row of table 3 the
    marginal 68% (dark) and 95% (light) intervals of g and of v against those of the defaults (grey
    band, dotted lines), and the distance D of the posterior means. Rows grouped by the four steps of
    figure 1, each in its own colour, with the name of the step written vertically on the left. No
    interval is cut by the prior range except the lower end of v of the IPTA hybrid template, whose bar ends
    in an arrow (asserted)."""
    rows, info = _budget(dataset)
    rows, pair_labels, _ = _figure_rows(rows)
    d = _budget_scan()
    G, LOGV, names = d["G"], d["LOGV"], list(d["names"])
    base = _intervals(d[dataset][names.index("nlo")], G, LOGV)
    colour = dict((c, col) for c, _, col in BLOCKS)
    GAP = 0.75                                  # extra vertical space between the steps
    entries, groups, yy = [("default", None, 0.0)], [], 0.0
    for code, title, col in BLOCKS:
        members = [r for r in rows if _block(SYMBOL[r[1]]) == code]
        if not members:
            continue
        yy -= GAP
        first = yy - 1.0
        for r in members:
            yy -= 1.0
            entries.append(("row", r, yy))
        groups.append((title, col, first, yy))
    y_all = [e[2] for e in entries]
    n_eff = -min(y_all) + 1
    stem = "fig_budget" if dataset == "NG15" else "fig_budget_ipta"
    fig = plt.figure(figsize=(width(stem), 0.158 * n_eff + 0.55), layout="constrained")
    gs = fig.add_gridspec(1, 4, width_ratios=[0.12, 1.0, 1.0, 0.42])
    axs = fig.add_subplot(gs[0]); axg = fig.add_subplot(gs[1], sharey=axs)
    axv = fig.add_subplot(gs[2], sharey=axs); axd = fig.add_subplot(gs[3], sharey=axs)
    # the axes span the 2 sigma bars of all rows with a margin, not the prior range, which reaches far beyond most
    # regions. One row is cut by the prior range, at its lower end of v: the hybrid template for IPTA, whose posterior
    # continues below 10 MeV, where T_reh lies below the BBN bound; its bar ends in an arrow
    cut_rows = [r[1] for r in rows if r[7]]
    allowed = {"IPTA2": {"LISA update 2019"}}.get(dataset, set())
    assert set(cut_rows) <= allowed, f"posteriors cut by the prior range: {sorted(set(cut_rows) - allowed)}"
    ivs_all = [base] + [_intervals(_row_array(d, dataset, r[1]), G, LOGV) for r in rows]
    lim = {k: (min(iv[k + "2"][0] for iv in ivs_all), max(iv[k + "2"][1] for iv in ivs_all)) for k in ("g", "v")}
    lim = {"g": (G[0] - 0.02, G[-1] + 0.02), "v": (lim["v"][0] - 0.25, lim["v"][1] + 0.25)}
    H = 0.5                                      # the same height for the 68% and the 95% bars
    ylo, yhi = min(y_all) - 0.6, 0.6
    # the posterior mean of each row, the centre that D is built from, marked on its intervals (not the maximum
    # likelihood, which can sit in the narrow strip at g ~ 0.41 outside the 68% intervals)
    def post_mean(arr):
        _, p_ = _levels(arr)
        w = p_ / p_.sum()
        return {"g": float((w * G[:, None]).sum()), "v": float((w * LOGV[None, :]).sum())}

    for ax, key in ((axg, "g"), (axv, "v")):
        span = lim[key][1] - lim[key][0]
        lo1, hi1 = base[key + "1"]; lo2, hi2 = base[key + "2"]
        ax.fill_betweenx([ylo, yhi], lo1, hi1, color="0.91", lw=0, zorder=0)
        for x_ in (lo2, hi2):
            ax.plot([x_, x_], [ylo, yhi], color="0.6", lw=0.6, ls=(0, (1, 1.5)), zorder=0)
        for kind, r, yy in entries:
            if kind == "default":
                iv, cdark, clight = base, "0.25", "0.72"
            else:
                iv = _intervals(_row_array(d, dataset, r[1]), G, LOGV)
                cdark = colour[_block(SYMBOL[r[1]])]
                clight = _tint(cdark, 0.38)
            bf_ = post_mean(d[dataset][names.index("nlo")] if kind == "default" else _row_array(d, dataset, r[1]))
            if bf_ is not None:
                ax.plot(bf_[key], yy, marker="D", ms=2.6, mfc="white", mec="0.15", mew=0.6, zorder=5, ls="none")
            # the best fit of each row on the grid
            arr_ = d[dataset][names.index("nlo")] if kind == "default" else _row_array(d, dataset, r[1])
            if np.isfinite(arr_).any():
                ib, jb = np.unravel_index(np.nanargmax(arr_), arr_.shape)
                ax.plot({"g": G[ib], "v": LOGV[jb]}[key], yy, marker="|", ms=5.0, mec="0.1", mew=1.0, zorder=5,
                        ls="none")
            for lvl, col, z in (("2", clight, 2), ("1", cdark, 3)):
                lo, hi = iv[key + lvl]
                ax.add_patch(Rectangle((lo, yy - H / 2), hi - lo, H, facecolor=col, lw=0, zorder=z))
            if kind != "default" and r[1] in cut_rows and key == "v":
                ax.plot(iv["v2"][0] - 0.06, yy, marker="<", ms=3.0, mfc=cdark, mec="none", ls="none", zorder=4)
        ax.set_xlim(*lim[key])
        ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
        ax.tick_params(axis="x", which="both", top=False)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    axg.set_xlabel(r"$g$")
    axv.set_xlabel(r"$v$  [GeV]")
    # major ticks at the decades inside the axis, minor ticks at 2 to 9 times each decade over the whole axis
    decs = [k for k in range(int(np.floor(lim["v"][0])), int(np.ceil(lim["v"][1])) + 1) if lim["v"][0] <= k <= lim["v"][1]]
    # every second decade labelled, as powers of ten, once the axis spans more than four decades
    every = 2 if len(decs) > 4 else 1
    axv.set_xticks(decs)
    axv.set_xticklabels([(f"$10^{{{k}}}$" if every == 2 else f"${10.0 ** k:g}$") if k % every == 0 else "" for k in decs])
    axv.set_xticks([k + np.log10(m) for k in range(decs[0] - 1, decs[-1] + 1) for m in range(2, 10)
                    if lim["v"][0] <= k + np.log10(m) <= lim["v"][1]], minor=True)
    # the distance of the posterior means; beyond the axis the value is written out
    DMAX = float(np.ceil(2 * max(r[5] for kind, r, _ in entries if kind == "row") + 1.2)) / 2   # just room for the largest D
    # guides at every integer D, below the key; interrupted at the rows whose D lies beyond the
    # axis, where the value is written out over them
    gaps = sorted(yy for kind, r, yy in entries if kind == "row" and r[5] > DMAX - 0.35)
    for x_ in range(1 if DMAX < 10 else 2, int(DMAX) + 1, 1 if DMAX < 10 else 2):     # every second D on a wide axis
        y0 = ylo
        for yg in gaps:
            axd.plot([x_, x_], [y0, yg - 0.5], color="0.8", lw=0.6, ls=(0, (1, 1.5)), zorder=0)
            y0 = yg + 0.5
        axd.plot([x_, x_], [y0, -0.6], color="0.8", lw=0.6, ls=(0, (1, 1.5)), zorder=0)
    for kind, r, yy in entries:
        if kind != "row":
            continue
        col = colour[_block(SYMBOL[r[1]])]
        x_ = min(r[5], DMAX - 0.35)
        if r[5] > DMAX - 0.35:
            axd.plot([0, x_ - 3.4], [yy, yy], color="0.75", lw=0.8, zorder=1)
            axd.text(x_ - 0.8, yy, f"{r[5]:.1f}", fontsize=6, ha="right", va="center", color="0.25")
        else:
            axd.plot([0, x_], [yy, yy], color="0.75", lw=0.8, zorder=1)
        axd.plot(x_, yy, "o", ms=3.6, mfc=col, mec=col, mew=0.9, zorder=3)
    axd.set_xlim(0, DMAX)
    axd.set_xticks(list(range(0, int(DMAX) + 1, 2 if DMAX < 10 else 4)))
    axd.xaxis.set_minor_locator(mpl.ticker.NullLocator())         # no minor ticks on the D axis
    axd.set_xlabel(r"$\mathcal{D}$")
    axd.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
    axd.tick_params(axis="x", which="both", top=False)
    for s_ in ("top", "right"):
        axd.spines[s_].set_visible(False)
    # the steps, written vertically next to their rows, each with a bar in its colour
    axs.set_xlim(0, 1)
    axs.axis("off")
    for title, col, y_top, y_bot in groups:
        axs.plot([0.97, 0.97], [y_bot - 0.4, y_top + 0.4], color=col, lw=2.0, solid_capstyle="butt")
        # the longest name that fits next to the rows of the step (the binaries and data step: "PTA data,
        # binaries" for NANOGrav, "Data" for IPTA, which has fewer rows)
        names = {"Percolation and reheating": ["Reheating"],
                 "Gravitational wave source": ["Gravitational wave source", "GW source"],
                 "Binaries and pulsar data": ["PTA data, binaries", "Data"]}.get(title, [title])
        from matplotlib.textpath import TextPath
        from matplotlib.font_manager import FontProperties
        room = (y_top - y_bot + 0.8) * 0.158 * 72           # the length of the step's bar, in points
        fits = lambda n: TextPath((0, 0), n, prop=FontProperties(size=7, weight="bold")).get_extents().width < room
        txt = next((n for n in names if fits(n)), names[-1])
        axs.text(0.25, 0.5 * (y_top + y_bot), txt, rotation=90, fontsize=7, fontweight="bold", color=col,
                 ha="center", va="center")
    # the row labels, on the right edge of the column of the steps
    axg.set_yticks([e[2] for e in entries])
    axg.set_yticklabels(["This work's default" if k == "default" else f"{SYMBOL[r[1]]}  "
                         + pair_labels.get(r[1], FIG_LABELS.get(r[1], LABELS.get(r[1], r[1]))) for k, r, _ in entries], fontsize=7)
    axg.tick_params(axis="y", which="both", labelleft=True, left=False)
    axg.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    axs.set_ylim(ylo, yhi)
    save(fig, stem + ".pdf")
    return rows


# ---------------------------------------------------------------- the table of the budget
SHORT = {
    "mu4 x0.5": r"$\mu_4/2$", "mu4 x2": r"$2\mu_4$", "mu3 x0.5": r"$\mu_3/2$", "mu3 x2": r"$2\mu_3$",
    "dimension six": "sextic operator", "prefactor x10": r"$\Adyn\times10$", "prefactor /10": r"$\Adyn/10$",
    "higher order +": r"$-x_3|\delta S|$", "higher order -": r"$+x_3|\delta S|$",
    "full leading order": "LO",
    "4D daisy RG": r"$V_{\rm 4D}$ + Daisy at the thermal scale", "4D daisy": r"$V_{\rm 4D}$ + Daisy at the vacuum scale",
    "percolation at 1/e": r"$P_{\rm t} = 1 - 1/\eu$", "local completion criterion": r"$a^3(1-P_{\rm t})$ decreasing at $\Tp$",
    "T_nuc instead of T_p": r"$T_{\rm nuc}$", "beta/H from the action slope": r"$\bHS$",
    "T_reh = T_p (1+alpha)^1/4": r"$\Tp(1+\alpha)^{1/4}$", "dark sector with its own temperature": r"$a \propto 1/T$",
    "false-vacuum background": r"false-vacuum $H$",
    "no expansion, bubbles R^-3": "no Hubble expansion", "expansion, bulk flow": "free shells", "sharp causality tail": "sharp tail",
    "no expansion, envelope": "envelope", "sound waves": "Higgsless (LISA 2024)", "LISA update 2019": "hybrid (LISA 2019)",
    "sound waves, guo": "Guo et al.", "binary foreground": r"$(A,\gamma)$", "binaryfog": r"\texttt{binaryfog}, NANOGrav",
    "binaryfog, Tomczak-Kormendy": r"\texttt{binaryfog}, Tomczak--Kormendy",
    "binaryfog, survey-averaged": r"\texttt{binaryfog}, survey-averaged",
    "2024 release": "pre-erratum", "HD instead of CURN": "Hellings--Downs", "DMGP instead of DMX": "DMGP",
    "expansion factors at the separation speed": r"at $\bHR$",
    "expansion factors at beta/H_V": r"at $(H/\HV)\bHG$",
    "template within its simulated range": r"$(H/\HV)\bHG > 4$ only",
    "expansion factors at the matched exponential rate": r"matched $\eu^{\beta t}$",
    "instantaneous reheating": "instantaneous",
    "bins: 5": "5 bins", "bins: 30": "30 bins", "bins: 30 (IPTA)": "30 bins",
    "IPTA DR2": r"IPTA DR2 instead of NANOGrav 15$\,$yr",
}


# The layout of switch_numbers.tex (the intervals themselves instead of shifts in per cent): one table per data set,
# nine columns.
D_FMT = ".1f"


# the prior sets of the population rows in data/bhb_population_dlnl.json
POPULATION_PRIOR = {"binaryfog": "nanograv", "binaryfog, survey-averaged": "recent", "binaryfog, Tomczak-Kormendy": "literature"}


def table_switch_numbers():
    """The numbers behind the budget figure as a LaTeX table, ../switch_numbers.tex (NANOGrav): per switch the marginal 68% and 95% intervals of g and of v (in GeV),
    none of them cut by the prior range except the lower end of v of the IPTA hybrid template (asserted); the distance D of the posterior means; the
    ratio of the 68% areas; and the change of the maximum ln L (where the likelihood is the same or, for the
    binaries, extends it: ln L without the prior term at the maximum of ln L + ln pi; not for the data switches B2 to B5). A dagger marks the rows measured against another row
    instead of the defaults (NOT_AGAINST_DEFAULT)."""
    d = dict(_grid_npz(_need(os.path.join(DATA, SCAN_NPZ))))
    names = list(d["names"])
    res = {}
    # the maxima of ln L from maximising over the plane (reproduce/plane/best_fits.py), not the grid
    bf = json.load(open(_need(os.path.join(DATA, "best_fits.json"))))
    lmax = lambda name, ds: (bf.get(name, {}).get(ds) or {}).get("max", np.nan)
    default = {}
    for ds in ("NG15", "IPTA2"):
        rows, info = _budget(ds)
        default[ds] = info["default"]
        rows, _, pair_of = _figure_rows(rows)
        lnl0 = lmax("nlo", ds)
        for r in rows:
            name = r[1]
            if name in names:
                dl = lmax(name, ds) - lnl0
            elif name == "binary foreground":
                # ln L alone, without the prior term, at the maximum of ln L + ln pi (reproduce/binaries/bhb_lnl_without_prior.py)
                dl = json.load(open(_need(os.path.join(DATA, "bhb_lnl_without_prior.json"))))[ds]["lnL_at_profile"] - lnl0
            elif name in POPULATION_PRIOR:
                # the population: ln L alone at the maximum of ln L + ln pi over (g, log10 v) and its six parameters,
                # against the maximum without binaries in the same likelihood (reproduce/binaries/bhb_population_dlnl.py)
                dl = json.load(open(_need(os.path.join(DATA, "bhb_population_dlnl.json"))))[ds][POPULATION_PRIOR[name]]["dlnL"]
            else:
                dl = np.nan
            key = "bins: 30" if name.startswith("bins: 30") else pair_of[name][0] if name in pair_of else name
            res.setdefault(key, {})[ds] = (r, dl)
    paired = {n for pair in FIGURE_PAIRS for n in pair[1:]}
    missing = [n for n in SYMBOL if n not in res and n != "bins: 30 (IPTA)" and n not in paired]
    if missing:
        print(f"  table_switch_numbers: no row in the scan for {missing}")
    order = sorted(res, key=_symkey)

    def fmt(x, f):
        """A signed number in math mode, without a negative zero."""
        if x is None or not np.isfinite(x):
            return "--"
        s = format(x, f)
        if float(s) == 0:
            s = s.lstrip("+-")
        return f"${s}$"

    def num(x, key):
        if key == "g":
            return f"{x:.3f}"
        v = 10.0 ** x                              # v in GeV, two significant digits, trailing zeros kept
        return f"{v:.2f}" if v < 1 else f"{v:.1f}" if v < 10 else f"{v:.0f}"

    def iv(cell, key):
        lo, hi, at_lo, at_hi = cell
        # an interval cut by the prior range: only the lower end of v, at 10 MeV (the IPTA hybrid template)
        assert not at_hi and not (at_lo and key != "v"), "an interval cut by the prior range"
        return f"{num(lo, key)}--{num(hi, key)}"     # an end cut by the prior range is named in the caption

    def intervals(ivs):
        return " & ".join(iv(ivs[k], k[0]) for k in ("g1", "g2", "v1", "v2"))

    names_ds = {"NG15": "NANOGrav 15$\\,$yr", "IPTA2": "IPTA DR2"}
    # one table per data set: the IPTA appendix compares the two
    for ds, path in (("NG15", "switch_numbers.tex"), ("IPTA2", "switch_numbers_ipta.tex")):
        out = [r"\begin{tabular}{@{}llllllrrr@{}}", r"\toprule",
               r" & & \multicolumn{2}{c}{$g$} & \multicolumn{2}{c}{$v$ [GeV]} & & & \\",
               r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
               r" & & $1\sigma$ & $2\sigma$ & $1\sigma$ & $2\sigma$ & $\mathcal{D}$ & Area & $\Delta\ln L$ \\",
               r"\midrule",
               r"\multicolumn{2}{@{}l}{This work's default} & " + intervals(default[ds]) + r" & & & \\"]
        prev = None
        for n in order:
            if ds not in res[n]:
                continue
            sym = SYMBOL.get(n, "B5")
            if sym[0] != prev:
                out.append(r"\midrule")
            prev = sym[0]
            short = next((FIGURE_PAIRS[p] for p in FIGURE_PAIRS if p[0] == n), FIG_LABELS.get(n, SHORT.get(n, n))).replace("\u2013", "--")
            r, dl = res[n][ds]
            short = short[0].upper() + short[1:] if short[:1].isalpha() and short[:2] != "LO" else short
            out.append(f"\\lk{{{sym}}} & {short}" + (r"$^\dagger$" if n in NOT_AGAINST_DEFAULT else "") + " & "
                       + intervals(r[9]) + " & " + f"{format(r[5], D_FMT)} & {r[6]:.2f} & {fmt(dl, '+.2f')}" + r" \\")
        out += [r"\bottomrule", r"\end{tabular}"]
        open(os.path.join(HERE, "..", path), "w").write("\n".join(out) + "\n")
    print("wrote switch_numbers.tex")


def table_budget():
    """budget_table.py: ../budget_table.tex from data/line_budget.json."""
    import runpy
    runpy.run_path(os.path.join(HERE, "budget_table.py"), run_name="__main__")


def fig_pipeline():
    """make_schematics.py: the schematic of figure 1."""
    import subprocess
    subprocess.run([sys.executable, os.path.join(HERE, "make_schematics.py")], check=True)


TABLE_OUTPUTS = {"table_switch_numbers": ["switch_numbers.tex", "switch_numbers_ipta.tex"],
                 "table_budget": ["budget_table.tex"]}


if __name__ == "__main__":
    import time
    # by default every figure and every table that main.tex includes
    which = sys.argv[1:] or ["fig_pipeline", "fig_rate", "fig_spectra", "fig_plane", "fig_hardening", "fig_budget",
                             "fig_foreground", "table_switch_numbers", "table_budget"]
    t_start = time.time()
    for w in which:
        globals()[w]()
    # the build is a gate: every requested output must have been written in this run
    outputs = [p for w in which for p in ([os.path.join(HERE, "..", t) for t in TABLE_OUTPUTS[w]]
                                          if w in TABLE_OUTPUTS else [os.path.join(HERE, w + ".pdf")])]
    # (the file system keeps whole seconds on some mounts, hence the floor)
    stale = [o for o in outputs if not os.path.exists(o) or os.path.getmtime(o) < np.floor(t_start)]
    assert not stale, f"not written in this run: {stale}"
    # and every figure has the width at which main.tex includes it
    size_table([(os.path.join(HERE, w + ".pdf"), FRAC[w] / SCALE.get(w, 1.0)) for w in which if w in FRAC and w != "fig_pipeline"])
    if os.environ.get("FIGCHECK_WARN_ONLY"):
        print("DRAFT BUILD: FIGCHECK_WARN_ONLY was set, violations were only printed")
    print("figure build complete, all checks passed, written in this run:",
          ", ".join(os.path.basename(o) for o in outputs))
