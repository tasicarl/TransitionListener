"""S_3/T of the leading-order 3D bounce from one precomputed table, for any potential of the class below.

The potential. One radial field phi with canonical kinetic term, in abelian_higgs_3d's conventions,

    V(phi) = m3sq phi^2 / 2 + lam3 phi^4 / 4
             - kappa/(12 pi) [ 2 (M2 phi^2)^(3/2) + (mD^2 + c M2 phi^2)^(3/2) - mD^3 ],

where M2 phi^2 is the mass squared of the gauge bosons, all degenerate, mD^2 + c M2 phi^2 the mass squared of their
temporal modes, and n_T = 2 kappa transverse and n_0 = kappa temporal modes run in the loop. c = h3 / (2 M2) is the
A0^2 phi^2 coupling h3 over its leading-order value; it is 1 + O(1e-2) for the conformal models with the NLO-matched h3. The arguments of action_over_T are abelian_higgs_3d's: g3sq is
M2, n_transverse is n_T, n_temporal is n_0. The U(1) has kappa = 1 and M2 = g3^2; an SU(2) doublet with degenerate
bosons has kappa = 3 and M2 = g3^2/4.

What qualifies: one radial field, degenerate gauge bosons, temporal mass mD^2 + c M2 phi^2, n_T : n_0 = 2 : 1, no scalar
loops in the leading-order potential. action_over_T refuses (ValueError) any counts with n_T != 2 n_0 or without gauge
loops. It cannot see the other conditions in its arguments: a potential that violates them must not be passed.

The rescaling. With phi = (mD/sqrt(M2)) f and r = (kappa M2 mD)^(-1/2) rho,

    S_3/T = kappa^(-1/2) (mD/M2)^(3/2) F(A, B, c),   A = m3sq / (kappa M2 mD),   B = lam3 mD / (kappa M2^2),

where F is the action of the one potential f^2 ( (A - h(f))/2 + B f^2/4 ), with
h(f) = (2 f + c (w + 1/(1 + w))) / (6 pi) and w = sqrt(1 + c f^2). So S_3/T is a function of three numbers for every
(g, y, v), and for both conformal models: their fermions enter through the matching alone.

The edges, exact and cheap. The origin is a minimum for A > c A_S, A_S = 1/(4 pi). For B > 0 the potential is negative
somewhere only for B < B_E(A), the double root of A/2 + B f^2/4 = h(f): A = h - f h'/2, B_E = h'/f, solved by Newton
iteration in _edge_B. Below A_S and from B_E on there is no bounce: action_over_T returns None without a solve, as the
direct path does. S_3/T ~ (A - A_S)^(3/2) at the first edge and diverges as (1 - B/B_E)^(-2) at the second.

The table. ln F on a uniform grid in

    s = ln(A - c A_S),   tau = -ln(1 - B/B_E(A, c)),   c,

in which both edges are at infinity and ln F is close to linear; tau = 0 is B = 0. The interpolant is a tensor-product
quintic spline (not-a-knot), C^4 in s, tau and c, so that dS/dT carries no kink and node noise enters with the
spline's small Lebesgue constant. The file tab_data/lo_bounce_tables/lo_bounce_table.npz holds the coefficients; its header records
commit, grid, solver tolerances and the node noise. Build it with the command tl-bounce-table (133882 nodes on s in
[-8, 5], tau in [-8, 6.5], step 0.1, and c in [0.94, 1.06], step 0.02; 47 core minutes).

Outside the grid action_over_T solves the bounce directly: the last part of the way to both edges, where S_3/T is far
below 1 or far above any value that enters a rate, c outside [0.94, 1.06], and m_debye_sq or g3sq that are not
positive.

Accuracy against abelian_higgs_3d.solve_LO_bounce: random nodes in (s, tau, c) agree with direct solves to 1e-6 in ln F
(tests/test_bounce_table.py). Across the grid with s > -6, tau > -6 the interpolation error in ln F is of order 1e-8,
largest at the top of tau. For s < -6 the direct solve itself scatters, by up to about 1e-4 in ln F; the temperature
lines of the conformal models pass there near tau = 0 only, at S_3/T < 0.2, where the nodes are good to of order 1e-8.
Another potential has to be checked: an SU(2) doublet has a larger prefactor and may reach the scatter at S_3/T of
order 10 (estimated, not tested). Along the temperature lines of the conformal models, for S_3/T from a few to several
thousand, ln S_3/T agrees with direct solves to the noise of the direct solve, of order 1e-8, and d ln S_3/d ln T to
of order 1e-7, largest at tau > -0.3 where dtau/d ln T grows.

Tests: tests/test_bounce_table.py.

Documentation: https://tasillo.de/TransitionListener/

Authors:
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""
import argparse
import json
import math
import os
import subprocess
import time

import numpy as np

from transitionlistener.eft_rate import abelian_higgs_3d
from transitionlistener.compiled.bounce_table import edge_B, log_F

TABLE_FILE = "lo_bounce_table.npz"
FORMAT = 2
A_S = 1.0 / (4.0 * np.pi)
#: the grid of the shipped table (see tl-bounce-table)
S_RANGE = (-8.0, 5.0)
TAU_RANGE = (-8.0, 6.5)
STEP = 0.1
#: c = h3 / (2 M2), the temporal mass's coupling over its leading-order value: 1 + O(1e-2) for the conformal models
C_RANGE = (0.94, 1.06)
C_STEP = 0.02

stats = {"table": 0, "none": 0, "direct": 0}
_table = {}


def temporal_ratio(g3sq, h3=None):
    """c = h3 / (2 M2), 1 for the leading-order temporal coupling (h3 None)."""
    return 1.0 if h3 is None else float(h3) / (2.0 * g3sq)


def coordinates(m3sq, lam3, g3sq, m_debye_sq, n_transverse=2, n_temporal=1):
    """(A, B, prefactor) of a potential: S_3/T = prefactor * F(A, B). Raises ValueError for a potential of another class."""
    if not (n_transverse > 0 and n_transverse == 2 * n_temporal):
        raise ValueError(f"the table holds n_T : n_0 = 2 : 1, not n_T = {n_transverse}, n_0 = {n_temporal}")
    kappa = 0.5 * n_transverse
    mD = math.sqrt(m_debye_sq)
    return (m3sq / (kappa * g3sq * mD), lam3 * mD / (kappa * g3sq * g3sq),
            (mD / g3sq) ** 1.5 / math.sqrt(kappa))


def load(path=None):
    """Read the table (once); returns its header."""
    if _table and path is None:
        return _table["header"]
    file = path or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tab_data", "lo_bounce_tables",
                                 TABLE_FILE)
    with np.load(file, allow_pickle=False) as d:
        header = json.loads(str(d["header"]))
        if header.get("format") != FORMAT:
            raise ValueError(f"{file}: format {header.get('format')}, this code reads {FORMAT}")
        coef = np.ascontiguousarray(d["c"], dtype=float)
    ns, nt, nc = header["n"]
    x0, x1 = header["s"]
    y0, y1 = header["tau"]
    z0, z1 = header["c"]
    hx, hy, hz = (x1 - x0) / (ns - 1), (y1 - y0) / (nt - 1), (z1 - z0) / (nc - 1)
    tx = _not_a_knot(x0 + hx * np.arange(ns))
    ty = _not_a_knot(y0 + hy * np.arange(nt))
    tz = _not_a_knot(z0 + hz * np.arange(nc))
    _table.update(header=header, args=(x0, x1, hx, y0, y1, hy, z0, z1, hz, tx, ty, tz, coef))
    return header


def _not_a_knot(x):
    return np.r_[[x[0]] * 6, x[3:-3], [x[-1]] * 6]


def _direct(m3sq, lam3, g3sq, m_debye_sq, n_transverse, n_temporal, h3=None):
    stats["direct"] += 1
    try:
        return abelian_higgs_3d.solve_LO_bounce(m3sq, lam3, g3sq, n_transverse, m_debye_sq=m_debye_sq,
                                     n_temporal=n_temporal, h3=h3)[0]
    except ValueError:       # no barrier, or the marginal edge
        return None


def action_over_T(m3sq, lam3, g3sq, m_debye_sq, n_transverse=2, n_temporal=1, *, h3=None):
    """S_3/T of the leading-order bounce, None where there is no bounce. Takes the parameters of the potential, in
    the convention of abelian_higgs_3d.solve_LO_bounce (see the module docstring for the class of potentials); from the table,
    or by solving the bounce outside the grid."""
    if not (m_debye_sq > 0.0 and g3sq > 0.0):
        if not (n_transverse > 0 and n_transverse == 2 * n_temporal):
            raise ValueError(f"the table holds n_T : n_0 = 2 : 1, not n_T = {n_transverse}, n_0 = {n_temporal}")
        return _direct(m3sq, lam3, g3sq, m_debye_sq, n_transverse, n_temporal, h3)
    A, B, prefactor = coordinates(m3sq, lam3, g3sq, m_debye_sq, n_transverse, n_temporal)
    if not _table:
        load()
    code, value = log_F(A, B, temporal_ratio(g3sq, h3), *_table["args"])
    if code == 0:
        stats["table"] += 1
        return prefactor * math.exp(value)
    if code == 1:
        stats["none"] += 1
        return None
    return _direct(m3sq, lam3, g3sq, m_debye_sq, n_transverse, n_temporal, h3)


def clear():
    """Forget the table in memory and the counters."""
    _table.clear()
    for k in stats:
        stats[k] = 0


def _solve_node(node):
    """ln F at (s, tau, c), solved on the canonical potential; nan where the solver gives none."""
    s, tau, c = node
    A = c * A_S + math.exp(s)
    B = float(edge_B(A, c)) * -math.expm1(-tau)
    try:
        return math.log(abelian_higgs_3d.solve_LO_bounce(A, B, 1.0, m_debye_sq=1.0, h3=2.0 * c)[0])
    except ValueError:
        return math.nan


def _pool_map(f, nodes, workers):
    from multiprocessing import Pool
    with Pool(workers) as pool:
        return pool.map(f, nodes, chunksize=max(1, len(nodes) // (32 * workers)))


def _noise(G):
    """rms noise of ln F along each axis, from sixth differences (a smooth function leaves its own sixth derivative)."""
    return [float(np.sqrt(np.mean(np.diff(G, 6, axis=k) ** 2) / 924.0)) for k in range(G.ndim)]


def _commit():
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        run = lambda *a: subprocess.run(["git", "-C", here, *a], capture_output=True, text=True, check=True).stdout.strip()
        return run("rev-parse", "HEAD") + ("+dirty" if run("status", "--porcelain", "--", "..") else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build(out, s_range=S_RANGE, tau_range=TAU_RANGE, step=STEP, workers=1, nodes_out=None, c_range=C_RANGE,
          c_step=C_STEP):
    """Solve the grid and write the table to `out`. Returns the header. Refuses a grid with a node the solver missed."""
    from scipy.interpolate import make_interp_spline

    s = np.linspace(*s_range, int(round((s_range[1] - s_range[0]) / step)) + 1)
    tau = np.linspace(*tau_range, int(round((tau_range[1] - tau_range[0]) / step)) + 1)
    cs = np.linspace(*c_range, int(round((c_range[1] - c_range[0]) / c_step)) + 1)
    nodes = [(float(a), float(b), float(q)) for a in s for b in tau for q in cs]
    t0 = time.time()
    G = np.array(_pool_map(_solve_node, nodes, workers)).reshape(s.size, tau.size, cs.size)
    seconds = time.time() - t0
    if nodes_out is not None:
        np.savez_compressed(nodes_out, s=s, tau=tau, c=cs, G=G)
    missed = np.argwhere(~np.isfinite(G))
    if missed.size:
        raise RuntimeError(f"{len(missed)} of {G.size} nodes have no bounce, first at (s, tau, c) = "
                           f"{(s[missed[0, 0]], tau[missed[0, 1]], cs[missed[0, 2]])}")
    coef = G
    for axis, x in enumerate((s, tau, cs)):
        coef = np.moveaxis(make_interp_spline(x, coef, k=5, axis=axis).c, 0, axis)
    sig = abelian_higgs_3d.solve_LO_bounce.__defaults__
    header = dict(format=FORMAT, commit=_commit(), n=[int(s.size), int(tau.size), int(cs.size)],
                  s=[float(x) for x in s_range], tau=[float(x) for x in tau_range], c=[float(x) for x in c_range],
                  quantity="ln F(s, tau, c), quintic not-a-knot spline coefficients",
                  solve_LO_bounce=dict(n_transverse=2, n_temporal=1, wall_multiple=sig[1], wall_power=sig[2],
                                       xtol=sig[3], phitol=sig[4]),
                  noise_rms_sixth_difference=_noise(G), core_seconds=seconds * workers,
                  built=time.strftime("%Y-%m-%d %H:%M:%S"))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez_compressed(out, c=coef, header=np.array(json.dumps(header)))
    return header


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the table of the leading-order S_3/T (see "
                                             "transitionlistener.eft_rate.abelian_lo_bounce_table).")
    ap.add_argument("--out", required=True, help="the .npz file to write")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--step", type=float, default=STEP, help="node spacing in s and tau")
    ap.add_argument("--s-range", type=float, nargs=2, default=S_RANGE)
    ap.add_argument("--tau-range", type=float, nargs=2, default=TAU_RANGE)
    ap.add_argument("--c-range", type=float, nargs=2, default=C_RANGE, help="range of c = h3 / (2 M2)")
    ap.add_argument("--c-step", type=float, default=C_STEP)
    ap.add_argument("--nodes-out", default=None, help="also keep the raw node values in this .npz file")
    args = ap.parse_args(argv)
    print(json.dumps(build(args.out, tuple(args.s_range), tuple(args.tau_range), args.step, args.workers,
                           args.nodes_out, tuple(args.c_range), args.c_step), indent=1))


if __name__ == "__main__":
    main()
