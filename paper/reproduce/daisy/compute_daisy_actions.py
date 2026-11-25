"""The four-dimensional daisy-resummed bounce action S3/T of the fermion-free conformal U(1), per
coupling, on a temperature grid, for the exact percolation of build_daisy_table.py.

Potential: TransitionListener's 4D one (TL_conformal_u1_no_fermions via TL_conformal_u1_no_fermions_3d, loaded the way
linescan/pipeline_one.py --model u1 --mode 4d loads it: load_potential on
models/TL_conformal_u1_no_fermions_3d.py; setup() sets nucleationConf.rate_mode "S3overT_4D", so that actionNuc
passes the 4D action through unchanged): one-loop Coleman-Weinberg with log(m^2(phi)/m^2(v)), Arnold-Espinosa
daisy, quartic from radiative symmetry breaking. Internal units: v = v_stable = 1000,
conversionFactor = v_GeV/1000.

At fixed g the potential in internal units does not depend on v (only the field-independent
radiation term carries the SM degrees of freedom at the real temperature), so S3/T is a function of
T_int = 1000 T/v alone; one curve per coupling covers every v (checked by --scalefree). The actions
are computed at v_ref = e^{1/6} GeV/g (the benchmark line, mu0 = 1 GeV).

Kinds:
  tl      TransitionListener's own action: bubbledynamics.calcAction(pot, T, symmetric phase,
          broken phase, ...) on the phases of transitionlistener.phases.Phases, i.e. fullTunneling
          with the model's tunnelling settings (V_spline_samples = 10000, ...). This is the action
          of TransitionListener's own 4D runs. Below T_int ~ 0.27 g (T/v = 1.3e-4
          at g = 0.5 to 2.1e-4 at g = 0.8; T_int_lowest_finite in the caches) it returns an
          infinite action (trantype 0): the path extension of
          pathDeformation.SplinePath (optimize.fmin from the false vacuum with an initial step of
          2.5e-4 path lengths = 0.25 internal field units) steps over the barrier, whose escape
          point is then below 0.25, and ends the path in the mirror minimum at -v. A solver
          limitation, not physics: the barrier exists at every T > 0.
          Grid: T_int = geomspace(0.09, 0.98 Tmax_broken), 80 per decade, plus a bisection of the
          lowest temperature with a finite action (to 1e-4 in ln T).
  direct  DIAGNOSTIC, not TransitionListener's default solver: TL's own one-dimensional instanton
          solver (tunneling1D.SingleFieldInstanton, with the model's findProfile settings) on the
          straight path from the broken minimum to phi = 0, in TL's path coordinate
          x = phi_broken - phi, on the unsplined potential, and with the derivatives taken by the
          solver's own finite differences with a step of 1e-4 T (TL's gradV uses 1e-3 internal
          units, which is the size of the barrier at T_int ~ 1e-3). Agrees with `tl` to 2e-5
          (median; at most 1.3e-3) relative at T_int >= 1; at TL's cutoff `tl` is higher by
          0.13-0.54 in S3/T (its 10000-sample spline of V no longer resolves the barrier), at twice
          the cutoff by 0.01-0.06. Continues smoothly below the `tl` cutoff; its noise (rms
          deviation from a local quadratic in ln T) is below 0.004 in S3/T for T_int > 1e-4,
          0.01-0.03 for 3e-5 < T_int < 1e-4 and 0.03-0.23 for 1e-5 < T_int < 3e-5.
          Grid: T_int = geomspace(1e-5, 0.98 Tmax_broken), 50 per decade.
          Bias: the solver's tolerances (phitol = xtol = 1e-10) are fractions of the
          distance between the two minima, v = 1000 internal units, so the field is resolved to
          1e-7 internal units while the bounce has phi(0) = 4-6 T. S3/T comes out low by 0.0012
          at T_int = 0.1, 0.19 at 1e-4, 6 at 1e-6, and collapses to 0 at 1e-7 (g = 0.46,
          against `bounce`). This, not double precision, ends the direct action.
  bounce  TL's one-dimensional instanton solver set up as for the 3D actions
          (abelian_higgs_3d.solve_bounce_1d): the false vacuum at the origin, V(phi) - V(0) of the unsplined
          4D potential, a wall |V(w)| (phi/w)^24 at w = 30 x the escape point, so that the
          solver's tolerances (1e-8) are fractions of the bounce, not of v; derivatives by
          fourth-order finite differences with a step of 1e-4 x the escape point. Independent of
          the wall (6, 30 and 100 x the escape point agree to 1e-9 relative), of the tolerance
          (1e-8 against 1e-13: 7e-4 in S3/T) and of the step (1e-3 to 1e-5) at T_int = 1e-7 to
          0.1; converged where the direct action is (T_int = 0.1: 235.1969 both). Grid:
          T_int = geomspace(1e-10, 1), 40 per decade; build_daisy_table.py joins it to the direct
          action above T_int = 1 (bias there below 3e-4).
  bounce_rg  the solver of bounce on the potential of models/TL_conformal_u1_no_fermions_rg.py, whose couplings
          are run to mu = MU_OVER_PIT pi T (environment variable, default 1); the grid, the phases and the unit
          conversion are those of the fixed-coupling 4D model TL_conformal_u1_no_fermions.py. Grid:
          T_int = geomspace(1e-10, 0.98 Tmax_broken), 40 per decade, no join. Temperatures where the running
          quartic reaches its Landau pole are recorded with trantype -2.

Writes <DAISY_CACHE>/daisy_<kind><tag>_g<g>.json (paths.DAISY_CACHE, paper/data/daisy_cache unless DAISY_CACHE is
set; couplings whose file exists are skipped): g, v_ref_GeV, conversionFactor, dV_GeV4 (vacuum energy at
v_ref), Tmax_broken_int, and rows {T_int, T_GeV (at v_ref), S3_over_T (NaN where infinite or
failed), trantype}.

Usage (the driver is reproduce/jobs/05_daisy.sh):
  python compute_daisy_actions.py --kind tl [--g 0.55 0.56 ...] [--workers 56] [--per-decade 80]
  python compute_daisy_actions.py --kind direct|bounce|bounce_rg --g ...
  MU_OVER_PIT=2 python compute_daisy_actions.py --kind bounce_rg --tag _mu2 --g 0.5
                                                         the scale band of bounce_rg
  python compute_daisy_actions.py --kind tl --tag probe --per-decade 20 --g 0.50 0.51 ...
                                                         the coarse edge probe
  python compute_daisy_actions.py --scalefree 0.6        the v-independence check (<work>/scalefree_check.txt)
"""
import argparse
import json
import math
import os
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np

sys.dont_write_bytecode = True     # no __pycache__ in the directories this script imports from
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import paths  # noqa: E402  (src/, models/ and the folders of paper/reproduce on sys.path)
ROOT = paths.CODE
MODEL_FILE = os.path.join(ROOT, "models", "TL_conformal_u1_no_fermions_3d.py")   # as pipeline_one.py --model u1
# kind bounce_rg: the same solver as bounce on the potential with the couplings run to mu = MU_OVER_PIT pi T
# (models/TL_conformal_u1_no_fermions_rg.py); the grid, the phases and the vacuum energy are those of MODEL_FILE
MODEL_FILE_RG = os.path.join(ROOT, "models", "TL_conformal_u1_no_fermions_rg.py")
MU_OVER_PIT = float(os.environ.get("MU_OVER_PIT", "1"))
# the phases (grid top) and the unit conversion of bounce_rg: the fixed-coupling 4D model itself, whose potential MODEL_FILE
# uses for its phases too (MODEL_FILE's default rate mode is the three-dimensional one)
MODEL_FILE_4D = os.path.join(ROOT, "models", "TL_conformal_u1_no_fermions.py")
SETUP_MODEL = [MODEL_FILE]
MU0 = 1.0
CACHE = paths.DAISY_CACHE

# the couplings of the table: 0.001 across the band where the edge of the tl action moves with v
# (the coarse edge probe: first percolation at g = 0.52-0.53 for log10 v = -1.125, 0.54-0.55 on the line,
# 0.55-0.57 for log10 v = 1.625), 0.005 to 0.6, then 0.65-0.8, and 0.50-0.51 below the edge so that
# the side without percolation is represented
G_D = ([0.50, 0.505, 0.51] + [round(float(x), 3) for x in np.arange(0.515, 0.5655, 0.001)]
       + [0.57, 0.575, 0.58, 0.585, 0.59, 0.595, 0.6, 0.65, 0.7, 0.75, 0.8])
G_VALIDATION_EXTRA = [0.9, 1.0]      # TL line couplings above the table, for the comparison on the line only
# the couplings of the direct (diagnostic) table: down to the edge of the direct action
G_DIRECT = [0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.535, 0.54, 0.545, 0.55,
            0.56, 0.57, 0.58, 0.59, 0.6, 0.65, 0.7, 0.75, 0.8]
# with the bounce action the edge of percolation lies at g = 0.43-0.47; steps of 0.0025
# there, so that the edge is set by nodes rather than by the interpolation between nodes 0.01 apart
G_DIRECT = sorted(set(G_DIRECT) | {round(0.42 + 0.0025 * k, 4) for k in range(1, 24)})
GRID = {"tl": dict(T_lo=0.09, per_decade=80), "direct": dict(T_lo=1e-5, per_decade=50),
        "bounce": dict(T_lo=1e-10, per_decade=40, T_hi=1.0),
        "bounce_rg": dict(T_lo=1e-10, per_decade=40)}       # up to 0.98 x the top of the broken phase, no join
# the couplings of the bounce_rg table: those of the bounce table and 0.39-0.415, where the NLO table starts
G_RG = sorted(set(G_DIRECT) | {0.39, 0.395, 0.40, 0.405, 0.41, 0.415})
# the further couplings of the line curves of figure 2 (daisy_line.py reads every cached coupling): densely at the
# edges where the transition stops completing, g = 0.4448-0.4495 for N5 and 0.399-0.404 for N4, and between the
# couplings of the tables, so that the curves resolve their bends
G_LINE_BOUNCE = [0.4448, 0.4449, 0.4451, 0.44511, 0.44512, 0.44514, 0.44516, 0.44518, 0.4452, 0.4453, 0.4454, 0.4455,
                 0.446, 0.4465, 0.447, 0.448, 0.4485, 0.449, 0.4495, 0.485, 0.495, 0.505, 0.515, 0.525]
G_LINE_BOUNCE += [round(0.555 + 0.005 * k, 3) for k in range(49) if round(0.555 + 0.005 * k, 3) not in G_DIRECT]
G_LINE_RG = [0.399, 0.3994, 0.3996, 0.3998, 0.39985, 0.3999, 0.39995, 0.4001, 0.4002, 0.4003, 0.4004, 0.4005, 0.4007,
             0.401, 0.402, 0.403, 0.404, 0.4825, 0.485, 0.4875, 0.4925, 0.495, 0.4975, 0.5025, 0.505, 0.5075, 0.5125,
             0.515, 0.5175, 0.62, 0.64, 0.66, 0.68, 0.72, 0.74, 0.76, 0.78]
# the lowest T_int of the couplings whose transition on the line percolates below the default window: the edges of
# completion of N5 (bounce) and N4 (bounce_rg)
EDGE_T_LO = {"bounce": {g: 1e-15 for g in (0.4448, 0.4449, 0.445, 0.4451, 0.44511, 0.44512, 0.44514, 0.44516,
                                             0.44518, 0.4452, 0.4453, 0.4454)},
             "bounce_rg": {**{g: 1e-16 for g in (0.399, 0.3994, 0.3996, 0.3998, 0.39985, 0.3999, 0.39995)},
                           **{g: 1e-13 for g in (0.4, 0.4001, 0.4002, 0.4003, 0.4004, 0.4005, 0.4007, 0.401,
                                                 0.402, 0.403, 0.404, 0.405)}}}
BOUNCE_WALL, BOUNCE_TOL, BOUNCE_STEP = 30.0, 1e-8, 1e-4


def v_ref(g):
    """The benchmark line v = e^{1/6} mu0/g in GeV."""
    return math.exp(1.0 / 6.0) * MU0 / g


_SETUP = {}


def setup(g, v=None):
    """(pot, symmetric phase, broken phase) of coupling g at v (default v_ref), cached per process."""
    v = v_ref(g) if v is None else v
    key = (g, v)
    if key not in _SETUP:
        from transitionlistener.interface.single_point import load_potential
        from transitionlistener.phases import Phases
        base = load_potential(SETUP_MODEL[0], "specific_potential")
        pot = base({"g": g, "v_GeV": v}, verbose=False)
        if SETUP_MODEL[0] == MODEL_FILE:
            # the action of the four-dimensional potential itself (the 3D model file defaults to rate_mode "model")
            pot.config.nucleationConf.rate_mode = "S3overT_4D"
        ph = Phases(pot, False)
        brk = [p for p in ph.phases.values() if abs(float(np.ravel(p.X[0])[0])) > 1.0]
        sym = [p for p in ph.phases.values() if abs(float(np.ravel(p.X[0])[0])) <= 1.0]
        if len(brk) != 1 or len(sym) != 1:
            raise RuntimeError(f"g={g}: expected one broken and one symmetric phase, found {len(ph.phases)}")
        _SETUP[key] = (pot, sym[0], brk[0])
    return _SETUP[key]


_RG = {}


def rg_potential(g, v=None):
    """The potential of MODEL_FILE_RG at (g, v), mu = MU_OVER_PIT pi T, cached per process."""
    v = v_ref(g) if v is None else v
    if (g, v) not in _RG:
        from transitionlistener.interface.single_point import load_potential
        base = load_potential(MODEL_FILE_RG, "specific_potential")
        _RG[(g, v)] = base({"g": g, "v_GeV": v, "mu_over_piT": MU_OVER_PIT}, verbose=False)
    return _RG[(g, v)]


def action_tl(pot, sym, brk, T):
    """(S3/T, trantype) from TransitionListener's calcAction, exactly as the percolation solvers
    call it (actionNuc included, a pass-through in the 4D mode)."""
    from transitionlistener import bubbledynamics as bd
    od = {}
    S = float(bd.calcAction(pot, T, sym, brk, od))
    tt = int(od[T].get("trantype", -1))
    return (S / T if np.isfinite(S) and S > 0 else float("nan")), tt


def action_direct(pot, brk, T):
    """(S3/T, 1) from tunneling1D.SingleFieldInstanton on the unsplined potential along TL's
    straight path, with finite-difference steps scaled to the temperature (see the docstring)."""
    from transitionlistener import tunneling1D
    from transitionlistener.phases import findLocalMinimum
    x1 = float(np.ravel(findLocalMinimum(brk.valAt(T), T, pot.gradV, pot.d2V))[0])   # as calcAction

    def V(x):
        return pot.Vtot(np.asarray(x1 - np.asarray(x, float))[..., None], T, include_radiation=False)
    prm = pot.config.tracingConf.tunneling_params
    try:
        inst = tunneling1D.SingleFieldInstanton(0.0, x1, V, None, None, phi_eps=1e-4 * T / x1)
        prof = inst.findProfile(**prm["tunneling_findProfile_params"])
        S = float(inst.findAction(prof))
        return (S / T if np.isfinite(S) and S > 0 else float("nan")), 1
    except Exception:          # noqa: BLE001 -- any failure is recorded as a missing action
        return float("nan"), -1


def action_bounce(pot, T):
    """(S3/T, 1) from tunneling1D.SingleFieldInstanton through abelian_higgs_3d.solve_bounce_1d: the false
    vacuum at the origin and a wall at BOUNCE_WALL x the escape point, so that the tolerances are
    relative to the bounce (see the docstring, kind bounce)."""
    from scipy.optimize import brentq
    from transitionlistener.eft_rate import abelian_higgs_3d
    Vpot = getattr(pot, "V_rg", pot.Vtot)         # the potential with running couplings of the _rg models
    V0 = float(np.ravel(Vpot(np.array([[0.0]]), T, include_radiation=False))[0])

    def W(p):
        a = np.asarray(p, float)
        out = np.asarray(Vpot(np.atleast_1d(a)[..., None], T, include_radiation=False),
                         float).reshape(np.atleast_1d(a).shape) - V0
        return float(out[0]) if a.ndim == 0 else out
    try:
        # W(phi) for phi << T is below the rounding of V(0) ~ T^4, so the search starts at 1e-2 T
        a = b = 1e-2 * T
        if not W(a) > 0:
            return float("nan"), -1
        while W(b) >= 0:
            b *= 1.3
        esc = brentq(W, a, b, xtol=1e-14 * T, rtol=1e-14)
        h = BOUNCE_STEP * esc
        dW = lambda p: (W(p - 2 * h) - 8 * W(p - h) + 8 * W(p + h) - W(p + 2 * h)) / (12 * h)
        S = abelian_higgs_3d.solve_bounce_1d(W, dW, esc, bounded=False, wall_multiple=BOUNCE_WALL,
                                  xtol=BOUNCE_TOL, phitol=BOUNCE_TOL)[0]
        return (S / T if np.isfinite(S) and S > 0 else float("nan")), 1
    except Exception:          # noqa: BLE001 -- any failure is recorded as a missing action
        return float("nan"), -1


def task(args):
    g, kind, Ts = args
    warnings.filterwarnings("ignore")
    pot, sym, brk = setup(g)
    rows = []
    for T in Ts:
        t0 = time.time()
        if kind == "tl":
            s, tt = action_tl(pot, sym, brk, float(T))
        elif kind == "bounce":
            s, tt = action_bounce(pot, float(T))
        elif kind == "bounce_rg":
            try:
                s, tt = action_bounce(rg_potential(g), float(T))
            except RuntimeError:
                # the running quartic reaches its Landau pole between mu0 and pi T (T/v below ~1e-11 at g = 0.8): no
                # perturbative potential, no action, recorded as trantype -2. Above the pole, at low T and g >= 0.6, the
                # running drives lambda(pi T) so negative that the barrier vanishes: action_bounce records trantype -1
                s, tt = float("nan"), -2
        else:
            s, tt = action_direct(pot, brk, float(T))
        rows.append(dict(T_int=float(T), S3_over_T=s, trantype=tt, sec=round(time.time() - t0, 3)))
    return g, kind, rows


def refine_cutoff(args):
    """Bisect the lowest temperature at which the tl action is finite, to 1e-4 in ln T, and add
    a few points above it (the percolation of the edge couplings happens right there)."""
    g, lo, hi = args            # lo: infinite action, hi: finite
    warnings.filterwarnings("ignore")
    pot, sym, brk = setup(g)
    rows = []
    while math.log(hi / lo) > 1e-4:
        mid = math.sqrt(lo * hi)
        s, tt = action_tl(pot, sym, brk, mid)
        rows.append(dict(T_int=mid, S3_over_T=s, trantype=tt))
        if np.isfinite(s):
            hi = mid
        else:
            lo = mid
    for f in (1.002, 1.005, 1.01, 1.02):
        s, tt = action_tl(pot, sym, brk, hi * f)
        rows.append(dict(T_int=hi * f, S3_over_T=s, trantype=tt))
    return g, rows, hi


def grid(g, kind, per_decade=None, T_lo=None):
    pot, sym, brk = setup(g)
    T_lo = T_lo or GRID[kind]["T_lo"]
    T_hi = GRID[kind].get("T_hi", 0.98 * float(brk.Tmax))
    n = int(math.ceil(math.log10(T_hi / T_lo) * (per_decade or GRID[kind]["per_decade"]))) + 1
    return np.geomspace(T_lo, T_hi, n), float(brk.Tmax)


def run(gs, kind, workers, per_decade=None, n_chunks=None, tag="", T_lo=None):
    import rate_history_u1 as R
    os.makedirs(CACHE, exist_ok=True)
    lowest = {g: T_lo or EDGE_T_LO.get(kind, {}).get(round(g, 6)) for g in gs}

    def done(g):
        """A cache that exists and reaches down to the window asked for."""
        path = cache_path(g, kind, tag)
        if not os.path.exists(path):
            return False
        need = lowest[g] or GRID[kind]["T_lo"]
        with open(path) as f:
            return min(r["T_int"] for r in json.load(f)["rows"]) <= need * (1 + 1e-9)
    gs = [g for g in gs if not done(g)]
    if not gs:
        return
    t0 = time.time()
    grids = {g: grid(g, kind, per_decade, lowest[g]) for g in gs}
    n_chunks = n_chunks or max(1, int(math.ceil(4 * workers / len(gs))))
    jobs = []
    for g, (Ts, _) in grids.items():
        # interleaved chunks: every chunk spans the whole range, so the chunks cost about the same
        for c in range(n_chunks):
            jobs.append((g, kind, Ts[c::n_chunks]))
    rows = {g: [] for g in gs}
    with Pool(workers) as pool:
        for g, _, r in pool.imap_unordered(task, jobs):
            rows[g] += r
        if kind == "tl":
            ref = []
            for g in gs:
                rr = sorted(rows[g], key=lambda r: r["T_int"])
                fin = [i for i, r in enumerate(rr) if np.isfinite(r["S3_over_T"])]
                if fin and fin[0] > 0:
                    ref.append((g, rr[fin[0] - 1]["T_int"], rr[fin[0]]["T_int"]))
            for g, r, hi in pool.imap_unordered(refine_cutoff, ref):
                rows[g] += r
    for g in gs:
        pot, sym, brk = setup(g)
        rr = sorted(rows[g], key=lambda r: r["T_int"])
        for r in rr:
            r["T_GeV"] = r["T_int"] * pot.conversionFactor
        fin = [r["T_int"] for r in rr if np.isfinite(r["S3_over_T"])]
        out = dict(g=g, kind=kind, v_ref_GeV=v_ref(g), conversionFactor=pot.conversionFactor,
                   dV_GeV4=R.vacuum_energy(g, v_ref(g)), Tmax_broken_int=grids[g][1],
                   T_int_lowest_finite=min(fin) if fin else None, lambda_quartic=float(pot.l),
                   tunneling_params=pot.config.tracingConf.tunneling_params,
                   model_file=MODEL_FILE_RG if kind == "bounce_rg" else MODEL_FILE,
                   mu_over_piT=MU_OVER_PIT if kind == "bounce_rg" else None,
                   seconds_total_run=time.time() - t0, rows=rr)
        json.dump(out, open(cache_path(g, kind, tag), "w"), indent=0)
        print(f"g={g:.3f} {kind}: {len(rr)} temperatures, {len(fin)} finite, lowest finite T_int = "
              f"{out['T_int_lowest_finite']}, {time.time() - t0:.0f} s", flush=True)


def cache_path(g, kind, tag=""):
    """<DAISY_CACHE>/daisy_<kind><tag>_g<g>.json with three decimals, and more
    decimals where three do not represent g or where the shorter name is taken by a different coupling (the
    steps of 0.0005 and 0.00001 at the edge of percolation on the benchmark line, daisy_line.py; some three-decimal
    caches hold half steps, e.g. g0.448 = 0.4475)."""
    for nd in (3, 4, 5, 6):
        p = os.path.join(CACHE, f"daisy_{kind}{tag}_g{g:.{nd}f}.json")
        if os.path.exists(p):
            try:
                same = abs(float(json.load(open(p))["g"]) - g) < 1e-9
            except (KeyError, ValueError, OSError):
                same = False
            if same:
                return p
            continue                      # the name holds a different coupling
        if abs(round(g, nd) - g) < 1e-9:
            return p
    raise ValueError(f"no cache name for g = {g!r}")


def scalefree(g, out_path):
    """S3/T at (g, v1, T_int) against (g, v2, T_int), i.e. (g, v2, T v2/v1) in GeV, for both kinds."""
    warnings.filterwarnings("ignore")
    vs = (v_ref(g), 0.1, 30.0)
    Ts = [0.15, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    lines = [f"Scale-free check at g = {g}: S3/T at the same T/v for v = " + ", ".join(f"{v:.4g}" for v in vs) + " GeV",
             "T_int = 1000 T/v; columns: T/v, S3/T (tl) for each v, largest relative difference; then the same for direct"]
    worst = {"tl": 0.0, "direct": 0.0}
    for kind in ("tl", "direct"):
        for T in Ts:
            vals = []
            for v in vs:
                pot, sym, brk = setup(g, v)
                vals.append(action_tl(pot, sym, brk, T)[0] if kind == "tl" else action_direct(pot, brk, T)[0])
            rel = (max(vals) - min(vals)) / abs(np.mean(vals))
            worst[kind] = max(worst[kind], rel)
            lines.append(f"  {kind:6s} T/v = {T / 1000:.2e}: " + "  ".join(f"{x:.8f}" for x in vals) + f"   max rel diff {rel:.1e}")
    lines.append(f"largest relative difference: tl {worst['tl']:.1e}, direct {worst['direct']:.1e}")
    open(out_path, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=("tl", "direct", "bounce", "bounce_rg"), default="tl")
    ap.add_argument("--g", type=float, nargs="*", default=None)
    ap.add_argument("--line", action="store_true", help="the further couplings of the line curves (G_LINE_BOUNCE for direct and\n"
                    "bounce, G_LINE_RG for bounce_rg)")
    ap.add_argument("--workers", type=int, default=56)
    ap.add_argument("--per-decade", type=int, default=None)
    ap.add_argument("--scalefree", type=float, default=None)
    ap.add_argument("--tag", default="", help="suffix of the cache name (e.g. probe for the coarse edge probe)")
    ap.add_argument("--t-lo", type=float, default=None,
                    help="lowest T_int of the grid instead of GRID[kind]['T_lo'] (the bounce action at the end of"
                         " percolation on the line, where T_p falls below the default window: figure 3)")
    a = ap.parse_args()
    if a.scalefree is not None:
        scalefree(a.scalefree, paths.work("scalefree_check.txt"))
    else:
        if a.line:
            gs = G_LINE_RG if a.kind == "bounce_rg" else G_LINE_BOUNCE
        else:
            gs = a.g if a.g else (G_D + G_VALIDATION_EXTRA if a.kind == "tl" else G_RG if a.kind == "bounce_rg" else G_DIRECT)
        if a.kind == "bounce_rg":
            SETUP_MODEL[0] = MODEL_FILE_4D
        run(gs, a.kind, a.workers, a.per_decade, tag=a.tag, T_lo=a.t_lo)
