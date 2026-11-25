"""The compiled kernels against scipy, against short references in this file and against numbers recorded from the Python
implementation they replace (tests/data/compiled_reference.npz: the percolation integral on every prefix, the post-peak
test of the remeshing and one whole run). Results agree to rounding; a shot or a root search that takes other steps
agrees within its tolerance."""
import json
import math
import os
import sys
import warnings
from unittest import mock

import numpy as np
import pytest
from scipy import interpolate, optimize, special

from transitionlistener import bubbledynamics as bd, errors, finiteT, hydrodynamics, pathDeformation as pdm

from transitionlistener.eft_rate import abelian_higgs_3d
from transitionlistener import percolation_adaptive_rate as pr, thermodynamics as td, tunneling1D
from transitionlistener.eft_rate import rg
from transitionlistener import percolation_adaptivestepsize as pa
from transitionlistener.eft_rate import nlo_tables
from transitionlistener.compiled import onefield_potential
from transitionlistener.compiled import roots, shooting, splines
from transitionlistener.helper_functions import cubicInterpFunction, load_potential
from transitionlistener.interface import _compute_single_point
from transitionlistener.phases import PhaseInfo, findLocalMinimum

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
if MODELS not in sys.path:
    sys.path.insert(0, MODELS)     # the 3D model files import their 4D parents by name
RECORDED = np.load(os.path.join(ROOT, "tests", "data", "compiled_reference.npz"))


def _model(name):
    return load_potential(os.path.join(MODELS, name), "specific_potential")


def _relative(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape
    scale = np.maximum(np.abs(a), np.abs(b))
    return float(np.max(np.where(scale > 0.0, np.abs(a - b) / np.where(scale > 0.0, scale, 1.0), 0.0))) if a.size else 0.0


def _profile_difference(r1, phi1, r2, phi2):
    """The largest distance of two profiles on their own grids. The end of the radial grid follows the rounding of the
    last shot, so the profiles are compared between their nodes, by a spline: a straight line would measure the grid."""
    r = np.linspace(0.0, min(r1[-1], r2[-1]), 2000)
    return np.max(np.abs(interpolate.CubicSpline(r1, phi1)(r) - interpolate.CubicSpline(r2, phi2)(r)))


# ----------------------------------------------------------------------------------------- 3D bounce
def _lo_cases():
    cls = _model("TL_conformal_u1_no_fermions_3d.py")
    out = []
    for g in (0.5, 0.7, 0.9):
        pot = cls(dict(g=g, v_GeV=math.exp(1 / 6) / g), verbose=False)
        for x in np.geomspace(3e-3, 0.3, 12):
            try:
                out.append(pot.matched_3d_parameters(x * pot.v_GeV))
            except Exception:
                continue
    assert len(out) >= 20
    return out


def test_the_closed_form_shooting_agrees_with_the_python_integrator():
    solved = 0
    for m3sq, lam3, g3sq, mD in _lo_cases():
        V = lambda p: abelian_higgs_3d.V3_LO(p, m3sq, lam3, g3sq, m_debye_sq=mD)             # noqa: E731
        dV = lambda p: abelian_higgs_3d.dV3_LO(p, m3sq, lam3, g3sq, m_debye_sq=mD)           # noqa: E731
        bounded = abelian_higgs_3d.is_bounded_below(lam3)
        try:
            compiled = abelian_higgs_3d.solve_LO_bounce(m3sq, lam3, g3sq, m_debye_sq=mD)
            escape = abelian_higgs_3d.escape_point_LO(m3sq, lam3, g3sq, m_debye_sq=mD)
        except ValueError:
            continue
        python = abelian_higgs_3d.solve_bounce_1d(V, dV, escape, bounded=bounded)
        solved += 1
        # both shoot to xtol = phitol = 1e-11, which fixes S3 to about 1e-9: going to 1e-12 moves either by up to 1.5e-9
        assert compiled[0] == pytest.approx(python[0], rel=1e-8)
        assert _profile_difference(compiled[1], compiled[2], python[1], python[2]) <= 1e-7 * np.max(np.abs(python[2]))
    assert solved >= 20


def test_bezier_root_agrees_with_brentq():
    rng = np.random.default_rng(23)
    n_roots = n_refused = 0
    for _ in range(3000):
        y0, y1, d0, d1 = rng.normal(size=(4, 2)) * 10.0 ** rng.uniform(-3, 1, (4, 2))
        f = cubicInterpFunction(y0, d0, y1, d1)
        target = float(rng.normal() * abs(y0[0] - y1[0]))
        p = [float(f.Y[i][0]) for i in range(4)]
        x, status = shooting.bezier_root(*p, target)
        try:
            ref = optimize.brentq(lambda t: f(t)[0] - target, 0, 1)
        except ValueError:
            assert status != roots.CONVERGED
            n_refused += 1
            continue
        assert status == roots.CONVERGED and x == pytest.approx(ref, rel=1e-12, abs=4e-12)
        n_roots += 1
    assert n_roots > 500 and n_refused > 200
    assert shooting.bezier_root(float("nan"), 1.0, 1.0, -1.0, 0.0)[1] != roots.CONVERGED


def _roots_on(f, a, b):
    state = np.empty(roots.STATE)
    value = f(b)
    x, status = roots.start(state, a, f(a), b, value)
    for _ in range(roots.MAXITER):
        if status != roots.RUNNING:
            break
        x, status = roots.step(state, value)
        if status == roots.RUNNING:
            value = f(x)
    return x, status


def test_the_root_search_takes_the_steps_of_scipy():
    for f, a, b in ((lambda x: x**3 - 2.0, 0.0, 3.0), (lambda x: math.cos(x) - x, 0.0, 1.0),
                    (lambda x: math.exp(x) - 3.0, -4.0, 4.0), (lambda x: x - 1.0, 1.0, 2.0)):
        calls = [[], []]
        x, status = _roots_on(lambda t: calls[0].append(t) or f(t), a, b)
        reference = optimize.brentq(lambda t: calls[1].append(t) or f(t), a, b)
        assert status == roots.CONVERGED and x == reference
        assert sorted(calls[0]) == sorted(calls[1])
    assert _roots_on(lambda x: x * x + 1.0, 0.1, 1.0)[1] == roots.SAME_SIGN
    assert _roots_on(lambda x: float("nan"), 0.1, 1.0)[1] == roots.NOT_A_NUMBER
    assert _roots_on(lambda x: x - 0.3 if x <= 0.2 or x == 1.0 else float("nan"), 0.1, 1.0)[1] == roots.NOT_A_NUMBER
    limit = _roots_on(lambda x: 1.0 if x > 0.3 else -1.0 + 1e-300 * x, 0.0, 1.0)
    assert limit[1] == roots.CONVERGED and limit[0] == pytest.approx(0.3, abs=1e-9)


# ----------------------------------------------------------------------------------------- shooting start-up
def _instanton():
    # a quartic barrier: V = phi^4/4 - (2/3) phi^3 + 0.4/2 phi^2, false vacuum at 0, true vacuum at 1 + sqrt(0.6),
    # reflected so that the true minimum sits at 0
    V = lambda p: 0.25 * p**4 - 2.0 / 3 * p**3 + 0.2 * p**2   # noqa: E731
    dV = lambda p: p**3 - 2.0 * p**2 + 0.4 * p                  # noqa: E731
    phi_true = 1.0 + math.sqrt(0.6)
    return tunneling1D.SingleFieldInstanton(0.0, phi_true, lambda x: V(phi_true - x), lambda x: -dV(phi_true - x))


def _exact_solution(alpha, r, phi0, dV, d2V):
    """SingleFieldInstanton's equation (2) with scipy.special."""
    beta = np.sqrt(abs(d2V))
    beta_r = beta * r
    nu = 0.5 * (alpha - 1)
    gamma, iv, jv = special.gamma, special.iv, special.jv
    if beta_r < 1e-2:
        s = +1 if d2V > 0 else -1
        terms = [(0.5 * beta_r) ** (2 * k - 2) * s**k / (gamma(k + 1) * gamma(k + 1 + nu)) for k in (1, 2, 3)]
        phi = sum(terms) * 0.25 * gamma(nu + 1) * r**2 * dV * s + phi0
        dphi = sum(t * 2 * k for t, k in zip(terms, (1, 2, 3))) * 0.25 * gamma(nu + 1) * r * dV * s
        return phi, dphi
    bessel, sign = (iv, 1.0) if d2V > 0 else (jv, -1.0)
    p = (0.5 * beta_r) ** -nu
    phi = (gamma(nu + 1) * p * bessel(nu, beta_r) - 1) * dV / d2V + phi0
    dphi = -nu * (p / r) * bessel(nu, beta_r) + p * 0.5 * beta * (bessel(nu - 1, beta_r) + sign * bessel(nu + 1, beta_r))
    return phi, dphi * gamma(nu + 1) * dV / d2V


def test_the_exact_solution_near_the_centre():
    inst = _instanton()
    n = 0
    for r in (1e-4, 0.05, 1.0, 7.0):
        for d in np.geomspace(1e-6, 0.9, 5) * inst.phi_metaMin:
            for curvature in (-0.8, 0.8, 1e-9):
                compiled = inst.exactSolution(r, inst.phi_absMin + d, 0.3, curvature)
                assert _relative(compiled, _exact_solution(inst.alpha, r, inst.phi_absMin + d, 0.3, curvature)) < 1e-12
                n += 1
    assert n == 60


def test_the_initial_conditions():
    inst = _instanton()
    cutoff = 1e-3 * inst.phi_metaMin
    for d in np.geomspace(1e-6, 0.9, 25) * inst.phi_metaMin:
        phi0, rmin = inst.phi_absMin + d, 1e-5
        dV, d2V = inst.dV_from_absMin(d), inst.d2V(phi0)
        r0, phi, dphi = inst.initialConditions(d, rmin, cutoff)
        field, slope = _exact_solution(inst.alpha, r0, phi0, dV, d2V)
        assert _relative((phi, dphi), (field, slope)) < 1e-12
        if r0 > rmin:
            assert abs(phi - inst.phi_absMin) == pytest.approx(abs(cutoff), rel=1e-9)
        else:
            assert abs(_exact_solution(inst.alpha, rmin, phi0, dV, d2V)[0] - inst.phi_absMin) > abs(cutoff) \
                or np.sign(dphi) != np.sign(d)


def test_the_profile_agrees_with_the_python_integrator():
    python = _instanton()
    profile = python.findProfile()
    phi_true = python.phi_metaMin
    closed = [phi_true, 0.4, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]        # dV = 0.4 p + p^3 - 2 p |p|
    compiled = tunneling1D.ClosedFormInstanton(python.phi_absMin, phi_true, python.V, python.dV, closed=closed)
    other = compiled.findProfile()
    assert compiled.findAction(other) == pytest.approx(python.findAction(profile), rel=1e-8)
    assert _profile_difference(other.R, other.Phi, profile.R, profile.Phi) < 1e-7


# ----------------------------------------------------------------------------------------- splines
def test_scalar_pchip_is_scipys():
    rng = np.random.default_rng(3)
    x = np.sort(rng.uniform(0, 10, 40))
    y = np.cumsum(rng.normal(size=40))
    for extrap in (True, False):
        a, b = interpolate.PchipInterpolator(x, y, extrapolate=extrap), splines.PchipInterpolator(x, y, extrapolate=extrap)
        pts = np.concatenate([rng.uniform(-1, 11, 300), x, [x[0], x[-1], np.nan]])
        for p in pts:
            ra, rb = a(float(p)), b(float(p))
            assert np.shape(ra) == np.shape(rb)
            assert (np.isnan(ra) and np.isnan(rb)) or abs(ra - rb) <= 1e-13 * max(abs(ra), 1.0)
        np.testing.assert_array_equal(a(pts), b(pts))               # array calls are scipy's


def test_the_spline_kernels_are_fitpack_and_interp1d():
    tables = onefield_potential._shared_tables()
    tb, _, tabb, wb, tf, _, tabf, wf, _ = tables["J"]
    G = tables["G"]
    rng = np.random.default_rng(11)
    for t, w, tab, tck, lo, hi in ((tb, wb, tabb, finiteT._tckb, finiteT._xbmin, finiteT._xbmax),
                                   (tf, wf, tabf, finiteT._tckf, finiteT._xfmin, finiteT._xfmax)):
        x = np.concatenate([rng.uniform(lo, hi, 3000), rng.uniform(lo, 30.0, 3000), t[3:-3]])
        x = x[(x >= lo) & (x <= hi)]
        assert _relative(interpolate.splev(x, tck, der=1), [splines._splder1(t, w, tab, float(v)) for v in x]) < 1e-11
    for i, spline in enumerate((td.Ieb, td.Ipb, td.Ief, td.Ipf)):
        x = np.concatenate([10 ** rng.uniform(-7, 2.84, 3000), spline.x[1:-2]])
        x = x[(x >= spline.x[0]) & (x < spline.x[-1])]
        assert _relative(spline(x), [splines._bspline(G[4 * i], G[4 * i + 1], float(v)) for v in x]) < 1e-11


# ----------------------------------------------------------------------------------------- thermodynamics
def _fields_geff_reference(bosons, fermions, T, kind, mask):
    """thermodynamics.potential_fields_geff one mode at a time."""
    fn = {"e": td.e_geff, "p": td.p_geff, "s": td.s_geff}[kind]
    total = 0.0
    for i, m2 in enumerate(bosons[0]):
        if mask is None or mask[0][i]:
            total = total + fn(np.sqrt(np.maximum(m2, 0.0)), T, bosons[1][i], "b")
    for i, m2 in enumerate(fermions[0]):
        if mask is None or mask[1][i]:
            total = total + fn(np.sqrt(np.maximum(m2, 0.0)), T, fermions[1][i], "f")
    return total


def test_the_degrees_of_freedom_of_the_fields():
    rng = np.random.default_rng(7)
    m2b, m2f = rng.uniform(0, 4, 6) ** 4, rng.uniform(0, 2, 3) ** 4
    bosons = (m2b, np.array([1, 1, 2, 3, 3, 1.0]), None, None)
    fermions = (m2f, np.array([4, 4, 2.0]))
    masks = (None, (np.array([1, 0, 1, 1, 0, 1], bool), np.array([1, 1, 0], bool)))
    for T in (0.7, np.geomspace(0.01, 30, 50)):
        for kind in ("e", "p", "s"):
            for mask in masks:
                assert _relative(td.potential_fields_geff(bosons, fermions, T, kind, mask),
                                 _fields_geff_reference(bosons, fermions, T, kind, mask)) < 1e-14


def test_the_degrees_of_freedom_of_a_spectrum_per_temperature():
    """Masses that depend on the temperature keep their modes on the first axis."""
    T = np.geomspace(0.1, 3.0, 5)
    m2b = np.array([1.0 + T**2, 2.0 * T**2])
    bosons, fermions = (m2b, np.array([1.0, 3.0]), None, None), (np.array([[0.5, 0.2]]).repeat(5, axis=0).T, np.array([2.0, 2.0]))
    assert _relative(td.potential_fields_geff(bosons, fermions, T, "e"),
                     _fields_geff_reference(bosons, fermions, T, "e", None)) < 1e-14


# ----------------------------------------------------------------------------------------- percolation
def _percolation_history(n=70, cold_nan=0):
    """A descending grid across a rate peak: S/T falls by hundreds, with the spacing uneven as the solver leaves it."""
    rng = np.random.default_rng(17)
    T = np.sort(np.geomspace(1.0, 1e-2, n) * rng.uniform(0.985, 1.015, n))[::-1]
    S = T * (140.0 + 300.0 * (np.log(T / 0.2)) ** 2)
    H = 1e-16 * (1.0 + T ** 2)
    if cold_nan:
        S[-cold_nan:] = np.nan
    return T, H, S


def _sweep(T, H, S, vw, factors=None, monkeypatch=None):
    if factors is not None:
        monkeypatch.setattr(bd, "_time_temperature_factors", lambda *a, **k: factors)
    return bd.percIntegralODE_full_sweep(T, H, S, vw, integral_method="double_integral")[0]


def _close(a, b, rel=1e-12):
    return bool(np.all((np.abs(np.asarray(a) - b) <= rel * np.abs(b)) | (np.isnan(a) & np.isnan(b))))


@pytest.mark.parametrize("cold_nan", [0, 12])
def test_the_sweep_against_the_python_integral(cold_nan, monkeypatch):
    """The bag history and the history of a sound speed, with and without the nan of a saturated action tail."""
    T, H, S = _percolation_history(cold_nan=cold_nan)
    bag = _sweep(T, H, S, 0.8)
    assert np.any(bag > 1e-12) and _close(bag, RECORDED[f"bag_{cold_nan}"])
    cs = 0.3 + 0.02 * np.sin(5 * np.log(T))
    a = np.exp(-np.cumsum(np.log(T / np.roll(T, 1)) * (T < T[0])))
    assert _close(_sweep(T, H, S, 0.8, (cs, a), monkeypatch), RECORDED[f"sound_{cold_nan}"])


def test_each_prefix_of_the_sweep_is_the_single_integral():
    T, H, S = _percolation_history()
    sweep = _sweep(T, H, S, 0.8)
    for i in range(0, T.size, 7):
        assert bd.percIntegral(T[:i + 1], H[:i + 1], S[:i + 1], vw=0.8) == pytest.approx(sweep[i], rel=1e-13, abs=1e-300)


def test_a_step_of_the_action_warns_of_the_cap():
    T, H, S = _percolation_history()
    S = S.copy()
    S[35:] = S[35:] * 0.002
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = _sweep(T, H, S, 1.0)
    assert _close(out, RECORDED["capped"], rel=1e-10)
    assert sum("limited by _SUBGRID_MAX_STEPS" in str(w.message) for w in caught) == 1


def test_a_history_that_starts_at_the_rate_peak_refines_on_fewer_than_four_nodes():
    T, H, S = _percolation_history()
    k = int(np.argmin(np.abs(T - 0.2)))
    out = _sweep(T[k:], H[k:], S[k:], 1.0)
    assert out[2] > 0.0 and _close(out, RECORDED["from_peak"])


def test_irregular_grids():
    T, H, S = _percolation_history()
    T2 = T.copy()
    T2[20] = T2[19]         # two equal temperatures
    assert _close(_sweep(T2, H, S, 1.0), RECORDED["equal_T"])
    T3 = T.copy()
    T3[25] = T3[24] * 1.001  # a step the wrong way inside the grid
    assert bd.percIntegral(T3, H, S) == pytest.approx(float(RECORDED["rising_T"][0]), rel=1e-10)
    bad = RECORDED["single_prefix_bad_H"]
    assert bd.percIntegral(T, np.where(T > 0.5, np.nan, H), S) == pytest.approx(float(bad[0]), rel=1e-10)
    assert bd.percIntegral(T, H, S, cooling_factor=0.9) == pytest.approx(float(bad[1]), rel=1e-10)
    with pytest.raises(errors.PercolationError, match="not decreasing"):
        _sweep(T[::-1], H, S, 1.0)
    assert bd.percIntegral(T[:1], H[:1], S[:1]) == 0.0
    np.testing.assert_array_equal(_sweep(T[:1], H[:1], S[:1], 1.0), np.zeros(1))


def test_three_nodes_are_scipys_pchip():
    from transitionlistener.compiled import percolation
    rng = np.random.default_rng(5)
    for values in (rng.normal(size=3) for _ in range(40)):
        u = np.array([-3.0, -2.2, -2.0])
        y = np.tile(values, (4, 1))
        s = np.empty((4, 3))
        percolation._pchip_slopes(3, np.arange(3), u, y, s)
        assert _relative(s[0], interpolate.PchipInterpolator(u, values).derivative()(u)) < 1e-13


def _grid_reference(values):
    """bubbledynamics._temperature_grid written out."""
    raw = np.asarray(values, dtype=float).reshape(-1)
    ordered = np.sort(raw[np.isfinite(raw)])[::-1]
    unique = [float(ordered[0])]
    for candidate in map(float, ordered[1:]):
        if not math.isclose(unique[-1], candidate, abs_tol=1e-10 * max(abs(unique[-1]), abs(candidate)), rel_tol=0.0):
            unique.append(candidate)
    return np.array(unique)


def test_near_duplicate_temperatures_are_merged():
    values = np.array([1.0, 1.0 - 5e-11, 1.0 - 3e-10, 0.5, 0.5 * (1.0 + 2e-9), 1e-8, 1e-8 * (1.0 + 5e-11), np.nan])
    np.testing.assert_array_equal(bd._temperature_grid(values), _grid_reference(values))
    assert bd._temperature_grid(values).size == 5
    assert bd._temperature_grid(None).size == 0 and bd._temperature_grid([np.nan]).size == 0
    rng = np.random.default_rng(2)
    many = np.concatenate([10 ** rng.uniform(-13, 2, 400), 10 ** rng.uniform(-13, 2, 40) * (1 + 3e-11)])
    np.testing.assert_array_equal(bd._temperature_grid(many), _grid_reference(many))


def test_a_support_point_counts_as_present_within_the_tolerance():
    bank = np.array([2.0, 1.0, 0.5, 0.25])
    free = np.array([2.0 * (1.0 + 5e-11), 1.0 * (1.0 + 5e-10), 0.5 * (1.0 + 2e-9)])
    assert pa._controller_added_support_points(bank, free) == 3
    assert pa._controller_added_support_points(bank, None) == 4
    assert pa._controller_added_support_points(None, free) == 0


def test_the_post_peak_test_answers_as_the_python_one():
    n = int(RECORDED["inert_n"][0])
    answers = []
    for i in range(n):
        high, low, vw, answer = RECORDED[f"inert_{i}_args"]
        got = pr._rate_interval_is_post_peak_inert(RECORDED[f"inert_{i}_T"], RECORDED[f"inert_{i}_S"],
                                                   RECORDED[f"inert_{i}_H"], high, low, None, vw=vw)
        assert got is bool(answer)
        answers.append(got)
    assert any(answers) and not all(answers)
    T = RECORDED["inert_0_T"]
    assert pr._rate_interval_is_post_peak_inert(T, None, T, 1.0, 0.5, None) is False
    assert pr._rate_interval_is_post_peak_inert(T[:2], T[:2], T[:2], 1.0, 0.5, None) is False


def test_the_post_peak_test_reads_the_shoulder_and_the_bound_on_the_integral():
    from transitionlistener.compiled.percolation import rate_interval_is_inert
    T = np.geomspace(1e4, 1e-3, 240)
    H = np.full(240, 1e-20)
    S = T * (100.0 + 40.0 * np.log(T / 0.1) ** 2)
    high, low = float(T[194]), float(T[195])
    assert pr._rate_interval_is_post_peak_inert(T, S, H, high, low, None) is True
    S_with_a_dip = S.copy()
    S_with_a_dip[183] *= 1.05         # the rate falls and rises again on the shoulder
    assert pr._rate_interval_is_post_peak_inert(T, S_with_a_dip, H, high, low, None) is False

    def test(largest_integral):
        return rate_interval_is_inert(T, S, H, high, low, 1.0, -3.0, -10.0, largest_integral, 2, 0.25, 2e-6)

    assert test(1e-4) and test(0.0) and not test(1e-40)


def test_the_scale_factor_follows_the_entropy_and_the_bag_where_the_entropy_cannot():
    from transitionlistener.compiled.percolation import scale_factor_from_entropy
    T = np.array([4.0, 2.0, 1.0, 0.5])
    assert scale_factor_from_entropy(T, 3.0 * T**3) == pytest.approx(T[0] / T, rel=1e-13)
    two_thirds = scale_factor_from_entropy(T, 3.0 * T**2)
    assert two_thirds == pytest.approx((T[0] / T) ** (2.0 / 3.0), rel=1e-13)
    assert scale_factor_from_entropy(T, np.array([1.0, np.nan, 1.0, 0.125])) == pytest.approx(T[0] / T, rel=1e-13)
    rising = np.array([1.0, 2.0, 4.0, 8.0])         # the universe would shrink as it cools
    assert scale_factor_from_entropy(T, rising) == pytest.approx(T[0] / T, rel=1e-13)
    assert scale_factor_from_entropy(T[:0], T[:0]).size == 0


class _Bath:
    """A broken phase for the criteria on a compiled potential."""

    def __init__(self):
        self.pot = _model("TL_conformal_u1_fermions_3d.py")(dict(g=0.6, y=0.45, v_GeV=1.0), verbose=False)
        v = float(self.pot.v)
        T = v * np.geomspace(1e-5, 2.0, 120)
        self.broken = PhaseInfo(0, (v * np.sqrt(np.maximum(1.0 - (T / (1.5 * v)) ** 2, 1e-3)))[:, None], T,
                                0.0 * T[:, None])
        self.symmetric = PhaseInfo(1, (1e-15 * T)[:, None], T, 0.0 * T[:, None])


def test_the_criteria_of_the_broken_phase_find_the_root_of_scipy():
    bath = _Bath()
    pot, broken, symmetric = bath.pot, bath.broken, bath.symmetric
    a, b = 0.02 * pot.v, 1.5 * pot.v
    T = 0.1 * pot.v
    e_sym = float(bd.energyDensity(pot, symmetric, float(T), include_decoupled=False))
    cases = [(bd.Tb_criterion, (float(T), symmetric, broken, pot)),
             (bd.energy_criterion_BRO, (e_sym, 0.8 * e_sym, 0.5, 0.1, broken, pot)),
             (bd.entropy_criterion_SYM_BRO, (0.3 * pot.v, (0.2 / 0.3) ** 3, broken, pot))]
    for criterion, args in cases:
        root = bd.criterion_root(criterion, a, b, args=args)
        assert root == pytest.approx(optimize.brentq(criterion, a, b, args=args), rel=1e-9)
        assert abs(criterion(root, *args)) < 1e-6


def test_a_criterion_without_a_root_gets_the_error_of_scipy():
    bath = _Bath()
    args = (0.1 * float(bath.pot.v), bath.symmetric, bath.broken, bath.pot)
    root = bd.criterion_root(bd.Tb_criterion, 0.02 * bath.pot.v, 1.5 * bath.pot.v, args=args)
    with pytest.raises(ValueError, match="different signs"):
        bd.criterion_root(bd.Tb_criterion, root * (1.0 + 1e-3), root * (1.0 + 2e-3), args=args)
    with pytest.raises(ValueError, match="different signs"):
        bd.criterion_root(lambda x: x * x + 1.0, 0.1, 1.0)


def test_a_whole_run_against_recorded_numbers(monkeypatch):
    """A point of the fermion model with the NLO rate: the numbers of the Python percolation code.

    They were recorded with the temporal mass at h3 = 2 g3^2, g3^2 at LO and the grid of that rate, which tests/data
    keeps, and with the monotone interpolation (PCHIP) of Delta S / S_LO between the grid's values of S_LO, which this test
    pins as well. With g3^2 pinned to that of the grid, the grid's g3^2 convention is not compared. The reheating in step 3
    converts the false vacuum with the symmetric splitting and redshifts with the scale factor of the percolation
    integral, s_sym a^3 = const.
    """
    import importlib
    cls = _model("TL_conformal_u1_fermions_3d.py")
    matching = importlib.import_module("dr_conformal_u1_fermions_matching")
    monkeypatch.setattr(matching, "gauge_coupling_sq", lambda g, T, mubar, order: matching.g3_sq_LO(g, T))
    monkeypatch.delitem(nlo_tables.CONVENTIONS, "g3sq_order")
    from scipy.interpolate import PchipInterpolator
    monkeypatch.setattr(nlo_tables, "make_interp_spline", lambda x, y, k: PchipInterpolator(x, y))

    def factory(p, *a, **k):
        pot = cls(p, *a, **k)
        c = pot.config.nucleationConf
        c.rate_mode, c.mu4_over_T, c.matching_order = "model", 7.0553, "NLO"
        c.input_scheme, c.mu3_choice, c.include_determinants = "zero_quartic", "gT", True
        c.include_dim6 = False
        c.temporal_coupling = "2g3sq"
        c.nlo_grid_file = os.path.join(ROOT, "tests", "data", "conformal_u1_fermions_2g3sq.npz")
        pot.config.percolationConf.integral_method = "double_integral"
        return pot

    context, error = _compute_single_point(dict(g=0.6, y=0.45, v_GeV=1.0), factory, timeout=-1, verbose=False,
                                           include_smbhb=False, max_stage=3)
    assert error is None
    observed = context["strongest_transition_observables"]
    for key, value in json.loads(str(RECORDED["run"])).items():
        assert observed[key] == pytest.approx(value, rel=1e-4), key


# ----------------------------------------------------------------------------------------- the 4D bounce
def _segment_bounce():
    pot = _model("TL_conformal_u1_fermions.py")(dict(g=0.7, y=0.01, v_GeV=0.1), verbose=False)
    v = float(pot.v)
    T = 0.02 * v
    true_vacuum = findLocalMinimum(np.array([v]), T, pot.gradV, pot.d2V)
    V = lambda x: pot.Vtot(x, T, include_radiation=False)  # noqa: E731
    dV = lambda x: pot.gradV(x, T)  # noqa: E731
    return pot, T, [true_vacuum, np.array([0.0])], V, dV


def test_the_compiled_integrator_agrees_with_the_python_one():
    """A one-field bounce through the path deformation: S3/T within 2e-3 and the profile within 1e-7. The profile is the
    sharp test: the action is stationary at the bounce, so a 1% error in the integrator's force moves S3/T by only 4e-5,
    but the profile by 3e-4."""
    pot, T, path, V, dV = _segment_bounce()
    kw = dict(pot.config.tracingConf.tunneling_params)
    python = pdm.fullTunneling(path, V, dV, **dict(kw, tunneling_class=tunneling1D.SingleFieldInstanton))
    with mock.patch.object(shooting, "bisect", wraps=shooting.bisect) as bisect:
        compiled = pdm.fullTunneling(path, V, dV, **kw)
    assert bisect.called
    assert np.isfinite(python.action) and abs(compiled.action / python.action - 1) < 2e-3
    a, b = python.profile1D, compiled.profile1D
    assert _profile_difference(a.R, a.Phi, b.R, b.Phi) < 1e-7 * np.max(np.abs(a.Phi))


def test_rg_flow_serves_the_scales_a_direct_solve_reaches_and_stops_at_a_pole():
    def rhs(_t, u):
        return [0.0, u[1] ** 2]

    rg._FLOWS.clear()
    # du/dt = u^2 from u(0) = 1 has its pole at t = 1
    start = rg.flow("pole", rhs, 0.0, [1.0, 1.0], 0.9, 1e-10, 1e-14)
    assert abs(start[1] - 10.0) < 1e-6
    assert rg.flow("pole", rhs, 0.0, [1.0, 1.0], 1.5, 1e-10, 1e-14) is None
    stored = rg._FLOWS[("pole", True)]
    assert rg.flow("pole", rhs, 0.0, [1.0, 1.0], 0.5, 1e-10, 1e-14)[1] == pytest.approx(2.0, rel=1e-7)
    assert rg._FLOWS[("pole", True)] is stored, "the solution ending at the pole is kept"
    assert not stored[2]
    # a flow that ended on its reach is extended, not refused
    rg._FLOWS.clear()
    easy = lambda _t, u: [-u[0]]    # noqa: E731
    assert rg.flow("easy", easy, 0.0, [1.0], 5.0, 1e-10, 1e-14)[0] == pytest.approx(np.exp(-5.0), rel=1e-7)
    assert rg.flow("easy", easy, 0.0, [1.0], 30.0, 1e-10, 1e-14)[0] == pytest.approx(np.exp(-30.0), rel=1e-6)


@pytest.mark.parametrize("alpha", [1e12, 7e15, 1e30])
def test_kappa_sw_of_a_runaway_is_finite_where_the_jouguet_velocity_rounds_to_one(alpha):
    value = hydrodynamics.kappa_sw(alpha, 1.0, 0.57)
    assert value == pytest.approx(alpha / (0.73 + 0.083 * np.sqrt(alpha) + alpha), rel=1e-14)
