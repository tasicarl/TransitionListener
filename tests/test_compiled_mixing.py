"""The compiled abelian_determinants.log_gy_mixed against scipy's Radau on the same callables: same value, on every path a run
takes."""
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "models"))

from transitionlistener.eft_rate import abelian_higgs_3d  # noqa: E402

from transitionlistener.eft_rate import abelian_determinants as fl  # noqa: E402
from transitionlistener.compiled import mixing as mx  # noqa: E402

G, Y, V, T = 0.7, 0.5, 1.5323027730238088, 0.01686325619024715
#: The Python integrator on this bounce: abelian_higgs_3d.mixing_log_det at l_max = 70 (808 s)
RECORDED_VALUE = -46.696396335804344
RECORDED_ON = {1: 18.85283827558043, 35: 1.391182929384827, 70: 0.6960419882088051}
RECORDED_OFF = {1: 20.944535805720534, 35: 1.3926463982228217, 70: 0.6962312368212772}


@pytest.fixture(scope="module", autouse=True)
def direct_rg():
    """The recorded numbers are for a direct solve of the RG equations; they resolve 1e-12 in the quartic."""
    from transitionlistener.eft_rate import rg
    flow = rg.flow
    rg.flow = lambda *a: None
    yield
    rg.flow = flow


@pytest.fixture(scope="module")
def bounce():
    import dr_conformal_u1_fermions_matching as dr
    mu0 = V / dr.vev_from_zero_quartic(G, Y, 1.0)
    p = dr.matched_soft_parameters_zero_quartic(G, Y, mu0, T, mu4_over_T=2.0 * math.pi, matching_order="NLO", mu3=G * T)
    p = dict(p, g3sq=dr.g3_sq_LO(p["g4"], T))     # the recorded numbers are for g3^2 at LO
    S3, r, phi, _ = abelian_higgs_3d.solve_LO_bounce(p["m3sq"], p["lam3"], p["g3sq"], m_debye_sq=p["mD2"])
    assert S3 == pytest.approx(147.30182784613382, rel=1e-8)
    return r, phi, p


def callables(bounce, dV=None):
    r, phi, p = bounce
    return abelian_higgs_3d._mixing_callables(r, phi, p["m3sq"], p["lam3"], p["g3sq"], m_debye_sq=p["mD2"], n_temporal=1, dV=dV)


def python_integrator(WX, mXp, dWG, l, rmax, mG, n_chunks=200):
    return fl._log_gy_mixed_on_callables(WX, mXp, dWG, l, rmax, mG, 1e-4, 1e-10, 1e-13, n_chunks)


@pytest.fixture
def compiled():
    fl._profile_cache.clear()
    fl._wave_cache.clear()
    return fl.log_gy_mixed


@pytest.mark.parametrize("l", sorted(RECORDED_ON))
def test_each_wave_agrees_with_the_recorded_python_value(bounce, compiled, l):
    WX, mXp, dWG, zero, rmax, mG = callables(bounce)
    assert compiled(WX, mXp, dWG, l, rmax, mG) == pytest.approx(RECORDED_ON[l], rel=1e-8)
    assert compiled(WX, zero, dWG, l, rmax, mG) == pytest.approx(RECORDED_OFF[l], rel=1e-8)


def test_a_wave_agrees_with_the_python_integrator_run_here(bounce, compiled):
    """Few chunks: the chunk count is an argument that has to reach the compiled solve, and this keeps it cheap."""
    WX, mXp, dWG, _, rmax, mG = callables(bounce)
    want = python_integrator(WX, mXp, dWG, 4, rmax, mG, n_chunks=12)
    assert compiled(WX, mXp, dWG, 4, rmax, mG, n_chunks=12) == pytest.approx(want, abs=1e-9)
    assert abs(compiled(WX, mXp, dWG, 4, rmax, mG, n_chunks=12) - compiled(WX, mXp, dWG, 4, rmax, mG)) > 1e-12


def test_the_richardson_value_of_the_whole_sum(bounce, compiled):
    r, phi, p = bounce
    m = abelian_higgs_3d.mixing_log_det(r, phi, p["m3sq"], p["lam3"], p["g3sq"], m_debye_sq=p["mD2"], l_max=70)
    assert m["richardson"]["L46_70"] == pytest.approx(RECORDED_VALUE, abs=1e-7)


def test_the_series_tail_sums_an_exact_series_to_its_limit():
    """A summand that is exactly a sum of the fitted powers: the remainder is recovered to rounding."""
    from scipy.special import zeta
    ells = np.arange(1.0, 31.0)
    nu = ells + 0.5
    coef = dict(zip(abelian_higgs_3d.SERIES_TAIL_POWERS, (-70.0, 12.0, -30.0, 45.0)))
    summand = sum(c * nu ** -p for p, c in coef.items())
    exact = sum(c * zeta(p, 31.5) for p, c in coef.items())
    assert abelian_higgs_3d._mixing_series_tail(ells, summand) == pytest.approx(exact, rel=1e-10)


def test_the_series_tail_converges_where_richardson_does_not(compiled):
    """On a bounce of the conformal benchmark the sum at l_max = 30 already has its l_max = 70 value (5e-6 measured);
    the Richardson estimate S(L) = S_inf - c/L moves by 2e-2 between the two."""
    nt, p, r, phi, T = _cold_bounce()
    r_u, phi_u, m3sq, lam3, g3sq, mD2 = nt._in_units(p, r, phi, T)
    at = {L: abelian_higgs_3d.mixing_log_det(r_u, phi_u, m3sq, lam3, g3sq, m_debye_sq=mD2, l_max=L) for L in (30, 70)}
    assert at[30]["value"] == pytest.approx(at[70]["value"], abs=1e-4)
    assert abs(at[30]["richardson"]["L20_30"] - at[70]["richardson"]["L46_70"]) > 1e-2


def test_scalar_sampling_gives_the_table_of_the_array_form(bounce, compiled):
    """A dV of the caller may be scalar only; the callable then has no array form and is sampled point by point."""
    r, phi, p = bounce
    WX, mXp, dWG, _, rmax, mG = callables(bounce)
    custom = lambda q: abelian_higgs_3d.dV3_LO(q, p["m3sq"], p["lam3"], p["g3sq"], m_debye_sq=p["mD2"], n_temporal=1)
    WX2, mXp2, dWG2, *_ = callables(bounce, dV=custom)
    assert not hasattr(dWG2, "array") and hasattr(dWG, "array")
    grid = np.geomspace(1e-4, rmax, 500)
    for f in (WX, mXp, dWG):
        assert np.array_equal(f.array(grid), np.array([f(q) for q in grid]))
    assert compiled(WX2, mXp2, dWG2, 7, rmax, mG) == pytest.approx(compiled(WX, mXp, dWG, 7, rmax, mG), abs=1e-11)


def test_the_massless_false_vacuum_branch():
    """m = 0 has no Bessel table: h = 1 and U_G = 2 (l + 1)/r."""
    W = lambda r: 0.3 * math.exp(-r * r / 4.0)
    p = lambda r: -0.1 * r * math.exp(-r * r / 8.0)
    d = lambda r: 0.2 * math.exp(-r * r / 4.0)
    want = python_integrator(W, p, d, 2, 20.0, 0.0, n_chunks=10)
    assert fl.log_gy_mixed(W, p, d, 2, 20.0, 0.0, n_chunks=10) == pytest.approx(want, abs=1e-8)


def test_a_callable_no_grid_represents_goes_to_the_python_integrator(compiled, monkeypatch):
    """A bump far narrower than a grid cell: the check between nodes has to notice, and the value has to stay right."""
    calls = []
    real = fl._log_gy_mixed_on_callables
    monkeypatch.setattr(fl, "_log_gy_mixed_on_callables", lambda *a: calls.append(a) or real(*a))
    monkeypatch.setattr(fl, "PROFILE_N_MAX", 2000)
    W = lambda r: 0.01 * math.exp(-((r - 3.0) / 2e-4) ** 2) + 1e-3
    zero = lambda r: 0.0
    fl.log_gy_mixed(W, zero, zero, 3, 10.0, 0.1, n_chunks=4)
    assert calls, "the table was used for a callable it does not represent"


def test_a_nan_in_a_callable_is_not_tabulated(compiled, monkeypatch):
    """nan passes a `>` tolerance test; here it must reach the Python integrator, whose error is the right one."""
    W = lambda r: float("nan") if r > 5.0 else 0.01
    zero = lambda r: 0.0
    with pytest.raises(fl.FluctuationSolverError):
        fl.log_gy_mixed(W, zero, zero, 3, 10.0, 0.1, n_chunks=4)


def test_a_failed_compiled_solve_raises_the_solver_error(compiled, monkeypatch):
    monkeypatch.setattr(fl, "_log_det", lambda *a: (mx.FAILED_STEP, float("nan"), 3))
    W = lambda r: 0.01
    zero = lambda r: 0.0
    with pytest.raises(fl.FluctuationSolverError, match="at l=3"):
        fl.log_gy_mixed(W, zero, zero, 3, 10.0, 0.1, n_chunks=8)


def _coefficients_of_a_cold_bounce():
    """A(r) where the shifted matrix needs pivoting: mu ~ 1e-14 against unit entries, as on a bounce of size 1e22."""
    return np.array([3.7e-41, -1.5e-24, 5.7e-62, -4.1e-62, 2.1e-24, 1.8e-41, 3.3e-19, 1e-18, 6.7e-19])


@pytest.mark.parametrize("mu", [1e-14, (1e-14 + 3e-14j), 1e-3])
def test_the_linear_solve_with_pivoting(mu):
    """The solve inside the Newton iteration. A wrong solve leaves the iteration stalled at rate ~ 1 without a wrong
    number: the bounce of size 1e22 then took 100 times the steps and never finished, and only pivoting shows it."""
    k = _coefficients_of_a_cold_bounce()
    A = np.zeros((6, 6))
    A[0, 3] = A[1, 4] = A[2, 5] = 1.0
    A[3, 3], A[4, 4], A[5, 5] = -k[6], -k[7], -k[8]
    A[3, 0], A[3, 2], A[4, 1], A[4, 2], A[5, 2], A[5, 0], A[5, 1] = k[0], k[1], k[0], k[2], k[5], k[3], k[4]
    dtype = complex if isinstance(mu, complex) else float
    M = np.empty((6, 6), dtype=dtype)
    piv = np.empty(6, dtype=np.int64)
    mx._lu(k, mu, M, piv)
    b = np.arange(1.0, 19.0).reshape(6, 3).astype(dtype)
    x = b.copy()
    mx._solve(M, piv, x)
    matrix = mu * np.identity(6) - A
    assert np.all(np.abs(matrix @ x - b) <= 1e-12 * (np.abs(matrix) @ np.abs(x) + np.abs(b)))


def _cold_bounce():
    import TL_conformal_u1_no_fermions_3d as m
    from transitionlistener.eft_rate import nlo_tables as nt
    pot = m.specific_potential({"g": 0.42, "v_GeV": 1.0}, verbose=False)
    conf = pot.config.nucleationConf
    conf.input_scheme, conf.mu3_choice, conf.mu4_over_T, conf.rate_mode = "zero_quartic", "gT", 7.0553, "model"
    conf.temporal_coupling = "2g3sq"   # the recorded value below, with g3^2 at LO
    soft = pot.soft_parameters
    pot.soft_parameters = lambda T_GeV: (lambda p: dict(p, g3sq=p["g4"] ** 2 * T_GeV))(soft(T_GeV))
    T = 1.1078220303368486e-20
    p, _, r, phi, _ = nt._bounce(pot, T)
    assert r[-1] > 1e22
    return nt, p, r, phi, T


def test_a_bounce_of_size_1e22_finishes_in_gev(compiled):
    """It is not a number to use (see below), but the integration has to end: this is what never ended."""
    nt, p, r, phi, T = _cold_bounce()
    WX, mXp, dWG, zero, rmax, mG = callables((r, phi, p))
    assert math.isfinite(compiled(WX, mXp, dWG, 1, rmax, mG, n_chunks=4))


@pytest.mark.parametrize("factor", [10.0, 0.1])
def test_in_units_of_T_the_sum_does_not_depend_on_the_unit(compiled, factor):
    """Measured: -17.49530 at unit/T from 0.1 to 10; in GeV the same bounce gives 8e5.

    -17.49535 before the profile was saved on radii that resolve the wall (tunneling1D._savedProfile).
    """
    nt, p, r, phi, T = _cold_bounce()
    def value(unit):
        r_u, phi_u, m3sq, lam3, g3sq, mD2 = nt._in_units(p, r, phi, unit)
        return abelian_higgs_3d.mixing_log_det(r_u, phi_u, m3sq, lam3, g3sq, m_debye_sq=mD2, l_max=70)["value"]
    assert value(T) == pytest.approx(-17.49530330, abs=2e-6)
    assert value(T * factor) == pytest.approx(value(T), abs=2e-6)


def test_a_nan_step_size_stops_the_integration(compiled, monkeypatch):
    """nan in the step control would never leave the step loop: h_abs < min_step is False for it."""
    W = lambda r: 0.01
    zero = lambda r: 0.0
    with pytest.raises(fl.FluctuationSolverError):
        fl.log_gy_mixed(W, zero, zero, 3, 10.0, 0.1, rtol=float("nan"), n_chunks=2)


def test_a_negative_or_nan_mass_goes_to_the_python_integrator(compiled, monkeypatch):
    seen = []
    monkeypatch.setattr(fl, "_log_gy_mixed_on_callables", lambda *a: seen.append(a) or 1.5)
    W = lambda r: 0.01
    assert fl.log_gy_mixed(W, W, W, 3, 10.0, float("nan")) == 1.5
    assert fl.log_gy_mixed(W, W, W, 3, 1e-5, 0.1) == 1.5
    assert len(seen) == 2
