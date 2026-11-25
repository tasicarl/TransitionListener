"""The compiled trials of findProfile against the Python loop, the trials it hands back, and the numbers of a SplinePath."""
import numpy as np
import pytest
from scipy import integrate, interpolate

from transitionlistener import errors, pathDeformation as pd, tunneling1D as t
from transitionlistener.compiled import deformation, shooting


class PythonTrials(t.SplinePathInstanton):
    _ordinary_trials = t.SingleFieldInstanton._ordinary_trials


def line_path(depth):
    """The spline of a quartic along a curved path of two fields; the minima lie at the ends."""
    s = np.linspace(0.0, 1.0, 120)
    pts = np.column_stack((s, 0.3 * np.sin(2.0 * s)))

    def V(X):
        X = np.asanyarray(X)
        x = X[..., 0]
        return 0.25 * x**2 * (x - 1.0)**2 * 8.0 + depth * x**2 * (2.0 * x - 3.0) + 0.1 * X[..., 1]**2

    return pd.SplinePath(pts, V, V_spline_samples=1000, extend_to_minima=False)


def on_the_kernel_spline(path):
    """The Python loop with dV and d2V from the numbers the kernel reads, so that the two share every rounding."""
    python = PythonTrials(path.L, 0.0, path.V, path.dV, path.d2V)
    brk, coef = python.P[1], python.P[2]
    python.dV = lambda phi: shooting._spline_slopes(float(phi), brk, coef)[0]
    python.d2V = lambda phi: shooting._spline_slopes(float(phi), brk, coef)[1]
    return python


@pytest.mark.parametrize("depth", [0.02, 0.1, 0.5])
@pytest.mark.parametrize("tol", [1e-4, 1e-8, 1e-10, 1e-13])
def test_the_trials_are_those_of_the_python_loop(depth, tol):
    path = line_path(depth)
    compiled = t.SplinePathInstanton(path.L, 0.0, path.V, path.dV, path.d2V)
    python = on_the_kernel_spline(path)
    a = compiled.findProfile(xtol=tol, phitol=tol)
    b = python.findProfile(xtol=tol, phitol=tol)
    assert compiled.findAction(a) == pytest.approx(python.findAction(b), rel=1e-8)
    assert set(compiled.bracket_branch_actions) == set(python.bracket_branch_actions)
    assert compiled.bracket_branch_actions == pytest.approx(python.bracket_branch_actions, rel=1e-7)
    if tol >= 1e-8:
        # below, numba and numpy round differently, which can flip the last trials: the tail of the profile follows
        np.testing.assert_allclose(a.R, b.R, rtol=1e-9)
        np.testing.assert_allclose(a.Phi, b.Phi, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("depth", [0.02, 0.1, 0.5])
def test_the_action_is_that_of_the_scipy_spline_calls(depth):
    path = line_path(depth)
    compiled = t.SplinePathInstanton(path.L, 0.0, path.V, path.dV, path.d2V)
    python = PythonTrials(path.L, 0.0, path.V, path.dV, path.d2V)
    a = compiled.findAction(compiled.findProfile(xtol=1e-8, phitol=1e-8))
    b = python.findAction(python.findProfile(xtol=1e-8, phitol=1e-8))
    assert a == pytest.approx(b, rel=1e-7)


def test_the_compiled_loop_takes_every_ordinary_trial(monkeypatch):
    path = line_path(0.1)
    instanton = t.SplinePathInstanton(path.L, 0.0, path.V, path.dV, path.d2V)
    monkeypatch.setattr(t.CompiledInstanton, "integrateProfile", lambda *a, **k: pytest.fail("a trial in Python"))
    assert instanton.findProfile(xtol=1e-6, phitol=1e-6).R.size > 0


def arguments(path, x):
    i = t.SplinePathInstanton(path.L, 0.0, path.V, path.dV, path.d2V)
    delta = i.phi_metaMin - i.phi_absMin
    rmin = 1e-4 * i.rscale
    return i, dict(xmin=1e-3, xmax=np.inf, xincrease=5.0, xtol=1e-6, absmin=float(i.phi_absMin), delta_phi=float(delta),
                   metamin=float(i.phi_metaMin), phi_eps=float(i.phi_eps), cutoff=0.01 * float(delta),
                   rscale=float(i.rscale), rmin=rmin, dr0=rmin, drmin=0.01 * rmin, rmax=1e4 * i.rscale, phitol=1e-6,
                   min_slope=1e-200, alpha=float(i.alpha), kept=np.zeros((3, 8))), x


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_an_x_that_is_not_finite_goes_to_python(bad):
    i, kw, x = arguments(line_path(0.1), bad)
    out = shooting.bisect(i.P, x, **kw)
    assert out[0] == shooting.TRIAL_FOR_PYTHON and out[4] == shooting.TRIAL_NONE and not kw["kept"].any()


def test_no_usable_slope_goes_to_python():
    i, kw, x = arguments(line_path(0.1), 1.0)
    kw["min_slope"] = np.inf
    out = shooting.bisect(i.P, x, **kw)
    assert out[0] == shooting.TRIAL_FOR_PYTHON and out[1] == 1.0 and not kw["kept"].any()


def test_a_slope_that_is_not_finite_goes_to_python():
    i, kw, x = arguments(line_path(0.1), 1.0)
    P = (i.P[0], i.P[1], i.P[2] * np.inf)
    assert shooting.bisect(P, x, **kw)[0] == shooting.TRIAL_FOR_PYTHON


def test_an_infinite_slope_away_from_the_minimum_goes_to_python():
    i, kw, x = arguments(line_path(0.1), 1.0)
    coef = i.P[2].copy()
    coef[-1, :] = -np.inf          # delta_phi is negative here, so this is a slope of +inf away from the minimum
    assert kw["delta_phi"] < 0.0
    assert shooting.bisect((i.P[0], i.P[1], coef), x, **kw)[0] == shooting.TRIAL_FOR_PYTHON
    assert not kw["kept"].any()


def test_a_slope_towards_the_minimum_goes_to_python():
    i, kw, x = arguments(line_path(0.1), 1.0)
    coef = i.P[2].copy()
    coef[-1, :] = 1e3              # delta_phi is negative here, so this slope drives the field towards the minimum
    assert shooting.bisect((i.P[0], i.P[1], coef), x, **kw)[0] == shooting.TRIAL_FOR_PYTHON
    assert not kw["kept"].any()


def test_a_start_that_cannot_be_found_goes_to_python():
    i, kw, x = arguments(line_path(0.1), 1.0)
    kw["cutoff"] = np.inf
    assert shooting.bisect(i.P, x, **kw)[0] == shooting.TRIAL_FOR_PYTHON
    assert not kw["kept"].any()


def test_a_bracket_below_the_floating_point_spacing_ends():
    path = line_path(0.1)
    profile = t.SplinePathInstanton(path.L, 0.0, path.V, path.dV, path.d2V).findProfile(xtol=1e-300, phitol=1e-8)
    assert profile.R.size > 0


@pytest.mark.parametrize("x", [1.0, 2.0, 3.0, 5.0])
def test_neighbouring_floats_end_the_bisection(x):
    """An undershoot with the overshoot one float above: the midpoint is one of the two, which only the comparison of
    x with the ends of the bracket stops."""
    i, kw, x = arguments(line_path(0.1), x)
    kw.update(xmin=0.5 * x, xmax=np.nextafter(x, 1.0), xtol=0.0)
    out = shooting.bisect(i.P, x, **kw)
    assert out[0] == shooting.BISECTION_DONE and out[4] == shooting.TRIAL_UNDERSHOOT


def test_a_shot_that_fails_goes_to_python_at_the_same_trial():
    i, kw, x = arguments(line_path(0.1), 1.0)
    kw["rmax"] = -1.0
    out = shooting.bisect(i.P, x, **kw)
    assert out[0] == shooting.TRIAL_FOR_PYTHON and out[1] == 1.0 and not kw["kept"].any()


def test_a_shot_that_fails_raises_as_in_python():
    path = line_path(0.1)
    for cls in (t.SplinePathInstanton, PythonTrials):
        with pytest.raises(errors.TunnelingError, match="r > rmax"):
            cls(path.L, 0.0, path.V, path.dV, path.d2V).findProfile(rmax=1e-3)


def test_a_rejected_trial_between_compiled_ones_leaves_the_same_bounce(monkeypatch):
    """Python takes a trial the kernel refused, and the bisection goes on in the kernel from where Python left it."""
    path = line_path(0.1)
    want = PythonTrials(path.L, 0.0, path.V, path.dV, path.d2V).findProfile(xtol=1e-6, phitol=1e-6)
    calls = []
    real = shooting.bisect

    def refuses_every_other_call(*a):
        calls.append(1)
        if len(calls) % 2:
            return shooting.TRIAL_FOR_PYTHON, a[1], a[2], a[3], shooting.TRIAL_NONE
        return real(*a)

    monkeypatch.setattr(shooting, "bisect", refuses_every_other_call)
    got = t.SplinePathInstanton(path.L, 0.0, path.V, path.dV, path.d2V).findProfile(xtol=1e-6, phitol=1e-6)
    assert len(calls) >= 2
    np.testing.assert_allclose(got.Phi, want.Phi, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("x", [-0.5, 0.0, 0.37, 1.0, 1.5])
def test_the_slopes_are_those_of_the_spline(x):
    knots = np.linspace(0.0, 1.0, 30)
    tck = interpolate.splrep(knots, np.sin(5.0 * knots), s=0)
    pp = interpolate.PPoly.from_spline(tck).derivative()
    value, slope = shooting._spline_slopes(x, pp.x, pp.c)
    assert value == pytest.approx(float(interpolate.splev(x, tck, der=1)), rel=1e-12, abs=1e-12)
    assert slope == pytest.approx(float(interpolate.splev(x, tck, der=2)), rel=1e-12, abs=1e-12)


def test_the_interior_is_the_exact_solution():
    R = np.linspace(0.0, 2.0, 40)
    phi, dphi = shooting.interior(R, 0.3, 0.2, -1.5, 1.5)
    assert phi[0] == 0.3 and dphi[0] == 0.0
    for i in (1, 7, 39):
        assert (phi[i], dphi[i]) == shooting.exact_solution(R[i], 0.3, 0.2, -1.5, 1.5)


def spline(pts, k=3):
    dp = pd._pathDeriv(pts)
    u = integrate.cumulative_trapezoid(np.sqrt(np.sum(dp * dp, axis=1)), initial=0.0)
    return interpolate.splprep(pts.T, u=u, s=0, k=k)[0], u


@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("fields", [1, 2, 4])
def test_the_arc_length_against_a_tight_integral(k, fields):
    s = np.linspace(0.0, 1.0, 80)
    pts = np.column_stack([(a + 1.0) * s + np.sin((a + 2) * s) for a in range(fields)])
    tck, u = spline(pts, k)

    def speed(_, x):
        return float(np.sqrt(np.sum(np.array(interpolate.splev(x, tck, der=1))**2)))

    tight = integrate.odeint(speed, 0.0, u, rtol=1e-11, atol=u[-1] * 1e-11, hmax=u[-1] / 200, mxstep=100000)[:, 0]
    parts = [interpolate.PPoly.from_spline((tck[0], c, k)).derivative() for c in tck[1]]
    length = deformation.arc_length(parts[0].x, np.stack([p.c for p in parts], axis=-1), u, pd._ARC_NODES, pd._ARC_WEIGHTS)
    assert length[0] == 0.0
    np.testing.assert_allclose(length, tight, atol=(1e-4 if k == 1 else 1e-8) * u[-1])


def python_walk(ladder, probes):
    """The walk of _extension_minimum in Python."""
    best_x, best_v, previous_x = 0.0, float(probes[0]), 0.0
    for i in range(1, len(probes)):
        current = float(probes[i])
        if not np.isfinite(current):
            break
        noise = 8.0 * np.finfo(float).eps * max(abs(current), abs(best_v), np.finfo(float).tiny)
        if current > best_v + noise:
            return i, previous_x, best_x
        if current < best_v:
            previous_x, best_x, best_v = best_x, float(ladder[i]), current
    return -1, previous_x, best_x


@pytest.mark.parametrize("seed", range(40))
def test_the_walk_along_the_ladder(seed):
    rng = np.random.default_rng(seed)
    ladder = np.concatenate(([0.0], np.geomspace(1e-3, 10.0, 49)))
    probes = np.cumsum(rng.normal(-0.2, 1.0, 50))
    if seed % 3 == 0:
        probes[rng.integers(5, 50)] = np.nan if seed % 2 else np.inf
    if seed % 5 == 0:
        probes = np.sort(probes)[::-1]
    assert deformation.first_rise(ladder, probes) == python_walk(ladder, probes)


def test_the_walk_stops_at_an_infinite_wall():
    ladder = np.arange(5.0)
    assert deformation.first_rise(ladder, np.array([3.0, 2.0, 1.0, np.inf, 0.0])) == (-1, 1.0, 2.0)


def test_a_plateau_within_rounding_is_no_rise():
    ladder = np.array([0.0, 1.0, 2.0, 3.0])
    probes = np.array([1.0, 1.0 + 1e-16, 0.5, 2.0])
    assert deformation.first_rise(ladder, probes) == (3, 0.0, 2.0)
