"""The kernels of Deformation_Spline against the numpy expressions of the same step, and the rare branches around them."""
import numpy as np
import pytest

from transitionlistener import errors

from transitionlistener import helper_functions as hf
from transitionlistener.compiled import deformation as kernels
from transitionlistener.pathDeformation import Deformation_Spline


def curve(n=60, fields=2, seed=3):
    rng = np.random.default_rng(seed)
    s = np.linspace(0.0, 1.0, n)
    phi = np.column_stack([s * rng.normal() + 0.3 * np.sin((a + 2) * s) for a in range(fields)])
    return phi, 0.2 + np.sin(np.pi * s)


def well(X):
    return X**3 - 0.4 * X + 0.1 * X[..., ::-1]


def numpy_forces(d):
    """Normal force of the path of d, in the numpy expressions the kernel replaces."""
    beta, dX, d2X = d._beta, d._dX, d._d2X
    dphi = np.sum(beta[np.newaxis, :, :] * dX[:, :, np.newaxis], axis=1) + (d.phi[-1] - d.phi[1])[np.newaxis, :]
    d2phi = np.sum(beta[np.newaxis, :, :] * d2X[:, :, np.newaxis], axis=1)
    dphi_sq = np.sum(dphi * dphi, axis=-1)[:, np.newaxis]
    dphids = dphi / np.sqrt(dphi_sq)
    d2phids2 = (d2phi - dphi * np.sum(dphi * d2phi, axis=-1)[:, np.newaxis] / dphi_sq) / dphi_sq
    dV = d.dV(d.phi)
    return d2phids2 * d.v2 - (dV - np.sum(dV * dphids, axis=-1)[:, np.newaxis] * dphids)


def numpy_fit(d, phi):
    """The spline fit of a moved path by lstsq: the path, the coefficients."""
    line = phi[:1] + (phi[-1:] - phi[:1]) * d._t
    beta = np.linalg.lstsq(d._X, phi - line, rcond=None)[0]
    return np.sum(beta[np.newaxis, :, :] * d._X[:, :, np.newaxis], axis=1) + line, beta


@pytest.mark.parametrize("kb", [1, 2, 3, 4])
def test_the_basis_against_the_numpy_recursion_and_its_derivatives(kb):
    t = np.append(np.append([0.0] * (kb - 1), np.linspace(0, 1, 13 - kb)), [1.0] * (kb - 1))
    x = np.concatenate(([1e-100], np.sort(np.random.default_rng(1).uniform(0, 1, 200)), np.unique(t[1:])))
    N, dN, d2N = hf.Nbspld2(t, x, kb)
    N1, dN1 = hf.Nbspld1(t, x, kb)
    np.testing.assert_allclose(N, hf.Nbspl(t, x, kb), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(dN, dN1, rtol=1e-13, atol=1e-11)
    if kb > 2:
        h = 1e-6
        inner = np.linspace(0.013, 0.987, 41)
        slope = (hf.Nbspld1(t, inner + h, kb)[1] - hf.Nbspld1(t, inner - h, kb)[1]) / (2 * h)
        np.testing.assert_allclose(hf.Nbspld2(t, inner, kb)[2], slope, rtol=1e-5, atol=1e-5)


def test_the_basis_takes_a_list_and_refuses_too_few_knots():
    assert hf.Nbspld2([0, 0.5, 1], [0.25], 1)[0].shape == (1, 1)
    with pytest.raises(Exception, match="require"):
        hf.Nbspld2([0.0, 1.0], [0.5], 3)


@pytest.mark.parametrize("fields", [1, 2, 5])
def test_the_force_of_the_kernel(fields):
    phi, speed = curve(fields=fields)
    d = Deformation_Spline(phi, speed, well)
    np.testing.assert_allclose(d.forces().F_norm, numpy_forces(d), rtol=1e-11, atol=1e-11)


def test_fixed_ends_have_no_force():
    phi, speed = curve()
    F = Deformation_Spline(phi, speed, well, fix_start=True, fix_end=True).forces().F_norm
    assert np.all(F[0] == 0.0) and np.all(F[-1] == 0.0) and np.all(F[1:-1] != 0.0)


def test_steps_against_the_numpy_expressions_also_through_a_reversal():
    phi, speed = curve()
    d = Deformation_Spline(phi, speed, well, nb=8)
    step, reversals = 1e-3, 0
    for _ in range(30):
        start, beta_start = d.phi, d._beta
        F = numpy_forces(d)
        dV = d.dV(d.phi)
        F = F * d._L / np.max(np.sqrt(np.sum(dV * dV, -1)))
        last, F_last = d._phi_prev, d._F_prev
        step, is_reversed, ratio = d.step(step, reverseCheck=0.05, stepDecrease=2.0, maxstep=2.0)
        reversals += is_reversed
        origin, force = (last, F_last) if is_reversed else (start, F)
        new, beta = numpy_fit(d, origin + force * step)
        np.testing.assert_allclose(d.phi, new, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(d._beta, beta, rtol=1e-8, atol=1e-8)
        fitted = (new - origin) / step
        assert ratio == pytest.approx(np.max(np.sqrt(np.sum(fitted * fitted, -1))) / d._L, rel=1e-8)
        if is_reversed:
            np.testing.assert_array_equal(d._phi_prev, last)
    assert reversals > 0


def test_a_step_leaves_the_stored_path_alone():
    phi, speed = curve()
    d = Deformation_Spline(phi, speed, well, save_all_steps=True)
    d.step(1e-3)
    d.step(1e-3)
    assert len(d.phi_list) == 2 and not np.array_equal(d.phi_list[0], d.phi_list[1])
    np.testing.assert_array_equal(d.phi_list[0], phi)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_a_gradient_that_is_not_finite_stops_the_step(bad):
    phi, speed = curve()

    def broken(X):
        out = well(X)
        out[7, 0] = bad
        return out

    d = Deformation_Spline(phi, speed, lambda X: well(X))
    d.dV = broken
    # refused before the compiled step, with the error the bounce's caller catches
    with pytest.raises(errors.DeformationError):
        d.step(1e-3)


def test_a_path_that_is_not_finite_is_refused():
    phi, speed = curve()
    phi[5, 1] = np.nan
    with pytest.raises(np.linalg.LinAlgError):
        Deformation_Spline(phi, speed, well)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_the_fit_reports_a_target_that_is_not_finite(bad):
    target = np.ones((4, 2))
    assert kernels.fit(np.zeros((3, 4)), target)[1]
    target[2, 1] = bad
    assert not kernels.fit(np.zeros((3, 4)), target)[1]


def test_largest_norm_returns_the_nan_as_numpy_does():
    a = np.array([[3.0, 4.0], [np.nan, 0.0], [10.0, 0.0]])
    assert np.isnan(kernels.largest_norm(a))
    assert kernels.largest_norm(a[[0, 2]]) == 10.0
    assert kernels.largest_norm(np.zeros((3, 2))) == 0.0


def test_a_path_in_reverse_or_a_view_is_taken():
    phi, speed = curve()
    d = Deformation_Spline(phi[::-1], speed[::-1], well)
    np.testing.assert_allclose(d.forces().F_norm, numpy_forces(d), rtol=1e-11, atol=1e-11)
