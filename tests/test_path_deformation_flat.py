"""A path whose potential gradient vanishes everywhere ends the deformation where the path is.

The gradient along a path enters the deformation as a divisor: a step moves each point by the
transverse force divided by the largest gradient on the path, and the force ratio the
deformation converges on is the same quotient. A gradient that vanishes at every point leaves
both of those dividing by zero, and the `nan` path coordinates that follow left the
least-squares fit of the next iteration raising `numpy.linalg.LinAlgError`, which nothing on
the bounce path turns into a result or an error code.

The tests below cover the three places that divide by that gradient, both deformation classes
and `fullTunneling`'s own measurement of the force ratio, and the two neighbouring values that
a test against zero does not catch, `nan` and a positive infinity.
"""
import numpy as np
import pytest

from transitionlistener import errors
from transitionlistener import pathDeformation as pd


def path():
    """A short path, of length 4.9e-08 in field units, that does not lie on a straight line."""
    t = np.linspace(0.0, 1.0, 101)
    phi = 5e-8 * np.column_stack((1e-9 * t, 1.0 - t)) + 1e-9 * np.column_stack(
        (np.sin(3.0 * t), t**2))
    return phi, 1.0 - t**2


def flat(X):
    """A potential gradient that is exactly zero at every point of the path."""
    return 0.0 * X


#: Overall amplitude of the test potential below, in the units `V` is given in. Over a field
#: scale of 1e-07 it makes the gradient of that potential of order 1e-193, whose square is
#: below the smallest subnormal double, 4.9e-324, so `sqrt(sum(dV*dV))` comes out as exactly
#: zero while the gradient itself is not zero. That underflow is how a run reaches the case
#: these tests cover, rather than a potential that is flat.
_UNDERFLOWING_POTENTIAL_AMPLITUDE = 1e-200


def test_a_flat_path_converges_where_it_is():
    """`Deformation_Spline` leaves the path alone and reports a force ratio of zero.

    Under `np.errstate`, so that the vanishing gradient is not divided by at all. A route
    that divides and then reports the quotient it got would otherwise satisfy the equality
    below whenever the numerator happened to be zero as well.
    """
    phi, dphidr = path()
    deformation = pd.Deformation_Spline(phi, dphidr, flat)
    with np.errstate(divide="raise", invalid="raise"):
        assert deformation.step(1e-3) == (1e-3, False, 0.0)
        assert np.array_equal(deformation.phi, phi)
        assert deformation.deformPath() is True
    assert np.array_equal(deformation.phi, phi)


def test_a_flat_path_converges_where_it_is_on_the_points_deformation():
    """`Deformation_Points`, the other deformation class, divides by the same gradient."""
    phi, dphidr = path()
    deformation = pd.Deformation_Points(phi, dphidr, flat)
    with np.errstate(divide="raise", invalid="raise"):
        assert deformation.step(1e-3, 1e-6) == (1e-3, 0.0)
        assert np.array_equal(deformation.phi, phi)
        # `deformPath` scales its stepsize by the same gradient before it takes any step.
        assert deformation.deformPath() is True
    assert np.array_equal(deformation.phi, phi)


def test_the_reported_force_ratio_of_a_flat_path_is_zero():
    """The ratio `fullTunneling` returns is the one the deformation converged on, not `inf`."""
    phi, dphidr = path()
    settings = pd._build_tunneling_settings(
        100, pd.tunneling1D.SingleFieldInstanton, None, None,
        pd.Deformation_Spline, None, None, False, False, None, None)
    with np.errstate(divide="raise", invalid="raise"):
        assert pd._compute_force_ratio(phi, dphidr, flat, settings) == 0.0


@pytest.mark.parametrize("deformation_class", [pd.Deformation_Spline, pd.Deformation_Points])
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_a_gradient_that_is_not_a_number_is_refused(deformation_class, value):
    """A gradient that was not evaluated is a failed deformation, not a converged one.

    `nan == 0.0` and `inf == 0.0` are both False, so a test against zero alone lets each of
    them through, and the step then fills every coordinate of the path with `nan`.
    """
    phi, dphidr = path()
    deformation = deformation_class(phi, dphidr, lambda X: np.full_like(X, value))
    with pytest.raises(errors.DeformationError):
        deformation.deformPath()
    assert np.array_equal(deformation.phi, phi)


def test_a_gradient_on_one_point_of_the_path_still_deforms_it():
    """The guard answers a gradient that vanishes, not a gradient that is small."""
    phi, dphidr = path()

    def one_point(X):
        out = 0.0 * X
        out[50, 0] = 1e-30
        return out

    deformation = pd.Deformation_Spline(phi, dphidr, one_point)
    stepsize, _, ratio = deformation.step(1e-3)
    assert ratio > 0.0 and np.isfinite(ratio) and stepsize > 0.0
    assert not np.array_equal(deformation.phi, phi)


def test_an_underflowing_gradient_comes_back_as_an_unresolved_bounce():
    """The contract through the entry point the pipeline uses, :func:`pathDeformation.bounceAction`.

    The potential here is an ordinary double well in two fields, scaled so that its gradient
    underflows: the two minima are 2e-07 apart in field units, far enough apart for the path
    validation to accept them, while the gradient along the path is of order 1e-193 and its
    square is zero. The bounce cannot be resolved, and the contract is that it is reported as
    one that was not resolved, `trantype` 0 with an infinite action, which every caller already
    reads as no tunnelling at that temperature. Before the guard the same input raised
    `numpy.linalg.LinAlgError` out of the least-squares fit of the second iteration, which no
    handler between the deformation and this entry point converts, so a temperature the
    solver cannot resolve left the bounce solver as a numpy error rather than as an outcome for
    that temperature.
    """
    amplitude, field_scale = _UNDERFLOWING_POTENTIAL_AMPLITUDE, 1e-7

    def V(X, T):
        X = np.asanyarray(X)
        u, w = X[..., 0] / field_scale, X[..., 1] / field_scale
        return amplitude * ((u * u - 1.0)**2 * (1.0 + 0.5 * w * w) - 0.1 * u + 0.3 * w * w)

    def dV(X, T):
        X = np.asanyarray(X)
        u, w = X[..., 0] / field_scale, X[..., 1] / field_scale
        du = amplitude / field_scale * (4.0 * u * (u * u - 1.0) * (1.0 + 0.5 * w * w) - 0.1)
        dw = amplitude / field_scale * ((u * u - 1.0)**2 * w + 0.6 * w)
        return np.stack((du, dw), axis=-1)

    true_vacuum = np.array([field_scale, 0.0])
    false_vacuum = np.array([-field_scale, 0.0])
    # The premise of the test: the gradient is not zero, its square is.
    gradient = dV(np.array([0.3 * field_scale, 0.1 * field_scale]), 1.0)
    assert np.all(np.abs(gradient) > 0.0)
    assert np.sqrt(np.sum(gradient * gradient)) == 0.0

    outdict = {}
    pd.bounceAction(1.0, V, dV, outdict,
                    dict(low_vev=true_vacuum, high_vev=false_vacuum))
    assert outdict[1.0]["action"] == np.inf
    assert outdict[1.0]["trantype"] == 0
