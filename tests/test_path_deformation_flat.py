"""A path along which the potential is flat to rounding stays where it is, and the deformation ends there."""
import numpy as np

from transitionlistener.pathDeformation import Deformation_Spline


def path():
    t = np.linspace(0.0, 1.0, 101)
    return 5e-8 * np.column_stack((1e-9 * t, 1.0 - t)) + 1e-9 * np.column_stack((np.sin(3.0 * t), t**2)), 1.0 - t**2


def test_a_flat_path_converges_where_it_is():
    phi, dphidr = path()
    deformation = Deformation_Spline(phi, dphidr, lambda X: 0.0 * X)
    assert deformation.step(1e-3) == (1e-3, False, 0.0)
    assert np.array_equal(deformation.phi, phi)
    assert deformation.deformPath() is True
    assert np.array_equal(deformation.phi, phi)


def test_a_gradient_on_one_point_of_the_path_still_deforms_it():
    phi, dphidr = path()

    def one_point(X):
        out = 0.0 * X
        out[50, 0] = 1e-30
        return out

    deformation = Deformation_Spline(phi, dphidr, one_point)
    stepsize, _, ratio = deformation.step(1e-3)
    assert ratio > 0.0 and np.isfinite(ratio) and stepsize > 0.0
    assert not np.array_equal(deformation.phi, phi)
