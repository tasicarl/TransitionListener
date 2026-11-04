r"""The second counterterm coupling of the dark flip-flop multiplies a quartic.

The docstring of `DarkFlipFlop.Vct` states the second counterterm as
`dlambda2 / 4 * phi_2 ** 4`, and that is what the Lagrangian needs it to be: a
counterterm for the quartic self-coupling of the second field has to carry the
same field power as the coupling it renormalises, or it shifts the mass of that
field instead. `TheSecondCountertermIsQuartic` sets the coupling by hand and
reads the power off.

`calcCounterTerms` leaves `dlambda2` at zero, and `dlambda2` is not one of the
model parameters, so the only way to make the term contribute is to assign the
attribute on an instantiated potential. `TheCountertermCouplingIsZeroInEveryPath`
pins both halves of that statement, and `TheChangeMovesNoNumber` pins what
follows from it: with the counterterms the model computes for itself, the
counterterm potential is bitwise what the pre-change expression gave.

Scope of the gates. The two power tests fail on the pre-change expression. The
no-op test does not: it passes with either power as long as `dlambda2` is zero,
which is what it is there to measure. It fails when `calcCounterTerms` stops
leaving the coupling at zero, because the claim that no published number moves
is false from that moment on.
"""

import contextlib
import io
import unittest
from pathlib import Path

import numpy as np
import yaml

from transitionlistener.errors import InitPotentialError
from transitionlistener.helper_functions import load_potential

_REPO = Path(__file__).resolve().parents[1]
_MODEL = _REPO / "models" / "TL_dark_flipflop.py"
_BENCHMARK_YAML = _REPO / "examples" / "bp_muares.yaml"
_REQUIRED = ("lambda0", "lambda1", "lambda12", "v_GeV", "y", "gamma")


def _benchmark_point():
    """Read the published benchmark parameters of the flip-flop model.

    The parameters are read from `examples/bp_muares.yaml` rather than copied,
    so the point the test uses cannot drift away from the published benchmark.
    """
    with open(_BENCHMARK_YAML, encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)
    point = {name: float(value) for name, value in spec["Parameters"].items()}
    missing = [name for name in _REQUIRED if name not in point]
    if missing:
        raise AssertionError(
            f"{_BENCHMARK_YAML} no longer sets the flip-flop parameters {missing}"
        )
    return point


def _flipflop_class():
    with contextlib.redirect_stdout(io.StringIO()):
        return load_potential(str(_MODEL), "DarkFlipFlop")


def _load_flipflop():
    with contextlib.redirect_stdout(io.StringIO()):
        return _flipflop_class()(_benchmark_point(), verbose=False)


def _Vct_with_a_quadratic_second_term(pot, X):
    """The counterterm potential as it read before the fix.

    The second counterterm multiplies `phi_2 ** 2`. The order of the operations
    is the order `DarkFlipFlop.Vct` uses, so that two results that agree agree
    bitwise and not only to a tolerance.
    """
    X = np.asarray(X)
    phi1 = X[..., 0]
    phi2 = X[..., 1]

    Vct = - 0.5 * pot.dmu1 * phi1**2 - 0.5 * pot.dmu2 * phi2**2
    Vct = Vct + 0.25 * pot.dlambda1 * phi1**4 + 0.25 * pot.dlambda2 * phi2**2
    Vct = Vct + 0.5 * pot.dlambda12 * phi1**2 * phi2**2
    return Vct


class TheSecondCountertermIsQuartic(unittest.TestCase):

    def setUp(self):
        self.pot = _load_flipflop()
        # Isolate the term: every other counterterm of Vct is switched off.
        self.pot.dmu1 = 0.0
        self.pot.dmu2 = 0.0
        self.pot.dlambda1 = 0.0
        self.pot.dlambda12 = 0.0
        self.pot.dlambda2 = 1.0

    def test_the_term_scales_as_the_fourth_power_of_the_second_field(self):
        phi2 = np.array([0.3, 0.6, 1.2])
        X = np.stack([np.zeros_like(phi2), phi2], axis=-1)
        Vct = np.asarray(self.pot.Vct(X), dtype=float)
        # Doubling phi_2 multiplies a quartic by 16 and a quadratic by 4.
        for a, b in zip(Vct[:-1], Vct[1:]):
            self.assertAlmostEqual(b / a, 16.0, places=10)

    def test_the_term_is_the_coupling_over_four_times_the_quartic(self):
        phi2 = 0.7
        X = np.array([0.0, phi2])
        self.assertAlmostEqual(
            float(self.pot.Vct(X)), 0.25 * phi2 ** 4, places=14,
            msg="Vct does not reproduce dlambda2 / 4 * phi_2 ** 4")

    def test_the_first_field_still_carries_its_own_quartic(self):
        self.pot.dlambda2 = 0.0
        self.pot.dlambda1 = 1.0
        phi1 = 0.7
        X = np.array([phi1, 0.0])
        self.assertAlmostEqual(float(self.pot.Vct(X)), 0.25 * phi1 ** 4, places=14)


class TheCountertermCouplingIsZeroInEveryPath(unittest.TestCase):
    """`dlambda2` is zero unless a user assigns the attribute by hand.

    The model fixes its counterterms from the gradient and the Hessian of the
    one-loop potential at the vacuum of the first field, which gives three
    conditions for `dmu1`, `dmu2` and `dlambda1`. No condition is left for
    `dlambda2`, and `calcCounterTerms` assigns it zero with no branch.
    """

    def test_calcCounterTerms_leaves_the_coupling_at_zero(self):
        pot = _load_flipflop()
        self.assertEqual(pot.dlambda2, 0.0)
        self.assertEqual(pot.dlambda12, 0.0)
        # The step of the derivative is an argument, so a caller can change it.
        # It reaches dmu2 and must not reach dlambda2.
        dmu2_default = pot.dmu2
        pot.calcCounterTerms(eps=1.0)
        self.assertEqual(pot.dlambda2, 0.0)
        self.assertNotEqual(pot.dmu2, dmu2_default)

    def test_the_coupling_is_zero_over_a_range_of_model_parameters(self):
        cls = _flipflop_class()
        point = _benchmark_point()
        for gamma in (0.5, 0.7532, 1.0, 2.0):
            for lambda1 in (0.001, 0.002144, 0.004):
                with self.subTest(gamma=gamma, lambda1=lambda1):
                    with contextlib.redirect_stdout(io.StringIO()):
                        pot = cls(dict(point, gamma=gamma, lambda1=lambda1),
                                  verbose=False)
                    self.assertEqual(pot.dlambda2, 0.0)

    def test_the_coupling_is_not_a_model_parameter(self):
        cls = _flipflop_class()
        self.assertNotIn("dlambda2", cls.model_parameters)
        with self.assertRaises(InitPotentialError):
            with contextlib.redirect_stdout(io.StringIO()):
                cls(dict(_benchmark_point(), dlambda2=1.0), verbose=False)


class TheChangeMovesNoNumber(unittest.TestCase):
    """No observable of the published benchmark moves with the fix.

    `dlambda2` is zero, so the term is zero with either field power. The test
    compares the counterterm potential with the expression the fix replaced,
    bitwise, over field values of up to three times the internal field scale of
    1000 internal field units, which is where the model is traced.
    """

    def test_the_counterterm_potential_is_bitwise_unchanged(self):
        pot = _load_flipflop()
        self.assertEqual(pot.dlambda2, 0.0)

        rng = np.random.default_rng(0)
        grid = rng.uniform(-3.0 * pot.v, 3.0 * pot.v, size=(2000, 2))
        corners = np.array([[0.0, 0.0],
                            [pot.v * np.sqrt(1 + pot.lambda1 / pot.lambda0), 0.0],
                            [0.0, pot.v / pot.gamma],
                            [pot.v, pot.v]])
        grid = np.vstack([grid, corners])

        new = np.asarray(pot.Vct(grid), dtype=float)
        old = np.asarray(_Vct_with_a_quadratic_second_term(pot, grid), dtype=float)
        self.assertTrue(np.isfinite(new).all())
        self.assertTrue(
            np.array_equal(new.view(np.int64), old.view(np.int64)),
            msg="the counterterm potential of the benchmark point is not "
                "bitwise what the pre-change expression gave")


if __name__ == "__main__":
    unittest.main()
