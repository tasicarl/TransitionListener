"""The counterterm derivatives must not depend on the phase-tracing accuracy."""

from __future__ import annotations

import types
import unittest
from pathlib import Path
from unittest import mock

import mpmath as mp
import numpy as np

from transitionlistener import generic_potential as gp
from transitionlistener import helper_functions
from transitionlistener.helper_functions import load_potential

REPO = Path(__file__).resolve().parents[1]
ABELIAN = (REPO / "models/TL_dark_U1_g_parameterization.py", "specific_potential",
           {"g": 1.0, "lambda": 0.03, "v_GeV": 0.1})
FLIPFLOP = (REPO / "models/TL_dark_flipflop.py", "DarkFlipFlop", {})


def build(spec, **overrides):
    path, cls, params = spec
    return load_potential(str(path), cls)(dict(params, **overrides), verbose=False)


class CountertermStepTests(unittest.TestCase):
    def test_counterterms_independent_of_tracing_accuracy(self):
        # The counterterm derivatives used to take the phase-tracing accuracy as
        # their finite-difference step, so tighter tracing changed the potential.
        def counterterms(**overrides):
            pot = build(ABELIAN, **overrides)
            return pot._dV1atvev, pot._d2V1atvev, pot.dl, pot.dmu2

        reference = counterterms()
        for overrides in ({"precision_mode": "benchmark"}, {"precision_mode": "xtrace"},
                          {"precision_mode": "robust"}, {"precision_trace_field_accuracy": 1e-5}):
            with self.subTest(**{k: str(v) for k, v in overrides.items()}):
                self.assertEqual(counterterms(**overrides), reference)

    def test_model_may_override_the_step(self):
        pot = build(ABELIAN)
        self.assertEqual(pot.counterterm_derivative_step, 1.0e-3)
        pot.counterterm_derivative_step = 0.1
        self.assertEqual(pot.counterterm_derivative_step, 0.1)


class RoundingOfTheCounterterms(unittest.TestCase):
    """The finite difference at the vev of a sum of order 1e9 is free of rounding, and the step still regulates."""

    @staticmethod
    def model(g=1.0, resummed=False, **attributes):
        cls = load_potential(str(REPO / "models/TL_dark_U1_g_parameterization.py"), "specific_potential")
        if not resummed:
            attributes["setGoldstoneShift"] = lambda self: None
        original = {name: getattr(cls, name) for name in attributes}
        for name, value in attributes.items():
            setattr(cls, name, value)
        try:
            return cls({"g": g, "lambda": 0.018012121212121212, "v_GeV": 0.1}, verbose=False)
        finally:
            for name, value in original.items():
                setattr(cls, name, value)

    def test_a_change_of_rounding_does_not_reach_the_counterterms(self):
        ref = self.model()
        for k in (-2, -1, 1, 2):
            other = self.model(1.0 + k * 1e-14)
            self.assertLess(abs(other.dmu2 / ref.dmu2 - 1), 1e-8)
            self.assertLess(abs(other.dl / ref.dl - 1), 1e-8)

    def test_the_second_derivative_is_the_stencil_in_exact_arithmetic(self):
        """The five-point stencil at the step 1e-3, in 60 digits; the heavy species agree with any step."""
        mp.mp.dps = 60
        g, lam, v = mp.mpf(1), mp.mpf("0.018012121212121212"), mp.mpf(1000)
        lam = mp.mpf(float(lam))
        mu2, Q2 = lam * v**2, v**2

        pot = self.model()
        _, dof, constants, _ = pot.boson_massSq(np.array([[1000.0]]), 0.0)

        def spectrum(phi):
            return [-mu2 + 3 * lam * phi**2, -mu2 + lam * phi**2, g**2 * phi**2, g**2 * phi**2]

        def V1(phi):
            return sum(mp.mpf(float(n)) * m2**2 * (mp.log(abs(m2 / Q2) + mp.mpf("1e-100")) - mp.mpf(float(c)))
                       for n, m2, c in zip(np.ravel(dof), spectrum(phi), np.ravel(constants))) / (64 * mp.pi**2)

        h = mp.mpf(float(1e-3))
        # The spectrum above is written out rather than taken from the model, so it has to be
        # pinned to the model. Without this the reference would drift silently.
        for phi in (v, v + 7 * h):
            np.testing.assert_allclose(
                np.ravel(pot.boson_massSq(np.array([[float(phi)]]), 0.0)[0]),
                [float(m2) for m2 in spectrum(phi)], rtol=1e-8, atol=1e-8,
                err_msg="the spectrum written out here is not the spectrum of the model any more")
        exact = (-V1(v - 2 * h) + 16 * V1(v - h) - 30 * V1(v) + 16 * V1(v + h) - V1(v + 2 * h)) / (12 * h**2)
        self.assertAlmostEqual(float(pot.d2V1atvev()) / float(exact), 1.0, delta=1e-9)

    def test_a_fermion_counts_once(self):
        """A fermion against the closed form of its second derivative, on top of the same bosons."""
        pot = self.model()
        without = float(pot.d2V1atvev())
        yukawa, dof = 0.3, 12.0
        pot.fermion_massSq = lambda X: ((yukawa * np.asarray(X)[..., :1])**2, np.array([dof]))
        del pot._d2V1atvev
        phi, Q2 = 1000.0, pot.renormScaleSq
        log = np.log(yukawa**2 * phi**2 / Q2)
        expected = -dof * yukawa**4 * phi**2 * (12.0 * (log - 1.5) + 14.0) / (64.0 * np.pi**2)
        self.assertAlmostEqual((float(pot.d2V1atvev()) - without) / expected, 1.0, delta=1e-9)

    def test_the_step_still_regulates_the_goldstone_boson(self):
        small, large = self.model(), self.model(counterterm_derivative_step=0.1)
        self.assertGreater(float(large.d2V1atvev()) - float(small.d2V1atvev()), 15.0)

    def test_a_species_at_the_step_of_the_model_when_its_mass_changes(self):
        pot = self.model(counterterm_derivative_step=1e-3)
        steps = []
        original = helper_functions.hessianFunction

        def spy(f, eps, Ndim, order):
            steps.append(eps)
            return original(f, eps=eps, Ndim=Ndim, order=order)

        del pot._d2V1atvev
        with mock.patch.object(helper_functions, "hessianFunction", spy):
            pot.d2V1atvev()
        self.assertEqual(steps.count(1e-3), 1)
        self.assertEqual(steps.count(1.0), len(steps) - 1)


# A spectrum written out here rather than taken from a model file, so that the field space has two
# dimensions whose vevs are a factor 200 apart and each species depends on one direction
# only. The one-loop potential is then separable, its Hessian at the vev is diagonal, and each
# diagonal entry is a one-dimensional stencil that 60-digit arithmetic reproduces exactly.
Q_SQ = 1.0e6              #: renormalisation scale squared, internal units
QUARTIC = 0.018           #: quartic coupling lambda of the first direction
VEV = np.array([1000.0, 5.0])   #: the two vacuum expectation values, internal field units
GAUGE = 1.0               #: gauge coupling g of the second direction
SEPARABLE_DOF = np.array([1.0, 1.0, 2.0])             #: degrees of freedom n_i
SEPARABLE_C = np.array([1.5, 1.5, 5.0 / 6.0])         #: renormalisation constants c_i


def separable_spectrum(X, T, dof=SEPARABLE_DOF):
    """Three species: one massive and one massless in direction 0, one massive in direction 1."""
    X = np.asanyarray(X)
    first, second = X[..., 0], X[..., 1]
    mu_sq = QUARTIC * VEV[0] ** 2
    masses_sq = np.stack([-mu_sq + 3 * QUARTIC * first ** 2,
                          QUARTIC * (first ** 2 - VEV[0] ** 2),
                          GAUGE ** 2 * second ** 2], axis=-1)
    return masses_sq, dof, SEPARABLE_C, np.array([True, False, True])


def two_field_potential(spectrum=separable_spectrum, vev=VEV, **overrides):
    """A namespace that answers what `_derivative_at_vev` asks of a potential, and nothing else."""
    pot = types.SimpleNamespace(
        v=vev, Ndim=np.size(vev), deriv_order=4, renormScaleSq=Q_SQ,
        counterterm_derivative_step=gp.generic_potential.counterterm_derivative_step,
        _SMOOTH_STEP_FRACTION=gp.generic_potential._SMOOTH_STEP_FRACTION,
        goldstone_shift=gp.generic_potential.goldstone_shift,
        boson_massSq=spectrum,
        fermion_massSq=lambda X: (np.zeros(np.shape(X)[:-1] + (0,)), np.zeros(0)))
    for name, value in overrides.items():
        setattr(pot, name, value)
    pot.V1 = types.MethodType(gp.generic_potential.V1, pot)
    return pot


def record_steps(pot, derivative=helper_functions.hessianFunction):
    """Return the derivative at the vev and the step handed to `derivative` for each species."""
    steps = []

    def spy(f, eps, Ndim, order):
        steps.append(np.array(eps, dtype=float).tolist())
        return derivative(f, eps=eps, Ndim=Ndim, order=order)

    return gp.generic_potential._derivative_at_vev(pot, pot.V1, spy), steps


class TheStepOfEachFieldDirection(unittest.TestCase):
    """Each direction is differenced at a fraction of its own vev, not of the largest vev."""

    def test_each_direction_takes_the_fraction_of_its_own_vev(self):
        """With vevs 1000.0 and 5.0 the smooth species take steps 1.0 and 0.005 in internal units.

        One common step would be 1.0 in both directions, which is 20 per cent of the second vev.
        """
        _, steps = record_steps(two_field_potential())
        smooth = [s for s in steps if isinstance(s, list)]
        self.assertEqual(len(smooth), 2, "the two species that are smooth at the vev")
        for step in smooth:
            self.assertEqual(step, [1.0, 0.005])
        self.assertEqual([s for s in steps if not isinstance(s, list)], [1.0e-3],
                         "the massless species keeps the model step, as a single number")

    def test_the_hessian_is_the_stencil_at_those_steps_in_exact_arithmetic(self):
        """The two-dimensional Hessian against 60-digit arithmetic, species by species.

        The entry for the first direction agrees to 1e-9 in relative terms. The entry for the
        second agrees to 1e-4: the first two species do not depend on the second direction, so
        their stencils in that direction cancel term by term, and cancelling numbers of order 1e7
        against a step of 0.005 leaves about 1e-5 in internal units out of 9.75. The same
        round-off is why this comparison cannot tell a per-direction step from one common step,
        which the test above is for.
        """
        hessian, steps = record_steps(two_field_potential())
        mp.mp.dps = 60
        lam, g = mp.mpf(float(QUARTIC)), mp.mpf(float(GAUGE))
        first, second, scale_sq = mp.mpf(float(VEV[0])), mp.mpf(float(VEV[1])), mp.mpf(float(Q_SQ))
        mu_sq = lam * first ** 2

        def term(mass_sq, dof, constant):
            return (mp.mpf(float(dof)) * mass_sq ** 2
                    * (mp.log(abs(mass_sq / scale_sq) + mp.mpf("1e-100")) - mp.mpf(float(constant)))
                    / (64 * mp.pi ** 2))

        species = [lambda x: term(-mu_sq + 3 * lam * x ** 2, SEPARABLE_DOF[0], SEPARABLE_C[0]),
                   lambda x: term(lam * (x ** 2 - first ** 2), SEPARABLE_DOF[1], SEPARABLE_C[1]),
                   lambda x: term(g * g * x ** 2, SEPARABLE_DOF[2], SEPARABLE_C[2])]

        def second_derivative(f, x, step):
            h = mp.mpf(float(step))
            return (-f(x - 2 * h) + 16 * f(x - h) - 30 * f(x) + 16 * f(x + h) - f(x + 2 * h)) / (12 * h * h)

        def step_of(index, direction):
            return steps[index][direction] if isinstance(steps[index], list) else steps[index]

        exact_first = (second_derivative(species[0], first, step_of(0, 0))
                       + second_derivative(species[1], first, step_of(1, 0)))
        exact_second = second_derivative(species[2], second, step_of(2, 1))
        self.assertEqual(hessian.shape, (2, 2))
        self.assertAlmostEqual(float(hessian[0, 0]) / float(exact_first), 1.0, delta=1e-9)
        self.assertAlmostEqual(float(hessian[1, 1]) / float(exact_second), 1.0, delta=1e-4)
        for off in (hessian[0, 1], hessian[1, 0]):
            self.assertLess(abs(off), 1e-6 * abs(float(hessian[0, 0])),
                            "a separable potential has no mixed second derivative")


class EverySpeciesIsInTheSum(unittest.TestCase):
    """The sum runs over the mass eigenvalues, whatever shape the degrees of freedom have."""

    def test_degrees_of_freedom_as_one_number_for_all_species(self):
        """A model may return n_i as a single number. Counting species from it would keep one.

        With n_i = 1 for all three species the sum has to be the same whether the model says so
        with the number 1.0 or with an array of three ones.
        """
        def hessian(dof):
            pot = two_field_potential(spectrum=lambda X, T: separable_spectrum(X, T, dof=dof))
            return gp.generic_potential._derivative_at_vev(
                pot, pot.V1, helper_functions.hessianFunction)

        from_array, from_scalar = hessian(np.ones(3)), hessian(1.0)
        np.testing.assert_allclose(from_scalar, from_array, rtol=1e-12, atol=0.0)
        self.assertGreater(abs(float(from_array[1, 1])), 1.0,
                           "the third species, which is the one at risk of being dropped, is not zero")

    def test_the_species_sum_is_the_derivative_of_the_whole_potential(self):
        """At one step for every species the sum over species is the derivative of the sum.

        `_SMOOTH_STEP_FRACTION` of 0.0 puts every species on `counterterm_derivative_step`, here
        1.0 in internal field units, which is large enough that differencing the whole potential at
        once is not limited by round-off. The two then have to agree to 1e-10 in relative terms.
        """
        pot = two_field_potential(_SMOOTH_STEP_FRACTION=0.0, counterterm_derivative_step=1.0)
        by_species = gp.generic_potential._derivative_at_vev(
            pot, pot.V1, helper_functions.hessianFunction)
        whole = helper_functions.hessianFunction(
            lambda X: pot.V1(pot.boson_massSq(X, 0.0), pot.fermion_massSq(X)),
            eps=1.0, Ndim=2, order=4)(np.array([pot.v]))
        np.testing.assert_allclose(by_species, np.squeeze(whole), rtol=1e-10, atol=1e-6)


class TheStepAtAVanishingVev(unittest.TestCase):
    """A vev of zero cannot be allowed to produce a step of zero."""

    @staticmethod
    def conformal_spectrum(X, T):
        """No quadratic term, so every mass squared vanishes at the origin."""
        X = np.asanyarray(X)
        first, second = X[..., 0], X[..., 1]
        masses_sq = np.stack([3 * QUARTIC * first ** 2, QUARTIC * first ** 2,
                              GAUGE ** 2 * second ** 2], axis=-1)
        return masses_sq, SEPARABLE_DOF, SEPARABLE_C, np.array([True, False, True])

    def test_a_vanishing_vev_keeps_the_model_step(self):
        pot = two_field_potential(spectrum=self.conformal_spectrum, vev=np.array([0.0, 0.0]))
        hessian, steps = record_steps(pot)
        self.assertTrue(np.all(np.isfinite(hessian)), f"the Hessian at the origin is {hessian}")
        for step in steps:
            np.testing.assert_allclose(step, 1.0e-3, rtol=0.0, atol=0.0)

    def test_one_vanishing_direction_keeps_the_model_step_in_that_direction(self):
        pot = two_field_potential(spectrum=self.conformal_spectrum, vev=np.array([1000.0, 0.0]))
        hessian, steps = record_steps(pot)
        self.assertTrue(np.all(np.isfinite(hessian)), f"the Hessian is {hessian}")
        self.assertIn([1.0, 1.0e-3], steps,
                      "the direction with a vev of zero takes the model step, the other 1.0")


class TheFermionSpectrumIsThePairTheModelReturns(unittest.TestCase):
    """`V1phys` reads the fermion spectrum the way `fermion_massSq` returns it."""

    def test_v1phys_takes_the_tuple_fermion_masssq_returns(self):
        """`V1phys` read two attributes off that tuple, so every call to it raised AttributeError."""
        pot = build(ABELIAN)
        X = np.array([[pot.v]], dtype=float)
        value = float(np.squeeze(pot.V1phys(pot.boson_massSq(X, 0.0), pot.fermion_massSq(X))))
        self.assertTrue(np.isfinite(value))
        masses_sq, dof, constants, physical = pot.boson_massSq(X, 0.0)
        by_hand = float(np.squeeze(np.sum(
            dof * masses_sq ** 2 * physical
            * (np.log(np.abs(masses_sq / pot.renormScaleSq) + 1e-100) - constants), axis=-1)
            / (64 * np.pi ** 2)))
        self.assertAlmostEqual(value / by_hand, 1.0, delta=1e-12)

    def test_the_first_derivative_without_the_goldstone_boson_is_finite(self):
        pot = build(ABELIAN)
        self.assertTrue(np.isfinite(float(np.squeeze(pot.dV1physatvev()))))


class AModelThatKeepsOneNumberForTwoFieldDirections(unittest.TestCase):
    """`models/TL_dark_flipflop.py` has `Ndim` of 2 and keeps its vev as a single number.

    The mass function of that model indexes both field directions, so a field point built from
    the shape the vev happens to be stored in carries one component and the spectrum raises
    `IndexError`. On the base commit c81ebf6f `dV1atvev` returned a two-component gradient, since
    the derivative helper broadcast that one-component point against its own stencil before the
    spectrum saw it, while `d2V1atvev` raised `IndexError` and `dV1physatvev` raised
    `AttributeError`. Differencing species by species evaluates the spectrum at the field point
    itself, which turned the first of the three into an `IndexError` as well. All three return a
    finite result of the model's own dimension now, the second derivative for the first time.
    """

    def setUp(self):
        self.pot = build(FLIPFLOP)

    def test_the_model_is_the_one_this_is_about(self):
        """Without this the three tests below would pass on any model at all."""
        self.assertEqual(self.pot.Ndim, 2)
        self.assertEqual(np.size(self.pot.v), 1, "the vev is kept as a single number")

    def test_the_gradient_carries_one_component_per_field_direction(self):
        gradient = np.asarray(self.pot.dV1atvev(), dtype=float)
        self.assertEqual(gradient.shape, (2,))
        self.assertTrue(np.all(np.isfinite(gradient)), f"the gradient is {gradient}")

    def test_the_hessian_carries_one_row_and_column_per_field_direction(self):
        hessian = np.asarray(self.pot.d2V1atvev(), dtype=float)
        self.assertEqual(hessian.shape, (2, 2))
        self.assertTrue(np.all(np.isfinite(hessian)), f"the Hessian is {hessian}")

    def test_the_gradient_without_the_goldstone_bosons_carries_the_same_components(self):
        gradient = np.asarray(self.pot.dV1physatvev(), dtype=float)
        self.assertEqual(gradient.shape, (2,))
        self.assertTrue(np.all(np.isfinite(gradient)), f"the gradient is {gradient}")


class GoldstoneResummation(unittest.TestCase):
    """The shift of arXiv:1406.2652 in the Coleman-Weinberg term of a model with counterterms at the vev."""

    model = staticmethod(lambda **k: RoundingOfTheCounterterms.model(resummed=True, **k))

    def test_the_shift_is_the_physical_derivative_of_the_one_loop_potential(self):
        pot = self.model()
        phi, Q2, lam, g = 1000.0, pot.renormScaleSq, pot.l, pot.g
        _, dof, constants, _ = pot.boson_massSq(np.array([[phi]]), 0.0)
        dof, constants = np.ravel(dof), np.ravel(constants)
        species = [(dof[0], 2.0 * lam * phi**2, 6.0 * lam * phi, constants[0]),
                   (dof[2], g**2 * phi**2, 2.0 * g**2 * phi, constants[2]),
                   (dof[3], g**2 * phi**2, 2.0 * g**2 * phi, constants[3])]
        slope = sum(n * m2 * dm2 * (2.0 * (np.log(m2 / Q2) - c) + 1.0) for n, m2, dm2, c in species)
        self.assertAlmostEqual(pot.goldstone_shift / (slope / (64.0 * np.pi**2) / phi), 1.0, delta=1e-9)

    def test_the_step_no_longer_regulates(self):
        values = [self.model(counterterm_derivative_step=step).dmu2 for step in (1e-5, 1e-3, 1e-1)]
        self.assertAlmostEqual(values[0] / values[2], 1.0, delta=1e-9)
        self.assertAlmostEqual(values[1] / values[2], 1.0, delta=1e-9)

    def test_the_shift_stays_out_of_the_thermal_masses(self):
        pot = self.model()
        X = np.array([[300.0]])
        m2 = np.ravel(pot.boson_massSq(X, 0.0)[0])
        self.assertAlmostEqual(m2[1], -pot.mu2 + pot.l * 300.0**2, delta=1e-9)

    def test_the_vev_and_the_scalar_mass_are_those_of_the_tree_level(self):
        pot = self.model()
        vev, T = np.array([1000.0]), 1e-6
        self.assertLess(abs(float(np.ravel(pot.gradV(vev, T))[0])), 1e-6 * pot.mDH**2 * 1000.0)
        self.assertAlmostEqual(float(np.ravel(pot.d2V(vev, T))[0]) / pot.mDH**2, 1.0, delta=1e-3)


if __name__ == "__main__":
    unittest.main()
