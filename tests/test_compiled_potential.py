"""The compiled potential of a one-field model: what compiles, what does not, and that it follows the model.

The reference is the generic numpy code of generic_potential, which stays as the potential of every model outside the
compiled form.
"""
import copy
import os
import sys
from unittest import mock

import numpy as np
import pytest

from transitionlistener import bubbledynamics as bd, errors, generic_potential as gp, phases
from transitionlistener.helper_functions import load_potential
from transitionlistener.particles import Fermion, MassSpectrum, Scalar

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
if MODELS not in sys.path:
    sys.path.insert(0, MODELS)     # the 3D model files import their 4D parents by name


def python_path(pot):
    """The same model on the generic numpy code."""
    pot.current_kernels()
    plain = copy.copy(pot)
    plain.__dict__.update(_kernels=None, _kernel_stamp=None)
    for name in ("_gradV", "_d2V", "_gradVT"):
        plain.__dict__.pop(name, None)
    return plain


class Singlet(gp.generic_potential):
    """A real singlet with a tree-level mass and cubic term, a Yukawa fermion with a mass of its own, and a counterterm.

    V0 = mu^2 phi^2 / 2 - c phi^3 / 3 + l phi^4 / 4, m_b^2 = mu^2 - 2 c phi + 3 l phi^2 + (l/4 + y^2/6) T^2,
    m_f = m + y phi."""

    model_parameters = {"y": {"default": 0.8, "min": 0, "max": 10}, "v_GeV": {"default": 100.0, "min": 0, "max": np.inf}}

    def init(self, inp, verbose=False):
        self.mp = self.set_modelparams(inp)
        self.Ndim = 1
        self.verbose = verbose
        self.y, self.v_GeV = self.mp["y"]["value"], self.mp["v_GeV"]["value"]
        self.X0 = np.array([1000.0])
        self.v = 1000.0
        self.daisy = "ArnoldEspinosa"
        self.conversionFactor = self.computeConversionFactor(1000.0, self.v_GeV)
        self.renormScaleSq = 1e6
        self.mu2, self.c, self.l, self.m = 2e4, 80.0, 0.1, 40.0
        self.delta = 3e-3         # a counterterm of the form of the quartic and the mass
        self.mass_spectrum = self.spectrum()

    def bosons(self, X, T):
        phi = np.asarray(X)[..., 0]
        m2 = self.mu2 - 2.0 * self.c * phi + 3.0 * self.l * phi * phi
        return (m2 + (self.l / 4.0 + self.y**2 / 6.0) * T * T)[..., np.newaxis]

    def fermions(self, X):
        return ((self.m + self.y * np.asarray(X)[..., 0]) ** 2)[..., np.newaxis]

    def spectrum(self):
        return MassSpectrum([Scalar(name="m_h", latex_name="m_h", dof=1, is_SM=False)], [],
                            [Fermion(name="m_f", latex_name="m_f", dof=4, is_SM=False)],
                            boson_massSq_fn=self.bosons, fermion_massSq_fn=self.fermions)

    def V0(self, X):
        phi = np.asanyarray(X)[..., 0]
        return self.mu2 * phi**2 / 2.0 - self.c * phi**3 / 3.0 + self.l * phi**4 / 4.0

    def Vct(self, X):
        phi = np.asanyarray(X)[..., 0]
        return self.delta * (phi**4 / 4.0 - self.mu2 * phi**2 / self.l)


class NonPolynomial(Singlet):
    """A fermion mass that is not polynomial in the field: it saturates."""

    def fermions(self, X):
        phi = np.asarray(X)[..., 0]
        return (self.y**2 * phi * phi / (1.0 + (phi / 800.0) ** 2))[..., np.newaxis]


def toy(cls=Singlet, **params):
    return cls(dict({"y": 0.8, "v_GeV": 100.0}, **params), verbose=False)


FIELDS = np.array([[0.0], [1.0], [300.0], [1000.0], [-200.0], [1500.0]])
TEMPERATURES = [1e-3, 1.0, 30.0, 300.0, 1500.0]


def assert_same_potential(pot, rel=1e-9):
    """Vtot, its gradient and second derivative, the temperature derivative and the energy density, against numpy.
    A finite difference has the rounding of the potential over its step as an absolute error."""
    plain = python_path(pot)
    for T in TEMPERATURES:
        for radiation in (True, False):
            a, b = pot.Vtot(FIELDS, T, radiation), plain.Vtot(FIELDS, T, radiation)
            assert a == pytest.approx(b, rel=rel, abs=1e-6)
        rounding = 2e-15 * float(np.max(np.abs(pot.Vtot(FIELDS, T, False))))
        step = float(pot.x_eps)
        assert pot.gradV(FIELDS, T) == pytest.approx(plain.gradV(FIELDS, T), rel=1e-6, abs=rounding / step)
        for x in FIELDS[1:]:
            assert pot.energyDensity(x, float(T)) == pytest.approx(plain.energyDensity(x, float(T)), rel=rel)
            assert pot.d2V(x, T) == pytest.approx(plain.d2V(x, T), rel=1e-6, abs=8 * rounding / step**2)
            assert pot.dgradV_dT(x, T) == pytest.approx(plain.dgradV_dT(x, T), rel=1e-5,
                                                        abs=rounding / (step * pot.T_eps))


# ------------------------------------------------------------------------------------------ what compiles
def test_a_toy_model_compiles_and_matches_its_python_potential():
    pot = toy()
    assert pot.compiled_status == "compiled"
    assert_same_potential(pot)


@pytest.mark.parametrize("daisy", ["ArnoldEspinosa", "Parwani", "off"])
def test_every_daisy_scheme(daisy):
    pot = toy()
    pot.daisy = daisy
    assert pot.compiled_status == "compiled"
    assert_same_potential(pot)


@pytest.mark.parametrize("deriv_order", [2, 4])
def test_both_stencils_of_the_derivatives(deriv_order):
    pot = toy()
    pot.deriv_order = deriv_order
    assert_same_potential(pot)


def test_array_temperatures_and_shapes_of_a_field_array():
    pot, X = toy(), np.linspace(1.0, 900.0, 12).reshape(3, 4, 1)
    T = np.linspace(1.0, 400.0, 4)
    plain = python_path(pot)
    for args in ((X, 50.0), (X, T), (X[0, 0], T), (X[0], T)):
        a, b = pot.Vtot(*args), plain.Vtot(*args)
        assert np.shape(a) == np.shape(b) and a == pytest.approx(b, rel=1e-10)
    assert pot.dgradV_dT(X, T).shape == plain.dgradV_dT(X, T).shape == X.shape


def test_tl_conformal_u1_fermions_compiles():
    pot = load_potential(os.path.join(MODELS, "TL_dark_U1.py"), "specific_potential")(
        dict(g_tilde=1.0, l=0.1, v_GeV=100.0), verbose=False)
    assert pot.compiled_status == "compiled"
    assert_same_potential(pot, rel=1e-8)


def test_the_conformal_models_compile_also_without_the_loader():
    import TL_conformal_u1_fermions as model
    pot = model.specific_potential(dict(g=0.7, y=0.3, v_GeV=2.0), verbose=False)
    assert pot.compiled_status == "compiled"
    assert_same_potential(pot)


def test_zero_yukawa_is_finite():
    """The model drops the fermion term at y = 0, where it is 0 log(0/0); the kernel must as well."""
    import TL_conformal_u1_fermions as model
    pot = model.specific_potential(dict(g=0.4, y=0.0, v_GeV=2.95), verbose=False)
    assert pot.compiled_status == "compiled"
    assert_same_potential(pot)


# ------------------------------------------------------------------------------------------ what does not compile
def test_a_non_polynomial_mass_stays_on_the_python_path_and_is_right():
    pot = toy(NonPolynomial)
    assert pot.compiled_status != "compiled" and "compiled form" in pot.compiled_status
    assert pot.current_kernels() is None
    x = np.array([500.0])
    m2 = 0.8**2 * 500.0**2 / (1.0 + (500.0 / 800.0) ** 2)
    assert pot.fermion_massSq(x)[0][0] == pytest.approx(m2)
    # the derivative of the potential at a field where the saturating mass matters, against a step of the potential
    h = 1e-2
    numeric = (pot.Vtot(x + h, 30.0, False) - pot.Vtot(x - h, 30.0, False)) / (2 * h)
    assert pot.gradV(x, 30.0)[0] == pytest.approx(numeric, rel=1e-6)
    assert np.isfinite(pot.energyDensity(x, 30.0))


class OverridesVtot(Singlet):
    def Vtot(self, X, T, include_radiation=True, include_decoupled=True):
        return super().Vtot(X, T, include_radiation, include_decoupled) + 7.0


class OverridesV1T(Singlet):
    def V1T(self, bosons, fermions, T):
        return 1.5 * super().V1T(bosons, fermions, T)


class OverridesEnergy(Singlet):
    def energyDensity(self, X, T, include_decoupled=True):
        return 11.0


@pytest.mark.parametrize("cls, entry", [(OverridesVtot, "Vtot"), (OverridesV1T, "V1T")])
def test_a_model_replacing_a_piece_of_the_potential_keeps_it(cls, entry):
    pot = toy(cls)
    assert pot.compiled_status == f"it replaces {entry}"
    plain = python_path(toy())
    x = np.array([500.0])
    if entry == "Vtot":
        assert pot.Vtot(x, 30.0) == pytest.approx(plain.Vtot(x, 30.0) + 7.0)
    else:
        difference = pot.Vtot(x, 30.0, False) - plain.Vtot(x, 30.0, False)
        assert difference == pytest.approx(0.5 * plain.V1T_from_X(x, 30.0), rel=1e-9)


class OverridesV1TFromX(Singlet):
    def V1T_from_X(self, X, T):
        return 1.5 * super().V1T_from_X(X, T)


class OverridesThermalToo(OverridesVtot):
    """The documented way to rewrite the potential: Vtot and its thermal part."""

    def V_thermal(self, X, T, include_decoupled=False):
        return generic_thermal(self, X, T, include_decoupled)


def generic_thermal(pot, X, T, include_decoupled):
    plain = python_path(toy())
    return gp.generic_potential.V_thermal(plain, X, T, include_decoupled)


@pytest.mark.parametrize("cls", [OverridesVtot, OverridesV1TFromX, OverridesThermalToo, OverridesEnergy])
def test_the_enthalpy_of_a_rewritten_model_is_e_plus_p_of_that_model(cls):
    """V_thermal refuses a rewritten potential, and its thermal part would not be this model's."""
    pot = toy(cls)
    x, T = np.zeros(1), 300.0
    vacuum = pot.V0(pot.X0) + pot.Vct(pot.X0) + pot.V1_from_X(pot.X0)
    for decoupled in (True, False):
        expected = (pot.energyDensity(x, T, include_decoupled=decoupled)
                    - (pot.Vtot(x, T, include_decoupled=decoupled) - vacuum))
        assert pot.enthalpyDensity(x, T, include_decoupled=decoupled) == pytest.approx(expected, rel=1e-12)


def test_an_instance_patch_of_the_potential_reaches_the_enthalpy():
    pot = toy()
    pot.Vtot = lambda X, T, include_radiation=True, include_decoupled=True: 5.0
    vacuum = pot.V0(pot.X0) + pot.Vct(pot.X0) + pot.V1_from_X(pot.X0)
    x = np.zeros(1)
    assert pot.enthalpyDensity(x, 300.0) == pytest.approx(pot.energyDensity(x, 300.0) - (5.0 - vacuum), rel=1e-12)


def test_the_strengths_of_a_model_that_rewrites_the_potential_and_its_thermal_part():
    """calcAlphas end to end: its enthalpy goes through the rewritten model."""
    pot = toy(OverridesThermalToo)
    assert pot.compiled_status.endswith("replaces Vtot, V_thermal")
    T = np.geomspace(1.0, 600.0, 60)
    field = 1000.0 * np.sqrt(np.maximum(1.0 - (T / 700.0) ** 2, 1e-3))
    broken = phases.PhaseInfo(0, field[:, None], T, 0.0 * field[:, None])
    symmetric = phases.PhaseInfo(1, 0.0 * field[:, None], T, 0.0 * field[:, None])
    assert all(np.isfinite(value) for value in bd.calcAlphas(100.0, pot, symmetric, broken))


def test_a_replaced_energy_density_keeps_the_potential_compiled_but_not_the_thermodynamics():
    pot = toy(OverridesEnergy)
    assert pot.compiled_status == "compiled" and pot.thermal_kernels() is None
    assert pot.energyDensity(np.array([500.0]), 30.0) == 11.0


def test_a_daisy_scheme_without_a_kernel_keeps_the_python_potential():
    pot = toy()
    pot.daisy = "none of these"
    assert "daisy" in pot.compiled_status and pot.current_kernels() is None


class Failing(Singlet):
    """Refuses a field value that only the compilation asks for."""

    def V0(self, X):
        if np.ndim(X) == 2 and np.shape(X)[0] == 7:
            raise self.failure
        return super().V0(X)


def test_a_model_that_fails_on_the_samples_stays_on_the_python_path():
    Failing.failure = ZeroDivisionError("seven points")
    pot = toy(Failing)
    assert pot.compiled_status == "seven points" and pot.current_kernels() is None
    assert np.isfinite(pot.Vtot(np.array([300.0]), 30.0))


def test_the_run_deadline_is_not_read_as_a_refusal():
    pot = toy(Failing)
    Failing.failure = errors.Timeout("deadline")
    pot.g_of_this_test = 1.0
    with pytest.raises(errors.Timeout):
        pot.Vtot(np.array([300.0]), 30.0)
    Failing.failure = ZeroDivisionError("seven points")


def test_a_refusal_is_not_repeated_forever():
    """A model that assigns to itself in its Python potential must not rebuild the kernels at every call."""
    class Restless(Failing):
        def boson_massSq(self, X, T):
            self.last_field = X
            return super().boson_massSq(X, T)

    Failing.failure = ZeroDivisionError("seven points")
    pot = toy(Restless)
    with mock.patch.object(gp.onefield_potential, "build", wraps=gp.onefield_potential.build) as build:
        for _ in range(3 * gp._MAX_KERNEL_FAILURES):
            pot.Vtot(np.array([300.0]), 30.0)
    assert build.call_count <= gp._MAX_KERNEL_FAILURES


# ------------------------------------------------------------------------------------------ stale kernels
X, T_TEST = np.array([500.0]), 30.0


def value(pot):
    return pot.Vtot(X, T_TEST)


def test_changing_a_coupling_after_init_reads_the_new_value():
    pot = toy()
    before = value(pot)
    pot.y = 1.3
    assert pot.compiled_status == "compiled"
    assert value(pot) != pytest.approx(before, rel=1e-6)
    assert value(pot) == pytest.approx(python_path(pot).Vtot(X, T_TEST), rel=1e-10)
    assert_same_potential(pot)


def test_changing_the_daisy_scheme_after_init():
    pot = toy()
    for daisy in ("off", "Parwani", "ArnoldEspinosa"):
        pot.daisy = daisy
        assert value(pot) == pytest.approx(python_path(pot).Vtot(X, T_TEST), rel=1e-10)


def test_set_modelparams_changes_the_parameters_in_place():
    class ReadsTheDict(Singlet):
        def bosons(self, X, T):
            return super().bosons(X, T) * self.model_parameters["y"]["value"]

    pot = toy(ReadsTheDict)
    assert pot.compiled_status == "compiled"
    pot.set_modelparams({"y": 0.1})
    assert value(pot) == pytest.approx(python_path(pot).Vtot(X, T_TEST), rel=1e-10)


def test_a_patch_on_the_class_and_its_removal():
    pot = toy()
    plain = python_path(pot)
    honest = value(pot)
    with mock.patch.object(Singlet, "V1T", lambda self, b, f, T: 0.0 * T):
        assert pot.compiled_status.startswith("it replaces V1T")
        assert value(pot) == pytest.approx(plain.Vtot(X, T_TEST), rel=1e-12) and value(pot) != honest
    assert pot.compiled_status == "compiled" and value(pot) == pytest.approx(honest, rel=1e-12)


def test_a_patch_on_a_class_attribute_a_kernel_reads_through_the_model():
    class Fixed(Singlet):
        extra = 0.0

        def V0(self, X):
            return super().V0(X) + self.extra * np.asanyarray(X)[..., 0] ** 2

    pot = toy(Fixed)
    before = value(pot)
    Fixed.extra = 10.0
    assert value(pot) == pytest.approx(python_path(pot).Vtot(X, T_TEST), rel=1e-10)
    assert value(pot) != before


def test_a_patch_on_the_instance_and_its_removal():
    pot = toy()
    honest = value(pot)
    pot.Vtot = lambda X, T, *a, **k: 3.5
    assert pot.Vtot(X, T_TEST) == 3.5 and "replaces Vtot" in pot.compiled_status and pot.current_kernels() is None
    del pot.Vtot
    assert pot.compiled_status == "compiled" and value(pot) == honest


def test_a_replaced_mass_function():
    pot = toy()
    pot.fermions = lambda X: ((1.0 + 0.5 * np.asarray(X)[..., 0]) ** 2)[..., np.newaxis]
    pot.mass_spectrum = pot.spectrum()
    assert pot.compiled_status == "compiled"
    assert value(pot) == pytest.approx(python_path(pot).Vtot(X, T_TEST), rel=1e-10)


def test_a_changed_bath_reaches_the_energy_density():
    pot = toy()
    pot.kin_coupled_e_geff = lambda T, cf: 3.0 + 0.0 * T
    pot.kin_coupled_p_geff = lambda T, cf: 3.0 + 0.0 * T
    pot.kin_decoupled_e_geff = lambda T, cf: 5.0 + 0.0 * T
    pot.kin_decoupled_p_geff = lambda T, cf: 5.0 + 0.0 * T
    x = np.array([500.0])
    plain = python_path(pot)
    for decoupled in (True, False):
        assert pot.energyDensity(x, T_TEST, decoupled) == pytest.approx(plain.energyDensity(x, T_TEST, decoupled),
                                                                        rel=1e-10)
    assert pot.energyDensity(x, T_TEST, True) != pot.energyDensity(x, T_TEST, False)
    assert value(pot) == pytest.approx(python_path(pot).Vtot(X, T_TEST), rel=1e-10)


def test_standard_model_fields_leave_the_thermodynamics_to_numpy():
    class WithStandardModelFields(Singlet):
        def spectrum(self):
            return MassSpectrum([Scalar(name="m_h", latex_name="m_h", dof=1, is_SM=True)], [],
                                [Fermion(name="m_f", latex_name="m_f", dof=4, is_SM=False)],
                                boson_massSq_fn=self.bosons, fermion_massSq_fn=self.fermions)

    pot = toy(WithStandardModelFields)
    assert pot.compiled_status == "compiled" and pot.thermal_kernels() is None


def test_bookkeeping_does_not_rebuild_the_kernels():
    pot = toy()
    kernels = pot.current_kernels()
    pot.config = pot.config
    pot.verbose = False
    pot.nlo_correction = None
    pot._gradV = None
    del pot._gradV
    assert pot.current_kernels() is kernels


def test_a_copy_follows_its_own_changes():
    pot = toy()
    twin = copy.copy(pot)
    honest = value(pot)
    twin.y = 2.0
    assert value(pot) == honest and value(twin) == pytest.approx(python_path(twin).Vtot(X, T_TEST), rel=1e-10)
    assert copy.deepcopy(pot).compiled_status == "compiled"


# ------------------------------------------------------------------------------------------ minimisers and thermodynamics
def test_the_newton_minimiser_goes_through_the_kernels_and_finds_the_python_minimum():
    pot = toy()
    plain = python_path(pot)
    kernels = pot.current_kernels()
    for T in (1.0, 30.0, 150.0):
        with mock.patch.object(kernels, "newton_minimum", wraps=kernels.newton_minimum) as spy:
            found = phases.findLocalMinimum(np.array([950.0]), T, pot.gradV, pot.d2V)
        assert spy.call_count == 1
        reference = phases.findLocalMinimum(np.array([950.0]), T, plain.gradV, plain.d2V)
        assert found == pytest.approx(reference, rel=1e-5)


def test_other_derivatives_keep_the_loop_in_python():
    pot = toy()
    kernels = pot.current_kernels()
    with mock.patch.object(kernels, "newton_minimum") as spy:
        found = phases.findLocalMinimum(np.array([1.0]), 0.0, lambda x, T: 2.0 * (x - 3.0),
                                        lambda x, T: np.array([[2.0]]))
        assert not spy.called and found[0] == pytest.approx(3.0, rel=1e-5)
        phases.findLocalMinimum(np.array([900.0]), 30.0, pot.dgradV_dT, pot.d2V)
        assert not spy.called, "a method of the potential that is not its gradient"


def phase(pot):
    T = np.geomspace(0.5, 400.0, 80)
    field = 1000.0 * np.sqrt(np.maximum(1.0 - (T / 700.0) ** 2, 1e-3))
    return phases.PhaseInfo(0, field[:, None], T, 0.0 * field[:, None])


def test_the_thermodynamics_along_a_phase_against_numpy():
    pot = toy()
    plain = python_path(pot)
    on_phase = phase(pot)
    assert pot.thermal_kernels() is not None and plain.thermal_kernels() is None
    temperatures = np.geomspace(0.7, 390.0, 40)
    for t in map(float, temperatures):
        for decoupled in (True, False):
            assert bd.energyDensity(pot, on_phase, t, decoupled) == pytest.approx(
                bd.energyDensity(plain, on_phase, t, decoupled), rel=1e-9)
        assert bd.h_eff_DS(t, pot, on_phase) == pytest.approx(bd.h_eff_DS(t, plain, on_phase), rel=1e-9)
        assert bd.h_eff_coupled_radiation(t, pot) == pytest.approx(bd.h_eff_coupled_radiation(t, plain), rel=1e-9)
        assert bd.entropy_density(pot, on_phase, t) == pytest.approx(bd.entropy_density(plain, on_phase, t), rel=1e-9)
    a = bd._time_temperature_factors(pot, on_phase, temperatures[::-1], "sound_speed", "dof_table")
    b = bd._time_temperature_factors(plain, on_phase, temperatures[::-1], "sound_speed", "dof_table")
    assert a[0] == pytest.approx(b[0], rel=1e-6) and a[1] == pytest.approx(b[1], rel=1e-9)
    usable = [[], []]
    bd._time_temperature_factors(pot, on_phase, temperatures[::-1], usable_out=usable[0])
    bd._time_temperature_factors(plain, on_phase, temperatures[::-1], usable_out=usable[1])
    assert usable[0] == usable[1] and usable[0][0] > 30


def test_a_replaced_trace_is_read_again():
    pot = toy()
    plain = python_path(pot)
    on_phase = phase(pot)
    bd.energyDensity(pot, on_phase, 30.0)
    on_phase.replaceTrace(1.01 * on_phase.X, on_phase.dXdT)
    assert bd.energyDensity(pot, on_phase, 30.0) == pytest.approx(bd.energyDensity(plain, on_phase, 30.0), rel=1e-9)
    assert bd.h_eff_DS(30.0, pot, on_phase) == pytest.approx(bd.h_eff_DS(30.0, plain, on_phase), rel=1e-9)


def test_a_phase_of_another_kind_takes_the_python_path():
    class Fixed:
        def valAt(self, T):
            return np.array([500.0])

    class FixedTrace(phases.PhaseInfo):
        def valAt(self, T, deriv=0):
            return np.array([500.0])

    pot = toy()
    traced = phase(pot)
    fixed = FixedTrace(0, traced.X, traced.T, traced.dXdT)
    assert bd.energyDensity(pot, fixed, 30.0) == pytest.approx(pot.energyDensity(np.array([500.0]), 30.0), rel=1e-12)
    traced.valAt = fixed.valAt
    assert bd.energyDensity(pot, traced, 30.0) == pytest.approx(pot.energyDensity(np.array([500.0]), 30.0), rel=1e-12)
    assert bd.energyDensity(pot, Fixed(), 30.0) == pytest.approx(pot.energyDensity(np.array([500.0]), 30.0), rel=1e-12)
    assert bd.h_eff_DS(30.0, pot, Fixed()) == pytest.approx(bd.h_eff_DS(30.0, python_path(pot), Fixed()), rel=1e-12)


def test_the_entropy_of_a_bath_that_is_not_the_table():
    pot = toy()
    pot.kin_coupled_e_geff = lambda T, cf: 3.0 + 0.0 * T
    pot.kin_coupled_p_geff = lambda T, cf: 3.0 + 0.0 * T
    assert not pot.thermal_kernels().thermal["sm_bath"]
    assert bd.h_eff_coupled_radiation(30.0, pot) == pytest.approx(3.0, rel=1e-12)


@pytest.mark.parametrize("name", ["TL_conformal_u1_fermions_3d.py", "TL_conformal_u1_no_fermions_3d.py"])
def test_the_3d_models_keep_the_origin_branch_in_the_kernels(name):
    args = dict(g=0.7, y=0.5, v_GeV=1.5) if "dark" in name else dict(g=0.7, v_GeV=1.5)
    pot = load_potential(os.path.join(MODELS, name), "specific_potential")(args, verbose=False)
    assert pot.compiled_status == "compiled"
    pot.config.nucleationConf.rate_mode = "model"
    plain = python_path(pot)
    v = float(pot.v)
    for T in v * np.geomspace(1e-4, 0.5, 6):
        for x in (1e-14 * v, 1e-3 * T, 0.05 * T, 0.3 * T, 0.2 * v, 0.9 * v, v):
            found = phases.findLocalMinimum(np.array([x]), T, pot.gradV, pot.d2V)
            reference = phases.findLocalMinimum(np.array([x]), T, plain.gradV, plain.d2V)
            assert found == pytest.approx(reference, rel=1e-4, abs=1e-5 * T)
    # colder than the step of the field: only the stencil on the scale of T sees the symmetric phase
    T = 1e-8 * v
    start = np.array([0.01 * T])
    moved = phases.findLocalMinimum(start, T, pot.gradV, pot.d2V) - start
    reference = phases.findLocalMinimum(start, T, plain.gradV, plain.d2V) - start
    assert reference[0] != 0.0 and moved == pytest.approx(reference, rel=1e-3)
    T = 1e-9 / pot.conversionFactor
    origin = np.array([1e-3 * T])
    assert pot.gradV(origin, T) == pytest.approx(plain.gradV(origin, T), rel=1e-5, abs=0.0)
    assert np.squeeze(pot.d2V(origin, T)) == pytest.approx(np.squeeze(plain.d2V(origin, T)), rel=1e-5, abs=0.0)
    assert np.squeeze(pot.d2V(origin, T)) > 0.0
    assert pot.dgradV_dT(np.zeros(1), 0.1)[0] == 0.0
    assert pot._origin_stencil() is not None
    pot.config.nucleationConf.rate_mode = "the 4D rate"
    assert pot._origin_stencil() is None


def test_the_enthalpy_is_finite_where_the_plasma_is_below_the_rounding_of_the_vacuum_energy():
    """Added as e + p the vacuum energy leaves nothing: 1e-16 of it is the size of the plasma here."""
    import TL_conformal_u1_fermions as model
    pot = model.specific_potential(dict(g=0.55, y=0.48, v_GeV=77.0), verbose=False)
    T, origin = 1e-2, np.zeros(1)
    vacuum = float(pot.energyDensity(origin, 0.0))
    w = float(pot.enthalpyDensity(origin, T, include_decoupled=False))
    assert 0.0 < w < 1e-13 * vacuum
    plasma = pot.radiationEnergyDensity(origin, T, include_decoupled=False)
    assert w == pytest.approx(4.0 / 3.0 * float(plasma), rel=0.05)
    hot = 300.0
    plain = pot.energyDensity(origin, hot, include_decoupled=False) - (
        pot.Vtot(origin, hot, include_decoupled=False) - (pot.V0(pot.X0) + pot.Vct(pot.X0) + pot.V1_from_X(pot.X0)))
    assert pot.enthalpyDensity(origin, hot, include_decoupled=False) == pytest.approx(plain, rel=1e-7)
