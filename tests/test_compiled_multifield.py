"""The compiled potential of a model with several fields: what a model supplies, that it is the Python potential, and
that it follows the model.

TwoSinglets is the template for a model author. The reference is the generic numpy code of generic_potential.
"""
import copy
import os
import subprocess
import sys
import textwrap
import warnings

import numpy as np
import pytest
from scipy import interpolate, optimize

from transitionlistener import bubbledynamics, errors, finiteT, generic_potential as gp, helper_functions, phases
from transitionlistener.compiled import multifield_potential, onefield_potential
from transitionlistener.compiled import multifield as kernels
from transitionlistener.compiled.splines import _dpiece, _jpiece
from transitionlistener.helper_functions import load_potential
from transitionlistener.particles import Fermion, MassSpectrum, Scalar

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")


def python_path(pot):
    """The same model on the generic numpy code."""
    pot.current_kernels()
    plain = copy.copy(pot)
    plain.__dict__.update(_kernels=None, _kernel_stamp=None)
    for name in ("_gradV", "_d2V", "_gradVT"):
        plain.__dict__.pop(name, None)
    return plain


def two_hdm():
    return load_potential(os.path.join(MODELS, "TL_2HDM.py"), "R2HDM")({}, verbose=False)


def two_hdm_bsmpt():
    return load_potential(os.path.join(MODELS, "TL_2HDM_BSMPT.py"), "R2HDM")({}, verbose=False)


def flipflop():
    return load_potential(os.path.join(MODELS, "TL_dark_flipflop.py"), "DarkFlipFlop")({}, verbose=False)


# ------------------------------------------------------------------------------------------- the template of a model
def tree(x, p):
    """V0 + Vct."""
    a, b = x[0], x[1]
    return 0.25 * p[0] * (a**2 - p[3])**2 + 0.25 * p[1] * b**4 + 0.5 * p[2] * a**2 * b**2 + p[5] * a**2


def bosons(x, T, p, m2):
    a, b = x[0], x[1]
    m2[0] = p[0] * (3 * a**2 - p[3]) + p[2] * b**2 + (p[0] + p[2]) / 4 * T**2
    m2[1] = 3 * p[1] * b**2 + p[2] * a**2 + (p[1] + p[2]) / 4 * T**2


def fermions(x, p, m2):
    m2[0] = (p[4] * x[1])**2


class TwoSinglets(gp.generic_potential):
    """Two real singlets and a fermion that couples to the second one, without mixing."""

    model_parameters = {"y": {"default": 0.5, "min": 0, "max": 10}, "v_GeV": {"default": 100.0, "min": 0, "max": np.inf}}

    def init(self, inp, verbose=False):
        self.mp = self.set_modelparams(inp)
        self.Ndim = 2
        self.verbose = verbose
        self.y, self.v_GeV = self.mp["y"]["value"], self.mp["v_GeV"]["value"]
        self.l1, self.l2, self.l12, self.w2, self.ct = 0.1, 0.2, 0.15, 1e6, 1e-3
        self.X0 = np.array([1000.0, 0.0])
        self.v = 1000.0
        self.daisy = "ArnoldEspinosa"
        self.conversionFactor = self.computeConversionFactor(1000.0, self.v_GeV)
        self.renormScaleSq = 1e6
        self.mass_spectrum = MassSpectrum(
            [Scalar(name="m_1", latex_name="m_1", dof=1, is_SM=False), Scalar(name="m_2", latex_name="m_2", dof=2, is_SM=False)],
            [], [Fermion(name="m_f", latex_name="m_f", dof=4, is_SM=False)],
            boson_massSq_fn=self.boson_masses, fermion_massSq_fn=self.fermion_masses)

    def boson_masses(self, X, T):
        a, b = np.asarray(X)[..., 0], np.asarray(X)[..., 1]
        T2 = np.asarray(T, dtype=float)**2
        return np.stack((self.l1 * (3 * a**2 - self.w2) + self.l12 * b**2 + (self.l1 + self.l12) / 4 * T2,
                         3 * self.l2 * b**2 + self.l12 * a**2 + (self.l2 + self.l12) / 4 * T2), axis=-1)

    def fermion_masses(self, X):
        return ((self.y * np.asarray(X)[..., 1])**2)[..., np.newaxis]

    def V0(self, X):
        a, b = np.asarray(X)[..., 0], np.asarray(X)[..., 1]
        return 0.25 * self.l1 * (a**2 - self.w2)**2 + 0.25 * self.l2 * b**4 + 0.5 * self.l12 * a**2 * b**2

    def Vct(self, X):
        return self.ct * np.asarray(X)[..., 0]**2

    def compiled_model(self):
        return tree, bosons, fermions, (self.l1, self.l2, self.l12, self.w2, self.y, self.ct)

    def approxZeroTMin(self):
        return [np.array([1000.0, 0.0])]


class WithoutInterface(TwoSinglets):
    compiled_model = gp.generic_potential.compiled_model


def toy():
    return TwoSinglets({}, verbose=False)


def one_field_tree(x, p):
    return p[0] * x[0]**4


def one_field_mass(x, T, p, m2):
    m2[0] = np.log1p(x[0]**2) + p[0] * T**2


FACTORIES = [two_hdm, two_hdm_bsmpt, flipflop, toy]


def sample(s, seed=1):
    """Fields and temperatures of a run, with the origin and the axes at low and high temperature."""
    rng = np.random.default_rng(seed)
    special = np.array([[0.0, 0.0], [s, 0.0], [0.0, s], [-s, 0.3 * s], [1e-6 * s, 1e-6 * s]])
    X = np.vstack((rng.uniform(-1.5 * s, 1.5 * s, (40, 2)), np.repeat(special, 4, axis=0)))
    T = np.concatenate((rng.uniform(0.0, 2.0 * s, 40), np.tile(s * np.array([0.0, 1e-8, 1e-2, 1.0]), 5)))
    return X, T


def scale(pot):
    return float(np.max(np.abs(pot.X0)))


# -------------------------------------------------------------------------------------- the kernels are the potential
@pytest.mark.parametrize("make", FACTORIES)
def test_the_kernels_are_the_python_potential_on_random_points(make):
    pot = make()
    assert pot.compiled_status == "compiled"
    plain = python_path(pot)
    s = scale(pot)
    X, T = sample(s)
    a, b = plain.Vtot(X, T, False, False), pot.Vtot(X, T, False, False)
    assert np.all(np.abs(a - b) <= 1e-12 * (np.abs(a) + s**4))
    single = [plain.Vtot(x, float(t), False, False) for x, t in zip(X, T)]
    assert np.all(np.abs(np.array(single) - b) <= 1e-12 * (np.abs(a) + s**4))
    assert type(pot.Vtot(X[0], float(T[0]), False, False)) is type(single[0])


@pytest.mark.parametrize("make", FACTORIES)
def test_the_potential_covers_negative_masses_and_the_origin(make):
    pot = make()
    s = scale(pot)
    X, T = sample(s)
    assert np.min(pot.boson_massSq(X, 0.0 * T)[0]) < 0.0
    plain = python_path(pot)
    at = np.array([[0.0, 0.0]])
    assert pot.Vtot(at, 0.0, False, False)[0] == pytest.approx(plain.Vtot(at, 0.0, False, False)[0], rel=1e-13, abs=1e-6)


@pytest.mark.parametrize("make", FACTORIES)
def test_the_radiation_bath_and_the_decoupled_bath(make):
    pot = make()
    plain = python_path(pot)
    s = scale(pot)
    X, T = sample(s)
    keep = T > 1e-3 * s
    for decoupled in (False, True):
        a, b = plain.Vtot(X[keep], T[keep], True, decoupled), pot.Vtot(X[keep], T[keep], True, decoupled)
        assert np.all(np.abs(a - b) <= 1e-12 * (np.abs(a) + s**4))
        a, b = plain.Vtot(X[0], float(T[0]), True, decoupled), pot.Vtot(X[0], float(T[0]), True, decoupled)
        assert abs(a - b) <= 1e-12 * (abs(a) + s**4)
        scalar = pot.Vtot(X[keep], float(T[0]), True, decoupled)
        assert np.allclose(scalar, plain.Vtot(X[keep], float(T[0]), True, decoupled), rtol=1e-12)


@pytest.mark.parametrize("make", [two_hdm, flipflop, toy])
def test_the_decoupled_bath_is_kept_apart_at_one_temperature(make):
    pot = make()
    pot.kin_decoupled_p_geff = lambda T, cf: 3.0 + 0.0 * T
    plain = python_path(pot)
    s = scale(pot)
    x, T = np.asarray(pot.X0, dtype=float), 0.2 * s
    for decoupled in (False, True, False):
        assert pot.Vtot(x, T, True, decoupled) == pytest.approx(plain.Vtot(x, T, True, decoupled), rel=1e-12)


def derivative_bound(pot, X, T, step):
    """What rounding of a potential of this size leaves in a difference with this step."""
    return 64 * np.finfo(float).eps * np.max(np.abs(pot.Vtot(X, T, False, False))) / step


@pytest.mark.parametrize("make", FACTORIES)
def test_the_derivatives_use_the_stencils_of_the_python_path(make):
    pot = make()
    plain = python_path(pot)
    s = scale(pot)
    X, T = sample(s)
    gradient = np.array([plain.gradV(x, t) for x, t in zip(X, T)])
    bound = derivative_bound(pot, X, T, pot.x_eps)
    assert np.max(np.abs(gradient - pot.gradV(X, T))) <= bound
    hessian = np.array([plain.d2V(x, t) for x, t in zip(X, T)])
    assert np.max(np.abs(hessian - pot.d2V(X, T))) <= bound / pot.x_eps
    a = np.array([plain.dgradV_dT(x, t) for x, t in zip(X, T)])
    assert np.max(np.abs(a - pot.dgradV_dT(X, T))) <= 8 * bound / pot.T_eps


@pytest.mark.parametrize("deriv_order", [2, 4])
def test_both_orders_of_the_stencils_on_a_smooth_model(deriv_order):
    pot = toy()
    pot.deriv_order = deriv_order
    pot.x_eps = 0.5          # large, so that the truncation error of the order shows above rounding
    plain = python_path(pot)
    assert pot.compiled_status == "compiled"
    X, T = sample(1000.0, seed=2)
    fine = python_path(pot)
    fine.__dict__.update(x_eps=1e-3, deriv_order=4)
    assert np.allclose(pot.gradV(X, T), np.array([fine.gradV(x, t) for x, t in zip(X, T)]), rtol=1e-6, atol=1e2)
    assert np.allclose(pot.d2V(X, T), np.array([plain.d2V(x, t) for x, t in zip(X, T)]), rtol=1e-8, atol=1e-1)
    assert np.allclose(pot.dgradV_dT(X, T), np.array([plain.dgradV_dT(x, t) for x, t in zip(X, T)]), rtol=1e-6, atol=1e1)


@pytest.mark.parametrize("make", [flipflop, toy])
@pytest.mark.parametrize("daisy", ["off", "Parwani", "ArnoldEspinosa"])
def test_every_daisy_scheme(make, daisy):
    pot = make()
    pot.daisy = daisy
    plain = python_path(pot)
    assert pot.compiled_status == "compiled"
    s = scale(pot)
    X, T = sample(s, seed=3)
    a, b = plain.Vtot(X, T, False, False), pot.Vtot(X, T, False, False)
    assert np.all(np.abs(a - b) <= 1e-12 * (np.abs(a) + s**4))
    assert np.max(np.abs(np.array([plain.dgradV_dT(x, t) for x, t in zip(X, T)]) - pot.dgradV_dT(X, T))) \
        <= 8 * derivative_bound(pot, X, T, pot.x_eps) / pot.T_eps
    assert np.max(np.abs(np.array([plain.gradV(x, t) for x, t in zip(X, T)]) - pot.gradV(X, T))) \
        <= derivative_bound(pot, X, T, pot.x_eps)


def stencil_of_the_potential(pot, X, T, step):
    """The fourth-order difference of Vtot, the gradient the chain rule replaces."""
    out = np.empty(X.shape)
    for j in range(X.shape[-1]):
        e = np.zeros(X.shape[-1])
        e[j] = step
        V = [pot.Vtot(X + k * e, T, False, False) for k in (-2, -1, 1, 2)]
        out[..., j] = (V[0] - 8 * V[1] + 8 * V[2] - V[3]) / (12 * step)
    return out


@pytest.mark.parametrize("make", FACTORIES)
def test_the_chain_rule_holds_for_tachyonic_masses_and_outside_the_tables(make):
    """Masses of both signs, x = m^2 / T^2 below the table of Jb and above those of Jb and Jf, and a very cold plasma."""
    pot = make()
    s = scale(pot)
    X = s * np.array([[0.05, 0.08], [0.4, 0.3], [1.0, 0.9], [-1.4, 1.2], [0.2, -0.05]])
    for T in s * np.array([1e-6, 1e-3, 0.03, 0.3]):
        Tvec = np.full(len(X), T)
        reached = (np.asarray(pot.boson_massSq(X, 0.0 * Tvec)[0]) / T**2)
        step = 1e-2
        reference = stencil_of_the_potential(pot, X, Tvec, step)
        size = np.max(np.abs(reference), axis=1, keepdims=True) + 1e-9 * s**3
        assert np.all(np.abs(pot.gradV(X, Tvec) - reference) <= 1e-5 * size + 1e-13 * s**4 / step), reached
    assert np.min(np.asarray(pot.boson_massSq(X, 0.0 * X[:, 0])[0])) < 0.0


def test_the_points_of_the_chain_rule_test_reach_the_ends_of_the_tables():
    pot = two_hdm()
    s = scale(pot)
    X = s * np.array([[0.05, 0.08], [0.4, 0.3], [1.0, 0.9], [-1.4, 1.2], [0.2, -0.05]])
    x = np.asarray(pot.boson_massSq(X, 0.0 * X[:, 0])[0]) / (1e-3 * s)**2
    assert np.min(x) < finiteT._xbmin and np.max(x) > finiteT._xbmax


def test_the_chain_rule_is_exact_where_a_thermal_mass_vanishes():
    """The derivative of the daisy term is sqrt(m(T)^2) at m(T)^2 = 0, so that a difference of the potential converges only
    as the square root of its step, and to the chain rule."""
    pot = two_hdm()
    s, T = scale(pot), 0.5 * scale(pot)

    def thermal_mass(first):
        return pot.boson_massSq(np.array([first, 0.3 * s]), T)[0][5]

    first = optimize.brentq(thermal_mass, 0.0, 1.5 * s, xtol=1e-9)
    X = np.array([first, 0.3 * s])
    chain = pot.gradV(X, T)
    errors_of_the_stencil = [np.max(np.abs(chain - stencil_of_the_potential(pot, X, T, step))) for step in (1e-3, 1e-4, 1e-5)]
    assert errors_of_the_stencil[0] > 2 * errors_of_the_stencil[1] > 4 * errors_of_the_stencil[2]
    assert errors_of_the_stencil[2] <= 1e-6 * np.max(np.abs(chain))


def test_the_chain_rule_across_a_crossing_of_two_eigenvalues():
    """On the axis of phi_1 the two eigenvalues of the flip-flop model cross, with a kink in each. Their sum in the potential
    is smooth, because they have the same degrees of freedom and constant."""
    pot = flipflop()
    s, T = scale(pot), 0.1 * scale(pot)

    def gap(first):
        m2 = pot.boson_massSq(np.array([first, 0.0]), T)[0]
        return m2[0] - m2[1]

    found = optimize.minimize_scalar(gap, bounds=(0.0, 1.5 * s), method="bounded", options={"xatol": 1e-9})
    assert found.fun < 1e-3 * abs(np.ravel(pot.boson_massSq(np.array([found.x, 0.0]), T)[0])[0]) + 1e-3
    for shift in (0.0, 2e-3, -2e-3, 3e-1):
        X = np.array([found.x + shift, 1e-5])
        reference = stencil_of_the_potential(pot, X, T, 1e-2)
        assert np.all(np.abs(pot.gradV(X, T) - reference) <= 1e-5 * np.max(np.abs(reference)))


@pytest.mark.parametrize("make", [two_hdm, flipflop, toy])
def test_the_entropy_degrees_of_freedom_along_a_phase(make):
    pot = make()
    s = scale(pot)
    plain = python_path(pot)
    T = s * np.linspace(0.02, 1.2, 30)
    field = np.outer(np.sqrt(np.clip(1.0 - (T / (1.3 * s))**2, 0.0, None)), np.asarray(pot.X0, dtype=float))
    phase = phases.PhaseInfo(0, field, T, 0.0 * field)
    for t in s * np.array([0.01, 0.03, 0.1, 0.37, 0.9, 1.1, 1.5]):
        reference = bubbledynamics.h_eff_DS(float(t), plain, phase)
        assert bubbledynamics.h_eff_DS(float(t), pot, phase) == pytest.approx(reference, rel=1e-12, abs=1e-12)
        assert bubbledynamics.h_eff_DS(np.float64(t), pot, phase) == pytest.approx(reference, rel=1e-12, abs=1e-12)


def test_the_entropy_degrees_of_freedom_take_the_kernels_and_only_for_a_plain_phase(monkeypatch):
    pot = two_hdm()
    s = scale(pot)
    T = s * np.linspace(0.02, 1.2, 30)
    field = np.outer(np.ones_like(T), np.asarray(pot.X0, dtype=float))
    phase = phases.PhaseInfo(0, field, T, 0.0 * field)

    class Moved(phases.PhaseInfo):
        def valAt(self, T, deriv=0):
            return np.zeros(2)

    moved = Moved(0, field, T, 0.0 * field)
    monkeypatch.setattr(kernels, "h_eff", lambda *a: -1.0)
    assert bubbledynamics.h_eff_DS(0.3 * s, pot, phase) == -1.0
    assert bubbledynamics.h_eff_DS(0.3 * s, pot, moved) != -1.0
    assert bubbledynamics.h_eff_DS(0.3 * s, python_path(pot), phase) != -1.0
    wrong = phases.PhaseInfo(0, field[:, :1], T, 0.0 * field[:, :1])
    assert bubbledynamics._on_multifield_kernels(pot, wrong) == (None, None)


def test_the_entropy_degrees_of_freedom_do_not_go_below_zero():
    pot = two_hdm()
    s = scale(pot)
    T = s * np.linspace(0.02, 1.2, 30)
    field = np.outer(np.ones_like(T), np.asarray(pot.X0, dtype=float))
    compiled, spline = bubbledynamics._on_multifield_kernels(pot, phases.PhaseInfo(0, field, T, 0.0 * field))
    compiled.ghosts = 1e6
    assert compiled.h_eff(spline, 0.3 * s) == 0.0


def test_the_daisy_scheme_of_the_2hdm_reaches_the_scalar_masses():
    pot = two_hdm()
    pot.daisy = "off"
    plain = python_path(pot)
    assert pot.compiled_status == "compiled"
    X, T = sample(246.0, seed=4)
    a, b = plain.Vtot(X, T, False, False), pot.Vtot(X, T, False, False)
    assert np.all(np.abs(a - b) <= 1e-12 * (np.abs(a) + 246.0**4))


def test_a_field_independent_debye_mass_is_the_exact_thermal_part():
    """At temperatures whose T^2 is lost against the mass, m(T)^2 - m(0)^2 is zero; c T^2 is not."""
    pot = toy()
    assert isinstance(pot._debye_state, np.ndarray)
    X = np.array([3000.0, 2000.0])
    T = 1e-6
    daisy = pot.Vdaisy_from_X(X, T)
    assert daisy != 0.0
    table = pot.current_kernels().P[0]
    assert table[kernels.CONSTANT_DEBYE] == 1.0
    work = np.empty(2 * 2 + 1)
    kernels._zero_temperature_masses(X, pot.current_kernels().P, work)
    assert kernels._zero_modes(X, T, pot.current_kernels().P, work) == pytest.approx(daisy, rel=1e-10, abs=0.0)


@pytest.mark.parametrize("make", [two_hdm, flipflop])
def test_a_mixing_model_takes_the_difference_of_the_two_spectra(make):
    pot = make()
    assert pot.current_kernels().P[0][kernels.CONSTANT_DEBYE] == 0.0


def test_a_model_without_fermions():
    class NoFermions(TwoSinglets):
        def init(self, inp, verbose=False):
            super().init(inp, verbose)
            self.mass_spectrum = MassSpectrum(self.mass_spectrum.bosons, [], boson_massSq_fn=self.boson_masses)

        def compiled_model(self):
            return tree, bosons, None, (self.l1, self.l2, self.l12, self.w2, self.y, self.ct)

    pot = NoFermions({}, verbose=False)
    assert pot.compiled_status == "compiled"
    plain = python_path(pot)
    X, T = sample(1000.0)
    assert np.allclose(pot.Vtot(X, T, False, False), plain.Vtot(X, T, False, False), rtol=1e-11, atol=1e-4)


def test_the_shapes_and_input_types_of_the_python_path():
    pot = toy()
    plain = python_path(pot)
    X = np.arange(1.0, 13.0).reshape(3, 2, 2)
    T = np.array([[10.0], [20.0], [30.0]])
    for x, t in ((X, 50.0), (X, T[:, :, None][:, 0]), (X[0, 0], T[:, 0]), ([1.0, 2.0], 50.0), ([1, 2], 50.0),
                 (np.asfortranarray(X[0]), 50.0), (np.broadcast_to(X[0, 0], (3, 2)), 50.0)):
        assert np.allclose(pot.Vtot(x, t, False, False), plain.Vtot(x, t, False, False), rtol=1e-12)
        assert np.shape(pot.gradV(x, t)) == np.shape(plain.gradV(x, t))
        assert np.shape(pot.d2V(x, t)) == np.shape(plain.d2V(x, t))
        assert np.shape(pot.dgradV_dT(x, t)) == np.shape(plain.dgradV_dT(x, t))


def test_any_input_reaches_one_compiled_specialisation():
    """A read-only or Fortran array would otherwise compile the kernels again, for seconds, in the middle of a run."""
    pot = toy()
    X = np.arange(1.0, 7.0).reshape(3, 2)
    readonly = np.broadcast_to(X[0], (4, 2))
    frozen = X.copy()
    frozen.flags.writeable = False
    before = len(kernels.vtot_many.signatures)
    for x in (X, readonly, frozen, np.asfortranarray(X), X[::-1], [[1.0, 2.0], [3.0, 4.0]]):
        pot.Vtot(x, 50.0, False, False)
        pot.Vtot(x, np.array([50.0]), False, False)
        pot.gradV(x, 50.0)
    assert len(kernels.vtot_many.signatures) <= max(before, 1)


@pytest.mark.parametrize("make", [two_hdm, flipflop])
def test_the_points_of_a_spline_path_are_read_row_by_row(make):
    """SplinePath hands over np.array(pts).T, which is in Fortran order: a row of it has a stride of its own."""
    pot = make()
    plain = python_path(pot)
    s = scale(pot)
    points = np.array([np.linspace(0.0, s, 5), np.linspace(0.3 * s, 0.1 * s, 5)]).T
    assert not points.flags.c_contiguous
    a, b = plain.Vtot(points, 0.2 * s, False, False), pot.Vtot(points, 0.2 * s, False, False)
    assert np.all(np.abs(a - b) <= 1e-12 * (np.abs(a) + s**4))
    assert np.max(np.abs(pot.gradV(points, 0.2 * s) - plain.gradV(points, 0.2 * s))) \
        <= derivative_bound(pot, points, 0.2 * s, pot.x_eps)


def test_a_non_float_temperature_takes_the_python_path():
    pot = toy()
    assert np.allclose(pot.Vtot(np.array([1.0, 2.0]), 50, False, False),
                       python_path(pot).Vtot(np.array([1.0, 2.0]), 50, False, False), rtol=1e-12)


def test_the_newton_minimiser_finds_the_python_minimum():
    pot = flipflop()
    plain = python_path(pot)
    X = pot.X0 * 1.02 + 1.0
    a = phases.findLocalMinimum(X, 10.0, plain.gradV, plain.d2V)
    b = phases.findLocalMinimum(X, 10.0, pot.gradV, pot.d2V)
    assert np.allclose(a, b, rtol=1e-6, atol=1e-4)


@pytest.mark.parametrize("make", [two_hdm, flipflop, toy])
def test_the_simplex_search_is_that_of_scipy(make):
    pot = make()
    plain = python_path(pot)
    s = scale(pot)

    def V(X, T):
        return plain.Vtot(X, T, False, False)

    calls = []
    for x, T in ((1.02 * pot.X0, 0.05 * s), (0.9 * pot.X0 + 0.01 * s, 0.2 * s), (np.array([0.0, 0.1 * s]), 0.3 * s),
                 (np.zeros(2), 0.5 * s), (np.array([0.05 * s, 0.0]), 0.1 * s)):
        tolerance = pot.x_eps
        expected = optimize.fmin(V, x, args=(T,), xtol=tolerance, ftol=np.inf, disp=False)
        found = pot.current_kernels().minimum(x, T, tolerance)
        assert found is not None and np.max(np.abs(found - expected)) <= 4 * tolerance
        calls.append(found.copy())
        coarse = 2e-3 * s       # the search stops on one simplex of the path, so the same answer means the same steps
        assert np.allclose(pot.current_kernels().minimum(x, T, coarse),
                           optimize.fmin(V, x, args=(T,), xtol=coarse, ftol=np.inf, disp=False), rtol=1e-9, atol=1e-9 * s)
    assert len({tuple(c) for c in calls}) > 1


def rough(x, p):
    return 1e9 * np.sin(3.0 * x[0]) * np.cos(2.0 * x[1])


def test_the_simplex_search_on_a_rough_landscape_takes_the_steps_of_scipy():
    """On a landscape of many minima the simplex shrinks on itself, a step the searches above never take. scipy works on
    the same compiled function here, so that the two paths agree to the last digit."""
    kernel = toy().current_kernels()
    addresses = kernel.P[7].copy()
    addresses[kernels.TREE] = multifield_potential._native_function(rough, kernels.TREE_SIGNATURE).address
    P = (*kernel.P[:7], addresses)

    def V(X, T):
        return kernels.veff(np.array(X, dtype=float), T, P, kernel.daisy)

    for x in np.random.default_rng(8).uniform(-40.0, 40.0, (40, 2)):
        ok, found = kernels.nelder_mead(x, 10.0, 1e-3, P, kernel.daisy)
        if ok:
            np.testing.assert_array_equal(found, optimize.fmin(V, x, args=(10.0,), xtol=1e-3, ftol=np.inf, disp=False))


def partly_undefined(x, p):
    return np.sqrt(1000.0 - x[0]) + x[1]**2


def test_the_simplex_search_leaves_a_simplex_with_an_undefined_corner_to_scipy():
    kernel = toy().current_kernels()
    addresses = kernel.P[7].copy()
    addresses[kernels.TREE] = multifield_potential._native_function(partly_undefined, kernels.TREE_SIGNATURE).address
    start = np.array([990.0, 10.0])     # the first step goes to 1039.5, where the tree level is nan
    ok, _ = kernels.nelder_mead(start, 0.0, 1e3, (*kernel.P[:7], addresses), kernel.daisy)
    assert not ok


def test_the_simplex_search_leaves_a_search_without_an_answer_to_scipy():
    pot = toy()
    kernel = pot.current_kernels()
    assert kernel.minimum(np.array([900.0, 10.0]), 100.0, -1.0) is None             # the budget runs out
    assert kernel.minimum(np.array([np.nan, 10.0]), 100.0, 1e-3) is None


def test_the_tracing_takes_the_compiled_search_for_several_fields():
    pot = toy()
    seen = []

    def V(X, T):
        raise AssertionError("the search stays in compiled code")

    V.compiled_minimum = lambda x, t, xeps: seen.append(len(x)) or pot.current_kernels().minimum(x, t, xeps)
    found = phases.traceLocalMinimum(V, np.array([900.0, 10.0]), 100.0, None, None, 1e-3)
    assert seen == [2] and found.shape == (2,)


def test_two_by_two_eigenvalues_are_those_of_the_python_helper():
    rng = np.random.default_rng(5)
    below_zero = (22739.233746429087, 0.0, 22739.233746429083)     # the discriminant is -2.4e-7 in floating point
    for a, b, c in [*rng.normal(size=(50, 3)) * 1e4, below_zero, (1.0, 0.0, 1.0), (np.nan, 1.0, 2.0), (0.0, 0.0, 0.0)]:
        np.testing.assert_allclose(kernels.eigenvalues_2x2(a, b, c), helper_functions.eigenvalues_2x2(a, b, c), rtol=1e-14)


@pytest.mark.parametrize("name, python", [("b", finiteT.Jb_spline), ("f", finiteT.Jf_spline)])
def test_the_polynomial_pieces_are_the_thermal_spline(name, python):
    tb, pb, tabb, tf, pf, tabf = onefield_potential._shared_tables()["pieces"]
    xbmin, xbmax, xfmin, xfmax, ybmin, yfmin = onefield_potential._shared_tables()["limits"]
    t, pieces, tab, xmin, xmax, ymin = ((tb, pb, tabb, xbmin, xbmax, ybmin) if name == "b"
                                        else (tf, pf, tabf, xfmin, xfmax, yfmin))
    rng = np.random.default_rng(7)
    x = np.concatenate((rng.uniform(1.1 * xmin, 1.1 * xmax, 2000), rng.normal(0.0, 1.0, 2000), t[3:-4], t[4:-3] - 1e-9,
                        [0.0, xmin, xmax, xmin * 1.001, xmax * 1.001, np.inf, -np.inf]))
    for guess in (3, t.size // 2, t.size - 5):
        ours = np.array([_jpiece(float(v), guess, t, pieces, tab, xmin, xmax, ymin)[0] for v in x])
        assert np.max(np.abs(ours - python(x))) <= 1e-12
    previous = 3
    for v in x[:400]:                   # the guess each evaluation hands to the next
        value, previous = _jpiece(float(v), previous, t, pieces, tab, xmin, xmax, ymin)
        assert abs(value - python(np.array([v]))[0]) <= 1e-12
        assert 3 <= previous <= t.size - 5
    assert np.isnan(_jpiece(np.nan, 3, t, pieces, tab, xmin, xmax, ymin)[0])


@pytest.mark.parametrize("name, tck", [("b", finiteT._tckb), ("f", finiteT._tckf)])
def test_the_slope_of_the_pieces_is_the_derivative_of_the_thermal_spline(name, tck):
    tb, pb, tabb, tf, pf, tabf = onefield_potential._shared_tables()["pieces"]
    xbmin, xbmax, xfmin, xfmax, _, _ = onefield_potential._shared_tables()["limits"]
    t, pieces, tab, xmin, xmax = (tb, pb, tabb, xbmin, xbmax) if name == "b" else (tf, pf, tabf, xfmin, xfmax)
    rng = np.random.default_rng(8)
    x = np.concatenate((rng.uniform(xmin, xmax, 2000), rng.normal(0.0, 1.0, 2000), t[4:-4]))
    x = x[(x >= xmin) & (x <= xmax)]
    reference = interpolate.splev(x, tck, der=1)
    for guess in (3, t.size // 2, t.size - 5):
        ours = np.array([_dpiece(float(v), guess, t, pieces, tab, xmin, xmax)[0] for v in x])
        assert np.max(np.abs(ours - reference)) <= 1e-9 * (1.0 + np.max(np.abs(reference)))
    assert _dpiece(1.001 * xmin, 3, t, pieces, tab, xmin, xmax)[0] == 0.0
    assert _dpiece(1.001 * xmax, 3, t, pieces, tab, xmin, xmax)[0] == 0.0
    assert np.isnan(_dpiece(np.nan, 3, t, pieces, tab, xmin, xmax)[0])


# ----------------------------------------------------------------------------------------- the kernels follow the model
@pytest.mark.parametrize("make, name", [
    *((flipflop, n) for n in ("lambda0", "gamma", "lambda1", "lambda12", "v2", "y", "dmu1", "dmu2", "dlambda1", "dlambda2",
                              "dlambda12")),
    *((two_hdm, n) for n in ("m11_sq", "m22_sq", "m12_sq", "lambda1", "lambda2", "lambda3", "lambda5", "lambda345",
                             "lambda34m5", "lambda45", "g1", "g2", "yt", "yb", "ytau", "CTempC1", "CTempC2", "dm11_sq",
                             "dm22_sq", "dm12_sq", "dlambda1", "dlambda2", "dlambda3", "dlambda4", "dlambda5")),
    (toy, "l12"), (toy, "ct")])
def test_changing_an_attribute_reaches_the_kernels(make, name):
    pot = make()
    s = scale(pot)
    X, T = sample(s, seed=6)
    before = pot.Vtot(X, T, False, False)
    setattr(pot, name, getattr(pot, name) * 1.05 + 1e-3)
    assert pot.compiled_status == "compiled"
    after = pot.Vtot(X, T, False, False)
    assert not np.allclose(before, after, rtol=1e-13, atol=0.0)
    assert np.allclose(after, python_path(pot).Vtot(X, T, False, False), rtol=1e-11, atol=1e-6 * s**2)


def test_a_counterterm_set_after_the_construction_is_not_stale():
    pot = two_hdm()
    x = np.array([100.0, 200.0])
    pot.Vtot(x, 50.0, False, False)
    pot.dlambda2 = pot.dlambda2 + 0.01
    pot.dm22_sq = pot.dm22_sq + 100.0
    assert pot.Vtot(x, 50.0, False, False) == pytest.approx(python_path(pot).Vtot(x, 50.0, False, False), rel=1e-12)


def test_set_modelparams_reaches_the_kernels():
    pot = toy()
    pot.current_kernels()
    pot.set_modelparams({"y": 0.9})
    pot.y = 0.9
    assert pot.current_kernels().P[0][-2] == 0.9


def test_a_patch_of_a_class_attribute_reaches_the_kernels():
    pot = toy()
    x = np.array([500.0, 700.0])
    before = pot.Vtot(x, 50.0, False, False)
    TwoSinglets.ct = 5e-3
    pot.__dict__.pop("ct")
    try:
        assert pot.Vtot(x, 50.0, False, False) != before
    finally:
        del TwoSinglets.ct


def test_a_deep_copy_keeps_working_and_follows_its_own_changes():
    pot = toy()
    x = np.array([500.0, 700.0])
    other = copy.deepcopy(pot)
    assert other.Vtot(x, 50.0, False, False) == pot.Vtot(x, 50.0, False, False)
    other.ct = 2e-3
    assert other.Vtot(x, 50.0, False, False) != pot.Vtot(x, 50.0, False, False)


def test_one_field_may_supply_a_compiled_model_too():
    """A logarithm in a mass is outside the form the one-field code reads from a model."""
    class OneField(gp.generic_potential):
        model_parameters = {"v_GeV": {"default": 100.0, "min": 0, "max": np.inf}}

        def init(self, inp, verbose=False):
            self.mp = self.set_modelparams(inp)
            self.Ndim, self.verbose, self.v_GeV = 1, verbose, self.mp["v_GeV"]["value"]
            self.X0 = np.array([1000.0])
            self.v = 1000.0
            self.daisy = "ArnoldEspinosa"
            self.conversionFactor = self.computeConversionFactor(1000.0, self.v_GeV)
            self.renormScaleSq = 1e6
            self.l = 0.1
            self.mass_spectrum = MassSpectrum([Scalar(name="m", latex_name="m", dof=1, is_SM=False)], [],
                                              boson_massSq_fn=self.mass)

        def mass(self, X, T):
            return (np.log1p(np.asarray(X)[..., 0]**2) + self.l * np.asarray(T, dtype=float)**2)[..., np.newaxis]

        def V0(self, X):
            return self.l * np.asarray(X)[..., 0]**4

        def compiled_model(self):
            return one_field_tree, one_field_mass, None, (self.l,)

    pot = OneField({}, verbose=False)
    assert pot.compiled_status == "compiled"
    X = np.array([[10.0], [200.0], [-700.0]])
    assert np.allclose(pot.Vtot(X, 50.0, False, False), python_path(pot).Vtot(X, 50.0, False, False), rtol=1e-11)
    assert pot.d2V(X[1], 50.0).shape == (1, 1)


def test_bookkeeping_does_not_rebuild_the_kernels():
    pot = toy()
    kernels_before = pot.current_kernels()
    pot.config = pot.config
    pot.derived_parameters = {}
    assert pot.current_kernels() is kernels_before


class OverridesVtot(TwoSinglets):
    def Vtot(self, X, T, include_radiation=True, include_decoupled=True):
        return super().Vtot(X, T, include_radiation, include_decoupled) + 1.0


class OverridesGradient(TwoSinglets):
    def gradV(self, X, T):
        return super().gradV(X, T) * 2.0


@pytest.mark.parametrize("cls, entry", [(OverridesVtot, "Vtot"), (OverridesGradient, "gradV")])
def test_a_replaced_entry_point_keeps_the_python_path(cls, entry):
    with pytest.warns(UserWarning, match=f"it replaces {entry}"):
        pot = cls({}, verbose=False)
    assert pot.current_kernels() is None
    assert entry in pot.compiled_status


def test_an_instance_patch_keeps_the_python_path_and_its_removal_compiles_again():
    pot = toy()
    pot.Vtot = lambda X, T, *a, **k: 0.0
    assert pot.current_kernels() is None
    del pot.Vtot
    assert pot.compiled_status == "compiled"


# --------------------------------------------------------------------------------------------- a model that disagrees
def wrong_tree(x, p):
    return 0.25 * p[0] * (x[0]**2 - p[3])**2 + 0.25 * p[1] * x[1]**4      # no portal and no counterterm


def wrong_bosons(x, T, p, m2):
    a, b = x[0], x[1]
    m2[0] = 3 * p[1] * b**2 + p[2] * a**2 + (p[1] + p[2]) / 4 * T**2      # the two modes exchanged
    m2[1] = p[0] * (3 * a**2 - p[3]) + p[2] * b**2 + (p[0] + p[2]) / 4 * T**2


def wrong_thermal(x, T, p, m2):
    a, b = x[0], x[1]
    m2[0] = p[0] * (3 * a**2 - p[3]) + p[2] * b**2 + (p[0] + p[2]) / 4 * T**2 * 1.001
    m2[1] = 3 * p[1] * b**2 + p[2] * a**2 + (p[1] + p[2]) / 4 * T**2


def wrong_fermions(x, p, m2):
    m2[0] = (p[4] * x[1])**2 * (1.0 + 1e-8)


def does_not_compile(x, T, p, m2):
    m2[0] = {"a": 1.0}


class WrongNumbers(TwoSinglets):
    def compiled_model(self):
        return tree, bosons, fermions, (self.l1, self.l2, self.l12 * (1.0 + 1e-9), self.w2, self.y, self.ct)


def model_with(functions):
    class Mismatch(TwoSinglets):
        def compiled_model(self):
            return (*functions, (self.l1, self.l2, self.l12, self.w2, self.y, self.ct))
    return Mismatch


@pytest.mark.parametrize("cls, reason", [
    (model_with((wrong_tree, bosons, fermions)), "tree level"),
    (model_with((tree, wrong_bosons, fermions)), "boson masses"),
    (model_with((tree, wrong_thermal, fermions)), "thermal boson masses"),
    (model_with((tree, bosons, wrong_fermions)), "fermion masses"),
    (WrongNumbers, "differ"),
])
def test_a_mismatch_warns_and_keeps_the_python_potential(cls, reason):
    with pytest.warns(UserWarning, match="runs on the Python potential"):
        pot = cls({}, verbose=False)
    assert pot.current_kernels() is None and reason in pot.compiled_status
    x = np.array([500.0, 700.0])
    assert pot.Vtot(x, 50.0, False, False) == python_path(toy()).Vtot(x, 50.0, False, False)


def test_a_replaced_one_loop_potential_is_caught_by_the_check():
    class OtherOneLoop(TwoSinglets):
        def V1(self, bosons, fermions):
            return super().V1(bosons, fermions) * 1.01

    with pytest.warns(UserWarning, match="the potential differ"):
        pot = OtherOneLoop({}, verbose=False)
    assert pot.current_kernels() is None


def test_a_function_numba_cannot_compile_warns_and_keeps_the_python_potential():
    with pytest.warns(UserWarning, match="runs on the Python potential"):
        pot = model_with((tree, does_not_compile, fermions))({}, verbose=False)
    assert pot.current_kernels() is None
    assert pot.Vtot(np.array([500.0, 700.0]), 50.0, False, False) != 0.0


def test_complex_masses_keep_the_python_potential():
    class Complex(TwoSinglets):
        def boson_masses(self, X, T):
            return super().boson_masses(X, T) + 0j

        def V0(self, X):
            return super().V0(X) + 0j

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pot = Complex({}, verbose=False)
    assert pot.current_kernels() is None and "complex" in pot.compiled_status


def test_fermions_of_a_model_need_a_function():
    class Forgotten(TwoSinglets):
        def compiled_model(self):
            return tree, bosons, None, (self.l1, self.l2, self.l12, self.w2, self.y, self.ct)

    with pytest.warns(UserWarning, match="no function for their masses"):
        pot = Forgotten({}, verbose=False)
    assert pot.current_kernels() is None


def test_the_run_deadline_is_not_read_as_a_refusal():
    class Slow(TwoSinglets):
        fail = False

        def compiled_model(self):
            if self.fail:
                raise errors.Timeout("deadline")
            return super().compiled_model()

    pot = Slow({}, verbose=False)
    pot.fail = True
    with pytest.raises(errors.Timeout):
        pot.Vtot(np.array([1.0, 2.0]), 50.0)
    assert pot._kernel_failures == 0


def test_two_fields_without_the_interface_warn_once_and_work():
    gp._warned_python_potential.discard(WithoutInterface)
    with pytest.warns(UserWarning, match="more than one field"):
        pot = WithoutInterface({}, verbose=False)
    assert pot.compiled_status == "more than one field"
    assert pot.current_kernels() is None and pot.thermal_kernels() is None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        again = WithoutInterface({}, verbose=False)
    X, T = sample(1000.0)
    assert np.allclose(pot.Vtot(X, T, False, False), toy().Vtot(X, T, False, False), rtol=1e-11, atol=1e-4)
    assert again.gradV(X, 50.0).shape == X.shape


# ------------------------------------------------------------------------------------------- what stays as it was
@pytest.mark.parametrize("make", [two_hdm, two_hdm_bsmpt, flipflop])
def test_the_energy_density_stays_on_the_python_expressions(make):
    pot = make()
    assert pot.thermal_kernels() is None
    plain = python_path(pot)
    x = np.asarray(pot.X0, dtype=float)
    assert pot.energyDensity(x, 50.0) == plain.energyDensity(x, 50.0)


@pytest.mark.parametrize("path, cls", [
    ("models/TL_2HDM_BSMPT_highacc.py", "R2HDM_HighAcc_NAct100_Conv1_FRatio005"),
    ("arxiv/reproducibility/models/TL_2HDM_BSMPT.py", "R2HDM"),
    ("arxiv/reproducibility/models/TL_2HDM_BSMPT_highacc.py", "R2HDM_HighAcc_NAct100_Conv1_FRatio005"),
])
def test_the_variants_of_the_2hdm_share_the_compiled_model(path, cls):
    if not os.path.exists(os.path.join(ROOT, path)):
        pytest.skip(f"{path} is not part of this checkout")
    assert load_potential(os.path.join(ROOT, path), cls)({}, verbose=False).compiled_status == "compiled"


def test_a_new_process_reads_kernels_and_model_from_the_cache(tmp_path):
    """The model file is loaded under no importable name, as every model of a run."""
    model = tmp_path / "toy_model.py"
    model.write_text(textwrap.dedent("""
        import numpy as np

        def tree(x, p):
            return p[0] * x[0]**4 + p[0] * x[1]**4

        def bosons(x, T, p, m2):
            m2[0] = np.sqrt(1.0 + x[0]**2 + T**2) + x[1]**2

        def fermions(x, p, m2):
            m2[0] = p[0] * x[1]**2
    """))
    script = textwrap.dedent(f"""
        import importlib.util
        import sys
        sys.path.insert(0, {ROOT!r})
        from tests.test_compiled_multifield import TwoSinglets, np
        from transitionlistener.compiled import multifield_potential as mp
        from transitionlistener.compiled import multifield as kernels
        spec = importlib.util.spec_from_file_location("not.importable", {str(model)!r})
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        compiled = [mp._native_function(module.tree, kernels.TREE_SIGNATURE),
                    mp._native_function(module.bosons, kernels.BOSONS_SIGNATURE),
                    mp._native_function(module.fermions, kernels.FERMIONS_SIGNATURE)]
        pot = TwoSinglets({{}}, verbose=False)
        assert pot.compiled_status == "compiled"
        pot.Vtot(np.array([1.0, 2.0]), 50.0, False, False)
        print(sum(f.cache_hits for f in compiled), sum(kernels.veff.stats.cache_hits.values()),
              sum(kernels.veff.stats.cache_misses.values()))
    """)
    env = dict(os.environ, NUMBA_CACHE_DIR=str(tmp_path / "cache"), PYTHONPATH=os.path.join(ROOT, "src"))

    def run():
        done = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, cwd=ROOT)
        assert done.returncode == 0, done.stderr
        return [int(n) for n in done.stdout.split()[-3:]]

    assert run() == [0, 0, 1]
    assert run() == [3, 1, 0]


def test_the_check_at_load_time_reads_only_the_grid():
    """The check has to stay small: one array call of each Python function."""
    pot = toy()
    calls = []
    original = TwoSinglets.boson_masses
    field, _ = multifield_potential._grid(pot, 1000.0)
    assert field.shape[0] <= 80

    def counted(self, X, T):
        calls.append(np.shape(X))
        return original(self, X, T)

    TwoSinglets.boson_masses = counted
    try:
        pot.mass_spectrum.boson_massSq_fn = pot.boson_masses
        pot._kernel_stamp = -1
        pot.current_kernels()
    finally:
        TwoSinglets.boson_masses = original
    assert len(calls) <= 16
