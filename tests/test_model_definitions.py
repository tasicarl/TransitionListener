"""The compiled functions of the 2HDM and the flip-flop model against the numpy formulas of the same models."""
from pathlib import Path

import numpy as np
import pytest

from transitionlistener.helper_functions import eigenvalues_2x2, load_potential

REPO = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(11)
FIELDS = RNG.uniform(-300.0, 300.0, (40, 2))
POINT_2HDM = {"lambda1": 0.28894, "lambda2": 0.26237, "lambda3": 5.80958300958301, "lambda4": -2.17494,
              "lambda5": -2.24170, "m12_sq_GeV2": 1573.171, "tan_beta": 21.3949}


def two_hdm(yukawa_type):
    return load_potential(str(REPO / "models/TL_2HDM.py"), "R2HDM")(dict(POINT_2HDM, yukawa_type=yukawa_type),
                                                                    verbose=False)


def flip_flop():
    pot = load_potential(str(REPO / "models/TL_dark_flipflop.py"), "DarkFlipFlop")(
        {"lambda0": 0.1, "gamma": 1.1, "lambda1": 0.02, "lambda12": 0.2, "v_GeV": 1.0, "y": 0.5}, verbose=False)
    pot.dlambda2, pot.dlambda12 = 3e-3, 2e-3
    return pot


def numpy_2hdm_tree(p, X):
    a, b = X[..., 0], X[..., 1]
    tree = 0.5 * p.m11_sq * a**2 + 0.5 * p.m22_sq * b**2 - p.m12_sq * a * b + 0.125 * p.lambda1 * a**4
    tree = tree + 0.125 * p.lambda2 * b**4 + 0.25 * p.lambda345 * a**2 * b**2
    counter = 0.5 * p.dm11_sq * a**2 + 0.5 * p.dm22_sq * b**2 - p.dm12_sq * a * b + 0.125 * p.dlambda1 * a**4
    counter = counter + 0.125 * p.dlambda2 * b**4 + 0.25 * (p.dlambda3 + p.dlambda4 + p.dlambda5) * a**2 * b**2
    return tree, counter


def numpy_2hdm_bosons(p, X, T):
    a1, a2 = X[..., 0], X[..., 1]
    T2, phi_sq = T * T, a1**2 + a2**2
    Pi1, Pi2 = p.CTempC1 * T2, p.CTempC2 * T2
    even = eigenvalues_2x2(p.m11_sq + 1.5 * p.lambda1 * a1**2 + 0.5 * p.lambda345 * a2**2 + Pi1,
                           -p.m12_sq + p.lambda345 * a1 * a2,
                           p.m22_sq + 1.5 * p.lambda2 * a2**2 + 0.5 * p.lambda345 * a1**2 + Pi2)
    odd = eigenvalues_2x2(p.m11_sq + 0.5 * p.lambda1 * a1**2 + 0.5 * p.lambda34m5 * a2**2 + Pi1,
                          -p.m12_sq + p.lambda5 * a1 * a2,
                          p.m22_sq + 0.5 * p.lambda2 * a2**2 + 0.5 * p.lambda34m5 * a1**2 + Pi2)
    charged = eigenvalues_2x2(p.m11_sq + 0.5 * p.lambda1 * a1**2 + 0.5 * p.lambda3 * a2**2 + Pi1,
                              -p.m12_sq + 0.5 * p.lambda45 * a1 * a2,
                              p.m22_sq + 0.5 * p.lambda2 * a2**2 + 0.5 * p.lambda3 * a1**2 + Pi2)
    a = 0.25 * p.g2**2 * phi_sq + 2 * p.g2**2 * T2
    b = -0.25 * p.g1 * p.g2 * phi_sq
    c = 0.25 * p.g1**2 * phi_sq + 2 * p.g1**2 * T2
    root = np.sqrt(np.maximum(a**2 + 4.0 * b**2 - 2.0 * a * c + c**2, 0.0))
    mW = 0.25 * p.g2**2 * phi_sq
    return np.stack((even[0], even[1], odd[1], odd[0], charged[1], charged[0], 0.25 * (p.g1**2 + p.g2**2) * phi_sq,
                     0.5 * (a + c + root), np.zeros_like(phi_sq), 0.5 * (a + c - root), mW,
                     mW + 2 * p.g2**2 * T2), axis=-1)


def numpy_2hdm_fermions(p, X):
    a1, a2 = X[..., 0], X[..., 1]
    return np.stack(((p.yt * a2)**2, (p.yb * (a2 if p._bottom_uses_phi2 else a1))**2,
                     (p.ytau * (a2 if p._lepton_uses_phi2 else a1))**2), axis=-1)


def numpy_flipflop_tree(p, X):
    a, b = X[..., 0], X[..., 1]
    tree = 0.25 * p.lambda0 * (a**2 + p.gamma**2 * b**2 - p.v2)**2 - 0.5 * p.lambda1 * p.v2 * a**2
    tree = tree + 0.5 * p.lambda12 * a**2 * b**2
    counter = -0.5 * p.dmu1 * a**2 - 0.5 * p.dmu2 * b**2 + 0.25 * p.dlambda1 * a**4 + 0.25 * p.dlambda2 * b**4
    return tree, counter + 0.5 * p.dlambda12 * a**2 * b**2


def numpy_flipflop_bosons(p, X, T):
    x, y = X[..., 0], X[..., 1]
    T2 = T * T
    common = x**2 + p.gamma**2 * y**2 - p.v2
    a = 2 * p.lambda0 * x**2 + p.lambda0 * common - p.lambda1 * p.v2 + p.lambda12 * y**2
    b = 2 * p.gamma**4 * p.lambda0 * y**2 + p.lambda0 * p.gamma**2 * common + p.lambda12 * x**2
    a = a + (p.y**2 + 2 * ((3 + p.gamma**2) * p.lambda0 + p.lambda12)) / 24.0 * T2
    b = b + (p.gamma**2 * p.lambda0 + 3 * p.gamma**4 * p.lambda0 + p.lambda12) / 12.0 * T2
    c = 2 * p.gamma**2 * p.lambda0 * x * y + 2 * p.lambda12 * x * y
    root = np.sqrt(0.25 * (a - b)**2 + c**2)
    return np.stack((0.5 * (a + b) + root, 0.5 * (a + b) - root), axis=-1)


@pytest.mark.parametrize("yukawa_type", [1, 2, 3, 4])
@pytest.mark.parametrize("T", [0.0, 37.0, 150.0])
def test_the_2hdm(yukawa_type, T):
    pot = two_hdm(yukawa_type)
    tree, counter = numpy_2hdm_tree(pot, FIELDS)
    np.testing.assert_allclose(pot.V0(FIELDS), tree, rtol=1e-12)
    np.testing.assert_allclose(pot.Vct(FIELDS), counter, rtol=1e-9, atol=1e-9)
    size = np.max(np.abs(numpy_2hdm_bosons(pot, FIELDS, T)))
    np.testing.assert_allclose(pot.boson_massSq(FIELDS, T)[0], numpy_2hdm_bosons(pot, FIELDS, T), atol=1e-10 * size)
    np.testing.assert_allclose(pot.fermion_massSq(FIELDS)[0], numpy_2hdm_fermions(pot, FIELDS), rtol=1e-13)


def test_the_flip_flop_model():
    pot = flip_flop()
    fields = FIELDS * 3.0
    tree, counter = numpy_flipflop_tree(pot, fields)
    np.testing.assert_allclose(pot.V0(fields), tree, rtol=1e-12)
    np.testing.assert_allclose(pot.Vct(fields), counter, rtol=1e-11)
    np.testing.assert_allclose(pot.boson_massSq(fields, 40.0)[0], numpy_flipflop_bosons(pot, fields, 40.0), rtol=1e-9,
                               atol=1e-9)
    np.testing.assert_allclose(pot.fermion_massSq(fields)[0][..., 0], 0.5 * pot.y**2 * fields[..., 0]**2, rtol=1e-13)


@pytest.mark.parametrize("model", [lambda: two_hdm(1), flip_flop], ids=["2HDM", "flip-flop"])
def test_every_shape_of_the_fields(model):
    pot = model()
    grid = FIELDS[:12].reshape(3, 4, 2)
    assert pot.V0(FIELDS[0]).shape == ()
    assert pot.V0(grid).shape == (3, 4)
    np.testing.assert_array_equal(pot.V0(grid).reshape(-1), pot.V0(grid.reshape(-1, 2)))
    bosons = pot.boson_massSq(grid, 10.0)[0]
    assert bosons.shape[:2] == (3, 4)
    assert pot.boson_massSq(FIELDS[0], 10.0)[0].shape == bosons.shape[2:]
    np.testing.assert_array_equal(pot.boson_massSq(FIELDS[::2][:3], 10.0)[0][1], pot.boson_massSq(FIELDS[2], 10.0)[0])


def test_temperatures_one_for_each_field():
    pot = two_hdm(1)
    T = np.linspace(0.0, 100.0, FIELDS.shape[0])
    np.testing.assert_allclose(pot.boson_massSq(FIELDS, T)[0], numpy_2hdm_bosons(pot, FIELDS, T), atol=1e-8)


def test_a_change_of_a_parameter_reaches_every_function():
    pot = two_hdm(1)
    before = pot.V0(FIELDS), pot.boson_massSq(FIELDS, 0.0)[0], pot.fermion_massSq(FIELDS)[0]
    pot.lambda1 = pot.lambda1 + 0.1
    pot.yt = pot.yt + 0.1
    assert not np.array_equal(before[0], pot.V0(FIELDS))
    assert not np.array_equal(before[1], pot.boson_massSq(FIELDS, 0.0)[0])
    assert not np.array_equal(before[2], pot.fermion_massSq(FIELDS)[0])


def test_the_masses_of_the_vacuum():
    pot = two_hdm(1)
    M2 = numpy_2hdm_bosons(pot, np.array([pot.v1, pot.v2]), 0.0)
    assert pot.derived_parameters["mH_GeV"] == pytest.approx(np.sqrt(M2[1]) * pot.conversionFactor, rel=1e-12)
    assert pot.derived_parameters["mA_GeV"] == pytest.approx(np.sqrt(M2[2]) * pot.conversionFactor, rel=1e-12)
    assert pot.derived_parameters["mHp_GeV"] == pytest.approx(np.sqrt(M2[4]) * pot.conversionFactor, rel=1e-12)
